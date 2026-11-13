"""
Supabase Storage backend & safe local temp-file bridge for HumayDrone Cloud Backend.

Keeps mobile SD-card sync (`captured-media` bucket) isolated by storing cloud inference
captures and U-Net/OpenCV mask overlays inside a dedicated public `inference-media` bucket
using Supabase's REST Storage API and `SUPABASE_SERVICE_ROLE_KEY`.
"""
import io
import mimetypes
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import Storage
from django.utils.deconstruct import deconstructible


def is_supabase_storage_configured() -> bool:
    """Returns True when SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are both set."""
    url = str(getattr(settings, "SUPABASE_URL", "") or "").strip()
    key = str(getattr(settings, "SUPABASE_SERVICE_ROLE_KEY", "") or "").strip()
    return bool(url and key)


@deconstructible
class SupabaseMediaStorage(Storage):
    """
    Django 6.0-compatible Storage backend uploading captures & masks to Supabase Storage.
    """

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        service_role_key: Optional[str] = None,
        bucket: Optional[str] = None,
        timeout: int = 45,
    ):
        self.supabase_url = (
            supabase_url
            if supabase_url is not None
            else str(getattr(settings, "SUPABASE_URL", "") or "").strip().rstrip("/")
        )
        self.service_role_key = (
            service_role_key
            if service_role_key is not None
            else str(getattr(settings, "SUPABASE_SERVICE_ROLE_KEY", "") or "").strip()
        )
        self.bucket = (
            bucket
            if bucket is not None
            else str(getattr(settings, "SUPABASE_STORAGE_BUCKET", "inference-media") or "inference-media").strip()
        )
        self.timeout = timeout

    def _normalize_name(self, name: str) -> str:
        return str(name or "").replace("\\", "/").lstrip("/")

    def _is_legacy_jwt(self, key: str) -> bool:
        """Legacy service_role JWTs have 3 dot-separated segments starting with 'eyJ'; sb_secret_ keys are not JWTs."""
        return bool(key and not key.startswith("sb_") and key.count(".") == 2 and key.startswith("eyJ"))

    def _headers(self, extra: Optional[dict] = None) -> dict:
        # New Supabase sb_secret_ / sb_publishable_ keys must ONLY be sent in `apikey`
        # (sending a non-JWT sb_secret_ in Authorization: Bearer causes Storage 400/401 errors).
        headers = {"apikey": self.service_role_key}
        if self._is_legacy_jwt(self.service_role_key):
            headers["Authorization"] = f"Bearer {self.service_role_key}"
        if extra:
            headers.update(extra)
        return headers

    def _object_endpoint(self, name: str) -> str:
        clean = self._normalize_name(name)
        encoded = urllib.parse.quote(clean, safe="/")
        return f"{self.supabase_url}/storage/v1/object/{self.bucket}/{encoded}"

    def _public_url(self, name: str) -> str:
        clean = self._normalize_name(name)
        encoded = urllib.parse.quote(clean, safe="/")
        return f"{self.supabase_url}/storage/v1/object/public/{self.bucket}/{encoded}"

    def _save(self, name: str, content) -> str:
        clean_name = self._normalize_name(name)
        if not self.supabase_url or not self.service_role_key:
            raise RuntimeError("Supabase Storage credentials are not configured.")

        if hasattr(content, "seek"):
            try:
                content.seek(0)
            except Exception:
                pass
        data = content.read()
        if isinstance(data, str):
            data = data.encode("utf-8")

        guessed_type, _ = mimetypes.guess_type(clean_name)
        content_type = getattr(content, "content_type", None) or guessed_type or "application/octet-stream"

        req = urllib.request.Request(
            self._object_endpoint(clean_name),
            data=data,
            method="POST",
            headers=self._headers(
                {
                    "Content-Type": content_type,
                    "x-upsert": "true",
                }
            ),
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise IOError(
                f"Supabase Storage upload failed ({exc.code}) for '{clean_name}': {body}"
            ) from exc
        return clean_name

    def _open(self, name: str, mode: str = "rb"):
        clean_name = self._normalize_name(name)
        req = urllib.request.Request(
            self._object_endpoint(clean_name),
            method="GET",
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = resp.read()
            return ContentFile(payload, name=clean_name)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise FileNotFoundError(f"Object not found in Supabase Storage: {clean_name}") from exc
            body = exc.read().decode("utf-8", errors="ignore")
            raise IOError(
                f"Supabase Storage read failed ({exc.code}) for '{clean_name}': {body}"
            ) from exc

    def exists(self, name: str) -> bool:
        clean_name = self._normalize_name(name)
        if not clean_name:
            return False
        req = urllib.request.Request(
            self._object_endpoint(clean_name),
            method="HEAD",
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=15):
                return True
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 404):
                return False
            return False
        except Exception:
            return False

    def delete(self, name: str) -> None:
        clean_name = self._normalize_name(name)
        if not clean_name:
            return
        req = urllib.request.Request(
            self._object_endpoint(clean_name),
            method="DELETE",
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=15):
                return
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return
            raise

    def size(self, name: str) -> int:
        clean_name = self._normalize_name(name)
        req = urllib.request.Request(
            self._object_endpoint(clean_name),
            method="HEAD",
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                length = resp.headers.get("Content-Length")
                return int(length) if length else 0
        except Exception:
            return 0

    def url(self, name: str) -> str:
        return self._public_url(name)

    def path(self, name: str):
        raise NotImplementedError("SupabaseMediaStorage does not support local filesystem paths.")


@contextmanager
def local_image_path(field_file) -> Iterator[Optional[str]]:
    """
    Yields a local filesystem path for an ImageFieldFile cleanly regardless of whether
    Django's default storage backend is local filesystem or remote Supabase Storage.
    """
    if not field_file or not getattr(field_file, "name", None):
        yield None
        return

    try:
        candidate = field_file.path
        if candidate and os.path.exists(candidate):
            yield candidate
            return
    except NotImplementedError:
        pass
    except Exception:
        pass

    ext = Path(str(field_file.name)).suffix or ".jpg"
    tmp_path = None
    try:
        with field_file.open("rb") as f:
            raw = f.read()
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        yield tmp_path
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
