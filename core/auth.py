"""
Supabase Bearer JWT verification helpers for authenticated Django mutations.

Verifies `Authorization: Bearer <token>` locally when HS256 + `SUPABASE_JWT_SECRET`
is set, and falls back to `/auth/v1/user` against `SUPABASE_URL` for asymmetric
(`ES256`/`RS256`) tokens or revoked-session checks. Never trusts client-supplied
identity fields when recording reviewer attribution.
"""
import base64
import hashlib
import hmac
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

from django.conf import settings


@dataclass
class AuthenticatedSupabaseUser:
    user_id: str
    email: str
    role: str = ""

    @property
    def display_identity(self) -> str:
        return self.email or self.user_id


def _b64url_decode(segment: str) -> bytes:
    padding = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + padding)


def _verify_hs256_jwt(token: str, secret: str) -> Optional[AuthenticatedSupabaseUser]:
    parts = token.split(".")
    if len(parts) != 3:
        return None
    header_b64, payload_b64, sig_b64 = parts
    try:
        header = json.loads(_b64url_decode(header_b64).decode("utf-8"))
        if header.get("alg") != "HS256":
            return None
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        expected_sig = hmac.new(
            secret.encode("utf-8"), signing_input, hashlib.sha256
        ).digest()
        actual_sig = _b64url_decode(sig_b64)
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None

        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except Exception:
        return None

    exp = payload.get("exp")
    if isinstance(exp, (int, float)) and time.time() >= float(exp):
        return None

    user_id = str(payload.get("sub") or "").strip()
    if not user_id:
        return None
    email = str(payload.get("email") or "").strip()
    role = str(payload.get("role") or "").strip()
    return AuthenticatedSupabaseUser(user_id=user_id, email=email, role=role)


def _verify_via_supabase_auth_api(token: str) -> Optional[AuthenticatedSupabaseUser]:
    supabase_url = str(getattr(settings, "SUPABASE_URL", "") or "").strip().rstrip("/")
    api_key = (
        str(getattr(settings, "SUPABASE_ANON_KEY", "") or "").strip()
        or str(getattr(settings, "SUPABASE_SERVICE_ROLE_KEY", "") or "").strip()
    )
    if not supabase_url or not api_key:
        return None

    req = urllib.request.Request(
        f"{supabase_url}/auth/v1/user",
        method="GET",
        headers={
            "Authorization": f"Bearer {token}",
            "apikey": api_key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError:
        return None
    except Exception:
        return None

    user_id = str(body.get("id") or "").strip()
    if not user_id:
        return None
    email = str(body.get("email") or "").strip()
    role = str(body.get("role") or "").strip()
    return AuthenticatedSupabaseUser(user_id=user_id, email=email, role=role)


def verify_supabase_bearer_token(
    request,
) -> Tuple[Optional[AuthenticatedSupabaseUser], Optional[str]]:
    """
    Validates `Authorization: Bearer <jwt>` from DRF/Django requests.
    Returns `(user, None)` on success or `(None, reason)` on rejection.
    """
    auth_header = (
        request.META.get("HTTP_AUTHORIZATION")
        or request.headers.get("Authorization")
        or ""
    ).strip()

    if not auth_header.lower().startswith("bearer "):
        return None, "Authentication required: provide a Supabase Bearer token."

    token = auth_header[7:].strip()
    if not token:
        return None, "Authentication required: empty Bearer token."

    jwt_secret = str(getattr(settings, "SUPABASE_JWT_SECRET", "") or "").strip()
    if jwt_secret:
        user = _verify_hs256_jwt(token, jwt_secret)
        if user is not None:
            return user, None

    user = _verify_via_supabase_auth_api(token)
    if user is not None:
        return user, None

    if not jwt_secret and not str(getattr(settings, "SUPABASE_URL", "") or "").strip():
        return (
            None,
            "Server authentication verifier is unconfigured (set SUPABASE_JWT_SECRET or SUPABASE_URL).",
        )

    return None, "Invalid or expired Supabase authentication token."


def _is_supabase_active_admin(user_id: str) -> bool:
    """
    Verifies against Supabase `public.profiles` that `user_id` is an active admin.
    Uses `apikey` always and `Authorization: Bearer` only for legacy 3-segment JWT keys.
    """
    supabase_url = str(getattr(settings, "SUPABASE_URL", "") or "").strip().rstrip("/")
    service_key = str(getattr(settings, "SUPABASE_SERVICE_ROLE_KEY", "") or "").strip()
    if not supabase_url or not service_key or not user_id:
        return False

    headers = {
        "apikey": service_key,
        "Accept": "application/json",
    }
    if not service_key.startswith("sb_") and service_key.count(".") == 2 and service_key.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {service_key}"

    import urllib.parse

    encoded_id = urllib.parse.quote(user_id.strip(), safe="")
    req = urllib.request.Request(
        f"{supabase_url}/rest/v1/profiles?id=eq.{encoded_id}&select=role,status&limit=1",
        method="GET",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            rows = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return False

    if not isinstance(rows, list) or not rows:
        return False
    profile = rows[0] if isinstance(rows[0], dict) else {}
    role = str(profile.get("role") or "").strip().lower()
    acct_status = str(profile.get("status") or "").strip().lower()
    return role == "admin" and acct_status in ("", "active")


def verify_supabase_admin_bearer_token(
    request,
) -> Tuple[Optional[AuthenticatedSupabaseUser], Optional[str], int]:
    """
    Validates Supabase Bearer JWT and confirms active `admin` role in `public.profiles`.
    Returns `(user, None, 200)` on success or `(None, reason, http_status)` on failure.
    """
    user, auth_err = verify_supabase_bearer_token(request)
    if user is None:
        return None, auth_err, 401

    if not _is_supabase_active_admin(user.user_id):
        return None, "Forbidden: Only active administrators may archive, restore, or permanently delete missions.", 403

    return user, None, 200
