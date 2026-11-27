"""
30-day archive retention and permanent cascade purge helpers for HumayDrone.
"""
import math
import threading
import time
from datetime import timedelta
from typing import Optional, Tuple

from django.utils import timezone

from .models import DroneFlight, SurveyMission

ARCHIVE_RETENTION_DAYS = 30
_LAZY_PURGE_INTERVAL_SECONDS = 300.0
_purge_lock = threading.Lock()
_last_purge_monotonic: Optional[float] = None


def is_archive_expired(archived_at, now=None) -> bool:
    """Returns True when an archived timestamp has reached the 30-day purge cutoff."""
    if not archived_at:
        return False
    ref = now or timezone.now()
    return archived_at <= (ref - timedelta(days=ARCHIVE_RETENTION_DAYS))


def compute_archive_metadata(archived_at, now=None) -> Tuple[Optional[str], Optional[int]]:
    """Returns `(purge_after_iso, days_remaining)` for an archived record."""
    if not archived_at:
        return None, None
    ref = now or timezone.now()
    purge_after = archived_at + timedelta(days=ARCHIVE_RETENTION_DAYS)
    seconds_left = (purge_after - ref).total_seconds()
    days_remaining = max(0, int(math.ceil(seconds_left / 86400.0)))
    return purge_after.isoformat(), days_remaining


def _safe_delete_field_file(field_file) -> None:
    if not field_file or not getattr(field_file, "name", None):
        return
    try:
        field_file.delete(save=False)
    except Exception:
        pass


def delete_flight_permanently(flight: DroneFlight) -> int:
    """
    Deletes stored original captures + segmentation masks for `flight` and
    removes the DroneFlight record (cascading images, inference results, reports).
    """
    images = list(flight.images.select_related("cloud_result").all())
    for img in images:
        _safe_delete_field_file(img.file)
        cloud = getattr(img, "cloud_result", None)
        if cloud is not None:
            _safe_delete_field_file(cloud.segmentation_mask)
    flight.delete()
    return len(images)


def purge_expired_archived_records(force: bool = False) -> dict:
    """
    Permanently removes DroneFlight and SurveyMission records archived >= 30 days ago.
    Rate-limited to run at most every 5 minutes during normal API requests unless `force=True`.
    """
    global _last_purge_monotonic
    now_mono = time.monotonic()
    if not force and _last_purge_monotonic is not None:
        if (now_mono - _last_purge_monotonic) < _LAZY_PURGE_INTERVAL_SECONDS:
            return {"flights_purged": 0, "images_purged": 0, "surveys_purged": 0}

    with _purge_lock:
        now_mono = time.monotonic()
        if not force and _last_purge_monotonic is not None:
            if (now_mono - _last_purge_monotonic) < _LAZY_PURGE_INTERVAL_SECONDS:
                return {"flights_purged": 0, "images_purged": 0, "surveys_purged": 0}

        cutoff = timezone.now() - timedelta(days=ARCHIVE_RETENTION_DAYS)
        expired_flights = list(
            DroneFlight.objects.filter(archived_at__isnull=False, archived_at__lte=cutoff)
        )
        flights_purged = 0
        images_purged = 0
        for flight in expired_flights:
            images_purged += delete_flight_permanently(flight)
            flights_purged += 1

        surveys_purged, _ = SurveyMission.objects.filter(
            archived_at__isnull=False,
            archived_at__lte=cutoff,
        ).delete()

        _last_purge_monotonic = time.monotonic()
        return {
            "flights_purged": flights_purged,
            "images_purged": images_purged,
            "surveys_purged": surveys_purged,
        }
