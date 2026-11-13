import json
import uuid
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from django.shortcuts import get_object_or_404
from django.utils import timezone

from .auth import verify_supabase_bearer_token
from .models import SurveyMission, DroneFlight, ImageCapture, EdgeInferenceResult, CloudInferenceResult, AnalyticsReport
from .serializers import (
    SurveyMissionSerializer,
    DroneFlightSerializer,
    ImageCaptureSerializer,
    CloudInferenceResultSerializer,
    AnalyticsReportSerializer
)
from .storage import local_image_path
from inference.tasks import process_cloud_inference_task


class SurveyMissionViewSet(viewsets.ModelViewSet):
    """
    REST endpoint for the `survey_missions` table structure, supporting
    list/retrieve/create and exact lat/lng disease pinpoint extraction.
    """
    queryset = SurveyMission.objects.all().order_by('-captured_at')
    serializer_class = SurveyMissionSerializer

    @action(detail=True, methods=['get'])
    def pinpoints(self, request, pk=None):
        mission = self.get_object()
        raw_points = mission.disease_points or []
        if isinstance(raw_points, str):
            try:
                raw_points = json.loads(raw_points)
            except Exception:
                raw_points = []

        raw_boundary = mission.boundary_points or []
        if isinstance(raw_boundary, str):
            try:
                raw_boundary = json.loads(raw_boundary)
            except Exception:
                raw_boundary = []

        grouped = {}
        blb_seq = 0
        wh_seq = 0

        for idx, pt in enumerate(raw_points):
            if not isinstance(pt, dict):
                continue
            label_raw = str(pt.get('disease_label') or pt.get('diseaseLabel') or 'Healthy')
            label_lower = label_raw.lower()
            if 'healthy' in label_lower or 'dead' in label_lower:
                continue

            is_wh = 'white' in label_lower
            norm_label = 'White Head' if is_wh else 'Bacterial Leaf Blight'
            lat = float(pt.get('lat', 0.0))
            lng = float(pt.get('lng', 0.0))
            conf = float(pt.get('confidence', 0.0))
            ts = int(pt.get('timestamp', 0))

            key = f"{norm_label}:{lat:.8f},{lng:.8f}"
            if key not in grouped:
                if is_wh:
                    wh_seq += 1
                    code = f"WH-{wh_seq:03d}"
                else:
                    blb_seq += 1
                    code = f"BLB-{blb_seq:03d}"
                grouped[key] = {
                    "id": key,
                    "code": code,
                    "lat": lat,
                    "lng": lng,
                    "disease_label": norm_label,
                    "confidence": conf,
                    "max_confidence": conf,
                    "avg_confidence": conf,
                    "occurrences": 1,
                    "timestamp": ts,
                    "point_index": idx,
                }
            else:
                g = grouped[key]
                total_conf = g["avg_confidence"] * g["occurrences"] + conf
                g["occurrences"] += 1
                g["avg_confidence"] = total_conf / g["occurrences"]
                if conf > g["max_confidence"]:
                    g["max_confidence"] = conf
                    g["confidence"] = conf

        pinpoints = list(grouped.values())
        features = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [p["lng"], p["lat"]],
                },
                "properties": p,
            }
            for p in pinpoints
        ]

        return Response({
            "mission_id": str(mission.id),
            "title": mission.title,
            "total_scanned_count": mission.total_scanned_count,
            "blight_count": mission.blight_count,
            "whitehead_count": mission.whitehead_count,
            "healthy_count": mission.healthy_count,
            "average_confidence": mission.average_confidence,
            "center_lat": mission.center_lat,
            "center_lng": mission.center_lng,
            "pinpoints_count": len(pinpoints),
            "boundary_points": raw_boundary,
            "pinpoints": pinpoints,
            "geojson": {
                "type": "FeatureCollection",
                "features": features,
            },
        }, status=status.HTTP_200_OK)


class DroneFlightViewSet(viewsets.ModelViewSet):
    queryset = DroneFlight.objects.all().order_by('-start_time')
    serializer_class = DroneFlightSerializer


_VALID_DIAGNOSTIC_LABELS = {
    "healthy": "Healthy",
    "blb": "Bacterial Leaf Blight",
    "blight": "Bacterial Leaf Blight",
    "bacterial leaf blight": "Bacterial Leaf Blight",
    "bacterial_leaf_blight": "Bacterial Leaf Blight",
    "whitehead": "White Head",
    "white head": "White Head",
    "white_head": "White Head",
}


def _normalize_review_label(raw_label: str):
    if raw_label is None:
        return None
    cleaned = str(raw_label).strip()
    if not cleaned:
        return ""
    return _VALID_DIAGNOSTIC_LABELS.get(cleaned.lower())


def _extract_dji_metadata_from_path(file_path: str):
    """
    Extracts DJI XMP `drone-dji:RelativeAltitude` (height above takeoff point, AGL)
    and EXIF GPS (latitude, longitude) from a saved DJI JPEG file if available.
    Does NOT substitute EXIF GPSAltitude (MSL elevation) as AGL height.
    """
    import os
    import re
    rel_alt = None
    lat = None
    lon = None
    if not file_path or not os.path.exists(file_path):
        return lat, lon, rel_alt
    try:
        with open(file_path, "rb") as f:
            head = f.read(131072)
        match = re.search(
            rb'(?:drone-dji:RelativeAltitude="([+-]?\d+(?:\.\d+)?)"|<drone-dji:RelativeAltitude>\s*([+-]?\d+(?:\.\d+)?)\s*</drone-dji:RelativeAltitude>)',
            head,
        )
        if match:
            raw_alt = match.group(1) or match.group(2)
            rel_alt = round(max(0.0, float(raw_alt.decode("ascii"))), 2)
    except Exception:
        pass
    try:
        from PIL import Image, ExifTags
        with Image.open(file_path) as img:
            exif = getattr(img, "_getexif", lambda: None)()
            if exif:
                gps_info = None
                for tag, val in exif.items():
                    if ExifTags.TAGS.get(tag, str(tag)) == "GPSInfo":
                        gps_info = val
                        break
                if gps_info:
                    def _dms(dms, ref):
                        dd = float(dms[0]) + (float(dms[1]) / 60.0) + (float(dms[2]) / 3600.0)
                        return -dd if ref in ("S", "W") else dd
                    if 2 in gps_info and 1 in gps_info:
                        lat = round(_dms(gps_info[2], gps_info[1]), 7)
                    if 4 in gps_info and 3 in gps_info:
                        lon = round(_dms(gps_info[4], gps_info[3]), 7)
    except Exception:
        pass
    return lat, lon, rel_alt


def _refine_image_metadata_from_storage(image_record: ImageCapture, update_coords: bool = False):
    """
    Safely inspects an ImageCapture file (supporting remote Supabase Storage)
    to extract DJI takeoff-relative altitude (`xmp_rel_alt >= 0`) and missing GPS coords.
    """
    try:
        with local_image_path(image_record.file) as path:
            if not path:
                return
            exif_lat, exif_lon, xmp_rel_alt = _extract_dji_metadata_from_path(path)
        updated_fields = []
        if xmp_rel_alt is not None and xmp_rel_alt >= 0.0 and image_record.altitude != xmp_rel_alt:
            image_record.altitude = xmp_rel_alt
            updated_fields.append('altitude')
        if update_coords and (image_record.latitude == 0.0 or image_record.longitude == 0.0) and exif_lat and exif_lon:
            image_record.latitude = exif_lat
            image_record.longitude = exif_lon
            updated_fields.extend(['latitude', 'longitude'])
        if updated_fields:
            image_record.save(update_fields=updated_fields)
    except Exception:
        pass


def _parse_optional_uuid(raw_value):
    if not raw_value:
        return None
    try:
        return uuid.UUID(str(raw_value).strip())
    except (ValueError, TypeError):
        return None


class ImageUploadAndInferView(APIView):
    """
    Ingests an aerial drone image along with its GPS metadata and edge inference data.
    Dispatches multi-model inference to a Celery background worker for non-blocking processing.
    """
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request, *args, **kwargs):
        flight_id = request.data.get('flight_id')
        if not flight_id:
            return Response({"error": "flight_id is required"}, status=status.HTTP_400_BAD_REQUEST)

        survey_mission_uuid = _parse_optional_uuid(request.data.get('survey_mission_id'))

        # Support offline-generated flight IDs by automatically finding or registering the flight session
        flight = DroneFlight.objects.filter(id=flight_id).first()
        if not flight:
            try:
                flight = DroneFlight.objects.create(
                    id=int(flight_id),
                    title=f"Field Mission #{flight_id}",
                    survey_mission_id=survey_mission_uuid,
                )
            except Exception:
                flight = DroneFlight.objects.create(
                    title=f"Field Mission #{flight_id}",
                    survey_mission_id=survey_mission_uuid,
                )
        elif survey_mission_uuid and not flight.survey_mission_id:
            flight.survey_mission_id = survey_mission_uuid
            flight.save(update_fields=['survey_mission_id'])

        file = request.FILES.get('file')
        if not file:
            return Response({"error": "Image file is required"}, status=status.HTTP_400_BAD_REQUEST)

        def _safe_float(val, default):
            if val is None or (isinstance(val, str) and not val.strip()):
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        lat = _safe_float(request.data.get('latitude'), 0.0)
        lon = _safe_float(request.data.get('longitude'), 0.0)
        alt = _safe_float(request.data.get('altitude'), None)

        # 1. Save Image Capture record
        image_record = ImageCapture.objects.create(
            flight=flight,
            latitude=lat,
            longitude=lon,
            altitude=alt,
            file=file
        )

        # Auto-refine altitude (AGL) and GPS from DJI XMP/EXIF headers if present
        _refine_image_metadata_from_storage(image_record, update_coords=True)

        # 2. Parse Edge Inference Detections if provided
        edge_raw = request.data.get('edge_detections')
        edge_data = []
        if edge_raw:
            try:
                parsed = json.loads(edge_raw) if isinstance(edge_raw, str) else edge_raw
                if isinstance(parsed, dict):
                    parsed = [parsed]
                if isinstance(parsed, list):
                    edge_data = [item for item in parsed if isinstance(item, dict)]

                if edge_data:
                    first = edge_data[0]
                    boxes_raw = first.get('boxes') or first.get('bounding_boxes') or []
                    boxes = boxes_raw if isinstance(boxes_raw, list) else []
                    EdgeInferenceResult.objects.create(
                        image=image_record,
                        model_used=str(first.get('model_used') or 'YOLOv8-Tiny'),
                        bounding_boxes=boxes,
                        classification_label=str(
                            first.get('classification_label') or first.get('label') or 'Uncertain'
                        ),
                        confidence_score=_safe_float(
                            first.get('confidence_score', first.get('confidence')), 0.0
                        )
                    )
            except Exception:
                edge_data = []

        # 3. Create initial PENDING CloudInferenceResult
        cloud_record = CloudInferenceResult.objects.create(
            image=image_record,
            status='PENDING'
        )

        # 4. Dispatch Asynchronous Celery Worker Task
        task = process_cloud_inference_task.delay(image_record.id, edge_data)

        # Return 202 Accepted with immediate task receipt
        return Response({
            "message": "Image accepted for processing",
            "image_id": image_record.id,
            "task_id": task.id,
            "status": "PENDING"
        }, status=status.HTTP_202_ACCEPTED)


class ImageDetailView(APIView):
    """
    Check real-time processing status of an uploaded image and its inference results.
    """
    def get(self, request, image_id, *args, **kwargs):
        image = get_object_or_404(ImageCapture, id=image_id)
        serializer = ImageCaptureSerializer(image)
        return Response(serializer.data, status=status.HTTP_200_OK)


class ImageListView(APIView):
    """
    Lists uploaded drone image captures with their edge and cloud inference results.
    Supports optional ?flight_id=<id> filtering for the Web Dashboard.
    """
    def get(self, request, *args, **kwargs):
        queryset = ImageCapture.objects.select_related(
            'flight', 'edge_result', 'cloud_result'
        ).order_by('-timestamp')
        flight_id = request.query_params.get('flight_id')
        if flight_id:
            queryset = queryset.filter(flight_id=flight_id)
        serializer = ImageCaptureSerializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class ImageRetryInferenceView(APIView):
    """
    Re-runs the cloud inference pipeline on an existing ImageCapture record.
    Used by the Web Dashboard's Upload & Processing Queue retry action.
    """
    def post(self, request, image_id, *args, **kwargs):
        image = get_object_or_404(ImageCapture, id=image_id)
        _refine_image_metadata_from_storage(image, update_coords=False)
        edge = getattr(image, 'edge_result', None)
        edge_data = []
        if edge and edge.model_used != 'yolov8n_rice.tflite':
            edge_data = [{
                "label": edge.classification_label,
                "confidence": edge.confidence_score,
                "boxes": edge.bounding_boxes,
                "model_used": edge.model_used,
            }]
        task = process_cloud_inference_task.delay(image.id, edge_data)
        image.refresh_from_db()
        serializer = ImageCaptureSerializer(image)
        return Response({
            "message": "Inference re-executed",
            "task_id": getattr(task, 'id', 'eager-sync'),
            "image": serializer.data
        }, status=status.HTTP_200_OK)


def _compute_and_store_flight_report(flight: DroneFlight) -> AnalyticsReport:
    """
    Computes diagnostic statistics and GeoJSON severity map for a flight, honoring
    technician verification statuses (`VERIFIED`, `NEEDS_INSPECTION`, `REJECTED`),
    corrected labels, and diagnostic field notes.
    """
    images = ImageCapture.objects.filter(flight=flight).select_related('cloud_result', 'edge_result').order_by('-timestamp')

    total_images = images.count()
    active_images = 0
    disease_counts = {}
    review_breakdown = {
        "PENDING": 0,
        "VERIFIED": 0,
        "NEEDS_INSPECTION": 0,
        "REJECTED": 0,
    }
    verified_findings = []
    inspection_checklist = []
    total_severity = 0.0
    geojson_features = []

    for img in images:
        cloud = getattr(img, 'cloud_result', None)
        edge = getattr(img, 'edge_result', None)

        review_status = (cloud.review_status if cloud else None) or 'PENDING'
        if review_status not in review_breakdown:
            review_status = 'PENDING'
        review_breakdown[review_status] += 1

        reviewed_label = (cloud.reviewed_label if cloud else '') or ''
        ai_label = cloud.classification_label if (cloud and cloud.classification_label) else (
            edge.classification_label if edge else "Unprocessed"
        )
        effective_label = reviewed_label or ai_label
        severity = cloud.severity_score if cloud else 0.0
        confidence = cloud.confidence_score if (cloud and cloud.confidence_score) else (
            edge.confidence_score if edge else 0.0
        )
        review_notes = (cloud.review_notes if cloud else '') or ''
        boxes = list(edge.bounding_boxes) if (edge and isinstance(edge.bounding_boxes, list)) else []

        if review_status != 'REJECTED':
            active_images += 1
            disease_counts[effective_label] = disease_counts.get(effective_label, 0) + 1
            total_severity += severity
            geojson_features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [img.longitude, img.latitude]
                },
                "properties": {
                    "image_id": img.id,
                    "timestamp": img.timestamp.isoformat() if img.timestamp else None,
                    "altitude": img.altitude,
                    "disease": effective_label,
                    "ai_disease": ai_label,
                    "confidence": confidence,
                    "severity": severity,
                    "status": cloud.status if cloud else "PENDING",
                    "model_used": cloud.model_used if cloud else "YOLOv8-Tiny",
                    "recommendation": cloud.treatment_recommendation if cloud else "",
                    "image_url": img.file.url if img.file else None,
                    "segmentation_mask": cloud.segmentation_mask.url if (cloud and cloud.segmentation_mask) else None,
                    "review_status": review_status,
                    "review_notes": review_notes,
                    "reviewed_label": reviewed_label,
                    "reviewed_by": (cloud.reviewed_by if cloud else '') or '',
                    "reviewed_at": cloud.reviewed_at.isoformat() if (cloud and cloud.reviewed_at) else None,
                    "boxes_count": len(boxes),
                }
            })

        entry_summary = {
            "image_id": img.id,
            "timestamp": img.timestamp.isoformat() if img.timestamp else None,
            "latitude": img.latitude,
            "longitude": img.longitude,
            "altitude": img.altitude,
            "label": effective_label,
            "effective_label": effective_label,
            "ai_label": ai_label,
            "severity": severity,
            "confidence": confidence,
            "boxes_count": len(boxes),
            "review_status": review_status,
            "review_notes": review_notes,
            "reviewed_by": (cloud.reviewed_by if cloud else '') or '',
            "reviewed_at": cloud.reviewed_at.isoformat() if (cloud and cloud.reviewed_at) else None,
            "recommendation": cloud.treatment_recommendation if cloud else "",
        }
        if review_status == 'VERIFIED':
            verified_findings.append(entry_summary)
        elif review_status == 'NEEDS_INSPECTION':
            inspection_checklist.append(entry_summary)

    avg_severity = (total_severity / active_images) if active_images > 0 else 0.0

    summary_stats = {
        "total_images_analyzed": active_images,
        "total_images_captured": total_images,
        "rejected_count": review_breakdown["REJECTED"],
        "average_severity_percentage": round(avg_severity, 2),
        "disease_breakdown": disease_counts,
        "review_breakdown": review_breakdown,
        "verified_findings": verified_findings,
        "inspection_checklist": inspection_checklist,
    }

    geojson_map = {
        "type": "FeatureCollection",
        "features": geojson_features
    }

    blb_count = disease_counts.get("Bacterial Leaf Blight", 0)
    wh_count = disease_counts.get("White Head", 0) + disease_counts.get("Whitehead", 0)
    advice_parts = []
    if blb_count > 0:
        advice_parts.append(
            f"{blb_count} capture(s) flagged Bacterial Leaf Blight — inspect leaf lesions, improve paddy drainage, and schedule copper-based bactericide only on confirmed plots."
        )
    if wh_count > 0:
        advice_parts.append(
            f"{wh_count} capture(s) flagged White Head — verify bleached panicles at listed GPS pinpoints and target localized stem borer management."
        )
    if inspection_checklist:
        advice_parts.append(
            f"{len(inspection_checklist)} capture(s) marked Needs Field Inspection require ground validation prior to treatment."
        )
    if not advice_parts:
        advice_parts.append(
            "All non-rejected captures indicate healthy canopy conditions with minimal discoloration."
            if active_images > 0
            else "No active analyzed captures yet for this flight."
        )

    report, _ = AnalyticsReport.objects.update_or_create(
        flight=flight,
        defaults={
            "summary_statistics": summary_stats,
            "severity_map": geojson_map,
            "recommendations": " ".join(advice_parts),
        }
    )
    return report


class ImageReviewView(APIView):
    """
    Persists technician detection verification status (`PENDING`, `VERIFIED`,
    `NEEDS_INSPECTION`, `REJECTED`), diagnostic notes, and optional 3-class label override.
    Requires a valid Supabase Bearer JWT (`Authorization: Bearer <token>`).
    """
    def patch(self, request, image_id, *args, **kwargs):
        verified_user, auth_err = verify_supabase_bearer_token(request)
        if verified_user is None:
            return Response({"error": auth_err}, status=status.HTTP_401_UNAUTHORIZED)

        image = get_object_or_404(ImageCapture, id=image_id)
        cloud_result, _ = CloudInferenceResult.objects.get_or_create(
            image=image,
            defaults={'status': 'PENDING'},
        )

        valid_statuses = {choice[0] for choice in CloudInferenceResult.REVIEW_STATUS_CHOICES}
        if 'review_status' in request.data:
            req_status = str(request.data.get('review_status') or '').strip().upper()
            if req_status not in valid_statuses:
                return Response(
                    {"error": f"Invalid review_status '{req_status}'. Allowed: {sorted(valid_statuses)}"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            cloud_result.review_status = req_status

        if 'review_notes' in request.data:
            notes = request.data.get('review_notes')
            cloud_result.review_notes = '' if notes is None else str(notes).strip()

        if 'reviewed_label' in request.data:
            raw_lbl = request.data.get('reviewed_label')
            norm_lbl = _normalize_review_label('' if raw_lbl is None else str(raw_lbl))
            if norm_lbl is None:
                return Response(
                    {"error": "Invalid reviewed_label. Allowed: Healthy, Bacterial Leaf Blight, White Head, or empty string."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            cloud_result.reviewed_label = norm_lbl

        cloud_result.reviewed_by = verified_user.display_identity
        cloud_result.reviewed_at = timezone.now()
        cloud_result.save(update_fields=[
            'review_status',
            'review_notes',
            'reviewed_label',
            'reviewed_by',
            'reviewed_at',
        ])

        if hasattr(image.flight, 'report'):
            _compute_and_store_flight_report(image.flight)

        image.refresh_from_db()
        return Response({
            "message": "Detection review updated",
            "image": ImageCaptureSerializer(image).data,
        }, status=status.HTTP_200_OK)


class FlightAnalyticsReportView(APIView):
    """
    Returns or computes the aggregate Analytics Report and GeoJSON for a flight.
    """
    def get(self, request, flight_id, *args, **kwargs):
        flight = get_object_or_404(DroneFlight, id=flight_id)
        report = _compute_and_store_flight_report(flight)
        serializer = AnalyticsReportSerializer(report)
        return Response(serializer.data, status=status.HTTP_200_OK)
