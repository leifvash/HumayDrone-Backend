import json
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from django.shortcuts import get_object_or_404
from django.utils import timezone

from .models import SurveyMission, DroneFlight, ImageCapture, EdgeInferenceResult, CloudInferenceResult, AnalyticsReport
from .serializers import (
    SurveyMissionSerializer,
    DroneFlightSerializer,
    ImageCaptureSerializer,
    CloudInferenceResultSerializer,
    AnalyticsReportSerializer
)
from inference.pipeline import CloudInferencePipeline

pipeline = CloudInferencePipeline()

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
            if 'healthy' in label_lower:
                continue

            is_wh = 'white' in label_lower or 'dead' in label_lower
            norm_label = 'Whitehead' if is_wh else 'Bacterial Leaf Blight'
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

        # Support offline-generated flight IDs by automatically finding or registering the flight session
        flight = DroneFlight.objects.filter(id=flight_id).first()
        if not flight:
            try:
                flight = DroneFlight.objects.create(
                    id=int(flight_id),
                    title=f"Field Mission #{flight_id}"
                )
            except Exception:
                flight = DroneFlight.objects.create(
                    title=f"Field Mission #{flight_id}"
                )

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
        alt = _safe_float(request.data.get('altitude'), 3.0)

        # 1. Save Image Capture record
        image_record = ImageCapture.objects.create(
            flight=flight,
            latitude=lat,
            longitude=lon,
            altitude=alt,
            file=file
        )

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


class FlightAnalyticsReportView(APIView):
    """
    Returns or computes the aggregate Analytics Report and GeoJSON for a flight.
    """
    def get(self, request, flight_id, *args, **kwargs):
        flight = get_object_or_404(DroneFlight, id=flight_id)
        images = ImageCapture.objects.filter(flight=flight).select_related('cloud_result', 'edge_result')

        total_images = images.count()
        disease_counts = {}
        total_severity = 0.0
        geojson_features = []

        for img in images:
            cloud = getattr(img, 'cloud_result', None)
            edge = getattr(img, 'edge_result', None)
            label = cloud.classification_label if (cloud and cloud.classification_label) else (
                edge.classification_label if edge else "Unprocessed"
            )
            severity = cloud.severity_score if cloud else 0.0
            confidence = cloud.confidence_score if (cloud and cloud.confidence_score) else (
                edge.confidence_score if edge else 0.0
            )

            disease_counts[label] = disease_counts.get(label, 0) + 1
            total_severity += severity

            # Build GeoJSON feature for GIS Discoloration Map
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
                    "disease": label,
                    "confidence": confidence,
                    "severity": severity,
                    "status": cloud.status if cloud else "PENDING",
                    "model_used": cloud.model_used if cloud else "YOLOv8-Tiny",
                    "recommendation": cloud.treatment_recommendation if cloud else "",
                    "image_url": img.file.url if img.file else None,
                    "segmentation_mask": cloud.segmentation_mask.url if (cloud and cloud.segmentation_mask) else None
                }
            })

        avg_severity = (total_severity / total_images) if total_images > 0 else 0.0

        summary_stats = {
            "total_images_analyzed": total_images,
            "average_severity_percentage": round(avg_severity, 2),
            "disease_breakdown": disease_counts
        }

        geojson_map = {
            "type": "FeatureCollection",
            "features": geojson_features
        }

        report, _ = AnalyticsReport.objects.update_or_create(
            flight=flight,
            defaults={
                "summary_statistics": summary_stats,
                "severity_map": geojson_map,
                "recommendations": (
                    "Schedule targeted spot-spraying in areas marked with high severity. "
                    "Refer to individual GPS coordinates on the discoloration map."
                )
            }
        )

        serializer = AnalyticsReportSerializer(report)
        return Response(serializer.data, status=status.HTTP_200_OK)
