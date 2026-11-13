from rest_framework import serializers
from .models import SurveyMission, DroneFlight, ImageCapture, EdgeInferenceResult, CloudInferenceResult, AnalyticsReport

class SurveyMissionSerializer(serializers.ModelSerializer):
    id = serializers.UUIDField(required=False)

    class Meta:
        model = SurveyMission
        fields = [
            'id',
            'user_id',
            'title',
            'captured_at',
            'total_scanned_count',
            'blight_count',
            'whitehead_count',
            'healthy_count',
            'average_confidence',
            'center_lat',
            'center_lng',
            'surveyed_area_m2',
            'surveyed_area_ha',
            'disease_points',
            'boundary_points',
            'created_at',
        ]

class EdgeInferenceResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = EdgeInferenceResult
        fields = ['id', 'model_used', 'bounding_boxes', 'classification_label', 'confidence_score', 'created_at']

class CloudInferenceResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = CloudInferenceResult
        fields = [
            'id', 'model_used', 'status', 'segmentation_mask',
            'severity_score', 'classification_label', 'confidence_score',
            'treatment_recommendation', 'error_message', 'processed_at',
            'review_status', 'review_notes', 'reviewed_label',
            'reviewed_by', 'reviewed_at',
        ]

class ImageCaptureSerializer(serializers.ModelSerializer):
    flight_title = serializers.CharField(source='flight.title', read_only=True)
    edge_result = EdgeInferenceResultSerializer(read_only=True)
    cloud_result = CloudInferenceResultSerializer(read_only=True)

    class Meta:
        model = ImageCapture
        fields = ['id', 'flight', 'flight_title', 'timestamp', 'latitude', 'longitude', 'altitude', 'file', 'edge_result', 'cloud_result']

class DroneFlightSerializer(serializers.ModelSerializer):
    images_count = serializers.IntegerField(source='images.count', read_only=True)

    class Meta:
        model = DroneFlight
        fields = [
            'id', 'title', 'survey_mission_id', 'start_time', 'end_time',
            'flight_path', 'telemetry_log', 'is_completed', 'images_count',
        ]

class AnalyticsReportSerializer(serializers.ModelSerializer):
    flight_id = serializers.IntegerField(source='flight.id', read_only=True)
    flight_title = serializers.CharField(source='flight.title', read_only=True)
    survey_mission_id = serializers.UUIDField(source='flight.survey_mission_id', read_only=True)

    class Meta:
        model = AnalyticsReport
        fields = [
            'id',
            'flight_id',
            'flight_title',
            'survey_mission_id',
            'created_at',
            'summary_statistics',
            'severity_map',
            'recommendations',
        ]

