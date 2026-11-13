import uuid
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class SurveyMission(models.Model):
    """
    Mirrors the Supabase `survey_missions` table structure containing
    autonomous drone mission counts, `boundary_points` perimeter vertices,
    and `disease_points` geotagged lat/lng pinpoints.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_id = models.UUIDField(null=True, blank=True, help_text="Supabase auth.users UUID")
    title = models.CharField(max_length=255, default="Survey Mission")
    captured_at = models.DateTimeField(default=timezone.now)
    total_scanned_count = models.IntegerField(default=0)
    blight_count = models.IntegerField(default=0)
    whitehead_count = models.IntegerField(default=0)
    healthy_count = models.IntegerField(default=0)
    average_confidence = models.FloatField(default=0.0)
    center_lat = models.FloatField(default=0.0)
    center_lng = models.FloatField(default=0.0)
    surveyed_area_m2 = models.FloatField(default=0.0)
    surveyed_area_ha = models.FloatField(default=0.0)
    disease_points = models.JSONField(
        default=list,
        blank=True,
        help_text="Array of {lat, lng, timestamp, confidence, disease_label} objects"
    )
    boundary_points = models.JSONField(
        default=list,
        blank=True,
        help_text="Array of {lat, lng} boundary polygon vertices"
    )
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = 'survey_missions'
        ordering = ['-captured_at']

    def __str__(self):
        return f"{self.title} ({self.total_scanned_count} scanned, {self.blight_count} BLB, {self.whitehead_count} Whitehead)"


class DroneFlight(models.Model):
    """
    Represents a specific autonomous or manual drone flight mission.
    Maps to 'drone_flight' in Capstone Section 3.2.3.2.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='flights', null=True, blank=True)
    title = models.CharField(max_length=200, default="Survey Mission")
    survey_mission_id = models.UUIDField(null=True, blank=True, db_index=True, help_text="Optional linked Supabase survey_missions UUID")
    start_time = models.DateTimeField(default=timezone.now)
    end_time = models.DateTimeField(null=True, blank=True)
    flight_path = models.JSONField(default=list, blank=True, help_text="List of waypoints and target coordinates")
    telemetry_log = models.JSONField(default=dict, blank=True, help_text="Logged speed, battery, altitude data")
    is_completed = models.BooleanField(default=False)

    def __str__(self):
        return f"Flight #{self.id} - {self.title} ({self.start_time.strftime('%Y-%m-%d %H:%M')})"


class ImageCapture(models.Model):
    """
    Individual aerial photo captured during a flight with attached geotag.
    Maps to 'image_capture' in Capstone Section 3.2.3.2.
    """
    flight = models.ForeignKey(DroneFlight, on_delete=models.CASCADE, related_name='images')
    timestamp = models.DateTimeField(default=timezone.now)
    latitude = models.FloatField()
    longitude = models.FloatField()
    altitude = models.FloatField(null=True, blank=True, default=None, help_text="Height above takeoff point (AGL) in meters")
    file = models.ImageField(upload_to='captures/%Y/%m/%d/')
    encrypted = models.BooleanField(default=False)
    retention_period = models.IntegerField(default=365, help_text="Retention period in days")

    def __str__(self):
        return f"Image #{self.id} (Lat: {self.latitude}, Lon: {self.longitude})"


class EdgeInferenceResult(models.Model):
    """
    Real-time lightweight edge detections performed on the Android phone
    (e.g., YOLOv8-Tiny, EfficientNet-Lite).
    Maps to 'edge_inference_result' in Capstone Section 3.2.3.2.
    """
    image = models.OneToOneField(ImageCapture, on_delete=models.CASCADE, related_name='edge_result')
    model_used = models.CharField(max_length=100, default="YOLOv8-Tiny")
    bounding_boxes = models.JSONField(default=list, help_text="Bounding boxes [xmin, ymin, xmax, ymax]")
    classification_label = models.CharField(max_length=100)
    confidence_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Edge [{self.classification_label} ({self.confidence_score*100:.1f}%)]"


class CloudInferenceResult(models.Model):
    """
    High-precision cloud verification results using heavier models
    (Full YOLOv8, U-Net, ResNet-50).
    Maps to 'cloud_inference_result' in Capstone Section 3.2.3.2.
    """
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('PROCESSING', 'Processing'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
    ]
    REVIEW_STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('VERIFIED', 'Verified Disease'),
        ('NEEDS_INSPECTION', 'Needs Field Inspection'),
        ('REJECTED', 'Rejected'),
    ]

    image = models.OneToOneField(ImageCapture, on_delete=models.CASCADE, related_name='cloud_result')
    model_used = models.CharField(max_length=150, default="YOLOv8 + U-Net + ResNet-50")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    segmentation_mask = models.ImageField(upload_to='masks/%Y/%m/%d/', null=True, blank=True)
    severity_score = models.FloatField(default=0.0, help_text="Percentage of discolored canopy area (0.0 to 100.0)")
    classification_label = models.CharField(max_length=100, blank=True)
    confidence_score = models.FloatField(default=0.0)
    treatment_recommendation = models.TextField(blank=True, help_text="Chemical or cultural control advisory")
    error_message = models.TextField(blank=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    review_status = models.CharField(max_length=24, choices=REVIEW_STATUS_CHOICES, default='PENDING')
    review_notes = models.TextField(blank=True, default='', help_text="Technician diagnostic notes for field validation and reporting")
    reviewed_label = models.CharField(max_length=100, blank=True, default='', help_text="Optional technician-corrected diagnosis (Healthy, Bacterial Leaf Blight, White Head)")
    reviewed_by = models.CharField(max_length=255, blank=True, default='')
    reviewed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"Cloud [{self.classification_label} - Severity: {self.severity_score:.1f}%]"


class AnalyticsReport(models.Model):
    """
    Aggregated agricultural diagnostic report for farmers and DA technicians.
    Maps to 'analytics_report' in Capstone Section 3.2.3.2.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='reports', null=True, blank=True)
    flight = models.OneToOneField(DroneFlight, on_delete=models.CASCADE, related_name='report')
    created_at = models.DateTimeField(auto_now_add=True)
    summary_statistics = models.JSONField(default=dict, help_text="Counts of BLB, White Head, Healthy, review statuses, overall damage %")
    severity_map = models.JSONField(default=dict, help_text="GeoJSON heat-zones and discolored coordinates")
    recommendations = models.TextField(blank=True, help_text="Actionable farm advice")

    def __str__(self):
        return f"Report for Flight #{self.flight_id}"

