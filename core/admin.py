from django.contrib import admin
from .models import DroneFlight, ImageCapture, EdgeInferenceResult, CloudInferenceResult, AnalyticsReport

@admin.register(DroneFlight)
class DroneFlightAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'start_time', 'end_time', 'is_completed')
    list_filter = ('is_completed', 'start_time')

@admin.register(ImageCapture)
class ImageCaptureAdmin(admin.ModelAdmin):
    list_display = ('id', 'flight', 'timestamp', 'latitude', 'longitude', 'altitude')
    list_filter = ('flight', 'timestamp')

@admin.register(EdgeInferenceResult)
class EdgeInferenceResultAdmin(admin.ModelAdmin):
    list_display = ('id', 'image', 'classification_label', 'confidence_score', 'created_at')
    list_filter = ('classification_label',)

@admin.register(CloudInferenceResult)
class CloudInferenceResultAdmin(admin.ModelAdmin):
    list_display = ('id', 'image', 'classification_label', 'severity_score', 'confidence_score', 'status', 'processed_at')
    list_filter = ('status', 'classification_label')

@admin.register(AnalyticsReport)
class AnalyticsReportAdmin(admin.ModelAdmin):
    list_display = ('id', 'flight', 'created_at')
