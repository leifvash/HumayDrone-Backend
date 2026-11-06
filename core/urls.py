from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    SurveyMissionViewSet,
    DroneFlightViewSet,
    ImageListView,
    ImageUploadAndInferView,
    ImageDetailView,
    ImageRetryInferenceView,
    FlightAnalyticsReportView,
)

router = DefaultRouter()
router.register(r'flights', DroneFlightViewSet, basename='flight')
router.register(r'survey-missions', SurveyMissionViewSet, basename='survey-mission')

urlpatterns = [
    path('', include(router.urls)),
    path('images/', ImageListView.as_view(), name='image-list'),
    path('images/upload/', ImageUploadAndInferView.as_view(), name='image-upload-infer'),
    path('images/<int:image_id>/', ImageDetailView.as_view(), name='image-detail'),
    path('images/<int:image_id>/retry/', ImageRetryInferenceView.as_view(), name='image-retry'),
    path('flights/<int:flight_id>/report/', FlightAnalyticsReportView.as_view(), name='flight-report'),
]


