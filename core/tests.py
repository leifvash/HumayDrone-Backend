import io
from PIL import Image
from django.test import TestCase, override_settings
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status

from .models import SurveyMission, DroneFlight, ImageCapture, CloudInferenceResult

class HumayDroneBackendTestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.flight = DroneFlight.objects.create(
            title="Opol Experimental Rice Plot Test",
            flight_path=[{"lat": 8.4542, "lon": 124.6319}, {"lat": 8.4545, "lon": 124.6322}]
        )

    def _generate_test_image(self):
        file = io.BytesIO()
        image = Image.new('RGB', (320, 320), color=(34, 139, 34)) # Rice canopy green
        image.save(file, 'jpeg')
        file.name = 'test_canopy.jpg'
        file.seek(0)
        return file

    def test_create_flight(self):
        response = self.client.post('/api/flights/', {
            "title": "Grid Flight 50x50",
            "flight_path": []
        }, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(DroneFlight.objects.count(), 2)

    @override_settings(CELERY_TASK_ALWAYS_EAGER=True)
    def test_image_upload_and_async_inference(self):
        test_img = self._generate_test_image()
        payload = {
            'flight_id': self.flight.id,
            'latitude': 8.45431,
            'longitude': 124.63205,
            'altitude': 3.0,
            'file': test_img,
            'edge_detections': '[{"label": "Bacterial Leaf Blight", "confidence": 0.89, "boxes": [[10, 10, 50, 50]]}]'
        }

        # 1. Test immediate 202 Accepted response with task_id
        response = self.client.post('/api/images/upload/', payload, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)
        data = response.json()
        self.assertIn('task_id', data)
        self.assertIn('image_id', data)
        image_id = data['image_id']

        # 2. Check that ImageCapture and CloudInferenceResult were created and completed by eager worker
        self.assertEqual(ImageCapture.objects.count(), 1)
        self.assertEqual(CloudInferenceResult.objects.count(), 1)

        result = CloudInferenceResult.objects.first()
        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(result.classification_label, "Bacterial Leaf Blight")
        self.assertGreater(result.severity_score, 0.0)
        self.assertTrue(len(result.treatment_recommendation) > 0)

        # 3. Test image detail retrieval
        detail_resp = self.client.get(f'/api/images/{image_id}/')
        self.assertEqual(detail_resp.status_code, status.HTTP_200_OK)
        detail_data = detail_resp.json()
        self.assertEqual(detail_data['cloud_result']['status'], 'COMPLETED')

    @override_settings(CELERY_TASK_ALWAYS_EAGER=True)
    def test_flight_analytics_report(self):
        # First upload an image
        test_img = self._generate_test_image()
        self.client.post('/api/images/upload/', {
            'flight_id': self.flight.id,
            'latitude': 8.45431,
            'longitude': 124.63205,
            'altitude': 3.0,
            'file': test_img
        }, format='multipart')

        # Retrieve report
        url = reverse('flight-report', kwargs={'flight_id': self.flight.id})
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIn('summary_statistics', data)
        self.assertIn('severity_map', data)
        self.assertEqual(data['summary_statistics']['total_images_analyzed'], 1)
        self.assertEqual(data['severity_map']['type'], 'FeatureCollection')

    def test_survey_missions_schema_and_pinpoints_endpoint(self):
        mission_uuid = "414dce7a-c0dc-4e64-ac0e-967a3fdc81bb"
        payload = {
            "id": mission_uuid,
            "user_id": "e7e41bd5-6ca5-42ba-b43a-37a6cd65484e",
            "title": "Survey Oct 28, 2026 • 04:59 PM",
            "captured_at": "2026-10-28T08:59:47.380000Z",
            "total_scanned_count": 1155,
            "blight_count": 448,
            "whitehead_count": 4,
            "healthy_count": 703,
            "average_confidence": 0.885655,
            "center_lat": 8.50448420249072,
            "center_lng": 124.575954162071,
            "surveyed_area_m2": 2402.62,
            "surveyed_area_ha": 0.2403,
            "disease_points": [
                {
                    "lat": 8.50438101714454,
                    "lng": 124.57592802312033,
                    "timestamp": 1793177581014,
                    "confidence": 0.8573869,
                    "disease_label": "Bacterial Leaf Blight",
                },
                {
                    "lat": 8.50438101714454,
                    "lng": 124.57592802312033,
                    "timestamp": 1793177581014,
                    "confidence": 0.57163274,
                    "disease_label": "Whitehead",
                },
                {
                    "lat": 8.504493026179066,
                    "lng": 124.57612826587429,
                    "timestamp": 1793177463689,
                    "confidence": 0.90392905,
                    "disease_label": "Healthy",
                },
            ],
            "boundary_points": [
                {"lat": 8.504494647786717, "lng": 124.57615862749517},
                {"lat": 8.504272521728408, "lng": 124.57600674778222},
                {"lat": 8.504352543866941, "lng": 124.57582099363208},
            ],
        }
        create_resp = self.client.post('/api/survey-missions/', payload, format='json')
        self.assertEqual(create_resp.status_code, status.HTTP_201_CREATED)
        self.assertEqual(SurveyMission.objects.count(), 1)

        pinpoints_resp = self.client.get(f'/api/survey-missions/{mission_uuid}/pinpoints/')
        self.assertEqual(pinpoints_resp.status_code, status.HTTP_200_OK)
        pdata = pinpoints_resp.json()
        self.assertEqual(pdata['mission_id'], mission_uuid)
        self.assertEqual(pdata['total_scanned_count'], 1155)
        self.assertEqual(pdata['blight_count'], 448)
        self.assertEqual(pdata['whitehead_count'], 4)
        self.assertEqual(pdata['healthy_count'], 703)
        self.assertEqual(pdata['pinpoints_count'], 2)
        codes = [p['code'] for p in pdata['pinpoints']]
        self.assertIn('BLB-001', codes)
        self.assertIn('WH-001', codes)
        self.assertEqual(pdata['geojson']['type'], 'FeatureCollection')
        self.assertEqual(len(pdata['geojson']['features']), 2)


