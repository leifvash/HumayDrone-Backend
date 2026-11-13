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

    @override_settings(CELERY_TASK_ALWAYS_EAGER=True, SUPABASE_JWT_SECRET="test-jwt-secret-key-at-least-32b")
    def test_review_workflow_report_aggregation_and_mission_link(self):
        import base64
        import hashlib
        import hmac
        import json

        mission_uuid = "914dce7a-c0dc-4e64-ac0e-967a3fdc8199"
        self.flight.survey_mission_id = mission_uuid
        self.flight.save(update_fields=["survey_mission_id"])

        # Upload 3 captures into linked flight
        ids = []
        for idx, edge_lbl in enumerate(["Bacterial Leaf Blight", "Whitehead", "Bacterial Leaf Blight"]):
            resp = self.client.post(
                '/api/images/upload/',
                {
                    'flight_id': self.flight.id,
                    'latitude': 8.45431 + idx * 0.0001,
                    'longitude': 124.63205,
                    'altitude': 1.5,
                    'file': self._generate_test_image(),
                    'edge_detections': json.dumps([
                        {"label": edge_lbl, "confidence": 0.85, "boxes": [[10, 10, 80, 80]]}
                    ]),
                },
                format='multipart',
            )
            self.assertEqual(resp.status_code, status.HTTP_202_ACCEPTED)
            ids.append(resp.json()['image_id'])

        # Unauthenticated review must fail 401
        unauth_resp = self.client.patch(
            f'/api/images/{ids[0]}/review/',
            {'review_status': 'VERIFIED', 'review_notes': 'Confirmed blight'},
            format='json',
        )
        self.assertEqual(unauth_resp.status_code, status.HTTP_401_UNAUTHORIZED)

        # Construct real HS256 Supabase JWT token
        def _b64(obj):
            raw = json.dumps(obj, separators=(',', ':')).encode('utf-8')
            return base64.urlsafe_b64encode(raw).decode('ascii').rstrip('=')

        hdr = _b64({"alg": "HS256", "typ": "JWT"})
        pld = _b64({
            "sub": "tech-user-123",
            "email": "agronomist@humaydrone.test",
            "role": "authenticated",
            "exp": 4102444800,
        })
        sig = hmac.new(b"test-jwt-secret-key-at-least-32b", f"{hdr}.{pld}".encode('ascii'), hashlib.sha256).digest()
        jwt_token = f"{hdr}.{pld}.{base64.urlsafe_b64encode(sig).decode('ascii').rstrip('=')}"
        auth_headers = {"HTTP_AUTHORIZATION": f"Bearer {jwt_token}"}

        # 1. Verify image 0 with label override to White Head + notes
        r0 = self.client.patch(
            f'/api/images/{ids[0]}/review/',
            {
                'review_status': 'VERIFIED',
                'reviewed_label': 'Whitehead',
                'review_notes': 'Panicles fully bleached near paddy bund.',
            },
            format='json',
            **auth_headers,
        )
        self.assertEqual(r0.status_code, status.HTTP_200_OK)
        self.assertEqual(r0.json()['image']['cloud_result']['reviewed_label'], 'White Head')
        self.assertEqual(r0.json()['image']['cloud_result']['reviewed_by'], 'agronomist@humaydrone.test')

        # 2. Mark image 1 NEEDS_INSPECTION with note
        r1 = self.client.patch(
            f'/api/images/{ids[1]}/review/',
            {
                'review_status': 'NEEDS_INSPECTION',
                'review_notes': 'Check tiller base manually on-site.',
            },
            format='json',
            **auth_headers,
        )
        self.assertEqual(r1.status_code, status.HTTP_200_OK)

        # 3. Mark image 2 REJECTED -> must be excluded from severity stats & geojson map
        r2 = self.client.patch(
            f'/api/images/{ids[2]}/review/',
            {
                'review_status': 'REJECTED',
                'review_notes': 'Sun glare on water puddle.',
            },
            format='json',
            **auth_headers,
        )
        self.assertEqual(r2.status_code, status.HTTP_200_OK)

        # 4. Excluded "Dead Heart" override must return 400
        bad = self.client.patch(
            f'/api/images/{ids[0]}/review/',
            {'review_status': 'VERIFIED', 'reviewed_label': 'Dead Heart'},
            format='json',
            **auth_headers,
        )
        self.assertEqual(bad.status_code, status.HTTP_400_BAD_REQUEST)

        # 5. Flight analytics report aggregates verified/active captures and excludes REJECTED
        report_resp = self.client.get(f'/api/flights/{self.flight.id}/report/')
        self.assertEqual(report_resp.status_code, status.HTTP_200_OK)
        rep = report_resp.json()
        self.assertEqual(rep['survey_mission_id'], mission_uuid)
        stats = rep['summary_statistics']
        self.assertEqual(stats['total_images_captured'], 3)
        self.assertEqual(stats['total_images_analyzed'], 2)
        self.assertEqual(stats['rejected_count'], 1)
        self.assertEqual(stats['review_breakdown']['VERIFIED'], 1)
        self.assertEqual(stats['review_breakdown']['NEEDS_INSPECTION'], 1)
        self.assertEqual(stats['review_breakdown']['REJECTED'], 1)
        self.assertEqual(len(rep['severity_map']['features']), 2)
        self.assertEqual(len(stats['verified_findings']), 1)
        self.assertEqual(stats['verified_findings'][0]['effective_label'], 'White Head')
        self.assertEqual(len(stats['inspection_checklist']), 1)

    @override_settings(CELERY_TASK_ALWAYS_EAGER=True)
    def test_dji_xmp_relative_altitude_ignores_exif_msl_fallback(self):
        # Construct a JPEG with EXIF GPSAltitude = 33.1m MSL AND DJI XMP RelativeAltitude = 0.50m AGL
        buf = io.BytesIO()
        img = Image.new('RGB', (160, 160), color=(34, 139, 34))
        exif = img.getexif()
        exif[0x8825] = {6: 33.1}  # GPSInfo IFD GPSAltitude = 33.1m sea-level
        xmp_payload = (
            b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
            b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            b'<rdf:Description drone-dji:RelativeAltitude="+0.50"/>'
            b'</rdf:RDF></x:xmpmeta>'
        )
        img.save(buf, format='JPEG', exif=exif.tobytes())
        raw_jpeg = buf.getvalue()
        # Inject XMP packet cleanly right after SOI marker (\xff\xd8)
        xmp_app1_body = b'http://ns.adobe.com/xap/1.0/\x00' + xmp_payload
        xmp_segment = b'\xff\xe1' + len(xmp_app1_body + b'\x00\x00').to_bytes(2, 'big') + xmp_app1_body
        injected = raw_jpeg[:2] + xmp_segment + raw_jpeg[2:]

        upload_file = io.BytesIO(injected)
        upload_file.name = 'dji_xmp_05m.jpg'
        resp = self.client.post(
            '/api/images/upload/',
            {'flight_id': self.flight.id, 'file': upload_file},
            format='multipart',
        )
        self.assertEqual(resp.status_code, status.HTTP_202_ACCEPTED)
        cap = ImageCapture.objects.get(id=resp.json()['image_id'])
        self.assertIsNotNone(cap.altitude)
        self.assertAlmostEqual(cap.altitude, 0.5, places=2)

    def test_supabase_storage_no_path_crash_with_local_image_path(self):
        from unittest.mock import patch, MagicMock
        from django.core.files.uploadedfile import SimpleUploadedFile
        from core.storage import SupabaseMediaStorage, local_image_path

        storage = SupabaseMediaStorage(
            supabase_url="https://example.supabase.co",
            service_role_key="service-role-secret",
            bucket="inference-media",
        )
        # Calling .path() on SupabaseMediaStorage raises NotImplementedError as expected
        with self.assertRaises(NotImplementedError):
            storage.path("captures/sample.jpg")

        sample_bytes = self._generate_test_image().read()
        resp_cm = MagicMock()
        resp_cm.__enter__ = lambda s: s
        resp_cm.__exit__ = lambda s, *a: None
        resp_cm.read.return_value = sample_bytes

        with patch.object(storage, "exists", return_value=False), patch(
            "core.storage.urllib.request.urlopen", return_value=resp_cm
        ):
            saved_name = storage.save("captures/sample.jpg", SimpleUploadedFile("sample.jpg", sample_bytes))
            self.assertEqual(saved_name, "captures/sample.jpg")
            self.assertEqual(
                storage.url(saved_name),
                "https://example.supabase.co/storage/v1/object/public/inference-media/captures/sample.jpg",
            )

            field_mock = MagicMock()
            field_mock.name = saved_name
            field_mock.storage = storage
            type(field_mock).path = property(lambda self: (_ for _ in ()).throw(NotImplementedError()))
            field_mock.open.return_value.__enter__ = lambda s: io.BytesIO(sample_bytes)
            field_mock.open.return_value.__exit__ = lambda s, *a: None

            with local_image_path(field_mock) as tmp_local:
                import os
                self.assertTrue(os.path.exists(tmp_local))


