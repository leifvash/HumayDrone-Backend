import os
import tempfile
import cv2
import numpy as np
import torch
from django.test import TestCase

from inference.pipeline import CloudInferencePipeline
from inference.models import (
    RiceCanopyUNet,
    RiceDiscolorationResNet50,
    YoloV8TFLiteDetector,
    DetectionResult,
)


class CloudNeuralInferenceTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.pipeline = CloudInferencePipeline()
        cls.repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        cls.demo_dir = os.path.join(
            cls.repo_root,
            "SampleCode-V5",
            "android-sdk-v5-sample",
            "src",
            "main",
            "assets",
            "demo_captures",
        )

    def test_weights_bundled_and_real_models_loaded(self):
        weights_dir = os.path.join(os.path.dirname(__file__), "weights")
        self.assertTrue(
            os.path.exists(os.path.join(weights_dir, "yolov8n_rice.tflite")),
            "yolov8n_rice.tflite must exist in inference/weights/",
        )
        self.assertTrue(
            os.path.exists(os.path.join(weights_dir, "unet_canopy.pth")),
            "unet_canopy.pth must exist in inference/weights/",
        )
        self.assertTrue(
            os.path.exists(os.path.join(weights_dir, "resnet50_disease.pth")),
            "resnet50_disease.pth must exist in inference/weights/",
        )
        self.assertTrue(self.pipeline.is_real_models_loaded)
        self.assertIsNotNone(self.pipeline.yolo_tflite_detector)
        self.assertIsNotNone(self.pipeline.unet_model)
        self.assertIsNotNone(self.pipeline.resnet_model)

    def test_yolo_tflite_detector_live_inference_and_tensor_decoding(self):
        detector = self.pipeline.yolo_tflite_detector
        self.assertIsInstance(detector, YoloV8TFLiteDetector)

        # 1. Run real TFLite interpreter on demo images
        blight_path = os.path.join(self.demo_dir, "demo_blight.jpg")
        dets, summary = detector.detect_with_raw_summary(blight_path)
        self.assertIsInstance(dets, list)
        self.assertIn("max_confidence", summary)
        self.assertEqual(summary["num_anchors"], 8400)

        # 2. Test decoding channels-first [1, 7, 8400] and [1, 8, 8400]
        tensor_cf7 = np.zeros((1, 7, 8400), dtype=np.float32)
        # Anchor 10: normalized coords [0.5, 0.5, 0.2, 0.2], class 1 (BLB) score = 0.88
        tensor_cf7[0, 0:4, 10] = [0.5, 0.5, 0.2, 0.2]
        tensor_cf7[0, 4 + 1, 10] = 0.88
        # Anchor 11: overlapping same-class box with lower confidence = 0.60 (should be suppressed by NMS)
        tensor_cf7[0, 0:4, 11] = [0.51, 0.50, 0.2, 0.2]
        tensor_cf7[0, 4 + 1, 11] = 0.60
        # Anchor 20: below threshold (0.30 < 0.45)
        tensor_cf7[0, 0:4, 20] = [0.2, 0.2, 0.1, 0.1]
        tensor_cf7[0, 4 + 2, 20] = 0.30

        decoded_cf7, _ = detector.decode_output(tensor_cf7, orig_width=1000, orig_height=500)
        self.assertEqual(len(decoded_cf7), 1)
        self.assertEqual(decoded_cf7[0].label, "Bacterial Leaf Blight")
        self.assertAlmostEqual(decoded_cf7[0].confidence, 0.88, places=4)
        self.assertAlmostEqual(decoded_cf7[0].box[0], 400.0, places=1)
        self.assertAlmostEqual(decoded_cf7[0].box[1], 200.0, places=1)
        self.assertAlmostEqual(decoded_cf7[0].box[2], 600.0, places=1)
        self.assertAlmostEqual(decoded_cf7[0].box[3], 300.0, places=1)

        # 3. Test decoding channels-last [1, 8400, 8] with pixel-scale coordinates
        tensor_cl8 = np.zeros((1, 8400, 8), dtype=np.float32)
        tensor_cl8[0, 42, 0:4] = [320.0, 320.0, 128.0, 128.0]
        tensor_cl8[0, 42, 4 + 3] = 0.91  # Class 3: White Head
        decoded_cl8, _ = detector.decode_output(tensor_cl8, orig_width=640, orig_height=640)
        self.assertEqual(len(decoded_cl8), 1)
        self.assertEqual(decoded_cl8[0].label, "White Head")
        self.assertAlmostEqual(decoded_cl8[0].confidence, 0.91, places=4)
        self.assertAlmostEqual(decoded_cl8[0].box[0], 256.0, places=1)
        self.assertAlmostEqual(decoded_cl8[0].box[2], 384.0, places=1)

    def test_unet_and_resnet_neural_forward_execution(self):
        # Verify U-Net and ResNet-50 execute real forward passes and distinguish healthy vs diseased rice
        healthy_path = os.path.join(self.demo_dir, "demo_healthy.jpg")
        blight_path = os.path.join(self.demo_dir, "demo_blight.jpg")
        whitehead_path = os.path.join(self.demo_dir, "demo_whitehead.jpg")

        healthy_res = self.pipeline._run_real_models(healthy_path)
        blight_res = self.pipeline._run_real_models(blight_path)
        whitehead_res = self.pipeline._run_real_models(whitehead_path)

        self.assertEqual(healthy_res["status"], "COMPLETED")
        self.assertEqual(healthy_res["classification_label"], "Healthy")
        self.assertEqual(healthy_res["severity_score"], 0.0)
        self.assertIn("YOLOv8n-TFLite + RiceCanopyUNet + ResNet-50", healthy_res["model_used"])

        self.assertEqual(blight_res["status"], "COMPLETED")
        self.assertEqual(blight_res["classification_label"], "Bacterial Leaf Blight")
        self.assertGreater(blight_res["severity_score"], 0.0)
        self.assertGreater(
            blight_res["unet_mean_activation"],
            healthy_res["unet_mean_activation"],
        )
        self.assertGreater(
            blight_res["resnet_probabilities"]["Bacterial Leaf Blight"],
            0.5,
        )

        self.assertEqual(whitehead_res["status"], "COMPLETED")
        self.assertEqual(whitehead_res["classification_label"], "White Head")
        self.assertGreater(whitehead_res["severity_score"], 0.0)
        self.assertGreater(
            whitehead_res["resnet_probabilities"]["White Head"],
            0.5,
        )

    def test_edge_cases_missing_image_and_dead_heart(self):
        # Missing image path should return graceful COMPLETED fallback
        missing_res = self.pipeline._run_real_models(
            "non_existent_path_12345.jpg",
            edge_detections=[{"label": "Dead Heart", "confidence": 0.87}],
        )
        self.assertEqual(missing_res["status"], "COMPLETED")
        self.assertEqual(missing_res["classification_label"], "Dead Heart")

        # Synthetic Dead Heart image on disk (both canonical and shifted/perturbed coordinates)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            img_bgr = np.full((300, 300, 3), [40, 130, 40], dtype=np.uint8)
            img_bgr[80:240, 110:190] = [30, 65, 150]
            cv2.imwrite(tmp_path, img_bgr)

            dh_res = self.pipeline.process_image(tmp_path)
            self.assertEqual(dh_res["status"], "COMPLETED")
            self.assertEqual(dh_res["classification_label"], "Dead Heart")
            self.assertGreater(dh_res["severity_score"], 0.0)
            self.assertTrue(len(dh_res["yolo_detections"]) > 0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_unicode_paths_small_whitehead_and_zero_score_anchors(self):
        # 1. Zero-score anchors must not pass even at confidence_threshold=0.0 (matching YoloV8Detector.kt)
        detector = self.pipeline.yolo_tflite_detector
        zero_tensor = np.zeros((1, 7, 8400), dtype=np.float32)
        zero_dets, _ = detector.decode_output(
            zero_tensor, orig_width=640, orig_height=640, confidence_threshold=0.0
        )
        self.assertEqual(len(zero_dets), 0)

        # 2. 2D grayscale and 4-channel BGRA arrays in YoloV8TFLiteDetector
        gray_2d = np.zeros((128, 128), dtype=np.uint8)
        bgra_4ch = np.zeros((128, 128, 4), dtype=np.uint8)
        self.assertIsInstance(detector.detect(gray_2d), list)
        self.assertIsInstance(detector.detect(bgra_4ch), list)

        # 3. Small / low-saturation White Head image (128x128 where white pixels < 2500 and S < 30)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            wh_path = tmp.name
        try:
            wh_small = np.full((128, 128, 3), [34, 139, 34], dtype=np.uint8)
            wh_small[30:90, 40:80] = [230, 235, 235]
            cv2.imwrite(wh_path, wh_small)
            wh_res = self.pipeline._run_real_models(wh_path)
            self.assertEqual(wh_res["status"], "COMPLETED")
            self.assertEqual(wh_res["classification_label"], "White Head")
            self.assertGreater(wh_res["severity_score"], 5.0)
        finally:
            if os.path.exists(wh_path):
                os.remove(wh_path)

        # 4. Unicode directory & filename on Windows
        import shutil
        uni_dir = tempfile.mkdtemp(prefix="humay_piña_•_")
        try:
            uni_path = os.path.join(uni_dir, "captura_ñ_•.jpg")
            blb_img = np.full((256, 256, 3), [34, 139, 34], dtype=np.uint8)
            blb_img[35:215, 45:210] = [35, 165, 215]
            self.assertTrue(self.pipeline._write_image_bgr(uni_path, blb_img))
            uni_res = self.pipeline.process_image(uni_path)
            self.assertEqual(uni_res["status"], "COMPLETED")
            self.assertEqual(uni_res["classification_label"], "Bacterial Leaf Blight")
            self.assertGreater(uni_res["severity_score"], 10.0)
            self.assertTrue(len(uni_res["mask_relative_path"]) > 0)
        finally:
            shutil.rmtree(uni_dir, ignore_errors=True)

    def test_upload_corrupted_and_malformed_edge_payloads(self):
        import io
        import json
        from PIL import Image
        from django.test import override_settings
        from rest_framework.test import APIClient
        from core.models import DroneFlight, CloudInferenceResult, EdgeInferenceResult

        client = APIClient()
        flight = DroneFlight.objects.create(title="Edge Robustness Flight")

        def make_png(mode="RGBA"):
            buf = io.BytesIO()
            color = 120 if mode == "L" else (34, 139, 34, 255)[: len(mode)]
            Image.new(mode, (128, 128), color=color).save(buf, "PNG")
            buf.name = f"test_{mode}.png"
            buf.seek(0)
            return buf

        with override_settings(CELERY_TASK_ALWAYS_EAGER=True):
            # 1. Dict edge_detections + blank coordinate strings + RGBA PNG
            resp1 = client.post(
                "/api/images/upload/",
                {
                    "flight_id": flight.id,
                    "latitude": "",
                    "longitude": "",
                    "altitude": "",
                    "file": make_png("RGBA"),
                    "edge_detections": json.dumps(
                        {"label": "Whitehead", "confidence": 0.91, "boxes": [[10, 10, 60, 60]]}
                    ),
                },
                format="multipart",
            )
            self.assertEqual(resp1.status_code, 202)
            img1_id = resp1.json()["image_id"]
            res1 = CloudInferenceResult.objects.get(image_id=img1_id)
            self.assertEqual(res1.status, "COMPLETED")
            self.assertEqual(res1.classification_label, "White Head")

            # 2. '[null]' edge_detections + Grayscale PNG + retry endpoint
            resp2 = client.post(
                "/api/images/upload/",
                {
                    "flight_id": flight.id,
                    "file": make_png("L"),
                    "edge_detections": "[null]",
                },
                format="multipart",
            )
            self.assertEqual(resp2.status_code, 202)
            img2_id = resp2.json()["image_id"]
            res2 = CloudInferenceResult.objects.get(image_id=img2_id)
            self.assertEqual(res2.status, "COMPLETED")
            edge2 = EdgeInferenceResult.objects.get(image_id=img2_id)
            self.assertEqual(edge2.model_used, "yolov8n_rice.tflite")

            retry_resp = client.post(f"/api/images/{img2_id}/retry/")
            self.assertEqual(retry_resp.status_code, 200)
            self.assertEqual(retry_resp.json()["image"]["cloud_result"]["status"], "COMPLETED")

