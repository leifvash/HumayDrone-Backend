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

        # 2. Test decoding channels-first [1, 6, 8400] (2-class) and [1, 7, 8400] (legacy 3-class)
        tensor_cf6 = np.zeros((1, 6, 8400), dtype=np.float32)
        tensor_cf6[0, 0:4, 8] = [0.5, 0.5, 0.2, 0.2]
        tensor_cf6[0, 4 + 1, 8] = 0.90  # BLB
        decoded_cf6, _ = detector.decode_output(tensor_cf6, orig_width=1000, orig_height=500)
        self.assertEqual(len(decoded_cf6), 1)
        self.assertEqual(decoded_cf6[0].label, "Bacterial Leaf Blight")

        tensor_cf7 = np.zeros((1, 7, 8400), dtype=np.float32)
        # Anchor 10: normalized coords [0.5, 0.5, 0.2, 0.2], class 1 (BLB) score = 0.88
        tensor_cf7[0, 0:4, 10] = [0.5, 0.5, 0.2, 0.2]
        tensor_cf7[0, 4 + 1, 10] = 0.88
        # Anchor 11: overlapping same-class box with lower confidence = 0.60 (should be suppressed by NMS)
        tensor_cf7[0, 0:4, 11] = [0.51, 0.50, 0.2, 0.2]
        tensor_cf7[0, 4 + 1, 11] = 0.60
        # Anchor 20: legacy class 2 ("White Head") must be suppressed even above threshold
        tensor_cf7[0, 0:4, 20] = [0.2, 0.2, 0.1, 0.1]
        tensor_cf7[0, 4 + 2, 20] = 0.93

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
        tensor_cl8[0, 41, 0:4] = [320.0, 320.0, 128.0, 128.0]
        tensor_cl8[0, 41, 4 + 1] = 0.91  # Class 1: Bacterial Leaf Blight
        # Legacy class 2 ("Dead Heart") and legacy class 3 ("White Head") must both be suppressed
        tensor_cl8[0, 42, 0:4] = [200.0, 200.0, 100.0, 100.0]
        tensor_cl8[0, 42, 4 + 3] = 0.94
        tensor_cl8[0, 43, 0:4] = [100.0, 100.0, 64.0, 64.0]
        tensor_cl8[0, 43, 4 + 2] = 0.95
        decoded_cl8, _ = detector.decode_output(tensor_cl8, orig_width=640, orig_height=640)
        self.assertEqual(len(decoded_cl8), 1)
        self.assertEqual(decoded_cl8[0].label, "Bacterial Leaf Blight")
        self.assertAlmostEqual(decoded_cl8[0].confidence, 0.91, places=4)
        self.assertAlmostEqual(decoded_cl8[0].box[0], 256.0, places=1)
        self.assertAlmostEqual(decoded_cl8[0].box[2], 384.0, places=1)

        # 4. Test rejecting 80-class COCO [1, 84, 8400] without bogus blight detections
        tensor_coco = np.zeros((1, 84, 8400), dtype=np.float32)
        tensor_coco[0, 0:4, 5] = [0.5, 0.5, 0.4, 0.4]
        tensor_coco[0, 4 + 1, 5] = 0.99
        decoded_coco, summary_coco = detector.decode_output(tensor_coco, orig_width=640, orig_height=640)
        self.assertEqual(len(decoded_coco), 0)
        self.assertTrue(summary_coco.get("ignored_non_rice_model"))

        # 5. Test 2-class PyTorch ResNet-50 architecture instantiation
        resnet_2cls = RiceDiscolorationResNet50(num_classes=2, pretrained=False)
        sample_batch = torch.zeros((2, 3, 224, 224), dtype=torch.float32)
        logits_2cls = resnet_2cls(sample_batch)
        self.assertEqual(tuple(logits_2cls.shape), (2, 2))

        # 6. Test RiceCanopyUNet forward_logits preserves shape and probability bounds
        unet_net = RiceCanopyUNet()
        unet_in = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
        self.assertEqual(tuple(unet_net.forward_logits(unet_in).shape), (1, 1, 256, 256))

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
        self.assertEqual(
            set(blight_res["resnet_probabilities"].keys()),
            {"Healthy", "Bacterial Leaf Blight"},
        )
        self.assertGreater(
            blight_res["resnet_probabilities"]["Bacterial Leaf Blight"],
            0.5,
        )

        # Even if a legacy Whitehead image is supplied, Cloud never classifies it as White Head
        self.assertEqual(whitehead_res["status"], "COMPLETED")
        self.assertIn(whitehead_res["classification_label"], ("Healthy", "Bacterial Leaf Blight"))

    def test_edge_cases_missing_image_and_excluded_classes(self):
        # Missing image path should return graceful COMPLETED fallback for active classes
        missing_res = self.pipeline._run_real_models(
            "non_existent_path_12345.jpg",
            edge_detections=[{"label": "Bacterial Leaf Blight", "confidence": 0.87}],
        )
        self.assertEqual(missing_res["status"], "COMPLETED")
        self.assertEqual(missing_res["classification_label"], "Bacterial Leaf Blight")

        # Excluded legacy "Dead Heart" and "White Head" hints on missing image normalize to Healthy
        for legacy_label in ("Dead Heart", "White Head", "Whitehead"):
            excluded_res = self.pipeline._run_real_models(
                "non_existent_path_12345.jpg",
                edge_detections=[{"label": legacy_label, "confidence": 0.92}],
            )
            self.assertEqual(excluded_res["status"], "COMPLETED")
            self.assertEqual(excluded_res["classification_label"], "Healthy")

        # Synthetic discolored lesion image maps cleanly onto 2-class taxonomy and emits JPEG mask bytes
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            img_bgr = np.full((300, 300, 3), [40, 130, 40], dtype=np.uint8)
            img_bgr[80:240, 110:190] = [30, 65, 150]
            cv2.imwrite(tmp_path, img_bgr)

            res = self.pipeline.process_image(tmp_path)
            self.assertEqual(res["status"], "COMPLETED")
            self.assertEqual(res["classification_label"], "Bacterial Leaf Blight")
            self.assertGreater(res["severity_score"], 0.0)
            self.assertTrue(len(res["yolo_detections"]) > 0)
            self.assertTrue(isinstance(res.get("mask_bytes"), bytes) and len(res["mask_bytes"]) > 0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # Verify large (>= 1400px) image executes tiled YOLO path cleanly
        large_img = np.full((1500, 1500, 3), [40, 130, 40], dtype=np.uint8)
        dets_large, summary_large = self.pipeline.yolo_tflite_detector.detect_tiled_with_raw_summary(large_img)
        self.assertIsInstance(dets_large, list)
        self.assertIn("max_confidence", summary_large)

    def test_unicode_paths_bright_reflection_exclusion_and_zero_score_anchors(self):
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

        # 3. Bright white non-BLB reflection/patch on green canopy must NOT emit White Head
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            wh_path = tmp.name
        try:
            wh_small = np.full((128, 128, 3), [34, 139, 34], dtype=np.uint8)
            wh_small[30:90, 40:80] = [230, 235, 235]
            cv2.imwrite(wh_path, wh_small)
            wh_res = self.pipeline._run_real_models(wh_path)
            self.assertEqual(wh_res["status"], "COMPLETED")
            self.assertEqual(wh_res["classification_label"], "Healthy")
            self.assertEqual(wh_res["severity_score"], 0.0)
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
            # 1. Dict edge_detections + blank coordinate strings + RGBA PNG (legacy Whitehead hint -> Healthy)
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
            self.assertEqual(res1.classification_label, "Healthy")

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

    def test_soil_mask_and_healthy_fusion_guards(self):
        from unittest.mock import patch

        # 1. Sunlit dull pale clay soil (V < 225, S <= 28) is excluded as soil, not whitehead/lesion
        # while pure bright white panicle [230, 235, 235] stays in white_mask
        patch_bgr = np.full((120, 120, 3), [34, 139, 34], dtype=np.uint8)
        patch_bgr[10:50, 10:50] = [175, 180, 185]  # dry gray-tan mud (V=185, S=14)
        patch_bgr[60:100, 60:100] = [230, 235, 235]  # bright white lesion
        masks = CloudInferencePipeline._compute_hsv_masks(patch_bgr)
        self.assertTrue(bool(np.all(masks["soil_mask"][15:45, 15:45])))
        self.assertFalse(bool(np.any(masks["white_mask"][15:45, 15:45])))
        self.assertFalse(bool(np.any(masks["yellow_brown_mask"][15:45, 15:45])))
        self.assertTrue(bool(np.all(masks["white_mask"][65:95, 65:95])))

        # 2. Minor dry specks (1%..10% HSV) + strong ResNet Healthy must NOT fall back to BLB
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            speck_path = tmp.name
        try:
            speck_bgr = np.full((200, 200, 3), [34, 139, 34], dtype=np.uint8)
            speck_bgr[80:115, 80:115] = [35, 165, 215]  # ~3% minor yellow-brown speck
            cv2.imwrite(speck_path, speck_bgr)

            with patch.object(
                self.pipeline,
                "_run_resnet_with_rois",
                return_value=(np.array([0.92, 0.08], dtype=np.float32), "Healthy", 0.92),
            ), patch.object(
                self.pipeline,
                "unet_model",
                None,
            ), patch.object(
                self.pipeline,
                "yolo_tflite_detector",
                None,
            ):
                res_healthy = self.pipeline._run_real_models(speck_path)
            self.assertEqual(res_healthy["classification_label"], "Healthy")
            self.assertEqual(res_healthy["severity_score"], 0.0)
        finally:
            if os.path.exists(speck_path):
                os.remove(speck_path)

        # 3. Dull pale reflections must stay in 2-class taxonomy and never emit White Head
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            blb_pale_path = tmp.name
        try:
            blb_pale = np.full((200, 200, 3), [34, 139, 34], dtype=np.uint8)
            blb_pale[20:140, 20:140] = [148, 178, 178]  # dull pale straw/mud (S=43, V=178)
            cv2.imwrite(blb_pale_path, blb_pale)

            with patch.object(
                self.pipeline,
                "_run_resnet_with_rois",
                return_value=(np.array([0.35, 0.65], dtype=np.float32), "Bacterial Leaf Blight", 0.65),
            ):
                res_blb = self.pipeline._run_real_models(blb_pale_path)
            self.assertEqual(res_blb["classification_label"], "Bacterial Leaf Blight")
        finally:
            if os.path.exists(blb_pale_path):
                os.remove(blb_pale_path)

    def test_cloud_overrides_false_positive_edge_label_and_calibrates_confidence(self):
        import io
        import json
        from unittest.mock import patch
        from PIL import Image
        from django.test import override_settings
        from rest_framework.test import APIClient
        from core.models import DroneFlight, CloudInferenceResult, EdgeInferenceResult

        # 1. High-res aerial frame where Edge guessed "White Head" (no boxes)
        #    but ResNet-50 sees Healthy (0.85) & U-Net activation is low -> Cloud corrects to Healthy.
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            aerial_path = tmp.name
        try:
            aerial = np.full((800, 1000, 3), [34, 139, 34], dtype=np.uint8)
            aerial[100:260, 100:260] = [180, 195, 200]
            cv2.imwrite(aerial_path, aerial)

            with patch.object(
                self.pipeline,
                "_run_resnet_with_rois",
                return_value=(np.array([0.85, 0.15], dtype=np.float32), "Healthy", 0.85),
            ):
                res_override = self.pipeline._run_real_models(
                    aerial_path,
                    edge_detections=[{"label": "White Head", "confidence": 0.83}],
                )
            self.assertEqual(res_override["classification_label"], "Healthy")
            self.assertEqual(res_override["severity_score"], 0.0)
            self.assertAlmostEqual(res_override["confidence_score"], 0.85, places=2)

            # 2. When lesion pixels exist and ResNet-50 predicts BLB (0.52) against an edge White Head
            #    hint (no boxes), Cloud chooses BLB with calibrated ~0.52 confidence (not >= 0.78 floor).
            blb_aerial = np.full((800, 1000, 3), [34, 139, 34], dtype=np.uint8)
            blb_aerial[120:420, 180:520] = [35, 165, 215]  # ~12.8% necrotic yellow-brown lesion
            cv2.imwrite(aerial_path, blb_aerial)
            with patch.object(
                self.pipeline,
                "_run_resnet_with_rois",
                return_value=(np.array([0.48, 0.52], dtype=np.float32), "Bacterial Leaf Blight", 0.52),
            ), patch.object(
                self.pipeline,
                "yolo_tflite_detector",
                None,
            ):
                res_blb = self.pipeline._run_real_models(
                    aerial_path,
                    edge_detections=[{"label": "White Head", "confidence": 0.83}],
                )
            self.assertEqual(res_blb["classification_label"], "Bacterial Leaf Blight")
            self.assertGreater(res_blb["severity_score"], 0.0)
            self.assertAlmostEqual(res_blb["confidence_score"], 0.52, places=2)
        finally:
            if os.path.exists(aerial_path):
                os.remove(aerial_path)

        # 3. Verify EdgeInferenceResult keeps original Edge label even if Cloud overrides to Healthy
        client = APIClient()
        flight = DroneFlight.objects.create(title="Edge Preservation Flight")
        buf = io.BytesIO()
        Image.new("RGB", (640, 480), color=(34, 139, 34)).save(buf, "JPEG")
        buf.name = "healthy_field.jpg"
        buf.seek(0)

        with override_settings(CELERY_TASK_ALWAYS_EAGER=True):
            resp = client.post(
                "/api/images/upload/",
                {
                    "flight_id": flight.id,
                    "file": buf,
                    "edge_detections": json.dumps(
                        {"label": "White Head", "confidence": 0.83, "boxes": []}
                    ),
                },
                format="multipart",
            )
        self.assertEqual(resp.status_code, 202)
        img_id = resp.json()["image_id"]
        edge_rec = EdgeInferenceResult.objects.get(image_id=img_id)
        cloud_rec = CloudInferenceResult.objects.get(image_id=img_id)
        self.assertEqual(edge_rec.classification_label, "White Head")
        self.assertAlmostEqual(edge_rec.confidence_score, 0.83, places=2)
        self.assertEqual(cloud_rec.classification_label, "Healthy")



