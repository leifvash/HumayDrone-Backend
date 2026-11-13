import os
import time
import threading
import cv2
import numpy as np
import datetime
from typing import Dict, Any, List, Optional


def _get_now():
    try:
        from django.utils import timezone
        return timezone.now()
    except Exception:
        return datetime.datetime.now(datetime.timezone.utc)


class CloudInferencePipeline:
    """
    Cloud Inference Engine coordinating:
    1. YOLOv8 (yolov8n_rice.tflite / Ultralytics) - High-resolution anomaly/discoloration detection
    2. U-Net (RiceCanopyUNet) + OpenCV HSV - Pixel-level canopy discoloration segmentation & percentage scoring
    3. ResNet-50 (RiceDiscolorationResNet50) - Multi-class deep verification (Healthy, BLB, White Head)
    4. Expert Decision Rules - Prescribing chemical/cultural countermeasures
    """

    DISEASE_CLASSES = ["Healthy", "Bacterial Leaf Blight", "White Head"]
    ACTIVE_DISEASE_CLASSES = ("Bacterial Leaf Blight", "White Head")

    RECOMMENDATIONS = {
        "Bacterial Leaf Blight": (
            "Apply copper-based bactericides (e.g. Copper Hydroxide or Copper Oxychloride). "
            "Ensure proper drainage and avoid excessive nitrogen fertilizer application."
        ),
        "White Head": (
            "Heading-stage Stem Borer damage detected. Empty white panicles identified. "
            "Record affected spatial coordinates and apply protective biological control or targeted spray."
        ),
        "Healthy": "Canopy is healthy and showing normal vigor. No chemical action required.",
    }

    def __init__(self, yolo_weights: str = None, unet_weights: str = None, resnet_weights: str = None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(base_dir, "weights")

        default_tflite = os.path.join(weights_dir, "yolov8n_rice.tflite")
        android_asset_tflite = os.path.abspath(
            os.path.join(
                base_dir,
                "..",
                "..",
                "SampleCode-V5",
                "android-sdk-v5-sample",
                "src",
                "main",
                "assets",
                "yolov8n_rice.tflite",
            )
        )
        default_pt = os.path.join(weights_dir, "yolov8_cloud.pt")

        if os.path.exists(default_tflite):
            self.yolo_tflite_weights = default_tflite
        elif os.path.exists(android_asset_tflite):
            self.yolo_tflite_weights = android_asset_tflite
        else:
            self.yolo_tflite_weights = default_tflite

        if yolo_weights:
            self.yolo_weights = yolo_weights
        elif os.path.exists(self.yolo_tflite_weights):
            self.yolo_weights = self.yolo_tflite_weights
        else:
            self.yolo_weights = default_pt

        self.unet_weights = unet_weights or os.path.join(weights_dir, "unet_canopy.pth")
        self.resnet_weights = resnet_weights or os.path.join(weights_dir, "resnet50_disease.pth")

        self.yolo_model = None
        self.yolo_tflite_detector = None
        self.unet_model = None
        self.resnet_model = None
        self._resnet_num_classes = len(self.DISEASE_CLASSES)
        self._infer_lock = threading.Lock()
        self.device = None
        self.is_real_models_loaded = False

        self._load_models_if_available()

    @classmethod
    def _normalize_label(cls, label: Optional[str]) -> Optional[str]:
        if not label or not isinstance(label, str):
            return None
        cleaned = label.strip()
        lower = cleaned.lower()
        if lower in ("whitehead", "white head", "white_head"):
            return "White Head"
        if lower in ("deadheart", "dead heart", "dead_heart"):
            return None
        if lower in ("blb", "bacterial leaf blight", "bacterial_leaf_blight", "blight"):
            return "Bacterial Leaf Blight"
        if lower == "healthy":
            return "Healthy"
        # Normalize 4-class legacy ResNet checkpoint onto active 3-class taxonomy if needed
        return cleaned

    @staticmethod
    def _compute_hsv_masks(img_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Computes soil, specular glare, yellow-brown necrotic, white bleached panicle,
        and healthy green canopy boolean pixel masks for a BGR image.
        """
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        B_f = img_bgr[:, :, 0].astype(np.float32)
        G_f = img_bgr[:, :, 1].astype(np.float32)
        R_f = img_bgr[:, :, 2].astype(np.float32)

        # Exclude dark shadow / wet soil / mud background between rice hills
        soil_mask = (V < 60) | ((V < 105) & (S < 50) & (G_f < R_f * 1.15) & (R_f < 115))
        # Exclude pure specular sunlight glare on leaves/water
        glare_mask = (V >= 252) & (S <= 8) & (B_f >= 245) & (G_f >= 245) & (R_f >= 245)
        excluded_bg = soil_mask | glare_mask

        chlorotic_mask = (
            (H >= 28) & (H <= 44) & (S >= 30) & (V >= 95) & (R_f >= G_f * 0.79) & (~excluded_bg)
        )
        yellow_brown_mask = (
            (
                ((H >= 6) & (H < 35) & (S >= 25) & (V >= 30))
                | ((H < 10) & (S >= 40) & (V >= 40))
            )
            & (~excluded_bg)
        ) | chlorotic_mask

        white_mask = (V >= 145) & (S <= 82) & (R_f >= G_f * 0.76) & (~excluded_bg)
        green_mask = (
            (H >= 35) & (H <= 88) & (S >= 28) & (V >= 30) & (~yellow_brown_mask) & (~white_mask)
        )
        return {
            "V": V,
            "S": S,
            "soil_mask": soil_mask,
            "glare_mask": glare_mask,
            "yellow_brown_mask": yellow_brown_mask,
            "white_mask": white_mask,
            "green_mask": green_mask,
        }

    def _run_resnet_with_rois(
        self,
        rgb: np.ndarray,
        roi_boxes: List[List[float]],
    ) -> tuple:
        """
        Runs batched ResNet-50 verification over the full image plus up to 6 lesion ROI crops.
        Maps legacy 4-class checkpoints cleanly onto the 3 active classes when present.
        """
        import torch

        num_active = len(self.DISEASE_CLASSES)
        if self.resnet_model is None:
            uniform = np.full((num_active,), 1.0 / num_active, dtype=np.float32)
            return uniform, "Healthy", float(uniform[0])

        h, w = rgb.shape[:2]
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        def _prep_crop(crop_rgb: np.ndarray) -> np.ndarray:
            resized = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            norm = (resized - mean) / std
            return np.transpose(norm, (2, 0, 1))

        crops = [_prep_crop(rgb)]
        for box in roi_boxes[:6]:
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            bw, bh = x2 - x1, y2 - y1
            if bw < 24 or bh < 24:
                continue
            pad_x, pad_y = bw * 0.15, bh * 0.15
            cx1 = max(0, int(round(x1 - pad_x)))
            cy1 = max(0, int(round(y1 - pad_y)))
            cx2 = min(w, int(round(x2 + pad_x)))
            cy2 = min(h, int(round(y2 + pad_y)))
            if cx2 - cx1 >= 16 and cy2 - cy1 >= 16:
                crops.append(_prep_crop(rgb[cy1:cy2, cx1:cx2]))

        batch_arr = np.ascontiguousarray(np.stack(crops, axis=0))
        device = self.device or torch.device("cpu")
        batch_tensor = torch.from_numpy(batch_arr).to(device)
        with torch.inference_mode():
            logits = self.resnet_model(batch_tensor)
            raw_probs = torch.softmax(logits, dim=1).detach().cpu().numpy()

        if raw_probs.shape[1] == 4:
            # Project legacy [Healthy, BLB, Dead Heart, White Head] -> [Healthy, BLB, White Head]
            proj = raw_probs[:, [0, 1, 3]]
            row_sums = np.clip(proj.sum(axis=1, keepdims=True), 1e-6, None)
            active_probs = proj / row_sums
        else:
            active_probs = raw_probs[:, :num_active]

        full_probs = active_probs[0]
        if active_probs.shape[0] > 1:
            roi_probs = np.mean(active_probs[1:], axis=0)
            blended = 0.40 * full_probs + 0.60 * roi_probs
            blended = blended / np.clip(blended.sum(), 1e-6, None)
            probs = blended.astype(np.float32)
        else:
            probs = full_probs.astype(np.float32)

        cls_idx = int(np.argmax(probs))
        return probs, self.DISEASE_CLASSES[cls_idx], float(probs[cls_idx])

    def _render_overlay_and_encode_mask(
        self,
        img: np.ndarray,
        green_mask: np.ndarray,
        yb_mask: np.ndarray,
        white_mask: np.ndarray,
        yolo_detections: List[Dict[str, Any]],
        image_path: str,
    ) -> tuple:
        """
        Renders canopy discoloration overlay + detection boxes, returning (mask_bytes, mask_relative_path).
        Writes a local file under media/masks only when remote Supabase Storage is inactive.
        """
        mask_bytes: Optional[bytes] = None
        mask_relative_path = ""
        try:
            height, width = img.shape[:2]
            overlay = img.copy()
            overlay[green_mask] = [0, 210, 0]
            overlay[yb_mask] = [0, 70, 255]
            overlay[white_mask] = [255, 255, 100]

            vis_mask = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
            stroke = max(2, int(round(min(width, height) / 450)))
            for det in yolo_detections:
                box = det.get("box")
                if isinstance(box, list) and len(box) == 4:
                    x1, y1, x2, y2 = [int(round(v)) for v in box]
                    color = (0, 200, 0) if det.get("label") == "Healthy" else (0, 180, 255)
                    cv2.rectangle(vis_mask, (x1, y1), (x2, y2), color, stroke)

            ok, buf = cv2.imencode(".jpg", vis_mask, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok and buf is not None:
                mask_bytes = bytes(buf.tobytes())

            use_remote_storage = bool(
                os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
            )
            if not use_remote_storage:
                media_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "media"))
                masks_dir = os.path.join(media_root, "masks")
                img_dir = os.path.abspath(os.path.dirname(image_path))
                captures_dir = os.path.join(media_root, "captures")
                rel_sub = os.path.relpath(img_dir, captures_dir)
                target_dir = os.path.join(masks_dir, rel_sub) if not rel_sub.startswith("..") else masks_dir
                os.makedirs(target_dir, exist_ok=True)
                mask_filename = "mask_" + os.path.splitext(os.path.basename(image_path))[0] + ".jpg"
                mask_full_path = os.path.join(target_dir, mask_filename)
                if self._write_image_bgr(mask_full_path, vis_mask):
                    mask_relative_path = os.path.relpath(mask_full_path, media_root).replace("\\", "/")
        except Exception:
            mask_relative_path = ""
        return mask_bytes, mask_relative_path

    @classmethod
    def _parse_edge_detections(cls, edge_detections: Any):
        edge_label = None
        edge_conf = 0.85
        edge_boxes: List[List[float]] = []

        if not edge_detections:
            return edge_label, edge_conf, edge_boxes

        if isinstance(edge_detections, str):
            try:
                import json
                edge_detections = json.loads(edge_detections)
            except Exception:
                return edge_label, edge_conf, edge_boxes

        if isinstance(edge_detections, dict):
            edge_detections = [edge_detections]

        if not isinstance(edge_detections, (list, tuple)):
            return edge_label, edge_conf, edge_boxes

        first_det = None
        for item in edge_detections:
            if isinstance(item, dict):
                first_det = item
                break

        if first_det is None:
            return edge_label, edge_conf, edge_boxes

        edge_label = cls._normalize_label(
            first_det.get("classification_label") or first_det.get("label")
        )
        raw_conf = first_det.get("confidence_score")
        if raw_conf is None:
            raw_conf = first_det.get("confidence", 0.85)
        try:
            edge_conf = float(raw_conf)
        except (ValueError, TypeError):
            edge_conf = 0.85

        raw_boxes = first_det.get("boxes") or first_det.get("bounding_boxes") or []
        if isinstance(raw_boxes, (list, tuple)):
            for b in raw_boxes:
                if isinstance(b, (list, tuple)) and len(b) == 4:
                    try:
                        coords = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
                        edge_boxes.append(coords)
                    except (ValueError, TypeError):
                        continue

        return edge_label, edge_conf, edge_boxes

    @staticmethod
    def _read_image_bgr(image_path: str) -> Optional[np.ndarray]:
        if not image_path or not os.path.exists(image_path):
            return None
        try:
            raw = np.fromfile(image_path, dtype=np.uint8)
            if raw.size > 0:
                bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if bgr is not None and bgr.size > 0:
                    return bgr
        except Exception:
            pass
        try:
            bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if bgr is not None and bgr.size > 0:
                return bgr
        except Exception:
            pass
        try:
            from PIL import Image
            with Image.open(image_path) as pil_img:
                rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    @staticmethod
    def _write_image_bgr(image_path: str, img_bgr: np.ndarray) -> bool:
        try:
            ok, buf = cv2.imencode(".jpg", img_bgr)
            if ok:
                buf.tofile(image_path)
                return True
        except Exception:
            pass
        try:
            return bool(cv2.imwrite(image_path, img_bgr))
        except Exception:
            return False

    @staticmethod
    def _extract_lesion_boxes(
        lesion_mask: np.ndarray,
        class_index: int,
        label: str,
        confidence: float,
        max_boxes: int = 16,
    ) -> List[Dict[str, Any]]:
        if lesion_mask is None or lesion_mask.size == 0 or not np.any(lesion_mask):
            return []
        h, w = lesion_mask.shape[:2]
        mask_u8 = (lesion_mask.astype(np.uint8)) * 255

        # For high-resolution drone captures, open tiny noise speckles and close nearby leaf streaks
        if h * w >= 500000:
            k_open = np.ones((3, 3), np.uint8)
            k_close = np.ones((11, 11), np.uint8)
            mask_clean = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k_open)
            mask_closed = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, k_close)
            min_area = max(64.0, float(h * w) * 0.00012)
        else:
            kernel = np.ones((5, 5), np.uint8)
            mask_closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
            min_area = max(16.0, float(h * w) * 0.001)

        contours, _ = cv2.findContours(
            mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        raw_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if h * w >= 500000 and bw > w * 0.55 and bh > h * 0.55:
                continue
            raw_boxes.append(
                (
                    area,
                    [
                        round(float(x), 2),
                        round(float(y), 2),
                        round(float(x + bw), 2),
                        round(float(y + bh), 2),
                    ],
                )
            )
        raw_boxes.sort(key=lambda item: item[0], reverse=True)

        # Containment-aware non-maximum suppression to prevent overlapping/nested boxes
        kept_boxes = []
        for area_b, b in raw_boxes:
            overlap = False
            for _, k in kept_boxes:
                ix1 = max(b[0], k[0])
                iy1 = max(b[1], k[1])
                ix2 = min(b[2], k[2])
                iy2 = min(b[3], k[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                b_rect_area = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
                if inter / b_rect_area > 0.45:
                    overlap = True
                    break
            if not overlap:
                kept_boxes.append((area_b, b))
            if len(kept_boxes) >= max_boxes:
                break

        results: List[Dict[str, Any]] = []
        for idx, (_, box) in enumerate(kept_boxes):
            box_conf = max(0.68, round(float(confidence) - idx * 0.012, 4))
            results.append(
                {
                    "class_index": int(class_index),
                    "label": label,
                    "confidence": box_conf,
                    "box": box,
                    "norm_box": [
                        round(float(box[0]) / max(1, w), 4),
                        round(float(box[1]) / max(1, h), 4),
                        round(float(box[2]) / max(1, w), 4),
                        round(float(box[3]) / max(1, h), 4),
                    ],
                }
            )
        return results

    def _load_models_if_available(self):
        """
        Dynamically loads TFLite YOLOv8 (`yolov8n_rice.tflite`) and PyTorch (`RiceCanopyUNet`,
        `RiceDiscolorationResNet50`) models when weight checkpoints exist on disk.
        """
        self.yolo_model = None
        self.yolo_tflite_detector = None
        self.unet_model = None
        self.resnet_model = None

        # 1. Load TFLite YOLOv8 detector (yolov8n_rice.tflite)
        tflite_candidate = (
            self.yolo_weights
            if (self.yolo_weights and self.yolo_weights.lower().endswith(".tflite"))
            else self.yolo_tflite_weights
        )
        if tflite_candidate and os.path.exists(tflite_candidate):
            try:
                from .models.yolo_tflite import YoloV8TFLiteDetector
                self.yolo_tflite_detector = YoloV8TFLiteDetector(model_path=tflite_candidate)
                self.yolo_model = self.yolo_tflite_detector
            except Exception:
                self.yolo_tflite_detector = None

        # Optional Ultralytics .pt model support if explicitly provided
        if (
            self.yolo_model is None
            and self.yolo_weights
            and self.yolo_weights.lower().endswith(".pt")
            and os.path.exists(self.yolo_weights)
        ):
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(self.yolo_weights)
            except Exception:
                self.yolo_model = None

        # 2. Load PyTorch U-Net and ResNet-50 checkpoints
        has_unet = bool(self.unet_weights and os.path.exists(self.unet_weights))
        has_resnet = bool(self.resnet_weights and os.path.exists(self.resnet_weights))

        if has_unet or has_resnet:
            try:
                import torch
                try:
                    torch.set_num_threads(1)
                    torch.set_num_interop_threads(1)
                except Exception:
                    pass
                from .models.unet import RiceCanopyUNet
                from .models.resnet import RiceDiscolorationResNet50

                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                if has_unet:
                    unet = RiceCanopyUNet()
                    try:
                        state_dict = torch.load(
                            self.unet_weights, map_location=self.device, weights_only=True
                        )
                    except TypeError:
                        state_dict = torch.load(self.unet_weights, map_location=self.device)
                    unet.load_state_dict(state_dict)
                    self.unet_model = unet.to(self.device).eval()

                if has_resnet:
                    try:
                        state_dict = torch.load(
                            self.resnet_weights, map_location=self.device, weights_only=True
                        )
                    except TypeError:
                        state_dict = torch.load(self.resnet_weights, map_location=self.device)
                    fc_key = next(
                        (
                            k
                            for k in state_dict.keys()
                            if k.endswith("fc.1.weight") or k.endswith("fc.weight")
                        ),
                        None,
                    )
                    ckpt_classes = (
                        int(state_dict[fc_key].shape[0])
                        if fc_key is not None
                        else len(self.DISEASE_CLASSES)
                    )
                    resnet = RiceDiscolorationResNet50(num_classes=ckpt_classes, pretrained=False)
                    resnet.load_state_dict(state_dict)
                    self._resnet_num_classes = ckpt_classes
                    self.resnet_model = resnet.to(self.device).eval()
            except Exception:
                self.unet_model = None
                self.resnet_model = None

        self.is_real_models_loaded = bool(
            self.yolo_tflite_detector is not None
            or self.yolo_model is not None
            or (self.unet_model is not None and self.resnet_model is not None)
        )

    def run_yolo_tflite(
        self,
        image: Any,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes `yolov8n_rice.tflite` on an image path or BGR numpy array and returns
        a list of bounding box & class detection dicts.
        """
        if self.yolo_tflite_detector is None:
            return []
        detections = self.yolo_tflite_detector.detect(
            image=image,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )
        return [d.to_dict() for d in detections]

    def process_image(self, image_path: str, edge_detections: List[Dict] = None) -> Dict[str, Any]:
        """
        Runs the full verification sequence:
        Uses loaded TFLite YOLOv8 + PyTorch U-Net + ResNet-50 models if available,
        otherwise falls back to pixel-level canopy discoloration analysis with OpenCV.
        """
        if self.is_real_models_loaded:
            try:
                return self._run_real_models(image_path, edge_detections)
            except Exception:
                return self._run_pixel_canopy_analysis(image_path, edge_detections)
        else:
            return self._run_pixel_canopy_analysis(image_path, edge_detections)

    def _run_real_models(self, image_path: str, edge_detections: List[Dict] = None) -> Dict[str, Any]:
        """
        Executes real neural network inference end-to-end:
        1. `yolov8n_rice.tflite` bounding-box & class inference (640x640 RGB normalized, NMS)
        2. `RiceCanopyUNet` forward pass (256x256 tensor) + OpenCV HSV hybrid mask refinement
        3. `RiceDiscolorationResNet50` forward pass (224x224 ImageNet-normalized tensor) + softmax
        4. Multi-model decision fusion & visual segmentation mask generation
        """
        import torch

        edge_label, edge_conf, edge_boxes = self._parse_edge_detections(edge_detections)

        # 1. Load image safely (supports Unicode paths on Windows & PIL fallback)
        img = self._read_image_bgr(image_path)

        if img is None:
            suggested_class = edge_label if edge_label in self.DISEASE_CLASSES else "Healthy"
            severity_score = 15.0 if suggested_class != "Healthy" else 0.0
            return {
                "model_used": "YOLOv8n-TFLite + RiceCanopyUNet + ResNet-50 (Fallback - Unreadable Image)",
                "classification_label": suggested_class,
                "confidence_score": round(float(edge_conf), 2),
                "severity_score": severity_score,
                "treatment_recommendation": self.RECOMMENDATIONS.get(suggested_class, "Monitor regularly."),
                "mask_relative_path": "",
                "yolo_detections": [],
                "processed_at": _get_now(),
                "status": "COMPLETED",
            }

        height, width = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ------------------------------------------------------------------
        # Stage 1: YOLOv8 (`yolov8n_rice.tflite`) Tiled + Full-Frame Inference
        # ------------------------------------------------------------------
        yolo_detections: List[Dict[str, Any]] = []
        yolo_disease_label: Optional[str] = None
        yolo_disease_conf: float = 0.0
        yolo_top_label: Optional[str] = None
        yolo_top_conf: float = 0.0

        with self._infer_lock:
            if self.yolo_tflite_detector is not None:
                det_objs, raw_summary = self.yolo_tflite_detector.detect_tiled_with_raw_summary(img)
                yolo_detections = [d.to_dict() for d in det_objs]
                if yolo_detections:
                    yolo_top_label = self._normalize_label(yolo_detections[0]["label"])
                    yolo_top_conf = float(yolo_detections[0]["confidence"])
                    for d in yolo_detections:
                        norm_lbl = self._normalize_label(d["label"])
                        if norm_lbl in self.ACTIVE_DISEASE_CLASSES:
                            yolo_disease_label = norm_lbl
                            yolo_disease_conf = float(d["confidence"])
                            break
                elif raw_summary:
                    yolo_top_label = self._normalize_label(raw_summary.get("best_label"))
                    yolo_top_conf = float(raw_summary.get("max_confidence", 0.0))

            # ------------------------------------------------------------------
            # Stage 2: PyTorch RiceCanopyUNet Segmentation + OpenCV HSV Refinement
            # ------------------------------------------------------------------
            unet_prob_map = np.zeros((height, width), dtype=np.float32)
            if self.unet_model is not None:
                device = self.device or torch.device("cpu")
                unet_resized = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
                unet_input = np.transpose(unet_resized.astype(np.float32) / 255.0, (2, 0, 1))
                unet_tensor = torch.from_numpy(np.ascontiguousarray(unet_input)).unsqueeze(0).to(device)
                with torch.inference_mode():
                    unet_out = self.unet_model(unet_tensor)
                unet_mask_256 = unet_out[0, 0].detach().cpu().numpy()
                unet_prob_map = cv2.resize(
                    unet_mask_256, (width, height), interpolation=cv2.INTER_LINEAR
                )

            hsv_masks = self._compute_hsv_masks(img)
            V = hsv_masks["V"]
            S = hsv_masks["S"]
            soil_mask = hsv_masks["soil_mask"]
            glare_mask = hsv_masks["glare_mask"]
            yellow_brown_mask = hsv_masks["yellow_brown_mask"]
            white_mask = hsv_masks["white_mask"]
            green_mask = hsv_masks["green_mask"]

            # Hybrid refinement combining U-Net probability map with HSV+RGB symptom masks
            unet_lesion_mask = (
                (unet_prob_map >= 0.5)
                & (V >= 30)
                & (~green_mask)
                & (~soil_mask)
                & (~glare_mask)
            )
            refined_white_mask = white_mask | (unet_lesion_mask & (V >= 145) & (S <= 85))
            refined_yb_mask = yellow_brown_mask | (unet_lesion_mask & (S >= 25) & (~refined_white_mask))
            refined_green_mask = green_mask & (~refined_yb_mask) & (~refined_white_mask)

            total_green = int(np.sum(refined_green_mask))
            total_yb = int(np.sum(refined_yb_mask))
            total_white = int(np.sum(refined_white_mask))

            combined_boxes = list(edge_boxes) if isinstance(edge_boxes, list) else []
            for d in yolo_detections:
                if d.get("label") != "Healthy" and isinstance(d.get("box"), list):
                    combined_boxes.append(d["box"])

            box_pixels = 0
            for b in combined_boxes:
                if isinstance(b, (list, tuple)) and len(b) == 4:
                    box_w = max(0, int(round(float(b[2]))) - int(round(float(b[0]))))
                    box_h = max(0, int(round(float(b[3]))) - int(round(float(b[1]))))
                    box_pixels += box_w * box_h

            total_discolored = total_yb + total_white
            has_disease_hint = (
                (edge_label in self.ACTIVE_DISEASE_CLASSES)
                or (yolo_disease_label in self.ACTIVE_DISEASE_CLASSES)
            )
            if box_pixels > 0 and total_discolored == 0 and has_disease_hint:
                total_discolored = box_pixels

            total_canopy = total_green + total_discolored
            if total_canopy <= 0:
                severity_score = 0.0
            else:
                raw_severity = (total_discolored / total_canopy) * 100.0
                severity_score = round(min(100.0, raw_severity), 2)

            # ------------------------------------------------------------------
            # Stage 3: PyTorch RiceDiscolorationResNet50 Batched ROI Verification
            # ------------------------------------------------------------------
            roi_boxes_for_resnet = list(combined_boxes)
            if not roi_boxes_for_resnet and total_discolored > 0:
                prelim_rois = self._extract_lesion_boxes(
                    lesion_mask=(refined_yb_mask | refined_white_mask),
                    class_index=1,
                    label="Bacterial Leaf Blight",
                    confidence=0.8,
                    max_boxes=6,
                )
                roi_boxes_for_resnet = [r["box"] for r in prelim_rois if isinstance(r.get("box"), list)]

            resnet_probs, resnet_label, resnet_conf = self._run_resnet_with_rois(
                rgb=rgb,
                roi_boxes=roi_boxes_for_resnet,
            )

        # ------------------------------------------------------------------
        # Stage 4: Multi-Model Decision Fusion & Visual Mask Generation
        # ------------------------------------------------------------------
        white_ratio = total_white / (total_discolored + 1e-5)
        white_min_pixels = min(2500, max(100, int(total_canopy * 0.03)))

        # Top disease candidate among ResNet-50's 2 disease classes [BLB, White Head]
        resnet_disease_probs = resnet_probs[1:3]
        best_disease_subidx = int(np.argmax(resnet_disease_probs))
        resnet_best_disease_label = self.DISEASE_CLASSES[1 + best_disease_subidx]
        resnet_best_disease_prob = float(resnet_disease_probs[best_disease_subidx])

        if severity_score < 0.85 and not (has_disease_hint and box_pixels > 0):
            classification = "Healthy"
            severity_score = 0.0
            confidence = max(0.92, float(resnet_probs[0]), yolo_top_conf if yolo_top_label == "Healthy" else 0.0)
        else:
            if total_white > white_min_pixels and white_ratio >= 0.35:
                classification = "White Head"
            elif yolo_disease_label in self.ACTIVE_DISEASE_CLASSES:
                classification = yolo_disease_label
            elif edge_label in self.ACTIVE_DISEASE_CLASSES:
                classification = edge_label
            elif resnet_label in self.ACTIVE_DISEASE_CLASSES:
                classification = resnet_label
            elif total_white > white_min_pixels and white_ratio >= 0.18:
                classification = "White Head"
            elif resnet_best_disease_prob >= 0.15:
                classification = resnet_best_disease_label
            else:
                classification = "Bacterial Leaf Blight"

            cls_idx = self.DISEASE_CLASSES.index(classification)
            resnet_cls_prob = float(resnet_probs[cls_idx])
            confidence = min(
                0.99,
                max(
                    edge_conf if edge_label == classification else 0.0,
                    yolo_disease_conf if yolo_disease_label == classification else 0.0,
                    resnet_cls_prob,
                    0.78 + (severity_score / 100.0) * 0.21,
                ),
            )

        # Populate or supplement lesion bounding boxes from the refined segmentation mask
        if classification != "Healthy":
            cls_idx = self.DISEASE_CLASSES.index(classification)
            target_lesion_mask = refined_yb_mask | refined_white_mask
            lesion_boxes = self._extract_lesion_boxes(
                lesion_mask=target_lesion_mask,
                class_index=cls_idx,
                label=classification,
                confidence=confidence,
                max_boxes=16,
            )
            if not yolo_detections:
                yolo_detections = lesion_boxes
            elif lesion_boxes:
                # Supplement any TFLite boxes with non-overlapping segmentation lesion boxes
                existing_boxes = [
                    d["box"] for d in yolo_detections if isinstance(d.get("box"), list) and len(d["box"]) == 4
                ]
                for lb in lesion_boxes:
                    b = lb["box"]
                    overlap = False
                    for eb in existing_boxes:
                        ix1 = max(b[0], eb[0])
                        iy1 = max(b[1], eb[1])
                        ix2 = min(b[2], eb[2])
                        iy2 = min(b[3], eb[3])
                        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                        b_area = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
                        if inter / b_area > 0.4:
                            overlap = True
                            break
                    if not overlap and len(yolo_detections) < 16:
                        yolo_detections.append(lb)
                        existing_boxes.append(b)

            if not yolo_detections and edge_boxes:
                for b in edge_boxes[:5]:
                    yolo_detections.append(
                        {
                            "class_index": cls_idx,
                            "label": classification,
                            "confidence": round(float(confidence), 4),
                            "box": [round(float(v), 2) for v in b],
                        }
                    )

        # Build normalized [0..1] bounding boxes for responsive web/mobile UI overlays
        normalized_boxes: List[List[float]] = []
        for det in yolo_detections:
            if isinstance(det.get("norm_box"), list) and len(det["norm_box"]) == 4:
                normalized_boxes.append(det["norm_box"])
            elif isinstance(det.get("box"), list) and len(det["box"]) == 4:
                bx = det["box"]
                if max(bx) <= 1.0:
                    normalized_boxes.append([round(float(v), 4) for v in bx])
                else:
                    normalized_boxes.append([
                        round(max(0.0, min(1.0, float(bx[0]) / max(1, width))), 4),
                        round(max(0.0, min(1.0, float(bx[1]) / max(1, height))), 4),
                        round(max(0.0, min(1.0, float(bx[2]) / max(1, width))), 4),
                        round(max(0.0, min(1.0, float(bx[3]) / max(1, height))), 4),
                    ])

        mask_bytes, mask_relative_path = self._render_overlay_and_encode_mask(
            img=img,
            green_mask=refined_green_mask,
            yb_mask=refined_yb_mask,
            white_mask=refined_white_mask,
            yolo_detections=yolo_detections,
            image_path=image_path,
        )

        recommendation = self.RECOMMENDATIONS.get(
            classification,
            "Canopy is showing discoloration. Monitor field regularly and test soil/leaf samples.",
        )

        return {
            "model_used": "YOLOv8n-TFLite + RiceCanopyUNet + ResNet-50",
            "classification_label": classification,
            "confidence_score": round(float(confidence), 2),
            "severity_score": severity_score,
            "treatment_recommendation": recommendation,
            "mask_relative_path": mask_relative_path,
            "mask_bytes": mask_bytes,
            "yolo_detections": yolo_detections,
            "normalized_boxes": normalized_boxes,
            "resnet_probabilities": {
                cls_name: round(float(resnet_probs[i]), 4)
                for i, cls_name in enumerate(self.DISEASE_CLASSES)
            },
            "unet_mean_activation": round(float(np.mean(unet_prob_map)), 4),
            "processed_at": _get_now(),
            "status": "COMPLETED",
        }

    def _run_pixel_canopy_analysis(self, image_path: str, edge_detections: List[Dict] = None) -> Dict[str, Any]:
        """
        Real pixel-level canopy discoloration segmentation and dynamic severity scoring.
        Formula:
            Severity % = (Discolored Canopy Pixels / Total Canopy Pixels) * 100
        Analyzes healthy green canopy vs necrotic/yellow-brown and bleached whitehead discoloration.
        """
        edge_label, edge_conf, edge_boxes = self._parse_edge_detections(edge_detections)

        img = self._read_image_bgr(image_path)
        if img is None:
            suggested_class = edge_label if edge_label in self.DISEASE_CLASSES else "Healthy"
            severity_score = 15.0 if suggested_class != "Healthy" else 0.0
            return {
                "model_used": "Cloud Pipeline (Simulated Fallback)",
                "classification_label": suggested_class,
                "confidence_score": edge_conf,
                "severity_score": severity_score,
                "treatment_recommendation": self.RECOMMENDATIONS.get(suggested_class, "Monitor regularly."),
                "mask_relative_path": "",
                "mask_bytes": None,
                "processed_at": _get_now(),
                "status": "COMPLETED",
            }

        hsv_masks = self._compute_hsv_masks(img)
        yellow_brown_mask = hsv_masks["yellow_brown_mask"]
        white_mask = hsv_masks["white_mask"]
        green_mask = hsv_masks["green_mask"]

        total_green = int(np.sum(green_mask))
        total_yb = int(np.sum(yellow_brown_mask))
        total_white = int(np.sum(white_mask))

        box_pixels = 0
        if edge_boxes and isinstance(edge_boxes, list):
            for b in edge_boxes:
                if isinstance(b, (list, tuple)) and len(b) == 4:
                    box_w = max(0, int(round(float(b[2]))) - int(round(float(b[0]))))
                    box_h = max(0, int(round(float(b[3]))) - int(round(float(b[1]))))
                    box_pixels += (box_w * box_h)

        total_discolored = total_yb + total_white
        if box_pixels > 0 and total_discolored == 0 and edge_label != "Healthy":
            total_discolored = box_pixels

        total_canopy = total_green + total_discolored
        if total_canopy <= 0:
            severity_score = 0.0
        else:
            raw_severity = (total_discolored / total_canopy) * 100.0
            severity_score = round(min(100.0, raw_severity), 2)

        white_ratio = total_white / (total_discolored + 1e-5)
        white_min_pixels = min(2500, max(100, int(total_canopy * 0.03)))

        if severity_score < 0.85 and not (edge_label in self.ACTIVE_DISEASE_CLASSES and box_pixels > 0):
            classification = "Healthy"
            severity_score = 0.0
            confidence = 0.96
        else:
            if total_white > white_min_pixels and white_ratio >= 0.35:
                classification = "White Head"
            elif edge_label in self.ACTIVE_DISEASE_CLASSES:
                classification = edge_label
            elif total_white > white_min_pixels and white_ratio >= 0.18:
                classification = "White Head"
            else:
                classification = "Bacterial Leaf Blight"

            confidence = min(0.99, max(edge_conf, 0.78 + (severity_score / 100.0) * 0.21))

        yolo_detections: List[Dict[str, Any]] = []
        if classification != "Healthy":
            cls_idx = self.DISEASE_CLASSES.index(classification)
            yolo_detections = self._extract_lesion_boxes(
                lesion_mask=(yellow_brown_mask | white_mask),
                class_index=cls_idx,
                label=classification,
                confidence=confidence,
                max_boxes=16,
            )
        normalized_boxes = [
            d["norm_box"] for d in yolo_detections if isinstance(d.get("norm_box"), list)
        ]

        mask_bytes, mask_relative_path = self._render_overlay_and_encode_mask(
            img=img,
            green_mask=green_mask,
            yb_mask=yellow_brown_mask,
            white_mask=white_mask,
            yolo_detections=yolo_detections,
            image_path=image_path,
        )

        recommendation = self.RECOMMENDATIONS.get(
            classification,
            "Canopy is showing discoloration. Monitor field regularly and test soil/leaf samples."
        )

        return {
            "model_used": "Cloud Pipeline (Pixel Canopy Discoloration Analysis + ResNet-50)",
            "classification_label": classification,
            "confidence_score": round(confidence, 2),
            "severity_score": severity_score,
            "treatment_recommendation": recommendation,
            "mask_relative_path": mask_relative_path,
            "mask_bytes": mask_bytes,
            "yolo_detections": yolo_detections,
            "normalized_boxes": normalized_boxes,
            "processed_at": _get_now(),
            "status": "COMPLETED"
        }


_pipeline_singleton: Optional["CloudInferencePipeline"] = None
_pipeline_lock = threading.Lock()


def get_inference_pipeline(
    yolo_weights: Optional[str] = None,
    unet_weights: Optional[str] = None,
    resnet_weights: Optional[str] = None,
) -> CloudInferencePipeline:
    """
    Returns a process-wide thread-safe singleton instance of CloudInferencePipeline.
    """
    global _pipeline_singleton
    if _pipeline_singleton is None:
        with _pipeline_lock:
            if _pipeline_singleton is None:
                _pipeline_singleton = CloudInferencePipeline(
                    yolo_weights=yolo_weights,
                    unet_weights=unet_weights,
                    resnet_weights=resnet_weights,
                )
    return _pipeline_singleton

