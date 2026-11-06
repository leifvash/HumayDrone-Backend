"""
Python TFLite YOLOv8 Rice Disease Detector (`yolov8n_rice.tflite`).
Mirrors the exact preprocessing, tensor decoding ([1, 7, 8400], [1, 8400, 7],
[1, 8, 8400], [1, 8400, 8], [1, 84, 8400]), confidence thresholding, and
class-aware Non-Maximum Suppression (NMS) from `YoloV8Detector.kt`.
"""
import os
import threading
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union

import cv2
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter as TFLiteInterpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
    except ImportError:
        try:
            import tensorflow as tf
            TFLiteInterpreter = tf.lite.Interpreter
        except ImportError:
            TFLiteInterpreter = None


@dataclass
class DetectionResult:
    class_index: int
    label: str
    confidence: float
    box: List[float]  # [left, top, right, bottom]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_index": self.class_index,
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "box": [round(float(coord), 2) for coord in self.box],
        }


class YoloV8TFLiteDetector:
    MODEL_INPUT_SIZE = 640
    NUM_CHANNELS = 3
    CONFIDENCE_THRESHOLD = 0.45
    IOU_THRESHOLD = 0.45
    NUM_THREADS = 4

    # Matches YoloV8Detector.kt LABELS while normalizing "Whitehead" -> "White Head" for pipeline consistency
    RAW_LABELS = [
        "Healthy",
        "Bacterial Leaf Blight",
        "Dead Heart",
        "Whitehead",
    ]

    LABELS = [
        "Healthy",
        "Bacterial Leaf Blight",
        "Dead Heart",
        "White Head",
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        num_threads: int = NUM_THREADS,
    ):
        if TFLiteInterpreter is None:
            raise RuntimeError(
                "No TFLite runtime available (install ai-edge-litert, tflite-runtime, or tensorflow)."
            )

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weights_tflite = os.path.join(base_dir, "weights", "yolov8n_rice.tflite")
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

        if model_path and os.path.exists(model_path):
            self.model_path = model_path
        elif os.path.exists(weights_tflite):
            self.model_path = weights_tflite
        elif os.path.exists(android_asset_tflite):
            self.model_path = android_asset_tflite
        else:
            raise FileNotFoundError(
                f"yolov8n_rice.tflite not found at {model_path or weights_tflite}"
            )

        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.num_threads = int(num_threads)
        self._lock = threading.Lock()

        self.interpreter = TFLiteInterpreter(
            model_path=self.model_path,
            num_threads=self.num_threads,
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

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
    def _ensure_bgr_uint8(img: np.ndarray) -> Optional[np.ndarray]:
        if img is None or not isinstance(img, np.ndarray) or img.size == 0:
            return None
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.ndim != 3 or img.shape[2] != 3:
            return None

        if img.dtype != np.uint8:
            if np.issubdtype(img.dtype, np.floating) and float(np.nanmax(img)) <= 1.0:
                img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Resizes BGR image to 640x640, converts to RGB, normalizes [0..255] -> [0.0..1.0],
        and formats tensor layout according to the TFLite input tensor shape.
        """
        bgr = self._ensure_bgr_uint8(image_bgr)
        if bgr is None:
            raise ValueError("Invalid or empty image array passed to preprocess()")

        orig_h, orig_w = bgr.shape[:2]
        if orig_w == self.MODEL_INPUT_SIZE and orig_h == self.MODEL_INPUT_SIZE:
            resized_bgr = bgr
        else:
            resized_bgr = cv2.resize(
                bgr,
                (self.MODEL_INPUT_SIZE, self.MODEL_INPUT_SIZE),
                interpolation=cv2.INTER_LINEAR,
            )

        rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        rgb_norm = rgb.astype(np.float32) / 255.0

        input_shape = self.input_details[0]["shape"]
        if len(input_shape) == 4 and input_shape[1] == self.NUM_CHANNELS:
            # NCHW [1, 3, 640, 640]
            tensor = np.expand_dims(np.transpose(rgb_norm, (2, 0, 1)), axis=0)
        else:
            # NHWC [1, 640, 640, 3]
            tensor = np.expand_dims(rgb_norm, axis=0)

        return np.ascontiguousarray(tensor, dtype=np.float32)

    def detect(
        self,
        image: Union[str, np.ndarray],
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> List[DetectionResult]:
        """
        Runs TFLite inference on an image path or BGR numpy array and returns NMS-filtered DetectionResults.
        """
        dets, _ = self.detect_with_raw_summary(
            image=image,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )
        return dets

    def detect_with_raw_summary(
        self,
        image: Union[str, np.ndarray],
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> Tuple[List[DetectionResult], Dict[str, Any]]:
        """
        Runs TFLite inference and returns both NMS-filtered detections and a summary of top raw anchor scores.
        """
        if isinstance(image, str):
            img_bgr = self._read_image_bgr(image)
        else:
            img_bgr = self._ensure_bgr_uint8(image)

        if img_bgr is None or img_bgr.size == 0:
            return [], {"max_confidence": 0.0, "best_class_index": 0, "best_label": "Healthy"}

        orig_height, orig_width = img_bgr.shape[:2]
        if orig_height <= 0 or orig_width <= 0:
            return [], {"max_confidence": 0.0, "best_class_index": 0, "best_label": "Healthy"}

        input_tensor = self.preprocess(img_bgr)

        input_index = self.input_details[0]["index"]
        output_index = self.output_details[0]["index"]

        with self._lock:
            self.interpreter.set_tensor(input_index, input_tensor)
            self.interpreter.invoke()
            output_tensor = self.interpreter.get_tensor(output_index).copy()

        return self.decode_output(
            output_tensor=output_tensor,
            orig_width=orig_width,
            orig_height=orig_height,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )

    def decode_output(
        self,
        output_tensor: np.ndarray,
        orig_width: int,
        orig_height: int,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> Tuple[List[DetectionResult], Dict[str, Any]]:
        """
        Decodes output tensor of shape [1, C, N] (e.g. [1, 7, 8400], [1, 8, 8400], [1, 84, 8400])
        or [1, N, C] (e.g. [1, 8400, 7], [1, 8400, 8], [1, 8400, 84]), matching YoloV8Detector.kt.
        """
        conf_thresh = (
            self.confidence_threshold
            if confidence_threshold is None
            else float(confidence_threshold)
        )
        nms_thresh = (
            self.iou_threshold if iou_threshold is None else float(iou_threshold)
        )

        if (
            output_tensor is None
            or not isinstance(output_tensor, np.ndarray)
            or output_tensor.ndim != 3
            or orig_width <= 0
            or orig_height <= 0
        ):
            return [], {"max_confidence": 0.0, "best_class_index": 0, "best_label": "Healthy"}

        dim1 = int(output_tensor.shape[1])
        dim2 = int(output_tensor.shape[2])

        is_channels_first = dim1 < dim2
        num_features = dim1 if is_channels_first else dim2
        num_anchors = dim2 if is_channels_first else dim1

        if num_features <= 4 or num_anchors <= 0:
            return [], {"max_confidence": 0.0, "best_class_index": 0, "best_label": "Healthy"}

        # Normalize to [num_anchors, num_features]
        anchors = np.nan_to_num(
            output_tensor[0].T if is_channels_first else output_tensor[0],
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

        num_classes = min(len(self.LABELS), num_features - 4)
        class_scores = anchors[:, 4 : 4 + num_classes]

        best_class_indices = np.argmax(class_scores, axis=1)
        max_confidences = np.max(class_scores, axis=1)

        # Summary across all anchors even if below threshold
        overall_best_anchor = int(np.argmax(max_confidences))
        overall_max_conf = float(max_confidences[overall_best_anchor])
        overall_best_cls = int(best_class_indices[overall_best_anchor]) if overall_max_conf > 0.0 else 0
        overall_best_label = (
            self.LABELS[overall_best_cls]
            if 0 <= overall_best_cls < len(self.LABELS)
            else "Unknown"
        )
        raw_summary = {
            "max_confidence": max(0.0, overall_max_conf),
            "best_class_index": overall_best_cls,
            "best_label": overall_best_label,
            "output_shape": list(output_tensor.shape),
            "num_anchors": num_anchors,
            "num_classes": num_classes,
        }

        scale_x = float(orig_width) / float(self.MODEL_INPUT_SIZE)
        scale_y = float(orig_height) / float(self.MODEL_INPUT_SIZE)

        # Match YoloV8Detector.kt: score > 0.0f and maxConfidence >= CONFIDENCE_THRESHOLD
        passing_indices = np.where((max_confidences >= conf_thresh) & (max_confidences > 0.0))[0]
        candidate_detections: List[DetectionResult] = []

        for idx in passing_indices:
            best_class_index = int(best_class_indices[idx])
            max_confidence = float(max_confidences[idx])

            cx = float(anchors[idx, 0])
            cy = float(anchors[idx, 1])
            w = float(anchors[idx, 2])
            h = float(anchors[idx, 3])

            # Scale up if model outputs normalized coordinates [0..1]
            if cx <= 1.0 and cy <= 1.0 and w <= 1.0 and h <= 1.0:
                cx *= self.MODEL_INPUT_SIZE
                cy *= self.MODEL_INPUT_SIZE
                w *= self.MODEL_INPUT_SIZE
                h *= self.MODEL_INPUT_SIZE

            x_min_640 = cx - w / 2.0
            y_min_640 = cy - h / 2.0
            x_max_640 = cx + w / 2.0
            y_max_640 = cy + h / 2.0

            left = float(np.clip(x_min_640 * scale_x, 0.0, float(orig_width)))
            top = float(np.clip(y_min_640 * scale_y, 0.0, float(orig_height)))
            right = float(np.clip(x_max_640 * scale_x, 0.0, float(orig_width)))
            bottom = float(np.clip(y_max_640 * scale_y, 0.0, float(orig_height)))

            label = (
                self.LABELS[best_class_index]
                if 0 <= best_class_index < len(self.LABELS)
                else "Unknown"
            )

            candidate_detections.append(
                DetectionResult(
                    class_index=best_class_index,
                    label=label,
                    confidence=max_confidence,
                    box=[left, top, right, bottom],
                )
            )

        selected = self.apply_nms(candidate_detections, nms_thresh)
        return selected, raw_summary

    @classmethod
    def apply_nms(
        cls, candidates: List[DetectionResult], iou_threshold: float
    ) -> List[DetectionResult]:
        """
        Class-wise Non-Maximum Suppression matching YoloV8Detector.applyNMS in Kotlin.
        """
        selected_detections: List[DetectionResult] = []
        grouped_by_class: Dict[int, List[DetectionResult]] = {}
        for det in candidates:
            grouped_by_class.setdefault(det.class_index, []).append(det)

        for _, class_candidates in grouped_by_class.items():
            sorted_candidates = sorted(
                class_candidates, key=lambda d: d.confidence, reverse=True
            )
            while sorted_candidates:
                best_candidate = sorted_candidates.pop(0)
                selected_detections.append(best_candidate)
                sorted_candidates = [
                    next_cand
                    for next_cand in sorted_candidates
                    if cls.calculate_iou(best_candidate.box, next_cand.box)
                    < iou_threshold
                ]

        selected_detections.sort(key=lambda d: d.confidence, reverse=True)
        return selected_detections

    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """
        Computes Intersection-over-Union (IoU) between two [left, top, right, bottom] boxes.
        """
        inter_left = max(box1[0], box2[0])
        inter_top = max(box1[1], box2[1])
        inter_right = min(box1[2], box2[2])
        inter_bottom = min(box1[3], box2[3])

        inter_width = max(0.0, inter_right - inter_left)
        inter_height = max(0.0, inter_bottom - inter_top)
        inter_area = inter_width * inter_height

        box1_area = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        box2_area = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0.0 else 0.0
