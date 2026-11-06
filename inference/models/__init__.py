def __getattr__(name):
    if name == "RiceCanopyUNet":
        from .unet import RiceCanopyUNet
        return RiceCanopyUNet
    if name == "RiceDiscolorationResNet50":
        from .resnet import RiceDiscolorationResNet50
        return RiceDiscolorationResNet50
    if name == "YoloV8TFLiteDetector":
        from .yolo_tflite import YoloV8TFLiteDetector
        return YoloV8TFLiteDetector
    if name == "DetectionResult":
        from .yolo_tflite import DetectionResult
        return DetectionResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "RiceCanopyUNet",
    "RiceDiscolorationResNet50",
    "YoloV8TFLiteDetector",
    "DetectionResult",
]
