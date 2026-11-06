import logging
from celery import shared_task
from django.utils import timezone
from core.models import ImageCapture, EdgeInferenceResult, CloudInferenceResult
from inference.pipeline import get_inference_pipeline

logger = logging.getLogger(__name__)

@shared_task(bind=True, max_retries=3)
def process_cloud_inference_task(self, image_id: int, edge_data: list = None):
    """
    Celery background worker task for running heavy multi-model inference:
    - YOLOv8 candidate detection (yolov8n_rice.tflite)
    - U-Net canopy discoloration segmentation & % calculation (RiceCanopyUNet)
    - ResNet-50 secondary verification & classification (RiceDiscolorationResNet50)
    - Chemical/Cultural treatment recommendations
    """
    try:
        image_record = ImageCapture.objects.get(id=image_id)
    except ImageCapture.DoesNotExist:
        logger.error(f"ImageCapture with ID {image_id} not found.")
        return

    # Update or create the pending record
    cloud_result, _ = CloudInferenceResult.objects.get_or_create(
        image=image_record,
        defaults={'status': 'PROCESSING'}
    )
    cloud_result.status = 'PROCESSING'
    cloud_result.save(update_fields=['status'])

    try:
        pipeline = get_inference_pipeline()
        if not pipeline.is_real_models_loaded:
            pipeline._load_models_if_available()

        # Run inference pipeline
        inference_out = pipeline.process_image(
            image_path=image_record.file.path,
            edge_detections=edge_data
        )

        cloud_result.model_used = inference_out.get('model_used', cloud_result.model_used)
        cloud_result.status = inference_out.get('status', 'COMPLETED')
        cloud_result.severity_score = inference_out.get('severity_score', 0.0)
        cloud_result.classification_label = inference_out.get('classification_label', 'Healthy')
        cloud_result.confidence_score = inference_out.get('confidence_score', 0.0)
        cloud_result.treatment_recommendation = inference_out.get('treatment_recommendation', '')
        cloud_result.processed_at = inference_out.get('processed_at', timezone.now())
        mask_path = inference_out.get('mask_relative_path')
        if mask_path:
            cloud_result.segmentation_mask = mask_path
        cloud_result.error_message = ''
        cloud_result.save()

        # Populate or refresh EdgeInferenceResult from yolov8n_rice.tflite
        yolo_dets = inference_out.get('yolo_detections') or []
        boxes = [d.get('box') for d in yolo_dets if isinstance(d, dict) and isinstance(d.get('box'), list)]
        top_label = yolo_dets[0].get('label') if yolo_dets else cloud_result.classification_label
        top_conf = float(yolo_dets[0].get('confidence', 0.0)) if yolo_dets else cloud_result.confidence_score

        existing_edge = EdgeInferenceResult.objects.filter(image=image_record).first()
        if existing_edge is None:
            EdgeInferenceResult.objects.create(
                image=image_record,
                model_used='yolov8n_rice.tflite',
                bounding_boxes=boxes,
                classification_label=top_label or 'Healthy',
                confidence_score=top_conf,
            )
        elif existing_edge.model_used == 'yolov8n_rice.tflite':
            existing_edge.bounding_boxes = boxes
            existing_edge.classification_label = top_label or 'Healthy'
            existing_edge.confidence_score = top_conf
            existing_edge.save(update_fields=['bounding_boxes', 'classification_label', 'confidence_score'])
        elif not existing_edge.bounding_boxes and boxes:
            existing_edge.bounding_boxes = boxes
            existing_edge.save(update_fields=['bounding_boxes'])

        logger.info(f"Cloud inference completed for Image #{image_id}: {cloud_result.classification_label} ({cloud_result.severity_score}%)")
        return {
            "image_id": image_id,
            "status": "COMPLETED",
            "classification": cloud_result.classification_label,
            "severity": cloud_result.severity_score
        }

    except Exception as exc:
        logger.exception(f"Error during inference on image {image_id}: {exc}")
        cloud_result.status = 'FAILED'
        cloud_result.error_message = str(exc)
        cloud_result.save(update_fields=['status', 'error_message'])
        raise self.retry(exc=exc, countdown=5)
