"""
Unified Detector
Stage 1: EfficientNet binary classifier on image A.
  - Class 0 -> return classification result
  - Class 1 -> run YOLO detector on image B and return detections
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
import os
from PIL import Image
import logging
from datetime import datetime
from ultralytics import YOLO


# ============== LOGGER ==============
def setup_logger():
    """Setup logger for detection"""
    log_filename = 'classifier_detection.log'
    logger = logging.getLogger('EfficientNetDetector')
    logger.setLevel(logging.DEBUG)
    
    # File handler
    file_handler = logging.FileHandler(log_filename, mode='a')
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()
# ====================================


# ============== CONFIG ==============
CONFIG = {
    # EfficientNet classifier
    'classifier_model_path': 'efficientnet_classifier_model.pth',
    'classifier_model_name': 'efficientnet_b0',
    'num_classes': 2,
    'class_names': ['class_0', 'class_1'],
    'confidence_threshold': 0.5,

    # YOLO detector (used only when classifier predicts class_1)
    'yolo_model_yaml': '/home/sadeepa/FYP_Group_10/Amana/ultralytics/ultralytics/cfg/models/11/yolo11.yaml',
    'yolo_weights_path': 'phase1_cv3_model.pt',
    'yolo_imgsz': 640,
    'yolo_device': 'cuda'
}
# ====================================


def get_transforms():
    """Get image transforms for inference"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform


def load_model(device, model_path=CONFIG['classifier_model_path'], model_name=CONFIG['classifier_model_name']):
    """Load pretrained EfficientNet model"""
    logger.info(f"Loading model from {model_path}...")
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create model
    model = getattr(models, model_name)(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, CONFIG['num_classes'])
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully: {model.__class__.__name__}")
    return model


def load_yolo_model():
    """Load pretrained YOLO model"""
    logger.info(f"Loading YOLO model from {CONFIG['yolo_weights_path']}...")

    if not os.path.exists(CONFIG['yolo_weights_path']):
        logger.error(f"YOLO weights not found: {CONFIG['yolo_weights_path']}")
        raise FileNotFoundError(f"YOLO weights not found: {CONFIG['yolo_weights_path']}")

    yolo = YOLO(CONFIG['yolo_model_yaml'])
    state_dict = torch.load(CONFIG['yolo_weights_path'])
    yolo.model.load_state_dict(state_dict)

    logger.info("YOLO model loaded successfully.")
    return yolo


def run_yolo(image_path, yolo_model):
    """Run YOLO detection on image_path and return structured results"""
    logger.info(f"Running YOLO detection on: {image_path}")

    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = yolo_model.predict(
        source=image_path,
        imgsz=CONFIG['yolo_imgsz'],
        device=CONFIG['yolo_device']
    )

    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            detections.append({
                'bbox': box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                'confidence': float(box.conf[0]),
                'class_id': int(box.cls[0]),
                'class_name': r.names[int(box.cls[0])]
            })

    logger.info(f"YOLO found {len(detections)} detection(s) in {image_path}")
    return {
        'image_path': image_path,
        'detections': detections,
        'num_detections': len(detections)
    }


def read_image(image_path):
    """Read image from file"""
    logger.debug(f"Reading image: {image_path}")
    
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = Image.open(image_path).convert('RGB')
    logger.debug(f"Image loaded: {image.size}")
    return image


def detect(image_path_a, image_path_b, classifier, yolo_model, transform, device,
           class_names=CONFIG['class_names']):
    """
    Unified detection pipeline.

    1. Run EfficientNet classifier on image_path_a.
    2. If class_0  -> return classification result.
    3. If class_1  -> run YOLO on image_path_b and return detections.
    """
    try:
        # --- Stage 1: EfficientNet classification on image A ---
        image = read_image(image_path_a)
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = classifier(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class_idx].item()

        predicted_class = class_names[predicted_class_idx]
        logger.info(f"Classifier result: {image_path_a} -> {predicted_class} (confidence: {confidence:.2%})")

        # --- Stage 2: Route based on class ---
        if predicted_class_idx == 0:
            return {
                'stage': 'classifier',
                'image_path': image_path_a,
                'class': predicted_class,
                'class_idx': predicted_class_idx,
                'confidence': confidence,
                'all_probabilities': {class_names[i]: probabilities[0, i].item() for i in range(len(class_names))}
            }
        else:
            # Class 1: run YOLO on image B
            yolo_result = run_yolo(image_path_b, yolo_model)
            return {
                'stage': 'yolo',
                'classifier_trigger': {
                    'image_path': image_path_a,
                    'class': predicted_class,
                    'confidence': confidence
                },
                **yolo_result
            }

    except Exception as e:
        logger.error(f"Error during detection: {str(e)}")
        raise


def detect_batch(pairs, classifier, yolo_model, transform, device):
    """
    Batch detection over a list of (image_a, image_b) pairs.

    Args:
        pairs: list of tuples (image_path_a, image_path_b)
    """
    logger.info(f"Starting batch detection for {len(pairs)} pairs...")

    results = []
    for image_path_a, image_path_b in pairs:
        try:
            result = detect(image_path_a, image_path_b, classifier, yolo_model, transform, device)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process pair ({image_path_a}, {image_path_b}): {str(e)}")
            results.append({
                'image_path_a': image_path_a,
                'image_path_b': image_path_b,
                'error': str(e)
            })

    logger.info(f"Batch detection completed. Processed {len(results)} pairs.")
    return results


def main():
    """Main detection function - Example usage"""
    logger.info("="*70)
    logger.info(f"Starting Unified Detector - {datetime.now()}")
    logger.info("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load models
    try:
        classifier = load_model(device)
        yolo_model = load_yolo_model()
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    transform = get_transforms()

    # Example usage
    image_a = './sample_image_a.jpg'   # classified by EfficientNet
    image_b = './sample_image_b.jpg'   # passed to YOLO if class_1

    result = detect(image_a, image_b, classifier, yolo_model, transform, device)
    logger.info(f"Result: {result}")
    

if __name__ == '__main__':
    main()
