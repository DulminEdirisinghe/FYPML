"""
Unified Detector Backend
EfficientNet -> 4-class drone classification (No Drone, Phantom, Matrice, Mavic)
YOLO -> (Removed, legacy compatibility maintained)
Fusion -> (Removed, legacy compatibility maintained)
"""

import os
import logging
from datetime import datetime
import time
import glob

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models


# ============== LOGGER ==============
def setup_logger():
    log_filename = 'classifier_detection.log'
    logger = logging.getLogger('EfficientNetDetector')
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_filename, mode='a')
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


logger = setup_logger()


# ============== CONFIG ==============
CONFIG = {
    'classifier_model_path': 'runs/efficientnet_v5_classification/20260429_205239/best_classifier.pth',
    'classifier_model_name': 'efficientnet_b0',
    'num_classes': 4,
    'class_names': ['no_drone', 'Phantom', 'Matrice', 'Mavic'],

    'output_dir': 'outputs',

    # Streaming folders
    'stream_folder_a': 'runs/folder_a',
    'stream_folder_b': 'runs/folder_b',
    'poll_interval': 5
}


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_model(device):
    model = getattr(models, CONFIG['classifier_model_name'])(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        CONFIG['num_classes']
    )

    if os.path.exists(CONFIG['classifier_model_path']):
        model.load_state_dict(
            torch.load(CONFIG['classifier_model_path'], map_location=device)
        )
    else:
        logger.warning(f"Could not load weights from {CONFIG['classifier_model_path']}")

    model = model.to(device)
    model.eval()

    return model


def load_yolo_model():
    # Dummy function to maintain compatibility for load_all_models and app_all.py
    return None


def read_image(image_path):
    return Image.open(image_path).convert('RGB')


def detect(image_path_a, image_path_b, classifier, yolo_model, transform, device):
    # EfficientNet 4-class classification
    image = read_image(image_path_a)
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = classifier(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        max_prob, preds = torch.max(probabilities, 1)

    eff_pred = preds.item()
    
    # P4, F, G dummy values for backward compatibility
    P4 = probabilities[0, 1:].sum().item() if eff_pred != 0 else probabilities[0, 0].item()
    F = max_prob.item()
    G = max_prob.item()

    logger.info(
        f"Model Scores - EfficientNet Class: {eff_pred}, Prob: {max_prob.item():.4f}"
    )

    base_result = {
        'P4': P4,
        'F': F,
        'G': G,
        'num_detections': 1 if eff_pred != 0 else 0,
        'saved_detection_image': image_path_b,
        'detections': []
    }

    if eff_pred == 0:
        return {
            **base_result,
            'final_decision': 'NOdrone',
            'drone_type': None,
            'status_message': 'No Drone Detected'
        }
    
    drone_type = CONFIG['class_names'][eff_pred]
    return {
        **base_result,
        'final_decision': 'DroneType',
        'drone_type': drone_type,
        'status_message': f"Drone Detected: {drone_type}"
    }


def load_all_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classifier = load_model(device)
    yolo_model = load_yolo_model()
    transform = get_transforms()

    return classifier, yolo_model, transform, device


def get_latest_file(folder_path):
    list_of_files = glob.glob(os.path.join(folder_path, '*'))

    image_files = [
        f for f in list_of_files
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    if not image_files:
        return None

    return max(image_files, key=os.path.getctime)


def main():
    logger.info("Loading models...")

    classifier, yolo_model, transform, device = load_all_models()

    print("Models loaded successfully. Starting real-time folder monitoring...")

    os.makedirs(CONFIG['stream_folder_a'], exist_ok=True)
    os.makedirs(CONFIG['stream_folder_b'], exist_ok=True)

    last_processed_time_a = None
    last_processed_time_b = None

    try:
        while True:
            latest_image_a = get_latest_file(CONFIG['stream_folder_a'])
            latest_image_b = get_latest_file(CONFIG['stream_folder_b'])

            if latest_image_a and latest_image_b:
                time_a = os.path.getctime(latest_image_a)
                time_b = os.path.getctime(latest_image_b)

                has_update_a = last_processed_time_a is None or time_a > last_processed_time_a
                has_update_b = last_processed_time_b is None or time_b > last_processed_time_b

                if has_update_a and has_update_b:
                    time.sleep(0.1)

                    try:
                        filename_a = os.path.basename(latest_image_a)
                        filename_b = os.path.basename(latest_image_b)

                        logger.info("Processing new pair")
                        logger.info(f"Folder A image: {latest_image_a} timestamp: {time_a}")
                        logger.info(f"Folder B image: {latest_image_b} timestamp: {time_b}")

                        result = detect(
                            latest_image_a,
                            latest_image_b,
                            classifier,
                            yolo_model,
                            transform,
                            device
                        )

                        print("\n" + "=" * 60)
                        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"Folder A: {filename_a}")
                        print(f"Folder B: {filename_b}")
                        print(f"EfficientNet P4 dummy: {result['P4']:.4f}")
                        print(f"YOLO F dummy: {result['F']:.4f}")
                        print(f"Fusion Score G dummy: {result['G']:.4f}")
                        print(f"Number of 'detections': {result['num_detections']}")
                        print(f"Decision: {result['status_message']}")
                        print("=" * 60 + "\n")

                        last_processed_time_a = time_a
                        last_processed_time_b = time_b

                    except Exception as e:
                        logger.error(f"Failed to process image pair: {e}")

                else:
                    logger.info(
                        f"NO update in 5s - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )

            else:
                if not latest_image_a:
                    logger.warning("No images found in Folder A")

                if not latest_image_b:
                    logger.warning("No images found in Folder B")

            time.sleep(CONFIG['poll_interval'])

    except KeyboardInterrupt:
        logger.info("Real-time monitoring stopped by user.")


if __name__ == "__main__":
    main()
