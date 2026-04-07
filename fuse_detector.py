"""
Unified Detector Backend (for GUI)
Fusion architecture:
- EfficientNet → P4
- YOLO → F
- Fusion → G
"""

import os
import logging
from datetime import datetime

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


# ============== LOGGER ==============
def setup_logger():
    log_filename = 'classifier_detection.log'
    logger = logging.getLogger('EfficientNetDetector')
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_filename, mode='a')
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


logger = setup_logger()


# ============== CONFIG ==============
CONFIG = {
    'classifier_model_path': 'weights/efficientnet_classifier_model.pth',
    'classifier_model_name': 'efficientnet_b0',
    'num_classes': 2,
    'class_names': ['no_drone', 'drone'],

    'T1': 0.5,
    'T2': 0.5,

    'fusion_w1': 1.0,
    'fusion_w2': 1.0,
    'fusion_b': 0.0,

    'yolo_model_yaml': 'ultralytics/cfg/models/11/yolo11.yaml',
    'yolo_weights_path': 'weights/phase1_cv3_model.pt',
    'yolo_imgsz': 640,
    'yolo_device': 'cuda',
    'yolo_nc': 1,
    'yolo_class_names': ['phantom'],

    'output_dir': 'outputs'  # 🔥 NEW
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
    model = getattr(models, CONFIG['classifier_model_name'])(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, CONFIG['num_classes'])

    model.load_state_dict(torch.load(CONFIG['classifier_model_path'], map_location=device))
    model = model.to(device)
    model.eval()
    return model


def load_yolo_model():
    yolo = YOLO(CONFIG['yolo_model_yaml'])

    # 🔥 NEW: rebuild model with correct nc
    yolo.model = DetectionModel(CONFIG['yolo_model_yaml'], nc=CONFIG['yolo_nc'])

    state_dict = torch.load(CONFIG['yolo_weights_path'], map_location='cpu')
    yolo.model.load_state_dict(state_dict)

    yolo.model.to(CONFIG['yolo_device'])
    yolo.model.eval()

    return yolo


def logistic_fusion(F, P4):
    z = CONFIG['fusion_w1'] * F + CONFIG['fusion_w2'] * P4 + CONFIG['fusion_b']
    G = 1.0 / (1.0 + torch.exp(torch.tensor(-z, dtype=torch.float32)))
    return float(G)


def read_image(image_path):
    return Image.open(image_path).convert('RGB')


def run_yolo(image_path, yolo_model):
    results = yolo_model.predict(
        source=image_path,
        imgsz=CONFIG['yolo_imgsz'],
        device=CONFIG['yolo_device'],
        verbose=False
    )

    detections = []
    max_conf = 0.0
    best_class_name = None

    # 🔥 NEW: create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    base_name = os.path.basename(image_path).split('.')[0]
    save_path = os.path.join(CONFIG['output_dir'], f"{base_name}_detected.jpg")  # 🔥 NEW

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            class_id = int(box.cls[0])

            class_name = CONFIG['yolo_class_names'][class_id]

            detections.append({
                'bbox': box.xyxy[0].tolist(),
                'confidence': conf,
                'class_name': class_name
            })

            if conf > max_conf:
                max_conf = conf
                best_class_name = class_name

        # 🔥 NEW: save plotted image
        plotted_img = r.plot()
        Image.fromarray(plotted_img[..., ::-1]).save(save_path)

    return {
        'detections': detections,
        'num_detections': len(detections),
        'max_confidence': max_conf,
        'best_class_name': best_class_name,
        'saved_detection_image': save_path  # 🔥 NEW
    }


def detect(image_path_a, image_path_b, classifier, yolo_model, transform, device):

    image = read_image(image_path_a)
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = classifier(image_tensor)
        probabilities = torch.softmax(output, dim=1)

    P4 = probabilities[0, 1].item()

    yolo_result = run_yolo(image_path_b, yolo_model)
    F = yolo_result['max_confidence']

    G = logistic_fusion(F, P4)

    # 🔥 NEW: base result (used in GUI)
    base_result = {
        'P4': P4,
        'F': F,
        'G': G,
        'num_detections': yolo_result['num_detections'],
        'saved_detection_image': yolo_result['saved_detection_image']  # 🔥 NEW
    }

    if G <= CONFIG['T1']:
        return {
            **base_result,
            'final_decision': 'NOdrone',
            'status_message': 'No drone detected'  # 🔥 NEW
        }

    if F > CONFIG['T2']:
        return {
            **base_result,
            'final_decision': 'DroneType',
            'drone_type': yolo_result['best_class_name'],
            'status_message': f"Drone detected: {yolo_result['best_class_name']}"  # 🔥 NEW
        }

    return {
        **base_result,
        'final_decision': 'Detected',
        'status_message': 'Drone detected but type uncertain'  # 🔥 NEW
    }


# 🔥 NEW: helper for GUI
def load_all_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = load_model(device)
    yolo_model = load_yolo_model()
    transform = get_transforms()
    return classifier, yolo_model, transform, device


def main():
    classifier, yolo_model, transform, device = load_all_models()

    image_a = 'data/SCD_Images_224/scd_Phanthom_5_3.00m.png'
    image_b = 'data/YOLO_data/split/moving/val/images/phantom_movingin_141.png'

    result = detect(image_a, image_b, classifier, yolo_model, transform, device)
    print(result)


if __name__ == "__main__":
    main()