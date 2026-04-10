import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import os
import re
import logging
import numpy as np
from PIL import Image
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ============== CONFIG ==============
CONFIG = {
    'train_dir': 'data/data/old/effientnet_grapgh/train',
    'test_dir': 'data/data/old/effientnet_grapgh/test',
    'output_root': 'runs/scd_train_full_range',
    'batch_size': 32,
    'num_epochs': 50,
    'patience': 5,            # Early stopping patience
    'learning_rate': 0.001,
    'model_name': 'efficientnet_b0'
}

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(CONFIG['output_root'], TIMESTAMP)
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, 'best_classifier.pth')
# ====================================

# ============== LOGGER ==============
def setup_logger():
    log_filename = os.path.join(OUTPUT_DIR, 'training.log')
    logger = logging.getLogger('EfficientNetTrainer')
    logger.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler(log_filename, mode='a')
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger

logger = setup_logger()

# ============== EARLY STOPPING ==============
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = None
        self.early_stop = False

    def __call__(self, val_acc):
        if self.best_acc is None:
            self.best_acc = val_acc
        elif val_acc < self.best_acc + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_acc = val_acc
            self.counter = 0

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images, self.labels, self.distances = [], [], []
        
        valid_extensions = ('.png', '.jpg', '.jpeg')
        dist_pattern = re.compile(r'(\d+\.?\d*)m')

        for file_name in os.listdir(root_dir):
            if file_name.lower().endswith(valid_extensions):
                img_path = os.path.join(root_dir, file_name)
                name_parts = os.path.splitext(file_name)[0].split('_')
                
                label = 0 if 'NO' in name_parts else 1
                dist_match = dist_pattern.search(file_name)
                distance = float(dist_match.group(1)) if dist_match else 0.0
                
                self.images.append(img_path)
                self.labels.append(label)
                self.distances.append(distance)

    def __len__(self): return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, self.labels[idx]

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# ============== EVALUATION ENGINE ==============
def run_evaluation(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    all_preds, all_labels = [], []
    
    # Bins for summary
    bins = {
        "Drone 0-3m":  {"y_true": [], "y_pred": []},
        "Drone 4-10m": {"y_true": [], "y_pred": []},
        "Drone 10-40m": {"y_true": [], "y_pred": []},
        "No Drone":     {"y_true": [], "y_pred": []}
    }

    logger.info("Running range-wise metric analysis...")
    
    with torch.no_grad():
        for i, (image, label) in enumerate(loader):
            image = image.to(device)
            output = model(image)
            _, pred = torch.max(output, 1)
            
            p, l = pred.item(), label.item()
            dist = dataset.distances[i]
            
            all_preds.append(p)
            all_labels.append(l)
            
            # Sort into bins
            if l == 0:
                bins["No Drone"]["y_true"].append(l)
                bins["No Drone"]["y_pred"].append(p)
            else:
                if 0 <= dist <= 3.0:
                    bins["Drone 0-3m"]["y_true"].append(l)
                    bins["Drone 0-3m"]["y_pred"].append(p)
                elif 3.0 < dist <= 10.0:
                    bins["Drone 4-10m"]["y_true"].append(l)
                    bins["Drone 4-10m"]["y_pred"].append(p)
                elif dist > 10.0:
                    bins["Drone 10-40m"]["y_true"].append(l)
                    bins["Drone 10-40m"]["y_pred"].append(p)

    def calculate_metrics(y_true, y_pred):
        if not y_true: return 0.0, 0.0, 0.0, 0.0
        # For subset Accuracy, we compare directly. 
        # For subset Precision/Recall, we use binary metrics relative to the original task.
        acc = accuracy_score(y_true, y_pred)
        # Note: Precision/Recall on a single-class slice can be tricky. 
        # We use 'macro' or 'binary' with zero_division handling to ensure the report stays clean.
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=1)
        return acc, prec, rec, f1

    # Overall Metrics
    o_prec, o_rec, o_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    o_acc = accuracy_score(all_labels, all_preds)

    report = [
        "="*85,
        f"FINAL PERFORMANCE SUMMARY - {TIMESTAMP}",
        "="*85,
        f"OVERALL SYSTEM PERFORMANCE:",
        f"  Accuracy:  {o_acc*100:.2f}% | Precision: {o_prec:.4f} | Recall: {o_rec:.4f} | F1: {o_f1:.4f}",
        "\n" + "-"*85,
        f"{'Category':<15} | {'Samples':<8} | {'Accuracy':<12} | {'F1-Score':<10}",
        "-"*85
    ]

    for cat_name, data in bins.items():
        acc, _, _, f1 = calculate_metrics(data['y_true'], data['y_pred'])
        report.append(f"{cat_name:<15} | {len(data['y_true']):<8} | {acc*100:<10.2f}% | {f1:<10.3f}")

    report_str = "\n".join(report)
    print(report_str)
    
    with open(os.path.join(OUTPUT_DIR, 'evaluation_report.txt'), 'w') as f:
        f.write(report_str)
    logger.info(f"Report saved to {OUTPUT_DIR}")

# ============== TRAINING LOGIC ==============
def train_model(model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    early_stopping = EarlyStopping(patience=CONFIG['patience'])
    best_val_acc = 0.0

    for epoch in range(CONFIG['num_epochs']):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward(); optimizer.step()
            
            train_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += lbls.size(0); correct += (pred == lbls).sum().item()

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                _, pred = torch.max(outputs, 1)
                val_total += lbls.size(0); val_correct += (pred == lbls).sum().item()

        val_acc = (val_correct / val_total)
        logger.info(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}] - Loss: {train_loss/len(train_loader):.4f}, Train Acc: {100*correct/total:.2f}%, Val Acc: {val_acc*100:.2f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            logger.info(f"Model Improved. Checkpoint saved.")

        # Early Stopping check
        early_stopping(val_acc)
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}.")
            break

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # transform = get_transforms()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),transforms.ToTensor(),
    ])

    train_set = CustomImageDataset(CONFIG['train_dir'], transform=transform)
    test_set = CustomImageDataset(CONFIG['test_dir'], transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(test_set, batch_size=CONFIG['batch_size'], shuffle=False)

    model = getattr(models, CONFIG['model_name'])(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model = model.to(device)
    
    train_model(model, train_loader, val_loader, device)
    
    logger.info("Evaluation on best checkpoint...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    run_evaluation(model, test_set, device)

if __name__ == '__main__':
    main()