"""
EfficientNet-B0 Drone Classifier
Classes: No Drone (0), Phantom (1), Mavic (2), Matrice (3)

Handles alias names in filenames:
  - No Drone  : NO, NO_DRONE, nodrone
  - Phantom   : Phantom, Phanthom, PHANTOM, PHANTHOM (common typo in dataset)
  - Mavic     : MAVIC3, Mavic3, mavic, MAVIC
  - Matrice   : MATRICE4, Matrice4, matrice, MATRICE

Range-wise accuracy bins (for drone samples only):
  - 0–3 m
  - >3–10 m
  - >10–40 m
  - No Drone (distance not applicable)
"""

from collections import Counter

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
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    'train_dir':    'data/v10/SCD_images/train',
    'test_dir':     'data/v10/SCD_images/test',
    'output_root':  'runs/scd_classification_v10',
    'batch_size':   32,
    'num_epochs':   50,
    'patience':     20,
    'learning_rate': 0.001,
    'model_name':   'efficientnet_b0',
}

# 4 canonical classes
CLASS_NAMES = ['No Drone', 'Phantom', 'Mavic', 'Matrice']
NUM_CLASSES  = len(CLASS_NAMES)

TIMESTAMP      = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR     = os.path.join(CONFIG['output_root'], TIMESTAMP)
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, 'best_classifier.pth')
# ============================================================


# ============================================================
# LOGGER
# ============================================================
def setup_logger() -> logging.Logger:
    log_filename = os.path.join(OUTPUT_DIR, 'training.log')
    logger = logging.getLogger('EfficientNetTrainer')
    logger.setLevel(logging.DEBUG)

    file_handler    = logging.FileHandler(log_filename, mode='a')
    console_handler = logging.StreamHandler()
    formatter       = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger

logger = setup_logger()


# ============================================================
# LABEL RESOLVER  (all alias → canonical class index)
# ============================================================
def resolve_label(filename: str) -> int:
    """
    Map a filename to one of the 4 canonical class indices.

    Priority order matters: check 'no drone' markers first so
    files that accidentally contain a drone keyword aren't mislabelled.

    Returns:
        0 – No Drone
        1 – Phantom
        2 – Mavic
        3 – Matrice
    """
    upper = filename.upper()

    # ---- No-drone aliases ----
    # Use lookarounds instead of \b — underscores are word chars so \b
    # fails on patterns like _NO_DRONE_ or _NO_
    no_drone_patterns = [
        r'(?<![A-Z])NO[_\-]?DRONE(?![A-Z])',   # NO_DRONE, NO-DRONE, NODRONE
        r'(?<![A-Z])NO(?![A-Z])',               # bare "NO" token e.g. scd_NO_42.png
        r'(?<![A-Z])NEGATIVE(?![A-Z])',
        r'(?<![A-Z])BACKGROUND(?![A-Z])',
    ]
    for pat in no_drone_patterns:
        if re.search(pat, upper):
            return 0

    # ---- Phantom aliases (includes common 'Phanthom' typo) ----
    # Use lookaround instead of \b so underscores act as boundaries too
    if re.search(r'(?<![A-Z])PHANT[A-Z]*OM\d*(?![A-Z])', upper):
        return 1

    # ---- Mavic aliases ----
    if re.search(r'(?<![A-Z])MAVIC\d*(?![A-Z])', upper):
        return 2

    # ---- Matrice aliases ----
    if re.search(r'(?<![A-Z])MATRICE\d*(?![A-Z])', upper):
        return 3

    # ---- Unknown → default to No Drone ----
    logger.warning(f"  [LABEL] Could not resolve class for '{filename}', defaulting to No Drone (0).")
    return 0


def parse_distance(filename: str) -> float:
    """Extract distance in metres from filename, e.g. '20.00m' → 20.0."""
    match = re.search(r'(\d+\.?\d*)m', filename, re.IGNORECASE)
    return float(match.group(1)) if match else 0.0


# ============================================================
# DATASET
# ============================================================
class DroneDataset(Dataset):
    VALID_EXT = ('.png', '.jpg', '.jpeg')

    def __init__(self, root_dir: str, transform=None):
        self.root_dir  = root_dir
        self.transform = transform
        self.images:    list[str]   = []
        self.labels:    list[int]   = []
        self.distances: list[float] = []

        for fname in sorted(os.listdir(root_dir)):
            if not fname.lower().endswith(self.VALID_EXT):
                continue
            label    = resolve_label(fname)
            distance = parse_distance(fname)
            self.images.append(os.path.join(root_dir, fname))
            self.labels.append(label)
            self.distances.append(distance)

        logger.info(f"[Dataset] {root_dir}  →  {len(self.images)} samples")
        dist = Counter(self.labels)
        for idx, name in enumerate(CLASS_NAMES):
            logger.info(f"  Class {idx} ({name:<10}): {dist.get(idx, 0):>5} samples")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


# ============================================================
# EARLY STOPPING
# ============================================================
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_acc   = None
        self.early_stop = False

    def __call__(self, val_acc: float):
        if self.best_acc is None:
            self.best_acc = val_acc
        elif val_acc < self.best_acc + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_acc = val_acc
            self.counter  = 0


# ============================================================
# EVALUATION
# ============================================================
def assign_bin(label: int, distance: float) -> str:
    """Return the evaluation bin name for a sample."""
    if label == 0:
        return 'No Drone'
    if distance <= 3.0:
        return 'Drone 0-3m'
    if distance <= 10.0:
        return 'Drone >3-10m'
    return 'Drone >10-40m'


def run_evaluation(model: nn.Module, dataset: DroneDataset, device: torch.device):
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_preds: list[int] = []
    all_labels: list[int] = []

    bins: dict[str, dict] = {
        'No Drone':      {'y_true': [], 'y_pred': []},
        'Drone 0-3m':    {'y_true': [], 'y_pred': []},
        'Drone >3-10m':  {'y_true': [], 'y_pred': []},
        'Drone >10-40m': {'y_true': [], 'y_pred': []},
    }

    logger.info("Running evaluation …")

    with torch.no_grad():
        for i, (image, label) in enumerate(loader):
            image   = image.to(device)
            output  = model(image)
            _, pred = torch.max(output, 1)

            p = pred.item()
            l = label.item()
            dist = dataset.distances[i]

            all_preds.append(p)
            all_labels.append(l)

            bin_name = assign_bin(l, dist)
            bins[bin_name]['y_true'].append(l)
            bins[bin_name]['y_pred'].append(p)

    # ---- Helpers ----
    def metrics(y_true, y_pred):
        if not y_true:
            return 0.0, 0.0, 0.0, 0.0
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        return acc, prec, rec, f1

    # ---- Overall ----
    o_acc  = accuracy_score(all_labels, all_preds)
    o_prec, o_rec, o_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    # ---- Per-class report ----
    per_class_txt = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    # ---- Confusion matrix ----
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    cm_header = f"{'':>12}" + "".join(f"{CLASS_NAMES[j]:>12}" for j in range(NUM_CLASSES))
    cm_rows    = "\n".join(
        f"{CLASS_NAMES[i]:>12}" + "".join(f"{cm[i][j]:>12}" for j in range(NUM_CLASSES))
        for i in range(NUM_CLASSES)
    )

    W = 90
    report_lines = [
        "=" * W,
        f"  FINAL EVALUATION REPORT  —  {TIMESTAMP}",
        "=" * W,
        "",
        "OVERALL SYSTEM PERFORMANCE (macro-averaged)",
        f"  Accuracy  : {o_acc*100:.2f}%",
        f"  Precision : {o_prec:.4f}",
        f"  Recall    : {o_rec:.4f}",
        f"  F1-Score  : {o_f1:.4f}",
        "",
        "-" * W,
        "RANGE-WISE ACCURACY",
        "-" * W,
        f"{'Bin':<18} {'Samples':>8} {'Accuracy':>10} {'Precision':>11} {'Recall':>9} {'F1':>9}",
        "-" * W,
    ]

    for bin_name, data in bins.items():
        acc, prec, rec, f1 = metrics(data['y_true'], data['y_pred'])
        n = len(data['y_true'])
        report_lines.append(
            f"{bin_name:<18} {n:>8} {acc*100:>9.2f}% {prec:>11.4f} {rec:>9.4f} {f1:>9.4f}"
        )

    report_lines += [
        "",
        "-" * W,
        "PER-CLASS REPORT",
        "-" * W,
        per_class_txt,
        "",
        "-" * W,
        "CONFUSION MATRIX  (rows = actual, cols = predicted)",
        "-" * W,
        cm_header,
        cm_rows,
        "",
        "=" * W,
    ]

    report_str = "\n".join(report_lines)
    print(report_str)

    report_path = os.path.join(OUTPUT_DIR, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_str)
    logger.info(f"Evaluation report saved → {report_path}")


# ============================================================
# TRAINING
# ============================================================
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device: torch.device,
):
    criterion     = nn.CrossEntropyLoss()
    optimizer     = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    early_stopping = EarlyStopping(patience=CONFIG['patience'])
    best_val_acc  = 0.0

    for epoch in range(CONFIG['num_epochs']):
        # ---- Train ----
        model.train()
        train_loss = correct = total = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred    = torch.max(outputs, 1)
            total      += lbls.size(0)
            correct    += (pred == lbls).sum().item()

        # ---- Validate ----
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs    = model(imgs)
                _, pred    = torch.max(outputs, 1)
                val_total  += lbls.size(0)
                val_correct += (pred == lbls).sum().item()

        val_acc   = val_correct / val_total
        train_acc = correct    / total
        avg_loss  = train_loss / len(train_loader)

        logger.info(
            f"Epoch [{epoch+1:>3}/{CONFIG['num_epochs']}]  "
            f"Loss: {avg_loss:.4f}  "
            f"Train Acc: {train_acc*100:.2f}%  "
            f"Val Acc: {val_acc*100:.2f}%"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            logger.info(f"  ↑ New best ({best_val_acc*100:.2f}%). Checkpoint saved.")

        early_stopping(val_acc)
        if early_stopping.early_stop:
            logger.info(f"Early stopping at epoch {epoch+1}. Best Val Acc: {best_val_acc*100:.2f}%")
            break


# ============================================================
# MAIN
# ============================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # ImageNet-normalised transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std= [0.229, 0.224, 0.225]),
    ])

    train_set = DroneDataset(CONFIG['train_dir'], transform=transform)
    test_set  = DroneDataset(CONFIG['test_dir'],  transform=transform)

    train_loader = DataLoader(train_set, batch_size=CONFIG['batch_size'], shuffle=True,  num_workers=2)
    val_loader   = DataLoader(test_set,  batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)

    # EfficientNet-B0 with 4-class head
    model = getattr(models, CONFIG['model_name'])(weights='DEFAULT')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model = model.to(device)

    logger.info(f"Model: {CONFIG['model_name']}  |  Output classes: {NUM_CLASSES}  ({', '.join(CLASS_NAMES)})")

    train_model(model, train_loader, val_loader, device)

    logger.info("Loading best checkpoint for final evaluation …")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    run_evaluation(model, test_set, device)


if __name__ == '__main__':
    main()