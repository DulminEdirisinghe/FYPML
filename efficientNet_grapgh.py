"""
EfficientNet Binary Classification Training Script
Now includes:
1. Training + validation
2. Range extraction from filename
3. Detection-rate vs range graph
4. Accuracy vs range graph

Filename examples supported:
- phantom_5m_001.png
- NO_DRONE_12m_003.png
- mavic_range_27_010.png
"""

import os
import re
import logging
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt


# ============== LOGGER ==============
def setup_logger():
    """Setup logger for training"""
    log_filename = 'classifier_train.log'
    logger = logging.getLogger('EfficientNetClassifier')
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_filename, mode='a')
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

logger = setup_logger()
# ====================================


# ============== CONFIG ==============
CONFIG = {
    'train_dir': 'data/data/old/effientnet_grapgh/train',
    'test_dir': 'data/data/old/effientnet_grapgh/test',
    'batch_size': 32,
    'num_epochs': 10,
    'learning_rate': 0.001,
    'checkpoint_path': 'efficientnet_classifier_grapgh_model.pth',
    'model_name': 'efficientnet_b0',

    # plotting
    'plot_dir': 'runs/range_analysis',
    'range_min': 0,
    'range_max': 40,
    'range_bin_size': 10
}
# ====================================


def extract_range_from_filename(file_name):
    """
    Extract numeric range (in meters) from filename.

    Supported patterns:
    - *_5m_*
    - *_12.5m_*
    - *range_27*
    - *range27*
    - *_27_meter_*

    Returns:
        float range_in_meters or None if not found
    """
    name = os.path.splitext(file_name)[0].lower()

    patterns = [
        r'(\d+(?:\.\d+)?)m\b',              # 5m, 12.5m
        r'range[_-]?(\d+(?:\.\d+)?)\b',     # range_27, range27
        r'(\d+(?:\.\d+)?)_meter\b',         # 27_meter
        r'(\d+(?:\.\d+)?)meters\b'          # 27meters
    ]

    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return float(match.group(1))

    return None


class CustomImageDataset(Dataset):
    """Custom Dataset for loading images, labels, and range"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        valid_extensions = ('.png', '.jpg', '.jpeg')

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Directory not found: {root_dir}")

        for file_name in os.listdir(root_dir):
            if file_name.lower().endswith(valid_extensions):
                img_path = os.path.join(root_dir, file_name)

                # label
                name_without_ext = os.path.splitext(file_name)[0]
                parts = name_without_ext.split('_')

                # 0 = no drone, 1 = drone
                if 'NO' in parts or 'no' in parts:
                    binary_label = 0
                else:
                    binary_label = 1

                # range
                range_m = extract_range_from_filename(file_name)

                self.samples.append({
                    'img_path': img_path,
                    'label': binary_label,
                    'range_m': range_m,
                    'file_name': file_name
                })

        logger.info(f"Loaded {len(self.samples)} images from {root_dir}")

        num_with_range = sum(1 for s in self.samples if s['range_m'] is not None)
        logger.info(f"Images with detected range info: {num_with_range}/{len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['img_path']
        image = Image.open(img_path).convert('RGB')
        label = sample['label']
        range_m = sample['range_m']

        if self.transform:
            image = self.transform(image)

        # if range is missing, store as -1
        if range_m is None:
            range_m = -1.0

        return image, label, float(range_m), sample['file_name']


def get_transforms():
    """Get image transforms for training and validation"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform


def _is_valid_path(path):
    return path is not None and str(path).strip().lower() != 'none'


def load_datasets(train_dir, test_dir, transform):
    """Load datasets"""
    logger.info("Loading datasets...")

    if not train_dir or not os.path.exists(train_dir):
        logger.error(f"Train directory not found: {train_dir}")
        return None, None, None

    full_train_dataset = CustomImageDataset(train_dir, transform=transform)
    train_dataset = full_train_dataset
    val_dataset = None
    test_dataset = None

    if _is_valid_path(test_dir) and os.path.exists(test_dir):
        test_dataset = CustomImageDataset(test_dir, transform=transform)
        val_dataset = test_dataset
        logger.info("Using TEST dataset as validation dataset.")
    else:
        total_train = len(full_train_dataset)
        train_size = int(total_train * 0.7)
        val_size = total_train - train_size

        if train_size == 0 or val_size == 0:
            logger.error("Not enough TRAIN samples to create a 70/30 split.")
            return None, None, None

        split_generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=split_generator,
        )
        logger.info("TEST path not provided. Using 70/30 split from TRAIN for train/validation.")

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    if test_dataset:
        logger.info(f"Test samples: {len(test_dataset)}")
    else:
        logger.info("Test samples: 0 (no TEST dataset provided)")

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) if train_dataset else None
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if test_dataset else None
    return train_loader, val_loader, test_loader


def create_model(device):
    """Create and initialize EfficientNet model"""
    model_name = CONFIG['model_name']
    logger.info(f"Loading {model_name}...")
    model = getattr(models, model_name)(pretrained=True)

    num_classes = 2
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    model = model.to(device)

    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels, ranges, file_names in train_loader:
        images = images.to(device)
        labels = labels.to(device, dtype=torch.long)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, ranges, file_names in val_loader:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, checkpoint_path='best_model.pth'):
    if not (train_loader and val_loader):
        logger.error("Train or Val loader not available.")
        return

    logger.info(f"\nStarting training for {num_epochs} epochs...")

    best_val_acc = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}] - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Model saved (best val acc: {best_val_acc:.2f}%)")

    logger.info("Training completed!")
    return best_val_acc


def make_range_bins(range_min=0, range_max=40, bin_size=5):
    """
    Example:
    0-5, 5-10, ..., 35-40
    """
    bins = []
    start = range_min
    while start < range_max:
        end = start + bin_size
        bins.append((start, end))
        start = end
    return bins


def get_bin_label(start, end):
    return f"{int(start)}-{int(end)}m"


def find_range_bin(value, bins):
    for start, end in bins:
        if start <= value < end:
            return (start, end)
    if value == bins[-1][1]:
        return bins[-1]
    return None


def evaluate_by_range(model, data_loader, device, config):
    """
    Evaluate per-range:
    - detection rate = predicted drone among actual drone samples
    - accuracy = overall classification accuracy in that range
    """
    model.eval()

    bins = make_range_bins(
        config['range_min'],
        config['range_max'],
        config['range_bin_size']
    )

    stats = {
        b: {
            'total': 0,
            'correct': 0,
            'actual_drone': 0,
            'predicted_drone_on_actual_drone': 0
        }
        for b in bins
    }

    with torch.no_grad():
        for images, labels, ranges, file_names in data_loader:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            preds = preds.cpu()
            labels = labels.cpu()
            ranges = ranges.cpu()

            for pred, label, range_m, fname in zip(preds, labels, ranges, file_names):
                range_m = float(range_m.item())

                # skip samples without valid range
                if range_m < 0:
                    continue

                bin_key = find_range_bin(range_m, bins)
                if bin_key is None:
                    continue

                stats[bin_key]['total'] += 1

                if pred.item() == label.item():
                    stats[bin_key]['correct'] += 1

                if label.item() == 1:  # actual drone
                    stats[bin_key]['actual_drone'] += 1
                    if pred.item() == 1:  # predicted drone
                        stats[bin_key]['predicted_drone_on_actual_drone'] += 1

    range_labels = []
    detection_rates = []
    accuracies = []
    sample_counts = []

    logger.info("\n===== Range-wise Evaluation =====")

    for b in bins:
        start, end = b
        label = get_bin_label(start, end)

        total = stats[b]['total']
        correct = stats[b]['correct']
        actual_drone = stats[b]['actual_drone']
        detected_drone = stats[b]['predicted_drone_on_actual_drone']

        accuracy = (100.0 * correct / total) if total > 0 else 0.0
        detection_rate = (100.0 * detected_drone / actual_drone) if actual_drone > 0 else 0.0

        range_labels.append(label)
        detection_rates.append(detection_rate)
        accuracies.append(accuracy)
        sample_counts.append(total)

        logger.info(
            f"{label}: total={total}, accuracy={accuracy:.2f}%, "
            f"actual_drone={actual_drone}, detection_rate={detection_rate:.2f}%"
        )

    return range_labels, detection_rates, accuracies, sample_counts


def plot_metric_vs_range(range_labels, values, ylabel, title, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(range_labels, values, marker='o')
    plt.xlabel("Range")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved plot: {save_path}")


def plot_sample_count_vs_range(range_labels, sample_counts, save_path):
    plt.figure(figsize=(10, 6))
    plt.bar(range_labels, sample_counts)
    plt.xlabel("Range")
    plt.ylabel("Number of Samples")
    plt.title("Sample Count per Range")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved plot: {save_path}")


def run_range_analysis(model, data_loader, device, config):
    os.makedirs(config['plot_dir'], exist_ok=True)

    range_labels, detection_rates, accuracies, sample_counts = evaluate_by_range(
        model, data_loader, device, config
    )

    plot_metric_vs_range(
        range_labels,
        detection_rates,
        ylabel="Detection Rate (%)",
        title="Detection Rate vs Range",
        save_path=os.path.join(config['plot_dir'], 'detection_rate_vs_range.png')
    )

    plot_metric_vs_range(
        range_labels,
        accuracies,
        ylabel="Accuracy (%)",
        title="Accuracy vs Range",
        save_path=os.path.join(config['plot_dir'], 'accuracy_vs_range.png')
    )

    plot_sample_count_vs_range(
        range_labels,
        sample_counts,
        save_path=os.path.join(config['plot_dir'], 'sample_count_vs_range.png')
    )


def main(config):
    logger.info("=" * 70)
    logger.info(f"Starting EfficientNet Classification Training - {datetime.now()}")
    logger.info("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Config: {config}")

    transform = get_transforms()

    train_dataset, val_dataset, test_dataset = load_datasets(
        config['train_dir'], config.get('test_dir'), transform
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, config['batch_size']
    )

    model = create_model(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        config['num_epochs'],
        device,
        config['checkpoint_path']
    )

    # load best model before analysis
    if os.path.exists(config['checkpoint_path']):
        model.load_state_dict(torch.load(config['checkpoint_path'], map_location=device))
        logger.info(f"Loaded best model from {config['checkpoint_path']}")

    # Use test_loader if available, otherwise val_loader
    analysis_loader = test_loader if test_loader is not None else val_loader

    if analysis_loader is not None:
        run_range_analysis(model, analysis_loader, device, config)
    else:
        logger.warning("No validation/test loader available for range analysis.")


if __name__ == '__main__':
    main(CONFIG)