"""
EfficientNet Binary Classification Training Script
Simple training pipeline for EfficientNet on custom image dataset with 2 classes.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import os
from PIL import Image
import logging
from datetime import datetime


# ============== LOGGER ==============
def setup_logger():
    """Setup logger for training"""
    log_filename = 'classifier_train.log'
    logger = logging.getLogger('EfficientNetClassifier')
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
    'train_dir': './dataset/train',
    'val_dir': './dataset/val',
    'test_dir': './dataset/test',
    'batch_size': 32,
    'num_epochs': 10,
    'learning_rate': 0.001,
    'checkpoint_path': 'efficientnet_classifier_model.pth',
    'model_name': 'efficientnet_b0'
}
# ====================================


class CustomImageDataset(Dataset):
    """Custom Dataset for loading images from folder structure"""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Path to dataset folder (train/val/test)
            transform: Transforms to apply to images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        
        # Load all image paths and labels
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(class_path, img_name))
                    self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms():
    """Get image transforms for training and validation"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # EfficientNet expects 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                             std=[0.229, 0.224, 0.225])
    ])
    return transform


def load_datasets(train_dir, val_dir, test_dir, transform):
    """Load train, validation, and test datasets"""
    logger.info("Loading datasets...")
    
    train_dataset = CustomImageDataset(train_dir, transform=transform) if os.path.exists(train_dir) else None
    val_dataset = CustomImageDataset(val_dir, transform=transform) if os.path.exists(val_dir) else None
    test_dataset = CustomImageDataset(test_dir, transform=transform) if os.path.exists(test_dir) else None
    
    if train_dataset:
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Classes: {train_dataset.classes}")
    
    if val_dataset:
        logger.info(f"Val samples: {len(val_dataset)}")
    
    if test_dataset:
        logger.info(f"Test samples: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    """Create DataLoader objects"""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) if train_dataset else None
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if test_dataset else None
    
    return train_loader, val_loader, test_loader


def create_model(device):
    """Create and initialize EfficientNet model"""
    model_name = CONFIG['model_name']
    logger.info(f"Loading {model_name}...")
    model = getattr(models, model_name)(pretrained=True)
    
    # Modify the classifier for binary classification
    num_classes = 2
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    model = model.to(device)
    
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
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
    """Train the model"""
    if not (train_loader and val_loader):
        logger.error("Train or Val loader not available. Please set up the dataset first.")
        return
    
    logger.info(f"\nStarting training for {num_epochs} epochs...")
    
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Model saved (best val acc: {best_val_acc:.2f}%)")
    
    logger.info("Training completed!")
    return best_val_acc



def main(config):
    """Main training function"""
    logger.info("="*70)
    logger.info(f"Starting EfficientNet Classification Training - {datetime.now()}")
    logger.info("="*70)
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Config: {config}")
    
    # Get transforms
    transform = get_transforms()
    
    # Load datasets
    train_dataset, val_dataset, test_dataset = load_datasets(
        config['train_dir'], config['val_dir'], config['test_dir'], transform
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, config['batch_size']
    )
    
    # Create model
    model = create_model(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Train
    train(model, train_loader, val_loader, criterion, optimizer, 
          config['num_epochs'], device, config['checkpoint_path'])


if __name__ == '__main__':
    main(CONFIG)
