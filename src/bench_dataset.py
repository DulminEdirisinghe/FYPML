import os
import glob
import re
from torch.utils.data import Dataset

class DronePairDataset(Dataset):
    def __init__(self, folder_a, folder_b):
        self.folder_a = folder_a
        self.folder_b_images = os.path.join(folder_b, 'images')
        self.folder_b_labels = os.path.join(folder_b, 'labels')

        self.pairs = []
        self._build_dataset()

    def _extract_distance_a(self, filename):
        """Extracts distance from format: scd_Phanthom_1_1.00m.png"""
        match = re.search(r'_(\d+\.\d+)m', filename)
        return float(match.group(1)) if match else None

    def _extract_class_a(self, filename):
        """Extracts class from format: scd_Phanthom_1_1.00m.png"""
        # Mapping for normalizing class names
        class_map = {
            'Phanthom': 'Phanthom',
            'MATRICE4': 'MATRICE4',
            'MAVIC3': 'MAVIC3'
        }
        for k in class_map:
            if k in filename:
                return class_map[k]
        return None

    def _extract_distance_b(self, filename):
        """Extracts distance from format: phantom_2m_0.png"""
        match = re.search(r'_(\d+)m_', filename)
        return float(match.group(1)) if match else None

    def _build_dataset(self):
        # 1. Read and group Folder B (YOLO) images by distance and class
        b_images = glob.glob(os.path.join(self.folder_b_images, '*.*'))
        b_dict = {}  # Map (distance, drone_class) -> list of (image_path, label_path)
        
        for b_img_path in b_images:
            basename = os.path.basename(b_img_path)
            dist = self._extract_distance_b(basename)
            if dist is None:
                continue
            
            # Construct expected label path
            label_basename = os.path.splitext(basename)[0] + '.txt'
            label_path = os.path.join(self.folder_b_labels, label_basename)
            
            # Determine Ground Truth
            # If label file is missing or empty, it's background (no drone)
            has_drone = False
            drone_class = None
            if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                has_drone = True
                # Map YOLO class IDs to class names
                # 0: Phantom, 1: Matrice, 2: Mavic
                class_map = {0: 'Phanthom', 1: 'MATRICE4', 2: 'MAVIC3'}
                try:
                    with open(label_path, 'r') as f:
                        line = f.readline().strip()
                        if line:
                            class_id = int(line.split()[0])
                            drone_class = class_map.get(class_id)
                except Exception:
                    raise ValueError(f"Error reading label file: {label_path}")
                
            key = (dist, drone_class)
            if key not in b_dict:
                b_dict[key] = []
            b_dict[key].append({
                'b_img_path': b_img_path,
                'has_drone': has_drone,
                'drone_class': drone_class
            })

        # 2. Read Folder A and match with Folder B based on distance and class
        a_images = glob.glob(os.path.join(self.folder_a, '*.*'))
        
        for a_img_path in a_images:
            basename = os.path.basename(a_img_path)
            dist = self._extract_distance_a(basename)
            drone_class = self._extract_class_a(basename)
            
            key = (dist, drone_class)
            
            if dist is not None and key in b_dict and len(b_dict[key]) > 0:
                # Pop an image from B to pair with A (1-to-1 matching based on distance and class)
                # If you have more A images than B images for a distance/class, this recycles B images
                b_data = b_dict[key].pop(0) 
                b_dict[key].append(b_data) # Put it back at the end to recycle if needed
                
                self.pairs.append({
                    'a_img_path': a_img_path,
                    'b_img_path': b_data['b_img_path'],
                    'distance': dist,
                    'gt_has_drone': b_data['has_drone'],
                    'gt_drone_class': b_data['drone_class']
                })

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]
