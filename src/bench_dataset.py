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

    def _extract_distance_b(self, filename):
        """Extracts distance from format: phantom_2m_0.png"""
        match = re.search(r'_(\d+)m_', filename)
        return float(match.group(1)) if match else None

    def _build_dataset(self):
        # 1. Read and group Folder B (YOLO) images by distance
        b_images = glob.glob(os.path.join(self.folder_b_images, '*.*'))
        b_dict = {}  # Map distance -> list of (image_path, label_path)
        
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
                drone_class = 'phantom' # Currently hardcoded based on your prompt, modify if you add classes
                
            if dist not in b_dict:
                b_dict[dist] = []
            b_dict[dist].append({
                'b_img_path': b_img_path,
                'has_drone': has_drone,
                'drone_class': drone_class
            })

        # 2. Read Folder A and match with Folder B based on distance
        a_images = glob.glob(os.path.join(self.folder_a, '*.*'))
        
        for a_img_path in a_images:
            basename = os.path.basename(a_img_path)
            dist = self._extract_distance_a(basename)
            
            if dist is not None and dist in b_dict and len(b_dict[dist]) > 0:
                # Pop an image from B to pair with A (1-to-1 matching based on distance)
                # If you have more A images than B images for a distance, this recycles B images
                b_data = b_dict[dist].pop(0) 
                b_dict[dist].append(b_data) # Put it back at the end to recycle if needed
                
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
