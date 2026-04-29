import os
import glob
import re
import random
from torch.utils.data import Dataset

class DronePairDataset(Dataset):
    def __init__(self, folder_a, folder_b):
        self.folder_a = folder_a
        self.folder_b_images = os.path.join(folder_b, 'images')
        self.folder_b_labels = os.path.join(folder_b, 'labels')
        
        # Mapping: Phanthom=0, MATRICE4=1, MAVIC3=2, Uncertain=4, Background=-1
        self.class_map = {0: 'Phanthom', 1: 'MATRICE4', 2: 'MAVIC3'}
        
        self.pairs = []
        self._build_dataset()

    def _extract_dist_yolo(self, filename):
        match = re.search(r'_(\d+)m_', filename)
        return float(match.group(1)) if match else None

    def _extract_class_yolo_fn(self, filename):
        if 'phantom' in filename.lower(): return 'Phanthom'
        if 'matrice' in filename.lower(): return 'MATRICE4'
        if 'mavic' in filename.lower(): return 'MAVIC3'
        raise ValueError(f"Could not determine class from YOLO filename: {filename}")

    def _extract_dist_scd(self, filename):
        match = re.search(r'_(\d+\.\d+)m', filename)
        return float(match.group(1)) if match else None

    def _extract_class_scd(self, filename):
        if 'Phanthom' in filename: return 'Phanthom'
        if 'MATRICE4' in filename: return 'MATRICE4'
        if 'MAVIC3' in filename: return 'MAVIC3'
        return 'NO_DRONE'

    def _build_dataset(self):
        # 1. Catalog SCD images for matching
        scd_images = glob.glob(os.path.join(self.folder_a, '*.*'))
        scd_catalog = {} # (dist, class_name) -> [paths]
        scd_no_drone = []

        for p in scd_images:
            fn = os.path.basename(p)
            c = self._extract_class_scd(fn)
            if c == 'NO_DRONE':
                scd_no_drone.append(p)
            else:
                d = self._extract_dist_scd(fn)
                key = (d, c)
                if key not in scd_catalog: scd_catalog[key] = []
                scd_catalog[key].append(p)

        # 2. Process YOLO images
        yolo_images = sorted(glob.glob(os.path.join(self.folder_b_images, '*.*')))
        
        for b_path in yolo_images:
            basename = os.path.basename(b_path)
            dist = self._extract_dist_yolo(basename)
            label_path = os.path.join(self.folder_b_labels, os.path.splitext(basename)[0] + '.txt')
            
            # Logic for Class ID and Matching
            if dist == 0:
                # Background Case
                class_id = -1
                if not scd_no_drone: raise ValueError("No SCD NO_DRONE images available.")
                a_path = random.choice(scd_no_drone)
            else:
                # Drone Case
                if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
                    class_id = 4 # Uncertain
                    class_name = self._extract_class_yolo_fn(basename)
                else:
                    with open(label_path, 'r') as f:
                        line = f.readline().strip()
                        class_id = int(line.split()[0])
                        class_name = self.class_map[class_id]

                # Match by distance and class
                key = (float(dist), class_name)
                if key not in scd_catalog or not scd_catalog[key]:
                    raise ValueError(f"No matching SCD image for {key} (Source: {basename})")
                
                # Pop and recycle SCD image
                a_path = scd_catalog[key].pop(0)
                scd_catalog[key].append(a_path)

            self.pairs.append({
                'a_path': a_path,
                'b_path': b_path,
                'class_id': class_id,
                'dist': dist,
                'has_drone': 1 if class_id != -1 else 0
            })

        self._print_stats()

    def _print_stats(self):
        print(f"\n--- Dataset Statistics ---")
        print(f"Total Pairs: {len(self.pairs)}")
        
        dist_stats = {}
        # build class stat dict dynamically to handle any unexpected class IDs
        class_stats = {}
        
        for p in self.pairs:
            dist_stats[p['dist']] = dist_stats.get(p['dist'], 0) + 1
            class_stats[p['class_id']] = class_stats.get(p['class_id'], 0) + 1
            
        print("\nRange Wise (Distance):")
        for d in sorted(dist_stats.keys()):
            print(f"  {d}m: {dist_stats[d]} images")
            
        print("\nClass Wise (ID):")
        class_names = {-1:'Background', 0:'Phantom', 1:'Matrice', 2:'Mavic', 4:'Uncertain'}
        for cid in sorted(class_stats.keys()):
            print(f"  Class {cid} ({class_names[cid]}): {class_stats[cid]} images")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        return p['a_path'], p['b_path'], p['class_id'], p['dist'], p['has_drone']