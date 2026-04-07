import os
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
from src.bench_dataset import DronePairDataset

# Import your existing functions from your main detector file
# Assuming your previous code is saved as `detector_backend.py`
from pipeline import load_all_models, detect, CONFIG

# ============== CONFIGURATIONS ==============
FOLDER_A_TEST = 'data/data/SCD_Images/test'
FOLDER_B_TEST = 'data/data/YOLO_data/stationary/test'





# ============== BENCHMARK LOOP ==============
def run_benchmark():
    print("Loading models...")
    classifier, yolo_model, transform, device = load_all_models()
    
    print("Building Dataset...")
    dataset = DronePairDataset(FOLDER_A_TEST, FOLDER_B_TEST)
    print(f"Total valid pairs found: {len(dataset)}")
    
    if len(dataset) == 0:
        print("No pairs matched. Please check your folder paths and naming conventions.")
        return

    # Tracking lists for metrics
    y_true_binary = []
    y_pred_binary = []
    
    y_true_class = []
    y_pred_class = []

    print("\nStarting Evaluation...")
    
    for i in range(len(dataset)):
        data = dataset[i]
        
        # 1. Run inference
        result = detect(
            data['a_img_path'], 
            data['b_img_path'], 
            classifier, 
            yolo_model, 
            transform, 
            device
        )
        
        # 2. Extract Ground Truths
        gt_is_drone = data['gt_has_drone']
        gt_class = data['gt_drone_class'] if gt_is_drone else 'no_drone'
        
        # 3. Extract Predictions based on your fusion rules
        pred_is_drone = result['final_decision'] in ['Detected', 'DroneType']
        
        if result['final_decision'] == 'DroneType':
            pred_class = result['drone_type']
        elif result['final_decision'] == 'Detected':
            pred_class = 'uncertain'
        else:
            pred_class = 'no_drone'
            
        # 4. Append to lists
        y_true_binary.append(gt_is_drone)
        y_pred_binary.append(pred_is_drone)
        
        y_true_class.append(gt_class)
        y_pred_class.append(pred_class)
        
        # Optional: Print progress every 10 images
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(dataset)} pairs...")

    # ============== CALCULATE METRICS ==============
    print("\n" + "="*50)
    print("🎯 BENCHMARK RESULTS")
    print("="*50)
    
    # 1. Binary Accuracy (Drone vs No Drone)
    bin_acc = accuracy_score(y_true_binary, y_pred_binary)
    print(f"\n1. Drone / No Drone Detection Accuracy: {bin_acc * 100:.2f}%")
    
    # 2. Classification Accuracy / Certainty
    # We evaluate this only on the images where a drone was ACTUALLY present
    true_drones_indices = [i for i, x in enumerate(y_true_binary) if x == True]
    
    if len(true_drones_indices) > 0:
        y_true_drones_only = [y_true_class[i] for i in true_drones_indices]
        y_pred_drones_only = [y_pred_class[i] for i in true_drones_indices]
        
        class_acc = accuracy_score(y_true_drones_only, y_pred_drones_only)
        print(f"\n2. Class Certainty Accuracy (When GT is Drone): {class_acc * 100:.2f}%")
        
        print("\nDetailed Class Report (On Actual Drones):")
        print(classification_report(y_true_drones_only, y_pred_drones_only, zero_division=0))
    else:
        print("\n2. Class Certainty Accuracy: N/A (No ground truth drones found in dataset)")

if __name__ == "__main__":
    run_benchmark()