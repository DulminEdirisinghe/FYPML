import os
import glob
import re
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
from src.bench_dataset import DronePairDataset

# Import your existing functions from your main detector file
from pipeline import load_all_models, detect, CONFIG

# ============== CONFIGURATIONS ==============
FOLDER_A_TEST = 'data/data/SCD_Images/test'
FOLDER_B_TEST = 'data/data/YOLO_data/stationary/test'

PLOT_SAVE_DIR = 'runs/benchmark_plots'  # Folder where all pair plots will be saved


# ============== PLOTTING UTILITY ==============
def save_pair_plot(res, idx, save_dir):
    """Plots a single pair side-by-side and saves it based on accuracy category."""
    
    # 1. Determine the accuracy category
    gt_is_drone = res['gt_is_drone']
    pred_is_drone = res['pred_is_drone']
    gt_class = res['gt_class']
    pred_class = res['pred_class']

    if gt_is_drone != pred_is_drone:
        category = "detection_error"
        color = "red"
    elif pred_class == 'uncertain':
        category = "class_uncertainty"
        color = "orange"
    elif gt_class != pred_class:
        category = "class_error"
        color = "red"
    else:
        category = "all_correct"
        color = "green"

    # 2. Setup the plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Title with GT, Pred, Scores, and Distance
    dist_str = f"{res['distance']}m" if res.get('distance') is not None else "Unknown Dist"
    title = (f"Status: {category.upper()} | Range: {dist_str}\n"
             f"GT: {gt_class} | Pred: {pred_class} | Fusion (G): {res['G']:.3f}")
    fig.suptitle(title, fontsize=14, fontweight='bold', color=color)

    # Load images
    img_a = Image.open(res['a_img_path']).convert('RGB')
    
    # Use bounding box image if available, else original B
    img_b_path = res.get('saved_detection_image', res['b_img_path'])
    img_b = Image.open(img_b_path).convert('RGB')

    # Display paths (keeping just the filename to prevent the text from overlapping)
    path_a_text = os.path.basename(res['a_img_path'])
    path_b_text = os.path.basename(res['b_img_path'])

    # Plot A (EfficientNet)
    axes[0].imshow(img_a)
    axes[0].axis('off')
    axes[0].set_title(f"A: EfficientNet (P4: {res['P4']:.3f})\nFile: {path_a_text}", fontsize=11)

    # Plot B (YOLO)
    axes[1].imshow(img_b)
    axes[1].axis('off')
    axes[1].set_title(f"B: YOLO (F: {res['F']:.3f})\nFile: {path_b_text}", fontsize=11)

    plt.tight_layout()
    
    # 3. Save the figure (added distance to filename for easier sorting)
    dist_prefix = f"{res['distance']}m_" if res.get('distance') is not None else ""
    filename = f"{category}_{dist_prefix}{idx:04d}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close(fig)  # Important to prevent memory leaks!


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

    # Tracking lists for overall metrics
    y_true_binary = []
    y_pred_binary = []
    y_true_class = []
    y_pred_class = []
    
    detailed_results = []
    
    # Tracking dictionary for RANGE-WISE metrics
    range_metrics = {}

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
        dist = data.get('distance', 'Unknown')
        
        # 3. Extract Predictions
        pred_is_drone = result['final_decision'] in ['Detected', 'DroneType']
        
        if result['final_decision'] == 'DroneType':
            pred_class = result['drone_type']
        elif result['final_decision'] == 'Detected':
            pred_class = 'uncertain'
        else:
            pred_class = 'no_drone'
            
        # 4. Append to overall metric lists
        y_true_binary.append(gt_is_drone)
        y_pred_binary.append(pred_is_drone)
        y_true_class.append(gt_class)
        y_pred_class.append(pred_class)
        
        # 5. Track Range-wise Metrics
        if dist not in range_metrics:
            range_metrics[dist] = {
                'total': 0, 'bin_correct': 0, 'cls_correct': 0, 
                'cls_total': 0, 'sum_F': 0.0, 'sum_G': 0.0
            }
            
        range_metrics[dist]['total'] += 1
        range_metrics[dist]['sum_F'] += result['F']
        range_metrics[dist]['sum_G'] += result['G']
        
        if gt_is_drone == pred_is_drone:
            range_metrics[dist]['bin_correct'] += 1
            
        if gt_is_drone:
            range_metrics[dist]['cls_total'] += 1
            if gt_class == pred_class:
                range_metrics[dist]['cls_correct'] += 1
        
        # 6. Save detailed results for plotting
        detailed_results.append({
            'a_img_path': data['a_img_path'],
            'b_img_path': data['b_img_path'],
            'saved_detection_image': result.get('saved_detection_image', data['b_img_path']),
            'gt_is_drone': gt_is_drone,
            'pred_is_drone': pred_is_drone,
            'gt_class': gt_class,
            'pred_class': pred_class,
            'G': result['G'],
            'F': result['F'],
            'P4': result['P4'],
            'distance': dist
        })
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(dataset)} pairs...")

    # ============== CALCULATE OVERALL METRICS ==============
    print("\n" + "="*50)
    print("🎯 OVERALL BENCHMARK RESULTS")
    print("="*50)
    
    bin_acc = accuracy_score(y_true_binary, y_pred_binary)
    print(f"\n1. Drone / No Drone Detection Accuracy: {bin_acc * 100:.2f}%")
    
    true_drones_indices = [i for i, x in enumerate(y_true_binary) if x == True]
    
    if len(true_drones_indices) > 0:
        y_true_drones_only = [y_true_class[i] for i in true_drones_indices]
        y_pred_drones_only = [y_pred_class[i] for i in true_drones_indices]
        
        class_acc = accuracy_score(y_true_drones_only, y_pred_drones_only)
        print(f"\n2. Class Certainty Accuracy (When GT is Drone): {class_acc * 100:.2f}%")
    else:
        print("\n2. Class Certainty Accuracy: N/A")

    # ============== CALCULATE RANGE-WISE METRICS ==============
    print("\n" + "="*60)
    print("📊 RANGE-WISE PERFORMANCE")
    print("="*60)
    print(f"{'Dist(m)':<8} | {'Pairs':<6} | {'Bin Acc':<8} | {'Cls Acc':<8} | {'Avg F':<6} | {'Avg G':<6}")
    print("-" * 60)
    
    # Sort numeric distances first, then any "Unknown" strings at the end
    sorted_dists = sorted([d for d in range_metrics.keys() if isinstance(d, (int, float))])
    sorted_dists += [d for d in range_metrics.keys() if not isinstance(d, (int, float))]

    for dist in sorted_dists:
        rm = range_metrics[dist]
        bin_acc = (rm['bin_correct'] / rm['total']) * 100
        cls_acc_str = "N/A"
        if rm['cls_total'] > 0:
            cls_acc = (rm['cls_correct'] / rm['cls_total']) * 100
            cls_acc_str = f"{cls_acc:6.2f}%"
            
        avg_F = rm['sum_F'] / rm['total']
        avg_G = rm['sum_G'] / rm['total']
        
        dist_label = f"{dist}" if isinstance(dist, str) else f"{dist:.1f}"
        print(f"{dist_label:<8} | {rm['total']:<6} | {bin_acc:6.2f}% | {cls_acc_str:<8} | {avg_F:5.3f} | {avg_G:5.3f}")

    # ============== GENERATE ALL PLOTS ==============
    print(f"\nGenerating plots for all {len(detailed_results)} pairs...")
    os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
    
    for idx, res in enumerate(detailed_results):
        save_pair_plot(res, idx, PLOT_SAVE_DIR)
        if (idx + 1) % 50 == 0:
            print(f"Saved {idx + 1}/{len(detailed_results)} plots...")

    print(f"\n✅ All plots saved successfully in the '{PLOT_SAVE_DIR}' directory.")


if __name__ == "__main__":
    run_benchmark()