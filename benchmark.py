import os
import glob
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from src.bench_dataset import DronePairDataset

from pipeline import load_all_models, detect, CONFIG

# ============== CONFIGURATIONS ==============
FOLDER_A_TEST = 'data/data/v3/SCD_images/test'
FOLDER_B_TEST = 'data/data/v3/YOLO/stationary/test'

PLOT_SAVE_DIR = 'runs/benchmark_v3'
RESULTS_TXT_PATH = os.path.join(PLOT_SAVE_DIR, 'benchmark_results.txt')
K_EXAMPLES = 5


# ============== PLOTTING UTILITY ==============
def save_pair_plot(res, idx, save_dir):
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    dist_str = f"{res['distance']}" if res.get('distance') is not None else "Unknown Dist"
    title = (f"Status: {category.upper()} | Range: {dist_str}\n"
             f"GT: {gt_class} | Pred: {pred_class} | Fusion (G): {res['G']:.3f}")
    fig.suptitle(title, fontsize=14, fontweight='bold', color=color)

    img_a = Image.open(res['a_img_path']).convert('RGB')
    img_b_path = res.get('saved_detection_image', res['b_img_path'])
    img_b = Image.open(img_b_path).convert('RGB')

    path_a_text = os.path.basename(res['a_img_path'])
    path_b_text = os.path.basename(res['b_img_path'])

    axes[0].imshow(img_a)
    axes[0].axis('off')
    axes[0].set_title(f"A: EfficientNet (P4: {res['P4']:.3f})\nFile: {path_a_text}", fontsize=11)

    axes[1].imshow(img_b)
    axes[1].axis('off')
    axes[1].set_title(f"B: YOLO (F: {res['F']:.3f})\nFile: {path_b_text}", fontsize=11)

    plt.tight_layout()
    dist_prefix = f"{res['distance']}_" if res.get('distance') is not None else ""
    filename = f"{category}_{dist_prefix}{idx:04d}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close(fig)


# ============== BENCHMARK LOOP ==============
def run_benchmark():
    print("Loading models...")
    classifier, yolo_model, transform, device = load_all_models()
    
    print("Building Dataset...")
    dataset = DronePairDataset(FOLDER_A_TEST, FOLDER_B_TEST)
    total_pairs = len(dataset)
    print(f"Total valid pairs found: {total_pairs}")
    
    if total_pairs == 0:
        return

    # Tracking Dictionaries
    def create_stats_dict():
        return {
            'gt_no_drone': 0, 'gt_drone': 0,
            'pred_no_drone': 0, 'pred_drone': 0,
            'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
            'class_certain_correct': 0
        }

    overall_stats = create_stats_dict()
    range_stats = {}
    detailed_results = []
    results_lines = []

    print("\nStarting Evaluation...")
    results_lines.append("Starting Evaluation...\n")
    
    for i in range(total_pairs):
        data = dataset[i]
        
        # 1. Image Paths & Base Predictions
        a_path = data['a_img_path']
        b_path = data['b_img_path']
        result = detect(a_path, b_path, classifier, yolo_model, transform, device)
        
        a_filename = os.path.basename(a_path).lower()
        b_filename = os.path.basename(b_path).lower()

        # 2. Extract Distance using Regex (Search both filenames to be safe)
        dist_val = 0.0
        dist_match = re.search(r'(\d+(?:\.\d+)?)m', a_filename)
        if not dist_match:
            dist_match = re.search(r'(\d+(?:\.\d+)?)m', b_filename)
            
        if dist_match:
            dist_val = float(dist_match.group(1))
            
        dist_str = f"{dist_val:g}m" # Formats cleanly (e.g. 2.00m -> 2m)

        # 3. Override Ground Truth based strictly on extracted distance
        if dist_val == 0.0:
            gt_is_drone = False
            gt_class = 'no_drone'
        else:
            gt_is_drone = True
            if 'phantom' in a_filename or 'phanthom' in a_filename:
                gt_class = 'phantom'
            else:
                gt_class = 'phantom'

        # 4. Resolve Prediction Decisions
        pred_is_drone = result['final_decision'] in ['Detected', 'DroneType']
        if result['final_decision'] == 'DroneType':
            pred_class = result['drone_type'].lower()
        elif result['final_decision'] == 'Detected':
            pred_class = 'uncertain'
        else:
            pred_class = 'no_drone'

        # 5. Determine Strict Pipeline Correctness
        is_pipeline_correct = False
        if gt_is_drone == pred_is_drone:
            if not gt_is_drone:
                is_pipeline_correct = True
            else:
                if gt_class == pred_class:
                    is_pipeline_correct = True

        # 6. Update Tracking Statistics
        def update_stats(stats_dict, gt_is, pred_is, gt_c, pred_c):
            stats_dict['gt_drone'] += int(gt_is)
            stats_dict['gt_no_drone'] += int(not gt_is)
            stats_dict['pred_drone'] += int(pred_is)
            stats_dict['pred_no_drone'] += int(not pred_is)
            
            if gt_is and pred_is:
                stats_dict['tp'] += 1
                if gt_c == pred_c:
                    stats_dict['class_certain_correct'] += 1
            elif gt_is and not pred_is:
                stats_dict['fn'] += 1
            elif not gt_is and pred_is:
                stats_dict['fp'] += 1
            elif not gt_is and not pred_is:
                stats_dict['tn'] += 1

        update_stats(overall_stats, gt_is_drone, pred_is_drone, gt_class, pred_class)

        if dist_str not in range_stats:
            range_stats[dist_str] = create_stats_dict()
        update_stats(range_stats[dist_str], gt_is_drone, pred_is_drone, gt_class, pred_class)

        detailed_results.append({
            'a_img_path': a_path, 'b_img_path': b_path,
            'saved_detection_image': result.get('saved_detection_image', b_path),
            'gt_is_drone': gt_is_drone, 'pred_is_drone': pred_is_drone,
            'gt_class': gt_class, 'pred_class': pred_class,
            'G': result['G'], 'F': result['F'], 'P4': result['P4'], 'distance': dist_str,
            'is_pipeline_correct': is_pipeline_correct
        })
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{total_pairs} pairs...")

    # ============== METRICS CALCULATION UTILITY ==============
    def calculate_metrics(stats):
        tp, fp, tn, fn = stats['tp'], stats['fp'], stats['tn'], stats['fn']
        gt_dr, gt_no_dr = stats['gt_drone'], stats['gt_no_drone']
        cc_correct = stats['class_certain_correct']

        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        
        no_drone_acc = (tn / gt_no_dr * 100) if gt_no_dr > 0 else 0.0
        drone_acc = (tp / gt_dr * 100) if gt_dr > 0 else 0.0
        class_acc = (cc_correct / gt_dr * 100) if gt_dr > 0 else 0.0

        return {
            'precision': precision, 'recall': recall,
            'no_drone_acc': no_drone_acc, 'drone_acc': drone_acc,
            'class_acc': class_acc
        }

    # ============== GENERATE CONSOLIDATED TABLE ==============
    table_lines = []
    
    header = (
        f"| {'Range/Category':<15} "
        f"| {'GT NoDr':<8} "
        f"| {'GT Dr':<6} "
        f"| {'Pred NoDr':<10} "
        f"| {'Pred Dr':<8} "
        f"| {'NoDr Acc(%)':<12} "
        f"| {'Dr Acc(%)':<10} "
        f"| {'Class Acc(%)':<13} "
        f"| {'Prec(%)':<8} "
        f"| {'Recall(%)':<9} |"
    )
    separator = "-" * len(header)
    
    table_lines.extend(["\n🎯 BENCHMARK RESULTS TABLE", separator, header, separator])
    
    def format_row(label, stats):
        m = calculate_metrics(stats)
        return (
            f"| {label:<15} "
            f"| {stats['gt_no_drone']:<8} "
            f"| {stats['gt_drone']:<6} "
            f"| {stats['pred_no_drone']:<10} "
            f"| {stats['pred_drone']:<8} "
            f"| {m['no_drone_acc']:>12.2f} "
            f"| {m['drone_acc']:>10.2f} "
            f"| {m['class_acc']:>13.2f} "
            f"| {m['precision']:>8.2f} "
            f"| {m['recall']:>9.2f} |"
        )
    
    # Add Overall Row
    table_lines.append(format_row("OVERALL", overall_stats))
    table_lines.append(separator)
    
    # Sort Range Rows (Numeric sort to handle '0m', '10m', '2m' correctly)
    def dist_sorter(x):
        try:
            return float(x.replace('m', ''))
        except ValueError:
            return 9999.0

    sorted_dists = sorted(range_stats.keys(), key=dist_sorter)
    for dist in sorted_dists:
        table_lines.append(format_row(dist, range_stats[dist]))
        
    table_lines.append(separator)
    
    table_output = "\n".join(table_lines)
    print(table_output)
    results_lines.append(table_output)

    # ============== SAVE RESULTS TO TXT ==============
    os.makedirs(os.path.dirname(RESULTS_TXT_PATH), exist_ok=True)
    with open(RESULTS_TXT_PATH, 'w') as f:
        f.write("\n".join(results_lines))

    print(f"\nBenchmark results written to: {RESULTS_TXT_PATH}")

    # ============== GENERATE LIMITED PLOTS ==============
    print(f"\nGenerating up to {K_EXAMPLES} correct and {K_EXAMPLES} wrong example plots...")
    os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

    correct_examples = [r for r in detailed_results if r['is_pipeline_correct']][:K_EXAMPLES]
    wrong_examples = [r for r in detailed_results if not r['is_pipeline_correct']][:K_EXAMPLES]

    for idx, res in enumerate(correct_examples):
        save_pair_plot(res, idx, PLOT_SAVE_DIR)
    for idx, res in enumerate(wrong_examples):
        save_pair_plot(res, idx + K_EXAMPLES, PLOT_SAVE_DIR)

    print(f"\n✅ Saved {len(correct_examples)} correct and {len(wrong_examples)} wrong plots in '{PLOT_SAVE_DIR}'.")

if __name__ == "__main__":
    run_benchmark()