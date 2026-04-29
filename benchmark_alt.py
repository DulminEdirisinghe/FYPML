import os
import glob
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from src.bench_dataset import DronePairDataset

# Import only what is needed for YOLO
from pipeline_all import load_yolo_model, run_yolo, CONFIG

# ============== CONFIGURATIONS ==============
FOLDER_A_TEST = 'data/v5/SCD_images/test'
FOLDER_B_TEST = 'data/v5/YOLO/stationary/test'

PLOT_SAVE_DIR = 'runs/benchmark_3test'
RESULTS_TXT_PATH = os.path.join(PLOT_SAVE_DIR, 'benchmark_test.txt')
K_EXAMPLES = 20

# Confidence threshold for YOLO detection
YOLO_CONF_THRESH = 0.80  


# ============== PLOTTING UTILITY ==============
def save_yolo_plot(res, idx, save_dir):
    gt_is_drone = res['gt_is_drone']
    pred_is_drone = res['pred_is_drone']
    gt_class = str(res['gt_class']).lower()
    pred_class = str(res['pred_class']).lower()

    # Class mapping for flexible matching (gt_normalized: [valid_preds])
    alias_map = {
        'phanthom': ['phantom', 'phanthom'],
        'matrice4': ['matrice', 'matrice4'],
        'mavic3': ['mavic', 'mavic3']
    }

    is_class_correct = (gt_class == pred_class)
    if not is_class_correct and gt_class in alias_map:
        if pred_class in alias_map[gt_class]:
            is_class_correct = True

    if gt_is_drone != pred_is_drone:
        category = "detection_error"
        color = "red"
    elif not is_class_correct and pred_is_drone:
        category = "class_error"
        color = "red"
    else:
        category = "all_correct"
        color = "green"

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    dist_str = f"{res['distance']}" if res.get('distance') is not None else "Unknown Dist"
    title = (f"YOLO ONLY | Status: {category.upper()} | Range: {dist_str}\n"
             f"GT: {gt_class} | Pred: {pred_class} | Conf (F): {res['F']:.3f}")
    fig.suptitle(title, fontsize=14, fontweight='bold', color=color)

    img_b_path = res.get('saved_detection_image', res['b_img_path'])
    img_b = Image.open(img_b_path).convert('RGB')
    path_b_text = os.path.basename(res['b_img_path'])

    ax.imshow(img_b)
    ax.axis('off')
    ax.set_title(f"File: {path_b_text}", fontsize=11)

    plt.tight_layout()
    dist_prefix = f"{res['distance']}_" if res.get('distance') is not None else ""
    filename = f"{category}_{dist_prefix}{idx:04d}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close(fig)


# ============== BENCHMARK LOOP ==============
def run_benchmark():
    print("Loading YOLO model...")
    yolo_model = load_yolo_model()
    
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

    print(f"\nStarting YOLO-Only Evaluation (Threshold = {YOLO_CONF_THRESH})...")
    results_lines.append(f"Starting YOLO-Only Evaluation (Threshold = {YOLO_CONF_THRESH})...\n")
    
    for i in range(total_pairs):
        data = dataset[i]
        
        a_path = data['a_img_path']
        b_path = data['b_img_path']
        
        a_filename = os.path.basename(a_path).lower()
        b_filename = os.path.basename(b_path).lower()

        # 1. Extract Distance using Regex
        dist_val = 0.0
        dist_match = re.search(r'(\d+(?:\.\d+)?)m', a_filename)
        if not dist_match:
            dist_match = re.search(r'(\d+(?:\.\d+)?)m', b_filename)
            
        if dist_match:
            dist_val = float(dist_match.group(1))
            
        dist_str = f"{dist_val:g}m" 

        # 2. Determine Ground Truth from Dataset
        gt_is_drone = data['gt_has_drone']
        gt_class = data['gt_drone_class'] if data['gt_drone_class'] else 'no_drone'

        if gt_class != 'no_drone':
            gt_class = gt_class.lower()

        # 3. Direct YOLO Inference
        yolo_result = run_yolo(b_path, yolo_model)
        F = yolo_result['max_confidence']

        # 4. Resolve Prediction Decisions (YOLO ONLY)
        if F > YOLO_CONF_THRESH:
            pred_is_drone = True
            pred_class = yolo_result['best_class_name'].lower() if yolo_result['best_class_name'] else 'uncertain'
        else:
            pred_is_drone = False
            pred_class = 'uncertain'

        # 5. Determine Strict Pipeline Correctness
        is_pipeline_correct = False
        if gt_is_drone == pred_is_drone:
            if not gt_is_drone:
                is_pipeline_correct = True
            else:
                # Class mapping for flexible matching (gt_normalized: [valid_preds])
                alias_map = {
                    'phanthom': ['phantom', 'phanthom'],
                    'matrice4': ['matrice', 'matrice4'],
                    'mavic3': ['mavic', 'mavic3']
                }
                
                gt_normalized = gt_class.lower()
                pred_normalized = pred_class.lower()
                
                if gt_normalized == pred_normalized:
                    is_pipeline_correct = True
                elif gt_normalized in alias_map and pred_normalized in alias_map[gt_normalized]:
                    is_pipeline_correct = True

        # 6. Update Tracking Statistics
        def update_stats(stats_dict, gt_is, pred_is, gt_c, pred_c):
            stats_dict['gt_drone'] += int(gt_is)
            stats_dict['gt_no_drone'] += int(not gt_is)
            stats_dict['pred_drone'] += int(pred_is)
            stats_dict['pred_no_drone'] += int(not pred_is)
            
            if gt_is and pred_is:
                stats_dict['tp'] += 1
                
                # Use same mapping for certain correct
                alias_map = {
                    'phanthom': ['phantom', 'phanthom'],
                    'matrice4': ['matrice', 'matrice4'],
                    'mavic3': ['mavic', 'mavic3']
                }
                gt_n = gt_c.lower()
                pred_n = pred_c.lower()
                
                if gt_n == pred_n or (gt_n in alias_map and pred_n in alias_map[gt_n]):
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
            'b_img_path': b_path,
            'saved_detection_image': yolo_result.get('saved_detection_image', b_path),
            'gt_is_drone': gt_is_drone, 'pred_is_drone': pred_is_drone,
            'gt_class': gt_class, 'pred_class': pred_class,
            'F': F, 'distance': dist_str,
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
    
    table_lines.extend(["\n🎯 YOLO-ONLY BENCHMARK RESULTS", separator, header, separator])
    
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
    
    # Sort Range Rows
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

    print(f"\nYOLO-Only Benchmark results written to: {RESULTS_TXT_PATH}")

    # ============== GENERATE LIMITED PLOTS ==============
    print(f"\nGenerating up to {K_EXAMPLES} correct and {K_EXAMPLES} wrong example plots...")
    os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

    correct_examples = [r for r in detailed_results if r['is_pipeline_correct']][:K_EXAMPLES]
    wrong_examples = [r for r in detailed_results if not r['is_pipeline_correct']][:K_EXAMPLES]

    for idx, res in enumerate(correct_examples):
        save_yolo_plot(res, idx, PLOT_SAVE_DIR)
    for idx, res in enumerate(wrong_examples):
        save_yolo_plot(res, idx + K_EXAMPLES, PLOT_SAVE_DIR)

    print(f"\n✅ Saved {len(correct_examples)} correct and {len(wrong_examples)} wrong plots in '{PLOT_SAVE_DIR}'.")

if __name__ == "__main__":
    run_benchmark()