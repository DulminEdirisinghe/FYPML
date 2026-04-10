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
FOLDER_A_TEST = 'data/data/v2/SCD_images/test'
FOLDER_B_TEST = 'data/data/v2/YOLO/stationary/test'

PLOT_SAVE_DIR = 'runs/benchmark_v2'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_TXT_PATH = os.path.join(SCRIPT_DIR, 'benchmark_results.txt')
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
    
    dist_str = f"{res['distance']}m" if res.get('distance') is not None else "Unknown Dist"
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
    dist_prefix = f"{res['distance']}m_" if res.get('distance') is not None else ""
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
    print(f"Total valid pairs found: {len(dataset)}")
    
    if len(dataset) == 0:
        return

    # Metrics Tracking
    y_true_binary = []
    y_pred_binary = []
    
    # Strict Pipeline Tracking
    pipeline_correct_count = 0
    total_pairs = len(dataset)

    # Error Diagnostic Tracking for F
    f_scores_misses = []         # GT=Drone, Pred=NoDrone
    f_scores_false_detects = []  # GT=NoDrone, Pred=Drone

    detailed_results = []
    range_metrics = {}
    results_lines = []

    print("\nStarting Evaluation...")
    results_lines.append("Starting Evaluation...\n")
    
    for i in range(total_pairs):
        data = dataset[i]
        result = detect(data['a_img_path'], data['b_img_path'], classifier, yolo_model, transform, device)
        
        gt_is_drone = data['gt_has_drone']
        gt_class = data['gt_drone_class'] if gt_is_drone else 'no_drone'
        dist = data.get('distance', 'Unknown')
        
        pred_is_drone = result['final_decision'] in ['Detected', 'DroneType']
        if result['final_decision'] == 'DroneType':
            pred_class = result['drone_type']
        elif result['final_decision'] == 'Detected':
            pred_class = 'uncertain'
        else:
            pred_class = 'no_drone'

        gt_class_eval = gt_class.lower() if isinstance(gt_class, str) else gt_class
        pred_class_eval = pred_class.lower() if isinstance(pred_class, str) else pred_class
            
        y_true_binary.append(gt_is_drone)
        y_pred_binary.append(pred_is_drone)

        # Evaluate Strict Pipeline Accuracy
        is_pipeline_correct = False
        if gt_is_drone == pred_is_drone:
            if not gt_is_drone:
                is_pipeline_correct = True  # Correctly identified 'no drone'
            else:
                if gt_class_eval == pred_class_eval:
                    is_pipeline_correct = True  # Correctly identified specific drone
        
        if is_pipeline_correct:
            pipeline_correct_count += 1

        # Evaluate YOLO Error Diagnostics
        if gt_is_drone and not pred_is_drone:
            f_scores_misses.append(result['F'])
        elif not gt_is_drone and pred_is_drone:
            f_scores_false_detects.append(result['F'])

        # Range Metrics Tracking
        if dist not in range_metrics:
            range_metrics[dist] = {'total': 0, 'pipeline_correct': 0}
            
        range_metrics[dist]['total'] += 1
        if is_pipeline_correct:
            range_metrics[dist]['pipeline_correct'] += 1
        
        detailed_results.append({
            'a_img_path': data['a_img_path'], 'b_img_path': data['b_img_path'],
            'saved_detection_image': result.get('saved_detection_image', data['b_img_path']),
            'gt_is_drone': gt_is_drone, 'pred_is_drone': pred_is_drone,
            'gt_class': gt_class, 'pred_class': pred_class,
            'G': result['G'], 'F': result['F'], 'P4': result['P4'], 'distance': dist,
            'is_pipeline_correct': is_pipeline_correct
        })
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{total_pairs} pairs...")

    # ============== CALCULATE OVERALL METRICS ==============
    header_overall = "\n" + "="*50
    print(header_overall)
    print("🎯 OVERALL BENCHMARK RESULTS")
    print("="*50)
    results_lines.extend([header_overall, "🎯 OVERALL BENCHMARK RESULTS", "="*50])

    bin_acc = accuracy_score(y_true_binary, y_pred_binary)
    strict_acc = (pipeline_correct_count / total_pairs) * 100

    line_bin = f"1. Drone / No Drone Detection Accuracy: {bin_acc * 100:.2f}%"
    line_strict = f"2. Strict End-to-End Pipeline Accuracy: {strict_acc:.2f}%"
    
    print(f"\n{line_bin}\n{line_strict}")
    results_lines.extend(["\n" + line_bin, line_strict])

    # F-Score Diagnostics
    avg_f_misses = np.mean(f_scores_misses) if f_scores_misses else 0.0
    avg_f_false_detects = np.mean(f_scores_false_detects) if f_scores_false_detects else 0.0

    diag_header = "\n⚠️ YOLO F-SCORE DIAGNOSTICS"
    line_miss = f"- Avg F on MISSES (False Negatives): {avg_f_misses:.3f} (over {len(f_scores_misses)} errors)"
    line_false = f"- Avg F on FALSE DETECTS (False Positives): {avg_f_false_detects:.3f} (over {len(f_scores_false_detects)} errors)"

    print(f"{diag_header}\n{line_miss}\n{line_false}")
    results_lines.extend([diag_header, line_miss, line_false])

    # ============== CALCULATE RANGE-WISE METRICS ==============
    header_range_1 = "\n" + "="*60
    print(header_range_1)
    print("📊 RANGE-WISE STRICT PIPELINE PERFORMANCE")
    print("="*60)
    header_range_2 = f"{'Dist(m)':<8} | {'Pairs':<6} | {'Strict Acc':<10}"
    sep_line = "-" * 60
    print(f"{header_range_2}\n{sep_line}")

    results_lines.extend([header_range_1, "📊 RANGE-WISE STRICT PIPELINE PERFORMANCE", "="*60, header_range_2, sep_line])
    
    sorted_dists = sorted([d for d in range_metrics.keys() if isinstance(d, (int, float))])
    sorted_dists += [d for d in range_metrics.keys() if not isinstance(d, (int, float))]

    for dist in sorted_dists:
        rm = range_metrics[dist]
        strict_acc_range = (rm['pipeline_correct'] / rm['total']) * 100
        dist_label = f"{dist}" if isinstance(dist, str) else f"{dist:.1f}"
        line_range = f"{dist_label:<8} | {rm['total']:<6} | {strict_acc_range:6.2f}%"
        print(line_range)
        results_lines.append(line_range)

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