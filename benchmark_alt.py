import os
import torch
import re
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from src.bench_dataset import DronePairDataset
from pipeline_all import load_yolo_model, run_yolo, load_model, get_transforms, detect

# ============== CONFIGURATIONS ==============
FOLDER_A = 'data/v5/SCD_images/test'
FOLDER_B = 'data/v5/YOLO/stationary/test'
CONF_THRESH = 0.70
PLOT_SAVE_DIR = 'runs/result_check/benchmark_plots_fused' # Directory to save the benchmark plots
K_EXAMPLES = 10  # Number of plots to save for each category
YOLO_ONLY = False# Set to False to run full pipeline (EfficientNet + YOLO fusion)

# ============== PLOTTING UTILITY ==============
def save_benchmark_plot(res, idx, save_dir):
    """Saves a side-by-side comparison of SCD (A) and YOLO (B) images."""
    gt_cid = res['gt_cid']
    pred_cid = res['pred_cid']
    is_correct = res['is_correct']
    
    name_map = {-1: "Background", 0: "Phantom", 1: "Matrice", 2: "Mavic"}
    
    # Determine error category for naming
    if is_correct:
        category = "all_correct"
        color = "green"
    elif gt_cid in [0, 1, 2] and pred_cid is None:
        category = "missed_drone" # FN
        color = "red"
    elif gt_cid == -1 and pred_cid is not None:
        category = "false_alarm" # FP
        color = "red"
    else:
        category = "class_error" # Wrong drone type
        color = "orange"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    title = (f"Status: {category.upper()} | Range: {res['dist']}m\n"
             f"GT: {name_map.get(gt_cid)} | Pred: {name_map.get(pred_cid, 'None')} | Conf: {res['max_conf']:.3f}")
    fig.suptitle(title, fontsize=14, fontweight='bold', color=color)

    # Load Images
    img_a = Image.open(res['a_path']).convert('RGB')
    # Use the saved detection image from YOLO if available, otherwise raw B image
    img_b = Image.open(res['saved_detection_image']).convert('RGB')

    axes[0].imshow(img_a)
    axes[0].axis('off')
    axes[0].set_title(f"A: SCD Source\n{os.path.basename(res['a_path'])}", fontsize=10)

    axes[1].imshow(img_b)
    axes[1].axis('off')
    axes[1].set_title(f"B: YOLO Detection\n{os.path.basename(res['b_path'])}", fontsize=10)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{category}_{res['dist']}m_{idx:04d}.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close(fig)

# ============== BENCHMARK LOOP ==============
def run_bench():
    print(f"Initializing Dataset and Model...")
    dataset = DronePairDataset(FOLDER_A, FOLDER_B)
    yolo_model = load_yolo_model()
    
    if not YOLO_ONLY:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        classifier = load_model(device)
        transform = get_transforms()
        
    range_stats = {}
    binary_stats = {'c': 0, 't': 0} # Drone vs No Drone
    class_stats = {-1: {'c': 0, 't': 0}, 0: {'c': 0, 't': 0}, 
                    1: {'c': 0, 't': 0}, 2: {'c': 0, 't': 0}}
    
    detailed_results = []

    print(f"\nEvaluating {len(dataset)} pairs (YOLO_ONLY={YOLO_ONLY})...")

    for i in tqdm(range(len(dataset)), desc="Benchmarking"):
        a_path, b_path, gt_cid, dist, _ = dataset[i]
        
        if YOLO_ONLY:
            res_dict = run_yolo(b_path, yolo_model)
            max_conf = res_dict.get('max_confidence', 0.0)
            detections = res_dict['detections']
            saved_img = res_dict.get('saved_detection_image', b_path)
            
            # Predict binary base on confidence alone for YOLO ONLY
            is_drone_pred = max_conf >= CONF_THRESH
        else:
            res_dict = detect(a_path, b_path, classifier, yolo_model, transform, device)
            max_conf = res_dict['F'] # YOLO max conf
            detections = res_dict['detections']
            saved_img = res_dict['saved_detection_image']
            
            # Fusion prediction
            is_drone_pred = res_dict['final_decision'] != 'NOdrone'
            
        pred_cid = None
        if detections:
            best_det = max(detections, key=lambda x: x['confidence'])
            if best_det['confidence'] >= CONF_THRESH:
                pred_cid = best_det['class_id']

        is_correct = False
        # Logic: -1 must be silent
        if gt_cid == -1:
            if max_conf < CONF_THRESH:
                is_correct = True
        else:
            # Must detect correct ID above threshold
            if pred_cid == gt_cid:
                is_correct = True

        # Binary accuracy for pipeline or YOLO
        is_drone_gt = gt_cid != -1
        if is_drone_pred == is_drone_gt:
            binary_stats['c'] += 1
        binary_stats['t'] += 1

        # Store results for statistics
        if dist not in range_stats: range_stats[dist] = {'c': 0, 't': 0, 'bin_c': 0}
        range_stats[dist]['t'] += 1
        if is_correct: range_stats[dist]['c'] += 1
        if is_drone_pred == is_drone_gt: range_stats[dist]['bin_c'] += 1
        
        class_stats[gt_cid]['t'] += 1
        if is_correct: class_stats[gt_cid]['c'] += 1

        # Store details for plotting
        detailed_results.append({
            'a_path': a_path, 'b_path': b_path,
            'gt_cid': gt_cid, 'pred_cid': pred_cid,
            'max_conf': max_conf, 'dist': dist,
            'is_correct': is_correct,
            'saved_detection_image': saved_img
        })

    # ============== OUTPUT TABLES ==============
    print("\n" + "═"*65 + "\nOVERALL BINARY ACCURACY (DRONE vs NO DRONE)\n" + "─"*65)
    bin_acc = (binary_stats['c'] / binary_stats['t'] * 100) if binary_stats['t'] > 0 else 0
    print(f"Total Accuracy: {bin_acc:>12.2f}% | Count: {binary_stats['t']:>7}")

    print("\n" + "═"*65 + "\nRANGE ACCURACY\n" + "─"*65)
    print(f"{'Range':<10} | {'Binary (Drone/No Drone)':<25} | {'YOLO (Class)':<15} | {'Count':>7}")
    for d in sorted(range_stats.keys()):
        s = range_stats[d]
        b_acc = (s['bin_c']/s['t']*100) if s['t']>0 else 0
        y_acc = (s['c']/s['t']*100) if s['t']>0 else 0
        print(f"{f'{d}m':<10} | {b_acc:>10.2f}%{' '*14} | {y_acc:>10.2f}%{' '*4} | {s['t']:>7}")

    # (Class table code here...)
    print("\n" + "═"*50 + "\nCLASS-WISE ACCURACY\n" + "─"*50)
    name_map = {-1: "Background", 0: "Phantom", 1: "Matrice", 2: "Mavic"}
    for cid in sorted(class_stats.keys()):
        s = class_stats[cid]
        if s['t'] == 0: continue
        print(f"{name_map[cid]:<15} | {(s['c']/s['t']*100):>12.2f}% | {s['t']:>7}")

    # ============== GENERATE PLOTS ==============
    print(f"\nSaving visual examples to {PLOT_SAVE_DIR}...")
    correct_samples = [r for r in detailed_results if r['is_correct']][:K_EXAMPLES]
    error_samples = [r for r in detailed_results if not r['is_correct']][:K_EXAMPLES]

    for idx, res in enumerate(correct_samples):
        save_benchmark_plot(res, idx, PLOT_SAVE_DIR)
    for idx, res in enumerate(error_samples):
        save_benchmark_plot(res, idx + K_EXAMPLES, PLOT_SAVE_DIR)

    print(f"✅ Visuals saved successfully.")

if __name__ == "__main__":
    run_bench()