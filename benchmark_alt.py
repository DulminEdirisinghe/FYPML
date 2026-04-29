import os
import torch
from tqdm import tqdm
from src.bench_dataset import DronePairDataset
from pipeline_all import load_yolo_model, run_yolo

# ============== CONFIG ==============
FOLDER_A = 'data/v5/SCD_images/test'
FOLDER_B = 'data/v5/YOLO/stationary/test'
CONF_THRESH = 0.80

def run_bench():
    print(f"Loading Dataset and Model...")
    dataset = DronePairDataset(FOLDER_A, FOLDER_B)
    yolo_model = load_yolo_model()

    # distance -> {'correct': 0, 'total': 0}
    range_stats = {}
    # class_id -> {'correct': 0, 'total': 0}
    class_stats = {-1: {'c': 0, 't': 0}, 0: {'c': 0, 't': 0}, 
                    1: {'c': 0, 't': 0}, 2: {'c': 0, 't': 0}, 4: {'c': 0, 't': 0}}

    print(f"\nEvaluating {len(dataset)} pairs...")

    for i in tqdm(range(len(dataset)), desc="Benchmarking"):
        # a_path, b_path, gt_cid, dist, has_drone
        _, b_path, gt_cid, dist, _ = dataset[i]
        
        yolo_res = run_yolo(b_path, yolo_model)
        
        # Extract numerical data from pipeline result
        
        max_conf = yolo_res.get('max_confidence')
        
        # Finding the class ID of the highest confidence detection
        pred_cid = None
        if yolo_res['detections']:
            # detections is a list of {'class_id': int, 'confidence': float, ...}
            best_det = max(yolo_res['detections'], key=lambda x: x['confidence'])
            pred_cid = best_det['class_id']

        is_correct = False

        # --- EVALUATION LOGIC ---
        if gt_cid in [-1, 4]:
            # Silence required: Correct if no detection exceeds threshold
            if max_conf < CONF_THRESH:
                is_correct = True
        else:
            # Drone detection required: Correct if max_conf is high AND class ID matches
            if max_conf >= CONF_THRESH and pred_cid == gt_cid:
                is_correct = True

        # --- UPDATE RANGE STATS ---
        if dist not in range_stats:
            range_stats[dist] = {'c': 0, 't': 0}
        range_stats[dist]['t'] += 1
        if is_correct: range_stats[dist]['c'] += 1

        # --- UPDATE CLASS STATS ---
        class_stats[gt_cid]['t'] += 1
        if is_correct: class_stats[gt_cid]['c'] += 1

    # ============== PRINT RANGE-WISE RESULTS ==============
    print("\n" + "═"*50)
    print(f"{'RANGE ACCURACY':^50}")
    print("─" * 50)
    print(f"{'Distance':<15} | {'Accuracy (%)':<15} | {'Samples'}")
    print("─" * 50)
    for d in sorted(range_stats.keys()):
        s = range_stats[d]
        acc = (s['c'] / s['t'] * 100) if s['t'] > 0 else 0
        print(f"{f'{d}m':<15} | {acc:>12.2f}% | {s['t']:>7}")

    # ============== PRINT CLASS-WISE RESULTS ==============
    print("\n" + "═"*50)
    print(f"{'CLASS-WISE ACCURACY':^50}")
    print("─" * 50)
    print(f"{'Class ID':<15} | {'Accuracy (%)':<15} | {'Samples'}")
    print("─" * 50)
    name_map = {-1: "Background", 0: "Phantom", 1: "Matrice", 2: "Mavic", 4: "Uncertain"}
    for cid in sorted(class_stats.keys()):
        s = class_stats[cid]
        if s['t'] == 0: continue
        acc = (s['c'] / s['t'] * 100)
        label = f"{cid} ({name_map[cid]})"
        print(f"{label:<15} | {acc:>12.2f}% | {s['t']:>7}")

    # OVERALL
    total_t = sum(s['t'] for s in class_stats.values())
    total_c = sum(s['c'] for s in class_stats.values())
    overall = (total_c / total_t * 100) if total_t > 0 else 0
    print("─" * 50)
    print(f"{'OVERALL':<15} | {overall:>12.2f}% | {total_t:>7}")
    print("═"*50)

if __name__ == "__main__":
    run_bench()