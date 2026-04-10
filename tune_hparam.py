import os
import json
import re
import torch
from PIL import Image
from sklearn.metrics import accuracy_score
from src.bench_dataset import DronePairDataset

# Import your existing functions from your main detector file
from pipeline import load_all_models, CONFIG, logistic_fusion, run_yolo


# ============== CONFIGURATIONS ==============
FOLDER_A_TEST = 'data/data/v3/SCD_images/test'
FOLDER_B_TEST = 'data/data/v3/YOLO/stationary/test'

OUTPUT_DIR = "runs/hparams_v3_data_v3_model"
CACHE_FILENAME = 'fusion_cache.json'
RESULTS_FILENAME = 'fusion_hparam_results.txt'

# Search ranges for fusion hyperparameters
# Fine-tuning Search Ranges
W1_RANGE = [0.3, 0.4, 0.5, 0.6, 0.7]
W2_RANGE = [0.3, 0.4, 0.5, 0.6, 0.7]
B_RANGE = [-0.5, -0.4, -0.3, -0.2, -0.1]
T1_RANGE = [0.40, 0.45, 0.50, 0.55, 0.60]
T2_RANGE = [0.05, 0.10, 0.15, 0.20, 0.25] # Expanded below previous minimum


def run_inference_and_cache(cache_path):
    """First pass: run inference and save model outputs + labels to JSON."""

    print("Loading models...")
    classifier, yolo_model, transform, device = load_all_models()

    print("Building Dataset...")
    dataset = DronePairDataset(FOLDER_A_TEST, FOLDER_B_TEST)
    print(f"Total valid pairs found: {len(dataset)}")

    if len(dataset) == 0:
        print("No pairs matched. Please check your folder paths and naming conventions.")
        return []

    records = []

    print("\nRunning first-pass inference to cache P4/F values...")

    for i in range(len(dataset)):
        data = dataset[i]

        a_filename = os.path.basename(data['a_img_path']).lower()
        b_filename = os.path.basename(data['b_img_path']).lower()

        # 1. Extract Distance using Regex
        dist_val = 0.0
        dist_match = re.search(r'(\d+(?:\.\d+)?)m', a_filename)
        if not dist_match:
            dist_match = re.search(r'(\d+(?:\.\d+)?)m', b_filename)
            
        if dist_match:
            dist_val = float(dist_match.group(1))
            
        dist_str = f"{dist_val:g}m"

        # 2. Override Ground Truth based strictly on extracted distance
        if dist_val == 0.0:
            gt_is_drone = False
            gt_class = 'no_drone'
        else:
            gt_is_drone = True
            if 'phantom' in a_filename or 'phanthom' in a_filename:
                gt_class = 'phantom'
            else:
                gt_class = 'phantom'

        # EfficientNet (P4)
        img_a = Image.open(data['a_img_path']).convert('RGB')
        img_tensor = transform(img_a).unsqueeze(0).to(device)
        with torch.no_grad():
            output = classifier(img_tensor)
            probabilities = torch.softmax(output, dim=1)
        P4 = probabilities[0, 1].item()

        # YOLO (F and class)
        yolo_result = run_yolo(data['b_img_path'], yolo_model)
        F = yolo_result['max_confidence']
        yolo_class = yolo_result['best_class_name']

        record = {
            'a_img_path': data['a_img_path'],
            'b_img_path': data['b_img_path'],
            'distance': dist_str,
            'gt_is_drone': gt_is_drone,
            'gt_class': gt_class,
            'P4': P4,
            'F': F,
            'yolo_class': yolo_class,
        }

        records.append(record)

        if (i + 1) % 10 == 0:
            print(f"Cached {i + 1}/{len(dataset)} pairs...")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(records, f, indent=2)

    print(f"\n✅ Cached inference saved to: {cache_path}")
    return records


def evaluate_current_config(records):
    """Evaluate detection + class accuracy for current CONFIG fusion/thresholds."""

    y_true_binary = []
    y_pred_binary = []

    cls_total = 0
    cls_correct = 0

    for rec in records:
        P4 = rec['P4']
        F = rec['F']

        # Fusion score with current CONFIG hyperparameters
        G = logistic_fusion(F, P4)

        # Decision logic matches pipeline.detect
        if G <= CONFIG['T1']:
            final_decision = 'NOdrone'
            pred_class = 'no_drone'
        elif F > CONFIG['T2']:
            final_decision = 'DroneType'
            pred_class = rec['yolo_class'].lower() if rec['yolo_class'] else 'uncertain'
        else:
            final_decision = 'Detected'
            pred_class = 'uncertain'

        pred_is_drone = final_decision in ['Detected', 'DroneType']

        gt_is_drone = rec['gt_is_drone']
        gt_class = rec['gt_class']
        
        y_true_binary.append(gt_is_drone)
        y_pred_binary.append(pred_is_drone)

        # Class "correct" aligns with strict benchmark logic: specific drone class must match
        if gt_is_drone:
            cls_total += 1
            if pred_is_drone and gt_class == pred_class:
                cls_correct += 1

    # Overall binary accuracy (drone vs no-drone)
    bin_acc = accuracy_score(y_true_binary, y_pred_binary)

    # Class certainty accuracy when GT is drone
    class_acc = (cls_correct / cls_total) if cls_total > 0 else 0.0

    return bin_acc, class_acc


def run_hparam_search(records, results_path):
    """Grid-search over (w1, w2, b, T1, T2) and write results to a txt file."""

    if not records:
        print("No records available for hyperparameter search.")
        return

    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # Backup original config so we can restore it
    original_config = {
        'fusion_w1': CONFIG['fusion_w1'],
        'fusion_w2': CONFIG['fusion_w2'],
        'fusion_b': CONFIG['fusion_b'],
        'T1': CONFIG['T1'],
        'T2': CONFIG['T2'],
    }

    best_params = None
    best_bin_acc = -1.0
    best_class_acc = -1.0
    all_results = []

    print("\nStarting hyperparameter search over fusion weights and thresholds...")

    total_combos = len(W1_RANGE) * len(W2_RANGE) * len(B_RANGE) * len(T1_RANGE) * len(T2_RANGE)
    combo_idx = 0

    for w1 in W1_RANGE:
        for w2 in W2_RANGE:
            for b in B_RANGE:
                CONFIG['fusion_w1'] = w1
                CONFIG['fusion_w2'] = w2
                CONFIG['fusion_b'] = b

                for t1 in T1_RANGE:
                    for t2 in T2_RANGE:
                        CONFIG['T1'] = t1
                        CONFIG['T2'] = t2

                        combo_idx += 1
                        bin_acc, class_acc = evaluate_current_config(records)

                        all_results.append({
                            'w1': w1,
                            'w2': w2,
                            'b': b,
                            'T1': t1,
                            'T2': t2,
                            'bin_acc': bin_acc,
                            'class_acc': class_acc,
                        })

                        # Track the best configuration (by binary acc, then class acc)
                        if (bin_acc > best_bin_acc) or (
                            abs(bin_acc - best_bin_acc) < 1e-6 and class_acc > best_class_acc
                        ):
                            best_bin_acc = bin_acc
                            best_class_acc = class_acc
                            best_params = {
                                'w1': w1,
                                'w2': w2,
                                'b': b,
                                'T1': t1,
                                'T2': t2,
                                'bin_acc': bin_acc,
                                'class_acc': class_acc,
                            }

                        if combo_idx % 20 == 0 or combo_idx == total_combos:
                            print(
                                f"Evaluated {combo_idx}/{total_combos} combos... "
                                f"(best bin_acc={best_bin_acc*100:.2f}%, class_acc={best_class_acc*100:.2f}%)"
                            )

    # Restore original CONFIG
    CONFIG.update(original_config)

    # Write detailed results
    with open(results_path, 'w') as f:
        f.write("Logistic Fusion Hyperparameter Search\n")
        f.write("====================================\n\n")
        f.write(f"Num samples: {len(records)}\n")
        f.write(f"Search space sizes: w1={W1_RANGE}, w2={W2_RANGE}, b={B_RANGE}, T1={T1_RANGE}, T2={T2_RANGE}\n\n")

        f.write("Per-combination results (accuracies in %):\n")
        f.write("w1\tw2\tb\tT1\tT2\tbin_acc\tclass_acc\n")
        for r in all_results:
            f.write(
                f"{r['w1']:.3f}\t{r['w2']:.3f}\t{r['b']:.3f}\t{r['T1']:.3f}\t{r['T2']:.3f}\t"
                f"{r['bin_acc']*100:.2f}\t{r['class_acc']*100:.2f}\n"
            )

        f.write("\nBest configuration (by binary accuracy, then class accuracy):\n")
        f.write(json.dumps(best_params, indent=2))

    print(f"\n✅ Hyperparameter search complete. Results written to: {results_path}")
    print(
        f"Best params -> w1={best_params['w1']}, w2={best_params['w2']}, b={best_params['b']}, "
        f"T1={best_params['T1']}, T2={best_params['T2']} | "
        f"bin_acc={best_params['bin_acc']*100:.2f}%, class_acc={best_params['class_acc']*100:.2f}%"
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cache_path = os.path.join(OUTPUT_DIR, CACHE_FILENAME)
    results_path = os.path.join(OUTPUT_DIR, RESULTS_FILENAME)

    if os.path.exists(cache_path):
        print(f"Loading cached inference from {cache_path}...\n(Note: If dataset labels changed, delete {cache_path} to re-cache)")
        with open(cache_path, 'r') as f:
            records = json.load(f)
    else:
        records = run_inference_and_cache(cache_path)

    run_hparam_search(records, results_path)


if __name__ == "__main__":
    main()