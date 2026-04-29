import os
import torch
import torch.nn as nn
from torchvision import transforms, models
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from src.bench_dataset import DronePairDataset

# ============== CONFIGURATIONS ==============
FOLDER_A = 'data/v5/SCD_images/test'
FOLDER_B = 'data/v5/YOLO/stationary/test'
PLOT_SAVE_DIR = 'runs/efficientnet_visuals'
K_EXAMPLES = 10

CONFIG = {
    'model_path': 'runs/efficientnet_v5_classification/20260429_205239/best_classifier.pth',
    'model_name': 'efficientnet_b0',
    'num_classes': 4,
}

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
    elif gt_cid in [0, 1, 2] and pred_cid == -1:
        category = "missed_drone" # FN
        color = "red"
    elif gt_cid == -1 and pred_cid in [0, 1, 2]:
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
    img_b = Image.open(res['b_path']).convert('RGB')

    axes[0].imshow(img_a)
    axes[0].axis('off')
    axes[0].set_title(f"A: SCD Source\n{os.path.basename(res['a_path'])}", fontsize=10)

    axes[1].imshow(img_b)
    axes[1].axis('off')
    axes[1].set_title(f"B: YOLO Source\n{os.path.basename(res['b_path'])}", fontsize=10)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{category}_{res['dist']}m_{idx:04d}.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close(fig)

# ============== BENCHMARK LOOP ==============
def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"Initializing Dataset...")
    dataset = DronePairDataset(FOLDER_A, FOLDER_B)

    print(f"Initializing {CONFIG['model_name']} model (num_classes={CONFIG['num_classes']})...")
    model = getattr(models, CONFIG['model_name'])(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, CONFIG['num_classes'])
    
    if os.path.exists(CONFIG['model_path']):
        model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device))
        print("✅ Model weights loaded successfully.")
    else:
        print(f"⚠️ WARNING: Model path '{CONFIG['model_path']}' not found!")
        
    model = model.to(device)
    model.eval()

    # Trackers for statistics
    range_stats = {}
    binary_stats = {'c': 0, 't': 0}
    class_stats = {-1: {'c': 0, 't': 0}, 0: {'c': 0, 't': 0}, 
                   1: {'c': 0, 't': 0}, 2: {'c': 0, 't': 0}}
                    
    detailed_results = []

    print(f"\nEvaluating {len(dataset)} pairs...")

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Benchmarking"):
            a_path, b_path, gt_cid, dist, _ = dataset[i]
            
            image = Image.open(a_path).convert('RGB')
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            max_prob, preds = torch.max(probs, 1)

            # EfficientNet outputs: 0=No Drone, 1=Phantom, 2=Matrice, 3=Mavic
            # Dataset gt_cid: -1=Background, 0=Phantom, 1=Matrice, 2=Mavic
            eff_pred = preds.item()
            pred_cid = -1 if eff_pred == 0 else eff_pred - 1
            max_conf = max_prob.item()

            is_correct_class = (gt_cid == pred_cid)
            is_drone_gt = (gt_cid != -1)
            is_drone_pred = (pred_cid != -1)
            is_correct_binary = (is_drone_gt == is_drone_pred)
            
            if is_correct_binary:
                binary_stats['c'] += 1
            binary_stats['t'] += 1

            if dist not in range_stats:
                range_stats[dist] = {'c': 0, 'bin_c': 0, 't': 0}
            
            range_stats[dist]['t'] += 1
            if is_correct_class:
                range_stats[dist]['c'] += 1
            if is_correct_binary:
                range_stats[dist]['bin_c'] += 1

            class_stats[gt_cid]['t'] += 1
            if is_correct_class:
                class_stats[gt_cid]['c'] += 1
                
            detailed_results.append({
                'a_path': a_path, 'b_path': b_path,
                'gt_cid': gt_cid, 'pred_cid': pred_cid,
                'max_conf': max_conf, 'dist': dist,
                'is_correct': is_correct_class,
            })

    # ============== OUTPUT TABLES ==============
    print("\n" + "═"*65 + "\nOVERALL BINARY ACCURACY (DRONE vs NO DRONE)\n" + "─"*65)
    bin_acc = (binary_stats['c'] / binary_stats['t'] * 100) if binary_stats['t'] > 0 else 0
    print(f"Total Accuracy: {bin_acc:>12.2f}% | Count: {binary_stats['t']:>7}")

    print("\n" + "═"*65 + "\nRANGE ACCURACY\n" + "─"*65)
    print(f"{'Range':<10} | {'Binary (Drone/No Drone)':<25} | {'Class-wise':<15} | {'Count':>7}")
    for d in sorted(range_stats.keys()):
        s = range_stats[d]
        b_acc = (s['bin_c']/s['t']*100) if s['t']>0 else 0
        c_acc = (s['c']/s['t']*100) if s['t']>0 else 0
        print(f"{f'{d}m':<10} | {b_acc:>10.2f}%{' '*14} | {c_acc:>10.2f}%{' '*5} | {s['t']:>7}")

    print("\n" + "═"*65 + "\nCLASS-WISE ACCURACY\n" + "─"*65)
    name_map = {-1: "Background", 0: "Phantom", 1: "Matrice", 2: "Mavic"}
    for cid in sorted(class_stats.keys()):
        s = class_stats[cid]
        if s['t'] == 0: continue
        acc = (s['c'] / s['t'] * 100)
        print(f"{name_map[cid]:<20} | {acc:>12.2f}% | {s['t']:>7}")
        
    # ============== GENERATE PLOTS ==============
    print(f"\nSaving visual examples to {PLOT_SAVE_DIR}...")
    correct_samples = [r for r in detailed_results if r['is_correct']][:K_EXAMPLES]
    error_samples = [r for r in detailed_results if not r['is_correct']][:K_EXAMPLES]

    for idx, res in enumerate(correct_samples):
        save_benchmark_plot(res, idx, PLOT_SAVE_DIR)
    for idx, res in enumerate(error_samples):
        save_benchmark_plot(res, idx + K_EXAMPLES, PLOT_SAVE_DIR)

    print(f"✅ Visuals saved successfully.")

if __name__ == '__main__':
    run_benchmark()
