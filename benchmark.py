import os
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
    
    # Title with GT, Pred, and Scores
    title = (f"Status: {category.upper()}\n"
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
    
    # 3. Save the figure
    filename = f"{category}_{idx:04d}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close(fig)  # Important to prevent memory leaks when saving hundreds of plots!


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
    
    detailed_results = []

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
            
        # 4. Append to metric lists
        y_true_binary.append(gt_is_drone)
        y_pred_binary.append(pred_is_drone)
        
        y_true_class.append(gt_class)
        y_pred_class.append(pred_class)
        
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
            'P4': result['P4']
        })
        
        # Optional: Print progress every 10 images
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(dataset)} pairs...")

    # ============== CALCULATE METRICS ==============
    print("\n" + "="*50)
    print("🎯 BENCHMARK RESULTS")
    print("="*50)
    
    bin_acc = accuracy_score(y_true_binary, y_pred_binary)
    print(f"\n1. Drone / No Drone Detection Accuracy: {bin_acc * 100:.2f}%")
    
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

    # ============== GENERATE ALL PLOTS ==============
    print(f"\nGenerating plots for all {len(detailed_results)} pairs...")
    
    # Create the output directory if it doesn't exist
    os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
    
    for idx, res in enumerate(detailed_results):
        save_pair_plot(res, idx, PLOT_SAVE_DIR)
        
        # Give progress update every 50 plots
        if (idx + 1) % 50 == 0:
            print(f"Saved {idx + 1}/{len(detailed_results)} plots...")

    print(f"\n✅ All plots saved successfully in the '{PLOT_SAVE_DIR}' directory.")


if __name__ == "__main__":
    run_benchmark()