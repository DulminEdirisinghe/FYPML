
from ultralytics import YOLO
import torch
from collections import defaultdict

# Configuration
model_path = 'ultralytics/cfg/models/11/yolo11.yaml'#'/home/sadeepa/FYP_Group_10/Amana/ultralytics/ultralytics/cfg/models/11/yolo11.yaml'
# model_path = '/home/sadeepa/FYP_Group_10/Amana/ultralytics/ultralytics/cfg/models/11/yolo11.yaml'
data_config = 'coco8.yaml'
expert1_weights = 'stationary_expert_cv3_weights.pt'
expert2_weights = 'moving_expert_cv3_weights.pt'
phase1_save = 'phase1_cv3_model.pt'
phase2_save = 'phase2_cv3_model.pt'

device = 'cuda'
imgsz = 640

def check_expert_status(model, phase_name):
    detect_layer = model.model.model[-1]
    print(f"\n=== Expert Status Check ({phase_name}) ===")
    
    for i, expert in enumerate([detect_layer.experts1, detect_layer.experts2], 1):
        params = list(expert.parameters())
        requires_grad = any(p.requires_grad for p in params)
        print(f"Expert {i}: {'TRAINABLE' if requires_grad else 'FROZEN'}")
        print(f"First layer: requires_grad={params[0].requires_grad}")
        print(f"Last layer: requires_grad={params[-1].requires_grad}")
        print(f"Total parameters: {sum(p.numel() for p in expert.parameters())}")
        print(f"Gradients enabled: {sum(p.requires_grad for p in expert.parameters())}/{len(params)}")
    
    print("\n=== Gating Networks ===")
    for i, gate in enumerate(detect_layer.gates):
        gate_params = list(gate.parameters())
        print(f"Gate {i+1}:")
        print(f"  - Linear layer weights: {gate[2].weight.shape}")
        print(f"  - Linear layer bias: {gate[2].bias.shape}")
        print(f"  - Total parameters: {sum(p.numel() for p in gate.parameters())}")
        print(f"  - Trainable: {any(p.requires_grad for p in gate.parameters())}")

def setup_model(phase=1):
    model = YOLO(model_path)
    detect_layer = model.model.model[-1]
    
    # Load expert weights
    detect_layer.experts1.load_state_dict(torch.load(expert1_weights))
    detect_layer.experts2.load_state_dict(torch.load(expert2_weights))
    
    # Initialize tracking
    detect_layer.gate_history = []
    detect_layer.class_expert_map = defaultdict(lambda: defaultdict(int))
    
    # Phase-specific initialization
    if phase == 1:
        for expert in [detect_layer.experts1, detect_layer.experts2]:
            for p in expert.parameters():
                p.requires_grad = False
    else:
        for expert in [detect_layer.experts1, detect_layer.experts2]:
            for p in expert.parameters():
                p.requires_grad = True
    
    return model

def train_phase(model, epochs, phase_num, save_name=None):
    detect_layer = model.model.model[-1]
    
    # Pre-training verification
    print(f"\n=== Phase {phase_num} Training Starting ===")
    check_expert_status(model, f"Phase {phase_num} Pre-Training")
    
    # Train
    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        save_period=10,
        pretrained=False
    )
    
    # Post-training verification
    check_expert_status(model, f"Phase {phase_num} Post-Training")
    
    # Additional verification during training
    print("\nGradient Flow Verification:")
    for i, expert in enumerate([detect_layer.experts1, detect_layer.experts2], 1):
        sample_param = next(expert.parameters())
        print(f"Expert {i} gradient norm: {sample_param.grad.norm().item() if sample_param.grad is not None else 0:.6f}")
    
    if save_name:
        torch.save(model.model.state_dict(), save_name)
    
    return results

def main():
    # Phase 1: Frozen experts
    print("\n" + "="*50)
    print("Starting Phase 1: Experts Frozen")
    print("="*50)
    model = setup_model(phase=1)
    train_phase(model, epochs=100, phase_num=1, save_name=phase1_save)
    
    #check_expert_status(model, f"Phase Pre-Training")
    
    # Phase 2: Unfrozen experts
    print("\n" + "="*50)
    print("Starting Phase 2: Experts Trainable")
    print("="*50)
    model = setup_model(phase=2)
    model.model.load_state_dict(torch.load(phase1_save))
    
    # Force unfreezing after load
    detect_layer = model.model.model[-1]
    for expert in [detect_layer.experts1, detect_layer.experts2]:
        for p in expert.parameters():
            p.requires_grad = True
    
    train_phase(model, epochs=3, phase_num=2, save_name=phase2_save)
    

if __name__ == "__main__":
    main()


