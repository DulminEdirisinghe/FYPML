from ultralytics import YOLO
import torch

# load the model architecture
model = YOLO('/home/sadeepa/FYP_Group_10/Amana/ultralytics/ultralytics/cfg/models/11/yolo11.yaml')

# load trained weights
state_dict = torch.load('phase1_cv3_model.pt')
model.model.load_state_dict(state_dict)

# evaluate
#results = model.predict(data='coco8.yaml', imgsz=640, device='cuda')
results = model.val(data='coco8.yaml', imgsz=640, device='cuda')


# Check MoE statistics
detect_layer = model.model.model[-1]
if hasattr(detect_layer, 'gate_history'):
    gates = torch.cat(detect_layer.gate_history)
    print(f"Expert 1 usage: {gates[:,0].mean():.3f}")
    print(f"Expert 2 usage: {gates[:,1].mean():.3f}")

