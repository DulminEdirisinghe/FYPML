# from ultralytics import YOLO
# import torch
# import copy

# # Configuration
# model_path = 'ultralytics/cfg/models/11/yolo11.yaml'#'/home/sadeepa/FYP_Group_10/Amana/ultralytics/ultralytics/cfg/models/11/yolo11.yaml'
# data_config = 'test_data.yaml'
# # save_expert1 = 'moving_expert_cv3_weights.pt'
# save_expert2 = 'stationary_expert_cv3_weights.pt'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# def save_experts(model, expert2_path):
#     """Extract and save cv3 weights as two experts."""
#     detect_layer = model.model.model[-1]  # Get the Detect head
#     # expert1 = copy.deepcopy(detect_layer.experts1)  # Expert 1 (original cv3), detect_layer = YOLO Detect head (P3, P4, P5)
#     expert2 = copy.deepcopy(detect_layer.experts2)  # Expert 2 (duplicate for MoE)
    
#     # Save state dicts ,Saves only the parameters ,Not the full model
#     torch.save(expert2.state_dict(), expert2_path)
#     #torch.save(expert2.state_dict(), expert2_path)
#     print(f"Experts saved to {expert2_path}")

# def train_yolo():
#     # Load model , Initializes weights, If pretrained weights exist, they can be loaded later
#     model = YOLO(model_path)
    
#     # Train the model
#     results = model.train(
#         data=data_config,
#         epochs=100,
#         imgsz=640,
#         device=device,
#         batch=16,
#         save=True,
#         pretrained=True,  # Use pretrained weights if available
#     )
    
#     # Save cv3 weights as experts
#     save_experts(model, save_expert2)
    
#     return results

# if __name__ == "__main__":
#     train_yolo()

from ultralytics import YOLO
import torch
import copy
import os

# Configuration
model_path = 'ultralytics/cfg/models/11/yolo11.yaml'
data_config_moving = 'moving_data.yaml'

save_expert1 = 'weights/v5/moving_expert_cv3_weights.pt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def save_moving_expert(model, expert_path):
    """Extract and save moving expert cv3 weights."""
    
    # Get YOLO Detect head
    detect_layer = model.model.model[-1]

    # Moving expert = experts1
    moving_expert = copy.deepcopy(detect_layer.experts1)

    # Create save folder if not exists
    os.makedirs(os.path.dirname(expert_path), exist_ok=True)

    # Save only parameters
    torch.save(moving_expert.state_dict(), expert_path)

    print(f"Moving expert saved to {expert_path}")


def train_yolo_moving():
    # Load YOLO model from yaml
    model = YOLO(model_path)

    # Train only on moving dataset
    results_moving = model.train(
        data=data_config_moving,
        epochs=100,
        imgsz=640,
        device=device,
        batch=16,
        save=True,
        pretrained=True,
    )

    # Save moving expert weights
    save_moving_expert(model, save_expert1)

    return results_moving


if __name__ == "__main__":
    train_yolo_moving()