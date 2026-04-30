from ultralytics import YOLO
import torch
import copy

# Configuration
model_path = 'ultralytics/cfg/models/11/yolo11.yaml'
data_config_moving = 'moving_data.yaml'
data_config_stationary = 'stationary_data.yaml'
save_expert1 = 'weights/v5/moving_expert_cv3_weights.pt'
save_expert2 = 'weights/v5/stationary_expert_cv3_weights.pt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def save_experts(model, expert):
    """Extract and save cv3 weights as two experts."""
    detect_layer = model.model.model[-1]  # Get the Detect head
    if(expert == 'moving'):
        expert_weights = copy.deepcopy(detect_layer.experts1)  
        torch.save(expert_weights.state_dict(), save_expert1)
        print(f"Moving expert saved to {save_expert1}")
    elif(expert == 'stationary'):
        expert_weights = copy.deepcopy(detect_layer.experts2)  
        torch.save(expert_weights.state_dict(), save_expert2)
        print(f"Stationary expert saved to {save_expert2}")


def train_yolo():
    # Load model , Initializes weights, If pretrained weights exist, they can be loaded later
    model = YOLO(model_path)
    
    # Train the model
    results_moving = model.train(
        data=data_config_moving,
        epochs=100,
        imgsz=640,
        device=device,
        batch=16,
        save=True,
        pretrained=True,  # Use pretrained weights if available
    )
    
    # Save cv3 weights as experts
    save_experts(model, 'moving')
    results_stationary = model.train(
        data=data_config_stationary,
        epochs=100,
        imgsz=640,
        device=device,
        batch=16,
        save=True,
        pretrained=True,  # Use pretrained weights if available
    )
    save_experts(model, 'stationary')

    return results_moving, results_stationary   

if __name__ == "__main__":
    train_yolo()
   
