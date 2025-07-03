import torch
import torch.nn as nn
import numpy as np
import os
from torchvision.models import resnet50, ResNet50_Weights

class SingleJointInflammationClassifier(nn.Module):
    def __init__(self, num_landmarks=32):
        super().__init__()
        self.cnn = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.cnn.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.fc = nn.Identity()

        self.mlp = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, image_input, landmark_input):
        x_img = self.cnn(image_input)
        x_landmark = self.mlp(landmark_input)
        x = torch.cat([x_img, x_landmark], dim=1)
        return self.classifier(x).squeeze(1)


def predict_joint_inflammation(image, landmarks, model_dir, device=None, logger=None):
    """
    Predict joint inflammation for all 32 joints by loading models sequentially.
    
    Args:
        image (np.array): 4-channel image (H, W, 4) or (4, H, W)
        landmarks (np.array or list): Array of shape (32, 2) containing landmark coordinates
        model_dir (str): Directory containing the trained .pt model files
        device (torch.device, optional): Device to run on. Auto-detects if None.
    
    Returns:
        list: List of 0/1 predictions for each joint
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Preprocess image
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()
    
    # Ensure correct shape (4, H, W)
    if len(image.shape) == 3:
        if image.shape[2] == 4:  # (H, W, 4) -> (4, H, W)
            image = image.permute(2, 0, 1)
        elif image.shape[0] != 4:
            raise ValueError(f"Expected 4 channels, got {image.shape[0]}")
    
    # Normalize per channel
    for c in range(image.shape[0]):
        min_val = image[c].min()
        max_val = image[c].max()
        image[c] = (image[c] - min_val) / (max_val - min_val + 1e-6)
    
    # Add batch dimension
    image = image.unsqueeze(0).to(device)
    
    # Convert landmarks to numpy if needed
    if isinstance(landmarks, list):
        landmarks = np.array(landmarks)
    
    predictions = []
    
    # Loop through each joint
    for joint_idx in range(32):
        print(f"Processing joint {joint_idx}...")
        
        # Load model for current joint
        model_path = os.path.join(model_dir, f"joint_{joint_idx}_resnet50.pt")
        
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} not found. Skipping joint {joint_idx}.")
            predictions.append(0)
            continue
        
        # Initialize model and load state dict from .pt file
        model = SingleJointInflammationClassifier().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Get landmark for current joint
        landmark = torch.tensor(landmarks[joint_idx], dtype=torch.float32).unsqueeze(0).to(device)
        
        # Make prediction for current joint
        with torch.no_grad():
            output = model(image, landmark)
            probability = torch.sigmoid(output).item()
            predictions.append(probability)
        
        print(f"Joint {joint_idx} prediction: {probability}")
        
        # Clear model from memory before next iteration
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return predictions