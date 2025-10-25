"""
Behavior Cloning for PiRobot5
Implements neural network-based behavior cloning for autonomous navigation
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import json
import pickle


@dataclass
class TrainingData:
    """Training data container"""
    images: np.ndarray  # Input images
    actions: np.ndarray  # Corresponding actions
    metadata: Dict[str, Any]  # Additional metadata


class BehaviorCloningNetwork(nn.Module):
    """Neural network for behavior cloning"""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (3, 480, 640)):
        super(BehaviorCloningNetwork, self).__init__()
        
        self.input_shape = input_shape
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        
        # Calculate flattened size
        self.flattened_size = self._get_flattened_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)  # 2 outputs: steering and throttle
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def _get_flattened_size(self) -> int:
        """Calculate the size after convolutional layers"""
        with torch.no_grad():
            x = torch.zeros(1, *self.input_shape)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            return x.numel()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x


class BehaviorCloning:
    """Behavior cloning system for autonomous navigation"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 logger_name: str = "PiRobot.BehaviorCloning"):
        self.logger = logging.getLogger(logger_name)
        self.model_path = model_path or "models/behavior_cloning.pth"
        
        # Initialize model
        self.model = BehaviorCloningNetwork()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Load model if exists
        if Path(self.model_path).exists():
            self.load_model()
        else:
            self.logger.info("No pre-trained model found, starting fresh")
            
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for neural network input"""
        # Resize to network input size
        image = cv2.resize(image, (640, 480))
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor
        
    def predict_action(self, image: np.ndarray) -> Tuple[float, float]:
        """Predict action from image"""
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            
            # Set model to evaluation mode
            self.model.eval()
            
            with torch.no_grad():
                # Get prediction
                output = self.model(image_tensor)
                
                # Extract steering and throttle
                steering = output[0, 0].item()
                throttle = output[0, 1].item()
                
                # Clamp values to valid ranges
                steering = np.clip(steering, -1.0, 1.0)
                throttle = np.clip(throttle, 0.0, 1.0)
                
                return steering, throttle
                
        except Exception as e:
            self.logger.error(f"Error in action prediction: {e}")
            return 0.0, 0.0  # Return neutral action on error
            
    def train(self, training_data: TrainingData, epochs: int = 100) -> Dict[str, float]:
        """Train the behavior cloning model"""
        try:
            self.logger.info(f"Starting training with {len(training_data.images)} samples")
            
            # Convert to tensors
            images = torch.from_numpy(training_data.images).float()
            actions = torch.from_numpy(training_data.actions).float()
            
            # Set model to training mode
            self.model.train()
            
            # Training loop
            losses = []
            for epoch in range(epochs):
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, actions)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                losses.append(loss.item())
                
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
                    
            # Save model
            self.save_model()
            
            return {
                'final_loss': losses[-1],
                'min_loss': min(losses),
                'max_loss': max(losses)
            }
            
        except Exception as e:
            self.logger.error(f"Error in training: {e}")
            return {'error': str(e)}
            
    def save_model(self) -> None:
        """Save the trained model"""
        try:
            # Create directory if it doesn't exist
            Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model state
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'input_shape': self.model.input_shape
            }, self.model_path)
            
            self.logger.info(f"Model saved to {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            
    def load_model(self) -> None:
        """Load a trained model"""
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.logger.info(f"Model loaded from {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            
    def collect_training_data(self, 
                             image: np.ndarray, 
                             steering: float, 
                             throttle: float,
                             metadata: Optional[Dict] = None) -> None:
        """Collect training data for future training"""
        # This would typically save to a dataset file
        # For now, just log the collection
        self.logger.debug(f"Collected training data: steering={steering}, throttle={throttle}")
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_path': self.model_path,
            'input_shape': self.model.input_shape
        }


# Global behavior cloning instance
behavior_cloning = BehaviorCloning()


def get_behavior_cloning() -> BehaviorCloning:
    """Get the global behavior cloning instance"""
    return behavior_cloning
