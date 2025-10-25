import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json
from typing import Dict, List, Tuple
import cv2
from pathlib import Path

class RoomMappingDataset(Dataset):
    """Dataset for room mapping training."""
    
    def __init__(self, 
                 data_dir: str,
                 transform=None,
                 target_transform=None):
        """Initialize the dataset."""
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_transform = target_transform
        
        # Load data files
        self.data_files = list(self.data_dir.glob('*.json'))
        if not self.data_files:
            raise ValueError(f"No data files found in {data_dir}")
            
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data_files)
        
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from the dataset."""
        # Load data file
        with open(self.data_files[idx], 'r') as f:
            data = json.load(f)
            
        # Extract input data (occupancy grid)
        occupancy_grid = np.array(data['occupancy_grid'], dtype=np.float32)
        
        # Convert to tensor and add channel dimension
        input_tensor = torch.from_numpy(occupancy_grid).unsqueeze(0)
        
        # Extract target data (obstacle positions)
        obstacles = np.array(data['obstacles'], dtype=np.float32)
        
        # Create target tensor
        target_tensor = torch.from_numpy(obstacles)
        
        # Apply transforms if specified
        if self.transform:
            input_tensor = self.transform(input_tensor)
        if self.target_transform:
            target_tensor = self.target_transform(target_tensor)
            
        return {
            'input': input_tensor,
            'target': target_tensor
        }
        
    @staticmethod
    def create_sample(occupancy_grid: np.ndarray,
                     obstacles: List[Tuple[float, float]],
                     save_path: str):
        """Create and save a new data sample."""
        data = {
            'occupancy_grid': occupancy_grid.tolist(),
            'obstacles': obstacles
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save data
        with open(save_path, 'w') as f:
            json.dump(data, f)
            
    @staticmethod
    def preprocess_image(image: np.ndarray,
                        target_size: Tuple[int, int] = (100, 100)) -> np.ndarray:
        """Preprocess an image for the dataset."""
        # Resize image
        image = cv2.resize(image, target_size)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image 