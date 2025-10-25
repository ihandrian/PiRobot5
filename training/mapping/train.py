import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
from pathlib import Path

from room_mapper import RoomMapper
from dataset import RoomMappingDataset

def train_model(
    data_dir: str,
    model_save_path: str,
    map_save_path: str,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    map_size: Tuple[int, int] = (100, 100),
    resolution: float = 0.1
):
    """Train the room mapping model."""
    # Initialize room mapper
    mapper = RoomMapper(
        map_size=map_size,
        resolution=resolution
    )
    
    # Create dataset and dataloader
    dataset = RoomMappingDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize optimizer and loss function
    optimizer = Adam(mapper.model.parameters(), lr=learning_rate)
    criterion = MSELoss()
    
    # Training loop
    for epoch in range(epochs):
        mapper.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
            # Move data to device
            inputs = batch['input'].to(mapper.device)
            targets = batch['target'].to(mapper.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = mapper.model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Print epoch statistics
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
        
        # Save model and map periodically
        if (epoch + 1) % 5 == 0:
            mapper.save_model(model_save_path)
            mapper.save_map(map_save_path)
            
    # Save final model and map
    mapper.save_model(model_save_path)
    mapper.save_map(map_save_path)
    
    return mapper

def generate_training_data(
    output_dir: str,
    num_samples: int = 1000,
    map_size: Tuple[int, int] = (100, 100),
    resolution: float = 0.1
):
    """Generate synthetic training data."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i in tqdm(range(num_samples), desc='Generating training data'):
        # Create random occupancy grid
        occupancy_grid = np.random.rand(*map_size)
        
        # Add some obstacles
        num_obstacles = np.random.randint(5, 15)
        obstacles = []
        
        for _ in range(num_obstacles):
            x = np.random.randint(0, map_size[0])
            y = np.random.randint(0, map_size[1])
            size = np.random.randint(3, 10)
            
            # Create circular obstacle
            for dx in range(-size, size + 1):
                for dy in range(-size, size + 1):
                    if dx*dx + dy*dy <= size*size:
                        px = x + dx
                        py = y + dy
                        if 0 <= px < map_size[0] and 0 <= py < map_size[1]:
                            occupancy_grid[px, py] = 1.0
                            
            obstacles.append((x * resolution, y * resolution))
            
        # Save sample
        sample_path = os.path.join(output_dir, f'sample_{i:04d}.json')
        RoomMappingDataset.create_sample(
            occupancy_grid=occupancy_grid,
            obstacles=obstacles,
            save_path=sample_path
        )

def main():
    # Configuration
    config = {
        'data_dir': 'training_data',
        'model_save_path': 'models/room_mapper.pth',
        'map_save_path': 'maps/room_map.json',
        'epochs': 20,
        'batch_size': 32,
        'learning_rate': 0.001,
        'map_size': (100, 100),
        'resolution': 0.1
    }
    
    # Create directories
    os.makedirs('training_data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('maps', exist_ok=True)
    
    # Generate training data
    print("Generating training data...")
    generate_training_data(
        output_dir=config['data_dir'],
        num_samples=1000,
        map_size=config['map_size'],
        resolution=config['resolution']
    )
    
    # Train model
    print("\nTraining model...")
    mapper = train_model(
        data_dir=config['data_dir'],
        model_save_path=config['model_save_path'],
        map_save_path=config['map_save_path'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        map_size=config['map_size'],
        resolution=config['resolution']
    )
    
    # Visualize final map
    print("\nVisualizing final map...")
    mapper.visualize_map()
    
if __name__ == '__main__':
    main() 