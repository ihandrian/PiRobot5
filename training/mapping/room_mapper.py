import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import DBSCAN
from scipy.spatial import Voronoi, voronoi_plot_2d
import os
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RoomMapper')

class RoomMapper:
    """Room mapping and path planning system."""
    
    def __init__(self, 
                 map_size: Tuple[int, int] = (100, 100),
                 resolution: float = 0.1,  # meters per pixel
                 min_cluster_size: int = 5,
                 eps: float = 0.5):
        """Initialize the room mapper."""
        self.map_size = map_size
        self.resolution = resolution
        self.min_cluster_size = min_cluster_size
        self.eps = eps
        
        # Initialize occupancy grid
        self.occupancy_grid = np.zeros(map_size, dtype=np.float32)
        self.visited = np.zeros(map_size, dtype=np.bool_)
        
        # Initialize graph for path planning
        self.graph = nx.Graph()
        
        # Initialize neural network for obstacle prediction
        self.model = self._create_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def _create_model(self) -> nn.Module:
        """Create neural network for obstacle prediction."""
        class ObstaclePredictor(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
                self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.fc1 = nn.Linear(64 * 12 * 12, 128)
                self.fc2 = nn.Linear(128, 2)  # x, y coordinates
                
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.max_pool2d(x, 2)
                x = torch.relu(self.conv2(x))
                x = torch.max_pool2d(x, 2)
                x = torch.relu(self.conv3(x))
                x = torch.max_pool2d(x, 2)
                x = x.view(x.size(0), -1)
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x
                
        return ObstaclePredictor()
        
    def update_map(self, 
                  position: Tuple[float, float],
                  sensor_data: np.ndarray,
                  confidence: float = 0.8):
        """Update the occupancy grid with new sensor data."""
        # Convert position to grid coordinates
        grid_x = int(position[0] / self.resolution)
        grid_y = int(position[1] / self.resolution)
        
        # Update visited areas
        self.visited[grid_x, grid_y] = True
        
        # Update occupancy grid
        for i in range(sensor_data.shape[0]):
            for j in range(sensor_data.shape[1]):
                if sensor_data[i, j] > 0:
                    x = grid_x + i - sensor_data.shape[0] // 2
                    y = grid_y + j - sensor_data.shape[1] // 2
                    if 0 <= x < self.map_size[0] and 0 <= y < self.map_size[1]:
                        self.occupancy_grid[x, y] = max(
                            self.occupancy_grid[x, y],
                            sensor_data[i, j] * confidence
                        )
                        
    def detect_obstacles(self) -> List[Tuple[float, float]]:
        """Detect obstacles in the occupancy grid."""
        # Threshold the occupancy grid
        binary_map = (self.occupancy_grid > 0.5).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(
            binary_map, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Extract obstacle centers
        obstacles = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                obstacles.append((cx * self.resolution, cy * self.resolution))
                
        return obstacles
        
    def build_graph(self):
        """Build a graph for path planning."""
        # Clear existing graph
        self.graph.clear()
        
        # Get obstacle positions
        obstacles = self.detect_obstacles()
        
        # Create Voronoi diagram
        points = np.array(obstacles)
        vor = Voronoi(points)
        
        # Add vertices to graph
        for i, vertex in enumerate(vor.vertices):
            self.graph.add_node(i, pos=vertex)
            
        # Add edges to graph
        for i, j in vor.ridge_vertices:
            if i >= 0 and j >= 0:  # Skip infinite edges
                self.graph.add_edge(i, j)
                
    def find_path(self, 
                 start: Tuple[float, float],
                 goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Find a path from start to goal."""
        # Convert positions to grid coordinates
        start_grid = (int(start[0] / self.resolution), 
                     int(start[1] / self.resolution))
        goal_grid = (int(goal[0] / self.resolution), 
                    int(goal[1] / self.resolution))
        
        # Add start and goal to graph
        start_node = len(self.graph.nodes)
        goal_node = start_node + 1
        self.graph.add_node(start_node, pos=start)
        self.graph.add_node(goal_node, pos=goal)
        
        # Connect start and goal to nearest vertices
        for node in self.graph.nodes:
            if node not in [start_node, goal_node]:
                pos = self.graph.nodes[node]['pos']
                if np.linalg.norm(np.array(pos) - np.array(start)) < 2.0:
                    self.graph.add_edge(start_node, node)
                if np.linalg.norm(np.array(pos) - np.array(goal)) < 2.0:
                    self.graph.add_edge(goal_node, node)
                    
        # Find shortest path
        try:
            path = nx.shortest_path(self.graph, start_node, goal_node)
            return [self.graph.nodes[node]['pos'] for node in path]
        except nx.NetworkXNoPath:
            logger.warning("No path found between start and goal")
            return []
            
    def train(self, 
              dataset: Dataset,
              epochs: int = 10,
              batch_size: int = 32,
              learning_rate: float = 0.001):
        """Train the obstacle prediction model."""
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}')
            
    def save_model(self, path: str):
        """Save the trained model."""
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path: str):
        """Load a trained model."""
        self.model.load_state_dict(torch.load(path))
        
    def save_map(self, path: str):
        """Save the current map state."""
        map_data = {
            'occupancy_grid': self.occupancy_grid.tolist(),
            'visited': self.visited.tolist(),
            'resolution': self.resolution,
            'map_size': self.map_size
        }
        with open(path, 'w') as f:
            json.dump(map_data, f)
            
    def load_map(self, path: str):
        """Load a saved map state."""
        with open(path, 'r') as f:
            map_data = json.load(f)
        self.occupancy_grid = np.array(map_data['occupancy_grid'])
        self.visited = np.array(map_data['visited'])
        self.resolution = map_data['resolution']
        self.map_size = tuple(map_data['map_size'])
        
    def visualize_map(self, path: Optional[List[Tuple[float, float]]] = None):
        """Visualize the current map state."""
        plt.figure(figsize=(10, 10))
        
        # Plot occupancy grid
        plt.imshow(self.occupancy_grid.T, cmap='gray', origin='lower')
        
        # Plot visited areas
        visited_mask = np.ma.masked_where(self.visited == 0, self.visited)
        plt.imshow(visited_mask.T, cmap='Blues', alpha=0.3, origin='lower')
        
        # Plot path if provided
        if path:
            path_array = np.array(path)
            plt.plot(path_array[:, 0] / self.resolution,
                    path_array[:, 1] / self.resolution,
                    'r-', linewidth=2)
            
        plt.colorbar(label='Occupancy Probability')
        plt.title('Room Map')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.grid(True)
        plt.show() 