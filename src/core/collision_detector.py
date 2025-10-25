import numpy as np
import cv2
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import logging
from .coral_tpu import CoralTPU

logger = logging.getLogger('PiRobot-Collision')

@dataclass
class CollisionPrediction:
    """Data structure for collision predictions"""
    time_to_collision: float
    collision_point: Tuple[float, float]
    severity: float  # 0-1 scale
    confidence: float  # 0-1 scale
    object_type: str  # Type of detected object

class CollisionDetector:
    """Hierarchical collision detection system with Coral TPU integration."""
    
    def __init__(self, 
                 camera_id: int = 0,
                 frame_size: Tuple[int, int] = (640, 480),
                 update_rate: float = 30.0,
                 buffer_size: int = 10,
                 tpu_model_path: str = None,
                 tpu_labels_path: str = None):
        """Initialize collision detector with Coral TPU support."""
        self.logger = logging.getLogger('PiRobot.Collision')
        
        # Camera setup
        self.camera_id = camera_id
        self.frame_size = frame_size
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
        
        # Performance optimization
        self.update_rate = update_rate
        self.update_interval = 1.0 / update_rate
        self.last_update_time = 0
        self.buffer_size = buffer_size
        
        # State tracking
        self.obstacle_buffer = deque(maxlen=buffer_size)
        self.velocity_buffer = deque(maxlen=buffer_size)
        self.acceleration_buffer = deque(maxlen=buffer_size)
        self.last_position = None
        self.last_velocity = None
        
        # Coral TPU setup
        if tpu_model_path and tpu_labels_path:
            try:
                self.tpu = CoralTPU(
                    model_path=tpu_model_path,
                    labels_path=tpu_labels_path,
                    update_rate=update_rate
                )
                self.use_tpu = True
                logger.info("Coral TPU integration enabled")
            except Exception as e:
                self.use_tpu = False
                logger.warning(f"Failed to initialize Coral TPU: {e}")
        else:
            self.use_tpu = False
            logger.info("Coral TPU not configured")
        
        # Start detection thread
        self.running = True
        self.detection_thread = threading.Thread(
            target=self._detection_loop,
            daemon=True
        )
        self.detection_thread.start()
        
    def _detection_loop(self):
        """Main collision detection loop with Coral TPU integration."""
        while self.running:
            try:
                current_time = time.time()
                
                # Check if it's time for an update
                if current_time - self.last_update_time < self.update_interval:
                    time.sleep(0.001)  # Small sleep to prevent CPU hogging
                    continue
                    
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    continue
                    
                # Update TPU with current frame
                if self.use_tpu:
                    self.tpu.update_frame(frame)
                    tpu_detections = self.tpu.get_detections()
                else:
                    tpu_detections = []
                    
                # Process frame
                processed_frame = self._process_frame(frame)
                
                # Detect obstacles
                obstacles = self._detect_obstacles(processed_frame, tpu_detections)
                
                # Update state tracking
                self._update_state_tracking(obstacles)
                
                # Predict collisions
                predictions = self._predict_collisions()
                
                # Store results
                self.obstacle_buffer.append(obstacles)
                
                # Update timing
                self.last_update_time = current_time
                
            except Exception as e:
                logger.error(f"Collision detection error: {e}")
                time.sleep(0.1)
                
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame for obstacle detection."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            return edges
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return frame
            
    def _detect_obstacles(self, frame: np.ndarray, tpu_detections: List[Dict]) -> List[Dict]:
        """Detect obstacles using both traditional and TPU-based methods."""
        try:
            obstacles = []
            
            # Process TPU detections
            for detection in tpu_detections:
                bbox = detection['bbox']
                score = detection['score']
                label = detection['label']
                
                # Convert bbox to obstacle format
                x, y, w, h = bbox
                center = (x + w/2, y + h/2)
                distance = self._calculate_distance(center)
                
                obstacles.append({
                    'center': center,
                    'size': (w, h),
                    'distance': distance,
                    'area': w * h,
                    'type': label,
                    'confidence': score,
                    'source': 'tpu'
                })
            
            # Traditional obstacle detection
            contours, _ = cv2.findContours(
                frame, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in contours:
                # Calculate contour properties
                area = cv2.contourArea(contour)
                if area < 100:  # Filter small contours
                    continue
                    
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate center and distance
                center = (x + w/2, y + h/2)
                distance = self._calculate_distance(center)
                
                # Check if this obstacle overlaps with TPU detections
                if not self._overlaps_with_tpu_detections(center, tpu_detections):
                    obstacles.append({
                        'center': center,
                        'size': (w, h),
                        'distance': distance,
                        'area': area,
                        'type': 'unknown',
                        'confidence': 0.5,
                        'source': 'traditional'
                    })
                
            return obstacles
            
        except Exception as e:
            logger.error(f"Obstacle detection error: {e}")
            return []
            
    def _overlaps_with_tpu_detections(self, center: Tuple[float, float], 
                                     tpu_detections: List[Dict]) -> bool:
        """Check if a point overlaps with TPU detections."""
        try:
            for detection in tpu_detections:
                bbox = detection['bbox']
                x, y, w, h = bbox
                
                # Check if center is within bbox
                if (x <= center[0] <= x + w and 
                    y <= center[1] <= y + h):
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Overlap check error: {e}")
            return False
            
    def _calculate_distance(self, point: Tuple[float, float]) -> float:
        """Calculate distance to point using camera calibration."""
        try:
            # Implement actual distance calculation
            # For now using simulated value
            return np.random.normal(1.0, 0.1)
        except Exception as e:
            logger.error(f"Distance calculation error: {e}")
            return float('inf')
            
    def _update_state_tracking(self, obstacles: List[Dict]):
        """Update state tracking for predictive collision avoidance."""
        try:
            if not obstacles:
                return
                
            # Calculate current position (average of obstacle centers)
            current_position = np.mean(
                [obs['center'] for obs in obstacles],
                axis=0
            )
            
            # Calculate velocity if we have a previous position
            if self.last_position is not None:
                time_delta = time.time() - self.last_update_time
                if time_delta > 0:
                    velocity = (current_position - self.last_position) / time_delta
                    self.velocity_buffer.append(velocity)
                    
                    # Calculate acceleration if we have a previous velocity
                    if self.last_velocity is not None:
                        acceleration = (velocity - self.last_velocity) / time_delta
                        self.acceleration_buffer.append(acceleration)
                        
                    self.last_velocity = velocity
                    
            self.last_position = current_position
            
        except Exception as e:
            logger.error(f"State tracking error: {e}")
            
    def _predict_collisions(self) -> List[CollisionPrediction]:
        """Predict potential collisions."""
        try:
            if not self.velocity_buffer or not self.obstacle_buffer:
                return []
                
            predictions = []
            
            # Get current state
            current_velocity = np.mean(self.velocity_buffer, axis=0)
            current_acceleration = np.mean(self.acceleration_buffer, axis=0)
            
            # Predict for each obstacle
            for obstacle in self.obstacle_buffer[-1]:
                # Calculate relative velocity
                relative_velocity = current_velocity
                
                # Calculate time to collision
                distance = obstacle['distance']
                if np.linalg.norm(relative_velocity) > 0:
                    time_to_collision = distance / np.linalg.norm(relative_velocity)
                else:
                    time_to_collision = float('inf')
                    
                # Calculate collision point
                collision_point = (
                    obstacle['center'][0] + relative_velocity[0] * time_to_collision,
                    obstacle['center'][1] + relative_velocity[1] * time_to_collision
                )
                
                # Calculate severity (based on distance and velocity)
                severity = min(1.0, 1.0 / (distance * np.linalg.norm(relative_velocity)))
                
                # Calculate confidence
                confidence = min(1.0, len(self.velocity_buffer) / self.buffer_size)
                
                predictions.append(CollisionPrediction(
                    time_to_collision=time_to_collision,
                    collision_point=collision_point,
                    severity=severity,
                    confidence=confidence,
                    object_type=obstacle['type']
                ))
                
            return predictions
            
        except Exception as e:
            logger.error(f"Collision prediction error: {e}")
            return []
            
    def get_collision_status(self) -> Dict:
        """Get current collision status."""
        try:
            return {
                'obstacles': self.obstacle_buffer[-1] if self.obstacle_buffer else [],
                'predictions': self._predict_collisions(),
                'velocity': np.mean(self.velocity_buffer, axis=0) if self.velocity_buffer else None,
                'acceleration': np.mean(self.acceleration_buffer, axis=0) if self.acceleration_buffer else None,
                'tpu_enabled': self.use_tpu
            }
        except Exception as e:
            logger.error(f"Status retrieval error: {e}")
            return {}
            
    def cleanup(self):
        """Cleanup resources."""
        self.running = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()
        if self.use_tpu:
            self.tpu.cleanup()
        self.obstacle_buffer.clear()
        self.velocity_buffer.clear()
        self.acceleration_buffer.clear()
        
    def __del__(self):
        """Cleanup resources."""
        self.cleanup() 