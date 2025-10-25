import time
import logging
import threading
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file

logger = logging.getLogger('PiRobot-Coral')

class CoralTPU:
    """Handles Coral TPU operations for object detection and collision avoidance."""
    
    def __init__(self,
                 model_path: str,
                 labels_path: str,
                 confidence_threshold: float = 0.5,
                 update_rate: float = 30.0):
        """Initialize Coral TPU with optimized settings."""
        self.logger = logging.getLogger('PiRobot.Coral')
        
        # Configuration
        self.model_path = model_path
        self.labels_path = labels_path
        self.confidence_threshold = confidence_threshold
        self.update_rate = update_rate
        self.update_interval = 1.0 / update_rate
        
        # Initialize TPU
        self._setup_tpu()
        
        # State tracking
        self.last_update_time = 0
        self.detections = []
        self.running = True
        
        # Start detection thread
        self.detection_thread = threading.Thread(
            target=self._detection_loop,
            daemon=True
        )
        self.detection_thread.start()
        
    def _setup_tpu(self):
        """Setup Coral TPU with error handling."""
        try:
            # Initialize interpreter
            self.interpreter = make_interpreter(self.model_path)
            self.interpreter.allocate_tensors()
            
            # Get model details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Load labels
            self.labels = read_label_file(self.labels_path)
            
            # Get input size
            self.input_size = common.input_size(self.interpreter)
            
            logger.info("Coral TPU initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Coral TPU: {e}")
            raise
            
    def _detection_loop(self):
        """Main detection loop for Coral TPU."""
        while self.running:
            try:
                current_time = time.time()
                
                # Check if it's time for an update
                if current_time - self.last_update_time < self.update_interval:
                    time.sleep(0.001)  # Small sleep to prevent CPU hogging
                    continue
                    
                # Process frame if available
                if hasattr(self, 'current_frame'):
                    detections = self._process_frame(self.current_frame)
                    self.detections = detections
                    
                # Update timing
                self.last_update_time = current_time
                
            except Exception as e:
                logger.error(f"Detection loop error: {e}")
                time.sleep(0.1)
                
    def _process_frame(self, frame: np.ndarray) -> List[Dict]:
        """Process frame using Coral TPU."""
        try:
            # Resize frame to model input size
            resized = cv2.resize(frame, self.input_size)
            
            # Convert to RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Add batch dimension
            input_data = np.expand_dims(rgb, axis=0)
            
            # Set input tensor
            common.set_input(self.interpreter, input_data)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get detection results
            objs = detect.get_objects(
                self.interpreter,
                self.confidence_threshold,
                (1, 1)
            )
            
            # Convert to list of detections
            detections = []
            for obj in objs:
                detection = {
                    'bbox': obj.bbox,
                    'score': obj.score,
                    'id': obj.id,
                    'label': self.labels.get(obj.id, 'unknown')
                }
                detections.append(detection)
                
            return detections
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return []
            
    def update_frame(self, frame: np.ndarray):
        """Update current frame for processing."""
        self.current_frame = frame.copy()
        
    def get_detections(self) -> List[Dict]:
        """Get current detections."""
        return self.detections
        
    def get_detection_status(self) -> Dict:
        """Get current detection status."""
        try:
            return {
                'detections': self.detections,
                'update_rate': self.update_rate,
                'confidence_threshold': self.confidence_threshold
            }
        except Exception as e:
            logger.error(f"Status retrieval error: {e}")
            return {}
            
    def cleanup(self):
        """Cleanup resources."""
        self.running = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1.0)
        del self.interpreter
        
    def __del__(self):
        """Cleanup resources."""
        self.cleanup() 