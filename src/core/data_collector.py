import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import json
import time
import os
import threading
import logging
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from pathlib import Path
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DataCollector')

class DataCollector:
    """Data collection interface for manual training."""
    
    def __init__(self, config_path: str = "config/data_collector.yaml"):
        """Initialize the data collector with configuration."""
        try:
            # Load configuration
            self.config = self._load_config(config_path)
            
            # Initialize variables
            self.is_recording = False
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.data_dir = Path(self.config['data_dir']) / self.session_id
            
            # Create data directory
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize camera
            self.cap = self._initialize_camera()
            
            # Initialize sensors
            self.sensor_data = {
                'ultrasonic': [],
                'position': [],
                'motor_commands': [],
                'timestamps': []
            }
            
            # Create UI
            self.root = tk.Tk()
            self.root.title("PiRobot Data Collector")
            self._create_ui()
            
            # Start update thread
            self.update_thread = threading.Thread(target=self._update, daemon=True)
            self.update_thread.start()
            
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            raise
            
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Validate required configuration
            required_keys = ['data_dir', 'camera', 'sensors']
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Missing required configuration key: {key}")
                    
            return config
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {str(e)}")
            raise
            
    def _initialize_camera(self) -> cv2.VideoCapture:
        """Initialize camera with configuration settings."""
        try:
            cap = cv2.VideoCapture(self.config['camera']['device_id'])
            if not cap.isOpened():
                raise RuntimeError("Cannot open camera")
                
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])
            cap.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
            
            return cap
            
        except Exception as e:
            logger.error(f"Camera initialization error: {str(e)}")
            raise
            
    def _create_ui(self):
        """Create the user interface."""
        try:
            # Main frame
            main_frame = ttk.Frame(self.root, padding="10")
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # Video display
            self.video_label = ttk.Label(main_frame)
            self.video_label.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
            
            # Control buttons
            self.record_button = ttk.Button(
                main_frame,
                text="Start Recording",
                command=self._toggle_recording
            )
            self.record_button.grid(row=1, column=0, padx=5, pady=5)
            
            self.save_frame_button = ttk.Button(
                main_frame,
                text="Save Current Frame",
                command=self._save_current_frame
            )
            self.save_frame_button.grid(row=1, column=1, padx=5, pady=5)
            
            self.emergency_stop_button = ttk.Button(
                main_frame,
                text="Emergency Stop",
                command=self._emergency_stop,
                style='Emergency.TButton'
            )
            self.emergency_stop_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
            
            # Status display
            self.status_label = ttk.Label(
                main_frame,
                text="Status: Ready",
                font=('Arial', 10)
            )
            self.status_label.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
            
            # Statistics
            self.stats_frame = ttk.LabelFrame(main_frame, text="Statistics")
            self.stats_frame.grid(row=4, column=0, columnspan=2, padx=5, pady=5)
            
            self.frames_label = ttk.Label(
                self.stats_frame,
                text="Frames: 0"
            )
            self.frames_label.grid(row=0, column=0, padx=5, pady=2)
            
            self.sensor_label = ttk.Label(
                self.stats_frame,
                text="Sensor Readings: 0"
            )
            self.sensor_label.grid(row=0, column=1, padx=5, pady=2)
            
            # Configure styles
            style = ttk.Style()
            style.configure('Emergency.TButton', foreground='red')
            
        except Exception as e:
            logger.error(f"UI creation error: {str(e)}")
            raise
            
    def _toggle_recording(self):
        """Toggle recording state."""
        try:
            self.is_recording = not self.is_recording
            if self.is_recording:
                self.record_button.config(text="Stop Recording")
                self.status_label.config(text="Status: Recording")
                logger.info("Started recording")
            else:
                self.record_button.config(text="Start Recording")
                self.status_label.config(text="Status: Ready")
                logger.info("Stopped recording")
                
        except Exception as e:
            logger.error(f"Error toggling recording: {str(e)}")
            messagebox.showerror("Error", f"Failed to toggle recording: {str(e)}")
            
    def _save_current_frame(self):
        """Save the current frame and sensor data."""
        if not self.is_recording:
            return
            
        try:
            timestamp = time.time()
            frame_path = self.data_dir / f"frame_{timestamp:.3f}.jpg"
            
            # Save frame
            cv2.imwrite(str(frame_path), self.current_frame)
            
            # Validate sensor data
            if not self._validate_sensor_data():
                raise ValueError("Invalid sensor data")
                
            # Save sensor data
            data = {
                'timestamp': timestamp,
                'frame_path': str(frame_path),
                'sensor_data': {
                    'ultrasonic': self.sensor_data['ultrasonic'][-1],
                    'position': self.sensor_data['position'][-1],
                    'motor_commands': self.sensor_data['motor_commands'][-1]
                }
            }
            
            with open(self.data_dir / f"data_{timestamp:.3f}.json", 'w') as f:
                json.dump(data, f, indent=4)
                
            logger.info(f"Saved frame and data at {timestamp}")
            
        except Exception as e:
            logger.error(f"Error saving frame: {str(e)}")
            messagebox.showerror("Error", f"Failed to save frame: {str(e)}")
            
    def _validate_sensor_data(self) -> bool:
        """Validate sensor data before saving."""
        try:
            # Check if all required sensor data is present
            if not all(key in self.sensor_data for key in ['ultrasonic', 'position', 'motor_commands']):
                return False
                
            # Check if data arrays are not empty
            if not all(len(self.sensor_data[key]) > 0 for key in ['ultrasonic', 'position', 'motor_commands']):
                return False
                
            # Validate ultrasonic sensor data
            ultrasonic_data = self.sensor_data['ultrasonic'][-1]
            if not isinstance(ultrasonic_data, list) or len(ultrasonic_data) != 4:
                return False
                
            # Validate position data
            position_data = self.sensor_data['position'][-1]
            if not isinstance(position_data, list) or len(position_data) != 2:
                return False
                
            # Validate motor commands
            motor_data = self.sensor_data['motor_commands'][-1]
            if not isinstance(motor_data, list) or len(motor_data) != 2:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating sensor data: {str(e)}")
            return False
            
    def _emergency_stop(self):
        """Emergency stop all operations."""
        try:
            self.is_recording = False
            self.record_button.config(text="Start Recording")
            self.status_label.config(text="Status: Emergency Stop")
            logger.warning("Emergency stop activated")
            
        except Exception as e:
            logger.error(f"Error in emergency stop: {str(e)}")
            
    def _update(self):
        """Update the interface and collect data."""
        frame_count = 0
        sensor_count = 0
        
        try:
            while True:
                # Read camera frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to grab frame")
                    break
                    
                # Update video display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (640, 480))
                photo = tk.PhotoImage(data=cv2.imencode('.ppm', frame_resized)[1].tobytes())
                self.video_label.config(image=photo)
                self.video_label.image = photo
                
                # Store current frame
                self.current_frame = frame
                
                if self.is_recording:
                    # Collect sensor data
                    # TODO: Replace with actual sensor readings
                    self.sensor_data['ultrasonic'].append([0.0] * 4)  # 4 ultrasonic sensors
                    self.sensor_data['position'].append([0.0, 0.0])  # x, y position
                    self.sensor_data['motor_commands'].append([0.0, 0.0])  # left, right motor
                    self.sensor_data['timestamps'].append(time.time())
                    
                    # Update statistics
                    frame_count += 1
                    sensor_count += 1
                    self.frames_label.config(text=f"Frames: {frame_count}")
                    self.sensor_label.config(text=f"Sensor Readings: {sensor_count}")
                    
                # Update UI
                self.root.update()
                
        except Exception as e:
            logger.error(f"Error in update loop: {str(e)}")
            messagebox.showerror("Error", f"Update loop error: {str(e)}")
            
    def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, 'cap'):
                self.cap.release()
            cv2.destroyAllWindows()
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            
    def run(self):
        """Run the data collector interface."""
        try:
            self.root.mainloop()
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
        finally:
            self.cleanup()
            
if __name__ == '__main__':
    try:
        collector = DataCollector()
        collector.run()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        messagebox.showerror("Fatal Error", f"Application failed to start: {str(e)}") 