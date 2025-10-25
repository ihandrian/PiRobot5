#!/usr/bin/env python3
# Standard library imports
import atexit
import gc
import json
import logging
import math
import os
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional

# Third-party imports
import cv2
import numpy as np
import psutil
import wiringpi as wp
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_socketio import SocketIO
import websockets

# Local imports
from src.core.resource_manager import ResourceManager
from src.core.safety_monitor import SafetyMonitor
from src.core.temperature_monitor import TemperatureMonitor
from src.core.error_handler import ErrorHandler
from src.core.motor_controller import MotorController
from src.core.gps_handler import GPSHandler
from src.core.lane_detector import LaneDetector
from src.core.behavior_cloning import BehaviorCloning
from src.core.waypoint_navigator import WaypointNavigator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/robot.log")
    ]
)
logger = logging.getLogger("PiRobot")

# Disable garbage collector for real-time operations
gc.disable()

# Set up paths
BASE_DIR = Path(__file__).parent.absolute()
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

class PersonDetector:
    """Handles person detection using TensorFlow Lite with optimized performance"""
    
    def __init__(self):
        self.running = True
        self.detection_enabled = False
        self.follow_mode = False
        self.detections = []
        self.detection_lock = threading.Lock()
        self.last_detection_time = 0
        self.detection_interval = 0.033  # ~30 FPS for detection
        
        # Initialize model with optimized settings
        try:
            import tflite_runtime.interpreter as tflite
            self.tflite_available = True
            
            # Path to model files
            model_dir = Path(__file__).parent / "models"
            model_dir.mkdir(exist_ok=True)
            
            model_path = model_dir / "mobilenet_ssd_v2_coco_quant.tflite"
            label_path = model_dir / "coco_labels.txt"
            
            # Download model if not exists
            if not model_path.exists():
                logger.info("Downloading person detection model...")
                self._download_model(
                    "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip",
                    model_dir
                )
            
            # Load model with optimized settings
            self.interpreter = tflite.Interpreter(
                model_path=str(model_path),
                num_threads=4  # Use multiple threads for inference
            )
            self.interpreter.allocate_tensors()
            
            # Get model details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_shape = self.input_details[0]['shape']
            self.height = self.input_shape[1]
            self.width = self.input_shape[2]
            
            # Load labels
            with open(label_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
                
            # Pre-allocate numpy arrays for better performance
            self.input_data = np.zeros(self.input_shape, dtype=np.uint8)
            
            logger.info("Person detector initialized with optimized settings")
            
        except Exception as e:
            logger.error(f"Failed to initialize person detector: {e}")
            self.tflite_available = False
            
    def _download_model(self, url, model_dir):
        """Download and extract model files"""
        try:
            import requests
            import zipfile
            import io
            
            # Download the file
            response = requests.get(url)
            z = zipfile.ZipFile(io.BytesIO(response.content))
            z.extractall(model_dir)
            
            # Rename files to expected names
            for file in model_dir.glob("*.tflite"):
                file.rename(model_dir / "mobilenet_ssd_v2_coco_quant.tflite")
                
            # Create labels file if not exists
            label_path = model_dir / "coco_labels.txt"
            if not label_path.exists():
                # COCO dataset labels
                coco_labels = [
                    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
                ]
                with open(label_path, 'w') as f:
                    for label in coco_labels:
                        f.write(f"{label}\n")
                        
            logger.info("Model downloaded and extracted successfully")
            
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            raise
    
    def enable_detection(self, enabled=True):
        """Enable or disable person detection"""
        self.detection_enabled = enabled and self.tflite_available
        logger.info(f"Person detection {'enabled' if self.detection_enabled else 'disabled'}")
        return self.detection_enabled
    
    def set_follow_mode(self, enabled=True):
        """Enable or disable person following mode"""
        self.follow_mode = enabled and self.detection_enabled
        logger.info(f"Person following {'enabled' if self.follow_mode else 'disabled'}")
        return self.follow_mode
    
    def detect_persons(self, frame):
        """Detect persons in the given frame with minimal latency"""
        if not self.detection_enabled or not self.tflite_available:
            return []
            
        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_interval:
            # Return cached detections if within interval
            with self.detection_lock:
                return self.detections
            
        try:
            # Resize and normalize image efficiently
            image = cv2.resize(frame, (self.width, self.height))
            self.input_data[0] = image
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], self.input_data)
            self.interpreter.invoke()
            
            # Get detection results
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
            
            # Filter for persons efficiently
            persons = []
            for i in range(len(scores)):
                if scores[i] >= 0.5 and classes[i] == 0:  # Person class with confidence > 0.5
                    box = boxes[i]
                    persons.append({
                        'box': [
                            int(box[1] * frame.shape[1]),  # xmin
                            int(box[0] * frame.shape[0]),  # ymin
                            int(box[3] * frame.shape[1]),  # xmax
                            int(box[2] * frame.shape[0])   # ymax
                        ],
                        'score': float(scores[i])
                    })
            
            # Update detections and timestamp
            with self.detection_lock:
                self.detections = persons
                self.last_detection_time = current_time
                
            return persons
            
        except Exception as e:
            logger.error(f"Error in person detection: {e}")
            return []
    
    def get_follow_target(self):
        """Get the nearest person to follow with minimal latency"""
        with self.detection_lock:
            if not self.detections:
                return None
                
            # Find the largest detection (assuming it's the closest)
            largest_area = 0
            target = None
            
            for person in self.detections:
                box = person['box']
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area > largest_area:
                    largest_area = area
                    target = person
                    
            return target
    
    def get_follow_direction(self, frame_width):
        """Calculate direction to move to follow the target person with minimal latency"""
        target = self.get_follow_target()
        if not target:
            return None
            
        box = target['box']
        center_x = (box[0] + box[2]) / 2
        frame_center = frame_width / 2
        
        # Calculate horizontal position relative to center
        position = (center_x - frame_center) / frame_center  # -1 to 1
        
        # Calculate area of bounding box (for distance estimation)
        area = (box[2] - box[0]) * (box[3] - box[1])
        area_ratio = area / (frame_width * frame_width)  # Normalized by frame size
        
        return {
            'position': position,  # -1 (far left) to 1 (far right)
            'area_ratio': area_ratio,  # Approximation of distance
            'box': box
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if hasattr(self, 'interpreter'):
            del self.interpreter
        logger.info("Person detector cleaned up")


class CameraController:
    """Handles camera operations with optimized performance"""
    
    def __init__(self, person_detector=None):
        self.cameras = []
        self.camera_index = 0
        self.current_camera = None
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.processed_frame = None
        self.running = True
        self.person_detector = person_detector
        self.detect_cameras()
        
        # Performance optimization settings
        self.resolution = (320, 240)  # Lower resolution for better performance
        self.frame_interval = 0.016   # ~60 FPS target
        self.frame_buffer = deque(maxlen=2)  # Minimal frame buffer
        self.processing_thread = None
        self.processing_queue = queue.Queue(maxsize=1)  # Single frame queue
        
        # Start optimized threads
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        self.processing_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.processing_thread.start()
        
        logger.info(f"Camera controller initialized with optimized settings. Found cameras: {self.cameras}")
        
    def detect_cameras(self):
        """Detect available cameras"""
        self.cameras = []
        for i in range(5):  # Check the first 5 camera indices
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    self.cameras.append(i)
                    cap.release()
            except Exception as e:
                logger.error(f"Error detecting camera {i}: {e}")
                
        if not self.cameras:
            logger.warning("No cameras detected!")
        
    def set_camera(self, camera_id):
        """Set the active camera"""
        if camera_id in self.cameras:
            self.camera_index = camera_id
            logger.info(f"Switched to camera {camera_id}")
            return True
        else:
            logger.warning(f"Camera {camera_id} not available")
            return False
            
    def _capture_frames(self):
        """Continuously capture frames with minimal latency"""
        while self.running:
            try:
                if self.current_camera is None or self.current_camera != self.camera_index:
                    if self.current_camera is not None:
                        cap = cv2.VideoCapture(self.current_camera)
                        cap.release()
                    
                    self.current_camera = self.camera_index
                    cap = cv2.VideoCapture(self.current_camera)
                    
                    # Optimize camera settings
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                    cap.set(cv2.CAP_PROP_FPS, 60)  # Request 60 FPS
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
                    
                    if not cap.isOpened():
                        logger.error(f"Failed to open camera {self.current_camera}")
                        time.sleep(0.1)  # Shorter sleep on error
                        continue
                
                # Read frame with timeout
                success, frame = cap.read()
                if not success:
                    logger.warning(f"Failed to read frame from camera {self.current_camera}")
                    time.sleep(0.01)  # Minimal sleep on error
                    continue
                
                # Update latest frame immediately
                with self.frame_lock:
                    self.latest_frame = frame
                
                # Queue frame for processing if detection is enabled
                if self.person_detector and self.person_detector.detection_enabled:
                    try:
                        # Non-blocking queue put with timeout
                        self.processing_queue.put(frame, block=False)
                    except queue.Full:
                        # Skip frame if queue is full
                        pass
                
                # Minimal sleep to maintain frame rate
                time.sleep(max(0, self.frame_interval - (time.time() % self.frame_interval)))
                
            except Exception as e:
                logger.error(f"Error in frame capture: {e}")
                time.sleep(0.01)  # Minimal sleep on error
                
    def _process_frames(self):
        """Process frames in separate thread to avoid blocking capture"""
        while self.running:
            try:
                # Get frame from queue with timeout
                frame = self.processing_queue.get(timeout=0.1)
                
                if frame is not None:
                    # Process frame for person detection
                    processed = frame.copy()
                    persons = self.person_detector.detect_persons(frame)
                    
                    # Draw detections
                    for person in persons:
                        box = person['box']
                        score = person['score']
                        cv2.rectangle(processed, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                        cv2.putText(processed, f"Person: {score:.2f}", (box[0], box[1] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Update processed frame
                    with self.frame_lock:
                        self.processed_frame = processed
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in frame processing: {e}")
                time.sleep(0.01)
                
    def generate_frames(self):
        """Generate frames for streaming with minimal latency"""
        while self.running:
            try:
                with self.frame_lock:
                    frame = self.processed_frame if self.processed_frame is not None else self.latest_frame
                    if frame is None:
                        time.sleep(0.01)
                        continue
                
                # Optimize JPEG encoding
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]  # Lower quality for better performance
                _, buffer = cv2.imencode('.jpg', frame, encode_param)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
            except Exception as e:
                logger.error(f"Error encoding frame: {e}")
                time.sleep(0.01)
                
    def cleanup(self):
        """Clean up camera resources"""
        self.running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=0.5)
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=0.5)
        logger.info("Camera controller cleaned up")


class PersonFollower:
    """Controls the robot to follow a person with minimal latency"""
    
    def __init__(self, motor_controller, camera_controller, person_detector):
        self.motor_controller = motor_controller
        self.camera_controller = camera_controller
        self.person_detector = person_detector
        self.running = True
        self.follow_thread = None
        self.follow_speed = 0.5  # Default follow speed
        
        # Optimized following parameters
        self.OPTIMAL_AREA_MIN = 0.15  # Minimum desired area ratio
        self.OPTIMAL_AREA_MAX = 0.30  # Maximum desired area ratio
        self.POSITION_THRESHOLD = 0.2  # Center position threshold
        self.TURN_SPEED_FACTOR = 0.8   # Increased turn speed for responsiveness
        self.FORWARD_SPEED_FACTOR = 0.8 # Speed factor for forward movement
        self.BACKWARD_SPEED_FACTOR = 0.4 # Speed factor for backward movement
        
        # Safety parameters
        self.MIN_DISTANCE = 0.5  # Minimum safe distance (meters)
        self.MAX_TURN_SPEED = 0.8  # Maximum turn speed
        self.EMERGENCY_STOP_DISTANCE = 0.3  # Distance for emergency stop
        
        # Performance optimization
        self.last_update_time = 0
        self.update_interval = 0.016  # ~60 Hz update rate
        self.last_direction = None
        self.direction_change_time = 0
        self.direction_change_delay = 0.1  # Delay before changing direction
        
    def start_following(self):
        """Start the person following thread"""
        if self.follow_thread is None or not self.follow_thread.is_alive():
            self.running = True
            self.follow_thread = threading.Thread(target=self._follow_loop, daemon=True)
            self.follow_thread.start()
            logger.info("Person following started with optimized settings")
            return True
        return False
        
    def stop_following(self):
        """Stop the person following thread"""
        self.running = False
        if self.follow_thread and self.follow_thread.is_alive():
            self.follow_thread.join(timeout=0.5)
            self.motor_controller.move("stop", 0)
            logger.info("Person following stopped")
            return True
        return False
        
    def set_follow_speed(self, speed):
        """Set the following speed"""
        self.follow_speed = max(0.1, min(1.0, float(speed)))
        logger.info(f"Follow speed set to {self.follow_speed}")
        
    def _follow_loop(self):
        """Main loop for person following with minimal latency"""
        while self.running and self.person_detector.follow_mode:
            try:
                current_time = time.time()
                if current_time - self.last_update_time < self.update_interval:
                    time.sleep(max(0, self.update_interval - (current_time - self.last_update_time)))
                    continue
                    
                # Get frame dimensions
                with self.camera_controller.frame_lock:
                    if self.camera_controller.latest_frame is None:
                        time.sleep(0.01)
                        continue
                    frame_width = self.camera_controller.latest_frame.shape[1]
                
                # Get direction to follow
                direction_info = self.person_detector.get_follow_direction(frame_width)
                
                if direction_info:
                    position = direction_info['position']
                    area_ratio = direction_info['area_ratio']
                    
                    # Check safety distance
                    if area_ratio > 0.5:  # Person too close
                        self.motor_controller.move("backward", self.follow_speed * self.BACKWARD_SPEED_FACTOR)
                        time.sleep(0.1)
                        continue
                        
                    # Calculate speed adjustments based on distance
                    distance_factor = self._calculate_distance_factor(area_ratio)
                    adjusted_speed = self.follow_speed * distance_factor
                    
                    # Determine movement direction with minimal latency
                    new_direction = self._determine_movement_direction(position, area_ratio)
                    
                    # Apply direction change delay for safety
                    if new_direction != self.last_direction:
                        if current_time - self.direction_change_time < self.direction_change_delay:
                            continue
                        self.direction_change_time = current_time
                        self.last_direction = new_direction
                    
                    # Execute movement with optimized parameters
                    self._execute_movement(new_direction, adjusted_speed, position)
                else:
                    # No person detected, stop movement
                    self.motor_controller.move("stop", 0)
                    self.last_direction = None
                
                self.last_update_time = current_time
                
            except Exception as e:
                logger.error(f"Error in follow loop: {e}")
                time.sleep(0.01)
                
    def _calculate_distance_factor(self, area_ratio):
        """Calculate speed factor based on distance"""
        if area_ratio < self.OPTIMAL_AREA_MIN:
            return 1.0  # Move faster when far
        elif area_ratio > self.OPTIMAL_AREA_MAX:
            return 0.5  # Slow down when close
        else:
            return 0.8  # Normal speed in optimal range
            
    def _determine_movement_direction(self, position, area_ratio):
        """Determine movement direction with minimal latency"""
        if abs(position) > self.POSITION_THRESHOLD:
            return "left" if position < 0 else "right"
        elif area_ratio < self.OPTIMAL_AREA_MIN:
            return "forward"
        elif area_ratio > self.OPTIMAL_AREA_MAX:
            return "backward"
        else:
            return "stop"
            
    def _execute_movement(self, direction, speed, position):
        """Execute movement with optimized parameters"""
        if direction == "left" or direction == "right":
            # Apply turn speed factor for responsive turning
            turn_speed = min(speed * self.TURN_SPEED_FACTOR, self.MAX_TURN_SPEED)
            self.motor_controller.move(direction, turn_speed)
        elif direction == "forward":
            self.motor_controller.move("forward", speed * self.FORWARD_SPEED_FACTOR)
        elif direction == "backward":
            self.motor_controller.move("backward", speed * self.BACKWARD_SPEED_FACTOR)
        else:
            self.motor_controller.move("stop", 0)
            
    def cleanup(self):
        """Clean up resources"""
        self.stop_following()
        logger.info("Person follower cleaned up")


class RobotWebServer:
    """Web server for robot control"""
    
    def __init__(self, motor_controller, camera_controller, person_detector, person_follower, host="0.0.0.0", port=5002):
        self.app = Flask(__name__)
        self.motor_controller = motor_controller
        self.camera_controller = camera_controller
        self.person_detector = person_detector
        self.person_follower = person_follower
        self.host = host
        self.port = port
        self.server_thread = None
        self.setup_routes()
        
        logger.info(f"Web server initialized on {host}:{port}")
        
    def setup_routes(self):
        """Set up Flask routes"""
        
        @self.app.route("/")
        def home():
            return render_template("index.html")
        
        @self.app.route("/control", methods=["POST"])
        def control():
            action = request.form["action"]
            speed = float(request.form["speed"])
            camera_id = request.form.get("camera_id", type=int)
            
            if camera_id is not None:
                self.camera_controller.set_camera(camera_id)
                
            self.motor_controller.move(action, speed)
            return "OK"
        
        @self.app.route("/video_feed")
        def video_feed():
            return Response(self.camera_controller.generate_frames(), 
                           mimetype='multipart/x-mixed-replace; boundary=frame')
                           
        @self.app.route("/detection", methods=["POST"])
        def detection_control():
            action = request.form.get("action")
            
            if action == "enable":
                enabled = self.person_detector.enable_detection(True)
                return jsonify({"status": "ok", "enabled": enabled})
            elif action == "disable":
                enabled = self.person_detector.enable_detection(False)
                return jsonify({"status": "ok", "enabled": enabled})
            else:
                return jsonify({"status": "error", "message": "Invalid action"})
                
        @self.app.route("/follow", methods=["POST"])
        def follow_control():
            action = request.form.get("action")
            
            if action == "start":
                # Enable detection and follow mode
                self.person_detector.enable_detection(True)
                self.person_detector.set_follow_mode(True)
                # Start following
                success = self.person_follower.start_following()
                return jsonify({"status": "ok", "following": success})
            elif action == "stop":
                # Disable follow mode
                self.person_detector.set_follow_mode(False)
                # Stop following
                success = self.person_follower.stop_following()
                return jsonify({"status": "ok", "following": not success})
            elif action == "speed":
                speed = request.form.get("speed", type=float)
                if speed is not None:
                    self.person_follower.set_follow_speed(speed)
                    return jsonify({"status": "ok", "speed": speed})
                else:
                    return jsonify({"status": "error", "message": "Invalid speed"})
            else:
                return jsonify({"status": "error", "message": "Invalid action"})
                
        @self.app.route("/detection_info")
        def detection_info():
            with self.camera_controller.frame_lock:
                if self.camera_controller.latest_frame is None:
                    return jsonify({"info": "No camera feed available"})
                    
            if not self.person_detector.tflite_available:
                return jsonify({"info": "Person detection not available (TFLite missing)"})
                
            if not self.person_detector.detection_enabled:
                return jsonify({"info": "Person detection disabled"})
                
            with self.person_detector.detection_lock:
                num_persons = len(self.person_detector.detections)
                
            if self.person_detector.follow_mode:
                target = self.person_detector.get_follow_target()
                if target:
                    return jsonify({
                        "info": f"Following person. {num_persons} person(s) detected.",
                        "following": True,
                        "persons": num_persons
                    })
                else:
                    return jsonify({
                        "info": f"Searching for person to follow. {num_persons} person(s) detected.",
                        "following": False,
                        "persons": num_persons
                    })
            else:
                return jsonify({
                    "info": f"{num_persons} person(s) detected.",
                    "following": False,
                    "persons": num_persons
                })
            
        @self.app.route("/system_info")
        def system_info():
            # Basic system information
            try:
                cpu = psutil.cpu_percent()
                memory = psutil.virtual_memory().percent
                temperature = None
                
                try:
                    # Try to get Raspberry Pi temperature
                    with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                        temperature = float(f.read()) / 1000.0
                except:
                    pass
                    
                return jsonify({
                    "cpu": cpu,
                    "memory": memory,
                    "temperature": temperature,
                    "cameras": self.camera_controller.cameras,
                    "active_camera": self.camera_controller.camera_index,
                    "detection_enabled": self.person_detector.detection_enabled,
                    "follow_mode": self.person_detector.follow_mode
                })
            except Exception as e:
                logger.error(f"Error getting system info: {e}")
                return jsonify({"error": "Failed to get system information"})
        
        @self.app.route("/navigation", methods=["POST"])
        def navigation_control():
            mode = request.form.get("mode")
            success = self.robot.set_navigation_mode(mode)
            return jsonify({
                "status": "ok" if success else "error",
                "mode": self.robot.navigation_mode
            })
        
        @self.app.route("/learning", methods=["POST"])
        def learning_control():
            action = request.form.get("action")
            if action == "enable":
                self.robot.toggle_learning(True)
                return jsonify({"status": "ok", "learning": True})
            elif action == "disable":
                self.robot.toggle_learning(False)
                return jsonify({"status": "ok", "learning": False})
            elif action == "save":
                filename = request.form.get("filename", "learned_behavior.pkl")
                self.robot.save_learned_behavior(filename)
                return jsonify({"status": "ok", "saved": True})
            else:
                return jsonify({"status": "error", "message": "Invalid action"})
        
        @self.app.route("/waypoints", methods=["GET", "POST", "DELETE"])
        def waypoint_control():
            if request.method == "GET":
                return jsonify({
                    "waypoints": [
                        {
                            "id": wp.id,
                            "lat": wp.lat,
                            "lon": wp.lon,
                            "visited": wp.visited
                        } for wp in self.robot.waypoint_navigator.waypoints
                    ],
                    "is_navigating": self.robot.waypoint_navigator.is_navigating
                })
                
            elif request.method == "POST":
                if request.form.get("action") == "start":
                    success = self.robot.waypoint_navigator.start_navigation()
                    return jsonify({"status": "ok", "started": success})
                    
                elif request.form.get("action") == "stop":
                    self.robot.waypoint_navigator.stop_navigation()
                    return jsonify({"status": "ok"})
                    
                elif request.form.get("action") == "clear":
                    self.robot.waypoint_navigator.clear_waypoints()
                    return jsonify({"status": "ok"})
                    
                else:
                    # Add new waypoint
                    lat = float(request.form["lat"])
                    lon = float(request.form["lon"])
                    waypoint_id = self.robot.waypoint_navigator.add_waypoint(lat, lon)
                    return jsonify({"status": "ok", "id": waypoint_id})
                    
            elif request.method == "DELETE":
                waypoint_id = request.form["id"]
                success = self.robot.waypoint_navigator.remove_waypoint(waypoint_id)
                return jsonify({"status": "ok" if success else "error"})
        
        @self.app.route('/motor', methods=['POST'])
        def motor_control():
            try:
                direction = request.form.get('direction')
                speed = float(request.form.get('speed', 0))
                self.motor_controller.move(direction, speed)
                return jsonify({'status': 'ok'})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)})
        
        @self.app.route('/calibrate', methods=['POST'])
        def calibrate_motors():
            try:
                action = request.form.get('action')
                if action == 'start':
                    robot.motor_controller.calibrate_straight()
                    return jsonify({'status': 'ok', 'message': 'Calibration completed'})
                elif action == 'save':
                    robot.motor_controller.save_calibration()
                    return jsonify({'status': 'ok', 'message': 'Calibration saved'})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)})
        
    def start(self):
        """Start the web server in a separate thread"""
        self.server_thread = threading.Thread(
            target=lambda: self.app.run(
                host=self.host, 
                port=self.port, 
                debug=False, 
                use_reloader=False,
                threaded=True
            ),
            daemon=True
        )
        self.server_thread.start()
        logger.info(f"Web server running at http://{self.host}:{self.port}")
        
    def cleanup(self):
        """Clean up web server resources"""
        logger.info("Web server shutting down")


class JoystickController:
    """Handles joystick control via WebSocket"""
    
    def __init__(self, motor_controller, host="0.0.0.0", port=5003):
        self.motor_controller = motor_controller
        self.host = host
        self.port = port
        self.server = None
        self.running = True
        
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connection"""
        try:
            # Add message size limits
            websocket.max_size = 1024  # Limit message size
            websocket.max_queue = 4    # Limit message queue
            
            async for message in websocket:
                if not self.running:
                    break
                    
                try:
                    data = json.loads(message)
                    if data['type'] == 'joystick':
                        # Convert joystick position to motor commands
                        x = float(data['x'])  # Left/Right
                        y = float(data['y'])  # Forward/Backward
                        speed = float(data['speed'])
                        
                        # Calculate motor commands based on joystick position
                        self._process_joystick(x, y, speed)
                        
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                except Exception as e:
                    logger.error(f"Error processing joystick data: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            # Stop motors when connection is closed
            self.motor_controller.move("stop", 0)
            
    def _process_joystick(self, x, y, speed):
        """Convert joystick position to motor commands"""
        # Dead zone to prevent small unwanted movements
        if abs(x) < 0.1 and abs(y) < 0.1:
            self.motor_controller.move("stop", 0)
            return
            
        # Calculate final speed
        final_speed = speed * math.sqrt(x*x + y*y)
        final_speed = min(1.0, final_speed)  # Ensure speed doesn't exceed 1.0
        
        # Determine direction based on joystick position
        if abs(x) > abs(y):
            # Turning left or right
            if x < 0:
                self.motor_controller.move("left", final_speed)
            else:
                self.motor_controller.move("right", final_speed)
        else:
            # Moving forward or backward
            if y < 0:
                self.motor_controller.move("forward", final_speed)
            else:
                self.motor_controller.move("backward", final_speed)
                
    async def start(self):
        """Start the WebSocket server"""
        self.server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port
        )
        logger.info(f"Joystick WebSocket server running on ws://{self.host}:{self.port}")
        
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.server:
            self.server.close()
        logger.info("Joystick controller cleaned up")


class AutonomousNavigation:
    def __init__(self, motor_controller, camera_controller):
        self.motor_controller = motor_controller
        self.camera_controller = camera_controller
        self.lane_detector = LaneDetector()
        self.running = True
        self.navigation_thread = None
        self.behavior_learning = BehaviorCloning()
        self.learning_enabled = False
        
        # Navigation parameters
        self.min_line_length = 50
        self.max_line_gap = 50
        self.turn_threshold = 0.3
        self.base_speed = 0.5
        self.lane_center_threshold = 50  # pixels
        self.turn_speed_multiplier = 0.7
        
    def enable_learning(self, enabled=True):
        """Enable/disable behavior learning"""
        self.learning_enabled = enabled
        logger.info(f"Behavior learning {'enabled' if enabled else 'disabled'}")
        
    def _calculate_steering(self, lane_info):
        """Calculate steering angle from lane information"""
        if not lane_info:
            return 0
            
        try:
            frame_width = lane_info['frame_width']
            center_x = frame_width // 2
            
            # Calculate average position of lanes
            left_x = 0
            right_x = frame_width
            
            if lane_info['left_lines']:
                left_x = np.mean([line[0] for line in lane_info['left_lines']])
            if lane_info['right_lines']:
                right_x = np.mean([line[0] for line in lane_info['right_lines']])
                
            # Calculate center of lane
            lane_center = (left_x + right_x) // 2
            
            # Calculate offset from center
            offset = lane_center - center_x
            
            # Normalize to [-1, 1]
            return offset / (frame_width // 2)
            
        except Exception as e:
            logger.error(f"Steering calculation error: {e}")
            return 0
            
    def _navigate(self, steering_angle, obstacles):
        """Execute navigation commands with improved lane following"""
        try:
            if obstacles:
                self.motor_controller.move("stop", 0)
                logger.warning("Obstacle detected - stopping")
                return
                
            # Calculate turn speed based on steering angle
            turn_speed = abs(steering_angle) * self.turn_speed_multiplier
            turn_speed = min(turn_speed, self.base_speed)
            
            if abs(steering_angle) < 0.1:
                # Go straight
                self.motor_controller.move("forward", self.base_speed)
            else:
                # Turn while moving forward
                direction = "left" if steering_angle < 0 else "right"
                self.motor_controller.move(direction, turn_speed)
                
        except Exception as e:
            logger.error(f"Navigation error: {e}")
            self.motor_controller.move("stop", 0)

    def start_navigation(self):
        """Start autonomous navigation"""
        if self.navigation_thread is None or not self.navigation_thread.is_alive():
            self.running = True
            self.navigation_thread = threading.Thread(target=self._navigation_loop)
            self.navigation_thread.start()
            return True
        return False
    
    def stop_navigation(self):
        """Stop autonomous navigation"""
        self.running = False
        if self.navigation_thread and self.navigation_thread.is_alive():
            self.navigation_thread.join(timeout=1.0)
            self.motor_controller.move("stop", 0)
            return True
        return False
    
    def _navigation_loop(self):
        """Main navigation loop with learning"""
        while self.running:
            try:
                with self.camera_controller.frame_lock:
                    frame = self.camera_controller.latest_frame
                    if frame is None:
                        continue
            
                # Detect lanes and obstacles
                lanes = self.lane_detector.detect_lanes(frame)
                obstacles = self.detect_obstacles(frame)
                
                # Calculate steering angle
                steering_angle = self._calculate_steering(lanes)
                
                # Store experience if learning is enabled
                if self.learning_enabled:
                    state = self._preprocess_frame(frame)
                    action = steering_angle
                    reward = self._calculate_reward(steering_angle, obstacles)
                    self.behavior_learning.add_experience(state, action, reward)
                
                # Navigate based on current conditions
                self._navigate(steering_angle, obstacles)
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Navigation error: {e}")
                time.sleep(0.5)


class HybridNavigation:
    def __init__(self, autonomous_nav, person_follower):
        self.autonomous_nav = autonomous_nav
        self.person_follower = person_follower
        self.running = True
        self.hybrid_thread = None
        
    def start(self):
        """Start hybrid navigation"""
        if self.hybrid_thread is None or not self.hybrid_thread.is_alive():
            self.running = True
            self.hybrid_thread = threading.Thread(target=self._hybrid_loop, daemon=True)
            self.hybrid_thread.start()
            logger.info("Hybrid navigation started")
            return True
        return False
        
    def stop(self):
        """Stop hybrid navigation"""
        self.running = False
        if self.hybrid_thread and self.hybrid_thread.is_alive():
            self.hybrid_thread.join(timeout=1.0)
            self.autonomous_nav.stop_navigation()
            self.person_follower.stop_following()
            logger.info("Hybrid navigation stopped")
            return True
        return False
        
    def _hybrid_loop(self):
        """Main loop for hybrid navigation"""
        while self.running:
            try:
                # Check if autonomous navigation is needed
                if self.autonomous_nav.running:
                    self.autonomous_nav._navigation_loop()
                else:
                    # Check if person following is needed
                    if self.person_follower.running:
                        self.person_follower._follow_loop()
                    else:
                        # No navigation needed, stop
                        self.autonomous_nav.stop_navigation()
                        self.person_follower.stop_following()
                
                time.sleep(0.1)  # Control loop rate
                
            except Exception as e:
                logger.error(f"Hybrid navigation error: {e}")
                time.sleep(0.5)


class Robot:
    """Main robot class that coordinates all components"""
    
    def __init__(self):
        # Initialize components
        self.person_detector = PersonDetector()
        self.motor_controller = MotorController()
        self.camera_controller = CameraController(self.person_detector)
        self.person_follower = PersonFollower(
            self.motor_controller, 
            self.camera_controller, 
            self.person_detector
        )
        self.joystick_controller = JoystickController(self.motor_controller)
        self.web_server = RobotWebServer(
            self.motor_controller, 
            self.camera_controller, 
            self.person_detector,
            self.person_follower
        )
        self.autonomous_nav = AutonomousNavigation(
            self.motor_controller, 
            self.camera_controller
        )
        self.hybrid_nav = HybridNavigation(
            self.autonomous_nav,
            self.person_follower
        )
        self.behavior_learning = BehaviorCloning()
        self.resource_manager = ResourceManager()
        self.error_handler = ErrorHandler(self)
        self.navigation_mode = "manual"  # manual/autonomous/hybrid
        self.learning_enabled = False
        self.gps = GPSHandler()
        self.waypoint_navigator = WaypointNavigator(self)
        
        # Start GPS in init
        if not self.gps.start():
            self.logger.warning("GPS initialization failed")
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)
        
        # Register cleanup handlers
        atexit.register(self.cleanup)
        
        logger.info("Robot initialized")
        
    def start(self):
        """Start all robot components"""
        try:
            self.web_server.start()
            asyncio.get_event_loop().run_until_complete(
                self.joystick_controller.start()
            )
            
            # Start resource monitoring
            self.monitor_thread = threading.Thread(
                target=self._monitor_resources,
                daemon=True
            )
            self.monitor_thread.start()
            
            asyncio.get_event_loop().run_forever()
            
        except Exception as e:
            self.error_handler.handle_error("startup", e)
            
    def _monitor_resources(self):
        """Monitor system resources"""
        while True:
            try:
                self.resource_manager.optimize_resources()
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")

    def cleanup(self, *args):
        """Cleanup method for graceful shutdown"""
        if hasattr(self, 'motor_controller'):
            self.motor_controller.cleanup()
        if hasattr(self, 'joystick_controller'):
            self.joystick_controller.cleanup()
        if hasattr(self, 'person_follower'):
            self.person_follower.cleanup()
        if hasattr(self, 'camera_controller'):
            self.camera_controller.cleanup()
        if hasattr(self, 'person_detector'):
            self.person_detector.cleanup()
        if hasattr(self, 'web_server'):
            self.web_server.cleanup()
        if hasattr(self, 'autonomous_nav'):
            self.autonomous_nav.stop_navigation()
        if hasattr(self, 'hybrid_nav'):
            self.hybrid_nav.stop()
        logger.info("Robot shutdown complete")

    def set_navigation_mode(self, mode):
        """Switch between navigation modes"""
        if mode not in ["manual", "autonomous", "hybrid"]:
            logger.error(f"Invalid navigation mode: {mode}")
            return False
            
        self.navigation_mode = mode
        logger.info(f"Switched to {mode} navigation mode")
        
        if mode == "autonomous":
            self.person_detector.enable_detection(False)
            self.person_follower.stop_following()
            self.autonomous_nav.start_navigation()
        elif mode == "hybrid":
            self.person_detector.enable_detection(True)
            self.hybrid_nav.start()
        else:  # manual
            self.autonomous_nav.stop_navigation()
            self.hybrid_nav.stop()
            
        return True

    def toggle_learning(self, enabled=True):
        """Toggle behavior learning"""
        self.learning_enabled = enabled
        self.autonomous_nav.enable_learning(enabled)
        logger.info(f"Behavior learning {'enabled' if enabled else 'disabled'}")
        
    def save_learned_behavior(self, filename):
        """Save learned behavior to file"""
        if hasattr(self.autonomous_nav, 'behavior_learning'):
            # Implementation depends on your model format
            pass

    def check_system_health(self):
        """Comprehensive system health check"""
        try:
            # Check component status
            component_status = {
                'camera': self.camera_controller.running,
                'motors': self._check_motors(),
                'detection': self.person_detector.tflite_available,
                'websocket': self.joystick_controller.running,
                'web_server': self._check_web_server()
            }
            
            # Get performance metrics
            performance = self.resource_manager.monitor_performance()
            
            # Combine status and metrics
            health_report = {
                'components': component_status,
                'performance': performance,
                'errors': self.error_handler.error_count,
                'mode': self.navigation_mode,
                'gps': self.gps.get_location()
            }
            
            return health_report
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return None
            
    def _check_motors(self):
        """Check motor controller status"""
        try:
            # Test motors with minimal movement
            self.motor_controller.move("forward", 0.1)
            time.sleep(0.1)
            self.motor_controller.move("stop", 0)
            return True
        except:
            return False
            
    def _check_web_server(self):
        """Check web server status"""
        return bool(self.web_server.server_thread 
                   and self.web_server.server_thread.is_alive())

    def get_status(self):
        # Add to existing status method
        status = {
            # ... existing status data ...
            'gps': self.gps.get_location()
        }
        return status


class ResourceManager:
    def __init__(self):
        self.memory_threshold = 0.8
        self.cpu_threshold = 0.9
        self.frame_buffer_size = 10
        self.current_fps = 30
        self.performance_history = deque(maxlen=100)
        self.last_check = time.time()
        
    def optimize_resources(self):
        try:
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            if memory_percent > self.memory_threshold * 100:
                self._optimize_memory()
                
            if cpu_percent > self.cpu_threshold * 100:
                self._optimize_cpu()
                
        except Exception as e:
            logger.error(f"Resource optimization error: {e}")
            
    def _optimize_memory(self):
        gc.collect()
        self.frame_buffer_size = max(3, self.frame_buffer_size - 2)
        logger.info(f"Reduced frame buffer size to {self.frame_buffer_size}")
        
    def _optimize_cpu(self):
        self.current_fps = max(10, self.current_fps - 5)
        interval = 1.0 / self.current_fps
        self.camera_controller.frame_interval = interval
        logger.info(f"Reduced frame rate to {self.current_fps} FPS")

    def monitor_performance(self):
        """Monitor and log system performance"""
        try:
            metrics = {
                'timestamp': time.time(),
                'cpu': psutil.cpu_percent(interval=1),
                'memory': psutil.virtual_memory().percent,
                'temperature': self._get_cpu_temperature(),
                'disk': psutil.disk_usage('/').percent,
                'frame_rate': self.current_fps
            }
            
            # Log warnings for high resource usage
            if metrics['cpu'] > 80:
                logger.warning(f"High CPU usage: {metrics['cpu']}%")
            if metrics['memory'] > 80:
                logger.warning(f"High memory usage: {metrics['memory']}%")
            if metrics['temperature'] > 75:
                logger.warning(f"High temperature: {metrics['temperature']}°C")
                
            return metrics
            
        except Exception as e:
            logger.error(f"Performance monitoring error: {e}")
            return None
            
    def _get_cpu_temperature(self):
        """Get CPU temperature on Raspberry Pi"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read()) / 1000.0
            return temp
        except:
            return None


class ErrorHandler:
    def __init__(self, robot):
        self.robot = robot
        self.error_count = {}
        self.max_retries = 3
        self.recovery_cooldown = 5  # seconds
        self.last_recovery = {}
        
    def handle_error(self, component, error):
        current_time = time.time()
        
        # Check cooldown
        if component in self.last_recovery:
            if current_time - self.last_recovery[component] < self.recovery_cooldown:
                logger.warning(f"Skipping {component} recovery - in cooldown")
                return
                
        self.error_count[component] = self.error_count.get(component, 0) + 1
        logger.error(f"{component} error: {error}")
        
        if self.error_count[component] <= self.max_retries:
            self.recover_component(component)
            self.last_recovery[component] = current_time
        else:
            logger.critical(f"{component} failed after {self.max_retries} retries")
            self.robot.cleanup()
            
    def recover_component(self, component):
        logger.info(f"Attempting to recover {component}")
        try:
            if component == "camera":
                self.robot.camera_controller.cleanup()
                time.sleep(1)
                self.robot.camera_controller.__init__(self.robot.person_detector)
            elif component == "motors":
                self.robot.motor_controller.move("stop", 0)
                time.sleep(1)
                self.robot.motor_controller.__init__()
            elif component == "detection":
                self.robot.person_detector.enable_detection(False)
                time.sleep(1)
                self.robot.person_detector.enable_detection(True)
        except Exception as e:
            logger.error(f"Recovery failed for {component}: {e}")


if __name__ == "__main__":
    robot = Robot()
    robot.start()

