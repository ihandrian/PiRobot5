#!/usr/bin/env python3
"""
PiRobot V.4 Start Script
This script starts the PiRobot V.4 system with optimized performance.
"""

import os
import sys
import time
import logging
import signal
import psutil
import json
from pathlib import Path
from typing import Dict, Optional
import threading
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/start.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('PiRobot-Start')

class SystemMonitor:
    """Monitor system resources."""
    
    def __init__(self):
        """Initialize system monitor."""
        self._stop_event = threading.Event()
        self._monitor_thread = None
        
    def start(self):
        """Start system monitoring."""
        self._monitor_thread = threading.Thread(target=self._monitor_resources, 
                                              daemon=True)
        self._monitor_thread.start()
        
    def stop(self):
        """Stop system monitoring."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join()
            
    def _monitor_resources(self):
        """Monitor system resources in background thread."""
        while not self._stop_event.is_set():
            try:
                # Check CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                if cpu_percent > 90:
                    logger.warning(f"High CPU usage: {cpu_percent}%")
                    
                # Check memory usage
                memory = psutil.virtual_memory()
                if memory.percent > 90:
                    logger.warning(f"High memory usage: {memory.percent}%")
                    
                # Check temperature
                try:
                    with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                        temp = float(f.read()) / 1000.0
                        if temp > 70:
                            logger.warning(f"High temperature: {temp}°C")
                except:
                    pass
                    
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(1)
                
class ProcessManager:
    """Manage system processes."""
    
    def __init__(self):
        """Initialize process manager."""
        self.processes = {}
        self._stop_event = threading.Event()
        
    def start_process(self, name: str, command: str):
        """Start a process."""
        try:
            process = psutil.Popen(command.split(), 
                                 stdout=psutil.PIPE,
                                 stderr=psutil.PIPE)
            self.processes[name] = process
            logger.info(f"Started {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to start {name}: {e}")
            return False
            
    def stop_process(self, name: str):
        """Stop a process."""
        if name in self.processes:
            try:
                process = self.processes[name]
                process.terminate()
                process.wait(timeout=5)
                logger.info(f"Stopped {name}")
                return True
            except Exception as e:
                logger.error(f"Failed to stop {name}: {e}")
                return False
        return False
        
    def stop_all(self):
        """Stop all processes."""
        for name in list(self.processes.keys()):
            self.stop_process(name)
            
class PiRobotStart:
    """PiRobot V.4 startup manager."""
    
    def __init__(self):
        """Initialize startup manager."""
        self.config = self._load_config()
        self.monitor = SystemMonitor()
        self.process_manager = ProcessManager()
        
    def _load_config(self) -> Dict:
        """Load configuration file."""
        try:
            config_path = Path("config/config.txt")
            if not config_path.exists():
                logger.error("Configuration file not found")
                sys.exit(1)
                
            with open(config_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)
            
    def _check_dependencies(self) -> bool:
        """Check system dependencies."""
        try:
            # Check Python version
            if sys.version_info < (3, 7):
                logger.error("Python 3.7 or higher required")
                return False
                
            # Check required packages
            import numpy
            import RPi.GPIO as GPIO
            import cv2
            import fastapi
            import uvicorn
            
            return True
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            return False
            
    def _check_resources(self) -> bool:
        """Check system resources."""
        try:
            # Check CPU architecture
            if not os.uname().machine.startswith('arm'):
                logger.warning("Not running on ARM architecture")
                
            # Check available memory
            memory = psutil.virtual_memory()
            if memory.available < 100 * 1024 * 1024:  # 100MB
                logger.warning("Low memory available")
                
            # Check disk space
            disk = psutil.disk_usage('/')
            if disk.free < 100 * 1024 * 1024:  # 100MB
                logger.warning("Low disk space")
                
            return True
        except Exception as e:
            logger.error(f"Resource check failed: {e}")
            return False
            
    def start(self):
        """Start PiRobot V.4 system."""
        try:
            logger.info("Starting PiRobot V.4...")
            
            # Check dependencies and resources
            if not self._check_dependencies() or not self._check_resources():
                sys.exit(1)
                
            # Start system monitor
            self.monitor.start()
            
            # Start core processes
            processes = [
                ("motor_controller", "python3 src/core/motor_controller.py"),
                ("encoder_handler", "python3 src/core/encoder_handler.py"),
                ("pid_controller", "python3 src/core/pid_controller.py"),
                ("temperature_monitor", "python3 src/core/temperature_monitor.py"),
                ("safety_monitor", "python3 src/core/safety_monitor.py")
            ]
            
            for name, command in processes:
                if not self.process_manager.start_process(name, command):
                    logger.error(f"Failed to start {name}")
                    self.stop()
                    sys.exit(1)
                    
            # Start web interface
            if self.config["web_interface"]["debug"]:
                self.process_manager.start_process(
                    "web_interface",
                    "python3 src/web/app.py"
                )
                
            logger.info("PiRobot V.4 started successfully")
            
            # Keep running until interrupted
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Received shutdown signal")
                self.stop()
                
        except Exception as e:
            logger.error(f"Startup error: {e}")
            self.stop()
            sys.exit(1)
            
    def stop(self):
        """Stop PiRobot V.4 system."""
        logger.info("Stopping PiRobot V.4...")
        
        # Stop all processes
        self.process_manager.stop_all()
        
        # Stop system monitor
        self.monitor.stop()
        
        logger.info("PiRobot V.4 stopped")
        
def main():
    """Main function."""
    robot = PiRobotStart()
    
    # Handle signals
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        robot.stop()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start robot
    robot.start()
    
if __name__ == "__main__":
    main() 