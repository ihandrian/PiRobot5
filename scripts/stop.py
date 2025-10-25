#!/usr/bin/env python3
"""
PiRobot V.4 Stop Script
This script stops the PiRobot V.4 system with optimized performance.
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
        logging.FileHandler('logs/stop.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('PiRobot-Stop')

class ProcessManager:
    """Manage system processes."""
    
    def __init__(self):
        """Initialize process manager."""
        self.processes = {}
        
    def find_processes(self, name: str) -> list:
        """Find processes by name."""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if name in proc.info['name'] or name in ' '.join(proc.info['cmdline']):
                    processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return processes
        
    def stop_process(self, proc: psutil.Process):
        """Stop a process gracefully."""
        try:
            proc.terminate()
            proc.wait(timeout=5)
            logger.info(f"Stopped process {proc.pid}")
            return True
        except psutil.TimeoutExpired:
            try:
                proc.kill()
                logger.warning(f"Killed process {proc.pid}")
                return True
            except Exception as e:
                logger.error(f"Failed to kill process {proc.pid}: {e}")
                return False
        except Exception as e:
            logger.error(f"Failed to stop process {proc.pid}: {e}")
            return False
            
    def stop_all(self, name: str):
        """Stop all processes with given name."""
        processes = self.find_processes(name)
        for proc in processes:
            self.stop_process(proc)
            
class ResourceCleanup:
    """Clean up system resources."""
    
    def __init__(self):
        """Initialize resource cleanup."""
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load configuration file."""
        try:
            config_path = Path("config/config.txt")
            if not config_path.exists():
                logger.error("Configuration file not found")
                return {}
                
            with open(config_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return {}
            
    def cleanup_gpio(self):
        """Clean up GPIO resources."""
        try:
            import RPi.GPIO as GPIO
            GPIO.cleanup()
            logger.info("GPIO cleanup completed")
        except Exception as e:
            logger.error(f"GPIO cleanup failed: {e}")
            
    def cleanup_camera(self):
        """Clean up camera resources."""
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                cap.release()
            logger.info("Camera cleanup completed")
        except Exception as e:
            logger.error(f"Camera cleanup failed: {e}")
            
    def cleanup_serial(self):
        """Clean up serial resources."""
        try:
            import serial
            ports = serial.tools.list_ports.comports()
            for port in ports:
                try:
                    ser = serial.Serial(port.device)
                    ser.close()
                except:
                    pass
            logger.info("Serial cleanup completed")
        except Exception as e:
            logger.error(f"Serial cleanup failed: {e}")
            
    def cleanup_files(self):
        """Clean up temporary files."""
        try:
            # Clean up log files
            log_dir = Path("logs")
            if log_dir.exists():
                for log_file in log_dir.glob("*.log.*"):
                    log_file.unlink()
                    
            # Clean up temp files
            temp_dir = Path("temp")
            if temp_dir.exists():
                for temp_file in temp_dir.glob("*"):
                    temp_file.unlink()
                    
            logger.info("File cleanup completed")
        except Exception as e:
            logger.error(f"File cleanup failed: {e}")
            
class PiRobotStop:
    """PiRobot V.4 shutdown manager."""
    
    def __init__(self):
        """Initialize shutdown manager."""
        self.process_manager = ProcessManager()
        self.resource_cleanup = ResourceCleanup()
        
    def stop(self):
        """Stop PiRobot V.4 system."""
        try:
            logger.info("Stopping PiRobot V.4...")
            
            # Stop core processes
            core_processes = [
                "motor_controller",
                "encoder_handler",
                "pid_controller",
                "temperature_monitor",
                "safety_monitor"
            ]
            
            for process in core_processes:
                self.process_manager.stop_all(process)
                
            # Stop web interface
            self.process_manager.stop_all("web_interface")
            
            # Clean up resources
            self.resource_cleanup.cleanup_gpio()
            self.resource_cleanup.cleanup_camera()
            self.resource_cleanup.cleanup_serial()
            self.resource_cleanup.cleanup_files()
            
            logger.info("PiRobot V.4 stopped successfully")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
            sys.exit(1)
            
def main():
    """Main function."""
    robot = PiRobotStop()
    
    # Handle signals
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        robot.stop()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Stop robot
    robot.stop()
    
if __name__ == "__main__":
    main() 