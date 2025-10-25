#!/usr/bin/env python3
"""
PiRobot V.4 - Raspberry Pi 5 Installation Test
Tests all components for Pi 5 compatibility and performance
"""

import sys
import platform
import time
import threading
import subprocess
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_status(message, status="INFO"):
    """Print status message with color coding"""
    colors = {
        "INFO": "\033[94m",    # Blue
        "SUCCESS": "\033[92m", # Green
        "WARNING": "\033[93m", # Yellow
        "ERROR": "\033[91m",   # Red
        "RESET": "\033[0m"     # Reset
    }
    print(f"{colors.get(status, '')}[{status}] {message}{colors['RESET']}")

def test_system_info():
    """Test system information and Pi 5 detection"""
    print_header("SYSTEM INFORMATION")
    
    # Check if running on Pi 5
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip()
        print(f"Device Model: {model}")
        
        if "Pi 5" in model or "Raspberry Pi 5" in model:
            print_status("✓ Raspberry Pi 5 detected", "SUCCESS")
        else:
            print_status("⚠ Not running on Pi 5 - some optimizations may not apply", "WARNING")
    except Exception as e:
        print_status(f"✗ Could not detect Pi model: {e}", "ERROR")
    
    # System info
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    
    # Check CPU info
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if "ARM" in cpuinfo:
                print("CPU: ARM-based processor")
            if "Cortex-A76" in cpuinfo:
                print_status("✓ Pi 5 Cortex-A76 CPU detected", "SUCCESS")
    except Exception as e:
        print_status(f"✗ Could not read CPU info: {e}", "ERROR")

def test_python_packages():
    """Test Python package compatibility"""
    print_header("PYTHON PACKAGE COMPATIBILITY")
    
    packages = [
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("torch", "PyTorch"),
        ("flask", "Flask"),
        ("RPi.GPIO", "RPi.GPIO"),
        ("spidev", "SPI Dev"),
        ("psutil", "PSUtil"),
        ("yaml", "PyYAML"),
        ("serial", "PySerial"),
        ("websockets", "WebSockets")
    ]
    
    for package, name in packages:
        try:
            __import__(package)
            print_status(f"✓ {name} imported successfully", "SUCCESS")
        except ImportError as e:
            print_status(f"✗ {name} import failed: {e}", "ERROR")

def test_hardware_interfaces():
    """Test hardware interfaces for Pi 5 compatibility"""
    print_header("HARDWARE INTERFACE TESTING")
    
    # Test GPIO
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        print_status("✓ GPIO interface working", "SUCCESS")
        GPIO.cleanup()
    except Exception as e:
        print_status(f"✗ GPIO test failed: {e}", "ERROR")
    
    # Test SPI
    try:
        import spidev
        spi = spidev.SpiDev()
        print_status("✓ SPI interface available", "SUCCESS")
    except Exception as e:
        print_status(f"✗ SPI test failed: {e}", "ERROR")
    
    # Test I2C
    try:
        import smbus
        bus = smbus.SMBus(1)
        print_status("✓ I2C interface available", "SUCCESS")
    except Exception as e:
        print_status(f"✗ I2C test failed: {e}", "ERROR")

def test_camera_system():
    """Test camera system for Pi 5"""
    print_header("CAMERA SYSTEM TESTING")
    
    try:
        import cv2
        
        # Test camera detection
        cameras = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append(i)
                cap.release()
        
        print(f"Detected cameras: {cameras}")
        
        if cameras:
            print_status(f"✓ {len(cameras)} camera(s) detected", "SUCCESS")
            
            # Test primary camera
            cap = cv2.VideoCapture(cameras[0])
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print_status(f"✓ Camera {cameras[0]} working - Frame size: {frame.shape}", "SUCCESS")
                else:
                    print_status(f"✗ Camera {cameras[0]} failed to capture frame", "ERROR")
                cap.release()
        else:
            print_status("⚠ No cameras detected", "WARNING")
            
    except Exception as e:
        print_status(f"✗ Camera test failed: {e}", "ERROR")

def test_performance():
    """Test performance metrics for Pi 5"""
    print_header("PERFORMANCE TESTING")
    
    try:
        import psutil
        import numpy as np
        import time
        
        # CPU test
        print("Testing CPU performance...")
        start_time = time.time()
        result = sum(i*i for i in range(1000000))
        cpu_time = time.time() - start_time
        print(f"CPU test completed in {cpu_time:.3f} seconds")
        
        # Memory test
        print("Testing memory performance...")
        start_time = time.time()
        arr = np.random.random((1000, 1000))
        result = np.dot(arr, arr)
        memory_time = time.time() - start_time
        print(f"Memory test completed in {memory_time:.3f} seconds")
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        print(f"CPU Usage: {cpu_percent}%")
        print(f"Memory Usage: {memory.percent}% ({memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB)")
        
        # Temperature check
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read()) / 1000.0
            print(f"CPU Temperature: {temp:.1f}°C")
            
            if temp > 80:
                print_status("⚠ High temperature detected", "WARNING")
            else:
                print_status("✓ Temperature within normal range", "SUCCESS")
        except Exception as e:
            print_status(f"✗ Could not read temperature: {e}", "ERROR")
            
    except Exception as e:
        print_status(f"✗ Performance test failed: {e}", "ERROR")

def test_network_connectivity():
    """Test network connectivity for Pi 5"""
    print_header("NETWORK CONNECTIVITY TESTING")
    
    try:
        import socket
        import subprocess
        
        # Test local network
        result = subprocess.run(['ping', '-c', '1', '8.8.8.8'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print_status("✓ Internet connectivity working", "SUCCESS")
        else:
            print_status("✗ Internet connectivity failed", "ERROR")
        
        # Test local network interface
        result = subprocess.run(['ip', 'addr', 'show'], capture_output=True, text=True)
        if 'inet' in result.stdout:
            print_status("✓ Network interfaces configured", "SUCCESS")
        else:
            print_status("✗ No network interfaces found", "ERROR")
            
    except Exception as e:
        print_status(f"✗ Network test failed: {e}", "ERROR")

def test_robot_components():
    """Test robot-specific components"""
    print_header("ROBOT COMPONENT TESTING")
    
    # Test motor controller
    try:
        from src.core.motor_controller import MotorController
        print_status("✓ Motor controller module imported", "SUCCESS")
    except Exception as e:
        print_status(f"✗ Motor controller import failed: {e}", "ERROR")
    
    # Test safety monitor
    try:
        from src.core.safety_monitor import SafetyMonitor
        print_status("✓ Safety monitor module imported", "SUCCESS")
    except Exception as e:
        print_status(f"✗ Safety monitor import failed: {e}", "ERROR")
    
    # Test main application
    try:
        import main
        print_status("✓ Main application module imported", "SUCCESS")
    except Exception as e:
        print_status(f"✗ Main application import failed: {e}", "ERROR")

def test_pi5_specific_features():
    """Test Pi 5 specific features"""
    print_header("PI 5 SPECIFIC FEATURES")
    
    # Check for Pi 5 specific optimizations
    try:
        with open('/boot/config.txt', 'r') as f:
            config = f.read()
            
        pi5_features = [
            'arm_freq=2400',
            'gpu_freq=800',
            'sdram_freq=5000',
            'dtparam=pciex1',
            'camera_auto_detect=1'
        ]
        
        found_features = []
        for feature in pi5_features:
            if feature in config:
                found_features.append(feature)
        
        if found_features:
            print_status(f"✓ Pi 5 optimizations found: {len(found_features)}", "SUCCESS")
            for feature in found_features:
                print(f"  - {feature}")
        else:
            print_status("⚠ No Pi 5 optimizations found in config.txt", "WARNING")
            
    except Exception as e:
        print_status(f"✗ Could not check Pi 5 features: {e}", "ERROR")

def main():
    """Main test function"""
    print_header("PIRobot V.4 - Raspberry Pi 5 Installation Test")
    print("Testing all components for Pi 5 compatibility and performance...")
    
    # Run all tests
    test_system_info()
    test_python_packages()
    test_hardware_interfaces()
    test_camera_system()
    test_performance()
    test_network_connectivity()
    test_robot_components()
    test_pi5_specific_features()
    
    print_header("TEST COMPLETED")
    print_status("Installation test completed. Check results above for any issues.", "INFO")
    print("\nIf all tests passed, your PiRobot V.4 is ready for Pi 5!")
    print("If any tests failed, please check the error messages and fix the issues.")

if __name__ == "__main__":
    main()
