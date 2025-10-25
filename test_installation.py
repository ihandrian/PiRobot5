import sys
import platform
import torch
import cv2
import numpy as np
import RPi.GPIO as GPIO

def test_installation():
    print("Testing PiRobot5 installation...")
    
    # Test Python version
    print(f"Python version: {sys.version}")
    
    # Test system info
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Machine: {platform.machine()}")
    
    # Test OpenCV
    print("\nTesting OpenCV...")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"OpenCV build info: {cv2.getBuildInformation()}")
    
    # Test PyTorch
    print("\nTesting PyTorch...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Test NumPy
    print("\nTesting NumPy...")
    print(f"NumPy version: {np.__version__}")
    
    # Test GPIO
    print("\nTesting GPIO...")
    try:
        GPIO.setmode(GPIO.BCM)
        print("GPIO test successful")
    except Exception as e:
        print(f"GPIO test failed: {e}")
    
    print("\nInstallation test completed!")

if __name__ == "__main__":
    test_installation()
