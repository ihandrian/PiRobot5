#!/usr/bin/env python3
"""
PiRobot V.4 Installation Script
This script handles the installation and setup of PiRobot V.4.
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version meets requirements."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)

def check_os():
    """Check if operating system is supported."""
    if platform.system() != "Linux":
        print("Warning: PiRobot V.4 is designed for Linux systems")
        print("Some features may not work correctly on other operating systems")

def create_virtual_environment():
    """Create and activate virtual environment."""
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("Virtual environment created successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        sys.exit(1)

def install_dependencies():
    """Install project dependencies."""
    try:
        # Determine pip path based on OS
        if platform.system() == "Windows":
            pip_path = "venv\\Scripts\\pip"
        else:
            pip_path = "venv/bin/pip"

        # Install production dependencies
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        print("Production dependencies installed successfully")

        # Install development dependencies if --dev flag is present
        if "--dev" in sys.argv:
            subprocess.run([pip_path, "install", "-r", "requirements-dev.txt"], check=True)
            print("Development dependencies installed successfully")

    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

def setup_configuration():
    """Set up configuration files."""
    config_dir = Path("config")
    if not config_dir.exists():
        config_dir.mkdir()

    # Copy example configuration if it doesn't exist
    config_file = config_dir / "config.txt"
    if not config_file.exists():
        shutil.copy("config/config.example.txt", config_file)
        print("Configuration file created from example")

def setup_logging():
    """Set up logging directory."""
    log_dir = Path("logs")
    if not log_dir.exists():
        log_dir.mkdir()
        print("Logging directory created")

def main():
    """Main installation function."""
    print("Starting PiRobot V.4 installation...")

    # Run installation checks
    check_python_version()
    check_os()

    # Create project structure
    create_virtual_environment()
    install_dependencies()
    setup_configuration()
    setup_logging()

    print("\nInstallation completed successfully!")
    print("\nNext steps:")
    print("1. Activate the virtual environment:")
    if platform.system() == "Windows":
        print("   .\\venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("2. Edit config/config.txt with your settings")
    print("3. Run the robot using: python src/main.py")

if __name__ == "__main__":
    main() 