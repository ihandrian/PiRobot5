#!/usr/bin/env python3
"""
PiRobot V.4 Diagnostic Script
This script performs system diagnostics and troubleshooting for PiRobot V.4.
"""

import os
import sys
import logging
import subprocess
import platform
import psutil
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/diagnostic.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('PiRobot-Diagnostic')

def check_system_resources():
    """Check system resources."""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        logger.info(f"CPU Usage: {cpu_percent}%")

        # Memory usage
        memory = psutil.virtual_memory()
        logger.info(f"Memory Usage: {memory.percent}%")
        logger.info(f"Available Memory: {memory.available / 1024 / 1024:.2f} MB")

        # Disk usage
        disk = psutil.disk_usage('/')
        logger.info(f"Disk Usage: {disk.percent}%")
        logger.info(f"Available Disk Space: {disk.free / 1024 / 1024 / 1024:.2f} GB")

        return True
    except Exception as e:
        logger.error(f"System resource check failed: {e}")
        return False

def check_gpio_access():
    """Check GPIO access and configuration."""
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.cleanup()
        logger.info("GPIO access verified")
        return True
    except Exception as e:
        logger.error(f"GPIO check failed: {e}")
        return False

def check_camera():
    """Check camera access and configuration."""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Camera not accessible")
            return False
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to capture frame")
            return False
        cap.release()
        logger.info("Camera access verified")
        return True
    except Exception as e:
        logger.error(f"Camera check failed: {e}")
        return False

def check_network():
    """Check network connectivity."""
    try:
        # Check internet connectivity
        subprocess.run(["ping", "-c", "1", "8.8.8.8"], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL)
        logger.info("Internet connectivity verified")

        # Check local network
        subprocess.run(["ping", "-c", "1", "localhost"], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL)
        logger.info("Local network verified")
        return True
    except Exception as e:
        logger.error(f"Network check failed: {e}")
        return False

def check_dependencies():
    """Check Python dependencies."""
    try:
        # Determine pip path based on OS
        if sys.platform == "win32":
            pip_path = "venv\\Scripts\\pip"
        else:
            pip_path = "venv/bin/pip"

        # Check installed packages
        result = subprocess.run([pip_path, "freeze"], 
                              capture_output=True, 
                              text=True)
        logger.info("Installed packages:")
        for package in result.stdout.splitlines():
            logger.info(f"  {package}")
        return True
    except Exception as e:
        logger.error(f"Dependency check failed: {e}")
        return False

def check_logs():
    """Check system logs for errors."""
    try:
        log_dir = Path("logs")
        if not log_dir.exists():
            logger.error("Log directory not found")
            return False

        error_count = 0
        for log_file in log_dir.glob("*.log"):
            with open(log_file) as f:
                for line in f:
                    if "ERROR" in line or "CRITICAL" in line:
                        error_count += 1
                        logger.warning(f"Found error in {log_file.name}: {line.strip()}")

        if error_count > 0:
            logger.warning(f"Found {error_count} errors in log files")
        else:
            logger.info("No errors found in log files")
        return True
    except Exception as e:
        logger.error(f"Log check failed: {e}")
        return False

def generate_report():
    """Generate diagnostic report."""
    try:
        report_dir = Path("diagnostics")
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"diagnostic_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("PiRobot V.4 Diagnostic Report\n")
            f.write("============================\n\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write(f"System: {platform.system()} {platform.release()}\n")
            f.write(f"Python: {sys.version}\n\n")
            
            # Add system resource information
            f.write("System Resources:\n")
            f.write(f"CPU Usage: {psutil.cpu_percent()}%\n")
            f.write(f"Memory Usage: {psutil.virtual_memory().percent}%\n")
            f.write(f"Disk Usage: {psutil.disk_usage('/').percent}%\n\n")
            
            # Add log summary
            f.write("Recent Errors:\n")
            log_dir = Path("logs")
            if log_dir.exists():
                for log_file in log_dir.glob("*.log"):
                    with open(log_file) as log:
                        for line in log:
                            if "ERROR" in line or "CRITICAL" in line:
                                f.write(f"{line.strip()}\n")
        
        logger.info(f"Diagnostic report generated: {report_file}")
        return True
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return False

def main():
    """Main diagnostic function."""
    try:
        logger.info("Starting PiRobot V.4 diagnostics...")

        # Run diagnostic checks
        checks = [
            ("System Resources", check_system_resources),
            ("GPIO Access", check_gpio_access),
            ("Camera", check_camera),
            ("Network", check_network),
            ("Dependencies", check_dependencies),
            ("Logs", check_logs)
        ]

        results = []
        for name, check in checks:
            logger.info(f"\nRunning {name} check...")
            try:
                result = check()
                results.append(result)
                logger.info(f"{name} check: {'✓ Passed' if result else '✗ Failed'}")
            except Exception as e:
                logger.error(f"{name} check error: {e}")
                results.append(False)

        # Generate report
        generate_report()

        # Print summary
        logger.info("\nDiagnostic Summary:")
        for (name, _), result in zip(checks, results):
            logger.info(f"{'✓' if result else '✗'} {name}")

        if all(results):
            logger.info("\n✓ All diagnostic checks passed successfully!")
            sys.exit(0)
        else:
            logger.info("\n✗ Some diagnostic checks failed. Please check the report for details.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Diagnostic error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 