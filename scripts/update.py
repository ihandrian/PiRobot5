#!/usr/bin/env python3
"""
PiRobot V.4 Update Script
This script handles system updates for PiRobot V.4.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/update.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('PiRobot-Update')

def backup_configuration():
    """Backup current configuration."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path("config/backups")
        backup_dir.mkdir(exist_ok=True)

        config_file = Path("config/config.txt")
        if config_file.exists():
            backup_file = backup_dir / f"config_{timestamp}.txt"
            import shutil
            shutil.copy2(config_file, backup_file)
            logger.info(f"Configuration backed up to {backup_file}")
            return True
        return False
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        return False

def update_dependencies():
    """Update Python dependencies."""
    try:
        # Determine pip path based on OS
        if sys.platform == "win32":
            pip_path = "venv\\Scripts\\pip"
        else:
            pip_path = "venv/bin/pip"

        # Update pip itself
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        logger.info("Pip updated successfully")

        # Update production dependencies
        subprocess.run([pip_path, "install", "--upgrade", "-r", "requirements.txt"], check=True)
        logger.info("Production dependencies updated successfully")

        # Update development dependencies if --dev flag is present
        if "--dev" in sys.argv:
            subprocess.run([pip_path, "install", "--upgrade", "-r", "requirements-dev.txt"], check=True)
            logger.info("Development dependencies updated successfully")

        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Dependency update failed: {e}")
        return False

def update_system():
    """Update system packages."""
    try:
        if sys.platform != "win32":
            # Update system packages on Linux
            subprocess.run(["sudo", "apt-get", "update"], check=True)
            subprocess.run(["sudo", "apt-get", "upgrade", "-y"], check=True)
            logger.info("System packages updated successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"System update failed: {e}")
        return False

def verify_installation():
    """Verify the installation after update."""
    try:
        # Run installation tests
        subprocess.run([sys.executable, "scripts/test_installation.py"], check=True)
        logger.info("Installation verification passed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Installation verification failed: {e}")
        return False

def main():
    """Main update function."""
    try:
        logger.info("Starting PiRobot V.4 update...")

        # Backup configuration
        if not backup_configuration():
            logger.error("Update aborted: Configuration backup failed")
            sys.exit(1)

        # Update dependencies
        if not update_dependencies():
            logger.error("Update aborted: Dependency update failed")
            sys.exit(1)

        # Update system packages
        if not update_system():
            logger.warning("System package update failed, but continuing...")

        # Verify installation
        if not verify_installation():
            logger.error("Update aborted: Installation verification failed")
            sys.exit(1)

        logger.info("PiRobot V.4 update completed successfully")
        logger.info("Please restart the robot to apply all updates")

    except Exception as e:
        logger.error(f"Update error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 