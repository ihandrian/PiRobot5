#!/usr/bin/env python3
"""
PiRobot V.4 Backup Script
This script handles system backups for PiRobot V.4.
"""

import os
import sys
import shutil
import logging
import zipfile
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backup.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('PiRobot-Backup')

def create_backup_directory():
    """Create backup directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path("backups") / f"backup_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir

def backup_configuration(backup_dir):
    """Backup configuration files."""
    try:
        config_dir = Path("config")
        if config_dir.exists():
            backup_config_dir = backup_dir / "config"
            backup_config_dir.mkdir(exist_ok=True)
            shutil.copytree(config_dir, backup_config_dir, dirs_exist_ok=True)
            logger.info("Configuration files backed up successfully")
            return True
        return False
    except Exception as e:
        logger.error(f"Configuration backup failed: {e}")
        return False

def backup_logs(backup_dir):
    """Backup log files."""
    try:
        log_dir = Path("logs")
        if log_dir.exists():
            backup_log_dir = backup_dir / "logs"
            backup_log_dir.mkdir(exist_ok=True)
            shutil.copytree(log_dir, backup_log_dir, dirs_exist_ok=True)
            logger.info("Log files backed up successfully")
            return True
        return False
    except Exception as e:
        logger.error(f"Log backup failed: {e}")
        return False

def backup_source_code(backup_dir):
    """Backup source code."""
    try:
        src_dir = Path("src")
        if src_dir.exists():
            backup_src_dir = backup_dir / "src"
            backup_src_dir.mkdir(exist_ok=True)
            shutil.copytree(src_dir, backup_src_dir, dirs_exist_ok=True)
            logger.info("Source code backed up successfully")
            return True
        return False
    except Exception as e:
        logger.error(f"Source code backup failed: {e}")
        return False

def create_zip_archive(backup_dir):
    """Create a zip archive of the backup."""
    try:
        zip_path = backup_dir.parent / f"{backup_dir.name}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(backup_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, backup_dir.parent)
                    zipf.write(file_path, arcname)
        logger.info(f"Backup archive created: {zip_path}")
        return True
    except Exception as e:
        logger.error(f"Archive creation failed: {e}")
        return False

def cleanup_old_backups():
    """Clean up old backups, keeping only the last 5."""
    try:
        backup_dir = Path("backups")
        if backup_dir.exists():
            backups = sorted(backup_dir.glob("backup_*"), key=os.path.getctime)
            if len(backups) > 5:
                for old_backup in backups[:-5]:
                    shutil.rmtree(old_backup)
                    logger.info(f"Removed old backup: {old_backup}")
        return True
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return False

def main():
    """Main backup function."""
    try:
        logger.info("Starting PiRobot V.4 backup...")

        # Create backup directory
        backup_dir = create_backup_directory()
        logger.info(f"Backup directory created: {backup_dir}")

        # Perform backups
        config_backup = backup_configuration(backup_dir)
        logs_backup = backup_logs(backup_dir)
        source_backup = backup_source_code(backup_dir)

        if not any([config_backup, logs_backup, source_backup]):
            logger.error("No data backed up")
            sys.exit(1)

        # Create zip archive
        if create_zip_archive(backup_dir):
            # Clean up the uncompressed backup directory
            shutil.rmtree(backup_dir)
            logger.info("Uncompressed backup directory removed")

        # Clean up old backups
        cleanup_old_backups()

        logger.info("PiRobot V.4 backup completed successfully")

    except Exception as e:
        logger.error(f"Backup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 