#!/usr/bin/env python3
"""
PiRobot V.4 Restore Script
This script handles system restoration from backups for PiRobot V.4.
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
        logging.FileHandler('logs/restore.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('PiRobot-Restore')

def list_available_backups():
    """List all available backups."""
    backup_dir = Path("backups")
    if not backup_dir.exists():
        logger.error("No backups directory found")
        return []

    backups = []
    for item in backup_dir.iterdir():
        if item.suffix == '.zip':
            backups.append(item)
    return sorted(backups, key=os.path.getctime, reverse=True)

def select_backup(backups):
    """Let user select a backup to restore."""
    if not backups:
        logger.error("No backups available")
        return None

    print("\nAvailable backups:")
    for i, backup in enumerate(backups, 1):
        print(f"{i}. {backup.name} ({datetime.fromtimestamp(backup.stat().st_mtime)})")

    while True:
        try:
            choice = int(input("\nSelect backup to restore (number): "))
            if 1 <= choice <= len(backups):
                return backups[choice - 1]
            print("Invalid selection")
        except ValueError:
            print("Please enter a number")

def extract_backup(backup_path):
    """Extract the backup archive."""
    try:
        temp_dir = Path("temp_restore")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir()

        with zipfile.ZipFile(backup_path, 'r') as zipf:
            zipf.extractall(temp_dir)
        logger.info("Backup extracted successfully")
        return temp_dir
    except Exception as e:
        logger.error(f"Backup extraction failed: {e}")
        return None

def restore_configuration(backup_dir):
    """Restore configuration files."""
    try:
        backup_config_dir = backup_dir / "config"
        if backup_config_dir.exists():
            config_dir = Path("config")
            if config_dir.exists():
                shutil.rmtree(config_dir)
            shutil.copytree(backup_config_dir, config_dir)
            logger.info("Configuration files restored successfully")
            return True
        return False
    except Exception as e:
        logger.error(f"Configuration restore failed: {e}")
        return False

def restore_logs(backup_dir):
    """Restore log files."""
    try:
        backup_log_dir = backup_dir / "logs"
        if backup_log_dir.exists():
            log_dir = Path("logs")
            if log_dir.exists():
                shutil.rmtree(log_dir)
            shutil.copytree(backup_log_dir, log_dir)
            logger.info("Log files restored successfully")
            return True
        return False
    except Exception as e:
        logger.error(f"Log restore failed: {e}")
        return False

def restore_source_code(backup_dir):
    """Restore source code."""
    try:
        backup_src_dir = backup_dir / "src"
        if backup_src_dir.exists():
            src_dir = Path("src")
            if src_dir.exists():
                shutil.rmtree(src_dir)
            shutil.copytree(backup_src_dir, src_dir)
            logger.info("Source code restored successfully")
            return True
        return False
    except Exception as e:
        logger.error(f"Source code restore failed: {e}")
        return False

def cleanup(backup_dir):
    """Clean up temporary files."""
    try:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        logger.info("Cleanup completed")
        return True
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return False

def main():
    """Main restore function."""
    try:
        logger.info("Starting PiRobot V.4 restore...")

        # List and select backup
        backups = list_available_backups()
        selected_backup = select_backup(backups)
        if not selected_backup:
            sys.exit(1)

        # Extract backup
        backup_dir = extract_backup(selected_backup)
        if not backup_dir:
            sys.exit(1)

        # Perform restore
        config_restore = restore_configuration(backup_dir)
        logs_restore = restore_logs(backup_dir)
        source_restore = restore_source_code(backup_dir)

        if not any([config_restore, logs_restore, source_restore]):
            logger.error("No data restored")
            sys.exit(1)

        # Clean up
        cleanup(backup_dir)

        logger.info("PiRobot V.4 restore completed successfully")
        logger.info("Please restart the robot to apply restored configuration")

    except Exception as e:
        logger.error(f"Restore error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 