#!/usr/bin/env python3
"""
PiRobot V.4 Format Script
This script formats code according to style guidelines for PiRobot V.4.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/format.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('PiRobot-Format')

def run_black():
    """Run Black code formatter."""
    try:
        logger.info("Running Black...")
        result = subprocess.run([
            "black",
            "src"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Black formatting completed")
            return True
        else:
            logger.error("Black formatting failed:")
            logger.error(result.stdout)
            return False
    except Exception as e:
        logger.error(f"Black formatting failed: {e}")
        return False

def run_isort():
    """Run isort import sorter."""
    try:
        logger.info("Running isort...")
        result = subprocess.run([
            "isort",
            "src"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("isort formatting completed")
            return True
        else:
            logger.error("isort formatting failed:")
            logger.error(result.stdout)
            return False
    except Exception as e:
        logger.error(f"isort formatting failed: {e}")
        return False

def run_autopep8():
    """Run autopep8 formatter."""
    try:
        logger.info("Running autopep8...")
        result = subprocess.run([
            "autopep8",
            "--in-place",
            "--recursive",
            "src"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("autopep8 formatting completed")
            return True
        else:
            logger.error("autopep8 formatting failed:")
            logger.error(result.stdout)
            return False
    except Exception as e:
        logger.error(f"autopep8 formatting failed: {e}")
        return False

def run_yapf():
    """Run YAPF formatter."""
    try:
        logger.info("Running YAPF...")
        result = subprocess.run([
            "yapf",
            "--in-place",
            "--recursive",
            "src"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("YAPF formatting completed")
            return True
        else:
            logger.error("YAPF formatting failed:")
            logger.error(result.stdout)
            return False
    except Exception as e:
        logger.error(f"YAPF formatting failed: {e}")
        return False

def generate_format_report():
    """Generate format report."""
    try:
        report_dir = Path("format_results")
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"format_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("PiRobot V.4 Format Report\n")
            f.write("=====================\n\n")
            f.write(f"Date: {datetime.now()}\n\n")
            
            # Add format results
            for tool in ['black', 'isort', 'autopep8', 'yapf']:
                result_file = report_dir / f"{tool}_results.txt"
                if result_file.exists():
                    f.write(f"\n{tool.title()} Results:\n")
                    f.write("-" * (len(tool) + 9) + "\n")
                    with open(result_file) as txt:
                        f.write(txt.read())
        
        logger.info(f"Format report generated: {report_file}")
        return True
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return False

def cleanup_format_results():
    """Clean up old format results."""
    try:
        report_dir = Path("format_results")
        if report_dir.exists():
            # Keep only the last 5 reports
            reports = sorted(report_dir.glob("format_report_*.txt"), 
                           key=os.path.getctime)
            if len(reports) > 5:
                for old_report in reports[:-5]:
                    old_report.unlink()
                    logger.info(f"Removed old format report: {old_report}")
        return True
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return False

def main():
    """Main format function."""
    try:
        logger.info("Starting PiRobot V.4 code formatting...")

        # Create format results directory
        Path("format_results").mkdir(exist_ok=True)

        # Run format tools
        format_tools = [
            ("Black", run_black),
            ("isort", run_isort),
            ("autopep8", run_autopep8),
            ("YAPF", run_yapf)
        ]

        results = []
        for name, format_func in format_tools:
            logger.info(f"\nRunning {name}...")
            try:
                result = format_func()
                results.append(result)
                logger.info(f"{name}: {'✓ Completed' if result else '✗ Failed'}")
            except Exception as e:
                logger.error(f"{name} error: {e}")
                results.append(False)

        # Generate report
        generate_format_report()

        # Clean up old results
        cleanup_format_results()

        # Print summary
        logger.info("\nFormat Summary:")
        for (name, _), result in zip(format_tools, results):
            logger.info(f"{'✓' if result else '✗'} {name}")

        if all(results):
            logger.info("\n✓ All formatting completed successfully!")
            sys.exit(0)
        else:
            logger.info("\n✗ Some formatting failed. Please check the format report for details.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Format error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 