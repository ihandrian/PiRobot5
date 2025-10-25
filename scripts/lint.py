#!/usr/bin/env python3
"""
PiRobot V.4 Lint Script
This script performs code quality and style checks for PiRobot V.4.
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
        logging.FileHandler('logs/lint.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('PiRobot-Lint')

def run_black():
    """Run Black code formatter."""
    try:
        logger.info("Running Black...")
        result = subprocess.run([
            "black",
            "--check",
            "src"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Black check passed")
            return True
        else:
            logger.error("Black check failed:")
            logger.error(result.stdout)
            return False
    except Exception as e:
        logger.error(f"Black check failed: {e}")
        return False

def run_flake8():
    """Run Flake8 linter."""
    try:
        logger.info("Running Flake8...")
        result = subprocess.run([
            "flake8",
            "src"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Flake8 check passed")
            return True
        else:
            logger.error("Flake8 check failed:")
            logger.error(result.stdout)
            return False
    except Exception as e:
        logger.error(f"Flake8 check failed: {e}")
        return False

def run_mypy():
    """Run MyPy type checker."""
    try:
        logger.info("Running MyPy...")
        result = subprocess.run([
            "mypy",
            "src"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("MyPy check passed")
            return True
        else:
            logger.error("MyPy check failed:")
            logger.error(result.stdout)
            return False
    except Exception as e:
        logger.error(f"MyPy check failed: {e}")
        return False

def run_pylint():
    """Run Pylint."""
    try:
        logger.info("Running Pylint...")
        result = subprocess.run([
            "pylint",
            "src"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Pylint check passed")
            return True
        else:
            logger.error("Pylint check failed:")
            logger.error(result.stdout)
            return False
    except Exception as e:
        logger.error(f"Pylint check failed: {e}")
        return False

def run_isort():
    """Run isort import sorter."""
    try:
        logger.info("Running isort...")
        result = subprocess.run([
            "isort",
            "--check",
            "src"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("isort check passed")
            return True
        else:
            logger.error("isort check failed:")
            logger.error(result.stdout)
            return False
    except Exception as e:
        logger.error(f"isort check failed: {e}")
        return False

def generate_lint_report():
    """Generate lint report."""
    try:
        report_dir = Path("lint_results")
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"lint_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("PiRobot V.4 Lint Report\n")
            f.write("=====================\n\n")
            f.write(f"Date: {datetime.now()}\n\n")
            
            # Add lint results
            for tool in ['black', 'flake8', 'mypy', 'pylint', 'isort']:
                result_file = report_dir / f"{tool}_results.txt"
                if result_file.exists():
                    f.write(f"\n{tool.title()} Results:\n")
                    f.write("-" * (len(tool) + 9) + "\n")
                    with open(result_file) as txt:
                        f.write(txt.read())
        
        logger.info(f"Lint report generated: {report_file}")
        return True
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return False

def cleanup_lint_results():
    """Clean up old lint results."""
    try:
        report_dir = Path("lint_results")
        if report_dir.exists():
            # Keep only the last 5 reports
            reports = sorted(report_dir.glob("lint_report_*.txt"), 
                           key=os.path.getctime)
            if len(reports) > 5:
                for old_report in reports[:-5]:
                    old_report.unlink()
                    logger.info(f"Removed old lint report: {old_report}")
        return True
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return False

def main():
    """Main lint function."""
    try:
        logger.info("Starting PiRobot V.4 lint checks...")

        # Create lint results directory
        Path("lint_results").mkdir(exist_ok=True)

        # Run lint tools
        lint_tools = [
            ("Black", run_black),
            ("Flake8", run_flake8),
            ("MyPy", run_mypy),
            ("Pylint", run_pylint),
            ("isort", run_isort)
        ]

        results = []
        for name, lint_func in lint_tools:
            logger.info(f"\nRunning {name}...")
            try:
                result = lint_func()
                results.append(result)
                logger.info(f"{name}: {'✓ Passed' if result else '✗ Failed'}")
            except Exception as e:
                logger.error(f"{name} error: {e}")
                results.append(False)

        # Generate report
        generate_lint_report()

        # Clean up old results
        cleanup_lint_results()

        # Print summary
        logger.info("\nLint Summary:")
        for (name, _), result in zip(lint_tools, results):
            logger.info(f"{'✓' if result else '✗'} {name}")

        if all(results):
            logger.info("\n✓ All lint checks passed successfully!")
            sys.exit(0)
        else:
            logger.info("\n✗ Some lint checks failed. Please check the lint report for details.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Lint error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 