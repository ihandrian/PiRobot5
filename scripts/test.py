#!/usr/bin/env python3
"""
PiRobot V.4 Test Script
This script runs system tests for PiRobot V.4.
"""

import os
import sys
import logging
import subprocess
import pytest
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('PiRobot-Test')

def run_unit_tests():
    """Run unit tests."""
    try:
        logger.info("Running unit tests...")
        result = pytest.main([
            "src/tests/unit",
            "-v",
            "--junitxml=test_results/unit_tests.xml"
        ])
        return result == 0
    except Exception as e:
        logger.error(f"Unit test execution failed: {e}")
        return False

def run_integration_tests():
    """Run integration tests."""
    try:
        logger.info("Running integration tests...")
        result = pytest.main([
            "src/tests/integration",
            "-v",
            "--junitxml=test_results/integration_tests.xml"
        ])
        return result == 0
    except Exception as e:
        logger.error(f"Integration test execution failed: {e}")
        return False

def run_performance_tests():
    """Run performance tests."""
    try:
        logger.info("Running performance tests...")
        result = pytest.main([
            "src/tests/performance",
            "-v",
            "--junitxml=test_results/performance_tests.xml"
        ])
        return result == 0
    except Exception as e:
        logger.error(f"Performance test execution failed: {e}")
        return False

def generate_test_report():
    """Generate test report."""
    try:
        report_dir = Path("test_results")
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"test_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("PiRobot V.4 Test Report\n")
            f.write("=====================\n\n")
            f.write(f"Date: {datetime.now()}\n\n")
            
            # Add test results
            for test_type in ['unit', 'integration', 'performance']:
                result_file = report_dir / f"{test_type}_tests.xml"
                if result_file.exists():
                    f.write(f"\n{test_type.title()} Tests:\n")
                    f.write("-" * (len(test_type) + 8) + "\n")
                    with open(result_file) as xml:
                        f.write(xml.read())
        
        logger.info(f"Test report generated: {report_file}")
        return True
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return False

def cleanup_test_results():
    """Clean up old test results."""
    try:
        report_dir = Path("test_results")
        if report_dir.exists():
            # Keep only the last 5 reports
            reports = sorted(report_dir.glob("test_report_*.txt"), 
                           key=os.path.getctime)
            if len(reports) > 5:
                for old_report in reports[:-5]:
                    old_report.unlink()
                    logger.info(f"Removed old test report: {old_report}")
        return True
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return False

def main():
    """Main test function."""
    try:
        logger.info("Starting PiRobot V.4 tests...")

        # Create test results directory
        Path("test_results").mkdir(exist_ok=True)

        # Run tests
        test_types = [
            ("Unit Tests", run_unit_tests),
            ("Integration Tests", run_integration_tests),
            ("Performance Tests", run_performance_tests)
        ]

        results = []
        for name, test_func in test_types:
            logger.info(f"\nRunning {name}...")
            try:
                result = test_func()
                results.append(result)
                logger.info(f"{name}: {'✓ Passed' if result else '✗ Failed'}")
            except Exception as e:
                logger.error(f"{name} error: {e}")
                results.append(False)

        # Generate report
        generate_test_report()

        # Clean up old results
        cleanup_test_results()

        # Print summary
        logger.info("\nTest Summary:")
        for (name, _), result in zip(test_types, results):
            logger.info(f"{'✓' if result else '✗'} {name}")

        if all(results):
            logger.info("\n✓ All tests passed successfully!")
            sys.exit(0)
        else:
            logger.info("\n✗ Some tests failed. Please check the test report for details.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Test error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 