#!/usr/bin/env python3
"""
Test runner script for CapybaraDB
"""
import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all tests with pytest"""
    test_dir = Path(__file__).parent
    project_root = test_dir.parent
    
    cmd = [
        sys.executable, "-m", "pytest",
        str(test_dir),
        "-v",
        "--tb=short",
        "--cov=capybaradb",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
    ]
    
    print("Running CapybaraDB tests...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)