#!/usr/bin/env sage -python
"""
Master Test Runner
Runs all RustMath vs SageMath validation tests
"""

import subprocess
import sys
import os

# Change to sage-tests directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

test_files = [
    "test_integers.py",
    "test_rationals.py",
    "test_matrix.py",
]

print("="*70)
print("RUSTMATH VS SAGEMATH: COMPLETE VALIDATION SUITE")
print("="*70)
print()

results = {}
for test_file in test_files:
    print(f"\nRunning {test_file}...")
    print("-"*70)

    try:
        result = subprocess.run(
            ["sage", "-python", test_file],
            capture_output=False,
            text=True
        )
        results[test_file] = (result.returncode == 0)
    except Exception as e:
        print(f"Error running {test_file}: {e}")
        results[test_file] = False

# Print summary
print("\n" + "="*70)
print("OVERALL SUMMARY")
print("="*70)

all_passed = True
for test_file, passed in results.items():
    status = "✓ PASSED" if passed else "✗ FAILED"
    print(f"{status:12} {test_file}")
    if not passed:
        all_passed = False

print("="*70)
if all_passed:
    print("✓ ALL TESTS PASSED!")
    sys.exit(0)
else:
    print("✗ SOME TESTS FAILED")
    sys.exit(1)
