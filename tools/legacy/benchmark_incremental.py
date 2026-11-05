#!/usr/bin/env python3
"""
Incremental RustMath vs SageMath Benchmark
Tests arbitrary precision support progressively with timeouts
"""

import time
import sys
import builtins

# Critical: Use Python builtins to avoid SageMath preparser
pyint = builtins.int
pyrange = builtins.range

print("Importing rustmath...")
sys.path.insert(0, '/home/user/RustMath/target/wheels')
import rustmath
PyInt = rustmath.PyInteger

print("✓ Import successful\n")

def test_basic_arbitrary_precision():
    """Test 1: Verify arbitrary precision works at all"""
    print("=" * 60)
    print("TEST 1: Basic Arbitrary Precision (20 digits)")
    print("=" * 60)

    # 20 digits - previously failed with i64
    a_str = "12345678901234567890"
    b_str = "98765432109876543210"

    print(f"Creating PyInteger from {len(a_str)}-digit string...")
    try:
        a = PyInt.from_string(a_str)
        print(f"✓ Created: {a}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    print(f"Creating second PyInteger from {len(b_str)}-digit string...")
    try:
        b = PyInt.from_string(b_str)
        print(f"✓ Created: {b}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    print("Computing GCD...")
    try:
        start = time.time()
        result = a.gcd(b)
        elapsed = time.time() - start
        print(f"✓ GCD result: {result}")
        print(f"  Time: {elapsed:.6f}s")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    print("\n✓ TEST 1 PASSED\n")
    return True

def test_medium_numbers():
    """Test 2: 50-digit numbers"""
    print("=" * 60)
    print("TEST 2: Medium Precision (50 digits)")
    print("=" * 60)

    a_str = "12345678901234567890" * 2 + "1234567890"  # 50 digits
    b_str = "98765432109876543210" * 2 + "9876543210"

    print(f"Testing {len(a_str)}-digit GCD...")

    try:
        a = PyInt.from_string(a_str)
        b = PyInt.from_string(b_str)

        start = time.time()
        rust_result = a.gcd(b)
        rust_time = time.time() - start

        print(f"✓ RustMath GCD: {rust_time:.6f}s")

        # Compare with SageMath
        sage_a = Integer(a_str)
        sage_b = Integer(b_str)

        start = time.time()
        sage_result = gcd(sage_a, sage_b)
        sage_time = time.time() - start

        print(f"✓ SageMath GCD: {sage_time:.6f}s")
        print(f"  Ratio: {rust_time/sage_time:.2f}x")

        # Verify correctness
        if str(rust_result) == str(sage_result):
            print(f"✓ Results match: {rust_result}")
        else:
            print(f"✗ MISMATCH!")
            print(f"  Rust: {rust_result}")
            print(f"  Sage: {sage_result}")
            return False

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✓ TEST 2 PASSED\n")
    return True

def test_100_digits():
    """Test 3: 100-digit numbers"""
    print("=" * 60)
    print("TEST 3: Large Precision (100 digits)")
    print("=" * 60)

    a_str = "12345678901234567890" * 5  # 100 digits
    b_str = "98765432109876543210" * 5

    print(f"Testing {len(a_str)}-digit GCD...")
    print("This may take several seconds...")

    try:
        a = PyInt.from_string(a_str)
        b = PyInt.from_string(b_str)

        start = time.time()
        rust_result = a.gcd(b)
        rust_time = time.time() - start

        print(f"✓ RustMath GCD: {rust_time:.6f}s")

        # Compare with SageMath
        sage_a = Integer(a_str)
        sage_b = Integer(b_str)

        start = time.time()
        sage_result = gcd(sage_a, sage_b)
        sage_time = time.time() - start

        print(f"✓ SageMath GCD: {sage_time:.6f}s")
        print(f"  Ratio: {rust_time/sage_time:.2f}x")

        # Verify correctness
        if str(rust_result) == str(sage_result):
            print(f"✓ Results match (GCD has {len(str(rust_result))} digits)")
        else:
            print(f"✗ MISMATCH!")
            return False

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✓ TEST 3 PASSED\n")
    return True

def test_arithmetic_operations():
    """Test 4: Other operations with large numbers"""
    print("=" * 60)
    print("TEST 4: Arithmetic Operations (100 digits)")
    print("=" * 60)

    a_str = "12345678901234567890" * 5
    b_str = "98765432109876543210" * 5

    try:
        a = PyInt.from_string(a_str)
        b = PyInt.from_string(b_str)

        print("Testing addition...")
        start = time.time()
        result = a + b
        print(f"✓ Addition: {time.time() - start:.6f}s")

        print("Testing subtraction...")
        start = time.time()
        result = a - b
        print(f"✓ Subtraction: {time.time() - start:.6f}s")

        print("Testing multiplication...")
        start = time.time()
        result = a * b
        print(f"✓ Multiplication: {time.time() - start:.6f}s")

        print("Testing division...")
        start = time.time()
        result = a / b
        print(f"✓ Division: {time.time() - start:.6f}s")

        print("Testing modulo...")
        start = time.time()
        result = a % b
        print(f"✓ Modulo: {time.time() - start:.6f}s")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✓ TEST 4 PASSED\n")
    return True

def test_primality_small():
    """Test 5: Primality testing with reasonable-sized numbers"""
    print("=" * 60)
    print("TEST 5: Primality Testing (20 digits)")
    print("=" * 60)

    # Use a known 20-digit prime
    prime_str = "12345678901234567891"  # This might not be prime, but that's ok for testing

    try:
        n = PyInt.from_string(prime_str)

        print(f"Testing is_prime() on {len(prime_str)}-digit number...")
        print("WARNING: This operation has known performance issues")

        start = time.time()
        result = n.is_prime()
        rust_time = time.time() - start

        print(f"✓ RustMath result: {result} ({rust_time:.6f}s)")

        # Compare with SageMath
        sage_n = Integer(prime_str)
        start = time.time()
        sage_result = sage_n.is_prime()
        sage_time = time.time() - start

        print(f"✓ SageMath result: {sage_result} ({sage_time:.6f}s)")
        print(f"  Ratio: {rust_time/sage_time:.2f}x")

        if result == sage_result:
            print(f"✓ Results match")
        else:
            print(f"✗ MISMATCH!")
            return False

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✓ TEST 5 PASSED\n")
    return True

# Run tests incrementally
if __name__ == "__main__":
    print("RustMath Arbitrary Precision Incremental Benchmark")
    print("=" * 60)
    print()

    tests = [
        ("Basic 20-digit", test_basic_arbitrary_precision),
        ("Medium 50-digit", test_medium_numbers),
        ("Large 100-digit", test_100_digits),
        ("Arithmetic ops", test_arithmetic_operations),
        ("Primality test", test_primality_small),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\nRunning: {name}")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"\n⚠️  {name} failed - stopping here")
                break
        except KeyboardInterrupt:
            print("\n\n⚠️  Test interrupted by user")
            break
        except Exception as e:
            print(f"\n✗ Unexpected error in {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            break

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if passed == len(tests):
        print("\n✓ ALL TESTS PASSED - Arbitrary precision is working!")
    else:
        print(f"\n⚠️  Stopped at test {passed + 1}")
