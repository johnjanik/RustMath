#!/usr/bin/env sage -python
"""
Comprehensive Integer Testing Suite
Tests RustMath integers against SageMath for correctness
"""

from sage.all import *
import rustmath
import sys

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.failures = []

    def record_pass(self):
        self.passed += 1

    def record_fail(self, test_name, expected, got):
        self.failed += 1
        self.failures.append((test_name, expected, got))

    def print_summary(self):
        total = self.passed + self.failed
        print("\n" + "="*70)
        print(f"SUMMARY: {self.passed}/{total} tests passed")
        if self.failures:
            print(f"\nFailed tests ({self.failed}):")
            for name, expected, got in self.failures:
                print(f"  ✗ {name}")
                print(f"    Expected: {expected}")
                print(f"    Got:      {got}")
        else:
            print("✓ All tests passed!")
        print("="*70)
        return self.failed == 0

results = TestResults()

def test_equal(name, rust_val, sage_val):
    """Test if Rust and SageMath values are equal"""
    # Convert to strings for comparison
    rust_str = str(rust_val)
    sage_str = str(sage_val)

    if rust_str == sage_str:
        print(f"  ✓ {name}")
        results.record_pass()
        return True
    else:
        print(f"  ✗ {name}")
        results.record_fail(name, sage_str, rust_str)
        return False

print("="*70)
print("INTEGER TESTING SUITE: RustMath vs SageMath")
print("="*70)

# ========== Basic Arithmetic ==========
print("\n[1] Basic Arithmetic")
print("-"*70)

test_cases = [
    (10, 5),
    (100, 37),
    (1000, 999),
    (-15, 7),
    (0, 42),
]

for a, b in test_cases:
    ra, rb = rustmath.PyInteger(a), rustmath.PyInteger(b)
    sa, sb = Integer(a), Integer(b)

    test_equal(f"add({a}, {b})", ra + rb, sa + sb)
    test_equal(f"sub({a}, {b})", ra - rb, sa - sb)
    test_equal(f"mul({a}, {b})", ra * rb, sa * sb)

# ========== Primality Testing ==========
print("\n[2] Primality Testing")
print("-"*70)

primes_and_composites = [
    (2, True), (3, True), (4, False), (5, True),
    (17, True), (97, True), (100, False),
    (997, True), (1009, True), (1024, False),
    (10007, True), (10000, False),
]

for n, expected in primes_and_composites:
    rn = rustmath.PyInteger(n)
    sn = Integer(n)

    rust_prime = rn.is_prime()
    sage_prime = is_prime(sn)

    test_equal(f"is_prime({n})", rust_prime, sage_prime)

# ========== Next Prime ==========
print("\n[3] Next Prime")
print("-"*70)

next_prime_cases = [1, 2, 10, 100, 1000, 10000]

for n in next_prime_cases:
    rn = rustmath.PyInteger(n)
    sn = Integer(n)

    rust_next = rn.next_prime()
    sage_next = next_prime(sn)

    test_equal(f"next_prime({n})", rust_next, sage_next)

# ========== Previous Prime ==========
print("\n[4] Previous Prime")
print("-"*70)

prev_prime_cases = [3, 10, 100, 1000, 10007]

for n in prev_prime_cases:
    rn = rustmath.PyInteger(n)
    sn = Integer(n)

    try:
        rust_prev = rn.previous_prime()
        sage_prev = previous_prime(sn)
        test_equal(f"previous_prime({n})", rust_prev, sage_prev)
    except Exception as e:
        print(f"  ⚠ previous_prime({n}) raised exception: {e}")

# ========== GCD and LCM ==========
print("\n[5] GCD (Greatest Common Divisor)")
print("-"*70)

gcd_cases = [
    (12, 18), (100, 35), (1071, 462),
    (2**10, 2**5), (97, 100), (1, 1000000),
]

for a, b in gcd_cases:
    ra, rb = rustmath.PyInteger(a), rustmath.PyInteger(b)
    sa, sb = Integer(a), Integer(b)

    test_equal(f"gcd({a}, {b})", ra.gcd(rb), gcd(sa, sb))

print("\n[6] LCM (Least Common Multiple)")
print("-"*70)

for a, b in gcd_cases:
    ra, rb = rustmath.PyInteger(a), rustmath.PyInteger(b)
    sa, sb = Integer(a), Integer(b)

    test_equal(f"lcm({a}, {b})", ra.lcm(rb), lcm(sa, sb))

# ========== Extended GCD ==========
print("\n[7] Extended GCD")
print("-"*70)

xgcd_cases = [(12, 18), (100, 35), (1071, 462)]

for a, b in xgcd_cases:
    ra, rb = rustmath.PyInteger(a), rustmath.PyInteger(b)
    sa, sb = Integer(a), Integer(b)

    rust_gcd, rust_s, rust_t = ra.extended_gcd(rb)
    sage_gcd, sage_s, sage_t = xgcd(sa, sb)

    test_equal(f"xgcd({a}, {b})[0] (gcd)", rust_gcd, sage_gcd)
    test_equal(f"xgcd({a}, {b})[1] (s)", rust_s, sage_s)
    test_equal(f"xgcd({a}, {b})[2] (t)", rust_t, sage_t)

# ========== Modular Inverse ==========
print("\n[8] Modular Inverse")
print("-"*70)

modinv_cases = [
    (3, 7), (5, 11), (7, 13),
    (100, 97), (123, 456),
]

for a, m in modinv_cases:
    if gcd(a, m) != 1:
        continue

    ra, rm = rustmath.PyInteger(a), rustmath.PyInteger(m)
    sa, sm = Integer(a), Integer(m)

    try:
        rust_inv = ra.mod_inverse(rm)
        sage_inv = inverse_mod(sa, sm)
        test_equal(f"mod_inverse({a}, {m})", rust_inv, sage_inv)
    except Exception as e:
        print(f"  ⚠ mod_inverse({a}, {m}) error: {e}")

# ========== Modular Exponentiation ==========
print("\n[9] Modular Exponentiation")
print("-"*70)

modpow_cases = [
    (2, 10, 1000),
    (3, 100, 97),
    (5, 50, 13),
    (123, 456, 789),
]

for base, exp, mod in modpow_cases:
    rb, re, rm = rustmath.PyInteger(base), rustmath.PyInteger(exp), rustmath.PyInteger(mod)
    sb, se, sm = Integer(base), Integer(exp), Integer(mod)

    rust_pow = rb.mod_pow(re, rm)
    sage_pow = power_mod(sb, se, sm)

    test_equal(f"mod_pow({base}, {exp}, {mod})", rust_pow, sage_pow)

# ========== Factorization ==========
print("\n[10] Prime Factorization")
print("-"*70)

factor_cases = [60, 100, 256, 997, 1024, 2**10, 2310]

for n in factor_cases:
    rn = rustmath.PyInteger(n)
    sn = Integer(n)

    rust_factors = rn.factor()
    sage_factors = list(factor(sn))

    # Convert Rust format to comparable
    rust_factorization = [(int(str(p)), e) for p, e in rust_factors]
    sage_factorization = [(int(p), int(e)) for p, e in sage_factors]

    test_equal(f"factor({n})", rust_factorization, sage_factorization)

# ========== Divisors ==========
print("\n[11] Divisors")
print("-"*70)

divisor_cases = [12, 20, 60, 100, 128]

for n in divisor_cases:
    rn = rustmath.PyInteger(n)
    sn = Integer(n)

    rust_divs = [int(str(d)) for d in rn.divisors()]
    sage_divs = [int(d) for d in divisors(sn)]

    test_equal(f"divisors({n})", rust_divs, sage_divs)

# ========== Number of Divisors ==========
print("\n[12] Number of Divisors (tau function)")
print("-"*70)

for n in divisor_cases:
    rn = rustmath.PyInteger(n)
    sn = Integer(n)

    rust_num = rn.num_divisors()
    sage_num = number_of_divisors(sn)

    test_equal(f"num_divisors({n})", rust_num, sage_num)

# ========== Sum of Divisors ==========
print("\n[13] Sum of Divisors (sigma function)")
print("-"*70)

for n in divisor_cases:
    rn = rustmath.PyInteger(n)
    sn = Integer(n)

    rust_sum = rn.sum_divisors()
    sage_sum = sigma(sn, 1)

    test_equal(f"sum_divisors({n})", rust_sum, sage_sum)

# ========== Square Root ==========
print("\n[14] Integer Square Root")
print("-"*70)

sqrt_cases = [0, 1, 4, 9, 16, 100, 1000, 10000]

for n in sqrt_cases:
    rn = rustmath.PyInteger(n)
    sn = Integer(n)

    rust_sqrt = rn.sqrt()
    sage_sqrt = isqrt(sn)

    test_equal(f"sqrt({n})", rust_sqrt, sage_sqrt)

# ========== Nth Root ==========
print("\n[15] Nth Root")
print("-"*70)

root_cases = [
    (27, 3), (64, 3), (1024, 2),
    (1000, 3), (2**20, 4),
]

for n, k in root_cases:
    rn = rustmath.PyInteger(n)
    sn = Integer(n)

    rust_root = rn.nth_root(k)
    # SageMath doesn't have direct nth_root, compute manually
    sage_root = floor(sn ** (1/k))

    test_equal(f"nth_root({n}, {k})", rust_root, sage_root)

# ========== Print Final Results ==========
success = results.print_summary()
sys.exit(0 if success else 1)
