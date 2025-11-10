#!/usr/bin/env sage -python
"""
Comprehensive Rational Number Testing Suite
Tests RustMath rationals against SageMath for correctness
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
    # For rationals, compare numerator and denominator
    rust_num = int(str(rust_val.numerator()))
    rust_den = int(str(rust_val.denominator()))
    sage_num = int(sage_val.numerator())
    sage_den = int(sage_val.denominator())

    if rust_num == sage_num and rust_den == sage_den:
        print(f"  ✓ {name}")
        results.record_pass()
        return True
    else:
        print(f"  ✗ {name}")
        results.record_fail(name, f"{sage_num}/{sage_den}", f"{rust_num}/{rust_den}")
        return False

print("="*70)
print("RATIONAL NUMBER TESTING SUITE: RustMath vs SageMath")
print("="*70)

# ========== Creation and Simplification ==========
print("\n[1] Creation and Automatic Simplification")
print("-"*70)

creation_cases = [
    (1, 2), (2, 4), (3, 6), (5, 10),
    (6, 8), (12, 18), (100, 150),
    (-1, 2), (1, -2), (-1, -2),
]

for num, den in creation_cases:
    rr = rustmath.PyRational(num, den)
    sr = QQ(num) / QQ(den)

    test_equal(f"Rational({num}, {den})", rr, sr)

# ========== Addition ==========
print("\n[2] Addition")
print("-"*70)

add_cases = [
    ((1, 2), (1, 3)),   # 1/2 + 1/3 = 5/6
    ((2, 3), (1, 4)),   # 2/3 + 1/4 = 11/12
    ((1, 5), (1, 7)),   # 1/5 + 1/7 = 12/35
    ((3, 4), (5, 6)),   # 3/4 + 5/6 = 19/12
]

for (n1, d1), (n2, d2) in add_cases:
    r1 = rustmath.PyRational(n1, d1)
    r2 = rustmath.PyRational(n2, d2)
    s1 = QQ(n1) / QQ(d1)
    s2 = QQ(n2) / QQ(d2)

    rust_sum = r1 + r2
    sage_sum = s1 + s2

    test_equal(f"{n1}/{d1} + {n2}/{d2}", rust_sum, sage_sum)

# ========== Subtraction ==========
print("\n[3] Subtraction")
print("-"*70)

for (n1, d1), (n2, d2) in add_cases:
    r1 = rustmath.PyRational(n1, d1)
    r2 = rustmath.PyRational(n2, d2)
    s1 = QQ(n1) / QQ(d1)
    s2 = QQ(n2) / QQ(d2)

    rust_diff = r1 - r2
    sage_diff = s1 - s2

    test_equal(f"{n1}/{d1} - {n2}/{d2}", rust_diff, sage_diff)

# ========== Multiplication ==========
print("\n[4] Multiplication")
print("-"*70)

mul_cases = [
    ((1, 2), (1, 3)),   # 1/2 * 1/3 = 1/6
    ((2, 3), (3, 4)),   # 2/3 * 3/4 = 1/2
    ((5, 7), (7, 11)),  # 5/7 * 7/11 = 5/11
    ((6, 8), (4, 9)),   # 6/8 * 4/9 = 1/3
]

for (n1, d1), (n2, d2) in mul_cases:
    r1 = rustmath.PyRational(n1, d1)
    r2 = rustmath.PyRational(n2, d2)
    s1 = QQ(n1) / QQ(d1)
    s2 = QQ(n2) / QQ(d2)

    rust_prod = r1 * r2
    sage_prod = s1 * s2

    test_equal(f"{n1}/{d1} * {n2}/{d2}", rust_prod, sage_prod)

# ========== Division ==========
print("\n[5] Division")
print("-"*70)

for (n1, d1), (n2, d2) in mul_cases:
    r1 = rustmath.PyRational(n1, d1)
    r2 = rustmath.PyRational(n2, d2)
    s1 = QQ(n1) / QQ(d1)
    s2 = QQ(n2) / QQ(d2)

    rust_quot = r1 / r2
    sage_quot = s1 / s2

    test_equal(f"{n1}/{d1} / {n2}/{d2}", rust_quot, sage_quot)

# ========== Negation ==========
print("\n[6] Negation")
print("-"*70)

neg_cases = [(1, 2), (3, 4), (5, 7), (-1, 3)]

for num, den in neg_cases:
    rr = rustmath.PyRational(num, den)
    sr = QQ(num) / QQ(den)

    rust_neg = -rr
    sage_neg = -sr

    test_equal(f"-({num}/{den})", rust_neg, sage_neg)

# ========== Absolute Value ==========
print("\n[7] Absolute Value")
print("-"*70)

abs_cases = [(1, 2), (-1, 2), (3, -4), (-5, -7)]

for num, den in abs_cases:
    rr = rustmath.PyRational(num, den)
    sr = QQ(num) / QQ(den)

    rust_abs = abs(rr)
    sage_abs = abs(sr)

    test_equal(f"abs({num}/{den})", rust_abs, sage_abs)

# ========== Reciprocal ==========
print("\n[8] Reciprocal")
print("-"*70)

recip_cases = [(1, 2), (3, 4), (5, 7), (12, 13)]

for num, den in recip_cases:
    rr = rustmath.PyRational(num, den)
    sr = QQ(num) / QQ(den)

    rust_recip = rr.reciprocal()
    sage_recip = 1 / sr

    test_equal(f"reciprocal({num}/{den})", rust_recip, sage_recip)

# ========== Floor ==========
print("\n[9] Floor")
print("-"*70)

floor_cases = [
    (1, 2), (3, 2), (7, 3), (10, 3),
    (-1, 2), (-3, 2), (-7, 3),
]

for num, den in floor_cases:
    rr = rustmath.PyRational(num, den)
    sr = QQ(num) / QQ(den)

    rust_floor = rr.floor()
    sage_floor = floor(sr)

    rust_val = int(str(rust_floor))
    sage_val = int(sage_floor)

    if rust_val == sage_val:
        print(f"  ✓ floor({num}/{den})")
        results.record_pass()
    else:
        print(f"  ✗ floor({num}/{den})")
        results.record_fail(f"floor({num}/{den})", sage_val, rust_val)

# ========== Ceiling ==========
print("\n[10] Ceiling")
print("-"*70)

for num, den in floor_cases:
    rr = rustmath.PyRational(num, den)
    sr = QQ(num) / QQ(den)

    rust_ceil = rr.ceil()
    sage_ceil = ceil(sr)

    rust_val = int(str(rust_ceil))
    sage_val = int(sage_ceil)

    if rust_val == sage_val:
        print(f"  ✓ ceil({num}/{den})")
        results.record_pass()
    else:
        print(f"  ✗ ceil({num}/{den})")
        results.record_fail(f"ceil({num}/{den})", sage_val, rust_val)

# ========== Round ==========
print("\n[11] Rounding")
print("-"*70)

round_cases = [
    (1, 2), (3, 2), (5, 2), (7, 2),
    (1, 3), (2, 3), (4, 3), (5, 3),
]

for num, den in round_cases:
    rr = rustmath.PyRational(num, den)
    sr = QQ(num) / QQ(den)

    rust_round = rr.round()
    sage_round = round(sr)

    rust_val = int(str(rust_round))
    sage_val = int(sage_round)

    if rust_val == sage_val:
        print(f"  ✓ round({num}/{den})")
        results.record_pass()
    else:
        print(f"  ✗ round({num}/{den})")
        results.record_fail(f"round({num}/{den})", sage_val, rust_val)

# ========== Power ==========
print("\n[12] Integer Powers")
print("-"*70)

pow_cases = [
    ((2, 3), 2),   # (2/3)^2 = 4/9
    ((1, 2), 3),   # (1/2)^3 = 1/8
    ((3, 4), 2),   # (3/4)^2 = 9/16
    ((5, 2), 2),   # (5/2)^2 = 25/4
    ((2, 3), -1),  # (2/3)^-1 = 3/2
    ((5, 7), -2),  # (5/7)^-2 = 49/25
]

for (num, den), exp in pow_cases:
    rr = rustmath.PyRational(num, den)
    sr = QQ(num) / QQ(den)

    rust_pow = rr.pow(exp)
    sage_pow = sr ** exp

    test_equal(f"({num}/{den})^{exp}", rust_pow, sage_pow)

# ========== Comparison ==========
print("\n[13] Comparison Operations")
print("-"*70)

comp_cases = [
    ((1, 2), (1, 3)),
    ((2, 3), (3, 4)),
    ((5, 6), (7, 8)),
]

for (n1, d1), (n2, d2) in comp_cases:
    r1 = rustmath.PyRational(n1, d1)
    r2 = rustmath.PyRational(n2, d2)
    s1 = QQ(n1) / QQ(d1)
    s2 = QQ(n2) / QQ(d2)

    # Test less than
    if (r1 < r2) == (s1 < s2):
        print(f"  ✓ {n1}/{d1} < {n2}/{d2}")
        results.record_pass()
    else:
        print(f"  ✗ {n1}/{d1} < {n2}/{d2}")
        results.record_fail(f"{n1}/{d1} < {n2}/{d2}", s1 < s2, r1 < r2)

    # Test greater than
    if (r1 > r2) == (s1 > s2):
        print(f"  ✓ {n1}/{d1} > {n2}/{d2}")
        results.record_pass()
    else:
        print(f"  ✗ {n1}/{d1} > {n2}/{d2}")
        results.record_fail(f"{n1}/{d1} > {n2}/{d2}", s1 > s2, r1 > r2)

# ========== String Parsing ==========
print("\n[14] String Parsing")
print("-"*70)

parse_cases = ["1/2", "3/4", "5/6", "10/15", "100/200"]

for s in parse_cases:
    rr = rustmath.PyRational.from_string(s)
    parts = s.split('/')
    sr = QQ(int(parts[0])) / QQ(int(parts[1]))

    test_equal(f"from_string('{s}')", rr, sr)

# ========== Print Final Results ==========
success = results.print_summary()
sys.exit(0 if success else 1)
