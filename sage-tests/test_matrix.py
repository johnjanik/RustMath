#!/usr/bin/env sage -python
"""
Comprehensive Matrix Testing Suite
Tests RustMath matrices against SageMath for correctness
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

def matrices_equal(rust_mat, sage_mat):
    """Check if RustMath and SageMath matrices are equal"""
    if rust_mat.rows() != sage_mat.nrows() or rust_mat.cols() != sage_mat.ncols():
        return False

    for i in range(rust_mat.rows()):
        for j in range(rust_mat.cols()):
            rust_val = int(str(rust_mat.get(i, j)))
            sage_val = int(sage_mat[i, j])
            if rust_val != sage_val:
                return False
    return True

def test_matrix_equal(name, rust_mat, sage_mat):
    """Test if matrices are equal"""
    if matrices_equal(rust_mat, sage_mat):
        print(f"  ✓ {name}")
        results.record_pass()
        return True
    else:
        print(f"  ✗ {name}")
        results.record_fail(name, sage_mat, rust_mat.to_list())
        return False

def test_scalar_equal(name, rust_val, sage_val):
    """Test scalar equality"""
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
print("MATRIX TESTING SUITE: RustMath vs SageMath")
print("="*70)

# ========== Matrix Creation ==========
print("\n[1] Matrix Creation")
print("-"*70)

# Test from_list
data_2x2 = [[1, 2], [3, 4]]
rm = rustmath.PyMatrix.from_list(data_2x2)
sm = Matrix(ZZ, data_2x2)
test_matrix_equal("from_list([[1,2],[3,4]])", rm, sm)

data_3x3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
rm = rustmath.PyMatrix.from_list(data_3x3)
sm = Matrix(ZZ, data_3x3)
test_matrix_equal("from_list(3x3)", rm, sm)

# Test zeros
rm = rustmath.PyMatrix.zeros(3, 3)
sm = Matrix(ZZ, 3, 3)
test_matrix_equal("zeros(3, 3)", rm, sm)

# Test identity
rm = rustmath.PyMatrix.identity(3)
sm = identity_matrix(ZZ, 3)
test_matrix_equal("identity(3)", rm, sm)

# ========== Matrix Properties ==========
print("\n[2] Matrix Properties")
print("-"*70)

data = [[1, 2, 3], [4, 5, 6]]
rm = rustmath.PyMatrix.from_list(data)
sm = Matrix(ZZ, data)

if rm.rows() == sm.nrows():
    print(f"  ✓ rows()")
    results.record_pass()
else:
    print(f"  ✗ rows()")
    results.record_fail("rows()", sm.nrows(), rm.rows())

if rm.cols() == sm.ncols():
    print(f"  ✓ cols()")
    results.record_pass()
else:
    print(f"  ✗ cols()")
    results.record_fail("cols()", sm.ncols(), rm.cols())

# Test square matrices
square_data = [[1, 2], [3, 4]]
rm = rustmath.PyMatrix.from_list(square_data)
sm = Matrix(ZZ, square_data)

if rm.is_square() == sm.is_square():
    print(f"  ✓ is_square() for square matrix")
    results.record_pass()
else:
    print(f"  ✗ is_square() for square matrix")
    results.record_fail("is_square()", sm.is_square(), rm.is_square())

nonsquare_data = [[1, 2, 3], [4, 5, 6]]
rm = rustmath.PyMatrix.from_list(nonsquare_data)
sm = Matrix(ZZ, nonsquare_data)

if rm.is_square() == sm.is_square():
    print(f"  ✓ is_square() for non-square matrix")
    results.record_pass()
else:
    print(f"  ✗ is_square() for non-square matrix")
    results.record_fail("is_square()", sm.is_square(), rm.is_square())

# ========== Matrix Transpose ==========
print("\n[3] Matrix Transpose")
print("-"*70)

test_cases = [
    [[1, 2], [3, 4]],
    [[1, 2, 3], [4, 5, 6]],
    [[1], [2], [3]],
    [[1, 2, 3]],
]

for data in test_cases:
    rm = rustmath.PyMatrix.from_list(data)
    sm = Matrix(ZZ, data)

    rust_t = rm.transpose()
    sage_t = sm.transpose()

    test_matrix_equal(f"transpose({data})", rust_t, sage_t)

# ========== Matrix Addition ==========
print("\n[4] Matrix Addition")
print("-"*70)

add_cases = [
    ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
    ([[1, 0], [0, 1]], [[0, 1], [1, 0]]),
    ([[1, 2, 3], [4, 5, 6]], [[6, 5, 4], [3, 2, 1]]),
]

for data1, data2 in add_cases:
    rm1 = rustmath.PyMatrix.from_list(data1)
    rm2 = rustmath.PyMatrix.from_list(data2)
    sm1 = Matrix(ZZ, data1)
    sm2 = Matrix(ZZ, data2)

    rust_sum = rm1 + rm2
    sage_sum = sm1 + sm2

    test_matrix_equal(f"addition", rust_sum, sage_sum)

# ========== Matrix Subtraction ==========
print("\n[5] Matrix Subtraction")
print("-"*70)

for data1, data2 in add_cases:
    rm1 = rustmath.PyMatrix.from_list(data1)
    rm2 = rustmath.PyMatrix.from_list(data2)
    sm1 = Matrix(ZZ, data1)
    sm2 = Matrix(ZZ, data2)

    rust_diff = rm1 - rm2
    sage_diff = sm1 - sm2

    test_matrix_equal(f"subtraction", rust_diff, sage_diff)

# ========== Matrix Multiplication ==========
print("\n[6] Matrix Multiplication")
print("-"*70)

mul_cases = [
    ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
    ([[1, 0], [0, 1]], [[1, 2], [3, 4]]),  # Identity multiplication
    ([[1, 2, 3]], [[1], [2], [3]]),  # 1x3 * 3x1
    ([[1, 2], [3, 4], [5, 6]], [[1, 2, 3], [4, 5, 6]]),  # 3x2 * 2x3
]

for data1, data2 in mul_cases:
    rm1 = rustmath.PyMatrix.from_list(data1)
    rm2 = rustmath.PyMatrix.from_list(data2)
    sm1 = Matrix(ZZ, data1)
    sm2 = Matrix(ZZ, data2)

    rust_prod = rm1 * rm2
    sage_prod = sm1 * sm2

    test_matrix_equal(f"multiplication", rust_prod, sage_prod)

# ========== Matrix Negation ==========
print("\n[7] Matrix Negation")
print("-"*70)

neg_cases = [
    [[1, 2], [3, 4]],
    [[1, -2], [-3, 4]],
    [[0, 0], [0, 0]],
]

for data in neg_cases:
    rm = rustmath.PyMatrix.from_list(data)
    sm = Matrix(ZZ, data)

    rust_neg = -rm
    sage_neg = -sm

    test_matrix_equal(f"negation", rust_neg, sage_neg)

# ========== Determinant ==========
print("\n[8] Determinant")
print("-"*70)

det_cases = [
    [[1, 2], [3, 4]],                    # det = -2
    [[2, 1], [1, 2]],                    # det = 3
    [[1, 0], [0, 1]],                    # det = 1 (identity)
    [[1, 2, 3], [0, 1, 4], [5, 6, 0]],   # 3x3
    [[2, 0, 0], [0, 3, 0], [0, 0, 5]],   # diagonal
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],   # singular (det = 0)
]

for data in det_cases:
    rm = rustmath.PyMatrix.from_list(data)
    sm = Matrix(ZZ, data)

    rust_det = rm.determinant()
    sage_det = sm.determinant()

    test_scalar_equal(f"det({data[0]}...)", rust_det, sage_det)

# ========== Trace ==========
print("\n[9] Trace")
print("-"*70)

trace_cases = [
    [[1, 2], [3, 4]],                    # trace = 5
    [[1, 0], [0, 1]],                    # trace = 2
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],   # trace = 15
    [[0, 0], [0, 0]],                    # trace = 0
]

for data in trace_cases:
    rm = rustmath.PyMatrix.from_list(data)
    sm = Matrix(ZZ, data)

    rust_trace = rm.trace()
    sage_trace = sm.trace()

    test_scalar_equal(f"trace({data[0]}...)", rust_trace, sage_trace)

# ========== Element Access ==========
print("\n[10] Element Access (get/set)")
print("-"*70)

data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
rm = rustmath.PyMatrix.from_list(data)
sm = Matrix(ZZ, data)

# Test get
for i in range(3):
    for j in range(3):
        rust_val = int(str(rm.get(i, j)))
        sage_val = int(sm[i, j])
        if rust_val == sage_val:
            results.record_pass()
        else:
            print(f"  ✗ get({i}, {j})")
            results.record_fail(f"get({i},{j})", sage_val, rust_val)

print(f"  ✓ All get() operations passed")

# Test set
rm.set(0, 0, 100)
sm[0, 0] = 100

rust_val = int(str(rm.get(0, 0)))
sage_val = int(sm[0, 0])

if rust_val == sage_val:
    print(f"  ✓ set(0, 0, 100)")
    results.record_pass()
else:
    print(f"  ✗ set(0, 0, 100)")
    results.record_fail("set(0,0,100)", sage_val, rust_val)

# ========== Print Final Results ==========
success = results.print_summary()
sys.exit(0 if success else 1)
