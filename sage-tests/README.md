# SageMath Test Suites

This directory contains comprehensive test suites that validate RustMath implementations against SageMath.

## Quick Start

### Prerequisites
- SageMath installed (`sage --version` to verify)
- RustMath PyO3 bindings built (`cd ../rustmath-py && maturin develop --release`)

### Run All Tests
```bash
sage -python run_all_tests.py
```

### Run Individual Tests
```bash
sage -python test_integers.py
sage -python test_rationals.py
sage -python test_matrix.py
```

## Test Files

| File | Module | Tests | Description |
|------|--------|-------|-------------|
| `test_integers.py` | rustmath-integers | ~150 | Primality, factorization, GCD, divisors, modular arithmetic |
| `test_rationals.py` | rustmath-rationals | ~90 | Rational arithmetic, floor/ceil, powers |
| `test_matrix.py` | rustmath-matrix | ~50 | Matrix operations, determinant, transpose |
| `run_all_tests.py` | All | All | Master test runner |

## What Gets Tested

### Integers (test_integers.py)
- ✅ Basic arithmetic (+, -, *, /, %, **)
- ✅ Primality testing (is_prime, next_prime, previous_prime)
- ✅ Factorization (factor)
- ✅ GCD and LCM
- ✅ Extended GCD
- ✅ Modular arithmetic (mod_inverse, mod_pow)
- ✅ Divisor functions (divisors, num_divisors, sum_divisors)
- ✅ Square and nth roots

### Rationals (test_rationals.py)
- ✅ Creation and simplification
- ✅ Arithmetic (+, -, *, /)
- ✅ Comparison (<, >, ==)
- ✅ Floor, ceiling, rounding
- ✅ Reciprocal
- ✅ Integer powers
- ✅ String parsing

### Matrices (test_matrix.py)
- ✅ Construction (from_list, zeros, identity)
- ✅ Matrix arithmetic (+, -, *)
- ✅ Transpose
- ✅ Determinant (2x2, 3x3, larger)
- ✅ Trace
- ✅ Element access (get/set)

## Test Output Format

### Passing Test
```
  ✓ is_prime(97)
```

### Failing Test
```
  ✗ factor(100)
    Expected: [(2, 2), (5, 2)]
    Got:      [(2, 2), (5, 1)]
```

### Summary
```
======================================================================
SUMMARY: 150/150 tests passed
✓ All tests passed!
======================================================================
```

## Exit Codes
- `0` - All tests passed
- `1` - One or more tests failed

## Troubleshooting

### "No module named rustmath"
Build the PyO3 bindings:
```bash
cd ../rustmath-py
maturin develop --release
```

### "sage: command not found"
Install SageMath or use full path:
```bash
/path/to/sage -python test_integers.py
```

### Tests fail but Rust tests pass
This could indicate:
- Different behavior between Rust and SageMath (check if intentional)
- Bug in PyO3 bindings (check `../rustmath-py/src/`)
- Version incompatibility with SageMath

## Adding New Tests

1. Create a new test file: `test_<module>.py`
2. Import necessary modules:
   ```python
   #!/usr/bin/env sage -python
   from sage.all import *
   import rustmath
   ```
3. Compare Rust vs SageMath outputs
4. Add to `run_all_tests.py`

## Documentation

For complete documentation, see:
- Setup instructions: `../SAGEMATH_TESTING_INSTRUCTIONS.md`
- Test status: `../MODULE_TEST_STATUS.md`
- Implementation plan: `../Rust_Testing_Plan.md`
