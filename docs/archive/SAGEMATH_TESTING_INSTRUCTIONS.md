# SageMath Testing Instructions

This document provides complete instructions for setting up and running the PyO3-based testing framework to validate RustMath against SageMath.

## Overview

The testing system uses PyO3 to create Python bindings for RustMath, allowing you to call Rust code directly from Python/SageMath. This enables comprehensive testing by comparing RustMath outputs with SageMath (the reference implementation).

## Prerequisites

### 1. SageMath Installation

You need SageMath installed on your system. Verify by running:
```bash
sage --version
```

If not installed:
- **Linux**: `sudo apt install sagemath` (Ubuntu/Debian) or download from https://www.sagemath.org/
- **macOS**: `brew install sagemath` or download from https://www.sagemath.org/
- **Windows**: Use WSL2 with Linux installation

### 2. Rust Toolchain

Ensure you have Rust installed:
```bash
rustc --version
cargo --version
```

If not installed, get it from https://rustup.rs/

### 3. Python Development Tools

Install maturin (PyO3 build tool):
```bash
pip install maturin

# Or using SageMath's Python:
sage -pip install maturin
```

## Setup Instructions

### Step 1: Build the PyO3 Bindings

From the RustMath root directory:

```bash
# Navigate to the PyO3 bindings crate
cd rustmath-py

# Build and install the module in development mode
# This installs into your Python/SageMath environment
maturin develop --release

# Alternatively, build a wheel for distribution
maturin build --release
```

**Important**:
- Use `--release` for optimized performance
- The module will be available as `import rustmath` in Python/SageMath
- You may need to use `sage -pip install maturin` if maturin isn't in SageMath's Python

### Step 2: Verify Installation

Test that the module loads correctly:

```bash
sage -python -c "import rustmath; print(rustmath.PyInteger(42))"
```

Expected output: `Integer(42)`

### Step 3: Make Test Scripts Executable

```bash
cd ../sage-tests
chmod +x test_integers.py
chmod +x test_rationals.py
chmod +x test_matrix.py
chmod +x run_all_tests.py
```

## Running Tests

### Option 1: Run Individual Test Suites

Test specific modules:

```bash
cd sage-tests

# Test integers (primality, factorization, GCD, etc.)
sage -python test_integers.py

# Test rationals (fractions, arithmetic)
sage -python test_rationals.py

# Test matrices (determinant, transpose, multiplication)
sage -python test_matrix.py
```

Each test suite will:
- Run dozens of test cases
- Print ✓ for passing tests, ✗ for failures
- Show a summary at the end
- Exit with code 0 (success) or 1 (failure)

### Option 2: Run Complete Test Suite

Run all tests at once:

```bash
cd sage-tests
sage -python run_all_tests.py
```

This will run all test suites sequentially and provide an overall summary.

### Option 3: Run from SageMath Interactive

You can also run tests interactively in SageMath:

```python
sage: import rustmath
sage:
sage: # Test integers
sage: n = rustmath.PyInteger(97)
sage: print(f"Is 97 prime? {n.is_prime()}")
sage: print(f"Next prime after 97: {n.next_prime()}")
sage:
sage: # Compare with SageMath
sage: sage_n = Integer(97)
sage: print(f"SageMath says: {is_prime(sage_n)}")
sage: print(f"SageMath next: {next_prime(sage_n)}")
sage:
sage: # Test rationals
sage: r = rustmath.PyRational(1, 2)
sage: s = rustmath.PyRational(1, 3)
sage: print(f"1/2 + 1/3 = {r + s}")
sage:
sage: # Compare with SageMath
sage: print(f"SageMath: {QQ(1)/QQ(2) + QQ(1)/QQ(3)}")
```

## Understanding Test Output

### Successful Test
```
  ✓ is_prime(97)
  ✓ next_prime(97)
  ✓ gcd(12, 18)
```

### Failed Test
```
  ✗ factor(100)
    Expected: [(2, 2), (5, 2)]
    Got:      [(2, 2), (5, 2)]
```

### Summary
```
======================================================================
SUMMARY: 150/150 tests passed
✓ All tests passed!
======================================================================
```

## Rebuilding After Code Changes

If you modify RustMath code, rebuild the Python module:

```bash
cd rustmath-py
maturin develop --release
```

Then re-run tests:

```bash
cd ../sage-tests
sage -python test_integers.py
```

## Troubleshooting

### Problem: "No module named rustmath"

**Solution**: The module wasn't installed. Run:
```bash
cd rustmath-py
maturin develop --release
```

### Problem: "maturin: command not found"

**Solution**: Install maturin using SageMath's pip:
```bash
sage -pip install maturin
```

### Problem: Compilation errors

**Solution**:
1. Ensure all RustMath dependencies build: `cargo build --workspace`
2. Check that you're in the `rustmath-py` directory
3. Try cleaning: `cargo clean && maturin develop --release`

### Problem: Test failures

**Solution**:
1. Check if it's a known limitation (see Module Coverage below)
2. Verify SageMath version: `sage --version` (tested with 9.x+)
3. Report the failure as a bug if unexpected

### Problem: "ImportError: dynamic module does not define module export function"

**Solution**: Rebuild with correct architecture:
```bash
cargo clean
maturin develop --release
```

## Module Coverage

### Currently Implemented

#### ✅ Integers (rustmath.PyInteger)
- Arithmetic: `+`, `-`, `*`, `/`, `%`, `**`, `-x`, `abs(x)`
- Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=`
- Number theory:
  - `is_prime()` - Miller-Rabin primality test
  - `next_prime()`, `previous_prime()`
  - `factor()` - Prime factorization
  - `divisors()` - All divisors
  - `num_divisors()` - Count of divisors (tau)
  - `sum_divisors()` - Sum of divisors (sigma)
  - `gcd(other)`, `lcm(other)`
  - `extended_gcd(other)` - Returns (gcd, s, t)
  - `mod_inverse(m)` - Modular inverse
  - `mod_pow(exp, m)` - Modular exponentiation
- Properties:
  - `is_even()`, `is_odd()`, `is_zero()`, `is_one()`
  - `signum()` - Sign: -1, 0, or 1
  - `bit_length()` - Number of bits
- Roots:
  - `sqrt()` - Integer square root
  - `nth_root(n)` - Nth root

#### ✅ Rationals (rustmath.PyRational)
- Arithmetic: `+`, `-`, `*`, `/`, `-r`, `abs(r)`
- Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=`
- Operations:
  - `numerator()`, `denominator()` - Access components
  - `reciprocal()` - 1/r
  - `floor()`, `ceil()`, `round()` - Rounding
  - `fract()` - Fractional part
  - `pow(exp)` - Integer power
- Properties:
  - `is_zero()`, `is_one()`, `is_integer()`
  - `to_float()` - Convert to f64
- Construction:
  - `PyRational(num, den)` - From integers
  - `PyRational.from_integer(n)` - From integer
  - `PyRational.from_string("3/4")` - Parse string

#### ✅ Matrices (rustmath.PyMatrix)
- Construction:
  - `PyMatrix.from_list([[1,2],[3,4]])` - From 2D list
  - `PyMatrix.zeros(rows, cols)` - Zero matrix
  - `PyMatrix.identity(n)` - Identity matrix
- Arithmetic: `+`, `-`, `*` (matrix multiply), `-M`
- Properties:
  - `rows()`, `cols()`, `shape()` - Dimensions
  - `is_square()` - Check if square
- Operations:
  - `transpose()` - Matrix transpose
  - `determinant()` - Compute determinant
  - `trace()` - Sum of diagonal
  - `get(i, j)`, `set(i, j, val)` - Element access
  - `to_list()` - Convert to nested list

### Not Yet Implemented

- ⬜ Polynomials (planned)
- ⬜ Finite Fields (planned)
- ⬜ Symbolic expressions (planned)
- ⬜ Advanced matrix operations (LU decomposition, eigenvalues, etc.)

## Writing Custom Tests

You can create your own test scripts following this template:

```python
#!/usr/bin/env sage -python
"""My Custom Test"""

from sage.all import *
import rustmath

# Test something
rust_result = rustmath.PyInteger(12).factor()
sage_result = list(factor(12))

print(f"Rust: {rust_result}")
print(f"Sage: {sage_result}")

# Assert they match
assert str(rust_result) == str(sage_result), "Mismatch!"
print("✓ Test passed")
```

Save as `my_test.py` and run:
```bash
sage -python my_test.py
```

## Performance Testing

Compare performance between Rust and SageMath:

```python
#!/usr/bin/env sage -python
import rustmath
import time
from sage.all import *

# Test primality for large number
n = 2**89 - 1  # Mersenne number

# Rust timing
start = time.time()
rust_result = rustmath.PyInteger(int(n)).is_prime()
rust_time = time.time() - start

# SageMath timing
start = time.time()
sage_result = is_prime(n)
sage_time = time.time() - start

print(f"Rust:  {rust_result} in {rust_time:.4f}s")
print(f"Sage:  {sage_result} in {sage_time:.4f}s")
print(f"Speedup: {sage_time/rust_time:.2f}x")
```

## Next Steps

1. **Run tests**: Start with `test_integers.py` to validate your setup
2. **Explore interactively**: Use SageMath console to compare implementations
3. **Report issues**: If you find discrepancies, check `MODULE_TEST_STATUS.md`
4. **Add tests**: Contribute new test cases for edge cases
5. **Expand coverage**: Help implement bindings for more modules

## Additional Resources

- **PyO3 Documentation**: https://pyo3.rs/
- **SageMath Documentation**: https://doc.sagemath.org/
- **RustMath Documentation**: Run `cargo doc --open` in project root
- **Test Status**: See `MODULE_TEST_STATUS.md` for detailed module testing status

## Getting Help

If you encounter issues:

1. Check the Troubleshooting section above
2. Review `MODULE_TEST_STATUS.md` for known limitations
3. Run `cargo test` to verify RustMath builds correctly
4. Check SageMath version compatibility (9.x or higher recommended)
5. Open an issue on the RustMath repository

---

**Happy Testing!** 🧪

This testing framework ensures RustMath maintains mathematical correctness by validating against the battle-tested SageMath implementation.
