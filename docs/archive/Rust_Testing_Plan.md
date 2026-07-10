# Rust Testing Plan: Validation Against SageMath

This document describes methods for testing RustMath modules against a functional SageMath installation to verify correctness.

## Overview

Since SageMath is the reference implementation, we can use it as an oracle to validate RustMath's output. This plan presents three approaches in order of increasing sophistication:

1. **CLI Testing** (Easiest): Rust command-line tools + Python test scripts
2. **PyO3 Bindings** (Recommended): Direct Python/SageMath integration
3. **Automated Test Suite**: Property-based testing with random inputs

---

## Method 1: CLI Testing (Quick Start)

### Concept
Create simple Rust executables that accept inputs and output results. Compare their output with SageMath's results via Python scripts.

### Setup

#### Step 1: Create Test Executables

Create a new crate for testing utilities:

```bash
mkdir -p test-tools/src/bin
```

**File: `test-tools/Cargo.toml`**
```toml
[package]
name = "test-tools"
version = "0.1.0"
edition = "2021"

[dependencies]
rustmath-integers = { path = "../rustmath-integers" }
rustmath-rationals = { path = "../rustmath-rationals" }
rustmath-matrix = { path = "../rustmath-matrix" }
rustmath-symbolic = { path = "../rustmath-symbolic" }
rustmath-polynomials = { path = "../rustmath-polynomials" }
serde_json = "1.0"

[[bin]]
name = "test-integers"
path = "src/bin/test_integers.rs"

[[bin]]
name = "test-rationals"
path = "src/bin/test_rationals.rs"

[[bin]]
name = "test-matrix"
path = "src/bin/test_matrix.rs"
```

#### Step 2: Example Test Executable

**File: `test-tools/src/bin/test_integers.rs`**
```rust
use rustmath_integers::Integer;
use std::env;
use std::io::{self, Write};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: test-integers <operation> [args...]");
        eprintln!("Operations: factor, is_prime, gcd, lcm, next_prime, divisors");
        std::process::exit(1);
    }

    let operation = &args[1];

    match operation.as_str() {
        "factor" => {
            let n = Integer::from(args[2].parse::<i64>().unwrap());
            let factors = n.factor();
            println!("{:?}", factors);
        },
        "is_prime" => {
            let n = Integer::from(args[2].parse::<i64>().unwrap());
            println!("{}", n.is_prime());
        },
        "gcd" => {
            let a = Integer::from(args[2].parse::<i64>().unwrap());
            let b = Integer::from(args[3].parse::<i64>().unwrap());
            println!("{}", a.gcd(&b));
        },
        "lcm" => {
            let a = Integer::from(args[2].parse::<i64>().unwrap());
            let b = Integer::from(args[3].parse::<i64>().unwrap());
            println!("{}", a.lcm(&b));
        },
        "next_prime" => {
            let n = Integer::from(args[2].parse::<i64>().unwrap());
            println!("{}", n.next_prime());
        },
        "divisors" => {
            let n = Integer::from(args[2].parse::<i64>().unwrap());
            let divs = n.divisors();
            println!("{:?}", divs);
        },
        "mod_inverse" => {
            let a = Integer::from(args[2].parse::<i64>().unwrap());
            let m = Integer::from(args[3].parse::<i64>().unwrap());
            match a.mod_inverse(&m) {
                Some(inv) => println!("{}", inv),
                None => println!("None"),
            }
        },
        _ => {
            eprintln!("Unknown operation: {}", operation);
            std::process::exit(1);
        }
    }
}
```

**File: `test-tools/src/bin/test_rationals.rs`**
```rust
use rustmath_rationals::Rational;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: test-rationals <operation> [args...]");
        std::process::exit(1);
    }

    let operation = &args[1];

    match operation.as_str() {
        "add" => {
            let a = parse_rational(&args[2]);
            let b = parse_rational(&args[3]);
            println!("{}", a + b);
        },
        "mul" => {
            let a = parse_rational(&args[2]);
            let b = parse_rational(&args[3]);
            println!("{}", a * b);
        },
        "simplify" => {
            let r = parse_rational(&args[2]);
            println!("{}", r);
        },
        _ => {
            eprintln!("Unknown operation: {}", operation);
            std::process::exit(1);
        }
    }
}

fn parse_rational(s: &str) -> Rational {
    if s.contains('/') {
        let parts: Vec<&str> = s.split('/').collect();
        Rational::new(
            parts[0].parse().unwrap(),
            parts[1].parse().unwrap()
        )
    } else {
        Rational::from(s.parse::<i64>().unwrap())
    }
}
```

#### Step 3: Create Python Test Scripts

**File: `sage-tests/test_integers.py`**
```python
#!/usr/bin/env sage -python
from sage.all import *
import subprocess
import sys

def run_rust(operation, *args):
    """Run Rust test executable and return output"""
    cmd = ['./target/release/test-integers', operation] + list(map(str, args))
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()

def test_factor():
    """Test prime factorization"""
    test_cases = [60, 100, 997, 1024, 2**31 - 1]

    for n in test_cases:
        sage_result = factor(n)
        rust_result = run_rust('factor', n)
        print(f"Testing factor({n}):")
        print(f"  SageMath: {sage_result}")
        print(f"  Rust:     {rust_result}")
        print()

def test_is_prime():
    """Test primality testing"""
    test_cases = [2, 3, 4, 97, 100, 997, 1009, 1024]

    for n in test_cases:
        sage_result = is_prime(n)
        rust_result = run_rust('is_prime', n)
        match = str(sage_result) == rust_result
        status = "✓" if match else "✗"
        print(f"{status} is_prime({n}): Sage={sage_result}, Rust={rust_result}")

def test_gcd():
    """Test GCD"""
    test_cases = [(12, 18), (100, 35), (1071, 462), (2**32, 2**16)]

    for a, b in test_cases:
        sage_result = gcd(a, b)
        rust_result = int(run_rust('gcd', a, b))
        match = sage_result == rust_result
        status = "✓" if match else "✗"
        print(f"{status} gcd({a}, {b}): Sage={sage_result}, Rust={rust_result}")

def test_next_prime():
    """Test next_prime"""
    test_cases = [1, 2, 10, 100, 1000, 10007]

    for n in test_cases:
        sage_result = next_prime(n)
        rust_result = int(run_rust('next_prime', n))
        match = sage_result == rust_result
        status = "✓" if match else "✗"
        print(f"{status} next_prime({n}): Sage={sage_result}, Rust={rust_result}")

if __name__ == '__main__':
    print("=" * 60)
    print("Testing RustMath Integers against SageMath")
    print("=" * 60)
    print()

    test_is_prime()
    print()
    test_gcd()
    print()
    test_next_prime()
    print()
    test_factor()
```

**File: `sage-tests/test_matrix.py`**
```python
#!/usr/bin/env sage -python
from sage.all import *
import json
import subprocess

def rust_matrix_det(matrix_data):
    """Call Rust to compute determinant"""
    cmd = ['./target/release/test-matrix', 'det']
    input_data = json.dumps(matrix_data)
    result = subprocess.run(cmd, input=input_data, capture_output=True, text=True)
    return result.stdout.strip()

def test_determinant():
    """Test matrix determinant"""
    test_matrices = [
        [[1, 2], [3, 4]],
        [[2, 1, 0], [0, 1, 2], [1, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Identity
    ]

    for mat_data in test_matrices:
        sage_mat = Matrix(mat_data)
        sage_det = sage_mat.determinant()
        rust_det = rust_matrix_det(mat_data)

        print(f"Matrix: {mat_data}")
        print(f"  SageMath det: {sage_det}")
        print(f"  Rust det:     {rust_det}")
        print()

if __name__ == '__main__':
    test_determinant()
```

#### Step 4: Build and Run Tests

```bash
# Build Rust test tools (release for speed)
cargo build --release -p test-tools

# Make Python scripts executable
chmod +x sage-tests/*.py

# Run tests
cd sage-tests
./test_integers.py
./test_matrix.py
```

### Advantages
- ✅ Simple to implement
- ✅ No special dependencies
- ✅ Easy to debug
- ✅ Works on any system with SageMath

### Disadvantages
- ❌ Process overhead (slower)
- ❌ Limited data type support (must serialize)
- ❌ Manual parsing of results

---

## Method 2: PyO3 Bindings (Recommended)

### Concept
Create Python bindings for Rust code using PyO3, allowing direct calls from SageMath/Python.

### Setup

#### Step 1: Create Python Bindings Crate

**File: `rustmath-py/Cargo.toml`**
```toml
[package]
name = "rustmath-py"
version = "0.1.0"
edition = "2021"

[lib]
name = "rustmath"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
rustmath-integers = { path = "../rustmath-integers" }
rustmath-rationals = { path = "../rustmath-rationals" }
rustmath-matrix = { path = "../rustmath-matrix" }
rustmath-symbolic = { path = "../rustmath-symbolic" }
```

#### Step 2: Implement Python Bindings

**File: `rustmath-py/src/lib.rs`**
```rust
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;

/// Python wrapper for RustMath Integer
#[pyclass]
struct PyInteger {
    inner: Integer,
}

#[pymethods]
impl PyInteger {
    #[new]
    fn new(value: i64) -> Self {
        PyInteger {
            inner: Integer::from(value),
        }
    }

    fn is_prime(&self) -> bool {
        self.inner.is_prime()
    }

    fn factor(&self) -> Vec<(String, u32)> {
        self.inner.factor()
            .into_iter()
            .map(|(p, e)| (p.to_string(), e))
            .collect()
    }

    fn gcd(&self, other: &PyInteger) -> PyInteger {
        PyInteger {
            inner: self.inner.gcd(&other.inner),
        }
    }

    fn lcm(&self, other: &PyInteger) -> PyInteger {
        PyInteger {
            inner: self.inner.lcm(&other.inner),
        }
    }

    fn next_prime(&self) -> PyInteger {
        PyInteger {
            inner: self.inner.next_prime(),
        }
    }

    fn divisors(&self) -> Vec<String> {
        self.inner.divisors()
            .into_iter()
            .map(|d| d.to_string())
            .collect()
    }

    fn mod_inverse(&self, modulus: &PyInteger) -> PyResult<PyInteger> {
        match self.inner.mod_inverse(&modulus.inner) {
            Some(inv) => Ok(PyInteger { inner: inv }),
            None => Err(PyValueError::new_err("No modular inverse exists")),
        }
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("Integer({})", self.inner)
    }
}

/// Python wrapper for RustMath Rational
#[pyclass]
struct PyRational {
    inner: Rational,
}

#[pymethods]
impl PyRational {
    #[new]
    fn new(numerator: i64, denominator: i64) -> PyResult<Self> {
        if denominator == 0 {
            return Err(PyValueError::new_err("Denominator cannot be zero"));
        }
        Ok(PyRational {
            inner: Rational::new(Integer::from(numerator), Integer::from(denominator)),
        })
    }

    fn __add__(&self, other: &PyRational) -> PyRational {
        PyRational {
            inner: self.inner.clone() + other.inner.clone(),
        }
    }

    fn __mul__(&self, other: &PyRational) -> PyRational {
        PyRational {
            inner: self.inner.clone() * other.inner.clone(),
        }
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// Module initialization
#[pymodule]
fn rustmath(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyInteger>()?;
    m.add_class::<PyRational>()?;
    Ok(())
}
```

#### Step 3: Build Python Module

```bash
# Install maturin (PyO3 build tool)
pip install maturin

# Build and install the Python module
cd rustmath-py
maturin develop --release

# Or build wheel for distribution
maturin build --release
```

#### Step 4: Create Integrated Tests

**File: `sage-tests/test_with_pyo3.py`**
```python
#!/usr/bin/env sage -python
from sage.all import *
import rustmath  # Our Rust module

def test_integers_comprehensive():
    """Comprehensive integer testing with random inputs"""
    test_cases = [2, 3, 17, 97, 100, 256, 1009, 10007]

    print("Testing Integer Operations")
    print("-" * 60)

    for n in test_cases:
        rust_int = rustmath.PyInteger(n)
        sage_int = Integer(n)

        # Test is_prime
        rust_prime = rust_int.is_prime()
        sage_prime = sage_int.is_prime()
        assert rust_prime == sage_prime, f"is_prime mismatch for {n}"

        # Test next_prime
        rust_next = int(str(rust_int.next_prime()))
        sage_next = next_prime(sage_int)
        assert rust_next == sage_next, f"next_prime mismatch for {n}"

        # Test factor (for composite numbers)
        if not sage_prime:
            rust_factors = rust_int.factor()
            sage_factors = list(factor(sage_int))
            print(f"  factor({n}): Rust={rust_factors}, Sage={sage_factors}")

    print("✓ All integer tests passed!")

def test_gcd_random():
    """Test GCD with random inputs"""
    import random

    print("\nTesting GCD with Random Inputs")
    print("-" * 60)

    for _ in range(20):
        a = random.randint(1, 10000)
        b = random.randint(1, 10000)

        rust_a = rustmath.PyInteger(a)
        rust_b = rustmath.PyInteger(b)
        rust_gcd = int(str(rust_a.gcd(rust_b)))

        sage_gcd = gcd(a, b)

        if rust_gcd != sage_gcd:
            print(f"✗ MISMATCH: gcd({a}, {b}): Rust={rust_gcd}, Sage={sage_gcd}")
        else:
            print(f"✓ gcd({a}, {b}) = {rust_gcd}")

def test_rationals():
    """Test rational number operations"""
    print("\nTesting Rational Operations")
    print("-" * 60)

    test_pairs = [(1, 2), (3, 4), (5, 6), (7, 11)]

    for (n1, d1), (n2, d2) in zip(test_pairs, test_pairs[1:]):
        rust_r1 = rustmath.PyRational(n1, d1)
        rust_r2 = rustmath.PyRational(n2, d2)

        sage_r1 = QQ(n1) / QQ(d1)
        sage_r2 = QQ(n2) / QQ(d2)

        # Test addition
        rust_sum = str(rust_r1 + rust_r2)
        sage_sum = str(sage_r1 + sage_r2)
        print(f"  {n1}/{d1} + {n2}/{d2}: Rust={rust_sum}, Sage={sage_sum}")

        # Test multiplication
        rust_prod = str(rust_r1 * rust_r2)
        sage_prod = str(sage_r1 * sage_r2)
        print(f"  {n1}/{d1} * {n2}/{d2}: Rust={rust_prod}, Sage={sage_prod}")

if __name__ == '__main__':
    print("=" * 60)
    print("RustMath vs SageMath Validation Suite (PyO3)")
    print("=" * 60)
    print()

    test_integers_comprehensive()
    test_gcd_random()
    test_rationals()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
```

### Advantages
- ✅ Direct integration (no serialization overhead)
- ✅ Native Python/SageMath interoperability
- ✅ Easy to write comprehensive tests
- ✅ Can pass complex data structures
- ✅ Fast execution

### Disadvantages
- ❌ Requires PyO3 setup
- ❌ More complex initial configuration
- ❌ Need to maintain bindings

---

## Method 3: Automated Property-Based Testing

### Concept
Generate random test cases and verify properties hold across both implementations.

### Setup

**File: `sage-tests/property_testing.py`**
```python
#!/usr/bin/env sage -python
from sage.all import *
import rustmath
import random
import sys

class PropertyTester:
    def __init__(self, num_tests=100):
        self.num_tests = num_tests
        self.passed = 0
        self.failed = 0
        self.failures = []

    def test_property(self, name, test_func):
        """Run a property test multiple times"""
        print(f"\nTesting: {name}")
        print("-" * 60)

        for i in range(self.num_tests):
            try:
                test_func()
                self.passed += 1
            except AssertionError as e:
                self.failed += 1
                self.failures.append((name, i, str(e)))
                print(f"  ✗ Test {i} failed: {e}")

        print(f"  Passed: {self.passed}/{self.num_tests}")

    def report(self):
        """Print final report"""
        print("\n" + "=" * 60)
        print(f"Total: {self.passed + self.failed} tests")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")

        if self.failures:
            print("\nFailures:")
            for name, test_num, error in self.failures:
                print(f"  {name} (test {test_num}): {error}")
        print("=" * 60)

def random_int(max_val=10000):
    """Generate random integer"""
    return random.randint(1, max_val)

def test_gcd_commutative():
    """Property: gcd(a, b) = gcd(b, a)"""
    a, b = random_int(), random_int()

    rust_a, rust_b = rustmath.PyInteger(a), rustmath.PyInteger(b)

    gcd1 = int(str(rust_a.gcd(rust_b)))
    gcd2 = int(str(rust_b.gcd(rust_a)))
    sage_gcd = gcd(a, b)

    assert gcd1 == gcd2, f"GCD not commutative: gcd({a},{b})={gcd1} but gcd({b},{a})={gcd2}"
    assert gcd1 == sage_gcd, f"GCD mismatch with Sage: Rust={gcd1}, Sage={sage_gcd}"

def test_gcd_lcm_identity():
    """Property: gcd(a, b) * lcm(a, b) = a * b"""
    a, b = random_int(), random_int()

    rust_a, rust_b = rustmath.PyInteger(a), rustmath.PyInteger(b)

    rust_gcd = int(str(rust_a.gcd(rust_b)))
    rust_lcm = int(str(rust_a.lcm(rust_b)))

    assert rust_gcd * rust_lcm == a * b, \
        f"GCD-LCM identity failed: gcd({a},{b}) * lcm({a},{b}) = {rust_gcd * rust_lcm} != {a * b}"

def test_prime_next_prime():
    """Property: next_prime(p) > p and is_prime(next_prime(p))"""
    n = random_int(1000)

    rust_n = rustmath.PyInteger(n)
    next_p = rust_n.next_prime()
    next_p_val = int(str(next_p))

    assert next_p_val > n, f"next_prime({n}) = {next_p_val} is not greater than {n}"
    assert next_p.is_prime(), f"next_prime({n}) = {next_p_val} is not prime"

    # Verify with SageMath
    sage_next = next_prime(n)
    assert next_p_val == sage_next, f"next_prime mismatch: Rust={next_p_val}, Sage={sage_next}"

def test_divisor_count():
    """Property: All divisors divide n evenly"""
    n = random_int(1000)

    rust_n = rustmath.PyInteger(n)
    divisors = [int(d) for d in rust_n.divisors()]

    for d in divisors:
        assert n % d == 0, f"{d} is not a divisor of {n}"

    # Compare count with SageMath
    sage_divs = divisors(n)
    assert len(divisors) == len(sage_divs), \
        f"Divisor count mismatch for {n}: Rust={len(divisors)}, Sage={len(sage_divs)}"

if __name__ == '__main__':
    random.seed(42)  # Reproducible tests

    tester = PropertyTester(num_tests=50)

    print("=" * 60)
    print("Property-Based Testing: RustMath vs SageMath")
    print("=" * 60)

    tester.test_property("GCD Commutativity", test_gcd_commutative)
    tester.test_property("GCD-LCM Identity", test_gcd_lcm_identity)
    tester.test_property("next_prime Properties", test_prime_next_prime)
    tester.test_property("Divisor Validity", test_divisor_count)

    tester.report()

    sys.exit(0 if tester.failed == 0 else 1)
```

---

## Module-Specific Testing Guides

### Testing Symbolic Computation

**File: `test-tools/src/bin/test_symbolic.rs`**
```rust
use rustmath_symbolic::{Expr, Variable};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let operation = &args[1];

    match operation.as_str() {
        "diff" => {
            // Example: ./test-symbolic diff "x^2 + 3*x + 2" x
            // For now, manually construct expression
            let x = Variable::new("x");
            let expr = Expr::from(x.clone()).pow(2) + Expr::from(x.clone()) * 3 + Expr::from(2);
            let derivative = expr.diff(&x);
            println!("{}", derivative);
        },
        _ => eprintln!("Unknown operation"),
    }
}
```

**File: `sage-tests/test_symbolic.py`**
```python
#!/usr/bin/env sage -python
from sage.all import *

def test_differentiation():
    """Compare symbolic differentiation"""
    var('x')

    test_expressions = [
        x^2,
        x^3 + 2*x^2 + x + 1,
        sin(x),
        exp(x),
        x^2 * exp(x),
    ]

    for expr in test_expressions:
        sage_diff = diff(expr, x)
        print(f"d/dx({expr}) = {sage_diff}")
        # TODO: Compare with Rust output once expression parsing is implemented
```

### Testing Polynomials

**File: `sage-tests/test_polynomials.py`**
```python
#!/usr/bin/env sage -python
from sage.all import *

def test_polynomial_gcd():
    """Test polynomial GCD"""
    R.<x> = ZZ[]

    test_cases = [
        (x^2 - 1, x - 1),
        (x^3 - 1, x^2 - 1),
        (x^4 - 1, x^3 - 1),
    ]

    for p1, p2 in test_cases:
        gcd_result = gcd(p1, p2)
        print(f"gcd({p1}, {p2}) = {gcd_result}")
        # TODO: Compare with Rust implementation
```

### Testing Finite Fields

**File: `sage-tests/test_finite_fields.py`**
```python
#!/usr/bin/env sage -python
from sage.all import *

def test_finite_field_ops():
    """Test finite field operations"""
    F = GF(7)

    for a in F:
        for b in F:
            if b != 0:
                result = a / b
                print(f"{a} / {b} = {result} (mod 7)")
                # Verify: result * b ≡ a (mod 7)
                assert (int(result) * int(b)) % 7 == int(a) % 7

    print("\n✓ All finite field tests passed")

def test_conway_polynomials():
    """Test Conway polynomial usage"""
    for p in [2, 3, 5, 7]:
        for n in [2, 3, 4]:
            try:
                F = GF(p^n, 'a', modulus='conway')
                print(f"GF({p}^{n}) modulus: {F.modulus()}")
            except Exception as e:
                print(f"Could not construct GF({p}^{n}): {e}")

if __name__ == '__main__':
    test_finite_field_ops()
    print()
    test_conway_polynomials()
```

---

## Recommended Workflow

### Phase 1: Quick Validation (Week 1)
1. Build CLI test tools for 2-3 modules (integers, rationals)
2. Write basic Python comparison scripts
3. Run on ~20 hand-picked test cases per module
4. Goal: Verify basic correctness

### Phase 2: PyO3 Integration (Week 2-3)
1. Set up PyO3 bindings for core modules
2. Rewrite tests to use direct Python calls
3. Add random test generation
4. Goal: Automate testing with 100+ cases per function

### Phase 3: Comprehensive Suite (Ongoing)
1. Add property-based testing
2. Create CI/CD integration (run tests automatically)
3. Build test coverage dashboard
4. Goal: Continuous validation against SageMath

---

## Tips and Best Practices

### Handling Large Integers
SageMath's `Integer` is arbitrary precision. Ensure your Rust tests use appropriate ranges:
```python
# Be careful with large numbers
n = 2^1000  # This works in SageMath
rust_n = rustmath.PyInteger(n)  # May need special handling
```

### Handling Different Output Formats
SageMath and Rust may format results differently:
```python
# Normalize before comparing
sage_result = str(factor(60)).replace(" ", "").replace("*", " ")
rust_result = rust_output.replace("*", " ")
```

### Performance Benchmarking
Add timing to compare performance:
```python
import time

start = time.time()
sage_result = some_computation()
sage_time = time.time() - start

start = time.time()
rust_result = rust_computation()
rust_time = time.time() - start

print(f"SageMath: {sage_time:.4f}s, Rust: {rust_time:.4f}s, Speedup: {sage_time/rust_time:.2f}x")
```

### Debugging Mismatches
When tests fail, use SageMath's introspection:
```python
# Understand what SageMath is doing
sage: n = 12345
sage: type(n.factor())  # See return type
sage: n.factor??        # View source code (if available)
```

---

## Next Steps

1. **Start with CLI testing** (easiest entry point)
2. **Choose 1-2 modules** to validate first (recommend: integers, rationals)
3. **Build iteratively**: Add tests as you implement features
4. **Automate**: Set up scripts to run on every commit
5. **Document mismatches**: Create issues for any discrepancies found

---

## Additional Resources

- **PyO3 Documentation**: https://pyo3.rs/
- **SageMath Documentation**: https://doc.sagemath.org/
- **Property-Based Testing**: Consider adding `hypothesis` for Python side
- **CI Integration**: Can run SageMath in Docker for automated testing

---

## Summary

This plan provides three approaches:

| Method | Setup Time | Flexibility | Speed | Recommended For |
|--------|-----------|-------------|-------|----------------|
| CLI Testing | 1-2 hours | Medium | Slow | Quick validation, getting started |
| PyO3 Bindings | 1-2 days | High | Fast | Comprehensive testing, production use |
| Property Testing | 2-3 days | Very High | Fast | Continuous validation, finding edge cases |

**Recommendation**: Start with CLI testing to validate 2-3 modules immediately, then invest in PyO3 bindings for long-term comprehensive testing.
