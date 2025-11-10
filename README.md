# RustMath

**A Rust-based mathematical computation library designed to eventually integrate with SageMath**

[![License: GPL v2+](https://img.shields.io/badge/License-GPL%20v2+-blue.svg)](https://www.gnu.org/licenses/gpl-2.0)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)

---

## ⚠️ Project Status: Active Development / Experimental

**Important**: RustMath is in **early experimental development**. While the codebase compiles, it is **not production-ready** and contains:
- Numerous compiler warnings
- Failing tests in several modules
- Incomplete implementations
- Unoptimized code paths
- APIs subject to change

**This project is a research prototype** exploring how Rust can provide performance-critical computational backends for Python-based computer algebra systems like SageMath.

---

## Overview

RustMath is an ambitious effort to create high-performance mathematical computation libraries in Rust that can:

1. **Serve as performance-critical backends** for existing SageMath installations via Python FFI (PyO3)
2. **Demonstrate Rust's advantages** for mathematical computing: memory safety, parallelism, and zero-cost abstractions
3. **Gradually enable migration** of computationally intensive SageMath components to native Rust

### Core Philosophy

**This is NOT a full rewrite of SageMath.** Instead, RustMath focuses on:
- **Selective porting** of performance-critical algorithms
- **Python interoperability** via PyO3 bindings for seamless SageMath integration
- **Incremental adoption** allowing coexistence with existing Python/Cython code

---

## Why RustMath?

### Advantages Over Pure Python/Cython

| Aspect | Python/Cython | RustMath (Target) |
|--------|---------------|-------------------|
| **Performance** | Fast with Cython | 2-10x faster for many operations |
| **Parallelism** | Limited by GIL | Native thread safety, fearless concurrency |
| **Memory Safety** | Runtime checks | Compile-time guarantees |
| **Type Safety** | Dynamic/gradual | Strong static typing with generics |
| **Packaging** | Complex dependencies | Single binary, easier deployment |

### Use Cases

- Replace computationally intensive loops in SageMath
- Parallelize algorithms that are sequential in Python
- Provide memory-safe implementations of cryptographic primitives
- Enable WebAssembly-based browser computation (future)

---

## Current Implementation Status

### Overall Progress: ~68% Feature Parity (364/539 tracked functions)

See [`THINGS_TO_DO.md`](THINGS_TO_DO.md) for detailed function-by-function tracking and [`SAGEMATH_RUST_ROADMAP.md`](SAGEMATH_RUST_ROADMAP.md) for the comprehensive porting strategy.

### ✅ Well-Developed Areas (80-100% complete)

- **Polynomials**: Univariate/multivariate with factorization (Berlekamp, Rational Root Theorem)
- **Finite Fields**: GF(p) and GF(p^n) with Conway polynomials
- **Combinatorics**: Permutations, partitions, tableaux, posets
- **Graph Theory**: BFS/DFS, shortest paths, matching, coloring
- **Symbolic Differentiation**: Chain rule, product/quotient rules
- **Coding Theory**: Linear codes, Hamming, Reed-Solomon, BCH, Golay
- **Group Theory**: Permutation groups, matrix groups, abelian groups

### 🔧 Partially Implemented (50-80% complete)

- **Integers**: Basic operations complete, advanced number theory in progress
- **Matrices**: Core operations work, some decompositions incomplete
- **Symbolic Integration**: Basic patterns work, advanced techniques needed
- **Elliptic Curves**: Point arithmetic works, rank computation incomplete

### ⚠️ Known Issues

- **Tests**: Some test suites have failures (e.g., Pollard's p-1 factorization)
- **Warnings**: Extensive unused variable/import warnings throughout
- **Real/Complex Numbers**: Currently use f64 (arbitrary precision MPFR planned)
- **Python Bindings**: Only 3 modules exposed (integers, rationals, matrices)
- **Documentation**: API docs incomplete in many areas

---

## Architecture

RustMath is organized as a **Cargo workspace** with 17 specialized crates:

```
rustmath/
├── rustmath-core/           # Trait system (Ring, Field, EuclideanDomain)
├── rustmath-integers/       # Arbitrary precision integers, primality, factorization
├── rustmath-rationals/      # Exact rational arithmetic
├── rustmath-reals/          # Real numbers (f64-based, MPFR planned)
├── rustmath-complex/        # Complex numbers
├── rustmath-polynomials/    # Univariate/multivariate with factorization
├── rustmath-finitefields/   # GF(p), GF(p^n), Conway polynomials
├── rustmath-padics/         # p-adic numbers with Hensel lifting
├── rustmath-powerseries/    # Formal power series
├── rustmath-matrix/         # Dense/sparse matrices, decompositions
├── rustmath-symbolic/       # Expression trees, differentiation, integration
├── rustmath-combinatorics/  # Permutations, partitions, tableaux
├── rustmath-graphs/         # Graph algorithms
├── rustmath-geometry/       # Computational geometry
├── rustmath-crypto/         # Cryptographic primitives
├── rustmath-groups/         # Group theory
├── rustmath-stats/          # Statistics and probability
├── rustmath-numerical/      # Numerical methods
├── rustmath-logic/          # SAT solving, Boolean logic
├── rustmath-dynamics/       # Dynamical systems, chaos, fractals
├── rustmath-coding/         # Error-correcting codes
└── rustmath-databases/      # OEIS, Cunningham tables, Cremona database
```

### Python Bindings (PyO3)

**Location**: `rustmath-py/` (separate crate built with maturin)

**Status**: Experimental - only 3 modules currently exposed

---

## Installation & Usage

### For Rust Developers

```bash
# Clone repository
git clone https://github.com/johnjanik/RustMath.git
cd RustMath

# Build all crates (expect many warnings)
cargo build

# Run tests (expect some failures)
cargo test

# Build optimized binaries
cargo build --release

# Generate documentation
cargo doc --open
```

### For SageMath Integration (Experimental)

**Prerequisites**:
- Python 3.8+
- Existing SageMath installation
- Rust toolchain (1.70+)
- maturin (`pip install maturin`)

**Build Python bindings**:

```bash
cd rustmath-py

# Build wheel
maturin build --release

# Install in current Python environment
maturin develop --release
```

**Use from Python/SageMath**:

```python
# Import RustMath module
import rustmath

# Create integers
a = rustmath.Integer(123456789)
b = rustmath.Integer(987654321)

# Perform operations
gcd = a.gcd(b)
print(f"GCD: {gcd}")

# Create rationals
r1 = rustmath.Rational(3, 4)
r2 = rustmath.Rational(5, 6)
sum_r = r1 + r2
print(f"Sum: {sum_r}")

# Matrix operations
m = rustmath.Matrix([[1, 2], [3, 4]])
det = m.det()
print(f"Determinant: {det}")
```

**⚠️ Current Limitations of Python Bindings**:
- Only integers, rationals, and matrices are exposed
- No symbolic computation or polynomial APIs yet
- Limited error handling
- Performance not yet optimized for Python interop

---

## Example: Rust API Usage

### Integer Arithmetic

```rust
use rustmath_integers::Integer;
use rustmath_integers::prime::*;

fn main() {
    // Arbitrary precision integers
    let a = Integer::from(12345678901234567890u64);
    let b = Integer::from(98765432109876543210u64);

    // GCD/LCM
    let gcd = a.gcd(&b);
    let lcm = a.lcm(&b);

    // Primality testing
    let n = Integer::from(1000000007);
    if is_prime(&n) {
        println!("{} is prime", n);
    }

    // Factorization (may be slow for large numbers)
    let factors = factor(&Integer::from(60));
    println!("Factors: {:?}", factors); // [(2, 2), (3, 1), (5, 1)]
}
```

### Symbolic Differentiation

```rust
use rustmath_symbolic::{Expr, differentiate};

fn main() {
    // Build expression: x² + 2x + 1
    let x = Expr::symbol("x");
    let expr = x.pow(Expr::from(2)) + x * Expr::from(2) + Expr::from(1);

    // Differentiate
    let deriv = differentiate(&expr, "x");
    println!("Derivative: {}", deriv); // 2*x + 2
}
```

### Graph Algorithms

```rust
use rustmath_graphs::Graph;

fn main() {
    let mut g = Graph::new(5);
    g.add_edge(0, 1);
    g.add_edge(1, 2);
    g.add_edge(2, 3);
    g.add_edge(3, 4);

    // BFS from node 0
    let visited = g.bfs(0);
    println!("BFS order: {:?}", visited);

    // Check if connected
    println!("Connected: {}", g.is_connected());
}
```

---

## Design Principles

### 1. Trait-Based Generic Programming

All mathematical structures use Rust's trait system for maximum code reuse:

```rust
pub trait Ring: Clone + PartialEq {
    fn zero() -> Self;
    fn one() -> Self;
    fn add(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn neg(&self) -> Self;
}

// Matrix works over ANY Ring
pub struct Matrix<R: Ring> {
    rows: usize,
    cols: usize,
    data: Vec<R>,
}
```

**Benefits**:
- One implementation works for integers, rationals, polynomials, finite fields
- Type safety prevents mixing incompatible operations
- Zero runtime overhead

### 2. Zero Unsafe Code

RustMath's computational core avoids `unsafe` code entirely, providing:
- No buffer overflows
- No use-after-free errors
- No data races in parallel code
- Compile-time invariant checking

### 3. Exact Arithmetic

Unlike many CAS systems, RustMath prioritizes exact computation:
- Rational arithmetic (no floating-point errors)
- Symbolic manipulation
- Integer-based algorithms where possible

---

## Testing & Validation

### Running Tests

```bash
# All tests (expect failures)
cargo test --workspace

# Specific crate
cargo test -p rustmath-integers

# With output
cargo test -- --nocapture --test-threads=1

# Specific test
cargo test -p rustmath-symbolic test_differentiation
```

### Known Test Failures

- `rustmath-integers`: Pollard's p-1 factorization test
- Various modules: Unused variable/import warnings

### Future: SageMath Equivalence Testing

Planned test infrastructure (see [`SAGEMATH_RUST_ROADMAP.md`](SAGEMATH_RUST_ROADMAP.md)):

```python
# Automated validation against SageMath
import pytest
from sage.all import *
import rustmath

def test_gcd_equivalence():
    sage_result = gcd(48, 18)
    rust_result = rustmath.Integer(48).gcd(rustmath.Integer(18))
    assert int(sage_result) == int(rust_result)
```

---

## Contributing

**We welcome contributions!** This is a massive undertaking requiring expertise in both Rust and mathematics.

### Good First Issues

**For Rust developers**:
- Fix compiler warnings (unused variables, imports)
- Add parallelism to algorithms using Rayon
- Improve error messages and documentation
- Optimize hot paths identified by profiling

**For mathematicians/SageMath developers**:
- Port algorithms from SageMath to Rust
- Add mathematical function implementations
- Create equivalence tests against SageMath
- Improve mathematical documentation

### Contribution Workflow

1. Read [`CLAUDE.md`](CLAUDE.md) for project architecture
2. Check [`THINGS_TO_DO.md`](THINGS_TO_DO.md) for implementation status
3. Review [`SAGEMATH_RUST_ROADMAP.md`](SAGEMATH_RUST_ROADMAP.md) for strategy
4. Fork the repository and create a feature branch
5. Implement with tests and documentation
6. Run `cargo test`, `cargo clippy`, `cargo fmt`
7. Submit a pull request

### Code Standards

All contributions must:
- [ ] Pass `cargo build` (warnings acceptable initially)
- [ ] Include tests for new functionality
- [ ] Pass `cargo clippy` (or document why warnings are acceptable)
- [ ] Be formatted with `cargo fmt`
- [ ] Include documentation comments (`///` for public APIs)
- [ ] Avoid `unsafe` code unless absolutely necessary and justified

---

## Performance Targets

| Operation | SageMath Baseline | RustMath Target | Status |
|-----------|-------------------|-----------------|--------|
| Integer GCD (large) | 1.0x | 1.5-2x | 🔧 In progress |
| Matrix determinant (100x100 rationals) | 1.0x | 5-10x | 🔧 In progress |
| Polynomial multiplication (dense) | 1.0x | 2-3x | ✅ Achieved |
| Graph algorithms (10K nodes) | 1.0x | 3-5x | ✅ Achieved |
| Gröbner bases (3 vars, deg 4) | 1.0x | 3-7x | 🔧 In progress |

*Benchmarks to be validated against SageMath 10.x*

---

## Roadmap

### Near-Term (2025)

- [ ] Expand PyO3 bindings to all 17 crates
- [ ] Fix all failing tests
- [ ] Create SageMath equivalence test suite
- [ ] Implement arbitrary precision reals (MPFR)
- [ ] Comprehensive API documentation

### Mid-Term (2026)

- [ ] Production deployment in 1-2 SageMath subsystems
- [ ] Parallel algorithms using Rayon
- [ ] WebAssembly builds for browser-based CAS
- [ ] Performance benchmarking framework

### Long-Term (2027+)

- [ ] GPU acceleration for linear algebra
- [ ] Distributed computation framework
- [ ] Standalone CLI tool
- [ ] Integration with other CAS systems

See [`SAGEMATH_RUST_ROADMAP.md`](SAGEMATH_RUST_ROADMAP.md) for comprehensive details.

---

## Comparison with Related Projects

| Project | Focus | Integration | Status |
|---------|-------|-------------|--------|
| **SageMath** | Full-featured CAS | Python-native | Mature (2M LOC) |
| **SymPy** | Symbolic math | Pure Python | Mature |
| **RustMath** | Performance backend | PyO3 for SageMath | Experimental |
| **Pari/GP** | Number theory | C library | Mature |
| **FLINT** | Number theory | C library | Mature |
| **Singular** | Commutative algebra | C++ | Mature |

**RustMath's niche**: Modern Rust implementation with Python FFI for gradual SageMath enhancement.

---

## License

**GPL-2.0-or-later** to maintain compatibility with SageMath.

See [LICENSE](LICENSE) for details.

---

## Acknowledgments

This project builds on decades of computer algebra research and software development:

- **[SageMath](https://www.sagemath.org/)** - The inspiration and integration target
- **[num-bigint](https://github.com/rust-num/num-bigint)** - Arbitrary precision integer foundation
- **[PyO3](https://pyo3.rs/)** - Rust-Python interoperability
- **The Rust Community** - Language and ecosystem
- **Mathematical Software Community** - Algorithms and theory (PARI, FLINT, GAP, Singular, etc.)

---

## Resources

### Documentation

- **Project Documentation**: (to be deployed on docs.rs)
- **SageMath Documentation**: https://doc.sagemath.org/
- **Rust Book**: https://doc.rust-lang.org/book/
- **PyO3 Guide**: https://pyo3.rs/

### Key Files

- [`CLAUDE.md`](CLAUDE.md) - Architecture and development guidance
- [`THINGS_TO_DO.md`](THINGS_TO_DO.md) - Detailed implementation checklist
- [`SAGEMATH_RUST_ROADMAP.md`](SAGEMATH_RUST_ROADMAP.md) - Comprehensive porting strategy

### Community

- **Issues**: https://github.com/johnjanik/RustMath/issues
- **Discussions**: (to be set up)

---

## Disclaimer

**This is an independent research project and is not officially affiliated with the SageMath development team.**

RustMath is experimental software. Use in production environments is **strongly discouraged** at this stage.

---

## FAQ

**Q: Can I use RustMath instead of SageMath right now?**
A: No. RustMath is experimental and incomplete. Use SageMath for production work.

**Q: Will RustMath replace SageMath?**
A: No. RustMath aims to provide performance-critical backends that SageMath can optionally use via Python FFI. The Python layer provides the user interface.

**Q: Why Rust instead of improving Cython?**
A: Rust provides memory safety guarantees, fearless concurrency, and modern tooling that complement Cython's strengths. Both can coexist.

**Q: How can I help?**
A: See the Contributing section above. We need both Rust developers and mathematicians!

**Q: When will this be production-ready?**
A: Optimistically, initial production deployments in specific SageMath subsystems by late 2026. See the roadmap for details.

**Q: Does this compile?**
A: Yes, the project compiles with warnings. However, some tests fail and the code is not optimized or production-ready.

---

**Status**: Active Development (as of November 2025)

**Star** this repository if you're interested in high-performance mathematical computing with Rust!
