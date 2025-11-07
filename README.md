# RustMath

A high-performance computer algebra system written in Rust, aiming to provide equivalent functionality to [SageMath](https://github.com/sagemath/sage) while leveraging Rust's performance, memory safety, and modern tooling.

## Overview

RustMath is an ambitious project to rewrite the SageMath computer algebra system in Rust. This provides:

- **Performance**: Zero-cost abstractions and efficient algorithms
- **Safety**: Memory safety and thread safety guaranteed by Rust
- **Modularity**: Clean separation of concerns with multiple specialized crates
- **Modern Tooling**: Easy integration with Rust ecosystem tools

## Project Status

🚧 **Early Development** - This project is in active development. Core functionality is being implemented.

### Completed Components

- ✅ Core algebraic traits (Ring, Field, EuclideanDomain, etc.)
- ✅ Arbitrary precision integers with GCD/LCM algorithms
- ✅ Rational numbers with automatic simplification
- ✅ Univariate polynomials over rings
- ✅ Modular arithmetic
- ✅ Prime number algorithms (trial division, Miller-Rabin)
- ✅ Integer factorization
- ✅ Matrix and vector operations
- ✅ Symbolic expression system
- ✅ Expression simplification
- ✅ Symbolic differentiation
- ✅ Basic combinatorics (factorial, binomial coefficients)

### In Progress

- 🔄 Multivariate polynomials
- 🔄 Advanced linear algebra (eigenvalues, decompositions)
- 🔄 Symbolic integration
- 🔄 Series expansions

### Planned

See [ROADMAP.md](ROADMAP.md) for the complete development plan.

## Architecture

RustMath is organized as a Cargo workspace with specialized crates:

```
rustmath/
├── rustmath-core/          # Core algebraic traits and types
├── rustmath-integers/      # Arbitrary precision integer arithmetic
├── rustmath-rationals/     # Rational numbers
├── rustmath-polynomials/   # Polynomial rings
├── rustmath-matrix/        # Linear algebra (matrices, vectors)
├── rustmath-symbolic/      # Symbolic expression engine
├── rustmath-calculus/      # Calculus (derivatives, integrals)
├── rustmath-numbertheory/  # Number theory algorithms
├── rustmath-combinatorics/ # Combinatorial structures
├── rustmath-geometry/      # Geometric objects
└── rustmath-graphs/        # Graph theory
```

## Quick Start

### Prerequisites

- Rust 1.70+ (2021 edition)
- Cargo

### Building

```bash
# Clone the repository
git clone https://github.com/johnjanik/RustMath.git
cd RustMath

# Build all crates
cargo build

# Run tests
cargo test

# Build with optimizations
cargo build --release
```

### Usage Examples

#### Basic Arithmetic

```rust
use rustmath_integers::Integer;

// Arbitrary precision integers
let a = Integer::from(12345678901234567890_u64);
let b = Integer::from(98765432109876543210_u64);
let sum = a + b;

// GCD and LCM
let x = Integer::from(48);
let y = Integer::from(18);
let gcd = x.gcd(&y);  // 6
let lcm = x.lcm(&y);  // 144
```

#### Rational Numbers

```rust
use rustmath_rationals::Rational;

let a = Rational::new(1, 2).unwrap();  // 1/2
let b = Rational::new(1, 3).unwrap();  // 1/3
let sum = a + b;  // 5/6 (automatically simplified)
```

#### Polynomials

```rust
use rustmath_polynomials::UnivariatePolynomial;
use rustmath_integers::Integer;

// Create polynomial: 1 + 2x + 3x²
let p = UnivariatePolynomial::new(vec![
    Integer::from(1),
    Integer::from(2),
    Integer::from(3),
]);

// Evaluate at x = 2: 1 + 4 + 12 = 17
let result = p.eval(&Integer::from(2));

// Compute derivative: 2 + 6x
let deriv = p.derivative();
```

#### Symbolic Computation

```rust
use rustmath_symbolic::Expr;
use rustmath_calculus::differentiate;

// Build expression: x² + 2x + 1
let x = Expr::symbol("x");
let expr = x.clone().pow(Expr::from(2))
         + Expr::from(2) * x.clone()
         + Expr::from(1);

// Differentiate: 2x + 2
let deriv = differentiate(&expr, "x");
```

#### Matrices

```rust
use rustmath_matrix::Matrix;

// Create a 2x2 matrix
let m = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();

// Matrix multiplication
let id = Matrix::identity(2);
let result = (m.clone() * id).unwrap();

// Transpose
let mt = m.transpose();
```

#### Prime Numbers

```rust
use rustmath_integers::Integer;
use rustmath_integers::prime::*;

// Check primality
let n = Integer::from(17);
assert!(is_prime(&n));

// Generate next prime
let next = next_prime(&Integer::from(20));  // 23

// Factor a number
let factors = factor(&Integer::from(60));
// [(2, 2), (3, 1), (5, 1)] representing 2² × 3 × 5
```

## Features

### Core Mathematical Structures

- **Algebraic Traits**: Ring, Field, EuclideanDomain, Group, Module
- **Number Types**: Arbitrary precision integers, rationals, modular integers
- **Polynomials**: Univariate and (planned) multivariate polynomials
- **Linear Algebra**: Matrices, vectors, linear operations

### Number Theory

- Prime testing (Miller-Rabin, trial division)
- Integer factorization
- GCD, LCM, extended Euclidean algorithm
- Modular exponentiation
- Chinese Remainder Theorem (planned)

### Symbolic Computation

- Expression building and manipulation
- Pattern matching and simplification
- Symbolic differentiation
- Expression evaluation

### Calculus

- Symbolic differentiation (power, chain, product, quotient rules)
- Trigonometric and exponential functions
- Integration (planned)
- Series expansions (planned)

### Linear Algebra

- Matrix operations (add, multiply, transpose)
- Vector spaces
- Determinants (planned)
- Eigenvalues and eigenvectors (planned)
- Matrix decompositions (planned)

## Development

### Running Tests

```bash
# Run all tests
cargo test

# Run tests for a specific crate
cargo test -p rustmath-integers

# Run with output
cargo test -- --nocapture
```

### Benchmarks

```bash
cargo bench
```

### Documentation

```bash
# Generate and open documentation
cargo doc --open

# Generate documentation for all crates
cargo doc --workspace --no-deps
```

## Contributing

Contributions are welcome! This is a massive undertaking that will benefit from community involvement.

### Areas Needing Help

- Implementing additional algorithms
- Performance optimization
- Documentation and examples
- Testing and validation
- Bug fixes

### Development Process

1. Fork the repository
2. Create a feature branch
3. Implement your changes with tests
4. Run `cargo test` and `cargo fmt`
5. Submit a pull request

## Comparison with SageMath

| Feature | SageMath | RustMath |
|---------|----------|----------|
| Language | Python/Cython | Rust |
| Performance | Fast (with Cython) | Very Fast (native) |
| Memory Safety | Runtime checks | Compile-time guarantees |
| Parallelism | Limited (GIL) | Native (rayon) |
| Installation | Complex (many deps) | Simple (cargo) |
| Lines of Code | ~2 million | Growing |

## Performance

RustMath aims to match or exceed SageMath's performance through:

- Zero-cost abstractions
- SIMD optimization where applicable
- Efficient memory management
- Parallel computation with rayon
- Native code compilation

## License

GPL-2.0-or-later (to maintain compatibility with SageMath)

## Acknowledgments

- [SageMath](https://www.sagemath.org/) - The original inspiration
- [num-bigint](https://github.com/rust-num/num-bigint) - Arbitrary precision integers
- The Rust community

## Links

- [Project Roadmap](ROADMAP.md)
- [SageMath Repository](https://github.com/sagemath/sage)
- [Documentation](https://docs.rs/rustmath)

---

**Note**: This is an independent reimplementation and is not affiliated with the official SageMath project.
