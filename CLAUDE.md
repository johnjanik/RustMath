# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RustMath is a Rust rewrite of SageMath's core mathematical functionality (~35% complete). The project provides exact arithmetic, symbolic computation, and comprehensive mathematical structures without any unsafe code.

## Build and Development Commands

```bash
# Build all crates
cargo build

# Run all tests
cargo test

# Run tests for a specific crate
cargo test -p rustmath-integers
cargo test -p rustmath-matrix

# Run a specific test
cargo test --package rustmath-symbolic --lib -- tests::test_differentiation

# Build with optimizations
cargo build --release

# Check compilation without building
cargo check

# Run clippy linter
cargo clippy

# Format code
cargo fmt

# Build documentation
cargo doc --open

# Clean build artifacts
cargo clean
```

## High-Level Architecture

### Core Design Pattern: Trait-Based Generics
The codebase uses Rust traits to define algebraic structures that work over ANY mathematical type:

```rust
// Matrix works over any Ring (integers, rationals, polynomials, finite fields, etc.)
pub struct Matrix<R: Ring> { ... }

// Operations generic over algebraic structures
impl<R: EuclideanDomain> Matrix<R> {
    pub fn det(&self) -> R { ... }
}
```

### Module Structure (17 crates)
- **rustmath-core**: Fundamental traits (Ring, Field, EuclideanDomain)
- **rustmath-integers**: Integer operations, factorization (Pollard's Rho), primality (Miller-Rabin)
- **rustmath-rationals**: Exact rational arithmetic
- **rustmath-matrix**: Generic matrix operations, LU/PLU decomposition, Gaussian elimination
- **rustmath-symbolic**: Expression trees, differentiation, assumptions, simplification
- **rustmath-polynomials**: Univariate/multivariate polynomials with factorization
- **rustmath-finitefields**: GF(p) and GF(p^n) with Conway polynomials
- **rustmath-padics**: p-adic numbers
- **rustmath-reals**: Real numbers (currently f64-based)
- **rustmath-complex**: Complex arithmetic
- **rustmath-combinatorics**: Permutations, partitions, Young tableaux
- **rustmath-graphs**: Graph algorithms (BFS/DFS, coloring, connectivity)
- **rustmath-geometry**: Points, lines, polytopes
- **rustmath-quadraticforms**: Quadratic forms for number theory
- **rustmath-powerseries**: Formal power series with Newton-Raphson inversion
- **rustmath-modulararithmetic**: Modular arithmetic as Ring composition
- **rustmath-sparsematrix**: Sparse matrix representations (CSR/CSC)

### Key Architectural Decisions

1. **Zero Unsafe Code**: All operations use safe Rust, preventing memory corruption
2. **Exact Arithmetic**: No floating-point errors; exact rational and symbolic computation
3. **Expression Evaluation Pipeline**: Symbolic expressions evaluate via: parse → build tree → substitute → simplify → compute
4. **Assumption Propagation**: Symbolic system tracks assumptions (e.g., x > 0) for intelligent simplification
5. **No Circular Dependencies**: Clean DAG structure between crates

### Non-Obvious Implementation Details

1. **Matrix Determinant**: Uses PLU decomposition with pivoting for numerical stability
2. **Symbolic Differentiation**: Recursive tree traversal with chain rule application
3. **Finite Field Construction**: Conway polynomials ensure compatible embeddings between fields
4. **Power Series Inversion**: Newton-Raphson iteration for efficient computation
5. **Modular Arithmetic**: Implemented as type composition `ModularInteger<R: Ring>`
6. **Quadratic Forms**: Specialized for number theory applications (not just linear algebra)

### Testing Strategy

Tests are organized within each crate's `src/lib.rs` or dedicated test modules:
- Unit tests: `#[cfg(test)] mod tests { ... }`
- Integration tests possible but not yet implemented
- ~60 test modules across all crates
- Focus on mathematical correctness over performance

### Known Issues and Limitations

1. **Sparse Matrix Tests**: Won't compile due to trait bound issues with generic parameters
2. **No Symbolic Integration**: Only differentiation implemented
3. **No Expression Parsing**: Cannot parse strings like "x^2 + 3*x + 2"
4. **Real Numbers**: Currently f64-based; arbitrary precision planned
5. **Gröbner Bases**: Partially implemented for multivariate polynomial ideals

### Development Workflow

1. Most development happens in feature branches
2. Test changes with `cargo test -p <crate-name>` before full test suite
3. The project follows standard Rust conventions (snake_case, module organization)
4. Documentation uses inline `///` comments for public APIs
5. THINGS_TO_DO.md tracks implementation progress by SageMath section

### Important Files

- **THINGS_TO_DO.md**: Detailed progress tracking (which SageMath sections are complete)
- **README.md**: Project overview and goals
- **Cargo.toml**: Workspace configuration with all 17 crates
- Each crate's **src/lib.rs**: Entry point with public API exports