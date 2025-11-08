# RustMath Architecture: Quick Reference Guide

## Project At A Glance

- **Purpose**: SageMath computer algebra system rewritten in Rust
- **Status**: ~35% complete, core modules working
- **Code**: ~20,700 lines across 17 crates
- **Tests**: 60+ test modules, some with compilation issues
- **License**: GPL-2.0-or-later

## The 17 Crates

| Crate | Purpose | Status | Key Type |
|-------|---------|--------|----------|
| `rustmath-core` | Trait definitions (Ring, Field, etc.) | ✅ | `trait Ring` |
| `rustmath-integers` | Arbitrary precision integers | ✅ | `Integer` |
| `rustmath-rationals` | Rational numbers auto-simplified | ✅ | `Rational` |
| `rustmath-reals` | Real numbers (f64-based) | 🚧 | `Real` |
| `rustmath-complex` | Complex numbers | ✅ | `Complex` |
| `rustmath-polynomials` | Univariate/multivariate polynomials | 🚧 | `UnivariatePolynomial<R>` |
| `rustmath-powerseries` | Truncated power series | ✅ | `PowerSeries<R>` |
| `rustmath-finitefields` | Finite fields GF(p), GF(p^n) | ✅ | `PrimeField`, `ExtensionField` |
| `rustmath-padics` | p-adic numbers/integers | ✅ | `PadicInteger`, `PadicRational` |
| `rustmath-matrix` | Linear algebra & matrices | 🚧 | `Matrix<R>`, `Vector<R>` |
| `rustmath-calculus` | Symbolic differentiation | 🚧 | Uses `Expr` |
| `rustmath-numbertheory` | Primes, factorization, CRT | 🚧 | Functions + `QuadraticForm` |
| `rustmath-combinatorics` | Permutations, partitions, binomial | ✅ | `Permutation`, `Partition` |
| `rustmath-graphs` | Graph theory algorithms | 🚧 | `Graph<V>` |
| `rustmath-geometry` | Geometry (placeholder) | ⬜ | - |
| `rustmath-crypto` | RSA encryption/decryption | 🚧 | `KeyPair` |
| `rustmath-symbolic` | Expression system & simplification | 🚧 | `Expr`, `Symbol` |

## Core Design Patterns

### 1. Generic Algorithms Over Traits
```rust
// Works over ANY ring, not just floats
fn determinant<R: Ring>(m: &Matrix<R>) -> R { ... }

// One implementation, infinite types:
// Matrix<Integer>, Matrix<Rational>, Matrix<Polynomial>, ...
```

### 2. No Unsafe Code
- Pure Rust, no raw pointers
- Memory safety guaranteed by compiler
- Borrow checker enforces correctness

### 3. Result-Based Error Handling
```rust
// Panics only in truly unrecoverable situations
// Invalid math returns Err, not panic
pub fn inverse(&self) -> Result<Self>
```

### 4. Exact Arithmetic
- Integer: Arbitrary precision via num-bigint
- Rational: Auto-simplified to lowest terms
- Polynomials: Exact coefficients
- **No floating-point accumulation errors**

### 5. Arc-Based Expression Trees
```rust
pub enum Expr {
    Binary(BinaryOp, Arc<Expr>, Arc<Expr>),  // Cheap cloning
    Unary(UnaryOp, Arc<Expr>),
    // ...
}
```

### 6. Modular Arithmetic as Ring Implementation
```rust
// ModularInteger implements Ring
// Same algorithms work over Z/nZ
pub struct ModularInteger { value: Integer, modulus: Integer }
```

### 7. Assumption-Based Simplification
```rust
// Expressions can have properties
assume(x, Property::Positive)
// Affects simplification: sqrt(x²) → x (not |x|)
```

## Dependency Flows (Downward Only)

```
rustmath-symbolic ─┐
rustmath-calculus ─┼─→ rustmath-polynomials ─┐
rustmath-matrix ───┼────────────────────┬─→ rustmath-rationals ─┐
rustmath-numbertheory ─┐                │    rustmath-integers ─┤
rustmath-graphs ──────┤                │    rustmath-finitefields
rustmath-combinatorics ├───────────────→ rustmath-core ←──┤
rustmath-crypto ──────┤                └─ rustmath-reals
rustmath-padics ──────┘                   rustmath-complex
rustmath-geometry ───────────────────────────────────────┘
```

Key properties:
- No circular dependencies
- Each crate can be used independently
- Core has minimal dependencies

## Algorithms You'll Find

### Integers (rustmath-integers)
- Euclidean GCD/LCM
- Miller-Rabin primality test
- Pollard's Rho factorization
- Chinese Remainder Theorem
- Modular exponentiation

### Matrix (rustmath-matrix)
- Gaussian elimination with pivoting
- LU/PLU decomposition
- Matrix inversion (Gauss-Jordan)
- RREF computation
- Determinant (multiple algorithms)
- Sparse CSR format

### Polynomials (rustmath-polynomials)
- Univariate arithmetic
- Multivariate (sparse)
- GCD algorithms
- Derivative/integral
- Square-free factorization
- Rational root finding

### Symbolic (rustmath-symbolic)
- Expression tree
- Simplification rules
- Differentiation (complete)
- Substitution/evaluation
- Assumption system
- 11 property types

### Combinatorics (rustmath-combinatorics)
- Permutation generation & properties
- Integer partitions
- Binomial coefficients
- Factorial, Catalan, Fibonacci
- Stirling numbers

### Graph (rustmath-graphs)
- BFS/DFS traversal
- Connectivity testing
- Shortest paths (BFS-based)
- Graph coloring (greedy)
- Bipartite detection

## Major Architectural Strengths

1. **Type Safety**: Ring/Field traits prevent invalid operations
2. **Modularity**: 17 independent crates
3. **Genericity**: One implementation, many types
4. **Exactness**: No floating-point corruption
5. **Safety**: No unsafe code, memory-safe by default
6. **Testability**: 60+ test modules
7. **Extensibility**: Assumption system allows custom logic

## Known Issues & Limitations

| Issue | Impact | Fix |
|-------|--------|-----|
| Sparse matrix test compilation | Tests don't build | Relax Field to Ring |
| No symbolic integration | Can't integrate | Implement Risch algorithm |
| No expression parsing | Must build by code | Add parser combinators |
| No arbitrary precision floats | Limited precision | Add rug feature |
| Gröbner bases incomplete | Can't solve ideals | Implement algorithms |

## Quick Usage Examples

### Basic Arithmetic
```rust
let a = Integer::from(12);
let b = Integer::from(18);
println!("{}", a.gcd(&b));  // 6
```

### Matrices
```rust
let m = Matrix::from_vec(2, 2, vec![1, 2, 3, 4])?;
let inv = m.inverse()?;
```

### Symbolic
```rust
let x = Expr::symbol("x");
let expr = x.clone().pow(Expr::from(2)) + Expr::from(1);
let deriv = differentiate(&expr, &Symbol::new("x"));
```

### Polynomials
```rust
let p = UnivariatePolynomial::new(vec![1, 2, 3]);  // 1 + 2x + 3x²
let root = p.eval(&Integer::from(2));  // 17
```

## Files Worth Reading

| File | Why |
|------|-----|
| `rustmath-core/src/traits.rs` | Core algebraic traits |
| `rustmath-symbolic/src/expression.rs` | Expression tree design |
| `rustmath-matrix/src/matrix.rs` | Generic matrix implementation |
| `rustmath-symbolic/src/differentiate.rs` | Differentiation rules |
| `rustmath-symbolic/src/simplify.rs` | Simplification rules |
| `THINGS_TO_DO.md` | Implementation checklist |
| `PROJECT_SUMMARY.md` | Detailed status |

## Development Workflow

1. **Build**: `cargo build --all`
2. **Test**: `cargo test --all` (some tests fail)
3. **Docs**: `cargo doc --open`
4. **Lint**: Watch for warnings (few)
5. **Format**: Rust convention followed

## Next High-Impact Work

1. **Fix sparse matrix tests** (quick win)
2. **Add symbolic integration** (medium effort)
3. **Complete linear algebra decompositions** (QR, SVD)
4. **Improve simplification** (ongoing)
5. **Add expression parsing** (enables REPL)

## Why This Architecture Works

- **Trait-based**: Captures essential algebraic structures
- **Rust's type system**: Enforces mathematical properties at compile time
- **No runtime overhead**: Zero-cost abstractions
- **Exact arithmetic**: Prevents subtle bugs
- **Modular**: Each component can be used independently

The architecture demonstrates that **Rust is an excellent language for mathematical software**, offering safety and performance unavailable in Python-based systems like SageMath.

