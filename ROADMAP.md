# RustMath: SageMath Rewrite in Rust

## Vision
Create a high-performance, memory-safe computer algebra system in Rust that provides equivalent functionality to SageMath while leveraging Rust's performance, safety guarantees, and modern tooling.

## Architecture

### Core Design Principles
1. **Modular**: Separate crates for different mathematical domains
2. **Type-Safe**: Leverage Rust's type system for mathematical correctness
3. **Performance**: Zero-cost abstractions and efficient algorithms
4. **Interoperability**: C FFI for integration with existing mathematical libraries
5. **Memory-Safe**: No undefined behavior, guaranteed thread safety

### Workspace Structure
```
rustmath/
├── rustmath-core/          # Core algebraic structures and traits
├── rustmath-integers/      # Integer arithmetic (arbitrary precision)
├── rustmath-rationals/     # Rational numbers
├── rustmath-polynomials/   # Polynomial rings
├── rustmath-matrix/        # Linear algebra
├── rustmath-calculus/      # Symbolic differentiation/integration
├── rustmath-numbertheory/  # Number theory algorithms
├── rustmath-combinatorics/ # Combinatorics
├── rustmath-geometry/      # Geometric objects and algorithms
├── rustmath-graphs/        # Graph theory
├── rustmath-crypto/        # Cryptographic primitives
├── rustmath-symbolic/      # Symbolic expression engine
└── rustmath-cli/           # Command-line interface
```

## Implementation Phases

### Phase 1: Foundation (Core Algebraic Structures)
**Goal**: Basic mathematical building blocks

#### 1.1 Core Traits and Types ✅
- [x] `Ring` trait (addition, multiplication, zero, one)
- [x] `Field` trait (division)
- [x] `EuclideanDomain` trait (division with remainder)
- [x] `Group` trait
- [x] `Module` trait
- [x] Error handling types

#### 1.2 Integer Arithmetic ✅
- [x] Arbitrary precision integers (using `num-bigint`)
- [x] GCD, LCM algorithms
- [x] Modular arithmetic
- [x] Prime testing (Miller-Rabin, etc.)
- [x] Integer factorization (trial division)
- [x] Pollard's Rho factorization
- [x] Chinese Remainder Theorem

#### 1.3 Rational Numbers ✅
- [x] Rational number type with automatic simplification
- [x] Arithmetic operations
- [x] Continued fractions

#### 1.4 Polynomial Rings 🚧
- [x] Univariate polynomials
- [x] Multivariate polynomials
- [x] Polynomial arithmetic
- [x] GCD for polynomials (basic Euclidean algorithm)
- [ ] Polynomial factorization (TODO)
- [ ] Advanced GCD algorithms (multivariate, subresultants)

### Phase 2: Linear Algebra 🚧
**Goal**: Matrix and vector operations

#### 2.1 Dense Matrices and Basic Operations ✅
- [x] Dense matrix implementation over generic rings
- [x] Matrix creation (zeros, identity, from_vec)
- [x] Element access (get, set)
- [x] Matrix arithmetic (addition, subtraction, multiplication)
- [x] Transpose
- [x] Row and column extraction
- [x] Trace
- [x] Determinant (cofactor expansion for small matrices)

#### 2.2 Linear System Solving ✅
- [x] Gaussian elimination
- [x] Row echelon form (REF)
- [x] Reduced row echelon form (RREF)
- [x] Rank computation
- [x] Matrix inversion (Gauss-Jordan)
- [x] Linear system solver (Ax = b)

#### 2.3 Matrix Decompositions ✅ (partial)
- [x] LU decomposition (Doolittle's algorithm)
- [x] PLU decomposition (with partial pivoting)
- [x] Determinant via LU (more efficient for large matrices)
- [ ] QR decomposition (TODO)
- [ ] Cholesky decomposition (TODO)
- [ ] SVD (Singular Value Decomposition) (TODO)

#### 2.4 Advanced Topics (TODO)
- [ ] Eigenvalues and eigenvectors
- [ ] Sparse matrices (CSR, COO formats)
- [ ] Iterative solvers (Conjugate Gradient, GMRES)
- [ ] Matrix norms and condition numbers

### Phase 3: Symbolic Computation
**Goal**: Expression manipulation and simplification

- [ ] Expression tree structure
- [ ] Expression parsing
- [ ] Simplification rules
- [ ] Pattern matching
- [ ] Substitution
- [ ] Expression ordering

### Phase 4: Calculus
**Goal**: Differentiation and integration

- [ ] Symbolic differentiation
- [ ] Chain rule, product rule, quotient rule
- [ ] Integration (pattern matching)
- [ ] Series expansion (Taylor, Laurent)
- [ ] Limits
- [ ] Numerical integration

### Phase 5: Number Theory
**Goal**: Advanced number-theoretic algorithms

- [ ] Primality testing (deterministic and probabilistic)
- [ ] Prime generation
- [ ] Integer factorization (Pollard's rho, etc.)
- [ ] Discrete logarithm
- [ ] Elliptic curves
- [ ] Modular forms

### Phase 6: Combinatorics
**Goal**: Combinatorial objects and algorithms

- [ ] Permutations
- [ ] Combinations
- [ ] Partitions
- [ ] Young tableaux
- [ ] Posets (Partially Ordered Sets)

### Phase 7: Graph Theory
**Goal**: Graph algorithms and structures

- [ ] Graph representations (adjacency list, matrix)
- [ ] Graph traversal (DFS, BFS)
- [ ] Shortest paths
- [ ] Spanning trees
- [ ] Graph coloring
- [ ] Matching algorithms

### Phase 8: Geometry
**Goal**: Geometric objects and computations

- [ ] Points, lines, planes
- [ ] Polygons and polyhedra
- [ ] Transformations
- [ ] Computational geometry algorithms

### Phase 9: Cryptography
**Goal**: Cryptographic primitives and protocols

- [ ] RSA
- [ ] Elliptic curve cryptography
- [ ] Hashing algorithms
- [ ] Random number generation

### Phase 10: User Interface
**Goal**: Interactive usage

- [ ] REPL (Read-Eval-Print Loop)
- [ ] Jupyter kernel
- [ ] LaTeX output
- [ ] Pretty printing
- [ ] Documentation system

## Technical Decisions

### Dependencies
- `num-bigint`: Arbitrary precision integers
- `num-rational`: Rational numbers
- `num-traits`: Numeric traits
- `ndarray`: N-dimensional arrays
- `rayon`: Parallel computation
- `serde`: Serialization
- `nom` or `pest`: Parsing

### Performance Targets
- Match or exceed Python/Cython SageMath performance
- Parallel computation where applicable
- SIMD optimization for critical paths
- Memory efficiency through zero-copy operations

### Testing Strategy
- Unit tests for all mathematical operations
- Property-based testing with `proptest`
- Benchmarks against SageMath
- Fuzzing for parser and critical algorithms

## Timeline Estimate

This is a multi-year project. Rough estimates:
- **Phase 1 (Foundation)**: 3-6 months
- **Phase 2 (Linear Algebra)**: 2-3 months
- **Phase 3 (Symbolic)**: 4-6 months
- **Phase 4 (Calculus)**: 3-4 months
- **Phase 5 (Number Theory)**: 3-4 months
- **Phase 6-9**: 6-12 months
- **Phase 10 (UI)**: 2-3 months

**Total**: 2-3 years for feature parity with core SageMath functionality

## Contributing
This is an ambitious project that will require community effort. Contributions welcome in all areas!

## License
To be determined (SageMath is GPL v2+)
