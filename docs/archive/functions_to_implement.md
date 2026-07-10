# Functions to Implement in RustMath

This document provides an exhaustive list of all remaining functions to be implemented in RustMath, organized by SageMath module.

**Total Functions**: 172 unimplemented (out of 539 tracked)
**Current Progress**: 367/539 (68%)

---

## 1. Basic Rings and Fields

### 1.1 Integers (sage.rings.integer) - 5 Missing
**SageMath Source**: `src/sage/rings/integer.pyx`, `integer_ring.py`

Missing functions (implied by 21/26 progress):
1. `.isqrt()` - Integer square root with remainder
2. `.is_perfect_power()` - Check if n = a^b for some a, b
3. `.perfect_power()` - Find (a, b) such that n = a^b if it exists
4. `.trial_division(bound)` - Trial division up to bound
5. `.nth_root(n, all=True)` - Return all nth roots (complex)

**Priority**: Medium (some are aliases/extensions of existing functions)

---

### 1.2 Rational Numbers (sage.rings.rational) - 1 Missing
**SageMath Source**: `src/sage/rings/rational.pyx`

Missing functions (implied by 11/12 progress):
1. `.height()` - Logarithmic height of rational number (max of |numerator|, |denominator|)

**Priority**: Low (specialized number theory function)

---

### 1.3-1.4 Real and Complex Numbers
**Note**: Current implementation uses f64. The main missing feature is:

**Missing**: Arbitrary precision arithmetic (MPFR/GMP bindings)
- `RealField(prec)` with true arbitrary precision
- `ComplexField(prec)` with true arbitrary precision
- Integration with `rug` crate or direct MPFR/GMP FFI

**Priority**: High (foundational for numerical work)

---

### 1.5 Polynomial Rings (sage.rings.polynomial) - Advanced Algorithms

**Missing advanced factorization algorithms**:
1. **Zassenhaus Algorithm** - Integer polynomial factorization
   - Source: `src/sage/rings/polynomial/polynomial_integer_dense_flint.pyx`
   - Lift factorizations from Z/pZ to Z

2. **LLL Algorithm** - Lattice basis reduction
   - Source: `src/sage/libs/fplll/`
   - Used in polynomial factorization and cryptography

3. **Hensel Lifting (general)** - Lift factorizations
   - Source: `src/sage/rings/polynomial/polynomial_element.pyx`
   - Already have basic version in p-adics

4. **Factorization over algebraic extensions**
   - Source: `src/sage/rings/polynomial/`
   - Factor over Q(α) for algebraic α

**Priority**: Medium (advanced features, current factorization works for common cases)

---

## 2. Linear Algebra (sage.matrix, sage.modules)

### 2.1 Matrices (sage.matrix.matrix) - 3 Missing
**SageMath Source**: `src/sage/matrix/`

Missing functions (implied by 32/35 progress):
1. `.minimal_polynomial()` - Minimal polynomial of a matrix
2. `.jordan_form()` - Jordan canonical form
3. `.rational_canonical_form()` - Already implemented but needs verification/extension

**Priority**: Medium (specialized linear algebra)

---

## 3. Number Theory (sage.rings.number_theory)

### 3.5 Quadratic Forms (sage.quadratic_forms) - 2 Missing
**SageMath Source**: `src/sage/quadratic_forms/`

**Explicitly marked as missing**:
1. ⬜ **Theta series** - `theta_series(q, var='q')`
   - Generating function: Σ q^(Q(v)) over lattice vectors v
   - Used in modular forms and number theory

2. ⬜ **Local densities** - `local_density(p, m)`
   - p-adic density of representations
   - Essential for Siegel mass formula

**Priority**: Low (specialized number theory, requires advanced theory)

---

## 4. Symbolic Computation (sage.symbolic)

### ✅ COMPLETE (21/21 functions)
All symbolic computation features are now implemented including:
- ✅ Factorization (`.factor()`)
- ✅ Radical canonicalization (`.canonicalize_radical()`)
- ✅ Equation solving (`.solve()`)
- ✅ Differentiation, integration, limits, series

---

## 5. Calculus (sage.calculus)

### 5.4 Series Expansions - 1 Missing
**SageMath Source**: `src/sage/calculus/calculus.py`

**Explicitly marked as missing**:
1. ⬜ **Puiseux series** - `puiseux_series(x, a, n)`
   - Laurent series with fractional exponents
   - f(x) = Σ a_n (x-a)^(n/k) for some k
   - Used in algebraic geometry

**Priority**: Low (specialized, Laurent series already implemented)

---

### 5.3 Limits - 1 Partially Implemented
**SageMath Source**: `src/sage/calculus/calculus.py`

**Partially complete**:
1. 🚧 **Limits at infinity** - `limit(f, x=oo)`
   - Basic support exists via substitution
   - Needs: Better handling of ∞ - ∞, ∞/∞ indeterminate forms
   - Asymptotic analysis improvements

**Priority**: Medium (commonly used in calculus)

---

## 6. Combinatorics (sage.combinat)

### ✅ COMPLETE (44/44 functions)

---

## 7. Graph Theory (sage.graphs)

### Missing Functions (~9-15 functions)

**7.1 Graph Construction**:
1. **Voronoi diagrams** ⬜ (explicitly marked)
   - Source: External computational geometry library
   - Partition of plane based on distance to points
   - Priority: Low (complex, requires computational geometry)

**7.2 Graph Algorithms** (Advanced):
2. **Maximum flow (advanced variants)**
   - Source: `src/sage/graphs/generic_graph.py`
   - Ford-Fulkerson with different implementations

3. **Minimum cut algorithms**
   - Source: `src/sage/graphs/generic_graph.py`
   - Stoer-Wagner algorithm

**7.5 Graph Coloring** (Advanced):
4. **Chromatic polynomial**
   - Source: `src/sage/graphs/chrompoly.pyx`
   - P(G,k) = number of proper k-colorings

5. **Edge chromatic number**
   - Source: `src/sage/graphs/graph_coloring.pyx`
   - Minimum colors needed for edge coloring

**7.6 Matching** (Advanced):
6. **Perfect matching enumeration**
7. **Maximum weight matching**

**7.7 Graph Properties**:
8. **Treewidth computation**
9. **Graph minors testing**

**7.8 Planar Graphs**:
10. **Planarity testing (advanced)**
11. **Planar embedding**

**Priority**: Low-Medium (specialized graph theory)

---

## 8. Geometry (sage.geometry)

### 8.3 Computational Geometry - 1 Missing

Already listed above under Graph Theory (Voronoi diagrams).

---

## 9. Algebraic Geometry (sage.schemes)

### 9.2 Elliptic Curves - 3 Missing
**SageMath Source**: `src/sage/schemes/elliptic_curves/`

**Explicitly marked as missing**:
1. ⬜ **Rank computation (rigorous)** - `rank(algorithm='mwrank')`
   - Mordell-Weil rank via descent
   - Requires: mwrank/eclib integration or pure Rust implementation
   - Very complex: involves 2-descent, 4-descent algorithms

2. ⬜ **Full L-functions** - `lseries()`
   - Complete analytic L-function L(E, s)
   - Requires: Complex analysis, Functional equation
   - Advanced: Modular parametrization

3. ⬜ **Modular forms** - `modular_form()`
   - Connection to modular forms via modularity theorem
   - Requires: Extensive modular forms infrastructure
   - Very advanced: q-expansions, Hecke operators

**Priority**: Low (very advanced, requires significant infrastructure)

---

## 10. Cryptography (sage.crypto)

### 10.3 Block Ciphers - 1 Missing
**SageMath Source**: `src/sage/crypto/block_cipher/`

**Explicitly marked as missing**:
1. ⬜ **AES (Advanced Encryption Standard)**
   - Source: `src/sage/crypto/block_cipher/sdes.py` (S-DES as reference)
   - Full AES-128, AES-192, AES-256
   - Components needed:
     - SubBytes (S-box substitution)
     - ShiftRows
     - MixColumns
     - AddRoundKey
     - Key expansion
   - Note: Simplified version via existing Feistel framework possible

**Priority**: Medium (widely used, but crypto crates exist)

---

## 11. Coding Theory (sage.coding)

### ✅ COMPLETE (13/13 functions)

---

## 12-20. Advanced Topics

### ✅ MOSTLY COMPLETE
- Group Theory: 100%
- Representation Theory: 100%
- Homological Algebra: 100%
- Category Theory: 100%
- Statistics: 100%
- Numerical Computation: 100%
- Logic and SAT: 100%
- Dynamics: 100%
- Databases: 100%

---

## High-Level Features Not Yet Decomposed (~130 functions)

These are major features mentioned in THINGS_TO_DO.md "Priority Areas" that haven't been broken down into specific functions yet:

### High Priority (Foundational)

#### 1. Arbitrary Precision Real/Complex Numbers (≈20 functions)
**Source**: `src/sage/rings/real_mpfr.pyx`, `src/sage/rings/complex_mpfr.pyx`

**Core Functions**:
- `RealField(prec, rnd='RNDN')` - Arbitrary precision reals
- `ComplexField(prec)` - Arbitrary precision complex
- All arithmetic operations with correct rounding
- Transcendental functions at arbitrary precision
- Interval arithmetic at arbitrary precision
- Conversion between precisions
- Special constants (π, e, etc.) at arbitrary precision

**Implementation Approach**:
- Option 1: FFI bindings to MPFR/GMP via `gmp-mpfr-sys`
- Option 2: Use `rug` crate (safe Rust wrapper)
- Option 3: Pure Rust implementation (very complex)

**Priority**: High (foundational for serious numerical work)

---

#### 2. Advanced Symbolic Integration (≈15 functions)
**Source**: `src/sage/symbolic/integration/`

**Current Status**: Basic table-based integration exists

**Missing Patterns**:
- Rational function integration (partial fractions)
- Trigonometric integrals (sin^m cos^n)
- Trigonometric substitutions
- Hyperbolic function integrals
- Integration by parts (pattern matching)
- Integration by substitution (automatic)
- Reduction formulas
- Special functions (Bessel, error function, etc.)
- Definite integral evaluation with limits
- Improper integrals
- Multiple integration (symbolic)
- Line integrals
- Surface integrals
- Vector calculus integrals

**Priority**: High (commonly used in calculus)

---

#### 3. Complete Gröbner Bases (≈10 functions)
**Source**: `src/sage/rings/polynomial/multi_polynomial_ideal.py`

**Current Status**: Basic Buchberger's algorithm implemented

**Missing**:
- F4 algorithm (Faugère)
- F5 algorithm (Faugère)
- Signature-based Gröbner bases
- Gröbner walk (conversion between orderings)
- FGLM algorithm (change of ordering)
- Elimination theory (symbolic)
- Syzygy computation
- Hilbert series from Gröbner basis
- Dimension and degree of ideals

**Priority**: Medium (important for algebraic geometry, but current implementation works)

---

#### 4. Extended Linear Algebra (≈15 functions)
**Source**: `src/sage/matrix/`

**Missing Decompositions**:
- SVD variants (thin, compact, truncated)
- QR variants (full, reduced)
- Schur decomposition
- Hessenberg form
- Cholesky with pivoting
- LDL decomposition
- Polar decomposition
- Matrix exponential (better algorithm)
- Matrix logarithm
- Matrix square root
- Matrix sign function
- Generalized eigenvalue problems
- Sparse matrix specialized algorithms

**Priority**: Medium (specialized linear algebra)

---

### Medium Priority (Frequently Used)

#### 5. Extended Number Theory (≈20 functions)
**Source**: `src/sage/rings/number_field/`, `src/sage/schemes/`

**Number Fields**:
- Algebraic number field construction
- Ideal arithmetic in number fields
- Class group computation
- Unit group computation
- Discriminant and different
- Galois group computation
- Embedding into C

**Diophantine Equations**:
- General Pell equation solver
- Generalized Pell equations
- Quadratic forms classification
- Ternary quadratic forms
- Binary quadratic forms (more complete)
- Thue equations
- Superelliptic equations

**Priority**: Medium (specialized number theory)

---

#### 6. Additional Graph Algorithms (≈15 functions)
**Source**: `src/sage/graphs/`

**Missing**:
- Graph isomorphism (general case, not just special cases)
- Subgraph isomorphism
- Graph homomorphism
- Strongly regular graphs
- Distance-regular graphs
- Spectral graph theory (eigenvalues of Laplacian)
- Expander graphs
- Ramsey theory computations
- Graph limits
- Graphons
- Network reliability
- Community detection
- Graph embedding (dimension reduction)

**Priority**: Low-Medium (specialized)

---

#### 7. Numerical Methods Extensions (≈10 functions)
**Source**: `src/sage/numerical/`

**ODE Solvers**:
- Adaptive step size ODE solvers
- Stiff ODE solvers (BDF methods)
- Symplectic integrators
- Implicit methods (backwards Euler)

**Optimization**:
- Constrained optimization
- Nonlinear least squares
- Trust region methods
- BFGS and L-BFGS
- Conjugate gradient

**Root Finding**:
- Multidimensional root finding
- Bracketing methods
- Hybrid methods

**Priority**: Medium (useful for applications)

---

### Lower Priority (Specialized)

#### 8. Extended Algebraic Geometry (≈20 functions)
**Source**: `src/sage/schemes/`, `src/sage/geometry/`

**Schemes**:
- General scheme arithmetic
- Morphisms between schemes
- Fiber products
- Blowups
- Normalization

**Varieties**:
- Dimension computation
- Degree computation
- Singular loci
- Resolution of singularities (2D)
- Intersection theory

**Toric Geometry**:
- Toric variety construction from fans
- Toric morphisms
- Toric divisors
- Cox ring

**Priority**: Low (very specialized)

---

#### 9. Modular Forms Infrastructure (≈15 functions)
**Source**: `src/sage/modular/`

**Modular Forms**:
- Modular forms space construction
- q-expansion
- Hecke operators
- Petersson inner product
- Atkin-Lehner involutions
- Newforms
- Oldforms

**Modular Symbols**:
- Modular symbols spaces
- Hecke operators on modular symbols
- L-values from modular symbols

**Priority**: Low (very specialized, needed for elliptic curves)

---

#### 10. Homological Algebra Extensions (≈10 functions)
**Source**: `src/sage/homology/`

**Current Status**: Basic chain complexes implemented

**Missing**:
- Spectral sequences
- Derived functors (Tor, Ext)
- Cohomology operations
- Cup products
- Steenrod algebra
- Adams spectral sequence
- Group cohomology
- Lie algebra cohomology

**Priority**: Low (very specialized)

---

## Summary by Priority

### High Priority (~50 functions)
1. Arbitrary precision arithmetic (MPFR/GMP) - 20 functions
2. Advanced symbolic integration - 15 functions
3. Complete Gröbner bases - 10 functions
4. Limits at infinity improvements - 1 function
5. Extended linear algebra - 15 functions

### Medium Priority (~60 functions)
1. Extended number theory - 20 functions
2. Additional graph algorithms - 15 functions
3. Numerical methods extensions - 10 functions
4. Advanced matrix decompositions - 3 functions
5. Polynomial factorization (Zassenhaus, LLL) - 4 functions
6. AES block cipher - 1 function
7. Various smaller features - 7 functions

### Low Priority (~62 functions)
1. Extended algebraic geometry - 20 functions
2. Modular forms infrastructure - 15 functions
3. Homological algebra extensions - 10 functions
4. Specialized number theory (theta series, local densities, etc.) - 5 functions
5. Advanced elliptic curve features (rigorous rank, L-functions, modular forms) - 3 functions
6. Puiseux series - 1 function
7. Voronoi diagrams - 1 function
8. Various specialized features - 7 functions

---

## Implementation Strategy

### Phase 1: High Priority Foundations
1. **Arbitrary Precision Arithmetic**
   - Integrate `rug` crate or MPFR/GMP
   - Update Real/Complex number types
   - Ensure all operations maintain precision

2. **Symbolic Integration**
   - Extend pattern matching
   - Add more integration rules
   - Implement reduction formulas

3. **Linear Algebra**
   - Complete SVD/QR variants
   - Add missing decompositions

### Phase 2: Medium Priority Extensions
1. Number theory extensions
2. Graph algorithms
3. Numerical methods
4. Polynomial factorization

### Phase 3: Specialized Features
1. Algebraic geometry
2. Modular forms
3. Advanced elliptic curves
4. Specialized homological algebra

---

## Notes

1. **Function Count**: The 172 missing functions include:
   - 9 explicitly marked incomplete (⬜ or 🚧)
   - ~30 implied by incomplete progress percentages
   - ~133 high-level features not yet decomposed into specific functions

2. **SageMath Scale**: SageMath has ~2 million lines of code across ~4,000 modules. This list focuses on core functionality that aligns with RustMath's goals.

3. **Pure Rust vs FFI**: Some features (especially arbitrary precision) require deciding between:
   - FFI to C libraries (MPFR, GMP) - faster, more mature
   - Pure Rust implementations - safer, more portable
   - Hybrid approach

4. **Testing**: All implemented functions should have comprehensive test coverage matching or exceeding SageMath's test suite.

5. **Documentation**: Each function should have:
   - Mathematical description
   - Usage examples
   - Performance characteristics
   - Comparison with SageMath behavior

---

## References

- **SageMath Documentation**: https://doc.sagemath.org/
- **SageMath Source**: https://github.com/sagemath/sage
- **SageMath Reference Manual**: https://doc.sagemath.org/html/en/reference/
- **MPFR**: https://www.mpfr.org/
- **GMP**: https://gmplib.org/
- **Rug Crate**: https://docs.rs/rug/
