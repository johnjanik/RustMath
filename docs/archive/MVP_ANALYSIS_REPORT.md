# RustMath MVP Analysis Report
## SageMath Core Module Analysis for Minimal Viable Product

**Date:** 2025-11-13
**Purpose:** Determine critical core modules needed for MVP to benchmark against SageMath
**Source:** Analysis of 13,852 SageMath entities across 51 top-level domains

---

## Executive Summary

Based on comprehensive analysis of the SageMath codebase tracker, RustMath has already implemented **approximately 60-70% of the core functionality** needed for an MVP. The remaining critical gaps are:

1. **Elementary mathematical functions** (sin, cos, exp, log, etc.)
2. **Expression parsing** (string to symbolic expression)
3. **Arbitrary precision real numbers** (replace f64 with MPFR equivalent)
4. **Symbolic integration** (only differentiation currently works)
5. **Enhanced symbolic simplification**

**Estimated time to MVP:** 11-17 weeks (3-4 months)

---

## SageMath Module Distribution

### Total Coverage
- **Total Entities Analyzed:** 13,852
- **Top-Level Domains:** 51
- **Core Domains for CAS:** 7-10

### Top 10 Domains by Size

| Rank | Domain | Entities | Classes | Functions | Modules |
|------|--------|----------|---------|-----------|---------|
| 1 | sage.combinat | 2,769 | 1,496 | 794 | 373 |
| 2 | sage.rings | 1,693 | 883 | 509 | 265 |
| 3 | sage.categories | 1,054 | 768 | 37 | 227 |
| 4 | sage.graphs | 750 | 50 | 606 | 78 |
| 5 | sage.schemes | 677 | 252 | 286 | 133 |
| 6 | sage.misc | 600 | 143 | 349 | 91 |
| 7 | sage.modular | 571 | 229 | 219 | 115 |
| 8 | sage.algebras | 518 | 292 | 102 | 109 |
| 9 | sage.interfaces | 356 | 147 | 154 | 52 |
| 10 | sage.groups | 347 | 170 | 99 | 77 |

---

## Priority Ranking for MVP

Based on weighted scoring (size, maturity, core functionality), the critical domains are:

| Rank | Domain | Priority Score | Entities | Status in RustMath |
|------|--------|----------------|----------|-------------------|
| 1 | sage.categories | 50.0 | 1,054 | ✓ Core traits implemented |
| 2 | sage.structure | 42.2 | 222 | ✓ Ring, Field, EuclideanDomain |
| 3 | sage.rings | 35.0 | 1,693 | ⚠ Mostly done, missing arbitrary precision reals |
| 4 | sage.modules | 27.7 | 266 | ✗ Not implemented |
| 5 | sage.matrix | 27.3 | 226 | ✓ Generic matrices, LU/PLU, determinant |
| 6 | sage.symbolic | 27.1 | 211 | ⚠ Partial - differentiation only |
| 7 | sage.functions | 26.8 | 177 | ✗ Not implemented |
| 8 | sage.arith | 26.0 | 102 | ✓ Basic arithmetic functions |

---

## Detailed Domain Analysis

### 1. sage.rings (1,693 entities) - CRITICAL

The most important domain for any computer algebra system.

#### Priority Sub-modules

| Sub-module | Entities | Description | RustMath Status |
|------------|----------|-------------|-----------------|
| **integer** | 8 | Basic integer arithmetic | ✓ rustmath-integers |
| **rational** | 9 | Rational numbers | ✓ rustmath-rationals |
| **polynomial** | 432 | Polynomial rings | ✓ rustmath-polynomials |
| **finite_rings** | 83 | Finite fields GF(p), GF(p^n) | ✓ rustmath-finitefields |
| **padics** | 195 | p-adic numbers | ✓ rustmath-padics |
| **real_mpfr** | 20 | Arbitrary precision reals | ✗ **CRITICAL GAP** |
| **real_double** | 6 | Double precision reals | ⚠ Using f64 |
| **complex_mpfr** | 11 | Arbitrary precision complex | ⚠ Need MPFR backend |
| **complex_double** | 7 | Double precision complex | ✓ rustmath-complex |
| **integer_ring** | 5 | Integer ring ZZ | ✓ In rustmath-integers |
| **rational_field** | 4 | Rational field QQ | ✓ In rustmath-rationals |

**Key Finding:** RustMath has excellent coverage of algebraic structures but lacks arbitrary precision real arithmetic.

#### Lower Priority (Specialized)
- number_field (200 entities) - Algebraic number fields
- function_field (160 entities) - Function fields
- asymptotic (79 entities) - Asymptotic expansions
- valuation (50 entities) - Valuation rings

---

### 2. sage.matrix (226 entities) - CRITICAL

Matrix operations are fundamental for benchmarking.

**RustMath Status:** ✓ **EXCELLENT**
- Generic matrix implementation over any Ring
- LU/PLU decomposition with pivoting
- Determinant calculation
- Gaussian elimination
- Matrix inversion

**Sub-modules:**
- matrix_integer_dense_hnf (26) - Hermite normal form
- special (26) - Special matrices
- benchmark (31) - **Ready for benchmarking!**

---

### 3. sage.symbolic (211 entities) - CRITICAL

Symbolic computation is a core CAS capability.

| Sub-module | Entities | RustMath Status |
|------------|----------|-----------------|
| **expression** | 67 | ⚠ Basic expression trees |
| **expression_conversions** | 36 | ✗ Not implemented |
| **integration** | 9 | ✗ **CRITICAL GAP** |
| **operators** | 5 | ⚠ Differentiation only |
| **constants** | 14 | ✗ Need mathematical constants |
| **assumptions** | 6 | ⚠ Basic support |
| **function** | 6 | ✗ **CRITICAL GAP** |

**Key Gaps:**
1. Symbolic integration (only differentiation works)
2. Expression parsing (cannot parse strings)
3. Mathematical constants (π, e, etc.)
4. Advanced simplification

---

### 4. sage.functions (177 entities) - CRITICAL GAP

Elementary mathematical functions are **completely missing** from RustMath.

**Function Categories:**

| Category | Examples | Count | Priority |
|----------|----------|-------|----------|
| Trigonometric | sin, cos, tan, arcsin, arccos, arctan | ~20 | **CRITICAL** |
| Exponential/Log | exp, log, ln, log10 | ~10 | **CRITICAL** |
| Hyperbolic | sinh, cosh, tanh, asinh, acosh | ~10 | **HIGH** |
| Special | Bessel, Gamma, Erf, Zeta | ~50 | MEDIUM |
| Power/Root | sqrt, cbrt, pow | ~5 | **CRITICAL** |
| Other | abs, sign, floor, ceiling | ~10 | HIGH |

**Recommendation:** Create `rustmath-functions` crate with trait-based design supporting:
- Symbolic representation in expression trees
- Numeric evaluation for various number types
- Integration with symbolic differentiation/integration

---

### 5. sage.modules (266 entities) - HIGH PRIORITY

Vector spaces and modules are important for linear algebra.

**RustMath Status:** ✗ **NOT IMPLEMENTED**

**Key Components:**
- free_module (24) - Free modules over rings
- with_basis (44) - Modules with basis
- free_quadratic_module (16) - Quadratic modules

**Recommendation:** Lower priority than elementary functions. Can defer to post-MVP.

---

### 6. sage.categories (1,054 entities) - FOUNDATION

Category theory framework for algebraic structures.

**RustMath Status:** ✓ **DONE via Rust traits**
- Ring, Field, EuclideanDomain traits implemented
- Generic programming enables same functionality
- No need for separate category framework

---

## Current RustMath Status

### ✓ COMPLETE (11 crates)

1. **rustmath-core** - Ring, Field, EuclideanDomain traits
2. **rustmath-integers** - Integer arithmetic, primality (Miller-Rabin), factorization (Pollard's Rho)
3. **rustmath-rationals** - Exact rational arithmetic
4. **rustmath-matrix** - Generic matrices, LU/PLU decomposition, determinant
5. **rustmath-polynomials** - Univariate/multivariate polynomials with factorization
6. **rustmath-finitefields** - GF(p) and GF(p^n) with Conway polynomials
7. **rustmath-padics** - p-adic numbers
8. **rustmath-complex** - Complex number arithmetic (double precision)
9. **rustmath-combinatorics** - Permutations, partitions, Young tableaux
10. **rustmath-graphs** - Graph algorithms (BFS/DFS, coloring, connectivity)
11. **rustmath-geometry** - Points, lines, polytopes

### ⚠ PARTIAL (2 crates)

1. **rustmath-symbolic** - Differentiation works, integration missing
2. **rustmath-reals** - Using f64, needs arbitrary precision (MPFR)

### ✗ MISSING (Critical Gaps)

1. **rustmath-functions** - Elementary functions (sin, cos, exp, log, sqrt)
2. **Expression parser** - Cannot parse "x^2 + 3*x + 2"
3. **Symbolic integration** - Only differentiation implemented
4. **Advanced simplification** - Current simplification is basic
5. **Modules/vector spaces** - Not implemented
6. **Number fields** - Algebraic numbers not supported

---

## MVP Implementation Roadmap

### MILESTONE 1: Elementary Functions (2-3 weeks)

**Goal:** Implement core mathematical functions for symbolic and numeric computation.

**Deliverables:**
- Create `rustmath-functions` crate
- Implement trait-based function framework
- Core functions:
  - Trigonometric: sin, cos, tan, arcsin, arccos, arctan, sec, csc, cot
  - Exponential: exp, exp2, exp10
  - Logarithmic: log, ln, log10, log2
  - Hyperbolic: sinh, cosh, tanh, asinh, acosh, atanh
  - Power/Root: sqrt, cbrt, pow
  - Other: abs, sign, floor, ceiling, round

**Implementation Strategy:**
```rust
// Trait for functions that can be evaluated symbolically or numerically
pub trait MathFunction<T> {
    fn eval(&self, x: T) -> T;
    fn to_symbolic(&self) -> Expression;
    fn derivative(&self) -> Box<dyn MathFunction<T>>;
}

// Example: Sine function
pub struct Sin;

impl MathFunction<f64> for Sin {
    fn eval(&self, x: f64) -> f64 { x.sin() }
    fn to_symbolic(&self) -> Expression { /* ... */ }
    fn derivative(&self) -> Box<dyn MathFunction<f64>> { Box::new(Cos) }
}
```

**Integration:**
- Add function nodes to symbolic expression trees
- Support function differentiation in rustmath-symbolic
- Enable numeric evaluation for all number types

---

### MILESTONE 2: Expression Parsing (1-2 weeks)

**Goal:** Parse mathematical expressions from strings.

**Deliverables:**
- Lexer/tokenizer for mathematical notation
- Recursive descent parser or parser combinator approach
- Support for:
  - Arithmetic operators: +, -, *, /, ^
  - Functions: sin(x), exp(x), log(x)
  - Variables and constants
  - Parentheses and operator precedence

**Example Usage:**
```rust
let expr = parse("x^2 + 3*x + 2").unwrap();
let result = expr.eval_at("x", 5); // 42

let expr2 = parse("sin(x) * exp(-x)").unwrap();
let derivative = expr2.diff("x");
```

**Libraries to Consider:**
- nom (parser combinator)
- pest (PEG parser)
- lalrpop (LR parser generator)

---

### MILESTONE 3: Arbitrary Precision Reals (2-3 weeks)

**Goal:** Replace f64 with arbitrary precision real numbers.

**Deliverables:**
- Integrate `rug` crate (Rust bindings for GMP/MPFR)
- Implement Ring/Field traits for MPFR reals
- Support configurable precision (default: 53 bits = f64 equivalent)
- Ensure compatibility with existing matrix/polynomial code

**Implementation:**
```rust
use rug::Float;

pub struct RealMPFR {
    value: Float,
    precision: u32, // bits of precision
}

impl Ring for RealMPFR { /* ... */ }
impl Field for RealMPFR { /* ... */ }
```

**Testing:**
- High-precision calculations (1000+ bits)
- Comparison with SageMath MPFR backend
- Performance benchmarks

---

### MILESTONE 4: Symbolic Integration (3-4 weeks)

**Goal:** Implement basic symbolic integration capabilities.

**Deliverables:**
- Power rule: ∫x^n dx = x^(n+1)/(n+1) + C
- Sum rule: ∫(f + g) dx = ∫f dx + ∫g dx
- Constant multiple: ∫k*f dx = k*∫f dx
- Integration by substitution (basic cases)
- Integration by parts
- Table lookup for standard integrals:
  - ∫sin(x) dx = -cos(x) + C
  - ∫cos(x) dx = sin(x) + C
  - ∫exp(x) dx = exp(x) + C
  - ∫1/x dx = log|x| + C
- Definite integral evaluation

**Example Usage:**
```rust
let expr = parse("x^2 + sin(x)").unwrap();
let integral = expr.integrate("x"); // x^3/3 - cos(x) + C

let definite = expr.integrate_definite("x", 0, PI);
```

---

### MILESTONE 5: Enhanced Simplification (2-3 weeks)

**Goal:** Improve symbolic expression simplification.

**Deliverables:**
- Algebraic simplification:
  - Combine like terms: x + 2x → 3x
  - Distribute: a(b + c) → ab + ac
  - Factor common terms: ab + ac → a(b + c)
- Trigonometric identities:
  - sin²(x) + cos²(x) → 1
  - sin(2x) → 2sin(x)cos(x)
  - cos(2x) → cos²(x) - sin²(x)
- Logarithm/exponential rules:
  - log(xy) → log(x) + log(y)
  - log(x^n) → n*log(x)
  - exp(x + y) → exp(x)*exp(y)
- Rational function simplification:
  - Cancel common factors in numerator/denominator
  - Partial fraction decomposition (basic)
- Expansion and factorization

**Example Usage:**
```rust
let expr = parse("sin(x)^2 + cos(x)^2").unwrap();
let simplified = expr.simplify(); // 1

let expr2 = parse("(x + 1)^2").unwrap();
let expanded = expr2.expand(); // x^2 + 2*x + 1
```

---

### MILESTONE 6: Benchmarking Suite (1-2 weeks)

**Goal:** Create comprehensive benchmarks comparing RustMath vs. SageMath.

**Benchmark Categories:**

#### 1. Integer Operations
- Large integer addition (10,000+ digits)
- Large integer multiplication
- Integer division with remainder
- GCD computation (Euclidean algorithm)
- Primality testing (Miller-Rabin, various bit sizes)
- Integer factorization (Pollard's Rho)

#### 2. Matrix Operations
- Matrix multiplication (100×100, 500×500, 1000×1000)
- Determinant calculation (various sizes)
- Matrix inversion
- LU decomposition
- PLU decomposition with pivoting
- Gaussian elimination

#### 3. Polynomial Operations
- Polynomial multiplication (univariate, various degrees)
- Polynomial division with remainder
- GCD of polynomials
- Polynomial factorization
- Multivariate polynomial operations

#### 4. Symbolic Computation
- Expression tree construction
- Symbolic differentiation (nested, complex expressions)
- Symbolic integration (standard integrals)
- Expression simplification
- Substitution and evaluation

#### 5. Finite Field Arithmetic
- Operations in GF(p) for various primes
- Operations in GF(p^n) for small n
- Polynomial arithmetic over finite fields

#### 6. Real Number Precision
- High-precision arithmetic (100, 1000, 10000 bits)
- Elementary function evaluation
- Convergent series computation

**Metrics to Track:**
- Execution time (microseconds/milliseconds)
- Memory usage
- Correctness (comparison with known values)
- Scalability (time vs. input size)

**Output Format:**
- Benchmark results in CSV/JSON
- Visualization graphs (time vs. size)
- Performance summary report

---

## Timeline Summary

| Milestone | Duration | Dependencies | Priority |
|-----------|----------|--------------|----------|
| 1. Elementary Functions | 2-3 weeks | None | **CRITICAL** |
| 2. Expression Parsing | 1-2 weeks | None | **CRITICAL** |
| 3. Arbitrary Precision Reals | 2-3 weeks | None | **HIGH** |
| 4. Symbolic Integration | 3-4 weeks | Milestones 1, 2 | **HIGH** |
| 5. Enhanced Simplification | 2-3 weeks | Milestones 1, 2 | **MEDIUM** |
| 6. Benchmarking Suite | 1-2 weeks | All above | **CRITICAL** |

**Total Estimated Time:** 11-17 weeks (3-4 months)

**Parallel Execution Opportunities:**
- Milestones 1, 2, 3 can be developed in parallel (weeks 1-3)
- Milestones 4 and 5 can overlap partially (weeks 4-9)
- Milestone 6 can begin once 1-4 are 80% complete (week 10+)

**Optimistic Timeline:** 11 weeks (2.5 months) with parallel development
**Realistic Timeline:** 15 weeks (3.5 months) accounting for testing/debugging
**Conservative Timeline:** 17 weeks (4 months) with buffer for unknowns

---

## Post-MVP Enhancements

After achieving MVP and benchmarking capability, consider:

### Phase 2: Advanced Algebra (3-6 months)
1. **Number Fields** - Algebraic numbers, minimal polynomials
2. **Modules and Vector Spaces** - Full abstraction for linear algebra
3. **Gröbner Bases** - Complete multivariate polynomial ideal algorithms
4. **Series Expansion** - Taylor/Laurent series, power series

### Phase 3: Analysis (3-6 months)
5. **Limits** - Symbolic limit computation
6. **Differential Equations** - Solve ODEs symbolically
7. **Multi-variable Calculus** - Partial derivatives, gradients, Jacobians
8. **Numerical Methods** - Root finding, optimization, integration

### Phase 4: Specialized Domains (6-12 months)
9. **Group Theory** - Permutation groups, group actions
10. **Combinatorics** - Advanced generating functions
11. **Graph Theory** - Advanced algorithms (matching, flows, planarity)
12. **Geometry** - Computational geometry, algebraic geometry basics

---

## Benchmarking Strategy

### Primary Comparison: SageMath

**Why SageMath?**
- Mature, comprehensive CAS (20+ years development)
- Python-based, similar design philosophy
- Excellent reference for correctness
- Well-documented performance characteristics

### Secondary Comparisons (Optional)
- **SymPy** - Pure Python symbolic math library
- **Mathematica** - Commercial CAS (if licensed copy available)
- **Maxima** - Open-source CAS

### Benchmark Metrics

1. **Performance:**
   - Raw speed (operations per second)
   - Scalability (time complexity)
   - Memory efficiency

2. **Correctness:**
   - Exact arithmetic accuracy
   - Symbolic manipulation correctness
   - Numerical precision

3. **Completeness:**
   - API coverage (% of SageMath features)
   - Edge case handling
   - Error handling

### Target Goals

**Performance Targets:**
- Integer operations: **2-10x faster** than SageMath (Rust vs. Python overhead)
- Matrix operations: **1-5x faster** (depends on BLAS backend)
- Polynomial operations: **1-3x faster**
- Symbolic operations: **0.5-2x** (SageMath uses highly optimized C++ backends)

**Correctness Target:**
- **100% agreement** with SageMath on test cases
- No arithmetic errors
- Proper edge case handling (division by zero, etc.)

**Completeness Target (MVP):**
- **~40% of SageMath core features**
- Focus on most commonly used operations
- Cover primary use cases for benchmarking

---

## Risk Assessment

### High Risk Items

1. **Arbitrary Precision Reals (Milestone 3)**
   - Risk: GMP/MPFR integration complexity
   - Mitigation: Use well-tested `rug` crate, extensive testing
   - Fallback: Start with `num-bigint` BigDecimal

2. **Symbolic Integration (Milestone 4)**
   - Risk: Complex algorithm, many edge cases
   - Mitigation: Start with table lookup + simple rules
   - Fallback: Defer advanced integration to post-MVP

3. **Simplification (Milestone 5)**
   - Risk: NP-hard problem, can't be perfect
   - Mitigation: Focus on common cases, use heuristics
   - Fallback: Basic simplification is acceptable for MVP

### Medium Risk Items

4. **Expression Parsing (Milestone 2)**
   - Risk: Operator precedence, ambiguity handling
   - Mitigation: Use existing parser libraries (nom, pest)
   - Fallback: Simple recursive descent parser

5. **Benchmarking Fairness (Milestone 6)**
   - Risk: Comparing Rust to Python isn't apples-to-apples
   - Mitigation: Benchmark against SageMath's C++ backends when possible
   - Clarify: Document what's being compared

### Low Risk Items

6. **Elementary Functions (Milestone 1)**
   - Risk: Low - well-defined mathematical functions
   - Standard libraries provide numeric implementations
   - Symbolic representation is straightforward

---

## Conclusion

RustMath is in an excellent position to achieve MVP status within 3-4 months. The project has already implemented most of the core algebraic structures and many specialized domains. The critical gaps are:

1. **Elementary functions** (highest priority)
2. **Expression parsing** (highest priority)
3. **Arbitrary precision reals** (high priority)
4. **Symbolic integration** (high priority)
5. **Enhanced simplification** (medium priority)

**Key Strengths:**
- ✓ Solid foundation with trait-based design
- ✓ Excellent coverage of algebraic structures
- ✓ Generic programming enables code reuse
- ✓ Zero unsafe code for memory safety

**Recommended Next Steps:**
1. Implement rustmath-functions crate (Milestone 1)
2. Add expression parser (Milestone 2)
3. Begin benchmarking infrastructure (Milestone 6 prep)
4. Integrate arbitrary precision reals (Milestone 3)
5. Implement symbolic integration (Milestone 4)
6. Enhance simplification (Milestone 5)
7. Complete comprehensive benchmarking (Milestone 6)

**Success Criteria:**
- RustMath can parse, manipulate, and evaluate mathematical expressions
- Performance meets or exceeds SageMath on integer/matrix/polynomial operations
- Symbolic differentiation and integration produce correct results
- Comprehensive test suite ensures correctness
- Benchmarking suite demonstrates viability

With focused effort on the identified gaps, RustMath can become a viable, high-performance alternative to SageMath for core mathematical computing tasks.

---

**Report Generated:** 2025-11-13
**Analysis Scripts:** `analyze_tracker.py`, `mvp_analysis.py`
**Data Source:** 14 split CSV files (13,852 SageMath entities)
