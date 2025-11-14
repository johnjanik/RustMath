# RustMath Top Implementation Priorities

**Quick Start Guide for Next Development Phase**

This document highlights the most impactful partial features to implement next. These were selected based on:
1. Frequency of use in mathematical computing
2. Foundational nature (other features depend on them)
3. Current partial implementation makes completion easier

---

## Top 20 Priority Features

### 1. Special Functions (sage.functions)
**Impact: HIGH** - Used extensively in scientific computing

- [x] `gamma(x)` - Gamma function
- [x] `beta(x, y)` - Beta function
- [x] `zeta(s)` - Riemann zeta function
- [x] `bessel_J(n, x)` - Bessel functions of first kind
- [x] `bessel_Y(n, x)` - Bessel functions of second kind
- [x] `erf(x)` - Error function
- [x] `erfc(x)` - Complementary error function

**Current Status:** ✅ COMPLETED - rustmath-special-functions crate fully implemented
**Location:** rustmath-special-functions/src/{gamma.rs, beta.rs, zeta.rs, bessel.rs, error.rs}

---

### 2. Enhanced Symbolic Simplification (sage.symbolic)
**Impact: HIGH** - Core to computer algebra system

- [x] `simplify_full()` - Full recursive simplification
- [x] `simplify_trig()` - Trigonometric simplification
- [x] `simplify_rational()` - Rational expression simplification
- [x] `simplify_log()` - Logarithm simplification
- [x] `expand_trig()` - Expand trig identities
- [x] `collect()` - Collect like terms
- [x] `combine()` - Combine fractions/powers

**Current Status:** ✅ COMPLETED - All simplification methods implemented
**Location:** rustmath-symbolic/src/simplify.rs and rustmath-symbolic/src/expand.rs

---

### 3. Polynomial Operations (sage.rings.polynomial)
**Impact: HIGH** - Foundation for algebraic operations

- [x] `polynomial.gcd()` - Polynomial GCD
- [x] `polynomial.resultant()` - Resultant of two polynomials
- [x] `polynomial.discriminant()` - Polynomial discriminant
- [x] `polynomial.squarefree_decomposition()` - Squarefree factorization
- [x] `polynomial.compose()` - Polynomial composition
- [x] `polynomial.quo_rem()` - Quotient and remainder

**Current Status:** ✅ COMPLETED - All polynomial operations implemented
**Location:** rustmath-polynomials/src/univariate.rs

---

### 4. Number-Theoretic Functions (sage.arith)
**Impact: MEDIUM-HIGH** - Useful for many applications

- [x] `divisors(n)` - All divisors of n
- [x] `euler_phi(n)` - Euler's totient function
- [x] `moebius(n)` - Möbius function
- [x] `sigma(n, k)` - Sum of k-th powers of divisors
- [x] `bernoulli(n)` - Bernoulli numbers
- [x] `harmonic_number(n)` - Harmonic numbers
- [x] `is_square(n)` - Test if perfect square
- [x] `is_power(n, k)` - Test if perfect k-th power

**Current Status:** ✅ COMPLETED - All number-theoretic functions implemented
**Location:** rustmath-integers/src/integer.rs and rustmath-rationals/src/special_numbers.rs

---

### 5. Number Fields (sage.rings.number_field)
**Impact: HIGH** - Essential for algebraic number theory

- [x] `NumberField(polynomial, 'a')` - Create number field
- [x] `number_field.class_number()` - Class number
- [x] `number_field.unit_group()` - Unit group
- [x] `number_field.discriminant()` - Field discriminant
- [x] `number_field.galois_closure()` - Galois closure

**Current Status:** ✅ COMPLETED - All number field methods implemented
**Location:** rustmath-numberfields/src/lib.rs

---

### 6. Ideal Theory (sage.rings.ideal)
**Impact: MEDIUM** - Important for ring theory

- [x] `ideal(generators)` - Create ideal
- [x] `ideal.reduce(element)` - Reduce modulo ideal
- [x] `ideal.groebner_basis()` - Gröbner basis of ideal
- [x] `ideal.quotient(other)` - Ideal quotient
- [x] `ideal.intersection(other)` - Ideal intersection

**Current Status:** ✅ COMPLETED - All ideal operations implemented
**Location:** rustmath-polynomials/src/ideal.rs and rustmath-polynomials/src/groebner.rs

---

### 7. Advanced Integration (sage.calculus.integration)
**Impact: MEDIUM-HIGH** - CAS core functionality

- [x] `integrate(f, x)` - Symbolic integration (enhanced)
- [x] Pattern matching for common integrals
- [ ] Integration by parts
- [ ] Trigonometric substitution
- [ ] Partial fraction decomposition for integration

**Current Status:** ⚠️ PARTIALLY COMPLETED (2/5) - Basic integration with pattern matching
**Location:** rustmath-symbolic/src/integrate.rs
**Next Step:** Implement integration by parts, trig substitution, and partial fractions

---

### 8. Equation Solving (sage.symbolic.solve)
**Impact: HIGH** - Frequently requested feature

- [x] `solve([equations], [vars])` - System of equations
- [x] `solve_polynomial_system()` - Polynomial systems via Gröbner
- [x] `solve_linear_system()` - Linear systems (symbolic)
- [ ] `solve_trig_equation()` - Trigonometric equations
- [ ] `solve_inequality()` - Symbolic inequalities

**Current Status:** ⚠️ MOSTLY COMPLETED (3/5) - General, polynomial, and linear systems
**Location:** rustmath-symbolic/src/solve.rs
**Next Step:** Add specialized trig equation and inequality solvers

---

### 9. Limits and Asymptotic Expansion (sage.calculus.limits)
**Impact: MEDIUM** - Important calculus feature

- [x] `limit(f, x, a, dir='+')` - Directional limits (enhanced)
- [x] L'Hôpital's rule (multiple applications)
- [x] `series(f, x, a, n)` - Asymptotic series
- [x] Big-O notation support

**Current Status:** ✅ COMPLETED - All features implemented
**Location:** rustmath-symbolic/src/{limits.rs, series.rs}
**Details:**
- Unified `series()` function provides flexible Taylor/Maclaurin/asymptotic series expansion
- `series_with_big_o()` returns series with Big-O remainder term
- Full Big-O, Little-o, Theta, and Omega notation support
- `limit_with_error()` computes limits with error analysis
- Enhanced integration between limits and series modules

---

### 10. Matrix Normal Forms (sage.matrix)
**Impact: MEDIUM** - Linear algebra foundation

- [x] `matrix.echelon_form()` - Enhanced with pivoting options
- [x] `matrix.jordan_form()` - Complete implementation
- [x] `matrix.rational_canonical_form()` - Complete implementation
- [x] `matrix.elementary_divisors()` - Elementary divisors

**Current Status:** ✅ COMPLETED - All matrix normal forms implemented
**Location:** rustmath-matrix/src/{linear_solve.rs, eigenvalues.rs, companion.rs, integer_forms.rs}

---

## Implementation Roadmap

### Phase 1: Foundation (Months 1-2)
Focus on completing partial implementations:
1. Special functions (new crate)
2. Enhanced symbolic simplification
3. Polynomial operations (GCD, resultants)
4. Number-theoretic functions

### Phase 2: Algebra (Months 3-4)
Build algebraic structures:
1. Number fields (new crate)
2. Ideal theory
3. Enhanced Gröbner bases
4. Quotient rings

### Phase 3: Analysis (Months 5-6)
Complete calculus features:
1. Advanced integration
2. Enhanced equation solving
3. Better limit computation
4. Asymptotic analysis

---

## Module-by-Module Breakdown

### sage.functions → rustmath-special-functions (NEW)
**68 partial features total**
- Priority: Gamma, Beta, Zeta, Bessel, Error functions
- Estimated effort: 2-3 months
- Dependencies: rustmath-reals (MPFR for precision)

### sage.symbolic → rustmath-symbolic (ENHANCE)
**211 partial features total**
- Priority: Simplification, solving, assumptions
- Estimated effort: 4-6 months ongoing
- Dependencies: rustmath-polynomials, rustmath-functions

### sage.rings.polynomial → rustmath-polynomials (ENHANCE)
**~200+ partial features in sage.rings**
- Priority: GCD, resultants, composition
- Estimated effort: 2-3 months
- Dependencies: rustmath-integers, rustmath-finitefields

### sage.arith → rustmath-integers (ENHANCE)
**102 partial features total**
- Priority: Divisors, Euler phi, Möbius, sigma
- Estimated effort: 1-2 months
- Dependencies: None (foundation)

### sage.calculus → rustmath-symbolic (ENHANCE)
**98 partial features total**
- Priority: Integration, limits, differential equations
- Estimated effort: 3-4 months
- Dependencies: rustmath-symbolic

---

## Quick Wins (Can Implement in 1-2 Weeks Each)

1. ✅ **Divisors function** - COMPLETED (already existed in rustmath-integers)
2. ✅ **Euler's totient** - COMPLETED (already existed in rustmath-integers)
3. ✅ **Möbius function** - COMPLETED (already existed in rustmath-integers)
4. ✅ **Sigma function** - COMPLETED (generalized version added to rustmath-integers)
5. ✅ **Polynomial GCD** - COMPLETED (already existed in rustmath-polynomials)
6. ✅ **Polynomial composition** - COMPLETED (already existed in rustmath-polynomials)
7. ✅ **Bernoulli numbers** - COMPLETED (added to rustmath-rationals/special_numbers)
8. ✅ **Harmonic numbers** - COMPLETED (added to rustmath-rationals/special_numbers)
9. ✅ **is_square/is_power** - COMPLETED (is_perfect_square existed, is_power added)
10. ✅ **Matrix elementary divisors** - COMPLETED (added to rustmath-matrix/integer_forms)

---

## External Dependencies Needed

For full implementation of partial features, consider:

1. **MPFR/GMP** - Already used in rustmath-reals
2. **FLINT** (optional) - Polynomial arithmetic optimization
3. **ARB** (optional) - Arbitrary precision ball arithmetic
4. **PARI/GP** (optional) - Number theory functions

Currently RustMath uses pure Rust, which is excellent. Consider:
- Keeping pure Rust as default
- Optional feature flags for C library integration
- Performance benchmarks to justify any external deps

---

## Testing Strategy

For each new feature:

1. **Unit tests** - Basic functionality
2. **Property tests** - Invariants and properties
3. **Cross-validation** - Compare with SageMath output
4. **Performance tests** - Ensure reasonable performance
5. **Documentation** - Examples and use cases

---

## Community Engagement

To accelerate development:

1. **Mark "good first issue"** - Quick wins above
2. **Create detailed specs** - For each priority feature
3. **Provide examples** - Expected behavior from SageMath
4. **Mentoring** - Guide contributors through first PR
5. **Celebrate progress** - Update tracker regularly

---

*This document should be updated as features are completed and priorities shift.*
