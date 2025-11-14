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

- [ ] `gamma(x)` - Gamma function
- [ ] `beta(x, y)` - Beta function
- [ ] `zeta(s)` - Riemann zeta function
- [ ] `bessel_J(n, x)` - Bessel functions of first kind
- [ ] `bessel_Y(n, x)` - Bessel functions of second kind
- [ ] `erf(x)` - Error function
- [ ] `erfc(x)` - Complementary error function

**Current Status:** rustmath-functions has elementary functions, but no special functions
**Next Step:** Create `rustmath-special-functions` crate

---

### 2. Enhanced Symbolic Simplification (sage.symbolic)
**Impact: HIGH** - Core to computer algebra system

- [ ] `simplify_full()` - Full recursive simplification
- [ ] `simplify_trig()` - Trigonometric simplification
- [ ] `simplify_rational()` - Rational expression simplification
- [ ] `simplify_log()` - Logarithm simplification
- [ ] `expand_trig()` - Expand trig identities
- [ ] `collect()` - Collect like terms
- [ ] `combine()` - Combine fractions/powers

**Current Status:** rustmath-symbolic has basic simplification
**Next Step:** Enhance `rustmath-symbolic/src/simplify.rs` with pattern matching

---

### 3. Polynomial Operations (sage.rings.polynomial)
**Impact: HIGH** - Foundation for algebraic operations

- [ ] `polynomial.gcd()` - Polynomial GCD
- [ ] `polynomial.resultant()` - Resultant of two polynomials
- [ ] `polynomial.discriminant()` - Polynomial discriminant
- [ ] `polynomial.squarefree_decomposition()` - Squarefree factorization
- [ ] `polynomial.compose()` - Polynomial composition
- [ ] `polynomial.quo_rem()` - Quotient and remainder

**Current Status:** rustmath-polynomials has basic operations and factorization
**Next Step:** Add to `rustmath-polynomials/src/polynomial.rs`

---

### 4. Number-Theoretic Functions (sage.arith)
**Impact: MEDIUM-HIGH** - Useful for many applications

- [ ] `divisors(n)` - All divisors of n
- [ ] `euler_phi(n)` - Euler's totient function
- [ ] `moebius(n)` - Möbius function
- [ ] `sigma(n, k)` - Sum of k-th powers of divisors
- [ ] `bernoulli(n)` - Bernoulli numbers
- [ ] `harmonic_number(n)` - Harmonic numbers
- [ ] `is_square(n)` - Test if perfect square
- [ ] `is_power(n, k)` - Test if perfect k-th power

**Current Status:** rustmath-integers has basic number theory (GCD, factorization)
**Next Step:** Extend `rustmath-integers/src/integer.rs`

---

### 5. Number Fields (sage.rings.number_field)
**Impact: HIGH** - Essential for algebraic number theory

- [ ] `NumberField(polynomial, 'a')` - Create number field
- [ ] `number_field.class_number()` - Class number
- [ ] `number_field.unit_group()` - Unit group
- [ ] `number_field.discriminant()` - Field discriminant
- [ ] `number_field.galois_closure()` - Galois closure

**Current Status:** Not implemented
**Next Step:** Create new `rustmath-numberfields` crate

---

### 6. Ideal Theory (sage.rings.ideal)
**Impact: MEDIUM** - Important for ring theory

- [ ] `ideal(generators)` - Create ideal
- [ ] `ideal.reduce(element)` - Reduce modulo ideal
- [ ] `ideal.groebner_basis()` - Gröbner basis of ideal
- [ ] `ideal.quotient(other)` - Ideal quotient
- [ ] `ideal.intersection(other)` - Ideal intersection

**Current Status:** Basic Gröbner bases in rustmath-polynomials
**Next Step:** Add ideal module to `rustmath-polynomials`

---

### 7. Advanced Integration (sage.calculus.integration)
**Impact: MEDIUM-HIGH** - CAS core functionality

- [ ] `integrate(f, x)` - Symbolic integration (enhanced)
- [ ] Pattern matching for common integrals
- [ ] Integration by parts
- [ ] Trigonometric substitution
- [ ] Partial fraction decomposition for integration

**Current Status:** rustmath-symbolic has basic integration
**Next Step:** Enhance `rustmath-symbolic/src/integrate.rs`

---

### 8. Equation Solving (sage.symbolic.solve)
**Impact: HIGH** - Frequently requested feature

- [ ] `solve([equations], [vars])` - System of equations
- [ ] `solve_polynomial_system()` - Polynomial systems via Gröbner
- [ ] `solve_linear_system()` - Linear systems (symbolic)
- [ ] `solve_trig_equation()` - Trigonometric equations
- [ ] `solve_inequality()` - Symbolic inequalities

**Current Status:** rustmath-symbolic has basic solve
**Next Step:** Enhance `rustmath-symbolic/src/solve.rs`

---

### 9. Limits and Asymptotic Expansion (sage.calculus.limits)
**Impact: MEDIUM** - Important calculus feature

- [ ] `limit(f, x, a, dir='+')` - Directional limits (enhanced)
- [ ] L'Hôpital's rule (multiple applications)
- [ ] `series(f, x, a, n)` - Asymptotic series
- [ ] Big-O notation support

**Current Status:** rustmath-symbolic has basic limits with L'Hôpital
**Next Step:** Enhance `rustmath-symbolic/src/limits.rs`

---

### 10. Matrix Normal Forms (sage.matrix)
**Impact: MEDIUM** - Linear algebra foundation

- [ ] `matrix.echelon_form()` - Enhanced with pivoting options
- [ ] `matrix.jordan_form()` - Complete implementation
- [ ] `matrix.rational_canonical_form()` - Complete implementation
- [ ] `matrix.elementary_divisors()` - Elementary divisors

**Current Status:** rustmath-matrix has basic normal forms
**Next Step:** Complete `rustmath-matrix/src/integer_forms.rs`

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

1. **Divisors function** - Use existing factorization
2. **Euler's totient** - Simple formula from prime factorization
3. **Möbius function** - Simple from factorization
4. **Sigma function** - Sum of divisor powers
5. **Polynomial GCD** - Extend existing GCD to polynomials
6. **Polynomial composition** - Simple recursion
7. **Bernoulli numbers** - Classic algorithm
8. **Harmonic numbers** - Simple sum
9. **is_square/is_power** - Integer testing
10. **Matrix elementary divisors** - Use Smith normal form

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
