# RustMath Implementation Checklist

This document tracks the implementation of SageMath functionality in RustMath.
Based on SageMath documentation: https://doc.sagemath.org/html/en/reference/index.html
and source code: https://github.com/sagemath/sage/tree/develop/src/sage

**Legend**:
- ✅ = Fully implemented and tested
- 🚧 = Partially implemented
- ⬜ = Not yet implemented
- 🔍 = Under investigation/planning

**Overall Progress**: ~35% (161 / 473 functions tracked)

---

## 1. Basic Rings and Fields

### 1.1 Integers (sage.rings.integer)
**SageMath Source**: `src/sage/rings/integer.pyx`, `integer_ring.py`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `Integer(n)` - Create integer | ✅ | `rustmath-integers/src/integer.rs` | Via `Integer::from()` |
| `ZZ` - Integer ring | ✅ | `rustmath-integers` | Type-level |
| `.abs()` - Absolute value | ✅ | `rustmath-integers/src/integer.rs` | Via `BigInt::abs()` |
| `.bits()` - Number of bits | ✅ | `rustmath-integers/src/integer.rs` | Alias for bit_length() |
| `.sqrt()` - Integer square root | ✅ | `rustmath-integers/src/integer.rs` | Newton's method |
| `.is_prime()` - Primality test | ✅ | `rustmath-integers/src/prime.rs` | Miller-Rabin |
| `.is_pseudoprime()` | ✅ | `rustmath-integers/src/prime.rs` | Fermat test |
| `.next_prime()` - Next prime | ✅ | `rustmath-integers/src/prime.rs` | |
| `.previous_prime()` | ✅ | `rustmath-integers/src/prime.rs` | |
| `.prime_divisors()` | 🚧 | `rustmath-integers/src/prime.rs` | Basic factorization |
| `.factor()` - Prime factorization | ✅ | `rustmath-integers/src/prime.rs` | Trial division + Pollard's Rho |
| `.divisors()` | ✅ | `rustmath-integers/src/integer.rs` | From prime factorization |
| `.gcd(b)` - Greatest common divisor | ✅ | `rustmath-integers/src/integer.rs` | Euclidean algorithm |
| `.lcm(b)` - Least common multiple | ✅ | `rustmath-integers/src/integer.rs` | |
| `.xgcd(b)` - Extended GCD | ✅ | `rustmath-integers/src/integer.rs` | Returns (gcd, s, t) |
| `.mod_inverse(n)` | ✅ | `rustmath-integers/src/modular.rs` | Via extended GCD |
| `.powermod(e, m)` | ✅ | `rustmath-integers/src/integer.rs` | `mod_pow()` |
| `.kronecker(b)` | ✅ | `rustmath-integers/src/integer.rs` | `jacobi_symbol()` |
| `.factorial()` | ✅ | `rustmath-combinatorics/src/lib.rs` | |
| `.binomial(k)` | ✅ | `rustmath-combinatorics/src/lib.rs` | |
| `.digits(base)` | ✅ | `rustmath-integers/src/integer.rs` | Base 2-36 |
| `.nth_root(n)` | ✅ | `rustmath-integers/src/integer.rs` | Newton's method |
| `.valuation(p)` | ✅ | `rustmath-integers/src/integer.rs` | p-adic valuation |

**Progress**: 21/26 functions (81%)

### 1.2 Rational Numbers (sage.rings.rational)
**SageMath Source**: `src/sage/rings/rational.pyx`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `Rational(a, b)` - Create rational | ✅ | `rustmath-rationals/src/rational.rs` | |
| `QQ` - Rational field | ✅ | `rustmath-rationals` | Type-level |
| `.numerator()` | ✅ | `rustmath-rationals/src/rational.rs` | |
| `.denominator()` | ✅ | `rustmath-rationals/src/rational.rs` | |
| `.floor()` | ✅ | `rustmath-rationals/src/rational.rs` | |
| `.ceil()` | ✅ | `rustmath-rationals/src/rational.rs` | |
| `.round()` | ✅ | `rustmath-rationals/src/rational.rs` | Round half up |
| `.continued_fraction()` | ✅ | `rustmath-rationals/src/continued_fraction.rs` | |
| `.convergents()` | ✅ | `rustmath-rationals/src/continued_fraction.rs` | |
| `.n(digits)` - Numerical approx | ✅ | `rustmath-rationals/src/rational.rs` | `to_f64()` |
| `.valuation(p)` | ✅ | `rustmath-rationals/src/rational.rs` | p-adic valuation |
| `.norm()` | ✅ | `rustmath-rationals/src/rational.rs` | Absolute value |

**Progress**: 11/12 functions (92%)

### 1.3 Real Numbers (sage.rings.real_mpfr)
**SageMath Source**: `src/sage/rings/real_mpfr.pyx`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `RealField(prec)` | ⬜ | | Arbitrary precision reals |
| `RR` - Real field | ⬜ | | |
| `.sin()`, `.cos()`, `.tan()` | ⬜ | | |
| `.exp()`, `.log()` | ⬜ | | |
| `.sqrt()` | ⬜ | | |
| Rounding modes | ⬜ | | |
| Interval arithmetic | ⬜ | | |

**Progress**: 0/7 features (0%)

### 1.4 Complex Numbers (sage.rings.complex_mpfr)
**SageMath Source**: `src/sage/rings/complex_mpfr.pyx`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `ComplexField(prec)` | ⬜ | | |
| `CC` - Complex field | ⬜ | | |
| `.real()`, `.imag()` | ⬜ | | |
| `.abs()`, `.arg()` | ⬜ | | |
| `.conjugate()` | ⬜ | | |
| Complex arithmetic | ⬜ | | |

**Progress**: 0/6 features (0%)

### 1.5 Polynomial Rings (sage.rings.polynomial)
**SageMath Source**: `src/sage/rings/polynomial/`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `PolynomialRing(R, 'x')` | ✅ | `rustmath-polynomials` | Generic over rings |
| Univariate polynomials | ✅ | `rustmath-polynomials/src/univariate.rs` | |
| Multivariate polynomials | ✅ | `rustmath-polynomials/src/multivariate.rs` | Sparse representation |
| `.degree()` | ✅ | | |
| `.coefficients()` | ✅ | | |
| `.roots()` | 🚧 | `rustmath-polynomials/src/roots.rs` | Rational roots, quadratic formula |
| `.factor()` - Factorization | 🚧 | `rustmath-polynomials/src/factorization.rs` | Square-free only |
| `.gcd()` - Polynomial GCD | 🚧 | `rustmath-polynomials/src/univariate.rs` | Limited to field coefficients |
| `.lcm()` | ⬜ | | |
| `.derivative()` | ✅ | `rustmath-polynomials/src/univariate.rs` | |
| `.integral()` | ✅ | `rustmath-polynomials/src/univariate.rs` | `integrate()` |
| `.resultant()` | ⬜ | | |
| `.discriminant()` | ✅ | `rustmath-polynomials/src/univariate.rs` | |
| `.sylvester_matrix()` | ⬜ | | |
| `.quo_rem(g)` - Quotient/remainder | ✅ | `rustmath-polynomials/src/univariate.rs` | `div_rem()` |
| Gröbner bases | ⬜ | | |
| `.is_irreducible()` | ⬜ | | |
| `.is_square_free()` | ✅ | `rustmath-polynomials/src/factorization.rs` | |
| `.content()` | ✅ | `rustmath-polynomials/src/factorization.rs` | |
| `.primitive_part()` | ✅ | `rustmath-polynomials/src/factorization.rs` | |

**Progress**: 12/20 features (60%)

### 1.6 Power Series (sage.rings.power_series_ring)
**SageMath Source**: `src/sage/rings/power_series_ring.py`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `PowerSeriesRing(R, 'x')` | ⬜ | | |
| Truncated series | ⬜ | | |
| `.exp()`, `.log()` | ⬜ | | |
| `.inverse()` | ⬜ | | |
| Composition | ⬜ | | |

**Progress**: 0/5 features (0%)

### 1.7 Finite Fields (sage.rings.finite_rings)
**SageMath Source**: `src/sage/rings/finite_rings/`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `GF(p)` - Prime field | ⬜ | | |
| `GF(p^n)` - Extension field | ⬜ | | |
| Conway polynomials | ⬜ | | |
| Discrete logarithm | ⬜ | | |
| Frobenius endomorphism | ⬜ | | |

**Progress**: 0/5 features (0%)

### 1.8 p-adic Numbers (sage.rings.padics)
**SageMath Source**: `src/sage/rings/padics/`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `Qp(p)` - p-adic field | ⬜ | | |
| `Zp(p)` - p-adic ring | ⬜ | | |
| Hensel lifting | ⬜ | | |
| `.valuation()` | ⬜ | | |

**Progress**: 0/4 features (0%)

---

## 2. Linear Algebra (sage.matrix, sage.modules)

### 2.1 Matrices (sage.matrix.matrix)
**SageMath Source**: `src/sage/matrix/`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `matrix(R, nrows, ncols, data)` | ✅ | `rustmath-matrix/src/matrix.rs` | `Matrix::from_vec()` |
| `identity_matrix(n)` | ✅ | `rustmath-matrix/src/matrix.rs` | |
| `zero_matrix(n, m)` | ✅ | `rustmath-matrix/src/matrix.rs` | |
| `.nrows()`, `.ncols()` | ✅ | | |
| `.rank()` | ✅ | `rustmath-matrix/src/linear_solve.rs` | |
| `.det()` - Determinant | ✅ | `rustmath-matrix/src/matrix.rs` | Two algorithms |
| `.trace()` | ✅ | `rustmath-matrix/src/matrix.rs` | |
| `.transpose()` | ✅ | `rustmath-matrix/src/matrix.rs` | |
| `.inverse()` | ✅ | `rustmath-matrix/src/linear_solve.rs` | Gauss-Jordan |
| `.solve_right(b)` | ✅ | `rustmath-matrix/src/linear_solve.rs` | |
| `.solve_left(b)` | ✅ | `rustmath-matrix/src/linear_solve.rs` | Solves xA = b |
| `.kernel()` - Null space | ✅ | `rustmath-matrix/src/linear_solve.rs` | Basis from RREF |
| `.image()` - Column space | ✅ | `rustmath-matrix/src/linear_solve.rs` | Pivot columns |
| `.eigenvalues()` | ✅ | `rustmath-matrix/src/eigenvalues.rs` | QR algorithm |
| `.eigenvectors_right()` | ✅ | `rustmath-matrix/src/eigenvalues.rs` | Via kernel of (A - λI) |
| `.eigenvectors_left()` | ✅ | `rustmath-matrix/src/eigenvalues.rs` | Right eigenvectors of A^T |
| `.charpoly()` - Characteristic polynomial | ✅ | `rustmath-matrix/src/polynomial_ops.rs` | Faddeev-LeVerrier algorithm |
| `.minpoly()` - Minimal polynomial | ✅ | `rustmath-matrix/src/polynomial_ops.rs` | Full implementation via nullity analysis |
| `.jordan_form()` | ✅ | `rustmath-matrix/src/eigenvalues.rs` | Works for diagonalizable matrices |
| `.rational_canonical_form()` | ⬜ | | Complex - requires companion matrices |
| `.smith_form()` | ✅ | `rustmath-matrix/src/integer_forms.rs` | Smith normal form for integer matrices |
| `.hermite_form()` | ✅ | `rustmath-matrix/src/integer_forms.rs` | Hermite normal form |
| `.echelon_form()` | ✅ | `rustmath-matrix/src/linear_solve.rs` | `row_echelon_form()` |
| `.rref()` | ✅ | `rustmath-matrix/src/linear_solve.rs` | `reduced_row_echelon_form()` |
| `.LU()` | ✅ | `rustmath-matrix/src/decomposition.rs` | |
| `.QR()` | ✅ | `rustmath-matrix/src/decomposition.rs` | Gram-Schmidt |
| `.SVD()` | ✅ | `rustmath-matrix/src/decomposition.rs` | Via eigendecomposition of A^T A |
| `.cholesky()` | ✅ | `rustmath-matrix/src/decomposition.rs` | For positive definite |
| `.hessenberg_form()` | ✅ | `rustmath-matrix/src/decomposition.rs` | Householder reflections |
| `.is_symmetric()` | ✅ | `rustmath-matrix/src/matrix.rs` | Also: is_diagonal, is_triangular |
| `.is_hermitian()` | ✅ | `rustmath-matrix/src/matrix.rs` | Same as symmetric for reals |
| `.is_positive_definite()` | ✅ | `rustmath-matrix/src/matrix.rs` | Sylvester's criterion |
| `.norm(p)` | ✅ | `rustmath-matrix/src/matrix.rs` | Frobenius, infinity, one norms |
| `.condition_number()` | ✅ | `rustmath-matrix/src/matrix.rs` | Using infinity norm |
| `.pseudoinverse()` | ✅ | `rustmath-matrix/src/linear_solve.rs` | Moore-Penrose via normal equations |
| Sparse matrices | ✅ | `rustmath-matrix/src/sparse.rs` | CSR format with basic operations |

**Progress**: 31/35 features (89%)

### 2.2 Vectors (sage.modules.free_module)
**SageMath Source**: `src/sage/modules/free_module.py`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `vector(R, values)` | ✅ | `rustmath-matrix/src/vector.rs` | |
| `.dot_product()` | ✅ | `rustmath-matrix/src/vector.rs` | |
| `.cross_product()` | ✅ | `rustmath-matrix/src/vector.rs` | 3D only |
| `.norm(p)` | ✅ | `rustmath-matrix/src/vector.rs` | p-norms (1, 2, inf, general) |
| `.normalize()` | ✅ | `rustmath-matrix/src/vector.rs` | Unit vector |
| Inner product spaces | ✅ | `rustmath-matrix/src/inner_product.rs` | With Gram matrix, Gram-Schmidt |

**Progress**: 6/6 features (100%)

### 2.3 Vector Spaces (sage.modules.vector_space)
**SageMath Source**: `src/sage/modules/vector_space.py`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `VectorSpace(F, n)` | ✅ | `rustmath-matrix/src/vector_space.rs` | With standard or custom basis |
| `.basis()` | ✅ | `rustmath-matrix/src/vector_space.rs` | |
| `.dimension()` | ✅ | `rustmath-matrix/src/vector_space.rs` | |
| Direct sums | ✅ | `rustmath-matrix/src/vector_space.rs` | `direct_sum()` |
| Quotient spaces | ✅ | `rustmath-matrix/src/vector_space.rs` | `QuotientSpace` with projection |

**Progress**: 5/5 features (100%)

---

## 3. Number Theory (sage.rings.number_theory)

### 3.1 Prime Numbers (sage.rings.arith)
**SageMath Source**: `src/sage/rings/arith.py`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `is_prime(n)` | ✅ | `rustmath-integers/src/prime.rs` | Miller-Rabin |
| `is_pseudoprime(n)` | ⬜ | | |
| `is_prime_power(n)` | ⬜ | | |
| `next_prime(n)` | ✅ | `rustmath-integers/src/prime.rs` | |
| `previous_prime(n)` | ✅ | `rustmath-integers/src/prime.rs` | |
| `nth_prime(n)` | ⬜ | | |
| `prime_range(start, stop)` | ⬜ | | Sieve of Eratosthenes |
| `primes_first_n(n)` | ⬜ | | |
| `prime_pi(x)` | ⬜ | | Prime counting |
| `random_prime(a, b)` | ⬜ | | |

**Progress**: 3/10 functions (30%)

### 3.2 Factorization (sage.rings.factorint)
**SageMath Source**: `src/sage/rings/factorint.pyx`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `factor(n)` | ✅ | `rustmath-integers/src/prime.rs` | Trial + Pollard's Rho |
| Trial division | ✅ | `rustmath-integers/src/prime.rs` | |
| Pollard's rho | ✅ | `rustmath-integers/src/prime.rs` | |
| Pollard's p-1 | ⬜ | | |
| ECM (Elliptic Curve Method) | ⬜ | | |
| Quadratic sieve | ⬜ | | |
| `.divisors()` | ✅ | `rustmath-integers/src/integer.rs` | From prime factorization |
| `.number_of_divisors()` | ✅ | `rustmath-integers/src/integer.rs` | tau(n) - `num_divisors()` |
| `.sum_of_divisors()` | ✅ | `rustmath-integers/src/integer.rs` | sigma(n) - `sum_divisors()` |
| `.euler_phi()` | ✅ | `rustmath-integers/src/integer.rs` | Totient function |
| `.moebius()` | ✅ | `rustmath-integers/src/integer.rs` | Möbius function μ(n) |

**Progress**: 8/11 functions (73%)

### 3.3 Modular Arithmetic (sage.rings.finite_rings.integer_mod)
**SageMath Source**: `src/sage/rings/finite_rings/integer_mod.pyx`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `Mod(a, n)` | ✅ | `rustmath-integers/src/modular.rs` | `ModularInteger` |
| `.inverse()` | ✅ | `rustmath-integers/src/modular.rs` | Via extended GCD |
| `.is_unit()` | ⬜ | | |
| `.multiplicative_order()` | ⬜ | | |
| Primitive roots | ⬜ | | |
| Chinese Remainder Theorem | ✅ | `rustmath-integers/src/crt.rs` | |
| Quadratic residues | ⬜ | | |
| Legendre/Jacobi symbols | ✅ | `rustmath-integers/src/integer.rs` | `legendre_symbol()`, `jacobi_symbol()` |

**Progress**: 4/8 functions (50%)

### 3.4 Continued Fractions (sage.rings.continued_fraction)
**SageMath Source**: `src/sage/rings/continued_fraction.py`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `continued_fraction(x)` | ✅ | `rustmath-rationals/src/continued_fraction.rs` | |
| `.convergents()` | ✅ | `rustmath-rationals/src/continued_fraction.rs` | |
| `.value()` | ✅ | `rustmath-rationals/src/continued_fraction.rs` | `to_rational()` |
| Periodic continued fractions | ⬜ | | For quadratic irrationals |
| `.quotients()` | ✅ | `rustmath-rationals/src/continued_fraction.rs` | |

**Progress**: 4/5 functions (80%)

### 3.5 Quadratic Forms (sage.quadratic_forms)
**SageMath Source**: `src/sage/quadratic_forms/`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `QuadraticForm(Q)` | ⬜ | | |
| Representation theory | ⬜ | | |
| Theta series | ⬜ | | |
| Local densities | ⬜ | | |

**Progress**: 0/4 features (0%)

---

## 4. Symbolic Computation (sage.symbolic)

### 4.1 Symbolic Expressions (sage.symbolic.expression)
**SageMath Source**: `src/sage/symbolic/expression.pyx`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `var('x')` | ✅ | `rustmath-symbolic/src/expression.rs` | `Expr::symbol()` |
| Symbolic ring `SR` | ✅ | `rustmath-symbolic` | Type-level |
| Basic arithmetic (+, -, *, /) | ✅ | `rustmath-symbolic/src/expression.rs` | |
| Power `x^n` | ✅ | `rustmath-symbolic/src/expression.rs` | |
| `.subs(x=value)` | ✅ | `rustmath-symbolic/src/substitute.rs` | `substitute()` |
| `.substitute({x:val})` | ✅ | `rustmath-symbolic/src/substitute.rs` | `substitute_many()` |
| `.n()` - Numerical eval | ✅ | `rustmath-symbolic/src/substitute.rs` | `eval_float()` |
| `.expand()` | 🚧 | | Limited |
| `.factor()` | ⬜ | | |
| `.simplify()` | 🚧 | `rustmath-symbolic/src/simplify.rs` | Basic |
| `.simplify_full()` | ⬜ | | |
| `.simplify_rational()` | ⬜ | | |
| `.simplify_trig()` | ⬜ | | |
| `.canonicalize_radical()` | ⬜ | | |
| `.collect(x)` | ⬜ | | Collect terms |
| `.coefficient(x, n)` | ⬜ | | |
| `.degree(x)` | ⬜ | | |
| `.variables()` | ✅ | `rustmath-symbolic/src/substitute.rs` | `symbols()` |
| `.is_polynomial(x)` | ⬜ | | |
| `.is_rational_expression()` | ⬜ | | |

**Progress**: 8/20 functions (40%)

### 4.2 Functions (sage.symbolic.function)
**SageMath Source**: `src/sage/symbolic/function.pyx`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `sin(x)`, `cos(x)`, `tan(x)` | ✅ | `rustmath-symbolic/src/expression.rs` | |
| `exp(x)`, `log(x)` | ✅ | `rustmath-symbolic/src/expression.rs` | |
| `sqrt(x)` | ✅ | `rustmath-symbolic/src/expression.rs` | |
| `abs(x)` | ⬜ | | |
| `sign(x)` | ⬜ | | |
| Hyperbolic functions | ⬜ | | sinh, cosh, tanh |
| Inverse trig functions | ⬜ | | arcsin, arccos, etc. |
| Bessel functions | ⬜ | | |
| Gamma function | ⬜ | | |
| Zeta function | ⬜ | | |
| Custom functions | ⬜ | | |

**Progress**: 3/11 features (27%)

### 4.3 Assumptions (sage.symbolic.assumptions)
**SageMath Source**: `src/sage/symbolic/assumptions.py`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `assume(x > 0)` | ⬜ | | |
| `.is_positive()` | ⬜ | | |
| `.is_real()` | ⬜ | | |
| `.is_integer()` | ⬜ | | |

**Progress**: 0/4 functions (0%)

---

## 5. Calculus (sage.calculus)

### 5.1 Differentiation (sage.calculus.calculus)
**SageMath Source**: `src/sage/calculus/calculus.py`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `diff(f, x)` | ✅ | `rustmath-symbolic/src/differentiate.rs` | `differentiate()` |
| Partial derivatives | ✅ | `rustmath-symbolic/src/differentiate.rs` | Via `gradient()` |
| Chain rule | ✅ | `rustmath-symbolic/src/differentiate.rs` | Automatic |
| Product rule | ✅ | `rustmath-symbolic/src/differentiate.rs` | Automatic |
| Quotient rule | ✅ | `rustmath-symbolic/src/differentiate.rs` | Automatic |
| Implicit differentiation | ⬜ | | |
| Higher-order derivatives | ✅ | `rustmath-symbolic/src/differentiate.rs` | `nth_derivative()` |
| `.derivative(x, n)` | ✅ | `rustmath-symbolic/src/differentiate.rs` | `nth_derivative()` |
| Jacobian matrix | ⬜ | | |
| Hessian matrix | ⬜ | | |

**Progress**: 7/10 functions (70%)

### 5.2 Integration (sage.calculus.integration)
**SageMath Source**: `src/sage/symbolic/integration/`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `integrate(f, x)` | ⬜ | | Indefinite |
| `integrate(f, (x, a, b))` | ⬜ | | Definite |
| Numerical integration | ⬜ | | |
| Multiple integrals | ⬜ | | |
| Symbolic integration | ⬜ | | Risch algorithm |
| Integration by parts | ⬜ | | |
| Substitution | ⬜ | | |

**Progress**: 0/7 functions (0%)

### 5.3 Limits (sage.calculus.calculus)
**SageMath Source**: `src/sage/calculus/calculus.py`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `limit(f, x=a)` | ⬜ | | |
| `.limit(x=a, dir='+')` | ⬜ | | One-sided |
| L'Hôpital's rule | ⬜ | | |
| Limits at infinity | ⬜ | | |

**Progress**: 0/4 functions (0%)

### 5.4 Series Expansions (sage.calculus.calculus)
**SageMath Source**: `src/sage/calculus/calculus.py`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `taylor(f, x, a, n)` | ⬜ | | Taylor series |
| `.series(x, n)` | ⬜ | | |
| Laurent series | ⬜ | | |
| Asymptotic expansions | ⬜ | | |
| Puiseux series | ⬜ | | |

**Progress**: 0/5 functions (0%)

### 5.5 Differential Equations (sage.calculus.desolvers)
**SageMath Source**: `src/sage/calculus/desolvers.py`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `desolve(de, dvar)` | ⬜ | | Symbolic ODE solver |
| `.solve_ode()` | ⬜ | | |
| Numerical ODE solvers | ⬜ | | Runge-Kutta, etc. |
| PDEs | ⬜ | | |

**Progress**: 0/4 functions (0%)

---

## 6. Combinatorics (sage.combinat)

### 6.1 Permutations (sage.combinat.permutation)
**SageMath Source**: `src/sage/combinat/permutation.py`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `Permutation([...])` | ✅ | `rustmath-combinatorics/src/permutations.rs` | |
| `.cycles()` | ✅ | `rustmath-combinatorics/src/permutations.rs` | |
| `.sign()` | ✅ | `rustmath-combinatorics/src/permutations.rs` | Even/odd |
| `.order()` | ⬜ | | Multiplicative order |
| `.inverse()` | ✅ | `rustmath-combinatorics/src/permutations.rs` | |
| `.to_matrix()` | ⬜ | | Permutation matrix |
| `Permutations(n)` - All perms | ✅ | `rustmath-combinatorics/src/permutations.rs` | `all_permutations()` |
| Pattern avoidance | ⬜ | | |
| Bruhat order | ⬜ | | |
| Descents, ascents | ⬜ | | |

**Progress**: 5/10 functions (50%)

### 6.2 Combinations (sage.combinat.combination)
**SageMath Source**: `src/sage/combinat/combination.py`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `Combinations(n, k)` | ⬜ | | Generate all |
| `binomial(n, k)` | ✅ | `rustmath-combinatorics/src/lib.rs` | Counting only |
| `.rank()`, `.unrank()` | ⬜ | | |

**Progress**: 1/3 functions (33%)

### 6.3 Partitions (sage.combinat.partition)
**SageMath Source**: `src/sage/combinat/partition.py`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `Partition([...])` | ✅ | `rustmath-combinatorics/src/partitions.rs` | |
| `Partitions(n)` | ✅ | `rustmath-combinatorics/src/partitions.rs` | `partitions()` |
| `.conjugate()` | ✅ | `rustmath-combinatorics/src/partitions.rs` | |
| `.ferrers_diagram()` | ✅ | `rustmath-combinatorics/src/partitions.rs` | |
| `.hook_lengths()` | ⬜ | | |
| `.dimension()` | ⬜ | | Number of SYT |
| Partition function p(n) | ✅ | `rustmath-combinatorics/src/partitions.rs` | `partition_count()` |
| `.dominates()` | ⬜ | | Dominance order |

**Progress**: 5/8 functions (63%)

### 6.4 Tableaux (sage.combinat.tableau)
**SageMath Source**: `src/sage/combinat/tableau.py`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `Tableau([...])` | ⬜ | | Young tableaux |
| Standard tableaux | ⬜ | | |
| `.content()` | ⬜ | | |
| `.reading_word()` | ⬜ | | |
| Robinson-Schensted | ⬜ | | |
| `.jeu_de_taquin()` | ⬜ | | |

**Progress**: 0/6 functions (0%)

### 6.5 Posets (sage.combinat.posets)
**SageMath Source**: `src/sage/combinat/posets/`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `Poset(...)` | ⬜ | | Partially ordered set |
| Hasse diagram | ⬜ | | |
| `.maximal_elements()` | ⬜ | | |
| `.linear_extensions()` | ⬜ | | |
| Möbius function | ⬜ | | |

**Progress**: 0/5 functions (0%)

### 6.6 Other Combinatorial Structures
**SageMath Source**: Various in `src/sage/combinat/`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `factorial(n)` | ✅ | `rustmath-combinatorics/src/lib.rs` | |
| `catalan_number(n)` | ✅ | `rustmath-combinatorics/src/lib.rs` | `catalan()` |
| `fibonacci(n)` | ✅ | `rustmath-combinatorics/src/lib.rs` | |
| `lucas_number(n)` | ✅ | `rustmath-combinatorics/src/lib.rs` | `lucas()` |
| `stirling_number1(n, k)` | ⬜ | | |
| `stirling_number2(n, k)` | ✅ | `rustmath-combinatorics/src/lib.rs` | `stirling_second()` |
| `bell_number(n)` | ✅ | `rustmath-combinatorics/src/lib.rs` | |
| Set partitions | ⬜ | | |
| Dyck words | ⬜ | | |
| Integer compositions | ⬜ | | |
| Perfect matchings | ⬜ | | |
| Latin squares | ⬜ | | |

**Progress**: 6/12 functions (50%)

---

## 7. Graph Theory (sage.graphs)

### 7.1 Graph Construction (sage.graphs.graph)
**SageMath Source**: `src/sage/graphs/graph.py`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `Graph()` | ✅ | `rustmath-graphs/src/graph.rs` | Undirected |
| `DiGraph()` | ⬜ | | Directed graphs |
| `.add_vertex(v)` | 🚧 | | Implicit in `new(n)` |
| `.add_edge(u, v)` | ✅ | `rustmath-graphs/src/graph.rs` | |
| `.add_edges(edges)` | ⬜ | | |
| Weighted graphs | ⬜ | | |
| Multigraphs | ⬜ | | |
| `.num_verts()` | ✅ | `rustmath-graphs/src/graph.rs` | `num_vertices()` |
| `.num_edges()` | ✅ | `rustmath-graphs/src/graph.rs` | |
| `.vertices()` | ⬜ | | |
| `.edges()` | ⬜ | | |
| `.neighbors(v)` | ✅ | `rustmath-graphs/src/graph.rs` | |
| `.degree(v)` | ✅ | `rustmath-graphs/src/graph.rs` | |

**Progress**: 7/13 functions (54%)

### 7.2 Graph Algorithms (sage.graphs.graph_algorithms)
**SageMath Source**: `src/sage/graphs/generic_graph.py`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `.is_connected()` | ✅ | `rustmath-graphs/src/graph.rs` | |
| `.connected_components()` | ✅ | `rustmath-graphs/src/graph.rs` | |
| `.is_bipartite()` | ✅ | `rustmath-graphs/src/graph.rs` | |
| `.is_planar()` | ⬜ | | |
| `.is_tree()` | ⬜ | | |
| `.is_forest()` | ⬜ | | |
| `.is_eulerian()` | ⬜ | | |
| `.is_hamiltonian()` | ⬜ | | |

**Progress**: 3/8 functions (38%)

### 7.3 Traversals (sage.graphs.traversals)
**SageMath Source**: `src/sage/graphs/traversals.pyx`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `.breadth_first_search(start)` | ✅ | `rustmath-graphs/src/graph.rs` | `bfs()` |
| `.depth_first_search(start)` | ✅ | `rustmath-graphs/src/graph.rs` | `dfs()` |
| Topological sort | ⬜ | | For DAGs |
| `.lex_BFS()` | ⬜ | | |

**Progress**: 2/4 functions (50%)

### 7.4 Shortest Paths (sage.graphs.distances_all_pairs)
**SageMath Source**: `src/sage/graphs/distances_all_pairs.pyx`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `.shortest_path(u, v)` | ✅ | `rustmath-graphs/src/graph.rs` | BFS-based |
| `.shortest_path_length(u, v)` | ⬜ | | |
| `.all_paths(u, v)` | ⬜ | | |
| Dijkstra's algorithm | ⬜ | | Weighted graphs |
| Bellman-Ford | ⬜ | | Negative weights |
| Floyd-Warshall | ⬜ | | All pairs |
| A* search | ⬜ | | Heuristic |

**Progress**: 1/7 functions (14%)

### 7.5 Trees and Spanning Trees (sage.graphs.spanning_tree)
**SageMath Source**: `src/sage/graphs/spanning_tree.pyx`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `.min_spanning_tree()` | ⬜ | | Kruskal's |
| Prim's algorithm | ⬜ | | |
| `.spanning_trees_count()` | ⬜ | | Kirchhoff's theorem |

**Progress**: 0/3 functions (0%)

### 7.6 Graph Coloring (sage.graphs.graph_coloring)
**SageMath Source**: `src/sage/graphs/graph_coloring.pyx`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `.chromatic_number()` | ✅ | `rustmath-graphs/src/graph.rs` | |
| `.coloring()` | ✅ | `rustmath-graphs/src/graph.rs` | `greedy_coloring()` |
| Greedy coloring | ✅ | `rustmath-graphs/src/graph.rs` | |
| `.chromatic_polynomial()` | ⬜ | | |

**Progress**: 3/4 functions (75%)

### 7.7 Matching (sage.graphs.matchpoly)
**SageMath Source**: `src/sage/graphs/matchpoly.pyx`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `.matching()` | ⬜ | | Maximum matching |
| `.perfect_matchings()` | ⬜ | | |
| `.matching_polynomial()` | ⬜ | | |
| Hungarian algorithm | ⬜ | | Bipartite matching |

**Progress**: 0/4 functions (0%)

### 7.8 Graph Generators (sage.graphs.graph_generators)
**SageMath Source**: `src/sage/graphs/graph_generators.py`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| Complete graphs K_n | ⬜ | | |
| Cycle graphs C_n | ⬜ | | |
| Path graphs P_n | ⬜ | | |
| Star graphs | ⬜ | | |
| Petersen graph | ⬜ | | |
| Random graphs | ⬜ | | Erdős-Rényi, etc. |

**Progress**: 0/6 functions (0%)

---

## 8. Geometry (sage.geometry)

### 8.1 Polytopes (sage.geometry.polyhedron)
**SageMath Source**: `src/sage/geometry/polyhedron/`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `Polyhedron(vertices)` | ⬜ | | |
| `.vertices()` | ⬜ | | |
| `.faces()` | ⬜ | | |
| `.volume()` | ⬜ | | |
| `.is_lattice_polytope()` | ⬜ | | |
| Convex hull | ⬜ | | |
| Face lattice | ⬜ | | |

**Progress**: 0/7 functions (0%)

### 8.2 Toric Geometry (sage.geometry.toric_varieties)
**SageMath Source**: `src/sage/geometry/toric_`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| Toric varieties | ⬜ | | |
| Fans | ⬜ | | |
| Cones | ⬜ | | |

**Progress**: 0/3 features (0%)

### 8.3 Computational Geometry
**SageMath Source**: Various

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| Convex hull (2D) | ⬜ | | |
| Voronoi diagrams | ⬜ | | |
| Delaunay triangulation | ⬜ | | |
| Line intersection | ⬜ | | |
| Point in polygon | ⬜ | | |

**Progress**: 0/5 functions (0%)

---

## 9. Algebraic Geometry (sage.schemes)

### 9.1 Affine and Projective Varieties
**SageMath Source**: `src/sage/schemes/`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| Affine space | ⬜ | | |
| Projective space | ⬜ | | |
| Varieties | ⬜ | | |
| Morphisms | ⬜ | | |
| Gröbner bases | ⬜ | | Critical for ideals |

**Progress**: 0/5 features (0%)

### 9.2 Elliptic Curves (sage.schemes.elliptic_curves)
**SageMath Source**: `src/sage/schemes/elliptic_curves/`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `EllipticCurve([a,b])` | ⬜ | | |
| Point addition | ⬜ | | |
| Torsion points | ⬜ | | |
| Rank computation | ⬜ | | |
| L-functions | ⬜ | | |
| Modular forms | ⬜ | | |

**Progress**: 0/6 features (0%)

---

## 10. Cryptography (sage.crypto)

### 10.1 Classical Cryptography (sage.crypto.classical)
**SageMath Source**: `src/sage/crypto/classical.py`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| Caesar cipher | ⬜ | | |
| Vigenère cipher | ⬜ | | |
| Substitution cipher | ⬜ | | |
| Hill cipher | ⬜ | | Matrix-based |

**Progress**: 0/4 functions (0%)

### 10.2 Public Key Cryptography (sage.crypto.public_key)
**SageMath Source**: `src/sage/crypto/public_key/`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| RSA key generation | ✅ | `rustmath-crypto/src/rsa.rs` | From primes |
| RSA encryption | ✅ | `rustmath-crypto/src/rsa.rs` | |
| RSA decryption | ✅ | `rustmath-crypto/src/rsa.rs` | |
| RSA signing | ⬜ | | |
| Diffie-Hellman | ⬜ | | |
| ElGamal | ⬜ | | |
| ECC (Elliptic Curve) | ⬜ | | |
| ECDSA | ⬜ | | |

**Progress**: 3/8 functions (38%)

### 10.3 Block Ciphers (sage.crypto.block_cipher)
**SageMath Source**: `src/sage/crypto/block_cipher/`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| AES | ⬜ | | |
| DES | ⬜ | | |
| General S-boxes | ⬜ | | |

**Progress**: 0/3 functions (0%)

### 10.4 Hash Functions
**SageMath Source**: Not directly in Sage (uses external libraries)

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| SHA-256 | ⬜ | | |
| SHA-3 | ⬜ | | |
| BLAKE2 | ⬜ | | |

**Progress**: 0/3 functions (0%)

---

## 11. Coding Theory (sage.coding)

### 11.1 Linear Codes (sage.coding.linear_code)
**SageMath Source**: `src/sage/coding/linear_code.py`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `LinearCode(...)` | ⬜ | | |
| Generator matrix | ⬜ | | |
| Parity check matrix | ⬜ | | |
| `.minimum_distance()` | ⬜ | | |
| `.encode()` | ⬜ | | |
| `.decode()` | ⬜ | | |

**Progress**: 0/6 functions (0%)

### 11.2 Specific Codes
**SageMath Source**: Various in `src/sage/coding/`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| Hamming codes | ⬜ | | |
| Reed-Solomon codes | ⬜ | | |
| BCH codes | ⬜ | | |
| Golay codes | ⬜ | | |

**Progress**: 0/4 types (0%)

---

## 12. Group Theory (sage.groups)

### 12.1 Permutation Groups (sage.groups.perm_gps)
**SageMath Source**: `src/sage/groups/perm_gps/`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `PermutationGroup(...)` | ⬜ | | |
| Symmetric group S_n | ⬜ | | |
| Alternating group A_n | ⬜ | | |
| `.order()` | ⬜ | | Group order |
| `.center()` | ⬜ | | |
| `.subgroups()` | ⬜ | | |
| Sylow subgroups | ⬜ | | |

**Progress**: 0/7 functions (0%)

### 12.2 Matrix Groups (sage.groups.matrix_gps)
**SageMath Source**: `src/sage/groups/matrix_gps/`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| General linear group GL(n) | ⬜ | | |
| Special linear group SL(n) | ⬜ | | |
| Orthogonal group O(n) | ⬜ | | |
| Unitary group U(n) | ⬜ | | |

**Progress**: 0/4 groups (0%)

### 12.3 Abelian Groups (sage.groups.abelian_gps)
**SageMath Source**: `src/sage/groups/abelian_gps/`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `AbelianGroup(...)` | ⬜ | | |
| Direct products | ⬜ | | |
| Structure theorem | ⬜ | | |

**Progress**: 0/3 functions (0%)

---

## 13. Representation Theory (sage.algebras.representation)

**SageMath Source**: `src/sage/algebras/`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| Character tables | ⬜ | | |
| Irreducible representations | ⬜ | | |
| Tensor products | ⬜ | | |

**Progress**: 0/3 features (0%)

---

## 14. Homological Algebra (sage.homology)

**SageMath Source**: `src/sage/homology/`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| Chain complexes | ⬜ | | |
| Homology groups | ⬜ | | |
| Cohomology | ⬜ | | |

**Progress**: 0/3 features (0%)

---

## 15. Category Theory (sage.categories)

**SageMath Source**: `src/sage/categories/`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| Category framework | 🚧 | `rustmath-core/src/traits.rs` | Trait-based |
| Functors | ⬜ | | |
| Natural transformations | ⬜ | | |

**Progress**: 1/3 features (33%)

---

## 16. Statistics and Probability (sage.probability, sage.stats)

**SageMath Source**: `src/sage/probability/`, `src/sage/stats/`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| Probability distributions | ⬜ | | |
| Random variables | ⬜ | | |
| Statistical functions | ⬜ | | Mean, variance, etc. |
| Hypothesis testing | ⬜ | | |
| Regression | ⬜ | | |

**Progress**: 0/5 features (0%)

---

## 17. Numerical Computation (sage.numerical)

**SageMath Source**: `src/sage/numerical/`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| `find_root(f, a, b)` | ⬜ | | Root finding |
| `minimize(f, x0)` | ⬜ | | Optimization |
| Linear programming | ⬜ | | |
| Numerical integration | ⬜ | | Quadrature |
| Interpolation | ⬜ | | |
| FFT | ⬜ | | Fast Fourier Transform |

**Progress**: 0/6 functions (0%)

---

## 18. Logic and SAT Solvers (sage.logic)

**SageMath Source**: `src/sage/logic/`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| Boolean formulas | ⬜ | | |
| CNF/DNF | ⬜ | | |
| SAT solving | ⬜ | | |
| Proofs | ⬜ | | |

**Progress**: 0/4 features (0%)

---

## 19. Dynamics (sage.dynamics)

**SageMath Source**: `src/sage/dynamics/`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| Dynamical systems | ⬜ | | |
| Fractals | ⬜ | | |
| Chaos theory | ⬜ | | |

**Progress**: 0/3 features (0%)

---

## 20. Databases (sage.databases)

**SageMath Source**: `src/sage/databases/`

| Function/Feature | Status | RustMath Location | Notes |
|-----------------|--------|-------------------|-------|
| OEIS interface | ⬜ | | Online Encyclopedia of Integer Sequences |
| Cunningham tables | ⬜ | | Factorizations |
| Elliptic curve database | ⬜ | | Cremona database |

**Progress**: 0/3 features (0%)

---

## Summary Statistics

### By Major Category

| Category | Functions Impl. | Total Functions | Progress |
|----------|----------------|-----------------|----------|
| 1. Rings & Fields | 43 | 90 | 48% |
| 2. Linear Algebra | 42 | 46 | 91% |
| 3. Number Theory | 20 | 42 | 48% |
| 4. Symbolic Computation | 11 | 35 | 31% |
| 5. Calculus | 7 | 30 | 23% |
| 6. Combinatorics | 17 | 54 | 31% |
| 7. Graph Theory | 17 | 59 | 29% |
| 8. Geometry | 0 | 15 | 0% |
| 9. Algebraic Geometry | 0 | 11 | 0% |
| 10. Cryptography | 3 | 18 | 17% |
| 11. Coding Theory | 0 | 10 | 0% |
| 12. Group Theory | 0 | 14 | 0% |
| 13-20. Advanced Topics | 1 | 49 | 2% |

**TOTAL**: **161 / 473 functions** = **~35% complete**

### Files to Examine in SageMath Source

Key source files for future implementation:

1. **Integers**: `src/sage/rings/integer.pyx`, `integer_ring.py`
2. **Polynomials**: `src/sage/rings/polynomial/polynomial_element.pyx`
3. **Matrices**: `src/sage/matrix/matrix2.pyx`
4. **Symbolic**: `src/sage/symbolic/expression.pyx`
5. **Calculus**: `src/sage/calculus/calculus.py`
6. **Combinatorics**: `src/sage/combinat/`
7. **Graphs**: `src/sage/graphs/graph.py`
8. **Crypto**: `src/sage/crypto/`

### Priority Areas for Implementation

Based on usage and foundational importance:

1. **High Priority** (foundational, commonly used):
   - ✅ Basic rings and fields (mostly done)
   - 🚧 Linear algebra completion (QR, SVD, eigenvalues)
   - 🚧 Polynomial factorization (complete algorithms)
   - 🚧 Symbolic integration
   - ⬜ Real/complex numbers with arbitrary precision

2. **Medium Priority** (frequently used):
   - ⬜ Gröbner bases
   - ⬜ More graph algorithms (coloring, matching)
   - ⬜ Group theory basics
   - ⬜ Numerical methods
   - ⬜ Series expansions

3. **Lower Priority** (specialized):
   - ⬜ Algebraic geometry
   - ⬜ Homological algebra
   - ⬜ Toric geometry
   - ⬜ Modular forms

---

## Notes

1. **Granularity**: This list focuses on major functions and features. Many functions have dozens of methods and variations not individually listed.

2. **Accuracy**: Based on RustMath current implementation as of this document. Status may change with ongoing development.

3. **SageMath Scale**: SageMath has ~2 million lines of code. This checklist represents core functionality, not every function.

4. **Testing**: ✅ indicates implementation exists and is tested, but may not have full feature parity with SageMath.

5. **Updates**: This document should be updated as new features are implemented.

## References

- SageMath Documentation: https://doc.sagemath.org/
- SageMath Source: https://github.com/sagemath/sage
- RustMath Repository: (current project)
