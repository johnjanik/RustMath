# Chapter 50 — Lazy Power Series Rings

**Handbook part:** VII — Local Arithmetic Fields
**Handbook pages:** 1349–1361 (PDF pages 1480–1495)

---

## Scope and overview

Chapter 50 introduces **lazy power series rings** (`RngLaz`) and their elements (`RngLazElt`),
which allow the creation of infinite-precision multivariate power series. The defining
characteristic of a lazy series is that all of its coefficients are *knowable* — computed on
demand by a formula — even though only finitely many can be known at any one time. Once a
coefficient is computed it is cached with the series for instant retrieval on subsequent calls.

The simplest construction gives a series by supplying a **map** whose input is the tuple of
variable exponents of a monomial and whose output is the coefficient of that monomial. More
complex lazy series arise from arithmetic on simpler ones: addition, multiplication, inversion,
evaluation, and square roots all yield new lazy series with automatically derived coefficient
formulas.

**Ordering convention.** Monomials are ordered by total degree (the "spiral" ordering, the same
default as for multivariate polynomials). This ordering governs `Coefficients`, `Valuation`,
`PrintToPrecision`, and `PrintTermsOfDegree`. An alternative non-spiral ordering is available
via `CoefficientsNonSpiral`.

**Limitation.** For series constructed by multiplication, inversion, evaluation, or square root,
computing the coefficient of the monomial with exponent vector `(i₁,…,iᵣ)` requires all
coefficients whose total degree is smaller to be known first. Consequently the individual
exponents must be small integers (< 2³⁰) in such cases.

---

## 50.1 Introduction

The `RngLaz` type provides infinite-precision power series over an arbitrary coefficient ring in
any number of variables. The Introduction section contains no intrinsics; its content is covered
in the scope overview above.

---

## 50.2 Creation of Lazy Series Rings

Both univariate and multivariate lazy series rings can be created.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LazyPowerSeriesRing(C, n)` | Returns the lazy power series ring with coefficient ring `C` and `n` variables. Any ring is valid for `C`; `n` must be a positive integer. | — |
| `ChangeRing(L, C)` | Given a lazy series ring `L` defined over ring `R` and a ring `C`, returns the lazy series ring with coefficient ring `C` and the same number of variables as `L`, together with a map from `L` to the new ring that coerces each coefficient into `C`. | — |

*Worked examples: H50E1 (creation and printing of lazy power series rings; `ChangeRing` to a cyclotomic maximal order).*

---

## 50.3 Functions on Lazy Series Rings

Lazy series rings expose their variables, variable names, and coefficient ring.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `R . i` | The `i`-th variable of the lazy power series ring `R`, where `i` is between 1 and the rank of `R`. | — |
| `AssignNames(~R, S)` | Given a lazy series ring `R` with `n` indeterminates and a sequence `S` of `n` strings, assigns the elements of `S` as the names of the variables of `R`. Names are used by `PrintToPrecision` but not in default series printing. | — |
| `BaseRing(R)` / `CoefficientRing(R)` | The coefficient ring of the lazy power series ring `R`. All series coefficients lie in this ring. | — |
| `Rank(R)` | The number of variables of the lazy power series ring `R`. | — |
| `R1 eq R2` | Returns `true` if the lazy series rings `R1` and `R2` are the same ring (same coefficient ring and same rank). | — |

*Worked examples: H50E2 (accessing variables, assigning names, `CoefficientRing`, `Rank`, equality test).*

---

## 50.4 Elements

Lazy series may be created in a number of ways, and arithmetic, coefficient extraction, and
predicates are all available.

### 50.4.1 Creation of Finite Lazy Series

Series with finitely many non-zero terms can be created by coercion from ring elements,
sequences, or polynomials/rational functions.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `R ! c` | Returns the series in `R` with constant term `c` (coercible into the coefficient ring) and every other coefficient 0. | — |
| `R ! s` | Returns the lazy series in `R` whose coefficients are those of the lazy series `s` coerced into the coefficient ring of `R`. | — |
| `R ! S` | Returns the series in `R` whose coefficients are the elements of sequence `S` (each coercible into the coefficient ring of `R`). Coefficients are given in the "spiral" order used by `Coefficients` and `PrintToPrecision`; any coefficient not given is taken to be zero. The result has only finitely many non-zero terms, all of which must be present in `S`. | — |
| `LazySeries(R, f)` | Creates a lazy series in `R` from the polynomial `f` (or rational function `p/q`): the series has the terms of `f` as its non-zero terms. The number of variables of `f`'s parent must match the rank of `R`; coefficients of `f` must be coercible into the coefficient ring of `R`. For a rational function `p/q` the result is `LazySeries(R, p) * LazySeries(R, q)^(-1)`. | — |
| `elt< R \| m >` | Creates a series in `R` from the map `m`. For univariate `R`, `m` takes a non-negative integer (the exponent) to a coefficient. For multivariate `R` with `r` variables, `m` takes a tuple of `r` integers `<i₁,…,iᵣ>` (the exponent vector) and returns the coefficient of `x₁^i₁ * … * xᵣ^iᵣ`. See §50.4.1.1 for details. | — |

*Worked examples: H50E3 (coercion of constants, lazy series, and rational functions; `AlgebraicClosure` coefficient ring). H50E4 (univariate map construction over the maximal order of Q(√5); spot-checking large-index coefficients). H50E5 (multivariate map construction; coefficient lookup by exponent tuple).*

#### 50.4.1.1 Creation of Lazy Series from Maps

A lazy series created from a map `m` has, by definition, no coefficient stored at creation time.
For a ring with variables `x₁,…,xᵣ`, the map `m` receives the exponent tuple `<i₁,…,iᵣ>` and
returns the coefficient of `x₁^i₁ * … * xᵣ^iᵣ`. For a univariate ring the map may take a plain
integer instead of a 1-tuple.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `elt< R \| m >` | Creates the series in `R` whose coefficient of the monomial with exponent vector `(i₁,…,iᵣ)` is `m(<i₁,…,iᵣ>)` (or `m(i)` for univariate `R`). The map domain must be compatible: `Integers()` for univariate, `car<Integers(),…,Integers()>` (`r`-fold cartesian product) for multivariate. The codomain must be the coefficient ring of `R`. | — |

---

### 50.4.2 Arithmetic with Lazy Series

All standard arithmetic operations are available; the result is always a new lazy series whose
coefficient formula is derived from those of the operands.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `s + t` | Sum of lazy series `s` and `t`. | — |
| `-s` | Negation of lazy series `s`. | — |
| `s - t` | Difference of lazy series `s` and `t`. | — |
| `s * t` | Product of lazy series `s` and `t`. | — |
| `s + r` / `r + s` | Sum of lazy series `s` and an element `r` of the coefficient ring of the parent of `s`. | — |
| `c * s` / `s * c` | Product of lazy series `s` and an element `c` of the coefficient ring of the parent of `s`. | — |
| `s * n` | Product of the lazy series `s` and the monomial `x^n`, where `n = [n₁,…,nᵣ]` is a sequence of exponents and `x₁,…,xᵣ` are the series variables. That is, multiplication by `x₁^n₁ * … * xᵣ^nᵣ`. | — |
| `s ^ n` | `n`-th power of lazy series `s` for integer `n`. Negative `n` is allowed; inverses are taken where possible. | — |

*Worked examples: H50E6 (combined arithmetic expression including monomial multiplication and inversion).*

---

### 50.4.3 Finding Coefficients of Lazy Series

Coefficients are computed on demand and cached. For series built via multiplication, inversion,
evaluation, or square root, computing the coefficient at exponent vector `(i₁,…,iᵣ)` requires
all coefficients of smaller total degree to be known first, so individual exponents must be < 2³⁰.
The default ordering is by total degree ("spiral" ordering). Any of these functions may be
interrupted at the Magma prompt.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Coefficient(s, i)` | Coefficient of `x^i` in the univariate lazy series `s`, where `i` is a non-negative integer. | — |
| `Coefficient(s, T)` | Coefficient of `x₁^T[1] * … * xᵣ^T[r]` in the multivariate lazy series `s`, where `T = [T₁,…,Tᵣ]` is a sequence of non-negative integers. | — |
| `Coefficients(s, n)` / `Coefficients(s, l, n)` | Sequence of coefficients of `s` whose monomials have total degree between `l` (default 0) and `n`, ordered by the "spiral" degree order. Both `l` and `n` must be non-negative integers. | — |
| `Valuation(s)` | The valuation of `s`: the exponent of the first non-zero monomial in spiral order. Returns `Infinity` for the zero series, an integer for the univariate case, and a sequence for the multivariate case. | — |
| `PrintToPrecision(s, n)` | Prints the sum of all terms of `s` of total degree at most `n` (non-negative integer), using the spiral ordering and the assigned variable names. | — |
| `PrintTermsOfDegree(s, l, n)` | Prints the sum of terms of `s` of total degree between `l` and `n` (non-negative integers), in spiral ordering. | — |
| `LeadingCoefficient(s)` | The first non-zero coefficient of `s` in spiral ordering (i.e. the coefficient whose monomial has exponent equal to the valuation of `s`). | — |
| `LeadingTerm(s)` | The first non-zero term of `s` in spiral ordering (monomial with exponent equal to the valuation of `s`, multiplied by its coefficient). | — |
| `CoefficientsNonSpiral(s, n)` | Coefficients of `s` for all monomials `x₁^i₁ * … * xᵣ^iᵣ` with `iⱼ ≤ n[j]` for each `j`. `n` may be a non-negative integer (univariate) or a sequence of non-negative integers of length `r`. The index of the `[i₁,…,iᵣ]`-th coefficient in the returned sequence is `Σⱼ iⱼ * (Πₖ>ⱼ (n[k]+1))`. | Alternative to spiral ordering; may be slower for series built from inversion as it may require higher-degree coefficients than asked for. |
| `Index(s, i, n)` | Returns the index in the return value of `CoefficientsNonSpiral(s, n)` of the monomial whose exponent vector is the sequence `i`. | — |

*Worked examples: H50E7 (coefficient lookup, `Coefficients` to degree 6, `PrintToPrecision` to degree 6, valuation of an invertible and zero series). H50E8 (`CoefficientsNonSpiral` vs. spiral — timing comparison for an inverted series).*

---

### 50.4.4 Predicates on Lazy Series

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `s eq t` | Returns `true` if the lazy series `s` and `t` are exactly the same series (same object). | — |
| `IsZero(s)` | Returns `true` if `s` was created as the zero series. | — |
| `IsOne(s)` | Returns `true` if `s` was created as the one (identity) series. | — |
| `IsMinusOne(s)` | Returns `true` if `s` was created as the minus-one series. | — |
| `IsUnit(s)` | Returns `true` if the lazy series `s` is a unit. | — |
| `IsWeaklyZero(s, n)` | Returns `true` if all terms of `s` of total degree at most `n` are zero. Called without the second argument, equivalent to `IsZero(s)`. | — |
| `IsWeaklyEqual(s, t, n)` | Returns `true` if `s` and `t` agree in all terms of total degree at most `n`. Called without the third argument, equivalent to `s eq t`. | — |

---

### 50.4.5 Other Functions on Lazy Series

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Derivative(s)` / `Derivative(s, v)` / `Derivative(s, v, n)` | Returns the (`n`-th, default 1) derivative of the lazy series `s` with respect to the `v`-th variable of the parent ring. If `v` is omitted the parent must be univariate and the unique variable is used. Valid values of `v` are 1 to `Rank(Parent(s))`; `n` must be a positive integer. | — |
| `Integral(s)` / `Integral(s, v)` | Returns the integral of `s` with respect to the `v`-th variable. If `v` is omitted the parent must be univariate. Valid values of `v` are 1 to `Rank(Parent(s))`. | — |
| `Evaluate(s, t)` / `Evaluate(s, T)` | Evaluates the lazy series `s` at the lazy series `t` (univariate or matching-rank case) or at the sequence `T` of lazy series. The series `t` (or each series in `T`) must have zero constant term so that every coefficient of the result can be finitely computed. | — |
| `SquareRoot(s)` / `Sqrt(s)` | The square root of the lazy series `s`. | — |
| `IsSquare(s)` | Returns `true` if `s` is a square, and if so returns the square root as the second value. | — |
| `PolynomialCoefficient(s, i)` | For a lazy series `s` whose coefficient ring is a (univariate or multivariate) polynomial ring: reinterprets `s` as a polynomial with lazy series as coefficients and returns the series which is the coefficient of the `i`-th power of the polynomial variable (univariate case, `i` a non-negative integer) or of `x₁^i[1] * … * xᵣ^i[r]` (multivariate case, `i` a sequence of non-negative integers). | — |

*Worked examples: H50E9 (bivariate `s` with all-1 map evaluated at a univariate series; composition and constant-shifted composition giving Taylor-like expansions).*

---

## 50.5 Bibliography

No bibliography is printed in Chapter 50 of the MAGMA Handbook. The chapter contains no
algorithmic citations or reference list.

---

## Algorithm-to-function quick reference

| Algorithm / concept | Functions |
|---------------------|-----------|
| Lazy (on-demand) coefficient evaluation via user-supplied map | `elt< R \| m >` |
| Coefficient caching and retrieval | `Coefficient`, `Coefficients`, `CoefficientsNonSpiral` |
| Total-degree ("spiral") monomial ordering | `Coefficients`, `Valuation`, `PrintToPrecision`, `PrintTermsOfDegree`, `LeadingCoefficient`, `LeadingTerm` |
| Non-spiral (box) monomial ordering | `CoefficientsNonSpiral`, `Index` |
| Lazy arithmetic (derived coefficient formulas) | `+`, `-`, `*`, `^`, `Derivative`, `Integral`, `Evaluate`, `SquareRoot`, `Sqrt` |
| Ring/element creation from polynomials and rational functions | `LazySeries`, `R ! S`, `R ! c`, `R ! s` |
| Weak (finite-precision) equality and zero testing | `IsWeaklyZero`, `IsWeaklyEqual` |
| Coefficient-ring reinterpretation | `PolynomialCoefficient`, `ChangeRing` |
