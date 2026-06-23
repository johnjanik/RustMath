# Chapter 52 — Algebraic Power Series Rings

**Authors:** Tobias Beck (RICAM, Linz, Austria); low-level adaptations by the Magma group
**Handbook part:** VII — Local Arithmetic Fields
**Handbook pages:** 1377–1387 (PDF pages 1508–1523)

---

## Scope and overview

Algebraic Power Series are a lazy representation of multivariate power series with fractional
exponents, which are roots of univariate polynomials with coefficients in multivariate polynomial
rings. The functionality allows the "lazy" computation of the power series expansion to any finite
degree, this being well-determined by the defining algebraic equation.

The package was designed with the computation of formal resolutions of singularities of surfaces
in mind but provides a useful general tool. As well as allowing definition directly from a
polynomial equation, users can compose algebraic power series and recursively define series that
are roots of polynomials whose coefficients are polynomial functions in other algebraic power
series. There are also functions for basic arithmetic operations and tests for exact equality.

The defined series must be expandable in fractional (positive) powers of the base variables.
This is true for all roots of a quasi-ordinary polynomial, possibly after a finite extension of
the base field.

The package was designed and implemented by Tobias Beck at the RICAM institute in Linz, Austria.
The algorithms are described in **[Bec07, Sec. 4]**.

---

## 52.1 Introduction

See scope and overview above.

---

## 52.2 Basics

In Magma, algebraic power series are represented in a hybrid lazy-exact way. Eventually every
power series is given by a defining polynomial and a sufficiently large initial segment.
Intermediate operations are represented in a lazy way. This makes it possible to compute both
quickly and to high precision if necessary.

Note that decision procedures may be very time intensive. The chapter text indicates for each
function whether it is fast or must be used with care.

### 52.2.1 Data Structures

Algebraic power series are of type `RngPowAlgElt`. There are two structural types:

- **Atomic (type A):** Given directly as the root of a univariate polynomial `p(z)` with a
  given initial expansion. `p(z)` comes from a polynomial `f(z)` with coefficients in a
  multivariate polynomial ring; `p = f` or `p` is determined from `f` by evaluating the
  coefficients at an array of algebraic power series, enabling recursive construction. The
  attribute `type` equals `0` for type A.

- **Substitution (type B):** Allows composition of algebraic power series. The principal
  defining data is an algebraic power series `s` in `n` variables and an array of `n` algebraic
  series substituted into it (through elements in the dual lattice of the exponent lattice of
  `s`). The attribute `type` equals `1` for type B.

Both types have an associated exponent lattice specified by a sublattice Γ of a standard integral
lattice and a positive integer `e` (the LCM of the denominators of the fractional exponents).
Finite expansions returned are always integral-exponent multivariate polynomials; the actual
mathematical expansion is obtained by dividing all exponents by `e`.

**Important:** The domain of an algebraic power series must be a multivariate polynomial ring
with a degree ordering (glex or grevlex). Polynomial rings with non-degree orderings will cause
a user error.

### 52.2.2 Verbose Output

A verbose flag `AlgSeries` exists which can take values `true`, `false`, `0`, or `1`. Setting
to `true` (or `1`) outputs information on the progress of potentially time-consuming intrinsics.

---

## 52.3 Constructors

Using constructors one can construct power series starting from polynomial data or using other
power series recursively.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PolyToSeries(s)` | Given a multivariate polynomial `s`, returns the series representation of `s`. | — |
| `AlgebraicPowerSeries(dp, ip, L, e)` | Define a power series root of a polynomial `p` using initial expansion `ip` and exponent lattice `(1/e)L`. The defining polynomial `p` is `dp` when optional parameter `subs` (default `[]`) is empty, or obtained by substituting the elements of `subs` into the variables of `dp`. The initial expansion must be sufficiently long to uniquely identify a root (see **[Bec07, Cond. 4.3]**). In expansions, variables represent `e`-th roots, so `x₁ x₂²` is really `x₁^{1/e} x₂^{2/e}`. Simpler overloads omit `L` (assuming the standard integral lattice) or `e` (assuming `e = 1`). No strong correctness checks are performed at construction time; incorrect initial data is only revealed during later expansion. | **[Bec07, Cond. 4.3]** |
| `EvaluationPowerSeries(s, nu, v)` | Given a series `s`, a sequence `nu` of vectors in the dual of its exponent lattice, and a sequence `v` (same length) of power series in a common domain with compatible coefficient field: returns the series obtained by substituting `x^μ ↦ ∏ᵢ v[i]^⟨nu[i],μ⟩`. Requires `nu` and `v` to fulfill a convergence condition (see **[Bec07, Cond. 4.6]**). | **[Bec07, Cond. 4.6]** |
| `ImplicitFunction(dp)` | The unique series with zero constant term defined by a polynomial `p ∈ k[x₁,…,xₙ][z]` or `k[[x₁,…,xₙ]][z]` fulfilling the implicit function theorem conditions: `p(0,…,0) = 0` and `∂p/∂z(0,…,0) ≠ 0`. `dp` may be substituted with the series in optional parameter `subs` (default `[]`). `dp` should have coefficients in a multivariate polynomial ring. | Implicit function theorem; **[Bec07]** |

*Worked examples: H52E1 (PolyToSeries, ImplicitFunction, AlgebraicPowerSeries with explicit lattice and initial expansion, EvaluationPowerSeries for composition, RationalPuiseux with both Duval := false and true).*

### 52.3.1 Rational Puiseux Expansions

Let `p ∈ k[[x₁,…,xₙ]][z]` be a quasi-ordinary polynomial over a field `k` of characteristic
zero: `p` is non-zero, squarefree, monic, and its discriminant `d = x₁^{e₁} ⋯ xₙ^{eₙ} u(x₁,…,xₙ)`
where `u` is a unit in the power series ring. The Theorem of Jung–Abhyankar states that `p` has
`deg(p)` distinct Puiseux series roots, i.e., power series roots with fractionary exponents and
coefficients in the algebraic closure of `k`.

These roots are computed by a generalization of the Newton–Puiseux algorithm. Duval's extension
for computing rational parametrizations has also been implemented. For further details see
**[Bec07, Sec. 4.3]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RationalPuiseux(p)` | Assumes `p` is a univariate polynomial over a multivariate polynomial ring `S = k[x₁,…,xᵣ]` and that `p` is quasi-ordinary (the user must ensure this for efficiency). Returns: (1) the exponent lattice `⟨Γ₀, e₀⟩` of the input polynomial; (2) a complete list of rational parametrizations in format `⟨λ, s, N, E⟩` where `λ` is a sequence of `r` field elements, `s` is a fractionary algebraic power series of type `RngPowAlgElt` solving the scaled polynomial, `N` is the index of `e₀⁻¹Γ₀` in the exponent lattice of `s`, and `E` is the degree of the coefficient field extension needed. Optional parameter `Gamma` (lattice, default `StandardLattice`) specifies Γ₀. Parameter `subs` (default `[]`): a sequence of `r` power series in a common domain — when provided, variables in `p` are substituted by the corresponding series and the resulting polynomial must be quasi-ordinary; `Gamma` has no effect in this case. Parameter `Duval` (BoolElt, default `false`): if `false`, returns a complete set of representatives (up to conjugacy) of Puiseux series roots with λ-vectors always vectors of ones; if `true`, applies Duval's trick, requiring automorphisms of `k[[x₁,…,xₙ]]` in advance but avoiding field extensions. Parameter `OnlySingular` (BoolElt, default `false`): if `true`, returns only parametrizations corresponding to singular branches. Parameters `ExtName` (MonStgElt, default `"gamma"`) and `ExtCount` (RngIntElt, default `0`) control naming of algebraic elements when the ground field must be extended; the last return value is `ExtCount` plus the number of field extensions introduced. | Newton–Puiseux algorithm with Duval's extension **[Bec07, Sec. 4.3]** |

*Worked examples: H52E1 (quasi-ordinary polynomial `z⁶ + 3xy²z⁴ + xyz³ + 3x²y⁴z² + x³y⁶`, Puiseux parametrizations with Duval := false showing field extensions, and Duval := true showing automorphism substitutions; verification that sum of extension degrees equals degree of qopol in both cases).*

---

## 52.4 Accessors and Expansion

The following functions provide an interface to extract information from a defined power series.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Domain(s)` | Return the multivariate polynomial ring used for approximating the series `s` by its truncations. | — |
| `ExponentLattice(s)` | Return the exponent lattice `(1/e)Γ` of the series as tuple `(Γ, e)` where `Γ` is an integral lattice and `e` is an integer. | — |
| `DefiningPolynomial(s)` | Return a defining polynomial of the series: a squarefree univariate polynomial over `Domain(s)`. For series defined with substitutions, the computation may be expensive and can involve recursive resultant computations. | Recursive resultant computation for substitution-type series |
| `Order(s)` | Return the integral order (total degree of smallest non-zero term) of the expansion as returned by `Expand`, i.e., its fractionary order times the exponent denominator. If `s` is zero, this function will not terminate. Optional parameter `TestZero` (BoolElt, default `false`): set to `true` to get return value `−1` for the zero series, but note this involves the computationally complex call `IsZero`. | — |
| `Expand(s, ord)` | Given the power series `β` represented by `s`, let `α` be the result of substituting `xᵢ ↦ xᵢᵉ` where `e` is taken from `ExponentLattice(s)`. Returns `true` and the truncation of `α` modulo terms of order ≥ `ord`. A return of `false` indicates the representation is inconsistent (only happens when `RationalPuiseux` is called with non-quasi-ordinary input, or `AlgebraicPowerSeries` is used inconsistently). | — |

*Worked examples: H52E2 (Domain, ExponentLattice, DefiningPolynomial, Order on the recursively-defined series s3; interpretation of results showing s3 ∈ Q(s)[[t^{1/3}]]).*

---

## 52.5 Arithmetic

Basic arithmetic operations on power series. These are fast operations.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AlgComb(c, ss)` | Given a polynomial `c` in `r` variables and a sequence `ss` of `r` power series in a common domain with compatible coefficient field: return the series obtained by substituting the elements of `ss` for the variables of `c`. Allows construction of completely arbitrary algebraic combinations. | — |
| `s + t` | Add two power series. | — |
| `s - t` | Subtract two power series. | — |
| `s * t` | Multiply two power series. | — |

*Worked examples: H52E3 (AlgComb to construct s0² + s1²; addition and multiplication of series; verification that h2 = 1 and h3 = 0).*

---

## 52.6 Predicates

Decision algorithms for algebraic power series. These may involve recursive resultant computations
and have high complexity; they should be used with care.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsZero(s)` | Decides if the series is zero. | Recursive resultant / algebraic; computationally expensive |
| `s eq t` | Decides if two series are equal. | Reduction to `IsZero` |
| `IsPolynomial(s)` | Decides whether the series is actually a polynomial (with integral exponents) in the multivariate polynomial domain returned by `Domain(s)`. In the positive case, also returns that polynomial. Relies on `SimplifyRep`. | Calls `SimplifyRep`; computationally expensive |

*Worked examples: H52E4 (IsPolynomial on h1 returning false; IsPolynomial on h2 returning true with value 1; IsEqual confirming h2 equals PolyToSeries(1); IsZero confirming h3 is zero).*

---

## 52.7 Modifiers

Functions that modify the representation of a power series or apply a simple automorphism.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ScaleGenerators(s, ls)` | Let `{γᵢ}ᵢ` be the basis (determined by Magma's representation) of the exponent lattice of `s`, and let `σ : x^{γᵢ} ↦ ls[i] · x^{γᵢ}`. Return the series `σ(s)`. | — |
| `ChangeRing(s, R)` | If `R` is a multivariate polynomial domain compatible with the approximation domain `Domain(s)`, return the same power series with new approximation domain `R`. Acts as a coercion between power series rings. | — |
| `SimplifyRep(s)` | "Simplifies" the internal representation of a series. The result is a series of atomic type without recursive (substitution) dependencies on other power series. With optional parameter `Factorizing` (BoolElt, default `true`), the defining polynomial of the simplified series will be irreducible and therefore a minimal polynomial over `Domain(s)` (if `Factorizing := false`, it is only guaranteed squarefree). After simplification, `DefiningPolynomial` returns this minimal polynomial, which can be useful (e.g., for `IsPolynomial`). **Warning:** If series leaves were constructed by `RationalPuiseux` with `Gamma` set, then calling `SimplifyRep` with `Factorizing := true` computes a minimal polynomial over the whole polynomial ground ring, which may not be the intended result. | Recursive resultant and polynomial factorization; expensive |

*Worked examples: H52E5 (ScaleGenerators mapping lattice generators of s2 by factors 3 and 4; ChangeRing to view h1 as a series in Q(i)[[u,v]]; SimplifyRep on h3 to obtain the explicit zero representation; DefiningPolynomial of the result returning z).*

---

## 52.8 Bibliography

| Key | Reference |
|-----|-----------|
| **[Bec07]** | Tobias Beck. *Formal Desingularization of Surfaces — The Jung Method Revisited —.* Technical Report 2007-31, RICAM, December 2007. URL: http://www.ricam.oeaw.ac.at/publications/reports/ |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Polynomial-to-series coercion | `PolyToSeries` |
| Implicit function theorem **[Bec07]** | `ImplicitFunction` |
| Newton–Puiseux algorithm with Duval's extension **[Bec07, Sec. 4.3]** | `RationalPuiseux` |
| Algebraic series root from initial expansion **[Bec07, Cond. 4.3]** | `AlgebraicPowerSeries` |
| Substitution / evaluation of series **[Bec07, Cond. 4.6]** | `EvaluationPowerSeries` |
| Exponent lattice extraction | `ExponentLattice`, `Domain` |
| Lazy truncation expansion | `Expand`, `Order`, `DefiningPolynomial` |
| Lazy arithmetic | `AlgComb`, `+`, `-`, `*` |
| Decision procedures (recursive resultant) | `IsZero`, `eq`, `IsPolynomial` |
| Representation simplification / minimization | `SimplifyRep`, `ChangeRing`, `ScaleGenerators` |
