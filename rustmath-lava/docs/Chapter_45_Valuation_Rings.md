# Chapter 45 — Valuation Rings

**Handbook part:** VII — Local Arithmetic Fields
**Handbook pages:** 1229–1231 (PDF pages 1362–1364)

---

## Scope and overview

Chapter 45 describes Magma's support for valuation rings arising from discrete non-Archimedean
valuations. Two families are supported:

1. **Rational valuation rings** — given the rational field **Q** and a finite prime p, the
   valuation ring consists of all rationals r = x/y with p ∤ y (i.e. vp(r) ≥ 0).

2. **Function-field valuation rings** — given the rational function field F = Frac(K[x]) and
   either a monic irreducible polynomial f ∈ K[x] (finite prime) or the infinite prime, the
   valuation ring consists of rational functions g/h with vf(g/h) ≥ 0 (finite case: f ∤ h)
   or deg(h) ≥ deg(g) (infinite case).

The chapter covers construction of valuation rings and their elements, standard ring-structure
queries, arithmetic and comparison operations on elements, and Euclidean-domain functions
(Euclidean norm, division with remainder, GCD, extended GCD).

No chapter bibliography is present; the algorithms for the Euclidean operations follow directly
from the valuation-theoretic definitions.

---

## 45.1 Introduction

Magma currently supports basic operations in valuation rings obtained either from the rational
field **Q** (and a finite prime p), or from a field of rational functions over a field (and an
irreducible polynomial, or the infinite prime).

---

## 45.2 Creation Functions

### 45.2.1 Creation of Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ValuationRing(Q, p)` | Given the rational field **Q** and a rational prime number p, create the valuation ring R corresponding to the discrete non-Archimedean valuation vp. R consists of rational numbers r = x/y ∈ **Q** such that p ∤ y (equivalently, vp(r) ≥ 0). | Direct construction from the definition of vp. |
| `ValuationRing(F, f)` | Given the rational function field F = Frac(K[x]) and a monic irreducible polynomial f ∈ K[x], create the valuation ring R corresponding to vf. R consists of rational functions g/h ∈ F with vf(g/h) ≥ 0, i.e. with f ∤ h. | Direct construction from the definition of vf. |
| `ValuationRing(F)` | Given the rational function field F = Frac(K[x]), create the valuation ring R corresponding to the infinite prime v∞. R consists of g/h ∈ F such that deg(h) ≥ deg(g). | Direct construction from the degree valuation at infinity. |

### 45.2.2 Creation of Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `V ! r` | Given a valuation ring V and an element r of the field of fractions F of V (from which V was created), coerce r into V. Only possible when the valuation on V is non-negative at r; an error occurs otherwise. | Valuation check followed by coercion. |

---

## 45.3 Structure Operations

### 45.3.1 Related Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Category(V)` | The Magma category of the valuation ring V. | — |
| `Parent(V)` | The parent structure of V. | — |
| `PrimeRing(V)` | The prime ring of V. | — |
| `Center(V)` | The center of V. | — |
| `FieldOfFractions(V)` | Returns the field of fractions of the valuation ring V, which is the rational field or the function field from which V was created. | — |

### 45.3.2 Numerical Invariants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Characteristic(V)` | The characteristic of the valuation ring V. | — |

---

## 45.4 Element Operations

### 45.4.1 Arithmetic Operations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `+ v`, `- v` | Unary plus and negation of a valuation ring element v. | — |
| `v + w`, `v - w`, `v * w`, `v ^ k`, `v / w` | Binary arithmetic on valuation ring elements. | — |
| `v +:= w`, `v -:= w`, `v *:= w` | In-place arithmetic assignments. | — |
| `v div w` | The quotient q of the division with remainder v = qw + r of valuation ring elements v and w, where the remainder has valuation less than that of w. If val(v) ≥ val(w), returns the quotient v/w; if val(w) > val(v), returns 0. | Valuation comparison, then exact division or zero. |

### 45.4.2 Equality and Membership

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `v eq w`, `v ne w` | Test equality or inequality of valuation ring elements v and w. | — |
| `v in V`, `v notin V` | Test membership of v in valuation ring V. | — |

### 45.4.3 Parent and Category

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Parent(v)` | The parent valuation ring of element v. | — |
| `Category(v)` | The Magma category of element v. | — |

### 45.4.4 Predicates on Ring Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsZero(n)`, `IsOne(n)`, `IsMinusOne(n)` | Test whether element n is zero, one, or minus one. | — |
| `IsNilpotent(n)`, `IsIdempotent(n)` | Test whether element n is nilpotent or idempotent. | — |
| `IsUnit(n)`, `IsZeroDivisor(n)`, `IsRegular(n)` | Test whether element n is a unit, a zero divisor, or a regular (non-zero-divisor) element. | — |

### 45.4.5 Other Element Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `EuclideanNorm(v)` | The Euclidean norm of an element v of a valuation ring. | Derived from the associated valuation. |
| `Valuation(v)` | Given an element v of a valuation ring V, return the valuation (associated with V) of v. | Direct evaluation of the discrete valuation. |
| `Quotrem(v, w)` | Given two elements v, w of a valuation ring V with associated valuation φ, return a quotient and remainder q and r in V such that v = qw + r and 0 ≤ φ(r) < φ(w). If φ(v) < φ(w), returns q = 0 and r = v; if φ(v) ≥ φ(w), returns q = v/w and r = 0. | Division algorithm in a discrete valuation ring, using the valuation to determine the case. |
| `GreatestCommonDivisor(v, w)`, `Gcd(v, w)` | Returns a greatest common divisor of two elements v, w in a valuation ring V. Returns u·m, where m = min(φ(v), φ(w)) and u is the uniformizing element of V (with φ(u) = 1). | GCD in a DVR is determined by the minimum valuation; the uniformizing element generates the maximal ideal. |
| `ExtendedGreatestCommonDivisor(v, w)`, `Xgcd(v, w)`, `XGCD(v, w)` | Returns a greatest common divisor z ∈ V of v, w together with multipliers x, y ∈ V such that xv + yw = z. The principal return value is z = u·m, where m = min(φ(v), φ(w)) and u is the uniformizing element of V (with φ(u) = 1). | Extended Euclidean algorithm in a DVR; Bezout coefficients derived from the valuation structure. |

---

## 45.5 Bibliography

This chapter contains no bibliography. The operations are based directly on the standard theory
of discrete valuation rings; no external algorithmic references are cited in the handbook text.

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Discrete valuation ring construction (rational prime) | `ValuationRing(Q, p)` |
| Discrete valuation ring construction (irreducible polynomial) | `ValuationRing(F, f)` |
| Discrete valuation ring construction (infinite prime / degree valuation) | `ValuationRing(F)` |
| Valuation evaluation | `Valuation(v)`, `EuclideanNorm(v)` |
| Division with remainder in a DVR | `v div w`, `Quotrem(v, w)` |
| GCD via minimum valuation in a DVR | `GreatestCommonDivisor(v, w)`, `Gcd(v, w)` |
| Extended GCD / Bezout in a DVR | `ExtendedGreatestCommonDivisor(v, w)`, `Xgcd(v, w)`, `XGCD(v, w)` |
