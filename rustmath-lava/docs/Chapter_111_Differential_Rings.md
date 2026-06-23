# Chapter 111 — Differential Rings

**Handbook part:** XIV — Commutative Algebra
**Handbook pages:** 3403–3466 (PDF pages 3532–3601)

---

## Scope and overview

Chapter 111 implements the algebraic infrastructure for **differential Galois theory** — the analogue
of classical Galois theory for linear differential equations — as well as general-purpose arithmetic
with differential operators. The foundational reference for the implementation is **[vdPS03]**
(van der Put and Singer, *Galois Theory of Linear Differential Equations*).

A **differential ring** `F` is a ring equipped with an additive derivation `δ_F : F → F` satisfying
the Leibniz rule `δ_F(a·b) = δ_F(a)·b + a·δ_F(b)`. Differential rings in Magma have type
`RngDiff`; their elements have type `RngDiffElt`. Every differential ring contains a subring of
constants on which the derivation acts as zero. A differential ring that is also a field is called
a **differential field**.

A **differential operator ring** `F[D]` over a differential field `F` is a non-commutative ring
whose elements are expressions `L = aₙDⁿ + … + a₁D + a₀` (with `aᵢ ∈ F`), with multiplication
determined by `D * a = aD + δ_F(a)`. The equation `L(y) = 0` is then the linear differential
equation `aₙδ_F^n(y) + … + a₁δ_F(y) + a₀y = 0`. Differential operator rings have type
`RngDiffOp`; their elements have type `RngDiffOpElt`.

The chapter covers six broad areas:

1. **Differential rings and fields** — creation (rational differential fields, Laurent series rings,
   completions, extensions) and all ring/element operations.
2. **Differential operator rings** — analogous creation, element, and map operations.
3. **Euclidean algorithms, GCDs, LCMs** — right/left Euclidean division, GCRD, GCLD, LCLM for the
   non-commutative operator ring.
4. **Local theory** — singular places, indicial polynomials, Newton polygons, rational solutions.
5. **Factorisation of operators** — coprime index 1 and LCLM factorisation methods over Laurent
   series rings, plus right-hand-factor computation using Riccati factors and semi-regular parts,
   based on **[vH97b]**.
6. **Symmetric powers** — via the algorithm of **[BMW97]**.

---

## 111.1 Introduction

The chapter introduces differential rings, the derivation rule, and the formalisation of linear
differential equations via the non-commutative ring `F[D]`. Readers are directed to **[vdPS03]**
for an introduction to differential Galois theory; that book forms the basis of the implementation.

No intrinsics appear in this introductory section.

---

## 111.2 Differential Rings and Fields

### 111.2.1 Creation

Two principal constructors create differential rings. A general constructor accepts any ring and
a user-specified derivation; a second constructor produces a rational differential field of
transcendence degree 1 whose derivation is specified by a differential form.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DifferentialRing(P, f, C)` | Returns the differential ring isomorphic to `P` with derivation `f : P → P` and constant ring `C` (a subring of `P` on which `f` is zero). | — |
| `RationalDifferentialField(C)` | The differential field in one variable over constant field `C`, with derivation `d/(1)d(F.1)`. `C` must be an exact field with polynomial GCD. | — |
| `DifferentialLaurentSeriesRing(C)` | The differential Laurent series ring in one variable over `C`, with projective derivation `F.1 · d/d(F.1)`. Parameter `Precision`. | — |
| `RingOfFractions(R)` | Returns the differential ring of fractions `R[r⁻¹ : r ∈ R not a zero divisor]` together with the inclusion map. | — |
| `FieldOfFractions(R)` | Returns the differential field of fractions of differential ring `R` and the inclusion map. | — |
| `AssignNames(~R, S)` | Assigns the strings in `S` as names of the indeterminates of `R`. Affects printing only. | — |

*Worked examples: H111E1 (general differential ring from polynomial ring), H111E2 (rational differential field), H111E3 (differential Laurent series ring).*

### 111.2.2 Creation of Differential Ring Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Name(R, i)` / `R . i` | The *i*-th indeterminate of differential ring `R`. | — |
| `R ! s` | Coerce `s` into `R` (coercible if coercible into the underlying ring of `R`). | — |
| `Zero(R)` | The zero element of `R`. | — |
| `One(R)` / `Identity(R)` | The identity element of `R`. | — |
| `SeparatingElement(F)` | Returns the separating element of the algebraic differential field `F`. | — |

*Worked example: H111E4 (elements of Q(z) with derivation d/dz).*

---

## 111.3 Structure Operations on Differential Rings

Differential rings form the Magma category `RngDiff`.

### 111.3.1 Category and Parent

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Category(R)` / `Type(R)` | The category (type) of the differential ring `R`. | — |
| `Parent(R)` | The power structure of `R`. | — |

### 111.3.2 Related Structures

The underlying ring and constant ring from which a differential ring was created, as well as the
base ring of a differential extension, can each be retrieved. If `M/F` is a differential extension,
then `F` is the base ring of `M`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `UnderlyingRing(R)` | The underlying ring of differential ring `R`. | — |
| `UnderlyingField(R)` | The underlying ring of `R`, provided it is a field. | — |
| `BaseRing(R)` | The base ring of `R`. | — |
| `BaseField(R)` | The base ring of `R`, provided it is a field. | — |
| `ConstantRing(R)` | The constant ring of `R` (on which the derivation is zero). | — |
| `ConstantField(R)` | The constant ring of `R`, provided it is a field. | — |
| `ExactConstantField(F)` | The exact constant field of `F` (algebraic closure in `F` of the constant field), together with the inclusion map. `F` must be a function field created with a differential. | — |
| `Generators(R)` | The list of generators of `R` (constructed from the underlying ring if not assigned). | — |

*Worked examples: H111E5 (related structures for Q(z) and an extension Q(z,√2)), H111E6 (related structures for differential Laurent series ring).*

### 111.3.3 Derivation and Differential

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Derivation(R)` | The derivation of the differential ring `R`. | — |
| `Differential(F)` | The differential belonging to the derivation of `F`. `F` must have been constructed so that its derivation is defined by a differential. | — |

*Worked example: H111E7 (derivation and differential of Q(z)).*

### 111.3.4 Numerical Invariants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Ngens(R)` | The number of indeterminates of `R`. | — |

### 111.3.5 Predicates and Booleans

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `R eq F` | True iff differential rings `R` and `F` are the same. | — |
| `IsIdentical(R, F)` | True iff `R` and `F` are identical. | — |
| `IsDomain(R)` | True iff `R` is a domain. | — |
| `IsField(R)` | True iff `R` is a field. | — |
| `IsDifferentialField(R)` | True iff `R` is a differential field. | — |
| `IsAlgebraicDifferentialField(R)` | True iff the field structure of `R` is an algebraic function field. | — |
| `IsDifferentialSeriesRing(R)` | True iff the underlying ring of `R` is a series ring. | — |
| `IsDifferentialLaurentSeriesRing(R)` | True iff the underlying ring of `R` is a Laurent series ring and `R` was created with a known constant ring. | — |
| `HasProjectiveDerivation(F)` | True iff `F` has derivation weakly of the form `(F.1) · d/d(F.1)`. | — |
| `HasZeroDerivation(F)` | True iff the algebraic differential field or differential series ring `F` has zero (or weakly zero) derivation. | — |

*Worked examples: H111E8 (booleans for various differential rings), H111E9 (HasProjectiveDerivation, HasZeroDerivation).*

### 111.3.6 Precision

Applicable to differential series rings.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RelativePrecision(F)` | The relative precision of the underlying series ring of `F`. | — |
| `RelativePrecisionOfDerivation(F)` | For a differential Laurent series ring `F`, the relative precision of the ring derivative of `F.1`. | — |
| `ChangePrecision(F, p)` | Returns the differential series ring isomorphic to `F` with relative precision `p`, and the induced map. | — |

*Worked examples: H111E10 (relative precision), H111E11 (precision of derivation), H111E12 (ChangePrecision).*

---

## 111.4 Element Operations on Differential Ring Elements

### 111.4.1 Category and Parent

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Category(s)` / `Type(s)` | The category (type) of differential ring element `s`. | — |
| `Parent(s)` | The parent of `s`. | — |

### 111.4.2 Arithmetic

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `s + t` | Sum of differential ring elements. | — |
| `-s` | Negation. | — |
| `s - t` | Difference. | — |
| `s * t` | Product. | — |
| `s ^ n` | `n`-th power of `s`; if `s` is invertible, `n` may be negative. | — |
| `s div t` | Exact division of `s` by `t`, if `s` is divisible by `t`. | — |
| `s / t` | Division in a differential field. | — |

### 111.4.3 Predicates and Booleans

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `s eq t` | True iff `s` and `t` are exactly equal. | — |
| `IsZero(s)` | True iff `s` is zero. | — |
| `IsOne(s)` | True iff `s` is the unity element. | — |
| `IsWeaklyEqual(s, t)` | True iff `s` and `t` are weakly equal (in series sense). | — |
| `IsWeaklyZero(s)` | True iff `s` is weakly zero. | — |
| `IsOrderTerm(s)` | True iff `s` is purely an order term of a differential series ring. | — |

*Worked example: H111E13 (booleans for differential ring elements).*

### 111.4.4 Coefficients and Terms

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `O(s)` | The order term of differential series `s`. | — |
| `Truncate(s)` | The known part (truncation) of differential series `s`. | — |
| `Eltseq(s)` | The coefficients of `s`. | — |
| `Exponents(s)` | The interval from the valuation of `s` to (inclusive) the degree of `s`. | — |

*Worked examples: H111E14 (Eltseq for algebraic extension), H111E15 (O, Truncate, Eltseq, Exponents for Laurent series).*

### 111.4.5 Conjugates, Norm and Trace

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `MinimalPolynomial(s)` | The minimal polynomial of differential field element `s` over the base field. | — |

*Worked example: H111E16 (minimal polynomial in a quadratic extension).*

### 111.4.6 Derivatives and Differentials

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Derivative(s)` | The image of `s` under the derivation of its parent. | — |
| `Differential(s)` | The differential of `s` in the algebraic differential field, as a differential in the differential space of the underlying ring. | — |

*Worked example: H111E17 (derivative and differential in Q(z) and Laurent series ring).*

---

## 111.5 Changing Related Structures

Functions to alter the derivation, differential, constant ring, or precision of an existing
differential ring or field.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ChangeDerivation(R, f)` | Returns a differential ring isomorphic to `R` with derivation `f · Derivation(R)` (`f` nonzero), and the isomorphism. Same underlying ring. | — |
| `ChangeDifferential(F, df)` | Returns the algebraic differential field with the same underlying ring as `F` but derivation with respect to differential `df`, together with the bijective map. | — |
| `ConstantFieldExtension(F, C)` | Returns the differential field isomorphic to `F` (algebraic function field) with constant field extended to `C`, and the isomorphism. | — |
| `Completion(F, p)` | The completion of differential field `F` with respect to place `p`, with naturally induced derivation, together with the embedding map. Parameter `Precision` (default ∞). | — |

*Worked examples: H111E18 (ChangeDerivation on Q(z)), H111E19 (ChangeDifferential), H111E20–H111E21 (ConstantFieldExtension for algebraic field and Laurent series ring), H111E22–H111E23 (Completion at a place, including genus-1 example).*

---

## 111.6 Ring and Field Extensions

Extensions induced by a differential operator or by an irreducible polynomial. Given
`L = aₙDⁿ + … + a₀ ∈ F[D]`, one constructs a ring/field extension of degree `n` whose
indeterminates are formal solutions of `L(y) = 0` and their derivatives.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DifferentialRingExtension(L)` | Constructs a differential ring extension `P = F[Y₁,…,Yₙ]` of the base ring of `L` by adding formal solution `Y₁` and its derivatives `Y₂,…,Yₙ` as indeterminates. Derivation: `δ_P(Yᵢ) = Yᵢ₊₁` for `i < n`; `aₙδ_P(Yₙ) = −aₙ₋₁Yₙ₋₁ − … − a₁Y₁`. | — |
| `DifferentialFieldExtension(L)` | As `DifferentialRingExtension(L)` but returns a differential field `M = F(Y₁,…,Yₙ)`. | — |
| `ext< F \| f >` | The differential field extension `F(α)` where `α` is a root of irreducible polynomial `f` over `F`. | — |
| `ExponentialFieldExtension(F, f)` | Returns differential field extension `F(E)` such that `δ(E) = f · E` (`f ∈ F`). | — |
| `LogarithmicFieldExtension(F, f)` | Returns differential field extension `F(L)` such that `δ(L) = f` (`f ∈ F`). | — |
| `PurelyRamifiedExtension(f)` | Creates a purely ramified field extension `M` of differential field `F` w.r.t. polynomial `f = Xⁿ − a·(F.1)` (constant `a`, positive integer `n`). Works for algebraic differential fields and differential Laurent series rings. Relative precision of `M` is `n` times that of `F`. Returns the field and the embedding (with partial inverse). | — |

*Worked examples: H111E24 (DifferentialRingExtension), H111E25 (DifferentialFieldExtension), H111E26 (ext< > for algebraic extension), H111E27 (ExponentialFieldExtension, LogarithmicFieldExtension), H111E28–H111E30 (PurelyRamifiedExtension for algebraic fields and Laurent series rings).*

---

## 111.7 Ideals and Quotient Rings

A differential ideal `I ⊆ R` is an ideal of `R` closed under the derivation. Restricted to
differential rings whose underlying rings are multivariate polynomial rings.

### 111.7.1 Defining Ideals and Quotient Rings

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DifferentialIdeal(L)` | Given a sequence `L` with entries in a differential ring `R` (underlying ring of type `RngMPol`), returns the differential ideal generated by the entries. Derivatives of generators are added as needed to close under derivation. | — |
| `QuotientRing(R, I)` | Given differential ring `R` and differential ideal `I`, returns the differential quotient ring `Q = R/I` (with induced derivation) and the quotient map. | — |

*Worked example: H111E31 (differential ideal and quotient ring).*

### 111.7.2 Boolean Operations on Ideals

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsDifferentialIdeal(R, I)` | Returns true iff `I` is a differential ideal of `R`. | — |

---

## 111.8 Wronskian Matrix

The Wronskian matrix of elements `y₁,…,yₙ ∈ R` is the `n × n` matrix `W` with `W[i,j] = δ_R^{i-1}(yⱼ)`. Its determinant is the Wronskian.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `WronskianMatrix(L)` | Given a sequence `L` of differential ring elements, returns the Wronskian matrix, with entries in the universe of `L`. | — |
| `WronskianDeterminant(L)` | Returns the Wronskian determinant and the Wronskian matrix of `L`. | — |

*Worked example: H111E32 (Wronskian of `{1, z, z²}` and `{z, z², 1/z}`).*

---

## 111.9 Differential Operator Rings

### 111.9.1 Creation

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DifferentialOperatorRing(F)` | Returns the differential operator ring over the differential field `F`. Magma category `RngDiffOp`. | — |
| `AssignNames(~R, S)` | Assigns strings in `S` as names for the indeterminates of `R`. Affects printing only. | — |

*Worked example: H111E33 (creating the differential operator ring).*

### 111.9.2 Creation of Differential Operators

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Name(R, i)` / `R . i` | The *i*-th indeterminate of `R` (must be 1). | — |
| `R ! s` | Coerce `s` into `R`. Coercible: elements of the underlying ring, sequences, differential operators over the base ring, or (when the base ring is algebraic) operators from other algebraic differential operator rings with the same underlying ring. | — |
| `Zero(R)` | The zero element of `R`. | — |
| `One(R)` | The identity element of `R`. | — |

*Worked example: H111E34 (creating operators; coercion between operator rings with different derivations).*

---

## 111.10 Structure Operations on Differential Operator Rings

Differential operator rings form the Magma category `RngDiffOp`.

### 111.10.1 Category and Parent

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Category(R)` / `Type(R)` | The category (type) of differential operator ring `R`. | — |
| `Parent(R)` | The power structure of `R`. | — |

### 111.10.2 Related Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BaseRing(R)` / `CoefficientRing(R)` | The base ring (coefficient ring) `F` of `R = F[D]`. | — |
| `ConstantRing(R)` | The constant ring of `R`. | — |

### 111.10.3 Derivation and Differential

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Derivation(R)` | The derivation of `R` (which is `δ_F`). | — |
| `Differential(R)` | The differential belonging to the derivation of `R`, if the derivation was constructed from a differential. | — |

*Worked example: H111E35 (BaseRing, Derivation, Differential of an operator ring).*

### 111.10.4 Predicates and Booleans

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `R eq F` | True iff `R` and `F` are the same differential operator rings. | — |
| `IsIdentical(R, F)` | True iff `R` and `F` are identical. | — |
| `IsDifferentialOperatorRing(R)` | True iff `R` is a differential operator ring. | — |
| `HasProjectiveDerivation(R)` | True iff `R` is defined over a ring with derivation weakly of the form `(F.1) · d/d(F.1)`. | — |
| `HasZeroDerivation(R)` | True iff the base ring of `R` has (weakly) zero derivation. | — |

*Worked examples: H111E36 (predicates for algebraic base), H111E37 (predicates for Laurent series base).*

### 111.10.5 Precision

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RelativePrecisionOfDerivation(R)` | The relative precision of the derivation of an operator ring over a Laurent series ring. | — |

*Worked example: H111E38 (relative precision of derivation).*

---

## 111.11 Element Operations on Differential Operators

### 111.11.1 Category and Parent

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Category(L)` / `Type(L)` | The category (type) of differential operator `L`. | — |
| `Parent(L)` | The parent of `L`. | — |

### 111.11.2 Arithmetic

Multiplication is **non-commutative** (determined by `D * a = aD + δ_F(a)`).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `s + t` | Sum. | — |
| `-s` | Negation. | — |
| `s - t` | Difference. | — |
| `s * t` | Product (non-commutative). | — |
| `s ^ n` | `n`-th power (`n ≥ 0`). | — |

*Worked example: H111E39 (non-commutative multiplication, powers).*

### 111.11.3 Predicates and Booleans

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `s eq t` | True iff `s` and `t` are exactly equal. | — |
| `IsZero(L)` | True iff `L` is zero. | — |
| `IsOne(L)` | True iff `L` is unity. | — |
| `IsMonic(L)` | True iff `L` is monic. | — |
| `IsWeaklyEqual(L, P)` | True iff all coefficients of `L` and `P` are weakly equal. | — |
| `IsWeaklyZero(L)` | True iff `L` is weakly equal to 0. | — |
| `IsWeaklyMonic(L)` | True iff the leading coefficient of `L` is weakly equal to 1. | — |

### 111.11.4 Coefficients and Terms

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Eltseq(L)` / `Coefficients(L)` | The sequence of coefficients of `L`, from constant term to leading term. | — |
| `Coefficient(L, i)` | The coefficient of `Dⁱ` in `L`. | — |
| `LeadingCoefficient(L)` | The leading (highest-order) coefficient of `L`. | — |
| `LeadingTerm(L)` | The leading term of `L`. | — |
| `Terms(L)` | Sequence of non-zero terms of `L`, ordered from lowest to highest order. | — |

*Worked example: H111E40 (Eltseq, LeadingTerm, Terms).*

### 111.11.5 Order and Degree

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Order(L)` / `Degree(L)` | The order of `L`. Defined to be −1 if `L` is identically 0. | — |
| `WeakOrder(L)` / `WeakDegree(L)` | Over a differential series ring: the exponent of the highest coefficient that is not weakly 0. | — |

*Worked example: H111E41 (Order, Degree, WeakOrder over Laurent series).*

### 111.11.6 Related Differential Operators

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `MonicDifferentialOperator(L)` | Returns `(1/c) · L` where `c` is the leading coefficient of `L`. | — |
| `Adjoint(L)` | The formal adjoint `L* = Σᵢ (−1)ⁱ Dⁱ * aᵢ` of `L = Σᵢ aᵢDⁱ`. The adjoint has the same order as `L` with leading coefficient `(−1)ⁿaₙ`. | — |
| `Translation(L, e)` | The operator obtained by replacing `R.1` by `R.1 + e` in `L`, together with the translation map. | — |
| `TruncateCoefficients(L)` | Over a differential series ring: the operator whose coefficients are the truncations of those of `L`. | — |

*Worked examples: H111E42 (MonicDifferentialOperator, Adjoint, Translation), H111E43 (TruncateCoefficients).*

### 111.11.7 Application of Operators

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Apply(L, f)` / `L(f)` / `f @ L` | Applies differential operator `L` to ring element `f`, returning the element `aₙδⁿ(f) + … + a₁δ(f) + a₀f` in the base ring of `L`. `f` must be coercible into the base ring. | — |

*Worked example: H111E44 (Apply for `D² − 2/z²` on `z` and `z²`).*

---

## 111.12 Related Maps

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `TranslationMap(R, e)` | Returns the map on differential operator ring `R` that replaces `R.1` by `R.1 + e`. | — |
| `LiftMap(m, R)` | For a differential map `m : F → M` on differential fields and operator ring `R` over `F`, lifts `m` to a map `R → S` where the base field of `S` is `M`. | — |

*Worked example: H111E45 (TranslationMap, LiftMap).*

---

## 111.13 Changing Related Structures

Functions to change the derivation, differential, constant field, or completion/localisation of a
differential operator ring.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ChangeDerivation(R, f)` | Returns a differential operator ring isomorphic to `R` with derivation `f * Derivation(R)` (`f` nonzero), and the isomorphism. The base ring gets `ChangeDerivation` applied as well. | — |
| `ChangeDifferential(R, df)` | Returns the differential operator ring with differential `df` (underlying base field ring unchanged), and the bijective map. | — |
| `ConstantFieldExtension(R, C)` | Operator ring with base ring isomorphic to that of `R` but with constant field `C`; derivation extended over `C`. Returns the new ring and the map. | — |
| `PurelyRamifiedExtension(R, f)` | Operator ring over the purely ramified extension of the base ring of `R` induced by polynomial `f = Xⁿ − a·(F.1)`. | — |
| `Completion(R, p)` | Operator ring `R̃` whose base ring is the completion of that of `R` at place `p`; returns `R̃` and the natural embedding. Parameter `Precision` (default ∞). | — |
| `Localization(R, p)` | Operator ring with derivation `t·d/dt` (where `t` is the uniformizer at place `p`), differential of valuation −1 at `p`. Returns the ring, the natural map, and the induced image of `p`. | — |
| `Localization(L, p)` | Localised operator of `L` at place `p`, plus the embedding map and induced image of `p`. | — |
| `Localization(R)` | For `R` over a differential Laurent series ring `C((t))`: the operator ring with derivation `t·d/dt`, and the natural map. | — |
| `Localization(L)` | Localised operator of `L` over a differential series ring, and the embedding map. | — |

*Worked examples: H111E46 (ChangeDifferential on operator ring), H111E47 (PurelyRamifiedExtension), H111E48 (ChangeDerivation, ConstantFieldExtension on operator rings), H111E49 (Completion on operator ring), H111E50 (Localization at a place and of an operator).*

---

## 111.14 Euclidean Algorithms, GCDs and LCMs

The differential operator ring shares many properties with a univariate polynomial ring, but
multiplication is non-commutative. GCD and LCM algorithms must therefore specify whether
multiplication is on the left or the right.

### 111.14.1 Euclidean Right and Left Division

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `EuclideanRightDivision(N, D)` | Returns `Q, R` with `N = Q·D + R` and `Degree(R) < Degree(D)`. Errors if `D = 0`. | Euclidean (right) division in `F[D]`. |
| `EuclideanLeftDivision(D, N)` | Returns `Q, R` with `N = D·Q + R` and `Degree(R) < Degree(D)`. Errors if `D = 0`. | Euclidean (left) division in `F[D]`. |

*Worked example: H111E51 (right and left Euclidean division).*

### 111.14.2 Greatest Common Right and Left Divisors

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `GreatestCommonRightDivisor(A, B)` / `GCRD(A, B)` | The unique monic operator generating the left ideal `RA + RB`. | Right-Euclidean algorithm in `F[D]`. |
| `ExtendedGreatestCommonRightDivisor(A, B)` | Returns `G, U, V` with `U·A + V·B = G`; `G` is the monic GCRD. | Extended right-Euclidean algorithm. |
| `GreatestCommonLeftDivisor(A, B)` / `GCLD(A, B)` | The unique monic operator generating the right ideal `AR + BR`. | Left-Euclidean algorithm in `F[D]`. |
| `ExtendedGreatestCommonLeftDivisor(A, B)` | Returns `G, U, V` with `A·U + B·V = G`; `G` is the monic GCLD. | Extended left-Euclidean algorithm. |

*Worked example: H111E52 (GCRD, extended GCRD, GCLD).*

### 111.14.3 Least Common Left Multiples

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LeastCommonLeftMultiple(L)` | For monic degree-1 `L = D − r` in `R = F[D]`: the LCLM of `L` and all its conjugates over the base ring of `F`. | — |
| `LeastCommonLeftMultiple(A, B)` / `LCLM(A, B)` | The unique monic operator generating the left ideal `RA ∩ RB`; order at most `Order(A) + Order(B)`. | — |
| `ExtendedLeastCommonLeftMultiple(A, B)` | Returns `L, U, V` with `L = U·A = V·B`; `L` is the monic LCLM. | — |
| `ExtendedLeastCommonLeftMultiple(S)` | For non-empty sequence `S`: the monic LCLM `L` and sequence `Q` with `L = Q[i]·S[i]` for all `i`. | — |

*Worked examples: H111E53 (LCLM, extended LCLM, ExtendedLeastCommonLeftMultiple with sequence), H111E54 (LeastCommonLeftMultiple of conjugate operators).*

---

## 111.15 Related Matrices

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CompanionMatrix(L)` | The companion matrix of monic `L = Dⁿ + aₙ₋₁Dⁿ⁻¹ + … + a₀`: the `n × n` matrix with 1s on the super-diagonal and `−a₀, …, −aₙ₋₁` in the last row. | — |

*Worked example: H111E55 (CompanionMatrix of a degree-3 operator).*

---

## 111.16 Singular Places and Indicial Polynomials

All functions in this section apply to differential operator rings over function fields of
transcendence degree 1, with derivation defined by a differential. A place `(q)` is **regular**
for `L` if none of the coefficients of the localised operator `L̃` (with `L̃` having differential
of valuation 0 at `(q)`) have negative valuation at `(q)`. A singular place is **regular singular**
if the coefficient of `D̃ⁱ` in `L̃` has valuation ≥ `i − n` for all `i`; otherwise it is
**irregular singular**.

### 111.16.1 Singular Places

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsRegularPlace(L, p)` | True iff place `p` is a regular place of `L`. Requires derivation defined by a differential. | Definition via localized operator valuations. |
| `IsRegularSingularPlace(L, p)` | True iff `p` is a regular singular place. | Definition via localized operator valuations. |
| `IsIrregularSingularPlace(L, p)` | True iff `p` is an irregular singular place. | Definition via localized operator valuations. |
| `SetsOfSingularPlaces(L)` | Returns two sets: the regular singular places and the irregular singular places of `L`. Requires derivation defined by a differential. | Definition via localized operator valuations. |
| `IsFuchsianOperator(L)` | True iff all singular places are regular singular. Also returns the set of all singular places (when Fuchsian). Requires differential derivation. | Checks `SetsOfSingularPlaces`. |
| `IsRegularSingularOperator(L)` | True iff `L` is regular singular. Works also over differential Laurent series rings (regular singular at `F.1`). If derivation is by a differential, returns `IsFuchsianOperator` values. | Definition-based check or Fuchsian test. |

*Worked examples: H111E56 (singular places of a Fuchsian and a non-Fuchsian operator), H111E57 (IsRegularSingularOperator over Laurent series).*

### 111.16.2 Indicial Polynomials

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IndicialPolynomial(L, p)` | The monic indicial polynomial of `L` at place `p`. Requires differential derivation; base ring must have one generator. Definition per §4.1 of **[vdPS03]**. | Indicial polynomial computation per **[vdPS03, §4.1]**. |

*Worked example: H111E58 (indicial polynomials of a Fuchsian operator at the three singular places).*

---

## 111.17 Rational Solutions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RationalSolutions(L)` | A basis of the nullspace of rational solutions of `L(y) = 0` in `F`, as a sequence. Requires differential derivation. | Algorithm of **[vdPS03, §4.1]**. |
| `HasRationalSolutions(L, g)` | For `L` with coefficients in `F` and `g ∈ F`: returns `true` if there exists `y ∈ F` with `L(y) = g`, plus a particular solution and the null-space basis if so; otherwise `false`. Requires differential derivation. | Algorithm of **[vdPS03, §4.1]**. |

*Worked example: H111E59 (RationalSolutions and HasRationalSolutions).*

---

## 111.18 Newton Polygons

The Newton polygon of a differential operator `L` with projective derivation `z · d/dz` is
defined in §3 of **[vH97b]**. For operators over function fields, the Newton polygon at place `(p)`
is the Newton polygon at `t = 0` after rewriting `L` with local parameter `t` and derivation
`t · d/dt`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `NewtonPolygon(L)` | Newton polygon of `L` over a differential Laurent series ring (rewriting to have derivation `t·d/dt` as needed). Also returns the rewritten operator `L̃`. | Newton polygon construction per **[vH97b, §3]**. |
| `NewtonPolygon(L, p)` | Newton polygon of `L` at place `p`. Derivation must be by a differential; base ring must have one generator. Returns the polygon and rewritten operator. | Newton polygon at a place per **[vH97b, §3]**. |
| `NewtonPolynomial(F)` | The Newton polynomial of face `F` of a Newton polygon (polygon must have been created for a differential operator). Well-defined up to scalar multiplication. Definition per **[vH97b, §3]**. | Newton polynomial per **[vH97b, §3]**. |
| `NewtonPolynomials(L)` | All Newton polynomials of `L` (one per face), and the corresponding slopes. | Newton polygon per **[vH97b, §3]**. |

*Worked examples: H111E60 (Newton polygon at a finite place), H111E61 (Newton polygon at the place at infinity), H111E62 (Newton polygon over Laurent series, corresponding to examples in [vdPS03, §3.46 and §3.49.2]).*

---

## 111.19 Symmetric Powers

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SymmetricPower(L, m)` | The `m`-th symmetric power of `L` (monic where possible). Degree at most `C(n+m−1, n−1)` where `n = Order(L)`. | Algorithm of **[BMW97]**. |

*Worked example: H111E63 (symmetric powers of `D²` and `D³ − 1`).*

---

## 111.20 Differential Operators of Algebraic Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DifferentialOperator(f)` | For irreducible `f(X) ∈ F[X]` (`F` a differential field): the monic differential operator over `F` of minimal degree to which a formal root of `f` is a solution. The algorithm uses the induced derivation on a root `g` via `f(g) = 0` to find the minimal linear relation among `g, δ(g), …, δⁿ(g)`. | Straightforward: constructs the algebraic extension `F(g)`, computes derivatives, finds minimal linear relation. |

*Worked example: H111E64 (differential operator for a cube root).*

---

## 111.21 Factorisation of Operators over Differential Laurent Series Rings

Factorisation of operators in `P[δ] := k((t))[δ]` with projective derivation `δ = t · d/dt`. The
factor structure relates to the Newton polygon: by Malgrange's theorem, an operator is reducible
if its Newton polygon has at least two slopes. Two complementary algorithms are implemented,
both based on **[vH97b]**: the **coprime index 1** method and the **LCLM** method.

The approach: for each distinct irreducible factor of the Newton polynomial of `f`, a local
lifting procedure (with respect to the slope valuation metric) computes a right hand factor
to a prescribed precision. No intermediate differential field extensions of `k((t))` are used.
The LCLM algorithm produces right hand factors whose LCLM is exactly `f` (up to precision);
the coprime index 1 algorithm may give factors whose LCLM only divides `f`.

For operators that cannot be factored by the above (e.g., Newton polynomial is a perfect power),
`RightHandFactors` uses **Riccati factors** and **semi-regular parts** (following **[vH97b, §5.1]**),
potentially over field extensions.

### 111.21.1 Slope Valuation of an Operator

The slope valuation of monomial `ctⁱδʲ` with respect to rational slope `s = n/d` (gcd = 1, `d > 0`) is `id − jn`. The slope valuation of operator `L` is the minimum over its nonzero monomials.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SlopeValuation(L, s)` | The slope valuation of `L` with respect to rational slope `s`, when the derivation of `L` is projective. Infinite if `L = 0`. | Definition-based; applicable to projective derivation rings. |

*Worked example: H111E65 (slope valuations for slopes 0, 1/2, 5).*

### 111.21.2 Coprime Index 1 and LCLM Factorisation

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Factorisation(L)` / `Factorization(L)` | Returns a sequence `M` of operator pairs `[A, B]` with `L = A·B`, where `B` has no non-trivial coprime index 1 or LCLM factorisation. A second sequence indicates whether each right factor `M[i][2]` is undisputedly irreducible. Parameters: `Precision` (lifting accuracy; default = relative precision of base ring), `Algorithm` (`"Default"`, `"LCLM"`, or `"CoprimeIndexOne"`). | Coprime index 1 / LCLM factorisation per **[vH97b]** with slope-valuation lifting. |

*Worked examples: H111E66 (operator with infinitely many factorisations; canonical representative), H111E67 (Example 3.46 from [vdPS03]), H111E68 (Example 3.49 from [vdPS03]), H111E69 (unfactorable Newton polynomial; square Newton polynomial), H111E70 (operator not fully recovered by factorisation), H111E71 (rational slope 1/5; effect of precision on coefficients).*

### 111.21.3 Right Hand Factors of Operators

A Riccati factor of `L ∈ k((t))[δ]` is a monic irreducible right hand factor `L̃ = δ̃ − r(t̃)` of `L`
in some field extension `k̃((t̃))[δ̃]`. The LCLM of all conjugates of `L̃` under the Galois group
of `k̃((t̃))/k((t))` yields a monic irreducible right hand factor of `L` over the original field.
Semi-regular parts `Rₑ(L)` yield further right hand factors, possibly over finite field extensions.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RightHandFactors(L)` | The canonical list of monic right hand factors of `L`, one per slope of the Newton polynomial. Second return: Boolean sequence indicating whether each factor is undisputedly irreducible. Parameter `Precision` (minimum absolute precision of coefficients; default −1). | Riccati factors via **[vH97b, §5.1]** and semi-regular parts; LCLM of Galois conjugates. |

*Worked examples: H111E72 (Example 3.49 from [vdPS03]; same factors as H111E68), H111E73 (Example 3.52 from [vdPS03]; handbook answer erroneous), H111E74 (square Newton polynomial; succeeds where Factorisation did not), H111E75 (Example 3.53 from [vdPS03]; illustrating Riccati factor computation), H111E76 (operators with Newton polynomial (T²+1)(T−1)(T+1); degrees 1, 1, 2), H111E77 (main example from [vH97b]; degree-9 operator).*

---

## 111.22 Bibliography

| Key | Reference |
|-----|-----------|
| **[BMW97]** | Manuel Bronstein, Thom Mulders, and Jacques-Arthur Weil. On symmetric powers of differential operators. In *Proceedings of the 1997 International Symposium on Symbolic and Algebraic Computation (Kihei, HI)*, pages 156–163 (electronic), New York, 1997. ACM. |
| **[vdPS03]** | Marius van der Put and Michael F. Singer. *Galois theory of linear differential equations*, volume 328 of *Grundlehren der Mathematischen Wissenschaften*. Springer-Verlag, Berlin, 2003. |
| **[vH97a]** | Mark van Hoeij. Factorization of differential operators with rational functions coefficients. *J. Symbolic Comput.*, 24(5):537–561, 1997. |
| **[vH97b]** | Mark van Hoeij. Formal solutions and factorization of differential operators with power series coefficients. *J. Symbolic Comput.*, 24(1):1–30, 1997. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Differential Galois theory (foundational framework) **[vdPS03]** | All functions in §§111.2–111.7, §§111.9–111.13 |
| Rational solutions of `L(y) = 0` **[vdPS03, §4.1]** | `RationalSolutions`, `HasRationalSolutions` |
| Indicial polynomial **[vdPS03, §4.1]** | `IndicialPolynomial` |
| Newton polygon construction **[vH97b, §3]** | `NewtonPolygon`, `NewtonPolynomial`, `NewtonPolynomials` |
| Symmetric powers of operators **[BMW97]** | `SymmetricPower` |
| Coprime index 1 factorisation **[vH97b]** | `Factorisation`/`Factorization` (Algorithm := "CoprimeIndexOne") |
| LCLM factorisation **[vH97b]** | `Factorisation`/`Factorization` (Algorithm := "LCLM") |
| Riccati factors and semi-regular parts **[vH97b, §5.1]** | `RightHandFactors` |
| Factorisation over rational function fields **[vH97a]** | (referenced context for `RightHandFactors`; canonical right-hand factors used in rational case) |
