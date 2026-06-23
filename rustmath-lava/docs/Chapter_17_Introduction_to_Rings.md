# Chapter 17 — Introduction to Rings

**Handbook part:** III — Basic Rings
**Handbook pages:** 259–275 (PDF pages 390–409)

---

## Scope and overview

Rings of various kinds form the richest source of algebraic structures in Magma. The chapter
surveys the principal ring and field types available (Tables 1 and 2 in the handbook), explains
the underlying ring-theoretic constructions (fractions, quotients, transcendental/algebraic
extension, completion), and then establishes the generic intrinsics that apply to *every* ring
and ring element in the system. Subsequent chapters cover the individual ring categories.

All rings in Magma can be built from Z by repeated application of a small set of fundamental
constructions: field of fractions, quotients (giving Z/mZ), transcendental extension (polynomial
rings), algebraic extension (transcendental extension followed by quotient), and completion
(leading to p-adic and power-series rings). Magma supports most of these via the `quo`, `ext`,
`loc`, and `comp` constructors, analogous to how `sub` creates sub-structures.

An important practical distinction: the *mathematical* properties of a ring and the properties
*Magma is aware of* may differ. Creating `IntegerRing(p)` for prime p yields a residue-class
ring object, not a finite field; field-extension operations will not be available. Similarly,
`Q[X]/(f)` produces a generic quotient ring rather than the full number-field type obtainable
from `NumberField(f)`.

Rings in Magma may be non-commutative (matrix rings, finitely presented algebras, and
polynomial rings over non-commutative coefficient rings) or non-unital (created via `sub`),
though most of the rings in Tables 1 and 2 are commutative and unital.

---

## 17.1 Overview

This section presents the two master tables of ring and field types:

- **Table 1 — Main ring types:** Z (`RngInt`, Ch. 18), Z/mZ (`RngIntRes`, Ch. 18),
  R[x] (`RngUPol`, Ch. 23), F[x]/f(x) (`RngUPolRes`, Ch. 23),
  R[x₁,…,xₘ] (`RngMPol`, Ch. 24), R[x₁,…,xₘ]^G (`RngInvar`, Ch. 110),
  R[[x]] (`RngSer`, Ch. 49), orders in number fields (`RngOrd`, Ch. 37),
  orders in function fields (`RngFunOrd`, Ch. 42),
  Zp (`RngPad`, Ch. 47), local rings (`RngLoc`, Ch. 47), valuation rings (`RngVal`, Ch. 45).

- **Table 2 — Main field types:** Q (`FldRat`, Ch. 20), Fq (`FldFin`, Ch. 21),
  F(x₁,…,xₘ) (`FldFunRat`, Ch. 41), F((x)) (`RngSerLaur`, Ch. 49),
  Q(√D) (`FldQuad`, Ch. 35), Q(ζₙ) (`FldCyc`, Ch. 36),
  Q(α) (`FldNum`, Ch. 34), F(x)(α) (`FldFun`, Ch. 42),
  Qp (`FldPad`, Ch. 47), local fields (`FldLoc`, Ch. 47),
  R (`FldRe`, Ch. 25), C (`FldCom`, Ch. 25).

*(No intrinsics are defined in this section.)*

---

## 17.2 The World of Rings

### 17.2.1 New Rings from Existing Ones

Describes how all rings in Table 1 arise from Z by repeated application of five fundamental
constructions: field of fractions, quotients, transcendental extension, algebraic extension
(transcendental + quotient), and completion. Notes that `exact` rings are those not requiring
approximations (excludes real/complex fields and p-adic/power-series types).

*(No intrinsics are defined in this subsection.)*

### 17.2.2 Attributes

*(This subsection has no standalone content or intrinsics in the handbook text.)*

---

## 17.3 Coercion

A ring element can often be coerced into a ring other than its parent. This is needed for
binary operations on elements from different structures, or when an intrinsic is invoked on
elements for which it has not been directly defined. The basic principle: coercion may be
performed whenever it is mathematically meaningful.

Two types of coercion are distinguished:
- **Automatic coercion** — Magma determines the unique target structure and an obvious
  homomorphism automatically (depends on the structures alone, not the element value).
- **Forced coercion** — the user specifies the target ring `R` via `R ! x`; success may
  depend on the particular element.

The `in` operator returns `true` if and only if forced coercion of an element into a ring
will succeed.

### 17.3.1 Automatic Coercion

Automatic coercion occurs only when there exists a *unique* target structure and an obvious
homomorphism from the parent to that target. In particular, there is always a natural
ring homomorphism Z → R for any ring R, so any integer can be coerced automatically into
any ring. **Table 3** in the handbook gives the full matrix of automatic coercions; key
entries include:

- Finite field elements (Fp^s) coerce only if one field is contained in the other or both
  share a common overstructure (symbol `⊂`).
- Elements of Z coerce automatically into any ring in the table (including polynomial
  rings R[X₁,…,Xₙ] and matrix rings Mₙ,ₙ(R) as the appropriate scalar).
- Q elements coerce into Q, Q(√Δ), Q(ζₙ), number fields L, Rₙ, and Cₙ.
- Cyclotomic field Q(ζₘ) × Q(ζₙ) → Q(ζ_lcm(m,n)).
- `-` denotes no automatic coercion (e.g. finite field elements with Z/nZ elements).
- `=` denotes coercion only when both structures are identical.

For polynomial rings: an element s from S is automatically coercible into R[X₁,…,Xₙ]
only if S = R[X₁,…,Xᵢ] for some i ≤ n, or s ∈ R (the coefficient ring itself). For matrix
rings Mₙ,ₙ(R): a scalar automatically coercible into R becomes the corresponding diagonal
matrix.

### 17.3.2 Forced Coercion

**Table 4** in the handbook details non-automatic (forced) coercions via `!`. Additional symbols:
- `+` — coercion always possible without restriction.
- `|`, `=` — conditions on ring parameters (e.g. Fp^s → Z/mZ requires s = 1 and m = p).
- `∋` — coercion applies only to certain elements (e.g. Q → Z only for integer-valued rationals).
- `or` — coercion is possible either if a parameter condition holds or on a subset of the domain.

Forced coercion rules for polynomial rings (steps tried in order): (a) identity if already in P;
(b) sequence coercion building Σ s[j] Xₙʲ⁻¹; (c) constant polynomial if s is in the
coefficient ring; (d) lift from R[X₁,…,Xₖ] for k ≤ n; (e) project down if s is constant in
Xₙ₊₁,…,Xₖ for k > n.

For matrix rings: s can be forced into Mₙ,ₙ(R) if it coerces into R (→ diagonal matrix), or
if s ∈ Mₙ,ₙ(R′) with R′ coercible into R, or if s is a sequence of n² elements coercible into R.
Elements of Mₙ,ₙ(R) can only be coerced out if n = 1.

Two-step forced coercion is possible even when direct coercion is not: `L ! (Q ! x)` can
move a rational-valued number-field element into another number field via the rationals.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `R ! x` | Force-coerce element `x` into ring `R`. If `x` is an integer, always succeeds, returning `x · 1_R`. | Follows Table 4 rules; depends on both the structure and the element value. |

---

## 17.4 Generic Ring Functions

The generic functions in this section apply in principle to every type of ring in Magma. For
some ring classes an algorithm to compute a given function may not be implemented, in which
case an error results.

### 17.4.1 Related Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Parent(R)` | The parent of ring `R`. Currently returns the power structure of the ring. | — |
| `Category(R)` / `Type(R)` | The Magma category to which ring `R` belongs. `ListCategories()` lists all categories. | — |
| `PrimeField(F)` | For a field `F`: returns Fp if the characteristic p > 0, or Q if characteristic 0. For extension fields, returns the field at the bottom of the extension tower. | — |
| `PrimeRing(R)` | For a unitary ring `R`: returns Z/nZ if the characteristic n > 0, or Z if characteristic 0. For extension rings, returns the ring at the bottom of the extension tower. | — |
| `Centre(R)` / `Center(R)` | The centre of ring `R`: the subring of all elements that commute with every element of `R`. | — |

### 17.4.2 Numerical Invariants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Characteristic(R)` | The characteristic of ring `R`: the smallest positive integer m such that m · r = 0 for every r ∈ R, or 0 if no such m exists. | — |
| `#R` | The cardinality of ring `R`; `R` must be finite. | — |

### 17.4.3 Predicates and Boolean Operations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsCommutative(R)` | `true` if `R` is known commutative; `false` if known non-commutative; error if unknown. | — |
| `IsUnitary(R)` | `true` if `R` has a multiplicative identity; `false` if `R` has no 1. | — |
| `IsFinite(R)` | `true` if `R` is known finite; `false` if known infinite; error if unknown. | — |
| `IsOrdered(R)` | `true` if `R` has a total ordering on its elements; `false` otherwise. | — |
| `IsField(R)` | `true` if `R` is known to be a field; `false` if known not a field; error if unknown. | — |
| `IsDivisionRing(R)` | `true` if `R` is known to be a division ring (every non-zero element invertible); `false` if known not; error if unknown. | — |
| `IsEuclideanDomain(R)` | `true` if `R` is known to be a Euclidean domain; `false` if known not; error if unknown. | — |
| `IsEuclideanRing(R)` | `true` if `R` is known to be Euclidean; `false` if known not; error if unknown. | — |
| `IsMagmaEuclideanRing(R)` | `true` iff `R` is a computable Euclidean ring within Magma (i.e., the necessary Euclidean operations are implemented and algorithms requiring a Euclidean ring will work). | — |
| `IsPID(R)` / `IsPrincipalIdealDomain(R)` | `true` if `R` is known to be a principal ideal domain; `false` if known not; error if unknown. | — |
| `IsPIR(R)` / `IsPrincipalIdealRing(R)` | `true` if `R` is known to be a principal ideal ring; `false` if `R` is known to have non-principal ideals; error if unknown. | — |
| `IsUFD(R)` / `IsUniqueFactorizationDomain(R)` | `true` if `R` is known to be a unique factorization domain; `false` if known not; error if unknown. | — |
| `IsDomain(R)` / `IsIntegralDomain(R)` | `true` if `R` is known to be an integral domain (no zero divisors); `false` if `R` is known to have zero divisors; error if unknown. | — |
| `HasGCD(R)` | `true` iff a GCD algorithm for elements of ring `R` is implemented in Magma. | — |
| `R eq S` | `true` if rings `R` and `S` refer to the same ring; `false` otherwise. May error if `R` and `S` belong to different categories. | — |
| `R ne S` | `true` if rings `R` and `S` refer to different rings; `false` otherwise. May error if they belong to different categories. | — |

---

## 17.5 Generic Element Functions

### 17.5.1 Parent and Category

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Parent(r)` | The (default) parent ring of ring element `r`. For literal integers, rationals, reals, and values returned by certain functions, a default parent is created in the background. | — |
| `Category(r)` / `Type(r)` | The Magma category to which ring element `r` belongs. `ListCategories()` lists all categories. | — |

### 17.5.2 Creation of Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Zero(R)` | The zero element of ring `R`; equivalent to `R ! 0`. | — |
| `One(R)` / `Id(R)` | The multiplicative identity 1 of ring `R`; equivalent to `R ! 1`. | — |
| `R ! a` | Coerce element `a` of some ring into ring `R`. If `a` is an integer, always succeeds, returning `a · 1_R`. | Follows coercion rules of §17.3. |
| `Random(R)` | A random element of finite ring `R` (uniform distribution over all elements). | — |
| `Representative(R)` / `Rep(R)` | A representative element of finite ring `R`. | — |

### 17.5.3 Arithmetic Operations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `+a` | The element `a` (unary plus). | — |
| `-a` | The negation (additive inverse) of element `a`. | — |
| `a + b` | Sum of ring elements `a` and `b`; if they do not belong to the same ring, a common overstructure is sought. | — |
| `a - b` | Difference of ring elements `a` and `b`; if they do not belong to the same ring, a common overstructure is sought. | — |
| `a * b` | Product of ring elements `a` and `b`; if they do not belong to the same ring, a common overstructure is sought. | — |
| `a ^ k` | k-th power of ring element `a` for small non-negative integer k. Requires k > 0 if a = 0. | — |
| `a ^ -k` | k-th power of the multiplicative inverse of unit `a`. | — |
| `a / b` | Quotient of element `a` by unit `b` in `R`. If `b` is not invertible, an error results (unless both are integers, in which case the rational a/b is returned). Common overstructure sought if `a`, `b` are in different rings. | — |
| `a +:= b` | Mutation assignment: replace `a` with `a + b`. | — |
| `a -:= b` | Mutation assignment: replace `a` with `a - b`. | — |
| `a *:= b` | Mutation assignment: replace `a` with `a * b`. | — |
| `a /:= b` | Mutation assignment: replace `a` with `a / b`. | — |
| `a ^:= k` | Mutation assignment: replace `a` with `a^k`. | — |

### 17.5.4 Equality and Membership

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `a eq b` | `true` if elements `a` and `b` of `R` are the same, otherwise `false`. | — |
| `a ne b` | `true` if elements `a` and `b` of `R` are distinct, otherwise `false`. | — |
| `R eq S` | `true` if rings `R` and `S` are the same, otherwise `false`. | — |
| `R ne S` | `true` if rings `R` and `S` are distinct, otherwise `false`. | — |
| `a in R` | `true` iff `a` is an element of ring `R` (equivalently, forced coercion of `a` into `R` will succeed). | — |
| `a notin R` | `true` iff `a` is not an element of ring `R`. | — |

### 17.5.5 Predicates on Ring Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsZero(a)` | `true` iff element `a` of `R` equals 0_R. | — |
| `IsOne(a)` | `true` iff element `a` of `R` equals 1_R. | — |
| `IsMinusOne(a)` | `true` iff element `a` of `R` equals −1 in `R`. | — |
| `IsUnit(a)` | `true` if `a` is a unit in its parent ring `R`, `false` otherwise. | — |
| `IsIdempotent(x)` | `true` iff x² = x. | — |
| `IsNilpotent(x)` | `true` iff some integer power xⁱ of `x` is zero. | — |
| `IsZeroDivisor(x)` | `true` iff `x` is a zero-divisor: there exists y in the parent `R` of `x` such that xy = 0. | — |
| `IsIrreducible(x)` | `true` iff the parent `R` is a domain and `x` is irreducible in `R`: `x` is a non-unit, and whenever ab divides `x` then `a` or `b` is a unit of `R`. | — |
| `IsPrime(x)` | `true` iff the parent `R` is a domain and `x` is a prime element of `R`: `x` is neither 0 nor a unit, and whenever `x` divides ab it divides `a` or `b`. | — |

### 17.5.6 Comparison of Ring Elements

Comparison operations are only defined on ordered ring types.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `a gt b` | `true` if ring element `a` is greater than `b`, otherwise `false`. | — |
| `a ge b` | `true` if ring element `a` is greater than or equal to `b`, otherwise `false`. | — |
| `a lt b` | `true` if ring element `a` is less than `b`, otherwise `false`. | — |
| `a le b` | `true` if ring element `a` is less than or equal to `b`, otherwise `false`. | — |
| `Maximum(a, b)` | The maximum of ring elements `a` and `b`; a common overstructure is sought if they are in different rings. | — |
| `Maximum(Q)` | The maximum of the sequence `Q` of ring elements. | — |
| `Minimum(a, b)` | The minimum of ring elements `a` and `b`; a common overstructure is sought if they are in different rings. | — |
| `Minimum(Q)` | The minimum of the sequence `Q` of ring elements. | — |

---

## 17.6 Ideals and Quotient Rings

The operations below apply to ideals in a commutative ring `R`. Operations on left and right
ideals in non-commutative rings are described in the chapters for those specific ring types.

### 17.6.1 Defining Ideals and Quotient Rings

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ideal< R \| a1, ..., ar >` | Create the ideal I of ring `R` generated by elements a₁, …, aᵣ. | — |
| `quo< R \| a1, ..., ar >` | Construct the quotient ring Q = R/I, where I is the ideal of `R` generated by a₁, …, aᵣ. | — |
| `R / I` | Given ring `R` and ideal `I` of `R`, construct the quotient ring Q = R/I together with the canonical map R → R/I. | — |
| `PowerIdeal(R)` | The set of all ideals of `R`; this is the parent of all ideals of `R`. | — |

### 17.6.2 Arithmetic Operations on Ideals

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `I + J` | The sum of ideals `I` and `J` of ring `R`: the ideal of all elements a + b with a ∈ I, b ∈ J. If I is generated by {a₁,…,aₖ} and J by {b₁,…,bₘ}, then I + J is generated by {a₁,…,aₖ, b₁,…,bₘ}. | — |
| `I * J` | The product of ideals `I` and `J` of ring `R`: the ideal generated by all elements a·b with a ∈ I, b ∈ J; consists of all sums a₁b₁ + … + aₙbₙ with aᵢ ∈ I, bⱼ ∈ J. | — |
| `I meet J` | The intersection of ideals `I` and `J` of ring `R`. | — |

### 17.6.3 Boolean Operators on Ideals

Throughout: `I` and `J` are ideals of the same ring `R`; `a` is an element of `R`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `a in I` | `true` iff element `a` is a member of ideal `I`. | — |
| `a notin I` | `true` iff element `a` is not a member of ideal `I`. | — |
| `I eq J` | `true` iff ideals `I` and `J` are equal. | — |
| `I ne J` | `true` iff ideals `I` and `J` are distinct. | — |
| `I subset J` | `true` iff ideal `I` is contained in ideal `J`. | — |
| `I notsubset J` | `true` iff ideal `I` is not contained in ideal `J`. | — |

---

## 17.7 Other Ring Constructions

Magma allows construction of residue fields, localizations, and completions. These
constructions create appropriate rings of different categories within Magma.

### 17.7.1 Residue Class Fields

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ResidueClassField(I)` | Given a maximal ideal `I` of a ring `R`, create the residue class field K = R/I together with a map sending elements of `R` to the corresponding elements of `K`. | — |

### 17.7.2 Localization

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `loc< R \| a1, ..., ar >` | Given ring `R` and elements a₁,…,aᵣ generating a prime ideal P of `R`, create the localization L of `R` at P together with a map R → L. | — |
| `Localization(R, P)` | Given ring `R` and prime ideal `P` of `R`, create the localization L of `R` at P together with a map R → L. | — |

### 17.7.3 Completion

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `comp< R \| a1, ..., ar >` | Given ring `R` and elements a₁,…,aᵣ generating a prime ideal or zero ideal P of `R`, create the completion C of `R` at P together with a map R → C. | — |
| `Completion(R, P)` | Given ring `R` and a prime ideal or zero ideal P of `R`, create the completion C of `R` at P together with a map R → C. | — |

### 17.7.4 Transcendental Extension

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ext< R \| >` | Create the univariate transcendental extension R[x] of ring `R`. Equivalent to `PolynomialRing(R)`. | — |
| `ext< R, n \| >` | Create the multivariate transcendental extension R[x₁,…,xₙ] of ring `R` for integer n ≥ 1. Equivalent to `PolynomialRing(R, n)`. | — |

---

## 17.8 Bibliography

No bibliography is present in Chapter 17. The chapter is foundational/introductory and cites
no external algorithmic references.

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Automatic coercion (ring homomorphisms, containment) | `R ! a`, `a + b`, `a - b`, `a * b`, `a / b`, `Maximum`, `Minimum` |
| Forced coercion (`!` operator, Table 4 rules) | `R ! a`, `a in R` |
| Ideal arithmetic | `ideal< >`, `quo< >`, `R / I`, `PowerIdeal`, `I + J`, `I * J`, `I meet J` |
| Field of fractions / localization | `loc< >`, `Localization(R, P)` |
| Completion | `comp< >`, `Completion(R, P)` |
| Transcendental extension | `ext< R \| >`, `ext< R, n \| >` |
| Residue class construction | `ResidueClassField(I)` |
| Ring-theoretic predicates | `IsCommutative`, `IsUnitary`, `IsFinite`, `IsOrdered`, `IsField`, `IsDivisionRing`, `IsEuclideanDomain`, `IsEuclideanRing`, `IsMagmaEuclideanRing`, `IsPID`, `IsPrincipalIdealDomain`, `IsPIR`, `IsPrincipalIdealRing`, `IsUFD`, `IsUniqueFactorizationDomain`, `IsDomain`, `IsIntegralDomain`, `HasGCD` |
| Element predicates | `IsZero`, `IsOne`, `IsMinusOne`, `IsUnit`, `IsIdempotent`, `IsNilpotent`, `IsZeroDivisor`, `IsIrreducible`, `IsPrime` |
