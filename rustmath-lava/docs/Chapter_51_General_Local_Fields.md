# Chapter 51 — General Local Fields

**Handbook part:** VII — Local Arithmetic Fields
**Handbook pages:** 1365–1374 (PDF pages 1496–1507)

---

## Scope and overview

Chapter 51 describes *general local fields* (`RngLocA`, elements `RngLocAElt`): extensions of
any Magma local field by an arbitrary irreducible polynomial. Unlike the fields described in
Chapter 47, these extensions are not constrained to be either totally ramified (Eisenstein
polynomial) or inertial (unramified polynomial) — a single polynomial may encode both the
ramified and unramified parts of an extension in one step.

Internally a general local field is represented as a polynomial quotient ring. A map into an
isomorphic `FldPad` can be constructed (via `RamifiedRepresentation`) and that isomorphic
field used for calculations that require the two-step representation.

The chapter covers: construction of extensions and subfields; field-level intrinsics
(degree, precision, ramification, inertia); the maximal order; homomorphisms;
automorphisms and Galois theory (including Frobenius, inertia, ramification and decomposition
groups, and fixed fields); element-level intrinsics (arithmetic, predicates, valuations); and
factorization of polynomials over general local fields.

No chapter-specific bibliography is provided in the handbook text; the algorithms reference the
standard p-adic machinery shared with Chapter 47.

---

## 51.1 Introduction

General local fields allow ramified and inertial extensions to be made in one step rather than
forcing a split into a ramified extension followed by an unramified extension (or vice versa).
They are typed `RngLocA` with elements `RngLocAElt`. For local fields restricted to totally
ramified or unramified extensions see Chapter 47.

---

## 51.2 Constructions

Local fields can be constructed as extensions of other local fields and as subfields of other
local fields.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LocalField(L, f)` | Construct a local field `F` as an extension of the local field `L` by the irreducible polynomial `f` over `L`. | Polynomial quotient ring construction; `f` need not be Eisenstein or inertial. |
| `sub< L \| a1, ..., an >` | Construct the local field `F` as a subfield of `L` containing the listed elements of `L`. Returns the subfield and an embedding map. | — |
| `sub< L \| S >` | Construct the local field `F` as a subfield of `L` containing the elements of the sequence `S`. Returns the subfield and an embedding map. | — |

*Worked examples: H51E1 (building a degree-6 extension of Q_7 in one step using the minimal polynomial of an element from a two-step construction; also constructing directly from an arbitrary irreducible polynomial). H51E2 (constructing a two-level tower over Q_5 then extracting a subfield via `sub<>` and inspecting the embedding map).*

---

## 51.3 Operations with Fields

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BaseRing(L)` | Return the coefficient field of the local field `L` (the field that was extended to construct `L`). | — |
| `CoefficientRing(L)` | Synonym for `BaseRing(L)`. | — |
| `DefiningPolynomial(L)` | Return the polynomial used to define `L` as an extension of its coefficient field. | — |
| `Degree(L)` | Return the degree of `L`, i.e. the degree of its defining polynomial. | — |
| `Degree(L, R)` | Return the degree of `L` as an extension of `R`, where `R` is some coefficient ring of `L`. | — |
| `InertiaDegree(L)` | Return the degree of the inertial subfield of `L` as an extension of the coefficient field of `L`. | — |
| `RamificationDegree(L)` | Return the degree of the totally ramified subfield of `L` as an extension of the coefficient field of `L`. | — |
| `RamificationIndex(L)` | Synonym for `RamificationDegree(L)`. | — |
| `Precision(L)` | Return the precision of `L`: the maximum number of p-adic digits that can appear in an element of `L` (the difference between the valuation of an element and the valuation of its highest-valuation term). | — |
| `Prime(L)` | Return the prime of `L`, which equals the prime of its coefficient field. | — |
| `QuotientRepresentation(L)` | Return the polynomial quotient ring isomorphic to `L` used to represent `L` internally. | — |
| `RamifiedRepresentation(L)` | Return the local field isomorphic to `L` constructed as an unramified extension followed by a totally ramified extension, together with the map from `L` into that isomorphic field. | — |
| `AssignNames(~L, S)` | Assign the name in sequence `S` to the generator of the extension defining `L`. | — |
| `Name(L, i)` | Return the generator of `L` whose name was assigned by `AssignNames`; the only valid value of `i` is 1. | — |
| `Discriminant(L)` | Return the discriminant of the local field `L`. | — |
| `ResidueClassField(L)` | Return the residue class field of the maximal order of `L` and the map between `L` and its residue class field. | — |
| `RelativeField(L, m)` | Return `L` as an extension of the domain of the map `m`, where `m` is a map from a subfield of `L` (having the same coefficient ring as `L`) into `L`. | — |

*Worked examples: H51E3 (printing the coefficient ring, defining polynomial, precision, prime, degree, ramification degree, and inertia degree of the degree-6 field from H51E1). H51E4 (computing `QuotientRepresentation` and `RamifiedRepresentation` of a one-step degree-6 extension over Q_7, then mapping the generator and its inverse image).*

### 51.3.1 Predicates on Fields

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsRamified(L)` | Return whether `L` has a non-trivial ramified subfield, i.e. the ramification degree of `L` is greater than 1. | — |
| `IsTamelyRamified(L)` | Return whether `L` is tamely ramified. | — |
| `IsWildlyRamified(L)` | Return whether `L` is wildly ramified. | — |
| `IsTotallyRamified(L)` | Return whether `L` is a totally ramified extension, i.e. has a trivial inertial subfield. | — |
| `IsUnramified(L)` | Return whether `L` equals its inertial subfield. | — |

---

## 51.4 Maximal Order

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IntegralBasis(L)` | Return a basis for the maximal order of the local field `L`. | — |
| `IsIntegral(a)` | Return whether the local field element `a` lies in the maximal order of its parent `L`; if so, also return a sequence giving the coordinates of `a` with respect to the integral basis of `L`. | — |

*Worked example: H51E5 (constructing the degree-6 field `x^6 - 49*x^2 + 686` over Q_7 and printing its integral basis).*

---

## 51.5 Homomorphisms from Fields

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `hom< L -> R \| a >` | Return the homomorphism from the local field `L` into the ring `R` whose image of the generator of `L` is `a`. | — |
| `hom< L -> R \| cfm, a >` | Return the homomorphism from `L` into `R` whose image of the generator is `a` and whose action on the coefficient field of `L` is given by `cfm`. | — |

---

## 51.6 Automorphisms and Galois Theory

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `FrobeniusAutomorphism(L)` | Return the automorphism of the unramified extension `L` that is the lift of the Frobenius automorphism on the residue class field of `L`. | p-adic Frobenius lift. |
| `AutomorphismGroup(L)` | Return the automorphism group of the local field `L` and a map from the group to the parent of automorphisms of `L`. | — |
| `DecompositionGroup(L)` | Return the subgroup of the automorphism group of `L` whose elements are the automorphisms σ (as group elements) such that v(σ(z) − z) ≥ 0 for all z; equivalently the (−1)-th ramification group. | Group-theoretic definition from the valuation. |
| `InertiaGroup(L)` | Return the 0th ramification group of `L`: those automorphisms σ with v(σ(z) − z) ≥ 1 for all z. | Group-theoretic definition. |
| `RamificationGroup(L, i)` | Return the i-th ramification group of `L`: those automorphisms σ (as group elements) with v(σ(z) − z) ≥ i + 1 for all z in the maximal order. The decomposition group is the (−1)-th ramification group; the inertia group is the 0th. | Group-theoretic definition from the valuation. |
| `FixedField(L, G)` | Return the subfield of `L` fixed by the automorphisms (represented as group elements) in the subgroup `G` of the automorphism group of `L`. | Galois correspondence. |

*Worked example: H51E6 (computing the automorphism group of the degree-6 extension `x^6 - 49*x^2 + 686` over Q_7, mapping a random element, computing the inertia group, and recovering the fixed fields of the full automorphism group and of the inertia group).*

---

## 51.7 Local Field Elements

### Element construction and generators

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `L ! r` | Return the element of the local field `L` described by `r`, where `r` is anything coercible into the quotient representation of `L`. | — |
| `L . i` | Return the generator of the local field `L`; the only valid value of `i` is 1. | — |
| `InertialElement(L)` | Return a generator for the inertial subfield of `L`. | — |
| `UniformizingElement(L)` | Return an element of `L` of valuation 1. | — |

### 51.7.1 Arithmetic

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `a * b` | Multiplication of local field elements. | — |
| `a + b` | Addition of local field elements. | — |
| `a - b` | Subtraction of local field elements. | — |
| `- a` | Negation of a local field element. | — |
| `a ^ n` | Exponentiation of a local field element by an integer `n`. | — |
| `a / b` | Division of local field elements. | — |

### 51.7.2 Predicates on Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `a eq b` | Return whether the local field elements `a` and `b` are considered equal. | — |
| `IsOne(a)` | Return whether `a` is known to be 1 to the precision of the field. | — |
| `IsMinusOne(a)` | Return whether `a` is known to be −1 to the precision of the field. | — |
| `IsWeaklyZero(a)` | Return whether `a` is not known to be non-zero (i.e. may be zero). | — |
| `IsZero(a)` | Return whether `a` is known to be zero. | — |

### 51.7.3 Other Operations on Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Valuation(a)` | Return the valuation of the element `a` in a local field. | — |
| `RelativePrecision(a)` | Return the relative precision of the element `a` in a local field. | — |
| `Eltseq(a)` | Return the coefficients of powers of the generator of the parent field in `a`. | — |
| `RepresentationMatrix(a)` | Return the representation matrix of the element `a` of a local field. | — |

*Worked example: H51E7 (computing `UniformizingElement` and `InertialElement` of the degree-6 field from H51E1, printing their valuations, and calling `Eltseq` on the uniformizing element).*

---

## 51.8 Polynomials over General Local Fields

Polynomials over general local fields can be factored and their roots computed.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Factorization(f)` | Factorization of the polynomial `f` over a local field defined by an arbitrary polynomial. Returns a sequence of tuples of prime polynomials with exponents, plus a scalar factor. Parameter: `Certificates` (BoolElt, default `false`): if `true`, also returns certificates proving the primality of each prime factor. | — |
| `SuggestedPrecision(f)` | For a polynomial `f` over a general local field, return a precision at which the factorization of `f` as given by `Factorization(f)` will be Hensel-liftable to the correct factorization. The returned precision is not guaranteed to be sufficient; a correct factorization may require slightly more precision. | Hensel lifting analysis. |
| `Roots(f)` | Return the roots of the polynomial `f` over the general local field that is the coefficient ring of `f`. | — |
| `Roots(f, R)` | Return the roots of the polynomial `f` over the general local field `R`. | — |

*Worked example: H51E8 (factoring `x^6 - 6*x^4 + 9*x^2 - 27` over Q_3 as a polynomial over the degree-6 local field `LocalField(Q3, x^6 - 6*x^4 + 9*x^2 - 27)`, showing all 6 linear factors).*

---

## 51.9 Bibliography

No dedicated bibliography is provided in Chapter 51. The algorithms used (polynomial quotient ring representation, Hensel lifting for factorization, p-adic Frobenius, ramification group definitions) are part of the general local-field infrastructure shared with Chapter 47 (p-adic Fields and Related Structures).

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Polynomial quotient ring representation of a mixed extension | `LocalField`, `QuotientRepresentation` |
| Two-step (unramified + totally ramified) isomorphic representation | `RamifiedRepresentation` |
| Subfield extraction | `sub< L \| ... >` |
| Ramification / inertia group theory (valuative definition) | `DecompositionGroup`, `InertiaGroup`, `RamificationGroup` |
| Galois correspondence (fixed fields) | `FixedField` |
| p-adic Frobenius lift | `FrobeniusAutomorphism` |
| Hensel lifting analysis for precision | `SuggestedPrecision` |
| Polynomial factorization over local fields | `Factorization`, `Roots` |
