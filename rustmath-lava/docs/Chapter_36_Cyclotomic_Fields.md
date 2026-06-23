# Chapter 36 — Cyclotomic Fields

**Handbook part:** VI — Global Arithmetic Fields
**Handbook pages:** 849–853 (PDF pages 980–987)

---

## Scope and overview

Cyclotomic fields are a subtype of number fields (`FldNum`) with additional functionality and more efficient implementations. Orders of cyclotomic fields belong to the category `RngCyc`; the fields themselves to `FldCyc`. Functions that apply generally to number fields, their orders, and elements are listed in Chapter 34.

Two representations of cyclotomic fields are available:

- **Dense representation:** the field is conceptually represented as Q(x)/f(x) where f is a cyclotomic polynomial — the minimal polynomial of a primitive root of unity. This has asymptotically fastest arithmetic.
- **Sparse representation:** given n = ∏ pᵢʳⁱ and nᵢ := pᵢʳⁱ, the field Q(ζₙ) = Q(ζₙ₁, …, ζₙᵣ) is represented as Q(x₁, …, xᵣ)/⟨fₙ₁(x₁), …, fₙᵣ(xᵣ)⟩. This allows much larger fields as long as elements have few coefficients.

The same trade-off as with non-simple representations of number fields applies: the sparse representation supports large fields but at the cost of arithmetic speed.

---

## 36.1 Introduction

Introductory section describing the two representations (dense and sparse) of cyclotomic fields and their respective trade-offs. No intrinsics are defined here.

---

## 36.2 Creation Functions

Functions are provided to create fields of the special type `FldCyc`. Orders and elements created from a `FldCyc` field have types `RngCyc` and `FldCycElt` respectively; elements created from orders have type `RngCycElt`. These types enable the extra functions and efficient implementations described in this chapter.

For elements of cyclotomic number fields, primitive roots of unity ζₘ are chosen so that ζₘ/ᵈ_m = ζ_d for every divisor d of m — equivalent to choosing ζₘ = e^(2πi/m) in the complex plane.

### 36.2.1 Creation of Cyclotomic Fields

Cyclotomic fields can be created from an integer specifying which roots of unity it should contain, or from a collection of elements of an existing field or order. Cyclotomic polynomials can also be retrieved independently of the fields and orders.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CyclotomicField(m)` | Given a positive integer m, creates the field Q(ζₘ) obtained by adjoining the m-th roots of unity to Q. A name can be assigned to the primitive m-th root of unity using angle brackets: `R<s> := CyclotomicField(m)`. Parameter: `Sparse` (Boolean, default `false`); if `true`, names for all generating elements can be assigned and the sparse representation is used. | — |
| `CyclotomicPolynomial(m)` | Given a positive integer m, creates the cyclotomic polynomial of order m. Equivalent to `DefiningPolynomial(CyclotomicField(m))`. | — |
| `MinimalCyclotomicField(a)` | Given an element a from a cyclotomic field F or ring R, returns the smallest cyclotomic field or order thereof (possibly Q or Z) E ⊆ F containing a. | — |
| `MinimalCyclotomicField(S)` | Given a set or sequence S of cyclotomic field or ring elements, returns the smallest cyclotomic field or ring (possibly Q or Z) G containing each element of S. | — |

*Worked examples: H36E1 (dense vs. sparse representation on the cyclotomic field of order 100; coercion between K1 and K2).*

### 36.2.2 Creation of Elements

Elements of cyclotomic fields and orders can also be created using coercion (`!`) and the `elt` constructor (`elt<|>`), where the left-hand side is the field or order the element will lie in. See Section 34.2.3 for details on coercion.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RootOfUnity(n)` | Creates the n-th root of unity ζₙ in Q(ζₙ). | — |
| `RootOfUnity(n, K)` | Given a cyclotomic field K = Q(ζₘ) and an integer n > 2, creates the n-th root of unity ζₙ in K. An error results if ζₙ ∉ K, i.e. if n does not divide m (or 2m when m is odd). | — |
| `Minimise(~a)` / `Minimize(~a)` | Procedure: given an element a in a cyclotomic field F or ring R, finds the minimal cyclotomic subfield E ⊆ F or subring E ⊆ R containing a, and coerces a into E. E may be Q or Z. | — |
| `Minimise(~s)` / `Minimize(~s)` | Procedure: given a set s of cyclotomic field or ring elements, finds the minimal cyclotomic field or ring E containing all of them, and coerces each element into E. The resulting set has universe E. E may be Q or Z. | — |
| `Minimise(a)` / `Minimize(a)` | Function: given an element a in a cyclotomic field F or ring R, finds the minimal cyclotomic subfield E ⊆ F or subring E ⊆ R containing a, and returns the coercion of a into E. E may be Q or Z. | — |
| `Minimise(s)` / `Minimize(s)` | Function: given a set s of cyclotomic field or ring elements, finds the minimal cyclotomic field E containing all of them, and returns the coercion of each element into E with universe E. E may be Q or Z. | — |

---

## 36.3 Structure Operations

In cyclotomic fields, the generic ring functions are supported (see Chapter 17). The functions listed below are additional to those for number fields. For the list of functions applying to general number fields see Sections 34.2 and 34.3.

### 36.3.1 Invariants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Conductor(K)` | The smallest n such that K ⊆ Q(ζₙ); for a cyclotomic field this is either the cyclotomic order m or half of it, depending on whether m ≡ 2 mod 4. The second return value is a sequence of the ramified real places of K. | — |
| `CyclotomicOrder(K)` | The value of m for the cyclotomic field Q(ζₘ). This will be the m with which the cyclotomic field was created. | — |
| `CyclotomicAutomorphismGroup(K)` | Returns the automorphism group of K as an abstract abelian group G and a map from G into the set of all automorphisms. Uses the fact that the automorphism group is already determined by the conductor. Similar functionality is available via `AutomorphismGroup`, but this function returns an abelian group directly. | — |
| `CyclotomicRelativeField(k, K)` | Given two cyclotomic fields k ⊆ K, computes a number field L/k that is isomorphic to K. | — |

---

## 36.4 Element Operations

For the full range of operations for elements of a number field or order see Section 34.4.

### 36.4.1 Predicates on Elements

Because of the nature of cyclotomic fields and orders, some properties of elements are easier to determine than in the general case.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsReal(a)` | Returns whether the cyclotomic field or ring element a is a real number, i.e. whether it is invariant under complex conjugation. | — |

### 36.4.2 Conjugates

Elements of cyclotomic fields and orders can have their complex conjugate computed. Conjugates are returned as cyclotomic elements (not reals), and the particular conjugate desired can be indicated by providing a primitive root of unity.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ComplexConjugate(a)` | The complex conjugate of cyclotomic field or ring element a. | — |
| `Conjugate(a, n)` | The image of a under the map ζ ↦ ζⁿ. The second argument n must be coprime to the conductor. | — |
| `Conjugate(a, r)` | The conjugate of the element a ∈ Q(ζₘ) or its order, obtained by applying the field automorphism ζₘ ↦ r, where r is a primitive root of unity. | — |

*Worked examples: H36E2 (Gaussian periods ηd for l = 13; generating a set W of minimal polynomials for degree-d cyclic subfields of Q(ζ₁₃) using divisors of l−1 and a primitive root modulo l).*

---

## 36.5 Bibliography

No bibliography is printed in this chapter. The chapter refers the reader to Chapter 34 for general number field references.

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Cyclotomic field construction (dense and sparse representations) | `CyclotomicField`, `CyclotomicPolynomial` |
| Minimal subfield search | `MinimalCyclotomicField`, `Minimise`/`Minimize` |
| Conductor and automorphism group via cyclotomic structure | `Conductor`, `CyclotomicAutomorphismGroup`, `CyclotomicOrder` |
| Relative cyclotomic field construction | `CyclotomicRelativeField` |
| Root of unity creation and coercion | `RootOfUnity` |
| Complex conjugation and field automorphisms on cyclotomic elements | `ComplexConjugate`, `Conjugate`, `IsReal` |
