# Chapter 44 — Artin Representations

**Handbook part:** VI — Global Arithmetic Fields
**Handbook pages:** 1217–1224 (PDF pages 1348–1357)

---

## Scope and overview

An Artin representation is a complex representation of Gal(Q̄/Q) that factors through some
finite quotient Gal(F/Q). In Magma, Artin representations are represented as characters of
Gal(F/Q), not as actual modules. They are allowed to be virtual, except in the L-function
machinery (see Chapter 127).

The chapter covers construction of Artin representations from number fields (computing all
irreducible ones simultaneously), basic invariants (degree, group, character, conductor,
decomposition into irreducibles), arithmetic operations (direct sum, difference, tensor
product, equality), and conversion between Artin representations and Dirichlet characters for
the one-dimensional case.

The algorithms for recognising Frobenius elements in Galois groups are described in [DD10].
They rely on cycle-type identification, Serre's trick for alternating groups, and the general
machinery from [DD10]. Magma is usually able to handle Galois groups of size < 10000 acting
on a small number of points easily, and much larger special groups such as Aₙ and Sₙ.

---

## 44.1 Overview

Artin representations in Magma are characters of a finite Galois group Gal(F/Q), where F is
the normal closure of some number field K. Virtual representations are permitted in general;
non-virtual representations are required by the L-function machinery. The implementation
identifies Frobenius elements via the algorithm of [DD10].

---

## 44.2 Constructing Artin Representations

Artin representations are constructed from a number field K (returning all irreducible ones
simultaneously), from abstract group characters, from the permutation action on embeddings,
or by converting Dirichlet characters.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ArtinRepresentations(K)` | Compute all irreducible Artin representations that factor through the normal closure F of the number field K. The Galois group G = Gal(F/K) is represented as a permutation group on the roots of a monic irreducible polynomial f with integer coefficients defining K (by default the defining polynomial of K over Q). Parameters: `Ramification` (BoolElt, default false) — pre-compute inertia groups at all ramified primes and conductors of all representations; `FactorDiscriminant` (BoolElt or tuple, default false) — whether to fully factorise the discriminant of f to determine which primes ramify (incomplete factorisation causes Magma to assume unramified primes in the unfactored part, and to print "(?)" after conductor values); `p0` (RngIntElt, default auto) — which p-adic field to use for roots of f, must be chosen so that `GaloisGroup(f: Prime := p0)` succeeds. | Frobenius-element recognition **[DD10]**. |
| `K !! ch` | Given the normal closure F of K/Q, convert an abstract group character of Gal(F/Q), or a sequence of its trace values, into an Artin representation. | — |
| `PermutationCharacter(K)` | Construct the permutation representation A of the absolute Galois group of Q on the embeddings of K into C. A has dimension [K : Q] and equals the permutation representation of Gal(F/Q) on the cosets of Gal(F/K), where F is the normal closure of K. | — |
| `Determinant(A)` | Construct the determinant of a given Artin representation. The result is a 1-dimensional Artin representation attached to the same field. | — |
| `ChangeField(A, K)` / `K !! A` | Given an Artin representation A (attached to some number field) known to factor through the Galois closure of K, attempt to recognise it as such. Returns the resulting Artin representation attached to K and true if successful, or 0 and false if it proves no such representation exists. Parameter: `MinPrimes` (RngIntElt, default 20) — number of additional primes for which to compare traces of Frobenius elements. | — |

*Worked examples: H44E1 (quadratic field K = Q(i): trivial and sign Artin representations; `PermutationCharacter`; `ChangeField` lifting from Gal(K/Q) to a D₄-extension L of K).*

---

## 44.3 Basic Invariants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Field(A)` | Number field K such that A factors through the Galois group of the normal closure of K. | — |
| `Degree(A)` | Degree (= dimension) of an Artin representation A. | — |
| `Group(A)` | The Galois group of the field through which A factors. | — |
| `Character(A)` | Character of an Artin representation A, represented as a complex-valued character of `Group(A)`. | — |
| `Conductor(A)` | Conductor of an Artin representation A (A must be a true representation, i.e. its character is not a generalised character). Computes all necessary local information if Artin representations were defined with `Ramification := false`; the first call may therefore take some time. | — |
| `Decomposition(A)` | Decompose an Artin representation A into irreducible constituents. Returns a sequence of tuples `[...<Aᵢ, nᵢ>...]` with Aᵢ irreducible and nᵢ its exponent in A (nonzero but possibly negative). | — |
| `DefiningPolynomial(A)` | Returns the polynomial whose roots `Group(A)` permutes. | — |
| `Minimize(A)` | Returns A attached to the smallest number field K such that A factors through its Galois closure. Parameter: `Optimize` (BoolElt, default true) — if true, attempts to minimise the defining polynomial of K using `OptimizedRepresentation`. | — |
| `Kernel(A)` | Smallest Galois extension K of the rationals through which A factors. Note that this field may be enormous and incomputable. | — |
| `IsIrreducible(A)` | Return true iff a given Artin representation is irreducible as a complex representation. | — |
| `IsRamified(A, p)` | Return true iff a given Artin representation is ramified at p. | — |
| `IsWildlyRamified(A, p)` | Return true iff a given Artin representation is wildly ramified at p. | — |
| `EulerFactor(A, p)` | The local polynomial (Euler factor) of an Artin representation A at the prime p. It is the inverse characteristic polynomial of (arithmetic) Frobenius at p on the inertia-invariant subspace of A. Parameter: `R` (Fld, default `ComplexField()`) — coefficient field of the resulting polynomial. | Frobenius-element recognition **[DD10]**. |
| `DirichletCharacter(A)` | Convert a one-dimensional Artin representation to a Dirichlet character. | — |
| `ArtinRepresentation(ch)` | Convert a Dirichlet character ch to a one-dimensional Artin representation A. Parameter: `field` (FldNum, default auto) — the minimal field through which A factors may be supplied to avoid recomputation. Uses class field theory (due to C. Fieker). | Class field theory. |

*Worked examples: H44E2 (S₄-extension of Q; `ArtinRepresentations`; `Minimize` of the 2-dimensional representation factoring through an S₃-quotient; `Kernel`). H44E3 (D₄-extension: splitting field of x⁴ − 3; `Degree`, `Field`, `Dimension`, `Character`, `Conductor`, `IsRamified`, `IsWildlyRamified`, `EulerFactor`). H44E4 (octic field with Galois group of order 576; `Determinant`, `DirichletCharacter`, `ArtinRepresentation`, `Minimize`, `Conductor`, `Factorization`, `Discriminant`).*

---

## 44.4 Arithmetic

Arithmetic on Artin representations is defined pointwise on characters. When representations
factor through different fields, operations involve the compositum of the fields.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `A1 + A2` | Direct sum of two Artin representations. | Character addition. |
| `A1 - A2` | Direct difference of two Artin representations. | Character subtraction. |
| `A1 * A2` | Tensor product of two Artin representations. | Character product. |
| `A1 eq A2` | Returns true iff the two Artin representations are equal. | Character equality. |
| `A1 ne A2` | Returns true iff the two Artin representations are not equal. | Character inequality. |

*Worked examples: H44E5 (same number field: `+`, `*`, equality of characters; x³ − 2 with `Ramification := true`). H44E6 (different fields — compositum: sign characters of Q(√2) and Q(√3) multiplied; their product minimises to the sign character of Q(√6); `sign1 * sign2 * sign3 eq triv1`).*

---

## 44.5 Implementation Notes

The algorithms for recognising Frobenius elements in Galois groups are described in [DD10].
They rely on cycle-type identification, Serre's trick for alternating groups, and the general
machinery from [DD10]. Magma is usually able to handle Galois groups of size < 10000 acting
on a small number of points easily, and much larger special groups such as Aₙ and Sₙ.

---

## 44.6 Bibliography

| Key | Reference |
|-----|-----------|
| **[DD10]** | T. Dokchitser and V. Dokchitser. *Identifying conjugacy classes in Galois groups.* 2010. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Frobenius-element recognition (cycle type, Serre's trick, general machinery) **[DD10]** | `ArtinRepresentations`, `EulerFactor`, `Conductor`, `IsRamified`, `IsWildlyRamified` |
| Permutation representation / character arithmetic | `PermutationCharacter`, `+`, `-`, `*`, `eq`, `ne`, `Decomposition` |
| Field minimisation / `OptimizedRepresentation` | `Minimize`, `ChangeField` |
| Class field theory (Fieker) | `ArtinRepresentation(ch)` |
| Dirichlet character ↔ Artin representation conversion | `DirichletCharacter`, `ArtinRepresentation` |
