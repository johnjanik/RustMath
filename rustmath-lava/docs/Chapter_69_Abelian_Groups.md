# Chapter 69 — Abelian Groups

**Handbook part:** X — Finitely-Presented Groups
**Handbook pages:** 2043–2075 (PDF pages 2174–2209)

---

## Scope and overview

Chapter 69 describes Magma's machinery for computing with abstract abelian groups. Two
distinct categories are provided:

- **GrpAb** — finitely presented abelian groups, specified either by a presentation (free group
  modulo relations) or by a sequence of cyclic-group orders. Groups may be finite or infinite,
  subject only to being finitely generated.
- **GrpAbGen** — generic abelian groups, where the user supplies a domain together with
  functions (or intrinsic names) for the identity, composition and inverse. The group structure
  (i.e. a presentation) is deduced automatically when needed, using either p-Sylow construction
  (when the order is known, algorithm of Michael Stoll) or a generator-based variant of the
  Pollard–Rho algorithm due to Edlyn Teske **[Tes98a]**.

The two key primitive operations for generic groups—computing the order of an element and
computing discrete logarithms—can be carried out without first computing the full group
structure, using baby-step giant-step (BSGS) **[BJT97]** or Pollard–Rho **[Tes98a, GH00]**
methods. Once the structure is known, all remaining operations (subgroups, quotients,
invariants, cohomology, homomorphisms, etc.) reduce to standard abelian-group linear algebra
over **Z**.

---

## 69.1 Introduction

Magma provides computing with abstract abelian groups in two forms: finitely presented
(category `GrpAb`) and generic (category `GrpAbGen`). In the finitely presented case the
groups may be finite or infinite (finitely generated). In the generic case the user supplies a
universe and group operations; the presentation, when needed, is computed automatically.

---

## 69.2 Construction of a Finitely Presented Abelian Group and its Elements

### 69.2.1 The Free Abelian Group

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `FreeAbelianGroup(n)` | Construct the free abelian group `F` on `n` generators (`n` a positive integer). The `i`-th generator is `F.i`. A special assignment `F<x,y,...> := FreeAbelianGroup(n)` names the generators. | — |

*Worked examples: H69E1 (creation of `FreeAbelianGroup(2)` with named generators).*

### 69.2.2 Relations

Relations are expressions `w1 = w2` over the generators of an abelian group and are not
automatically added to the group's defining relation set; they are used as arguments to
constructors such as `quo< >`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `w1 = w2` | Given words `w1` and `w2` over the generators of an abelian group `A`, create the relation `w1 = w2`. | — |
| `r[1]` / `LHS(r)` | Left-hand side of relation `r`; returns a word over the generators of `A`. | — |
| `r[2]` / `RHS(r)` | Right-hand side of relation `r`; returns a word over the generators of `A`. | — |
| `Parent(r)` | The group over which the relation `r` is taken. | — |

*Worked examples: H69E2 (defining relations over `FreeAbelianGroup(2)`; replacing one side of a relation).*

### 69.2.3 Specification of a Presentation

An abelian group with non-trivial relations is constructed as a quotient of an existing
(possibly free) abelian group. Each term of the relation list `R` may be a word (interpreted
as a relator, i.e. `w = 0`), a relation, a relation list, or a subgroup of `F`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AbelianGroup< X \| R >` | Given generator list `X = x1,...,xr` and relation list `R`, let `F` be the free abelian group on `X`; returns (a) the quotient group `A ≅ F/⟨R⟩` and (b) the natural homomorphism `φ: F → A`. | Smith normal form for finitely generated abelian groups. |
| `AbelianGroup([n1,...,nr])` | Construct the direct product `Cn1 × ... × Cnr` of cyclic groups, where `C0` denotes the infinite cyclic group **Z**. | — |

*Worked examples: H69E3 (presentation `< a,b,c | 7a+4b+c, 8a+5b+2c, 9a+6b+3c >` reduced to `Z/3 + Z`); H69E4 (`AbelianGroup([2,3,4,5,6,0,0])` simplified to `Z/2 + Z/6 + Z/60 + Z + Z`).*

### 69.2.4 Accessing the Defining Generators and Relations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `A . i` | The `i`-th defining generator of `A`. | — |
| `Generators(A)` | A set containing the generators of `A`. | — |
| `NumberOfGenerators(A)` / `Ngens(A)` | The number of generators of `A`. | — |
| `Parent(u)` | The parent group `A` of the word `u`. | — |
| `Relations(A)` | A sequence containing the defining relations of `A`. | — |
| `RelationMatrix(A)` | A matrix whose rows correspond to the defining relations of `A`. | — |

---

## 69.3 Construction of a Generic Abelian Group

A *generic abelian group* is specified by a domain `U` (any Magma structure or aggregate of
elements) together with functions for identity, addition and inverse. If these are not
supplied explicitly, the intrinsics of `U` (or `Universe(U)`) are used.

Computing the group structure (i.e. a presentation) is often expensive and is deferred until
needed. Two strategies are available: p-Sylow construction when the group order is known
(Stoll), and a Pollard–Rho generator-based method when only generators are available
(**[Tes98a]**).

### 69.3.1 Specification of a Generic Abelian Group

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `GenericAbelianGroup(U: parameters)` | Construct the generic abelian group `A` over domain `U`. Parameters: `IdIntrinsic` (name of the identity function), `AddIntrinsic` (name of the binary addition function), `InverseIntrinsic` (name of the binary inverse function; must be binary), `UseRepresentation` (bool, default `false`; if `true`, all elements are stored as linear combinations of structure generators — trivialises arithmetic but requires a full structure computation), `Order` (known group order; speeds up Sylow construction and DLP), `UserGenerators` (generating set; enables generator-based structure computation), `ProperSubset` (bool, default `false`; must be set when `A ⊊ U`), `RandomIntrinsic` (name of a random-element function for `U`; required when `A ⊊ U` and `UserGenerators` is not set), `ComputeStructure` (bool, default `false`; compute structure at creation time). When `ComputeStructure := true` additional parameters `UseUserGenerators` (bool, default `false`), `PollardRhoRParam` (default 20), `PollardRhoTParam` (default 8), `PollardRhoVParam` (default 3) control the Pollard–Rho structure algorithm. | p-Sylow construction (Stoll) when order is known; otherwise Pollard–Rho variant **[Tes98a]**. |

*Worked examples: H69E5 (unit group of Z/34384Z as `GenericAbelianGroup`; subgroup of class group of imaginary quadratic field of discriminant −4000004 using `QuadraticForms`).*

### 69.3.2 Accessing Generators

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Universe(A)` | The universe over which the generic abelian group `A` is defined. | — |
| `A . i` | The `i`-th generator of the generic abelian group `A`. | — |
| `Generators(A)` | A sequence of generators of `A` as elements of `A`; the reduced set obtained during structure computation (may differ from `UserGenerators`). Triggers structure computation if not yet done. | — |
| `UserGenerators(A)` | A sequence of the user-supplied generators of `A` as elements of `A`. | — |
| `NumberOfGenerators(A)` / `Ngens(A)` | The number of generators of `A`. | — |

### 69.3.3 Computing Abelian Group Structure

If the order of `A` is available, structure is found by constructing each p-Sylow subgroup
from random elements (Stoll). If only generators are available, a generator-based algorithm
using a variant of the Pollard–Rho method (Teske **[Tes98a]**) is used, which avoids
computing the order but requires solving discrete logarithm problems. When `A` is a subgroup
of a known generic group, the ambient structure is computed first.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AbelianGroup(A: parameters)` | Compute the group structure of the generic abelian group `A` (or a subgroup thereof). Returns the abstract abelian group and the invertible map from it into `A`. Parameters: `UseUserGenerators` (bool, default `false`; if `true`, uses the user-supplied generators via the Pollard–Rho method rather than Sylow construction), `PollardRhoRParam` (size of r-adding walks, default 20), `PollardRhoTParam` (size of internal element storage, default 8), `PollardRhoVParam` (efficient periodic-segment finding, default 3). Defaults are conjectured optimal **[Tes98b]**. | p-Sylow construction (Stoll) when `UseUserGenerators := false`; Pollard–Rho structure algorithm (Teske **[Tes98a]**) when `UseUserGenerators := true`. |

*Worked examples: H69E6 (structure of the unit group of Z/34384Z: `Z/2 + Z/2 + Z/6 + Z/612`).*

---

## 69.4 Elements

Unless otherwise stated, element operations apply to both fp-abelian groups and generic
abelian groups.

### 69.4.1 Construction of Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `A ! [a1,...,an]` | Given abelian group `A` with generators `e1,...,er` and a sequence of integers `[a1,...,ar]`, construct `a1*e1 + ... + ar*er`. | — |
| `A ! e` | Given a generic abelian group `A` and an element `e` of the domain, return `e` as an element of `A`. If `A` is a proper subset, `e` must be a linear combination of generators of `A`. | — |
| `A ! g` | Given a generic abelian group `A` and an element `g` of the underlying set `X`, return `g` as an element of `A`. | — |
| `A ! n` | Given an abelian group `A` with exactly one generator `x`, construct `n*x`. | — |
| `Random(A)` | A random element of a finite fp-abelian or generic abelian group `A`. | — |
| `Identity(A)` / `Id(A)` / `A ! 0` | The identity element (zero) of the abelian group `A`. | — |

### 69.4.2 Representation of an Element

An element `g` of an abelian group `A` can be represented as a linear combination of
generators. For fp-groups the generating set is the defining one; for generic groups it is
either the structure-computation generators or the user-supplied generators.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Representation(g)` / `ElementToSequence(g)` / `Eltseq(g)` | Let `A` have generating set `e1,...,en` and `g = a1*e1 + ... + an*en`. Returns the sequence `Q` of `n` integers with `Q[i] = ai` reduced modulo the order of the `i`-th generator. | — |
| `UserRepresentation(g)` | For a generic abelian group `A` with user-supplied generators `u1,...,un` and `g = a1*u1 + ... + an*un`: returns the sequence `Q` of `n` integers with `Q[i] = ai` reduced modulo the order of the `i`-th generator. | — |
| `Representation(S, g)` | For a generic abelian group `A`, a sequence `S = [s1,...,sm]` of elements of `A`, and an element `g` with `b*g = a1*s1 + ... + am*sm`: returns as first value the coefficient sequence `Q` and as second value the coefficient `b` of `g` (which may not be 1). | — |

*Worked examples: H69E7 (representation of elements in the quadratic forms group).*

### 69.4.3 Arithmetic with Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `u + v` | Sum of elements `u` and `v` in the same abelian group `A`. | — |
| `-u` | Inverse (negation) of element `u`. | — |
| `u - v` | `u + (-v)` for elements `u`, `v` in the same abelian group `A`. | — |
| `m * u` / `u * m` | For integer `m`: `u+u+...+u` (`|m|` summands) if `m > 0`, or `(-u)+...+(-u)` if `m < 0`. | — |

---

## 69.5 Construction of Subgroups and Quotient Groups

These operations apply to both free abelian groups and arbitrary abelian groups.

### 69.5.1 Construction of Subgroups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `sub< A \| L >` | Construct the subgroup `B` of the fp-abelian group `A` generated by the elements specified in `L`. Each term of `L` may be: (a) an element liftable to `A`; (b) a sequence of integers representing an element of `A`; (c) a subgroup of `A`; (d) a set or sequence of type (a), (b), or (c). | Smith normal form. |
| `sub< A \| L: parameters >` | Construct the subgroup of the generic abelian group `A` generated by elements in `L`. Elements liftable into `A` may be elements of `A` itself or of `U` (the domain). Parameters: `Order`, `RandomIntrinsic`, `ComputeStructure` (default `false`), `UseUserGenerators` (default `false`), `PollardRhoRParam` (default 20), `PollardRhoTParam` (default 8), `PollardRhoVParam` (default 3). When order and a random function are given, `L` may be empty (p-Sylow construction from random elements). | p-Sylow construction (Stoll) or Pollard–Rho **[Tes98a]**. |

*Worked examples: H69E8 (subgroup of `Z/2+Z/6+Z/60+Z+Z`); H69E9 (subgroup of the quadratic forms group).*

### 69.5.2 Construction of Quotient Groups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `quo< F \| R >` | Given an abelian group `F` and a relation set `R` in the generators of `F`, construct (a) the quotient `A ≅ F/⟨R⟩` and (b) the natural homomorphism `φ: F → A`. The possibilities for `R` are the same as for `AbelianGroup< >`. | Smith normal form. |
| `A / B` | Given a subgroup `B` of the abelian group `A`, construct the quotient group `A/B`. | Smith normal form. |

---

## 69.6 Standard Constructions and Conversions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AbelianGroup(GrpAb, Q)` / `AbelianGroup(Q)` | Let `Q = [a1,...,ar]` be a sequence of non-negative integers. Creates the abelian group `Z1 + ... + Zr` where `Zi` is cyclic of order `|ai|` if `ai ≠ 0`, or infinite cyclic (**Z**) if `ai = 0`. | — |
| `AbelianGroup(G)` | Given an abelian permutation, matrix, or polycyclic group `G`, represent it as an abelian group `A`. Also returns the isomorphism `φ: G → A`. | — |
| `AbelianQuotient(G)` | Given a finitely presented, permutation, matrix, or polycyclic group `G`, return the maximal abelian quotient `A` of `G` and the natural homomorphism `φ: G → A`. | Smith normal form for finitely generated abelian groups. |
| `DirectSum(A, B)` | The direct sum of abelian groups `A` and `B`. | — |
| `PCGroup(A)` | A pc-group representation `G` of `A`, and the isomorphism `φ: A → G`. | — |
| `PermutationGroup(A)` | A permutation group representation of `A` generated by disjoint cycles of lengths equal to the abelian invariants; returns `G` and the isomorphism `φ: G → A`. | — |
| `FPGroup(A)` | An fp-group representation of `A` generated by commuting generators of orders equal to the abelian invariants; returns `G` and the isomorphism `φ: G → A`. | — |
| `CommutatorSubgroup(G)` / `DerivedSubgroup(G)` | The derived subgroup of `G`; trivial since `G` is abelian. | — |
| `CommutatorSubgroup(H, K)` / `CommutatorSubgroup(G, H, K)` | The commutator subgroup of groups `H` and `K` in their common overgroup `G`. | — |
| `Centralizer(G, a)` / `Centraliser(G, a)` | The centraliser of `a` in `G`; equals `G` since `G` is abelian. | — |
| `Core(G, H)` | The maximal normal subgroup of `G` contained in subgroup `H`; equals `H` since `G` is abelian. | — |
| `Centre(G)` / `Center(G)` | The centre of `G`; equals `G` itself. | — |

---

## 69.7 Operations on Elements

### 69.7.1 Order of an Element

For fp-abelian groups, element order is a direct computation. For generic groups the
following algorithms are available, selected according to what is known:

- **T baby-step giant-step algorithm** (Buchmann–Jacobson–Teske **[BJT97]**): used when group
  order is unknown and `ComputeGroupOrder := false`.
- **T Pollard–Rho algorithm** (Teske **[Tes98a]**): space-efficient alternative.
- **Shanks algorithm** (standard BSGS): used when upper and lower bounds on the order are known.
- **Gaudry–Harley Pollard–Rho variant** (**[GH00]**): used when order bounds are known;
  recommended for very large groups (smaller space than BSGS).

When the group order is known beforehand, element order computation is trivial.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Order(x)` | Order of element `x` in an fp-abelian group. Returns 0 if infinite order. | Direct from the invariant factorisation. |
| `Order(g: parameters)` | Order of element `g` in a generic abelian group. Parameters: `ComputeGroupOrder` (bool, default `true`; if `true`, computes group order first and uses it trivially; if `false`, uses T BSGS), `BSGSLowerBound` (lower bound on the order, default 0), `BSGSStepWidth` (step width in the BSGS, default 0). | T BSGS **[BJT97]** or (if group order known) trivial. |
| `Order(g, l, u: parameters)` | Order of element `g` given that the order of `g` or the group order lies in `[l, u]`. Parameters: `Alg` (`"Shanks"` (default) or `"PollardRho"`), `UseInversion` (bool, default `false`; halves search space when element inversion is fast). | Shanks (standard BSGS) or Gaudry–Harley Pollard–Rho **[GH00]**. |
| `Order(g, l, u, n, m: parameters)` | Order of `g` given order bounds `[l, u]` and congruence `Order(g) ≡ n (mod m)` (or `#A ≡ n (mod m)`). Same `Alg` and `UseInversion` parameters. | Shanks or Gaudry–Harley Pollard–Rho **[GH00]** with congruence constraint. |

*Worked examples: H69E10 (orders of elements in `Z/2 × Z/3 × Z/4 × Z/5 × Z`).*

### 69.7.2 Discrete Logarithm

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Log(g, d: parameters)` | Given elements `g` and `d` of a generic abelian group `A`, return the discrete logarithm of `d` to the base `g`. Parameters: `ComputeGroupOrder` (bool, default `true`; computes group order then order of `g`, then uses Pohlig–Hellman), `AlInPohligHellmanLoop` (`"BSGS"` (default) or `"PollardRho"`; selects the inner algorithm used per prime-power component), `BSGSStepWidth` (default 0), `PollardRhoRParam` (default 20), `PollardRhoTParam` (default 8), `PollardRhoVParam` (default 3). If `ComputeGroupOrder := false`, uses T BSGS directly. | Pohlig–Hellman decomposition with T baby-step giant-step **[BJT97]** or T Pollard–Rho **[Tes98a]** per prime-power component. |

*Worked examples: H69E11 (discrete logarithm in the quadratic forms group with various algorithm options).*

### 69.7.3 Equality and Comparison

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `u eq v` | True if elements `u` and `v` are identical (as elements of the appropriate free abelian group). | — |
| `u ne v` | True if elements `u` and `v` are not identical. | — |
| `IsIdentity(u)` / `IsId(u)` | True if element `u` is the identity (zero) of its abelian group `A`. | — |

---

## 69.8 Invariants of an Abelian Group

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ElementaryAbelianQuotient(G, p)` | The maximal p-elementary abelian quotient of `G` as `GrpAb`; returns the natural epimorphism as second value. | — |
| `FreeAbelianQuotient(G)` | The maximal free abelian quotient of `G` as `GrpAb`; returns the natural epimorphism as second value. | — |
| `Invariants(A)` | The invariants of the abelian group `A`; infinite cyclic factors represented by 0. | Smith normal form. |
| `TorsionFreeRank(A)` | The torsion-free rank of `A`. | — |
| `TorsionInvariants(A)` | The torsion invariants of `A`. | — |
| `PrimaryInvariants(A)` | The primary invariants of `A`. | — |
| `pPrimaryInvariants(A, p)` | The p-primary invariants of `A`. | — |

---

## 69.9 Canonical Decomposition

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `TorsionFreeSubgroup(A)` | The torsion-free subgroup of the abelian group `A`. | — |
| `TorsionSubgroup(A)` | The torsion subgroup of the abelian group `A`. | — |
| `pPrimaryComponent(A, p)` | The p-primary component of the abelian group `A`. | — |

---

## 69.10 Set-Theoretic Operations

### 69.10.1 Functions Relating to Group Order

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Order(G)` / `#G` | Order of the group `G` as an ordinary integer; returns 0 if `G` is infinite. For generic groups, triggers structure computation if the order is unknown. | — |
| `FactoredOrder(G)` | Factored order of `G` as a sequence of prime-exponent pairs; returns the empty sequence if `G` is infinite. Triggers structure computation for generic groups. | — |
| `Exponent(G)` | Exponent of `G`; returns 0 if `G` is infinite. Triggers structure computation for generic groups. | — |
| `IsFinite(G)` | True if `G` is finite. | — |
| `IsInfinite(G)` | True if `G` is infinite. | — |

### 69.10.2 Membership and Equality

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `g in G` | True if element `g` belongs to group `G`. | — |
| `g notin G` | True if element `g` does not belong to group `G`. | — |
| `S subset G` | Given a group `G` and a set `S` of elements of a group `H` (where `G` and `H` share a common covering group), true if `S ⊆ G`. | — |
| `S notsubset G` | True if `S ⊄ G` (same conditions as above). | — |
| `H subset G` | Given groups `G` and `H` (subgroups of a common overgroup), true if `H ≤ G`. | — |
| `H notsubset G` | True if `H` is not a subgroup of `G`. | — |
| `G eq H` | True if groups `G` and `H` (subgroups of a common overgroup) are identical. | — |
| `G ne H` | True if groups `G` and `H` are distinct. | — |

### 69.10.3 Set Operations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `NumberingMap(G)` | A bijection from the finite group `G` onto `{1,...,|G|}`. The actual mapping depends on the choice of standard generators. | — |
| `RandomProcess(G: parameters)` | Create a process for generating random elements of the finite group `G`. Parameters: `Slots` (number of stored elements, at least `Ngens(G)+1`; default 10), `Scramble` (number of initial mixing operations; default 100). Based on the product-replacement algorithm **[CLGM+95]** with an accumulator; cannot produce well-distributed random elements of an infinite group. | Product-replacement algorithm **[CLGM+95]**. |
| `Random(P)` | Next random element of `G` from the process `P` created by `RandomProcess(G)`. | Product-replacement **[CLGM+95]**. |
| `Random(G)` | A random element of the finite group `G`. | — |
| `Rep(G)` | A representative element of `G`. | — |

---

## 69.11 Coset Spaces

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Transversal(G, H)` / `RightTransversal(G, H)` | Given group `G` and subgroup `H`, returns (a) an indexed set `T` forming a right transversal for `G` over `H`, and (b) the transversal mapping `φ: G → T` with `φ(g) = ti` where `g ∈ H*ti`. | — |

### 69.11.1 Coercions Between Groups and Subgroups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `G ! g` | Given element `g` of subgroup `H` of `G`, rewrite `g` as an element of `G`. | — |
| `H ! g` | Given element `g` of group `G` and subgroup `H` of `G` containing `g`, rewrite `g` as an element of `H`. | — |
| `K ! g` | Given element `g` of group `H` and group `K` where `H` and `K` are subgroups of a common `G` both containing `g`, rewrite `g` as an element of `K`. | — |
| `Morphism(H, G)` | The integer matrix defining the inclusion monomorphism from subgroup `H` into `G`. | — |

---

## 69.12 Subgroup Constructions

Many standard subgroup constructors are trivial for abelian groups but are provided for
uniformity. Only meaningful operations are documented here.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `H meet K` | Intersection of subgroups `H` and `K` of some group `G`. | — |
| `H meet:= K` | Replace `H` with the intersection of `H` and `K`. | — |
| `H + K` | Smallest subgroup of `G` containing both `H` and `K`. | — |
| `n * G` | For integer `n` and abelian group `G`, the subgroup `nG`. Returns also the map `G → G` sending `g ↦ n*g`. | — |
| `FrattiniSubgroup(G)` | The Frattini subgroup of the finite abelian group `G`. | — |
| `SylowSubgroup(G, p: parameters)` / `Sylow(G, p: parameters)` | The Sylow p-subgroup of `G`. Parameter: `Structure` (bool, default `false`; if `true` or if the group structure of `A` is already known, the group structure of the Sylow subgroup is computed). | p-Sylow construction (Stoll) for generic groups when structure is required. |

*Worked examples: H69E12 (Sylow 2-subgroup of the unit group of Z/34384Z: `Z/2+Z/2+Z/2+Z/4`).*

---

## 69.13 Subgroup Chains

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CompositionSeries(G)` | A composition series for the finite abelian group `G`, returned as a sequence of subgroups. | — |
| `Agemo(G, i)` | For a finite p-group `G` and positive integer `i`: the characteristic subgroup generated by `{ x^(p^i) : x ∈ G }`. | — |
| `Omega(G, i)` | For a finite p-group `G` and positive integer `i`: the characteristic subgroup generated by elements of order dividing `p^i`. | — |

---

## 69.14 General Group Properties

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsCyclic(G)` | True if `G` is cyclic. | — |
| `IsElementaryAbelian(G)` | True if `G` is elementary abelian. | — |
| `IsFree(G)` | True if `G` is free. | — |
| `IsMixed(G)` | True if `G` is mixed (neither torsion nor free). | — |
| `IspGroup(G)` | True if the finite group `G` is a p-group (all element orders are powers of `p`). | — |

### 69.14.1 Properties of Subgroups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsMaximal(G, H)` | True if subgroup `H` of the finite group `G` is maximal. | — |
| `Index(G, H)` | Index of subgroup `H` in `G` as an ordinary integer; returns 0 if infinite index. | — |
| `FactoredIndex(G, H)` | Factored index of `H` in `G` as a sequence of prime-exponent pairs; returns the empty sequence if infinite index. | — |
| `IsPure(G, H)` | True if subgroup `H` of the finite group `G` is pure: `nG ∩ H = nH` for all `n`. | — |
| `IsNeat(G, H)` | True if subgroup `H` is neat: `pG ∩ H = pH` for all primes `p`. | — |

### 69.14.2 Enumeration of Subgroups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `MaximalSubgroups(G)` | The maximal subgroups of the finite group `G` as a sequence of subgroups. | — |
| `Subgroups(G: parameters)` | Subgroups of the finite group `G` as a sequence of records with fields `subgroup`, `order`, and `length` (conjugacy class length; always 1 for abelian groups). Parameters: `Sub` (sequence of positive integers in ascending divisibility order; restrict to subgroups with those invariants), `Quot` (sequence of positive integers; restrict to subgroups whose quotient has those invariants). | — |
| `NumberOfSubgroupsAbelianPGroup(A)` | For an abelian p-group `G` with `A = [a1,a2,...]` and `G = Ca1 × Ca2 × ...`: returns a sequence whose `m`-th entry is the number of subgroups of order `p^m`. | — |
| `HasComplement(G, U)` | For a finite abelian group `G` and subgroup `U`: decides if a complementary subgroup `V` exists with `G = U + V` and `U ∩ V = {0}`; if so, returns `V` as the second value. | — |

*Worked examples: H69E13 (subgroups of `AbelianGroup([2,6])` of order 12; `Sub := [2,2]` for the Z/2+Z/2 subgroup; `Quot := [2]` for index-2 subgroups).*

---

## 69.15 Representation Theory

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CharacterTable(G)` | The table of irreducible characters for the abelian group `G`. | — |

---

## 69.16 The Hom Functor

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Hom(G, H)` | For finite abelian groups `G` and `H`: returns an abelian group `A ≅ Hom(G, H)` and a transfer map `t` such that `t(a)` gives the corresponding Magma Map homomorphism from `G` to `H`. | — |
| `HomGenerators(G, H)` | For finite abelian groups `G` and `H`: a sequence of **Z**-module generators of `Hom(G, H)` returned as actual Magma Map homomorphisms. Since `Hom(G, H)` is generally not free, generators alone do not enumerate all homomorphisms uniquely (use `Hom` or `Homomorphisms` for that). | — |
| `Homomorphisms(G, H)` | For finite abelian groups `G` and `H`: a sequence of all elements of `Hom(G, H)` as actual Magma Map homomorphisms. Implemented by calling `Hom` and transferring each element. | — |

*Worked examples: H69E14 (Hom(Z/2+Z/3, Z/4+Z/6) ≅ Z/2+Z/6; enumeration of all 12 homomorphisms).*

---

## 69.17 Automorphism Groups

The full automorphism group of the abelian group `G` can be computed (no additional
parameters documented in this chapter; the intrinsic is accessed via the standard
`AutomorphismGroup` function).

---

## 69.18 Cohomology

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Dual(G)` | Computes the dual group `G*` of the finite abelian group `G` and a map `M: G × G* → Z/mZ` (where `m` is the exponent of `G`) that allows `G*` to act on `G`. | — |
| `H2GQmodZ(G)` | Computes `H := H2(G, Q/Z)` and a map `f: H → (G × G → Z/mZ)` giving cocycles as maps `G × G → Z/mZ`, where `m = #G`. | Group cohomology. |
| `ResH2GQmodZ(U, H2)` | For a subgroup `U` of `G` and `H2 = H2(G, Q/Z)` (as returned by `H2GQmodZ`): computes `H2(U, Q/Z)` compatibly with the restriction map into `H2`. Requires `H2` to be the direct output of `H2GQmodZ` (attributes stored therein are used). | Restriction in group cohomology. |

---

## 69.19 Homomorphisms

Two constructors are provided for homomorphisms or isomorphisms between groups, at least
one of which may be a generic abelian group.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `hom< A -> B \| L >` | Construct a homomorphism `φ: A → B` mapping generators `g1,...,gn` of `A` to `h1,...,hn` in `B`. The list `L` may be: (a) `n` 2-tuples `<gi, hi>` (order-independent); (b) `n` arrow-pairs `gi → hi` (order-independent); (c) `h1,...,hn` (order-dependent). If `A` is a generic abelian group, option (a) or (b) relaxes the requirement that `gi` be the defining generators; the set `{g1,...,gn}` need only generate `A`. | — |
| `Homomorphism(A, B, X, Y)` | Creates a homomorphism from `A` into `B` mapping the sequence of elements `X` to `Y`. `A` and `B` may be any group type including generic abelian groups. Elements of `X` need not be generators of `A` as returned by `Generators(A)`, but if they fail to generate `A` then subsequent map application will fail. | — |
| `iso< A -> B \| L >` | Construct an isomorphism `φ: A → B`. Same list-format options as `hom< >`. If `A` is a generic abelian group, elements `gi` need not be defining generators; the set `{h1,...,hn}` must generate the whole of `B`. | — |
| `Isomorphism(A, B, X, Y)` | Creates an isomorphism from `A` into `B` mapping `X` to `Y`. Same flexibility and caveats as `Homomorphism`. | — |

*Worked examples: H69E15 (constructing a homomorphism between two subgroups of the unit-group generic group; showing that an iso with incompatible images gives a runtime error).*

---

## 69.20 Bibliography

| Key | Reference |
|-----|-----------|
| **[BJT97]** | J. Buchmann, M. J. Jacobson, Jr., and E. Teske. *On Some Computational Problems in Finite Abelian Groups.* Mathematics of Computation, **66**:1663–1687, 1997. |
| **[Bos00]** | Wieb Bosma, editor. *ANTS IV*, volume 1838 of LNCS. Springer-Verlag, 2000. |
| **[CLGM+95]** | Frank Celler, Charles R. Leedham-Green, Scott H. Murray, Alice C. Niemeyer, and E. A. O'Brien. *Generating random elements of a finite group.* Comm. Algebra, **23**(13):4931–4948, 1995. |
| **[GH00]** | P. Gaudry and R. Harley. *Counting Points on Hyperelliptic Curves over Finite Fields.* In Bosma [Bos00], pages 313–332. |
| **[Tes98a]** | E. Teske. *A Space Efficient Algorithm for Group Structure Computation.* Mathematics of Computation, **67**:1637–1663, 1998. |
| **[Tes98b]** | E. Teske. *Better Random Walks for Pollard's Rho Method.* 1998. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Smith normal form for finitely generated abelian groups | `AbelianGroup< >`, `quo< >`, `A / B`, `AbelianQuotient`, `Invariants`, `sub< >` (fp case) |
| p-Sylow construction from random elements (Stoll) | `AbelianGroup(A)` (when order known), `SylowSubgroup`/`Sylow`, `sub< >` (generic, order-based) |
| Pollard–Rho structure algorithm **[Tes98a]** | `AbelianGroup(A: UseUserGenerators := true)`, `GenericAbelianGroup(:ComputeStructure)`, `sub< >` (generic, generator-based), `Log` |
| T baby-step giant-step **[BJT97]** | `Order(g)`, `Log(g, d)` (when `ComputeGroupOrder := false`) |
| Gaudry–Harley Pollard–Rho **[GH00]** | `Order(g, l, u: Alg := "PollardRho")`, `Order(g, l, u, n, m: Alg := "PollardRho")` |
| Pohlig–Hellman decomposition | `Log(g, d)` (when group order is known) |
| Product-replacement random elements **[CLGM+95]** | `RandomProcess`, `Random(P)` |
| Group cohomology | `H2GQmodZ`, `ResH2GQmodZ`, `Dual` |
| Hom-group structure | `Hom`, `HomGenerators`, `Homomorphisms` |
