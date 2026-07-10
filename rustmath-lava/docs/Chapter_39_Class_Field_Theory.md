# Chapter 39 — Class Field Theory

**Handbook part:** VI — Global Arithmetic Fields
**Handbook pages:** 999–1033 (PDF pages 1130–1167)

---

## Scope and overview

Chapter 39 presents the Magma facilities for class field theory over number fields. The
primary objects are **abelian extensions** (type `FldAb`) of number fields, constructed via
ray class groups and their quotients, together with maps between finite abelian groups and
ideal groups.

**Mathematical setting.** Class field theory parametrises all abelian extensions K/k of a
number field k by quotients of ray class groups Clm. A ray class group modulo a modulus
m = (m0, m∞) (finite ideal m0 and set of real places m∞) is the quotient Im/Pm. The
Artin map gives a canonical isomorphism Clm/H ≅ Gal(K/k) for the appropriate subgroup
H; the smallest m for which this works is the conductor of H. The Hilbert class field
(m = (1o, ∅)) is the maximal unramified abelian extension; all ideals of k become principal
in it (Hilbert–Furtwängler theorem).

**Algorithmic approach.** In Magma, ideal groups are represented as maps from finite
abelian groups (`GrpAb`) to power structures of ideals (`PowIdeal`). Ray class groups are
computed by a mixture of Pauli's approach (following Hasse **[Pau96, HPP97]**) and
Cohen's method **[CDO96, CDO97, Coh00]**. Defining equations for abelian extensions are
computed by Fieker's algorithm **[Fie00, Coh00]**, which handles each cyclic factor of
prime power degree independently using Kummer theory and the Artin map. Maximal
orders exploit the Kummer structure **[Sut12]**. Many invariants (degree, discriminant,
conductor, decomposition types) can be read off directly from the ideal group without
computing defining equations.

**Scope.** This chapter covers number fields only; global function fields are in Chapter 43
and p-adic local fields in Section 47.14. The norm equation machinery handles fields up to
degree 80+ by exploiting the class field structure to reduce to smaller subfields.

---

## 39.1 Introduction

### 39.1.1 Overview

Theoretical background: ray class groups, the Artin map, conductor-discriminant relations,
the Hilbert class field, and norm groups. No intrinsics; see the scope section above for the
mathematical content.

### 39.1.2 Magma

Explains the programming model: maps `m : G → Ik` (G a finite abelian group, Ik the
ideals of a maximal order) represent ideal groups. Quotients of the class group map are
composed to define sub-extensions. Worked example H39E1 constructs the Hilbert class
field of Q(α), α³ + α² + 3α − 6 = 0, with class group C4, verifies the discriminant is 1,
checks that the class group generator becomes principal, and demonstrates the step-by-step
construction of a quadratic subfield.

*Worked examples: H39E1 (Hilbert class field of a cubic field, capitulation in a quadratic subfield).*

---

## 39.2 Creation

### 39.2.1 Ray Class Groups

The classical ideal-theoretic approach to class field theory uses ray class groups. In addition
to the functions below, the CRT functions (Chapter 38, page 946) are relevant here.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RayClassGroup(I)` `RayClassGroup(I, T)` | For an integral ideal I of the maximal order of a number field: the ray class group modulo I (with optional sequence T of indices of real infinite places to include in the modulus). Returns an abelian group A and a map from A to ideal representatives. Requires the class group to be known (computed heuristically if not stored; use `ClassGroup` with proof level beforehand for guaranteed results). | Mixture of Pauli–Hasse **[Pau96, HPP97]** and Cohen **[CDO96, CDO97, Coh00]**. |
| `RayClassGroup(D)` | For a divisor (or place) of an absolute number field: the ray class group modulo the divisor. Same class group requirement as above. Returns A and a map. | **[Pau96, HPP97, CDO96, CDO97, Coh00]**. |
| `RayResidueRing(I)` `RayResidueRing(I, T)` | For an integral ideal I of the maximal order: the unit group of the maximal order modulo I (the group of units mod*(I, T)), extended by one C2 factor per element of T (conditions on signs at real places). Returns a finite abelian group and a map to the order. | Direct computation of (o/I)* with sign conditions. |
| `RayResidueRing(D)` | For an effective divisor D of a number field: the unit group of the residue class ring modulo D (elements approximating 1 at finite places and positive at real infinite places of the support of D). | Direct computation. |

*Worked examples: H39E2 (ray class groups of Q(√10) modulo growing moduli, illustrating growth; effect of including infinite places).*

### 39.2.2 Selmer Groups

The p-Selmer group of a finite set S of prime ideals in K is Kp(S) = {x ∈ K×/(K×)p | vQ(x) ≡ 0 mod p ∀Q ∉ S}, a finite abelian group of exponent p.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `pSelmerGroup(p, S)` | For a prime integer p and set S of prime ideals in a number field K: returns the p-Selmer group Kp(S) as an abstract group G with map m from K to G (and its inverse). Parameters: `Integral` (default `true`) ensures integral representatives by enlarging generators to include uniformizing elements; `Nice` (default `true`); `Raw` (default `false`) returns exponent vectors instead of field elements and reduces by the lattice of p-th powers of S̃-units (elements may lose integrality unless `Integral` is also set). With `Raw`, returns additionally a map from G to an exponent-vector space and the basis sequence B such that x = PowerProduct(B, v). | Identifies the class of x in Kp(S) by computing multiplicative orders of residues of x and group generators modulo unrelated primes; the S̃-unit group is enlarged until the p-part of the class group is generated by ideals in S̃. |

*Worked examples: H39E3 (3-Selmer group of Q(√10) for primes above 2, 3, 11; demonstration of Raw mode and PowerProduct reconstruction).*

### 39.2.3 Maps

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `InducedMap(m1, m2, h, c)` | For maps m1 : G1 → I1, m2 : G2 → I2 from finite abelian groups into ideals of some maximal order, and a map h : I1 → I2 on ideals, and a multiple c of the minimum of the defining moduli: computes the map on abelian groups induced by h, defined by `hom<G1 -> G2 \| [h(r1(G1.x)) @@ r2 : x in [1..Ngens(G1)]]>`. Faster than the naïve approach for large moduli by using small prime ideal generators for both groups. | Finds "small" (small-norm) generators for both ray class groups to avoid expensive discrete logarithms for large ideals. |
| `InducedAutomorphism(r, h, c)` | Abbreviation for `InducedMap(r, r, h, c)`. | Same as above. |

*Worked examples: H39E4 (induced automorphisms of a "large" ray class group over Q(√10, ζ16); timing comparison with direct approach for growing moduli).*

### 39.2.4 Abelian Extensions

The main creation functions for abelian extensions (type `FldAb`) defined by ideal groups.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RayClassField(m)` `AbelianExtension(m)` `RayClassField(m, I, T)` `AbelianExtension(m, I, T)` `RayClassField(m, I)` `AbelianExtension(m, I)` | For a map m : G → Ik (G a finite abelian group, Ik ideals of an absolute maximal order): construct the class field defined by m. The inverse m−1 must be a homomorphism from a ray class group R onto G; the Galois group is isomorphic to R/ker(m−1) via the Artin map. If I (or I and T) are given they define R explicitly; otherwise Magma extracts the modulus from m. Note: Magma cannot validate the map — invalid input produces garbage output. | Object construction; defining equations are deferred until `EquationOrder` or `NumberField` is called. |
| `AbelianExtension(I)` | Creates the full ray class field modulo the integral ideal I. | — |
| `RayClassField(D)` | Creates the full ray class field modulo a divisor D: an abelian extension unramified outside the support of D whose automorphism group is canonically isomorphic to the ray class group modulo D. | — |
| `AbelianpExtension(m, p)` | For a map m as in `AbelianExtension` and a prime p: the maximal p-subfield, i.e. the maximal subfield of p-power degree. | Projection to the p-part of the defining group. |
| `AbelianExtension(I, P)` | Creates the full ray class field modulo the ideal I and the infinite places in P. | — |
| `HilbertClassField(K)` | The Hilbert class field of K: the maximal unramified abelian extension of K. Equivalent to `AbelianExtension(1*MaximalOrder(K))`. | — |
| `MaximalAbelianSubfield(M)` `MaximalAbelianSubfield(F)` `MaximalAbelianSubfield(K)` | For a number field K with coefficient field k: the maximal abelian extension A of k inside K. Parameter `Conductor` (a multiple of the true conductor; defaults to discriminant of K). Based on heuristics — correctness is not guaranteed. | Heuristic algorithm similar to **[Coh00, Algorithm 4.4.3]**. |
| `AbelianExtension(K)` `AbelianExtension(M)` | For a number field K (or its order M) with coefficient field k: an abelian extension of k isomorphic to K. Parameter `Conductor` (multiple of the true conductor; defaults to discriminant of the maximal order, or the defining order for fields of type `FldOrd`). Provided K is abelian, always computes a correct answer (contrast with `MaximalAbelianSubfield`). | Norm group identification via the conductor-discriminant relation. |

*Worked examples: H39E5 (computing Q(ζ12) as a class field; extracting the 5-part of a ray class field over Q(√10), verifying the Galois group is C5 × C5). H39E6 (Hilbert class field of a sextic field with class group C3 × C3; comparison of direct HilbertClassField vs step-by-step AbelianExtension, including maximal order via Discriminant algorithm).*

### 39.2.5 Binary Operations

Binary operations on abelian extensions with the same base field, computable directly from the ideal group without defining equations.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `A eq B` | Decide if two abelian extensions with the same base field are equal. | Comparison of defining ideal groups. |
| `A subset B` | Decide if one abelian extension is contained in the other. | Subgroup containment in the ideal group. |
| `A * B` | The smallest abelian extension containing both A and B (compositum). | Intersection of norm groups (join in the subgroup lattice). |
| `A meet B` | The largest common subfield of A and B. | Join of norm groups (meet in the subgroup lattice). |

---

## 39.3 Galois Module Structure

If the base field k for class field constructions is normal over some subfield k0 (k/k0 normal
with Galois group g), and if the defining modulus of the ideal group is g-invariant, then g
acts on the ideal group. The functions below treat ideal groups as Galois modules.

Given an abelian extension A with parameters `All` and `Over`: let k = BaseField(A), k1 =
coefficient field of k. If `All` is true, g := Aut(k/k1); otherwise g := ⟨Over⟩. Then
k0 := Fix(k, g). If k is normal over k1, then k0 = k1 and g is the full Galois group.

### 39.3.1 Predicates

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsAbelian(A)` | Returns true iff the abelian extension A is abelian over k0. Parameters: `All` (bool, default `false`), `Over` ([Map], default `[]`). | Tests whether the defining ideal group is a g-module with trivial action. |
| `IsNormal(A)` | Returns true iff A is normal over k0. Parameters: `All` (default `false`), `Over` (default `[]`). | Tests whether the defining ideal group is a g-module. |
| `IsCentral(A)` | Returns true iff A is central over k0 (for k cyclic over k0 this is equivalent to being abelian over k0). Tests whether the norm group extension `1 → N → G → g → 1` is central. Parameters: `All` (default `false`), `Over` (default `[]`). | Group extension centrality check on the norm group. |

### 39.3.2 Constructions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `GenusField(A)` | The genus field: the maximal abelian extension of k0 contained in the abelian extension A. Returns an abelian extension of k0. Parameters: `All` (default `false`), `Over` (default `[]`). | Fixed field of the g-invariant part of the norm group. |
| `H2GA(A)` | For A normal over Q with a base field k that is also normal over Q: compute the 2nd cohomology group of the Galois group of k acting on the ideal group defining A. | Cohomology of a Galois module (see Chapter 68). |
| `NormalSubfields(A)` | For an abelian extension normal over Q and defined over a normal number field k: a list of all normal intermediate fields. Parameter `Quot` (sequence of abelian invariants; restricts to fields whose norm group has those invariants). | Lattice of g-stable subgroups. |
| `AbelianSubfield(A, U)` `FixedField(A, U)` | For an abelian extension A with norm group map G → I and a subgroup U < G: the field corresponding to G/U (the field fixed by U). Parameter `IsNormal` (bool, default `false`; if set, transfers any available cohomology information to the new field). | Galois correspondence via quotient of the norm group. |
| `CohomologyModule(A)` | For an abelian extension A defined over a normal field k/Q: computes the cohomology module (see Chapter 68). Returns three maps: (1) automorphism group of k as a permutation group to actual field automorphisms (third return of `AutomorphismGroup`); (2) ideal group of A to a standard representation; (3) standard representation of the norm group to the Z-module. | — |

---

## 39.4 Conversion to Number Fields

Although an abelian extension theoretically determines a number field and all its properties,
not all are directly accessible from the ideal group. These functions perform the conversion.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `EquationOrder(A)` | Computes defining equations for the abelian extension A. For each cyclic factor of prime power degree, one polynomial is constructed. May be very time consuming depending on the sizes of the cyclic factors. Verbose level `ClassField` (max 5). | Algorithm of Fieker **[Fie00, Coh00]**: Kummer theory via the Artin map, one cyclic factor at a time. |
| `NumberField(A)` | Converts the abelian extension A into a number field. Equivalent to `NumberField(EquationOrder(A))`. | Calls `EquationOrder` then `NumberField`. |
| `MaximalOrder(A)` | Computes the maximal order of A, exploiting the Kummer structure for speed. Parameters: `Al` (`"Kummer"` (default) — uses Kummer theory on each component then intersects **[Sut12]**; `"Round2"` — applies ordinary Round 2 algorithm to components; `"Discriminant"` — passes discriminant into the maximal order computation). `Partial` (bool, default `false`; if true, stops after combining component maximal orders without the final `MaximalOrder` call). | **[Sut12]** for `Al := "Kummer"`. |
| `Components(A)` | Returns a list of relative extensions, one per cyclic prime-power factor of the defining group. Verbose level `ClassField` (max 5). | Decomposition of the defining group into cyclic factors. |
| `Generators(A)` | Returns: (1) a sequence of generating elements for `NumberField(A)`; (2) the same elements viewed in the Kummer extension; (3) the images of (2) under the generator of the automorphism group for each cyclic factor. | Kummer-theoretic generator extraction. |

---

## 39.5 Invariants

Many invariants of an abelian extension can be computed directly from the ideal group
without first computing defining equations.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Discriminant(A)` | Discriminant of the class field A and its signature (second return value). Does not require defining equations. | Conductor-discriminant formula **[Coh00, Section 3.5.2]**. |
| `AbsoluteDiscriminant(A)` | The absolute discriminant of A as a number field over Q. | — |
| `Conductor(A)` | The conductor of A: the smallest ideal and smallest set of infinite places needed to define A. | Algorithm of **[Pau96, HPP97]**. |
| `Degree(A)` | The degree of the abelian extension A (over its base field). | Read from the defining group. |
| `AbsoluteDegree(A)` | The degree of A over Q. | Product of degree and absolute degree of the base field. |
| `CoefficientRing(A)` `CoefficientField(A)` `BaseField(A)` | The base field of A: `FieldOfFractions(BaseRing(A))`. | — |
| `BaseRing(A)` `CoefficientRing(A)` | The base ring of A: the maximal order used to define the underlying ray class group. | — |
| `NormGroup(A)` | The norm group (the subgroup H of Im used to define A via the Artin map). | Direct access to the defining data. |
| `DecompositionField(p, A)` | The decomposition field of the finite prime p (or place p) in A, returned as an abelian (sub)extension. | Fixed field of the decomposition group. |
| `DecompositionGroup(p, A)` | The decomposition group of the finite prime p (or place p) in A, as a subgroup of the norm group. | Local analysis via the ideal group. |
| `DecompositionType(A, p)` | The decomposition type of the finite prime ideal p (or place p, or prime number p) in A, as a sequence of pairs ⟨f, e⟩ (residue degree and ramification index). For the prime-number variant, parameter `Normal` (bool, default `false`; if true assumes the base field is normal, speeds up computation). | Artin map / local class field theory. |
| `DecompositionTypeFrequency(A, l)` | Decomposition types of all elements in the list l (anything for which `DecompositionType` is defined), returned as a multiset. Parameter `Normal` (default `false`). | Calls `DecompositionType` for each element. |
| `DecompositionTypeFrequency(A, a, b)` | Decomposition types over Q of all primes a ≤ p ≤ b in A, returned as a multiset. Parameter `Normal` (default `false`). | Sieve over primes in [a, b]. |

---

## 39.6 Automorphisms

The relative automorphism group of an abelian extension is isomorphic to the defining ideal
group via the Artin map. After defining equations are known, ideals coprime to the modulus
can be mapped explicitly to automorphisms.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ArtinMap(A)` | Returns a map from the defining group (viewed as a "subgroup" of the ideals of the base ring) to the automorphisms of A over the base field. By the defining property of class fields, induces an isomorphism of the defining group onto the relative automorphisms. May trigger a lengthy defining-equation computation. | Artin map via Kummer theory / class field construction. |
| `FrobeniusAutomorphism(A, p)` | The relative automorphism of A that is the Frobenius automorphism of p. May trigger a defining-equation computation. | Artin map evaluation at p. |
| `AutomorphismGroup(A)` | If `IsNormal` is true for A (with given `All`/`Over` parameters), returns the automorphism group of A over k0. May trigger a defining-equation computation. Parameters: `All` (bool, default `false`), `Over` ([Map], default `[]`). | Group extension construction from the ideal group structure. |
| `ProbableAutomorphismGroup(A)` | For A and its base field k both normal over Q: sets up the group extension problem for Gal(A/Q) as an extension of the abelian part by Gal(k/Q), uses `DistinctExtensions` to find all group-theoretic possibilities, then attempts to select the correct one by cycle-type frequency analysis. Result is heuristic (not guaranteed). Parameter `Factor` (int, default 1; passed to `ImproveAutomorphismGroup`). | Group extension enumeration + cycle-type frequency analysis. Some groups cannot be distinguished even with large `Factor`. |
| `ImproveAutomorphismGroup(F, E)` | Given the output of `ProbableAutomorphismGroup` or a previous call to `ImproveAutomorphismGroup`: splits more primes to gather additional cycle-type frequency data and attempts to narrow the candidate list. Parameter `Factor` (int, default 1; controls how many primes are split). | Extended cycle-type frequency analysis. |
| `AbsoluteGaloisGroup(A)` | The Galois group of A over Q (abstract automorphism group of a Q-normal closure), returned as a permutation group with roots in a local field. Requires defining equations to be known, but is considerably faster than calling `GaloisGroup` directly on the number field. | Exploits abelian structure; third return value gives data for further computations. |
| `TwoCocycle(A)` | For A normal over Q and defined over a normal base field k/Q: computes an explicit 2-cocycle with values in the norm group representing Gal(A/Q) as a group extension of Gal(k/Q) by the abelian part. Usable as an element of the second cohomology group of `CohomologyModule(A)`. | 2-cohomology of the Galois module structure. |

*Worked examples: H39E7 (ProbableAutomorphismGroup for C2²-extensions of Q(√10); disambiguation via cycle-type orders and ImproveAutomorphismGroup with verbose output).*

---

## 39.7 Norm Equations

For cyclic fields, Hasse's norm theorem guarantees that local solvability everywhere implies
global solvability of norm equations. For non-cyclic fields (e.g. Klein V4) this local-global
principle can fail; the obstruction is measured by the **knot** — the quotient of everywhere-
local-norms by global norms. Local solvability is decidable from the ideal groups alone
(without defining equations), so "local" functions are fast.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsLocalNorm(A, x, p)` | True iff x is a local norm in A at the finite prime p (equivalently, x is a norm in the completion at p). | Local class field theory on the ideal group. |
| `IsLocalNorm(A, x, i)` | True iff x is a local norm in A at the infinite prime i. | Sign condition at the real place. |
| `IsLocalNorm(A, x, p)` | True iff x is a local norm in A at the place p. | Local class field theory. |
| `IsLocalNorm(A, x)` | True iff x is a local norm everywhere in A. | Local class field theory at all places. |
| `Knot(A)` | The knot: the quotient group of everywhere-local-norms by global norms in the base field of A. Trivial knot means Hasse's theorem applies. For cyclic fields the knot is always trivial. | Computed from the ideal group structure without defining equations. |
| `NormEquation(A, x)` | Checks if x is a global norm, and if so returns a preimage element. Steps: (1) verify local norms; (2) compute defining equations; (3) combine solutions from maximal p-subfields. If the knot is non-trivial, step (3) may fail. | Local-global patching using maximal p-subfields **[Coh00]**. |
| `IsNorm(A, x)` | Tests whether x is a global norm. If locally a norm everywhere and the knot is trivial, returns true. If the knot is non-trivial, calls `NormEquation`. | Local check then knot test; delegates to `NormEquation` if needed. |

*Worked examples: H39E8 (norm equations for elements of norm 2 and 5 in Q(ζ5)(η), degree 80 over Q; using conductor to speed up, demonstrating non-trivial knot C2, and solving NormEquation reducing to degree 16).*

---

## 39.8 Attributes

Technical section exposing the internal representation of abelian extensions as read-only
attributes, useful for extending the package.

### 39.8.1 Orders

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `o'CyclotomicExtensions` | A read-only attribute on an order o. If defined, a list of records — one per cyclic prime-power factor of the class group — each describing a cyclotomic extension O = o[ζl] for a prime power l = pⁿ. Record components: `Abs` (maximal order Oa of O as an absolute extension, in optimised representation); `Rel` (O as an extension of o); `p2n` (the order l); `Zeta` (a primitive l-th root of unity in Abs); `Aut` (list of records for generators of Aut(O/o) — if p is odd, length 1 — each with: `Aut'Abs` the automorphism of Oa, `Aut'Rel` the automorphism of O (not necessarily an o-automorphism), `Aut'Order` the order, `Aut'r` where the automorphism sends ζl to ζlʳ). | Stored as a side effect of `EquationOrder`/`NumberField`. |

*Worked examples: H39E9 (Hilbert class field of Q(√−1001) with class group C2 × C2 × C10; inspecting two CyclotomicExtensions records of orders 2 and 5; verifying ζ5 images under the degree-4 automorphism).*

### 39.8.2 Abelian Extensions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `A'Components` | Read-only attribute. A record for each cyclic prime-power factor of the abelian extension, assigned after `NumberField(A)` or `EquationOrder(A)` is called. Record components: `Basis` (matrix B of order elements forming a multiplicative basis); `GenRaw` (exponent vector G with a = ∏ B[1,i]^G[i,1]); `UnitsRaw` (matrix U with uj = ∏ B[1,i]^U[j,i]); `S` (list of prime ideals in the S-unit support); `Gen` (a as an element of Oa); `GenAut` (image of the class field generator in the field of fractions of O under a generator of the cyclic automorphism group); `GenInv` (1/a in the field of fractions of Oa); `O` (the big Kummer extension Oa(a^(1/l))); `ClassField` (equation order for the cyclic extension over o); `Artin` (Artin map on the big Kummer extension). | Set by `EquationOrder`/`NumberField`. |
| `A'DefiningGroup` `A'NormGroup` | Read-only attributes. `DefiningGroup` is the ideal group map used to create A. `NormGroup` is the group defined modulo the conductor (requires prior call to `Conductor`). Both are records with components: `Map` (the map G → ideals), `m0` (finite part of the modulus), `m_inf` (infinite part), `RcgMap` (ray class group map, if present), `GrpMap` (the "rest" of Map such that Map = RcgMap ∘ GrpMap). | Set at construction; `NormGroup` set after `Conductor`. |
| `A'IsAbelian` | Stores the result of a prior call to `IsAbelian(A : All := true)`. | Cached boolean. |
| `A'IsNormal` | Stores the result of a prior call to `IsNormal(A : All := true)`. | Cached boolean. |
| `A'IsCentral` | Stores the result of a prior call to `IsCentral(A : All := true)`. | Cached boolean. |

*Worked examples: H39E10 (3-part of the ray class field modulo 36 over Z[√10]; inspecting DefiningGroup record structure; computing NumberField to assign Components; reconstructing the Kummer generator from Basis/GenRaw; verifying S-unit group structure).*

---

## 39.9 Group Theoretic Functions

### 39.9.1 Generic Groups

Generic groups are finite groups defined by generators with implicit relations, requiring
user-supplied multiplication and equality functions. Used in the class field package for
automorphism groups when one knows certain automorphisms (as maps) and wishes to
enumerate the group they generate. All functions enumerate all elements, so the group must
be small.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `GenericGroup(X)` | Creates the group G generated by the elements of X (assumed finite). Returns G and a map from G to the list of elements of the same type as X. Parameters: `Mult` (multiplication function, default `'*'`), `Eq` (equality function, default `'eq'`), `Id` (identity element), `Verbose` (level for `GrpGen`, max 3). | Full group enumeration (all elements). |
| `AddGenerator(G, x)` | Adds a new generator x to the generic group G (returned by `GenericGroup`). If x was already in G, returns false with other values unassigned. Otherwise returns the new group and the corresponding map. | Version of Dimino's algorithm **[But91a]** to enumerate all elements with minimal operations. |
| `FindGenerators(G)` | For a generic group G (returned by `GenericGroup`): find a small set of generators. | — |

---

## 39.10 Bibliography

| Key | Reference |
|-----|-----------|
| **[But91a]** | Gregory Butler. *Dimino's Algorithm*, pages 13–23. Volume 559 of LNCS [But91b], 1991. |
| **[But91b]** | Gregory Butler. *Fundamental Algorithms for Permutation Groups*, volume 559 of LNCS. Springer-Verlag, 1991. |
| **[CDO96]** | Henri Cohen, Francisco Diaz y Diaz, and Michel Olivier. *Computing Ray Class Groups, Conductors and Discriminants.* In Cohen [Coh96], pages 52–59. |
| **[CDO97]** | Henri Cohen, Francisco Diaz y Diaz, and Michel Olivier. *Computing Ray Class Groups, Conductors and Discriminants.* Submitted to Math. Comp., 1997. |
| **[Coh96]** | Henri Cohen, editor. *ANTS II*, volume 1122 of LNCS. Springer-Verlag, 1996. |
| **[Coh00]** | Henri Cohen. *Advanced Topics in Computational Number Theory.* Springer, Berlin–Heidelberg–New York, 2000. |
| **[Fie00]** | Claus Fieker. *Computing Class Fields via the Artin Map.* Math. Comput., **70**(235):1293–1303, 2000. |
| **[HPP97]** | Florian Heß, Sebastian Pauli, and Michael E. Pohst. *On the computation of the multiplicative group of residue class rings.* Math. Comp., 1997. |
| **[Pau96]** | Sebastian Pauli. *Zur Berechnung von Strahlklassengruppen.* Diplomarbeit, Technische Universität Berlin, 1996. URL: http://www.math.tu-berlin.de/~kant/publications/diplom/pauli.ps.gz. |
| **[Sut12]** | Nicole Sutherland. *Efficient Computation of Maximal Orders of Radical (including Kummer) Extensions.* Journal of Symbolic Computation, **47**(5):552–567, 2012. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Ray class group (Pauli–Hasse + Cohen) **[Pau96, HPP97, CDO96, CDO97, Coh00]** | `RayClassGroup`, `RayResidueRing`, `Conductor` |
| p-Selmer groups (residue-order identification) | `pSelmerGroup` |
| Induced maps on ray class groups (small-norm generators) | `InducedMap`, `InducedAutomorphism` |
| Class field construction via Artin map / Kummer theory **[Fie00, Coh00]** | `EquationOrder`, `NumberField`, `Components`, `Generators`, `ArtinMap`, `FrobeniusAutomorphism` |
| Maximal order of Kummer extensions **[Sut12]** | `MaximalOrder(:Al:="Kummer")` |
| Conductor-discriminant formula **[Coh00, §3.5.2]** | `Discriminant`, `AbsoluteDiscriminant`, `Conductor` |
| Heuristic abelian-subfield identification **[Coh00, Algorithm 4.4.3]** | `MaximalAbelianSubfield` |
| Artin map / Frobenius / decomposition | `ArtinMap`, `FrobeniusAutomorphism`, `DecompositionField`, `DecompositionGroup`, `DecompositionType`, `DecompositionTypeFrequency` |
| Galois module / cohomology structure | `IsAbelian`, `IsNormal`, `IsCentral`, `GenusField`, `H2GA`, `NormalSubfields`, `AbelianSubfield`, `FixedField`, `CohomologyModule`, `TwoCocycle` |
| Local class field theory / norm equations | `IsLocalNorm`, `Knot`, `NormEquation`, `IsNorm` |
| Group extension enumeration + cycle-type analysis | `ProbableAutomorphismGroup`, `ImproveAutomorphismGroup`, `AbsoluteGaloisGroup`, `AutomorphismGroup` |
| Dimino's algorithm (generic group enumeration) **[But91a]** | `GenericGroup`, `AddGenerator`, `FindGenerators` |
