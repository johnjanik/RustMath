# Chapter 110 — Invariant Theory

**Handbook part:** XIV — Commutative Algebra
**Handbook pages:** 3354–3398 (PDF pages 3486–3531)

---

## Scope and overview

Chapter 110 documents Magma's invariant theory module, which computes generators of
invariant rings and fields of finite groups and linear algebraic groups. The algorithms for
finite-group invariant theory descend from G. Kemper's Invar package (originally in Maple)
**[Kem96]**, with many new ideas described in a subsequent joint paper with A. Steel **[KS97]**;
readers making serious use of these functions are directed to that paper. Since V2.14, Magma
also implements Derksen's algorithm for linearly reductive algebraic groups **[Der99]** and the
Beth–Müller-Quade algorithm for invariant fields **[MQB99]**, both using code by G. Kemper.

The ground field K may have arbitrary characteristic; the modular case (char(K) divides
|G|) is of particular interest because many theoretical questions remain open. For a finite
group G acting on the n-dimensional vector space V ≅ Kⁿ with basis x₁,…,xₙ, the invariant
ring R = K[V]^G is the set of polynomials fixed by the group action; its category is
`RngInvar`. Invariant fields K(V)^G (category `FldInvar`) are handled analogously.

The approach for finite groups has two major steps: (1) compute **primary invariants**
f₁,…,fₙ — algebraically independent homogeneous invariants such that R is a finitely
generated module over A = K[f₁,…,fₙ]; (2) compute **secondary invariants** generating R
as an A-module. Kemper's degree-optimal algorithm for primary invariants **[Kem99]** is used
by default. For algebraic groups, Derksen's algorithm **[Der99]** applies when the group is
linearly reductive.

Chapter organisation: §§110.2–110.7 set up the general framework; §110.8 constructs
invariants of specified degree; §§110.9–110.16 cover finite-group invariant rings; §§110.17–
110.18 present utility functions also useful outside invariant theory; §110.19 documents
low-level attribute control; §§110.20–110.21 cover algebraic groups and invariant fields;
§110.22 treats symmetric group invariants.

---

## 110.1 Introduction

*(Introductory prose; no intrinsics.)*

---

## 110.2 Invariant Rings of Finite Groups

### 110.2.1 Creation

The invariant ring R = K[V]^G is a lazy structure — no computation is triggered at
construction time.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `InvariantRing(G)` | Construct the invariant ring R = K[V]^G of a finite **matrix** group G; the coefficient field K is taken from G. The polynomial ring P is constructed automatically. | Lazy; no computation performed. |
| `InvariantRing(G, K)` | Construct the invariant ring R = K[V]^G of a finite **permutation** group G over the field K. | Lazy; no computation performed. |

### 110.2.2 Access

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Group(R)` | Return the group G of the invariant ring R = K[V]^G. | — |
| `CoefficientRing(R)` / `CoefficientField(R)` | Return the coefficient field K of R = K[V]^G. | — |
| `PolynomialRing(R)` | Return the polynomial ring P = K[x₁,…,xₙ] in which invariants of R lie (variable names "x1", "x2", …). | — |
| `f in R` | Test whether polynomial f belongs to R = K[V]^G (parent of f remains P, not R). | — |

---

## 110.3 Group Actions on Polynomials

*(Descriptive section; group actions are defined and the following two sections elaborate.)*

---

## 110.4 Permutation Group Actions on Polynomials

Sym(n) acts on a polynomial ring in n indeterminates by permuting indices: f(x₁,…,xₙ) ↦ f(x_{g(1)},…,x_{g(n)}).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `f ^ g` | For polynomial f in a ring of n indeterminates and permutation g ∈ Sym({1,…,n}): return the image of f under g. | Direct index permutation. |
| `f ^ G` | For polynomial f and permutation group G ⊆ Sym({1,…,n}): return the orbit of f under G as a set. | Orbit enumeration. |
| `IsInvariant(f, g)` | For polynomial f and permutation g (or matrix-group element g) of appropriate degree with matching coefficient ring: return whether f^g = f. | Equality test. |
| `IsInvariant(f, G)` | For polynomial f and permutation group G (or matrix group G) of appropriate degree: return whether f^g = f for all g ∈ G. | Orbit-stabiliser / direct check. |

---

## 110.5 Matrix Group Actions on Polynomials

GL(n, S) acts on K[x₁,…,xₙ] by (a·f)(x) = f(x·a).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `f ^ a` | For polynomial f with n indeterminates over ring S and matrix a in a subgroup of GL(n, S): return f(x·a). | Linear substitution. |
| `f ^ G` | For polynomial f and subgroup G of GL(n, S): return the orbit of f under G. | Orbit enumeration. |

*Worked example: H110E1 (action of a cyclic subgroup of GL(2, Q(√2)) on a quadratic form).*

---

## 110.6 Algebraic Group Actions on Polynomials

A linear algebraic group G is given by polynomials in t₁,…,tₘ defining G as an affine variety
over K̄. A G-module is given by an n×n matrix A with entries in K[t₁,…,tₘ]; a group element
(η₁,…,ηₘ) acts on K̄ⁿ by substituting ηᵢ into A. The action on polynomials is σ(f) = f∘σ⁻¹.
Since algorithms work only over K (not K̄), individual group elements are never directly used.

*(No new intrinsics; the framework is used via `InvariantRing(I, A)` in §110.20.)*

---

## 110.7 Verbosity

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SetVerbose("Invariants", v)` | (Procedure.) Set verbose output level for invariant theory algorithms. Legal values: `false`/`0` (silent), `true`/`1`–`4`. Level 1 gives minimal useful information; higher levels show more detail. For primary invariants, displays candidate degree lists; for secondary invariants, shows the loop over degrees (non-modular) or subgroup-based computation (modular). | — |

---

## 110.8 Construction of Invariants of Specified Degree

The homogeneous invariants of degree d in R form the vector space R_d over K. Two
construction methods are supported:

- **Reynolds operator method** — applies the averaging operator to a monomial to produce
  one non-zero invariant (or zero). Works for finite groups in the non-modular case; for
  permutation groups the simplified form is always available regardless of characteristic.
- **Linear algebra method** — finds a basis for all of R_d in one step, in both the modular
  and non-modular cases and (with modifications) for algebraic groups.

The default `"Both"` strategy applies an appropriate combination. See **[KS97]** for details.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ReynoldsOperator(f, G)` | Apply the Reynolds operator of matrix group G to polynomial f (need not be a monomial or homogeneous). Returns an invariant of the same degree, or zero. | Reynolds averaging (non-modular finite groups) **[KS97]**. |
| `InvariantsOfDegree(R, d)` / `InvariantsOfDegree(G, d)` / `InvariantsOfDegree(G, K, d)` / `InvariantsOfDegree(G, P, d)` | Compute a K-basis of R_d (homogeneous invariants of degree d) as a sequence of polynomials. Parameter `Invariants` selects the method: `"Reynolds"`, `"Linear"`, or `"Both"` (default). | Reynolds operator + linear algebra, combined adaptively **[KS97]**. |
| `InvariantsOfDegree(R, d, k)` / `InvariantsOfDegree(G, d, k)` / `InvariantsOfDegree(G, K, d, k)` / `InvariantsOfDegree(G, P, d, k)` | As above but compute only k ≤ dim(R_d) linearly independent invariants. Parameter `Invariants` as above. | Reynolds operator + linear algebra **[KS97]**. |
| `SetAllInvariantsOfDegree(R, d, Q)` | (Procedure.) Override the internally stored basis of R_d with the sequence Q. Q must span R_d; if the Hilbert series is known, its dimension is checked. | — |

*Worked examples: H110E2 (Reynolds operator and `InvariantsOfDegree` for a 3D matrix group over Q(ζ₅) and for C₄ over GF(2)); H110E3 (using `SetAllInvariantsOfDegree` to pre-supply invariants of degree 2 before calling `PrimaryInvariants`).*

---

## 110.9 Construction of G-modules

Finite-dimensional K[G]-modules arising from the action of G on polynomials.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `GModule(G, P, d)` | For finite permutation or matrix group G of degree n, polynomial ring P = K[x₁,…,xₙ], and non-negative integer d: the K[G]-module M for the action of G on homogeneous polynomials of degree d of P, plus isomorphism f and index set of monomials of degree d. | Module representation via the group action on monomials. |
| `GModule(G, I, J)` | For group G, ideal I, and zero-dimensional subideal J of I: the K[G]-module M for the action of G on the finite-dimensional quotient I/J, plus isomorphism f and monomial basis. | Quotient module from the group action on I/J. |
| `GModule(G, Q)` | For group G and finite-dimensional quotient ring Q = I/J: the K[G]-module M for the action of G on Q, plus isomorphism f and monomial basis. | Quotient module from the group action on Q. |

*Worked example: H110E4 (GL(3, GF(5)) acting on K[x,y,z]/(x⁵−x, y⁵−y, z⁵−z); constituents of the module for degree-4 polynomials).*

---

## 110.10 Molien Series

The Molien series of a finite group G, when it exists, equals the Hilbert series of K[V]^G.
For permutation groups it always exists; for matrix groups in the non-modular case it exists
and can be computed.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `MolienSeries(G)` | Molien series of G as an element of Z(t). For permutation groups: always exists and equals the Hilbert series for any field K. For matrix groups: requires char(K) coprime to |G|. | Molien's formula (sum over conjugacy classes of 1/det(I−tg)). |
| `MolienSeriesApproximation(G, n)` | Approximation to the Molien series of a permutation group G as a Laurent series with n known coefficients. Can handle far larger groups than `MolienSeries`. | Truncated Molien formula. |

*Worked example: H110E5 (Molien series of a matrix group over Q(ζ₅); verifying coefficients match counts from `InvariantsOfDegree`).*

---

## 110.11 Primary Invariants

Primary invariants {f₁,…,fₙ} are algebraically independent homogeneous invariants such
that R is a finitely generated module over A = K[f₁,…,fₙ]. They always exist, and Magma's
algorithm (due to Kemper **[Kem99]**) guarantees that the degrees found are optimal (minimal
product then minimal sum). Results are stored in R for reuse.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PrimaryInvariants(R)` | Construct optimal primary invariants for R = K[V]^G as a sorted sequence of n polynomials (increasing degree), where n is the degree of G. | Kemper's optimal-degree algorithm **[Kem99]**; detailed in **[KS97]**. |

*Worked example: H110E6 (primary invariants of "first A₅ in SL(F₂)" of degree 4; degrees 3,5,8,12 are proved optimal by **[Kem96]**).*

---

## 110.12 Secondary Invariants

Secondary invariants are generators of R as a module over A = K[f₁,…,fₙ]. They are
minimal (a minimal generating set for R over A) and stored in R. Different algorithms are
required for the modular and non-modular cases (see **[KS97]**).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SecondaryInvariants(R)` | Construct secondary invariants for R = K[V]^G (primary invariants computed first if necessary) as a sorted sequence of polynomials. Minimal generating set for R over the primary algebra. | Non-modular: Reynolds operator loop over required degrees. Modular: subgroup-based secondary + module syzygy computation **[KS97]**. |
| `SecondaryInvariants(R, H)` | Variant for modular R only: use subgroup H ≤ G. Secondary invariants for K[V]^H (with respect to the primaries of G) are computed first, then used as in **[KS97]**. Mainly useful when a particular subgroup H is better than the automatic strategy. | Subgroup strategy **[KS97]**. |
| `IrreducibleSecondaryInvariants(R)` | Return the irreducible secondary invariants (primary invariants + these generate R as an algebra over K). In the non-modular case these may be a proper subset of the secondary invariants; in the modular case they coincide with the secondary invariants minus 1. The expression of secondaries in terms of irreducible ones is the second return value of `Algebra(R)`. | Computed from secondary invariants. |

*Worked example: H110E7 (cyclic group C₄ over GF(2) in the modular case; Noether's degree bound is violated; primary + secondary invariants displayed).*

---

## 110.13 Fundamental Invariants

A set of fundamental invariants is a generating set for R as an algebra over K.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `FundamentalInvariants(R)` | Construct fundamental invariants for R = K[V]^G as a sorted sequence of polynomials (increasing degree). Parameter `Al`: `"King"` (default for non-modular, since V2.15) **[Kin07]** or `"MinPrimSec"` (minimises the union of primary + secondary invariants, always used in the modular case). Parameter `MaxDegree`: set to d if an a-priori degree bound is known, allowing early termination in the King algorithm (non-modular). | Non-modular: S. King's algorithm **[Kin07]** by default. Modular (or `Al := "MinPrimSec"`): minimisation of primary + secondary union. |

*Worked examples: H110E8 (two copies of S₃ in degree 6 over Q; fundamental invariants vs. primary+secondary); H110E9 (all transitive groups of degrees 7 and 8 in characteristic 0 and over Fₚ for non-dividing p, times reported; illustrates **[Kin07]** speedup); H110E10 (degree-10 representation of S₅; fundamental invariants over GF(7); comparison of permutation and matrix representations).*

---

## 110.14 The Module of an Invariant Ring

Given primary invariants {f₁,…,fₙ} and secondary invariants {g₁,…,gₘ}, Magma constructs
a graded polynomial algebra A′ = K[t₁,…,tₙ] (with deg(tᵢ) = deg(fᵢ)) and a graded module
M = A′ᵐ/Q where Q encodes the module syzygies of the gᵢ. The isomorphism f: R → M lets
one express any invariant as a sum Σ aᵢgᵢ with aᵢ ∈ A (unique up to syzygies).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Module(R)` | The graded module M isomorphic to R = K[V]^G, together with the isomorphism f: R → M. Coefficient ring of M has variables "t1","t2",…; weighted degrees equal degrees of primary invariants. | Module construction from primary + secondary invariants and their syzygies. |

*Worked example: H110E11 (4×4 Jordan block over GF(3); module structure displayed; invariant expressed in terms of primaries and secondaries).*

---

## 110.15 The Algebra of an Invariant Ring and Algebraic Relations

R is generated as an algebra by the primary invariants f₁,…,fₙ and irreducible secondary
invariants h₁,…,hᵣ. Magma constructs a polynomial algebra A with variables "f1",…,"fn",
"h1",…,"hr" (variable weights = degrees of respective invariants) and computes algebraic
relations, yielding a presentation R ≅ A/⟨relations⟩.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Algebra(R)` | Return the polynomial algebra A = K[f₁,…,fₙ, h₁,…,hᵣ] and a sequence Q expressing each secondary invariant as a monomial in the hᵢ (in A). The constant 1 is not an irreducible secondary so has no corresponding h-variable. | From primary + irreducible secondary invariants. |
| `Relations(R)` | Return a sorted sequence of algebraic relations among the algebra generators, as elements of A. So R ≅ A/⟨relations⟩. | Gröbner basis computation in A. |
| `RelationIdeal(R)` | The ideal in A generated by `Relations(R)`. | Ideal of algebraic relations. |
| `PrimaryAlgebra(R)` | The graded polynomial ring corresponding to the primary invariants of R (with weights = degrees of the primary invariants). | — |
| `PrimaryIdeal(R)` | The ideal in P generated by the primary invariants of R (stored in R). | — |

*Worked example: H110E12 (C₃×C₃ permutation group in degree 6 over Q; algebra A with 8 variables; two algebraic relations; homomorphism from A to R and Hilbert series check).*

---

## 110.16 Properties of Invariant Rings

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HilbertSeries(R)` | Hilbert series of R = K[V]^G as an element of Z(t). Uses Molien series if available; otherwise computes secondary invariants. | Molien series (non-modular permutation/matrix case) or secondary-invariant construction (modular case). |
| `HilbertSeriesApproximation(R, n)` | Hilbert series of R as a Laurent series with n known terms, computed using conjugacy classes of G. | Conjugacy-class averaging. |
| `IsCohenMacaulay(R)` | Whether R is Cohen-Macaulay. Always true in the non-modular case; in the modular case secondary invariants are constructed to determine the result. | Auslander–Buchsbaum + module structure (modular case). |
| `FreeResolution(R)` | Free resolution of (the module of) R; same as `FreeResolution(Module(R))`. Returns a sequence F: F[1] = M, F[i+1] = syzygy module of F[i], last element is free. | Syzygy computation over the primary algebra. |
| `MinimalFreeResolution(R)` | Minimal free resolution of (the module of) R; same as `MinimalFreeResolution(Module(R))`. | Minimal syzygy computation. |
| `HomologicalDimension(R)` | Homological dimension of R: length of the minimal free resolution minus 1 (M itself is included). | From minimal free resolution. |
| `Depth(R)` | Depth of R; equal to n − d by the Auslander–Buchsbaum formula, where n is the rank and d is the homological dimension. | Auslander–Buchsbaum formula. |

*Worked example: H110E13 (degree-5 Jordan block over GF(2); minimal free resolution; depth 3 verified via `HomologicalDimension`).*

---

## 110.17 Steenrod Operations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SteenrodOperation(f, i)` | The i-th Steenrod operation P^i(f) of multivariate polynomial f with coefficients in a finite field; i must be a non-negative integer. | Steenrod algebra action on polynomial rings in characteristic p. |

*Worked example: H110E14 (group F₄ over GF(3); applying Steenrod operations to obtain degree-4 and degree-10 invariants; membership test in R).*

---

## 110.18 Minimalization and Homogeneous Module Testing

General-purpose functions (also documented in the Multivariate Polynomials chapter) that
are heavily used in invariant theory to express invariants in terms of primaries and secondaries.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `MinimalAlgebraGenerators(L)` | For a set or sequence L = {p₁,…,pₖ} of polynomials in R = K[x₁,…,xₙ]: return a minimal generating set of the subalgebra K[p₁,…,pₖ] as a sorted sequence of elements from L. | Gröbner basis / membership test iteratively removing redundant generators. |
| `HomogeneousModuleTest(P, S, F)` | For homogeneous P = [p₁,…,pₖ], S = [s₁,…,sᵣ], and a single element F ∈ R: test whether F lies in the A-module M = A[s₁,…,sᵣ] (A = K[p₁,…,pₖ]). If true, also return coefficients cᵢ ∈ K[t₁,…,tᵣ] such that F = Σ cᵢ(p₁,…,pₖ)·sᵢ. | Gröbner basis over A. |
| `HomogeneousModuleTest(P, S, L)` | As above but for a sequence L of l elements homogeneous of degree d. Returns parallel boolean sequence B and coefficient sequence V. B[i] = true iff L[i] ∈ M; V[i] = the coefficients (or zero coefficients if B[i] = false). | Gröbner basis over A applied to each element. |

*Worked examples: H110E15 (using `MinimalAlgebraGenerators` on P∪S to compute fundamental invariants of two copies of S₃ in degree 6); H110E16 (`HomogeneousModuleTest` for C₄ over GF(2): expressing S[2]² in terms of P and S; writing all 14 degree-5 invariants in terms of P and S).*

---

## 110.19 Attributes of Invariant Rings and Fields

Low-level attributes for direct inspection or setting of cached data. Setting an incorrect
value causes unpredictable results; use `assigned` to test whether an attribute is set before
referring to it with the `'` operator.

| Attribute | Description |
|-----------|-------------|
| `R'PrimaryInvariants` | Primary invariants of R. Read: returns current value or errors if unset. Write: sequence of n algebraically independent invariants of G; if already set, new value must match. Useful for supplying specially constructed primaries before computing secondaries. |
| `R'SecondaryInvariants` | Secondary invariants of R. Read: returns current value or errors if unset. Write: requires primary invariants already defined; Q must be secondary invariants w.r.t. those primaries. If already set, new value must match. |
| `R'HilbertSeries` | Hilbert series of R. Read: computes and returns if unset. Write: H must be a rational function in Z(t) equal to the Hilbert series of R. If already set, new value must match. |

*Worked example: H110E17 (manually setting `R'HilbertSeries`; cross-setting primary invariants of a subgroup ring RH to the primaries of RG; then computing secondary invariants of RH with respect to those primaries).*

---

## 110.20 Invariant Rings of Linear Algebraic Groups

A linear algebraic group G is defined by an ideal I of polynomials (the affine variety defining G)
and an n×n representation matrix A with polynomial entries. Magma makes no check that the
variety is actually a group or that A defines a morphism; incorrect input gives unpredictable
results. The user must declare reductivity properties at creation time. Category is `RngInvar`.

### 110.20.1 Creation

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `InvariantRing(I, A)` | Construct the invariant ring R for the algebraic group G defined by ideal I and representation matrix A. Parameters: `Reductive` (BoolElt, default `false`) — assume G is reductive; `LinearlyReductive` (BoolElt, default `false`) — assume G is linearly reductive (required for Derksen's algorithm); `PolynomialRing` (RngMPol, default auto) — use a specified polynomial ring P for the invariants. | Lazy construction; Derksen's algorithm **[Der99]** invoked when `FundamentalInvariants` is called (requires `LinearlyReductive`). |
| `BinaryForms(N, p)` / `BinaryForms(n, p)` | For N = [n₁,…,nₖ] a sequence of positive integers and p a prime or 0: define the action of G = SL₂(K̄) (char K = p) on a direct sum of spaces of binary forms of degrees nᵢ. Returns: ideal IG defining G, representation matrix A (as sequence of sequences of polynomials), and a polynomial ring with appropriate variable names. Second form takes a single integer n and sets N = [n]. | Standard SL₂ action on binary forms. |

### 110.20.2 Access

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `GroupIdeal(R)` | Return the ideal I defining the algebraic group G of invariant ring R. | — |
| `Representation(R)` | Return the representation matrix A for the algebraic group G of invariant ring R. | — |

### 110.20.3 Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `InvariantsOfDegree(R, d)` | K-basis of R_d (homogeneous invariants of degree d) in the invariant ring R of an algebraic group, as a sequence of polynomials. | Linear algebra method (works for algebraic groups). |
| `FundamentalInvariants(R)` | Fundamental invariants of R (algebraic group), as a sequence. Parameters: `Optimize` (BoolElt, default `true`) — extend basis at each degree by multiplying lower-degree invariants by monomials; `Minimize` (BoolElt, default `true`) — minimise the generators; `MinimizeHilbert` (BoolElt, default `true`) — minimise the Hilbert ideal basis; `Force` (BoolElt, default `false`) — force Derksen's algorithm even if group may not be linearly reductive. Requires `LinearlyReductive` (or `Force := true`). | Derksen's algorithm **[Der99]**. |
| `DerksenIdeal(R)` | Generators of the Derksen ideal of R: an ideal D of P[y₁,…,yₙ] defined as the intersection of ideals ⟨y₁−g(x₁),…,yₙ−g(xₙ)⟩ over all g ∈ G; geometrically the vanishing ideal of {(x,g(x)) : x ∈ Kⁿ, g ∈ G}. Returned with a Gröbner basis. | Derksen's algorithm **[Der99]**. |
| `HilbertIdeal(R)` | Hilbert ideal of R: the ideal in P generated by all non-constant homogeneous invariants. Returns a sequence of homogeneous generators (not necessarily invariant). Parameters: `Minimize` (BoolElt, default `true`); `Force` (BoolElt, default `false`). | Via Derksen's algorithm **[Der99]**. |

*Worked examples: H110E18 (SL₂(Q) acting simultaneously on 3 vectors; 3 fundamental invariants = 2×3 matrix minors); H110E19 (SL₂(Q)×SL₂(Q)×SL₂(Q) via tensor product; single fundamental invariant); H110E20 (SL₂(Q) acting on binary forms of degrees [1,1,2,2] via `BinaryForms`; 13 fundamental invariants); H110E21 (non-reductive algebraic group; `FundamentalInvariants` raises a runtime error; `InvariantsOfDegree` of degrees 1 and 2 computed successfully).*

---

## 110.21 Invariant Fields

The invariant field K(x₁,…,xₙ)^G is the subfield of K(V) fixed by G. The category is
`FldInvar`. Arguments and access functions mirror those of `InvariantRing`.

### 110.21.1 Creation

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `InvariantField(G, K)` | Create the invariant field for permutation group G over field K. | Lazy construction. |
| `InvariantField(G)` | Create the invariant field for matrix group G (field K from G). | Lazy construction. |
| `InvariantField(I, A)` | Create the invariant field for the algebraic group defined by ideal I and representation matrix A. Parameters: `Reductive`, `LinearlyReductive`, `FunctionField` (FldFunRat, default auto). | Lazy construction. |

### 110.21.2 Access

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `FunctionField(F)` | Return the underlying rational function field of invariant field F. | — |
| `Group(F)` | Return the underlying group of invariant field F. | — |
| `GroupIdeal(F)` | For invariant field F over an algebraic group: return the ideal I defining G. | — |
| `Representation(F)` | For invariant field F over an algebraic group: return the representation matrix A. | — |

### 110.21.3 Functions for Invariant Fields

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `FundamentalInvariants(F)` | Fundamental invariants of invariant field F as a sequence generating F as an algebra over the base field. Parameters: `Al` (`"BethMuellerQuade"` default, or `"FleischmannKemperWoodcock"` for the Fleischmann–Kemper–Woodcock alternative); `Minimize` (BoolElt, default `true`) — return irredundant set; `Min` (RngIntElt, default 0) — stop when generating set of this size is reached; `BottomUpTo` (RngIntElt, default 0) — for minimisation, first try small subsets. | Beth–Müller-Quade algorithm **[MQB99]** by default; Fleischmann–Kemper–Woodcock alternative via parameter. |
| `DerksenIdeal(F)` | Derksen ideal of invariant field F: an ideal D in K[y₁,…,yₙ] (K = k(x₁,…,xₙ)) defined as the intersection of ideals ⟨y₁−g(x₁),…,yₙ−g(xₙ)⟩ for g ∈ G. Returned with a Gröbner basis. | Derksen-ideal construction. |
| `MinimizeGenerators(L)` | For L a set or sequence of non-constant elements of a rational function field: select a minimal (irredundant) subset generating the same subfield. Parameters: `Min` (RngIntElt, default 0) — stop at a generating set of this size; `BottomUpTo` (RngIntElt, default 0) — use bottom-up approach up to this size. Returns a sequence of minimal generators. | Subfield membership testing. |
| `QuadeIdeal(L)` | For L a non-empty set or sequence of non-constant elements from F = k(x₁,…,xₙ) generating K = k(L): the Quade ideal in F[y₁,…,yₙ] (kernel of K[y₁,…,yₙ] → F, yᵢ ↦ xᵢ), introduced in **[MQS99]**, returned with a Gröbner basis. Parameters: `Fy` (polynomial ring of rank n over F, to make result an ideal of a specific ring); `LargeIdeal` (BoolElt, default `false`) — return an ideal in a larger ring whose intersection with F[y₁,…,yₙ] is the Quade ideal. | Quade ideal construction **[MQS99]**. |

*Worked examples: H110E22 (C₃ over Q; `FundamentalInvariants` and `DerksenIdeal` of an invariant field; Gröbner basis displayed); H110E23 (non-reductive algebraic group; invariant field; `FundamentalInvariants` succeeds (not restricted for fields); `DerksenIdeal` displayed).*

---

## 110.22 Invariants of the Symmetric Group

Basic functions for symmetric polynomials (invariants of the symmetric group).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ElementarySymmetricPolynomial(P, k)` | For polynomial ring P of rank n and 1 ≤ k ≤ n: return the k-th elementary symmetric polynomial of P. | Direct construction: sum of all squarefree degree-k monomials in the variables. |
| `IsSymmetric(f)` | For polynomial f in a ring P of rank n: test whether f is symmetric in all n variables. If true, also return a polynomial g (in a new ring of rank n) such that f = g(e₁,…,eₙ) where eᵢ is the i-th elementary symmetric polynomial of P. | Rewriting f in terms of elementary symmetric polynomials. |
| `IsSymmetric(f, S)` | As `IsSymmetric(f)`, but g is returned as a member of the specified polynomial ring S of rank n (for predetermined variable names). | As above. |

*Worked examples: H110E24 (symmetric polynomial in Q[a,b,c,d] expressed as e₁−e₂e₃+e₂e₄; a parametric family of symmetric polynomials in 5 variables over k(a,b) expressed in elementary symmetric polynomials).*

---

## 110.23 Bibliography

| Key | Reference |
|-----|-----------|
| **[AM94]** | A. Adem and R. J. Milgram. *Cohomology of Finite Groups.* Grundlehren der Mathematischen Wissenschaften. Springer, Berlin–New York–Heidelberg, 1994. |
| **[Der99]** | Harm Derksen. *Computation of Invariants for Reductive Groups.* Adv. Math., 141:366–384, 1999. |
| **[Kem96]** | Gregor Kemper. *Calculating Invariant Rings of Finite Groups over Arbitrary Fields.* J. Symbolic Comp., 21(3):351–366, 1996. |
| **[Kem99]** | Gregor Kemper. *An Algorithm to Calculate Optimal Homogeneous Systems of Parameters.* J. Symbolic Comp., 27(2):171–184, 1999. |
| **[Kin07]** | Simon King. *Minimal generating sets of non-modular invariant rings of finite groups.* URL: http://arxiv.org/abs/math/0703035, 2007. |
| **[KS97]** | Gregor Kemper and Allan Steel. *Some Algorithms in Invariant Theory of Finite Groups.* In P. Dräxler, G. O. Michler, and C. M. Ringel, editors, Computational Methods for Representations of Groups and Algebras, Euroconference in Essen, April 1–5 1997, number 173 in Progress in Mathematics, Basel, 1997. Birkhäuser. |
| **[MQB99]** | Jörg Müller-Quade and Thomas Beth. *Calculating Generators for Invariant Fields of Linear Algebraic Groups.* In Applied Algebra, Algebraic Algorithms and Error-Correcting Codes (Honolulu, HI, 1999), number 1719 in LNCS, pages 392–403, Berlin, 1999. Springer. |
| **[MQS99]** | Jörg Müller-Quade and Rainer Steinwandt. *Basic algorithms for rational function fields.* J. Symbolic Comp., 27(2):143–170, 1999. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Reynolds operator (non-modular finite groups) **[KS97]** | `ReynoldsOperator`, `InvariantsOfDegree` |
| Linear algebra method for invariants **[KS97]** | `InvariantsOfDegree`, `InvariantsOfDegree(R,d)` (algebraic groups) |
| Kemper optimal primary invariants **[Kem99]** | `PrimaryInvariants` |
| Secondary invariants (modular + non-modular) **[KS97]** | `SecondaryInvariants`, `IrreducibleSecondaryInvariants` |
| King's fundamental invariants **[Kin07]** | `FundamentalInvariants(R)` (non-modular, default Al) |
| Minimisation of primary + secondary union | `FundamentalInvariants(R: Al:="MinPrimSec")` (modular) |
| Molien series | `MolienSeries`, `MolienSeriesApproximation`, `HilbertSeries(R)` |
| Module/algebra structure of R | `Module`, `Algebra`, `Relations`, `RelationIdeal`, `PrimaryAlgebra`, `PrimaryIdeal` |
| Free resolution / depth / Cohen-Macaulay | `FreeResolution`, `MinimalFreeResolution`, `HomologicalDimension`, `Depth`, `IsCohenMacaulay` |
| Steenrod operations | `SteenrodOperation` |
| Algebra minimisation / module membership | `MinimalAlgebraGenerators`, `HomogeneousModuleTest` |
| Derksen's algorithm (linearly reductive algebraic groups) **[Der99]** | `FundamentalInvariants(R)` (algebraic group), `DerksenIdeal(R)` |
| Beth–Müller-Quade algorithm (invariant fields) **[MQB99]** | `FundamentalInvariants(F)` (default Al) |
| Quade ideal **[MQS99]** | `QuadeIdeal` |
| Field generator minimisation | `MinimizeGenerators`, `FundamentalInvariants(F)` |
| Elementary symmetric polynomials | `ElementarySymmetricPolynomial`, `IsSymmetric` |
