# Chapter 68 — Cohomology and Extensions

**Handbook part:** IX — Finite Groups
**Handbook pages:** 2013–2037 (PDF pages 2144–2173)

---

## Scope and overview

Chapter 68 provides a flexible set of tools for computing with first and second cohomology groups of any type of finite group acting on any reasonable module, including a module defined by an action on an arbitrary finitely generated abelian group. First (but not second) cohomology groups can also be calculated for infinite groups defined by a finite presentation.

Zero-cocycles, one-cocycles and two-cocycles may be computed and identified. Extensions of modules by groups can be constructed as finitely presented groups, or as PC-groups when the acting group is a PC-group. A representative set of extensions of the module by the group — each distinct up to group isomorphism fixing the module — can also be computed. These functions complement, but do not completely supplant, an older collection of functions pertaining to cohomology groups, Schur multiplicators and covering groups that apply to permutation groups (see Chapter 58 on Permutation Groups).

H¹(G, M) is calculated as the nullspace of a certain matrix; see Section 5 of **[CCH01]** for details. H²(G, M) can also be found as the nullspace of a suitable matrix, but this matrix can be uncomfortably large in big examples. For soluble groups defined by a PC-presentation, the matrix corresponds to solving the consistency equations for a PC-presentation of a general extension of the module by the group, which depends on the number of group generators rather than its order, making it manageable for quite large groups. For permutation and matrix groups G the matrix is much larger, but can often be reduced using a base and strong generating set. When only the dimension of H²(G, M) is required and M is a module over a finite field of prime order p, the calculation can be reduced to H²(Q, M) for a suitable collection of p-subgroups Q of G, carried out efficiently via the PC-presentation approach **[Hol85b]**.

Section 68.10 adds a separate framework for computing the first cohomology H¹(Γ, A) of a finite group with coefficients in a finite (not necessarily abelian) group, based on **[Hal05]**.

---

## 68.1 Introduction

Introductory section; see Scope and overview above.

---

## 68.2 Creation of a Cohomology Module

To compute cohomology of a group with respect to a G-module M, the user must first construct a cohomology module object. All subsequent cohomology functions take this object as their first argument. For PC-groups the PC-presentation must be conditioned beforehand via `G := ConditionedGroup(G)`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CohomologyModule(G, M)` | Given a group G (finite permutation group, finite matrix group, PC-group, or any finitely presented group) and a G-module M with acting group G, returns a cohomology module for the action of G on M. | Constructs internal data structures for nullspace-based cohomology computation **[CCH01]**. |
| `CohomologyModule(G, Q, T)` | Given a group G acting on a finitely generated abelian group with invariants specified by the integer sequence Q and action described by T (a sequence of d×d integer matrices, one per generator of G), returns a cohomology module. G may be any of the types above; PC-presentation must be conditioned. | Action on finitely generated abelian group; same internal framework. |
| `CohomologyModule(G, A, M)` | For a permutation group G acting on some abelian group A through M, where M is either a map from G into the endomorphisms of A or a sequence of endomorphisms of A (one per generator of G), returns the cohomology module. | Direct action-on-abelian-group variant. |

*Worked example: H68E1 (PSL(3,2) acting on a dimension-3 GF(2)-module).*

---

## 68.3 Accessing Properties of the Cohomology Module

Each function in this section takes a cohomology module CM returned by `CohomologyModule` and retrieves properties used to define it.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Module(CM)` | Returns the K[G]-module used to define CM. Errors if CM was defined by an action on a finitely generated abelian group. | — |
| `Invariants(CM)` | If CM was defined by an action on a finitely generated abelian group A, returns the invariants of A. Errors if not defined that way. | — |
| `Dimension(CM)` | If CM was defined by a group acting on an R-module M, returns the dimension of M. If defined by an action on a finitely generated abelian group A, returns the rank of A. | — |
| `Ring(CM)` | The ring over which the module defining CM is defined. If defined by an action on a finitely generated abelian group A, the ring is the integers if A is infinite, or the integers modulo the exponent of A if A is finite. | — |
| `Group(CM)` | The group used to define the action on CM. | — |
| `FPGroup(CM)` | Given CM with associated group G, returns a finitely presented group F isomorphic to G and the isomorphism F → G. On a strong generating set if G is a permutation or matrix group. Used in the construction of presentations of extensions returned by `Extension`. | — |
| `MatrixOfElement(CM, g)` | The matrix representing the action of element g in the group of CM on the module of CM. | — |

---

## 68.4 Calculating Cohomology

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CohomologyGroup(CM, n)` | For cohomology module CM (group G acting on module M) and non-negative integer n ∈ {0, 1, 2}: returns the cohomology group Hⁿ(G, M). For modules over the integers only, n = 3 is also allowed (H³ computed as the second cohomology of M regarded as a module over Q/Z). If G was a finitely presented group, only n = 0 or 1 is allowed. | Nullspace of a suitable matrix **[CCH01]**; for PC-groups solves consistency equations for a PC-presentation of an extension **[Hol85b]**. |
| `CohomologicalDimension(CM, n)` | For cohomology module CM with module M over a finite field K and n ∈ {0, 1, 2}: returns the K-dimension of Hⁿ(G, M). When n = 2 this is faster than `CohomologyGroup(CM, 2)` but does not enable subsequent computation of explicit extensions or two-cocycles. Only applicable when CM was created via `CohomologyModule(G, M)` with M a module over a finite field. | For n = 2 and G a permutation or matrix group: reduces to computing H²(Q, M) for a collection of p-subgroups Q **[Hol85b]**. |
| `CohomologicalDimension(M, n)` | For K[G]-module M (K a finite field, G a finite group) and n ≥ 0: computes and returns the K-dimension of Hⁿ(G, M). For n = 0 and 1, uses `CohomologicalDimension(CM, n)`. For n ≥ 2, uses projective covers and dimension shifting recursively to reduce to n = 1. | Recursive dimension-shifting via projective covers. |
| `CohomologicalDimensions(M, n)` | For K[G]-module M (K a finite field, G a finite group) and positive integer n: returns the sequence of K-dimensions of Hᵏ(G, M) for 1 ≤ k ≤ n. Quicker than computing them all individually due to the recursive method. | Same recursive dimension-shifting; computed jointly. |
| `CohomologicalDimension(G, M, n)` | Given permutation group G, K[G]-module M (K a finite field of prime order), and integer n ∈ {1, 2}: returns the dimension of the n-th cohomology group. Invokes Derek Holt's original C cohomology code. May be faster than the cohomology-module-based functions for small examples. | Holt's original cohomology algorithm **[Hol85b]**. |

*Worked examples: H68E2 (H¹ and H² of A₈ comparing old and new functions); H68E3 (Ω⁻(8,3) where the new function succeeds but the old runs out of space).*

---

## 68.5 Cocycles

Before invoking functions in this section, it is necessary to first call `CohomologyGroup(CM, n)` for the appropriate n. Cocycles use right actions (slightly different from many textbooks); the relations satisfied are:

- 0-cocycles z: z(⟨⟩)ᵍ = z(⟨⟩)
- 1-cocycles o: o(⟨gh⟩) = o(⟨g⟩)ʰ + o(⟨h⟩)
- 2-cocycles t: t(⟨gh, k⟩) + t(⟨g, h⟩)ᵏ = t(⟨g, hk⟩) + t(⟨h, k⟩)

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ZeroCocycle(CM, s)` | Given cohomology module CM for G acting on M and an element s of H⁰(G, M) (either as a cohomology group element or as a sequence of integers defining one), returns the corresponding zero-cocycle as a function of the 0-tuple ⟨⟩ whose image is an element of the fixed point submodule of M. | Inverse to `IdentifyZeroCocycle`. |
| `IdentifyZeroCocycle(CM, s)` | Given CM and a zero-cocycle s as a function of the 0-tuple ⟨⟩ with image in the fixed point submodule of M, returns the corresponding element of H⁰(G, M). Inverse to `ZeroCocycle`. | — |
| `OneCocycle(CM, s)` | Given CM and an element s of H¹(G, M) (cohomology group element or sequence of integers), returns a corresponding one-cocycle as a function from G to M (elements of G represented as 1-tuples ⟨g⟩). | Inverse to `IdentifyOneCocycle`. |
| `IdentifyOneCocycle(CM, s)` | Given CM and a one-cocycle s specified as a function from G to M, returns the corresponding element of H¹(G, M). Inverse to `OneCocycle`. | — |
| `IsOneCoboundary(CM, s)` | Given CM and a one-cocycle s (function from G to M), determines whether the cocycle is a 1-coboundary (zero element of H¹(G, M)). If so, also returns a 0-cochain t(⟨⟩) satisfying s(⟨g⟩) = t(⟨⟩) − t(⟨⟩)ᵍ for all g ∈ G. | — |
| `TwoCocycle(CM, s)` | Given CM and an element s of H²(G, M) (cohomology group element or sequence of integers), returns a corresponding two-cocycle as a function from G × G to M (elements represented as 2-tuples ⟨g₁, g₂⟩). | Inverse to `IdentifyTwoCocycle`. |
| `IdentifyTwoCocycle(CM, s)` | Given CM and a two-cocycle s (function from G × G to M), returns the corresponding element of H²(G, M). Inverse to `TwoCocycle`. | — |
| `IsTwoCoboundary(CM, s)` | Given CM and a two-cocycle s (function G × G → M), determines whether it is a 2-coboundary (zero element of H²(G, M)). If so, returns a 1-cochain t(⟨g⟩) satisfying s(⟨g, h⟩) = t(⟨g⟩)ʰ + t(⟨h⟩) − t(⟨gh⟩) for all g, h ∈ G. | — |

*Worked example: H68E4 (cyclic group of order 4 acting on an abelian group with invariants [2,4,4]; computing H⁰, H¹, H², zero-cocycles, one-cocycles, two-cocycles and their identification functions).*

---

## 68.6 The Restriction to a Subgroup

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Restriction(CM, H)` | Given a cohomology module for a group G and a subgroup H of G, returns the restriction of CM to H (a new cohomology module CMH). Restriction maps on H¹ and H² can then be defined by combining `OneCocycle`/`IdentifyOneCocycle` and `TwoCocycle`/`IdentifyTwoCocycle` between CM and CMH. | Induced module restriction. |

*Worked example: H68E5 (GL(3,2) acting on its natural module; computing restriction of H¹ and H² to a Sylow 2-subgroup and to a subgroup of order 21; verifying coboundary condition).*

---

## 68.7 Other Operations on Cohomology Modules

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CorestrictionMapImage(G, C, c, i)` `CorestrictCocycle(G, C, c, i)` | Given an i-cochain c for cohomology module C defined with respect to some subgroup U of G, returns the corestriction of c to Hⁱ(G, …). | Corestriction (transfer) map. |
| `InflationMapImage(M, c)` `LiftCocycle(M, c)` | Given a cochain c : Gⁱ → X and a (transversal) map H → G, returns the inflation (lift) of c to H, i.e., a cochain d : Hⁱ → X defined by d(h) := c(M(h)). Parameters: `Level` (RngIntElt, default false; if given, c is assumed to be in the cohomology group at that level, i.e. i := Level; otherwise Magma guesses the level); `NewCodomain` (Any, default false; if given, values of d are coerced into this structure). | Inflation map in the Hochschild–Serre spectral sequence. |
| `CoboundaryMapImage(M, i, c)` | For cohomology module M, level i, and i-cochain c (as a user program), returns an (i+1)-coboundary obtained from the cohomological coboundary operator. | Coboundary operator δ. |

---

## 68.8 Constructing Extensions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Extension(CM, s)` | Given cohomology module CM for G acting on M and an element s of H²(G, M) (cohomology group element or sequence of integers), returns the corresponding extension E of M by G as a finitely presented group. Generators of E: generators of G (or strong generators if G is a permutation/matrix group) come first; generators of M come last. Also returns the projection E → G and the injection (abelian group isomorphic to M) → E. Only applicable when M is defined over a finite field of prime order, the integers, or as an abelian group via `CohomologyModule(G, Q, T)`. | Group extension via 2-cocycle **[CCH01]**. |
| `SplitExtension(CM)` | Given CM, returns the split extension E of M by G as a finitely presented group (same as `Extension(CM, s)` with s the zero element of H²(G, M)), but faster and without requiring H²(G, M) to be computed first. Also works when G was a finitely presented group. Also returns the projection E → G and the injection (abelian group isomorphic to M) → E. Same module type restrictions as `Extension`. | Split extension (semidirect product) construction. |
| `pMultiplicator(G, p)` | Given permutation group G and a prime p dividing |G|, returns the invariant factors of the p-part of the Schur multiplicator of G. | Holt's original cohomology code **[Hol84]**. |
| `pCover(G, F, p)` | Given permutation group G, finitely presented group F (an epimorphic image of which G is obtained, with generators of F in one-to-one correspondence with those of G and all relations of F satisfied in G), and a prime p: returns a presentation for the p-cover of G constructed as an extension of the p-multiplier by F. The mapping taking the i-th generator of F to the i-th generator of G must be an epimorphism (usually an isomorphism). | Holt's covering group algorithm **[Hol85a]**. |

*Worked examples: H68E6 (A₈ acting on a permutation module over GF(3); constructing a non-split extension of order 3⁸·|A₈| = 60480, verifying normality and absence of complements); H68E7 (A₅ acting on a permutation module over the integers; computing H⁰, H¹, H² with H² ≅ Z/3Z; constructing and examining the extension as an FPGroup; demonstrating `DistinctExtensions` fails over Z).*

---

## 68.9 Constructing Distinct Extensions

Two extensions E₁, E₂ of M by G are considered distinct if there is no group isomorphism from one to the other that maps the subgroup of E₁ corresponding to M to the subgroup of E₂ corresponding to M.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DistinctExtensions(CM)` | Given cohomology module CM for G acting on M, returns a sequence of all distinct extensions of M by G, each in the form returned by `Extension(CM, s)`. Only applicable when M is defined over a finite field of prime order, the integers, or as an abelian group via `CohomologyModule(G, Q, T)`. | Enumerates H²(G, M), quotients by the action of Aut(G) × Aut(M) to classify up to isomorphism fixing M **[CCH01]**. |
| `ExtensionsOfElementaryAbelianGroup(p, d, G)` | Given a prime p, a positive integer d, and a permutation group G, returns a list of finitely presented groups isomorphic to the distinct extensions of an elementary abelian group N of order pᵈ by G. Each extension E is defined on d + r generators (r = number of generators of G); the last d generate N and the quotient by N presents G on its own generators. | Enumeration over all K[G]-module structures on N, then `DistinctExtensions` for each. |
| `ExtensionsOfSolubleGroup(H, G)` | Given permutation groups G and H (H soluble), returns a sequence of finitely presented groups isomorphic to the distinct extensions of H by G. Each extension E is defined on d + r generators; the last d generate a copy of H (as a PC-generating sequence, not the original generators of H) and the quotient by H presents G on its own generators. | Iterates through the composition factors of H and applies cohomological extension at each step. |
| `IsExtensionOf(G)` | Given a permutation group G, finds normal abelian subgroups A < G such that G can be obtained by extending G/A by A. Parameters: `Degree` (RngIntElt, default 0; if given, G/A must have exactly Degree elements); `MaxId` (RngIntElt, default 15; identifier cutoff for the transitive groups database); `DegreeBound` (RngIntElt, default ∞; only considers A with |G/A| < DegreeBound). Returns a sequence of tuples containing: the cohomology module of G/A acting on A; the 2-cocycle element in H²(G/A, A) corresponding to G; the actual 2-cocycle as a user-defined function; a pair ⟨a, b⟩ giving the degree a of the transitive group G/A and the database identifier b (or a hash if b > MaxId); the abelian invariants of A; and a set of all pairs ⟨a, b⟩ obtainable through this extension process. The list considered contains only subgroups that are maximal under the given restrictions. | Database-assisted extension recognition. |
| `IsExtensionOf(L)` | For a list L of groups, applies `IsExtensionOf` to all groups in L. Returns: a minimal sequence of tuples (as in `IsExtensionOf(G)`) such that all groups in L can be generated using the cohomology modules in the sequence; and a set of pairs ⟨a, b⟩ describing all transitive groups obtainable through the processes. Parameters: `Degree`, `MaxId`, `DegreeBound` as above. | Batch application of `IsExtensionOf`. |

*Worked examples: H68E8 (Z₂×Z₂ acting trivially on GF(2); H² has dimension 3 giving 8 equivalence classes but only 4 distinct up to isomorphism: Z₂³, Z₄×Z₂, D₄, Q₈); H68E9 (A₄ acting on Z₂×Z₂; 4 distinct extensions, split/non-split identified via `Complements`); H68E10 (D₄ extending D₄; 20 distinct extensions); H68E11 (cyclic group of order 4 acting on Z/2Z × Z/4Z × Z/4Z; 3 distinct extensions verified non-isomorphic).*

---

## 68.10 Finite Group Cohomology

This section provides functions for computing the first cohomology group H¹(Γ, A) of a finite group with coefficients in a finite (not necessarily abelian) group A. Based on **[Hal05]**.

Let Γ be a group. A group A on which Γ acts by group automorphisms from the right is called a Γ-group. Given a Γ-group A:

- H⁰(Γ, A) := {a ∈ A | aᵟ = a for all σ ∈ Γ}
- A 1-cocycle α : Γ → A satisfies αₛₜ = (αₛ)ᵗ αₜ for all σ, τ ∈ Γ
- Two cocycles α, β are cohomologous (with respect to a) if there exists a ∈ A with βₛ = a⁻ᵟ · αₛ · a for all σ ∈ Γ
- H¹(Γ, A) is the set of equivalence classes of 1-cocycles (a pointed set)
- A twisted form Aᵝ of A by cocycle β has the same underlying group but with action a ∗ σ := aᵟ αₛ

### 68.10.1 Creation of Gamma-groups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `GammaGroup(Gamma, A, action)` | Given a group A and a group Γ acting on it via the map action (a homomorphism from Γ to the automorphism group of A), returns a Γ-group object of type GGrp. | — |
| `InducedGammaGroup(A, B)` | Given a Γ-group A and a normal subgroup B of A normalised by the action of Γ, returns the induced Γ-group A/B. | Natural quotient action. |
| `IsNormalised(B, action)` | Returns true if group B is normalised by the action. | — |
| `IsInduced(AmodB)` | Returns true iff the Γ-group AmodB was created as an induced Γ-group. If it is, also returns the Γ-groups A, B, the projection map, and the representative map. | — |

*Worked example: H68E12 (Γ = A₄ acting by conjugation on A = S₄; computing the induced Γ-group S₄/A₄ ≅ Z₂).*

### 68.10.2 Accessing Information

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Group(A)` | Returns the group A as a Grp object. | — |
| `GammaAction(A)` | Returns the action of Γ on A as a map. | — |
| `ActingGroup(A)` | Returns the group Γ acting on A. | — |

### 68.10.3 One Cocycles

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `OneCocycle(A, imgs)` `OneCocycle(A, alpha)` | If the map α : Γ → A or the sequence imgs of images of generators Γ.1, ..., Γ.n defines a 1-cocycle, returns the 1-cocycle. Parameter: `Check` (BoolElt, default true; if false, disables the cocycle-condition check). | Verifies cocycle relation αₛₜ = (αₛ)ᵗ αₜ **[Hal05]**. |
| `TrivialOneCocycle(A)` | Returns the trivial 1-cocycle (constant map σ ↦ 1). | — |
| `IsOneCocycle(A, imgs)` `IsOneCocycle(A, alpha)` | Returns true if the map or sequence defines a 1-cocycle; returns the cocycle as a second value if true. Does not abort with an error if the map does not define a cocycle (unlike `OneCocycle`). | — |
| `AreCohomologous(alpha, beta)` | Returns true iff the 1-cocycles α and β are cohomologous. If true, returns the intertwining element as the second return value. | — |
| `CohomologyClass(alpha)` | Returns the cohomology class of the 1-cocycle α (the set of all cocycles cohomologous to α). | — |
| `InducedOneCocycle(AmodB, alpha)` `InducedOneCocycle(A, B, alpha)` | Given a 1-cocycle on A, returns the induced 1-cocycle on AmodB. The second form generates the induced Γ-group A/B first. | Natural projection of cocycles. |
| `ExtendedOneCocycle(alpha)` | Given a 1-cocycle on an induced Γ-group A/B, returns the set of all pairwise non-cohomologous 1-cocycles on A that induce to α. Parameter: `OnlyOne` (BoolElt, default false; if true, the set contains at most one 1-cocycle). Returns an empty set if α is not extendible. | Extension via the Hochschild–Serre sequence **[Hal05]**. |
| `ExtendedCohomologyClass(alpha)` | Given a 1-cocycle on an induced Γ-group A/B, returns the set of all pairwise non-cohomologous 1-cocycles on A that induce to a cocycle in the cohomology class of α. Returns an empty set if no such cocycles on A exist. | — |
| `GammaGroup(alpha)` | Returns the Γ-group on which α is defined. | — |
| `CocycleMap(alpha)` | Returns the Map object corresponding to α. | — |

### 68.10.4 Group Cohomology

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Cohomology(A, n)` | Given a finite Γ-group A and integer n (currently restricted to n = 1), returns the n-th cohomology group Hⁿ(Γ, A). Since A is not assumed abelian, only n = 0, 1 can be defined; only n = 1 is currently implemented. (H⁰ equals the subgroup of A centralised by Γ, constructible via group-theoretic means.) | First cohomology of non-abelian group **[Hal05]**. |
| `OneCohomology(A)` | Returns H¹(Γ, A) as a set of representatives of all cohomology classes. If A is abelian, uses existing code by Derek Holt (see Chapter 68); otherwise uses **[Hal05]**. | **[Hol85b]** for abelian A; **[Hal05]** for non-abelian A. |
| `TwistedGroup(A, alpha)` | Given the Γ-group A and a 1-cocycle α on it, returns the twisted group Aα (same underlying group A but with Γ-action a ∗ σ := aᵟ αₛ). | Twisting construction **[Hal05]**. |

*Worked example: H68E13 (D₁₆ in S₈ with Γ = Normaliser; computing cohomology class of the trivial cocycle on A/B where B = Centre(A); extending a cocycle from A/B to A; constructing the twisted group Aβ).*

---

## 68.11 Bibliography

| Key | Reference |
|-----|-----------|
| **[CCH01]** | J.J. Cannon, B. Cox, and D.F. Holt. Computing the subgroup lattice of a permutation group. *J. Symbolic Comp.*, 31:149–161, 2001. |
| **[Hal05]** | Sergei Haller. *Computing Galois Cohomology and Forms of Linear Algebraic Groups.* PhD thesis, Technical University of Eindhoven, 2005. |
| **[Hol84]** | D.F. Holt. The calculation of the Schur multiplier of a permutation group. In *Computational group theory (Durham, 1982)*, pages 307–319. Academic Press, London, 1984. |
| **[Hol85a]** | D.F. Holt. A computer program for the calculation of a covering group of a finite group. *J. Pure Appl. Algebra*, 35(3):287–295, 1985. |
| **[Hol85b]** | D.F. Holt. The mechanical computation of first and second cohomology groups. *J. Symbolic Comp.*, 1(4):351–361, 1985. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Nullspace computation for H¹ and H² **[CCH01]** | `CohomologyGroup`, `CohomologicalDimension(CM, n)` |
| PC-presentation consistency equations for H² (soluble groups) **[Hol85b]** | `CohomologyGroup`, `CohomologicalDimension(CM, 2)` |
| p-subgroup reduction for dim H² over prime field **[Hol85b]** | `CohomologicalDimension(CM, 2)`, `CohomologicalDimension(G, M, n)` |
| Recursive dimension-shifting via projective covers | `CohomologicalDimension(M, n)`, `CohomologicalDimensions(M, n)` |
| Holt's original C cohomology code **[Hol85b]** | `CohomologicalDimension(G, M, n)` |
| 2-cocycle / group extension construction **[CCH01]** | `Extension`, `SplitExtension`, `DistinctExtensions` |
| Schur multiplicator (p-part) **[Hol84]** | `pMultiplicator` |
| p-cover / covering group **[Hol85a]** | `pCover` |
| Restriction / corestriction / inflation maps | `Restriction`, `CorestrictionMapImage`, `CorestrictCocycle`, `InflationMapImage`, `LiftCocycle` |
| Coboundary operator | `CoboundaryMapImage`, `IsOneCoboundary`, `IsTwoCoboundary` |
| Non-abelian H¹ for Γ-groups **[Hal05]** | `Cohomology`, `OneCohomology`, `OneCocycle(A,…)`, `IsOneCocycle`, `AreCohomologous`, `CohomologyClass`, `TwistedGroup` |
| Cocycle induction / extension (Hochschild–Serre) **[Hal05]** | `InducedOneCocycle`, `ExtendedOneCocycle`, `ExtendedCohomologyClass` |
| Gamma-group creation and quotients | `GammaGroup`, `InducedGammaGroup`, `IsNormalised`, `IsInduced` |
| Extension-of-group recognition via database | `IsExtensionOf` |
