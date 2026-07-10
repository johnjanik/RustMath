# Chapter 95 — Coxeter Systems

**Handbook part:** XIII — Lie Theory
**Handbook pages:** 2805–2825 (PDF pages 2936–2959)

---

## Scope and overview

This chapter covers the basic combinatorial descriptions of Coxeter systems used throughout
the Lie Theory part of Magma. A **Coxeter system** is a group G with a finite generating set
S = {s₁, …, sₙ}, defined by relations sᵢ² = 1 and the braid relations sᵢsⱼsᵢ⋯ = sⱼsᵢsⱼ⋯
(each side of length mᵢⱼ ≥ 2). The value mᵢⱼ = ∞ (represented in Magma by 0) indicates
that no braid relation is imposed. The integer n is the **rank** of the Coxeter system.
A Coxeter system is **reducible** if the generators split into two non-communicating subsets;
in that case the Coxeter group is an internal direct product. Two Coxeter groups are
**Coxeter isomorphic** if they are isomorphic via a map taking Coxeter generators to
Coxeter generators (i.e. equal up to renumbering of generators).

Four equivalent data structures are supported for encoding a Coxeter system: Coxeter
matrices, Coxeter graphs, Cartan matrices, and Dynkin digraphs. Conversion functions
among all four forms are provided. The chapter also handles the classification of finite and
affine Coxeter groups via Cartan names (strings such as `"A5"`, `"B~3"`, `"I2(7)"`), hyperbolic
Coxeter groups, and shortcuts for constructing related objects (root systems, root data,
Coxeter groups, Lie algebras, groups of Lie type) from any of these descriptions.

See **[Bou68]** for the theoretical background on Coxeter groups, and Chapters 96–99 and
100, 103 for the related Magma structures.

---

## 95.1 Introduction

The introductory section defines the key concepts and sets notation. No intrinsics are
listed; the mathematical definitions are given as prose (summarised in the Scope section
above).

---

## 95.2 Coxeter Matrices

A **Coxeter matrix** is the symmetric n×n integer matrix M = (mᵢⱼ) with mᵢᵢ = 1,
mᵢⱼ = mⱼᵢ ∈ {2, 3, 4, …} ∪ {0} for i ≠ j (where 0 represents ∞). Each entry encodes
the order of the product sᵢsⱼ in the Coxeter group.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsCoxeterMatrix(M)` | Returns `true` iff the matrix `M` is the Coxeter matrix of some Coxeter group. | Checks symmetry, diagonal = 1, and off-diagonal entries in {0, 2, 3, 4, …}. |
| `CoxeterMatrix(G)` / `CoxeterMatrix(C)` / `CoxeterMatrix(D)` | The Coxeter matrix corresponding to a Coxeter graph `G`, Cartan matrix `C`, or Dynkin digraph `D`. | Conversion via the defining bijection between data structures. |
| `IsCoxeterIsomorphic(M1, M2)` | Returns `true` iff Coxeter matrices `M1` and `M2` give rise to Coxeter-isomorphic groups. Also returns a sequence encoding the permutation of the basis taking `M1` to `M2`. | Checks equality of matrices up to simultaneous row/column permutation. |
| `CoxeterGroupOrder(M)` / `CoxeterGroupFactoredOrder(M)` | The (factored) order of the Coxeter group with Coxeter matrix `M`. | — |
| `IsCoxeterIrreducible(M)` | Returns `true` iff `M` is the Coxeter matrix of an irreducible Coxeter system. If reducible, also returns a nontrivial subset I ⊆ {1,…,n} with mᵢⱼ = 2 for i ∈ I, j ∉ I. | Connectivity check on the associated Coxeter graph. |
| `IsSimplyLaced(M)` | Returns `true` iff all entries of the Coxeter matrix `M` are in {1, 2, 3} (no edge labels > 3). | — |

*Worked examples: H95E1 (construction and validation of a 3×3 Coxeter matrix), H95E2 (`IsCoxeterIsomorphic` and `CoxeterGroupOrder` for two 3×3 matrices), H95E3 (`IsCoxeterIrreducible` for reducible and irreducible cases).*

---

## 95.3 Coxeter Graphs

A **Coxeter graph** is an undirected labelled graph with vertices {1, …, n}; an edge between
i and j exists whenever mᵢⱼ > 2, labelled by mᵢⱼ (with the label omitted when mᵢⱼ = 3).
Infinity (mᵢⱼ = ∞) is represented by 0. A Coxeter system is irreducible iff its Coxeter
graph is connected. Two Coxeter graphs give Coxeter-isomorphic groups iff they are
isomorphic as labelled graphs. See Chapter 149 for graph operations.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsCoxeterGraph(G)` | Returns `true` iff the graph `G` is the Coxeter graph of some Coxeter group. | Checks that the graph is standard (vertex set {1,…,n}), undirected, and all edge labels are in {0, 3, 4, 5, …}. |
| `CoxeterGraph(M)` / `CoxeterGraph(C)` / `CoxeterGraph(D)` | The Coxeter graph corresponding to a Coxeter matrix `M`, Cartan matrix `C`, or Dynkin digraph `D`. | Conversion via the defining bijection. |
| `CoxeterGroupOrder(G)` / `CoxeterGroupFactoredOrder(G)` | The (factored) order of the Coxeter group with Coxeter graph `G`. | — |
| `IsSimplyLaced(G)` | Returns `true` iff the Coxeter graph `G` is simply laced (unlabelled, i.e. all edges have mᵢⱼ = 3). | — |

*Worked examples: H95E4 (path graph with labels, `IsCoxeterGraph`, infinite order; `CoxeterGraph` from a matrix, edge label retrieval), H95E5 (`IsSimplyLaced` before and after adding a label).*

---

## 95.4 Cartan Matrices

A **Cartan matrix** is a real-valued matrix C = (cᵢⱼ) satisfying: (1) cᵢᵢ = 2; (2) cᵢⱼ ≤ 0
for i ≠ j; (3) cᵢⱼ = 0 iff cⱼᵢ = 0; and (4) if nᵢⱼ := cᵢⱼcⱼᵢ < 4 then nᵢⱼ = 4cos²(π/mᵢⱼ)
for some integer mᵢⱼ ≥ 2. In Magma, Cartan matrices may be defined over Z, Q, number
fields (Chapter 34), or cyclotomic fields (Chapter 36); the real field is excluded as it lacks
infinite precision. A Cartan matrix is **crystallographic** if all entries are integers.

The Cartan matrix is not unique for a given Coxeter system; it specifies a faithful representation
of the Coxeter group as a real reflection group via the reflections sᵢ : v ↦ v − (v, αᵢ*)αᵢ.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsCartanMatrix(C)` | Returns `true` iff `C` is a Cartan matrix. Parameter `RealInjection` (default `false`): for number-field or cyclotomic-field base, supply an injection into R (resp. C) so that conditions (2) and (4) can be checked; without it those conditions are skipped. | Checks the four defining conditions; conditions (2) and (4) require a real injection for non-real base fields. |
| `CartanMatrix(M)` / `CartanMatrix(G)` | A Cartan matrix for the Coxeter matrix `M` or Coxeter graph `G`. Default: cᵢⱼ = −4cos²(π/mᵢⱼ), cⱼᵢ = −1 for mᵢⱼ ≠ 2 and i < j (crystallographic when a crystallographic form exists). Parameter `Symmetric` (default `false`): if true, returns the symmetric form cᵢⱼ = cⱼᵢ = −2cos(π/mᵢⱼ). Parameter `BaseField` (default `"NumberField"`): `"NumberField"`, `"Cyclotomic"` / `"SparseCyclotomic"`, or `"DenseCyclotomic"` (overridden to Z when crystallographic). | Conversion from Coxeter data; default form is chosen to be crystallographic when possible. |
| `CartanMatrix(D)` | The crystallographic Cartan matrix corresponding to the Dynkin digraph `D`. | Reads edge labels as −cᵢⱼ entries of the Cartan matrix. |
| `IsCoxeterIsomorphic(C1, C2)` | Returns `true` iff Cartan matrices `C1` and `C2` give Coxeter-isomorphic groups (i.e. their Coxeter matrices are equal up to basis permutation). Also returns the permutation sequence. | Reduces to comparing Coxeter matrices up to permutation. |
| `IsCartanEquivalent(C1, C2)` | Returns `true` iff crystallographic Cartan matrices `C1` and `C2` are Cartan equivalent (equal up to basis permutation). Also returns the permutation sequence. Stronger than Coxeter isomorphism (e.g. B₅ and C₅ are Coxeter isomorphic but not Cartan equivalent). | Checks equality of matrices up to simultaneous row/column permutation. |
| `NumberOfPositiveRoots(C)` / `NumPosRoots(C)` | The number of positive roots of the root system with Cartan matrix `C`. See §96.1.3 for the definition. | — |
| `CoxeterGroupOrder(C)` / `CoxeterGroupFactoredOrder(C)` | The (factored) order of the Coxeter group with Cartan matrix `C`. | — |
| `FundamentalGroup(C)` | The fundamental group Zⁿ/Γ of the crystallographic Cartan matrix `C`, where Γ is the lattice generated by the rows of `C`. Also returns the natural map Zⁿ → Zⁿ/Γ. | Abelian-group quotient by the row lattice. |
| `IsCoxeterIrreducible(C)` | Returns `true` iff `C` is the Cartan matrix of an irreducible Coxeter system. If reducible, also returns a nontrivial subset I with cᵢⱼ = 0 for i ∈ I, j ∉ I. | Connectivity check on the associated Coxeter graph. |
| `IsCrystallographic(C)` | Returns `true` iff the Cartan matrix `C` is crystallographic (all entries integers). | Checks integrality of matrix entries. |
| `IsSimplyLaced(C)` | Returns `true` iff `C` is simply laced (all entries of its Coxeter matrix are in {1, 2, 3}). | — |

*Worked examples: H95E6 (`IsCartanMatrix`, `CoxeterMatrix`, `CartanMatrix` with `Symmetric` and `BaseField := "Cyclotomic"` options), H95E7 (`IsCoxeterIsomorphic` vs `IsCartanEquivalent` on B-type and C-type matrices), H95E8 (`FundamentalGroup` of path graph of rank 4, yielding Z/5).*

---

## 95.5 Dynkin Digraphs

A **Dynkin digraph** is a directed labelled graph describing a crystallographic Cartan matrix
C = (cᵢⱼ). It has vertices {1, …, n}; there is an edge from i to j labelled −cᵢⱼ whenever
cᵢⱼ < 0 (label omitted when cᵢⱼ = −1). A Dynkin digraph is bidirectional (if there is an
edge i → j there is also j → i, though with possibly different labels), so strong and weak
connectivity coincide. The system is irreducible iff the Dynkin digraph is connected.

The chapter reserves the term "Dynkin diagram" for printed displays (see §95.6); the
Dynkin digraph is the graph-theoretic object. Two Dynkin digraphs give Cartan-equivalent
matrices iff they are isomorphic as labelled digraphs. Note that there is no function to
compute a Dynkin digraph from a non-crystallographic Coxeter matrix or Coxeter graph.
See Chapter 149 for digraph operations.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsDynkinDigraph(D)` | Returns `true` iff the digraph `D` is the Dynkin digraph of some crystallographic Cartan matrix. | Checks that the graph is standard, and that the edge labels are consistent with a valid Cartan matrix. |
| `DynkinDigraph(C)` | The Dynkin digraph of the crystallographic Cartan matrix `C`. | Reads off the negative off-diagonal entries of `C` as directed edge labels. |
| `CoxeterGroupOrder(D)` / `CoxeterGroupFactoredOrder(D)` | The (factored) order of the Coxeter group with Dynkin digraph `D`. | — |
| `FundamentalGroup(D)` | The fundamental group Zⁿ/Γ of the Dynkin digraph `D`, where Γ is the row lattice of the corresponding Cartan matrix. Also returns the natural map Zⁿ → Zⁿ/Γ. | Abelian-group quotient by the row lattice. |
| `IsSimplyLaced(D)` | Returns `true` iff the Dynkin digraph `D` is simply laced (unlabelled, i.e. all edges have label 1). | — |

*Worked example: H95E10 (construction of a rank-4 Dynkin digraph with labels, `IsDynkinDigraph`, `CartanMatrix`, `FundamentalGroup` yielding Z/2 + Z/8).*

---

## 95.6 Finite and Affine Coxeter Groups

The classification of finite and affine Coxeter groups is due to Cartan **[Car52]** and Coxeter
**[Cox34]**. A Coxeter group is **finite** (spherical) iff it acts discretely and properly as a
reflection group of the sphere; it is **affine** iff it is infinite and acts discretely and properly
as an affine reflection group. A Coxeter group is finite iff all irreducible components are
finite; affine iff all components are finite or affine with at least one affine component.

The irreducible finite crystallographic types are: Aₙ (n ≥ 1), Bₙ (n ≥ 2), Cₙ (n ≥ 3),
Dₙ (n ≥ 4), E₆, E₇, E₈, F₄, G₂. The irreducible finite non-crystallographic types are:
H₃, H₄, I₂(m) for m = 5 and m > 6. Note the redundancies A₁ = B₁ = C₁, A₂ = I₂(3),
B₂ = C₂ = I₂(4), D₂ = A₁+A₁, D₃ = A₃, G₂ = I₂(6); for n ≥ 3, types Bₙ and Cₙ have
the same Coxeter matrix but inequivalent crystallographic Cartan matrices (for n > 2).
All irreducible affine groups are crystallographic; they are denoted Ã₁, Ãₙ, B̃ₙ, C̃ₙ, D̃ₙ,
Ẽ₆, Ẽ₇, Ẽ₈, F̃₄, G̃₂ (written `A~1`, `A~n`, etc. in Magma strings).

Cartan names are strings such as `"A5"`, `"A~2B2"`, `"I2(7)"`, `"BC3"`. The parser is
flexible: letters and numbers must alternate, except that I-type requires brackets.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsCoxeterFinite(M)` / `IsCoxeterFinite(G)` / `IsCoxeterFinite(C)` / `IsCoxeterFinite(D)` / `IsCoxeterFinite(N)` | Returns `true` iff the corresponding Coxeter group is finite. Input may be a Coxeter matrix `M`, Coxeter graph `G`, Cartan matrix `C`, Dynkin digraph `D`, or Cartan name string `N`. | Classification **[Car52, Cox34]**: tests whether all irreducible components match a finite type. |
| `IsCoxeterAffine(M)` / `IsCoxeterAffine(G)` / `IsCoxeterAffine(C)` / `IsCoxeterAffine(D)` / `IsCoxeterAffine(N)` | Returns `true` iff the corresponding Coxeter group is affine. Input as above. | Classification **[Car52, Cox34]**: tests whether all components are finite or affine and at least one is affine. |
| `CoxeterMatrix(N)` | The Coxeter matrix with Cartan name given by string `N`. | Lookup from the classification table. |
| `CoxeterGraph(N)` | The Coxeter graph with Cartan name given by string `N`. | Lookup from the classification table. |
| `CartanMatrix(N)` | The Cartan matrix with Cartan name given by string `N`. Default: crystallographic for crystallographic types; otherwise cᵢⱼ = −4cos²(π/mᵢⱼ), cⱼᵢ = −1 for i < j. Parameters `Symmetric` and `BaseField` as in §95.4. | Lookup from the classification table; field determined by `BaseField` unless crystallographic. |
| `DynkinDigraph(N)` | The Dynkin digraph with Cartan name `N`. The name must be crystallographic (not H₃, H₄, or I₂(m)). | Lookup from the classification table. |
| `IrreducibleCoxeterMatrix(X, n)` | The irreducible Coxeter matrix with Cartan name `Xn` (or `I2(n)` if `X = "I"`). | Lookup from the classification table; useful in loops over types. |
| `IrreducibleCoxeterGraph(X, n)` | The irreducible Coxeter graph with Cartan name `Xn` (or `I2(n)` if `X = "I"`). | Lookup from the classification table. |
| `IrreducibleCartanMatrix(X, n)` | The irreducible Cartan matrix with Cartan name `Xn`. Parameters `Symmetric` and `BaseField` as in §95.4. | Lookup from the classification table. |
| `IrreducibleDynkinDigraph(X, n)` | The irreducible Dynkin digraph with Cartan name `Xn`. The name must be crystallographic. | Lookup from the classification table. |
| `IsCoxeterIsomorphic(N1, N2)` | Returns `true` iff Cartan name strings `N1` and `N2` correspond to Coxeter-isomorphic groups. | Reduces to comparing the corresponding Coxeter matrices up to permutation. |
| `IsCartanEquivalent(N1, N2)` | Returns `true` iff Cartan name strings `N1` and `N2` correspond to Cartan-equivalent Cartan matrices. The names must be crystallographic. | Reduces to comparing the corresponding Cartan matrices up to permutation. |
| `IsSimplyLaced(N)` | Returns `true` iff the Coxeter matrix with Cartan name `N` is simply laced. | — |
| `CoxeterGroupOrder(N)` / `CoxeterGroupFactoredOrder(N)` | The (factored) order of the Coxeter group with Cartan name `N`. | — |
| `NumberOfPositiveRoots(N)` / `NumPosRoots(N)` | The number of positive roots of the Coxeter group with Cartan name `N`. See §96.1.3. | — |
| `FundamentalGroup(N)` | The fundamental group Zⁿ/Γ of the crystallographic Cartan matrix with Cartan name `N`. Also returns the natural map Zⁿ → Zⁿ/Γ. | Abelian-group quotient by the row lattice. |
| `CartanName(M)` / `CartanName(G)` / `CartanName(C)` / `CartanName(D)` | The Cartan name (string) of a Coxeter matrix `M`, Coxeter graph `G`, Cartan matrix `C`, or Dynkin digraph `D`. Raises an error if the Coxeter group is neither finite nor affine. | Reverse lookup in the classification table. |
| `DynkinDiagram(M)` / `DynkinDiagram(G)` / `DynkinDiagram(C)` / `DynkinDiagram(D)` / `DynkinDiagram(N)` | Prints the Dynkin diagram of the given Coxeter data. Raises an error if the group is neither affine nor crystallographic. | Formatted ASCII display from the classification table. |
| `CoxeterDiagram(M)` / `CoxeterDiagram(G)` / `CoxeterDiagram(C)` / `CoxeterDiagram(D)` / `CoxeterDiagram(N)` | Prints the Coxeter diagram of the given Coxeter data. Raises an error if the group is not affine or not crystallographic. | Formatted ASCII display from the classification table. |

*Worked examples: H95E11 (`IsCoxeterAffine` / `IsCoxeterFinite` on name strings), H95E12 (`CoxeterMatrix`, `CoxeterGraph`, `CartanMatrix` with `Symmetric`, `DynkinDigraph` from name strings; flexible name parser), H95E13 (`IrreducibleCoxeterGraph` in a loop over ranks; join of Coxeter graphs for rank-4 types), H95E14 (`IsCoxeterIsomorphic` and `IsCartanEquivalent` for A1A1/D2 and B5/C5), H95E15 (`CoxeterGroupOrder`, `CoxeterGroupFactoredOrder`, `NumPosRoots`, `FundamentalGroup` for F₄), H95E16 (`CartanName` from symmetric matrices including detection of affine type Ã₂ and error on non-classified type), H95E17 (`DynkinDiagram` for a compound name "A~5 D4 BC3"), H95E18 (`CoxeterDiagram` for the same compound name).*

---

## 95.7 Hyperbolic Groups

A Coxeter group is **hyperbolic** if it is infinite, non-affine, and has a representation as a
discrete, properly acting hyperbolic reflection group whose Tits cone consists entirely of
vectors with negative norm (see **[Bou68]**). It is **compact hyperbolic** if the fundamental
region is compact. Every infinite non-affine Coxeter group of rank 3 is hyperbolic. There
are exactly 72 hyperbolic Coxeter groups of rank > 3; they are numbered 1 to 72 (the
numbering is essentially arbitrary).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsCoxeterHyperbolic(M)` / `IsCoxeterCompactHyperbolic(M)` | Returns `true` iff the matrix `M` is the Coxeter matrix of a (compact) hyperbolic Coxeter group. | Classification: rank-3 groups are automatically hyperbolic if infinite and non-affine; rank > 3 is checked against the table of 72 groups. |
| `IsCoxeterHyperbolic(G)` / `IsCoxeterCompactHyperbolic(G)` | Returns `true` iff the graph `G` is the Coxeter graph of a (compact) hyperbolic Coxeter group. | As above but for graph input. |
| `HyperbolicCoxeterMatrix(i)` | The Coxeter matrix of the i-th hyperbolic Coxeter group of rank > 3 (i = 1, …, 72). | Table lookup. |
| `HyperbolicCoxeterGraph(i)` | The Coxeter graph of the i-th hyperbolic Coxeter group of rank > 3 (i = 1, …, 72). | Table lookup. |

*Worked example: H95E19 (iterating over all 72 hyperbolic groups and printing those that are compact hyperbolic; compact ones are numbered 1–14).*

---

## 95.8 Related Structures

Functions for constructing other Magma objects from any of the four Coxeter data types
(Coxeter matrix, Coxeter graph, Cartan matrix, Dynkin digraph) or a Cartan name string.
Detailed documentation is in the referenced chapters.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RootSystem(M)` / `RootSystem(G)` / `RootSystem(C)` / `RootSystem(D)` / `RootSystem(N)` | The finite root system from a Coxeter matrix `M`, Coxeter graph `G`, Cartan matrix `C`, Dynkin digraph `D`, or Cartan name `N`. Errors if the Coxeter group is infinite. See Chapter 96. | — |
| `RootDatum(C)` / `RootDatum(M)` / `RootDatum(G)` / `RootDatum(D)` / `RootDatum(N)` | The finite root datum from a crystallographic Cartan matrix `C`, Coxeter matrix `M`, Coxeter graph `G`, Dynkin digraph `D`, or Cartan name `N`. Errors if the Coxeter group is infinite. See Chapter 97. | — |
| `CoxeterGroup(GrpFPCox, M)` / `CoxeterGroup(GrpFPCox, G)` / `CoxeterGroup(GrpFPCox, C)` / `CoxeterGroup(GrpFPCox, D)` / `CoxeterGroup(GrpFPCox, N)` | The Coxeter group as a finitely presented group (`GrpFPCox`) from any of the four data types or a Cartan name. See Chapter 98. | — |
| `CoxeterGroup(GrpPermCox, M)` / `CoxeterGroup(GrpPermCox, G)` / `CoxeterGroup(GrpPermCox, C)` / `CoxeterGroup(GrpPermCox, D)` / `CoxeterGroup(GrpPermCox, N)` | The Coxeter group as a permutation group (`GrpPermCox`) from any of the four data types or a Cartan name. Errors if the Coxeter group is infinite. See Chapter 98. | — |
| `CoxeterGroup(M)` / `CoxeterGroup(G)` / `CoxeterGroup(C)` / `CoxeterGroup(D)` / `CoxeterGroup(N)` | The Coxeter group from any data type or Cartan name. Returns a permutation group if finite, a finitely presented group otherwise. See Chapter 98. | — |
| `ReflectionGroup(M)` / `ReflectionGroup(G)` / `ReflectionGroup(C)` / `ReflectionGroup(D)` / `ReflectionGroup(N)` | The reflection group from any data type or Cartan name. See Chapter 99. | — |
| `LieAlgebra(C, k)` / `LieAlgebra(D, k)` / `LieAlgebra(N, k)` | The Lie algebra over ring `k` of a crystallographic Cartan matrix `C`, Dynkin digraph `D`, or Cartan name `N`. Errors if the Coxeter group is infinite. See Chapter 100. | — |
| `MatrixLieAlgebra(C, k)` / `MatrixLieAlgebra(D, k)` / `MatrixLieAlgebra(N, k)` | The matrix Lie algebra over ring `k` of a crystallographic Cartan matrix `C`, Dynkin digraph `D`, or Cartan name `N`. Errors if the Coxeter group is infinite. See Chapter 100. | — |
| `GroupOfLieType(C, k)` / `GroupOfLieType(D, k)` / `GroupOfLieType(N, k)` | The group of Lie type over ring `k` of a crystallographic Cartan matrix `C`, Dynkin digraph `D`, or Cartan name `N`. Errors if the Coxeter group is infinite. See Chapter 103. | — |

---

## 95.9 Bibliography

| Key | Reference |
|-----|-----------|
| **[Bou68]** | N. Bourbaki. *Éléments de mathématique. Fasc. XXXIV. Groupes et algèbres de Lie. Chapitre IV: Groupes de Coxeter et systèmes de Tits. Chapitre V: Groupes engendrés par des réflexions. Chapitre VI: Systèmes de racines.* Hermann, Paris, 1968. |
| **[Car52]** | Elie Cartan. *Œuvres complètes. Partie I. Groupes de Lie.* Gauthier-Villars, Paris, 1952. |
| **[Cox34]** | H. S. M. Coxeter. Discrete groups generated by reflections. *Ann. of Math.*, 35:588–621, 1934. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Coxeter matrix ↔ graph ↔ Cartan matrix ↔ Dynkin digraph conversions | `CoxeterMatrix`, `CoxeterGraph`, `CartanMatrix`, `DynkinDigraph` (all overloads) |
| Classification of finite and affine Coxeter groups **[Car52, Cox34]** | `IsCoxeterFinite`, `IsCoxeterAffine`, `CartanName`, `IsCoxeterIsomorphic`, `IsCartanEquivalent`, `IsCoxeterIrreducible`, `CoxeterGroupOrder`, `CoxeterGroupFactoredOrder`, `NumberOfPositiveRoots`, `FundamentalGroup`, `IsSimplyLaced`, `IsCrystallographic` (all overloads) |
| Cartan-name string lookup and parsing | `CoxeterMatrix(N)`, `CoxeterGraph(N)`, `CartanMatrix(N)`, `DynkinDigraph(N)`, `IrreducibleCoxeterMatrix`, `IrreducibleCoxeterGraph`, `IrreducibleCartanMatrix`, `IrreducibleDynkinDigraph` |
| ASCII diagram printing | `DynkinDiagram`, `CoxeterDiagram` (all overloads) |
| Hyperbolic reflection group classification **[Bou68]** | `IsCoxeterHyperbolic`, `IsCoxeterCompactHyperbolic`, `HyperbolicCoxeterMatrix`, `HyperbolicCoxeterGraph` |
| Construction of root systems (Ch. 96) | `RootSystem` (all overloads) |
| Construction of root data (Ch. 97) | `RootDatum` (all overloads) |
| Construction of Coxeter groups (Ch. 98) | `CoxeterGroup(GrpFPCox, …)`, `CoxeterGroup(GrpPermCox, …)`, `CoxeterGroup` (all overloads) |
| Construction of reflection groups (Ch. 99) | `ReflectionGroup` (all overloads) |
| Construction of Lie algebras (Ch. 100) | `LieAlgebra`, `MatrixLieAlgebra` |
| Construction of groups of Lie type (Ch. 103) | `GroupOfLieType` |
