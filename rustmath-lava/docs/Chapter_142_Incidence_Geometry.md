# Chapter 142 — Incidence Geometry

**Handbook part:** XIX — Geometry
**Handbook pages:** 4751–4770 (PDF pages 4882–4905)

---

## Scope and overview

This chapter presents the functions for constructing and computing with **incidence
geometries** and **coset geometries**. The standard references are the *Handbook of Incidence
Geometry* **[Bue95]** (ed. F. Buekenhout) and A. Pasini's *Diagram Geometries* **[Pas94]**.

An *incidence geometry* in Magma is a four-tuple Γ(X, ∼, t, I), where X and I are finite sets,
t : X → I is a surjective *type function*, and ∼ is a reflexive, symmetric *incidence relation*
on X satisfying that any two incident elements of the same type are equal (∀ x, y ∈ X,
x ∼ y and t(x) = t(y) ⇒ x = y). X is the set of *elements*, I the set of *types*, and the
cardinality of I is the *rank* of Γ. Note that this is **not** a geometry in the sense of
Buekenhout, since flags (cliques of the incidence graph) are not required to lie in a chamber
(a clique containing one element of each type); when every flag does lie in a chamber, the
incidence geometry is a Buekenhout geometry. The category name is `IncGeom`.

A *coset geometry* is built from a group and some of its subgroups by an algorithm of Jacques
Tits **[Tit62]**. Given a group G and subgroups {Gᵢ : i ∈ I}, set X = {Gᵢg : g ∈ G, i ∈ I},
type function t : Gᵢg ↦ i, and incidence Gᵢg ∼ Gⱼh iff Gᵢg ∩ Gⱼh ≠ ∅. The {Gᵢ} are the
*maximal parabolic subgroups*, ∩ᵢ Gᵢ is the *Borel subgroup*, and the
{∩_{j∈I\{i}} Gⱼ : i ∈ I} are the *minimal parabolic subgroups*. Coset geometries let one build
huge incidence geometries from very little data. The category name is `CosetGeom`.

The chapter covers construction and inter-conversion of the two kinds of geometry, elementary
invariants, residues, truncations, shadows and shadow spaces, automorphism and correlation
groups, a battery of property tests (firm / thin / thick / residually connected, flag
transitivity, intersection properties, primitivity properties, local two-transitivity), and
the Buekenhout *diagram* of a firm, residually connected, flag-transitive geometry.

---

## 142.1 Introduction

Introductory section (definitions only — see overview above). Category names: incidence
geometry `IncGeom`; coset geometry `CosetGeom`.

---

## 142.2 Construction of Incidence and Coset Geometries

### 142.2.1 Construction of an Incidence Geometry

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IncidenceGeometry(G)` | Construct the incidence geometry `IG` having the (labelled) graph `G` as incidence graph. If `G` is unlabelled, the elements of `IG` are the vertices and edges of `G`, the types are `{@ 1, 2 @}` (type 1 = vertices, type 2 = edges), and a type-1 element `x` is incident to a type-2 element `y` iff vertex `x` lies on edge `y`. If `G` is labelled (vertices carry labels), the elements of `IG` are the vertices of `G`, the types are the set of vertex labels, and the incidence graph is `G` itself. | Direct construction from the incidence graph. |

*Worked examples:* H142E1 (Petersen graph as a rank-two incidence geometry); H142E2 (rank-three
geometry of vertices/edges/faces of the cube, via a 26-vertex labelled graph);
H142E3 (Hoffman-Singleton graph `HoSi` as a rank-two incidence geometry, after **[BCon]**);
H142E4 (Neumaier geometry: a rank-four incidence geometry built from `HoSi`, after **[BCon]**).

### 142.2.2 Construction of a Coset Geometry

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CosetGeometry(G, S, I)` | Construct the coset geometry `CG` with set of types `I` from group `G` and a set `S` of subgroups of `G`; `S` and `I` must have the same cardinality. If `G` is a permutation group and `S` a set of subgroups, the coset geometry is built by Tits' algorithm; when `S` and `I` are indexed, the cosets of `S[i]` are the elements of type `I[i]` for `i ∈ {0,…,n−1}`, `n = #S`. | Tits' algorithm **[Tit62]**. |
| `CosetGeometry(G, S)` | As above but with the default type set: each subgroup in `S` is assigned a number `0,…,n−1` (`n = #S`). | Tits' algorithm **[Tit62]**. |

*Worked examples:* H142E5 (Petersen graph as a rank-two coset geometry of `Sym(5)`);
H142E6 (cube geometry as a rank-three coset geometry of a subgroup of `Sym(8)`);
H142E7 (a rank-six coset geometry for `Sym(6)` from stabilizer chains).

---

## 142.3 Elementary Invariants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Points(D)` / `Elements(D)` | The set of elements of incidence geometry `D` (the points of its incidence graph). | — |
| `Types(D)` | The set of types of incidence geometry `D`. | — |
| `Types(C)` | The set of types of coset geometry `C`. | — |
| `Rank(D)` | The rank of incidence geometry `D` (cardinality of its type set). | — |
| `Rank(C)` | The rank of coset geometry `C`. | — |
| `IncidenceGraph(D)` | The incidence graph of incidence geometry `D`, together with its vertex set and edge set. Not implemented for coset geometries (convert via `IncidenceGeometry` first). | — |
| `Group(C)` | The group from which coset geometry `C` is constructed. | — |
| `MaxParabolics(C)` / `MaximalParabolics(C)` | An indexed set containing the maximal parabolic subgroups of coset geometry `C`. | — |
| `MinParabolics(C)` / `MinimalParabolics(C)` | An indexed set containing the minimal parabolic subgroups of coset geometry `C`. | — |
| `Borel(C)` / `BorelSubgroup(C)` | The Borel subgroup of `C`, i.e. the intersection of all maximal parabolic subgroups of `C`. | — |
| `Kernel(C)` | A permutation group that is the kernel of `C`: the subgroup of the Borel subgroup that fixes all elements of the geometry `C`. | — |
| `Kernels(C)` | A sequence containing the *i*-kernel `Kᵢ` of each maximal parabolic subgroup `Gᵢ` of `C`; the *i*-kernel of `Gᵢ` is the subgroup of all elements of `Gᵢ` that fix every element of the residue of `Gᵢ`. | — |
| `Quotient(C, K)` | For a coset geometry `C = (G; (Gᵢ)_{i∈I})` and a permutation group `K` that is a normal subgroup of `G` and of all maximal parabolic subgroups of `C`: the coset geometry `(G/K; (Gᵢ/K)_{i∈I})`. | Quotient by a common normal subgroup. |

---

## 142.4 Conversion Functions

Functions converting incidence geometries and coset geometries into other objects.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IncidenceGeometry(C)` | Construct the incidence geometry `IG` from coset geometry `C`. | Tits' algorithm (see §142.1) **[Tit62]**. |
| `CosetGeometry(D)` | Convert incidence geometry `D` into a coset geometry. The group `G` of the resulting coset geometry is `AutomorphismGroup(D)`; Magma finds a chamber `C` of `D` and to each element `x ∈ C` associates its stabilizer `Gₓ` in `G`, taking the `(Gₓ)` as maximal parabolics. To obtain a coset geometry combinatorially isomorphic to `D`, `G` must be transitive on every rank-two truncation of `D`. Returns a boolean (`true` if successful) and, when `true`, the coset geometry `CG`; otherwise returns `false`. | Stabilizers of a chamber under the automorphism group. |
| `Graph(D)` | If `IsGraph(D)` is `true`, the undirected graph corresponding to incidence geometry `D`. | — |
| `Graph(C)` | If `IsGraph(C)` is `true`, the undirected graph corresponding to coset geometry `C`. | — |

*Worked example:* H142E8 (converting the Neumaier geometry into a coset geometry).

---

## 142.5 Residues

Let Γ(X, ∼, t, I) be an incidence geometry and `F` a flag (a clique of the incidence graph).
An element `x` is incident to `F` (written `x ∼ F`) iff it is incident to all elements of `F`.
The *residue* Γ_F is the geometry whose elements are {x ∈ X : x ∼ F}\F, with types I\t(F) and
the restricted type function and incidence relation. For a flag-transitive coset geometry
Γ(G; (Gᵢ)), the residue of a flag `F` is the coset geometry
Γ(∩_{j∈F} Gⱼ ; (Gᵢ ∩ (∩_{j∈F} Gⱼ))_{i∈I\t(F)}).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Residue(D, f)` | For incidence geometry `D` and a flag `f` of `D`: the residue of `f` as an incidence geometry. | Restriction of the incidence graph to elements incident to `f`. |
| `Residue(C, f)` | For coset geometry `C` and a subset `f` of its type set: the residue of the flag consisting of the maximal parabolics of `C` whose type is in `f`. | Intersection of the relevant maximal parabolics. |

---

## 142.6 Truncations

Let Γ(X, ∼, t, I) be an incidence geometry and `J ⊆ I`. The *J-truncation* of Γ is the geometry
with elements t⁻¹(J) and the restricted type function and incidence relation. For a coset
geometry Γ(G; (Gᵢ)_{i∈I}), the J-truncation is Γ(G; (Gⱼ)_{j∈J}).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Truncation(D, t)` | For incidence geometry `D` and a subset `t` of its type set: the `t`-truncation of `D` as an incidence geometry. | Restriction to the chosen types. |
| `Truncation(C, t)` | For coset geometry `C` and a subset `t` of its type set: the `t`-truncation of `C` as a coset geometry. | Restriction to the chosen maximal parabolics. |

---

## 142.7 Shadows

Let Γ(X, ∼, t, I) be an incidence geometry, `J ⊆ I`, and `F` a flag with t(F) ∩ J = ∅. The
*J-shadow* of `F`, denoted σ_J(F), is the set of flags of type `J` in the residue of `F`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Shadow(D, I, F)` | For incidence geometry `D`, a subset `I` of its type set, and a flag `F`: the `I`-shadow of `F`, returned as an indexed set of subsets of points of `D`. | Flags of type `I` within the residue of `F`. |

---

## 142.8 Shadow Spaces

Let Γ(X, ∼, t, I) be an incidence geometry and `J ⊆ I`. The *shadow space* Γ(J) is the
incidence structure whose point set is the set of flags of type `J` and whose blocks are the
`J`-shadows of the flags of Γ.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ShadowSpace(D, I)` | For incidence geometry `D` and a subset `I` of its type set: the shadow space `D(I)` as an incidence structure. | Flags of type `I` as points; `I`-shadows as blocks. |

---

## 142.9 Automorphism Group and Correlation Group

These functions are currently only available for incidence geometries. An *automorphism* α of
Γ(X, ∼, t, I) is an automorphism of the incidence graph that preserves types
(t(α(x)) = t(x) ∀ x); the automorphism group is Aut(Γ). A *correlation* α is an automorphism of
the incidence graph that preserves type-equality
(t(x) = t(y) ⇒ t(α(x)) = t(α(y))); the correlation group is Cor(Γ). Aut(Γ) is a subgroup of
Cor(Γ).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AutomorphismGroup(D)` | The group of type-preserving automorphisms of incidence geometry `D`, as a `GrpPerm` acting on the elements of `D`. | Type-preserving automorphisms of the incidence graph. |
| `CorrelationGroup(D)` | The group of (type-class preserving) automorphisms of incidence geometry `D` (the correlation group), as a `GrpPerm` acting on the elements of `D`. | Automorphisms of the incidence graph preserving type-equality. |

---

## 142.10 Properties of Incidence Geometries and Coset Geometries

An incidence geometry Γ is *flag-transitive* if for every two flags `x`, `y` of the same type
there is `g ∈ Aut(Γ)` with `g(x) = y` (equivalently Aut(Γ) acts flag-transitively); it is a
*flag-transitive geometry* if additionally it contains at least one chamber. A coset geometry
Γ(G; (Gᵢ)_{i∈I}) is flag-transitive if for every two flags of the same type some `g ∈ G` maps
one to the other; then `{(Gᵢ)_{i∈I}}` is a chamber, so it is a flag-transitive geometry.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsFTGeometry(D)` | For incidence geometry `D`: `true` iff its automorphism group acts flag-transitively on `D` and `D` has at least one chamber. | — |
| `IsFTGeometry(C)` | For coset geometry `C`: `true` iff the group of `C` acts flag-transitively on `C`. | — |
| `IsFirm(X)` | For a flag-transitive coset or incidence geometry `X`: `true` iff every flag of `X` is contained in at least two chambers. | — |
| `IsThin(X)` | For a flag-transitive coset or incidence geometry `X`: `true` iff every flag of `X` is contained in exactly two chambers. | — |
| `IsThick(X)` | For a flag-transitive coset or incidence geometry `X`: `true` iff every flag is contained in exactly three chambers. | — |
| `IsResiduallyConnected(X)` / `IsRC(X)` | For a flag-transitive coset or incidence geometry `X`: `true` iff every residue of rank at least two of `X` has a connected incidence graph. | — |
| `IsGraph(D)` | For incidence geometry `D`: tests whether `D` corresponds to a graph — `D` must be rank two such that, for one of the two types `e`, every element of type `e` is incident with exactly two elements of the other type (type-`e` elements ↦ edges, the others ↦ vertices). | — |
| `IsGraph(C)` | For coset geometry `C`: tests whether `C` corresponds to a graph — `C` must be rank two and one of the two maximal parabolics, say `Gₑ`, must contain the Borel subgroup as a subgroup of index 2 (cosets of `Gₑ` ↦ edges, cosets of the other maximal parabolic ↦ vertices). | — |

*Worked examples:* H142E9 (Petersen graph: `IsFirm`, `IsRC`, `IsFTGeometry`, then `Diagram`).

---

## 142.11 Intersection Properties of Coset Geometries

For Γ(X, ∗, t, I), a type `i ∈ I`, and a flag `F`, the *i-shadow* σᵢ(F) is the set of elements
of type `i` incident with `F`. The *intersection property* (IP), as in **[Bue79]**: for every
type `i`, the intersection of the `i`-shadows of a variety `x` and a flag `F` is empty or is the
`i`-shadow of a flag incident to `x` and `F` (and likewise on residues). The condition (IP)₂
(due to Buekenhout, Dehon, Cara and others) asks that all rank-two residues satisfy (IP). For a
geometry of rank `n`, (IP)_k (k = 2,…,n): for every residue `R` of rank `k` and every type `i`
in the types of `R`, the intersection of the `i`-shadows of a variety `x` and a flag `F` is
empty or the `i`-shadow of a flag incident to `x` and `F`. Algorithms to test these are from
Jacobs and Leemans **[JL04]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HasIntersectionPropertyN(C, n)` | For coset geometry `C` and positive integer `n`: returns two booleans — whether `C` satisfies the intersection property of rank `n`, and whether `C` satisfies the weak intersection property of rank `n`. | Algorithm of Jacobs–Leemans **[JL04]**. |
| `HasIntersectionProperty(C)` | `true` iff coset geometry `C` satisfies the intersection property. | Algorithm of Jacobs–Leemans **[JL04]**. |
| `HasWeakIntersectionProperty(C)` | `true` iff coset geometry `C` satisfies the weak intersection property. | Algorithm of Jacobs–Leemans **[JL04]**. |

---

## 142.12 Primitivity Properties on Coset Geometries

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsPrimitive(C)` / `IsPRI(C)` | `true` iff `C` is a primitive geometry, i.e. all of its maximal parabolic subgroups are maximal subgroups of its group. | — |
| `IsWeaklyPrimitive(C)` / `IsWPRI(C)` | `true` iff `C` is a weakly primitive geometry, i.e. at least one of its maximal parabolic subgroups is a maximal subgroup of its group. | — |
| `IsResiduallyPrimitive(C)` / `IsRPRI(C)` | `true` iff `C` is a primitive geometry and all of its residues are as well. | — |
| `IsResiduallyWealyPrimitive(C)` / `IsRWPRI(C)` / `IsRWP(C)` | `true` iff `C` is a weakly primitive geometry and all of its residues are as well. | — |
| `IsLocallyTwoTransitive(C)` / `Is2T1(C)` | `true` iff `C` is locally two-transitive, i.e. all of its minimal parabolic subgroups have a two-transitive action on the cosets of the Borel subgroup. | — |

*(The intrinsic name `IsResiduallyWealyPrimitive` is transcribed as printed in the handbook — the
spelling "Wealy" appears to be a typo for "Weakly", but is reproduced verbatim.)*

---

## 142.13 Diagram of an Incidence Geometry

As defined by Buekenhout **[Bue79]**, the *diagram* of a firm, residually connected,
flag-transitive geometry Γ is a complete graph `K` whose vertices are the types `I` of Γ.
To each vertex `i ∈ I` is attached the order `sᵢ = |Γ_F| − 1` (`F` any flag of type `I\{i}`)
and the number `nᵢ` of type-`i` elements. To each edge `{i, j}` are attached three positive
integers `d_ij`, `g_ij`, `d_ji`, where the *gonality* `g_ij` equals half the girth of the
incidence graph of a residue Γ_F of type `{i, j}`, and `d_ij` (resp. `d_ji`) is the
*i-diameter* (resp. *j-diameter*): the greatest distance from a fixed `i`-element (resp.
`j`-element) to any other element in Γ_F. When `g_ij = d_ij = d_ji = n ≠ 2`, only `g_ij` is
written; when `n = 2`, the edge `{i, j}` is not drawn.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Diagram(D)` / `Diagram(C)` | For a firm, residually connected, flag-transitive incidence geometry `D` or coset geometry `C`: a complete graph `K` whose vertices and edges are labelled. Each vertex `i` is labelled by the sequence `[sᵢ, nᵢ]` (order and number of type-`i` elements); each edge `{i, j}` is labelled by the sequence `[d_ij, g_ij, d_ji]`. The two versions compute the same diagram, but `Diagram(C)` is much faster (it uses the group to compute the parameters) — it is strongly advised to first convert an incidence geometry to a coset geometry before computing its diagram. | Computes order/number per vertex and diameter/gonality/diameter per edge from residues (group-based for coset geometries). |

*Worked examples:* H142E9 (Petersen-graph diagram); H142E10 (cube diagram);
H142E11 (Hoffman-Singleton diagram); H142E12 (Neumaier-geometry diagram).

---

## 142.14 Bibliography (canonical references)

| Key | Reference |
|-----|-----------|
| **[BCon]** | Francis Buekenhout and Arjeh M. Cohen. *Diagram geometry.* In preparation. |
| **[Bue79]** | Francis Buekenhout. *Diagrams for geometries and groups.* J. Combin. Theory Ser. A, **27**(2):121–151, 1979. |
| **[Bue95]** | Francis Buekenhout (ed.). *Handbook of incidence geometry.* North-Holland, Amsterdam, 1995. |
| **[JL04]** | Pascale Jacobs and Dimitri Leemans. *An algorithmic analysis of the intersection property.* LMS J. Comput. Math., **7**:284–299 (electronic), 2004. |
| **[Pas94]** | Antonio Pasini. *Diagram geometries.* Oxford Science Publications. The Clarendon Press, Oxford University Press, New York, 1994. |
| **[Tit62]** | Jacques Tits. *Géométries polyédriques et groupes simples.* Atti 2a Riunione Groupem. Math. Express. Lat. Firenze, pages 66–88, 1962. |

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Tits' coset-geometry algorithm **[Tit62]** | `CosetGeometry(G, S, I)`, `CosetGeometry(G, S)`, `IncidenceGeometry(C)` |
| Construction from / conversion to the incidence graph | `IncidenceGeometry(G)`, `IncidenceGraph(D)`, `Graph(D)`, `Graph(C)`, `CosetGeometry(D)` |
| Residue / truncation / shadow restrictions | `Residue`, `Truncation`, `Shadow`, `ShadowSpace` |
| Incidence-graph automorphisms (type-preserving / type-class preserving) | `AutomorphismGroup(D)`, `CorrelationGroup(D)` |
| Flag-transitivity and chamber-counting tests | `IsFTGeometry`, `IsFirm`, `IsThin`, `IsThick`, `IsResiduallyConnected`/`IsRC` |
| Intersection-property tests **[Bue79, JL04]** | `HasIntersectionProperty`, `HasIntersectionPropertyN`, `HasWeakIntersectionProperty` |
| Primitivity / local-transitivity tests | `IsPrimitive`, `IsWeaklyPrimitive`, `IsResiduallyPrimitive`, `IsResiduallyWealyPrimitive`, `IsLocallyTwoTransitive` |
| Buekenhout diagram **[Bue79]** | `Diagram(D)`, `Diagram(C)` |
