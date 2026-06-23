# Chapter 141 — Finite Planes

**Handbook part:** XIX — Geometry
**Handbook pages:** 4713–4748 (PDF pages 4846–4881)

---

## Scope and overview

This chapter presents the Magma machinery for finite **projective** and **affine** planes. The
two category names are `PlaneProj` (projective) and `PlaneAff` (affine). Within each category
Magma distinguishes *classical* planes — those defined by a vector space of dimension 3 (for
projective planes) or 2 (for affine planes) over a finite field `F_q` — from arbitrary
combinatorially-defined planes. Some functions apply to all planes; others are specific to
projective, affine, or classical planes.

The key objects are:

- **Points** (special type `PlanePt`) and **lines** (special type `PlaneLn`). Points and lines
  may be defined over any Magma object, which improves efficiency. A point is created by
  coercing a suitable object into a parent structure called the **point-set** `V` (category
  `PlanePtSet`); a line is created by coercing into the **line-set** `L` (category
  `PlaneLnSet`). The point-set and line-set are *not* actual Magma sets — they act purely as
  parent structures, supporting creation via the `!` and `.` operators, retrieval of the
  *i*-th element, and random elements.

- Every plane-constructing function returns **three values**: the plane `P`, its point-set `V`,
  and its line-set `L`.

The chapter covers construction (combinatorial and classical), the point-set/line-set
abstraction, the genuine indexed point/line sets, defining-data recovery (`Support`),
subplanes, associated structures (vector space, incidence matrix, dual), numerical invariants
(order, `p`-rank), properties (Desarguesian, self-dual), isomorphism testing, the
projective/affine correspondence, point/line operations, **arcs**, **conics**, **unitals**, the
**collineation (automorphism) group** with its `G`-set action mechanism, central collineations,
transitivity, **translation planes** (Baer/oval derivation), conversion to and from designs, and
the graphs and codes associated with a plane.

---

## 141.1 Introduction

Introduces the categories `PlaneProj` and `PlaneAff`, and the notion of *classical* planes
(defined by a dimension-2 or dimension-3 vector space). Points have the special type `PlanePt`,
lines the special type `PlaneLn`; the point-set and line-set are their parent structures. No
intrinsics are defined in this section.

---

## 141.2 Construction of a Plane

All plane-constructing functions return three values: the plane, its point-set, and its
line-set. The constructors below build either combinatorial planes (from a list of lines /
incidence data) or classical planes (from a vector space or field).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `FiniteProjectivePlane< v \| X : Check >` / `FiniteProjectivePlane< V \| X : Check >` | Construct the projective plane `P` with point-set the indexed set `V` (or `{@1,…,v@}` if an integer `v` is given) and line-set `L = {L₁,…,L_b}` given by `X`. `X` may be: (a) a list of subsets of `V`; (b) a sequence/set/indexed set of subsets of `V`; (c) a list of lines of an existing plane; (d) a sequence/set/indexed set of lines of an existing plane; (e) a combination of the above; (f) a `v × b` (0,1)-matrix over any coefficient ring, interpreted as the incidence matrix; (g) a set of codewords of a linear code of length `v` (line-set = supports of the codewords). Parameter `Check` (default `true`) verifies the projective-plane axioms. | Direct construction from incidence data; optional axiom check. |
| `FiniteProjectivePlane(W)` / `FiniteProjectivePlane(F)` / `FiniteProjectivePlane(q)` | Given a 3-dimensional vector space `W` over `F = F_q` (or the field `F`, or the prime power `q`): construct the classical projective plane `PG(2, q)` defined by the 1- and 2-dimensional subspaces of `W`. | Classical construction via subspaces of `W`. |
| `FiniteAffinePlane< v \| X : Check >` / `FiniteAffinePlane< V \| X : Check >` | Construct the affine plane `P` with point-set `V` (or `{@1,…,v@}`) and line-set given by `X`, where `X` takes the same forms (a)–(g) as for `FiniteProjectivePlane`. Parameter `Check` (default `true`) verifies the affine-plane axioms. | Direct construction from incidence data; optional axiom check. |
| `FiniteAffinePlane(W)` / `FiniteAffinePlane(F)` / `FiniteAffinePlane(q)` | Given a 2-dimensional vector space `W` over `F = F_q` (or `F`, or `q`): construct the classical affine plane `AG(2, q)` defined by the cosets of the subspaces of `W`. | Classical construction via cosets of subspaces of `W`. |

*Worked example: H141E1 (classical `PG(2,3)`; non-classical affine plane of order 2 from `Subsets`; effect of `Check := true` vs `false` when recreating `PG(2,16)`).*

---

## 141.3 The Point-Set and Line-Set of a Plane

### 141.3.1 Introduction

A plane consists of three objects: the plane `P`, the point-set `V`, and the line-set `L`. `V`
and `L` are not actual Magma sets — they are parent structures for points and lines, supporting
creation via the `!` and `.` operators. `V` belongs to category `PlanePtSet`, `L` to
`PlaneLnSet`. No intrinsics are defined directly in this subsection.

### 141.3.2 Creating Point-Sets and Line-Sets

| Intrinsic | Description |
|-----------|-------------|
| `PointSet(P)` | The point-set `V` of a plane `P`. |
| `LineSet(P)` | The line-set `L` of a plane `P`. |

### 141.3.3 Using the Point-Set and Line-Set to Create Points and Lines

| Intrinsic | Description |
|-----------|-------------|
| `V . i` | The *i*-th point of `P`, given the point-set `V`. |
| `V ! [a, b, c]` | For the point-set of a classical projective plane `P = PG₂(K)` and `a,b,c ∈ K`: the projective point `(a : b : c)`. |
| `V ! [a, b]` | For the point-set of a classical affine plane `P = AG₂(K)` and `a,b ∈ K`: the point `(a, b)`. |
| `V ! x` | The point of `P` corresponding to `x`, where `x` is coercible into the underlying point set (for classical planes, coercible to a vector). |
| `Representative(V)` / `Rep(V)` | A representative point of `P`. |
| `Random(V)` | A random point of `P`. |
| `L . i` | The *i*-th line of `P`, given the line-set `L`. |
| `L ! [a, b, c]` | For the line-set of a classical plane over `K` and `a,b,c ∈ K`: the line `⟨a : b : c⟩`, i.e. `ax+by+cz = 0` if `P` is projective or `ax+by+c = 0` if `P` is affine. |
| `L ! [m, b]` | For the line-set of a classical affine plane `P = AG₂(K)` and `m,b ∈ K`: the affine line `y = mx + b`. |
| `L ! S` | For a set or sequence `S` of collinear points of `P`: the line containing the points of `S`. |
| `L ! l` | Given a line `l` of a (possibly) different plane (generally a subplane of `P`): the line of `P` corresponding to `l`. |
| `Representative(L)` / `Rep(L)` | A representative line of `P`. |
| `Random(L)` | A random line of `P`. |

*Worked example: H141E2 (creating points/lines of `PG(2,5)` via `.`, `!`, equations, point lists; non-classical affine plane of order 2; point/line retrieval).*

### 141.3.4 Retrieving the Plane from Points, Lines, Point-Sets and Line-Sets

| Intrinsic | Description |
|-----------|-------------|
| `ParentPlane(V)` | The plane `P` for which `V` is the point-set. |
| `ParentPlane(L)` | The plane `P` for which `L` is the line-set. |
| `ParentPlane(p)` | The plane `P` for which `p` is a point. |
| `ParentPlane(l)` | The plane `P` for which `l` is a line. |

---

## 141.4 The Set of Points and Set of Lines

These functions return *genuine* enumerated indexed sets (in contrast to the point-set/line-set,
which are not true Magma sets).

| Intrinsic | Description |
|-----------|-------------|
| `Points(P)` | An indexed set whose elements are the points of `P` (a standard indexed set, not the point-set; contrast `PointSet`). |
| `Lines(P)` | An indexed set containing the lines of `P` (a standard indexed set; contrast `LineSet`). |

---

## 141.5 The Defining Points of a Plane

Recovers the objects originally used to define the points — elements of the defining indexed
set, or (for a classical plane) the underlying vector-space elements. The results have their
"real" types: they are no longer of type `PlanePt`.

| Intrinsic | Description |
|-----------|-------------|
| `Support(P)` | An indexed set `E` which is the underlying point set of `P` (elements with their real types). |
| `Support(l)` | The set of underlying points contained in the line `l` of `P` (elements with their real types). |
| `Support(P, p)` / `Support(p)` | The Magma object corresponding to the point `p` of `P`. |

*Worked example: H141E3 (`Points` vs `Support` on an affine plane — same elements, different universes; classical `PG(2,2)` where `Support` yields vectors).*

---

## 141.6 Subplanes

The `sub` constructor builds subplanes of a projective or affine plane; for classical planes
`SubfieldSubplane` is also provided.

| Intrinsic | Description |
|-----------|-------------|
| `sub< P \| L >` | The subplane of `P` generated by the points specified by `L`, where `L` is a list of one or more of: (a) a point of `P`; (b) a set/sequence of points of `P`; (c) a subplane of `P`; (d) a set/sequence of subplanes of `P`. The defined point set `S` must include a quadrangle if `P` is projective, or three non-collinear points if `P` is affine. Returns the smallest subplane of `P` containing `S`. |
| `SubfieldSubplane(P, F)` | The plane obtained from the classical plane `P` by taking only those points of `P` whose coordinates all lie in `F`, where `F` must be a subfield of `Field(P)`. |

*Worked example: H141E4 (subplane of `PG₂(4)` generated by a quadrangle, giving a `PG(2,2)`; `SubfieldSubplane` of `AG₂(4)` over `F₂`).*

---

## 141.7 Structures Associated with a Plane

| Intrinsic | Description |
|-----------|-------------|
| `VectorSpace(P)` | The vector space underlying the classical plane `P`. |
| `Field(P)` | The field over which the classical plane `P` is defined. |
| `IncidenceMatrix(P)` | The incidence matrix of the plane `P`. |
| `Dual(P)` | The dual of the projective plane `P`. |

*Worked example: H141E5 (`VectorSpace`/`Field` of `AG₂(4)`; `IncidenceMatrix` of a `PG(2,2)` and `Dual` — `IncidenceMatrix(D) eq Transpose(IncidenceMatrix(P))`).*

---

## 141.8 Numerical Invariants of a Plane

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Order(P)` | The order of the plane `P`. | — |
| `NumberOfPoints(P)` / `#V` | The cardinality `v` of the point-set `V` of `P`. | — |
| `NumberOfLines(P)` / `#L` | The cardinality of the line-set `L` of `P`. | — |
| `pRank(P)` | The `p`-rank of the plane `P` of order `pᵗ`. | Rank of the incidence matrix over `F_p`. |
| `pRank(P, p)` | The `p`-rank of the plane `P` (with `p` specified). | Rank of the incidence matrix over `F_p`. |

*Worked example: H141E6 (`PG₂(8)`: `NumberOfLines` = 73, `Order` = 8, `pRank` = 28, and `pRank(P, p)` for `p = 2, 3, 5`).*

---

## 141.9 Properties of Planes

| Intrinsic | Description |
|-----------|-------------|
| `IsDesarguesian(P)` | Returns `true` iff the plane `P` is a Desarguesian plane. |
| `IsSelfDual(P)` | Returns `true` iff the projective plane `P` is self-dual. |

---

## 141.10 Identity and Isomorphism

Two planes are equal iff their point sets are equal and they have the same lines.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `P eq Q` | Returns `true` if the planes `P` and `Q` are equal, otherwise `false`. | — |
| `P ne Q` | Returns `true` if the planes `P` and `Q` are not equal, otherwise `false`. | — |
| `IsIsomorphic(P, Q : AutomorphismGroups)` | Returns `true` if `P` and `Q` are isomorphic, plus an isomorphism `f : P → Q` if so. Parameter `AutomorphismGroups` (default `"None"`; values `"Both"`, `"Left"`, `"Right"`, `"None"`) chooses which automorphism group(s) to construct first, to assist the test. | Computes none/one/both automorphism groups to guide isomorphism testing. |
| `P subset Q` | Returns `true` if `P` is a subplane of `Q`, otherwise `false`. | — |

---

## 141.11 The Connection between Projective and Affine Planes

Natural constructions converting between projective and affine planes.

| Intrinsic | Description |
|-----------|-------------|
| `FiniteAffinePlane(P, l)` | The affine plane obtained by removing the line `l` from the projective plane `P`, together with the point-set and line-set of the affine plane, plus the embedding map from the affine plane into `P`. Here `l` becomes the line at infinity for this embedding. |
| `ProjectiveEmbedding(P)` | The projective completion of the affine plane `P`, together with the point-set and line-set of the projective plane, plus the embedding map from `P` into the projective plane. The adjoined line at infinity is always the last line in the projective plane's line-set. |

*Worked example: H141E7 (`ProjectiveEmbedding` of `AG₂(3)`, recovering an isomorphic affine plane via `FiniteAffinePlane(P, Random(LineSet(P)))`; using the embedding map `f` with `@` / `@@` to map points and lines between affine and projective, and identifying the line at infinity).*

---

## 141.12 Operations on Points and Lines

### 141.12.1 Elementary Operations

| Intrinsic | Description |
|-----------|-------------|
| `p eq q` | Returns `true` if the points `p` and `q` are equal, otherwise `false`. |
| `p ne q` | Returns `true` if the points `p` and `q` are not equal, otherwise `false`. |
| `l eq m` | Returns `true` if the lines `l` and `m` are equal, otherwise `false`. |
| `l ne m` | Returns `true` if the lines `l` and `m` are not equal, otherwise `false`. |
| `p in l` | Returns `true` if point `p` lies on the line `l`, otherwise `false`. |
| `p notin l` | Returns `true` if point `p` does not lie on the line `l`, otherwise `false`. |
| `S subset l` | For a subset `S` of the point set and a line `l`: `true` if the points of `S` lie on `l`. |
| `S notsubset l` | For a subset `S` of the point set and a line `l`: `true` if the points of `S` do not all lie on `l`. |
| `l meet m` | The unique point common to the lines `l` and `m`. |
| `Representative(l)` / `Rep(l)` | A representative point of `P` incident with `l`. |
| `Random(l)` | A random point of `P` incident with `l`. |

### 141.12.2 Deconstruction Functions

| Intrinsic | Description |
|-----------|-------------|
| `Index(P, p)` | For a point `p` from the point-set `V` of `P`: the index `i` such that `p` is `V.i`. |
| `Index(P, l)` | For a line `l`: the index `i` such that `l` is `L.i` (where `L` is the line-set of `P`). |
| `p[i]` | The *i*-th coordinate of the point `p` (classical plane only). For a projective plane `1 ≤ i ≤ 3`; for an affine plane `1 ≤ i ≤ 2`. |
| `l[i]` | The *i*-th coordinate of the line `l` (classical plane only), `1 ≤ i ≤ 3`. In a classical plane `⟨a : b : c⟩` represents `ax+by+cz = 0` (projective) or `ax+by+c = 0` (affine). |
| `Coordinates(P, p)` | For a point `p = (a : b : c)` from a classical projective plane (or `p = (a, b)` from a classical affine plane): the sequence `[a,b,c]` (or `[a,b]`). |
| `Coordinates(P, l)` | For a line `l = ⟨a : b : c⟩` from a classical plane: the coordinate sequence `[a,b,c]`. |
| `ElementToSequence(p)` / `Eltseq(p)` | The coordinate sequence of a point `p` from a classical projective (`[a,b,c]`) or affine (`[a,b]`) plane. |
| `ElementToSequence(l)` / `Eltseq(l)` | The coordinate sequence `[a,b,c]` of a line `l` from a classical plane. |
| `Set(l)` | The set of points contained in the line `l`. |

*Worked example: H141E8 (`PG₂(4)`: build a line via `L![1,0,1]`, list its points with `Set`, get `Coordinates`/`l[1]`/`Index`, test point/set membership, form the line through a point set with `L!S`, and find `l meet l2`).*

### 141.12.3 Other Point and Line Functions

| Intrinsic | Description |
|-----------|-------------|
| `IsCollinear(P, S)` | Returns `true` if the set `S` of points of `P` are collinear; if so, the line they define is also returned. |
| `IsConcurrent(P, R)` | Returns `true` if the set `R` of lines of `P` are concurrent; if so, their common point is also returned. |
| `ContainsQuadrangle(P, S)` | Returns `true` if the set `S` of points of `P` contains a quadrangle. |
| `Pencil(P, p)` | The pencil of lines passing through the point `p` in the plane `P`. |
| `Slope(l)` | The slope of the line `l` of a classical affine plane `P`. |
| `IsParallel(P, l, m)` | Returns `true` if the line `l` is parallel to the line `m` in the affine plane `P`. |
| `ParallelClass(P, l)` | The parallel class containing the line `l` of an affine plane `P`. |
| `ParallelClasses(P)` | The partition into parallel classes of the lines of the affine plane `P`. |

*Worked example: H141E9 (`AG₂(3)`: `Slope` of `y = 2x+1`, `ParallelClass` and matching slopes, `Pencil` through a point).*

---

## 141.13 Arcs

A `k`-*arc* in a projective or affine plane `P` is a set of `k` points, no three collinear. A
`k`-arc is *complete* if it cannot be extended to a `(k+1)`-arc. A *tangent* meets the arc
exactly once, a *secant* exactly twice, and a *passant* (external line) not at all.

| Intrinsic | Description |
|-----------|-------------|
| `kArc(P, k)` | A `k`-arc for the plane `P`. |
| `CompleteKArc(P, k)` | A complete `k`-arc for the plane `P` (if one exists). |
| `IsArc(P, A)` | Returns `true` if the set of points `A` is an arc in `P`, i.e. no three points of `A` are collinear. |
| `IsComplete(P, A)` | Returns `true` if the `k`-arc `A` is complete in `P`. |
| `Conic(P, S)` | Given a set `S` of five points of a classical projective plane `P` of order `n > 3` in general position: the unique conic passing through them. |
| `QuadraticForm(S)` | Given a set `S` of five points of a classical projective plane of order `n > 3` in general position: the quadratic form defining the conic through the five points. |
| `Tangent(P, A, p)` | Given an arc `A` in `P` and a point `p` on `A`: a tangent to `A` at `p`. |
| `AllTangents(P, A)` | Given an arc `A` in `P`: the set of tangent lines to `A`. |
| `AllSecants(P, A)` | Given an arc `A` in `P`: the set of secant lines to `A`. |
| `ExternalLines(P, A)` / `AllPassants(P, A)` | Given an arc `A` in `P`: the set of external lines (passants) to `A`. |
| `Knot(P, C)` | Given a conic `C` in the projective plane `P` of even order: the knot of `C`, i.e. the intersection point of the tangents to `C`. |
| `Exterior(P, C)` | Given a conic `C` in the projective plane `P` of odd order: the exterior points of `C`, i.e. the points lying on two tangents of `C`. |
| `Interior(P, C)` | Given a conic `C` in the projective plane `P` of odd order: the interior points of `C`, i.e. the points lying on no tangent of `C`. |

*Worked example: H141E10 (oval design from `PG₂(16)` via `kArc`, `ExternalLines`, `Design`; `PG₂(9)`: `kArc`, `Conic`, `Interior`, then `SubfieldSubplane`, `kArc`/`IsArc`/`IsComplete` on a subplane, `sub`, `Tangent`/`AllTangents`).*

---

## 141.14 Unitals

A unital in the classical projective plane `PG₂(q²)` is a set of `q³ + 1` points such that every
line meeting two of them meets exactly `q + 1` of them.

| Intrinsic | Description |
|-----------|-------------|
| `IsUnital(P, U)` | For a set of points `U` of a projective plane `P` over a field of cardinality `q²`: returns `true` if `U` is a unital. |
| `AllTangents(P, U)` | Given a unital set of points `U` in the projective plane `P`: the set of tangents to the points of `U`. |
| `UnitalFeet(P, U, p)` | The set of intersections of the unital `U` with the tangents to `U` in `P` which pass through the point `p`. |

*Worked example: H141E11 (Hermitian unital `x^(q+1) + y^(q+1) + z^(q+1) = 0` in `PG₂(q²)` for `q = 3`; `IsUnital`, `UnitalFeet`, and forming the resulting 2-(28,4,1) design).*

---

## 141.15 The Collineation Group of a Plane

The automorphism (collineation) group `A` of a plane `P` is presented as a permutation group `G`
acting on the standard support `{1,…,v}`, where `v` is the number of points of `P`. Because a
group acting directly on the plane's objects would print unreadable permutations, `G` acts on
the abstract support, and `G`-sets transfer the action to sets associated with `P` (most
importantly the point set and the line set). The mapping structure `Aut(P)` denotes the set of
collineations of `P` viewed as actual mappings of `P` into itself; a transfer map converts a
permutation of `G` into a mapping in `Aut(P)`.

### 141.15.1 The Collineation Group Function

| Intrinsic | Description |
|-----------|-------------|
| `CollineationGroup(P)` / `AutomorphismGroup(P)` / `PointGroup(P)` | The collineation group `G` of `P` as a permutation group on the standard support `{1,…,v}`. Also returns: a `G`-set `Y` (the point set acted on by `G`); a `G`-set `W` (the line set acted on by `G`); a power structure `S`; and a transfer map `t`. For `g ∈ G`, `f = t(g)` is the map in `S` from `P` to `P` (mapping both point sets and line sets). The `G`-sets `Y`, `W` are used when computing stabilizers or similar subgroups so the appropriate action is used. |
| `LineGroup(P)` | The collineation group `G` of `P` in its action on the lines of `P`, as a permutation group on `{1,…,l}` (`l` = number of lines). Also returns a power structure `S` and transfer map `t`, so `f = t(g)` maps `L` to `L`. |
| `CollineationGroupStabilizer(P, k)` | A subgroup `G` of the collineation group stabilizing the first `k` base points, together with the points `G`-set, the lines `G`-set, the power structure `A` of all automorphisms of `P`, and the transfer map `t` from `G` into `A`. |
| `CollineationSubgroup(P)` | A subgroup `G` of the collineation group generated by one element, together with the points `G`-set, the lines `G`-set, the power structure `A` of all automorphisms, and the transfer map `t` from `G` into `A`. |

### 141.15.2 General Action of Collineations

The collineation group acts on the standard support; the two basic `G`-sets correspond to the
action on the point set `V` and the line set `L`, returned by `AutomorphismGroup`. Additional
`G`-sets may be built with the `G`-set constructors, and studied with the permutation-group
functions that take a `G`-set argument (see the permutation-groups chapter). The `^` operator
also yields actions on the plane.

| Intrinsic | Description |
|-----------|-------------|
| `y ^ g` | For `G` a subgroup of the collineation group, `g ∈ G`, and `y` a point or line of `P`: the image of `y` under `g`. |
| `y ^ G` | For `y` a point or line of `P`: the orbit of `y` under `G`. |
| `Image(g, Y, y)` | For a `G`-set `Y` and `y` belonging to `Y` (or a `G`-set derived from `Y`): the image of `y` under `g`. |
| `Orbit(G, Y, y)` | For a `G`-set `Y` and `y` in `Y` (or derived from it): the orbit of `y` under `G`. |
| `Orbits(G, Y)` | For a `G`-set `Y`: the orbits of the action of `G` on `Y`. |
| `Stabilizer(G, Y, y)` | For a `G`-set `Y` and `y` in `Y` (or derived from it): the stabilizer of `y` in `G`. |
| `Action(G, Y)` | For a `G`-set `Y`: the homomorphism `φ : G → L` giving the action of `G` on `Y`. Returns (a) `φ`; (b) the induced group `L`; (c) the kernel of the action. |
| `ActionImage(G, Y)` | For a `G`-set `Y`: the permutation group `L` giving the action of `G` on `Y`. |
| `ActionKernel(G, Y)` | For a `G`-set `Y`: the kernel of the action of `G` on `Y`. |

*Worked examples: H141E12 (`CollineationGroup` of `PG(2,3)`; `Stabilizer` on the points `G`-set `Y` and lines `G`-set `W`, equality with `CollineationGroupStabilizer`, orbit of a line via `^`). H141E13 (function `Bundle` constructing a projective bundle in `PG₂(q)` using `CollineationGroup`, `Orbit`, `Conic`, `SubfieldSubplane`).*

### 141.15.3 Central Collineations

Let `p` be a point and `l` a line of a projective plane `P`. A `(p, l)`-*central collineation*
is a collineation `α` of `P` fixing `l` pointwise and `p` linewise; `l` is the *axis* and `p`
the *centre* of `α`.

| Intrinsic | Description |
|-----------|-------------|
| `CentralCollineationGroup(P, p, l)` | The group `G` of `(p, l)`-central collineations of `P`. Also returns a power structure `S` and transfer map `t`, so `f = t(g)` represents `g` as a mapping in `S` from `P` to `P` (mapping both point sets and line sets). |
| `CentralCollineationGroup(P, p)` | The group of central collineations with centre `p` of `P` (with `S` and `t` as above). |
| `CentralCollineationGroup(P, l)` | The group of central collineations with axis `l` of `P` (with `S` and `t` as above). |
| `IsCentralCollineation(P, g)` | Returns `true` iff the collineation `g` of `P` is a central collineation; if `true`, also returns the centre and axis of `g`. The support of the parent of `g` must be the point set of `P` or the standard support `{1,…,v}`. |

*Worked example: H141E15 (a `PG(2,3)`-style combinatorial plane: `CentralCollineationGroup(P, p, l)` of order 3, `IsCentralCollineation` recovering centre and axis, and lines through the centre fixed by the group).*

### 141.15.4 Transitivity Properties

| Intrinsic | Description |
|-----------|-------------|
| `IsPointTransitive(P)` / `IsTransitive(P)` | Returns `true` iff the collineation group of `P` acts transitively on the points of `P`. |
| `IsLineTransitive(P)` | Returns `true` iff the collineation group of `P` acts transitively on the lines of `P`. |

*Worked example: H141E16 (`AG₂(4)`: both `IsPointTransitive` and `IsLineTransitive` return `true`).*

---

## 141.16 Translation Planes

Functions for constructing translation planes by derivation (original code due to Jenny Key).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BaerDerivation(q2)` | An affine plane constructed by the technique of derivation with respect to a Baer subplane, where `q2` is an even power of a prime. | Baer-subplane derivation. |
| `BaerSubplane(P)` | A Baer subplane of the projective plane `P`. | — |
| `OvalDerivation(q : HallOval, Print)` | A translation plane from `PG(2, q)` (`q` a power of 2), computed by derivation with respect to an oval. By default the oval is that defined by the points of the conic `y² = xz` together with the nucleus, in `PG(2, q)`. When `q = 16` and `HallOval := true`, a Hall oval is used. `Print := true` prints information during the computation. | Oval derivation. |

*Worked examples: H141E14 (function `BaerDerivation` building an affine plane via `SubfieldSubplane`, `Stabilizer`, `Orbit` on Baer subplanes). The chapter also illustrates these constructions in §141.15 (H141E14 listed there).*

---

## 141.17 Planes and Designs

Projective and affine planes can be viewed as special kinds of designs.

| Intrinsic | Description |
|-----------|-------------|
| `Design(P)` | The design corresponding to the points and lines of the plane `P`. |
| `FiniteAffinePlane(D)` | The affine plane corresponding to the incidence structure `D`. |
| `FiniteProjectivePlane(D)` | The projective plane corresponding to the incidence structure `D`. |

*Worked example: H141E17 (development of a Singer difference set `SingerDifferenceSet(2,3)` → `Development` → a 2-(13,4,1) design → `FiniteProjectivePlane` giving a `PG(2,3)`).*

---

## 141.18 Planes, Graphs and Codes

Functions creating the graphs and codes naturally associated to a projective or affine plane.

| Intrinsic | Description |
|-----------|-------------|
| `LineGraph(P)` | The line graph of the plane `P`. |
| `IncidenceGraph(P)` | The incidence graph of the plane `P`: a bipartite graph whose vertex set is the union of the point set `V` and line set `L`, with `p ∈ V` adjacent to `l ∈ L` iff `p ∈ l`. |
| `LinearCode(P, K)` | The linear code over the field `K` associated with the plane `P` (its rows being the incidence-matrix rows). |

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Combinatorial plane construction (incidence data / matrix / code supports, with axiom check) | `FiniteProjectivePlane<…>`, `FiniteAffinePlane<…>` |
| Classical plane construction (vector space / field subspaces and cosets) | `FiniteProjectivePlane(W/F/q)`, `FiniteAffinePlane(W/F/q)` |
| Point-set / line-set coercion and retrieval | `PointSet`, `LineSet`, `V!…`, `L!…`, `Representative`, `Random` |
| Defining-data recovery | `Support` |
| Subplane generation | `sub<…>`, `SubfieldSubplane` |
| Associated linear-algebra structures | `VectorSpace`, `Field`, `IncidenceMatrix`, `Dual` |
| Numerical invariants (incidence-matrix rank) | `Order`, `NumberOfPoints`, `NumberOfLines`, `pRank` |
| Isomorphism via automorphism groups | `IsIsomorphic`, `IsDesarguesian`, `IsSelfDual` |
| Projective ↔ affine correspondence | `FiniteAffinePlane(P, l)`, `ProjectiveEmbedding` |
| Arcs, conics, ovals | `kArc`, `CompleteKArc`, `IsArc`, `IsComplete`, `Conic`, `QuadraticForm`, `Tangent`, `AllTangents`, `AllSecants`, `ExternalLines`, `Knot`, `Exterior`, `Interior` |
| Unitals | `IsUnital`, `AllTangents`, `UnitalFeet` |
| Collineation group as permutation group + `G`-set action | `CollineationGroup`, `AutomorphismGroup`, `PointGroup`, `LineGroup`, `CollineationGroupStabilizer`, `CollineationSubgroup`, `Orbit(s)`, `Stabilizer`, `Action`, `ActionImage`, `ActionKernel` |
| Central collineations | `CentralCollineationGroup`, `IsCentralCollineation` |
| Transitivity | `IsPointTransitive`, `IsLineTransitive` |
| Translation planes by derivation (Baer / oval) | `BaerDerivation`, `BaerSubplane`, `OvalDerivation` |
| Plane ↔ design conversion | `Design`, `FiniteAffinePlane(D)`, `FiniteProjectivePlane(D)` |
| Associated graphs and codes | `LineGraph`, `IncidenceGraph`, `LinearCode` |
