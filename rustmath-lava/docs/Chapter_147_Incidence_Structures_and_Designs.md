# Chapter 147 — Incidence Structures and Designs

**Handbook part:** XX — Combinatorics
**Handbook pages:** 4871–4905 (PDF pages 5007–5039)

---

## Scope and overview

This chapter covers Magma's machinery for *incidence structures* and the families of objects
that specialise them: near-linear spaces, linear spaces and *t*-designs. The basic object is a
triple `D = (P, B, I)` where `P` is a set of *points*, `B` a set of *blocks*, and `I ⊆ P × B`
an incidence relation whose elements are called *flags*. Usually blocks are subsets of `P`
(so `p ∈ b` replaces `(p,b) ∈ I`); repeated blocks are allowed, and a structure with no
repeated blocks is *simple*.

Specialisations form a hierarchy of Magma categories:

- **Incidence structure** (`Inc`) — the most general object.
- **Near-linear space** (`IncNsp`) — every block (called a *line*) has at least two points and
  any two points lie in at most one line.
- **Linear space** (`IncLsp`) — a near-linear space in which any two points lie in *exactly*
  one line.
- **t-design** (`Dsgn`) — a simple, uniform (blocksize `k`) incidence structure on `v` points
  that is *t-balanced*: every *t*-subset of `P` lies in exactly `λ` blocks. Written
  `t-(v, k, λ)`; `λ` is the *index*. With `v = b` and `t ≥ 2` it is *symmetric*; with `λ = 1`
  it is a *Steiner* design; a *trivial* (every `k`-subset is a block) design is *complete*.

The chapter describes constructors (including from `(0,1)`-matrices, codes, difference sets,
Witt/Mathieu designs), related-structure operations (complement, dual, contraction, residual,
restriction), elementary invariants, properties and balance testing, resolutions and
parallelisms, conversion between categories, identity/isomorphism testing, automorphism-group
computation with its associated G-set action machinery, conversions to graphs and codes, and
automorphism groups of matrices regarded as designs.

**Note on the point-set and block-set.** A Magma incidence structure consists of three objects:
the *point-set* `P` (category `IncPtSet`), the *block-set* `B` (category `IncBlkSet`), and the
structure `D` itself. `P` and `B` are not true Magma sets; they are parent structures for the
points (category `IncPt`) and blocks (category `IncBlk`), enabling creation via the `!` and `.`
operators. Most constructors return all three (`D`, `P`, `B`).

---

## 147.2 Construction of Incidence Structures and Designs

Constructors take a point specification (an integer `v` giving `P = {@ 1,…,v @}`, or an
explicit indexed set `P`) and a block/line specification `X`, which may be a list/sequence/set
of subsets of `P`, blocks of an existing structure, a `v × b` `(0,1)`-matrix interpreted as an
incidence matrix, or a set of codewords (blocks = supports of codewords). Each returns three
values: the structure, its point-set, and its block-set.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IncidenceStructure< v \| X >` / `IncidenceStructure< P \| X >` | Construct the incidence structure `D` with point set `P = {@ 1,…,v @}` (or the given indexed set `P`) and block set `B = {B₁,…,B_b}` given by `X`. `X` may be (a) a list of subsets of `P`; (b) a sequence/set/indexed set of subsets; (c) a list of blocks of an existing structure; (d) a sequence/set/indexed set of such blocks; (e) a combination; (f) a `v × b` `(0,1)`-matrix `A` over any coefficient ring (the incidence matrix); (g) a set of codewords of a length-`v` linear code (blocks = supports). Returns `D`, point-set `P`, block-set `B`. | Direct construction; repeated blocks permitted. |
| `NearLinearSpace< v \| X : parameters >` / `NearLinearSpace< P \| X : parameters >` | Construct the near-linear space `S` on points `P` with lines `L` given by `X` (same forms (a)–(g) as above). Lines must each contain ≥ 2 points and any two points must lie on at most one line. `Check` (default `true`) toggles verification of these two properties. Returns `S`, `P`, line-set `L`. | Direct construction with optional near-linear-space property check. |
| `LinearSpace< v \| X : parameters >` / `LinearSpace< P \| X : parameters >` | Construct the linear space `S` on points `P` with lines `L` given by `X` (forms (a)–(g)). Lines must each contain ≥ 2 points and any two points must lie on *precisely* one line. `Check` (default `true`) toggles verification. Returns `S`, `P`, line-set `L`. | Direct construction with optional linear-space property check. |
| `Design< t, v \| X : parameters >` / `Design< t, P \| X : parameters >` | Construct the `t-(v, k, λ)` design `D` on points `P` with blocks `B` given by `X` (forms (a)–(g)). Blocks must all have the same size `k`, every `t`-subset of `P` must lie in the same number `λ > 0` of blocks, and there must be no repeated blocks. `Check` (default `true`) toggles verification; `Al` (default `"NoOrbits"`) selects the balance-testing algorithm (see §147.8). Returns `D`, `P`, `B`. | Direct construction with uniformity / t-balance / simplicity checks; balance test per `Al`. |

*Worked examples:* H147E1 (Fano plane as `Design<2,7|…>`; an incidence structure with repeated
blocks; a linear space from an incidence matrix via `RMatrixSpace`; a `4-(23,7,1)` design from
the minimum-weight words of the unextended binary Golay code with `Check := false`).

---

## 147.3 The Point-Set and Block-Set of an Incidence Structure

### 147.3.1 Introduction

A Magma incidence structure consists of the point-set `P`, the block-set `B`, and `D` itself.
`P` and `B` are parent structures (not true sets): `P` is of category `IncPtSet`, `B` of
category `IncBlkSet`. They support creation of points and blocks via the `!` and `.` operators.

### 147.3.2 Creating Point-Sets and Block-Sets

| Intrinsic | Description |
|-----------|-------------|
| `PointSet(D)` | The point-set `P` of incidence structure `D`. |
| `BlockSet(D)` | The block-set `B` of incidence structure `D`. |

### 147.3.3 Creating Points and Blocks

Points and blocks have dedicated types `IncPt` and `IncBlk`.

| Intrinsic | Description |
|-----------|-------------|
| `Point(D, i)` | The *i*-th point of `D`. |
| `P . i` | The *i*-th point of `D`, given the point-set `P` and an integer `i`. |
| `Representative(P)` / `Rep(P)` | A representative point of `D`, given point-set `P`. |
| `Random(P)` | A random point of `D`, given point-set `P`. |
| `P ! x` | The point of `D` corresponding to element `x` of the indexed set used to create `D`. |
| `Block(D, i)` | The *i*-th block of `D`. |
| `B . i` | The *i*-th block of `D`, given the block-set `B` and an integer `i`. |
| `Representative(B)` / `Rep(B)` | A representative block of `D`, given block-set `B`. |
| `Random(B)` | A random block of `D`, given block-set `B`. |
| `B ! S` | Tries to coerce a set `S` into the block-set `B`. |
| `Representative(b)` / `Rep(b)` | Given a block `b` of `D`, a representative point of `D` incident with `b`. |
| `Random(b)` | Given a block `b` of `D`, a random point incident with `b`. |

*Worked example:* H147E2 (creating points/blocks of an incidence structure on 5 points; `B.2`,
`P.4`, `P!4`, `P.5 eq Point(D,5)`, `Random(B)`, `Rep(b)`, `Parent`, `B!{…}`).

---

## 147.4 General Design Constructions

Each construction in §147.4.1 returns three values: the incidence structure `D`, its point-set
`P`, and its block-set `B`.

### 147.4.1 The Construction of Related Structures

All operations defined for incidence structures apply also to near-linear spaces, linear spaces
and designs.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Complement(D)` | The complement of the incidence structure `D`. | — |
| `Dual(D)` | The dual of the incidence structure `D`. | Interchange of roles of points and blocks. |
| `Contraction(D, p)` | For `D = (P, B)` and point `p ∈ P`: the structure `E = (P − {p}, {b − {p} : b ∈ B, p ∈ b})`, i.e. delete `p` and retain only the blocks incident with it (minus `p`). | Direct definition. |
| `Contraction(D, b)` | For `D = (P, B)` and block `b ∈ B`: the structure `E = (b, {b ∩ c : c ∈ B, c ≠ b})`, i.e. point set `b` with blocks the non-empty intersections of `b` with the other blocks. | Direct definition. |
| `Residual(D, b)` | For `D = (P, B)` and block `b ∈ B`: the structure `E = (P − b, B − {b})`; blocks are the non-empty intersections of `P − b` with the blocks of `D`. | Direct definition. |
| `Residual(D, p)` | For `D = (P, B)` and point `p ∈ P`: the structure `E = (P − {p}, {x : x ∈ B, p ∉ x})`; blocks are those of `D` not containing `p`. | Direct definition. |
| `Simplify(D)` | Simplify `D`, i.e. remove repeated blocks. | — |
| `Sum(Q)` | For a sequence `Q = [D₁,…,D_l]` of incidence structures over the same point set `P`: the structure with the union of the block sets, `D = (P, B₁ ∪ … ∪ B_l)`. | Block-set union. |
| `Union(D, E)` | The union of `D = (P, B)` and `E = (Q, C)`: `U = (P ∪ Q, B ∪ C)`. Point sets `P` and `Q` must be disjoint. | Disjoint union. |
| `Restriction(D, S)` | The restriction of the (near-)linear space `D` to the set of points `S`. | — |

*Worked example:* H147E3 (a `3-(8,4,1)` design `K`; `Contraction(K, Point(K,8))` gives the
Fano `2-(7,3,1)`; `Residual(K, Block(K,1))`; `Simplify` of the residual).

### 147.4.2 The Witt Designs

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `WittDesign(n)` | The Witt 5-design on `n` points, where `n = 12` or `24` (the small `5-(12,6,1)` and large `5-(24,8,1)` Mathieu designs). | Standard construction of the Witt/Mathieu designs. |

*Worked example:* H147E4 (`WittDesign(24)` = `5-(24,8,1)`; its contraction at a point is the
`4-(23,7,1)` design from the unextended binary Golay code).

### 147.4.3 Difference Sets and their Development

For a group `G` of order `v` and integers `k, λ` with `1 < k < v`, a `(v, k, λ)` *difference
set* is a set `D` of `k` group elements such that `{ g h⁻¹ : g, h ∈ D, g ≠ h }` contains every
non-identity element of `G` exactly `λ` times.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DifferenceSet(p, t)` | The difference set of type `t` (one of `"Q"`, `"H6"`, `"T"`, `"B"`, `"B0"`, `"O"`, `"OO"`, `"W4"`) corresponding to prime `p`. | Types as in Marshall Hall, **[Hal86]**, pp. 141–142. |
| `SingerDifferenceSet(n, q)` | The Singer difference set corresponding to a hyperplane of `PG(n, q)`. | Singer cycle on the projective geometry `PG(n, q)`. |
| `IsDifferenceSet(B)` | Returns `true` iff `B` is a difference set over an integer residue class ring or a finite group (with an iterator); if `true`, also returns `λ` (the number of times each non-identity group/ring element appears as a difference of elements of `B`). | Direct verification of the difference-set property. |
| `Development(B)` | For `B` a subset of a magma `A` (one of `Z/mZ`, a finite abelian group, or an arbitrary finite group with an iterator) that is a difference set relative to `A`: constructs the symmetric design with point set `A` whose blocks are the translates of `B` by each element of `A`. | Translation of `B` by the elements of `A`. |
| `Development(T)` | For a difference family `T = {B₁,…,B_l}` of subsets of a magma `A` (as above): constructs the incidence structure with point set `A` whose *i*-th block is `{B₁ ∪ … ∪ B_l}` translated by the *i*-th element of `A`. | Translation of the difference family by the elements of `A`. |

*Worked example:* H147E5 (`{1,3,4,5,9}` mod 11 is an `(11,5,2)` difference set; `IsDifferenceSet`,
`Development` give a `2-(11,5,2)` design; the twin-primes `DifferenceSet(17,"T")` mod 323 gives a
`2-(323,161,80)` design).

---

## 147.5 Elementary Invariants of an Incidence Structure

All operations apply also to near-linear spaces, linear spaces and designs.

| Intrinsic | Description |
|-----------|-------------|
| `NumberOfPoints(D)` / `#P` | The cardinality `v` of the point set `P` of `D`. |
| `Points(D)` | An indexed set `E` whose elements are the points of `D`. (A standard set, *not* the point-set, in contrast to `PointSet`.) |
| `Support(D)` | An indexed set `E` which is the underlying point set of `D` (elements have their "real" types, no longer in category `IncPt`). |
| `PointDegrees(D)` | A sequence whose *i*-th term is the number of blocks containing the *i*-th point of `D`. |
| `NumberOfBlocks(D)` / `#B` | The number of blocks `b` of `D` with block-set `B`. |
| `Blocks(D)` | An indexed set containing the blocks of `D` (a standard set, in contrast to `BlockSet`). |
| `BlockDegrees(D)` / `BlockSizes(D)` | A sequence whose *i*-th term is the number of points in the *i*-th block of `D`. |
| `Covalence(D, S)` | For a subset `S` of the point set of `D`: the number of blocks of `D` that contain `S`. |
| `IncidenceMatrix(D)` | The incidence matrix of `D`. |
| `pRank(D, p)` | The *p*-rank of the incidence structure `D`. |

---

## 147.6 Elementary Invariants of a Design

The following functions can be applied only to designs.

| Intrinsic | Description |
|-----------|-------------|
| `Parameters(D)` | The parameters `t-(v, b, r, k, λ)` of the design `D`, returned as a record. |
| `ReplicationNumber(D)` | The number of blocks `r` containing any point of the `t-(v,k,λ)` design `D` (`t > 0`). |
| `BlockDegree(D)` / `BlockSize(D)` | The number of points in a block of the design `D`. |
| `Covalence(D, s)` | For an integer `s` with `0 ≤ s ≤ t`: the value `λ_s`, i.e. the number of blocks that contain an arbitrary `s`-subset of the points of `D`. |
| `Order(D)` | The order of the `t-(v,k,λ)` design `D`. Defined only for designs with `t ≥ 2`. |
| `IntersectionNumber(D, i, j)` | The block intersection number `λᵢʲ`, i.e. the number of blocks of `D` containing an `i`-set and disjoint from a `j`-set. Requires `i + j ≤ t`. |
| `PascalTriangle(D)` | The "Pascal triangle" of `D`, returned as a sequence; the *i*-th element is the sequence `[λᵢ₋₁ⁱ⁻¹, λᵢ₋₁ⁱ⁻², …, λᵢ₋₁⁰]` representing the *i*-th row. For a Steiner `t`-design the triangle has `k+1` rows (`k` the blocksize); otherwise `t+1` rows. |

*Worked example:* H147E6 (a Fano `Design<2,7|…>` and an `IncidenceStructure`; `Points`, `Blocks`,
`IncidenceMatrix`, `Points`/`Support` with `Universe`, `Covalence`, `Order`, `PascalTriangle`).

---

## 147.7 Operations on Points and Blocks

In incidence structures, blocks are basically sets, so the set operations `join`, `meet` and
`subset` work on blocks. However blocks are not true Magma enumerated sets; `Set` and `Support`
convert a block to an enumerated set of points.

| Intrinsic | Description |
|-----------|-------------|
| `p in B` | Returns `true` if point `p` lies in block `B`, else `false`. |
| `p notin B` | Returns `true` if point `p` does not lie in block `B`, else `false`. |
| `S subset B` | For a subset `S` of the point set of `D` and a block `B`: `true` if `S` lies in `B`, else `false`. |
| `S notsubset B` | For a subset `S` of the point set and a block `B`: `true` if `S` does not lie in `B`, else `false`. |
| `PointDegree(D, p)` | The number of blocks of `D` that contain the point `p`. |
| `BlockDegree(D, B)` / `BlockSize(D, B)` / `#B` | The number of points contained in the block `B` of `D`. |
| `Set(B)` | The set of points contained in the block `B`. |
| `Support(B)` | The set of underlying points contained in block `B` (elements have their "real" types, no longer in category `IncPt`). |
| `IsBlock(D, S)` | Returns `true` iff the set (or block) `S` represents a block of `D`; if `true`, also returns one such block. |
| `Line(D, p, q)` / `Block(D, p, q)` | A block of `D` containing the points `p` and `q` (if one exists). In linear spaces such a block exists and is unique (for `p ≠ q`). |
| `ConnectionNumber(D, p, B)` | The connection number `c(p, B)`, i.e. the number of blocks joining `p` to `B` in `D`. |

*Worked example:* H147E7 (a `2-(7,4,2)` design; `P.1 in B.1`, `subset`, `Block(D,P.1,P.2)`,
`meet`, `Set`, `Support` with `Universe`).

---

## 147.8 Elementary Properties of Incidence Structures and Designs

**Testing the t-balance: the parameter `Al`.** A parameter `Al` selects the balance-testing
algorithm. The default `"NoOrbits"` applies a brute-force test. `"Orbits"` uses the orbits of
`t`-sets under the automorphism group — much faster in some cases, slower in others.
`"FastBalanceTest"` is another brute-force test whose implementation is dramatically more
efficient, especially for larger `t` (`t ≥ 4`), but it may require more memory than is available
(possible out-of-memory error); for this reason it is not the default, though its use is
strongly recommended as execution should succeed in most cases.

| Intrinsic | Description |
|-----------|-------------|
| `IsSimple(D)` | `true` iff `D` has no repeated blocks. |
| `IsTrivial(D)` | `true` iff `D` is a trivial incidence structure. |
| `IsSelfDual(D)` | `true` iff `D` is self-dual (isomorphic to its dual). |
| `IsUniform(D)` | `true` iff `D` is uniform (every block has the same number of points); if `true`, also returns the blocksize. |
| `IsNearLinearSpace(D)` | `true` iff `D` is a near-linear space. |
| `IsLinearSpace(D)` | `true` iff `D` is a linear space. |
| `IsDesign(D, t : parameters)` | `true` iff `D` is a `t`-design; if `true`, also returns the number of blocks of `D` containing a general `t`-set. Parameter `Al` (default `"NoOrbits"`) selects the balance-test algorithm (see above). |
| `IsBalanced(D, t : parameters)` | `true` iff `D` is balanced with respect to `t`; if `true`, also returns the number of blocks containing a general `t`-set. Parameter `Al` (default `"NoOrbits"`). |
| `IsComplete(D)` | `true` iff `D` is the complete design. |
| `IsSymmetric(D)` | `true` iff the design `D` is symmetric. |
| `IsSteiner(D, t)` | `true` iff the design `D` is a Steiner `t`-design. Parameter `Al` (default `"NoOrbits"`). |
| `IsPointRegular(D)` | `true` iff the (near-)linear space `D` is point regular; if `true`, also returns the point regularity. |
| `IsLineRegular(D)` | `true` iff the (near-)linear space `D` is line regular; if `true`, also returns the line regularity. |

---

## 147.9 Resolutions, Parallelisms and Parallel Classes

For an incidence structure `D` with `v` points, a *resolution* is a partition of the blocks into
classes `Cᵢ` each of which is a 1-design with `v` points and index `λ` (the *index* of the
resolution). A resolution with `λ = 1` is a *parallelism*, and its classes are *parallel
classes*. The functions `HasParallelism`, `AllParallelisms`, `HasParallelClass`,
`IsParallelClass` and `AllParallelClasses` require the structure to be uniform; `IsParallelism`
applies generally.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HasResolution(D)` | `true` iff `D` has a resolution; if `true`, also returns one resolution and its index. | Search. |
| `HasResolution(D, λ)` | `true` iff `D` has a resolution with index `λ`; if `true`, also returns one such resolution. | Search. |
| `AllResolutions(D)` | All resolutions of `D`. | Full backtrack search. |
| `AllResolutions(D, λ)` | All resolutions of `D` with index `λ`. (For all parallelisms `λ = 1` in a uniform design, prefer `AllParallelisms`.) | Full backtrack search. |
| `IsResolution(D, P)` | `true` iff the set `P` of blocks (or sets) is a resolution of `D`; if `true`, also returns the index. | Direct verification. |
| `HasParallelism(D : parameters)` | `true` iff the uniform structure `D` has a parallelism; if `true`, also returns one parallelism. Parameter `Al` (default `"Backtrack"`): `"Backtrack"` — a backtrack search, generally very efficient when `D` is parallelizable, comparing favourably with `"Clique"`. `"Clique"` — recommended when no parallelism is suspected; build graph `G₁` on blocks (adjacent iff disjoint), find all cliques of size `v/k` (each a parallel class), then build `G₂` on parallel classes (adjacent iff disjoint) and search for cliques of size `b/(v/k)` (each yielding a parallelism). | `"Backtrack"` (default) or `"Clique"` (communicated by Vladimir Tonchev). Clique may need considerable memory; try Backtrack first, then Clique if it does not complete in reasonable time. |
| `AllParallelisms(D)` | All parallelisms of the uniform structure `D`. Preferred to `AllResolutions(D, 1)` when `D` is uniform (uses the Clique algorithm rather than a full backtrack). | Clique-based (see `HasParallelism`). |
| `IsParallelism(D, P)` | `true` iff the set `P` of blocks (or sets) is a parallelism of `D`. | Direct verification. |
| `HasParallelClass(D)` | `true` iff the uniform structure `D` has a parallel class. | Search. |
| `IsParallelClass(D, B, C)` | `true` iff the uniform structure `D` has a parallel class containing blocks `B` and `C`; if so, one is returned. | Search. |
| `AllParallelClasses(D)` | All parallel classes of the uniform structure `D`. | Clique-finding in the disjointness graph. |

*Worked example:* H147E8 (an incidence structure on 6 points; `HasResolution`,
`HasResolution(D,2)`, `AllResolutions`, `HasParallelism`, `IsParallelism` with `PowerSet`,
`AllParallelClasses`).

---

## 147.10 Conversion Functions

| Intrinsic | Description |
|-----------|-------------|
| `IncidenceStructure(I)` | Given any-type structure `I`, return it as a "true" incidence structure (category `Inc`). |
| `NearLinearSpace(I)` | Given any-type structure `I`, return it as a near-linear space (category `IncNsp`), if possible. |
| `LinearSpace(I)` | Given any-type structure `I`, return it as a linear space (category `IncLsp`), if possible. |
| `Design(I, t)` | Given any-type structure `I`, return it as a `t`-design (category `Dsgn`), if possible. |

*Worked example:* H147E9 (`IsDesign(I,1)` returns `true 3`; `Design(I,1)` gives a `1-(8,4,3)`
design; `IsSteiner`, `IsNearLinearSpace`).

---

## 147.11 Identity and Isomorphism

Incidence structures `D₁ = (P₁, B₁, I₁)` and `D₂ = (P₂, B₂, I₂)` are *identical* if `P₁ = P₂`,
`B₁ = B₂` and `I₁ = I₂`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `D eq E` | `true` if `D` and `E` are identical, else `false`. | Componentwise equality. |
| `D ne E` | `true` if `D` and `E` are not identical, else `false`. | — |
| `IsIsomorphic(D, E : parameters)` | `true` if `D` and `E` are isomorphic, else `false`; if isomorphic, an isomorphism is returned as the second value. Parameter `AutomorphismGroups` (default `"None"`; values `"Both"`, `"Left"`, `"Right"`, `"None"`) specifies which of the left/right automorphism groups to construct first — in difficult cases this may significantly speed up the test. | Computes none/one/both automorphism groups (per `AutomorphismGroups`) to accelerate the isomorphism test. |

---

## 147.12 The Automorphism Group of an Incidence Structure

The automorphism group `A` of `D` is always presented as a permutation group `G` acting on the
*standard support*: the point set when `D` is simple, or the disjoint union of the point set
with the block set when `D` has repeated blocks (using a complicated internal support directly
would yield unreadable permutations). `G` does not act directly on `D`; instead **G-sets**
transfer the action of `G` to associated sets — the two most important being the action on the
point set and on the block set, returned by the construction functions. The mapping-structure
`Aut(D)` is the *parent* of the automorphisms; `Aut(D)` creates a shell rather than the group,
and a transfer map converts a permutation of `G` into a mapping in `Aut(D)`.

### 147.12.1 Construction of Automorphism Groups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AutomorphismGroup(D)` | The automorphism group `G` of `D`. If `D` is simple, `G` acts on the standard support `{1,…,v}` (point `i` ↔ `i`); if not simple, on `{1,…,v+b}` (`1 ≤ i ≤ v` ↔ point `i`, `v+1 ≤ i ≤ v+b` ↔ block `i−v`). Returns: (i) `G`; (ii) a G-set `Y` for the action on points; (iii) a G-set `W` for the action on blocks; (iv) the `Aut` structure `S`; (v) a transfer map `t : G → S` (for `g ∈ G`, `t(g)` is the corresponding mapping of `D` into itself). | Permutation-group computation on the standard support. |
| `AutomorphismSubgroup(D)` | A cyclic subgroup `H` of `Aut(D)`, to terminate the search as soon as a non-trivial automorphism is found. Returns: (i) `H`; (ii) the `Aut` structure `S`; (iii) a transfer map `t : G → S`. | Search halting at the first non-trivial automorphism. |
| `AutomorphismGroupStabilizer(D, k)` | The subgroup `H` of `Aut(D)` stabilizing the first `k` base points of `G` — sometimes easier to compute than all of `G`. Returns: (i) `H`; (ii) the `Aut` structure `S`; (iii) a transfer map `t : G → S`. | Base-point stabilizer of the automorphism group. |
| `PointGroup(D)` | The automorphism group of `D` in its action on the point set, together with the points G-set. | Action of `Aut(D)` on points. |
| `BlockGroup(D)` | The automorphism group of `D` in its action on the block set. | Action of `Aut(D)` on blocks. |
| `Aut(D)` | The power structure `A` of all automorphisms of `D`, together with the transfer map `t : Sym(n) → A` (points of `Sym(n)` in one-to-one correspondence with the natural support for the automorphism group). | Shell structure (parent of the automorphisms). |

*Worked example:* H147E10 (a `3-(16,8,3)` Hadamard design via `HadamardRowDesign` on a tensor
product of Hadamard matrices; `AutomorphismGroup` of order 322560; `CompositionFactors`).

### 147.12.2 Action of Automorphisms

The action of `G` on `D` is obtained via the G-set mechanism. The two basic G-sets (action on
points and on blocks) are returned by `AutomorphismGroup` or may be built directly. Only a few
of the available functions are described here (see the G-sets section for the full list).

| Intrinsic | Description |
|-----------|-------------|
| `Image(g, Y, y)` | For `G` a subgroup of `Aut(D)`, `Y` a G-set, and `y ∈ Y` (or in a G-set derived from `Y`): the image of `y` under `G`. |
| `Orbit(G, Y, y)` | The orbit of `y` under `G`, for a G-set `Y` and `y` belonging to `Y` or a derived G-set. |
| `Orbits(G, Y)` | The orbits of the action of `G` on the G-set `Y`. |
| `Stabilizer(G, Y, y)` | The stabilizer of `y` in `G`, for a G-set `Y` and `y` belonging to `Y` or a derived G-set. |
| `Action(G, Y)` | The homomorphism `φ : G → L` where `L` gives the action of `G` on G-set `Y`. Returns (a) the natural homomorphism `φ`; (b) the induced group `L`; (c) the kernel of the action. |
| `ActionImage(G, Y)` | The permutation group `L` giving the action of `G` on the G-set `Y`. |
| `ActionKernel(G, Y)` | The kernel of the action of `G` on the G-set `Y`. |
| `IsPointTransitive(D)` | `true` iff the automorphism group of `D` acts transitively on the point set. |
| `IsBlockTransitive(D)` | `true` iff the automorphism group of `D` acts transitively on the block set. |

*Worked example:* H147E11 (`AutomorphismGroup` of `WittDesign(12)` of order 95040; `Image`,
`SylowSubgroup`, `Stabilizer` on G-sets, `IsPointTransitive`, `IsBlockTransitive`,
`ActionImage` of order 95040 on 132 blocks).

---

## 147.13 Incidence Structures, Graphs and Codes

| Intrinsic | Description |
|-----------|-------------|
| `IncidenceStructure(G)` | The incidence structure `D` corresponding to graph `G`, with blocks corresponding to the edges of `G`. |
| `PointGraph(D)` | The point graph `G` of `D`: same point set as `D`, with vertices `u`, `v` adjacent whenever some block of `D` contains both. |
| `BlockGraph(D)` | The block graph of `D`, i.e. the point graph of the dual of `D`. |
| `IncidenceGraph(D)` | The incidence graph of `D`: a bipartite graph on `P ∪ B`, with `p ∈ P` adjacent to `b ∈ B` whenever `p ∈ b`. |
| `LinearCode(D, K)` | For `D` with `v` points and finite field `K`: the length-`v` linear code `C` generated by the characteristic functions of the blocks of `D` as vectors of `K^(v)`. |

*Worked example:* H147E12 (`LinearCode(WittDesign(24), GF(2))` equals the extended Golay code
over `GF(2)`).

---

## 147.14 Automorphisms of Matrices

A matrix may be regarded as a design, the entry in a given row and column defining the incidence
of that row and column. The automorphism group of a matrix is the set of permutations of its
rows and columns that leave the matrix unchanged.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `M ^ x` | The action of a permutation `x` on a matrix `M` by permuting rows and columns. For `M` with `r` rows and `c` columns, `x` must have degree `r+c` and fix `R = {1..r}`; the action of `x` on `R` permutes the rows, the remainder permutes the columns. | Permutation action on rows/columns. |
| `AutomorphismGroup(M)` | The group of all permutations `x` with `Mˣ = M`. | Constructs a graph from `M` and applies the graph `AutomorphismGroup`. |
| `IsIsomorphic(M, N)` | Finds a permutation `x` with `Mˣ = N` if one exists; returns `true` and the permutation, else `false`. | Constructs graphs from `M` and `N` and applies the graph `IsIsomorphic`. |

*Worked example:* H147E13 (a matrix from a smooth Fano polytope via `VertexFacetHeightMatrix`;
`AutomorphismGroup(M)` of order 24, `Orbits`; dualize and test `IsIsomorphic(M, D)`).

---

## 147.15 Bibliography (canonical references)

| Key | Reference |
|-----|-----------|
| **[Hal86]** | Marshall Hall. *Combinatorial Theory.* New York: Wiley, 2nd edition, 1986. |

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Direct construction (lists/sets/matrices/codes) | `IncidenceStructure`, `NearLinearSpace`, `LinearSpace`, `Design` |
| Related-structure constructions | `Complement`, `Dual`, `Contraction`, `Residual`, `Simplify`, `Sum`, `Union`, `Restriction` |
| Witt / Mathieu designs | `WittDesign` |
| Difference sets and development **[Hal86]** | `DifferenceSet`, `SingerDifferenceSet`, `IsDifferenceSet`, `Development` |
| Balance / t-design testing (`"NoOrbits"` / `"Orbits"` / `"FastBalanceTest"`) | `Design`, `IsDesign`, `IsBalanced`, `IsSteiner`, `Covalence`, `IntersectionNumber`, `PascalTriangle` |
| Resolution backtrack search | `HasResolution`, `AllResolutions`, `IsResolution` |
| Parallelism (Backtrack / Clique, Tonchev) | `HasParallelism`, `AllParallelisms`, `HasParallelClass`, `AllParallelClasses`, `IsParallelism`, `IsParallelClass` |
| Permutation-group automorphisms (via G-sets) | `AutomorphismGroup`, `AutomorphismSubgroup`, `AutomorphismGroupStabilizer`, `PointGroup`, `BlockGroup`, `Aut`, `IsPointTransitive`, `IsBlockTransitive` |
| G-set action machinery | `Image`, `Orbit`, `Orbits`, `Stabilizer`, `Action`, `ActionImage`, `ActionKernel` |
| Isomorphism (optionally via automorphism groups) | `IsIsomorphic` (structures), `IsIsomorphic` (matrices) |
| Conversions to graphs / codes | `IncidenceStructure(G)`, `PointGraph`, `BlockGraph`, `IncidenceGraph`, `LinearCode` |
| Matrix-as-design automorphisms (graph reduction) | `AutomorphismGroup(M)`, `IsIsomorphic(M,N)`, `M ^ x` |
