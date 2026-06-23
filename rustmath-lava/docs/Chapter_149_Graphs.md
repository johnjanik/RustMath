# Chapter 149 — Graphs

**Handbook part:** XX — Combinatorics
**Handbook pages:** 4917–4997 (PDF pages 5055–5131)

---

## Scope and overview

This chapter documents Magma's facilities for **simple graphs and simple digraphs** — graphs in
which each edge joins two distinct vertices, with at most one edge between any fixed pair of
vertices. (Loops and multiple edges are not permitted; graphs with such features — multigraphs,
multidigraphs and networks — are covered in Chapter 150.) For historical reasons Magma *does*
allow loops in digraphs. There are five Magma graph object types: the undirected simple graph
`GrphUnd`, the directed simple graph `GrphDir`, the undirected multigraph `GrphMultUnd`, the
directed multigraph `GrphMultDir`, and the network `GrphNet`. Simple graphs are of type `Grph`;
multigraphs (including networks) are of type `GrphMult`.

**Two internal representations.** Magma represents a graph either as an **adjacency matrix** (the
*dense* representation) or as an **adjacency list** (the *sparse* representation). The dense form
is quadratic in the number of vertices; the sparse form is linear in the number of edges, allowing
much larger low-density graphs. Memory for any single object cannot exceed 2³² bytes, giving graph
order bounds of n ≤ 65535 (dense) and n ≤ 134217722 (sparse). Many functions work with either
representation and convert automatically and transparently when required; the sparse representation
is required for multigraphs, the triconnectivity tester, planarity testing, and flow/shortest-path
algorithms based on the adjacency list. The default for simple graphs/digraphs is the dense
representation unless `SparseRep := true` is given.

**Enriched types.** A constructed graph consists of three objects: the **vertex-set** `V`, the
**edge-set** `E`, and the graph `G` itself; `V` and `E` are returned as the second and third
results of all construction functions. Graphs may carry a *support set* (the labelling set for
vertices) and *vertex/edge decorations* (labels, capacities, weights), the latter convenient for
shortest-path and flow algorithms (fully documented in Chapter 150). Standard construction
functions respect the support and decorations of their input graph.

**Algorithmic content.** Beyond construction and elementary invariants, the chapter covers
connectedness, triconnectivity (Hopcroft–Tarjan **[HT73]**, with corrections from
Gutwenger–Mutzel **[GM01]**), maximum matching and vertex/edge connectivity (flow-based, Dinic and
push-relabel), distances/paths, colourings, cliques and independent sets (Bron–Kerbosch
**[BK73]** and Myrvold's dynamic-programming method **[WM]**, with Brélaz *dsatur* bounding
**[Bre79]**), planarity (Boyer–Myrvold **[BM01]**), the automorphism-group interface to McKay's
**nauty** **[McK81]**, symmetry/regularity tests, and graph databases / graph generation.

---

## 149.1 Introduction

Introductory prose only; no intrinsics. Defines simple graph / digraph, multigraph, network; the
five graph types; the dense vs. sparse representation distinction; and notes that standard
construction functions (Subsections 149.6.1, 149.6.2) respect the support and decorations of the
original graph from V2.11 onwards. References **[Eve79]** (general graph theory/algorithms),
**[TCR90]** (graph algorithms) and **[RAO93]** (flow problems).

---

## 149.2 Construction of Graphs and Digraphs

Any enumerated or indexed set `S` may be used as a graph's vertex-set; the constructor copies it,
converts to an indexed set, and flags its type as `GrphVertSet`.

### 149.2.1 Bounds on the Graph Order

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `GraphSizeInBytes(n, m : parameters)` | Memory (in bytes) required for a graph of order `n` and size `m`. Parameters: `IsDigraph` (default `false`), `SparseRep` (default `false`). By default assumes an undirected dense graph. | Direct size formula; dense is quadratic in `n`, sparse linear in `m`. |

*Worked example: H149E1 (dense representation infeasible for n > 65535; sparse representation of large orders).*

### 149.2.2 Construction of a General Graph

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Graph< n \| edges : parameters >` / `Graph< S \| edges : parameters >` | Construct the graph `G` with vertex-set `V = {@ v₁,…,vₙ @}` (vᵢ = i in the first form, or the i-th element of enumerated/indexed set `S`) and edge-set from the list `edges`. Returns three values: `G`, the vertex-set `V`, the edge-set `E`. Parameter `SparseRep` (default `false`) gives a sparse representation. The items of `edges` may be: (a) a pair `{vᵢ,vⱼ}`; (b) a tuple `<vᵢ, Nᵢ>` with `Nᵢ` a set of neighbours of `vᵢ`; (c) a sequence `[N₁,…,Nₙ]` of neighbour-sets; (d) an edge `e` of any graph/digraph/multigraph/network (directed edges become undirected); (e) an edge-set `E`; (f) a graph/digraph/etc. `H` (all its edges added); (g) an n×n symmetric (0,1)-matrix `A` (interpreted as an adjacency matrix); (h) a set of any of (a)/(b)/edges/graphs; (j) a sequence of tuples `<vᵢ,Nᵢ>`. | Reads the edge specification and builds the adjacency structure. |
| `IncidenceGraph(A)` | For an n×m (0,1)-matrix `A` with exactly two 1s in each column: the graph `G` of order `n` and size `m` having `A` as its incidence matrix (rows ↔ vertices, columns ↔ edges). | Incidence-matrix interpretation. |

*Worked examples: H149E2 (Petersen graph constructed by edges, by neighbour-lists, sparsely, and by adjacency matrix); H149E3 (Tutte's 8-cage via PΓL(2,9) and the Lorimer construction **[Lor89]**).*

### 149.2.3 Construction of a General Digraph

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Digraph< n \| edges : parameters >` / `Digraph< S \| edges : parameters >` | Construct the digraph `G` with vertex-set `V` (as for `Graph<>`) and directed edge-set from `edges`. Returns `G`, `V`, `E`. Parameter `SparseRep` (default `false`). The items of `edges` may be: (a) a pair `[vᵢ,vⱼ]` (directed edge vᵢ→vⱼ); (b) a tuple `<vᵢ,Nᵢ>` with `Nᵢ` a set of out-neighbours; (c) a sequence of neighbour-sets; (d) an edge `e` (an undirected edge {u,v} adds both [u,v] and [v,u]); (e) an edge-set; (f) a graph/digraph/etc. `H`; (g) an n×n (0,1)-matrix `A` (adjacency matrix of a digraph); (h) a set of pairs/tuples/edges/graphs; (j) a sequence of tuples `<vᵢ,Nᵢ>`. | Reads the directed-edge specification and builds the structure. |
| `IncidenceDigraph(A)` | For an n×m matrix `A` with at most one +1 and at most one −1 per column: the digraph of order `n`, size `m` with `A` as its incidence matrix (+1 in row i, −1 in row j ⇒ edge vᵢvⱼ; single non-zero in row i ⇒ loop vᵢvᵢ). | Incidence-matrix interpretation. |

*Worked example: H149E4 (a 5-vertex digraph built from edges, sparsely, and from its adjacency matrix; `IsIsomorphic` check).*

### 149.2.4 Operations on the Support

The set `S` over which `G` is defined is the *support* of `G`.

| Intrinsic | Description |
|-----------|-------------|
| `Support(G)` / `Support(V)` | The indexed set used in constructing `G` (or the graph whose vertex-set is `V`), or `{@ 1,…,n @}` if none was given. |
| `ChangeSupport(G, S)` | For `G` of order `n` and an indexed set `S` of cardinality `n`: a new graph `H` structurally equal to `G` with support `S` and the same vertex/edge decorations. |
| `ChangeSupport(~G, S)` | Procedural version of the above. |
| `StandardGraph(G)` | The graph `H` equal to `G` but defined on the standard support, with the same decorations. |

*Worked example: H149E5 (the Odd Graph O₃ and its `StandardGraph`).*

### 149.2.5 Construction of a Standard Graph

Some of these create relatively sparse graphs and accept `SparseRep`.

| Intrinsic | Description |
|-----------|-------------|
| `BipartiteGraph(m, n)` | The complete bipartite graph `K_{m,n}`. |
| `CompleteGraph(n)` | The complete graph `Kₙ` on `n` vertices. |
| `KCubeGraph(n : parameters)` | The graph of the n-dimensional cube `Qₙ`. Parameter `SparseRep` (default `false`). |
| `MultipartiteGraph(Q)` | For a sequence `Q = [m₁,…,mᵣ]` of positive integers: the complete multipartite graph `K_{m₁,…,mᵣ}`. |
| `EmptyGraph(n : parameters)` | The graph on `n` vertices with no edges. Parameter `SparseRep`. |
| `NullGraph( : parameters)` | The graph with no vertices and no edges. Parameter `SparseRep`. |
| `PathGraph(n : parameters)` | The path graph on `n` vertices (vᵢ adj vⱼ iff j = i+1). Parameter `SparseRep`. |
| `PolygonGraph(n : parameters)` | For `n ≥ 3`: the polygon (cycle) graph on `n` vertices (vᵢ adj vⱼ iff j = i+1, or i = 1 and j = n). Parameter `SparseRep`. |
| `RandomGraph(n, r : parameters)` | A random graph on `n` vertices where each pair is adjacent with probability `r ∈ [0,1]`. Parameter `SparseRep`. |
| `RandomTree(n : parameters)` | A random tree on `n` vertices. Parameter `SparseRep`. |

### 149.2.6 Construction of a Standard Digraph

| Intrinsic | Description |
|-----------|-------------|
| `CompleteDigraph(n)` | The complete symmetric digraph on `n` vertices. |
| `EmptyDigraph(n : parameters)` | The null digraph on `n` vertices. Parameter `SparseRep`. |
| `RandomDigraph(n, r : parameters)` | A random digraph on `n` vertices where each ordered pair (u,v) is an edge with probability `r ∈ [0,1]`. Parameter `SparseRep`. |

*Worked example: H149E6 (`CompleteDigraph(5)` and a `RandomDigraph(5, 0.75)`).*

---

## 149.3 Graphs with a Sparse Representation

Prose only; no new intrinsics on the listing page. Reiterates the advantages of the sparse
(adjacency-list) representation: larger low-density graphs; linear-in-edges algorithms (planarity,
triconnectivity); required for multigraphs. Conversion between representations is automatic and
transparent; when conversion occurs the original representation is *not* deleted. The chapter lists
which function groups handle both representations natively versus those needing an adjacency-list
(149.5, 149.13.3/4/5, 149.21) or an adjacency-matrix (149.9, 149.14.2, 149.16, 149.18–149.20,
149.22, 149.23).

| Intrinsic | Description |
|-----------|-------------|
| `HasSparseRep(G)` | Whether `G` has a sparse representation. |
| `HasDenseRep(G)` | Whether `G` has a dense representation. |
| `HasSparseRepOnly(G)` | Whether `G` has only the sparse representation. |
| `HasDenseRepOnly(G)` | Whether `G` has only the dense representation. |
| `HasDenseAndSparseRep(G)` | Whether `G` has both representations. |

*Worked example: H149E7 (the four cases of representation conversion under `AutomorphismGroup` and `IsPlanar`).*

---

## 149.4 The Vertex-Set and Edge-Set of a Graph

A graph created by Magma consists of the vertex-set `V`, the edge-set `E`, and the graph `G`; `V`
and `E` are *enriched* sets (types in their own right) providing a convenient mechanism for
referring to vertices and edges.

### 149.4.1 Introduction

Prose only.

### 149.4.2 Creating Edges and Vertices

| Intrinsic | Description |
|-----------|-------------|
| `EdgeSet(G)` | The edge-set of `G`. |
| `Edges(G)` | A set `E` whose elements are the edges of `G` (an indexed set, not the edge-set). |
| `VertexSet(G)` | The vertex-set of `G`. |
| `Vertices(G)` | A set `V` whose elements are the vertices of `G` (an indexed set, not the vertex-set). |
| `V ! v` | For vertex-set `V` of `G` and an element `v` of the support of `V`: the corresponding vertex of `G`. |
| `V . i` | For vertex-set `V` and `1 ≤ i ≤ #V`: the vertex `vᵢ`. |
| `Index(v)` | For a vertex `v` of `G`: the index of `v` in the (indexed) vertex-set. |
| `E ! {u, v}` | For edge-set `E` of graph `G` and adjacent support elements `u, v`: the edge `uv`. |
| `E ! [u, v]` | For edge-set `E` of digraph `G` and adjacent support elements `u, v`: the directed edge `uv`. |
| `E . i` | For edge-set `E` and `1 ≤ i ≤ #E`: the i-th edge of `E`. |

*Worked example: H149E8 (vertices and edges of the Odd Graph O₃; coercion, `Index`, `Type`).*

### 149.4.3 Operations on Vertex-Sets and Edge-Sets

`S`, `T` may be a vertex-set or edge-set; `s` may be a vertex or edge. Vertex-sets and edge-sets
support all standard set operations.

| Intrinsic | Description |
|-----------|-------------|
| `#S` | Cardinality of the set `S`. |
| `s in S` / `s notin S` | Whether vertex/edge `s` lies (does not lie) in `S`. |
| `S subset T` / `S notsubset T` | Whether `S` is (is not) contained in `T`. |
| `S eq T` / `S ne T` | Whether vertex-sets/edge-sets `S` and `T` are (are not) equal. |
| `s eq t` / `s ne t` | Whether vertices/edges `s` and `t` are (are not) equal. |
| `ParentGraph(S)` | The graph `G` for which `S` is the vertex-set (edge-set). |
| `ParentGraph(s)` | The graph `G` for which `s` is a vertex (edge). |
| `Random(S)` | A random element of `S`. |
| `Representative(S)` / `Rep(S)` | Some element of `S`. |
| `for x in S do ... end for;` | `S` as the range in a `for`-statement. |
| `for random x in S do ... end for;` | `S` as the range in a `for random`-statement. |

### 149.4.4 Operations on Edges and Vertices

| Intrinsic | Description |
|-----------|-------------|
| `EndVertices(e)` | The two end-vertices of edge `e` (a set; a sequence if `G` is a digraph). |
| `InitialVertex(e)` | For an edge `e` from `u` to `v`: the vertex `u` (indicates traversal direction in the undirected case). |
| `TerminalVertex(e)` | For an edge `e` from `u` to `v`: the vertex `v`. |
| `IncidentEdges(u)` | For a vertex `u`: the set of all edges incident with `u` (incident into and from `u` if `G` is directed). |

---

## 149.5 Labelled, Capacitated and Weighted Graphs

Prose only; the decoration functions are fully documented in Section 150.4 of Chapter 150. A vertex
labelling is a partial map from `V` into a set `L`; an edge labelling a partial map from `E` into
`L`; a capacitated graph a partial map from `E` into ℤ⁺; a weighted graph a partial map from `E`
into any ring `R` with a total order. Capacities and weights are convenient for shortest-path and
flow algorithms.

---

## 149.6 Standard Constructions for Graphs

The two main ways to build a new graph from an old one are taking a subgraph and modifying the
original; a third is the quotient graph. For subgraphs and modifications the support set and
vertex/edge decorations are retained.

### 149.6.1 Subgraphs and Quotient Graphs

| Intrinsic | Description |
|-----------|-------------|
| `sub< G \| list >` | The subgraph `H` of `G`. Returns `H`, its vertex-set `V`, its edge-set `E`. Support and decorations are transferred. Items of `list` may be: (a) a vertex of `G` (the induced subgraph on those vertices); (b) an edge of `G` (subgraph with vertex-set `VertexSet(G)` and those edges); (c) a set of vertices or of edges. |
| `quo< G \| P >` | The quotient graph `Q` of `G` defined by a partition `P` of `V(G)` (given as a set of subsets). The cells `P₁,…,Pᵣ` become the vertices of `Q`; vᵢ adj vⱼ in `Q` iff some vertex of `Pᵢ` is joined in `G` to a vertex of `Pⱼ`. |

*Worked examples: H149E9 (subgraph of K₅; mapping subgraph vertices to/from the supergraph by coercion; K₆ minus a 1-factor); H149E10 (quotient of K₉ by a partition into three 3-sets).*

### 149.6.2 Incremental Construction of Graphs

Unless stated otherwise, each function returns the graph `G`, its vertex-set `V`, its edge-set `E`.

#### 149.6.2.1 Adding Vertices

| Intrinsic | Description |
|-----------|-------------|
| `G + n` | Add `n` new vertices to `G` (`n ≥ 0`). Decorations retained; support becomes standard. |
| `G +:= n` / `AddVertex(~G)` / `AddVertices(~G, n)` | Procedural version; `AddVertex` adds one vertex. |
| `AddVertex(~G, l)` | Add a new vertex with label `l` to `G`. Decorations retained; support becomes standard. |
| `AddVertices(~G, n, L)` | Add `n` new vertices with labels from the sequence `L`. Decorations retained; support becomes standard. |

#### 149.6.2.2 Removing Vertices

| Intrinsic | Description |
|-----------|-------------|
| `G - v` / `G - U` | Remove vertex `v` or set of vertices `U` from `G`. Support and decorations retained. |
| `G -:= v` / `G -:= U` / `RemoveVertex(~G, v)` / `RemoveVertices(~G, U)` | Procedural versions. |

#### 149.6.2.3 Adding Edges

| Intrinsic | Description |
|-----------|-------------|
| `G + { u, v }` / `G + [ u, v ]` | Add the edge described by the pair (set for undirected, sequence for directed). Returns the modified graph and the new edge. Support and decorations retained. |
| `G + { { u, v } }` / `G + { [ u, v ] }` | Add the set of edges described by these pairs. Support and decorations retained. |
| `G +:= { u, v }` / `G +:= [ u, v ]` / `G +:= { { u, v } }` / `G +:= { [ u, v ] }` | Procedural versions of the previous four. |
| `AddEdge(G, u, v)` | Add a new edge between vertices `u`, `v`. Returns the modified graph and the new edge. |
| `AddEdge(G, u, v, l)` | Add a new edge with label `l` between `u` and `v`. Returns the modified graph and the new edge. |
| `AddEdge(~G, u, v)` / `AddEdge(~G, u, v, l)` | Procedural versions. |
| `AddEdges(G, S)` | Add the edges in set `S` (each a set or sequence of two vertices). Support and decorations retained. |
| `AddEdges(G, S, L)` | Add the edges in sequence `S` with corresponding labels in sequence `L` (same length). |
| `AddEdges(~G, S)` / `AddEdges(~G, S, L)` | Procedural versions. |

#### 149.6.2.4 Removing Edges

| Intrinsic | Description |
|-----------|-------------|
| `G - e` / `G - { e }` | Remove edge `e` or the edges in set `S` from `G`. Support and decorations retained. |
| `G - { { u, v } }` / `G - { [u, v] }` | Remove the edges between the pairs `{u,v}` (or `[u,v]`) in the set `S`. Support and decorations retained. |
| `G -:= e` / `G -:= { e }` / `G -:= { { u, v } }` / `G -:= { [u, v] }` / `RemoveEdge(~G, e)` / `RemoveEdges(~G, S)` / `RemoveEdge(~G, u, v)` | Procedural versions of the previous functions. |

### 149.6.3 Constructing Complements, Line Graphs; Contraction, Switching

Unless stated otherwise, these apply to both graphs and digraphs and return `G`, `V`, `E`.

| Intrinsic | Description |
|-----------|-------------|
| `Complement(G)` | The complement of `G`. |
| `Contract(e)` | For an edge `e = {u,v}`: the graph obtained by identifying `u` and `v` and removing resulting multiple edges (and loops if undirected). Support and decorations retained. |
| `Contract(u, v)` | Identify vertices `u`, `v` (as above). |
| `Contract(S)` | Identify all vertices in the set `S` (as above). |
| `InsertVertex(e)` | Insert a new degree-2 vertex into edge `e`. The two replacement edges inherit `e`'s capacity and weight; they are unlabelled. Vertex labels and edge decorations retained; resulting graph has standard support. |
| `InsertVertex(T)` | Insert a degree-2 vertex into each edge of the set `T`. |
| `LineGraph(G)` | The line graph of the non-empty graph `G`. |
| `Switch(u)` | Construct `H` from undirected `G` with the same vertex/edge-sets except that the neighbours and non-neighbours of `u` are interchanged (Seidel switching at `u`). Support and vertex labels are *not* retained. |
| `Switch(S)` | Apply `Switch(u)` to each vertex of the set `S` in turn. Support and vertex labels are *not* retained. |

*Worked example: H149E11 (the Grötzsch graph via `CompleteGraph`, `InsertVertex`, `Union`).*

---

## 149.7 Unions and Products of Graphs

The support and decorations of the original graphs are *not* retained in any union.

| Intrinsic | Description |
|-----------|-------------|
| `Union(G, H)` / `G join H` | For graphs with disjoint vertex-sets: their union (vertex-set V(G)∪V(H), edge-set E(G)∪E(H)). |
| `EdgeUnion(G, H)` | For graphs with the same number of vertices: the edge union `K` identifying the i-th vertices, with `u,v` adjacent iff adjacent in `G` or in `H`. |
| `CompleteUnion(G, H)` | For disjoint vertex-sets: `Union(G,H)` together with all edges uv for u ∈ V(G), v ∈ V(H). |
| `CartesianProduct(G, H)` | The product `K = G × H`; vertex-set V(G)×V(H); (u₁,u₂) adj (v₁,v₂) iff (u₁=v₁ and u₂ adj v₂) or (u₂=v₂ and u₁ adj v₁). |
| `LexProduct(G, H)` | The lexicographic product; (u₁,u₂) adj (v₁,v₂) iff u₁ adj v₁, or (u₁=v₁ and u₂ adj v₂). |
| `TensorProduct(G, H)` | The tensor product; (u₁,u₂) adj (v₁,v₂) iff u₁ adj v₁ and u₂ adj v₂. |
| `G ^ n` | The n-th power of `G`: same vertex-set, `u` adj `v` iff the distance between them in `G` is ≤ `n`. |

---

## 149.8 Converting between Graphs and Digraphs

`UnderlyingGraph` and `UnderlyingDigraph` also give a copy of `G` without its support and decorations.

| Intrinsic | Description |
|-----------|-------------|
| `OrientatedGraph(G)` | A digraph `D` with the same vertex-set as `G`; each edge {u,v} is directed from the lower- to the higher-numbered vertex. Support and decorations not retained. |
| `UnderlyingGraph(D)` | The underlying (undirected) graph of `D`: u adj v iff there is an edge u→v or v→u in `D`. Support and decorations not retained. |
| `UnderlyingDigraph(G)` | The underlying digraph of `G`: if `G` undirected, each edge {u,v} gives both u→v and v→u. Support and decorations not retained. |

---

## 149.9 Construction from Groups, Codes and Designs

### 149.9.1 Graphs Constructed from Groups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CayleyGraph(A)` / `CayleyGraph(A : parameter)` / `UnlabelledCayleyGraph(A)` | For a finite group `A` on generating set `X`: the Cayley graph `C`; vertices ↔ elements of `A`, with u adj v iff some x ∈ X has u∗x = v. Parameter `Labelled` (default `true`); when `true`, vertices are labelled by elements of `A` and directed edges by the generator `x`. | Definition via right multiplication by generators. |
| `SchreierGraph(A, B)` / `UnlabelledSchreierGraph(A, B)` | For a finite group `A` on generating set `X` and a subgroup `B`: the Schreier coset graph `S` for `A` over `B` relative to `X`; vertices ↔ cosets of `B`, u adj v iff some x ∈ X has u∗x = v. Labelled and unlabelled versions. | Coset action of the generators. |
| `OrbitalGraph(P, u, T)` | For a transitive permutation group `P` on Ω = {1,…,n}, `u ∈ Ω`, and `T = {t₁,…,tᵣ} ⊆ Ω`: the underlying graph of the digraph for the union of `P`-orbits containing the pairs (u,t₁),…,(u,tᵣ). If `T` is a self-paired orbit of the stabiliser of `u`, this is the orbital graph of the associated orbit. | Union of point-pair orbits under `P`. |
| `ClosureGraph(P, G)` | For a permutation group `P` on {1,…,n} and a graph (digraph) `G` on v₁,…,vₙ: add the minimum number of edges to `G` to produce a graph (digraph) `H` left invariant by `P`. | Orbit closure of the edge-set under `P`. |
| `PaleyGraph(q)` | The Paley graph of 𝔽_q (`q` a prime power ≡ 1 mod 4); vertices ↔ 𝔽_q, distinct elements adjacent when their difference is a square. | Quadratic-residue adjacency. |
| `PaleyTournament(q)` | The Paley tournament of 𝔽_q (`q` a prime power ≡ 3 mod 4); edge u→v when u ≠ v and v − u is a square. | Quadratic-residue adjacency. |

### 149.9.2 Graphs Constructed from Designs

| Intrinsic | Description |
|-----------|-------------|
| `IncidenceGraph(D)` | For an incidence structure `D = (X, B)`: the incidence graph `G` with vertices X ∪ B; x ∈ X adjacent to B ∈ B iff x ∈ B (no X–X or B–B edges). |
| `PointGraph(D)` | For `D = (X, B)`: the point graph `G` with vertex-set X; x adj y iff some block contains both. |
| `BlockGraph(D)` | The block graph of `D`: the point graph of the dual of `D`. |
| `IncidenceGraph(P)` | For a plane `P` with point-set V and line-set L: the incidence graph with vertices V ∪ L; v ∈ V adjacent to a ∈ L iff v lies on a. |
| `PointGraph(P)` | For a plane `P`: the point graph with vertex-set V; u,v adj iff some line contains both. |
| `LineGraph(P)` | For a plane `P`: the line graph with vertex-set L; a,b adj iff some point lies on both. |
| `HadamardGraph(H : parameters)` | The graph of the ±1 matrix `H` per B. D. McKay's "Hadamard equivalence via graph isomorphism". Parameter `Labels` (default `false`); when `true`, row-vertices labelled "row" (those given loops in McKay's paper) and column-vertices "col". |

### 149.9.3 Miscellaneous Graph Constructions

| Intrinsic | Description |
|-----------|-------------|
| `Converse(G)` | The converse `H` of digraph `G`: if [u,v] is an edge of `G` then [v,u] is an edge of `H`. |
| `OddGraph(n)` | The n-th odd graph: vertices are (n−1)-subsets of a (2n−1)-set, adjacent iff disjoint. |
| `TriangularGraph(n)` | The n-th triangular graph: vertices are 2-subsets of an n-set, adjacent iff unequal and not disjoint. |
| `SquareLatticeGraph(n)` | The n-th square lattice graph: the Cartesian product of the n-th complete graph with itself. |
| `ClebschGraph()` / `ShrikhandeGraph()` / `GewirtzGraph()` | The named (strongly regular) graph. |
| `ChangGraphs()` | A sequence of the three Chang graphs. |

---

## 149.10 Elementary Invariants of a Graph

| Intrinsic | Description |
|-----------|-------------|
| `Order(G)` / `NumberOfVertices(G)` | The number of vertices of `G`. |
| `Size(G)` / `NumberOfEdges(G)` | The number of edges of `G`. |
| `CharacteristicPolynomial(G)` | The characteristic polynomial (over ℤ) of the adjacency matrix of `G`. |
| `Spectrum(G)` | The spectrum of `G`: the roots of the characteristic polynomial, returned as a set of tuples (root, multiplicity). |

---

## 149.11 Elementary Graph Predicates

Defines (in prose) when two graphs are equal and when `H` is a subgraph of `G`, taking support,
labels and capacities into account.

| Intrinsic | Description |
|-----------|-------------|
| `u adj v` | For vertices of the same graph: whether `u`,`v` are adjacent (an edge u→v if directed). |
| `e adj f` | For edges of the same graph: whether they share a common vertex (in the directed case, whether the terminal vertex of one is the initial vertex of the other). |
| `u notadj v` / `e notadj f` | Negation of `adj` for vertices / edges. |
| `u in e` | Whether vertex `u` is an end-vertex of edge `e`. |
| `u notin e` | Negation of the `in` predicate for a vertex w.r.t. an edge. |
| `G eq H` | Whether `G` and `H` are identical (same structure, support, labels, and — where applicable — capacities). |
| `IsSubgraph(G, H)` | Whether `H` is a subgraph of `G` (structural subgraph plus matching support, labels and ≥ capacities). |
| `IsBipartite(G)` | Whether `G` is bipartite. |
| `IsComplete(G)` | Whether `G` on `n` vertices is the complete graph. |
| `IsEulerian(G)` | Whether `G` is Eulerian (all vertices of even degree; for a digraph, OutDegree(v) = InDegree(v) for all v). |
| `IsForest(G)` | Whether `G` is a forest (acyclic). |
| `IsEmpty(G)` | Whether `G` is empty (edge-set empty). |
| `IsNull(G)` | Whether `G` is null (vertex-set empty). |
| `IsPath(G)` | Whether `G` is a path graph. |
| `IsPolygon(G)` | Whether `G` is a polygon graph. |
| `IsRegular(G)` | Whether `G` is regular. |
| `IsTree(G)` | Whether `G` is a tree. |

---

## 149.12 Adjacency and Degree

### 149.12.1 Adjacency and Degree Functions for a Graph

| Intrinsic | Description |
|-----------|-------------|
| `Degree(u)` | The degree of vertex `u` (number of incident edges). |
| `Alldeg(G, n)` | The set of all vertices of `G` of degree exactly `n`. |
| `MaximumDegree(G)` / `Maxdeg(G)` | The maximum degree; returns the degree and a vertex achieving it. |
| `MinimumDegree(G)` / `Mindeg(G)` | The minimum degree; returns the degree and a vertex achieving it. |
| `DegreeSequence(G)` | For `G` of maximum degree `r`: a sequence `D` of length `r+1` with `D[i]` = number of vertices of degree `i−1`. |
| `Valence(G)` | For a regular `G`: the valence (common degree). |
| `Neighbours(u)` / `Neighbors(u)` | The set of vertices adjacent to `u`. |
| `IncidentEdges(u)` | The set of edges incident with `u`. |
| `Bipartition(G)` | For a bipartite `G`: its two partite sets, as a pair of subsets of V(G). |
| `MinimumDominatingSet(G)` | A minimum dominating set of `G` (a smallest set `S` such that S together with the vertices adjacent to `S` is V(G)). Algorithm: backtrack search **[Chr75]** (p. 41). |

### 149.12.2 Adjacency and Degree Functions for a Digraph

| Intrinsic | Description |
|-----------|-------------|
| `InDegree(u)` | The number of edges directed into vertex `u`. |
| `OutDegree(u)` | The number of edges of the form uv (out of `u`). |
| `Degree(u)` | The total degree of `u` (in-degree + out-degree). |
| `Alldeg(G, n)` | The set of all vertices of total degree `n`. |
| `MaximumInDegree(G)` / `Maxindeg(G)` | The maximum indegree; returns the value and the first vertex achieving it. |
| `MaximumOutDegree(G)` / `Maxoutdeg(G)` | The maximum outdegree; value and first achieving vertex. |
| `MinimumInDegree(G)` / `Minindeg(G)` | The minimum indegree; value and first achieving vertex. |
| `MinimumOutDegree(G)` / `Minoutdeg(G)` | The minimum outdegree; value and first achieving vertex. |
| `MaximumDegree(G)` / `Maxdeg(G)` | The maximum total degree; value and first achieving vertex. |
| `MinimumDegree(G)` / `Mindeg(G)` | The minimum total degree; value and first achieving vertex. |
| `DegreeSequence(G)` | For digraph `G` of maximum degree `r`: a sequence `D` of length `r+1`, `D[i]` = number of vertices of degree `i−1`. |
| `InNeighbours(u)` / `InNeighbors(u)` | The set of starting points of all edges directed into `u`. |
| `OutNeighbours(u)` / `OutNeighbors(u)` | The set of end-points of all edges directed from `u`. |
| `IncidentEdges(u)` | The set of all edges incident with `u`. |

---

## 149.13 Connectedness

### 149.13.1 Connectedness in a Graph

| Intrinsic | Description |
|-----------|-------------|
| `IsConnected(G)` | Whether `G` is connected. |
| `Components(G)` | The connected components of `G`, as a sequence of subsets of V(G). |
| `Component(u)` | The subgraph corresponding to the connected component containing `u`. Support and decorations not retained. |
| `IsSeparable(G)` | Whether `G` is connected and has at least one cut vertex. |
| `IsBiconnected(G)` | Whether `G` is biconnected. |
| `CutVertices(G)` | The set of cut vertices of the connected graph `G`. |
| `Bicomponents(G)` | The biconnected components of `G`, as a sequence of subsets of V(G). `G` may be disconnected. |

### 149.13.2 Connectedness in a Digraph

| Intrinsic | Description |
|-----------|-------------|
| `IsStronglyConnected(G)` | Whether the digraph `G` is strongly connected. |
| `IsWeaklyConnected(G)` | Whether the digraph `G` is weakly connected. |
| `StronglyConnectedComponents(G)` | The strongly connected components of `G`, as a sequence of subsets of V(G). |
| `Component(u)` | The subgraph corresponding to the connected component of the digraph containing `u`. Support and decorations not retained. |

### 149.13.3 Graph Triconnectivity

The linear-time triconnectivity algorithm of Hopcroft and Tarjan **[HT73]** is implemented with
corrections of Magma's own and from Gutwenger and Mutzel **[GM01]**. Requires a sparse
representation; the input may be disconnected or have cut vertices. The algorithm splits the graph
into *split components* (which become triconnected on adding the separation-pair edges). Applies
only to undirected graphs.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsTriconnected(G)` | Whether `G` is triconnected. | Hopcroft–Tarjan **[HT73, GM01]**. |
| `Splitcomponents(G)` | The split components of `G`, as a sequence of subsets of V(G) (may be disconnected). Second return: the cut vertices and separation pairs. | Hopcroft–Tarjan **[HT73, GM01]**. |
| `SeparationVertices(G)` | The cut vertices and/or separation pairs of `G`, as a sequence of vertices or vertex-pairs. Second return: `G`'s split components. | Hopcroft–Tarjan **[HT73, GM01]**. |

*Worked example: H149E12 (split components and separation pairs of an 11-vertex graph; making a split component triconnected by adding a separation-pair edge).*

### 149.13.4 Maximum Matching in Bipartite Graphs

The maximum matching algorithm is flow-based; two maximum-flow algorithms are implemented — the
Dinic algorithm and a push-relabel algorithm (the latter almost always the more efficient,
outperformed by Dinic only for very sparse graphs). Full discussion in Section 151.4 of Chapter 150.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `MaximumMatching(G)` | A maximum matching in the bipartite graph `G`, returned as a sequence of edges. Parameter `Al` (`MonStgElt`, default `"PushRelabel"`; alternative `"Dinic"`) selects the underlying max-flow algorithm. | Flow-based: push-relabel or Dinic. |

*Worked example: H149E13 (random bipartite graph, `IsBipartite`, `Bipartition`, `MaximumMatching`).*

### 149.13.5 General Vertex and Edge Connectivity in Graphs and Digraphs

The vertex- and edge-connectivity algorithms are flow-based (Dinic and push-relabel; Dinic
outperforms push-relabel only for very sparse graphs, notably in edge connectivity). For the flow
algorithms see Section 151.4 of Chapter 150; for the connectivity implementation see **[Eve79]**.
These apply to both undirected and directed graphs.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `VertexSeparator(G)` | A smallest set of vertices `S` such that every (directed, if `G` is directed) path between any two vertices passes through `S`; returned as a sequence of vertices. Parameter `Al` (default `"PushRelabel"`; alternative `"Dinic"`). | Flow-based **[Eve79]**. |
| `VertexConnectivity(G)` | The vertex connectivity of `G` (size of a minimum vertex separator); also returns a vertex separator. Parameter `Al`. | Flow-based **[Eve79]**. |
| `IsKVertexConnected(G, k)` | Whether `G`'s vertex connectivity is at least `k`. Parameter `Al`. | Flow-based **[Eve79]**. |
| `EdgeSeparator(G)` | A smallest set of edges `T` such that every (directed) path between any two vertices passes through `T`; returned as a sequence of edges. Parameter `Al`. | Flow-based **[Eve79]**. |
| `EdgeConnectivity(G)` | The edge connectivity of `G` (size of a minimum edge separator); also returns an edge separator. Parameter `Al`. | Flow-based **[Eve79]**. |
| `IsKEdgeConnected(G, k)` | Whether `G`'s edge connectivity is at least `k`. Parameter `Al`. | Flow-based **[Eve79]**. |

*Worked example: H149E14 (vertex and edge connectivity of a small graph; `IsKVertexConnected`/`IsKEdgeConnected` checks across the threshold).*

---

## 149.14 Distances, Paths and Circuits in a Graph

The distance functions apply to graphs and digraphs.

### 149.14.1 Distances, Paths and Circuits in a Possibly Weighted Graph

These take edge weights into account; if unweighted, distance is the shortest-path length. For full
details on weighted-graph distance/path functions see Section 150.12 of Chapter 150.

| Intrinsic | Description |
|-----------|-------------|
| `Reachable(u, v)` | Whether `u` and `v` lie in the same component of `G`. |
| `Distance(u, v)` | The length (or total weight, if weighted) of a shortest path from `u` to `v`; error if no path exists. |
| `Geodesic(u, v)` | A sequence of vertices forming a shortest path from `u` to `v`; error if no path exists. |

### 149.14.2 Distances, Paths and Circuits in a Non-Weighted Graph

These ignore edge weights; distance is the usual shortest-path length.

| Intrinsic | Description |
|-----------|-------------|
| `Diameter(G)` | The length of the longest shortest-path in `G` (i.e. the diameter); `−1` if `G` is not connected. (Uses the automorphism group — see 149.22.3.) |
| `DiameterPath(G)` | A sequence of vertices defining a longest shortest-path if connected, else the empty sequence. (Uses the automorphism group.) |
| `Ball(u, n)` | The set of vertices at distance `≤ n` from `u`. |
| `Sphere(u, n)` | The set of vertices at distance exactly `n` from `u`. |
| `DistancePartition(u)` | The partition P₀ ∪ P₁ ∪ … ∪ P_d of V(G) where Pᵢ is the set of vertices at distance `i` from `u` (vertices not connected to `u` form the last cell). |
| `IsEquitable(G, P)` | Whether the partition `P` of V(G) is equitable. |
| `EquitablePartition(P, G)` | The coarsest equitable partition of V(G) that refines `P`. |
| `Girth(G)` | The girth (length of a shortest cycle). (Uses the automorphism group.) |
| `GirthCycle(G)` | A cycle of shortest length. (Uses the automorphism group.) |

---

## 149.15 Maximum Flow, Minimum Cut, and Shortest Paths

Prose only. Whenever edges carry capacities, maximum flow and minimum cut can be computed; whenever
they carry weights, shortest paths can be found (defaulting to capacity/weight one if unassigned).
The flow-based and shortest-paths functionality is fully documented in Chapter 150, Sections 151.4
and 150.12.

---

## 149.16 Matrices and Vector Spaces Associated with a Graph or Digraph

| Intrinsic | Description |
|-----------|-------------|
| `AdjacencyMatrix(G)` | The adjacency matrix of the (p,q) graph `G` as an element of the matrix ring M_p(ℤ). |
| `DistanceMatrix(G)` | The distance matrix `A` of `G` (M_p(ℤ)); the (i,j) entry is the distance between vᵢ and vⱼ. |
| `IncidenceMatrix(G)` | The incidence matrix `M` of `G` (matrix bimodule M^{p×q}(ℤ)). For a graph, M(i,j)=1 if vᵢ lies on edge eⱼ. For a digraph, M(i,j)=1 if vᵢ is the initial vertex of eⱼ, −1 if the final vertex (a loop entry may be ±1). |
| `IntersectionMatrix(G, P)` | For an ordered equitable partition `P = P₁ ∪ … ∪ Pᵣ` of V(G): the intersection matrix `T`, with T[i,j] = number of vertices of Pⱼ adjacent to a vertex of Pᵢ. |

---

## 149.17 Spanning Trees of a Graph or Digraph

| Intrinsic | Description |
|-----------|-------------|
| `SpanningTree(G)` | A spanning tree for the connected (undirected) graph `G`, rooted at an arbitrary vertex, returned as a subgraph. Support and decorations not retained. |
| `SpanningForest(G)` | A spanning forest for `G`, returned as a subgraph. Support and decorations not retained. |
| `BreadthFirstSearchTree(u)` / `BFSTree(u)` | A breadth-first search tree of `G` rooted at `u`, returned as a subgraph. Support and decorations not retained; `G` may be disconnected. |
| `DepthFirstSearchTree(u)` / `DFSTree(u)` | A depth-first search tree `T` of `G` rooted at `u`, returned as a subgraph. The fourth return value gives, for each vertex, its tree (visit) order; non-tree vertices receive orders from Order(T)+1 to Order(G). Support and decorations not retained. |

---

## 149.18 Directed Trees

The graph is assumed to be a directed (rooted) tree: a tree with a root vertex and all edges
directed away from it.

| Intrinsic | Description |
|-----------|-------------|
| `IsRootedTree(G)` | Whether the digraph `G` is a tree with a vertex `v` from which all edges are directed away; returns the root `v` as a second value. |
| `Root(G)` | The root vertex of a rooted tree. |
| `IsRoot(v)` | Whether the graph containing `v` is a rooted tree with `v` as root. |
| `RootSide(v)` | The unique neighbour of `v` closer to the root (in a rooted tree); `v` itself if `v` is the root. |
| `VertexPath(u, v)` | A sequence of vertices forming a path in the directed tree from `u` to `v` (first tracing back to a common ancestor, then following edge directions to `v`). |
| `BranchVertexPath(u, v)` | The sequence of vertices on the vertex path from `u` to `v` that have valency at least 3. |

---

## 149.19 Colourings

These functions are not applicable to digraphs.

| Intrinsic | Description |
|-----------|-------------|
| `ChromaticNumber(G)` | The chromatic number of `G` (minimum colours for a proper vertex colouring). |
| `OptimalVertexColouring(G)` | An optimal vertex colouring of `G` as a sequence of sets of vertices (each set one colour class). |
| `ChromaticIndex(G)` | The chromatic index of `G` (minimum colours for a proper edge colouring). |
| `OptimalEdgeColouring(G)` | An optimal edge colouring of `G` as a sequence of sets of edges. |
| `ChromaticPolynomial(G)` | The chromatic polynomial p_G(x) of `G` (number of colourings with colours 1,…,x). |

*Worked example: H149E15 (`ChromaticNumber`, `OptimalVertexColouring`, `ChromaticIndex`, `OptimalEdgeColouring`, `ChromaticPolynomial` of a 5-vertex graph).*

---

## 149.20 Cliques, Independent Sets

These functions are not applicable to digraphs. Two algorithms are used:

- **BranchAndBound** — the branch-and-bound algorithm of Bron and Kerbosch **[BK73]**, adapted to
  find cliques of specific size; some pruning heuristics are built in.
- **Dynamic** — finds a clique of size exactly `k` (not necessarily maximal) by recursion and
  dynamic programming, due to Wendy Myrvold **[WM]**.

In general *BranchAndBound* does better for maximum cliques, but *Dynamic* outperforms it on large
(> 400 vertices) high-density (> 0.5%) random graphs.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HasClique(G, k)` | Whether `G` has a maximal clique of size `k`; if so, returns such a clique. | BranchAndBound **[BK73]**. |
| `HasClique(G, k, m : parameters)` | If `m` is `true`, tests for a maximal clique of size `k`; if `m` is `false`, for any clique of size `k`. Returns a clique if found. Parameter `Al` (default `"BranchAndBound"`; alternative `"Dynamic"`), ignored when `m` is `true`. | **[BK73]** / **[WM]**. |
| `HasClique(G, k, m, f : parameters)` | As above, with flag `f`: when `m` is `true`, `f = 0` tests size exactly `k`, `f = 1` size ≥ k, `f = −1` size ≤ k. When `m` is `false`, tests for a (not necessarily maximal) clique of size `k` and `f` is ignored. Parameter `Al`. | **[BK73]** / **[WM]**. |
| `MaximumClique(G : parameters)` | A maximum clique, returned as a set of vertices. Parameter `Al` (default `"BranchAndBound"`); with `"Dynamic"`, a *dsatur* colouring (Brélaz **[Bre79]**) gives a lower bound `l`, then the largest clique of size `k ≥ l` is found by Dynamic. | BranchAndBound **[BK73]**, or *dsatur* bound **[Bre79]** + Dynamic **[WM]**. |
| `CliqueNumber(G : parameters)` | The size of a maximum clique. Parameter `Al` (default `"BranchAndBound"`; `"Dynamic"` as in `MaximumClique`). | **[BK73]** / **[Bre79, WM]**. |
| `AllCliques(G : parameters)` | All maximal cliques as a sequence of sets of vertices. Parameter `Limit` (default 0; if positive, return that many). | BranchAndBound **[BK73]**. |
| `AllCliques(G, k : parameters)` | All maximal cliques of size `k`. Parameter `Limit` (default 0). | BranchAndBound **[BK73]**. |
| `AllCliques(G, k, m : parameters)` | If `m` is `true`, all maximal cliques of size `k`; if `false`, all (not necessarily maximal) cliques of size `k`. Parameters `Limit` (default 0) and `Al` (default `"BranchAndBound"`, ignored when `m` is `true`). Limit does not apply when `m` is `false` and `Al` is `"Dynamic"`. | **[BK73]** / **[WM]**. |
| `MaximumIndependentSet(G : parameters)` | A maximum independent set, as a set of vertices (a maximum clique in the complement). Parameter `Al` (default `"BranchAndBound"`; alternative `"Dynamic"`). | Via maximum clique of the complement. |
| `IndependenceNumber(G : parameters)` | The size of a maximum independent set. Parameter `Al` (default `"BranchAndBound"`). | Via clique number of the complement. |

*Worked example: H149E16 (`HasClique`, `MaximumClique`, `AllCliques` with the various flags on a 9-vertex graph).*

---

## 149.21 Planar Graphs

A linear-time algorithm of Boyer and Myrvold **[BM01]** tests planarity and, if non-planar,
isolates a Kuratowski subgraph; it requires a sparse representation. The idea is a depth-first
search followed by embedding back edges by post-order tree traversal so they do not cross.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsPlanar(G)` | Whether the undirected graph `G` is planar (may be disconnected); if non-planar, returns a Kuratowski subgraph (homeomorphic to K₅ or K₃,₃). Support and decorations not retained in that subgraph. | Boyer–Myrvold **[BM01]**. |
| `Obstruction(G)` | A Kuratowski obstruction if `G` is non-planar, else the empty graph. Support and decorations not transferred. | Boyer–Myrvold **[BM01]**. |
| `IsHomeomorphic(G : parameters)` | Whether `G` is homeomorphic to K₅ or K₃,₃. Parameter `Graph` (`MonStgElt`, no default) must be `"K5"` or `"K33"`. | Homeomorphism test. |
| `Faces(G)` | The faces of planar `G` as sequences of bordering edges; an isolated vertex `v` gives the face `[v]`. | Planar embedding. |
| `Face(u, v)` | The face bordered by the directed edge `[u,v]` as an ordered list of edges (assuming a clockwise plane orientation, neighbours taken anti-clockwise). | Planar embedding. |
| `Face(e)` | For edge `e = {u,v}`: `Face(u,v)`. | Planar embedding. |
| `NFaces(G)` / `NumberOfFaces(G)` | The number of faces of planar `G` (an isolated vertex counts for one face). | Planar embedding. |
| `Embedding(G)` | The planar embedding as a sequence `S`, `S[i]` a sequence of edges incident from vertex `i`. | Planar embedding. |
| `Embedding(v)` | The ordered list of edges (clockwise) incident from vertex `v`. | Planar embedding. |
| `PlanarDual(G)` | The dual `G'` of planar `G`; vertices of `G'` numbered as the faces returned by `Faces(G)`. | Planar embedding. |

*Worked example: H149E17 (a disconnected planar graph, `Faces`/`Embedding`; a non-planar K₃,₃ obstruction via `IsPlanar` and `IsHomeomorphic`).*

---

## 149.22 Automorphism Group of a Graph or Digraph

The automorphism-group functionality is an interface to B. McKay's **nauty** V 2.2 program
**[McK81]** (user's manual **[McK]**).

### 149.22.1 The Automorphism Group Function

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AutomorphismGroup(G : parameters)` | The automorphism group of `G`: returns the group `A`, the vertex and edge `G`-sets, the power structure 𝒜 of all automorphisms, and the transfer map from `A` to 𝒜. `G` may be directed or undirected; for < 500 vertices the canonically labelled graph is also returned. Parameters: `Canonical` (`BoolElt`, default `false`; the canonical labelling depends on all parameters used); `Stabilizer` (a partition `P` of V(G) — computes the subgroup preserving `P`; uncovered vertices form an extra cell); `Invariant` (`MonStgElt` — a named nauty invariant, default `"Null"` for graphs / `"adjacencies"` for digraphs, with associated parameters `Minlevel`, `Maxlevel`, `Arg`); `Print` (`RngIntElt`, default 0 — printing level 0–3); `IgnoreLabels` (`BoolElt`, default `false` — treat all vertices as identically labelled). | nauty **[McK81]**. |

### 149.22.2 nauty Invariants

Invariants supplement nauty's built-in partition-refinement code; each takes three parameters
`Minlevel`, `Maxlevel`, `Arg`. The available invariants (named strings) are: `"default"` (none),
`"twopaths"`, `"adjtriang"`, `"triples"`, `"quadruples"`, `"celltrips"`, `"cellquads"`,
`"cellquins"`, `"cellfano"`, `"cellfano2"`, `"distances"`, `"indsets"` (independent sets of size
`Arg`, default 3), `"cliques"` (cliques of size `Arg`, default 3), `"cellind"`, `"cellcliq"`, and
`"adjacencies"` (the default for digraphs). Further described in the nauty manual **[McK]**.

| Intrinsic | Description |
|-----------|-------------|
| `IsPartitionRefined(G : parameters)` | Whether the invariant in `Invariant` refines `G`'s vertex-set partition (default refinement if none set). Parameters `Stabilizer`, `Invariant` (`MonStgElt`), `Arg` (`RngIntElt`), `IgnoreLabels` (default `false`) — same meaning as for `AutomorphismGroup`. |

### 149.22.3 Graph Colouring and Automorphism Group

Prose only. The graph colouring (set by `AssignLabels` etc.) is an *intrinsic* property; the
default automorphism group computed by nauty is the group of the *coloured* graph. Functions using
the automorphism group split into two groups: Group 1 (use it only for efficiency) — `Diameter`,
`DiameterPath`, `IsDistanceRegular`, `GirthCycle`, `Girth`; Group 2 (always use the default group)
— `IsDistanceTransitive`, `IsTransitive`, `IsPrimitive`, `IsSymmetric`, `IsIsomorphic`,
`IsEdgeTransitive`, `EdgeGroup`, `OrbitsPartition`, `CanonicalGraph`. Setting `Invariant` can drive
the group computation and is remembered until reset.

### 149.22.4 Variants of Automorphism Group

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CanonicalGraph(G)` | The canonically labelled graph isomorphic to `G`. The result depends on the invariant used in computing the automorphism group. (See 149.22.3.) | nauty canonical labelling **[McK81]**. |
| `EdgeGroup(G)` | The automorphism group of `G` in its action on the edges, plus the `G`-set of edges. (See 149.22.3.) | nauty **[McK81]**. |
| `IsIsomorphic(G, H : parameters)` | Whether `G` and `H` are isomorphic; if so, returns a vertex mapping. Parameters as for `AutomorphismGroup` except: `Stabilizer` sets the stabiliser for both graphs; `Stabilizer2` sets it for `H` only; `IgnoreLabels` (default `false`) applies to both. Isomorphism maps colours to colours and stabilisers to stabilisers; stabiliser cells must be compatible (same-sized cells in the same order). | nauty **[McK81]**. |

*Worked examples: H149E18 (`AutomorphismGroup` of a labelled 5-cycle with various stabilisers; `CanonicalGraph`); H149E19 (`IsIsomorphic` for coloured complete graphs, stabilisers, `IgnoreLabels`).*

### 149.22.5 Action of Automorphisms

The automorphism group `A` acts on the standard support, not directly on `G`; the action on `G` is
obtained via the `G`-set mechanism. The two basic `G`-sets are the vertex-set and edge-set returned
by `AutomorphismGroup`.

| Intrinsic | Description |
|-----------|-------------|
| `Image(a, Y, y)` | For `a ∈ A`, a `G`-set `Y` and `y` in `Y` (or a derived `G`-set): the image of `y` under `a`. |
| `Orbit(A, Y, y)` | The orbit of `y` under the subgroup `A`, for `G`-set `Y`. |
| `Orbits(A, Y)` | The orbits of the action of `A` on the `G`-set `Y`. |
| `Stabilizer(A, Y, y)` | The stabiliser of `y` in `A`, for `G`-set `Y`. |
| `Action(A, Y)` | The homomorphism φ: A → L giving the action of `A` on `Y`; returns φ, the induced group `L`, and the kernel. |
| `ActionImage(A, Y)` | The permutation group `L` giving the action of `A` on `Y`. |
| `ActionKernel(A, Y)` | The kernel of the action of `A` on `Y`. |

*Worked example: H149E20 (Clebsch graph: `AutomorphismGroup` of order 1920, `CompositionFactors`, `FittingSubgroup`, `EARNS`, stabiliser orbits, `ActionImage`/`IsSymmetric`).*

---

## 149.23 Symmetry and Regularity Properties of Graphs

These rely on the graph's (default) automorphism group via nauty (see 149.22.3).

| Intrinsic | Description |
|-----------|-------------|
| `IsTransitive(G)` / `IsVertexTransitive(G)` | Whether the automorphism group of `G` is (vertex-)transitive. |
| `IsEdgeTransitive(G)` | Whether the automorphism group is transitive on the edges (i.e. the edge group is transitive). |
| `OrbitsPartition(G)` | The partition of V(G) into the orbits of the automorphism group, as a set system. |
| `IsPrimitive(G)` | Whether `G` is primitive (its automorphism group is primitive). |
| `IsSymmetric(G)` | Whether `G` is symmetric (for all u adj v and w adj t, some automorphism maps u↦w, v↦t). |
| `IsDistanceTransitive(G)` | Whether the connected graph `G` is distance transitive. |
| `IsDistanceRegular(G)` | Whether `G` is distance regular. |
| `IntersectionArray(G)` | The intersection array of the distance-regular graph `G`, returned as `[k, b(1),…,b(d−1), 1, c(2),…,c(d)]` (k = valency, d = diameter). |

*Worked example: H149E21 (symmetry functions on the 8-cube and on the incidence graph of a finite projective plane; effect of colouring on `OrbitsPartition`/`IsSymmetric`).*

---

## 149.24 Graph Databases and Graph Generation

Magma provides interfaces to several graph databases (downloaded separately from the Magma website)
and to McKay's graph-generation program.

### 149.24.1 Strongly Regular Graphs

A catalogue of strongly regular graphs assembled by B. McKay (cs.anu.edu.au/~bdm/data/). Graphs are
indexed by four parameters: order, degree, common neighbours of adjacent pairs, common neighbours
of non-adjacent pairs.

| Intrinsic | Description |
|-----------|-------------|
| `StronglyRegularGraphsDatabase()` | Opens the strongly-regular-graphs database. |
| `Classes(D)` | All parameter sequences indexing the graphs in database `D`. |
| `NumberOfClasses(D)` | The number of classes in `D`. |
| `NumberOfGraphs(D)` | The number of graphs in `D`. |
| `NumberOfGraphs(D, S)` | The number of graphs in `D` with parameter sequence `S`. |
| `Graphs(D, S)` | All graphs in `D` with parameter sequence `S` (in a sequence). |
| `Graph(D, S, i)` | The i-th graph in `D` with parameter sequence `S`. |
| `RandomGraph(D)` | A random graph in `D`. |
| `RandomGraph(D, S)` | A random graph in `D` with parameter sequence `S`. |
| `for G in D do ... end for;` | The database as a range in a `for`-statement. |

*Worked example: H149E22 (`StronglyRegularGraphsDatabase`, `Classes`, `NumberOfClasses`, `NumberOfGraphs`, `Graphs`, `Graph`).*

### 149.24.2 Small Graphs

Databases of small graphs (simple, Eulerian, planar connected, self-complementary) from McKay
(cs.anu.edu.au/~bdm/data/graphs.html).

#### 149.24.2.1 Creation of Small Graph Databases

| Intrinsic | Description |
|-----------|-------------|
| `SmallGraphDatabase(n : parameters)` | The database of simple graphs with `n` vertices (2 ≤ n ≤ 10). Parameter `IncludeDisconnected` (`Bool`, default `false`). |
| `EulerianGraphDatabase(n : parameters)` | The database of Eulerian graphs with `n` vertices (3 ≤ n ≤ 11; 2 ≤ n ≤ 12 if disconnected). Parameter `IncludeDisconnected` (default `false`). |
| `PlanarGraphDatabase(n)` | The database of planar connected graphs with `n` vertices (2 ≤ n ≤ 11). |
| `SelfComplementaryGraphDatabase(n)` | The database of self-complementary graphs with `n` vertices, n ∈ {4,5,8,9,12,13,16,17,20} (n = 20 not a complete enumeration). |

#### 149.24.2.2 Access functions

| Intrinsic | Description |
|-----------|-------------|
| `#D` | The number of graphs in the database `D`. |
| `Graph(D, i)` | The i-th graph in `D`. |
| `Random(D)` | A random graph from `D`. |
| `for G in D do ... end for;` | The database as a range in a `for`-statement. |

### 149.24.3 Generating Graphs

An interface to McKay's graph-generation program **[McK98]** (downloaded from
cs.anu.edu.au/~bdm/nauty/, compiled to `geng`, with `MAGMA_NAUTY` set to its path). The program runs
in a Unix pipe, so this facility is **only available on Unix platforms**.

| Intrinsic | Description |
|-----------|-------------|
| `GenerateGraphs(n : parameters)` | Opens a pipe to generate all graphs of order `n`. Parameters: `FirstGraph` (`RngIntElt`, default 1), `MinEdges`/`MaxEdges` (`RngIntElt`), `Classes`/`Class` (`RngIntElt`, default 1 — divide output into disjoint classes and write only one), `Connected`/`Biconnected`/`TriangleFree`/`FourCycleFree`/`Bipartite` (`BoolElt`, default `false`), `MinDeg`/`MaxDeg` (`RngIntElt`), `Canonical` (`BoolElt`, default `false`), `SparseRep` (`BoolElt`, default `false` — Sparse6 format). |
| `NextGraph(F : parameters)` | Returns `true` iff file/pipe `F` is not at end, with the next graph as a second value. `F` must contain graphs in Graph6 or Sparse6 format. Parameter `SparseRep` (`Bool`, default `false`). |

*Worked example: H149E23 (`GenerateGraphs` of order 12 with multiple filters; reading with `NextGraph`; sparse variant).*

### 149.24.4 A General Facility

`OpenGraphFile` opens either a graph file or a Unix pipe to a graph-generation program, giving
access to a stream of graphs (read via `NextGraph`). Opening a pipe is Unix-only; the stream must be
in Graph6 or Sparse6 format (cs.anu.edu.au/~bdm/data/formats.html).

| Intrinsic | Description |
|-----------|-------------|
| `OpenGraphFile(s, f, p)` | Opens a graph file/pipe at position `p`. For a pipe, `s` must be `"cmd command"`; for a file, `s` is `"filename"`. The integer `f` flags fixed record length (true for Graph6 with constant order, permitting rapid positioning; use `f = 0` if in doubt). Pipes are Unix-only; the file/pipe must contain Graph6/Sparse6 graphs. |

*Worked example: H149E24 (reading a `.g6` file with `OpenGraphFile`/`NextGraph`; driving `geng` via `OpenGraphFile`, including the `geng -help` listing).*

---

## 149.25 Bibliography (canonical references)

| Key | Reference |
|-----|-----------|
| **[BK73]** | C. Bron and J. Kerbosch. *Finding All Cliques of an Undirected Graph.* Communications of the ACM 9, 16(9):575–577, 1973. |
| **[BM01]** | J. Boyer and W. Myrvold. *Simplified O(n) Planarity Algorithms.* Submitted, 2001. |
| **[Bre79]** | D. Brelaz. *New Methods to Color the Vertices of a Graph.* Communications of the ACM, 22(9):251–256, 1979. |
| **[Chr75]** | N. Christofides. *Graph Theory, An Algorithm Approach.* Academic Press, 1975. |
| **[Eve79]** | Shimon Even. *Graph Algorithms.* Computer Science Press, 1979. |
| **[GM01]** | C. Gutwenger and P. Mutzel. *A Linear Time Implementation of SPQR-Trees.* In J. Marks, editor, Graph Drawing 2000, volume 1984 of LNCS, pages 70–90. Springer-Verlag, 2001. |
| **[HT73]** | J. E. Hopcroft and R. E. Tarjan. *Dividing a Graph into Triconnected Components.* SIAM J. Comput., 2(3):135–158, 1973. |
| **[Lor89]** | P. Lorimer. *The construction of Tutte's 8-cage and the Conder graph.* J. of Graph Theory, 13(5):553–557, 1989. |
| **[McK]** | B. D. McKay. *nauty User's Guide (Version 2.2).* URL: http://cs.anu.edu.au/~bdm/nauty/nug.pdf. |
| **[McK81]** | B. D. McKay. *Practical Graph Isomorphism.* Congressus Numerantium, 30:45–87, 1981. |
| **[McK98]** | B. D. McKay. *Isomorph-free exhaustive generation.* J. Algorithms, 26:306–324, 1998. |
| **[RAO93]** | R. K. Ahuja, T. L. Magnanti and J. B. Orlin. *Network Flows, Theory, Algorithms, and Applications.* Prentice Hall, 1993. |
| **[TCR90]** | T. H. Cormen, C. E. Leiserson and R. L. Rivest. *Introduction to Algorithms.* MIT Press, 1990. |
| **[WM]** | N. Walker, W. Myrvold, T. Prsa. *A Dynamic Programming Approach for Timing and Designing Clique Algorithms.* Available at URL: http://www.csr.uvic.ca/~wendym/. |

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Hopcroft–Tarjan triconnectivity **[HT73, GM01]** | `IsTriconnected`, `Splitcomponents`, `SeparationVertices` |
| Flow-based matching / connectivity (Dinic, push-relabel) **[Eve79]** | `MaximumMatching`, `VertexSeparator`, `VertexConnectivity`, `IsKVertexConnected`, `EdgeSeparator`, `EdgeConnectivity`, `IsKEdgeConnected` |
| Backtrack search **[Chr75]** | `MinimumDominatingSet` |
| Bron–Kerbosch branch-and-bound cliques **[BK73]** | `HasClique`, `MaximumClique`, `CliqueNumber`, `AllCliques`, `MaximumIndependentSet`, `IndependenceNumber` |
| Brélaz *dsatur* lower bound **[Bre79]** + Myrvold dynamic programming **[WM]** | `MaximumClique`, `CliqueNumber` (with `Al := "Dynamic"`), `HasClique`/`AllCliques` (Dynamic) |
| Boyer–Myrvold planarity **[BM01]** | `IsPlanar`, `Obstruction`, `IsHomeomorphic`, `Faces`, `Face`, `NFaces`, `Embedding`, `PlanarDual` |
| nauty graph isomorphism / automorphism **[McK81, McK]** | `AutomorphismGroup`, `IsPartitionRefined`, `CanonicalGraph`, `EdgeGroup`, `IsIsomorphic`, all of 149.23, and (for efficiency) `Diameter`/`DiameterPath`/`Girth`/`GirthCycle`/`IsDistanceRegular` |
| Isomorph-free exhaustive generation **[McK98]** | `GenerateGraphs`, `NextGraph`, `OpenGraphFile` |
| Spectral / matrix invariants | `CharacteristicPolynomial`, `Spectrum`, `AdjacencyMatrix`, `DistanceMatrix`, `IncidenceMatrix`, `IntersectionMatrix` |
