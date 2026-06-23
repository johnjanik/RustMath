# Chapter 150 — Multigraphs

**Handbook part:** XX — Combinatorics
**Handbook pages:** 4999–5046 (PDF pages 5132–5180)

---

## Scope and overview

*Multigraphs* and *multidigraphs* are graphs and digraphs that may have multiple
(i.e., parallel) edges and loops — in contrast to the simple graphs of Chapter 149. In this
chapter the term "graph" is used generically for the vertex–edge incidence structure, and
"multigraph" is used whenever the possible existence of multiple edges and loops is to be
emphasised. Multigraphs may be directed or undirected; the precise meaning of the terms must
be inferred from context.

Multigraphs are represented internally as an **adjacency list**. As with simple graphs, the
vertices and edges may carry *decorations*: vertices and edges may be **labelled**, and edges
may additionally be assigned a **capacity** (non-negative integer; used for network-flow
problems, with loops forced to capacity 0) and/or a **weight** (an element of a totally ordered
ring; used for shortest-path problems). Almost all the standard graph construction functions
preserve the graph's support set and its vertex/edge decorations.

For convenience Magma also provides *networks*: a network is a multidigraph whose edges are
*always* given a capacity. Any function taking a network will usually accept any graph whose
edges carry capacities. Networks are covered in Chapter 151, but network-specific behaviour is
noted here where it differs from general multi(di)graphs.

A Magma multigraph object has type `GrphMultUnd`, a multidigraph has type `GrphMultDir`, and a
network has type `GrphNet`. All three are of type `GrphMult`. The order *n* of a multigraph is
bounded by 134217722.

Because the multigraph facilities reproduce a large subset of the simple-graph functionality,
several sections of this chapter closely mirror their counterparts in Chapter 149. For clarity
the chapter documents the *complete* multigraph functionality even where it duplicates simple
graphs; the key conceptual difference is in **edge identification**: since there may be several
edges from *u* to *v*, each is identified by its **index** in the adjacency list (Section 150.3).

---

## 150.2 Construction of Multigraphs

The order *n* of a multigraph or multidigraph is bounded by 134217722.

### 150.2.1 Construction of a General Multigraph

Undirected multigraphs are constructed similarly to simple graphs.

| Intrinsic | Description |
|-----------|-------------|
| `MultiGraph< n \| edges >` / `MultiGraph< S \| edges >` | Construct the multigraph `G` with vertex-set `V = {@ v₁,…,vₙ @}` (where `vᵢ = i` for the first form, or the *i*-th element of the enumerated/indexed set `S` otherwise) and edge-set `E = {e₁,…,e_q}`. Returns three values: the multigraph `G`, the vertex-set `V`, and the edge-set `E`. Items of `edges` may be: (a) a pair `{vᵢ,vⱼ}` (an undirected edge); (b) a tuple `⟨vᵢ, Nᵢ⟩` with `Nᵢ` a set of neighbours of `vᵢ`; (c) a sequence `[N₁,…,Nₙ]` of *n* neighbour-sets; (d) an edge `e` of any graph/digraph/multi(di)graph/network of order `n`; (e) an edge-set `E`; (f) a graph/digraph/multi(di)graph/network `H` of order `n` (all its edges added); (g) a set of any of pairs / tuples `⟨vᵢ,Nᵢ⟩` / edges / graphs of order `n`; (h) a sequence of tuples `⟨vᵢ,Nᵢ⟩`. |

*Worked example: H150E1 (construction from neighbour tuples; multiple edges shown by repeated entries in the adjacency listing).*

### 150.2.2 Construction of a General Multidigraph

Multidigraphs are constructed in the same way as digraphs.

| Intrinsic | Description |
|-----------|-------------|
| `MultiDigraph< n \| edges >` / `MultiDigraph< S \| edges >` | Construct the multidigraph `G` with vertex-set `V = {@ v₁,…,vₙ @}` and edge-set `E = {e₁,…,e_q}`. Returns three values: `G`, the vertex-set `V`, and the edge-set `E`. Items of `edges` may be: (a) a pair `[vᵢ,vⱼ]` (a directed edge from `vᵢ` to `vⱼ`); (b) a tuple `⟨vᵢ,Nᵢ⟩` with `Nᵢ` a set of out-neighbours of `vᵢ`; (c) a sequence `[N₁,…,Nₙ]` of *n* out-neighbour sets; (d) an edge `e` of any graph/digraph/multi(di)graph/network of order `n` (an undirected edge `u`–`v` adds both `[u,v]` and `[v,u]`); (e) an edge-set `E`; (f) a graph `H` of order `n`; (g) a set of pairs `[vᵢ,vⱼ]` / tuples `⟨vᵢ,Nᵢ⟩` / edges / graphs of order `n`; (h) a sequence of tuples `⟨vᵢ,Nᵢ⟩`. |

*Worked example: H150E2 (construction of a multidigraph from out-neighbour tuples).*

### 150.2.3 Printing of a Multi(di)graph

A multi(di)graph is displayed by listing, for each vertex, all of its adjacent vertices. If
there are multiple edges from *u* to *v*, the adjacency list of *u* contains as many copies of
*v* as there are edges from *u* to *v*. The vertices in the adjacency list are not ordered; they
appear in the order in which they were created. (See examples H150E1 and H150E2.)

### 150.2.4 Operations on the Support

The support of a multi(di)graph is subject to exactly the same operations as for simple graphs.

| Intrinsic | Description |
|-----------|-------------|
| `Support(G)` / `Support(V)` | The indexed set used in the construction of `G` (or of the graph for which `V` is the vertex-set), or the standard set `{@ 1,…,n @}` if none was given. |
| `ChangeSupport(G, S)` | For `G` with *n* vertices and `S` an indexed set of cardinality *n*: a new graph `H` equal to `G` but with support `S`. `H` is structurally equal to `G` and its vertex/edge decorations are the same. |
| `ChangeSupport(~G, S)` | Procedural version of the above. |
| `StandardGraph(G)` | A graph `H` isomorphic to `G` but defined on the standard support; structurally equal to `G` with the same decorations. |

*Worked example: H150E3 (`MultiGraph` over a string support; `StandardGraph` re-bases onto `{@1,2,3@}`).*

---

## 150.3 The Vertex–Set and Edge–Set of Multigraphs

Much of the simple-graph functionality (Section 149.4) applies to multigraphs. The functions
on the vertex-set are not repeated here; instead the focus is on **edges**, since multigraph
edges are created and accessed differently from simple-graph edges.

Because a multigraph may have multiple edges from *u* to *v*, a scheme is needed to uniquely
identify each one. Each edge from *u* to *v* is identified by its **index** in the adjacency
list. Edge coercion into a multigraph therefore requires two vertices *u*, *v* (with *v* a
neighbour of *u*) and a valid index *i*: position *i* in the adjacency list of *u* is the index
of an edge from *u* to *v*.

| Intrinsic | Description |
|-----------|-------------|
| `EdgeIndices(u, v)` / `Indices(u, v)` | The indices of the possibly multiple edge from *u* to *v* in multigraph `G`. |
| `EdgeMultiplicity(u, v)` / `Multiplicity(u, v)` | The multiplicity of the edge from *u* to *v*. Returns 0 if *u* is not adjacent to *v*. |
| `Edges(u, v)` | All the edges from *u* to *v* as a sequence of elements of the edge-set of `G`. |
| `IncidentEdges(u)` | All edges incident to *u* as a set of edge-set elements. For an undirected multigraph: edges incident to *u*; for a multidigraph: all edges incident *to and from* *u*. |
| `E ! < { u, v }, i >` | For the edge-set `E` of an **undirected** multigraph `G` and adjacent `u`, `v` in the support: the edge from *u* to *v* with index *i* in the adjacency list. Requires the edge at *i* to be an edge from *u* to *v*. |
| `E ! < [ u, v ], i >` | As above for the edge-set `E` of a **multidigraph** `G`: the directed edge from *u* to *v* with adjacency-list index *i*. |
| `E . i` | For the edge-set `E` of `G`: if `G` is simple, as in 149.4.2; if `G` is a multi(di)graph, the edge at index *i* in the adjacency list of `G`, provided *i* is valid. |
| `EndVertices(e)` | The end vertices of edge `e`: a set `{u,v}` (undirected) or a sequence `[u,v]` (directed). |
| `InitialVertex(e)` | For `e = {u,v}` or `e = [u,v]`: vertex *u*. Indicates, where relevant, the direction in which an undirected edge has been traversed. |
| `TerminalVertex(e)` | For `e = {u,v}` or `e = [u,v]`: vertex *v*. As above. |
| `Index(e)` | The index of edge `e` in the adjacency list of `G`. |
| `s eq t` | `true` iff edges `s` and `t` are equal; for edges of a multi(di)graph, `true` iff they have the same index in the adjacency list of `G`. |

*Worked example: H150E4 (loops and multiple edges; `EdgeSet`, `EdgeIndices`, coercion `E!<{u,v},i>`, `Index`, `EndVertices`).*

---

## 150.4 Vertex and Edge Decorations

### 150.4.1 Vertex Decorations: Labels

Only **labels** may be assigned as vertex decorations. A vertex labelling of `G` is a partial
map from the vertex-set into a set `L` of labels.

| Intrinsic | Description |
|-----------|-------------|
| `AssignLabel(~G, u, l)` | Assigns the label `l` to vertex `u` in `G`. |
| `AssignLabels(~G, S, L)` | Assigns the labels in `L` to the vertices in the sequence/indexed set `S`. If for a vertex the corresponding `L`-entry is undefined, any existing label is removed. |
| `AssignVertexLabels(~G, L)` | Assigns the labels in `L` to the corresponding vertices of `G`. |
| `IsLabelled(u)` | `true` iff vertex `u` has a label. |
| `IsLabelled(V)` | `true` iff the vertex-set `V` is labelled. |
| `IsVertexLabelled(G)` | `true` iff the vertices of `G` are labelled. |
| `Label(u)` | The label of `u`. Error if `u` is not labelled. |
| `Labels(S)` | The sequence `L` of labels of the vertices in sequence `S` (undefined entry where unlabelled). |
| `Labels(V)` | The sequence of labels of the vertices in the vertex-set `V` (undefined where unlabelled). |
| `VertexLabels(G)` | The sequence of labels of all vertices of `G` (undefined where unlabelled). |
| `DeleteLabel(~G, u)` | Removes the label of vertex `u`. |
| `DeleteLabels(~G, S)` | Removes the labels of the vertices in `S`. |
| `DeleteVertexLabels(~G)` | Removes the labels of all vertices in `G`. |

*Worked examples: H150E5 (2-colouring of K₃,₄ via vertex labels and `Distance`); H150E6 (Cayley graph of Sym(4): vertex and edge labelling so a cycle's edge labels multiply to the identity).*

### 150.4.2 Edge Decorations

Edges may carry three kinds of decoration: a **label**, a **capacity**, and a **weight**. Edge
labels may be of any Magma type. Capacities must be non-negative integers (loops must have
capacity 0). Weights must be elements of a totally ordered ring. Not all edges need be assigned
a given decoration; if some edges have a capacity/weight, any remaining unassigned edge is
taken to have capacity/weight zero.

#### 150.4.2.1 Assigning Edge Decorations

| Intrinsic | Description |
|-----------|-------------|
| `AssignLabel(~G, e, l)` / `AssignCapacity(~G, e, c)` / `AssignWeight(~G, e, w)` | Assigns the label `l`, capacity `c`, or weight `w` to edge `e`. Capacity must be a non-negative integer (0 for a loop); weight must be from a totally ordered ring. |
| `AssignLabels(~G, S, D)` / `AssignCapacities(~G, S, D)` / `AssignWeights(~G, S, D)` | Assigns the labels/capacities/weights in sequence `D` to the corresponding edges in the sequence/indexed set `S`. Undefined `D`-entry ⇒ existing decoration removed. Same capacity/weight constraints apply. |
| `AssignEdgeLabels(~G, D)` / `AssignCapacities(~G, D)` / `AssignWeights(~G, D)` | Assigns the labels/capacities/weights in sequence `D` to the edges of `G`: the edge decorated by `D[i]` is `E.i` (i.e. `E ! ⟨…,i⟩`). Undefined entry ⇒ existing decoration removed. Same constraints apply. |

#### 150.4.2.2 Testing for Edge Decorations

An edge is *labelled* iff it has been assigned a label. For capacity/weight there is a default
of zero: if any edge of `G` has been assigned a capacity (weight), the edge-set is *capacitated*
(*weighted*) and an unassigned edge has the default value zero; if no edge has been assigned
one, the edge-set is *uncapacitated* (*unweighted*) and asking for the capacity (weight) of an
edge is an error. By contrast there is no default label.

| Intrinsic | Description |
|-----------|-------------|
| `IsLabelled(e)` | `true` iff edge `e` has a label. |
| `IsLabelled(E)` | `true` iff the edge-set is labelled (at least one edge of `E` labelled). |
| `IsEdgeLabelled(G)` | `true` iff the edge-set of `G` is labelled. |
| `IsCapacitated(E)` | `true` iff the edge-set is capacitated (at least one edge of `E` assigned a capacity). |
| `IsEdgeCapacitated(G)` | `true` iff the edge-set of `G` is capacitated. |
| `IsWeighted(E)` | `true` iff the edge-set is weighted (at least one edge of `E` assigned a weight). |
| `IsEdgeWeighted(G)` | `true` iff the edge-set of `G` is weighted. |

#### 150.4.2.3 Reading Edge Decorations

| Intrinsic | Description |
|-----------|-------------|
| `Label(e)` | The label of edge `e`. (An error is raised if `e` has not been assigned a label.) |
| `Capacity(e)` | The capacity of edge `e`. Error if the edge-set of the parent graph is uncapacitated; if capacitated but `e` unassigned, returns zero (the default). |
| `Weight(e)` | The weight of edge `e`. Error if the edge-set is unweighted; if weighted but `e` unassigned, returns zero (the default). |
| `Labels(S)` / `Capacities(S)` / `Weights(S)` | The sequence `D` of labels/capacities/weights of the edges in sequence `S`. If the parent edge-set is unlabelled/uncapacitated/unweighted, `D` is the null sequence. Unlabelled element ⇒ undefined entry; un-capacitated/-weighted element of a capacitated/weighted set ⇒ default zero. |
| `Labels(E)` / `Capacities(E)` / `Weights(E)` | As above, for the edges in the edge-set `E`. The entry *i* in `D` corresponds to the edge `e = E.i`. |
| `EdgeLabels(G)` / `EdgeCapacities(G)` / `EdgeWeights(G)` | The sequence `D` of labels/capacities/weights of the edges in the edge-set of `G`. Same null/undefined/default-zero rules; entry *i* corresponds to edge `E.i`. |

#### 150.4.2.4 Deleting Edge Decorations

| Intrinsic | Description |
|-----------|-------------|
| `DeleteLabel(~G, e)` / `DeleteCapacity(~G, e)` / `DeleteWeight(~G, e)` | Removes the label/capacity/weight of edge `e`. |
| `DeleteLabels(~G, S)` / `DeleteCapacities(~G, S)` / `DeleteWeights(~G, S)` | Removes the labels/capacities/weights of the edges in `S`. |
| `DeleteEdgeLabels(~G)` / `DeleteCapacities(~G)` / `DeleteWeights(~G)` | Removes the labels/capacities/weights of all edges in the edge-set of `G`. |

*Worked example: H150E7 (random labels/capacities/weights on a multigraph; why undirected edges produce undefined entries in `EdgeLabels` — an undirected edge is stored twice and `Index` returns the odd position; loop has capacity zero by default).*

### 150.4.3 Unlabelled, or Uncapacitated, or Unweighted Graphs

These functions return a graph isomorphic (as a simple graph) to `G` but with selected
decorations removed.

| Intrinsic | Description |
|-----------|-------------|
| `UnlabelledGraph(G)` | The (vertex and edge) unlabelled graph structurally identical to `G`, whose edges keep the same capacities and weights as in `G`. The support of `G` is retained. |
| `UncapacitatedGraph(G)` | The uncapacitated graph structurally identical to `G`, keeping vertex/edge labels and edge weights. The support is retained. |
| `UnweightedGraph(G)` | The unweighted graph structurally identical to `G`, keeping vertex/edge labels and edge capacities. The support is retained. |

---

## 150.5 Standard Construction for Multigraphs

Most functions in this section correctly handle a graph's support and vertex/edge decorations:
these attributes are inherited by the resulting graph.

### 150.5.1 Subgraphs

Subgraph construction mirrors that of simple graphs. The support set, vertex labels and edge
decorations are transferred from the supergraph to the subgraph.

| Intrinsic | Description |
|-----------|-------------|
| `sub< G \| list >` | Construct the multigraph `H` as a subgraph of `G`. Returns three values: `H`, its vertex-set `V`, and its edge-set `E`. If `G` has a support and/or vertex/edge labels and/or edge capacities/weights, *all* are transferred to `H`. Items of `list` may be: (a) a vertex of `G` (induced subgraph on those vertices); (b) an edge of `G` (subgraph on `VertexSet(G)` with those edges); (c) a set of vertices of `G` or a set of edges of `G`. Duplicate elements are ignored. |

*Worked example: H150E8 (`MultiDigraph` over a support with vertex and edge decorations; a `sub<>` subgraph retains support and all decorations; correspondence of edges via end-vertices, since edge coercion needs the index too).*

### 150.5.2 Incremental Construction of Multigraphs

The full simple-graph machinery for adding/removing vertices or edges is available. Some
edge-adding functions also return the **newly created edge** (useful for determining its
adjacency-list index, e.g. when adding parallel edges); removing an edge given by a vertex pair
removes *all* edges between those vertices. Existing vertex labels and edge decorations are
retained when adding/removing; the support is retained in all cases **except when adding a new
vertex** (which reverts to standard support). Unless otherwise specified, each function returns
three values: `G`, its vertex-set `V`, and its edge-set `E`.

#### 150.5.2.1 Adding Vertices

| Intrinsic | Description |
|-----------|-------------|
| `G + n` | Adds `n` new vertices to `G` (`n ≥ 0`). Existing vertex/edge decorations retained; support becomes standard. |
| `G +:= n` / `AddVertex(~G)` / `AddVertices(~G, n)` | Procedural version. `AddVertex` adds exactly one vertex. |
| `AddVertex(~G, l)` | Adds a new vertex with label `l`. Decorations retained; support becomes standard. |
| `AddVertices(~G, n, L)` | Adds `n` new vertices with labels from the length-`n` sequence `L`. Decorations retained; support becomes standard. |

#### 150.5.2.2 Removing Vertices

| Intrinsic | Description |
|-----------|-------------|
| `G - v` / `G - U` | Removes vertex `v` (or the set `U` of vertices) from `G`. Support, vertex labels and edge decorations retained. |
| `G -:= v` / `G -:= U` / `RemoveVertex(~G, v)` / `RemoveVertices(~G, U)` | Procedural versions. |

#### 150.5.2.3 Adding Edges

| Intrinsic | Description |
|-----------|-------------|
| `G + { u, v }` / `G + [ u, v ]` | Adds the edge described by the pair to `G` (set for undirected, sequence for directed). For a network the edge gets capacity 1 (0 for a loop). Two return values: the modified graph and the newly created edge (useful for parallel edges). Support and decorations retained. |
| `G + { { u, v } }` / `G + [ { u, v } ]` / `G + { [ u, v ] }` / `G + [ [ u, v ] ]` | Adds the edges described by a set or sequence of vertex-pairs. Network ⇒ capacity 1 (0 for loop). Support and decorations retained. |
| `G +:= { u, v }` / `G +:= [ u, v ]` / `G +:= { { u, v } }` / `G +:= [ { u, v } ]` / `G +:= { [ u, v ] }` / `G +:= [ [ u, v ] ]` | Procedural versions of the previous edge-adding functions. |
| `AddEdge(G, u, v)` | Returns a new edge between `u` and `v` (network ⇒ capacity 1, 0 for loop). Two return values: the modified graph and the new edge. Support and decorations retained. |
| `AddEdge(G, u, v, l)` | For `G` not a network: adds a new edge with label `l` between `u` and `v`. Two return values (graph, new edge). Support and decorations retained. |
| `AddEdge(G, u, v, c)` | For a network `G`: adds a new edge from `u` to `v` with capacity `c`. Two return values (graph, new edge). |
| `AddEdge(G, u, v, c, l)` | For a network `G`: adds a new edge from `u` to `v` with capacity `c` and label `l`. Two return values (graph, new edge). |
| `AddEdge(~G, u, v)` / `AddEdge(~G, u, v, l)` / `AddEdge(~G, u, v, c)` / `AddEdge(~G, u, v, c, l)` | Procedural versions of the previous edge-adding functions. |
| `AddEdges(G, S)` | Adds the edges given by the set/sequence `S` of vertex-pairs (sets for undirected, sequences for directed). Network ⇒ capacity 1 (0 for loop). Support and decorations retained. |
| `AddEdges(G, S, L)` | As above, with a sequence `L` of labels of the same length assigning the corresponding label to each added edge. |
| `AddEdges(~G, S)` / `AddEdges(~G, S, L)` | Procedural versions. |

#### 150.5.2.4 Removing Edges

| Intrinsic | Description |
|-----------|-------------|
| `G - e` / `G - { e }` | The graph with edge `e` (or the set of edges) removed: vertex-set `V(G)`, edge-set `E(G)∖{e}` (resp. `E(G)∖S`). Support, vertex labels and edge labels retained on the remaining edges. |
| `G - { { u, v } }` / `G - { [u, v] }` | The graph with edge-set `E(G) − S` for a set `S` of vertex-pairs: *all* edges specified by pairs in `S` are removed (set for undirected, sequence for directed). Support, vertex and edge labels retained. |
| `G -:= e` / `G -:= { e }` / `G -:= { { u, v } }` / `G -:= { [u, v] }` / `RemoveEdge(~G, e)` / `RemoveEdges(~G, S)` / `RemoveEdge(~G, u, v)` | Procedural versions. When an edge is given as a vertex-pair, *all* edges from `u` to `v` are removed. |

### 150.5.3 Vertex Insertion, Contraction

A vertex may be inserted into a multigraph edge. The two new edges replacing an edge `e` from
`u` to `v` (with capacity `c`, weight `w`) are unlabelled and *both* keep capacity `c` and
weight `w` (this rule applies whether or not the edge-set is capacitated/weighted). Contraction
applies only to a *pair* of vertices (contracting a single multigraph edge with parallel edges
would be meaningless); support and vertex/edge decorations are retained when contracting.
Each function returns three values: `G`, the vertex-set `V`, and the edge-set `E`.

| Intrinsic | Description |
|-----------|-------------|
| `InsertVertex(e)` | Inserts a new degree-2 vertex into edge `e`. The two replacement edges share `e`'s capacity and weight (if appropriate) and are unlabelled. Vertex labels and edge decorations of `G` retained; resulting graph has standard support. |
| `InsertVertex(T)` | For a set `T` of edges: inserts a degree-2 vertex into each edge in `T`. |
| `Contract(e)` | For an edge `e = {u,v}`: removes `e` and identifies `u` and `v`. New parallel edges and loops may result; any new loop gets zero capacity. With that exception, edge decorations, support and vertex labels are retained. |
| `Contract(u, v)` | Identifies vertices `u` and `v`. Same handling of new parallel edges/loops and decorations as `Contract(e)`. |
| `Contract(S)` | For a set `S` of vertices: identifies all vertices in `S`. |

### 150.5.4 Unions of Multigraphs

Only `Union` and `EdgeUnion` are implemented for multigraphs (other union operations are easy
to write in Magma). In contrast with other standard constructions, support, vertex labels and
edge decorations are generally *not* handled — the result has standard support and no
decorations — **except** for networks, where edge capacities are properly handled.

| Intrinsic | Description |
|-----------|-------------|
| `Union(G, H)` / `G join H` | For multi(di)graphs `G`, `H` with disjoint vertex-sets: the union with vertex-set `V(G)∪V(H)` and edge-set `E(G)∪E(H)`. Standard support, no decorations. |
| `Union(N, H)` / `N join H` | For networks `N`, `H` with disjoint vertex-sets: the union network with vertex-set `V(N)∪V(H)` and edge-set `E(N)∪E(H)`. Standard support and capacities; no vertex/edge labels or weights. |
| `& join S` | The union of the multigraphs or networks in the sequence or set `S`. |
| `EdgeUnion(G, H)` | For multi(di)graphs `G`, `H` with the same number of vertices: identifies the *i*-th vertices and forms the edge union `K` with an edge `u→v` iff there is one in `G` or `H`. Standard support, no decorations. |
| `EdgeUnion(N, H)` | For networks `N`, `H` with the same number of vertices: the edge union, where `[u,v]` with capacity `c` is in `K` iff present (with capacity `c`) in `N` or `H`. Standard support, inherited capacities; no vertex/edge labels or weights. |

---

## 150.6 Conversion Functions

Conversion functions do **not** preserve a graph's support and vertex/edge decorations — the
result has standard support and no decorations — with a slight exception when the result is a
network (see `UnderlyingNetwork`).

### 150.6.1 Orientated Graphs

The rules for building an orientated graph from an undirected graph are the same as for simple
graphs.

| Intrinsic | Description |
|-----------|-------------|
| `OrientatedGraph(G)` | For a multigraph `G`: a multidigraph `D` with the same vertex-set whose edges are those of `G`, each directed from the lower- to the higher-numbered vertex. An edge `{u,v}` becomes `[u,v]` if `u < v`, else `[v,u]`; a loop at `u` becomes a directed loop at `u`. |

### 150.6.2 Converse

| Intrinsic | Description |
|-----------|-------------|
| `Converse(G)` | For a multidigraph `G` with edge-set `E`: a multidigraph `D` with the same vertex-set and edge-set `{[u,v] : [v,u] ∈ E}`. |

### 150.6.3 Converting between Simple Graphs and Multigraphs

Any simple (di)graph can be converted to a multi(di)graph and vice versa. The result has
standard support and no decorations, unless it is a network (then edges get capacity 1, 0 for
loops). For an edge `e` from `u` to `v` of `G`: if `G` and `H` are both undirected or both
directed, `e` is an edge of `H`; if `G` is undirected and `H` directed, both `[u,v]` and
`[v,u]` are edges of `H`; if `G` directed and `H` undirected, `{u,v}` is an edge of `H`. Since
these functions drop support/decorations, they may also be used to obtain a decoration-free copy.

| Intrinsic | Description |
|-----------|-------------|
| `UnderlyingGraph(G)` | The underlying simple graph of `G`. Support and vertex/edge decorations not retained. |
| `UnderlyingDigraph(G)` | The underlying simple digraph of `G`. Support and decorations not retained. |
| `UnderlyingMultiGraph(G)` | The underlying multigraph of `G`. Support and decorations not retained. |
| `UnderlyingMultiDigraph(G)` | The underlying multidigraph of `G`. Support and decorations not retained. |
| `UnderlyingNetwork(G)` | The underlying network of `G`. Support and decorations not retained, except that if `G` is a network only the edge capacities are retained. If `G` is not a network, all capacities are set to 1 (0 for loops). |

---

## 150.7 Elementary Invariants and Predicates for Multigraphs

Most (but not all) invariants and predicates that apply to simple graphs (Sections 149.10,
149.11) also apply to multigraphs.

**Equality.** `G` and `H` are equal iff: same type; structurally identical; same support;
identical vertex and edge labels; and (if applicable) the total capacity from `u` to `v` in `G`
equals that in `H`. **Subgraph.** `H` is a subgraph of `G` iff: same type; `H` is a structural
subgraph of `G`; each vertex `v` in `H` has the same support and label as `VertexSet(G)!v`;
each edge `e` in `H` has the same label as `EdgeSet(G)!e`; and (if applicable) the total
capacity from `u` to `v` in `G` is at least that in `H`. Neither test depends on edge weights.

| Intrinsic | Description |
|-----------|-------------|
| `Order(G)` / `NumberOfVertices(G)` | The number of vertices of `G`. |
| `Size(G)` / `NumberOfEdges(G)` | The number of edges of `G`. |
| `u adj v` | For vertices: `true` iff `u`, `v` are adjacent (undirected) / there is an edge `u→v` (directed). |
| `e adj f` | For edges: `true` iff `e`, `f` share a vertex (undirected) / the terminal vertex of `e` (`f`) is the initial vertex of `f` (`e`) (directed). |
| `u notadj v` | Negation of `adj` for vertices. |
| `e notadj f` | Negation of `adj` for edges. |
| `u in e` | `true` iff vertex `u` is an end-vertex of edge `e`. |
| `u notin e` | Negation of `in`. |
| `G eq H` | `true` iff `G` and `H` are equal (structurally equal and compatible w.r.t. support, vertex/edge labels and edge capacities). |
| `IsSubgraph(G, H)` | `true` iff `H` is a subgraph of `G` (structural subgraph, compatible w.r.t. support, vertex/edge labels and edge capacities). |
| `IsBipartite(G)` | `true` iff `G` is bipartite. |
| `Bipartition(G)` | For a bipartite `G`: its two partite sets as a pair of subsets of `V(G)`. |
| `IsRegular(G)` | `true` iff `G` is regular. |
| `IsComplete(G)` | `true` iff `G` on `n` vertices is the complete graph on `n` vertices. |
| `IsEmpty(G)` | `true` iff the edge-set of `G` is empty. |
| `IsNull(G)` | `true` iff the vertex-set of `G` is empty. |
| `IsSimple(G)` | `true` iff `G` is a simple graph. |
| `IsUndirected(G)` | `true` iff `G` is undirected. |
| `IsDirected(G)` | `true` iff `G` is directed. |

---

## 150.8 Adjacency and Degree

The adjacency and degree functionality of simple graphs (Section 149.12) applies similarly to
multigraphs.

### 150.8.1 Adjacency and Degree Functions for Multigraphs

| Intrinsic | Description |
|-----------|-------------|
| `Degree(u)` | The degree of vertex `u`, i.e. the number of edges incident to `u`. |
| `Alldeg(G, n)` | The set of all vertices of `G` of degree `n` (`n ≥ 0`). |
| `MaximumDegree(G)` / `Maxdeg(G)` | The maximum degree of `G`; returns the maximum degree and a vertex achieving it. |
| `MinimumDegree(G)` / `Mindeg(G)` | The minimum degree of `G`; returns the minimum degree and a vertex achieving it. |
| `DegreeSequence(G)` | For `G` with maximum degree `r`: a sequence `D` of length `r+1` with `D[i]` the number of vertices of degree `i−1`. |
| `Neighbours(u)` / `Neighbors(u)` | The set of vertices of `G` adjacent to `u`. |
| `IncidentEdges(u)` | The set of all edges incident with `u`. |

### 150.8.2 Adjacency and Degree Functions for Multidigraphs

| Intrinsic | Description |
|-----------|-------------|
| `InDegree(u)` | The number of edges directed into `u`. |
| `OutDegree(u)` | The number of edges of the form `[u,v]`. |
| `MaximumInDegree(G)` / `Maxindeg(G)` | The maximum in-degree; returns it and the first vertex achieving it. |
| `MinimumInDegree(G)` / `Minindeg(G)` | The minimum in-degree; returns it and the first vertex achieving it. |
| `MaximumOutDegree(G)` / `Maxoutdeg(G)` | The maximum out-degree; returns it and the first vertex achieving it. |
| `MinimumOutDegree(G)` / `Minoutdeg(G)` | The minimum out-degree; returns it and the first vertex achieving it. |
| `Degree(u)` | The total degree of `u` (sum of in- and out-degree). |
| `MaximumDegree(G)` / `Maxdeg(G)` | The maximum total degree; returns it and the first vertex achieving it. |
| `MinimumDegree(G)` / `Mindeg(G)` | The minimum total degree; returns it and the first vertex achieving it. |
| `Alldeg(G, n)` | The set of all vertices of total degree `n` (`n ≥ 0`). |
| `DegreeSequence(G)` | As for multigraphs, by total degree. |
| `InNeighbours(u)` / `InNeighbors(u)` | The set of vertices `v` with `[v,u]` an edge (initial vertices of edges into `u`). |
| `OutNeighbours(u)` / `OutNeighbors(u)` | The set of vertices `v` with `[u,v]` an edge (terminal vertices of edges from `u`). |
| `IncidentEdges(u)` | The set of all edges incident into and from `u`. |

---

## 150.9 Connectedness

All connectivity functions are the same for simple graphs and multigraphs; for the algorithms,
see Section 149.13.1 and its subsections.

### 150.9.1 Connectedness in a Multigraph

| Intrinsic | Description |
|-----------|-------------|
| `IsConnected(G)` | `true` iff the undirected graph `G` is connected. |
| `Components(G)` | The connected components of undirected `G` as a sequence of subsets of `V(G)`. |
| `Component(u)` | The subgraph for the connected component containing `u`. Support and decorations *not* retained in the resulting structural subgraph. |
| `IsSeparable(G)` | `true` iff `G` is connected and has at least one cut vertex. |
| `IsBiconnected(G)` | `true` iff `G` is biconnected. `G` must be undirected. |
| `CutVertices(G)` | The set of cut vertices for the connected undirected graph `G`. |
| `Bicomponents(G)` | The biconnected components of undirected `G` as a sequence of subsets of `V(G)` (`G` may be disconnected). |

### 150.9.2 Connectedness in a Multidigraph

| Intrinsic | Description |
|-----------|-------------|
| `IsStronglyConnected(G)` | `true` iff the multidigraph `G` is strongly connected. |
| `IsWeaklyConnected(G)` | `true` iff `G` is weakly connected. |
| `StronglyConnectedComponents(G)` | The strongly connected components of `G` as a sequence of subsets of `V(G)`. |
| `Component(u)` | The subgraph for the connected component containing `u`. Support and decorations *not* retained. |

### 150.9.3 Triconnectivity for Multigraphs

See Section 149.13.3 for the triconnectivity algorithm and the meaning of `Splitcomponents`.
The algorithm applies to undirected graphs only.

| Intrinsic | Description |
|-----------|-------------|
| `IsTriconnected(G)` | `true` iff `G` is triconnected. `G` must be undirected. |
| `Splitcomponents(G)` | The split components of undirected `G` as a sequence of subsets of `V(G)` (may be disconnected). Second return value: the cut vertices and the separation pairs as sequences of one or two vertices respectively. |
| `SeparationVertices(G)` | The cut vertices and/or separation pairs of undirected `G` as a sequence of sequences of one and/or two vertices. Second return value: the split components of `G`. |

### 150.9.4 Maximum Matching in Bipartite Multigraphs

See Section 149.13.4 for the maximum-matching algorithm.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `MaximumMatching(G : Al)` | A maximum matching in the bipartite graph `G`, returned as a sequence of edges of `G`. Parameter `Al` (`MonStg`, default `"PushRelabel"`) selects the algorithm: `"PushRelabel"` or `"Dinic"`. | Push–relabel (default) or Dinic max-flow-based matching. |

### 150.9.5 General Vertex and Edge Connectivity in Multigraphs and Multidigraphs

See Section 149.13.5 for the underlying algorithms. These functions apply to both undirected
and directed graphs. Each takes a parameter `Al` (`MonStg`, default `"PushRelabel"`) selecting
`"PushRelabel"` or `"Dinic"`.

| Intrinsic | Description |
|-----------|-------------|
| `VertexSeparator(G : Al)` | A vertex separator of `G` (smallest set `S` of vertices such that every (directed) path between any `u`, `v` passes through some vertex of `S`), returned as a sequence of vertices. |
| `VertexConnectivity(G : Al)` | The vertex connectivity of `G` (size of a minimum vertex separator). Second return value: a vertex separator for `G`. |
| `IsKVertexConnected(G, k : Al)` | `true` iff the vertex connectivity of `G` is at least `k`. |
| `EdgeSeparator(G : Al)` | An edge separator of `G` (smallest set `T` of edges such that every (directed) path between any `u`, `v` passes through some edge of `T`), returned as a sequence of edges. |
| `EdgeConnectivity(G : Al)` | The edge connectivity of `G` (size of a minimum edge separator). Second return value: an edge separator for `G`. |
| `IsKEdgeConnected(G, k : Al)` | `true` iff the edge connectivity of `G` is at least `k`. |

*Worked example: H150E9 (`EdgeConnectivity`/`EdgeSeparator` correctly accounting for multiple edges).*

---

## 150.10 Spanning Trees

All trees returned below are structural subgraphs of the original graph; their support and
vertex/edge decorations are *not* retained.

| Intrinsic | Description |
|-----------|-------------|
| `SpanningTree(G)` | For connected undirected `G`: a spanning tree rooted at an arbitrary vertex, as a structural subgraph (no support/decorations). |
| `SpanningForest(G)` | For a graph `G`: a spanning forest, as a structural subgraph (no support/decorations). |
| `BreadthFirstSearchTree(u)` / `BFSTree(u)` | A breadth-first search tree for `G` rooted at `u`, as a structural subgraph (no support/decorations). `G` may be disconnected. |
| `DepthFirstSearchTree(u)` / `DFSTree(u)` | A depth-first search tree `T` for `G` rooted at `u`, as a structural subgraph (no support/decorations). `G` may be disconnected. Fourth return value: the tree order of each vertex (order of visiting in the DFS); vertices not in `T` get tree order `Order(T)+1` to `Order(G)`. |

---

## 150.11 Planar Graphs

The Magma planarity algorithm tests whether an undirected graph or multigraph is planar. If
planar, an embedding is produced; otherwise a Kuratowski subgraph is identified. For a thorough
discussion of the algorithm, implementation and complexity, see Section 149.21.

| Intrinsic | Description |
|-----------|-------------|
| `IsPlanar(G)` | Tests whether the undirected graph `G` is planar (`G` may be disconnected). If non-planar, returns a Kuratowski subgraph (a subgraph homeomorphic to `K₅` or `K₃,₃`). Support and vertex/edge decorations *not* retained in the structural subgraph. |
| `Obstruction(G)` | A Kuratowski obstruction if `G` is non-planar, or the empty graph if planar. The Kuratowski graph is a structural subgraph of `G`; support and decorations not retained. |
| `IsHomeomorphic(G : Graph)` | Tests whether `G` is homeomorphic to `K₅` or `K₃,₃`. Parameter `Graph` (`MonStg`, no default) must be `"K5"` or `"K33"`. |
| `Faces(G)` | The faces of planar `G` as sequences of bordering edges. If `G` is disconnected, the face of an isolated vertex `v` is given as `[v]`. |
| `Face(u, v)` | The face of planar `G` bordered by the directed edge `[u,v]` as an ordered list of edges. (A directed edge and an orientation determine a face uniquely: the face of `e = [u₁,v₁]` is the ordered set `[u₁,v₁],[u₂,v₂],…,[u_m,v_m]` with `vᵢ = u_{i+1}` and `v_m = u₁`, where at each `vᵢ = u_{i+1}` the neighbours `uᵢ`, `u_{i+1}` are consecutive in `vᵢ`'s adjacency list ordered anti-clockwise.) |
| `Face(e)` | For an (undirected) edge `e = u,v` of planar `G`: `Face(u,v)`, the face bordered by directed `[u,v]` as a sequence of edges. |
| `NFaces(G)` / `NumberOfFaces(G)` | The number of faces of planar `G`. For a disconnected graph an isolated vertex counts as one face. |
| `Embedding(G)` | The planar embedding of `G` as a sequence `S` where `S[i]` is the sequence of edges incident from vertex `i`. |
| `Embedding(v)` | The ordered list of edges (e.g. clockwise) incident from vertex `v`. |

*Worked examples: H150E10 (embedding and faces of a multigraph with multiple edges and loops); H150E11 (constructing the dual `D` of a planar graph from its `Faces`, the bijection `e_star` between `G`'s and `D`'s edge-sets, and computing minimal cuts generating the cut space of `D`; cf. cycle/cut-space theory, **[Die00]**).*

---

## 150.12 Distances, Shortest Paths and Minimum Weight Trees

Two single-source shortest-path algorithms are implemented: **Dijkstra's algorithm** for graphs
without negative-weight cycles, and **Bellman–Ford** for graphs with negative-weight cycles.
Dijkstra is implemented with either a priority queue (binary heap) or a Fibonacci heap; the
Fibonacci heap is asymptotically faster for sparse graphs, but for most practical (small-order)
graphs the binary heap outperforms it and is the default. **Johnson's algorithm** is used for
all-pairs shortest paths (it outperforms Floyd's, especially on larger graphs). **Prim's
algorithm** implements the minimum-weight-tree computation for undirected graphs (the tree is
spanning iff the graph is connected).

All functions apply to graphs whose edges are weighted; an unweighted graph is treated as having
all edge weights 1. All functions accept negatively weighted edges. Each function takes a
parameter `UseFibonacciHeap` (`Bool`, default `false`) selecting the Fibonacci-heap variant of
Dijkstra.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Reachable(u, v : UseFibonacciHeap)` | `true` iff there is a path from `u` to `v`; if so, also returns the distance from `u` to `v`. | Dijkstra (binary or Fibonacci heap). |
| `Distance(u, v : UseFibonacciHeap)` | The distance from `u` to `v`. Error if no path exists. | Dijkstra. |
| `Distances(u : UseFibonacciHeap)` | The sequence `D` of distances from `u` to every vertex `v` (`D[Index(v)]`); undefined where no path exists. | Dijkstra. |
| `PathExists(u, v : UseFibonacciHeap)` | `true` iff there is a path from `u` to `v` in the parent graph; if so, also returns a shortest path as a sequence of edges. | Dijkstra. |
| `Path(u, v : UseFibonacciHeap)` / `ShortestPath(u, v : UseFibonacciHeap)` | A shortest path from `u` to `v` as a sequence of edges. Error if no path exists. | Dijkstra. |
| `Paths(u : UseFibonacciHeap)` / `ShortestPaths(u : UseFibonacciHeap)` | The sequence `P` of shortest paths (as edge sequences) from `u` to every vertex; `P[Index(v)]` undefined where no path exists. | Dijkstra. |
| `GeodesicExists(u, v : UseFibonacciHeap)` | `true` iff there is a path from `u` to `v`; if so, also returns a shortest path as a sequence of *vertices*. | Dijkstra. |
| `Geodesic(u, v : UseFibonacciHeap)` | A shortest path from `u` to `v` as a sequence of vertices. Error if no path exists. | Dijkstra. |
| `Geodesics(u : UseFibonacciHeap)` | The sequence `P` of shortest paths (as vertex sequences) from `u` to every vertex; undefined where no path exists. | Dijkstra. |
| `HasNegativeWeightCycle(u : UseFibonacciHeap)` | `true` iff there is a negative-weight cycle reachable from `u`. | Bellman–Ford. |
| `HasNegativeWeightCycle(G)` | `true` iff `G` has any negative-weight cycle. | Bellman–Ford. |
| `AllPairsShortestPaths(G : UseFibonacciHeap)` | All-pairs shortest paths. Returns two sequences `S₁`, `S₂`: `S₁[i][j]` (if defined) is the distance from `u` to `v`, and `S₂[i][j]` (if defined) is the vertex preceding `v` on a shortest path from `u` to `v` (`i = Index(u)`, `j = Index(v)`). Error if `G` has a negative-weight cycle. | **Johnson's algorithm**. |
| `MinimumWeightTree(u : UseFibonacciHeap)` | A minimum-weight tree rooted at `u` (of an undirected graph), as a subgraph of `G`. Spans `G` iff `G` is connected. Support and vertex/edge decorations *are* transferred to the tree. | **Prim's algorithm**. |

*Worked examples: H150E12 (weighted multidigraph; `HasNegativeWeightCycle`, `Reachable`, `Path`, `Geodesic`, path-weight verification); H150E13 (weighted multigraph with a negative-weight edge; `MinimumWeightTree` vs a DFS spanning tree via `DFSTree`, confirming the minimum-weight tree is no heavier; note: for an undirected graph, an edge `{u,v}` with negative weight creates negative-weight cycles `{u,u}` and `{v,v}`, so undirected graphs require non-negative weights).*

---

## 150.13 Bibliography (canonical references)

| Key | Reference |
|-----|-----------|
| **[Die00]** | Reinhard Diestel. *Graph Theory, Second Edition.* Springer, 2000. |

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Adjacency-list edge identification by index | `EdgeIndices`/`Indices`, `Edges`, `E.i`, `E!<…,i>`, `Index`, `EdgeMultiplicity`/`Multiplicity` |
| Push–relabel / Dinic max-flow (matching, connectivity) | `MaximumMatching`, `VertexSeparator`, `VertexConnectivity`, `IsKVertexConnected`, `EdgeSeparator`, `EdgeConnectivity`, `IsKEdgeConnected` |
| Triconnectivity (split components) | `IsTriconnected`, `Splitcomponents`, `SeparationVertices` |
| Breadth-/depth-first search | `BreadthFirstSearchTree`/`BFSTree`, `DepthFirstSearchTree`/`DFSTree`, `SpanningTree`, `SpanningForest` |
| Planarity testing, embedding, Kuratowski extraction | `IsPlanar`, `Obstruction`, `IsHomeomorphic`, `Faces`, `Face`, `NFaces`/`NumberOfFaces`, `Embedding` |
| Cycle/cut-space theory **[Die00]** | dual-graph and minimal-cut construction via `Faces`/`Embedding` (Example H150E11) |
| Dijkstra (binary or Fibonacci heap) | `Reachable`, `Distance`, `Distances`, `PathExists`, `Path`/`ShortestPath`, `Paths`/`ShortestPaths`, `GeodesicExists`, `Geodesic`, `Geodesics` |
| Bellman–Ford (negative-weight cycles) | `HasNegativeWeightCycle` |
| Johnson's algorithm (all-pairs shortest paths) | `AllPairsShortestPaths` |
| Prim's algorithm (minimum weight tree) | `MinimumWeightTree` |
