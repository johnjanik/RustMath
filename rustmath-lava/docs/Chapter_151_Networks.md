# Chapter 151 — Networks

**Handbook part:** XX — Combinatorics
**Handbook pages:** 5049–5065 (PDF pages 5180–5201)

---

## Scope and overview

A *network* is a directed graph whose arcs carry a cost and a capacity, and may have
multiple (parallel) edges. Networks model communication systems and dependence problems.
Magma provides a dedicated network object of type `GrphNet`, which differs from the Magma
multidigraph type `GrphMultDir` in exactly one respect: a network's edges are always assumed
to be capacitated. By default each edge has capacity 1 (loops have capacity 0) unless a
capacity is explicitly assigned. Since networks are a specialisation of multidigraphs, all the
functions that apply to multidigraphs also apply to networks; this chapter documents only the
functions that specifically concern networks — their construction, incremental modification,
and the maximum-flow / minimum-cut machinery.

The fundamental network-flow problem is the **minimum cost flow problem**: determine a maximum
flow at minimum cost from a specified source to a specified sink. Specialisations and related
problems include the shortest-path problem (no capacity constraint; covered in §150.12), the
maximum-flow problem (no cost constraint), the minimum spanning-tree problem, the matching
problem, and the multicommodity-flow problem. The comprehensive reference is **[RAO93]**.

For flows, Magma implements two algorithms: the **Dinic algorithm** **[Eve79]** and a generic
**push-relabel** method **[CGM+98, CG97]**. The Dinic algorithm (theoretical complexity
O(|V|²|E|), improvable to O(|E|^{3/2}) on zero-one networks and O(|V|^{2/3}|E|) on parallel-edge-free
zero-one networks) performs best on zero-one and very sparse networks. The push-relabel method
(O(|V|²|E|), improvable to at least O(|V||E|log(|V|²/|E|)) with heuristics — global relabelling,
gap relabelling, and discharge-ordering) is the default for the `MinimumCut` and `MaximumFlow`
intrinsics; only on very sparse zero-one networks may Dinic outperform it. The order n of a
network is bounded by 134217722.

---

## 151.1 Introduction

Introductory prose only (no intrinsics). Defines networks, the family of flow problems, and the
choice of the `GrphNet` type. See the overview above.

---

## 151.2 Construction of Networks

Networks are constructed in a manner closely analogous to multidigraphs (§150.2.2). Whenever an
edge `[u,v]` with `u ≠ v` is added to a network `N`, its capacity is set to 1 (or 0 for a loop)
unless the capacity is explicitly supplied at construction, or the object being added is itself
a network edge — in which case the edge retains the capacity it had in the source network.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Network< n \| edges >` | Construct the network `N` with vertex-set `V = {@ v_1,…,v_n @}` (with `v_i = i`) and edge-set `E = {e_1,…,e_q}`. Returns three values: the network `N`, its vertex-set `V`, and its edge-set `E`. The `edges` list items may be: (a) a pair `[v_i,v_j]` (directed edge, capacity 1, or 0 if a loop); (b) a tuple `<v_i, N_i>` (`N_i` a set of out-neighbours of `v_i`); (c) a tuple `<[v_i,v_j], c>` (directed edge with non-negative capacity `c`); (d) a sequence `[N_1,…,N_n]` of out-neighbour sets; (e) an edge `e` of a graph/di/multi/multidigraph/network of order `n` (a network edge keeps its source capacity); (f) an edge-set `E` of such an object; (g) a whole graph/di/multidigraph/network `H` of order `n`; (h) an n×n (0,1)-matrix `A` interpreted as an adjacency matrix; (i) a set of pairs / out-neighbour tuples / capacity tuples / edges / graphs of order `n`; (j) a sequence of out-neighbour tuples or capacity tuples. | Multidigraph-style constructor specialised to capacitated edges. |
| `Network< S \| edges >` | As above, but the vertices are taken from the enumerated or indexed set `S` (`v_i` is the `i`-th element of `S`). Same `edges` forms and return values. | As above. |

*Worked examples: H151E1 (`Network< n \| D >` from a `RandomDigraph` — all non-loop edges get capacity 1); H151E2 (network from a set of `<[vertex,vertex], capacity>` tuples, exhibiting parallel/multiple edges and `Edges(N)`).*

### 151.2.1 Magma Output: Printing of a Network

Magma displays a network `N` as a list of vertices, each followed by its outgoing capacitated
edges, with each edge's end-point printed followed by the capacity of the edge in brackets. If
there are multiple edges from `u` to `v`, each is printed separately with its own capacity. The
end-points in an adjacency list are not ordered; they appear in the order in which they were
created. (No intrinsics — formatting description only.)

---

## 151.3 Standard Construction for Networks

### 151.3.1 Subgraphs

Sub-network construction parallels sub-multidigraph construction (§150.5.1), with extra
flexibility for setting edge capacities. Two constraints apply (see §150.7): for any vertices
`u, v` of the subgraph `H`, the edge multiplicity from `u` to `v` must be no greater in `H` than
in `N`, and the total capacity from `u` to `v` in `H` must be no greater than in `N`. Violating
either yields a run-time error. Concretely, if the total capacity from `u` to `v` is `C_N` in
`N` and `C_H` in `H` before adding an edge, then adding an edge of capacity `c` requires
`C_H + c ≤ C_N`. Adding `[u,v]` without specifying a capacity assumes capacity `C_N` (and hence
requires the prior `C_H` to be zero); adding `[u,v]` with capacity `c` requires `C_H + c ≤ C_N`.
The support set, vertex labels, and edge labels/weights (if any) are transferred from `N` to the
sub-network.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `sub< N \| list >` | Construct the network `H` as a subgraph (sub-network) of `N`. Returns three values: the network `H`, its vertex-set `V`, and its edge-set `E`. The `list` items may be: (a) a vertex of `N` (subgraph induced on those vertices); (b) an edge of `N` (subgraph on `VertexSet(N)` whose edges are those listed, subject to the multiplicity/capacity constraints); (c) a pair `[v_i,v_j]` of vertices (edge `[v_i,v_j]` with capacity assumed to be the total capacity from `v_i` to `v_j` in `N`); (d) a tuple `<[v_i,v_j], c>` (edge with non-negative capacity `c` to be added to `H`); (e) a set of vertices / edges / vertex-pairs / capacity tuples of `N`. All attributes (support set, vertex/edge labels, weights) are transferred; edge capacities are transferred unless explicitly set. | Sub-network constructor enforcing the multiplicity and capacity constraints. |

*Worked example: H151E3 (build a network with multiple edges, then `sub< N \| V!1, V!3, V!4 >`; demonstrates `IsSubgraph`, the vertex maps between `N` and `H`, `Capacity`, `EdgeMultiplicity`, and the various run-time errors when the capacity/multiplicity constraints are violated).*

### 151.3.2 Incremental Construction: Adding Edges

Almost all multidigraph functions for adding or removing vertices/edges also apply to networks
(see §150.5.2) and are not re-listed here. Adding an edge through a general multidigraph function
(which cannot specify a capacity) always gives the edge capacity 1 (or 0 if a loop). The one
multidigraph function that differs is `AddEdge(G, u, v, l)`: for networks it is replaced by the
capacitated forms `AddEdge(G, u, v, c)` and `AddEdge(G, u, v, c, l)`, together with the
additional specialised edge-adding functions below.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `N + < [u, v], c >` | Add an edge from vertex `u` to vertex `v` of network `N` with non-negative integer capacity `c`. Returns two values: the modified network and the newly created edge (useful for parallel edges). The support and edge capacities are retained. | Functional edge addition. |
| `N + { < [u,v], c > }` / `N + [ < [u,v], c > ]` | Add the edges given by a set or sequence of `<[u,v], c>` tuples (`u, v` vertices, `c` a non-negative integer capacity). A sequence is useful when there are duplicates. Support and capacities retained. | Functional bulk edge addition. |
| `N +:= < [u,v], c >` / `N +:= { < [u,v], c > }` / `N +:= [ < [u,v], c > ]` | Procedural versions of the previous three functions. | As above. |
| `AddEdge(N, u, v, c)` | Add an edge from `u` to `v` with non-negative integer capacity `c`. Returns the modified network and the newly created edge (useful for parallel edges). Support and edge capacities retained. | Capacitated edge addition. |
| `AddEdge(N, u, v, c, l)` | Add an edge from `u` to `v` with capacity `c` and label `l`. Returns the modified network and the newly created edge. Support and capacities retained. | Capacitated, labelled edge addition. |
| `AddEdges(N, S)` | Given a set or sequence `S` of tuples `<[u,v], c>` (`u, v` vertices, `c` a non-negative integer), add all the specified edges. Support and existing vertex/edge decorations retained. | Bulk capacitated edge addition. |
| `AddEdge(~N, u, v, c)` / `AddEdge(~N, u, v, c, l)` / `AddEdges(~N, S)` | Procedural versions of the previous edge-adding functions. Tuples may be in a set or a sequence (a sequence is useful for duplicates). | Procedural edge addition. |

### 151.3.3 Union of Networks

A new network can be constructed from the union of two networks; see §150.5.4 for details. (No
intrinsics listed in this chapter.)

---

## 151.4 Maximum Flow and Minimum Cut

All functions in this section apply to general (multi)graphs whose edges carry a capacity, i.e.
networks; an uncapacitated graph is treated as having every edge of capacity one. To assign
capacities to graph edges see §150.4.2; to create a `GrphNet` object see §151.2.

**Definitions.** Let `G` be a network with vertex-set `V` and edge-set `E`; write `c(u,v)` for
the capacity of edge `[u,v]` (and `c(u,v)=0` if there is no such edge). An undirected edge
`{u,v}` of capacity `c` is treated as the two directed edges `[u,v]` and `[v,u]` each of capacity
`c`. Distinguishing a source `s` and a sink `t`, a *flow* is an integer-valued function
`f : V×V → Z` satisfying (i) the capacity constraint `f(u,v) ≤ c(u,v)`; (ii) skew symmetry
`f(u,v) = −f(v,u)`; and (iii) flow conservation `Σ_{v∈V} f(u,v) = 0` for all `u ∈ V\{s,t}`. The
*value* of a flow is `F = Σ_{v∈V} f(s,v)`. A *cut* `{S,T}` is a partition of `V` with `s ∈ S` and
`t ∈ T`; its capacity `c(S)` is the sum of `c(u,v)` over edges with `u ∈ S`, `v ∈ T`. Always
`F ≤ c(S)`, with equality iff `F` is a maximum flow and `S` is a minimum cut. (Flow could be
real-valued if capacities were real-valued.)

**Algorithms.** Two maximum-flow algorithms are implemented (see §149.13.4, §149.13.5):

- **The Dinic algorithm** **[Eve79]**. Phase 1 builds a layered network of the "useful" edges
  (those with `f(u,v) < c(u,v)`), with `s` and `t` in the first/last layers; if no such layered
  network exists the flow is maximum. Phase 2 finds a maximal flow by depth-first path
  construction in the layered network (every `s`–`t` path has a saturated edge). Complexity
  O(|V|²|E|), → O(|E|^{3/2}) for zero-one networks, → O(|V|^{2/3}|E|) for zero-one networks with no
  parallel edges. Best on zero-one and very sparse networks.
- **The generic push-relabel method** **[CGM+98, CG97]**. Pushes maximal flow out of `s` (whose
  height is initialised to |V|), then discharges excess "downward" from higher to lower vertices,
  relabelling (lifting) vertices as needed; terminates because heights are bounded by 2|V|−1.
  Complexity O(|V|²|E|), → at least O(|V||E|log(|V|²/|E|)) with heuristics. Magma incorporates global
  relabelling, gap relabelling, and discharge-order heuristics. For zero-one networks the
  smallest-height vertex is chosen next; for general flow the largest-height vertex is chosen.

For `MinimumCut` and `MaximumFlow` the `PushRelabel` algorithm is used by default; only on a very
sparse zero-one network may `Dinic` outperform it.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `MinimumCut(s, t : -)` | Given source `s` and sink `t` of a network `G`, return the subset `S` defining a minimum cut `{S,T}` of `V` (with `s ∈ S`, `t ∈ T`) corresponding to the maximum flow `F` from `s` to `t`; `S` is a sequence of vertices. The maximum flow `F` is the second return value. Parameter `Al` (`MonStgElt`, default `"PushRelabel"`) selects the algorithm: `"PushRelabel"` or `"Dinic"`. | Push-relabel **[CGM+98, CG97]** (default) or Dinic **[Eve79]**. |
| `MinimumCut(Ss, Ts : -)` | Given sequences `Ss` and `Ts` of vertices of `G`, return the subset `S` defining a minimum cut `{S,T}` with `Ss ⊆ S` and `Ts ⊆ T`, corresponding to the maximum flow `F` from the vertices in `Ss` to those in `Ts`; `S` is a sequence of vertices. `F` is the second return value. Parameter `Al` (`MonStgElt`, default `"PushRelabel"`): `"PushRelabel"` or `"Dinic"`. | As above. |
| `MaximumFlow(s, t : -)` | Given source `s` and sink `t` of a network `G`, return the maximum flow `F` from `s` to `t`. The subset `S` defining a minimum cut `{S,T}` of `VertexSet(N)` (with `s ∈ S`, `t ∈ T`) corresponding to `F` is returned as the second value (a sequence of vertices). Parameter `Al` (`MonStgElt`, default `"PushRelabel"`): `"PushRelabel"` or `"Dinic"`. | As above. |
| `MaximumFlow(Ss, Ts : -)` | Given sequences `Ss` and `Ts` of vertices of `G`, return the maximum flow `F` from the vertices in `Ss` to those in `Ts`. The subset `S` defining a minimum cut `{S,T}` of `VertexSet(N)` (with `Ss ⊆ S`, `Ts ⊆ T`) corresponding to `F` is the second value (a sequence of vertices). Parameter `Al` (`MonStgElt`, default `"PushRelabel"`): `"PushRelabel"` or `"Dinic"`. | As above. |
| `Flow(e)` | Given an edge `e` of a network `G` (whose edges must have explicitly assigned capacities — see §150.4.2 or §151.2), return the flow on `e` as an integer. Edges carry a flow only if a flow has been constructed from a source to a sink; otherwise all flows are zero. | Reads the flow value computed by a prior `MaximumFlow`/`MinimumCut`. |
| `Flow(u, v)` | For adjacent vertices `u, v` of a network `G` with explicitly assigned capacities, return the total net flow from `u` to `v` as an integer, defined as total outgoing flow from `u` into `v` minus total ingoing flow into `u` from `v`. Satisfies skew symmetry `Flow(u, v) = -Flow(v, u)`. Zero if no flow has been constructed. | Net flow between adjacent vertices. |

*Worked example: H151E4 (replicates the maximum-matching implementation of §149.13.4: build a bipartite graph, add source/sink, form a capacitated `MultiDigraph`, then `MaximumFlow(V!s, V!t)` and `MinimumCut(V!s, V!t)`; verify the cut capacity equals `F` via `Capacity`/`Flow`, recover the matching from the saturated edges, and check the capacity constraint, skew symmetry, and flow conservation using `Flow`, `Capacity`, `OutNeighbours`, `InNeighbours`, `EndVertices`).*

---

## 151.5 Bibliography (canonical references)

| Key | Reference |
|-----|-----------|
| **[CG97]** | B. V. Cherkassky and A. V. Golberg. *On Implementing the Push-Relabel Method for the Maximum Flow Problem.* Algorithmica, **19**:390–410, 1997. |
| **[CGM+98]** | B. V. Cherkassky, A. V. Golberg, Paul Martin, J. C. Setubal, and J. Stolfi. *Augment or Push? A Computational Study of Bipartite Matching and Unit Capacity Flow Algorithms.* Technical report, NEC Research Institute, 1998. |
| **[Eve79]** | Shimon Even. *Graph Algorithms.* Computer Science Press, 1979. |
| **[RAO93]** | R. K. Ahuja, T. L. Magnanti, and J. B. Orlin. *Network Flows, Theory, Algorithms, and Applications.* Prentice Hall, 1993. |

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Network construction (capacitated multidigraph) | `Network< n \| edges >`, `Network< S \| edges >`, `sub< N \| list >` |
| Incremental capacitated edge addition | `N + <[u,v],c>`, `N +:= …`, `AddEdge(N,u,v,c)`, `AddEdge(N,u,v,c,l)`, `AddEdges(N,S)`, `AddEdge(~N,…)`, `AddEdges(~N,S)` |
| Push-relabel maximum flow / minimum cut **[CGM+98, CG97]** (default) | `MaximumFlow`, `MinimumCut` (`Al := "PushRelabel"`) |
| Dinic maximum flow / minimum cut **[Eve79]** | `MaximumFlow`, `MinimumCut` (`Al := "Dinic"`) |
| Flow inspection | `Flow(e)`, `Flow(u, v)` |
| Network flow theory / applications reference | **[RAO93]** |
