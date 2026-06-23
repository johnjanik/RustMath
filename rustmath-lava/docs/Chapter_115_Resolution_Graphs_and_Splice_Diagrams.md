# Chapter 115 — Resolution Graphs and Splice Diagrams

**Handbook part:** XV — Algebraic Geometry
**Handbook pages:** 3743–3756 (PDF pages 3874–3889)

---

## Scope and overview

Resolution graphs and splice diagrams are labelled graph-like structures used to encode geometric data arising from resolutions of singularities in algebraic geometry. A typical use case is a configuration of curves on a surface: the dual graph of such a configuration has vertices corresponding to individual curves, edges corresponding to intersections, and vertex labels carrying self-intersection numbers, multiplicities, canonical class data, and transverse intersection counts.

The chapter defines two enhanced graph types: `GrphRes` (resolution graphs) and `GrphSpl` (splice diagrams). Neither is a literal Magma graph; both hold a directed graph as the **underlying graph** and cache associated numerical data in sequences. Dedicated vertex types expose the convenient idiom of Magma's graph package, but edge types are absent. Because these types wrap an underlying graph together with auxiliary data, standard graph surgery routines cannot be applied directly — purpose-built surgery functions (Connect, Disconnect, etc.) are provided instead.

Resolution graphs record the dual graph of a blowup process. Each vertex v corresponds to a rational exceptional curve Ev on a blown-up surface S. The data at v is the quadruple (sv, mv, kv, tv): self-intersection Ev², the coefficient of Ev in a pullback divisor, the coefficient of Ev in the canonical class, and the number of transverse intersections of the birational transform of the original curve with Ev. This data is sufficient for basic intersection-theory calculations, including computing the genus contribution of a singularity.

Splice diagrams are fully described in [EN85]. Each edge carries a pair of integer labels; each vertex carries a count of arrows. Magma creates splice diagrams from curve singularities, jacobian pencils, or resolution graphs. At present, Magma implements translation from resolution graphs to splice diagrams (reduction procedure using valency-2 vertex removal and determinant calculation) but not the reverse continued-fraction construction.

---

## 115.1 Introduction

No intrinsics are defined in this section. It provides the conceptual background described in the overview above.

---

## 115.2 Resolution Graphs

### 115.2.1 Graphs, Vertices and Printing

Resolution graphs do not have associated vertex and edge sets in the Magma sense. A `GrphResVert` vertex type is provided so vertices can be passed between intrinsics. Graph printing displays each vertex as its integer index together with a label `[s, m, k, t]` (or a shorter form when data is incomplete), followed by the neighbours in the underlying directed graph.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `g eq h` | Returns `true` if and only if resolution graphs `g` and `h` are the same Magma object. | Identity test. |
| `ResolutionGraphVertex(g, i)` / `g ! i` | The vertex of resolution graph `g` with index `i`. | — |
| `Vertex(v)` | The underlying directed-graph vertex of the resolution graph vertex `v`. | — |
| `ResolutionGraph(v)` | The resolution graph of which `v` is a vertex. | — |
| `IsVertex(g, v)` | Returns `true` if and only if `v` is a vertex of resolution graph `g`. | — |
| `Index(v)` | The integer index of the resolution graph vertex `v` (the identifier shown when the graph is printed). | — |
| `v eq w` | Returns `true` if and only if resolution graph vertices `v` and `w` are the same Magma object. | Identity test. |

### 115.2.2 Creation from Curve Singularities

Let C be a reduced plane curve (affine or projective, see Chapter 114) and p a singular point of C. The resolution is thought of as a morphism f : S → P² of projective surfaces. The target is the **minimum transverse (log) resolution**, in which the birational transform eC of C on S is nonsingular and transverse to the exceptional locus; a larger resolution may be calculated in some circumstances without affecting the geometric data.

The calculation strings together sequences of blowups recursively using a standard **Newton polygon** argument. This argument automatically makes the curve transverse to all coordinate axes (not merely to exceptional curves), producing extra blowups that are essential when resolving irregular fibres of pencils but that do not invalidate any numerical calculations.

Maps to the projective plane are computed only at *significant* vertices (where the blowup branches or where eC meets the exceptional locus), as compositions of patch maps, to limit expense.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ResolutionGraph(C, p)` | Calculate a transverse resolution graph of the plane curve singularity of C at the point p. If p is omitted and C is affine, the resolution is computed at the origin (parameters then take default values). Parameter `M` (default 1): when 1, compute pullback multiplicities [mv]; when 0, omit. Parameter `K` (default 1): when 1, compute canonical multiplicities [kv]; when 0, omit. Returns the resolution graph, with patch maps cached at significant vertices. | Newton polygon-driven recursive blowup algorithm. **[EN85]** provides the theoretical context. |

*Worked example: H115E1 — resolution of the singularity of (x²−y³)²+xy⁶ at the origin: 6 blowups, single place, canonical class identifies blowup order.*

### 115.2.3 Creation from Pencils

Let P be a jacobian pencil f(x,y) = c in the affine plane A². See Chapter 112 for pencil creation. The **regular resolution graph** is the minimal sequence of transverse blowups resolving the rational map from the projective plane to the projective line determined by P. The root vertex (index 1) corresponds to the line at infinity of A².

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ResolutionGraph(P)` | The resolution at infinity of the jacobian pencil P. The multiplicities are those of the fibre of f at infinity. The graph is rooted at the vertex corresponding to the line at infinity. | Blowup algorithm for pencils; automatic accounting of regular fibre multiplicities. |
| `ResolutionGraph(P, a, b)` | The resolution graph at infinity of the union of the two fibres of P above values a and b. Multiplicities and canonical class are not computed automatically. | Blowup algorithm; only self-intersections and transverse intersections are calculated. |

*Worked example: H115E2 — pencil x²y−x from [Neu99]: 7-vertex regular resolution graph, followed by explicit resolution of fibres above 0 and 1 revealing irregular behaviour.*

### 115.2.4 Creation by Hand

A resolution graph can be created explicitly by supplying a directed graph (see Chapter 149) and numerical data. The underlying graph should normally be a directed tree, with root at vertex index 1, when reduction algorithms are later invoked.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `MakeResolutionGraph(g, s, t)` / `MakeResolutionGraph(g, s)` | The resolution graph on underlying directed graph `g`. Self-intersections are given by integer sequence `s`; transverse intersection counts by integer sequence `t` (omitted in the two-argument form). | Direct construction. |
| `MakeResolutionGraph(N)` | The resolution graph corresponding to Newton polygon `N`. | Newton polygon to resolution graph conversion. |
| `UnderlyingGraph(g)` | The underlying directed graph of the resolution graph `g`. | Attribute retrieval. |

### 115.2.5 Modifying Resolution Graphs

Modification functions perform linear algebra calculations typical of resolution graph numerics, but do not validate overall data consistency after each step. Surgery functions manage both the underlying graph and associated data.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Connect(v, w)` | If `v` and `w` are vertices of distinct resolution graphs, returns the union graph joined by an edge from `v` to `w`. Self-intersections are inherited; multiplicities, canonical class, and transverse intersections are inherited when calculated on both components. | Graph union with data concatenation. |
| `Disconnect(v, w)` | Removes any edge joining vertices `v` and `w` in their resolution graph. The result may be disconnected; only self-intersections and transverse intersections are preserved. | Edge removal with selective data preservation. |
| `Component(v)` | The connected component of the resolution graph containing vertex `v`. | Graph traversal. |
| `CalculateCanonicalClass(~g)` | Compute the canonical class supported on `g` from the self-intersections of the Ev, assuming the Ev are nonsingular rational curves meeting transversely. Uses only self-intersections already present on `g`. | Adjunction / intersection theory on the exceptional divisor. |
| `CalculateMultiplicities(~g)` | Compute pullback multiplicities of `g` from self-intersections, assuming nonsingular rational curves meeting transversely, and from the transverse intersection counts of eC with the Ev. When `g` arises from a curve singularity with cached multiplicity, the first exceptional curve (canonical multiplicity 1) fixes a unique solution. For graphs arising from two fibres of a pencil, returns the unique divisor linearly equivalent to zero when added to the birational transforms of the two affine fibre patches. | Linear algebra on the intersection matrix; multiplicity of the singularity breaks degeneracy. |
| `CalculateTransverseIntersections(~g)` | Compute the number of transverse intersections of eC with each Ev from self-intersection numbers and multiplicities already in `g`. | Intersection theory. |
| `ModifySelfintersection(~v, n)` | Set the self-intersection at vertex `v` to `n`. | Direct data assignment. |
| `ModifyTransverseIntersection(~v, n)` | Set the transverse intersection count at vertex `v` to `n`. | Direct data assignment. |

### 115.2.6 Numerical Data Associated to a Graph

The functions below retrieve the cached numerical attributes of a resolution graph. Many can also be applied to a single vertex (e.g. `Selfintersection(v)`, `CanonicalMultiplicity(v)`).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Size(g)` | Number of vertices of the underlying graph of the resolution graph `g`; typically the number of exceptional curves in the resolution. | Attribute retrieval. |
| `SelfIntersections(g)` | Sequence of self-intersections of all vertices of `g`. | Attribute retrieval. |
| `Multiplicities(g)` | Sequence of multiplicities (pullback coefficients) of all vertices of `g` in some divisor. | Attribute retrieval. |
| `CanonicalClass(g)` | Sequence of canonical multiplicities (coefficients in a local representative of the canonical class) of all vertices of `g`. | Attribute retrieval. |
| `TransverseIntersections(g)` | Sequence giving the number of transverse intersections of the resolved curve with each vertex of `g`. | Attribute retrieval. |
| `GenusContribution(g)` | The contribution to the geometric genus of a plane curve of a singularity whose resolution graph is `g`. | Intersection-theory formula. |
| `CartanMatrix(g)` | The incidence matrix of the undirected underlying graph of `g` with self-intersections on the diagonal. | Matrix assembly from graph and self-intersection data. |
| `Determinant(g)` | The determinant of the Cartan matrix of `g`. | Linear algebra. |

---

## 115.3 Splice Diagrams

Splice diagrams (type `GrphSpl`) are graphs decorated with integer labels at each end of each edge and a count of arrows at each vertex. They are fully described in [EN85]. Two features are not yet supported: omitting labels equal to 1, and arrow weights. Additional data stored with the diagram includes vertex multiplicities, canonical multiplicities, and total linking numbers. The distinction between a splice diagram and its underlying directed graph (and between a splice diagram vertex and its underlying vertex) is often left implicit.

### 115.3.1 Creation of Splice Diagrams

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SpliceDiagram(C, p)` | The splice diagram of the plane curve singularity of C at the point p. | Derived from the resolution graph via the reduction procedure (§115.4). |
| `RegularSpliceDiagram(P)` | The regular splice diagram at infinity of the jacobian pencil P. Rooted at vertex 1 (the line at infinity); underlying graph directed away from this root. | Derived from the regular resolution graph of P. |
| `MakeSpliceDiagram(g, e, a)` | A splice diagram on directed graph `g`. Sequence `e` supplies edge labels: the i-th element is a sequence of pairs `[a, b]` assigned to the i-th edge (in the order of `Edges(g)`) with `a` as the near label and `b` as the far label. Sequence `a` gives the number of arrows at each vertex. | Direct construction from graph and data sequences. |
| `MakeSpliceDiagram(e, l, a)` | The splice diagram described by sequences `e` (directed edges, each a pair of vertex indices), `l` (edge labels, same format as the previous function), and `a` (arrow counts per vertex). | Direct construction from data sequences. |
| `SpliceDiagramVertex(s, i)` | The vertex of splice diagram `s` with index `i`. | — |
| `SpliceDiagram(v)` | The splice diagram containing vertex `v`. | — |
| `UnderlyingGraph(s)` | The underlying directed graph of splice diagram `s`. | Attribute retrieval. |
| `UnderlyingVertex(v)` / `Vertex(v)` | The underlying directed-graph vertex corresponding to splice diagram vertex `v`. | — |
| `Vertices(s)` | The vertices of splice diagram `s`. | — |
| `RootVertex(s)` | The root vertex of splice diagram `s` (the tree is directed away from its root). | — |
| `Index(v)` | The index of vertex `v` of a splice diagram. | — |
| `s eq t` / `v eq w` | Returns `true` if and only if splice diagrams `s` and `t` (or vertices `v` and `w`) are the same Magma object. | Identity test. |

### 115.3.2 Numerical Functions of Splice Diagrams

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `EdgeLabels(s)` | The integer labels on the edges of splice diagram `s`. | Attribute retrieval. |
| `VertexLabels(s)` | The integer labels on the vertices of splice diagram `s`. | Attribute retrieval. |
| `TotalLinking(v)` | The total linking number of vertex `v` of a splice diagram. | Computed from edge labels; cf. **[EN85]**. |
| `LinkingNumbers(s)` | The total linking numbers of all vertices of splice diagram `s`. | Attribute retrieval / computation; cf. **[EN85]**. |
| `Linking(u, v)` | The linking number of vertices `u` and `v` of a splice diagram. | Product of off-path edge labels; **[EN85]**. |
| `EdgeDeterminant(u, v)` | The edge determinant of the edge joining vertex `u` to vertex `v` of a splice diagram. | Determinant of a subgraph; **[EN85]**. |
| `Valency(v)` | The splice valency of vertex `v`: valency in the underlying graph plus the number of arrows at `v`. | Graph degree + arrow count. |
| `IsRegular(s)` | Returns `true` if and only if splice diagram `s` is regular. | Criterion from **[EN85]**. |
| `IsReduced(s)` | Returns `true` if and only if splice diagram `s` is reduced (no valency-2 nodes, no weight-1 leaves). | Structural check. |
| `HasIrregularFibres(s)` | Returns `true` if and only if `s` has a vertex with zero total linking number. | Linking number check; **[Neu99]**. |
| `Degree(s)` | The linking number of the first vertex of splice diagram `s`. | `TotalLinking` at vertex 1. |
| `EulerCharacteristic(s)` | The Euler characteristic of splice diagram `s`. | Combinatorial formula from **[EN85]**. |
| `Size(s)` | The number of vertices of splice diagram `s`. | Attribute retrieval. |
| `Arrows(s)` | A sequence of integers: the i-th entry is the number of arrows at the vertex of index i. Can also be applied to a single vertex, returning a single integer. | Attribute retrieval. |
| `VertexPath(u, v)` | A sequence of vertices on the path from `u` to `v` in the splice diagram. The second return value is the sequence of products of off-path edge weights at each vertex on the path. | Tree path traversal with edge-weight accumulation. |

---

## 115.4 Translation Between Graphs

Splice diagrams arise from resolution graphs by a reduction procedure; the reverse (resolution graph from splice diagram via continued fractions) is not yet implemented in Magma. When a splice diagram has been constructed from a curve singularity, the corresponding resolution graph is cached and can be recovered via `CorrespondingResolutionGraph`; the vertex correspondence is available via `CorrespondingVertices`.

### 115.4.1 Splice Diagrams from Resolution Graphs

The translation proceeds in two steps: (1) reduce the underlying graph of `g` by removing all valency-2 vertices (where valency counts arrows); (2) compute edge labels as determinants of subgraphs. By default, Magma produces the **reduced** splice diagram to avoid unnecessary determinant computations.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SpliceDiagram(g)` | A splice diagram of resolution graph `g`. Parameter `Reduced` (default 1): if 1, produce the reduced splice diagram (valency-2 vertices removed); if 0, keep the underlying graph of `g`. Parameter `L` (default 0): if 1, compute total linking numbers of vertices. Parameter `K` (default 0): if 1, compute canonical class of vertices. | Two-step reduction: valency-2 vertex removal then edge-label computation via subgraph determinants; **[EN85]**. |
| `SpliceDiagram(g, v)` | The splice diagram of resolution graph `g` with the constraint that vertex `v` is not removed by reduction. | As above, with forced retention of `v`. |

---

## 115.5 Bibliography

| Key | Reference |
|-----|-----------|
| **[EN85]** | D. Eisenbud and W.D. Neumann. *Three-dimensional link theory and invariants of plane curve singularities*, volume 110 of Annals of Mathematics Studies. Princeton University Press, Princeton, NJ, 1985. |
| **[Neu99]** | W.D. Neumann. *Irregular links at infinity of complex affine plane curves.* Quart. J. Math. Ox. Ser. (2), 50:301–320, 1999. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Newton polygon-driven recursive blowup (log resolution of plane curve singularities) | `ResolutionGraph(C, p)` |
| Blowup resolution of jacobian pencils at infinity | `ResolutionGraph(P)`, `ResolutionGraph(P, a, b)` |
| Newton polygon to resolution graph conversion | `MakeResolutionGraph(N)` |
| Canonical class from adjunction / intersection theory | `CalculateCanonicalClass(~g)` |
| Pullback multiplicities by linear algebra on intersection matrix | `CalculateMultiplicities(~g)` |
| Transverse intersection count from intersection theory | `CalculateTransverseIntersections(~g)` |
| Cartan matrix and determinant | `CartanMatrix(g)`, `Determinant(g)` |
| Genus contribution formula | `GenusContribution(g)` |
| Splice diagram creation from singularity / pencil | `SpliceDiagram(C, p)`, `RegularSpliceDiagram(P)` |
| Valency-2 reduction + subgraph determinants for splice diagram **[EN85]** | `SpliceDiagram(g)`, `SpliceDiagram(g, v)` |
| Linking numbers and edge determinants **[EN85]** | `Linking(u, v)`, `EdgeDeterminant(u, v)`, `TotalLinking(v)`, `LinkingNumbers(s)` |
| Irregular fibre detection **[Neu99]** | `HasIrregularFibres(s)` |
| Tree path traversal with off-path weight products | `VertexPath(u, v)` |
