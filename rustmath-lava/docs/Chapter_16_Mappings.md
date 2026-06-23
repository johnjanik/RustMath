# Chapter 16 — Mappings

**Handbook part:** II — Sets, Sequences, and Mappings
**Handbook pages:** 247–254 (PDF pages 380–387)

---

## Scope and overview

Mappings are one of the fundamental datatypes in Magma, reflecting their central role
throughout algebra and mathematics. The most general way to define a mapping f : A → B
is to write a function which, given any element of A, returns its image in B. Magma goes
beyond this, providing mappings as an independent datatype (Magma category `Map`) with
compact constructors, dedicated operations, and support for important structural classes
such as homomorphisms and partial maps.

Maps are created either through the three main mapping constructors (`map< >`,
`pmap< >`, `hom< >`) or via standard functions that return mappings as primary or
secondary values. All constructors share the same general form: inside angle brackets,
domain A and codomain B are separated by `->`, with image specification to the right of
a `|`. Images can be specified as a graph, as a rule (expression involving a free variable),
or, for homomorphisms, as generator images.

The principal distinctions between the three map types are: a **partial map** need not be
defined on every element of the domain; a **homomorphism** must be structure-preserving
and imposes additional requirements on domain and codomain. Homomorphisms may be
specified by generator images (requiring a finitely presented domain) or by a rule.
Checking of "correctness" is limited: Magma verifies uniqueness and totality for
graph-defined maps, but cannot in general verify that a rule is defined on all of the
domain, nor that generator images do in fact define a valid homomorphism.

---

## 16.1 Introduction

### 16.1.1 The Map Constructors

There are three main mapping constructors: `map< >` (general map), `hom< >` (homomorphism),
and `pmap< >` (partial map). All share the form `map< A -> B | specification >`. The
domain and codomain may be arbitrary magmas. When a full map is constructed from a graph,
the domain must be finite. Homomorphisms are restricted to structure-preserving maps and
may be specified by generator images (requiring a finitely presented domain) or by a rule.

### 16.1.2 The Graph of a Map

A **graph** of A × B is a subset G of the cartesian product such that every element of A
appears exactly once as a first component. A **subgraph** relaxes this to at most once.
Elements of a (sub)graph may be given as tuples `<a, b>` or as arrow pairs `a -> b`;
the specification in a constructor may be a comma-separated list, a set, or a sequence of
such pairs (mixing forms is permitted).

### 16.1.3 Rules for Maps

A rule is specified using a free variable and an expression separated by `:->`, for
example `x :-> 3*x - 1`. The scope of the free variable is restricted to the
map-constructor. General expressions are allowed, including calls to intrinsic or
user-defined functions and in-line function definitions.

### 16.1.4 Homomorphisms

A homomorphism is uniquely determined by the images of any generating set (for domains
belonging to a variety). The `hom< >` constructor uses this: the kind of homomorphism
is determined entirely by the type of the domain (group, ring, etc.), and the codomain
is often required to belong to the same variety. For generator-image specification, the
domain must be finitely presented.

### 16.1.5 Checking of Maps

Graph-defined maps: Magma checks that no element of the domain has multiple images and
that every element of the domain has an image (unless a partial map is defined).
Rule-defined maps: it cannot be verified that the rule is defined on all of the domain.
Generator-image homomorphisms: the user is responsible for ensuring the images actually
define a valid homomorphism.

---

## 16.2 Creation Functions

### 16.2.1 Creation of Maps

Maps between structures A and B may be specified by a full graph or by an expression
rule.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `map< A -> B \| G >` | Given a finite structure A, a structure B, and a full graph G of A × B (set, sequence, or list of tuples or arrow-pairs), construct the mapping f : A → B. Every element of A must appear exactly once as a first component. | — |
| `map< A -> B \| x :-> e(x) >` | Given a set or structure A, a set or structure B, a variable x, and an expression e(x), construct the mapping f : A → B defined by e(x). The user must ensure a value is defined for every x ∈ A. Scope of x is restricted to the constructor. | — |
| `map< A -> B \| x :-> e(x), y :-> i(y) >` | As above, but also specifies the inverse map: f⁻¹ : B → A defined by y ↦ i(y). The user must ensure e and i are true inverses defined everywhere. Scope of x and y is restricted to the constructor. | — |

### 16.2.2 Creation of Partial Maps

Partial maps need not be defined for every element of the domain.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `pmap< A -> B \| G >` | Given a finite structure A of cardinality n, a structure B, and a subgraph G of A × B (set, sequence, or list of tuples or arrow-pairs), construct the partial map f : A → B defined by G. Elements of A not in G are unmapped. | — |
| `pmap< A -> B \| x :-> e(x) >` | Given a set A, a set B, a variable x, and an expression e(x), construct the partial map f : A → B. Scope of x is restricted to the constructor. | — |
| `pmap< A -> B \| x :-> e(x), y :-> i(y) >` | Same as the two-rule map constructor, but the result is marked as a partial map. | — |

### 16.2.3 Creation of Homomorphisms

The principal construction is the generator-image form, where the images of the
generators of the domain are listed. The kind and number of generators expected, as well
as features such as checking and preimage support, depend on the types of domain and
codomain. Refer to the appropriate handbook chapters for domain-specific details.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `hom< A -> B \| G >` | Given a finitely generated algebraic structure A, a structure B, and a graph G of A × B (set, sequence, or list of tuples or arrow-pairs), construct the homomorphism f : A → B by extending the generator map to all of A. Detailed requirements are module-dependent. | — |
| `hom< A -> B \| y1, ..., yn >` | Module-dependent constructor; after the bar, images for all generators of A must be specified. | — |
| `hom< A -> B \| x1 -> y1, ..., xn -> yn >` | Same as above with explicit generator–image arrow pairs. | — |
| `hom< A -> B \| x :-> e(x) >` | Given a structure A, a structure B, a variable x, and an expression e(x), construct the homomorphism f : A → B defined by e(x). Scope of x is restricted to the constructor. | — |
| `hom< A -> B \| x :-> e(x), y :-> i(y) >` | Same as the two-rule map constructor, but the result is marked as a homomorphism. | — |

### 16.2.4 Coercion Maps

Magma has a sophisticated coercion machinery; non-automatic coercion is usually
performed via the `!` operator. The functions below return the coercion map
corresponding to `!` for a specific pair of structures.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Coercion(D, C)` | Given structures D and C such that elements from D can be coerced into C, return the map m that performs this coercion; domain of m is D and codomain is C. | — |
| `Bang(D, C)` | Synonym for `Coercion(D, C)`. | — |

---

## 16.3 Operations on Mappings

### 16.3.1 Composition

Although compatible maps can be composed by repeated application (e.g. `g(f(x))`), it is
also possible to create an explicit composite map object.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `f * g` | Given f : A → B and g : B → C, construct the composition h = g ∘ f : A → C. | — |
| `Components(f)` | Returns the sequence of maps which were composed to form f. | — |

### 16.3.2 (Co)Domain and (Co)Kernel

The domain and codomain of any map can be accessed directly. Image, kernel, and cokernel
formation are available only for some intrinsic maps and for maps with certain domain/
codomain types.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Domain(f)` | The domain of the mapping f. | — |
| `Codomain(f)` | The codomain of the mapping f. | — |
| `Image(f)` | Given a mapping f with domain A and codomain B, return the image of A in B as a substructure of B. Currently supported only for some intrinsic maps and maps with certain domains and codomains. | — |
| `Kernel(f)` | Given the homomorphism f with domain A and codomain B, return the kernel of f as a substructure of A. Currently supported only for some intrinsic maps and maps with certain domains and codomains. | — |

### 16.3.3 Inverse

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Inverse(m)` | The inverse map of the map m. | — |

### 16.3.4 Function

For a map defined by a rule, it is possible to retrieve the rule as a user-defined function.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Function(f)` | The function underlying the mapping f. Only available if f was defined by the user via a rule map (i.e., an expression for the image of an arbitrary element of the domain). | — |

---

## 16.4 Images and Preimages

Standard mathematical notation is used for map images. Preimages are available only for
mappings defined by certain system intrinsics and constructors; they are **not** available
for mappings defined via the mapping constructor. For homomorphisms, the full preimage of
an element y is the coset K ∗ (y @@ f), where K is the kernel.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `a @ f` / `f(a)` | Given mapping f : A → B and element a ∈ A, return the image of a under f as an element of B. | — |
| `S @ f` / `f(S)` | Given mapping f : A → B and a finite enumerated set, indexed set, or sequence S of elements of A, return the image of S under f as an enumerated set, indexed set, or sequence of elements of B. | — |
| `C @ f` / `f(C)` | Given homomorphism f : A → B and a substructure C of A, return the image of C under f as a substructure of B. | — |
| `y @@ f` | Given mapping f : A → B supporting preimages and element y ∈ B, return a preimage of y as an element of A. For homomorphisms, this is a single element; the full preimage requires K ∗ (y @@ f) where K = Kernel(f). | — |
| `R @@ f` | Given mapping f : A → B supporting preimages and a finite enumerated set, indexed set, or sequence R of elements of B, return the preimage of R as an enumerated set, indexed set, or sequence of elements of A. | — |
| `D @@ f` | Given mapping f : A → B supporting preimages with computable kernel, and a substructure D of B, return the preimage of D under f as a substructure of A. | — |
| `HasPreimage(x, f)` | Return whether the preimage of x under f can be taken, and the preimage as a second return value if it can. | — |

---

## 16.5 Parents of Maps

Parents of maps are structures that know a domain and a codomain; they arise typically in
automorphism group calculations where a map is returned from an automorphism group into
the set of all automorphisms of some structure. All parents of maps inherit from the type
`PowMap`; parents of automorphisms additionally inherit from `PowMapAut`. There is also a
power structure of maps (type `PowStr`, analogous to other structures) used as a common
overstructure of the different parents.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Parent(m)` | The parent of m. | — |
| `Domain(P)` | The domain of the maps for which P is the parent. | — |
| `Codomain(P)` | The codomain of the maps for which P is the parent. | — |
| `Maps(D, C)` | The parent of all maps from D to C. | — |
| `Iso(D, C)` | The parent of isomorphisms from D to C. Returns a structure different from `Maps(D, C)` only if specifically implemented for such maps. | — |
| `Aut(S)` | The parent of automorphisms of S. | — |

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Graph-based map construction (finite domain) | `map< A -> B \| G >`, `pmap< A -> B \| G >`, `hom< A -> B \| G >` |
| Rule-based map construction | `map< A -> B \| x :-> e(x) >`, `pmap< A -> B \| x :-> e(x) >`, `hom< A -> B \| x :-> e(x) >` |
| Two-rule map with explicit inverse | `map< A -> B \| x :-> e(x), y :-> i(y) >`, `pmap<…>`, `hom<…>` |
| Generator-image homomorphism construction | `hom< A -> B \| y1, ..., yn >`, `hom< A -> B \| x1 -> y1, ..., xn -> yn >` |
| Coercion via `!` operator | `Coercion`, `Bang` |
| Map composition | `f * g`, `Components` |
| Domain / codomain / image / kernel access | `Domain`, `Codomain`, `Image`, `Kernel` |
| Inverse map | `Inverse` |
| Rule extraction | `Function` |
| Image computation | `@ (element)`, `@ (set/sequence)`, `@ (substructure)` |
| Preimage computation | `@@`, `HasPreimage` |
| Map parents and automorphism parents | `Parent`, `Maps`, `Iso`, `Aut` |
