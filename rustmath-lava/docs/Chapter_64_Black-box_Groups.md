# Chapter 64 — Black-box Groups

**Handbook part:** IX — Finite Groups
**Handbook pages:** 1871–1874 (PDF pages 2002–2007)

---

## Scope and overview

This chapter describes the category of black-box groups (BB-groups) in Magma, whose category
name is `GrpBB`. Black-box groups are built on top of Magma's other concrete group types: the
basic constructor wraps any concrete group `H` and returns a corresponding black-box group
whose element set is essentially the same as `H`'s and whose group operations are inherited
from `H`. The abstraction is useful when algorithms should treat the group purely as a
black box — accessible only through multiplication, inversion, and random sampling — without
exploiting the particular representation. A typical application is finding standard generators
of a group known only up to isomorphism (e.g. M24), as illustrated in the worked example.

---

## 64.1 Introduction

The `GrpBB` category models the black-box group abstraction: the underlying generators and
arithmetic come from a concrete Magma group, but the interface deliberately hides
representation-specific structure. There are no intrinsics unique to this section; it is a
prose introduction to the category.

---

## 64.2 Construction of an SLP-Group and its Elements

### 64.2.1 Structure Constructors

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `NaturalBlackBoxGroup(H)` | Construct the natural black-box group from the concrete group `H`. The element set of the BB-group is essentially the same as the element set of `H`, and the group operations are inherited from `H`. | — |

### 64.2.2 Construction of an Element

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Identity(G)` / `Id(G)` / `G ! 1` | Construct the identity element for the BB-group `G`. | — |

---

## 64.3 Arithmetic with Elements

Standard group arithmetic is inherited from the underlying concrete group.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `u * v` | Construct the product of elements `u` and `v` of the BB-group `G`. | — |
| `u ^ m` | Given an integer `m` and `u`, an element of BB-group `G`, return the element of `G` corresponding to the m-th power of `u`. | — |
| `u ^ v` | Given `u` and `v`, elements of BB-group `G`, return the element of `G` corresponding to the conjugate of `u` by `v`, i.e. v⁻¹ ∗ u ∗ v. | — |
| `(u, v)` | Commutator of the elements `u` and `v`, i.e. the element u⁻¹ ∗ v⁻¹ ∗ u ∗ v. Here `u` and `v` must belong to the same BB-group `G`. | — |

### 64.3.1 Accessing the Defining Generators

The functions below provide access to basic information stored for a BB-group `G`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `G . i` | The i-th generator for `G`. | — |
| `Generators(G)` | A set containing the generators for `G`. | — |
| `NumberOfGenerators(G)` / `Ngens(G)` | The number of generators for `G`. | — |

---

## 64.4 Operations on Elements

### 64.4.1 Equality and Comparison

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `u eq v` | Returns true if and only if the underlying concrete group elements for `u` and `v` are equal. | — |
| `u ne v` | Returns true if and only if the underlying concrete group elements for `u` and `v` are not equal. | — |

### 64.4.2 Attributes of Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Parent(u)` | The parent group `G` of the element `u`. | — |
| `UnderlyingElement(u)` | The concrete group element corresponding to the BB-group element `u`. Use this to extract the result for further computation within the underlying group. | — |
| `Order(u)` | The order of the underlying concrete group element of `u`. | — |

*Worked example: H64E1 (finding standard generators of M24 from an anonymous black-box group via `PseudoRandom`, `Order`, and conjugation; extracting concrete elements with `UnderlyingElement`).*

---

## 64.5 Set-Theoretic Operations

### 64.5.1 Membership and Equality

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `g in G` | Return true if and only if `G` is the parent group of `g`, or the parent group of `g` is a subgroup of `G`. | — |

### 64.5.2 Set Operations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PseudoRandom(G)` | Return a pseudo-random element of the BB-group `G`. | Product-replacement with accumulator. |
| `Rep(G)` | A representative element of `G`. | — |

### 64.5.3 Coercions Between Related Groups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `G ! g` | Given an element `g` belonging to a subgroup of the BB-group `G`, rewrite `g` as an element of `G`. | — |

---

## 64.6 Bibliography

No bibliography entries appear in Chapter 64. The chapter contains no attributed algorithm
references.

---

## Algorithm-to-function quick reference

| Algorithm / method | Functions |
|--------------------|-----------|
| Natural black-box wrapping of a concrete group | `NaturalBlackBoxGroup` |
| Product-replacement with accumulator (pseudo-random element generation) | `PseudoRandom` |
| Inherited group arithmetic from underlying concrete group | `*`, `^` (power), `^` (conjugate), `( , )` (commutator) |
| Equality via underlying element comparison | `eq`, `ne` |
| Access to underlying concrete element | `UnderlyingElement` |
