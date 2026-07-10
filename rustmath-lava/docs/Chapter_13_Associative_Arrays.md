# Chapter 13 — Associative Arrays

**Handbook part:** II — Sets, Sequences, and Mappings
**Handbook pages:** 229–231 (PDF pages 362–364)

---

## Scope and overview

An associative array in Magma is an array indexed by arbitrary elements of an index structure
`I`. Unlike ordinary sequences (which are indexed by integers), the indexing may be by any
objects coercible into `I`; these are called the **keys**. For each current key there is an
associated **value**. The values need not lie in a fixed universe — they may be of any type.

Two construction forms are provided: a null array whose index universe is determined by the
first assignment, and a typed array whose index universe is specified at creation time. The
index universe may be widened automatically when an element not coercible into the current
universe is assigned.

---

## 13.1 Introduction

Associative arrays provide an efficient, hash-based key-value store for arbitrary Magma
structures. Because the keys can be elements of any Magma structure (rationals, group
elements, ring elements, etc.), they generalise both ordinary sequences and mathematical
functions. No algorithm references are given in this chapter; the data structure is a
fundamental Magma language feature.

---

## 13.2 Operations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AssociativeArray()` | Create the null associative array with no index universe. The first assignment to the array determines its index universe. | — |
| `AssociativeArray(I)` | Create the empty associative array with index universe `I`. | — |
| `A[x] := y` | Set the value in `A` associated with index `x` to `y`. If `x` is not coercible into the current index universe `I` of `A`, an attempt is first made to lift the index universe of `A` to contain both `I` and `x`. | — |
| `A[x]` | Given an index `x` coercible into the index universe `I` of `A`, return the value associated with `x`. Raises an error if `x` is not among the current keys of `A`. | — |
| `IsDefined(A, x)` | Given an index `x` coercible into the index universe `I` of `A`, return whether `x` is currently in the keys of `A`; if so, also return the value `A[x]`. | — |
| `Remove(~A, x)` | (Procedure.) Destructively remove the value indexed by `x` from array `A`. If `x` is not present as an index, nothing happens (no error is raised). | — |
| `Universe(A)` | Return the index universe `I` of associative array `A`, in which the current keys lie. | — |
| `Keys(A)` | Return the current keys of `A` as a set. Constructs a new copy of the key set; not intended as a fast accessor — call only when the full set is needed. | — |

*Worked examples: H13E1 (associative array indexed by rationals, then by elements of the symmetric group S3; demonstrates assignment, lookup, `IsDefined`, `Remove`, `Keys`, `Universe`).*

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Hash-based key-value store (built-in language primitive) | `AssociativeArray`, `A[x] := y`, `A[x]`, `IsDefined`, `Remove`, `Universe`, `Keys` |
