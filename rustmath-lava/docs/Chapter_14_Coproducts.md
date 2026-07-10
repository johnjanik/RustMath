# Chapter 14 — Coproducts

**Handbook part:** II — Sets, Sequences, and Mappings
**Handbook pages:** 235–237 (PDF pages 366–371)

---

## Scope and overview

Coproducts provide a unified container for objects of entirely different types. A coproduct
holds elements drawn from any number of constituent structures (its "summands"), acting as a
single parent for all of them. The proper parent of each element is recorded internally and
restored whenever the element is retrieved from the coproduct. Coproducts are therefore useful
when heterogeneous collections must be treated uniformly.

The chapter covers construction of coproducts and coproduct elements, access to structural
information (injection maps, constituents, element indices), retrieval of elements back to their
original parents, flattening of nested coproducts, and construction of the universal map from a
coproduct to another structure.

---

## 14.1 Introduction

Coproducts can be useful in various situations, as they may contain objects of entirely
different types. Although the coproduct structure serves as a single parent for such diverse
objects, the proper parents of the elements are recorded internally and restored whenever the
element is retrieved from the coproduct.

---

## 14.2 Creation Functions

There are two versions of the coproduct constructor. Ordinarily, coproducts are constructed
from a list of structures — the *constituents* of the coproduct. A single sequence argument is
also accepted to allow convenient creation of coproducts of parameterised families of
structures.

### 14.2.1 Creation of Coproducts

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `cop< S1, S2, ..., Sk >` | Given a list of two or more structures S1, S2, ..., Sk, creates and returns their coproduct C together with a sequence of injection maps [m1, m2, ..., mk] where mi : Si → C. | — |
| `cop< [ S1, S2, ..., Sk ] >` | As above but accepts a sequence (rather than an explicit list) of constituent structures. Useful for parameterised families. Returns C and the sequence of injection maps. | — |

### 14.2.2 Creation of Coproduct Elements

Coproduct elements are usually created by the injection maps returned as the second return
value from the `cop<>` constructor. The bang (`!`) operator may also be used, but only if the
type of the relevant constituent is unique within the particular coproduct.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `m(e)` | Given a coproduct injection map m and an element e of the corresponding constituent, returns the coproduct element version of e. | — |
| `C ! e` | Given a coproduct C and an element e of one of the constituents of C whose type is unique within C, returns the coproduct element version of e. | — |

---

## 14.3 Accessing Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Injections(C)` | Given a coproduct C, returns the sequence of injection maps (the same sequence returned as the second argument from the `cop<>` constructor). | — |
| `#C` | Given a coproduct C, returns the number of constituents of C. | — |
| `Constituent(C, i)` | Given a coproduct C and an integer i between 1 and the length of C, returns the i-th constituent structure of C. | — |
| `Index(x)` | Given an element x from a coproduct C, returns the constituent number (index) of C to which x belongs. | — |

---

## 14.4 Retrieve

The function described here restores an element of a coproduct to its original state.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Retrieve(x)` | Given an element x of some coproduct C, returns x as an element of the structure that formed its parent before it was mapped into C. | — |

*Worked examples: H14E1 (basic uses of coproduct constructors and functions: creating a coproduct of the integer ring and strings, injecting elements, checking equality, and retrieving elements back to their original parents).*

---

## 14.5 Flattening

The function described here enables the "concatenation" of coproducts into a single one.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Flat(C)` | Given a coproduct C whose constituents may themselves be coproducts, returns the coproduct of the base structures considered in depth-first order. | — |

---

## 14.6 Universal Map

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `UniversalMap(C, S, [ n1, ..., nm ])` | Given maps n1, ..., nm from structures S1, ..., Sm that compose the coproduct C to some structure S, returns the universal map C → S. | — |

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Coproduct construction | `cop< >` (list and sequence forms) |
| Element injection | `m(e)`, `C ! e` |
| Structural access | `Injections`, `#`, `Constituent`, `Index` |
| Element retrieval to original parent | `Retrieve` |
| Nested coproduct flattening (depth-first) | `Flat` |
| Universal map construction | `UniversalMap` |
