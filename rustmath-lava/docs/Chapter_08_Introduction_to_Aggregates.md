# Chapter 8 — Introduction to Aggregates

**Handbook part:** II — Sets, Sequences, and Mappings
**Handbook pages:** 155–161 (PDF pages 288–295)

---

## Scope and overview

Chapter 8 is a conceptual introduction to the four main aggregate types in Magma — sets,
sequences, tuples, and lists — and to the rules governing their universes, parents, and
compatibility. It does not introduce standalone intrinsics; instead it explains the semantic
machinery that underpins all subsequent chapters on aggregates.

Sets collect objects from a common structure and emphasise membership testing; each
element appears at most once. Sequences also require a common structure but emphasise
ordered access and allow repeated elements. Both enumerated forms are finite and store
elements explicitly (with the exception of arithmetic progressions); formal counterparts
may be infinite and use a Boolean predicate for membership. Indexed sets are a hybrid
offering set semantics with sequence-style indexing. Tuples are fixed-length elements of
Cartesian products, with each component constrained to its factor structure. Lists are
arbitrary finite ordered collections with no type restriction, intended for storage rather
than performance-critical access.

The chapter explains the universe–parent duality: the *universe* of a set or sequence is
the common parent of all its elements (stored once rather than per element), while the set
or sequence itself is parented by the corresponding *power set* or *power sequence*. Universe
coercion (automatic or explicit via `!`) and overstructure resolution govern when binary
operations between sets or sequences of different universes are legal. The nested-aggregate
section covers multi-indexing syntax for nested sequences and the distinction between
multi-indexing and subsequence extraction.

---

## 8.1 Introduction

This section gives an overview of the four aggregate types (sets, sequences, tuples, lists)
and their intended purposes. It is purely prose; no intrinsics are introduced.

---

## 8.2 Restrictions on Sets and Sequences

The universe of a set or sequence is the common structure to which all elements must
belong. When a set or sequence is constructed without an explicit universe declaration,
Magma determines a universe automatically through coercion (e.g., a set containing both
integers and a rational will have the rational field as its universe). Modifying a non-null
set or sequence does not change its universe; new elements must be coercible into the
current universe. Binary operations (e.g., `join`, `cat`, `meet`) require *compatible*
universes — either the same, or related by automatic coercion or a common overstructure.

### 8.2.1 Universe of a Set or Sequence

The universe may be declared explicitly in the constructor (e.g., `{ IntegerRing() | 1, 2, 3 }`),
inferred from the elements, or left undefined (the *null* set/sequence, written `{ }` or `[ ]`).
A null set/sequence has no universe. An empty set/sequence created with an explicit universe
declaration, or one that becomes empty during computation, retains its universe.

When the universe is itself a set or sequence, the actual parent of an element is the
universe of that universe (recursively), until a non-aggregate structure is reached. Thus
coercing into a sequence universe performs a bounds check against that sequence but
assigns the element the universe of the sequence as its parent.

This section is conceptual; no intrinsics are listed.

### 8.2.2 Modifying the Universe of a Set or Sequence

Once a non-null set or sequence `S` has been created, its universe is fixed. Mutations
(adding elements, changing entries in place) succeed only if the new element can be
coerced into the existing universe. To change the universe, the user must coerce `S`
explicitly into a new power set or power sequence via the `!` operator. The functions
`PowerSet` and `PowerSequence` construct those parent structures.

Compatibility rules for binary operations on `S` (universe `A`) and `T` (universe `B`):

1. Every set/sequence is compatible with the null set/sequence.
2. Two sets/sequences with the same universe are compatible.
3. `S` and `T` are compatible if elements of `A` can be automatically coerced into `B`, or vice versa.
4. More generally, they are compatible if Magma can automatically find a common overstructure for `A` and `B`.
5. Nested sets/sequences are compatible only when they are of the same depth and type (sets and sequences appear in the same recursive order in both) and their universes are compatible.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PowerSet(A)` | Returns the parent structure for sets of elements of structure `A` (the power set of `A`). Elements of the returned structure are all subsets of `A`. Used with `!` to recast a set's universe. | — |
| `PowerSequence(A)` | Returns the parent structure for sequences of elements of structure `A`. Used with `!` to recast a sequence's universe. | — |

### 8.2.3 Parents of Sets and Sequences

The universe of a set/sequence `S` is the common parent for its elements; `S` itself has a
parent too. The parent of a set with universe `A` is `PowerSet(A)`; the parent of a
sequence with universe `A` is `PowerSequence(A)`.

Rules for finding a common overstructure of `A` and `B` (when at least one is a set,
sequence, or its parent):

1. The overstructure of `A` and `B` equals that of `B` and `A`.
2. If `A` is the null set/sequence, the overstructure is `B`.
3. If `A` is a set/sequence with universe `U`, the overstructure of `A` and `B` is the overstructure of `U` and `B`; in particular, the overstructure of `A` and `A` is `U`.
4. If `A` is a power set, a common overstructure exists only if `B` is also a power set, in which case the overstructure is the power set of the overstructure of their respective universes. Likewise for power sequences.

This section is conceptual; no additional intrinsics are introduced beyond `PowerSet` and
`PowerSequence` described above.

---

## 8.3 Nested Aggregates

Enumerated sets and sequences may be arbitrarily nested (sets of sets, sequences of sets,
etc.). Tuples can be nested and freely mixed with sets and sequences provided the proper
Cartesian product parent can be created. Lists may be nested and may contain sets,
sequences, or tuples.

### 8.3.1 Multi-indexing

Because sequences (and lists) can be nested, assignment and mutation operators accept a
*multi-index* `i1, i2, …, ir` to reach `r` levels deep. For example, for
`S = [ [1, 2], [2, 3] ]`, the assignments `S[2][2] := 4` and `S[2,2] := 4` are equivalent.

Rules for multi-indexing:
- All `ij` must be greater than 0.
- An error is raised if any `ij` indexes beyond the length at level `j`, **except** that the
  last index `ir` may extend beyond the current length when used on the left-hand side of
  an assignment; intermediate positions become undefined.
- Multi-indexing is distinct from using a *sequence* as an index to extract a subsequence
  (e.g., `S[ [2, 3] ]`). These two constructs can be combined: `S[ [2,3], 2 ]` extracts
  the subsequence consisting of the second and third elements and then takes the second
  element of that subsequence (equivalent to `S[3]` in a flat case), while
  `S[ [2,3], [2] ]` returns that element wrapped in a one-element sequence.

This section is conceptual; no intrinsics are introduced.

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Universe construction and explicit coercion | `PowerSet`, `PowerSequence` |
| Automatic coercion / overstructure resolution | Underlying mechanism for `join`, `cat`, `meet`, `!`, and all set/sequence constructors |
| Multi-index assignment (nested sequences/lists) | Assignment syntax `S[i1, i2, …, ir] := v` (language feature, not a named intrinsic) |
