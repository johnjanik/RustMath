# Chapter 9 — Sets

**Handbook part:** II — Sets, Sequences, and Mappings
**Handbook pages:** 165–190 (PDF pages 296–323)

---

## Scope and overview

A set in Magma is a (usually unordered) collection of objects belonging to some common structure called the *universe* of the set. Chapter 9 covers four distinct set types and their full complement of constructors, accessors, modifiers, boolean operators, and combinatorial operations:

1. **Enumerated sets** (`SetEnum`) — finite sets whose elements are stored explicitly (except for arithmetic progressions, which are stored lazily). Specified by listing elements, by comprehension expressions over finite structures, or by arithmetic progressions.

2. **Formal sets** (`SetFormal`) — potentially infinite sets defined implicitly by a carrier set and a membership predicate. Only union, intersection, difference, symmetric difference, and membership testing are supported.

3. **Indexed sets** (`SetIndx`) — restricted enumerated sets with a numbering on elements; allow index-based access (`S[i]`), `Index`/`Position` lookup, and sequence-like appending/pruning. Equality ignores the index.

4. **Multisets** (`SetMulti`) — enumerated sets with element repetition; the `^^` operator specifies multiplicity. All finite.

The four types are not interconvertible via the binary operators (no mixing); conversion functions are provided. Universe compatibility rules for enumerated, indexed, and multisets are shared with sequences (see Chapter 8). Formal sets are additionally restricted to a single carrier structure.

No algorithmic bibliography is associated with this chapter; the content is definitional and language-specification in nature.

---

## 9.1 Introduction

### 9.1.1 Enumerated Sets

Enumerated sets are finite and can be specified in three basic ways: by listing all elements; by a set comprehension expression over a finite structure; or by an arithmetic progression. Arithmetic-progression sets store elements lazily until a modification forces explicit storage.

### 9.1.2 Formal Sets

A formal set consists of the subset of elements of some carrier set (structure) for which a given predicate is true. The only permitted set-theoretic operations are union, intersection, difference, symmetric difference, and element membership testing.

### 9.1.3 Indexed Sets

Indexed sets are enumerated sets equipped with an index map numbering the elements. They support element membership testing and sequence-like operations (index-based access, `Index`/`Position`, append, prune). Equality testing ignores the indexing.

### 9.1.4 Multisets

Multisets are enumerated sets allowing repeated elements. The number of times an object `x` occurs is its *multiplicity*. The `^^` operator (`x^^n`) specifies element `x` with multiplicity `n` in constructors and function arguments.

### 9.1.5 Compatibility

Binary set operators require both operands to be of the same type (no mixing of enumerated, formal, indexed, or multiset). Converting an enumerated set to a formal set is done via `{! x in R !}`. Functions exist to convert between enumerated sets, indexed sets, multisets, and sequences. Universe compatibility rules follow Chapter 8.

### 9.1.6 Notation

| Symbol | Meaning |
|--------|---------|
| `U` | Universe: any Magma structure |
| `E` | Carrier set for enumerated sets: any enumerable structure |
| `F` | Carrier set for formal sets: any structure for which `in` is defined |
| `x` | Free variable ranging over `E` (or `F`) |
| `P` | Boolean expression (usually involving `x`, `x1`, …, `xk`) |
| `e` | Expression (usually involving `x`, `x1`, …, `xk`) |

---

## 9.2 Creating Sets

The customary braces `{ }` delimit enumerated sets. Formal sets use `{! !}`. Indexed sets use `{@ @}`. Multisets use `{* *}`.

### 9.2.1 The Formal Set Constructor

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `{! x in F \| P(x) !}` | Form the formal set consisting of the subset of elements `x` of carrier set `F` for which `P(x)` is true. If `P` is always true, abbreviate to `{! x in F !}`. The universe of the resulting formal set equals `F`. | — |

### 9.2.2 The Enumerated Set Constructor

All general constructors have an optional universe `U` up front allowing explicit specification of the coercion target. An error results if any element cannot be coerced into `U`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `{ }` | The null (empty) enumerated set with no universe defined. | — |
| `{ U \| }` | The empty enumerated set with universe `U`. | — |
| `{ e1, e2, ..., en }` | Enumerated set of elements `a1, …, an` defined by the expressions `e1, …, en`, all automatically coercible into a common structure `U`. | — |
| `{ U \| e1, e2, ..., en }` | As above, but coerces all elements into the explicitly given universe `U`. | — |
| `{ e(x) : x in E \| P(x) }` | Comprehension: the set of values `e(x)` for all `x ∈ E` satisfying predicate `P(x)`. `E` must be finite and enumerable. `P` may be omitted if always true. | — |
| `{ U \| e(x) : x in E \| P(x) }` | As above, with all values coerced into `U`. `P` may be omitted if always true. | — |
| `{ e(x1,...,xk) : x1 in E1, ..., xk in Ek \| P(x1,...,xk) }` | Multi-variable comprehension over the Cartesian product `E1 × … × Ek`. Successive identical structures may be abbreviated `xi, xi+1 in Ei`. `P` may be omitted if always true. | — |
| `{ U \| e(x1,...,xk) : x1 in E1, ..., xk in Ek \| P(x1,...,xk) }` | As above, coercing all values into `U`. | — |

*Worked examples: H9E1 (creating a set of integers vs. rationals using universe specification); H9E2 (multi-variable comprehension — finding integer near-misses to Fermat's Last Theorem).*

### 9.2.3 The Indexed Set Constructor

Creation mirrors the enumerated set constructors; delimiters are `{@ @}`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `{@ @}` | The null (empty) indexed set with no universe defined. | — |
| `{@ U \| @}` | The empty indexed set with universe `U`. | — |
| `{@ e1, e2, ..., en @}` | Indexed set of elements `a1, …, an`, all coercible into a common structure `U`. | — |
| `{@ U \| e1, e2, ..., em @}` | As above, with universe explicitly `U`. | — |
| `{@ e(x) : x in E \| P(x) @}` | Comprehension over `E`; resulting indexed set has the order in which elements are encountered. `P` may be omitted if always true. | — |
| `{@ U \| e(x) : x in E \| P(x) @}` | As above, coercing into `U`. `P` may be omitted if always true. | — |
| `{@ e(x1,...,xk) : x1 in E1, ..., xk in Ek \| P(x1,...,xk) @}` | Multi-variable comprehension for indexed sets. Successive identical structures may be abbreviated. `P` may be omitted if always true. | — |
| `{@ U \| e(x1,...,xk) : x1 in E1, ..., xk in Ek \| P(x1,...,xk) @}` | As above, coercing into `U`. | — |

*Worked example: H9E3 (using indexed sets to retrieve both the pair and the cube root index in the near-Fermat search).*

### 9.2.4 The Multiset Constructor

Creation mirrors enumerated sets; delimiters are `{* *}`. The `^^` operator may be used in element lists to specify multiplicities.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `{* *}` | The null (empty) multiset with no universe defined. | — |
| `{* U \| *}` | The empty multiset with universe `U`. | — |
| `{* e1, e2, ..., en *}` | Multiset of elements `a1, …, an` (with repetition); elements coercible into a common structure. The `^^` multiplicity operator may appear. | — |
| `{* U \| e1, e2, ..., em *}` | As above, coercing into `U`. | — |
| `{* e(x) : x in E \| P(x) *}` | Comprehension multiset: `e(x)` for each `x ∈ E` with `P(x)` true; repetitions retained. `P` may be omitted if always true. | — |
| `{* U \| e(x) : x in E \| P(x) *}` | As above, coercing into `U`. `P` may be omitted if always true. | — |
| `{* e(x1,...,xk) : x1 in E1, ..., xk in Ek \| P(x1,...,xk) *}` | Multi-variable comprehension multiset. Successive identical structures may be abbreviated. `P` may be omitted if always true. | — |
| `{* U \| e(x1,...,xk) : x1 in E1, ..., xk in Ek \| P(x1,...,xk) *}` | As above, coercing into `U`. | — |

*Worked example: H9E4 (multiset of repeated integers; frequency of digits in the first 1000 digits of π using `{* I[i] : i in [1..#I] *}`).*

### 9.2.5 The Arithmetic Progression Constructors

Special constructors for enumerated sets of integers in arithmetic progression; stored efficiently without explicit enumeration until modified. Only the integer ring is a valid universe.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `{ i..j }` / `{ U \| i..j }` | Enumerated set `{i, i+1, i+2, …, j}` of integers. If `j < i`, the empty set is created. The only valid universe `U` is the integer ring. | — |
| `{ i .. j by k }` / `{ U \| i .. j by k }` | Enumerated set of the arithmetic progression `i, i+k, i+2k, …` up to `j` (inclusive), with step `k ≠ 0`. If `k > 0` and `j < i`, or `k < 0` and `j > i`, the empty set is created. | — |

*Worked example: H9E5 (demonstrating that `{ FiniteField(13) | 1..10 }` is a runtime error; correct approaches use comprehension or `PowerSet` coercion).*

---

## 9.3 Power Sets

The `PowerSet` family returns structures comprising all subsets of a given structure `R`; primarily useful as parent structures for set constructors. Permitted operations on power sets: printing, membership testing, and coercion.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PowerSet(R)` | The structure comprising all enumerated subsets of structure `R`. | — |
| `PowerIndexedSet(R)` | The structure comprising all indexed subsets of structure `R`. | — |
| `PowerMultiset(R)` | The structure comprising all submultisets of structure `R`. | — |
| `PowerFormalSet(R)` | The structure comprising all formal subsets of structure `R`. | — |
| `S in P` (P a `PowerSet`) | Returns `true` if enumerated set `S` is in power set `P` (i.e., all elements of `S` are in or coercible into `R`). | — |
| `S in P` (P a `PowerIndexedSet`) | Returns `true` if indexed set `S` is in the power indexed set `P`. | — |
| `S in P` (P a `PowerMultiset`) | Returns `true` if multiset `S` is in the power multiset `P`. | — |
| `P ! S` (P a `PowerSet`) | Returns a set with universe `R` consisting of the elements of set `S` coerced into `R`. Error if any element cannot be coerced. | — |
| `P ! S` (P a `PowerIndexedSet`) | Returns an indexed set with universe `R` consisting of the elements of `S` coerced into `R`. | — |
| `P ! S` (P a `PowerMultiset`) | Returns a multiset with universe `R` consisting of the elements of `S` coerced into `R`. | — |

*Worked example: H9E6 (constructing `PowerSet({ 1..10 })`, testing membership, and using `P ! F` to change universe from Rational Field to `{ 1..10 }`).*

### 9.3.1 The Cartesian Product Constructors

Cartesian products of sets (or any structures) are created with `car< >` and `CartesianProduct( )`, but the result is of type `CartesianProduct` (not a set) and its elements are tuples. See Chapter 11 for details.

---

## 9.4 Sets from Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Set(M)` | Given a finite structure `M` that allows explicit enumeration of its elements, returns the enumerated set containing all its elements, with `M` as universe. | — |
| `FormalSet(M)` | Given a structure `M`, returns the formal set consisting of all its elements. | — |

---

## 9.5 Accessing and Modifying Sets

Enumerated sets can be modified by inserting or removing elements. Indexed sets additionally support sequence-like access by index.

### 9.5.1 Accessing Sets and their Associated Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `#R` | Cardinality of enumerated, indexed, or multiset `R`. For multisets, repetitions count (result may exceed the number of distinct elements). | — |
| `Category(S)` / `Type(S)` | The Magma category of `S`. For sets: `SetEnum`, `SetIndx`, `SetMulti`, or `SetFormal`. For power sets: `PowSetEnum`, `PowSetIndx`, `PowSetMulti`. | — |
| `Parent(R)` | Returns the parent structure of `R` (the power set / structure of all sets over the universe of `R`). | — |
| `Universe(R)` | Returns the common structure (universe) to which all elements of the set `R` belong. Signals an error if `R` is the null set. | — |
| `Index(S, x)` / `Position(S, x)` | For indexed set `S` and element `x`: returns the index `i` such that `S[i] = x`, or `0` if `x ∉ S`. Attempts coercion into the universe of `S` if `x` is not already there. | — |
| `S[i]` | Returns the `i`-th element of indexed set `S`. Error if `i < 1` or `i > #S`. Indexing on the left-hand side is not allowed. | — |
| `S[I]` | Returns the indexed set `{S[i1], …, S[ir]}` for integer sequence `I`. Error if any term of `I` is out of range. If `I` is empty, returns the empty set with the same universe as `S`. | — |

*Worked example: H9E7 (indexed set of sets; demonstrating `#`, `Universe`, `Parent`, `Category`, `Index`, element access).*

### 9.5.2 Selecting Elements of Sets

`random{ }` selects a random element without constructing the full set first. `rep{ }` returns a representative element and aborts construction as soon as one is found. Note: `random{ e(x) : x in E | P(x) }` returns `e(y)` for a *random* `y ∈ E` satisfying `P`, not a random element of the resulting set of values.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Random(R)` | A random element of enumerated, indexed, or multiset `R`. Equal probability for enumerated/indexed sets; weighted by multiplicity for multisets. Successive calls return independently chosen elements. Error if `R` is empty. | — |
| `random{ e(x) : x in E \| P(x) }` | Returns `e(y)` for a randomly chosen `y ∈ E` satisfying `P(y)`. `P` may be omitted if always true. | — |
| `random{ e(x1,...,xk) : x1 in E1, ..., xk in Ek \| P(x1,...,xk) }` | Returns `e(y1,…,yk)` for a randomly chosen `<y1,…,yk> ∈ E1 × … × Ek` satisfying `P`. `P` may be omitted; successive identical structures may be abbreviated. | — |
| `Representative(R)` / `Rep(R)` | An arbitrary element chosen from enumerated, indexed, or multiset `R`. | — |
| `ExtractRep(~R, ~r)` | Assigns an arbitrary element of enumerated set `R` to `r` and removes it from `R` (both `R` and `r` are modified via reference). Error if `R` is empty. | — |
| `rep{ e(x) : x in E \| P(x) }` | Returns `e(y)` for the *first* element `y ∈ E` satisfying `P(y)`. Error if no element satisfies `P`. | — |
| `rep{ e(x1,...,xk) : x1 in E1, ..., xk in Ek \| P(x1,...,xk) }` | Returns `e(y1,…,yk)` for the first `<y1,…,yk> ∈ E1 × … × Ek` satisfying `P`. Error if none exists. `P` may be omitted; successive identical structures may be abbreviated. | — |
| `Minimum(S)` / `Min(S)` | Returns the minimum element of non-empty enumerated, indexed, or multiset `S` (requires `lt` and `eq` defined on the universe). For indexed sets, also returns the position. | — |
| `Maximum(S)` / `Max(S)` | Returns the maximum element of non-empty enumerated, indexed, or multiset `S`. For indexed sets, also returns the position. | — |
| `Hash(x)` | Returns the hash value of Magma object `x` as used by the set machinery: a fixed non-negative integer ≤ max C `unsigned long`. Equal objects always have equal hash values, regardless of internal structure. | — |

*Worked example: H9E8 (finding a random primitive element of `GF(p)` via `Random(proots)` vs. `random{ x : x in F | IsPrimitive(x) }`, illustrating the efficiency advantage of `random{ }`). H9E9 (using `ExtractRep` to search for cube-sum representations while iterating through a shrinking set).*

### 9.5.3 Modifying Sets

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Include(~S, x)` / `Include(S, x)` | Adds element `x` to enumerated, indexed, or multiset `S`. For enumerated/indexed sets, no-op if `x` is already present (indexed set appends at the end). For multisets, increases the multiplicity of `x`. Attempts coercion into the universe. Procedural form (`~S`) modifies `S` in place (more efficient); functional form returns the new set. | — |
| `Exclude(~S, x)` / `Exclude(S, x)` | Removes element `x` from `S`. For enumerated sets, no-op if `x ∉ S`. For multisets, decreases the multiplicity. Attempts coercion. Procedural form (more efficient) vs. functional form. | — |
| `ChangeUniverse(~S, V)` / `ChangeUniverse(S, V)` | Constructs a new set of the same type as `S` with universe `V`, coercing all elements of `S` into `V`. Procedural form (`~S`) modifies `S` in place (more efficient). | — |
| `CanChangeUniverse(S, V)` | Attempts to construct a new set `T` with universe `V` by coercing elements of `S`. Returns `true, T` on success, or `false` on failure. | — |
| `SetToIndexedSet(E)` | Given enumerated set `E`, returns an indexed set with the same elements and universe. | — |
| `IndexedSetToSet(S)` / `Isetset(S)` | Given indexed set `S`, returns an enumerated set with the same elements and universe. | — |
| `IndexedSetToSequence(S)` / `Isetseq(S)` | Given indexed set `S`, returns a sequence with the same elements and universe. | — |
| `MultisetToSet(S)` | Given multiset `S`, returns an enumerated set with the same (distinct) elements and universe. | — |
| `SetToMultiset(E)` | Given enumerated set `E`, returns a multiset with the same elements and universe (all multiplicities 1). | — |
| `SequenceToMultiset(Q)` | Given enumerated sequence `Q`, returns a multiset with the same elements and universe. | — |

*Worked example: H9E10 (using `Include` and `Exclude` to find cubes summing to elements of a given set `R`, then reconstructing the pairs).*

---

## 9.6 Operations on Sets

### 9.6.1 Boolean Functions and Operators

When elements are extracted from a set, their parent is the universe of the set (not the set itself). Equality testing on set elements is therefore equality in the underlying algebraic structure. Subset testing requires the left operand to be an enumerated or indexed set (not formal).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsNull(R)` | Returns `true` iff enumerated, indexed, or multiset `R` is empty and has no universe defined. | — |
| `IsEmpty(R)` | Returns `true` iff enumerated, indexed, or multiset `R` is empty. | — |
| `x eq y` | Returns `true` iff elements `x` and `y` are equal in their common overstructure `W` (see Chapter 8). | — |
| `x ne y` | Returns `true` iff elements `x` and `y` are distinct in their common overstructure. | — |
| `x in R` | Returns `true` iff element `x` is a member of set `R`. Attempts coercion into the universe of `R`; error if coercion fails. | — |
| `x notin R` | Returns `true` iff element `x` is not a member of set `R`. Attempts coercion; error if coercion fails. | — |
| `R subset S` | Returns `true` if enumerated, indexed, or multiset `R` is a subset of set `S`. For multisets, multiplicity of each element in `R` must be ≤ its multiplicity in `S`. Attempts coercion; error if coercion fails. | — |
| `R notsubset S` | Returns `true` if `R` is not a subset of `S`. | — |
| `R eq S` | Returns `true` iff `R` and `S` are identical sets (enumerated, indexed, or multisets). For indexed sets, the indexing is irrelevant. For multisets, multiplicities must also match. Attempts coercion. | — |
| `R ne S` | Returns `true` iff `R` and `S` are distinct sets. | — |
| `IsDisjoint(R, S)` | Returns `true` iff enumerated, indexed, or multisets `R` and `S` are disjoint. Attempts coercion; error if coercion fails. | — |

### 9.6.2 Binary Set Operators

For each operator below, `R` and `S` must be the same set type. For formal sets, both must have been constructed with the same carrier structure `F`. For enumerated, indexed, or multisets, their universes must be compatible (see Chapter 8). Note: `{! x in R !}` converts enumerated set `R` to a formal set.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `R join S` | Union of `R` and `S`. For multisets, matching multiplicities are added. | — |
| `R meet S` | Intersection of `R` and `S`. For multisets, the minimum of matching multiplicities is retained. | — |
| `R diff S` | Set difference: elements of `R` not in `S`. For multisets, elements of `S` are removed from `R` the appropriate number of times. | — |
| `R sdiff S` | Symmetric difference: elements in `R` or `S` but not both. Equivalently, `(R diff S) join (S diff R)`. | — |

*Worked example: H9E11 (demonstrating `join`, `meet`, `diff`, `sdiff` on `{1,2,3}` and `{1, 1/2, 1/3}`).*

### 9.6.3 Other Set Operations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Multiplicity(S, x)` | Returns the multiplicity of element `x` in multiset `S`. Returns `0` if `x ∉ S`. | — |
| `Multiplicities(S)` | Returns the sequence of multiplicities of distinct elements of multiset `S`, in internal enumeration order. | — |
| `Subsets(S)` | Returns the set of all subsets of `S`. | — |
| `Subsets(S, k)` | Returns the set of all subsets of `S` of size `k`. Empty if `k > #S`. | — |
| `RandomSubset(S, k)` | Returns a random subset of `S` of size `k`. Error if `k > #S`. | — |
| `Multisets(S, k)` | Returns the set of all multisets consisting of `k` (not necessarily distinct) elements of `S`. | — |
| `Subsequences(S, k)` | Returns the set of all sequences of length `k` with elements from `S`. | — |
| `Permutations(S)` | Returns the set of all permutations of the elements of `S` (stored as sequences). | — |
| `Permutations(S, k)` | Returns the set of all permutations (stored as sequences) of each subset of `S` of cardinality `k`. | — |

---

## 9.7 Quantifiers

`exists` and `forall` allow short-circuit evaluation: construction of the underlying set is aborted as soon as the answer is determined. `exists(t){ e(x) : x in E | P(x) }` assigns `e(y)` to `t` for the first `y` satisfying `P`, not a random element of `{e(x)}`. `forall` aborts and assigns the counterexample as soon as one is found.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `exists(t){ e(x) : x in E \| P(x) }` | Returns `true` if any `x ∈ E` satisfies `P(x)`, assigning `e(y)` to `t` for the first such `y`. Returns `false` and leaves `t` unassigned if none found. The clause `(t)` may be omitted. `P` may be omitted if always true. | Short-circuit iteration |
| `exists(t1,...,tr){ e(x) : x in E \| P(x) }` | As above; `e(y)` must be a tuple of length `r`; each `ti` is assigned the corresponding component. | Short-circuit iteration |
| `exists(t){ e(x1,...,xk) : x1 in E1, ..., xk in Ek \| P(x1,...,xk) }` | Returns `true` if any `<y1,…,yk> ∈ E1 × … × Ek` satisfies `P`, assigning `e(y1,…,yk)` to `t`. Successive identical structures may be abbreviated. `P` may be omitted. | Short-circuit iteration |
| `exists(t1,...,tr){ e(x1,...,xk) : x1 in E1, ..., xk in Ek \| P }` | Multi-variable multi-target form; `e(y1,…,yk)` is a tuple of length `r`. | Short-circuit iteration |
| `forall(t){ e(x) : x in E \| P(x) }` | Returns `true` if `P(x)` is true for every `x ∈ E`. If `P(y)` is false for some `y`, returns `false` and assigns `e(y)` to `t`. `t` left unassigned if result is `true`. `(t)` may be omitted. `P` may be omitted. | Short-circuit iteration |
| `forall(t1,...,tr){ e(x) : x in E \| P(x) }` | As above with tuple assignment to `t1,…,tr` upon counterexample. | Short-circuit iteration |
| `forall(t){ e(x1,...,xk) : x1 in E1, ..., xk in Ek \| P(x1,...,xk) }` | Multi-variable form of `forall`. | Short-circuit iteration |
| `forall(t1,...,tr){ e(x1,...,xk) : x1 in E1, ..., xk in Ek \| P }` | Multi-variable multi-target form. | Short-circuit iteration |

*Worked example: H9E12 (nested `exists` checking whether integers are sums of cubes; illustrating that `t` remains undeclared when the result is `false`). H9E13 (nested `forall`/`exists` to find the first `m ≡ 1 (mod 4)` in `[5..1000]` that is not `m + z` for any `|z| ≤ 1` expressible as a sum of two squares; answer: `m = 77`).*

---

## 9.8 Reduction and Iteration over Sets

Both enumerated and indexed sets allow enumeration of their elements via `x in S`; formal sets do not. Indexed sets enumerate in index order.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `x in S` | Enumerate the elements of enumerated or indexed set `S`. Used in `for` loops and in set/sequence constructors. | — |
| `&o S` | Reduction: given set `S = {a1, a2, …, an}` and an associative binary operator `◦ : U × U → U`, returns `ai1 ◦ ai2 ◦ … ◦ ain` for some permutation of indices. Supported operators for enumerated sets: `+`, `*`, `and`, `or`, `join`, `meet`; for indexed sets: `+`, `*`, `and`, `or`. Warning: result is unambiguous only if `◦` is commutative on the arguments. | — |

Return values for reduction on empty/null sets:

| Operator | Empty set (with universe `U`) | Null set |
|----------|-------------------------------|----------|
| `&+` | `U ! 0` | error |
| `&*` | `U ! 1` | error |
| `&and` | `true` | `true` |
| `&or` | `false` | `false` |
| `&join` | empty set | null set |
| `&meet` | error | error |

*Worked example: H9E14 (recursive `choose` function computing all `k`-element subsets of a set `S` using `&join` and set comprehension).*

---

### Algorithm-to-function quick reference

| Algorithm / operation | Functions |
|-----------------------|-----------|
| Set construction (enumerated) | `{ }`, `{ U | }`, `{ e1,...,en }`, `{ e(x) : x in E | P(x) }`, and multi-variable variants |
| Set construction (formal) | `{! x in F | P(x) !}` |
| Set construction (indexed) | `{@ @}` and variants |
| Set construction (multiset) | `{* *}` and variants |
| Arithmetic-progression construction | `{ i..j }`, `{ i..j by k }` and universe variants |
| Power-set construction and coercion | `PowerSet`, `PowerIndexedSet`, `PowerMultiset`, `PowerFormalSet`, `P ! S`, `S in P` |
| Sets from structures | `Set`, `FormalSet` |
| Cardinality / metadata | `#`, `Category`, `Type`, `Parent`, `Universe` |
| Index-based access | `Index`, `Position`, `S[i]`, `S[I]` |
| Element selection | `Random`, `random{ }`, `Representative`, `Rep`, `ExtractRep`, `rep{ }` |
| Min / max | `Minimum`, `Min`, `Maximum`, `Max` |
| Hashing | `Hash` |
| Modification | `Include`, `Exclude`, `ChangeUniverse`, `CanChangeUniverse` |
| Type conversion | `SetToIndexedSet`, `IndexedSetToSet`, `Isetset`, `IndexedSetToSequence`, `Isetseq`, `MultisetToSet`, `SetToMultiset`, `SequenceToMultiset` |
| Boolean / membership | `IsNull`, `IsEmpty`, `eq`, `ne`, `in`, `notin`, `subset`, `notsubset`, `IsDisjoint` |
| Binary set operations | `join`, `meet`, `diff`, `sdiff` |
| Multiplicity (multisets) | `Multiplicity`, `Multiplicities` |
| Combinatorial enumeration | `Subsets`, `RandomSubset`, `Multisets`, `Subsequences`, `Permutations` |
| Quantifiers (short-circuit) | `exists{ }`, `forall{ }` |
| Iteration / reduction | `x in S`, `&+`, `&*`, `&and`, `&or`, `&join`, `&meet` |
