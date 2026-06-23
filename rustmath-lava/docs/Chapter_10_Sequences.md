# Chapter 10 — Sequences

**Handbook part:** II — Sets, Sequences, and Mappings
**Handbook pages:** 193–212 (PDF pages 324–345)

---

## Scope and overview

A sequence in Magma is a linearly ordered collection of objects belonging to some common
structure called the *universe* of the sequence. There are two types:

- **Enumerated sequences** — all elements stored explicitly (always finite; length must be
  less than 2³⁰). An enumerated sequence of length *l* may be *complete* (all terms from
  index 1 to *l* defined) or *incomplete* (some interior terms undefined). Enumerated
  sequences of Booleans are stored as optimised bit-vectors.

- **Formal sequences** — elements stored implicitly by a predicate on a range set; membership
  can be tested but very few other operations are available. Formal sequences may be infinite.

Binary operators do not allow mixing of formal and enumerated types. Universe compatibility
rules for sequences are the same as for sets (Chapter 8): a common over-structure is sought
automatically, and an error results if none exists.

The chapter covers constructors (formal, enumerated, arithmetic-progression, and literal),
power sequences, access/selection/modification operators, predicates, recursion, reduction,
and iteration.

---

## 10.1 Introduction

### 10.1.1 Enumerated Sequences

An enumerated sequence of length *l* is an indefinitely sized array of which only finitely
many terms — including the *l*-th, but no term of larger index — have been defined to be
elements of some common structure. A sequence is *complete* if all terms from index 1 to *l*
are defined. Incomplete sequences are permitted as a programming convenience; most Magma
modules assume that any sequence passed in is complete.

Enumerated sequences of Booleans are highly optimised (stored as bit-vectors).

### 10.1.2 Formal Sequences

A formal sequence consists of those elements of some range set *F* for which a given predicate
is true. Only a very limited set of operations (notably membership testing) is available. Formal
sequences may be infinite.

### 10.1.3 Compatibility

Binary operators do not mix formal and enumerated sequence types. Compatibility of elements
and universe coercion follow the same rules as for sets (Chapter 8).

---

## 10.2 Creating Sequences

Square brackets delimit enumerated sequences; `[!` and `!]` delimit formal sequences.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `U` | Universe: any Magma structure (used as optional prefix in constructors). | — |
| `E` | Range set for enumerated sequences: any enumerable structure. | — |
| `F` | Range set for formal sequences: any structure supporting `in`. | — |

### 10.2.1 The Formal Sequence Constructor

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `[! x in F \| P(x) !]` | Create the formal sequence of elements `x` of `F` for which `P(x)` is true. If `P(x)` is true for every element, may be abbreviated to `[! x in F !]`. | — |

### 10.2.2 The Enumerated Sequence Constructor

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `[ ]` | The null sequence (empty, no universe specified). | — |
| `[ U \| ]` | The empty sequence with universe `U`. | — |
| `[ e1, e2, ..., en ]` | Given expressions `e1, …, en` evaluating to elements coercible into a single structure `U`, create the sequence `[a1, a2, …, an]`. The expression `x^^n` repeats `x` n times. | — |
| `[ U \| e1, e2, ..., em ]` | As above, with explicit universe `U`; an error results if any element is not coercible into `U`. | — |
| `[ e(x) : x in E \| P(x) ]` | Sequence of values `e(x)` for those `x ∈ E` with `P(x)` true. Abbreviated to `[ e(x) : x in E ]` when `P` is always true. | — |
| `[ U \| e(x) : x in E \| P(x) ]` | As above, coerced into universe `U`. | — |
| `[ e(x1,...,xk) : x1 in E1, ..., xk in Ek \| P(x1,...,xk) ]` | Sequence of values `e(x1,…,xk)` for tuples satisfying the predicate. Successive identical ranges may be abbreviated `xi, xi+1 in Ei`; predicate may be omitted if always true. | — |
| `[ U \| e(x1,...,xk) : x1 in E1, ..., xk in Ek \| P(x1,...,xk) ]` | As above, with all values coerced into `U`. | — |

### 10.2.3 The Arithmetic Progression Constructors

Since integer sequences arise frequently, special constructors create arithmetic progressions
efficiently. The universe must be the ring of integers, and Magma makes some effort to
preserve the arithmetic-progression storage format under subsequent operations.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `[ i..j ]` / `[ U \| i..j ]` | Enumerated sequence of integers `i, i+1, …, j`. If `j < i`, returns the empty sequence. `U`, if specified, must be the ring of integers. | — |
| `[ i .. j by k ]` / `[ U \| i .. j by k ]` | Sequence `i, i+k, i+2*k, …` (with `k ≠ 0`). If `k > 0`, last term ≤ j; if `k < 0`, last term ≥ j. Empty if the range is vacuous. `U`, if specified, must be the ring of integers. | — |

*Worked example: H10E1 (arithmetic progression constructors for sequences over non-integer rings).*

### 10.2.4 Literal Sequences

A literal sequence loads an enumerated sequence of integers very fast and space-efficiently.
Currently only integer literals are supported.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `\[ m1, ..., mn ]` | Given literal integers `m1, …, mn`, build the enumerated sequence `[m1, …, mn]` in a time- and space-efficient way. | — |

---

## 10.3 Power Sequences

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PowerSequence(R)` | Returns the structure comprising all enumerated sequences of elements of structure `R`. If `R` is itself a sequence or set, returns the power structure of its universe. Permitted operations on power sequences: printing, membership testing (`in`), and coercion (`!`). | — |
| `S in P` | Returns `true` if enumerated sequence `S` is an element of the power sequence `P` (i.e. all elements of `S` are in or coercible into the base structure `R`); `false` otherwise. | — |
| `P ! S` | Return a sequence with universe `R` consisting of the entries of enumerated sequence `S`, coerced into `R`. Error if any element cannot be coerced. | — |

*Worked example: H10E2 (PowerSequence of an integer range; coercion of rational sequences).*

---

## 10.4 Operators on Sequences

### 10.4.1 Access Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `#S` | The length of enumerated sequence `S` — the index of the last defined term. Length of the empty sequence is zero. | — |
| `Parent(S)` | The parent structure of sequence `S` — the power sequence over the universe of `S`. | — |
| `Universe(S)` | The common structure (universe) to which all elements of `S` belong. Error if `S` is the null sequence. | — |
| `S[i]` | The `i`-th term of sequence `S`. Error if `i ≤ 0`, or `i > #S + 1`, or `S[i]` is undefined. `i` may be a multi-index. Also valid as the left-hand side of an assignment `S[i] := x`; assigning beyond `#S` extends the sequence (with undefined intermediate terms). | — |

### 10.4.2 Selection Operators on Enumerated Sequences

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `S[I]` | Subsequence `[S[i1], …, S[ir]]` selected by the integer sequence `I`. Error if any term of `I` is out of range. Unlike `[ S[i] : i in I ]`, undefined entries in range are copied (not an error). | — |
| `Minimum(S)` / `Min(S)` | For a non-empty complete sequence `S` with `lt` and `eq` defined on its universe: returns the minimal element `s` and the first position `i` with `s = S[i]`. | — |
| `Maximum(S)` / `Max(S)` | For a non-empty complete sequence `S` with `gt` and `eq` defined: returns the maximal element and its first position. | — |
| `Index(S, x)` / `Index(S, x, f)` / `Position(S, x)` / `Position(S, x, f)` | Position of the first occurrence of `x` in `S`, or zero if not present. The two-argument form `f` starts the search at position `f`, saving time on repeated searches. | — |
| `Representative(R)` / `Rep(R)` | An arbitrary element chosen from enumerated sequence `R`. | — |
| `Random(R)` | A uniformly random element chosen from enumerated sequence `R`. Successive calls return independently chosen elements. Error if `R` is empty. | — |
| `Explode(R)` | For sequence `R` of length `r`, returns the `r` entries as separate return values (in order). | — |
| `Eltseq(R)` | Returns the enumerated sequence `R` itself (included for completeness/uniformity). | — |

### 10.4.3 Modifying Enumerated Sequences

Each modification is available both as a **procedure** (modifying `S` in place via reference
`~S`, more efficient as no copy is made) and as a **function** (returning the modified
sequence, leaving `S` unchanged).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Append(~S, x)` / `Append(S, x)` | Add `x` to the end of `S`, giving `[s1, …, sn, x]`. | — |
| `Exclude(~S, x)` / `Exclude(S, x)` | Remove the first occurrence of `x` from `S`; if `x` is not in `S`, `S` is unchanged. | — |
| `Include(~S, x)` / `Include(S, x)` | Add `x` to the end of `S` only if no term of `S` equals `x`; otherwise `S` is unchanged. | — |
| `Insert(~S, i, x)` / `Insert(S, i, x)` | Insert `x` at position `i`, shifting `S[i], …, S[n]` down one place. If `i > n`, the new length is `i` and intermediate terms are undefined. | — |
| `Insert(~S, k, m, T)` / `Insert(S, k, m, T)` | Replace terms `S[k], …, S[m]` with the terms of `T`, giving `[s1, …, sk-1, t1, …, tl, sm+1, …, sn]`. Error if `k ≤ 0` or `k > m+1`. If `T` is empty, terms `sk, …, sm` are deleted. | — |
| `Prune(~S)` / `Prune(S)` | Remove the last term of `S`, giving `[s1, …, sn-1]`. Error if `S` is empty. | — |
| `Remove(~S, i)` / `Remove(S, i)` | Remove the `i`-th term from `S`. Error if `i < 1` or `i > n`. | — |
| `Reverse(~S)` / `Reverse(S)` | Reverse the order of terms in the complete sequence `S`, giving `[sn, …, s1]`. | — |
| `Rotate(~S, p)` / `Rotate(S, p)` | Cyclically rotate complete sequence `S` by `p` positions: positive `p` rotates right, negative `p` rotates `|p|` positions left, zero does nothing. | — |
| `Sort(~S)` / `Sort(S)` | Sort complete sequence `S` into increasing order using quicksort, requiring `lt` and `eq` on the universe. | Quicksort. |
| `Sort(~S, C)` / `Sort(~S, C, ~p)` / `Sort(S, C)` | Sort `S` using comparison function `C(x, y)` returning negative/zero/positive. The three-return-value procedure form also sets the permutation `p` applied to `S`. | Quicksort with custom comparator. |
| `ParallelSort(~S, ~T)` | Sort `S` in place and simultaneously apply the same transpositions to `T`. | — |
| `Undefine(~S, i)` / `Undefine(S, i)` | Make the `i`-th term of `S` undefined. `i` may exceed `#S`; `i ≤ 0` produces an error. | — |
| `ChangeUniverse(~S, V)` / `ChangeUniverse(S, V)` | Coerce all elements of `S` into structure `V` (which must contain the current universe `U`). Procedure form modifies in place; function form returns the new sequence. | — |
| `CanChangeUniverse(S, V)` | Attempt to coerce all elements of `S` into `V`; returns `true` and the new sequence on success, `false` otherwise. | — |

*Worked example: H10E3 (three constructions of the Farey series Fn: iterative, recursive using `Insert`, and one-liner using `Sort` and `Setseq`).*

### 10.4.4 Creating New Enumerated Sequences from Existing Ones

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `S cat T` | Concatenation of `S` and `T`, giving `[s1, …, sn, t1, …, tm]`. A common over-structure is sought if universes differ; error if none exists. | — |
| `S cat:= T` | Mutation assignment: replaces `S` with `S cat T`. | — |
| `Partition(S, p)` | For complete non-empty `S` where `p` divides `#S`: sequence of sub-sequences each of length `p`. | — |
| `Partition(S, P)` | For complete non-empty `S` and sequence of positive integers `P` with sum `#S`: sequence of sub-sequences of lengths `P[1], P[2], …`. | — |
| `Setseq(S)` / `SetToSequence(S)` | Given a set `S`, construct a sequence whose terms are the elements of `S` in some arbitrary order. | — |
| `Seqset(S)` / `SequenceToSet(S)` | Given a sequence `S`, create a set whose elements are the distinct terms of `S`. | — |

*Worked example: H10E4 (Egyptian fraction decomposition using `Append`, `Remove`, `cat:=`, `Maximum`, `IntegerToString`; bibliography: [Bee93]).*

#### 10.4.4.1 Operations on Sequences of Booleans

These operations work pointwise on Boolean sequences of equal length.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `And(S, T)` / `And(~S, T)` | Pointwise logical `and` of `S` and `T`. Reference form stores result in `S`. | — |
| `Or(S, T)` / `Or(~S, T)` | Pointwise logical `or` of `S` and `T`. Reference form stores result in `S`. | — |
| `Xor(S, T)` / `Xor(~S, T)` | Pointwise logical `xor` of `S` and `T`. Reference form stores result in `S`. | — |
| `Not(S)` / `Not(~S)` | Pointwise logical `not` of `S`. Reference form stores result in `S`. | — |

---

## 10.5 Predicates on Sequences

Boolean operators for sequences test whether entries are defined, test membership and
containment, and compare sequences lexicographically. On formal sequences only membership
testing is available.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsComplete(S)` | Returns `true` iff every term `S[i]` for `1 ≤ i ≤ #S` is defined. | — |
| `IsDefined(S, i)` | Returns `true` iff `S[i]` is defined. Returns `false` if `i > #S`; error if `i < 1`. Supports multi-indices. | — |
| `IsEmpty(S)` | Returns `true` iff enumerated sequence `S` is empty. | — |
| `IsNull(S)` | Returns `true` iff `S` is empty and its universe is undefined (the null sequence). | — |

### 10.5.1 Membership Testing

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `x in S` | Returns `true` if `x` occurs as a term of enumerated or formal sequence `S`. Coercion is attempted if `x` is not in the universe; error if coercion fails. | — |
| `x notin S` | Returns `true` if `x` does not occur as a term of `S`. Same coercion behaviour. | — |
| `IsSubsequence(S, T)` / `IsSubsequence(S, T: Kind := option)` | Returns `true` if `S` appears as a subsequence of `T`. Default `Kind := "Consecutive"` (consecutive elements); `"Sequential"` (in order, not necessarily consecutive); `"Setwise"` (set containment). Coercion is attempted if universes differ. | — |
| `S eq T` | Returns `true` if sequences `S` and `T` are equal. Coercion attempted if universes differ. | — |
| `S ne T` | Returns `true` if `S` and `T` are not equal. | — |

### 10.5.2 Testing Order Relations

Sequences are ordered lexicographically: `S lt T` iff at the first differing index `k`,
`S[k] < T[k]`, or `S` is a proper prefix of `T`. Requires a common over-structure on
whose elements `eq`, `le`, `lt`, `gt`, `ge` are defined.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `S lt T` | `true` iff `S` strictly precedes `T` in lexicographic order: `S[k] < T[k]` at the first difference, or `S` is a proper prefix of `T`. | Lexicographic comparison. |
| `S le T` | `true` iff `S lt T` or `S eq T`. | Lexicographic comparison. |
| `S ge T` | `true` iff `S gt T` or `S eq T`. | Lexicographic comparison. |
| `S gt T` | `true` iff `S` strictly follows `T`: `S[k] > T[k]` at the first difference, or `T` is a proper prefix of `S`. | Lexicographic comparison. |

---

## 10.6 Recursion, Reduction, and Iteration

### 10.6.1 Recursion

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Self(n)` / `Self()` | Inside a sequence constructor, refers to the already-computed entry `s[n]` of the sequence under construction, or the sequence itself. Allows recursive sequence definitions. | — |

*Worked example: H10E5 (first 100 Fibonacci numbers defined recursively using `Self(i-2)+Self(i-1)`; sum via `&+`).*

### 10.6.2 Reduction

Rather than looping, the reduction operator `&` applies a binary associative operator to all
elements of a complete enumerated sequence.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `&◦S` | For a complete sequence `S = [a1, …, an]` and associative operator `◦`, computes `a1 ◦ a2 ◦ … ◦ an`. Supported operators: `+`, `*`, `and`, `or`, `join`, `meet`, `cat`. For a single-element sequence returns that element. For empty sequences with universe: `&+` → `0`, `&*` → `1`, `&and` → `true`, `&or` → `false`, `&join` → empty, `&cat` → empty; `&meet` → error. For the null sequence: `&and` and `&or` return `true`/`false`; all others error. | — |

---

## 10.7 Iteration

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `for x in S do statements; end for;` | Iterate over the defined terms of enumerated sequence `S`. Undefined terms are skipped. When multiple range sequences are used in a constructor, the first range forms the innermost loop (last-varying variable). | — |

*Worked example: H10E6 (nested iteration in sequence constructors vs. explicit nested for-loops; scope rules for loop variables in multi-range constructors).*

---

## 10.8 Bibliography

| Key | Reference |
|-----|-----------|
| **[Bee93]** | L. Beeckmans. *The splitting algorithm for Egyptian fractions.* J. Number Th., **43**:173–185, 1993. |

---

## Algorithm-to-function quick reference

| Algorithm / operation | Functions |
|-----------------------|-----------|
| Arithmetic-progression storage | `[ i..j ]`, `[ i..j by k ]`, `[ U \| i..j ]`, `[ U \| i..j by k ]` |
| Quicksort (default ordering) | `Sort(~S)`, `Sort(S)` |
| Quicksort (custom comparator) | `Sort(~S, C)`, `Sort(~S, C, ~p)`, `Sort(S, C)` |
| Parallel sort | `ParallelSort(~S, ~T)` |
| Lexicographic ordering | `lt`, `le`, `ge`, `gt` on sequences |
| Recursive sequence definition | `Self(n)`, `Self()` |
| Reduction (fold) | `&+`, `&*`, `&and`, `&or`, `&join`, `&meet`, `&cat` |
| Power-sequence coercion | `PowerSequence`, `in`, `!` |
| Set ↔ sequence conversion | `Setseq` / `SetToSequence`, `Seqset` / `SequenceToSet` |
| Pointwise Boolean ops on bit-vectors | `And`, `Or`, `Xor`, `Not` |
| Egyptian-fraction splitting [Bee93] | (example only — no dedicated intrinsic) |
