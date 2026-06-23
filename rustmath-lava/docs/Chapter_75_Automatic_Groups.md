# Chapter 75 — Automatic Groups

**Handbook part:** X — Finitely-Presented Groups
**Handbook pages:** 2359–2375 (PDF pages 2490–2509)

---

## Scope and overview

Chapter 75 provides a Magma-level interface to Derek Holt's KBMAG programs, specifically to the `autgroup` component that computes the automata associated with a short-lex automatic group. The chapter is based largely on the KBMAG documentation **[Hol97]**.

An **automatic group** G is a finitely presented group in which key operations — notably word equality and word enumeration — are decided by a collection of finite state automata. Words in G are ordered using the **short-lex ordering**: shorter words precede longer ones, and words of equal length are compared lexicographically according to the given generator ordering.

The Magma category for automatic groups is `GrpAtc`; elements are of type `GrpAtcElt`. Construction proceeds in three steps: (i) build a free group FG; (ii) take a quotient F of FG; (iii) create a monoid presentation for F and run the Knuth–Bendix completion procedure to construct and verify the automata. The construction may fail (if limits are exceeded or if the group is not automatic). Four automata are produced on success: the first and second word-difference machines, the word acceptor, and the word multiplier.

Because the Knuth–Bendix procedure can run indefinitely, configurable limits on the number of reduction equations, hash-table sizes, and other parameters govern termination. The chapter describes these parameters in detail and provides guidance on escalating from default to `Large` to `Huge` settings for harder examples.

---

## 75.1 Introduction

### 75.1.1 Terminology

An automatic group is a finitely presented group in which equality between words and word enumeration are decidable via automata. Words are compared under the short-lex ordering (shorter first; equal-length words ordered lexicographically by the generator ordering).

### 75.1.2 The Category of Automatic Groups

The family of all automatic groups forms a category in Magma. Objects are automatic groups (`GrpAtc`); morphisms are group homomorphisms. Elements are of type `GrpAtcElt`.

### 75.1.3 The Construction of an Automatic Group

Construction is a three-step process:

1. Construct a free group FG.
2. Construct a quotient F of FG (the finitely presented group).
3. Create a monoid presentation for F and run procedures that attempt to build the automata for G and prove them correct.

If G is not automatic the procedures cannot succeed. Failure to converge within the specified limits returns a non-confluent result.

---

## 75.2 Creation of Automatic Groups

### 75.2.1 Construction of an Automatic Group

Internally a monoid presentation P of F is constructed. By default the generators of P are g₁, g₁⁻¹, …, gₙ, gₙ⁻¹ where g₁, …, gₙ are the generators of F. Relations of P are the relations of F plus the trivial inverse relations. The short-lex word ordering is used. The Knuth–Bendix completion procedure for monoids is run on P to calculate the word-difference automata, which are then used to derive the finite state automata of a short-lex automatic group. In successful cases the automata are proved correct in a final verification step.

`AutomaticGroup` returns an automatic group on success, and does not return a value on failure. `IsAutomaticGroup` returns `(true, G)` on success and `false` on failure.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AutomaticGroup(F: -)` | Attempt to construct an automatic structure for the finitely presented group F. Returns the automatic group G on success; no return value on failure. | Knuth–Bendix completion for monoids → word-difference automata → word acceptor + word multiplier construction and verification **[Hol97]** |
| `IsAutomaticGroup(F: -)` | As `AutomaticGroup`, but returns `(true, G)` on success and `false` on failure. | Same as above **[Hol97]** |

*Worked examples: H75E1 (fundamental group of the torus; default parameters, 36-state word acceptor).*

### 75.2.2 Modifying Limits

The parameters below control the execution of the Knuth–Bendix completion procedure and the associated automata construction. If a first attempt fails it should be re-run with `Large := true`, then `Huge := true` if necessary.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AutomaticGroup(F: -)` | Full parameter form (see below for parameter details). | Knuth–Bendix with KBMAG-style limit controls **[Hol97]** |
| `IsAutomaticGroup(F: -)` | Full parameter form; same parameters as `AutomaticGroup`. | As above **[Hol97]** |
| `SetVerbose("KBMAG", v)` | Set the verbose printing level for the Knuth–Bendix completion algorithm. Legal values: 0 (silent), 1 (default), 2 (`-v` verbose, small extra output), 3 (`-vv` very verbose, full diagnostics). | — |

**Parameters for `AutomaticGroup` / `IsAutomaticGroup`:**

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| `Large` | `BoolElt` | `false` | Use large hash tables; sets TidyInt=500, MaxRelations=262144, MaxStates=unlimited, HaltingFactor=100, MinTime=20, ConfNum=0. Recommended only after a default attempt has failed. |
| `Huge` | `BoolElt` | `false` | Doubles the hash tables and MaxRelations over the `Large` setting. Use only after `Large` has also failed. |
| `MaxRelations` | `RngIntElt` | 200 | Limit on the maximum number of reduction equations. |
| `TidyInt` | `RngIntElt` | 20 | After finding this many new reduction equations, interrupt to tidy the equation set (eliminate redundancies, reduce left/right sides). Small values are better for easy examples; large values (e.g. 1000) suit very difficult ones. |
| `GeneratorOrder` | `SeqEnum` | `[g₁,g₁⁻¹,…,gₙ,gₙ⁻¹]` | Ordering for the generators and their inverses in the monoid presentation alphabet. |
| `MaxWordDiffs` | `RngIntElt` | (dynamic) | Maximum number of word differences. By default the limit grows dynamically; usually need not be set explicitly. |
| `HaltingFactor` | `RngIntElt` | 100 | Percentage increase threshold: after each tidy, if both the equation count and state count have grown by more than this percentage since the last decrease in word-differences, halt. |
| `MinTime` | `RngIntElt` | 5 | Minimum CPU seconds before the HaltingFactor check can trigger; prevents very early premature halting. |

*Worked examples: H75E2 (Listing's knot group — fails with defaults, succeeds with `Large := true`); H75E3 (trefoil knot fundamental group with explicit `GeneratorOrder`).*

### 75.2.3 Accessing Group Information

The functions in this section access basic data stored in a constructed automatic group G.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `G . i` | The i-th defining generator of G. The integer i must lie in [−r, r] where r is the number of generators of G. | — |
| `Generators(G)` | A sequence containing the defining generators of G. | — |
| `NumberOfGenerators(G)` / `Ngens(G)` | The number of defining generators of G. | — |
| `FPGroup(G)` | Returns the finitely presented group F used to construct G, and the isomorphism from F to G. | — |
| `WordAcceptor(G)` | A record describing the word acceptor automaton stored in G. | — |
| `WordAcceptorSize(G)` | The number of states of the word acceptor automaton stored in G, and the size of its alphabet. | — |
| `WordDifferenceAutomaton(G)` | A record describing the word difference automaton stored in G. | — |
| `WordDifferenceSize(G)` | The number of states of the 2nd word difference automaton stored in G, and the size of its alphabet. | — |
| `WordDifferences(G)` | The labels of the states of the word difference automaton, as a sequence of elements of the finitely presented group F used to construct G. | — |
| `GeneratorOrder(G)` | The value of the `GeneratorOrder` parameter used in the construction of G; a sequence of generators and their inverses from F. | — |

*Worked examples: H75E4 (Von Dyck (2,3,5) group ≅ A₅; illustrates `G.i`, `Generators`, `Ngens`, `Relations`, `Nrels`, `Ordering`).*

---

## 75.3 Properties of an Automatic Group

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsFinite(G)` | Returns `true` if the automatic group G has finite order and `false` otherwise. If G is finite also returns the order of G as a second value. | Order computation via the automata associated with G **[Hol97]** |
| `Order(G)` / `#G` | The order of G as an integer. If G is known to be infinite, returns the symbol ∞. | As above **[Hol97]** |

*Worked examples: H75E5 (Z ≀ C₂ — `Order(G)` returns `Infinity`); H75E6 (3-fold cover of A₆ — `IsFinite(G)` returns `true 1080`).*

---

## 75.4 Arithmetic with Words

### 75.4.1 Construction of a Word

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `G ! [ i1, ..., is ]` | Constructs the word G.\|i₁\|^ε₁ ∗ … ∗ G.\|iₛ\|^εₛ in the automatic group G, where εⱼ = +1 if iⱼ > 0 and −1 if iⱼ < 0; each iⱼ ∈ [−r, r] \ {0}. Returned in reduced form. | Reduction via the second word-difference machine **[Hol97]** |
| `Identity(G)` / `Id(G)` / `G ! 1` | Constructs the identity word in G. | — |
| `Parent(w)` | Returns the parent automatic group G of the word w. | — |

*Worked examples: H75E7 (two-generator two-relator group; constructs words via `G![…]` and operator expressions).*

### 75.4.2 Operations on Elements

Word arithmetic is performed by first forming the product in the underlying free group, then reducing using the second word-difference machine. Note: (i) reduction can increase word length and is subject to an internal length limit — any word operation involving reduction can fail if this limit is exceeded; (ii) the implementation prioritises speed over space, so the reduction machine is always used and can be space-consuming when the generator count is large.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `u * v` | Product of words u and v in a common automatic group. | Concatenation then reduction via 2nd word-difference machine **[Hol97]** |
| `u / v` | Product u ∗ v⁻¹ for words u and v in a common automatic group. | As above |
| `u ^ n` | The n-th power of the word u. | Repeated multiplication + reduction |
| `u ^ v` | Conjugate of u by v, i.e. v⁻¹ ∗ u ∗ v. | Multiplication + reduction |
| `Inverse(w)` | The inverse of the word w. | Reduction via 2nd word-difference machine |
| `(u, v)` | Commutator of words u and v, i.e. u⁻¹ v⁻¹ u v. | Multiplication + reduction |
| `(u1, ..., ur)` | Left-normed commutator of r words u₁, …, uᵣ, evaluated left to right. | Iterated commutator + reduction |
| `u eq v` | Returns `true` if u and v reduce to the same normal form. If G is confluent this tests genuine equality; if non-confluent, equal words may reduce to different normal forms. | Normal-form comparison via word-difference machine |
| `u ne v` | Returns `false` if u and v reduce to the same normal form, `true` otherwise. Same caveat as `eq` for non-confluent G. | Normal-form comparison |
| `IsId(w)` / `IsIdentity(w)` | Returns `true` if w is the identity word. | Normal-form check |
| `#u` | The length of the word u. | — |
| `ElementToSequence(u)` / `Eltseq(u)` | Decomposes the element u = G.i₁^e₁ ∗ … ∗ G.iₘ^eₘ into a sequence Q where Q[j] = iⱼ if eⱼ = +1 and Q[j] = −iⱼ if eⱼ = −1. | — |

*Worked examples: H75E8 (fundamental group of a 3-manifold with 5 generators; illustrates `*`, `/`, `^`, `Inverse`, commutator `(c,d)`, `IsIdentity`, `#`).*

---

## 75.5 Homomorphisms

### 75.5.1 General Remarks

Groups in the category `GrpAtc` are currently accepted as codomains only in certain special situations. The most important cases where an automatic group can be used as a codomain are group homomorphisms whose domain is in one of the categories `GrpFP`, `GrpGPC`, `GrpRWS`, or `GrpAtc`.

### 75.5.2 Construction of Homomorphisms

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `hom< A -> G \| S >` | Returns the homomorphism from automatic group A to group G defined by S. S may be: (i) a list/sequence/indexed set of n images for A.1, …, A.n (order matters); or (ii) a list/sequence/set of n tuples `< xi, yi >` or arrow pairs `xi -> yi` giving the image of each generator xi (order irrelevant). No checking is performed — the user must ensure the images define a valid homomorphism. Defining a homomorphism from an arbitrary generating set (not the canonical one) is not currently supported. | — |

---

## 75.6 Set Operations

Functions for enumerating sets of elements of an automatic group G.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Random(G, n)` | A random word of length at most n in the generators of G. | — |
| `Random(G)` | A random word (of length at most `Order(G)`) in the generators of G. | — |
| `Representative(G)` / `Rep(G)` | An element chosen from G. | — |
| `Set(G, a, b)` | The set of words w in G with a ≤ length(w) ≤ b. Parameter `Search` (`MonStgElt`, default `"DFS"`): `"DFS"` enumerates lexicographically; `"BFS"` enumerates in short-lex order (by length, then lexicographically). DFS is marginally faster. The result is a set, so internal enumeration order may not be reflected. | Word-acceptor traversal **[Hol97]** |
| `Set(G)` | The full carrier set of G. Parameter `Search` as above. | Word-acceptor traversal **[Hol97]** |
| `Seq(G, a, b)` | A sequence of words w in G with a ≤ length(w) ≤ b, appearing in the order specified by `Search` (default `"DFS"`; `"BFS"` gives short-lex order). | Word-acceptor traversal **[Hol97]** |
| `Seq(G)` | A sequence of all words in G, in the order specified by `Search` (default `"DFS"`). | Word-acceptor traversal **[Hol97]** |

*Worked examples: H75E9 (dihedral group D₂₂ with 6 generators; illustrates `Representative`, `Random`, `Random(G,5)`, `Set(G)`, `Seq(G : Search := "BFS")`).*

---

## 75.7 The Growth Function

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `GrowthFunction(G)` | Computes the growth function of the word acceptor automaton of G. Returns a rational function in a single variable x (quotient of two integral polynomials); the coefficient of xⁿ in the Taylor expansion about 0 equals the number of words of length n accepted by the word acceptor. Parameter `Primes` (`SeqEnum`, default `[]`) is no longer used but retained for backward compatibility. | Algorithm by Derek Holt; rational growth function of a DFA **[Hol97]** |

*Worked examples: H75E10 (dihedral group of order 10 — growth function is a polynomial `2*x^3 + 4*x^2 + 3*x + 1`; infinite dihedral group — rational function `(-x^2 - 2*x - 1)/(x - 1)` with coefficient extraction via `PowerSeriesRing`).*

---

## 75.8 Bibliography

| Key | Reference |
|-----|-----------|
| **[Hol97]** | Derek Holt. *KBMAG — Knuth-Bendix in Monoids and Automatic Groups.* University of Warwick, 1997. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Knuth–Bendix completion for monoids (short-lex, KBMAG) **[Hol97]** | `AutomaticGroup`, `IsAutomaticGroup` |
| Word-difference automata construction and verification **[Hol97]** | `AutomaticGroup`, `IsAutomaticGroup` |
| Word acceptor finite state automaton **[Hol97]** | `WordAcceptor`, `WordAcceptorSize`, `Set`, `Seq`, `GrowthFunction` |
| Word multiplier / reduction via 2nd word-difference machine **[Hol97]** | `u * v`, `u / v`, `u ^ n`, `u ^ v`, `Inverse`, `(u,v)`, `u eq v`, `u ne v`, `G ! [...]` |
| Rational growth function of a DFA (Derek Holt's algorithm) **[Hol97]** | `GrowthFunction` |
| Order computation via automata **[Hol97]** | `IsFinite`, `Order` / `#G` |
