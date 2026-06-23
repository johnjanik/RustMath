# Chapter 74 — Groups Defined by Rewrite Systems

**Handbook part:** X — Finitely-Presented Groups
**Handbook pages:** 2341–2356 (PDF pages 2472–2489)

---

## Scope and overview

Chapter 74 describes the class of finitely presented groups defined by finite rewrite systems
(category `GrpRWS`, elements `GrpRWSElt`). The implementation wraps Derek Holt's **KBMAG**
programs **[Hol97]**, specifically the Knuth–Bendix completion procedure for groups given by a
finite monoid presentation.

A **rewrite group** G is a finitely presented group whose elements (words or strings) are
equipped with reduction relations codified into a finite-state automaton called the *reduction
machine*. Words and reduction relations are totally ordered; supported orderings include
short-lex, recursive, right-recursive, weighted-lex, and wreath-product orderings. A rewrite
group is either *confluent* or *non-confluent*. When confluent, the reduction machine reduces
every word to its unique irreducible normal form, solving the word problem efficiently.

Construction proceeds in three steps: (i) build a free group FG; (ii) form a quotient F of FG;
(iii) run the Knuth–Bendix completion procedure on the corresponding monoid presentation to
produce G. The completion procedure may fail (returning a non-confluent monoid), in which case
the user may adjust ordering and limit parameters and retry. Much of the material is drawn
directly from the KBMAG documentation **[Hol97]**.

---

## 74.1 Introduction

### 74.1.1 Terminology

A *rewrite group* G is a finitely presented group in which equality between elements may be
decided via a sequence of reduction rules codified in a reduction machine (a finite-state
automaton). Words are ordered under one of several supported orderings; the group is confluent
if every word reduces to a unique normal form. The Magma category is `GrpRWS`; elements are
`GrpRWSElt`.

### 74.1.2 The Category of Rewrite Groups

Rewrite groups form a category whose objects are rewrite groups and whose morphisms are group
homomorphisms. Category name: `GrpRWS`. Elements: `GrpRWSElt`.

### 74.1.3 The Construction of a Rewrite Group

Three-step process: (i) construct a free group FG; (ii) form a quotient F of FG; (iii) create
a monoid presentation of F and run the Knuth–Bendix completion procedure to obtain G. If the
procedure succeeds the result is confluent. If it fails the user may need to adjust parameters
and repeat.

---

## 74.2 Constructing Confluent Presentations

### 74.2.1 The Knuth-Bendix Procedure

Internally, `RWSGroup` converts the group presentation into a monoid M whose generators are
g₁, g₁⁻¹, …, gₙ, gₙ⁻¹ together with the trivial relations gᵢgᵢ⁻¹ = gᵢ⁻¹gᵢ = 1. The
Knuth–Bendix completion procedure for monoids is then applied to M. Whether or not the
procedure succeeds, the result is a rewrite monoid containing a reduction machine and a set of
reduction relations. If the procedure succeeds M is marked confluent and the word problem is
decidable; otherwise M is non-confluent and reductions are correct in F but do not guarantee
unique normal forms. The default generator ordering is induced by the generators of F; the
default string ordering is ShortLex. Execution is bounded by user-specified limits on
variables such as the number of reduction relations.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RWSGroup(F: -)` | Given a finitely presented group F, attempt to construct a confluent rewrite group G by running the Knuth–Bendix completion procedure on the monoid presentation of F. Returns a `GrpRWS` object, confluent or not. Default generator ordering is that of F's generators; default string ordering is `ShortLex`. Parameters controlling the procedure are described in §74.2.2 (orderings) and §74.2.3 (limits). | Knuth–Bendix completion for monoids, via KBMAG **[Hol97]** |

*Worked example: H74E1 (Von Dyck (2,3,5) group with default ShortLex ordering; 39-state reduction machine).*

### 74.2.2 Defining Orderings

The second appearance of `RWSGroup` in the index corresponds to the ordering parameters
described in this subsection. These govern both the generator order and the ordering on strings
used by the Knuth–Bendix procedure.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RWSGroup(F: GeneratorOrder, Ordering, Levels, Weights)` | Attempt to construct a confluent presentation for F. `GeneratorOrder` (SeqEnum, default: order induced by F): ordering of the 2n monoid generators. `Ordering` (MonStgElt, default: `"ShortLex"`): one of `"ShortLex"`, `"Recursive"`, `"RtRecursive"`, `"WtLex"`, `"Wreath"`. `Weights` (SeqEnum): required for `"WtLex"`; non-negative integer weight per generator. `Levels` (SeqEnum): required for `"Wreath"`; non-negative integer level per generator — see **[Sim94, pp. 46–50]** for the complete definition. Note: recursive ordering is the special case of wreath-product ordering with level of generator i equal to i. | Knuth–Bendix completion with caller-specified term ordering, via KBMAG **[Hol97]**; wreath-product ordering defined in **[Sim94]** |

*Worked examples: H74E2 (infinite non-Hopfian group b⁻¹a²b = a³ with Recursive ordering); H74E3 (free nilpotent group of rank 2 and class 2 with Recursive ordering and explicit GeneratorOrder placing lower nilpotency-class generators first).*

### 74.2.3 Setting Limits

Parameters controlling the termination conditions of the Knuth–Bendix procedure. The
procedure is guaranteed to run forever on some inputs, so limits must be placed on various
internal variables. **Warning:** changing `MaxStoredLen`, `MaxOverlapLen`, `Sort`, or
`MaxOpLen` from their defaults may cause the procedure to terminate without finding a confluent
presentation, or may change the underlying group.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RWSMonoid(F: MaxRelations, TidyInt, RabinKarp, MaxStates, MaxReduceLen, ConfNum, MaxStoredLen, MaxOverlapLen, Sort, MaxOpLen)` | Attempt to construct a confluent presentation for F, with fine-grained control over the Knuth–Bendix execution. `MaxRelations` (RngIntElt, default 32767): maximum number of reduction equations. `TidyInt` (RngIntElt, default 100): after finding this many new equations, interrupt and tidy (eliminate redundant equations, reduce LHS/RHS). `RabinKarp` (Tup `<l,n>`, default none): use the Rabin–Karp algorithm for words of length ≥ l when there are ≥ n equations; uses less space than the default automaton but is slower — most useful when collapse occurs (large intermediate equation set shrinks to a small confluent set). `MaxStates` (RngIntElt, default none): cap on FSA states for the reduction automaton; normally not needed. `MaxReduceLen` (RngIntElt, default 32767): cap on word length during reduction; mainly relevant for the recursive ordering. `ConfNum` (RngIntElt, default 500): number of overlaps processed without finding new equations before a fast confluence check is performed; 0 means check only after the overlap search is fully complete. `MaxStoredLen` (Tup `<l,r>`, default none): discard equations whose LHS exceeds length l or RHS exceeds length r; can speed convergence in collapse cases but may change the group. `MaxOverlapLen` (RngIntElt, default none): ignore overlaps of total length exceeding this value; may produce a non-confluent result. `Sort` (BoolElt, default false) + `MaxOpLen` (RngIntElt, default 0): if Sort is true, sort equations by increasing LHS length; if MaxOpLen > 0, only output equations with LHS length ≤ MaxOpLen (danger: may change the group). | Knuth–Bendix completion with explicit resource limits, via KBMAG **[Hol97]** |
| `SetVerbose("KBMAG", v)` | Set the verbose printing level for the Knuth–Bendix completion algorithm. Legal values: 0 (silent, equivalent to KBMAG `-silent`), 1 (default), 2 (verbose, equivalent to KBMAG `-v`), 3 (very verbose diagnostic output, equivalent to KBMAG `-vv`). | — |

### 74.2.4 Accessing Group Information

Basic structural information stored in a rewrite group G.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `G . i` | The i-th defining generator of G. The integer i must lie in [−r, r] where r = Ngens(G). | — |
| `Generators(G)` | A sequence containing the defining generators of G. | — |
| `NumberOfGenerators(G)` / `Ngens(G)` | The number of defining generators of G. | — |
| `Relations(G)` | A sequence of the defining relations of G, expressed as equations between elements of the free group of which G is a quotient. The LHS of each relation is always greater than the RHS under the ordering used to construct G. | — |
| `NumberOfRelations(G)` / `Nrels(G)` | The number of relations in G. | — |
| `Ordering(G)` | The string ordering of G (e.g. `ShortLex`, `Recursive`). | — |

*Worked example: H74E4 (Z ≀ C₂ presented as a rewrite group; illustrating G.i, Generators, Ngens, Relations, Nrels, Ordering).*

---

## 74.3 Properties of a Rewrite Group

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsConfluent(G)` | Returns `true` if G is confluent, `false` otherwise. | Checks the confluence flag set during the Knuth–Bendix procedure. |
| `IsFinite(G)` | Given a confluent group G, returns `true` if G has finite order and `false` otherwise. If finite, also returns the order of G as a second value. | Requires confluence; finiteness test via KBMAG **[Hol97]** |
| `Order(G)` / `#G` | The order of G as an integer, or the symbol `∞` if G is known to be infinite. | Requires confluence; KBMAG enumeration **[Hol97]** |

*Worked examples: H74E5 (Weyl group E₈ of order 696,729,600, constructed via RWSGroup and verified finite by IsFinite); H74E6 (2-generator 2-relator infinite group; Order returns Infinity).*

---

## 74.4 Arithmetic with Words

### 74.4.1 Construction of a Word

Elements (words) in a rewrite group are constructed by coercion and reduced using the
reduction machine.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Identity(G)` / `Id(G)` / `G ! 1` | Construct the identity word in G. | — |
| `G ! [i1, ..., is]` | Given integers i₁, …, iₛ in [−r, r] \ {0} (r = Ngens(G)), construct the word G.\|i₁\|^ε₁ * … * G.\|iₛ\|^εₛ where εⱼ = +1 if iⱼ > 0 and −1 if iⱼ < 0. The result is reduced by the reduction machine. | Word reduction via the reduction machine (FSA). |
| `Parent(w)` | The parent group G of the word w. | — |

*Worked example: H74E7 (Fibonacci group F(2,7); constructing the identity, coercing 1, and building G![1,2]).*

### 74.4.2 Element Operations

For u, v ∈ G, the product u * v is formed in the underlying free group and then reduced by the
reduction machine. If G is confluent the result is the unique minimal word under the group
ordering; if non-confluent, equal elements may reduce to distinct words. Note: reduction can
increase word length, and an internal length limit will raise an error if exceeded.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `u * v` | Product of words u and v in the same group. | FSA-based word reduction |
| `u / v` | Product u * v⁻¹. | FSA-based word reduction |
| `u ^ n` | n-th power of word u. | FSA-based word reduction |
| `u ^ v` | Conjugate of u by v: v⁻¹ * u * v. | FSA-based word reduction |
| `Inverse(w)` | The inverse of word w. | FSA-based word reduction |
| `(u, v)` | Commutator of u and v: u⁻¹v⁻¹uv. | FSA-based word reduction |
| `(u1, ..., ur)` | Left-normed commutator of r words: evaluated left to right. | FSA-based word reduction |
| `u eq v` | `true` if u and v reduce to the same normal form, `false` otherwise. Tests equality if G is confluent; only a sufficient condition if non-confluent. | FSA-based word reduction |
| `u ne v` | `false` if u and v reduce to the same normal form, `true` otherwise. Dual of `eq`. | FSA-based word reduction |
| `IsId(w)` / `IsIdentity(w)` | Returns `true` if w is the identity word. | FSA-based word reduction |
| `#u` | The length of the word u. | — |
| `ElementToSequence(u)` / `Eltseq(u)` | Decompose u = G.i₁^e₁ * … * G.iₘ^eₘ (each eⱼ = ±1) into the sequence Q where Q[j] = iⱼ if eⱼ = +1 and Q[j] = −iⱼ if eⱼ = −1. | — |

*Worked example: H74E8 (Fibonacci group F(2,5); illustrating *, /, ^, conjugation, Inverse, commutator, eq, IsIdentity, #).*

---

## 74.5 Operations on the Set of Group Elements

For enumeration of elements. The `Search` parameter is common to `Set` and `Seq` variants:
`"DFS"` (depth-first, lexicographic order, default, marginally faster) or `"BFS"`
(breadth-first, short-lex order by length then lexicographically within each length).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Random(G, n)` | A random word of length at most n in the generators of G. | — |
| `Random(G)` | A random word (of length at most the order of G) in the generators of G. | — |
| `Representative(G)` / `Rep(G)` | An element chosen from G. | — |
| `Set(G, a, b)` | The set of reduced words w in G with a ≤ length(w) ≤ b. Parameter `Search` (MonStgElt, default `"DFS"`): enumeration order. | FSA-based enumeration; DFS or BFS traversal |
| `Set(G)` | The set of all reduced words (carrier set) of G. Parameter `Search` as above. | FSA-based enumeration |
| `Seq(G, a, b)` | The sequence of reduced words w with a ≤ length(w) ≤ b, in the order determined by `Search`. Parameter `Search` as above. | FSA-based enumeration |
| `Seq(G)` | A sequence of all reduced words in G, ordered by `Search`. | FSA-based enumeration |

*Worked example: H74E9 (group D₂₂; illustrating Representative, Random, Random(G,5), Set(G), Seq(G: Search:="DFS")).*

---

## 74.6 Homomorphisms

### 74.6.1 General Remarks

Rewrite groups (`GrpRWS`) are currently accepted as codomains only in some special situations.
The main cases in which a `GrpRWS` can serve as a codomain are homomorphisms whose domain
belongs to one of the categories `GrpFP`, `GrpGPC`, `GrpRWS`, or `GrpAtc`. For a general
description of homomorphisms, see Chapter 16.

### 74.6.2 Construction of Homomorphisms

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `hom< R -> G \| S >` | Returns the homomorphism from the rewrite group R to the group G defined by S. S can be: (i) a list/sequence/indexed set of images for generators R.1, …, R.n (order matters); or (ii) a list/sequence/set of tuples `<xᵢ, yᵢ>` or arrow pairs `xᵢ -> yᵢ` covering all generators (order does not matter). No checking is performed that the provided images yield a well-defined homomorphism — this is the user's responsibility. It is not currently possible to define a homomorphism from an arbitrary generating set of R. | — |

---

## 74.7 Conversion to a Finitely Presented Group

There is a standard idiom for converting a rewrite group into a finitely presented group using
`Relations` and `Simplify`. No new intrinsics are introduced in this section; it demonstrates
the use of `Relations(G)` (§74.2.4) together with `quo<...>` and the general `Simplify`
function.

*Worked example: H74E10 (two-generator free abelian group constructed as a rewrite group and converted to an FP group via `Simplify(quo< FG | Relations(G) >)`).*

---

## 74.8 Bibliography

| Key | Reference |
|-----|-----------|
| **[Hol97]** | Derek Holt. *KBMAG — Knuth-Bendix in Monoids and Automatic Groups.* University of Warwick, 1997. |
| **[Sim94]** | Charles C. Sims. *Computation with finitely presented groups.* Cambridge University Press, Cambridge, 1994. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Knuth–Bendix completion for monoids (KBMAG **[Hol97]**) | `RWSGroup`, `RWSMonoid` |
| Word ordering: ShortLex (default) | `RWSGroup(:Ordering:="ShortLex")` |
| Word ordering: Recursive / RtRecursive | `RWSGroup(:Ordering:="Recursive")`, `RWSGroup(:Ordering:="RtRecursive")` |
| Word ordering: Weighted-lex | `RWSGroup(:Ordering:="WtLex", :Weights:=...)` |
| Word ordering: Wreath-product **[Sim94, pp. 46–50]** | `RWSGroup(:Ordering:="Wreath", :Levels:=...)` |
| FSA-based word reduction (reduction machine) | `u*v`, `u/v`, `u^n`, `u^v`, `Inverse`, `(u,v)`, `eq`, `ne`, `IsId`, `IsIdentity`, `G![...]` |
| FSA-based element enumeration (DFS/BFS) | `Set`, `Seq`, `Random` |
| Confluence detection | `IsConfluent` |
| Finiteness / order computation (KBMAG **[Hol97]**) | `IsFinite`, `Order`, `#G` |
