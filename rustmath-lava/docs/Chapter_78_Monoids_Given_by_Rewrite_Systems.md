# Chapter 78 — Monoids Given by Rewrite Systems

**Handbook part:** X — Finitely-Presented Groups
**Handbook pages:** 2401–2415 (PDF pages 2532–2551)

---

## Scope and overview

Chapter 78 describes Magma's category of monoids defined by finite sets of rewrite rules
(`MonRWS`), which provides a high-level interface to Derek Holt's KBMAG programs — specifically
to KBMAG's Knuth–Bendix completion procedure applied to finitely presented monoids. Much of the
chapter's documentation is drawn directly from the KBMAG documentation **[Hol97]**.

A **rewrite monoid** M is a finitely presented monoid in which equality between elements (called
words or strings) is decidable via a sequence of rewriting equations called reduction relations,
rules, or equations. For efficiency the reduction rules are encoded in a finite state automaton
called a **reduction machine**. Words and reduction relations in M are ordered; supported
orderings include short-lex, recursive (and RTRecursive), weighted short-lex (`WTShortLex`), and
wreath-product (`Wreath`). A rewrite monoid can be **confluent** or **non-confluent**: if
confluent, the reduction machine reduces every word to its unique irreducible normal form under
the given ordering, making the word problem efficiently decidable.

Construction of a rewrite monoid is a three-step process: (i) define a free monoid F of the
appropriate rank; (ii) form a quotient Q of F by the desired relations; (iii) apply the
Knuth–Bendix completion procedure to Q to produce M. The completion procedure may or may not
terminate successfully; if it fails, the user may need to adjust parameters (e.g. ordering,
`MaxStoredLen`, `TidyInt`) and re-run, sometimes using the partial result as a warm start as
illustrated in Example H78E4 for the Fibonacci group F(2, 7).

Elements of a rewrite monoid are of type `MonRWSElt`. The category accepts monoid
homomorphisms, and the chapter documents conversion back to a finitely presented monoid via
`Relations`.

---

## 78.1 Introduction

### 78.1.1 Terminology

A rewrite monoid M is a finitely presented monoid in which equality of words is decidable via
reduction relations codified in a finite state automaton (the reduction machine). Words are
ordered; supported orderings are: short-lex, recursive, RTRecursive, weighted short-lex
(`WTShortLex`), and wreath-product (`Wreath`).

### 78.1.2 The Category of Rewrite Monoids

The family of all rewrite monoids forms a category whose objects are rewrite monoids and whose
morphisms are monoid homomorphisms. The Magma type designator is `MonRWS`; elements are
`MonRWSElt`.

### 78.1.3 The Construction of a Rewrite Monoid

Construction proceeds in three steps: (i) define a free monoid F; (ii) form a quotient Q of F;
(iii) apply the Knuth–Bendix completion procedure to Q to produce M. The procedure may fail, in
which case the result is a non-confluent rewrite monoid. If it succeeds, the result is marked
confluent and the word problem is decidable.

---

## 78.2 Construction of a Rewrite Monoid

The Knuth–Bendix completion procedure is run on Q (a quotient of a free monoid), taking Q's
relations as the initial reduction rules. Regardless of whether the procedure succeeds or fails,
the result is a rewrite monoid M containing a reduction machine and a sequence of reduction
relations. If the procedure succeeds M is marked confluent; otherwise M is marked non-confluent
and retains the partial results up to the point of failure.

Because the Knuth–Bendix procedure may run indefinitely, limits on internal variables are
required to force termination. The optimal limit values depend on the example.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RWSMonoid(Q: parameters)` | Runs the Knuth–Bendix completion procedure for monoids with the relations of `Q` as initial reduction rules. Returns a rewrite monoid M (confluent if successful, non-confluent otherwise). Parameters (see below). | Knuth–Bendix completion **[Hol97]** with the ordering and limits selected by the parameters. |
| `SetVerbose("KBMAG", v)` | Sets the verbose printing level for the Knuth–Bendix algorithm. Legal values: 0 (silent), 1 (default), 2 (`-v` verbose, small extra output), 3 (`-vv` very verbose, large amount of diagnostic output). | — |

**Parameters of `RWSMonoid`:**

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| `MaxRelations` | `RngIntElt` | 32767 | Maximum number of reduction equations. |
| `GeneratorOrder` | `SeqEnum` | (from Q) | Ordering for the generators, affects word ordering in the alphabet. |
| `Ordering` | `MonStgElt` | `"ShortLex"` | Word ordering: `"ShortLex"`, `"Recursive"`, `"RTRecursive"`, `"WTShortLex"`, or `"Wreath"`. |
| `Weights` | `SeqEnum` | — | One non-negative integer weight per generator; used with `Ordering := "WTShortLex"`. |
| `Levels` | `SeqEnum` | — | One non-negative integer level per generator; used with `Ordering := "Wreath"`. A complete definition of the wreath-product ordering is in **[Sim94, pp. 46–50]**. |
| `TidyInt` | `RngIntElt` | 100 | After finding `TidyInt` new equations, interrupt to tidy (eliminate redundant equations and reduce existing ones). |
| `RabinKarp` | `Tup` | — | `<l, n>`: use the Rabin–Karp algorithm for word-reduction on words of length ≥ l when there are ≥ n equations. Uses less space than the default automaton but is slower; most useful when collapse occurs. |
| `MaxStates` | `RngIntElt` | (no limit) | Maximum states of the word-reduction finite state automaton. |
| `MaxReduceLen` | `RngIntElt` | 32767 | Maximum allowed length a word may reach during reduction. Most relevant with recursive ordering. |
| `ConfNum` | `RngIntElt` | 500 | If `ConfNum` overlaps are processed with no new equations found, interrupt and perform a fast confluence check. Set to 0 to check only when the overlap search is complete. |
| `MaxStoredLen` | `Tup` | — | `<l, r>`: keep only equations whose left and right hand sides have lengths ≤ l and ≤ r respectively. Useful for collapse examples; may change the monoid or lose confluence. |
| `MaxOverlapLen` | `RngIntElt` | (no limit) | Process only overlaps of total length ≤ `MaxOverlapLen`. May leave the result non-confluent. |
| `Sort` | `BoolElt` | `false` | If `true`, sort equations by increasing left-hand-side length rather than discovery order. |
| `MaxOpLen` | `RngIntElt` | 0 | If positive, output only equations with left-hand-side length ≤ `MaxOpLen`. If 0, sort all equations. Used together with `Sort`. |

*Worked examples: H78E1 (alternating group A4, default ShortLex ordering — confluent result with 11 rewrite relations and a 12-state reduction machine); H78E2 (second Neumann trivial monoid presentation, large TidyInt := 3000, confluent result showing all generators equal the identity); H78E3 (submonoid of a nilpotent group, Recursive ordering, confluent, infinite order); H78E4 (Fibonacci group F(2,7), difficult two-pass approach: first run with Recursive ordering and MaxStoredLen := <15,15> yields a non-confluent monoid, original equations appended to the partial relations, second run immediately finds a confluent set — illustrating the warm-start technique).*

---

## 78.3 Basic Operations

### 78.3.1 Accessing Monoid Information

The functions in this section provide access to basic information stored for a rewrite monoid M.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `M . i` | The i-th defining generator of M. | — |
| `Generators(M)` | A sequence containing the defining generators of M. | — |
| `NumberOfGenerators(M)` / `Ngens(M)` | The number of defining generators of M. | — |
| `Relations(M)` | A sequence containing the defining relations of M as equations between elements of the free monoid of which M is a quotient. In each relation the left-hand side is greater than the right-hand side under the ordering used to construct M. | — |
| `NumberOfRelations(M)` / `Nrels(M)` | The number of relations in M. | — |
| `Ordering(M)` | The ordering of M (e.g. `ShortLex`, `Recursive`). | — |
| `Parent(w)` | The parent rewrite monoid M for the word w. | — |

*Worked example: H78E5 (presentation of S4 on two generators; illustrates M.1, M.1*M.2, Generators, Ngens, Relations, Nrels, Ordering, and Order).*

### 78.3.2 Properties of a Rewrite Monoid

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsConfluent(M)` | Returns `true` if M is confluent, `false` otherwise. | — |
| `IsFinite(M)` | Given a confluent monoid M, returns `true` if M has finite order and `false` otherwise. If M has finite order, also returns the order as a second value. | — |
| `Order(M)` / `#M` | Given a confluent presentation of M, returns the cardinality of M. Returns `Infinity` if M has infinite order. | Finite-order enumeration using the reduction machine of the confluent presentation. |

*Worked examples: H78E6 (threefold cover of A6, Order 1080, IsConfluent true); H78E7 (2-generator free abelian group, Order returns Infinity); H78E8 (Weyl group E8 on 8 generators and 36 relations, IsFinite returns true and order 696729600).*

### 78.3.3 Construction of a Word

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Identity(M)` / `Id(M)` / `M ! 1` | Constructs the identity word in M. | — |
| `M ! [i1, ..., is]` | Given a rewrite monoid M on r generators and a sequence of integers in [1, r], constructs the word M.i1 * M.i2 * ... * M.is. | — |

*Worked example: H78E9 (Fibonacci group F(2,7) on 14 generators; Id(M) and M!1 both print as `Id(M)`; Order returns 29).*

### 78.3.4 Arithmetic with Words

Word arithmetic proceeds by (i) forming the product in the underlying free monoid, then (ii) reducing using the reduction machine of M. If M is confluent the result is the unique minimal word under the ordering; if M is non-confluent, equal words may reduce to distinct forms. Internal limits on word length during reduction may cause word operations to fail with an error.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `u * v` | Product of words u and v in M: form the free-monoid product then reduce via the reduction machine. | Reduction machine of M. |
| `u ^ n` | The n-th power of word u (n a non-negative integer); reduces via the reduction machine. | Reduction machine of M. |
| `u eq v` | Returns `true` if u and v (in the same monoid) reduce to the same normal form, `false` otherwise. If M is confluent this is exact equality; if M is non-confluent two equal words may reduce differently. | Normal-form comparison via the reduction machine. |
| `u ne v` | Returns `false` if u and v reduce to the same normal form, `true` otherwise. Subject to the same caveat for non-confluent M. | Normal-form comparison via the reduction machine. |
| `IsId(w)` / `IsIdentity(w)` | Returns `true` if the word w is the identity word. | — |
| `#u` | The length of the word u. | — |
| `ElementToSequence(u)` / `Eltseq(u)` | Decomposes element u of a rewrite monoid into its constituent generators: if u = M.i1 * ... * M.im, returns the sequence [i1, ..., im]. | — |

*Worked example: H78E10 (Fibonacci monoid FM(2,5) on 5 generators; illustrates a*b*c*d = b^2, (c*d)^4 eq a is true, IsIdentity(a^0) is true, IsIdentity(b^2*e) is false).*

---

## 78.4 Homomorphisms

For a general treatment of homomorphisms see Chapter 16. Rewrite monoids in the category `MonRWS` are currently accepted as codomains only for monoid homomorphisms whose codomain is also a rewrite monoid or a group.

### 78.4.1 General Remarks

Monoids in the category `MonRWS` currently are accepted as codomains only for monoid homomorphisms whose codomain is a rewrite monoid as well.

### 78.4.2 Construction of Homomorphisms

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `hom< M -> N \| S >` | Returns the homomorphism from rewrite monoid M to monoid N defined by S. S may be: (i) a list/sequence/indexed set of images of the n generators M.1, ..., M.n (order matters); or (ii) a list/sequence/enumerated set of n tuples `<xi, yi>` or arrow pairs `xi -> yi`, where xi ranges over all generators of M and yi ∈ N (order does not matter). No checking is performed that the images give a well-defined homomorphism. N must be either a rewrite monoid or a group. | — |

---

## 78.5 Set Operations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Random(M, n)` | A random word of length at most n in the generators of M. | — |
| `Random(M)` | A random word (of length at most the order of M) in the generators of M. | — |
| `Representative(M)` / `Rep(M)` | An element chosen from M. | — |
| `Set(M, a, b)` | Creates the set of words w in M with a ≤ length(w) ≤ b. Parameter `Search` (`"DFS"` default for depth-first/lexicographical enumeration, or `"BFS"` for breadth-first/short-lex-order enumeration). Words may not appear in the set in the enumeration order since the result is a set. | — |
| `Set(M)` | Creates the set of all words that is the carrier set of M. Parameter `Search` (`"DFS"` default, or `"BFS"`). | — |
| `Seq(M, a, b)` | Creates a sequence S of words w in M with a ≤ length(w) ≤ b. Parameter `Search` (`"DFS"` default — words appear in lexicographical order; `"BFS"` — short-lex order). | — |
| `Seq(M)` | Creates a sequence S of words from the carrier set of M. Parameter `Search` (`"DFS"` default, or `"BFS"`). | — |

*Worked example: H78E11 (group D22 on 6 generators; illustrates Order(M) = 22, Representative, Random(M), Random(M,5), Set(M) and Seq(M : Search := "BFS")).*

---

## 78.6 Conversion to a Finitely Presented Monoid

There is a standard way to convert a rewrite monoid to a finitely presented monoid using `Relations`. The relations of the confluent rewrite monoid (which form a complete rewriting system) are passed to `quo` over the same free monoid to produce a finitely presented monoid that carries the same algebraic structure.

*Worked example: H78E12 (Fibonacci monoid FM(2,4) on 4 generators; Order(M) = 11; the confluent rewrite relations (19 equations) are fed into quo to produce a finitely presented monoid P with generators w, x, y, z).*

---

## 78.7 Bibliography

| Key | Reference |
|-----|-----------|
| **[Hol97]** | Derek Holt. *KBMAG — Knuth-Bendix in Monoids and Automatic Groups.* University of Warwick, 1997. |
| **[Sim94]** | Charles C. Sims. *Computation with finitely presented groups.* Cambridge University Press, Cambridge, 1994. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Knuth–Bendix completion **[Hol97]** | `RWSMonoid` |
| Short-lex word ordering | `RWSMonoid(:Ordering := "ShortLex")` |
| Recursive / RTRecursive word ordering | `RWSMonoid(:Ordering := "Recursive")`, `RWSMonoid(:Ordering := "RTRecursive")` |
| Weighted short-lex ordering | `RWSMonoid(:Ordering := "WTShortLex", Weights := ...)` |
| Wreath-product ordering **[Sim94, pp. 46–50]** | `RWSMonoid(:Ordering := "Wreath", Levels := ...)` |
| Rabin–Karp algorithm (space-efficient reduction) | `RWSMonoid(:RabinKarp := <l,n>)` |
| Reduction machine (finite state automaton) | `*`, `^`, `eq`, `ne`, `IsId`/`IsIdentity`, `Order`, `IsFinite`, `Set`, `Seq` |
| Monoid homomorphisms | `hom< M -> N \| S >` |
| Conversion to finitely presented monoid | `Relations(M)` used with `quo` |
