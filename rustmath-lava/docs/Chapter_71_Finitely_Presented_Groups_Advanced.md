# Chapter 71 — Finitely Presented Groups: Advanced

**Handbook part:** X — Finitely-Presented Groups
**Handbook pages:** 2205–2247 (PDF pages 2336–2381)

---

## Scope and overview

Chapter 71 covers advanced techniques for computing with finitely presented groups (fp-groups)
in Magma. The material is organised around three broad themes:

1. **Low-level operations on presentations and words** — primitive machinery (adding/deleting
   generators and relations, substring matching, word substitution and rotation) intended
   primarily for authors of user-written functions that manipulate fp-group presentations directly.

2. **Interactive coset enumeration** — a process-based wrapper around the ACE3 Todd–Coxeter
   implementation by George Havas and Colin Ramsay. The process object (category
   `GrpFPCosetEnumProc`) lets the user start, interrupt, adjust parameters, and restart an
   enumeration incrementally, adding relators or subgroup generators between runs while
   preserving accumulated coset-table information. All Todd–Coxeter parameters documented here
   also govern the standard `CosetTable` / `Index` functions described in Chapter 70.

3. **Process versions of quotient algorithms** — (a) the p-quotient process (interactive,
   class-by-class construction of a power-conjugate presentation for the largest p-quotient of
   a given class) and (b) the soluble quotient algorithm, with a detailed account of the
   underlying theory (cohomological extension, tail systems, relevant-prime calculation) and
   all tuning parameters.

The chapter warns throughout that interactive use of the p-quotient and soluble quotient
processes requires care: calling steps out of order or with incorrect assumptions can produce
incomplete or incorrect results without generating an error message.

---

## 71.1 Introduction

No intrinsics are defined in this section. It provides the motivating overview summarised above.

---

## 71.2 Low Level Operations on Presentations and Words

Low-level machinery for manipulating fp-group presentations and their elements (words). The
primary intended audience is authors of user-written Magma functions. All functions that create
a new fp-group from an existing one do so without establishing any formal relationship between
the two groups.

### 71.2.1 Modifying Presentations

Each function constructs a new fp-group by adding, deleting, or replacing a single generator
or relation in an existing presentation.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AddGenerator(G)` | Given fp-group `G` with presentation `⟨X | R⟩`, return a new fp-group with presentation `⟨X ∪ {z} | R⟩` where `z` is a fresh symbol. | Direct presentation construction. |
| `AddGenerator(G, w)` | As above, but the new generator `z` is added with the defining relation `z = w`, so the new presentation is `⟨X ∪ {z} | R ∪ {z = w}⟩`. | Direct presentation construction. |
| `AddRelation(G, r)` | Return a new fp-group whose presentation is that of `G` augmented by the relation `r`. | — |
| `AddRelation(G, g)` | Return a new fp-group whose presentation is that of `G` augmented by `g = Id(G)`. | — |
| `AddRelation(G, r, i)` | As `AddRelation(G, r)` but the new relation is inserted after the `i`-th existing relation. | — |
| `AddRelation(G, g, i)` | As `AddRelation(G, g)` but inserted after the `i`-th existing relation. | — |
| `DeleteGenerator(G, x)` | Return a new fp-group `⟨X \ {z} | R'⟩` where `R'` is obtained from `R` by deleting every relation containing an occurrence of the generator `z`. | — |
| `DeleteRelation(G, r)` | Return a new fp-group with the specified relation `r` removed from the presentation of `G`. | — |
| `DeleteRelation(G, g)` | Return a new fp-group with the relation `g = Id(G)` removed from the presentation of `G`. | — |
| `DeleteRelation(G, i)` | Return a new fp-group with the `i`-th relation of `G` deleted. | — |
| `ReplaceRelation(G, s, r)` / `ReplaceRelation(G, h, r)` / `ReplaceRelation(G, s, g)` / `ReplaceRelation(G, h, g)` | Return a new fp-group in which an existing relation `s` (or `h = Id(G)`) is replaced by `r` (or `g = Id(G)`). | — |
| `ReplaceRelation(G, i, r)` | Return a new fp-group in which the `i`-th relation is replaced by `r`. | — |
| `ReplaceRelation(G, i, g)` | Return a new fp-group in which the `i`-th relation is replaced by `g = Id(G)`. | — |

*Worked example: H71E1 (`ReplaceRelation` varying one relation in a 6-generator presentation across 24 parameter combinations; orders and subgroup indices are tabulated).*

### 71.2.2 Low Level Operations on Words

String-level operations on elements (words) of fp-groups: substitution, elimination, matching,
rotation, and subword extraction.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Eliminate(u, x, v)` | In word `u`, replace every occurrence of generator `x` by `v` and every occurrence of `x⁻¹` by `v⁻¹`. Returns the resulting word. | Direct string rewriting. |
| `Eliminate(U, x, v)` | Apply `Eliminate(·, x, v)` to every word in the set `U`. Returns the set of resulting words. | Direct string rewriting. |
| `Match(u, v, f)` | Starting from position `f` in `u`, find the least position `l ≥ f` such that `v` occurs as a subword of `u` beginning at letter `l`. Returns `true, l` if found; `false` otherwise. | Sequential substring search. |
| `RotateWord(u, n)` | Cyclic rotation of `u` by `n` places (positive = left-to-right; negative = right-to-left; 0 = identity). | Direct index arithmetic on the word. |
| `Substitute(u, f, n, v)` | Replace the subword of `u` of length `n` starting at position `f` by `v`. If `v = Id(G)` the subword is deleted. | Direct string replacement. |
| `Subword(u, f, n)` | Return the subword of `u` consisting of `n` consecutive letters starting at position `f`. | Direct index extraction. |

*Worked example: H71E2 (free group on `x, y, z`; demonstrates `Eliminate`, `Match`, `Substitute`, `RotateWord`, `ExponentSum`, `GeneratorNumber`).*

---

## 71.3 Interactive Coset Enumeration

### 71.3.1 Introduction

The Todd–Coxeter implementation in Magma is based on the stand-alone programme **ACE3**
by George Havas and Colin Ramsay (University of Queensland). An interactive coset
enumeration is realised as a `GrpFPCosetEnumProc` object. It can be created, modified
(additional relators or subgroup generators added), started, interrupted, and restarted with
modified parameters, with as much information from prior runs reused as possible.

Canonical references: **[CDHW73]** (Todd–Coxeter algorithm analysis), **[Hav91]** (coset
enumeration strategies), **[Ram]** (ACE3 manual and source).

### 71.3.2 Constructing and Modifying a Coset Enumeration Process

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CosetEnumerationProcess(G, H: -)` | Create (but do not start) a coset enumeration process for the cosets of subgroup `H` in fp-group `G`. All ACE3 enumeration parameters are settable here (see full parameter list below). | Todd–Coxeter coset enumeration via **ACE3** **[CDHW73, Hav91, Ram]**. |
| `AddRelator(∼P, w)` | Add word `w` to the defining relations of the underlying group `G`, transforming the process for cosets of `H` in `G` into one for cosets of `π(H)` in `π(G)`, where `π` kills the normal closure of `⟨w⟩`. | Modifies the process in place. |
| `AddSubgroupGenerator(∼P, w)` | Add element `w` to the generators of the subgroup, transforming the process into one for cosets of `⟨H, w⟩` in `G`. | Modifies the process in place. |
| `SetProcessParameters(∼P: -)` | Change enumeration parameters of process `P`. Accepts the same parameters as `CosetEnumerationProcess`. Parameters not explicitly set retain their current values. The workspace can only be increased, not decreased, once enumeration has started. | — |

**Parameters of `CosetEnumerationProcess` / `SetProcessParameters` / enumeration functions:**

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| `Compact` | `RngIntElt` | 10 | Trigger compaction when the percentage of dead cosets exceeds this value. |
| `CosetLimit` | `RngIntElt` | 0 | Maximum number of simultaneously active cosets (0 = determined by `Workspace`). |
| `Workspace` | `RngIntElt` | 4 000 000 | Number of words allocated for the coset table. |
| `FillFactor` | `RngIntElt` | 0 | Fill fraction = 1/`FillFactor`; 0 selects `⌊(5(c+2))/4⌋` where `c` = number of columns. |
| `CTFactor` | `RngIntElt` | 1000 | Number of C-style coset definitions per step. |
| `RTFactor` | `RngIntElt` | 2000/l | Number of R-style coset applications per step (`l` = total relator length). |
| `Style` | `MonStgElt` | `"R CR"` | Enumeration style: `"R"` (HLT), `"C"` (Felsch), `"R CR"`, `"Rc"`, `"Cr"`, `"Rt"`, `"CR"`. |
| `Lookahead` | `RngIntElt` | 0 | 0 = none; 1 = partial R-style; 2 = complete C-style; 3 = complete R-style; 4 = partial C-style. |
| `Mendelsohn` | `BoolElt` | `false` | If true, apply cosets at all cyclic permutations of relators. |
| `RelationsInSubgroup` | `RngIntElt` | −1 | Whether relators are used as subgroup generators at the start: −1 = all, 0 = none, n > 0 = first n. |
| `RowFilling` | `BoolElt` | `true` | In R-style, fill holes encountered during row scanning. |
| `PrefDefMode` | `RngIntElt` | 3 | How preferred definitions (length-1 gaps) are handled: 0 = Felsch-style; 1 = immediate fill; 2 = immediate fill+deduction; 3 = queued. |
| `PrefDefSize` | `RngIntElt` | 8 | Preferred definition queue size (ring buffer of size 2n). |
| `DeductionMode` | `RngIntElt` | 4 | Deduction stack overflow behaviour: 0 = discard; 1 = discard + purge top; 2 = discard + purge all; 3 = discard entire stack; 4 = double stack + purge. |
| `DeductionSize` | `RngIntElt` | 1000 | Initial deduction stack size in words. |
| `PathCompression` | `BoolElt` | `false` | Reduce data movement during coincidence processing (may affect run time but not result). |
| `TimeLimit` | `RngIntElt` | −1 | Time limit in seconds; −1 = unlimited; 0 = one main-loop pass. |
| `LoopLimit` | `RngIntElt` | 0 | Limit on total state-machine steps; 0 = unlimited. |
| `LowerBound` | `RngIntElt` | 1 | Known lower bound for the index; enumeration terminates early when active cosets equal this bound and the table has no holes. |
| `Print` | `BoolElt` | `false` | Print a summary message at completion and (with `Messages`) progress during enumeration. |
| `Messages` | `RngIntElt` | 0 | If non-zero, print a progress message every `|n|` actions; negative enables hole monitoring. |
| `Strategy` | `MonStgElt` | — | Select a predefined strategy (see tables below); individual parameters may still override. |

**Predefined strategies** (all set `PrefDefSize := 8`, `DeductionSize := 1000`, `PathCompression := false`, `LoopLimit := 0`):

| Strategy | `Compact` | `Workspace` (×10⁶) | `FillFactor` | `CTFactor` | `RTFactor` | `Style` | `Lookahead` | `Mendelsohn` | `RelationsInSubgroup` | `RowFilling` | `PrefDefMode` | `DeductionMode` |
|----------|-----------|---------------------|--------------|------------|------------|---------|-------------|--------------|----------------------|--------------|----------------|-----------------|
| `"Default"` | 10 | 4 | 0 | 1000 | 2000/l | R CR | 0 | false | −1 | true | 3 | 4 |
| `"Easy"` | 100 | 1 | 1 | 0 | 1000 | R | 0 | false | 0 | true | 0 | 0 |
| `"Hard"` | 10 | 10 | 0 | 1000 | 1 | CR | 0 | false | −1 | true | 3 | 4 |
| `"Felsch"` | 10 | 4 | 0 | 1000 | 0 | C | 0 | false | −1 | false | 3 | 4 |
| `"HLT"` | 10 | 4 | 1 | 0 | 1000 | R | 1 | false | 0 | true | 0 | 0 |
| `"CT"` | 100 | 4 | 1 | 1000 | 0 | C | 0 | false | 0 | false | 0 | 4 |
| `"RT"` | 100 | 4 | 1 | 0 | 1000 | R | 0 | false | 0 | false | 0 | 0 |
| `"Sims1"` | 10 | 4 | 1 | 0 | 1000 | R | 0 | false | 0 | true | 0 | 0 |
| `"Sims3"` | 10 | 4 | 1 | 0 | 1000 | Rt | 0 | false | 0 | true | 0 | 4 |
| `"Sims5"` | 10 | 4 | 1 | 0 | 1000 | R | 0 | true | 0 | true | 0 | 0 |
| `"Sims7"` | 10 | 4 | 1 | 0 | 1000 | Rt | 0 | true | 0 | true | 0 | 4 |
| `"Sims9"` | 10 | 4 | 1 | 1000 | 0 | C | 0 | false | 0 | false | 0 | 4 |

### 71.3.3 Starting and Restarting an Enumeration

All four functions accept the same parameter set as `CosetEnumerationProcess`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `StartEnumeration(∼P: -)` | Discard all information in `P` and start a fresh enumeration. May be called at any time. | ACE3 Todd–Coxeter **[CDHW73, Hav91]**. |
| `RedoEnumeration(∼P: -)` | Retain the current coset table and restart from coset 1. Legal only if `P` contains a valid coset table (`CanRedoEnumeration`). Intended for use after additional relators/subgroup generators have been added. | Reuses existing coset table rows; ACE3. |
| `CanRedoEnumeration(P)` | Returns `true` if calling `RedoEnumeration` is legal for `P`. | — |
| `ContinueEnumeration(∼P: -)` | Resume a previously interrupted enumeration from the coset at which it stopped. Legal only if `P` has a valid coset table and the subgroup has not changed (`CanContinueEnumeration`). | Minimal-overhead continuation in ACE3. |
| `CanContinueEnumeration(P)` | Returns `true` if calling `ContinueEnumeration` is legal for `P`. | — |
| `ResumeEnumeration(∼P: -)` | Resume in the cheapest permitted way: calls `ContinueEnumeration` if legal, else `RedoEnumeration` if legal, else `StartEnumeration`. | Dispatches to one of the above. |

### 71.3.4 Accessing Information

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CosetsSatisfying(P : -)` | Given a process `P` with a valid coset table, return a set of coset representatives satisfying the conditions specified by parameters `First`, `Last`, `Limit`, `Normalizing`, `Order`, `Print`. Requires a valid coset table (`HasValidCosetTable`). | Coset table lookup. |
| `CosetSatisfying(P : -)` | As `CosetsSatisfying` but returns at most one coset (other than `H`); equivalent to `CosetsSatisfying` with `Limit := 1`, `First := 2`. | Coset table lookup. |
| `CosetTable(P)` | Return the current coset table of `P` as a map `f : {1,…,r} × G → {0,…,r}` where `f(i, x)` is the coset index to which coset `i` maps under the action of `x`; 0 indicates an unknown entry (incomplete table). Requires a valid coset table. | — |
| `HasValidCosetTable(P)` | Returns `true` if `P` contains a valid (but not necessarily closed) coset table. | — |
| `HasClosedCosetTable(P)` / `HasCompleteCosetTable(P)` | Returns `true` if `P` contains a closed, valid coset table (no unknown entries). | — |
| `ExcludedConjugate(P)` | Return a set containing at most one word of the form `gᵢ⁻¹hⱼgᵢ` (where `gᵢ` is a generator of `G` and `hⱼ` a generator of `H`) that is not known to lie in `H` from the coset table. Empty set implies `H` is normal in `G`. | Coset table lookup. |
| `ExcludedConjugates(P)` | As above but returns all such words. | Coset table lookup. |
| `ExistsCosetSatisfying(P : -)` | Returns whether there exists a coset (other than `H`) satisfying the specified conditions; if so, also returns a representing word. Same parameters as `CosetSatisfying`. | Coset table lookup. |
| `ExistsExcludedConjugate(P)` | Returns whether there exists a word `gᵢ⁻¹hⱼgᵢ` not known to lie in `H`; if so, also returns it. A negative result always proves `H` is normal (even for incomplete tables). | Coset table lookup. |
| `ExistsNormalisingCoset(P)` / `ExistsNormalizingCoset(P)` | Returns whether an element of `G \ H` normalising `H` can be detected from the coset table; if so, also returns such an element. | Coset table lookup. |
| `Group(P)` | Returns the group underlying `P` as a finitely presented group. | — |
| `Index(P)` | Returns `[G : H]`. Legal only if the last enumeration completed with a finite index (`HasValidIndex`). | — |
| `HasValidIndex(P)` | Returns `true` if the last enumeration completed successfully with a finite index. | — |
| `MaximalNumberOfCosets(P)` | Returns the maximum number of simultaneously active cosets during the last enumeration (1 if no enumeration has run). Useful for assessing parameter efficiency. | — |
| `Subgroup(P)` | Returns the subgroup `H` underlying `P` as a subgroup of the underlying group `G`. | — |
| `TotalNumberOfCosets(P)` | Returns the total number of cosets defined during the last enumeration (1 if no enumeration has run). | — |

**Parameters of `CosetsSatisfying` / `CosetSatisfying` / `ExistsCosetSatisfying`:**

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| `First` | `RngIntElt` | 1 | Starting coset number for the search (set to 2 to exclude `H`). |
| `Last` | `RngIntElt` | 0 | Stopping coset number (0 = search all active cosets from `First`). |
| `Limit` | `RngIntElt` | 0 | Abort after finding this many satisfying cosets (0 = no limit; not available in `CosetSatisfying`). |
| `Normalizing` | `BoolElt` | `false` | If true, select cosets `x` such that `x⁻¹hᵢx ∈ H` is known from the coset table for every generator `hᵢ` of `H`. |
| `Order` | `RngIntElt` | 0 | Select cosets `x` such that `xⁿ ∈ H` is known from the coset table. |
| `Print` | `RngIntElt` | 0 | If positive, print the satisfying coset representatives found. |

*Worked examples: H71E3 (Harada–Norton group index 1 140 000; overflow, `HasValidCosetTable`, `CosetTable`, `SetProcessParameters` with strategy Hard, `ContinueEnumeration`). H71E4 (finite group of order 10 752; copying processes, `AddSubgroupGenerator`, `RedoEnumeration`, `AddRelator`, `ContinueEnumeration`, `Group`, `Subgroup`). H71E5 (infinite group; `ExistsNormalisingCoset` from a partial table). H71E6 (`ExistsExcludedConjugate`, `ExcludedConjugates`, `AddSubgroupGenerator`, `RedoEnumeration`, `ExistsCosetSatisfying`, `CosetsSatisfying` with `Order` and `Normalizing`).*

### 71.3.5 Induced Permutation Representations

Given `H` of finite index in `G`, the right-multiplication action of `G` on cosets of `H`
defines a permutation representation `ρ : G → S_{[G:H]}` with kernel = core of `H` in `G`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CosetAction(P)` | Given a process with a valid finite index, return: (a) the permutation representation `ρ` of `G`; (b) the image group `ρ(G)`; (c) the kernel of `ρ` (when computable). Requires `HasValidIndex`. | Right-coset action; Todd–Coxeter. |
| `CosetImage(P)` | Return the image of `ρ` as a permutation group on `[G:H]` points. Requires `HasValidIndex`. | Right-coset action. |
| `CosetKernel(P)` | Return the kernel of `ρ`. Only available when the index is sufficiently small. Requires `HasValidIndex`. | Right-coset action. |

### 71.3.6 Coset Spaces and Transversals

The (right) indexed coset space is a G-set on `{1, …, m}` where `i` represents the coset
`cᵢ = H·tᵢ`; the action is `f(cᵢ, x) = cₖ ⟺ cᵢ · x = cₖ`. When some products are unknown
the coset space is *incomplete*.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CosetSpace(P)` | Return the (possibly incomplete) coset space defined by the current state of `P`. Requires a valid coset table. | Derived from coset table. |
| `RightCosetSpace(P)` | Return the explicit right coset space (elements are pairs `⟨H, x⟩` for transversal elements `x`). Requires a valid coset table. | Derived from coset table. |
| `LeftCosetSpace(P)` | Return the explicit left coset space (elements are pairs `⟨x, H⟩`). Requires a valid coset table. | Derived from coset table. |
| `Transversal(P)` / `RightTransversal(P)` | Given a process with valid finite index: return (a) an indexed set `T` of elements of `G` forming a right transversal for `H`; (b) the transversal map `φ : G → T` where `φ(g) = tᵢ` whenever `g ∈ H · tᵢ`. Requires `HasValidIndex`. | Derived from closed coset table. |

*Worked examples: H71E7 (subgroup of index 448 in a group of order 10 752; `Transversal`, `CosetAction`, faithful permutation representation). H71E8 (infinite group; incomplete coset space via `CosetSpace`, `IsComplete`).*

---

## 71.4 p-Quotients (Process Version)

Let `F` be an fp-group, `p` a prime and `c` a positive integer. The p-quotient algorithm
constructs a consistent power-conjugate presentation (pcp) for the largest p-quotient of `F`
of lower exponent-p class at most `c`. The reference algorithm is **[NO96]**.

Generators `{a₁, …, ad}` (where `d` is the Frattini rank) generate the quotient; generators
`{ad+1, …, an}` are defined by relations. Each pcp generator carries a weight (the class at
which it was introduced).

### 71.4.1 The p-Quotient Process

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `pQuotientProcess(F, p, c: -)` | Create a p-quotient process: initialise by computing a pcp for the largest p-quotient of `F` of class at most `c` (class limit 127 if `c = 0`). Returns a process `P`. Accepts the same parameters as `pQuotient`. | p-quotient algorithm **[NO96]**. |
| `NextClass(∼P : -)` | Advance `P` from a class-`c` pcp to a class-`(c+1)` pcp. | p-quotient algorithm **[NO96]**. |
| `NextClass(∼P, k : -)` | Continue advancing until the class-`k` pcp is constructed. Parameters: `Exponent` (enforce exponent law), `Metabelian` (restrict to metabelian quotient), `Print`, `MaxOccurrence` (sequence bounding occurrences of class-1 generators in new definitions). | p-quotient algorithm **[NO96]**. |

### 71.4.2 Using p-Quotient Interactively

Assumes a pcp for the class-`c` quotient has been computed; the following steps construct
the class-`(c+1)` quotient. Steps must be executed in order; violations may produce silent
incorrect output.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `StartNewClass(∼P: -)` | Commence construction of the next class. Must be called first. | p-quotient algorithm **[NO96]**. |
| `Tails(∼P: -)` | Add all tails for the current class to the pcp. | Tails step of p-quotient **[NO96]**. |
| `Tails(∼P, k: -)` | Add tails for weight `k` only (assumes tails for weights `c+1, c, …, k+1` have been added). Valid range `k ∈ {2,…,c+1}`. Parameter `Metabelian` restricts to the metabelian p-quotient. | Tails step **[NO96]**. |
| `Consistency(∼P: -)` | Apply the full consistency algorithm to compute redundancies among all tails. Assumes all tails have been added. Parameter `Metabelian`. | Consistency check **[NO96]**. |
| `Consistency(∼P, k: -)` | Apply the consistency algorithm for tails of weight `k` only. Valid range `k ∈ {3,…,c+1}`. | Consistency check **[NO96]**. |
| `CollectRelations(∼P)` | Collect the defining relations of `F` in the current pcp. Must be called after the tails step is complete; incorrect results if tails are incomplete. | Collection algorithm. |
| `ExponentLaw(∼P : -)` | Enforce the supplied exponent law on the current pcp for all weights. Assumes tails are complete. Parameters: `Exponent` (the exponent `m` to enforce), `Print`, `Trial` (gather statistics without actually collecting), `ShortList` (restrict enforcement words), `DisplayList`, `IdentifyFilters`, `InitialSegment`, `Report`. | Exponent enforcement **[NO96]**. |
| `ExponentLaw(∼P, Start, Fin: -)` | Enforce the exponent law for weights in `[Start, Fin]`. Same parameters as above. | Exponent enforcement **[NO96]**. |
| `EliminateRedundancy(∼P)` | Eliminate all redundant generators from the pcp. May be applied at any time. | Gaussian elimination on pcp. |
| `Display(P)` / `Display(P, DisplayLevel)` | Print the pcp. `DisplayLevel`: 1 = order and class; 2 = also non-trivial relations; 3 = also generator structure and map from generators of `F` to pcp generators. | — |
| `RevertClass(∼P)` | Revert from a class-`(c+1)` pcp back to the class-`c` pcp. Can only be applied once per class during construction. | — |
| `pCoveringGroup(∼P)` / `pCoveringGroup(G)` | Compute a pcp for the p-covering group. In the process case, equivalent to `Tails(∼P); Consistency(∼P); EliminateRedundancy(∼P)`. | p-covering group **[NO96]**. |
| `GeneratorStructure(P)` | Display the structure (definitions) of all pcp generators. | — |
| `GeneratorStructure(P, Start, Fin)` | Display generator structure for pcp generators numbered `Start` to `Fin`. | — |
| `Jacobi(∼P, c, b, a, ∼r)` / `Jacobi(∼P, c, b, a)` | Calculate the Jacobi identity for generators `c, b, a` and echelonise the result against the current pcp. If a redundant generator is found, `r` is set to its number; otherwise `r = 0`. | Jacobi identity; echelonisation. |
| `Collect(P, Q)` | Collect the word in pcp generators specified by the sequence `Q` of generator-exponent pairs; return the normal word as an exponent vector. | Collection in pcp. |
| `EcheloniseWord(∼P, ∼r)` / `EcheloniseWord(∼P)` | Echelonise the word most recently collected by `Collect` against the pcp relations. If a redundant generator results, `r` is its number; otherwise `r = 0`. Must be called immediately after `Collect`. | Echelonisation. |
| `SetDisplayLevel(∼P, Level)` | Set the display level for the process to `Level`. | — |
| `ExtractGroup(P)` | Extract the group `G` defined by the current pcp as a member of `GrpPC` (finite soluble groups), together with: the natural homomorphism `π : F → G`; a sequence `S` of definition descriptors; and a flag indicating whether `G` is the maximal p-quotient of `F`. Each entry `S[k]` is `[0,r]` (defined by image of `F.r`), `[r,0]` (defined by power relation for `G.r`), or `[r,s]` (defined by commutator `[G.r, G.s]`). | — |
| `Order(P)` | Return the order of the group defined by the current pcp. | — |
| `FactoredOrder(P)` | Return the factored order of the group defined by the current pcp. | — |
| `NumberOfPCGenerators(P)` | Return the number of pc-generators of the group defined by the current pcp. | — |
| `pClass(P)` | Return the lower exponent-p class of the group defined by the current pcp. | — |
| `NuclearRank(G)` / `NuclearRank(P)` | Return the rank of the p-multiplicator of the p-group `G` (supplied directly or via process `P`). | — |
| `pMultiplicatorRank(G)` / `pMultiplicatorRank(P)` | Synonym for `NuclearRank`. | — |

*Worked examples: H71E9 (two-generator exponent-9 group; class-4 setup via `pQuotientProcess`, then `NextClass` to class 5, then manual class 6 via `StartNewClass`, `Tails`, `Consistency`, `CollectRelations`, `ExponentLaw`, `EliminateRedundancy`, `Display`). H71E10 (free product C₅ * C₅; `NextClass` with `MaxOccurrence` bounds). H71E11 (R(2,5) exponent-5 group; `StartNewClass`, `Tails`, `Consistency`, `ExponentLaw` with `InitialSegment := [<1,2>]` and `Trial := true`). H71E12 (7-generator 7-group; `GeneratorStructure`, `Jacobi`, `Collect`, `EcheloniseWord`, `CollectRelations`, `Consistency`, `EliminateRedundancy`, `Display`).*

---

## 71.5 Soluble Quotients

### 71.5.1 Introduction

The soluble quotient algorithm constructs epimorphisms `ε : F →→ G` from a finitely
presented group `F` to finite soluble groups `G`. It iterates the process of extending a
known quotient by an irreducible module section.

### 71.5.2 Construction (theory)

A finite soluble group `G` has a refined chief series `G = G⁽⁰⁾ > G⁽¹⁾ > … > G⁽ⁿ⁾ = ⟨1⟩`
in which each section `G⁽ⁱ⁾/G⁽ⁱ⁺¹⁾` is elementary abelian of prime power order and
irreducible as a `G/G⁽ⁱ⁾`-module. The algorithm builds `G` by successively choosing
irreducible modules `Mᵢ` and 2-cocycle classes `ζᵢ ∈ H²(G/G⁽ⁱ⁾, Mᵢ)` defining extensions.

Given a current quotient `δ : F →→ H` and a candidate `H`-module `M`, a lift `ε : F →→ H.M`
exists iff the images `δ(fᵢ) = hᵢ` admit corrections `xᵢ ∈ M` with `ε(fᵢ) = hᵢxᵢ` satisfying
the defining relations of `F`. This yields a linear equation system (together with consistency
equations) whose solution space `S ⊆ Mʳ × H²(H, M)` parametrises valid lifts. The split
extension sub-space `SS ⊆ S` and its sub-space `SC` of non-surjective lifts are computed;
the multiplicity `a = dim_K(S/SS)` and `b = dim_K(SS/SC)` are determined. After taking a
maximal extension `ε : F →→ H.Mᵇ`, the module `M` need not be revisited for split extensions.

### 71.5.3 Calculating the Relevant Primes (theory)

The relevant primes are the prime divisors of `|G|`. For each rational representation `Δ` of
`H`, the prime set `P_Δ` is computed; their union (plus primes dividing `|G|`) is a superset
of the actual prime divisors. The algorithm detects infinite abelian sections (all primes
relevant, no maximal finite soluble quotient exists) and uses known prime information from
prior quotients to reduce the search.

### 71.5.4 The Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SolubleQuotient(F, n : -)` / `SolvableQuotient(F, n : -)` | Compute a soluble quotient `ε : F →→ G` of order `n` (if `n > 0`; `n = 0` means order unconstrained). Returns: `G`, the epimorphism `ε`, a sequence describing the series/modules by which `G` was built, and a string giving the termination reason. | Soluble quotient algorithm (cohomological extension + relevant-prime search). |
| `SolubleQuotient(F : -)` / `SolvableQuotient(F : -)` | As above with `n = 0` (unconstrained order). | Same algorithm. |
| `SolubleQuotient(F, P : -)` / `SolvableQuotient(F, P : -)` | As above where `P` is a set of relevant primes; computes the largest quotient whose order has prime divisors only in `P`. | Same algorithm. |

**Termination conditions:**
1. (Normal) A quotient of the requested order has been constructed.
2. (Normal) A maximal quotient (w.r.t. the given constraints) has been constructed.
3. (Abort with warning) A bound on the series or subseries length has been hit.
4. (Abort with warning) A limit on the quotient or section size has been hit.
5. (Abort with warning) A free abelian section has been detected.

**Parameters of `SolubleQuotient` / `SolvableQuotient`:**

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| `SeriesLength` | `RngIntElt` | 0 | Maximum length of the chief series (0 = unlimited). |
| `SubseriesLength` | `RngIntElt` | 0 | Maximum length of a subseries within a section (0 = unlimited; for derived series, maximum exponent in a prime-power-order section). |
| `#QuotientSize` | `RngIntElt` | 0 | If > 0, abort when a quotient of order ≥ this value is found. |
| `#SectionSize` | `RngIntElt` | 0 | If > 0, abort when a section of order ≥ this value is encountered. |
| `#MSQ Series` | `MonStgElt` | `"sag"` | Series to use: `"sag"` (usually most efficient), `"derived"`, `"lowercentral"` (nilpotent quotients only), `"pcentral"` (p-group quotients only). |
| `MSQ PrimeSearchModus` | `RngIntElt` | 3 | When relevant-prime search is invoked: 0 = never; 1 = after termination; 2 = as 1 but continues on new primes; 3 = after each main series step; 4 = after each elementary abelian layer; 5 = after each successful lift. |
| `MSQ ModulCalcModus` | `RngIntElt` | 0 | Use tensor/skew-symmetric products to restrict modules in sag-series: 0 = disabled; 1 = fast version; 2 = full version (recommended only for large quotients). |
| `MSQ CollectorModus` | `RngIntElt` | 2 | Symbolic collector setup: 0 = full precalculation; 1 = partial; 2 = dynamic (default). |
| `Print` | `RngIntElt` | 0 | Timing/status verbosity (0–5). |
| `Verbose MSQ Messages` | — | max 2 | If ≥ 1, print sizes of new soluble quotients found. |
| `Verbose MSQ PrimeSearch` | — | max 15 | Bitflag: 1 = rational-representation timings; 2 = integral-representation timings; 4 = prime-finding timings; 8 = print new relevant primes. |
| `Verbose MSQ RepsCheck` | — | max 3 | 1 = extension-checking timings; 2 = module statistics. |
| `Verbose MSQ RepsCalc` | — | max 3 | 1 = module-calculation timings; 2 = module-calculation statistics. |
| `Verbose MSQ Collector` | — | max 1 | If 1, print timing for symbolic collector setup. |
| `Verbose MSQ TraceFunc` | — | max 2 | Trace main function calls (1) or most function calls (2). |

---

## 71.6 Bibliography

| Key | Reference |
|-----|-----------|
| **[CDHW73]** | John J. Cannon, Lucien A. Dimino, George Havas, and Jane M. Watson. *Implementation and analysis of the Todd-Coxeter algorithm.* Math. Comp., 27:463–490, 1973. |
| **[Hav91]** | G. Havas. *Coset enumeration strategies.* In ISSAC'91, pages 191–199. ACM Press, 1991. |
| **[NO96]** | M. F. Newman and E. A. O'Brien. *Application of computers to questions like those of Burnside. II.* Internat. J. Algebra Comput., 6(5):593–605, 1996. |
| **[Ram]** | Colin Ramsay. *ACE.* URL: http://www.csee.uq.edu.au/~cram/. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Todd–Coxeter coset enumeration (ACE3) **[CDHW73, Hav91, Ram]** | `CosetEnumerationProcess`, `StartEnumeration`, `RedoEnumeration`, `ContinueEnumeration`, `ResumeEnumeration`, `CosetAction`, `CosetImage`, `CosetKernel`, `Transversal`, `RightTransversal` |
| Coset table inspection | `CosetTable`, `HasValidCosetTable`, `HasClosedCosetTable`, `HasCompleteCosetTable`, `CosetsSatisfying`, `CosetSatisfying`, `ExistsCosetSatisfying`, `ExcludedConjugate`, `ExcludedConjugates`, `ExistsExcludedConjugate`, `ExistsNormalisingCoset`, `ExistsNormalizingCoset`, `Index`, `HasValidIndex`, `MaximalNumberOfCosets`, `TotalNumberOfCosets`, `Group`, `Subgroup` |
| Coset spaces and transversals | `CosetSpace`, `RightCosetSpace`, `LeftCosetSpace`, `Transversal`, `RightTransversal` |
| p-quotient algorithm **[NO96]** | `pQuotientProcess`, `NextClass`, `StartNewClass`, `Tails`, `Consistency`, `CollectRelations`, `ExponentLaw`, `EliminateRedundancy`, `pCoveringGroup`, `RevertClass` |
| Power-conjugate presentation utilities | `Display`, `GeneratorStructure`, `SetDisplayLevel`, `Jacobi`, `Collect`, `EcheloniseWord`, `ExtractGroup`, `Order`, `FactoredOrder`, `NumberOfPCGenerators`, `pClass`, `NuclearRank`, `pMultiplicatorRank` |
| Soluble quotient algorithm (cohomological extension + relevant primes) | `SolubleQuotient`, `SolvableQuotient` |
| Direct presentation manipulation | `AddGenerator`, `AddRelation`, `DeleteGenerator`, `DeleteRelation`, `ReplaceRelation` |
| Word-level operations | `Eliminate`, `Match`, `RotateWord`, `Substitute`, `Subword` |
