# Chapter 73 — Braid Groups

**Handbook part:** X — Finitely-Presented Groups
**Handbook pages:** 2291–2338 (PDF pages 2422–2471)

---

## Scope and overview

Chapter 73 covers the Magma category `GrpBrd` for braid groups — a specialised class of
finitely presented groups for which the word problem is solvable. The braid group B_n on n
strings was introduced by Artin **[Art47]** and admits two standard presentations implemented
in Magma:

- **Artin presentation** — n−1 generators σ₁,…,σ_{n−1} with the braid relations
  σᵢσⱼ = σⱼσᵢ (|i−j| > 1) and σᵢσᵢ₊₁σᵢ = σᵢ₊₁σᵢσᵢ₊₁.
- **BKL presentation** (Birman–Ko–Lee **[BKL98]**) — n(n−1)/2 generators a_{r,t}
  (n ≥ r > t ≥ 1) with a different but equivalent set of relations; the BKL generator
  a_{r,t} corresponds to (σ_{r−1}···σ_{t+1})σ_t(σ_{t+1}⁻¹···σ_{r−1}⁻¹) in the Artin
  generators.

Both presentations are special cases of Garside groups **[Deh02]**, and both give B_n a
lattice structure with a fundamental element Δ, left/right partial orderings, and a
distinguished set of simple elements (permutation elements). Every braid admits a
canonical factor product (CFP) representation Δˡc₁···c_k; bringing this into **left or
right normal form** is the key computational task, with cost O(k²n log n) for the Artin
presentation and O(k²n) for the BKL presentation **[ECH+92, BKL98]**.

A central application motivating the implementation is **public-key cryptography based on
braid groups** **[AAG99, KLC+00]**. The hardness assumptions rely on variants of the
conjugacy problem. Magma provides the **super summit set** **[Gar69, ERM94]** and the
**ultra summit set** **[Geb03]** as conjugacy-class invariants; ultra summit sets are
generally much smaller and can be computed for braids on up to ~100 strings with
canonical length up to ~1000 **[Geb03]**. Recent attacks on particular cryptosystems
**[GKT+02, HS03, Hug02, LL02, LP03]** and advances in conjugacy-problem analysis
**[GM02, Geb03]** have cast doubt on the security of braid-group cryptosystems; the
question of whether they can be made secure by suitable parameter choice remains open.

---

## 73.1 Introduction

### 73.1.1 Lattice Structure and Simple Elements

The positive monoid B⁺ embeds in B. Two partial orderings are defined: u ⪯ v (left
divisibility, a ⪯ b iff a⁻¹b ∈ B⁺) and u ⪰ v (right divisibility, a ⪰ b iff ab⁻¹ ∈ B⁺).
B is a lattice under both orderings, giving left-gcd, left-lcm, right-gcd, and right-lcm for
any pair. The **fundamental element** Δ is the left-lcm (= right-lcm) of all generators.
**Simple elements** are positive elements c satisfying c ⪯ Δ; they correspond bijectively
to permutations on n points (Artin: all n! permutations; BKL: the (2n)!/(n!(n+1)!)
permutations that are products of parallel descending cycles). Every braid can be written
as Δˡc₁···c_k (a CFP). In Magma the ordering ⪯ is operator `le` and ⪰ is `ge`.

### 73.1.2 Representing Elements of a Braid Group

Elements can be represented internally as (a) words in Artin generators, (b) words in BKL
generators, (c) products of Artin simple elements, or (d) products of BKL simple elements.
Automatic conversions occur when mixing representations. The default presentation is
selected at construction time via `BraidGroup` and can be changed with `SetPresentation`.
By default, group operations use the CFP representation (`SetForceCFP` controls this).
Print format is controlled by `SetElementPrintFormat` with values `"Word"`, `"CFP"`, or
`"Both"` (default).

### 73.1.3 Normal Form for Elements of a Braid Group

A CFP Δˡc₁···c_k is in **left normal form** if c₁ ≠ Δ, c_k ≠ 1, and (cᵢ⁻¹Δ) ∧_l c_{i+1} = 1
for all i. Similarly for **right normal form**. The infimum, canonical length, and supremum
of an element are the exponent l, the count k, and l+k, respectively, and agree between
left and right normal forms. The normalisation algorithm for the Artin presentation has
complexity O(k²n log n) **[ECH+92]**; for BKL it is O(k²n) **[BKL98]**.

### 73.1.4 Mixed Canonical Form and Lattice Operations

The **left-mixed canonical form** of x is a pair ⟨a, b⟩ with a, b positive in left normal
form, x = a⁻¹b, and left-gcd(leading factor of a, leading factor of b) = 1; the **right-mixed
canonical form** similarly uses x = ab⁻¹ and right normality. Lattice operations follow:
left-gcd(u, v) = u · a⁻¹ where ⟨a, b⟩ is the left-mixed canonical form of u⁻¹v; left-lcm(u,
v) = u · a where ⟨a, b⟩ is the right-mixed canonical form of u⁻¹v; etc. **[ECH+92]**.

### 73.1.5 Conjugacy Testing and Conjugacy Search

Two elements x, y are conjugate iff their **super summit sets** S_x = S_y or their **ultra
summit sets** U_x = U_y. Three class invariants are defined:

- **P_x** — positive conjugates (empty when ss-inf(x) < 0).
- **S_x** (super summit set) — conjugates achieving maximum infimum and minimum
  supremum simultaneously.
- **U_x** (ultra summit set) — elements of S_x fixed by some iterate of the cycling
  operation c.

Both S_x and U_x are non-empty and finite, and representatives can be reached from any
element by finitely many cycling and decycling operations **[ECH+92, BKL98, Geb03]**.
Closure-under-minimal-simple-elements algorithms for computing these sets are given in
**[FGM03]** (for P_x and S_x) and **[Geb03]** (for U_x). `IsConjugate` uses ultra summit
sets for conjugacy testing and search.

---

## 73.2 Constructing and Accessing Braid Groups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BraidGroup(n: -)` / `BraidGroup(GrpBrd, n: -)` | Returns the braid group on n strings (n−1 Artin generators). Parameter `Presentation` (`"Artin"` default or `"BKL"`) selects the default presentation. | Artin or BKL presentation **[Art47, BKL98]**. |
| `GetPresentation(B)` | Returns a string `"Artin"` or `"BKL"` indicating the current default presentation for B. | — |
| `SetPresentation(~B, s)` | Sets the default presentation for B to `s` (`"Artin"` or `"BKL"`). | — |
| `GetForceCFP(B)` | Returns whether arithmetic operations always convert arguments to CFP (product of simple elements) form. | — |
| `SetForceCFP(~B, b)` | If `b` is false, arithmetic operations on word-represented elements are performed without converting to CFP first. Default is true (always use CFP). | — |
| `GetElementPrintFormat(B)` | Returns current print format string: `"Word"`, `"CFP"`, or `"Both"`. | — |
| `SetElementPrintFormat(~B, s)` | Sets the print format for elements of B to `s` (`"Word"`, `"CFP"`, or `"Both"`). Default is `"Both"`. | — |
| `NumberOfStrings(B)` | Returns the number of strings on which B is defined (= n). | — |
| `NumberOfGenerators(B)` / `Ngens(B)` | Returns the number of Artin generators of B (= n−1), regardless of the selected presentation. | — |

*Worked example: H73E1 (constructing B_6, BKL fundamental element, Artin/BKL simple elements, `SetElementPrintFormat`, `SetPresentation`, `IsProductOfParallelDescendingCycles`).*

---

## 73.3 Creating Elements of a Braid Group

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Representative(B)` / `Rep(B)` | Returns a representative element of B. | — |
| `Identity(B)` / `Id(B)` / `B ! 1` | Returns the identity element of B. | — |
| `FundamentalElement(B: -)` | Returns the fundamental element Δ for the presentation indicated by the parameter `Presentation` (`"Artin"` or `"BKL"`; defaults to B's current presentation). | Δ is the left-lcm (= right-lcm) of all generators **[ECH+92, BKL98]**. |
| `Generators(B: -)` | Returns a sequence of generators for the presentation indicated by `Presentation` (defaults to B's current presentation). | — |
| `B . i` | For integer i with 0 < \|i\| < n: returns the \|i\|-th Artin generator σ_{\|i\|} if i > 0, or its inverse if i < 0. | — |
| `B . T` | For tuple T = ⟨r, t⟩ with 1 ≤ \|t\| < \|r\| ≤ n: returns the BKL generator a_{\|r\|,\|t\|} if r, t > 0, or its inverse otherwise. | — |
| `B ! [i1, ..., ik]` | For a sequence of nonzero integers with \|iⱼ\| < n: returns the product σ_{sgn(i₁),\|i₁\|}···σ_{sgn(iₖ),\|iₖ\|} as an element of B. | — |
| `B ! [T1, ..., Tk]` | For a sequence of tuples Tⱼ = ⟨rⱼ, tⱼ⟩: returns the corresponding product of BKL generators (or their inverses) as an element of B. | — |
| `B ! p` | For a permutation p on n points: returns the simple element defined by p in the current presentation. In the BKL presentation, p must be a product of parallel descending cycles (see `IsProductOfParallelDescendingCycles`); otherwise a runtime error occurs. | — |
| `B ! [p1, ..., pk]` | For a sequence of permutations: returns the product c₁···c_k where cⱼ is the simple element defined by pⱼ in the current presentation. | — |
| `B ! T` | For a tuple T = ⟨s, l, S, r⟩ where s is `"Artin"` or `"BKL"`, l and r are integers, and S is a sequence of permutations: returns the element Δˡc₁···c_k Δʳ. | — |
| `IsProductOfParallelDescendingCycles(p)` | Returns true if permutation p on n points is a product of parallel descending cycles, i.e. corresponds to a valid BKL simple element. | — |
| `Random(B, r, s, m, n: -)` / `RandomCFP(B, r, s, m, n: -)` | Returns a pseudo-random element Δᵉ c₁···c_l where e is drawn uniformly from [r,s] and l from [m,n], and each cᵢ is a uniformly random simple element. Parameter `Presentation` selects Artin or BKL. One-argument versions are short for `Random(B, 0, 0, 0, 42)`. | Uniform sampling over simple elements. |
| `Random(B: -)` / `RandomCFP(B: -)` | Short for `Random(B, 0, 0, 0, 42)`. | — |
| `Random(B, m, n: -)` / `RandomWord(B, m, n: -)` | Returns a pseudo-random element as a word of length l ∈ [m,n] in generators ∪ their inverses (avoiding consecutive cancellation). Parameter `Presentation` selects the generator set. | Uniform length then uniform random walk. |
| `RandomWord(B: -)` | Short for `RandomWord(B, 0, 42)`. | — |

*Worked example: H73E1 (continued — `Random`, `RandomWord`, coercion from integers/tuples/permutations, `IsProductOfParallelDescendingCycles`, switching presentation).*

---

## 73.4 Working with Elements of a Braid Group

### 73.4.1 Accessing Information

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Parent(u)` | Returns the parent braid group B of element u. | — |
| `#u` | Returns the length of the word in generators of the current presentation representing u. Not an invariant of u. | — |
| `CanonicalFactorRepresentation(u: -)` / `CFP(u: -)` | Returns a tuple T = ⟨s, l, S, r⟩ describing the CFP representation of u in the presentation indicated by `Presentation` (default: current presentation of B). s is `"Artin"` or `"BKL"`, l and r are integers, S is a sequence of permutations, and the element is Δˡc₁···c_k Δʳ. | — |
| `WordToSequence(u: -)` / `ElementToSequence(u: -)` / `Eltseq(u: -)` | Returns a sequence describing the word in generators of the indicated presentation. For Artin: a sequence of integers [e₁i₁,…,eₖiₖ]. For BKL: a sequence of tuples [⟨e₁r₁,e₁t₁⟩,…]. Parameter `Presentation`. | — |
| `InducedPermutation(u)` | Returns the permutation on n points induced by u acting on the strings of B. | Direct from CFP or word. |
| `CanonicalLength(u: -)` | Returns the canonical length k of u in the left normal form for the indicated presentation. Converts u to left normal form. | Left normal form **[ECH+92, BKL98]**. |
| `Infimum(u: -)` | Returns the infimum l (the leading power of Δ) of u in the left normal form. Converts u to left normal form. | Left normal form **[ECH+92, BKL98]**. |
| `Supremum(u: -)` | Returns the supremum l+k of u. Converts u to left normal form. | Left normal form **[ECH+92, BKL98]**. |
| `SuperSummitCanonicalLength(u: -)` | Returns the canonical length of a super summit representative of u (= minimal canonical length over all conjugates). | Super summit set computation **[Gar69, ERM94]**. |
| `SuperSummitInfimum(u: -)` | Returns the infimum of a super summit representative of u (= maximal infimum over all conjugates). | Super summit set computation **[Gar69, ERM94]**. |
| `SuperSummitSupremum(u: -)` | Returns the supremum of a super summit representative of u (= minimal supremum over all conjugates). | Super summit set computation **[Gar69, ERM94]**. |

*Worked example: H73E2 (B_6 with BKL presentation, `#u`, `WordToSequence`, `InducedPermutation`, `CanonicalFactorRepresentation`, `CanonicalLength`, `SuperSummitCanonicalLength` for both presentations).*

### 73.4.2 Computing Normal Forms of Elements

All normal form functions accept a `Presentation` parameter (`"Artin"` or `"BKL"`; default
is the current presentation of B). Functional versions return a new element; procedural
versions (with `~u`) modify u in place.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LeftNormalForm(u: -)` / `NormalForm(u: -)` | Returns a new element equal to u in left normal form with respect to the indicated presentation. | Left normal form algorithm **[ECH+92]** (Artin, O(k²n log n)) or **[BKL98]** (BKL, O(k²n)). |
| `LeftNormalForm(~u: -)` / `NormalForm(~u: -)` | Brings u into left normal form in place. | As above. |
| `RightNormalForm(u: -)` | Returns a new element in right normal form. | Analogous to left normal form **[ECH+92, BKL98]**. |
| `RightNormalForm(~u: -)` | Brings u into right normal form in place. | As above. |
| `LeftMixedCanonicalForm(u: -)` / `MixedCanonicalForm(u: -)` | Returns two tuples T1, T2 such that T1 and T2 define positive products v₁···v_k and w₁···w_l in left normal form with left-gcd(v₁, w₁) = 1 and u = (v₁···v_k)⁻¹(w₁···w_l). Tuples are in the format of `CFP`. | Left-mixed canonical form **[ECH+92]**. |
| `RightMixedCanonicalForm(u: -)` | Returns two tuples T1, T2 such that T1, T2 define positive products v₁···v_k, w₁···w_l in right normal form with right-gcd of trailing factors trivial, and u = (v₁···v_k)(w₁···w_l)⁻¹. | Right-mixed canonical form **[ECH+92]**. |

*Worked example: H73E3 (B_6, `LeftNormalForm` in Artin and BKL presentations, `CFP`, `RightNormalForm`, reading infimum/canonical length/supremum from the CFP tuple).*

*Worked example: H73E4 (performance of normalisation in iteration: comparing no normalisation vs. every step vs. every third step — illustrates that moderate normalisation frequency gives optimal results).*

### 73.4.3 Arithmetic Operators and Functions for Elements

All operations are performed with respect to the current presentation of B (or CFP form
by default). Complexity is linear in the length of input representations; no automatic
normalisation is done. It is recommended to call `NormalForm` periodically in loops.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `u * v` / `u *:= v` | Product uv; in-place version replaces u. | Concatenation of CFP or word representations. |
| `u / v` / `u /:= v` | Product uv⁻¹; in-place version replaces u. | — |
| `u ^ n` / `u ^:= n` | Power uⁿ (n an integer); in-place version replaces u. | Repeated multiplication. |
| `u ^ v` / `u ^:= v` | Conjugate uᵛ = v⁻¹uv; in-place version replaces u. | — |
| `Inverse(u)` / `Inverse(~u)` | Returns u⁻¹ as a new element; procedural version replaces u. | — |
| `LeftConjugate(u, v)` / `LeftConjugate(~u, v)` | Returns the left conjugate vuv⁻¹; procedural version replaces u. | — |
| `LeftDiv(u, v)` / `LeftDiv(u, ~v)` | Returns u⁻¹v; procedural version replaces v with u⁻¹v. | — |
| `Cycle(u: -)` / `Cycle(~u: -)` | For u with left normal form Δˡc₁···c_k: returns (procedural: sets u to) the cycling conjugate u·(c₁Δ⁻ˡ), in left normal form. Parameter `Presentation`. | Cycling operation **[ECH+92, BKL98]**. |
| `Decycle(u: -)` / `Decycle(~u: -)` | For u with left normal form Δˡc₁···c_k: returns (procedural: sets u to) the decycling conjugate u·(c_k⁻¹), in left normal form. Parameter `Presentation`. | Decycling operation **[ECH+92, BKL98]**. |

### 73.4.4 Boolean Predicates for Elements

Unless stated otherwise, computations use the presentation of B or the value of the
parameter `Presentation` (`"Artin"` or `"BKL"`).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `u in B` / `u notin B` | Membership test: returns true if u ∈ B. | — |
| `IsEmptyWord(u: -)` | Returns true if u is represented by the empty word in the indicated presentation. | — |
| `AreIdentical(u, v: -)` | Returns true if u and v are represented by identical words in the indicated presentation (syntactic identity, not group equality). | — |
| `IsSimple(u: -)` | Returns true if u is a simple element in the indicated presentation. Converts u to normal form. | Normal form then check c ⪯ Δ. |
| `IsSuperSummitRepresentative(u: -)` | Returns true if u is in its own super summit set. Converts u to normal form. | Compares (Infimum, Supremum) with (SuperSummitInfimum, SuperSummitSupremum). |
| `IsUltraSummitRepresentative(u: -)` | Returns true if u is in its own ultra summit set. Converts u to normal form. | Ultra summit set algorithm **[Geb03]**. |
| `IsIdentity(u: -)` / `IsId(u: -)` | Returns true if u is the identity. Converts to normal form. | — |
| `u eq v` / `u ne v` | Equality / inequality of elements. Both arguments converted to normal form. | Normal form comparison. |
| `u le v` / `IsLE(u, v: -)` / `IsLe(u, v: -)` | Returns true if u ⪯ v, i.e. u⁻¹v is positive (infimum ≥ 0 in left normal form). `Presentation` parameter not available for operator version. | Left normal form of u⁻¹v. |
| `u ge v` / `IsGE(u, v: -)` / `IsGe(u, v: -)` | Returns true if u ⪰ v, i.e. uv⁻¹ is positive. `Presentation` parameter not available for operator version. | Left normal form of uv⁻¹. |
| `IsConjugate(u, v: -)` | Returns true and a conjugating element c with u^c = v if u and v are conjugate; false otherwise. Uses ultra summit sets: computes USS representatives of u and v, then searches the USS of u for the USS representative of v. Parameter `Presentation`. | Ultra summit set algorithm **[Geb03]**; see Example H73E8 for the step-by-step procedure. |

*Worked example: H73E5 (super summit and ultra summit membership, partial orderings differ between Artin and BKL, `IsConjugate` with conjugating element, `InducedPermutation` as a quick invariant).*

### 73.4.5 Lattice Operations

All lattice functions accept a `Presentation` parameter. Results depend on the presentation
and partial ordering. Mixed canonical forms are used internally (see §73.1.4).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LeftGCD(u, v: -)` / `LeftGcd(u, v: -)` / `LeftGreatestCommonDivisor(u, v: -)` / `GCD(u, v: -)` / `Gcd(u, v: -)` / `GreatestCommonDivisor(u, v: -)` | Left-gcd of u and v: the ⪯-maximal d with d ⪯ u and d ⪯ v. | Left-mixed canonical form of u⁻¹v **[ECH+92]**. |
| `RightGCD(u, v: -)` / `RightGcd(u, v: -)` / `RightGreatestCommonDivisor(u, v: -)` | Right-gcd of u and v: the ⪰-maximal d with u ⪰ d and v ⪰ d. | Right-mixed canonical form of uv⁻¹ **[ECH+92]**. |
| `LeftGCD(S: -)` / `LeftGcd(S: -)` / `LeftGreatestCommonDivisor(S: -)` / `GCD(S: -)` / `Gcd(S: -)` / `GreatestCommonDivisor(S: -)` | Left-gcd of all elements in set or sequence S. | Iterated pairwise left-gcd. |
| `RightGCD(S: -)` / `RightGcd(S: -)` / `RightGreatestCommonDivisor(S: -)` | Right-gcd of all elements in S. | Iterated pairwise right-gcd. |
| `LeftLCM(u, v: -)` / `LeftLcm(u, v: -)` / `LeftLeastCommonMultiple(u, v: -)` / `LCM(u, v: -)` / `Lcm(u, v: -)` / `LeastCommonMultiple(u, v: -)` | Left-lcm of u and v: the ⪯-minimal d with u ⪯ d and v ⪯ d. | Right-mixed canonical form of u⁻¹v **[ECH+92]**. |
| `RightLCM(u, v: -)` / `RightLcm(u, v: -)` / `RightLeastCommonMultiple(u, v: -)` | Right-lcm of u and v: the ⪰-minimal d with d ⪰ u and d ⪰ v. | Left-mixed canonical form of uv⁻¹ **[ECH+92]**. |
| `LeftLCM(S: -)` / `LeftLcm(S: -)` / `LeftLeastCommonMultiple(S: -)` / `LCM(S: -)` / `Lcm(S: -)` / `LeastCommonMultiple(S: -)` | Left-lcm of all elements in S. | Iterated pairwise left-lcm. |
| `RightLCM(S: -)` / `RightLcm(S: -)` / `RightLeastCommonMultiple(S: -)` | Right-lcm of all elements in S. | Iterated pairwise right-lcm. |

*Worked example: H73E6 (Δ as left- and right-lcm of generators in both presentations; verifying left normal form condition via `LeftGCD`; left-lcm ≠ right-lcm in general).*

### 73.4.6 Invariants of Conjugacy Classes

Functions for computing the positive conjugate set P_x, the super summit set S_x, and
the ultra summit set U_x (see §73.1.5). All depend on the presentation; parameter
`Presentation` applies throughout. In practice, super summit sets can become infeasibly
large for braids on more than ~5–10 strings with moderate canonical length; ultra summit
sets remain tractable up to ~100 strings and canonical length ~1000 **[Geb03]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PositiveConjugates(u: -)` | Returns an indexed set of all conjugates of u that can be represented as positive words in the indicated presentation. | Closure under cycling/decycling from a positive representative **[ECH+92, BKL98]**; convexity result **[ERM94, FGM03]**. |
| `SuperSummitRepresentative(u: -)` | Returns a super summit representative u_s of u and an element c with u^c = u_s. | Iterated cycling/decycling to increase infimum **[Gar69, ERM94]**. |
| `SuperSummitSet(u: -)` | Returns the full super summit set S_u as an indexed set. | Closure under minimal simple element conjugation **[ERM94, FGM03]**. |
| `UltraSummitRepresentative(u: -)` | Returns an ultra summit representative u_s of u and an element c with u^c = u_s. u_s lies in S_u and is a positive conjugate if infimum ≥ 0. | Iterated cycling from a super summit representative **[Geb03]**. |
| `UltraSummitSet(u: -)` | Returns the full ultra summit set U_u as an indexed set. | Closure under minimal simple element conjugation for ultra summit **[Geb03]**. |

*Worked example: H73E7 (B_4, positive conjugates and super summit sets in Artin vs BKL (sizes 10, 36 and 2, 12); comparing super summit sets to prove non-conjugacy; B_8 example illustrating super summit set size 10972 vs ultra summit set size 36).*

#### 73.4.6.1 Computing Class Invariants Interactively

Process versions allow computing conjugacy-class invariants one element at a time,
suitable for large sets where only partial computation suffices (e.g. conjugacy search).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PositiveConjugatesProcess(u: -)` | Returns a process for constructing positive conjugates of u. Initially contains the first positive conjugate if one exists; empty otherwise. | Closure algorithm **[ERM94, FGM03]**, element by element. |
| `SuperSummitProcess(u: -)` | Returns a process for constructing super summit elements of u. Initially contains the first super summit element. | Closure algorithm **[ERM94, FGM03]**, element by element. |
| `UltraSummitProcess(u: -)` | Returns a process for constructing ultra summit elements of u. Initially contains the first ultra summit element. | Closure algorithm **[Geb03]**, element by element. |
| `BaseElement(P)` | Returns the element used to construct the process P. | — |
| `#P` | Returns the number of elements found so far by process P. | — |
| `Representative(P)` / `Rep(P)` | Returns the most recently found element of P. Runtime error if P is empty. | — |
| `IsEmpty(P)` | Returns true if P is exhausted. | — |
| `Elements(P)` | Returns an indexed set of all elements found so far by P. | — |
| `u in P` | Returns true and a conjugating element c (with base^c = u) if u has been found by P; false otherwise. | — |
| `u notin P` | Returns false if u has been found by P; true otherwise. | — |
| `NextElement(~P)` | Advances P to find the next element. If found, accessible via `Representative`. If no more elements exist, P becomes empty. | One step of the closure algorithm. |
| `Complete(~P)` | Runs the search to completion; P becomes empty; all elements accessible via `Elements`. | Completes the closure algorithm. |

*Worked example: H73E8 (user-defined `MyIsConjugate` using `UltraSummitProcess` and `in` operator, illustrating the algorithm used internally by `IsConjugate`; conjugacy attack on braid-group key exchange).*

#### 73.4.6.2 Computing Minimal Simple Elements

Functions for the minimal simple elements ι_x(s) (the ⪯-minimal extension of a simple
element s that conjugates x into the class invariant) and for transport/pullback maps
defined in **[Geb03]**. All accept parameters `Presentation` and `CheckArguments`
(default `true`; set to `false` to skip validity checks for performance).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `MinimalElementConjugatingToPositive(x, s: -)` | For a positive element x and simple element s: returns the minimal simple element r_x(s) with s ⪯ r_x(s) and x^{r_x(s)} ∈ B⁺. | Minimal simple element algorithm **[FGM03]**. |
| `MinimalElementConjugatingToSuperSummit(x, s: -)` | For x ∈ S_x and simple element s: returns the minimal simple element ρ_x(s) with s ⪯ ρ_x(s) and x^{ρ_x(s)} ∈ S_x. | Minimal simple element algorithm **[FGM03]**. |
| `MinimalElementConjugatingToUltraSummit(x, s: -)` | For x ∈ U_x and simple element s: returns the minimal simple element c_x(s) with s ⪯ c_x(s) and x^{c_x(s)} ∈ U_x. | Minimal simple element algorithm **[Geb03]**. |
| `Transport(x, s: -)` | For x, x^s both super summit elements and s a simple element: returns the transport φ_x(s) = (Δ ∧_l xΔ^{−inf(x)})⁻¹ · s · (Δ ∧_l x^s Δ^{−inf(x)}), satisfying c(x^s) = c(x)^{φ_x(s)}. φ_x(s) is a simple element. | Transport formula **[Geb03]**. |
| `Pullback(x, s: -)` | For x ∈ S_x and simple element s: returns the pullback π_x(s), the ⪯-minimal element satisfying x^{π_x(s)} ∈ S_x and s ⪯ φ_x(π_x(s)). | Pullback formula **[Geb03]**. |

*Worked example: H73E9 (user-defined `MyUltraSummitSet` using `MinimalElementConjugatingToUltraSummit`, reproducing the algorithm of `UltraSummitSet`).*

---

## 73.5 Homomorphisms

### 73.5.1 General Remarks

Homomorphisms from a braid group B to a group G are supported generally. A special
optimised case handles embeddings of B_n into B_m induced by σᵢ ↦ σ'_{k+εi} (ε = ±1,
k constant); these are detected automatically and use faster evaluation. Preimage
computation (`@@`) is currently only supported for this special embedding case.

### 73.5.2 Constructing Homomorphisms

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `hom< B -> G \| S : parameters >` | Returns the homomorphism from B to G defined by assignment S. S may be: (i) a list/sequence/indexed set of images of the k Artin generators B.1,…,B.k in order; or (ii) a list of tuples ⟨xᵢ, yᵢ⟩ or arrow pairs xᵢ → yᵢ for all Artin generators. Parameter `Check` (default `true`): verifies that the images satisfy the braid relations. Checking may be disabled for FPGroup codomains where it is unsupported. | Verification by substitution into defining relations. |

### 73.5.3 Accessing Homomorphisms

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `e @ f` / `f(e)` | Image of element e of B under homomorphism f. | Evaluate on generators then multiply in G. Optimised for the embedding case. |
| `B @ f` / `f(B)` | Image of B under f as a subgroup of the codomain. Not supported for all codomain categories. | — |
| `u @@ f` | Preimage of element u of Image(f) under f. Only supported for the special embedding σᵢ ↦ σ'_{k+εi}; may fail even when u ∈ Image(f). | — |
| `Domain(f)` | The domain of f. | — |
| `Codomain(f)` | The codomain of f. | — |
| `Image(f)` | Image of f as a subgroup of the codomain. Not supported for all codomain categories. | — |

*Worked example: H73E10 (epimorphism B_{10} → S_{10}; BKL key exchange protocol from **[KLC+00]** with l=6, r=7; conjugacy search attack using `MyIsConjugate`/`IsConjugate` on 100-string braid group; super summit set memory exhaustion).*

### 73.5.4 Representations of Braid Groups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SymmetricRepresentation(B)` | Returns the natural epimorphism from B onto the symmetric group Sym(n), where σᵢ ↦ transposition (i, i+1). | Artin's original permutation representation **[Art47]**. |
| `BurauRepresentation(B)` | Returns the Burau representation of B as a homomorphism to GL(n, Q(t)), the matrix algebra of degree n over the rational function field over Z. | Burau (1935) matrix representation. |
| `BurauRepresentation(B, p)` | Returns the p-modular Burau representation of B as a homomorphism to GL(n, F_p(t)). | Reduction of Burau mod p. |

*Worked example: H73E11 (Burau representation of B_4, explicit matrices for σ₁, σ₂, σ₃).*

---

## 73.6 Bibliography

| Key | Reference |
|-----|-----------|
| **[AAG99]** | Iris Anshel, Michael Anshel, and Dorian Goldfeld. *An algebraic method for public-key cryptography.* Math. Res. Lett., 6(3-4):287–291, 1999. |
| **[Art47]** | E. Artin. *Theory of braids.* Ann. of Math. (2), 48:101–126, 1947. |
| **[BKL98]** | Joan Birman, Ki Hyoung Ko, and Sang Jin Lee. *A new approach to the word and conjugacy problems in the braid groups.* Adv. Math., 139(2):322–353, 1998. |
| **[Deh02]** | Patrick Dehornoy. *Groupes de Garside.* Ann. Sci. École Norm. Sup. (4), 35(2):267–306, 2002. |
| **[ECH+92]** | David B. A. Epstein, James W. Cannon, Derek F. Holt, Silvio V. F. Levy, Michael S. Paterson, and William P. Thurston. *Word processing in groups.* Jones and Bartlett Publishers, Boston, MA, 1992. |
| **[ERM94]** | Elsayed A. El-Rifai and H. R. Morton. *Algorithms for positive braids.* Quart. J. Math. Oxford Ser. (2), 45(180):479–497, 1994. |
| **[FGM03]** | Nuno Franco and Juan González-Meneses. *Conjugacy problem for braid groups and Garside groups.* J. Algebra, 266(1):112–132, 2003. |
| **[Gar69]** | F. A. Garside. *The braid group and other groups.* Quart. J. Math. Oxford Ser. (2), 20:235–254, 1969. |
| **[Geb03]** | Volker Gebhardt. *A new approach to the conjugacy problem in Garside groups.* Preprint; arXiv:math.GT/0306199, 2003. |
| **[GKT+02]** | D. Garber, S. Kaplan, M. Teicher, B. Tsaban, and U. Vishne. *Length-based conjugacy search in the braid group.* Preprint; arXiv:math.GR/0209267, 2002. |
| **[GM02]** | Juan González-Meneses. *Improving an algorithm to solve Multiple Simultaneous Conjugacy Problems in braid groups.* Preprint; arXiv:math.GT/0212150, 2002. |
| **[HS03]** | D. Hofheinz and R. Steinwandt. *A practical attack on some braid group based cryptographic primitives.* In Public Key Cryptography, PKC 2003, LNCS 2567, pp. 187–198. Springer, 2003. |
| **[Hug02]** | J. Hughes. *A linear algebraic attack on the AAFG1 braid group cryptosystem.* In ACISP 2002, LNCS 2384, pp. 176–189. Springer, 2002. |
| **[KLC+00]** | Ki Hyoung Ko, Sang Jin Lee, Jung Hee Cheon, Jae Woo Han, Ju-sung Kang, and Choonsik Park. *New public-key cryptosystem using braid groups.* In Advances in Cryptology — CRYPTO 2000, pp. 166–183. Springer, Berlin, 2000. |
| **[LL02]** | Sang Jin Lee and Eonkyung Lee. *Potential Weakness of the Commutator Key Agreement Protocol Based on Braid Groups.* In Advances in Cryptology — EuroCrypt 2002, LNCS 2332, pp. 14–28. Springer, 2002. |
| **[LP03]** | E. Lee and J. H. Park. *Cryptanalysis of the public key encryption based on braid groups.* In Advances in Cryptology — EuroCrypt 2003, LNCS 2656, pp. 477–490. Springer, 2003. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Artin presentation and braid relations **[Art47]** | `BraidGroup`, `SymmetricRepresentation` |
| BKL presentation **[BKL98]** | `BraidGroup(:Presentation="BKL")`, all functions with `Presentation` parameter |
| Left/right normal form (Artin, O(k²n log n)) **[ECH+92]** | `LeftNormalForm`, `NormalForm`, `RightNormalForm`, `CanonicalLength`, `Infimum`, `Supremum` |
| Left/right normal form (BKL, O(k²n)) **[BKL98]** | As above with BKL presentation |
| Left/right mixed canonical form and lattice operations **[ECH+92]** | `LeftMixedCanonicalForm`, `RightMixedCanonicalForm`, `LeftGCD`, `RightGCD`, `LeftLCM`, `RightLCM` (and all aliases) |
| Cycling and decycling operations **[ECH+92, BKL98]** | `Cycle`, `Decycle`, `SuperSummitRepresentative`, `UltraSummitRepresentative` |
| Super summit set **[Gar69, ERM94]** | `SuperSummitRepresentative`, `SuperSummitSet`, `SuperSummitProcess`, `SuperSummitCanonicalLength`, `SuperSummitInfimum`, `SuperSummitSupremum`, `IsSuperSummitRepresentative` |
| Minimal simple elements for positive / super summit **[FGM03]** | `MinimalElementConjugatingToPositive`, `MinimalElementConjugatingToSuperSummit`, `PositiveConjugates`, `PositiveConjugatesProcess` |
| Ultra summit set **[Geb03]** | `UltraSummitRepresentative`, `UltraSummitSet`, `UltraSummitProcess`, `IsUltraSummitRepresentative`, `MinimalElementConjugatingToUltraSummit`, `Transport`, `Pullback` |
| Conjugacy testing and search via ultra summit sets **[Geb03]** | `IsConjugate` |
| Braid-group key exchange cryptosystem **[KLC+00]** | `hom<>`, `NormalForm`, `IsConjugate` (see H73E10) |
| Burau matrix representation | `BurauRepresentation` |
