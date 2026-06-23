# Chapter 70 — Finitely Presented Groups

**Handbook part:** X — Finitely-Presented Groups
**Handbook pages:** 2081–2201 (PDF pages 2210–2335)

---

## Scope and overview

Chapter 70 presents the facilities for computing with **finitely-presented groups** (fp-groups,
Magma category `GrpFP`). Every fp-group is realised as a quotient of a free group via a set of
defining relations. The chapter covers what is sometimes called *combinatorial group theory*.

Key facilities include:

- Construction of fp-groups by generators and relations (free groups, quotient constructors,
  the `Group< >` constructor, conversion from permutation/matrix/polycyclic groups);
- Quotient algorithms: **abelian quotient** (Smith normal form), **p-quotient** (ANU p-Quotient
  program, **[NO96]**), **nilpotent quotient** (ANU NQ, **[Nic96]**), and **soluble quotient**;
- Index determination and subgroup construction via the **Todd-Coxeter coset enumeration
  algorithm** (ACE3, **[CDHW73], [Hav91]**);
- Coset tables, coset spaces, and the induced permutation representation;
- All subgroups of bounded index via the **low-index subgroups algorithm** (**[Sim94, §5.6]**);
- Subgroup presentations via **Reidemeister-Schreier** rewriting (**[MKS76], [AR84], [HKRR84]**);
- Presentation simplification via **Tietze transformations** (**[HKRR84]**);
- Homomorphisms to finite groups (backtrack search), `PSL(2,q)` quotients via the
  **L2-quotient algorithm** (Plesken-Fabianska, **[PF09]**), and isomorphism search;
- Representation theory: `𝔽_p[G]`-modules from conjugation on elementary abelian sections;
- Small-group identification (using the database of small groups, Chapter 66).

For a general reference on algorithms for finitely presented groups, the chapter refers to
**[Sim94]**.

---

## 70.1 Introduction

### 70.1.1 Overview of Facilities

The facilities are grouped as: construction of fp-groups; abelian/p-/nilpotent/soluble
quotient computation; Todd-Coxeter index determination and subgroup building; operations on
finite-index subgroups via coset tables; low-index subgroup enumeration; coset-space
permutation representations; Reidemeister-Schreier rewriting; presentation simplification.

*No intrinsics in this subsection.*

### 70.1.2 The Construction of Finitely Presented Groups

Every group is a quotient of a free group. Two constructors are primary: `FreeGroup(n)` and
`quo< F | R >`. Generators are referenced as `G.i`; the special assignment `G<v1,...,vr> :=
construction` binds names to generators.

*No additional intrinsics beyond the constructors described in §70.2–§70.3.*

---

## 70.2 Free Groups and Words

### 70.2.1 Construction of a Free Group

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `FreeGroup(n)` | Construct the free group `F` of rank `n` (positive integer). The `i`-th generator is `F.i`. Supports the named-generator assignment `F<v1,...,vn> := FreeGroup(n)`. | Free group construction; no algorithm needed. |

*Worked examples: H70E1 (creation of the rank-2 free group with named and unnamed generators).*

### 70.2.2 Construction of Words

The operations apply to both free groups and arbitrary fp-groups.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `G ! [i1, ..., is]` | Given a group `G` on `r` generators and a sequence of integers in `[−r, r] \ {0}`, construct the word `G.\|i1\|^ε1 * ... * G.\|is\|^εs` (εj = +1 if ij > 0, −1 otherwise). | Word formation by concatenation. |
| `Identity(G)` / `Id(G)` / `G ! 1` | The identity element (empty word) of the fp-group `G`. | — |
| `Random(G, m, n)` | A random word of length `l` in the generators of `G`, where `m ≤ l ≤ n`. | Uniform random generator letters. |

### 70.2.3 Access Functions for Words

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `#w` | The length (number of letters including inverses) of the word `w`. | — |
| `ElementToSequence(w)` / `Eltseq(w)` | The integer sequence `[i1,...,im]` decomposing `w` into constituent generators/inverses (positive index = generator, negative = inverse). | — |
| `ExponentSum(w, x)` / `Weight(w, x)` | The sum of exponents of generator `x` in word `w`. | — |
| `GeneratorNumber(w)` | Returns `0` if `w = Id`, `i` if `w = G.i * w'`, `−i` if `w = G.i^{−1} * w'`. | — |
| `LeadingGenerator(w)` | If `w = x^ε * w'`, returns `x^ε`; if `w = Id`, returns `Id`. | — |
| `Parent(w)` | The parent group `G` of word `w`. | — |

*Worked examples: H70E2 (abelianised relation matrix via `Weight`), H70E3 (`Random`, `LeadingGenerator`, stripping letters one by one).*

### 70.2.4 Arithmetic Operators for Words

Only free reduction is applied when operators are used in a group with non-trivial relations.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `u * v` | Product of words `u` and `v` in the same fp-group. | Free reduction. |
| `u ^ n` | Power of word `u` (integer `n`); `n = −1` gives inverse; `n = 0` gives identity. | Free reduction. |
| `u ^ v` | Conjugate: `v^{−1} * u * v`. | Free reduction. |
| `(u, v)` | Commutator: `u^{−1} * v^{−1} * u * v`. | Free reduction. |
| `(u1, ..., un)` | Left-normed commutator: `((u1, u2), u3, ...)`. | Free reduction, left-normed. |

### 70.2.5 Comparison of Words

Comparison is purely on free reductions (not modulo relations). Words are ordered first by length, then lexicographically by `G.1 < G.1^{−1} < G.2 < G.2^{−1} < ...`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `u eq v` | True if free reductions of `u` and `v` are identical. | Free reduction + string equality. |
| `u ne v` | True if free reductions differ. | — |
| `u lt v` | True if `u` precedes `v` in the length-then-lex order. | — |
| `u le v` | True if `u` precedes or equals `v`. | — |
| `u ge v` | True if `u` follows or equals `v`. | — |
| `u gt v` | True if `u` follows `v`. | — |

### 70.2.6 Relations

A relation is an equality between two words. A relation type is provided for working with relations as objects.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `w1 = w2` | Create the relation `w1 = w2` over the generators of an fp-group. The relation is not automatically added to the group's defining set. | — |
| `r[1]` / `LHS(r)` | Left-hand side of relation `r` as a word. | — |
| `r[2]` / `RHS(r)` | Right-hand side of relation `r` as a word. | — |
| `r[1] := w` | Redefine the left-hand side of relation `r` to be word `w`. | — |
| `r[2] := w` | Redefine the right-hand side of relation `r` to be word `w`. | — |
| `f(r)` | Image of relation `r` under homomorphism `f`. | — |
| `Parent(r)` | The group over which relation `r` is taken. | — |

*Worked examples: H70E4 (defining a set of relations and replacing one side).*

---

## 70.3 Construction of an FP-Group

### 70.3.1 The Quotient Group Constructor

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `quo< F \| R >` | Given fp-group `F` and relation set `R` (words = relators, relations `w1=w2`, relation lists `w1=w2=...=wr`, or subgroups), construct the quotient `G = F / ncl(R)`. Returns `(G, φ: F → G)`. May implicitly invoke coset enumeration; controllable via `SetGlobalTCParameters`. Terms of `R` may use `$` for `F` and `1` for identity. | Quotient by normal closure; may use Todd-Coxeter coset enumeration **[CDHW73]**. |
| `G / H` | Quotient of `G` by the normal closure of subgroup `H`. Adds Schreier generators of `H` as relators. | Normal closure quotient. |

*Worked examples: H70E5 (S4 as quo with relators, relations list, and set), H70E6 (`$` symbol), H70E7 (quotient of a non-free group).*

### 70.3.2 The FP-Group Constructor

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Group< X \| R >` | Construct an fp-group directly: given generator names `X = x1,...,xr` and relations `R`, internally constructs `FreeGroup(r)` then takes the quotient. Same relation syntax as `quo<>`. Returns `(G, φ: F → G)`. | Combines `FreeGroup` + `quo<>`. |

*Worked examples: H70E8 (binary tetrahedral group), H70E9 (3-generator group), H70E10 (parametrised Coxeter family (l,m|n,k)).*

### 70.3.3 Construction from a Finite Permutation or Matrix Group

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `FPGroup(G)` | Given finite `G` in `GrpPerm` or `GrpMat`, return an fp-group `F` isomorphic to `G` and the isomorphism `φ: F → G`. Generators of `F` correspond to generators of `G`. Practical only for groups of order at most a few million. | Computes defining relations for the given generating set. |
| `FPGroupStrong(G)` | As `FPGroup`, but generators of `F` correspond to a **strong generating set** of `G` (computed if not known). More practical for large permutation groups; see Chapters 58/59. | Strong generating set method; detailed description in Ch. 58. |
| `FPGroupStrong(G, N)` | Given permutation group `G` and normal subgroup `N`, return an fp-group isomorphic to `G/N` and a homomorphism `φ: G → F`. See Chapter 58. | — |

*Worked examples: H70E11 (A5: `FPGroup` giving 2 generators, `FPGroupStrong` giving 3 strong generators).*

### 70.3.4 Construction of the Standard Presentation for a Coxeter Group

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CoxeterGroup(GrpFP, W)` | Given a finite Coxeter group `W` (in `GrpFPCox` or `GrpPermCox`), return an fp-group `F` isomorphic to `W` given by the standard Coxeter presentation, plus the isomorphism `W → F`. Parameter `Local` (default `false`): if `true`, `F` is the appropriate subgroup of the fp-version of the overgroup of `W`. | Standard Coxeter presentation; see Chapter 98. |

*Worked examples: H70E12 (C5 Coxeter group, mapping words between `W` and `F`).*

### 70.3.5 Conversion from a Special Form of FP-Group

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `FPGroup(G)` | Given `G` in `GrpPC`, `GrpGPC`, or `GrpAb`, return an fp-group `H` isomorphic to `G` and the isomorphism `φ: G → H`. Converts a special presentation into a general one. | Direct relabelling of PC/abelian relations. |

*Worked examples: H70E13 (infinite dihedral group from `GrpGPC` to `GrpFP`).*

### 70.3.6 Construction of a Standard Group

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AbelianGroup(GrpFP, [n1,...,nr])` | Abelian group `C_n1 × ... × C_nr` as an fp-group; `C_0` = infinite cyclic. | Standard abelian presentation. |
| `AlternatingGroup(GrpFP, n)` / `Alt(GrpFP, n)` | Alternating group of degree `n` as an fp-group. Generators: `(3,4,...,n)` and `(1,2,3)` for `n` odd; `(1,2)(3,4,...,n)` and `(1,2,3)` for `n` even. | Standard alternating group presentation. |
| `BraidGroup(GrpFP, n)` | Braid group on `n` strings (`n−1` Artin generators) as an fp-group. | Artin presentation. |
| `CoxeterGroup(GrpFP, t)` | Coxeter group of Cartan type `t` (string, e.g. `"F4"`) as an fp-group; standard Coxeter presentation. See Chapter 97. | Standard Coxeter presentation. |
| `CyclicGroup(GrpFP, n)` | Cyclic group of order `n` as an fp-group. | — |
| `DihedralGroup(GrpFP, n)` | Dihedral group of order `2n` as an fp-group (`n > 2`); for `n = 0`, the infinite dihedral group. | — |
| `ExtraSpecialGroup(GrpFP, p, n : parameters)` | Extra-special group of order `p^{2n+1}` as an fp-group. Parameter `Type` (`"+"` default or `"−"`): `"+"` gives exponent-`p` for `p` odd, central product of `n` D8's for `p=2`; `"−"` gives exponent-`p^2` for `p` odd, Q8 × (n−1) D8's for `p=2`. | — |
| `SymmetricGroup(GrpFP, n)` / `Sym(GrpFP, n)` | Symmetric group of degree `n` as an fp-group. Generators: `(1,2,...,n)` and `(1,2)`. | Standard symmetric group presentation. |

*Worked examples: H70E14 (S8 as fp-group, Coxeter group F4 as fp-group).*

### 70.3.7 Construction of Extensions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Darstellungsgruppe(G)` | Given fp-group `G`, construct a **maximal central extension** `G̃` of `G` as an fp-group. | Schur multiplier / representation group construction. |
| `DirectProduct(G, H)` | Direct product of fp-groups `G` and `H`. | Adds commutativity relations between disjoint generator sets. |
| `DirectProduct(Q)` | Direct product of the sequence `Q = [Q[1], ..., Q[r]]` of fp-groups. | — |
| `FreeProduct(G, H)` | Free product of fp-groups `G` and `H`. | Juxtaposition of presentations. |
| `FreeProduct(Q)` | Free product of the sequence `Q = [Q[1], ..., Q[r]]` of fp-groups. | — |

*Worked examples: H70E15 (`Darstellungsgruppe` of a group of order 36, producing a group of order 108), H70E16 (direct product A5 × Z2).*

### 70.3.8 Accessing the Defining Generators and Relations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `G . i` | The `i`-th defining generator of `G`; `i < 0` gives inverse; `G.0 = Identity(G)`. | — |
| `Generators(G)` | The set of generators of `G`. | — |
| `NumberOfGenerators(G)` / `Ngens(G)` | The number of generators of `G`. | — |
| `PresentationLength(G)` | Total length of all relators of `G`. | — |
| `Relations(G)` | Sequence of defining relations of `G`. | — |

---

## 70.4 Homomorphisms

### 70.4.1 General Remarks

The kernel of a homomorphism `f` with domain `GrpFP` can be computed when the codomain is
`GrpGPC`, `GrpPC`, `GrpAb`, `GrpPerm`, `GrpMat`, `ModAlg`, or `ModGrp` (Chapters 63, 69, 58,
59, 89) and the image is finite and of moderate order — via a regular permutation representation
of the image. Kernel computation may be very time/memory-intensive.

### 70.4.2 Construction of Homomorphisms

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `hom< P -> G \| S >` | Homomorphism from fp-group `P` to group `G`. `S` is either a list/sequence giving images of `P.1,...,P.n` in order, or a list of tuples/arrow-pairs `<xi, yi>` / `xi -> yi` (any order). Caller is responsible for well-definedness; use `IsSatisfied` to check. | — |
| `IsSatisfied(U, E)` | Given words or relations `U` over an `n`-generator fp-group `H`, and elements `E = [e1,...,en]` in some group `G`: checks whether all relations/relators in `U` are satisfied under `H.i → ei`. Returns `true`/`false`. | Substitution and equality check. |

### 70.4.3 Accessing Homomorphisms

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `w @ f` / `f(w)` | Image of element `w` of the fp-group domain under `f`. | Word evaluation. |
| `H @ f` / `f(H)` | Image of subgroup `H` under `f` (not supported by all maps). | — |
| `g @@ f` | Preimage of element `g` of the image under `f` (not supported by all maps). | — |
| `H @@ f` | Preimage of subgroup `H` of the codomain under `f`. Requires kernel to be computable (moderate index). | — |
| `Domain(f)` | Domain of homomorphism `f`. | — |
| `Codomain(f)` | Codomain of homomorphism `f`. | — |
| `Image(f)` | Image of `f` as a subgroup of the codomain (not supported by all maps). | — |
| `Kernel(f)` | Kernel of `f` as a subgroup of the domain (represented by coset table). Requires moderate index. | Todd-Coxeter coset enumeration **[CDHW73]**. |

*Worked examples: H70E17 (braid group B5 → S5 epimorphism, `IsSatisfied`, `Kernel`, `GeneratingWords`).*

### 70.4.4 Computing Homomorphisms to Finite Groups

Two overloaded versions exist: one for codomain a permutation group, one for a polycyclic group.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Homomorphisms(F, G, A : parameters)` / `Homomorphisms(F, G : parameters)` | Find representatives of classes of homomorphisms `F → G` modulo automorphisms induced by `A` (`G ◁ A`; default `A = G`). Parameters: `Surjective` (default `true`), `Limit` (default 0 = no limit), `TimeLimit` (default 0), `CosetEnumeration` (default `true`), `CacheCosetAction` (default `true`). Works for `G`, `A` permutation groups **or** finite polycyclic groups. | Backtrack search on A-class representatives in `G`; coset enumerations to prune candidate images. |
| `HomomorphismsProcess(F, G, A : parameters)` / `HomomorphismsProcess(F, G : parameters)` | Process version of `Homomorphisms` for permutation-group codomain: returns one representative at a time. Same parameters. Use `NextElement`, `Complete`, `IsEmpty`, `IsValid`, `DefinesHomomorphism`, `Homomorphism`, `#P`, `Homomorphisms(P)`. | As above, interactive. |
| `NextElement(∼P)` | Continue search until a new class is found; marks `P` empty if search completes, invalid if limit reached. | — |
| `Complete(∼P)` | Run search to completion (or until a limit is reached). | — |
| `IsEmpty(P)` | True if all classes have been found. | — |
| `IsValid(P)` | False if a limit set for `P` has been reached. | — |
| `DefinesHomomorphism(P)` | True if `P` currently defines a homomorphism extractable by `Homomorphism(P)`. | — |
| `Homomorphism(P)` | Extract the most recently found homomorphism from process `P`. | — |
| `#P` | Number of homomorphisms found so far by `P`. | — |
| `Homomorphisms(P)` | All homomorphisms found by `P` (complete only when `P` is empty). | — |

*Worked examples: H70E18 (`Homomorphisms` proving F maps onto A5), H70E19 (interactive process B4 → PSL(2,16) seeking maximal image), H70E20 (proving infiniteness via kernel with infinite abelian quotient).*

#### 70.4.4.2 Finding Homomorphisms onto Simple Groups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SimpleQuotients(F, deg1, deg2, ord1, ord2 : parameters)` / `SimpleQuotients(F, ord1, ord2 : parameters)` / `SimpleQuotients(F, ord2 : parameters)` | Find epimorphisms from `F` onto simple groups in a fixed list (all non-abelian simple groups of order ≤ 10^9, plus PSL(2,q) as an infinite family). Degree bounds `[deg1, deg2]` (default 5–10^7) and order bounds `[ord1, ord2]`. Parameters: `Family` (default `"All"`; also `"PSL"`, `"PSL2"`, `"Mathieu"`, `"Alt"`, `"PSp"`, `"PSU"`, `"Other"`, `"notPSL2"`), `Limit` (default 1), `HomLimit` (default 0). Returns list of sequences of epimorphisms grouped by target group. | Calls `Homomorphisms` against a fixed list of simple groups. |
| `SimpleQuotientProcess(F, deg1, deg2, ord1, ord2 : parameters)` | Process version: sets up the record, conducts initial search, then use `NextSimpleQuotient`, `IsEmptySimpleQuotientProcess`, `SimpleEpimorphisms`. Parameter `Family`. | — |
| `NextSimpleQuotient(∼P)` | Advance process `P` to the next simple quotient found. | — |
| `IsEmptySimpleQuotientProcess(P)` | True if process `P` has found all groups in the search range. | — |
| `SimpleEpimorphisms(P)` | Extract the most recently found epimorphisms and an info tuple `<degree, order, name>`. | — |

*Worked examples: H70E21 (perfect group with L(2,13) and L(3,3) as images; PSU process).*

### 70.4.5 The L2-Quotient Algorithm

Given a 2-generator fp-group `G`, the algorithm of Plesken and Fabianska **[PF09]** computes all
quotients isomorphic to some PSL(2,q), simultaneously for all prime powers `q`. It handles
infinitely many quotients and large prime powers. Does not currently return PSL(2,2), PSL(2,3),
PSL(2,4)=PSL(2,5).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `L2Quotients(G)` | Main method. Takes a 2-generator fp-group; returns a list of prime ideals of Z[x1, x2, x12] encoding all L2-quotient information. | L2-quotient algorithm via trace ideal / affine scheme **[PF09]**; constructs the scheme X of SL2 representations. |
| `L2Type(P)` | For prime ideal `P` in Z[x1,x2,x12]: returns a string describing the L2-type. Possible types (from `L2Quotients`/`L2Ideals`): `"PGL(2,q)"`, `"PSL(2,q)"`, `"infinite (characteristic zero)"`, `"infinite (characteristic p)"`. Other types (`"reducible"`, `"dihedral"`, `"Alt(4)"`, `"Sym(4)"`, `"Alt(5)"`) are eliminated internally. | — |
| `L2Generators(P)` | Given a maximal ideal `M` in Z[x1,x2,x12], compute two matrices with entries in Z[x1,x2,x12]/M corresponding to the generator images in SL(2, ...). | — |
| `L2Ideals(I)` | Given an ideal `I` in Z[x1,x2,x12] (e.g. `<p> + P` for a prime ideal `P`): compute minimal associated primes giving rise to L2-quotients. Used to analyse infinite families mod a specific prime. | Associated prime decomposition. |

*Worked examples: H70E22 (one-relator quotients of the modular group; PSL(2,25), PGL(2,13)), H70E23 (Coxeter presentations, many L2 images), H70E24 (infinite L2-quotient family, reducing mod p), H70E25 (infinitely many L2 quotients in characteristic 41 only).*

### 70.4.6 Infinite L2 Quotients

Uses the L2-quotient algorithm in characteristic zero to test for the existence of infinite
PSL2(K) quotients (K a characteristic-0 field). Constructs an affine scheme X (the "trace ideal"
scheme) whose points classify SL2 representations. If the complement `U = X \ Y` (where `Y`
encodes reducible/dihedral/A4/S4/A5 images) is non-empty, the group is infinite and has PSL2
images over infinitely many finite fields.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HasInfinitePSL2Quotient(G)` | For 2-generator fp-group `G`: returns `true` if `G` has an infinite quotient in PSL2(K) for K a characteristic-0 field. Parameters: `signs` (sequence of 0/1/−1 selecting sign combinations for defining relations in SL2 lifts; default 0 = unconstrained), `full` (if `true`, continue after first infinite image found and also return a sequence of (signs, types) pairs), `Verbose IsInfGrp` (level 0 or 1). | Computes affine scheme X, removes Y, analyses dimension-zero locus **[PF09, Fab09]**. |

*Worked examples: H70E26 (two quotients of the modular group; one infinite, one finite A5).*

### 70.4.7 Searching for Isomorphisms

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SearchForIsomorphism(F, G, m : parameters)` | Attempt to find an isomorphism `φ: F → G` (fp-groups) restricting to homomorphisms where the sum of word-lengths of images of generators of `F` in `G` is at most `m`. Returns `(true, φ, φ^{−1})` if found, else `(false, _, _)`. An error results if any generator of `F` is trivial. Parameters: `All` (default `false`: halt after first iso; `true`: return all isomorphisms found as list of `<φ, φ^{−1}>`), `IsomsOnly` (default `true`; if `false`, return all homomorphisms), `MaxRels` (default `250*m`, passed to `RWSGroup`), `CycConjTest` (default `true`: reject images that have an earlier cyclic conjugate, usually faster). Verbose flag: `"IsoSearch"`. | Backtrack search with word-length bound; uses rewriting system (`RWSGroup`) to test equality. |

*Worked examples: H70E27 (Hillman's isomorphism question, `SearchForIsomorphism` at length 7), H70E28 (Neumann's question; using `Simplify` + auxiliary generator to reduce length).*

---

## 70.5 Abelian, Nilpotent and Soluble Quotient

### 70.5.1 Abelian Quotient

Functions may invoke coset enumeration (controllable via `SetGlobalTCParameters`). Use
`HasComputableAbelianQuotient` / `HasInfiniteComputableAbelianQuotient` to avoid errors.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AbelianQuotient(G)` | Maximal abelian quotient `G/G'` as `GrpAb`. Returns `(G/G', π: G → G/G')`. | Smith normal form of the relation matrix over Z. |
| `ElementaryAbelianQuotient(G, p)` | Maximal p-elementary abelian quotient of `G` as `GrpAb`. Returns `(Q, π: G → Q)`. | Smith normal form mod p. |
| `AbelianQuotientInvariants(G)` / `AQInvariants(G)` | Elementary divisors of `G/G'` as a sequence of integers. The algorithm of Havas is used for the reduction over Z (with heuristics to minimize coefficient growth). | Havas's integer Smith normal form algorithm. |
| `AbelianQuotientInvariants(H)` / `AQInvariants(H)` | Elementary divisors of `H/H'` for subgroup `H` of fp-group `G`, using the Reidemeister-Schreier presentation for `H`. | Reidemeister-Schreier rewriting **[MKS76]** then Smith normal form. |
| `AbelianQuotientInvariants(G, T)` / `AQInvariants(G, T)` | As above but accepts coset table `T` defining `H`. | As above. |
| `AbelianQuotientInvariants(G, n)` / `AQInvariants(G, n)` | Divisors of `G/N` where `N = ⟨G', all n-th powers⟩`. Constructs relation matrix and computes Smith normal form over Z/nZ. Particularly efficient for small prime `n`. | Smith normal form over Z/nZ. |
| `AbelianQuotientInvariants(H, n)` / `AQInvariants(H, n)` / `AbelianQuotientInvariants(G, T, n)` / `AQInvariants(G, T, n)` | As `AQInvariants(G, n)` but for the subgroup `H` (or `H` defined by coset table `T`): divisors of `H/N` where `N = ⟨H', all n-th powers⟩`. Uses Reidemeister-Schreier then Smith normal form over Z/nZ. | Reidemeister-Schreier **[MKS76]** + Smith normal form over Z/nZ. |
| `HasComputableAbelianQuotient(G)` | Tests whether the abelian quotient of fp-group `G` can be computed. If yes, returns `(true, A, π: G → A)`; otherwise `false`. Avoids runtime errors in loops. | — |
| `HasInfiniteComputableAbelianQuotient(G)` | Tests whether the abelian quotient can be computed and is infinite. Heuristic: checks modular abelian invariants for small primes first. Returns `(true, A, π)` or `false`. | Modular invariant pre-check, then Smith normal form. |
| `IsPerfect(G)` | Tests whether fp-group `G` is perfect (abelian quotient trivial). | Checks `AbelianQuotient`. |
| `TorsionFreeRank(G)` | Torsion-free rank of `G/G'`. | Smith normal form, counts zero invariants. |

*Worked examples: H70E29 (Fibonacci group F(7): AQ invariants → cyclic of order 29), H70E30 ((8,7|2,3): `AQInvariants(H, 2)`).*

### 70.5.2 p-Quotient

A p-quotient algorithm constructs a consistent power-conjugate presentation (pcp) for the
largest p-quotient of `F` having lower exponent-p class at most `c`. The implementation is the
**ANU p-Quotient program** **[NO96]**. Results are returned in category `GrpPC` (Chapter 63).
A process version with finer control is described in Chapter 71.

### 70.5.3 The Construction of a p-Quotient

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `pQuotient(F, p, c : parameters)` | Given fp-group `F`, prime `p`, positive integer `c` (or `c=0` for class limit 127): construct a pcp for the largest p-quotient of `F` of lower exponent-p class ≤ c. Returns `(G: GrpPC, π: F → G, S: sequence of generator definitions, flag: whether G is the maximal p-quotient)`. Each element of `S` is `[0,r]` (generator defined via image of `F.r`), `[r,0]` (via power relation for `G.r`), or `[r,s]` (via conjugate relation `G.r^G.s`). Parameters: `Exponent` (enforce `x^m = 1`; default 0), `Metabelian` (default `false`: largest metabelian p-quotient), `Print` (0–3; default from `GetVerbose("pQuotient")`), `Workspace` (default 5000000). | ANU p-Quotient algorithm **[NO96]**: constructs the lower exponent-p central series level by level. |

*Worked examples: H70E31 (largest 2-quotient of class 6 for a 2-relator group: order 2^19), H70E32 (3-quotient class 6 with `Exponent := 9`), H70E33 (metabelian 5-quotient), H70E34 (largest 2-generator group of exponent 5: order 5^34).*

### 70.5.4 Nilpotent Quotient

A nilpotent quotient algorithm constructs a polycyclic presentation of a nilpotent quotient of an
fp-group. The implementation is the **ANU Nilpotent Quotient program** (version 2.2, January
2007), described in **[Nic96]**. Results are in category `GrpGPC`. Algorithm: computes factor
groups modulo successive terms of the lower central series, using "nilpotent presentations". Has
highly efficient code for enforcing n-Engel identities and arbitrary identical relations ("free
variables").

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `NilpotentQuotient(G, c : parameters)` | Class-`c` nilpotent quotient of `G` as `GrpGPC`, plus epimorphism `π: G → Q`. `c = 0` attempts the maximal nilpotent quotient. Parameters: `NumberOfEngelGenerators` (default 1), `LeftEngel` (default 0; enforce first `k` generators to be left `n`-Engel), `RightEngel` (default 0; right `n`-Engel), `Engel` (default 0; enforce the `n`-th Engel law globally), `SemigroupOnly` (default `true`), `SemigroupFirst` (default `false`), `ReverseOrder` (default `false`), `ReverseEngel` (default `false`), `CheckFewInstances` (default `false`), `Nickel` (default `false`; enforce `x^8` and `[[x1,x2,x3],[x4,x5,x6]]`), `NumberOfFreeVariables` (default 0; treat last `n` generators as "identical generators"), `PrintResult` (default `false`). | ANU NQ algorithm **[Nic96]**: inductively builds lower central series quotients using nilpotent polycyclic presentations; Engel-law enforcement with semigroup-word optimisation. |
| `SetVerbose("NilpotentQuotient", n)` | Set verbosity (0–3) for the nilpotent quotient algorithm. `n = 0` suppresses output; higher values print more detail. | — |

*Worked examples: H70E35 (group with infinite abelian quotient → class-2 nilpotent quotient), H70E36 (free nilpotent group of rank 2, class 3), H70E37 (Baumslag-Solitar groups BS(1,4) and BS(2,4), class-4 nilpotent quotients), H70E38 (maximal nilpotent quotient of class 6; metabelian quotient via free variables; 4th-Engel quotient).*

### 70.5.5 Soluble Quotient

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SolvableQuotient(G : parameters)` / `SolubleQuotient(G : parameters)` | Largest finite soluble quotient of fp-group `G`, as `GrpPC`. | Soluble quotient algorithm; see Chapter 71 for full parameter list. |
| `SolvableQuotient(F, n : parameters)` / `SolubleQuotient(F, n : parameters)` | Soluble quotient of `F` with a specified order `n` (0 = unknown; algorithm determines relevant primes). | As above. |
| `SolvableQuotient(F, P : parameters)` / `SolubleQuotient(F, P : parameters)` | Soluble quotient of `F` whose order has prime divisors only in the set `P` of primes. | As above. |

*Worked examples: H70E39 (soluble quotient of order 1920), H70E40 (group with soluble quotient of order 165888 = 2^11·3^4; using prime set `{2,3}` reduces time by 3×; confirmed by Todd-Coxeter).*

---

## 70.6 Subgroups

### 70.6.1 Specification of a Subgroup

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `sub< G \| L >` | Subgroup of fp-group `G` generated by words/sets/sequences/subgroups in list `L`. Empty list gives trivial subgroup. | — |
| `sub< G \| f >` | Subgroup of `G` afforded by homomorphism `f: G → Sym(n)` (the stabiliser of a point, via the corresponding permutation representation). | Coset-table stabiliser. |
| `ncl< G \| L >` | Normal closure in `G` of the subgroup generated by the words in list `L`. May be applied when `H = ⟨L⟩` has infinite index in `G`, provided the normal closure has finite index. Uses coset table of the trivial subgroup with additional relators. | Todd-Coxeter coset enumeration **[CDHW73]** (may invoke `SetGlobalTCParameters`). |
| `ncl< G \| f >` | Normal closure of the subgroup afforded by homomorphism `f` (stabiliser preimage). | As above. |
| `CommutatorSubgroup(G)` / `DerivedSubgroup(G)` / `DerivedGroup(G)` | Derived subgroup `G'` of fp-group `G` as a finite-index subgroup. Fails if no presentation is known or index is too large. May invoke coset enumeration (controllable via `SetGlobalTCParameters`). | Todd-Coxeter coset enumeration **[CDHW73]**. |

*Worked examples: H70E41 ((8,7|2,3): subgroup of index 448), H70E42 (subgroup via permutation homomorphism, `GeneratingWords`).*

### 70.6.2 Index of a Subgroup: The Todd-Coxeter Algorithm

The implementation is ACE3 (George Havas and Colin Ramsay, University of Queensland)
**[CDHW73, Hav91]** (manual and sources at **[Ram]**). Parameters `CosetLimit`, `Workspace`,
`Strategy` (`"Easy"` or `"Hard"`) are the most important; full parameter list is in
`CosetEnumerationProcess` (Chapter 71).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ToddCoxeter(G, H : parameters)` | Attempt to enumerate cosets of `H` in `G` by the Todd-Coxeter procedure. Returns `(index, coset-table map, max simultaneous cosets, total cosets defined)`. Returns index 0 if enumeration fails. Accepts same parameters as `CosetEnumerationProcess` (Chapter 71). | Todd-Coxeter coset enumeration **[CDHW73, Hav91]**, ACE3 implementation **[Ram]**. |
| `Index(G, H : parameters)` | Index of `H` in `G` via Todd-Coxeter; returns 0 if enumeration fails. | Todd-Coxeter **[CDHW73, Hav91]**, ACE3. |
| `FactoredIndex(G, H : parameters)` | Factored index of `H` in `G`; reports error on failure. | Todd-Coxeter **[CDHW73, Hav91]**, ACE3. |
| `Order(G : parameters)` | Attempt to determine the order of fp-group `G` (or prove infinity): returns the order (positive integer), `Infinity`, or 0 (indeterminate). Multi-step strategy: checks if `G` is free, checks deficiency, checks known subgroups/supergroups, coset enumerations, Reidemeister-Schreier rewriting. Parameters `UseRewrite` (default `true`), `MinIndex` (default 10), `MaxIndex` (default 1000) control when Reidemeister-Schreier is applied. | Combination of Todd-Coxeter **[CDHW73]** and Reidemeister-Schreier rewriting **[MKS76]**. |
| `FactoredOrder(G : parameters)` | As `Order`, but returns factored order; reports error in non-finite cases. | As above. |

*Worked examples: H70E43 (448 cosets of H in (8,7|2,3)), H70E44 (`Order` of (8,7|2,3) = 10752), H70E45 (Harada-Norton group: `CosetLimit`, `Strategy := "Hard"`, `Lookahead := 2`; index 1,140,000), H70E46 (parametrised family, batch `Order` computations with `Strategy := "Easy"`).*

### 70.6.3 Implicit Invocation of the Todd-Coxeter Algorithm

Functions such as `meet`, `Normaliser`, and `CommutatorSubgroup` may indirectly invoke coset
enumeration. The global parameters for such implicit calls are controlled by:

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SetGlobalTCParameters(: parameters)` | Set parameter values for all implicitly invoked Todd-Coxeter coset enumerations. Parameters as in `CosetEnumerationProcess` (Chapter 71). | — |
| `UnsetGlobalTCParameters()` | Restore default values for globally set Todd-Coxeter parameters. | — |

*Worked examples: H70E47 (Harada-Norton: default `Normaliser` fails; `SetGlobalTCParameters(: Strategy := "Hard")` makes it work; H is self-normalising).*

### 70.6.4 Constructing a Presentation for a Subgroup

#### 70.6.4.1 Introduction

Given `H` of finite index in fp-group `G`, a presentation for `H` can be obtained on Schreier
generators or on given generators via **Reidemeister-Schreier rewriting** **[MKS76]**, if necessary
together with **extended coset enumeration** **[AR84, HKRR84]**. For the abelian quotient of `H`
only, use `AbelianQuotientInvariants` directly (more efficient: abelianises each relator as it
is constructed).

#### 70.6.4.2 Rewriting

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Rewrite(G, H : parameters)` | Given `H` of finite index in `G`, return an fp-group `R ≅ H` with presentation on (some) Schreier generators of `H` in `G`, plus the isomorphism `H → R`. `R` is created as a subgroup of `G`. Parameters: `Simplify` (default `true`: apply Tietze transformations after rewriting), `EliminationLimit` (default 100), `ExpandLimit` (default 150), `GeneratorsLimit` (default 0), `LengthLimit` (default ∞), `SaveLimit` (default 10), `SearchSimultaneous` (default 20), `Iterations` (default 10000), `Print` (default 0). May invoke coset enumeration (global params apply). | Reidemeister-Schreier rewriting **[MKS76]** + optional Tietze simplification **[HKRR84]**. |
| `Rewrite(G, ∼H : parameters)` | Compute defining relations for `H` (of finite index in `G`) **on its existing generators**, using extended coset enumeration + Reidemeister-Schreier, and modify `H` in-place. Preserves isomorphism type and embedding. Parameter: `Simplify` (default `true`: apply substring-search Tietze transformations, not modifying generating set). Full `CosetEnumerationProcess` parameters also accepted. | Extended coset enumeration **[AR84, HKRR84]** + Reidemeister-Schreier **[MKS76]** + optional Tietze **[HKRR84]**. |

*Worked examples: H70E48 (subgroup K of index 3 in a 4-relator group: `Rewrite`, AQ invariants Z/2³, 2-quotient of class 30 has order 2^62), H70E49 (`Rewrite(F, ∼H)` to get a presentation of L2(7) on generators a=(xy)^2 and b=y).*

---

## 70.7 Subgroups of Finite Index

### 70.7.1 Low Index Subgroups

Algorithm due to Sims **[Sim94, §5.6]**: builds all coset tables via a backtrack search, defining
coset table entries systematically and using group relations to deduce entries or detect dead ends.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LowIndexSubgroups(G, R : parameters)` | Determine conjugacy classes of subgroups whose indices lie in range `R` (integer `n` → `[1,n]`; tuple `<a,b>` → `[a,b]`). Returns sequence sorted by increasing index. Parameters: `ColumnMajor` (default `false`; `true` is better when generators have large order), `GeneratingSets` (default `false`; if `true` return compact generating sets rather than subgroup objects), `Limit` (default ∞), `Long` (sequence of relator indices to defer until a complete table is found), `Print` (0–3), `Subgroup` (only find subgroups containing this subgroup; default trivial subgroup), `TimeLimit` (default 0 = no limit). | Low-index subgroup algorithm **[Sim94, §5.6]**: systematic backtrack through all coset table completions. |
| `LowIndexProcess(G, R : parameters)` | Create a process for `LowIndexSubgroups`: returns conjugacy classes one at a time. Same parameters except no `Limit`. Setting `TimeLimit` limits total time across all calls to `NextSubgroup`. | As above. |
| `NextSubgroup(∼P)` / `NextSubgroup(∼P, ∼G)` | Advance process `P` to the next conjugacy class. | — |
| `ExtractGroup(P)` | Extract the representative subgroup for the class currently defined by process `P`. | — |
| `ExtractGenerators(P)` | Extract a generating set for the current representative subgroup. | — |
| `IsEmpty(P)` | True if the process has found all conjugacy classes. | — |
| `IsValid(P)` | True if no time limit has been exceeded. | — |
| `LowIndexNormalSubgroups(G, n : parameters)` | All normal subgroups of fp-group `G` up to index `n` (n ≤ 100,000). Returns a sequence of records with fields `Group` (presentation), `Index`, `Supergroups` (set of positions in the sequence). Parameters: `PrintLevel` (0–2), `Simplify` (`"No"` default, `"Yes"`, or `"LengthLimit"`). | Uses `LowIndexSubgroups` internally, filtering for normality. |

*Worked examples: H70E50 (Lorimer: Tutte's 8-cage and Conder graph as two subgroups of index 10), H70E51 (PGL2(9): low-index to enumerate all 26 conjugacy classes up to index 720), H70E52 (triangle group ⟨a,b|a²,b³,(ab)⁷⟩: 16 classes up to index 15), H70E53 (process: prove PSL(2,8) is a homomorphic image), H70E54 (process: detect infinite group via subgroup with infinite abelian quotient).*

### 70.7.2 Subgroup Constructions

Most operations require a closed coset table. Implicit coset enumerations are controlled via
`SetGlobalTCParameters`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `H ^ u` / `Conjugate(H, u)` | Conjugate of subgroup `H` by word `u` in the same ambient group. | — |
| `H meet K` | Intersection of two subgroups `H`, `K` of finite index in `G`. Requires closed coset tables for both. | Coset table intersection. |
| `Core(G, H)` | Core of subgroup `H` (of finite index) in `G`. Requires closed coset table. | Coset table intersection of all conjugates. |
| `GeneratingWords(G, H)` | Set of words in the generators of `G` generating `H`. The returned set need not correspond to internal generators. | Schreier system from coset table. |
| `MaximalOvergroup(G, H)` | A maximal subgroup of `G` containing `H` (the group `G` itself if `H` is already maximal). Requires closed coset table. | Coset table methods. |
| `MinimalOvergroup(G, H)` | A minimal overgroup of `H` in `G` (a subgroup `K` such that `H` is maximal in `K`; returns `G` if `H` is maximal). Requires closed coset table. | Coset table methods. |
| `H ^ G` / `NormalClosure(G, H)` | Normal closure of `H` (of finite index) in `G`. Requires closed coset table. | Coset table. |
| `Normaliser(G, H)` / `Normalizer(G, H)` | Normaliser of `H` (of finite index) in `G`. Requires closed coset table. | Todd-Coxeter **[CDHW73]** on normaliser subgroup. |
| `SchreierGenerators(G, H : parameters)` | Schreier generators for `H` (finite index) as words in `G`. Parameter `Simplify` (default `true`): heuristically eliminate redundant Schreier generators. Requires closed coset table. | Schreier generator construction from coset table; optional reduction. |
| `SchreierSystem(G, H)` / `Transversal(G, H)` | Right Schreier system of coset representatives for `H` in `G`, and the Schreier coset function. Returns `(set of words, coset function)`. Requires closed coset table. | Schreier system from coset table. |
| `Transversal(G, H, K)` | Double coset representatives `HuK` for subgroups `H`, `K` of finite index in `G`, and a map from `G` to representatives. Requires closed coset table for `H`. | Orbits of right cosets of `H` under action of generators of `K`. |

*Worked examples: H70E55 (space group p4g: normal closure, minimal/maximal overgroup, transversal, intersection, core), H70E56 (triangle group ⟨x,y|x²,y³,(xy)⁷⟩: low-index, core of index-7 subgroup, `SchreierGenerators` with and without simplification).*

### 70.7.3 Properties of Subgroups

All operations require a closed coset table. Implicit enumerations are controlled via
`SetGlobalTCParameters`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `u in H` | True if word `u` in an fp-group `K` (sharing ambient group with `H`) lies in `H` (of finite index). Requires closed coset table for `H`. | Coset table lookup. |
| `u notin H` | True if `u` does not lie in `H`. | — |
| `H eq K` | True if subgroups `H` and `K` (both of finite index in `G`) are equal. | Coset table comparison. |
| `H ne K` | True if `H ≠ K`. | — |
| `H subset K` | True if `H ⊆ K` (both of finite index). Requires closed coset table for `K`. | — |
| `H notsubset K` | True if `H ⊄ K`. | — |
| `IsConjugate(G, H, K)` | True if `H` and `K` (both of finite index) are conjugate in `G`; returns conjugating element as second value. Requires closed coset tables for both. | Coset table permutation representation. |
| `IsNormal(G, H)` | True if `H` is normal in `G` (of finite index). Requires closed coset table. | Coset table check. |
| `IsMaximal(G, H)` | True if `H` is maximal in `G` (of finite index). Requires closed coset table. | Coset table. |
| `IsSelfNormalizing(G, H)` | True if `H` is self-normalizing in `G`. Requires closed coset table. | `Normaliser(G,H) eq H` via coset table. |

*Worked examples: H70E57 (p4g: normality, maximality, conjugacy, subset), H70E58 (Hall-Janko group J2: random subgroup search → maximal subgroups).*

---

## 70.8 Coset Spaces and Tables

### 70.8.1 Coset Tables

A coset table `T` for subgroup `H` (index `r`) in fp-group `G` is a mapping `{1,...,r} × G → {0,...,r}`: `T(i, x) = k` means coset `i` maps to coset `k` under `x ∈ G`. The value 0 denotes unknown images (incomplete table).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CosetTable(G, H : parameters)` | (Right) coset table for `G` over `H`, by the Todd-Coxeter procedure. Codomain includes 0 if table is not closed. Accepts parameters as `CosetEnumerationProcess` (Chapter 71). | Todd-Coxeter coset enumeration **[CDHW73, Hav91]**, ACE3 **[Ram]**. |
| `CosetTableToRepresentation(G, T)` | Given coset table `T` for `H` in `G`: returns `(φ: G → P, P: permutation group, K: kernel)`. | Coset table columns define the permutation representation. |
| `CosetTableToPermutationGroup(G, T)` | The permutation group image `P` from coset table `T` (second return value of `CosetTableToRepresentation`). | — |

*Worked examples: H70E59 (infinite dihedral group: coset table of 10-element subgroup, permutation representation, kernel).*

### 70.8.2 Coset Spaces: Construction

An **indexed coset space** `V` represents cosets as integers `{1,...,m}`; an **explicit coset space** represents cosets as pairs `<H, x>`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CosetSpace(G, H : parameters)` | Indexed right coset space for `H` in `G` (possibly incomplete if enumeration fails). Accepts parameters as `CosetEnumerationProcess` (Chapter 71). | Todd-Coxeter **[CDHW73]**, ACE3 **[Ram]**. |
| `RightCosetSpace(G, H : parameters)` | Explicit right coset space: elements are pairs `<H, x>`. Uses Todd-Coxeter. | Todd-Coxeter **[CDHW73]**, ACE3 **[Ram]**. |
| `LeftCosetSpace(G, H : parameters)` | Explicit left coset space: elements are pairs `<x, H>`. Uses Todd-Coxeter. | Todd-Coxeter **[CDHW73]**, ACE3 **[Ram]**. |

### 70.8.3 Coset Spaces: Elementary Operations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `H * g` | Right coset of subgroup `H` as an element of the right coset space, with representative `g ∈ G`. | — |
| `C * g` | Right coset obtained by the right action of `g ∈ G` on coset `C`. | Coset table lookup. |
| `C * D` | Product of two right cosets of the same normal subgroup (as a coset). | — |
| `g in C` | True if `g ∈ G` lies in coset `C`. | — |
| `g notin C` | True if `g ∈ G` does not lie in coset `C`. | — |
| `C1 eq C2` | True if cosets `C1` and `C2` are equal. | — |
| `C1 ne C2` | True if cosets are not equal. | — |

### 70.8.4 Accessing Information

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `#V` | Cardinality of coset space `V`. | — |
| `Action(V)` | The coset table (mapping `V × G → V`) giving the action of `G` on `V`. | — |
| `<i, w> @ T` / `T(i, w)` | Image of coset `i` in coset table `T` under the action of word `w`. | Coset table lookup. |
| `ExplicitCoset(V, i)` | Explicit coset corresponding to indexed coset `i` (in an explicit coset space). | — |
| `IndexedCoset(V, w)` | Indexed coset in `V` corresponding to element `w ∈ G`. | Coset table lookup. |
| `IndexedCoset(V, C)` | Indexed coset corresponding to explicit coset `C`. | — |
| `Group(V)` | The group `G` for which `V` is a coset space. | — |
| `Subgroup(V)` | The subgroup `H` such that `V` is a coset space for `G` over `H`. | — |
| `IsComplete(V)` | True if coset space `V` is complete (no undefined images). | — |
| `ExcludedConjugates(V)` / `ExcludedConjugates(T)` | Given partial or complete coset space `V` (or coset table `T`): returns the set `E = {gi^{−1} hj gi | gi generator of G, hj generator of H, and gi^{−1} hj gi ∉ H mod V}`. If `E = ∅`, `H` is normal. Adding elements of `E` to generators of `H` typically grows the subgroup towards the normal closure. Useful for the Todd-Coxeter algorithm when seeking small-index subgroups. | Coset action check. |
| `Transversal(G, H)` / `RightTransversal(G, H)` | Returns `(T: right transversal as set of words, φ: G → T)` where `φ(g) = ti` s.t. `g ∈ H*ti`. May invoke coset enumeration (global params apply). | Coset table → transversal. |

*Worked examples: H70E60 (infinite dihedral group: right transversal and transversal map), H70E61 (G=(8,7|2,3): indexed and explicit coset spaces, `ExplicitCoset`, `IndexedCoset`, coset multiplication), H70E62 (derived subgroup via `CosetSpace` + `ExcludedConjugates`), H70E63 (building normal closure incrementally using `ExcludedConjugates`).*

### 70.8.5 Double Coset Spaces: Construction

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DoubleCoset(G, H, g, K)` | The double coset `H*g*K` in `G`. | — |
| `DoubleCosets(G, H, K)` | Set of all double cosets `H*g*K` in `G`. May invoke coset enumeration (global params via `SetGlobalTCParameters`). | Orbits of right cosets of `H` under `K` action. |

*Worked examples: H70E64 (infinite dihedral group: H-H double cosets).*

### 70.8.6 Coset Spaces: Selection of Cosets

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CosetsSatisfying(T, S : parameters)` / `CosetsSatisfying(V, S : parameters)` | Given coset table `T` or coset space `V`, and subgroup generators `S`: return set of coset representatives satisfying conditions given by parameters. Parameters: `First` (start at coset `i`; default 1), `Last` (stop after coset `j`; default `#V`), `Limit` (stop after `l` found; default ∞), `Normalizing` (if `true`, select cosets `x` s.t. `x^{−1} h_j x ∈ H` for each `hj ∈ S`), `Order` (select cosets `x` s.t. `x^n ∈ H`), `Print` (if > 0, print found cosets). | Coset table search with condition filtering. |
| `CosetSatisfying(T, S : parameters)` / `CosetSatisfying(V, S : parameters)` | As `CosetsSatisfying` with `Limit := 1`: returns a set containing a single coset representative (or empty). | — |

*Worked examples: H70E65 (braid group B4: `CosetSatisfying` with `Normalizing := true` to find an element normalising H).*

### 70.8.7 Coset Spaces: Induced Homomorphism

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CosetAction(G, H)` | Permutation representation of `G` by action on cosets of `H`. Returns `(φ, φ(G), kernel)`. `G` may be infinite; requires finite index. Uses Todd-Coxeter. | Todd-Coxeter **[CDHW73]** + regular action on coset table. |
| `CosetAction(V)` | Permutation representation from action on coset space `V`. Returns `(φ, φ(G))`. | — |
| `CosetImage(G, H)` | Permutation group image of `G` acting on cosets of `H`. | — |
| `CosetImage(V)` | Permutation group image from coset space `V`. | — |
| `CosetKernel(G, H)` | Kernel of `G` in its action on cosets of `H`. Only available for very small index. May invoke coset enumeration (global params apply). | — |
| `CosetKernel(V)` | Kernel of `G` in its action on coset space `V`. | — |

*Worked examples: H70E66 (first Conway group Co1: `CosetSpace` with `FillFactor`, `CosetImage`, degree 98280), H70E67 (G2(3): `CosetAction`, degree 351, Sylow 2-subgroup).*

---

## 70.9 Simplification

### 70.9.1 Reducing Generating Sets

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ReduceGenerators(G)` | Attempt to construct a presentation `H` on fewer generators. `H` is returned as a subgroup of `G` (so element coercion works); isomorphism is the second return value. If a presentation for `G` is known, uses Tietze simplification; otherwise rewrites with respect to a supergroup. | Tietze transformations or supergroup rewriting. |

### 70.9.2 Tietze Transformations

Presentation simplification by **Tietze transformations** and **substring searching** **[HKRR84]**.
The core operations are: (1) eliminate generators using short relators; (2) shorten relators by
substituting substrings matching sides of existing relations.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Simplify(G : parameters)` | Simplify the presentation of fp-group `G` by repeatedly eliminating generators and then shortening relators by substring substitution. Returns a new group `K ≅ G` (as a subgroup of `G`) and the isomorphism `f: G → K`. Parameters: `Preserve` (indices of generators not to eliminate; default []), `Iterations` (default 10000), `EliminationLimit` (max generators eliminated per elimination phase; default 100), `LengthLimit` (max total relator length; default ∞), `ExpandLimit` (max growth in total relator length per elimination phase, in %; default 150), `GeneratorsLimit` (min generators to retain; default 0), `SaveLimit` (repeat simplification phase if reduction > n%; default 10), `SearchSimultaneous` (relators processed simultaneously; default 20), `Print` (default from `GetVerbose("Tietze")`). | Tietze transformations **[HKRR84]**: generator elimination + substring search for relator shortening. |
| `SimplifyLength(G : parameters)` | As `Simplify` but terminates when total relator length starts to increase with further generator elimination. Same parameters as `Simplify`. | Tietze transformations **[HKRR84]**, length-sensitive stopping criterion. |
| `TietzeProcess(G : parameters)` | Create a Tietze process with the presentation of `G` as starting point. Same parameters as `Simplify` (set defaults for subsequent operations). | — |
| `ShowOptions(∼P : parameters)` | Display all current control parameters of Tietze process `P`. | — |
| `SetOptions(∼P : parameters)` | Permanently override control parameters of process `P`. | — |
| `Simplify(∼P : parameters)` / `SimplifyPresentation(∼P : parameters)` | Apply default strategy (eliminate + shorten) to process `P` until no further progress. Parameters may be overridden. | Tietze transformations **[HKRR84]**. |
| `SimplifyLength(∼P : parameters)` | Apply default strategy to process `P`, stopping when total length starts to increase. | Tietze transformations **[HKRR84]**. |
| `Eliminate(∼P : parameters)` / `EliminateGenerators(∼P : parameters)` | Eliminate generators from process `P`: first trivial generators (relators of length 1), then those appearing once in a length-2 relator. Parameters overridable: `EliminationLimit`, `ExpandLimit`, `GeneratorsLimit`, `LengthLimit`. Additional: `Relator` (use specific relator; default 0), `Generator` (eliminate specific generator; default 0). | Tietze generator elimination. |
| `Search(∼P : parameters)` | Simplify by searching for common substrings in pairs of relators where the substring length > half the shorter relator's length; also apply length-1 and length-2 relators. Parameters overridable: `SaveLimit`, `SearchSimultaneous`. | Tietze substring search **[HKRR84]**. |
| `SearchEqual(∼P : parameters)` | As `Search` but for substrings of exactly half the shorter relator's length. Parameter: `SearchSimultaneous`. | Tietze substring search **[HKRR84]**. |
| `Group(P)` | Extract the fp-group `G` defined by the current presentation of process `P`, plus the isomorphism to the original group. `G` is returned as a subgroup of the original group. | — |
| `NumberOfGenerators(P)` / `Ngens(P)` | Number of generators in the current presentation of process `P`. | — |
| `NumberOfRelations(P)` / `Nrels(P)` | Number of relations in the current presentation of process `P`. | — |
| `PresentationLength(P)` | Total relator length in the current presentation of process `P`. | — |

*Worked examples: H70E68 (Fibonacci group F(8): `Simplify` → 2-generator presentation), H70E69 (F(7): Tietze process with explicit `Eliminate` + `Search` steps → 2-generator; also `Simplify` with `Iterations := 5`), H70E70 (F(8): `Simplify` with `Preserve := [1,2]` to rewrite in terms of x1, x2), H70E71 (F(9) infinite: Newman's theorem via `pQuotient`, `ncl`, `Rewrite`, `TietzeProcess`/`pQuotientProcess`), H70E72 (PSL3(7):2 identification: `Rewrite` → 133 generators; `SimplifyLength` → 48 generators, length 7152, enumerable in 289s).*

---

## 70.10 Representation Theory

Functions for creating R[G]-modules for fp-groups. All may require a closed coset table; if not
present, coset enumeration is invoked (controllable via `SetGlobalTCParameters`). See also
Chapter 89.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `GModulePrimes(G, A)` | For fp-group `G` and normal subgroup `A` of finite index: determine all primes `p` for which the maximal p-elementary abelian quotient of `A` (as an F_p[G]-module M_p via conjugation) is non-trivial. Returns a multiset `S`; if `0 ∉ S` then the abelian quotient of `A` is finite and `mult(p, S) = dim(M_p)`; if `0 ∈ S` with multiplicity `m`, there are `m` copies of Z in the abelian quotient and `dim(M_p) = m + mult(p, S)` for every `p`. | Reidemeister-Schreier + Smith normal form to compute abelian section dimensions. |
| `GModulePrimes(G, A, B)` | As above but for the section `A/B` (both `A`, `B` normal in `G`, `B ⊆ A`). | — |
| `GModule(G, A, p)` | For fp-group `G`, normal subgroup `A` of finite index, prime `p`: create the F_p[G]-module `M` for the conjugation action on the maximal p-elementary abelian quotient of `A`. Returns `(M, π: A → M)`. Normality of `A` is not checked. | Coset table + Smith normal form mod p. |
| `GModule(G, A, B, p)` / `GModule(G, A, B)` | As above for section `A/B`. Prime `p` may be omitted if the quotient is automatically a p-group. Faster if `p` is specified. Returns `(M, π: A → M)`. | As above. |
| `Pullback(f, N)` | For map `f: A → M` (where `A` is a normal subgroup of fp-group `G` and `M` an F_p[G]-module) and submodule `N`: compute the preimage of `N` under `f` as a subgroup of `A`. Generally faster and more concise than using `@@`. Reports error if pullback fails. | Fast submodule pullback (avoids full coset enumeration for the preimage). |

*Worked examples: H70E73 (G of order 3753792: `GModulePrimes`, `GModule(G,H,3)`, `Submodules`, `Pullback`, `ReduceGenerators`), H70E74 (abelian normal subgroup: loop over primes, `IsDecomposable`).*

---

## 70.11 Small Group Identification

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IdentifyGroup(G)` | Locate `<o, n>` such that `SmallGroup(o, n) ≅ G`. Errors if the construction of a permutation representation fails or `G` is not in the database (Ch. 66.2). Internally enumerates cosets of the trivial subgroup with coset limit `100·o` (unless order is already known to be ≤ o, in which case global TC parameters apply). Use `Order(G)` first for hard cases, then `SetGlobalTCParameters` before calling `IdentifyGroup`. | Todd-Coxeter **[CDHW73]** to build regular representation, then database lookup. |

*Worked examples: H70E75 (group of order 6 fails initial `IdentifyGroup`; `Order(G : Print := true)` shows ≈10^6 cosets needed; then `SetGlobalTCParameters(: Strategy := "Hard")` and `IdentifyGroup(G)` succeeds with `<6,1>`).*

### 70.11.1 Concrete Representations of Small Groups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PermutationGroup(G)` | Faithful permutation representation of (small) fp-group `G` and the isomorphism. Computes order, then takes regular representation and reduces degree. Restricted to small groups. | Regular permutation representation + degree reduction. |
| `PCGroup(G)` | Faithful PC-group representation of (small, soluble) fp-group `G` and the isomorphism. Computes order, then computes soluble quotient with that order. Restricted to small soluble groups. | Soluble quotient algorithm. |

---

## 70.12 Bibliography

| Key | Reference |
|-----|-----------|
| **[AR84]** | D. G. Arrell and E. F. Robertson. *A modified Todd-Coxeter algorithm.* In Computational group theory (Durham, 1982), pages 27–32. Academic Press, London, 1984. |
| **[CDHW73]** | John J. Cannon, Lucien A. Dimino, George Havas, and Jane M. Watson. *Implementation and analysis of the Todd-Coxeter algorithm.* Math. Comp., 27:463–490, 1973. |
| **[CHN11]** | M. Conder, G. Havas, and M. Newman. *On one-relator quotients of the modular group.* In Proc. Groups St Andrews 2009 in Bath, number 387 in London Mathematical Society Lecture Note Series, pages 183–197. Cambridge University Press, 2011. |
| **[COS08]** | A. Cavicchioli, E. O'Brien, and F. Spaggiari. *On some questions about a family of cyclically presented groups.* J. Algebra, 320(11):4063–4072, 2008. |
| **[Fab09]** | Anna Fabianska. *Algorithmic analysis of presentations of groups and modules.* Dissertation, RWTH Aachen University, 2009. |
| **[Hav91]** | G. Havas. *Coset enumeration strategies.* In ISSAC'91, pages 191–199. ACM Press, 1991. |
| **[HH10]** | G. Havas and D. F. Holt. *On Coxeter's families of group presentations.* J. Algebra, 324(5):1076–1082, 2010. |
| **[HKRR84]** | George Havas, P. E. Kenne, J. S. Richardson, and E. F. Robertson. *A Tietze transformation program.* In Computational group theory (Durham, 1982), pages 69–73. Academic Press, London, 1984. |
| **[MKS76]** | Wilhelm Magnus, Abraham Karrass, and Donald Solitar. *Combinatorial group theory.* Dover Publications Inc., New York, revised edition, 1976. Presentations of groups in terms of generators and relations. |
| **[Nic96]** | Werner Nickel. *Computing nilpotent quotients of finitely presented groups.* In Geometric and computational perspectives on infinite groups (Minneapolis, MN and New Brunswick, NJ, 1994), pages 175–191. Amer. Math. Soc., Providence, RI, 1996. |
| **[NO96]** | M. F. Newman and E. A. O'Brien. *Application of computers to questions like those of Burnside. II.* Internat. J. Algebra Comput., 6(5):593–605, 1996. |
| **[PF09]** | W. Plesken and A. Fabianska. *An L2-quotient algorithm for finitely presented groups.* J. Algebra, 322(3):914–935, 2009. |
| **[Ram]** | Colin Ramsay. *ACE.* URL: http://www.csee.uq.edu.au/~cram/. |
| **[Sim94]** | Charles C. Sims. *Computation with finitely presented groups.* Cambridge University Press, Cambridge, 1994. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Key functions |
|--------------------|---------------|
| Todd-Coxeter coset enumeration **[CDHW73, Hav91]**, ACE3 **[Ram]** | `ToddCoxeter`, `Index`, `FactoredIndex`, `Order`, `FactoredOrder`, `CosetTable`, `CosetSpace`, `CosetAction`, `CosetImage`, `CosetKernel`, `sub<>` (via normal closure), `ncl<>`, `DerivedSubgroup`, `Normaliser`, `Core`, `H meet K`, `DoubleCosets`, `IdentifyGroup`, `PermutationGroup` |
| Reidemeister-Schreier rewriting **[MKS76, AR84, HKRR84]** | `Rewrite(G, H)`, `Rewrite(G, ∼H)`, `AbelianQuotientInvariants(H)`, `AQInvariants(H, n)` |
| Tietze transformations **[HKRR84]** | `Simplify`, `SimplifyLength`, `TietzeProcess`, `Eliminate`, `Search`, `SearchEqual`, `ReduceGenerators`, `Rewrite` (with `Simplify := true`) |
| Low-index subgroups (Sims backtrack) **[Sim94, §5.6]** | `LowIndexSubgroups`, `LowIndexProcess`, `LowIndexNormalSubgroups` |
| p-Quotient algorithm (ANU pQ) **[NO96]** | `pQuotient` |
| Nilpotent quotient algorithm (ANU NQ) **[Nic96]** | `NilpotentQuotient`, `SetVerbose("NilpotentQuotient", n)` |
| Soluble quotient algorithm | `SolvableQuotient` / `SolubleQuotient` |
| Smith normal form (abelian quotient) | `AbelianQuotient`, `ElementaryAbelianQuotient`, `AbelianQuotientInvariants` / `AQInvariants`, `IsPerfect`, `TorsionFreeRank` |
| L2-quotient algorithm (Plesken-Fabianska) **[PF09, Fab09]** | `L2Quotients`, `L2Type`, `L2Generators`, `L2Ideals`, `HasInfinitePSL2Quotient` |
| Backtrack search for homomorphisms to finite groups | `Homomorphisms`, `HomomorphismsProcess`, `SimpleQuotients`, `SimpleQuotientProcess` |
| Isomorphism search (word-length bounded backtrack) | `SearchForIsomorphism` |
| F_p[G]-module theory (conjugation on elementary abelian sections) | `GModulePrimes`, `GModule`, `Pullback` |
