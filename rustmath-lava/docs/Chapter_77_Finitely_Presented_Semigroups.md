# Chapter 77 — Finitely Presented Semigroups

**Handbook part:** X — Finitely-Presented Groups
**Handbook pages:** 2389–2397 (PDF pages 2520–2531)

---

## Scope and overview

This chapter presents the functions designed for computing with finitely-presented semigroups
(fp-semigroups for short). The chapter covers the full lifecycle of fp-semigroup computation:
construction of free semigroups and monoids; element (word) construction; arithmetic and
comparison on words; specification of presentations by generators and relations; access to
subsemigroups, ideals, and quotients; direct and free products; elementary Tietze
transformations for modifying presentations; and a suite of string (word) operations.

Words in an fp-semigroup are ordered first by length and then lexicographically, with the
ordering on generators S.1 < S.2 < S.3 < · · ·. Equality of words is syntactic (identity as
elements of the underlying free semigroup), not semantic modulo the given relations.

No algorithmic attribution or bibliography entries appear in this chapter; it is a reference
for syntactic/structural operations on finitely-presented semigroups and their words.

---

## 77.1 Introduction

This section introduces the fp-semigroup framework in Magma. No intrinsics are defined here;
the section serves as a brief orientation to the chapter.

---

## 77.2 The Construction of Free Semigroups and their Elements

### 77.2.1 Structure Constructors

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `FreeSemigroup(n)` | Construct the free semigroup `F` on `n` generators, where `n` is a positive integer. The i-th generator is referenced as `F.i`. A special assignment form `F<x,y,...> := FreeSemigroup(n)` assigns names to the generators. | — |
| `FreeMonoid(n)` | Construct the free monoid `F` on `n` generators, where `n` is a positive integer. The i-th generator is referenced as `F.i`. The same named-generator assignment form is available. | — |

*Worked examples: H77E1 (constructing a free semigroup on two generators, both unnamed and with named generators `x`, `y`).*

### 77.2.2 Element Constructors

A word is defined inductively: a generator is a word; the product `uv` of words `u` and `v` is a word; the power `u^n` of a word `u` (for integer `n`) is a word. Suppose `S` is an fp-semigroup for which generators have already been defined.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `S ! [i1, ..., is]` | Given a semigroup `S` defined on `r` generators and a sequence `Q = [i1, …, is]` of integers in `[1, r]`, construct the word `S.i1 * S.i2 * … * S.is`. | — |
| `Id(M)` / `M ! 1` | Construct the identity element (empty word) for the fp-monoid `M`. | — |

---

## 77.3 Elementary Operators for Words

The word operations defined here may be applied to words of a free semigroup or to words of a semigroup with non-trivial relations.

### 77.3.1 Multiplication and Exponentiation

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `u * v` | Given words `u` and `v` belonging to the same fp-semigroup `S`, return the product of `u` and `v`. | — |
| `u ^ n` | The `n`-th power of the word `u`, where `n` is a positive integer. | — |
| `G ! Q` | Given a sequence `Q` of words belonging to the fp-semigroup `G`, return the product `Q[1] * Q[2] * … * Q[n]` as a word in `G`. | — |

### 77.3.2 The Length of a Word

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `#u` | The length of the word `u`. | — |

### 77.3.3 Equality and Comparison

Words of an fp-semigroup are ordered first by length and then lexicographically, with the ordering on generators S.1 < S.2 < S.3 < · · ·. Here `u` and `v` are words belonging to some common fp-semigroup.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `u eq v` | Returns `true` if the words `u` and `v` are identical (as elements of the appropriate free semigroup), `false` otherwise. | — |
| `u ne v` | Returns `true` if the words `u` and `v` are not identical (as elements of the appropriate free semigroup), `false` otherwise. | — |
| `u lt v` | Returns `true` if `u` precedes `v` with respect to the length-then-lexicographic ordering, `false` otherwise. | — |
| `u le v` | Returns `true` if `u` precedes or is equal to `v` with respect to the ordering, `false` otherwise. | — |
| `u ge v` | Returns `true` if `u` follows or is equal to `v` with respect to the ordering, `false` otherwise. | — |
| `u gt v` | Returns `true` if `u` follows `v` with respect to the ordering. | — |
| `IsOne(u)` | Returns `true` if the word `u`, belonging to the monoid `M`, is the identity word, `false` otherwise. | — |

---

## 77.4 Specification of a Presentation

### 77.4.1 Relations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `w1 = w2` | Given words `w1` and `w2` over the generators of an fp-semigroup `S`, create the relation `w1 = w2`. This relation is not automatically added to the existing defining relations of `S`; it may be added via the `quo`-constructor or other means. | — |
| `LHS(r)` | Given a relation `r` over the generators of `S`, return the left-hand side of `r` as a word over the generators of `S`. | — |
| `RHS(r)` | Given a relation `r` over the generators of `S`, return the right-hand side of `r` as a word over the generators of `S`. | — |

### 77.4.2 Presentations

A semigroup with non-trivial relations is constructed as a quotient of an existing semigroup, possibly a free semigroup.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Semigroup< generators \| relations >` | Given a generators clause `x1, …, xr` and a set of relations over these generators, construct the free semigroup `F` on those generators and then the quotient of `F` corresponding to the ideal defined by `relations`. Returns: (a) the quotient semigroup `S`; (b) the natural homomorphism φ: F → S. The statement `S<y1,...,yr> := Semigroup<x1,...,xr \| w1,...,ws>` is an abbreviation for constructing `FreeSemigroup(r)` followed by `quo< F \| w1,...,ws >`. | — |
| `Monoid< generators \| relations >` | Given a generators clause `x1, …, xr` and a set of relations over these generators, construct the free monoid `F` on those generators and then the quotient of `F` corresponding to the ideal defined by `relations`. Returns: (a) the quotient monoid `M`; (b) the natural homomorphism φ: F → M. The statement `M<y1,...,yr> := Monoid<x1,...,xr \| w1,...,ws>` is an abbreviation for constructing `FreeMonoid(r)` followed by `quo< F \| w1,...,ws >`. | — |

*Worked examples: H77E2 (creating the monoid defined by the presentation `< x, y | x^2, y^2, (xy)^2 >`).*

### 77.4.3 Accessing the Defining Generators and Relations

The functions in this group provide access to basic information stored for a finitely-presented semigroup `S`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `S . i` | The `i`-th defining generator for `S`. | — |
| `Generators(S)` | A set containing the generators for `S`. | — |
| `NumberOfGenerators(S)` / `Ngens(S)` | The number of generators for `S`. | — |
| `Parent(u)` | The parent semigroup `S` of the word `u`. | — |
| `Relations(S)` | A sequence containing the defining relations for `S`. | — |

---

## 77.5 Subsemigroups, Ideals and Quotients

### 77.5.1 Subsemigroups and Ideals

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `sub< S \| L1, ..., Lr >` | Construct the subsemigroup `R` of the fp-semigroup `S` generated by the words specified by the terms of the generator list `L1, …, Lr`. A term `Li` may be: a word; a set or sequence of words; a sequence of integers representing a word; a set or sequence of sequences of integers representing words; a subsemigroup of an fp-semigroup; or a set or sequence of subsemigroups. All words and semigroups must belong to `S`. Repetitions and occurrences of the identity are removed (unless `R` is trivial). | — |
| `ideal< S \| L1, ..., Lr >` | Construct the two-sided ideal `I` of the fp-semigroup `S` generated by the words specified by the generator list. The possible forms of a term `Li` are the same as for the `sub`-constructor. | — |
| `lideal< G \| L1, ..., Lr >` | Construct the left ideal `I` of the fp-semigroup `S` generated by the words specified by the generator list. The possible forms of a term `Li` are the same as for the `sub`-constructor. | — |
| `rideal< G \| L1, ..., Lr >` | Construct the right ideal `I` of the fp-semigroup `S` generated by the words specified by the generator list. The possible forms of a term `Li` are the same as for the `sub`-constructor. | — |

### 77.5.2 Quotients

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `quo< F \| relations >` | Given an fp-semigroup `F` and a list of relations over the generators of `F`, construct the quotient of `F` by the ideal defined by `relations`. Each term of the list may be: a relation (a pair of words separated by `=`); a relation list (`w1 = w2 = … = wr`, interpreted as `w1 = wr, …, wr-1 = wr`); or, if `F` is a monoid, a word (treated as a relator). The identity of a monoid may be represented by the digit `1`. Returns: (a) the quotient semigroup `S`; (b) the natural homomorphism φ: F → S. | — |

---

## 77.6 Extensions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DirectProduct(R, S)` | Given two fp-semigroups `R` and `S`, construct the direct product of `R` and `S`. | — |
| `FreeProduct(R, S)` | Given two fp-semigroups `R` and `S`, construct the free product of `R` and `S`. | — |

---

## 77.7 Elementary Tietze Transformations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AddRelation(S, r)` / `AddRelation(S, r, i)` | Given an fp-semigroup `S` and a relation `r` in the generators of `S`, create the quotient semigroup obtained by adding `r` to the defining relations of `S`. If an integer `i` is supplied as the third argument, insert the new relation after the `i`-th existing relation; otherwise `r` is appended to the end. | — |
| `DeleteRelation(S, r)` | Given an fp-semigroup `S` and a relation `r` that occurs among the defining relations of `S`, create the semigroup `T` with the same generating set as `S` but with `r` removed. | — |
| `DeleteRelation(S, i)` | Given an fp-semigroup `S` and an integer `i`, 1 ≤ `i` ≤ `m` (where `m` is the number of defining relations), create the semigroup `T` with the same generating set as `S` but with the `i`-th relation omitted. | — |
| `ReplaceRelation(S, r1, r2)` | Given an fp-semigroup `S` and relations `r1` and `r2` (where `r1` is one of the defining relations of `S`), create the semigroup `T` with the same generating set as `S` but with `r1` replaced by `r2`. | — |
| `ReplaceRelation(S, i, r)` | Given an fp-semigroup `S`, an integer `i`, 1 ≤ `i` ≤ `m`, and a relation `r`, create the semigroup `T` with the same generating set as `S` but with the `i`-th relation replaced by `r`. | — |
| `AddGenerator(S)` | Given an fp-semigroup `S` with presentation `< X \| R >`, create the semigroup `T` with presentation `< X ∪ {y} \| R >`, where `y` is a new generator. | — |
| `AddGenerator(S, w)` | Given an fp-semigroup `S` with presentation `< X \| R >` and a word `w` in the generators of `S`, create the semigroup `T` with presentation `< X ∪ {y} \| R ∪ {y = w} >`, where `y` is a new generator. | — |
| `DeleteGenerator(S, y)` | Given an fp-semigroup `S` with presentation `< X \| R >` and a generator `y` of `S` such that either `S` has no relations involving `y`, or has a single relation `r` containing a single occurrence of `y`, create the semigroup `T` with presentation `< X − {y} \| R − {r} >`. | — |

---

## 77.8 String Operations on Words

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Eliminate(u, x, v)` | Given words `u` and `v`, and a generator `x`, belonging to a semigroup `S`, return the word obtained from `u` by replacing each occurrence of `x` by `v`. | — |
| `Match(u, v, f)` | Given words `u` and `v` in the same semigroup `S` and an integer `f` with 1 ≤ `f` ≤ `#u`: if `v` is a subword of `u`, return `true` and the least integer `l` ≥ `f` such that `v` appears as a subword of `u` starting at the `l`-th letter. If no such `l` exists, return only `false`. | — |
| `Random(S, m, n)` | A random word of length `l` in the generators of the semigroup `S`, where `m` ≤ `l` ≤ `n`. | — |
| `RotateWord(u, n)` | The word obtained by cyclically permuting the word `u` by `n` places. Positive `n`: rotation from left to right; negative `n`: rotation from right to left; `n = 0`: returns `u` unchanged. | — |
| `Substitute(u, f, n, v)` | Given words `u` and `v` in a semigroup `S` and non-negative integers `f` and `n`, replace the substring of `u` of length `n` starting at position `f` by the word `v`. If `u` and `v` belong to a monoid `M` and `v = Id(M)`, the substring is deleted. | — |
| `Subword(u, f, n)` | The subword of `u` comprising the `n` consecutive letters commencing at the `f`-th letter of `u`. | — |
| `ElementToSequence(u)` / `Eltseq(u)` | The sequence obtained by decomposing `u` into the indices of its constituent generators. If `u = xi1 * … * xim`, the sequence returned is `[i1, i2, …, im]`. | — |

---

## 77.9 Bibliography

This chapter contains no bibliography. No algorithmic references are attributed in the source text; the intrinsics are structural/syntactic operations with no cited method.

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Free semigroup / monoid construction | `FreeSemigroup`, `FreeMonoid` |
| Word construction from generator sequences | `S ! [i1,...,is]`, `Id(M)` |
| Word arithmetic | `u * v`, `u ^ n`, `G ! Q` |
| Word comparison (length-then-lexicographic order) | `u eq v`, `u ne v`, `u lt v`, `u le v`, `u ge v`, `u gt v`, `IsOne` |
| Presentation specification | `Semigroup< >`, `Monoid< >`, `quo< >` |
| Relation/generator access | `S . i`, `Generators`, `NumberOfGenerators`/`Ngens`, `Parent`, `Relations` |
| Subsemigroup / ideal construction | `sub< >`, `ideal< >`, `lideal< >`, `rideal< >` |
| Extensions | `DirectProduct`, `FreeProduct` |
| Tietze transformations | `AddRelation`, `DeleteRelation`, `ReplaceRelation`, `AddGenerator`, `DeleteGenerator` |
| String operations on words | `Eliminate`, `Match`, `Random`, `RotateWord`, `Substitute`, `Subword`, `ElementToSequence`/`Eltseq` |
