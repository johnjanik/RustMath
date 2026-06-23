# Chapter 92 — Representations of Symmetric Groups

**Handbook part:** XII — Representation Theory
**Handbook pages:** 2781–2785 (PDF pages 2912–2919)

---

## Scope and overview

This chapter describes functions available in Magma for computations concerning the
non-modular representation theory of the symmetric group and the alternating group. It
covers three categories of computation:

1. **Matrix representations** — explicit irreducible representing matrices for elements of the
   symmetric group, available in integral, seminormal, and orthogonal forms. The integral
   representations support three independent algorithms (James–Kerber, Boerner, and Specht's
   original construction); the seminormal and orthogonal forms are constructed by the
   method of James and Kerber **[JK81, §3.3]**.

2. **Characters** — computation of individual character values (using the Murnaghan–Nakayama
   recursion **[JK81, p. 60]** with the hook-length formula **[JK81, p. 56]** for dimension
   values), full irreducible characters, and complete character tables, for both the symmetric
   group and the alternating group.

3. **Alternating group** — the chapter exploits the fact that, in most cases, irreducible
   characters of the symmetric group remain irreducible on restriction to the alternating group.
   Characters of the alternating group are therefore indexed by partitions in the same way as
   those of the symmetric group, taking one from each conjugate pair. When a partition is
   self-conjugate the restriction of the corresponding symmetric-group character is no longer
   irreducible but splits into two irreducible constituents; both are accessible by an index
   parameter `i ∈ {1, 2}`. This method follows **[JK81]**.

Throughout, irreducible representations of the symmetric group of degree n are indexed by
**partitions of weight n**. For background on partitions see Section 145.2 of the Magma handbook.

---

## 92.1 Introduction

*(No intrinsics — see Scope and overview above.)*

---

## 92.2 Representations of the Symmetric Group

For the symmetric group of degree n the irreducible representations can be indexed by
partitions of weight n.

### 92.2.1 Integral Representations

Representing matrices can be defined over the integers. Three algorithms are available,
selectable via the `Al` parameter.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SymmetricRepresentation(pa, pe)` | Given a partition `pa` of weight n and a permutation `pe` in a symmetric group of degree n, return an irreducible representing matrix for `pe`, indexed by `pa`, over the integers. Parameter: `Al` (default `"JamesKerber"`) selects the construction method. | `Al := "JamesKerber"`: method of James and Kerber **[JK81]**. `Al := "Boerner"`: method from Boerner **[Boe67]**. `Al := "Specht"`: direct implementation of Specht's original 1935 construction **[Spe35]**. All three methods yield similar matrices (verified by `IsSimilar` over Q). |

*Worked examples: H92E1 (computing a representing matrix for `[3,2]` and `(3,4,5) ∈ Sym(5)` with the `"Boerner"` and `"Specht"` algorithms; confirming the results are similar over Q).*

### 92.2.2 The Seminormal and Orthogonal Representations

The seminormal and orthogonal representations involve matrices which are not necessarily
integral. The method Magma uses to construct these matrices is described in **[JK81, §3.3]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SymmetricRepresentationSeminormal(pa, pe)` | Given a partition `pa` of weight n and a permutation `pe` in a symmetric group of degree n, return the matrix of the seminormal representation for `pe`, indexed by `pa`, over the rationals. | Young's seminormal construction **[JK81, §3.3]**. |
| `SymmetricRepresentationOrthogonal(pa, pe)` | Given a partition `pa` of weight n and a permutation `pe` in a symmetric group of degree n, return the matrix of the orthogonal representation for `pe`, indexed by `pa`. An orthogonal basis is used; entries may lie in a cyclotomic field. | Young's orthogonal construction **[JK81, §3.3]** (orthogonal basis). |

*Worked examples: H92E2 (comparing seminormal and orthogonal representations of `(3,4,5) ∈ Sym(5)` for partition `[3,2]`; confirming similarity and that both matrices have order 3, matching the permutation order).*

---

## 92.3 Characters of the Symmetric Group

### 92.3.1 Single Values

The method used to compute the value of a character on a permutation is the recursion
formula of Murnaghan and Nakayama **[JK81, p. 60]**, except when computing the value of
a character on the identity permutation, for which the hook-length formula for the dimension
of the representation indexed by the partition is used **[JK81, p. 56]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SymmetricCharacterValue(pa, pe)` | Computes the value of the irreducible character of the symmetric group of degree n indexed by the partition `pa` of weight n on the permutation `pe`. When `pe` is the identity, the hook-length (dimension) formula is used instead of the recursion. | Murnaghan–Nakayama recursion **[JK81, p. 60]**; hook-length formula for dimension **[JK81, p. 56]**. |

### 92.3.2 Irreducible Characters

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SymmetricCharacter(pa)` | Return the character of the representation of the symmetric group of degree n indexed by the partition `pa` of weight n. | Built from `SymmetricCharacterValue` via the Murnaghan–Nakayama method **[JK81]**. |

### 92.3.3 Character Table

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SymmetricCharacterTable(d)` | Return the character table of the symmetric group of degree `d`. | Full table assembled via Murnaghan–Nakayama recursion and the hook-length formula **[JK81]**. |

---

## 92.4 Representations of the Alternating Group

*(No separate intrinsics listed in this section — representations of the alternating group are
accessed through the character/table machinery described in §92.5. The theoretical background
is given in the section prose: irreducible characters of the symmetric group restrict to
irreducible characters of the alternating group except when the indexing partition is
self-conjugate, in which case the character splits into two irreducibles. See §92.5 for
the corresponding intrinsics.)*

---

## 92.5 Characters of the Alternating Group

Routines exploit the fact that in most cases the irreducible characters of the symmetric
group, which can be computed quickly, are also irreducible in the alternating group. Irreducible
characters of the alternating group may therefore be indexed by partitions in the same way.
As the restriction of the character indexed by partition λ equals the restriction of the
character indexed by its conjugate partition, only one from each conjugate pair is needed.
When a partition is self-conjugate the symmetric-group character is no longer irreducible
but is the sum of two irreducibles, accessible via an index `i ∈ {1, 2}`. This method is
described in **[JK81]**.

### 92.5.1 Single Values

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AlternatingCharacterValue(pa, pe)` | Return the value of the character of the alternating group of degree n indexed by the partition `pa` of weight n on the permutation `pe`. The partition `pa` and its conjugate must be distinct (non-self-conjugate). | Restriction from symmetric-group characters via Murnaghan–Nakayama **[JK81]**. |
| `AlternatingCharacterValue(pa, i, pe)` | Return the value of the `i`-th character (i = 1 or 2) of the alternating group of degree n indexed by the self-conjugate partition `pa` of weight n on the permutation `pe`. Required when `pa` equals its own conjugate, since the symmetric-group character then splits into two irreducibles. | Splitting of self-conjugate characters **[JK81]**. |

### 92.5.2 Irreducible Characters

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AlternatingCharacter(pa)` | Return the character of the alternating group of degree n indexed by the partition `pa` of weight n. The partition `pa` and its conjugate must be distinct. | Restriction from the symmetric-group character **[JK81]**. |
| `AlternatingCharacter(pa, i)` | Return the `i`-th character (i = 1 or 2) of the alternating group of degree n indexed by the self-conjugate partition `pa` of weight n. Required when `pa` equals its own conjugate. | Splitting of self-conjugate characters **[JK81]**. |

### 92.5.3 Character Table

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AlternatingCharacterTable(d)` | Returns the character table of the alternating group of degree `d`. | Full table via restriction and splitting of symmetric-group characters **[JK81]**. |

---

## 92.6 Bibliography

| Key | Reference |
|-----|-----------|
| **[Boe67]** | H. Boerner. *Darstellungen von Gruppen.* 2. Aufl. Berlin-Heidelberg-New York: Springer-Verlag. XIV, 317 S., 1967. |
| **[JK81]** | Gordon James and Adalbert Kerber. *The representation theory of the symmetric group.* Addison-Wesley Publishing Co., Reading, Mass., 1981. With a foreword by P. M. Cohn, with an introduction by Gilbert de B. Robinson. |
| **[Spe35]** | Wilhelm Specht. *Die irreduziblen Darstellungen der symmetrischen Gruppe.* Math. Z., 39:696–711, 1935. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| James–Kerber integral representation **[JK81]** | `SymmetricRepresentation(:Al:="JamesKerber")` |
| Boerner's integral representation **[Boe67]** | `SymmetricRepresentation(:Al:="Boerner")` |
| Specht's original (1935) integral representation **[Spe35]** | `SymmetricRepresentation(:Al:="Specht")` |
| Young's seminormal construction **[JK81, §3.3]** | `SymmetricRepresentationSeminormal` |
| Young's orthogonal construction **[JK81, §3.3]** | `SymmetricRepresentationOrthogonal` |
| Murnaghan–Nakayama recursion **[JK81, p. 60]** | `SymmetricCharacterValue`, `SymmetricCharacter`, `SymmetricCharacterTable`, `AlternatingCharacterValue`, `AlternatingCharacter`, `AlternatingCharacterTable` |
| Hook-length (dimension) formula **[JK81, p. 56]** | `SymmetricCharacterValue` (identity permutation case) |
| Self-conjugate partition splitting **[JK81]** | `AlternatingCharacterValue(pa, i, pe)`, `AlternatingCharacter(pa, i)` |
