# Chapter 94 — Introduction to Lie Theory

**Handbook part:** XIII — Lie Theory
**Handbook pages:** 2799–2802 (PDF pages 2930–2935)

---

## Scope and overview

Chapter 94 is the introductory overview for Part XIII (Lie Theory) of the Magma Handbook. It
surveys the range of Lie-theoretic structures that Magma supports and signals which later
chapters treat each topic in detail. No intrinsics are defined here; the chapter is entirely
descriptive. The structures covered are:

1. **Coxeter matrices, Coxeter graphs, Cartan matrices, Dynkin diagrams, and Cartan's naming
   system** for Coxeter groups (Chapter 95).
2. **Finite root systems and finite root data** — the foundational combinatorial data for
   semisimple Lie algebras and groups of Lie type (Chapters 96–97).
3. **Coxeter groups in three formats:** as finitely presented groups, as permutation groups
   (acting on the set of roots), and as reflection groups over a number field, together with
   all finite complex reflection groups (Chapters 98–99).
4. **Lie algebras** — as structure constant algebras, matrix Lie algebras, or finitely
   presented algebras. Algorithms of de Graaf **[dG00]** determine the structure of a
   finite-dimensional Lie algebra over a field; if the algebra is reductive, its root system
   and highest-weight representations can be recovered.
5. **Groups of Lie type** (connected reductive algebraic groups and their split and twisted
   forms) — presented via the Steinberg presentation. Canonical forms and arithmetic
   algorithms for the split case are given in **[CMT04, CHM08]**; twisted groups use Galois
   cohomology following **[Hal05]** (Chapter 100).
6. **Highest-weight representations** of Lie algebras and connected reductive Lie groups —
   construction **[dG01]** plus full combinatorial weight machinery (all functionality of the
   LiE system **[vLCL92]**) (Chapter 104).
7. **Universal enveloping algebras and quantum groups** — an integral basis for the universal
   enveloping algebra of a semisimple Lie algebra over characteristic zero (Section 100.17),
   and quantised versions (quantum groups) (Chapter 102).

---

## 94.1 Descriptions of Coxeter Groups

A *Coxeter system* is a group G with a finite generating set S = {s₁, …, sₙ} defined by
power relations sᵢ² = 1 and braid relations sᵢ sⱼ sᵢ ⋯ = sⱼ sᵢ sⱼ ⋯ (each side of length
mᵢⱼ ≥ 2). Magma uses mᵢⱼ = 0 (rather than the traditional ∞) to signal that the
corresponding relation is omitted. The preferred generating set of every Magma group
identifies the Coxeter group with its Coxeter system.

Descriptions used:
- **Coxeter matrix** M = (mᵢⱼ): the primary encoding.
- **Coxeter graph**: vertices 1, …, n with an edge labeled mᵢⱼ whenever mᵢⱼ ≥ 3.
- **Cartan matrix / Dynkin digraph**: encode a particular reflection representation; for
  finite and affine groups, Cartan's naming system is also available.

Functions for manipulating these descriptions are in **Chapter 95**.

*This section defines no intrinsics.*

---

## 94.2 Root Systems and Root Data

A (real) *reflection* is a linear automorphism that negates a one-dimensional subspace (the
root) and fixes a hyperplane (described by a coroot). A *root system* is a
root/coroot-pair collection closed under the corresponding reflections. Only finite root
systems are supported. Root systems are used to classify semisimple Lie algebras; the
closely related *root data* are used to classify groups of Lie type.

This is described in **Chapters 96 and 97**.

*This section defines no intrinsics.*

---

## 94.3 Coxeter and Reflection Groups

Three methods are provided for computing with Coxeter groups:

- **Coxeter presentation** — the most generally useful form. Elements are held in standard
  normal form (lexicographically least word of minimal length). Normalisation and
  multiplication use Robert Howlett's highly efficient method based on **[BH93]**.
- **Permutation representation on roots** — preferred when the Coxeter group is finite;
  elements are permutations of the full root set (not the minimal-degree representation).
- **Reflection representation over a number field** — Coxeter groups as real (in practice
  number-field) reflection groups. All finite complex reflection groups can also be
  constructed. Fewer facilities are available for reflection groups over an arbitrary field.

Efficient conversion between all three forms is provided.

This is described in **Chapters 98 and 99**.

*This section defines no intrinsics.*

---

## 94.4 Lie Algebras and Groups of Lie Type

Lie algebras can be constructed in three ways:

1. **Structure constant algebras** — the most general form.
2. **Lie matrix algebras** — elements are matrices with Lie bracket [A, B] = AB − BA.
3. **Finitely presented algebras** — generators and relations.

Most functionality targets finite-dimensional algebras over a field. Algorithms designed
and implemented by de Graaf **[dG00]** determine the structure of an arbitrary Lie algebra.
If the algebra is reductive, its root system and highest-weight representations can be
found.

Groups of Lie type (reductive algebraic groups and their split/twisted forms) are given by
the **Steinberg presentation**. Canonical forms and word-arithmetic algorithms for the
split case: **[CMT04, CHM08]**. Twisted groups use a modified presentation via Galois
cohomology: **[Hal05]**. These presentations are *not* in the category `GrpFP` because
generators are parametrised by field elements, so the groups need not be finitely generated.

This is described in **Chapter 100**.

*This section defines no intrinsics.*

---

## 94.5 Highest Weight Representations

Representations of Lie algebras and connected reductive Lie groups are classified by highest
weights. Magma can construct these representations following **[dG01]** and compute the full
combinatorics of their weights. This includes all functionality of the LiE system
**[vLCL92]**.

This is described in **Chapter 104**.

*This section defines no intrinsics.*

---

## 94.6 Universal Enveloping Algebras and Quantum Groups

Given a semisimple Lie algebra over a field of characteristic zero, Magma can construct an
integral basis for its **universal enveloping algebra**; functionality for computing in
these algebras is in **Section 100.17**. The *quantised* versions (quantum groups) are
supported separately and described in **Chapter 102**.

*This section defines no intrinsics.*

---

## 94.7 Bibliography

| Key | Reference |
|-----|-----------|
| **[BH93]** | Brigitte Brink and Robert B. Howlett. A finiteness property and an automatic structure for Coxeter groups. *Math. Ann.*, 296(1):179–190, 1993. |
| **[CHM08]** | Arjeh M. Cohen, Sergei Haller, and Scott H. Murray. Computing in unipotent and reductive algebraic groups. *LMS J. Comput. Math.*, 11:343–366, 2008. |
| **[CMT04]** | Arjeh M. Cohen, Scott H. Murray, and D. E. Taylor. Computing in groups of Lie type. *Math. Comp.*, 73(247):1477–1498, 2004. |
| **[dG00]** | W. A. de Graaf. *Lie Algebras: Theory and Algorithms*. Number 56 in North-Holland Mathematical Library. Elsevier, 2000. |
| **[dG01]** | W. A. de Graaf. Constructing representations of split semisimple Lie algebras. *J. Pure Appl. Algebra*, 164(1-2):87–107, 2001. Effective methods in algebraic geometry (Bath, 2000). |
| **[Hal05]** | Sergei Haller. *Computing Galois Cohomology and Forms of Linear Algebraic Groups*. PhD thesis, Technical University of Eindhoven, 2005. |
| **[vLCL92]** | M. A. A. van Leeuwen, A. M. Cohen, and B. Lisser. *LiE, A Package for Lie Group Computations*. CAN, Amsterdam, 1992. |

---

## Algorithm-to-function quick reference

| Algorithm / method | Bibliography | Described in |
|--------------------|--------------|--------------|
| Howlett's automatic-structure normal form for Coxeter groups | **[BH93]** | Chapter 95 |
| Root system and root datum constructions | — | Chapters 96–97 |
| Conversion between presentation / permutation / reflection forms | — | Chapters 98–99 |
| de Graaf's Lie algebra structure algorithms | **[dG00]** | Chapter 100 |
| Steinberg presentation arithmetic (split groups) | **[CMT04, CHM08]** | Chapter 100 |
| Twisted groups via Galois cohomology | **[Hal05]** | Chapter 100 |
| Universal enveloping algebra (integral basis) | — | Section 100.17 |
| Quantum groups | — | Chapter 102 |
| Highest-weight representation construction | **[dG01]** | Chapter 104 |
| LiE weight combinatorics | **[vLCL92]** | Chapter 104 |
