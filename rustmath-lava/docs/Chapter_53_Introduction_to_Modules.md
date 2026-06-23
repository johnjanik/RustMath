# Chapter 53 — Introduction to Modules

**Handbook part:** VIII — Modules
**Handbook pages:** 1393–1394 (PDF pages 1526–1527)

---

## Scope and overview

Chapter 53 is the gateway chapter for Part VIII of the Magma Handbook, which covers linear
algebra and module theory. It explains the conceptual framework used throughout the part and
describes how the major mathematical structures — vector spaces, inner product spaces, general
R-modules, R[G]-modules, and R-module homomorphisms — are represented and organised in
Magma.

The chapter also establishes two key conventions that govern every subsequent module chapter:

1. **Module classification** — the three classes of definable module (abstract modules, modules
   with scalar action, and modules with matrix action), each carrying a different level of
   structural support.

2. **Submodule presentation** — the two dual presentations for submodules (embedded vs.
   reduced), the naming convention that signals which presentation a creation function adopts
   (`QualifierSpace` for embedded, `QualifierModule` for reduced), and the rule that all
   descendant modules inherit the presentation chosen at creation time.

No intrinsics are defined in this chapter; it is entirely expository.

---

## 53.1 Overview

This section introduces the scope of the Modules part of the Handbook. The structures covered
include:

- Vector spaces
- Inner product spaces
- Modules defined over any ring or algebra
- R[G]-modules, where R is a ring and G is a group
- Linear transformations and R-module homomorphisms

Although vector spaces are subsumed under general modules, they receive separate treatment
both because of their importance and because their theory is cleaner than that of a general
module. Magma users unfamiliar with module theory are directed to Chapter 28 (vector spaces)
for a self-contained introduction.

Rectangular matrices are regarded in Magma as forming a module (a bimodule). A rectangular
m × n matrix over a ring R is considered an element of HomR(M, N), reflecting its dual
nature as both a module element and a homomorphism. The HomR(M, N) operations accordingly
include standard module-theoretic operations as well as operations that interpret the matrix
as a homomorphism.

*No intrinsics are defined in this section.*

---

## 53.2 General Modules

A module M is always regarded as a submodule or quotient module of the free module S(n) for
some ring or algebra S. The types of module definable in the system fall into three classes:

**(a) Abstract Modules.** Given a ring R, a set M, and a mapping φ : R × M → M, the pair
(M, φ) is an abstract R-module. Because of the very general nature of this construction, only
the basic arithmetic operations may be applied to modules of this type.

**(b) Modules with Scalar Action.** Given a general ring R, an R-module with scalar action is
a submodule or quotient module of the free R-module R(n), where the action is that of ring
multiplication in R.

**(c) Modules with Matrix Action.** Let R be a PIR and S an R-algebra, so there exists a ring
homomorphism φ : R → S making S a left R-module with action r ∗ s = φ(r) ∗ s. Any S-module M
is then also a left R-module via r ∗ m = φ(r) ∗ m; if φ(R) lies in the centre of S, then S
acts on M as a ring of R-module endomorphisms. Taking M to be the free R-module R(n), the
action of S on M is given by the action of a subring of Mn(R) on M. An S-module of this form
may be specified by giving M together with a homomorphism of S into Mn(R).

*No intrinsics are defined in this section.*

---

## 53.3 The Presentation of Submodules

Let N be a submodule of M = R(m). Assuming N is free of dimension n < m, there are two ways
to present N:

**(a) Embedded presentation.** N is viewed as a submodule embedded in M; its elements are
regarded as elements of M. This is the familiar presentation from elementary linear algebra.

**(b) Reduced presentation.** N is presented as R(n) represented on a reduced basis, together
with a morphism φ defining the inclusion of N into M.

The embedded presentation is familiar but inconvenient for advanced applications — for example,
many major functions for studying an R[G]-module N require N to be given relative to a reduced
basis. Magma supports both forms for the important classes of modules.

**Naming convention.** The two forms are signalled by the creation function name:

- Functions of the form `QualifierSpace` select the **embedded** form (e.g. `KSpace(K, n)`
  constructs the n-dimensional vector space over K with submodules in embedded form).
- Functions of the form `QualifierModule` select the **reduced** form (e.g. `RModule(K, n)`
  constructs the n-dimensional vector space over K with submodules in reduced form).

Once a choice has been made, all descendants of the initial module (submodules, quotient
modules, extensions) follow the same presentation convention.

*No intrinsics are defined in this section.*

---

## 53 Bibliography

This chapter contains no bibliography. It is a purely expository introduction; all references
for module and linear algebra algorithms appear in the subsequent chapters of Part VIII.

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| (none — introductory chapter only) | — |
