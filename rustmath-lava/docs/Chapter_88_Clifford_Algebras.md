# Chapter 88 — Clifford Algebras

**Handbook part:** XI — Algebras
**Handbook pages:** 2681–2682 (PDF pages 2814–2815)

---

## Scope and overview

Given a quadratic form Q defined on a vector space V over a field F, the Clifford algebra
of Q is an associative F-algebra C with a vector space homomorphism f : V → C such that
f(v)² = Q(v) for all v ∈ V. The triple (C, V, f) has the universal property that if A is
any associative algebra with a homomorphism g : V → A satisfying g(v)² = Q(v) for all
v ∈ V, then there is a unique algebra homomorphism h : C → A such that hf = g. The map
f is injective, so V may be identified with its image in C. If dim V = n then dim C = 2ⁿ.

Clifford algebras are represented in Magma as **structure constant algebras** (Chapter 79),
and all functions from that chapter apply. The Magma type is `AlgClff`. Every Clifford
algebra carries three attributes:

- `space` — the quadratic space from which the Clifford algebra is derived;
- `embedding` — the standard embedding of the quadratic space into the Clifford algebra;
- `mainInvolutionMatrix` — the matrix of the antiautomorphism that reverses multiplication.

Given a basis e₁, e₂, …, eₙ for V, a basis for C is the set of all products
e₁^{i₁} e₂^{i₂} ⋯ eₙ^{iₙ} where each iₖ ∈ {0, 1}. The function k ↦ iₖ is the
characteristic function of a subset S = {k | iₖ = 1} of {1, 2, …, n}, and the map
S ↦ 1 + Σ_{k ∈ S} 2^{k−1} is a bijection from the subsets of {1, …, n} to the integers
in [1 … 2ⁿ]. Elements of C are thus represented as sequences of pairs ⟨S, a⟩ where S
is a subset of {1, …, n} and a is a field element. Multiplication is determined by:

- v² = Q(v) · 1 for all v ∈ V, and
- uv + vu = β(u, v) · 1 for all u, v ∈ V,

where β is the polar form of Q.

The primary references are **[Che97]** and **[Art57]**.

---

## 88.2 Clifford Algebras and their Elements

### Construction

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CliffordAlgebra(Q)` | Given a quadratic form Q, returns a triple C, V, f where C is the Clifford algebra of Q, V is the quadratic space of Q, and f is the standard embedding of V into C. | Structure constant algebra construction; theory from **[Che97]**, **[Art57]**. |
| `CliffordAlgebra(V)` | Given a quadratic space V with quadratic form Q, returns the pair C, f where C is the Clifford algebra of Q and f is the standard embedding of V into C. | As above. |

### 88.2.1 Elements of a Clifford Algebra

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `elt< C \| r1, r2, ..., rm >` | Given a Clifford algebra C of dimension m = 2ⁿ over a field F and field elements r1, r2, …, rm ∈ F, constructs the element r1 ∗ C.1 + r2 ∗ C.2 + ⋯ + rm ∗ C.m of C. | Linear combination of basis elements. |
| `C ! L` | Given a Clifford algebra C of dimension m = 2ⁿ and a sequence L = [r1, r2, …, rm] of elements of the base ring R of C, constructs the element r1 ∗ C.1 + r2 ∗ C.2 + ⋯ + rm ∗ C.m of C. | Coercion from sequence of coefficients. |
| `BasisProduct(A, i, j)` | Returns the product of the i-th and j-th basis elements of the Clifford algebra C. | Direct multiplication in the structure constant algebra. |
| `BasisElement(C, L)` | Returns the basis element C.j of the Clifford algebra C corresponding to the subset L of {1, 2, …, n}, where j = 1 + Σ_{k ∈ L} 2^{k−1}. If e1, e2, …, eₙ is the standard basis for the underlying vector space, this corresponds to the product e_{i1} ∗ e_{i2} ∗ ⋯ ∗ e_{ih}, where L = {i1, i2, …, ih} and i1 < i2 < ⋯ < ih. | Bijection between subsets and basis index; theory from **[Che97]**. |

---

## 88.3 Bibliography

| Key | Reference |
|-----|-----------|
| **[Art57]** | E. Artin. *Geometric Algebra.* Interscience Publishers, New York, 1957. |
| **[Che97]** | Claude Chevalley. *The algebraic theory of spinors and Clifford algebras.* Springer-Verlag, Berlin, 1997. Collected works, Vol. 2. Edited and with a foreword by Pierre Cartier and Catherine Chevalley, with a postface by J.-P. Bourguignon. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Structure constant algebra construction **[Che97, Art57]** | `CliffordAlgebra(Q)`, `CliffordAlgebra(V)` |
| Linear combination / coercion of basis elements | `elt< >`, `C ! L` |
| Basis multiplication (polar form / multiplication rule) | `BasisProduct` |
| Subset-to-basis-index bijection **[Che97]** | `BasisElement` |
