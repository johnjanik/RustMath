# Chapter 76 — Groups of Straight-line Programs

**Handbook part:** X — Finitely-Presented Groups
**Handbook pages:** 2379–2385 (PDF pages 2510–2519)

---

## Scope and overview

This chapter describes the category `GrpSLP` — straight-line program groups (SLP-groups). A **straight-line program (SLP)** is formally a sequence [s₁, s₂, …, sₙ] where each sᵢ is one of:

1. A generator of the SLP-group;
2. A product sⱼsₖ, j < i, k < i;
3. A power sⱼⁿ, j < i;
4. A conjugate sⱼˢᵏ, j < i, k < i.

Effectively, a straight-line program is a word in the generators stored as an **expression tree** rather than a flat list of generator-exponent pairs. The key advantage is that common subexpressions need only be evaluated once when applying a homomorphism, and powers or conjugates can be computed more efficiently in the target group than by a linear product. This can yield dramatic speed-ups: the worked examples show a word of length 52307 (as a free-group word) reduced to 85 SLP operations, with a 80× speedup in homomorphism evaluation.

SLP-groups exist primarily as an efficient vehicle for evaluating homomorphisms. Magma provides facilities for arithmetic on SLPs, adding important subexpressions as new (redundant) generators, constructing homomorphisms, and random element generation via a product-replacement accumulator.

---

## 76.1 Introduction

*(Covered in Scope and overview above.)*

---

## 76.2 Construction of an SLP-Group and its Elements

### 76.2.1 Structure Constructors

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SLPGroup(n)` | Construct the free group of straight-line programs on `n` generators (n a non-negative integer). The i-th generator is referenced as `F.i` for i = 1, …, n. | — |

*Worked examples: H76E1 (creating a 2-generator SLP-group, referencing `F.1` and `F.2`).*

### 76.2.2 Construction of an Element

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Identity(G)` / `Id(G)` / `G ! 1` | Construct the identity element of the SLP-group G — the straight-line program [] of length 0. | — |

---

## 76.3 Arithmetic with Elements

Elements of an SLP-group support the standard group arithmetic operators, each building a new expression tree.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `u * v` | Given SLPs u = [u₁, …, uₘ] and v = [v₁, …, vₙ] in the same SLP-group G, returns an SLP representing the product u·v. The formal SLP [u₁, …, uₘ, v₁, …, vₙ, uₘvₙ] suffices; in practice, shared subexpressions may shorten the result. | — |
| `u ^ m` | Given an integer m and an SLP u, returns an SLP representing the m-th power of u. | — |
| `u ^ v` | Given SLPs u and v, returns an SLP representing the conjugate of u by v. | — |
| `#u` | Returns the number of multiplication, power, or conjugate operations required to evaluate a homomorphism on u (i.e. the evaluation cost of the SLP). | — |

### 76.3.1 Accessing the Defining Generators and Relations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `G . i` | The i-th generator for the SLP-group G. | — |
| `Generators(G)` | A set containing the generators for G. | — |
| `NumberOfGenerators(G)` / `Ngens(G)` | The number of generators for G. | — |
| `Parent(u)` | The parent SLP-group G of the straight-line program u. | — |

---

## 76.4 Addition of Extra Generators

When a particular expression in the original generators is expected to appear frequently as a common subexpression in subsequent programs, Magma allows it to be promoted to generator status in a new, related SLP-group.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AddRedundantGenerators(G, Q)` | Given SLP-group G with n generators and a sequence Q of q elements, returns an SLP-group H on n + q generators. The identification G.i ↔ H.i (for i ≤ n) and Q[i] ↔ H.(n+i) is maintained, enabling coercion between G and H and simple definition of homomorphisms from H. | — |

---

## 76.5 Creating Homomorphisms

SLP-groups support an extended homomorphism constructor. Because evaluation is the primary use-case, the constructor requires the user to supply correct input and (by default) verifies the codomain. For evaluating individual words without constructing an explicit homomorphism, the `Evaluate` function uses the same efficient evaluation mechanism.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `hom< G -> H \| L : parameters >` | Returns the group homomorphism φ: G → H defined by the list L. L may contain: (i) elements of the codomain (positional, one per generator of G); (ii) generator-image pairs of the form `G.i -> x` or `<G.i, x>`; (iii) a homomorphism ψ from an SLP-group B to H (where G was built by adding redundant generators to B) — must appear first; unassigned images are then computed from ψ. Parameter: `CheckCodomain` (BoolElt, default `true`) — if `false`, generator images are assumed to lie in the codomain without verification. | — |
| `Evaluate(u, Q)` / `Evaluate(u, G)` | Evaluate the single SLP u by substituting elements of sequence Q as images of the generators of the parent of u (Q must have at least as many elements as the parent has generators). When the second argument is a group G, Q is taken as the sequence of generators of G. | — |
| `Evaluate(v, Q)` / `Evaluate(v, G)` | Evaluate all SLPs in the sequence v simultaneously using images Q (or the generators of G). Simultaneous evaluation is generally faster than repeated individual calls. | — |

*Worked examples: H76E2 (AddRedundantGenerators and homomorphism construction; x becomes a redundant generator reducing evaluation length from 75 to 1–2 operations, yielding ~1000× speedup).*

---

## 76.6 Operations on Elements

### 76.6.1 Equality and Comparison

Note that equality here is **syntactic** (identical expression trees), not semantic: two SLPs may evaluate to the same group element without being equal under `eq`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `u eq v` | Returns `true` if and only if the SLPs u and v are identical as expression trees (not necessarily evaluating to the same word). | — |
| `u ne v` | Returns `true` if and only if the SLPs u and v are not identical as expression trees. | — |

---

## 76.7 Set-Theoretic Operations

### 76.7.1 Membership and Equality

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `g in G` | Given an SLP g and SLP-group G, returns `true` if g is an element of G, `false` otherwise. | — |
| `g notin G` | Given an SLP g and SLP-group G, returns `true` if g is not an element of G, `false` otherwise. | — |
| `S subset G` | Given SLP-group G and a set S of elements belonging to a related group H, returns `true` if S is a subset of G. | — |
| `S notsubset G` | Given SLP-group G and a set S of elements belonging to a related group H, returns `true` if S is not a subset of G. | — |

### 76.7.2 Set Operations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RandomProcess(G)` | Create a process to generate random elements from SLP-group G. Based on the product-replacement algorithm of **[CLGM+95]**, modified with an accumulator. At all times, N elements are stored where N = max(Slots, Ngens(G)+1); initially these are the generators. An extra accumulator element (initially the identity) is also kept. Parameters: `Slots` (RngIntElt, default 10) — size of the slot pool; `Scramble` (RngIntElt, default 100) — number of product-replacement steps performed before the process is returned. Note: only suitable for finite groups (or when the homomorphic target is finite), since all elements produced are products of generators only, not their inverses. | Product-replacement algorithm **[CLGM+95]**. |
| `Random(P)` | Given a random element process P (created by `RandomProcess(G)`), produce a random element of G: chooses a slot, multiplies it into the accumulator, and replaces that slot by its product with another randomly chosen slot. Returns the new accumulator value. The expanded generating set stored with P is updated. | Product-replacement step **[CLGM+95]**. |
| `Rep(G)` | A representative element of G. | — |

*Worked examples: H76E3 (comparing SLP evaluation vs. free-group word evaluation in GeneralOrthogonalGroup(7,3): an SLP of length 85 versus a free-group word of length 52307; homomorphism via SLP takes 0.020s vs. 1.640s for the free-group route).*

### 76.7.3 Coercions Between Related Groups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `G ! g` | Given an element g belonging to an SLP-group H related to G, rewrite g as an element of G. | — |

---

## 76.8 Bibliography

| Key | Reference |
|-----|-----------|
| **[CLGM+95]** | Frank Celler, Charles R. Leedham-Green, Scott H. Murray, Alice C. Niemeyer, and E. A. O'Brien. Generating random elements of a finite group. *Comm. Algebra*, 23(13):4931–4948, 1995. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| SLP expression-tree arithmetic | `SLPGroup`, `*`, `^` (power), `^` (conjugate), `#` |
| Redundant-generator promotion | `AddRedundantGenerators` |
| SLP homomorphism evaluation | `hom< >` (with `CheckCodomain`), `Evaluate` |
| Product-replacement random generation **[CLGM+95]** | `RandomProcess`, `Random` |
| Coercion between related SLP-groups | `G ! g` |
