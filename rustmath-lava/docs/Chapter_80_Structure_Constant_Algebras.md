# Chapter 80 — Structure Constant Algebras

**Handbook part:** XI — Algebras
**Handbook pages:** 2433–2439 (PDF pages 2564–2573)

---

## Scope and overview

A structure constant algebra A of dimension n over a ring R is defined by giving the n³ structure
constants a^k_ij ∈ R (1 ≤ i, j, k ≤ n) such that, for the basis e₁, e₂, …, eₙ of A,
eᵢ ∗ eⱼ = Σₖ a^k_ij ∗ eₖ. Structure constant algebras may be defined over any unital ring R;
however, many operations require R to be a Euclidean domain or even a field.

Magma supports three formats for specifying structure constants: a full dense sequence of n³ ring
elements, products eᵢ ∗ eⱼ as elements of the underlying free module Rⁿ, or sparse quadruples
listing only the non-zero constants. The internal storage format (dense, sparse, or partial) is
selected by the optional parameter `Rep`.

The chapter covers construction of the algebra and its elements, basic structural tests
(commutativity, associativity, Jacobi/Lie condition), module-structure accessors, indexed
element access, homomorphism construction, and a worked example building the 8-dimensional
real Cayley algebra and exploring its connection to E₈ and the groups G₂(q).

---

## 80.1 Introduction

Structure constant algebras provide a fully general framework for defining finite-dimensional
algebras in Magma. No special structure (associativity, Lie identity, etc.) is assumed at
construction time; the algebra is a free R-module of rank n with a bilinear product determined
entirely by the n³ scalars a^k_ij. The chapter notes that many individual operations additionally
require R to be a Euclidean domain or a field.

---

## 80.2 Construction of Structure Constant Algebras and Elements

### 80.2.1 Construction of a Structure Constant Algebra

There are three ways to specify structure constants for an algebra of dimension n. The first gives
n³ ring elements (or n² vectors of length n, or n sequences of n sequences of length n) in dense
form. The second identifies A with the module M = Rⁿ and gives products eᵢ ∗ eⱼ as elements
of M. The third specifies only the non-zero structure constants as quadruples ⟨i, j, k, a^k_ij⟩.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Algebra<R, n \| Q : parameters>` | Creates the structure constant algebra A over Rⁿ with standard basis e₁, …, eₙ. The sequence Q may be: (i) n sequences of n sequences of length n (the j-th element of the i-th sequence gives the coefficients of eᵢ ∗ eⱼ); (ii) n² sequences of length n, or n² elements of M (position (i−1)∗n+j gives eᵢ ∗ eⱼ); or (iii) a flat sequence of n³ ring elements with ordering a¹₁₁, a²₁₁, …, aⁿₙₙ (element a^k_ij at position (i−1)∗n²+(j−1)∗n+k). Optional `Rep`: `"Dense"` (default), `"Sparse"`, or `"Partial"`. Dense stores n² vectors of length n; sparse stores positions and values of non-zero constants; partial stores the vectors but records positions of non-zero constants. | Direct specification of structure constants; no algorithmic attribution. |
| `Algebra<M \| Q : parameters>` | Creates the structure constant algebra over the free module M = Rⁿ; Q specifies the structure constants as above. `Rep` parameter as above (default `"Dense"`). | — |
| `Algebra<R, n \| T : parameters>` | Creates the structure constant algebra over R with non-zero structure constants given as a sequence T of quadruples ⟨i, j, k, a^k_ij⟩; all other structure constants are 0. `Rep` parameter (default `"Sparse"`). | — |
| `ChangeBasis(A, B)` | Creates a new structure constant algebra A′ isomorphic to A by recomputing structure constants with respect to basis B. B may be a set or sequence of elements of A, a set or sequence of vectors, or a matrix. Returns A′ and the isomorphism A → A′. `Rep` parameter selects representation of A′ (default `"Dense"`, regardless of A's representation). | — |

### 80.2.2 Construction of Elements of a Structure Constant Algebra

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `elt<A \| r1, r2, …, rn>` | Given a structure constant algebra A of dimension n over R and ring elements r₁, …, rₙ ∈ R, constructs the element r₁∗e₁ + r₂∗e₂ + … + rₙ∗eₙ of A. | — |
| `A ! Q` | Given A of dimension n and a sequence Q = [r₁, …, rₙ] of base-ring elements, constructs r₁∗e₁ + … + rₙ∗eₙ. | — |
| `BasisProduct(A, i, j)` | Returns the product of the i-th and j-th basis elements of A. | — |
| `BasisProducts(A)` | Returns the products of all basis elements of A. `Rep` parameter controls format: `"Dense"` (default) returns a sequence Q of n sequences of n elements of A with Q[i][j] = eᵢ ∗ eⱼ; `"Sparse"` returns quadruples (i, j, k, aᵢⱼₖ) encoding eᵢ ∗ eⱼ = Σₖ aᵢⱼₖ bₖ. | — |

---

## 80.3 Operations on Structure Constant Algebras and Elements

### 80.3.1 Operations on Structure Constant Algebras

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsCommutative(A)` | Returns `true` if algebra A is commutative, `false` otherwise. | Direct check of structure constants. |
| `IsAssociative(A)` | Returns `true` if algebra A is associative, `false` otherwise. Requires up to n³ tests for an algebra of dimension n. | Direct check of the associativity identity; up to n³ tests. |
| `IsLie(A)` | Returns `true` if algebra A is a Lie algebra, `false` otherwise. Requires about n³/3 tests of the Jacobi identity for an algebra of dimension n. | Jacobi identity tests; approximately n³/3 checks. |
| `DirectSum(A, B)` | Constructs a structure constant algebra of dimension n + m (where n = dim A and m = dim B). The basis is the concatenation of the bases of A and B; products a ∗ b for a ∈ A, b ∈ B are defined to be 0. | — |

*Worked example: H80E1 (defining a Jordan algebra from a 2×2 matrix algebra over GF(3); verifying commutativity, checking the Jordan identity (x²∗y)∗x = x²∗(y∗x) on all elements; inspecting BasisProducts).*

### 80.3.2 Indexing Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `a[i]` | For element a of a structure constant algebra A of dimension n and positive integer 1 ≤ i ≤ n, returns the i-th component of a as an element of the base ring R. | — |
| `a[i] := r` | For element a of a structure constant algebra of dimension n over R, positive integer 1 ≤ i ≤ n, and r ∈ R, redefines the i-th component of a to be r. | — |

### 80.3.3 The Module Structure of a Structure Constant Algebra

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Module(A)` | Returns the module Rⁿ underlying the structure constant algebra A. | — |
| `Degree(A)` | Returns the degree (= dimension) of the module underlying A. | — |
| `Degree(a)` | For an element belonging to a structure constant algebra A of dimension n, returns n. | — |
| `ElementToSequence(a)` / `Eltseq(a)` | Returns the sequence of coefficients of the structure constant algebra element a. | — |
| `Coordinates(S, a)` | For element a of a structure constant algebra A and subalgebra S of A containing a, returns the coefficients of a with respect to the basis of S. | — |
| `InnerProduct(a, b)` | Returns the Euclidean inner product of the coefficient vectors of a and b, where a and b are elements of some structure constant algebra A. | — |
| `Support(a)` | Returns the support of structure constant algebra element a: the set of indices of non-zero components of a. | — |

### 80.3.4 Homomorphisms

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `hom<A -> B \| Q>` | For a structure constant algebra A of dimension n over R and either a structure constant algebra B or module B over R, constructs the homomorphism A → B specified by Q. Q may be [b₁, …, bₙ] with bᵢ ∈ B (mapping the i-th basis element of A to bᵢ), or [⟨a₁,b₁⟩, …, ⟨aₙ,bₙ⟩] mapping aᵢ ↦ bᵢ where {aᵢ} must be a basis of A. Note: only module homomorphism structure is guaranteed; it is not verified whether the map is an algebra homomorphism. | — |

*Worked example: H80E2 (constructing the real Cayley algebra of dimension 8 over Q using the sparse quadruple notation and the quat helper function; verifying non-associativity of the full algebra and associativity of sub-algebras; computing MinimalPolynomial and inverses of representative units; connecting the 240 units (after rescaling by √2) to the E₈ root lattice; applying ChangeRing to GF(3) and GF(5) to obtain finite Cayley algebras and permutation representations of G₂(3) and G₂(5)).*

---

## 80.4 Bibliography

No bibliography is printed in Chapter 80 of the MAGMA Handbook. The chapter presents the
structure constant algebra framework as an infrastructure layer with no attributed external
algorithmic references.

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Structure constant algebra construction (dense/sparse/partial) | `Algebra<R,n\|Q>`, `Algebra<M\|Q>`, `Algebra<R,n\|T>`, `ChangeBasis` |
| Element construction | `elt<A\|…>`, `A!Q`, `BasisProduct`, `BasisProducts` |
| Structural identity tests (commutativity, associativity, Jacobi) | `IsCommutative`, `IsAssociative`, `IsLie` |
| Module structure accessors | `Module`, `Degree`, `ElementToSequence`/`Eltseq`, `Coordinates`, `InnerProduct`, `Support` |
| Direct sum construction | `DirectSum` |
| Homomorphism construction | `hom<A->B\|Q>` |
