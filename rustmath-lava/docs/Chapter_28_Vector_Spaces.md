# Chapter 28 — Vector Spaces

**Handbook part:** IV — Matrices and Linear Algebra
**Handbook pages:** 585–606 (PDF pages 716–739)

---

## Scope and overview

Chapter 28 covers K-vector spaces and their linear transformations as implemented in Magma.
The standard K-vector space is the set of n-tuples over a field K, written K(n) (a *tuple
space*). Rectangular matrices over K are elements of the vector space of all m × n matrices
over K, written K(m×n) (a *matrix space*). Both kinds are collectively called *vector spaces*
or *K-vector spaces*; the distinction is drawn only when necessary.

The set of all linear transformations from a K-vector space U of dimension m to a K-vector
space V of dimension n is denoted Hom_K(U, V), and once bases are chosen is identified with
K(m×n). Thus K(m×n) is first a vector space (all normal vector-space operations apply) and
also a set of mappings (additional operations arising from this characterisation apply).

Every vector space V defined over a field K is created either as a subspace of the row space
K(n) (tuple spaces) or as a subspace of K(m×n) (matrix modules). Constructing a general
vector space is therefore a two-step process: first create the appropriate ambient space K(n)
or K(m×n), then define the required subspace or quotient space within it.

Subspace presentation depends on the constructor used: `VectorSpace`/`KSpace` give subspaces
in *embedded* form (basis elements are elements of the ambient space); `KModule` gives
subspaces in *reduced* form. There is no reduced mode for matrix spaces; they always use
embedded form.

---

## 28.1 Introduction

### 28.1.1 Vector Space Categories

The family of all finite-dimensional vector spaces over a given field K forms a category.
Tuple-space categories carry the name `ModTupFld`; matrix-space categories carry the name
`ModMatFld`. Objects are vector spaces and morphisms are linear transformations.

### 28.1.2 The Construction of a Vector Space

Every vector space V over K is a subspace of K(n) or K(m×n). Construction is a two-step
process: (i) create the appropriate ambient row space; (ii) define the required V as a
subspace or quotient of that ambient space.

---

## 28.2 Creation of Vector Spaces and Arithmetic with Vectors

### 28.2.1 Construction of a Vector Space

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `VectorSpace(K, n)` / `KSpace(K, n)` | Create the n-dimensional vector space V = K(n) over field K, with respect to the standard basis e₁, …, eₙ. Subspaces of V are presented in embedded form. | — |
| `KModule(K, n)` | Create the n-dimensional vector space V = K(n) over field K with the standard basis. Subspaces of V are presented in reduced form. Otherwise identical to `KSpace`. | — |
| `KMatrixSpace(K, m, n)` | Create the vector space K(m×n) of all m × n matrices over K, with standard basis {E_ij}. Subspaces are always in embedded form (no reduced mode). | — |
| `Hom(V, W)` | If V = K(m) and W = K(n), create Hom_K(V, W) as the matrix space K(m×n) with standard basis {E_ij}. | — |

*Worked examples: H28E1 (6-tuple space over Q); H28E2 (3×5 matrix space over Q(√5)).*

### 28.2.2 Construction of a Vector Space with Inner Product Matrix

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `VectorSpace(K, n, F)` / `KSpace(K, n, F)` | Create the n-dimensional vector space K(n) (embedded form) with inner product matrix F (a square n×n symmetric matrix over K). The functions `Norm` and `InnerProduct` operate with respect to F instead of the standard dot product. | — |

### 28.2.3 Construction of a Vector

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `elt< V \| L >` | Construct a vector (or matrix) from the element list L as an element of V. For a subspace of K(n) the list supplies n elements of K; for a subspace of K(m×n) the list supplies mn elements of K. Errors if the result is not an element of V. | — |
| `V ! Q` | Coerce sequence/list Q into V. Same semantics as `elt<>`. Errors if the result is not an element of V. | — |
| `CharacteristicVector(V, S)` | Given a subspace V of K(n) and a set S of integers in [1, n], return the characteristic vector of S as an element of V (1 in positions in S, 0 elsewhere). | — |
| `V ! 0` / `Zero(V)` | The zero element of the vector space V. | — |
| `Random(V)` | A random vector in V (V must be defined over a finite field). | — |

*Worked examples: H28E3 (5-dimensional space over F₄, constructing u = (1, w, 1+w, 0, 0)); H28E4 (element of 3×4 matrix space over Q(w), w a root of x⁷ − 7x + 3).*

### 28.2.4 Deconstruction of a Vector

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ElementToSequence(u)` / `Eltseq(u)` | Given an element u of the K-vector space V, return u as a sequence Q of elements of K. If u ∈ K(n), then Q[i] = u[i] for 1 ≤ i ≤ n. | — |

### 28.2.5 Arithmetic with Vectors

Vectors u and v must belong to the same vector space (same tuple space K(n) or same matrix
space K(m×n)). The scalar a must belong to K.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `u + v` | Sum of vectors u and v in the same vector space. | — |
| `-u` | Additive inverse of the vector u. | — |
| `u - v` | Difference of vectors u and v in the same vector space. | — |
| `x * u` / `u * x` | Scalar product of vector u and field element x ∈ K. | — |
| `u / x` | Scalar product of u and 1/x (x non-zero). | — |
| `NumberOfColumns(u)` / `Ncols(u)` | The number of columns in the vector u. | — |
| `Depth(u)` | Index of the first non-zero entry of u (0 if none). | — |
| `(u, v)` / `InnerProduct(u, v)` | Inner product of u and v with respect to the inner product defined on the space. If an inner product matrix F was supplied at construction: u · F · vᵀ; otherwise u · vᵀ. | — |
| `IsZero(u)` | Returns true iff u is the zero element of the vector space. | — |
| `Norm(u)` | Norm of u with respect to the inner product on the space. If F was supplied: u · F · uᵀ; otherwise u · uᵀ. | — |
| `Normalise(u)` / `Normalize(u)` | For a non-zero u, return (1/a)·u where a is the first non-zero component of u, so the result has leading non-zero entry equal to 1. Returns u if u is the zero vector. | — |
| `Rotate(u, k)` | Return the vector obtained from u by rotating k coordinate positions. | — |
| `Rotate(~u, k)` | Destructively rotate u by k coordinate positions (procedure). | — |
| `NumberOfRows(u)` / `Nrows(u)` | Number of rows in u (always 1 for a tuple; included for completeness). | — |
| `Support(u)` | Set of integers giving the positions of non-zero components of u. | — |
| `TensorProduct(u, v)` | Tensor (Kronecker) product of vectors u and v. The resulting vector has degree equal to the product of the degrees of u and v. | — |
| `Trace(u, F)` / `Trace(u)` | Given u ∈ K(n) and a subfield F of K, return the vector obtained by replacing each component of u with its trace over F. If F is the prime field of K it may be omitted. | — |
| `Weight(u)` | The number of non-zero components of u. | — |

*Worked examples: H28E5 (arithmetic in 4-dimensional space over Q(w), w an 8th root of unity: +, -, scalar *, Normalize, InnerProduct, Support); H28E6 (non-trivial inner product defined via a symmetric matrix F; Norm and InnerProduct relative to F).*

### 28.2.6 Indexing Vectors and Matrices

Indexing behaviour depends on whether V is a tuple space or a matrix space.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `u[i]` | For a subspace of K(n): return the i-th component of u as an element of K. For a subspace of K(m×n): return the i-th row of the matrix u as an element of K(n). | — |
| `u[i, j]` | For a subspace of K(m×n): return the (i, j)-th component of u as an element of K. | — |
| `u[i] := x` | For a subspace of K(n): redefine the i-th component of u to be x ∈ K. For a subspace of K(m×n): redefine the i-th row of u to be the vector x ∈ K(n). | — |
| `u[i, j] := x` | For a subspace of K(m×n): redefine the (i, j)-th component of u to be x ∈ K. | — |

*Worked examples: H28E7 (indexing a 3-dimensional tuple space and a 2×3 matrix space over Q(w), w an 8th root of unity).*

---

## 28.3 Subspaces, Quotient Spaces and Homomorphisms

Presentation conventions: subspaces and quotient spaces of a V created with `VectorSpace` or
`MatrixSpace` are given in embedded form; those of a V created with `RModule` are given in
reduced form.

### 28.3.1 Construction of Subspaces

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `sub< V \| L >` | Construct the subspace U of K-vector space V generated by the elements specified by list L. Each term of L may be: a sequence of n elements of K; a set or sequence of elements of V; a subspace of V; or a set or sequence of subspaces of V. Returns U and the inclusion homomorphism f : U → V. Repetitions and zero vectors are removed (unless U is trivial). | — |
| `Morphism(U, V)` | Assuming U was created as a subspace of V, return the matrix defining the embedding of U into V. | — |

*Worked examples: H28E8 (ternary Golay code as a 6-dimensional subspace of F₃(11) via VectorSpace — embedded basis); H28E9 (same code via RModule — reduced basis; Morphism returns the 6×11 generator matrix).*

### 28.3.2 Construction of Quotient Vector Spaces

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `quo< V \| L >` | Construct the quotient vector space W = V/U where U is the subspace generated by the elements specified by list L (same term types as `sub<>`). Returns W and the natural homomorphism f : V → W. | — |
| `V / U` | Given a subspace U of V, construct the quotient space W of V by U. If r = dim(V) − dim(U), W is created as an r-dimensional space relative to the standard basis. Returns W and the natural homomorphism f : V → W. | — |

*Worked examples: H28E10 (quotient of F₃(11) by the Golay code: 5-dimensional result); H28E11 (same quotient as a complement subspace using `Complement`); H28E12 (subspace and quotient in Q(3×4) via `Hom`).*

---

## 28.4 Changing the Coefficient Field

The standard R-module field-change constructions from section 31.5 also apply to vector
spaces. In addition, the following functions extend or restrict the field of scalars.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ExtendField(V, L)` | Given a K-vector space V and an extension L of K, construct the L-vector space U = V ⊗_K L. Returns (a) the vector space U and (b) the inclusion homomorphism φ : V → U. | — |
| `RestrictField(V, L)` | Given a K-vector space V and a subfield L of K, construct the L-vector space U consisting of those vectors of V with all components in L. Returns (a) U and (b) the restriction homomorphism φ : V → U. | — |
| `VectorSpace(V, F)` / `KSpace(V, F)` / `KMatrixSpace(V, F)` / `KModule(V, F)` | Given an n-dimensional K-vector space V and a subfield F of a finite field or cyclotomic field K such that [K : F] = m, construct a vector space U of dimension mn over F. Returns (a) U and (b) a mapping φ : V → U sending (v₁, …, vₙ) to (u₁₁, …, u₁ₙ, …, uₙ₁, …, uₙₙ), where (uᵢ₁, …, uᵢₘ) is vᵢ written as a vector over F. | — |

---

## 28.5 Basic Operations

### 28.5.1 Accessing Vector Space Invariants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `V . i` | The i-th generating element of vector space V (positive integer i). | — |
| `CoefficientField(V)` / `BaseField(V)` | The coefficient field K of the K-vector space V. | — |
| `Degree(V)` | For V a subspace of K(n), return n (the over-dimension / ambient degree). | — |
| `Degree(u)` | For u an element of a subspace of K(n), return n. | — |
| `Dimension(V)` | The dimension of the vector space V. | — |
| `Generators(V)` | The generators of V, returned as a set. | — |
| `NumberOfGenerators(M)` / `Ngens(M)` | The number of generators of V. | — |
| `OverDimension(V)` | For V a subspace of K(n), return n (alias for `Degree(V)`). | — |
| `OverDimension(u)` | For u in a subspace of K(n), return n. | — |
| `Generic(V)` | The generic (full ambient) vector space containing V. | — |
| `Parent(V)` | The power structure for V (the set of all finite-dimensional vector spaces). | — |

### 28.5.2 Membership and Equality

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `v in V` | Returns true if element v lies in vector space V (v and V belong to a common space). | — |
| `v notin V` | Returns true if element v does not lie in vector space V. | — |
| `U subset V` | Returns true if K-vector space U is contained in K-vector space V (both subspaces of a common space). | — |
| `U notsubset V` | Returns true if U is not contained in V. | — |
| `U eq V` | Returns true if subspaces U and V are equal (both in a common vector space). | — |
| `U ne V` | Returns true if subspaces U and V are not equal. | — |

### 28.5.3 Operations on Subspaces

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `U + V` | Sum of subspaces U and V (both must be subspaces of a common vector space). | — |
| `U meet V` | Intersection of subspaces U and V (both must be subspaces of a common vector space). | — |
| `U meet:= V` | Replace U with the intersection of subspaces U and V. | — |
| `&meet S` | Intersection of the subspaces in a set or sequence S (all subspaces of a common vector space). | — |
| `TensorProduct(U, V)` | Tensor (Kronecker) product of vector spaces U and V, generated by all tensor products of elements of U by elements of V. The result has degree equal to the product of the degrees of U and V. | — |
| `Complement(V, U)` | Given a subspace U of vector space V, construct a complement for U in V (a subspace of V disjoint from U, with U + complement = V). | — |
| `Transversal(V, U)` | Given a subspace U of V over a finite field, return a transversal for U in V as a set of vectors (one representative from each coset). | — |

---

## 28.6 Reducing Vectors Relative to a Subspace

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ReduceVector(W, v)` | (Function.) Given a vector v from a tuple module V and a submodule W of V, return the reduction of v with respect to W — the canonical representative of the coset v + W. | — |
| `ReduceVector(W, ~v)` | (Procedure.) Given a vector v from a tuple module V and a submodule W of V, replace v in-place with its reduction with respect to W. | — |
| `DecomposeVector(U, v)` | Given a vector v from a tuple module V and a submodule U of V, return the unique u ∈ U and w in the complement to U in U + ⟨v⟩ such that v = u + w. | — |

---

## 28.7 Bases

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `VectorSpaceWithBasis(Q)` / `VectorSpaceWithBasis(a)` | Create a vector space having as basis the terms of sequence Q (or the rows of matrix a). | — |
| `KSpaceWithBasis(Q)` / `KSpaceWithBasis(a)` | Create a K-space having as basis the terms of sequence Q (or the rows of matrix a). | — |
| `KModuleWithBasis(Q)` | Create a K-module having as basis the terms of sequence Q. | — |
| `Basis(V)` | The current basis of vector space V, returned as a sequence of vectors. | — |
| `BasisElement(V, i)` | The i-th basis element of V. | — |
| `BasisMatrix(V)` | The current basis of V returned as the rows of a matrix in K(m×n), where m = dim(V) and n = overdim(V). | — |
| `Coordinates(V, v)` | For v in the r-dimensional K-vector space V with basis v₁, …, vᵣ, return the sequence [a₁, …, aᵣ] of elements of K such that v = a₁·v₁ + ⋯ + aᵣ·vᵣ. | — |
| `Dimension(V)` | The dimension of the vector space V (also listed here for completeness). | — |
| `ExtendBasis(Q, U)` | Given a sequence Q of r linearly independent vectors in U, extend Q to a basis for U. Returns a sequence T with T[i] = Q[i] for i = 1, …, r. | — |
| `ExtendBasis(U, V)` | Given an r-dimensional subspace U of V, return a basis for V as a sequence T of elements such that the first r elements correspond to the basis vectors for U. | — |
| `IsIndependent(S)` | Given a set S of elements of V, return true if the elements of S are linearly independent. | — |
| `IsIndependent(Q)` | Given a sequence Q of elements of V, return true if the terms of Q are linearly independent. | — |

*Worked examples: H28E13 (ternary Golay code G3 in F₃(11): Dimension, Basis, ExtendBasis extending G3 to a basis of V11, Complement, G3 + C3, G3 meet C3, Random, Coordinates, reconstruction from coordinates).*

---

## 28.8 Operations with Linear Transformations

Throughout this section V is a subspace of K(m), W is a subspace of K(n), and a is a linear
transformation belonging to Hom_K(V, W). Many additional functions from the general matrices
chapter (e.g. `EchelonForm`) also apply to such matrices.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `v * a` / `a(v)` | Given v ∈ V and a ∈ Hom_K(V, W), return the image of v under the linear transformation a as an element of W. | — |
| `a * b` | Given a ∈ K(m×n) and b ∈ K(n×p), form the product as an element of K(m×p). | — |
| `Domain(a)` | The domain of the linear transformation a ∈ Hom_K(V, W), returned as a subspace of V. | — |
| `Codomain(a)` | The codomain of a ∈ Hom_K(V, W), returned as a subspace of W. | — |
| `Image(a)` | The image of a ∈ Hom_K(V, W), returned as a subspace of W. | — |
| `Rank(a)` | The dimension of the image of a (the rank of the matrix a). | — |
| `Kernel(a)` / `NullSpace(a)` | The kernel of a ∈ Hom_K(V, W), returned as a subspace of V. | — |
| `Cokernel(a)` | The cokernel of a ∈ Hom_K(V, W). | — |

*Worked examples: H28E14 (linear map operations: construct H23 and H34, compute composition a*b via Hom, apply Domain, Codomain, Image, Kernel, Rank, EchelonForm).*

---

## 28.9 Bibliography

Chapter 28 contains no bibliography. The material presented is standard linear algebra
implemented directly in Magma without citation to external references.

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Standard K-vector space construction (tuple and matrix spaces) | `VectorSpace`, `KSpace`, `KModule`, `KMatrixSpace`, `Hom` |
| Vector space with custom inner product | `VectorSpace(K, n, F)`, `KSpace(K, n, F)`, `Norm`, `InnerProduct` |
| Subspace construction (embedded / reduced) | `sub<>`, `Morphism` |
| Quotient space construction | `quo<>`, `V / U` |
| Field extension / restriction | `ExtendField`, `RestrictField`, `VectorSpace(V, F)`, `KSpace(V, F)`, `KMatrixSpace(V, F)`, `KModule(V, F)` |
| Basis management | `VectorSpaceWithBasis`, `KSpaceWithBasis`, `KModuleWithBasis`, `Basis`, `BasisElement`, `BasisMatrix`, `Coordinates`, `ExtendBasis`, `IsIndependent` |
| Subspace operations (sum, intersection, complement, transversal, tensor) | `+`, `meet`, `meet:=`, `&meet`, `TensorProduct`, `Complement`, `Transversal` |
| Vector reduction relative to a subspace | `ReduceVector`, `DecomposeVector` |
| Linear transformations (image, kernel, rank, cokernel) | `v * a`, `a(v)`, `a * b`, `Domain`, `Codomain`, `Image`, `Rank`, `Kernel`, `NullSpace`, `Cokernel` |
