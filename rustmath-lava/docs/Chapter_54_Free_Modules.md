# Chapter 54 — Free Modules

**Handbook part:** VIII — Modules
**Handbook pages:** 1397–1418 (PDF pages 1528–1551)

---

## Scope and overview

Chapter 54 describes Magma's facilities for free modules in two principal representations:

1. **Tuple modules** — modules whose elements are n-tuples over a fixed ring R, i.e. R(n). The ring R acts on the right by scalar multiplication.
2. **Matrix modules** — modules whose elements are homomorphisms of modules, i.e. HomR(M, N), represented as m × n matrices over R.

If R is not a Euclidean Domain, only arithmetic with vectors is supported. In particular, submodule and quotient-module operations are restricted to cases where R is a field or Euclidean Domain. Modules over fields (vector spaces) have extensive additional functionality described in the separate vector spaces chapter.

Two submodule presentation conventions are available:

- **Embedded presentation** (`RSpace`): submodules are presented in terms of a generating set of elements of the ambient module.
- **Reduced presentation** (`RModule`): submodules are presented as a module S(r) in terms of a reduced basis.

The category name for finitely generated modules is `ModRng`, with subcategories `ModTupFld`, `ModMatFld`, `ModTupEd`, `ModTupRng`, and `ModMatRng`.

---

## 54.1 Introduction

### 54.1.1 Free Modules

The chapter covers two representations: tuple modules R(n) and matrix modules HomR(M, N). Elementary module-theoretic operations are the same for both; the difference is only in input and display of elements. Special operations for matrices are described in the matrices chapter.

### 54.1.2 Module Categories

| Category | Description |
|----------|-------------|
| `ModRng` | Family of all finitely generated modules over any ring R (parent category) |
| `ModTupFld` | Modules of n-tuples over a field |
| `ModMatFld` | Modules of m × n matrices over a field |
| `ModTupEd` | Modules of n-tuples over a Euclidean domain |
| `ModTupRng` | Modules of n-tuples over a ring |
| `ModMatRng` | Modules of m × n matrices over a ring |

### 54.1.3 Presentation of Submodules

Let N be a free submodule of rank r of the R-module M. Two presentations are available: an **embedded presentation** (N given by generators that are elements of M) and a **reduced presentation** (N presented as S(r) with action induced from M). The choice is made at module creation time via `RModule` (reduced) or `RSpace` (embedded).

### 54.1.4 Notation

Throughout the chapter, R denotes a ring (possibly a field) and K denotes a field. M and N denote modules; U and V denote vector spaces.

---

## 54.2 Definition of a Module

### 54.2.1 Construction of Modules of n-tuples

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RSpace(R, n)` | Free right R-module R(n) of all n-tuples over ring R, with standard basis e1, …, en. Created in **embedded** mode. | — |
| `RModule(R, n)` | Free right R-module R(n) of all n-tuples over ring R, with standard basis e1, …, en. Created in **reduced** mode. | — |
| `RSpace(R, n, F)` | Free right R-module R(n) in embedded form, with inner product matrix F (a symmetric n × n matrix). `Norm` and `InnerProduct` are computed with respect to F. | — |

*Worked example: H54E1 (constructing a module of 6-tuples over the integers with `RModule`).*

### 54.2.2 Construction of Modules of m × n Matrices

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RMatrixSpace(R, m, n)` | The module comprising all m × n matrices over the ring R. | — |

### 54.2.3 Construction of a Module with Specified Basis

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RModuleWithBasis(Q)` | Given a sequence Q of k independent vectors each lying in a module M, construct the submodule of M of dimension k with basis Q. The basis is echelonized internally; functions depending on the basis (e.g. `Coordinates`) use the given basis. | — |
| `RSpaceWithBasis(Q)` | As `RModuleWithBasis(Q)` but in embedded mode. | — |
| `RSpaceWithBasis(a)` | Given a matrix a, construct the submodule of M with basis equal to the rows of a. Basis echelonized internally. | — |
| `RMatrixSpaceWithBasis(Q)` | The module of m × n matrices whose basis is given by the linearly independent matrices of the sequence Q. | — |

---

## 54.3 Accessing Module Information

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `M . i` | The i-th generator of the R-module M. The integer i must lie in [1, r] where r is the number of generators. | — |
| `CoefficientRing(M)` / `BaseRing(M)` | Given an R-module M defined as a submodule of S(n), return the ring S. | — |
| `CoefficientField(M)` / `BaseField(M)` | As above, when S is a field. | — |
| `Generators(M)` | The generators for the R-module M, returned as a set. | — |
| `OverDimension(M)` | Given an embedded submodule M of S(n), return n (the ambient dimension). | — |
| `OverDimension(u)` | Given an element u of an embedded submodule of S(n), return n. | — |
| `Moduli(M)` | The column moduli of the module M over a Euclidean domain. | — |
| `Parent(u)` | Given an element u belonging to the R-module M, return M. | — |
| `Generic(M)` | Given an R-module M which is a submodule of R(n), return the ambient module R(n). | — |

---

## 54.4 Standard Constructions

Given one or more existing modules, various standard constructions produce new modules.

### 54.4.1 Changing the Coefficient Ring

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ChangeRing(M, S)` | Given module M with base ring R and ring S, construct the module N with base ring S by coercing the components of elements of M into S. Returns N and the homomorphism M → N. | — |
| `ChangeRing(M, S, f)` | As above, but using a specified homomorphism f : R → S to map components. Returns N and the homomorphism M → N. | — |
| `ChangeUniverse(~x, R)` | Change the coefficient ring of x to be R (in-place, destructive). | — |

### 54.4.2 Direct Sums

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DirectSum(M, N)` | Direct sum D of R-modules M and N. Returns D, the embedding maps from M and N into D, and the projection maps from D onto M and N. | — |
| `DirectSum(Q)` | Direct sum D of a sequence Q of R-modules. Returns D, embedding maps from each module into D, and projection maps from D onto each module. | — |

---

## 54.5 Elements

Elements of a tuple module R(n) are n-tuples over R. Elements of a matrix module HomR(M, N) are m × n matrices over R. All operations defined for tuple module elements also apply to matrix module elements.

---

## 54.6 Construction of Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `elt< M \| a1, ..., an >` | Construct the element (a1, …, an) of module M with base module S(n). Error if the result is not an element of M. | — |
| `M ! Q` | Construct the element (a1, …, an) of M from a sequence Q = [a1, …, an] of elements of S. Error if result is not in M. | — |
| `CharacteristicVector(M, S)` | Given a submodule M of R(n) and a set S of integers in [1, n], return the characteristic vector of S as a vector of R. | — |
| `Zero(M)` / `M ! 0` | The zero element of the R-module M. | — |
| `Random(M)` | A random vector from module M, where M is defined over a finite ring or field. | — |

*Worked example: H54E2 (creating elements of the module of 4-tuples over Z[x]).*

### 54.6.1 Deconstruction of Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ElementToSequence(u)` / `Eltseq(u)` | Given an element u of R-module M, return u as a sequence Q of elements of R. For u ∈ R(n): Q[i] = u[i], 1 ≤ i ≤ n. For u ∈ R(m×n): Q[(i−1)n+j] = u[i,j]. | — |

### 54.6.2 Operations on Module Elements

#### Arithmetic

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `u + v` | Sum of elements u and v in the same R-module M. | — |
| `-u` | Additive inverse of element u. | — |
| `u - v` | Difference of elements u and v in the same R-module M. | — |
| `x * u` | Left scalar product: x ∈ R, u in a left R-module M; returns x ∗ u ∈ M. | — |
| `u * x` | Right scalar product: x ∈ R, u in a right R-module M; returns u ∗ x ∈ M. | — |
| `u / x` | Scalar quotient: x a non-zero element of field K, u in right K-module M; returns u ∗ (1/x) ∈ M. | — |

#### Indexing

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `u[i]` | The i-th component of element u in submodule M of R(n), 1 ≤ i ≤ n. Returns an element of R. | — |
| `u[i] := x` | Redefine the i-th component of u to be x ∈ R. The parent of u is changed to the generic module R(n) since the modified element need not lie in M. | — |

#### Normalization

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Normalize(u)` / `Normalise(u)` | Normalize the non-zero vector u. If R is a field: returns (1/a) ∗ u where a is the first non-zero component. If R = Z: returns ε ∗ u where ε = +1 if the first non-zero component is positive, −1 otherwise. If R = K[x] (K a field): returns (1/a) ∗ u where a is the leading coefficient of the first non-zero polynomial component. Returns the zero vector unchanged. | — |
| `Rotate(u, k)` | Return the vector obtained from u by rotating by k coordinate positions. | — |
| `Rotate(~u, k)` | Destructively rotate u by k coordinate positions (in-place). | — |

*Worked example: H54E3 (arithmetic operators on 4-tuples over Z[x], including `ElementToSequence` and `Support`).*

### 54.6.3 Properties of Vectors

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsZero(u)` | Returns `true` if element u of R-module M is the zero element. | — |
| `Depth(v)` | The index of the first non-zero entry of vector v; returns 0 if no such entry exists. | — |
| `Support(u)` | A set of integers giving the positions of the non-zero components of vector u. | — |
| `Weight(u)` | The number of non-zero components of vector u. | — |

### 54.6.4 Inner Products

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `(u, v)` / `InnerProduct(u, v)` | Inner product of vectors u and v with respect to the inner product defined on the space. If an inner product matrix F was given at creation, this is u · F · v^tr; otherwise it is u · v^tr. | — |
| `Norm(u)` | Norm of vector u with respect to the inner product. If an inner product matrix F was given, this is u · F · u^tr; otherwise u · u^tr. | — |

---

## 54.7 Bases

Basis functions are restricted to vector spaces or torsion-free modules over a Euclidean Domain. For modules over a field, the reader is referred to the vector spaces chapter.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Basis(M)` | The current basis for free R-module M (R an ED), returned as a sequence of module elements. | — |
| `Rank(M)` | The rank of the free R-module M. | — |
| `Coordinates(M, u)` | Given vector u in the rank-r free R-module M (R an ED) with basis u1, …, ur, return a sequence [a1, …, ar] such that u = a1 ∗ u1 + … + ar ∗ ur. | — |

---

## 54.8 Submodules

Submodules may be defined for any type of module. However, functions that depend on membership testing are only implemented for modules over Euclidean Domains.

### 54.8.1 Construction of Submodules

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `sub< M \| L >` | Construct the submodule N of R-module M generated by elements specified by list L. Each term Li may be: (a) a sequence of n ring elements defining an element of M; (b) a set or sequence of elements of M; (c) a submodule of M; or (d) a set or sequence of submodules of M. Returns N and the inclusion homomorphism f : N → M. Zero elements and repetitions are removed (unless N is trivial). | — |

*Worked example: H54E4 (submodule of the 4-dimensional vector space over the rational function field F5(x)).*

### 54.8.2 Operations on Submodules

The following operations are only available for submodules of R(n), HomR(M, N), and R[G] where R is a Euclidean Domain. For R[G]-modules the operators refer to the underlying R-module.

### 54.8.3 Membership and Equality

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `u in M` | Returns `true` if element u lies in R-module M (both from the same ambient module). | — |
| `u notin M` | Returns `true` if element u does not lie in R-module M. | — |
| `N subset M` | Returns `true` if R-module N is contained in R-module M (common ambient module). | — |
| `N notsubset M` | Returns `true` if R-module N is not contained in R-module M. | — |
| `M eq N` | Returns `true` if R-modules M and N are equal (common ambient module). | — |
| `M ne N` | Returns `true` if R-modules M and N are not equal. | — |

### 54.8.4 Operations on Submodules

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `M + N` | Sum of submodules M and N (both from a common R-module). | — |
| `M meet N` | Intersection of submodules M and N (both from a common R-module). | — |

---

## 54.9 Quotient Modules

### 54.9.1 Construction of Quotient Modules

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `quo< M \| L >` | Given R-module M, construct the quotient module P = M/N, where N is the submodule generated by elements specified by list L. Each term Li may be: (a) a sequence of n ring elements defining an element of M; (b) a set or sequence of elements of M; (c) a submodule of M; or (d) a set or sequence of submodules of M. Returns the quotient module P and the natural homomorphism f : M → P. | — |

---

## 54.10 Homomorphisms

Throughout this section R is a commutative ring. M and N are assumed to be free R-modules with bases present. HomR(M, N) is identified with the module of m × n matrices over R; an element of HomR(M, N) is represented as a matrix relative to the bases of the generic modules for M and N. These modules are called **matrix modules**.

Submodules of HomR(M, N) are always presented in embedded form. To obtain reduced-form presentation, use the natural isomorphism between R(m×n) and R(mn).

All operations defined for tuple modules, their elements, and submodules also apply to matrix modules.

### 54.10.1 HomR(M, N) for R-modules

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Hom(M, N)` | If M = R(m) and N = R(n), create HomR(M, N) as the (R, R)-bimodule R(m×n) (all m × n matrices over R) with standard basis {Eij}. | — |
| `RMatrixSpace(R, m, n)` | Given ring R and positive integers m, n, construct Hom(M, N) where M = R(m) and N = R(n), as the free (R, R)-bimodule R(m×n) with standard basis {Eij}. The modules M and N are created by this function and accessible as `Domain(H)` and `Codomain(H)`. | — |

*Worked example: H54E5 (constructing the module of homomorphisms from a 3-dimensional to a 4-dimensional space over GF(2)).*

### 54.10.2 HomR(M, N) for Matrix Modules

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Hom(M, N, "right")` | Given matrix module M (elements are a × b matrices, domain D, codomain C) and matrix module N (elements are a × c matrices, domain D, codomain C'), construct H = Hom(M, N) with **right** multiplication action. Elements of H are b × c matrices over R that multiply an element of M on the right to yield an element of N. If M and N are proper submodules, the correct basis of H is explicitly constructed. | — |
| `Hom(M, N, "left")` | Given matrix module M (elements are a × c matrices, domain D, codomain C) and matrix module N (elements are b × c matrices, domain D', codomain C), construct H = Hom(M, N) with **left** multiplication action. Elements of H are b × a matrices over R that multiply an element of M on the left to yield an element of N. | — |

*Worked examples: H54E6 (constructing Hom(H1, H2, "right") over Q, then "left" action, with `Image` and `Kernel`).*

### 54.10.3 Modules HomR(M, N) with Given Basis

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RMatrixSpaceWithBasis(Q)` | Given a sequence Q of k independent matrices in a matrix space H = Hom(M, N) (M = R(m), N = R(n)), construct the subspace of H of dimension k with basis Q. Basis is echelonized internally; functions depending on the basis (e.g. `Coordinates`) use the given Q. | — |
| `KMatrixSpaceWithBasis(Q)` | As `RMatrixSpaceWithBasis(Q)` but for K a field (M = K(m), N = K(n)). | — |

### 54.10.4 The Endomorphism Ring

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `EndomorphismAlgebra(M)` | If M is the free R-module R(m), create the matrix algebra Matm(R) with standard basis {Eij \| i = 1…m, j = 1…m}. | — |

*Worked example: H54E7 (endomorphism ring of the 4-dimensional rational vector space).*

### 54.10.5 The Reduced Form of a Matrix Module

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Reduce(H)` | For a matrix module H with domain A (degree a, dimension d) and codomain B (degree b, dimension e): construct the **reduced module** H' whose elements are represented with respect to the actual bases of A and B (not the generic embedded bases). Returns H' and the epimorphism f : H → H'. The kernel of f consists of all matrices in H that map all elements of A to the zero element of B. Also handles the case where A and B are themselves matrix modules (right or left action). If A and B are already in reduced form, H' = H. | — |

*Worked examples: H54E8 (Reduce for a homomorphism module from one vector space to another); H54E9 (Reduce for a homomorphism module from one matrix module to another — note the reduced module has the same dimension but larger degrees).*

### 54.10.6 Construction of a Matrix

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `M ! Q` | Given matrix bimodule M over R and sequence Q = [a11, …, a1n, a21, …, amn] of elements of R, construct the m × n matrix with entries aij as an element of M. In `sub` or `quo` constructors the coercion clause `M !` may be omitted. | — |

*Worked example: H54E10 (creating the 4 × 4 Hilbert matrix as an element of the endomorphism ring of the 4-dimensional rational vector space).*

### 54.10.7 Element Operations

All operations for tuple module elements also apply to matrix module elements. Operations specific to matrix modules (where a ∈ HomR(M, N), R a Euclidean Domain) are:

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `u * a` / `a(u)` | Given element u of module M, return the image of u under homomorphism a as an element of module N. | — |
| `a * b` | Given homomorphism a ∈ Hom(M, N) and b ∈ Hom(N, P), return the composition a ∘ b ∈ Hom(M, P). If Hom(M, P) does not exist it is created. | — |
| `a ^ -1` | Given bijective homomorphism a ∈ Hom(M, N) (M and N of equal dimension), return the inverse of a as an element of Hom(N, M). | — |
| `Codomain(S)` | Given submodule S of Hom(M, N), return the module N. | — |
| `Codomain(a)` | The codomain N of homomorphism a ∈ Hom(M, N). | — |
| `Cokernel(a)` | The cokernel of homomorphism a ∈ Hom(M, N). | — |
| `Domain(S)` | The domain M of submodule S of Hom(M, N). | — |
| `Domain(a)` | The domain M of homomorphism a ∈ Hom(M, N). | — |
| `Image(a)` | The image of homomorphism a ∈ Hom(M, N), returned as a submodule of N. For matrix module domains/codomains, image is with respect to the appropriate action (right or left). | — |
| `Kernel(a)` / `NullSpace(a)` | The kernel of homomorphism a ∈ Hom(M, N), returned as a submodule of M. For matrix module domains/codomains, kernel is with respect to the appropriate action. | — |
| `Morphism(M, N)` | Assuming R-module M was created as a submodule of N, return the matrix defining the inclusion homomorphism φ : M → N as an element of HomR(M, N). Gives the correspondence between elements of M (relative to M's standard basis) and elements of N. | — |
| `Rank(a)` | The dimension of the image of homomorphism a (i.e. the rank of a). | — |
| `IsBijective(a)` | Returns `true` if homomorphism a ∈ Hom(M, N) is a bijective mapping. | — |
| `IsInjective(a)` | Returns `true` if homomorphism a ∈ Hom(M, N) is an injective mapping. | — |
| `IsSurjective(a)` | Returns `true` if homomorphism a ∈ HomR(M, N) is a surjective mapping. | — |

*Worked example: H54E11 (element operations on Hom(V4, V3) over GF(8): constructing a matrix, computing Rank, Image, Kernel, Cokernel).*

---

## 54.11 Bibliography

No bibliography is provided at the end of Chapter 54. The chapter describes elementary linear-algebraic operations over rings; no external algorithmic references are cited in the text.

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| n-tuple free module construction (embedded / reduced) | `RSpace`, `RModule`, `RSpaceWithBasis`, `RModuleWithBasis` |
| Matrix module / HomR construction | `RMatrixSpace`, `Hom`, `RMatrixSpaceWithBasis`, `KMatrixSpaceWithBasis` |
| Direct sum | `DirectSum` |
| Coefficient ring change | `ChangeRing`, `ChangeUniverse` |
| Submodule construction and membership (over EDs) | `sub< >`, `in`, `notin`, `subset`, `notsubset`, `eq`, `ne`, `+`, `meet` |
| Quotient module construction | `quo< >` |
| Basis and coordinates (over EDs / fields) | `Basis`, `Rank`, `Coordinates` |
| Normalization | `Normalize` / `Normalise` |
| Inner product and norm | `InnerProduct`, `Norm` |
| Reduction to actual-basis representation | `Reduce` |
| Endomorphism algebra | `EndomorphismAlgebra` |
| Homomorphism image / kernel / cokernel | `Image`, `Kernel` / `NullSpace`, `Cokernel`, `Rank(a)`, `Morphism` |
| Injectivity / surjectivity / bijectivity tests | `IsInjective`, `IsSurjective`, `IsBijective` |
| Vector properties | `IsZero`, `Depth`, `Support`, `Weight` |
