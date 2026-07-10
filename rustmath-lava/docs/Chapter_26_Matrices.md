# Chapter 26 — Matrices

**Handbook part:** IV — Matrices and Linear Algebra
**Handbook pages:** 521–555 (PDF pages 650–689)

---

## Scope and overview

This chapter describes all the basic operations available for creating and working with matrices in Magma. Matrices arise in many different contexts; several matrix types exist within Magma but most operations listed here apply to all of them.

The parent of any matrix is one of several matrix-structure types (module, matrix algebra, matrix group, etc.); those structure types are documented in other chapters, together with operations peculiar to their elements. There is also a virtual type `Mtrx` from which all matrix types inherit — package intrinsics that accept any matrix type should declare their argument as `Mtrx`.

Key algorithmic capabilities include: a fast p-adic algorithm for nullspaces of matrices over Z and Q (with sparse-matrix optimisations); a modular determinant algorithm based on Abbott–Bronstein–Mulders **[ABM99]**; classical and fast modular Hermite normal form (the modular variant by Allan Steel, available since V2.13); Smith normal form using sparse preprocessing **[HHR93]** followed by iterated Hermite reduction or the modular algorithm of Lübeck **[Lüb02]**; canonical forms over fields based on the single unified algorithm of Steel **[Ste97]**; and order computation for matrices over finite fields using the Cunningham database **[CLG97]**.

---

## 26.1 Introduction

Introductory section; no intrinsics. See scope notes above.

---

## 26.2 Creation of Matrices

Elementary constructs for creating a matrix or vector. The parent of the result is determined as follows: (a) vectors → appropriate R-space (`ModTupRng` or `ModTupFld`); (b) square matrices → appropriate matrix algebra (`AlgMatElt`); (c) non-square matrices → appropriate R-matrix space (`ModMatRng` or `ModMatFld`). Matrices and vectors may also be created by coercing a sequence of ring elements into the appropriate parent structure.

### 26.2.1 General Matrix Construction

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Matrix(R, m, n, Q)` | Given ring `R`, integers `m, n ≥ 0`, and sequence `Q`, return the `m × n` matrix over `R` with entries from `Q` (coerced into `R`) in row-major order. `Q` may be: (a) a flat sequence of length `mn`; (b) a sequence of sparse tuples `<i,j,x>`; (c) a sequence of `m` sequences each of length `n`; or (d) a sequence of `m` vectors each of length `n`. Either of `m` or `n` may be 0, in which case `Q` must have length 0. | Direct construction. |

*Worked examples: H26E1 (creating 2×2 matrix over Z; 2×3 matrix over GF(23); sparse 5×10 matrix over Q; sparse 10×10 matrix over GF(101)).*

### 26.2.2 Shortcuts

Shortcut versions of the general creation function where some arguments are inferred by Magma.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Matrix(m, n, Q)` | Given integers `m, n ≥ 0` and a flat sequence `Q` of length `mn` over ring `R`, return the `m × n` matrix over `R` in row-major order. | — |
| `Matrix(m, n, Q)` | Given integers `m` and `n` and a sequence `Q` of `m` sequences each of length `n` over ring `R`, return the `m × n` matrix over `R`. | — |
| `Matrix(Q)` | Given a sequence `Q` of `m` vectors each of length `n` over ring `R`, return the `m × n` matrix over `R`. | — |
| `Matrix(R, n, Q)` | Given ring `R`, integer `n ≥ 0`, and flat sequence `Q` of length `l` where `n` divides `l`, return the `(l/n) × n` matrix over `R` (entries coerced). | — |
| `Matrix(n, Q)` | Given integer `n ≥ 0` and flat sequence `Q` of length `l` where `n` divides `l` over ring `R`, return the `(l/n) × n` matrix over `R`. | — |
| `Matrix(Q)` | Given a sequence `Q` of `m` sequences each of length `n` over ring `R`, return the `m × n` matrix over `R`. | — |
| `Matrix(R, Q)` | Given a sequence `Q` of `m` sequences each of length `n` over ring `S` and ring `R`, return the `m × n` matrix over `R` (entries coerced). | — |

*Worked examples: H26E2 (alternative creation of matrices from H26E1 using shortcut forms).*

### 26.2.3 Construction of Structured Matrices

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ZeroMatrix(R, m, n)` | Given ring `R` and integers `m, n ≥ 0`, return the `m × n` zero matrix over `R`. | — |
| `ScalarMatrix(n, s)` | Given integer `n ≥ 0` and element `s` of ring `R`, return the `n × n` scalar matrix over `R` with `s` on the diagonal and zeros elsewhere. Equivalent to `MatrixRing(Parent(s), n)!s`. | — |
| `ScalarMatrix(R, n, s)` | As above, with explicit ring `R`; `s` is coerced into `R`. Equivalent to `MatrixRing(R, n)!s`. | — |
| `DiagonalMatrix(R, n, Q)` | Given ring `R`, integer `n ≥ 0`, and sequence `Q` of `n` elements, return the `n × n` diagonal matrix over `R` with diagonal entries from `Q` (coerced). | — |
| `DiagonalMatrix(R, Q)` | Given ring `R` and sequence `Q` of `n` elements, return the `n × n` diagonal matrix over `R`. | — |
| `DiagonalMatrix(Q)` | Given sequence `Q` of `n` elements from ring `R`, return the `n × n` diagonal matrix over `R`. | — |
| `Matrix(A)` | Given a matrix `A` of any type, return the same matrix re-parented to the appropriate matrix algebra (if square) or R-matrix space (otherwise). Useful for converting matrix-group elements or square R-matrix-space elements to a general matrix algebra. | — |
| `LowerTriangularMatrix(Q)` | Given sequence `Q` of length `l = n(n+1)/2` over ring `R`, return the `n × n` lower-triangular matrix whose lower triangular part (row-major) is given by `Q`. | — |
| `LowerTriangularMatrix(R, Q)` | As above, with explicit ring `R`; entries coerced. | — |
| `UpperTriangularMatrix(Q)` | Given sequence `Q` of length `l = n(n+1)/2` over ring `R`, return the `n × n` upper-triangular matrix whose upper triangular part (row-major) is given by `Q`. | — |
| `UpperTriangularMatrix(R, Q)` | As above, with explicit ring `R`; entries coerced. | — |
| `SymmetricMatrix(Q)` | Given sequence `Q` of length `l = n(n+1)/2` over ring `R`, return the `n × n` symmetric matrix whose lower triangular part (row-major) is given by `Q`. Avoids specifying the redundant upper triangle. | — |
| `SymmetricMatrix(R, Q)` | As above, with explicit ring `R`; entries coerced. | — |
| `AntisymmetricMatrix(Q)` | Given sequence `Q` of length `l = n(n-1)/2` over ring `R`, return the `n × n` antisymmetric matrix whose proper lower triangular part (row-major) is given by `Q`; diagonal is zero, proper upper triangle is the negation. | — |
| `AntisymmetricMatrix(R, Q)` | As above, with explicit ring `R`; entries coerced. | — |
| `PermutationMatrix(R, Q)` | Given ring `R` and sequence `Q` which is a permutation of `[1..n]`, return the `n × n` permutation matrix over `R` corresponding to `Q`. | — |
| `PermutationMatrix(R, x)` | Given ring `R` and permutation `x` of degree `n`, return the `n × n` permutation matrix over `R` corresponding to `x`. | — |

*Worked examples: H26E3 (3×3 scalar matrix over Z; 3×3 diagonal matrix over GF(23); 3×3 symmetric matrix over Q; lower/upper-triangular matrices of various sizes).*

### 26.2.4 Construction of Random Matrices

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RandomMatrix(R, m, n)` | Given finite ring `R` and positive integers `m` and `n`, construct a random `m × n` matrix over `R`. | Uniform random. |
| `RandomUnimodularMatrix(M, n)` | Given positive integers `M` and `n`, construct a random integral `n × n` matrix with determinant ±1; most entries in `[-M, M]`. | — |
| `RandomSLnZ(n, k, l)` | A random element of `SLn(Z)`, obtained by multiplying `l` random matrices of the form `I + E`, where `E` has exactly one nonzero off-diagonal entry with absolute value at most `k`. | — |
| `RandomGLnZ(n, k, l)` | A random element of `GLn(Z)`, obtained similarly to `RandomSLnZ`. | — |
| `RandomSymplecticMatrix(g, m)` | Given positive integers `g` and `m`, construct a (somewhat) random `2g × 2g` symplectic matrix over the integers; entries have the same order of magnitude as `m`. | — |

### 26.2.5 Creating Vectors

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Vector(n, Q)` | Given integer `n` and sequence `Q` of length `n` over ring `R`, return the vector of length `n` with entries from `Q`. Equivalent to `RSpace(Universe(Q), n)!Q`. | — |
| `Vector(Q)` | Given sequence `Q` of length `l` over ring `R`, return the vector of length `l` with entries from `Q`. Equivalent to `RSpace(Universe(Q), #Q)!Q`. | — |
| `Vector(R, n, Q)` | Given ring `R`, integer `n`, and sequence `Q` of length `n` over ring `S`, return the vector of length `n` with entries coerced into `R`. Equivalent to `RSpace(R, n)!Q`. | — |
| `Vector(R, Q)` | Given ring `R` and sequence `Q` of length `l` over ring `S`, return the vector of length `l` with entries coerced into `R`. Equivalent to `RSpace(R, #Q)!Q`. | — |

---

## 26.3 Elementary Properties

The following functions yield elementary properties of matrices and may be applied to matrices of any type, including vectors.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `NumberOfRows(A)` / `Nrows(A)` | Given an `m × n` matrix `A`, return `m`, the number of rows. | — |
| `NumberOfColumns(A)` / `Ncols(A)` | Given an `m × n` matrix `A`, return `n`, the number of columns. | — |
| `NumberOfNonZeroEntries(A)` / `NNZEntries(A)` | Given a matrix `A`, return the number of non-zero entries. | — |
| `Density(A)` | Given a matrix `A`, return the density as a real number: number of non-zero entries divided by the product of rows and columns (zero if `A` has zero rows or columns). | — |
| `BaseRing(A)` / `CoefficientRing(A)` | Given a matrix `A` with entries in ring `R`, return `R`. | — |
| `ElementToSequence(A)` / `Eltseq(A)` | Given an `m × n` matrix `A` over ring `R`, return the `mn` entries of `A` in row-major order as a sequence. | — |
| `RowSequence(A)` | Returns the entries of `A` as a sequence of rows, where each row is a sequence of entries. | — |

---

## 26.4 Accessing or Modifying Entries

### 26.4.1 Indexing

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `A[i]` | Given matrix `A` with `m` rows and integer `1 ≤ i ≤ m`, return the `i`-th row of `A` as a vector of length `n`. | — |
| `A[i, j]` | Given matrix `A` and integers `1 ≤ i ≤ m`, `1 ≤ j ≤ n`, return the `(i,j)`-th entry of `A` as an element of the ring `R`. | — |
| `A[Q]` / `A[i .. j]` | Given matrix `A` and sequence `Q` of integers in `[1..m]`, return the sequence of rows of `A` specified by `Q`. The range form `A[i..j]` may be used to specify a contiguous range. | — |
| `A[i] := v` | Given matrix `A`, integer `1 ≤ i ≤ m`, and vector `v` of length `n` over `R` (or `0` for the zero vector), modify the `i`-th row of `A` to be `v`. | — |
| `A[i, j] := x` | Given matrix `A`, integers `1 ≤ i ≤ m`, `1 ≤ j ≤ n`, and ring element `x` coercible into `R`, modify the `(i,j)`-th entry of `A` to be `x`. | — |

*Worked examples: H26E4 (accessing and modifying rows and entries of a 3×4 matrix over Z).*

### 26.4.2 Extracting and Inserting Blocks

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Submatrix(A, i, j, p, q)` / `ExtractBlock(A, i, j, p, q)` | Given `m × n` matrix `A` and integers `i, j, p, q` with `1 ≤ i ≤ i+p ≤ m+1` and `1 ≤ j ≤ j+q ≤ n+1`, return the `p × q` submatrix of `A` rooted at `(i,j)`. Either of `p` or `q` may be zero. | — |
| `SubmatrixRange(A, i, j, r, s)` / `ExtractBlockRange(A, i, j, r, s)` | Given `m × n` matrix `A` and integers `i, j, r, s` with `1 ≤ i`, `i-1 ≤ r ≤ m`, `1 ≤ j`, `j-1 ≤ s ≤ n`, return the `(r-i+1) × (s-j+1)` submatrix from `(i,j)` to `(r,s)` inclusive. | — |
| `Submatrix(A, I, J)` | Given `m × n` matrix `A` and integer sequences `I` and `J`, return the submatrix of `A` with row indices from `I` and column indices from `J`. | — |
| `InsertBlock(A, B, i, j)` / `InsertBlock(~A, B, i, j)` | Given `m × n` matrix `A` over ring `R`, `p × q` matrix `B` over `R`, and integers `i, j` such that `1 ≤ i ≤ i+p ≤ m+1` and `1 ≤ j ≤ j+q ≤ n+1`, insert `B` at position `(i,j)` in `A`. Functional form returns new matrix; procedural form (`~A`) modifies in place. | — |
| `RowSubmatrix(A, i, k)` | Given `m × n` matrix `A` and integers `i, k` with `1 ≤ i ≤ i+k ≤ m+1`, return the `k × n` submatrix consisting of rows `[i..i+k-1]`. | — |
| `RowSubmatrix(A, i)` | Given `m × n` matrix `A` and integer `0 ≤ i ≤ m`, return the `i × n` submatrix consisting of the first `i` rows. | — |
| `RowSubmatrixRange(A, i, j)` | Given `m × n` matrix `A` and integers `i, j` with `1 ≤ i` and `i-1 ≤ j ≤ m`, return the `(j-i+1) × n` submatrix of rows `[i..j]`. | — |
| `ColumnSubmatrix(A, i, k)` | Given `m × n` matrix `A` and integers `i, k` with `1 ≤ i ≤ i+k ≤ n+1`, return the `m × k` submatrix consisting of columns `[i..i+k-1]`. | — |
| `ColumnSubmatrix(A, i)` | Given `m × n` matrix `A` and integer `0 ≤ i ≤ n`, return the `m × i` submatrix of the first `i` columns. | — |
| `ColumnSubmatrixRange(A, i, j)` | Given `m × n` matrix `A` and integers `i, j` with `1 ≤ i` and `i-1 ≤ j ≤ n`, return the `m × (j-i+1)` submatrix of columns `[i..j]`. | — |

*Worked examples: H26E5 (submatrix extraction and block insertion on a 6×6 integer matrix; row/column submatrix operations).*

### 26.4.3 Row and Column Operations

For each operation there is a functional form (returns new matrix, `A` unchanged) and a procedural form (`~A`, modifies in place).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SwapRows(A, i, j)` / `SwapRows(~A, i, j)` | Swap rows `i` and `j` of matrix `A`. | — |
| `SwapColumns(A, i, j)` / `SwapColumns(~A, i, j)` | Swap columns `i` and `j` of matrix `A`. | — |
| `ReverseRows(A)` / `ReverseRows(~A)` | Reverse all rows of matrix `A`. | — |
| `ReverseColumns(A)` / `ReverseColumns(~A)` | Reverse all columns of matrix `A`. | — |
| `AddRow(A, c, i, j)` / `AddRow(~A, c, i, j)` | Given matrix `A` over ring `R`, element `c` coercible into `R`, and row indices `i, j`, add `c` times row `i` to row `j`. | — |
| `AddColumn(A, c, i, j)` / `AddColumn(~A, c, i, j)` | Given matrix `A` over ring `R`, element `c` coercible into `R`, and column indices `i, j`, add `c` times column `i` to column `j`. | — |
| `MultiplyRow(A, c, i)` / `MultiplyRow(~A, c, i)` | Given matrix `A` over ring `R`, element `c` coercible into `R`, and row index `i`, multiply row `i` by `c` on the left. | — |
| `MultiplyColumn(A, c, i)` / `MultiplyColumn(~A, c, i)` | Given matrix `A` over ring `R`, element `c` coercible into `R`, and column index `i`, multiply column `i` by `c` on the left. | — |
| `RemoveRow(A, i)` / `RemoveRow(~A, i)` | Given matrix `A` and row index `i`, remove row `i`, leaving an `(m-1) × n` matrix. | — |
| `RemoveColumn(A, j)` / `RemoveColumn(~A, j)` | Given matrix `A` and column index `j`, remove column `j`, leaving an `m × (n-1)` matrix. | — |
| `RemoveRowColumn(A, i, j)` / `RemoveRowColumn(~A, i, j)` | Given matrix `A` and indices `i, j`, remove row `i` and column `j`, leaving an `(m-1) × (n-1)` matrix. | — |
| `RemoveZeroRows(A)` / `RemoveZeroRows(~A)` | Remove all zero rows of matrix `A`. | — |

*Worked examples: H26E6 (row/column swap, row addition, row/column removal on a 5×6 integer matrix).*

---

## 26.5 Building Block Matrices

Block matrices constructed by listing blocks or by joining smaller matrices horizontally, vertically, or diagonally.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BlockMatrix(m, n, blocks)` | Construct the matrix from a sequence of `m·n` block matrices (all the same dimensions), listed in row-major order. | — |
| `BlockMatrix(m, n, rows)` / `BlockMatrix(rows)` | Construct the matrix from a sequence of `m` rows, each containing `n` block matrices. | — |
| `HorizontalJoin(X, Y)` | Join matrix `X` (r × c) and matrix `Y` (r × d) horizontally, both over ring `R`, returning an `r × (c+d)` matrix. | — |
| `HorizontalJoin(Q)` / `HorizontalJoin(T)` | Join a sequence `Q` or tuple `T` of matrices (same number of rows, same ring) horizontally in order. | — |
| `VerticalJoin(X, Y)` | Join matrix `X` (r × c) and matrix `Y` (s × c) vertically, both over ring `R`, returning an `(r+s) × c` matrix. | — |
| `VerticalJoin(Q)` / `VerticalJoin(T)` | Join a sequence `Q` or tuple `T` of matrices (same number of columns, same ring) vertically in order. | — |
| `DiagonalJoin(X, Y)` | Join matrix `X` (a × b) and matrix `Y` (c × d), both over ring `R`, diagonally: returns an `(a+c) × (b+d)` matrix with `X` upper-left, `Y` lower-right, and zero blocks elsewhere. | — |
| `DiagonalJoin(Q)` / `DiagonalJoin(T)` | Join a sequence `Q` or tuple `T` of matrices (same ring) diagonally in order. | — |
| `KroneckerProduct(A, B)` | Given `m × n` matrix `A` and `p × q` matrix `B` over ring `R`, return the `mp × nq` Kronecker product `C` where the `((i-1)p+r, (j-1)q+s)`-th entry of `C` is `A[i,j] * B[r,s]`. | — |

---

## 26.6 Changing Ring

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ChangeRing(A, R)` / `Matrix(R, A)` | Given `m × n` matrix `A` over ring `S` and ring `R`, return the `m × n` matrix over `R` by coercing entries from `S` into `R`. The `Matrix(R, A)` form is consistent with the matrix creation convention of leading-ring argument. | — |
| `ChangeRing(A, R, f)` / `ChangeRing(A, f)` | Given matrix `A` over ring `S`, ring `R`, and map `f : S → R`, return the matrix over `R` obtained by applying `f` to each entry. `R` may be omitted (taken as the codomain of `f`). | — |

---

## 26.7 Elementary Arithmetic

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `A + B` | Given `m × n` matrices `A` and `B` over ring `R`, return `A + B`. | — |
| `A - B` | Given `m × n` matrices `A` and `B` over ring `R`, return `A - B`. | — |
| `A * B` | Given `m × n` matrix `A` and `n × p` matrix `B` over ring `R`, return the `m × p` product `A·B`. Parent is chosen to preserve maximal information (e.g. same algebra if both are square in the same algebra). | — |
| `x * A` / `A * x` | Given `m × n` matrix `A` over ring `R` and ring element `x` coercible into `R`, return the scalar product `x·A`. | — |
| `-A` | Given matrix `A`, return `-A`. | — |
| `A ^ -1` | Given invertible square matrix `A` over ring `R` (a field, Euclidean domain, or ring with exact division and characteristic 0 or > m), return the inverse `B` with `A·B = B·A = 1`. | — |
| `A ^ n` | Given square matrix `A` over ring `R` and integer `n`, return `Aⁿ`. `A^0` is the identity. If `n < 0`, `A` must be invertible. | — |
| `Transpose(A)` | Given `m × n` matrix `A` over ring `R`, return the `n × m` transpose whose `(i,j)`-th entry is the `(j,i)`-th entry of `A`. | — |
| `AddScaledMatrix(A, s, B)` | Given matrix `A` over ring `R`, scalar `s` coercible into `R`, and matrix `B` over `R` with the same shape, return `A + s·B`. Generally faster than `A + s*B`. | — |
| `AddScaledMatrix(~A, s, B)` | As above but modifies `A` in place to `A + s·B`. Generally faster than `A := A + s*B`. | — |

---

## 26.8 Nullspaces and Solutions of Systems

Magma possesses a rich suite of internal algorithms for computing nullspaces of matrices efficiently, including a fast p-adic algorithm for matrices over Z and Q, and algorithms that take advantage of sparsity.

The functions compute nullspaces of matrices (solving `V · A = 0`), or solve systems of the form `V · A = W` for given `A` and `W`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Nullspace(A)` / `Kernel(A)` | Given `m × n` matrix `A` over ring `R`, return the nullspace of `A`: the R-space of all vectors `v` of length `m` such that `v·A = 0`. If the parent of `A` is an R-matrix space, the result is a submodule of the domain. `Kernel(A)` also returns the inclusion map into the domain. | Fast p-adic algorithm for Z, Q; sparse-aware for general rings. |
| `NullspaceMatrix(A)` / `KernelMatrix(A)` | Given `m × n` matrix `A` over ring `R`, return a basis matrix `N` of the nullspace of `A` (`N·A = 0`), with `m` columns. The result is not an R-space, avoiding echelonisation overhead. | As above. |
| `NullspaceOfTranspose(A)` | Equivalent to `Nullspace(Transpose(A))`, but may be more efficient in space for large matrices since the transpose may not need to be explicitly constructed. | As above. |
| `IsConsistent(A, W)` | Given `m × n` matrix `A` over `R` and vector `W` of length `n` (or `r × n` matrix `W`) over `R`, return `true` iff `V·A = W` is consistent. If consistent, also return: (a) a particular solution `V`; (b) the nullspace `N` of `A`. | — |
| `IsConsistent(A, Q)` | Given `m × n` matrix `A` over `R` and sequence `Q` of vectors of length `n`, return `true` iff `V[i]*A = Q[i]` is consistent for all `i`. If consistent, also return a solution sequence `V` and the nullspace `N`. | — |
| `Solution(A, W)` | Given `m × n` matrix `A` over `R` and vector `W` of length `n` (or `r × n` matrix `W`) over `R`, solve `V·A = W` and return: (a) a particular solution `V`; (b) the nullspace `N` of `A`. Errors if no solution exists. | — |
| `Solution(A, Q)` | Given `m × n` matrix `A` over `R` and sequence `Q` of vectors of length `n`, solve `V[i]*A = Q[i]` for each `i` and return: (a) a solution sequence `V`; (b) the nullspace `N`. Errors if no solution exists. | — |

*Worked examples: H26E7 (nullspace of a 301×300 random integer matrix; nullity 1, null vector entries ~455 decimal digits); H26E8 (enumerating all solutions to `V·X = W` over GF(3)).*

---

## 26.9 Predicates

Test various properties of matrices. (See also the Lattices chapter for `IsPositiveDefinite` and related functions.)

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsZero(A)` | Return `true` iff `A` is the `m × n` zero matrix. | — |
| `IsOne(A)` | Return `true` iff square matrix `A` is the identity matrix. | — |
| `IsMinusOne(A)` | Return `true` iff square matrix `A` is the negation of the identity matrix. | — |
| `IsScalar(A)` | Return `true` iff square matrix `A` is scalar (a scalar multiple of the identity). | — |
| `IsDiagonal(A)` | Return `true` iff square matrix `A` is diagonal (all off-diagonal entries zero). | — |
| `IsSymmetric(A)` | Return `true` iff square matrix `A` equals its transpose. | — |
| `IsUpperTriangular(A)` | Return `true` iff matrix `A` is upper triangular (all entries below the diagonal are zero). | — |
| `IsLowerTriangular(A)` | Return `true` iff matrix `A` is lower triangular (all entries above the diagonal are zero). | — |
| `IsUnit(A)` | Return `true` iff square matrix `A` is a unit (has an inverse). Computed by testing if the determinant is a unit; works for any commutative ring `R`. | Determinant test. |
| `IsSingular(A)` | Return `true` iff square `m × m` matrix `A` is singular (determinant is zero, equivalently rank < m). Note: not singular does not imply invertible when `R` is not a field. Works for any commutative ring. | Determinant test. |
| `IsSymplecticMatrix(A)` | Given `m × m` integer matrix `A`, return `true` iff `A` is a symplectic matrix, i.e., `AJtA = J` where `J = [[0, 1g], [-1g, 0]]`. | Direct definition. |

---

## 26.10 Determinant and Other Properties

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Determinant(A: MonteCarloLevel, Proof, pAdic, Divisor)` | Given square matrix `A` over commutative ring `R`, return the determinant of `A`. The determinant of the `0 × 0` matrix is `R!1`. Parameters: `MonteCarloLevel` (RngIntElt, default 0 — if set to positive integer `s`, uses probabilistic Monte-Carlo modular technique terminating when residue is stable for `s` steps); `Proof` (BoolElt, default true — setting to false equivalent to `MonteCarloLevel := 2`); `pAdic` (BoolElt, default true); `Divisor` (RngIntElt, default 0 — if set to known exact divisor `d` of the determinant, algorithm is sped up). | For Z or Q: modular algorithm of **[ABM99]** — computes a divisor `d` of `det(A)` via fast p-adic nullspace, then computes `det(A)/d` by evaluating the determinant modulo enough small primes to exceed the Hadamard bound divided by `d`. |
| `Trace(A)` | Given square matrix `A` over ring `R`, return the trace (sum of diagonal elements) as an element of `R`. | — |
| `TraceOfProduct(A, B)` | Given square matrices `A` and `B` of the same size over ring `R`, return the trace of `A·B`. Generally much faster than `Trace(A*B)`. | — |
| `Rank(A)` | Given `m × n` matrix `A` over ring `R`, return the rank of `A` (the largest `r` such that a non-zero `r × r` subdeterminant of `A` exists). May require computing Smith form or echelon form. | Smith or echelon form computation. |
| `Minor(M, i, j)` | Return the determinant of the submatrix of square matrix `M` obtained by removing row `i` and column `j`. | — |
| `Minor(M, I, J)` | Return the determinant of the submatrix of `M` given by row indices `I` and column indices `J`. | — |
| `Minors(M, r)` | Return a sequence of all `r × r` minors of matrix `M`. | — |
| `Cofactor(M, i, j)` | Return `(-1)^(i+j)` times the `(i,j)`-minor of `M`. | — |
| `Cofactors(M)` | Return a sequence of all cofactors of square matrix `M`. | — |
| `Cofactors(M, r)` | Return a sequence of all `r × r` cofactors of matrix `M`. | — |
| `Pfaffian(M)` | Given antisymmetric square matrix `M`, return its Pfaffian (the canonical square root of the determinant given by a universal polynomial in the entries). Computed by Pfaffian row-expansion. | Pfaffian row-expansion. |
| `Pfaffian(M, I, J)` | Return the Pfaffian of the submatrix of `M` described by index sequences `I` and `J`. | Pfaffian row-expansion. |
| `Pfaffians(M, r)` | Return the sequence of Pfaffians of all `C(n,r)` principal `r × r` submatrices of `M` (`n` = number of rows). | Pfaffian row-expansion. |

---

## 26.11 Minimal and Characteristic Polynomials and Eigenvalues

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `MinimalPolynomial(A: Proof)` | Given square matrix `A` over ring `R` (field or Z), return the minimal polynomial of `A`: the unique monic polynomial `f(x)` of minimal degree with `f(A) = 0`. `f(x)` always divides the characteristic polynomial. Parameter: `Proof` (BoolElt, default true). | — |
| `CharacteristicPolynomial(A: Al, Proof)` | Given square matrix `A` over commutative ring `R`, return the characteristic polynomial `det(x - A) ∈ R[x]`. Parameters: `Al` (MonStg, default `"Modular"`) selects algorithm: `"Modular"` (default, for Z and Q, very fast since V2.8); `"Hessenberg"` (fields, via Hessenberg form reduction); `"Interpolation"` (Z, Q, by evaluation and interpolation); `"Trace"` (fields, via traces of powers). `Proof` (BoolElt, default true). | p-adic modular algorithm (default for Z, Q) **[ABM99]**-style; Hessenberg, interpolation, and trace-power variants also available. |
| `MinimalAndCharacteristicPolynomials(A: Proof)` / `MCPolynomials(A)` | Given square matrix `A` over ring `R`, return both the minimal and characteristic polynomials. More efficient than calling each separately for some rings. Parameter: `Proof` (BoolElt, default true). | — |
| `FactoredMinimalPolynomial(A: Proof)` | Given square matrix `A` over ring `R`, return the factorization of the minimal polynomial. Equivalent to `Factorization(MinimalPolynomial(A))` but may be faster for Z and Q. Parameter: `Proof` (BoolElt, default true). | — |
| `FactoredCharacteristicPolynomial(A: Al, Proof)` | Return the factorization of the characteristic polynomial. Same result as `Factorisation(CharacteristicPolynomial(A))` but may be faster. Parameters as for `CharacteristicPolynomial`. | — |
| `FactoredMinimalAndCharacteristicPolynomials(A: Al, Proof)` / `FactoredMCPolynomials(A: Al, Proof)` | Return the factorizations of both the minimal and characteristic polynomials. More efficient for some rings than calling each separately. Parameters as for `CharacteristicPolynomial`. | — |
| `Eigenvalues(A)` | Given square matrix `A` over ring `R`, return the eigenvalues as a set of pairs `(value, multiplicity)`. Factorization of polynomials over `R` must be possible. | Roots of characteristic polynomial. |
| `Eigenspace(A, e)` | Given square matrix `A` over ring `R` and element `e ∈ R`, return the eigenspace `Nullspace(A - e)`. Returns trivial nullspace if `e` is not an eigenvalue. | — |

---

## 26.12 Canonical Forms

### 26.12.1 Canonical Forms over General Rings

Applies to matrices over fields or Euclidean domains. (See also the Lattices chapter for LLL and related basis-reduction functions.)

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `EchelonForm(A)` | Given `m × n` matrix `A` over ring `R`, return the (reduced) row echelon form `E` of `A`, and an invertible `m × m` transformation matrix `T` over `R` such that `T·A = E`. If `R` is a Euclidean domain, `HermiteForm` is invoked (parameters for `HermiteForm` cannot be set via this route). | Row reduction; delegates to `HermiteForm` for Euclidean domains. |
| `Adjoint(A)` | Given square `m × m` matrix `A` over ring `R` (a ring with exact division whose characteristic is 0 or > m), return the adjoint of `A`. | — |

### 26.12.2 Canonical Forms over Fields

Applies to square matrices over fields supporting univariate polynomial factorization. The single unified algorithm underlying `PrimaryRationalForm`, `JordanForm`, `RationalForm`, and related functions is that of Steel **[Ste97]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PrimaryRationalForm(A)` | Given square matrix `A` over field `K` (with polynomial factorization), return: (a) the primary rational canonical form `F` of `A` (blocks are companion matrices of powers of irreducible polynomials); (b) invertible `T` with `T·A·T⁻¹ = F`; (c) a sequence of pairs `(irreducible polynomial, multiplicity)` describing the blocks (same as `PrimaryInvariantFactors(A)`). | **[Ste97]** |
| `JordanForm(A)` | Given square matrix `A` over field `K`, return: (a) the generalized Jordan form `F` (blocks are Jordan blocks from powers of irreducible polynomials; reduces to the usual Jordan form when the minimal polynomial splits over `K`); (b) invertible `T` with `T·A·T⁻¹ = F`; (c) sequence of pairs `(irreducible polynomial, multiplicity)` (same as `PrimaryInvariantFactors(A)`). | **[Ste97]** |
| `RationalForm(A)` | Given square matrix `A` over field `K`, return: (a) the rational form `F` of `A` (each block's polynomial divides the next); (b) invertible `T` with `T·A·T⁻¹ = F`; (c) sequence of polynomials for successive blocks (same as `InvariantFactors(A)`). | **[Ste97]** |
| `PrimaryInvariantFactors(A)` | Given square matrix `A` over field `K`, return the primary invariant factors: a sequence of pairs `(irreducible polynomial, multiplicity)`. This is the third return value of `PrimaryRationalForm(A)` or `JordanForm(A)`. | **[Ste97]** |
| `InvariantFactors(A)` | Given square matrix `A` over field `K`, return the invariant factors: a sequence of polynomials where each divides the next. This is the third return value of `RationalForm(A)`. | **[Ste97]** |
| `IsSimilar(A, B)` | Given square `m × m` matrices `A` and `B` over field `K`, return `true` iff `A` is similar to `B`, and if so, also return an invertible `m × m` matrix `T` with `T·A·T⁻¹ = B`. | **[Ste97]** |
| `HessenbergForm(A)` | Given square `m × m` matrix `A` over field `R`, return the Hessenberg form of `A` (zero entries above the super-diagonal). Used in one of the characteristic polynomial algorithms. | — |
| `FrobeniusFormAlternating(A)` | Given non-singular `2n × 2n` alternating integer matrix `A`, return the (alternating) Frobenius form `F = [[0, D], [-D, 0]]` where `D` is diagonal with positive entries `d₁ | d₂ | … | dₙ`, and the change-of-basis matrix `B` with `B·A·tB = F`. | — |

*Worked examples: H26E9 (5×5 matrix over GF(5): primary invariant factors, Jordan form, rational form, transformation matrix; verification via Smith form of the characteristic matrix).*

### 26.12.3 Canonical Forms over Euclidean Domains

Applies to matrices over Euclidean domains. (See also the Lattices chapter for LLL and related functions, which are very useful for integer matrices.)

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HermiteForm(A: Al, Optimize, Integral)` | Given `m × n` matrix `A` over Euclidean ring `R`, return the Hermite form `H` of `A` and an invertible `m × m` transformation matrix `T` over `R` with `T·A = H`. Parameters: `Al` (MonStg, default `"Default"`) selects `"Modular"` (fast modular, preferred for Z since V2.13) or `"Classical"` (Kannan–Bachem, no bad coefficient growth); `Optimize` (BoolElt, default true — if true and `R = Z` and `T` is requested, uses LLL on the kernel of `A` to minimise entries of `T`); `Integral` (BoolElt, default true — if true, uses integral de Weger LLL in the optimisation step instead of floating-point). | Classical Kannan–Bachem **[KB79, CC82]**; modular algorithm by Allan Steel (default for Z since V2.13). |
| `SmithForm(A)` | Given `m × n` matrix `A` over Euclidean ring `R`, return: (a) the Smith normal form `S`; (b) unimodular matrices `P` and `Q` with `P·A·Q = S`. Prefer `ElementaryDivisors` unless the transformation matrices are needed (saves memory). | Sparse preprocessing **[HHR93]** → iterated Hermite reduction or modular algorithm **[Lüb02]**. |
| `ElementaryDivisors(A)` | Given `m × n` matrix `A` over Euclidean ring or field `R`, return the elementary divisors: the non-zero diagonal entries of the Smith form, as a sequence `[e₁, …, eᵣ]` where `eᵢ | eᵢ₊₁` and `r` is the rank of `A`. Divisors are normalised. Over a field, always returns `r` ones. If `m = n = r` and `R` is a domain, `eᵣ` is the lowest common denominator of the inverse of `A` over the fraction field. | Smith normal form (see `SmithForm`). |
| `Saturation(A)` | Given `m × n` integer matrix `A` of rank `r`, return an `m × n` matrix `S` over Z whose first `r` rows form a basis of the saturation of the row space with respect to Q: the set of all `v` over Z such that `s·v` is in the Z-module spanned by the rows of `A` for some non-zero scalar `s`. | — |

*Worked examples: H26E10 (EchelonForm of a 4×3 matrix over GF(8); HermiteForm, SmithForm, ElementaryDivisors of a 4×5 integer matrix; verification of rank 4).*

---

## 26.13 Orders of Invertible Matrices

Magma can efficiently compute the order of an invertible matrix over a finite field using the Cunningham database to factorise numbers of the form `pⁿ - 1` which arise; the algorithm is that of **[CLG97]**. Magma also contains efficient algorithms for rigorously proving whether a matrix over Z, Q, an algebraic number field, a cyclotomic field, or a quadratic field has finite order or not.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HasFiniteOrder(A)` | Given invertible square matrix `A` over ring `R` (finite field, Z, Q, algebraic number field, cyclotomic field, or quadratic field), return `true` iff `Aⁿ = 1` for some positive integer `n`. Rigorously proven over all supported rings. | Rigorous finiteness test. |
| `Order(A: Proof)` | Given invertible square matrix `A` over any commutative ring, return the order of `A`. If `R` is a ring for which a finite order proof exists, errors if `A` has infinite order. Over other rings, may loop indefinitely if the order is infinite. Parameter: `Proof` (BoolElt, default true). | **[CLG97]** for finite fields (Cunningham database); rigorous methods for Z, Q, number fields. |
| `FactoredOrder(A: Proof)` | Given invertible square matrix `A` over a finite field, return the order in factored form. Same result as `Factorization(Order(A))` but the factorization is computed as part of the order computation at no extra cost. Parameter: `Proof` (BoolElt, default true). | **[CLG97]** |
| `ProjectiveOrder(A: Proof)` | Given invertible square matrix `A` over finite field `K`, return the projective order `n` of `A` (smallest `n` with `Aⁿ = sI` for some scalar `s`) and the scalar `s ∈ K`. The projective order always divides the true order. Parameter: `Proof` (BoolElt, default true). | **[CLG97]** |
| `FactoredProjectiveOrder(A: Proof)` | Given invertible square matrix `A` over finite field `K`, return the projective order in factored form and the scalar `s ∈ K` with `Aⁿ = sI`. Parameter: `Proof` (BoolElt, default true). | **[CLG97]** |

---

## 26.14 Miscellaneous Operations on Matrices

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `FrobeniusImage(A, e)` | Given matrix `A` over finite field `K` of characteristic `p`, return the matrix obtained from `A` by mapping each entry `Aᵢⱼ` to `(Aᵢⱼ)^(pᵉ)`. | Frobenius endomorphism. |
| `NumericalEigenvectors(M, e)` | Given square matrix `M` coercible into the complexes and an approximation `e` to an eigenvalue, attempt to find eigenvectors. Intended for cases with no numerical worries. | Numerical (floating-point) eigenvector computation. |

---

## 26.15 Bibliography

| Key | Reference |
|-----|-----------|
| **[ABM99]** | John Abbott, Manuel Bronstein, and Thom Mulders. *Fast Deterministic Computation of Determinants of Dense Matrices.* In Sam Dooley, editor, Proceedings ISSAC'99, pages 197–204, New York, 1999. ACM Press. |
| **[CC82]** | T. W. J. Chou and G. E. Collins. *Algorithms for the solution of systems of linear Diophantine equations.* SIAM J. Computing, 11(4):687–708, 1982. |
| **[CLG97]** | Frank Celler and Charles R. Leedham-Green. *Calculating the Order of an Invertible Matrix.* In Larry Finkelstein and William M. Kantor, editors, Groups and Computation II, volume 28 of DIMACS Series in Discrete Mathematics and Theoretical Computer Science, pages 55–60. AMS, 1997. |
| **[HHR93]** | George Havas, Derek F. Holt, and Sarah Rees. *Recognizing badly presented Z-modules.* Linear Algebra and its Applications, 192:137–164, 1993. |
| **[KB79]** | R. Kannan and A. Bachem. *Polynomial algorithms for computing the Smith and Hermite normal forms of an integer matrix.* SIAM J. Computing, 9:499–507, 1979. |
| **[Lüb02]** | F. Lübeck. *On the computation of elementary divisors of integer matrices.* J. Symbolic Comp., 33:57–65, 2002. |
| **[Ste97]** | Allan Steel. *A New Algorithm for the Computation of Canonical Forms of Matrices over Fields.* J. Symbolic Comp., 24(3):409–432, 1997. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Modular determinant **[ABM99]** | `Determinant` |
| Nullspace / linear systems (p-adic, sparse) | `Nullspace`, `Kernel`, `NullspaceMatrix`, `KernelMatrix`, `NullspaceOfTranspose`, `IsConsistent`, `Solution` |
| Canonical forms over fields **[Ste97]** | `PrimaryRationalForm`, `JordanForm`, `RationalForm`, `PrimaryInvariantFactors`, `InvariantFactors`, `IsSimilar`, `CharacteristicPolynomial(:Al="Hessenberg")`, `HessenbergForm` |
| Characteristic polynomial, modular (p-adic) | `CharacteristicPolynomial`, `FactoredCharacteristicPolynomial`, `MinimalAndCharacteristicPolynomials`, `FactoredMCPolynomials` |
| Hermite normal form: classical **[KB79, CC82]** / modular (Steel) | `HermiteForm`, `EchelonForm` |
| Smith normal form: sparse preprocessing **[HHR93]** + modular **[Lüb02]** | `SmithForm`, `ElementaryDivisors`, `Saturation` |
| Order of invertible matrix over finite fields (Cunningham) **[CLG97]** | `Order`, `FactoredOrder`, `ProjectiveOrder`, `FactoredProjectiveOrder`, `HasFiniteOrder` |
| Pfaffian row-expansion | `Pfaffian`, `Pfaffians` |
| Frobenius endomorphism | `FrobeniusImage` |
