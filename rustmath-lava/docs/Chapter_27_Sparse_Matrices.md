# Chapter 27 — Sparse Matrices

**Handbook part:** IV — Matrices and Linear Algebra
**Handbook pages:** 559–582 (PDF pages 690–715)

---

## Scope and overview

Magma provides a dedicated sparse matrix type (`MtrxSprs`) distinct from the ordinary dense-representation matrices. All sparse matrices over a given ring `R` share the same parent structure (`MtrxSprsStr`). The sparse representation stores only non-zero entries, enabling the construction and manipulation of extremely large matrices that would be infeasible in dense form.

The operations supported include dynamic construction (the matrix can grow as entries are set, without knowing final dimensions in advance), standard linear-algebraic invariants (rank, determinant, nullspace, Smith form), and a high-performance linear-system solver tailored to index-calculus problems. A key design goal is supporting algorithms that first generate a very large sparse relation matrix and then extract structural information (a null vector or the elementary divisors) from it.

The main algorithmic workhorse for the non-trivial properties (rank, determinant, nullspace, Smith form) is **sparse Gaussian elimination via Markowitz pivoting** [DEJ84], which reduces the sparse system to a smaller dense matrix before invoking the corresponding dense algorithm. For index-calculus linear systems, **Structured Gaussian Elimination** [LO91b] is used to reduce the sparse system dramatically before calling the dense modular nullspace solver; optionally the **Lanczos algorithm** [LO91b] may be substituted when memory is the limiting constraint.

An extended worked example (H27E3) implements the first stage of the basic linear sieve [COS86, LO91a] for discrete logarithms in a prime finite field, demonstrating end-to-end use of dynamic sparse-matrix construction via `SetEntry` and linear-system solution via `ModularSolution`.

---

## 27.1 Introduction

A separate type is provided for sparse matrices to allow users to construct such matrices and apply algorithms that take advantage of sparsity. Sparse matrices support dynamic construction, simple property queries, and a number of non-trivial and important invariants (rank, determinant, nullspace, elementary divisors). The type name for the category of sparse matrices is `MtrxSprs`; the parent structure for all sparse matrices over a given ring `R` is `MtrxSprsStr`.

---

## 27.2 Creation of Sparse Matrices

### 27.2.1 Construction of Initialized Sparse Matrices

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SparseMatrix(R, m, n, Q)` / `SparseMatrix(m, n, Q)` | Return the m × n sparse matrix over ring `R` (derived from entries if omitted) whose non-zero entries are specified by sequence `Q`. `Q` may be a sequence of tuples `<i, j, x>` (position and value), or a "flat" compact integer sequence encoding row non-zero counts followed by column-value pairs. In the flat form entries must lie in the prime ring of `R`. | Direct construction from entry list. |
| `SparseMatrix(R, m, n)` | Return the m × n zero sparse matrix over `R`. | — |
| `SparseMatrix(m, n)` | Return the m × n zero sparse matrix over **Z**. | — |

*Worked examples: H27E1 (creating matrices over **Z**, GF(23), GF(2^4) using both tuple and flat-integer forms; printing in Magma format).*

### 27.2.2 Construction of Trivial Sparse Matrices

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SparseMatrix(R)` / `SparseMatrix()` | Create the 0 × 0 sparse matrix over `R` (or over **Z** if no argument). Intended for dynamic construction where final dimensions are unknown; `SetEntry` will extend the matrix automatically as entries are added. | — |

### 27.2.3 Construction of Structured Matrices

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IdentitySparseMatrix(R, n)` | Return the n × n identity sparse matrix over `R`. | — |
| `ScalarSparseMatrix(n, s)` | Return the n × n scalar sparse matrix with `s` on the diagonal and zeros elsewhere; `R` is derived from `s`. | — |
| `ScalarSparseMatrix(R, n, s)` | Return the n × n scalar sparse matrix over `R` with `s` (coerced into `R`) on the diagonal and zeros elsewhere. | — |
| `DiagonalSparseMatrix(R, n, Q)` | Return the n × n diagonal sparse matrix over `R` whose diagonal entries are the elements of sequence `Q` (length n), coerced into `R`. | — |
| `DiagonalSparseMatrix(R, Q)` | Return the n × n diagonal sparse matrix over `R` whose diagonal entries are the elements of `Q` (n = #Q), coerced into `R`. | — |
| `DiagonalSparseMatrix(Q)` | Return the n × n diagonal sparse matrix over the ring of the elements of `Q` whose diagonal entries are the elements of `Q`. | — |

### 27.2.4 Parents of Sparse Matrices

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SparseMatrixStructure(R)` | Create the structure containing all sparse matrices (of any shape) over ring `R`. The parent is created automatically when a sparse matrix is constructed, but this function allows explicit creation. | — |

---

## 27.3 Accessing Sparse Matrices

### 27.3.1 Elementary Properties

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BaseRing(A)` / `CoefficientRing(A)` | Return the coefficient ring `R` of the sparse matrix `A`. | — |
| `NumberOfRows(A)` / `Nrows(A)` | Return the number of rows of the m × n sparse matrix `A`. | — |
| `NumberOfColumns(A)` / `Ncols(A)` | Return the number of columns of the m × n sparse matrix `A`. | — |
| `ElementToSequence(A)` / `Eltseq(A)` | Return the entries of `A` as a sequence of tuples `<i, j, x>` for all non-zero entries. It is always true that `SparseMatrix(Nrows(A), Ncols(A), Eltseq(A))` equals `A`. | — |
| `NumberOfNonZeroEntries(A)` / `NNZEntries(A)` | Return the number of non-zero entries in `A`. | — |
| `Density(A)` | Return the density of `A` as a real number: number of non-zero entries divided by the product of row count and column count (zero if either dimension is zero). | — |
| `Support(A, i)` | Return the column indices of the non-zero entries of row `i` of `A`. | — |
| `Support(A)` | Return the sequence of all pairs `<i, j>` such that the [i, j]-th entry of `A` is non-zero. | — |

### 27.3.2 Weights

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RowWeight(A, i)` | Return the weight (number of non-zero entries) of row `i` of `A`. | — |
| `RowWeights(A)` | Return the length-m sequence whose i-th entry is the weight of row `i`. | — |
| `ColumnWeight(A, j)` | Return the weight (number of non-zero entries) of column `j` of `A`. | — |
| `ColumnWeights(A)` | Return the length-n sequence whose j-th entry is the weight of column `j`. | — |

---

## 27.4 Accessing or Modifying Entries

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `A[i]` | Return row `i` of `A` as a dense vector of length n (in R^n). | — |
| `A[i, j]` | Return the (i, j)-th entry of `A` as an element of `R`. | — |
| `A[i, j] := x` | Set the (i, j)-th entry of `A` to `x` (coercible into `R`). Indices must be within the current dimensions of `A`. | — |
| `SetEntry(~A, i, j, x)` | (Procedure.) Set the (i, j)-th entry of `A` to `x` (coercible into `R`). Unlike `A[i,j] := x`, the indices `i` and `j` may exceed the current dimensions; `A` is automatically extended to have at least `i` rows and `j` columns. Used for dynamic matrix construction (e.g., in index-calculus methods). | — |

*Worked examples: H27E2 (accessing rows and entries, extending a matrix dynamically with `SetEntry`, starting from the 0 × 0 sparse matrix and building to 200 × 3876 with 4 non-zero entries).*

### 27.4.1 Extracting and Inserting Blocks

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Submatrix(A, i, j, p, q)` / `ExtractBlock(A, i, j, p, q)` | Return the p × q submatrix of `A` rooted at position (i, j). Either p or q may be zero. | — |
| `SubmatrixRange(A, i, j, r, s)` / `ExtractBlockRange(A, i, j, r, s)` | Return the submatrix of `A` from row `i` to row `r` and column `j` to column `s` inclusive. | — |
| `Submatrix(A, I, J)` | Return the submatrix of `A` indexed by integer sequences `I` (rows) and `J` (columns). | — |
| `InsertBlock(A, B, i, j)` / `InsertBlock(~A, B, i, j)` | Insert p × q sparse matrix `B` into `A` at position (i, j). Functional version returns new matrix; procedural version modifies `A` in place. | — |
| `RowSubmatrix(A, i, k)` | Return the k × n submatrix consisting of rows [i .. i+k−1]. | — |
| `RowSubmatrix(A, i)` | Return the i × n submatrix consisting of the first `i` rows. | — |
| `RowSubmatrixRange(A, i, j)` | Return the submatrix consisting of rows [i .. j]. | — |
| `ColumnSubmatrix(A, i, k)` | Return the m × k submatrix consisting of columns [i .. i+k−1]. | — |
| `ColumnSubmatrix(A, i)` | Return the m × i submatrix consisting of the first `i` columns. | — |
| `ColumnSubmatrixRange(A, i, j)` | Return the submatrix consisting of columns [i .. j]. | — |

### 27.4.2 Row and Column Operations

Each operation has both a functional form (returns a new sparse matrix, leaving `A` unchanged) and a procedural form (modifies `A` in place via `~A`).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SwapRows(A, i, j)` / `SwapRows(~A, i, j)` | Swap rows `i` and `j` of `A`. | — |
| `SwapColumns(A, i, j)` / `SwapColumns(~A, i, j)` | Swap columns `i` and `j` of `A`. | — |
| `ReverseRows(A)` / `ReverseRows(~A)` | Reverse all rows of `A`. | — |
| `ReverseColumns(A)` / `ReverseColumns(~A)` | Reverse all columns of `A`. | — |
| `AddRow(A, c, i, j)` / `AddRow(~A, c, i, j)` | Add `c` times row `i` to row `j` of `A`. | — |
| `AddColumn(A, c, i, j)` / `AddColumn(~A, c, i, j)` | Add `c` times column `i` to column `j` of `A`. | — |
| `MultiplyRow(A, c, i)` / `MultiplyRow(~A, c, i)` | Multiply row `i` of `A` by `c` (on the left). | — |
| `MultiplyColumn(A, c, i)` / `MultiplyColumn(~A, c, i)` | Multiply column `i` of `A` by `c` (on the left). | — |
| `RemoveRow(A, i)` / `RemoveRow(~A, i)` | Remove row `i` from `A`, yielding an (m−1) × n matrix. | — |
| `RemoveColumn(A, j)` / `RemoveColumn(~A, j)` | Remove column `j` from `A`, yielding an m × (n−1) matrix. | — |
| `RemoveRowColumn(A, i, j)` / `RemoveRowColumn(~A, i, j)` | Remove row `i` and column `j` from `A`, yielding an (m−1) × (n−1) matrix. | — |
| `RemoveZeroRows(A)` / `RemoveZeroRows(~A)` | Remove all zero rows from `A`. | — |

---

## 27.5 Building Block Matrices

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HorizontalJoin(A, B)` | Return the r × (c+d) sparse matrix obtained by joining `A` (r × c) and `B` (r × d) horizontally (B placed to the right of A). Both must have the same coefficient ring. | — |
| `VerticalJoin(A, B)` | Return the (r+s) × c sparse matrix obtained by joining `A` (r × c) and `B` (s × c) vertically (B placed underneath A). Both must have the same coefficient ring. | — |
| `DiagonalJoin(A, B)` | Return the (a+c) × (b+d) sparse matrix obtained by joining `A` (a × b) and `B` (c × d) diagonally, with zero blocks in the off-diagonal positions. Both must have the same coefficient ring. | — |

---

## 27.6 Conversion to and from Dense Matrices

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Matrix(A)` | Return the normal (dense-representation) matrix equal to sparse matrix `A`. Should only be used for reasonably small matrices. Useful for displaying `A` as a normal matrix. | — |
| `SparseMatrix(A)` | Return the sparse matrix equal to normal (dense-representation) matrix `A`. Note: if a fast sparse algorithm is available, there is no need to convert — Magma applies it automatically. | — |

---

## 27.7 Changing Ring

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ChangeRing(A, R)` / `SparseMatrix(R, A)` | Return the m × n sparse matrix over ring `R` obtained by coercing the entries of sparse matrix `A` (over ring `S`) into `R`. The two forms are provided for consistency: `ChangeRing(A, R)` is consistent with other `ChangeRing` usage; `SparseMatrix(R, A)` is consistent with the creation functions where the destination ring is the first argument. | — |

---

## 27.8 Predicates

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `A eq B` | Return `true` if and only if sparse matrices `A` and `B` are equal. | — |
| `IsZero(A)` | Return `true` iff `A` is the m × n zero sparse matrix. | — |
| `IsOne(A)` | Return `true` iff square `A` is the identity sparse matrix. | — |
| `IsMinusOne(A)` | Return `true` iff square `A` is the negation of the identity sparse matrix. | — |
| `IsScalar(A)` | Return `true` iff square `A` is scalar (a scalar multiple of the identity). | — |
| `IsDiagonal(A)` | Return `true` iff square `A` is diagonal (only non-zero entries on the diagonal). | — |
| `IsSymmetric(A)` | Return `true` iff square `A` equals its transpose. | — |
| `IsUpperTriangular(A)` | Return `true` iff `A` has non-zero entries only on or above the diagonal. | — |
| `IsLowerTriangular(A)` | Return `true` iff `A` has non-zero entries only on or below the diagonal. | — |

---

## 27.9 Elementary Arithmetic

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `A + B` | Sum of m × n sparse matrices `A` and `B` over `R`. | — |
| `A - B` | Difference of m × n sparse matrices `A` and `B` over `R`. | — |
| `A * B` | Product of m × n sparse matrix `A` and n × p sparse matrix `B` over `R`; returns m × p sparse matrix. | — |
| `x * A` / `A * x` | Scalar product of ring element `x` (coercible into `R`) and sparse matrix `A`. | — |
| `-A` | Negation of sparse matrix `A`. | — |
| `A ^ -1` | Inverse of invertible square sparse matrix `A` over `R`. `R` must be a field, a Euclidean domain, or a commutative ring with exact division and characteristic 0 or > m. | — |
| `A ^ n` | Matrix power A^n for integer `n`. A^0 is the identity for any square `A`. If `n < 0`, `A` must be invertible. | — |
| `Transpose(A)` | Return the transpose of m × n sparse matrix `A`: the n × m sparse matrix whose (i, j)-th entry is the (j, i)-th entry of `A`. | — |

---

## 27.10 Multiplying Vectors or Matrices by Sparse Matrices

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `v * A` / `V * A` | Multiply dense vector `v` or dense matrix `V` (with `c` columns) by sparse c × n matrix `A`, returning the product `v·A` or `V·A`. Generally fast if `A` is sparse, and uses minimal memory. | Sparse matrix-vector product. |
| `MultiplyByTranspose(v, A)` / `MultiplyByTranspose(V, A)` | Multiply dense vector `v` or dense matrix `V` (with `c` columns) by the transpose of sparse n × c matrix `A`. Much faster than forming the transpose of `A` first. Particularly useful for iterative algorithms (e.g., Lanczos) requiring the product v·A·A^T — call `MultiplyByTranspose(v*A, A)` to avoid forming the dense A·A^T. | Transpose-implicit sparse product. |

---

## 27.11 Non-trivial Properties

### 27.11.1 Nullspace and Rowspace

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Nullspace(A)` / `Kernel(A)` | Return the nullspace (kernel) of m × n sparse matrix `A` over `R`: the R-space of all vectors `v` of length m such that `v·A = 0`. Result is given in dense representation, so both the nullity and the number of rows must be reasonably small. | Sparse elimination via Markowitz pivoting **[DEJ84, Sec. 9.2]** to reduce to a smaller dense matrix, then the dense nullspace algorithm. |
| `NullspaceMatrix(A)` / `KernelMatrix(A)` | Return a dense basis matrix `N` of the nullspace of `A` with m columns, maximum independent rows subject to `N·A = 0`. Avoids returning the nullspace as an R-space, so echelonisation of the result may be skipped. | Same sparse elimination + dense nullspace algorithm as `Nullspace`. |
| `NullspaceOfTranspose(A)` | Equivalent to `Nullspace(Transpose(A))` but more memory-efficient for large matrices, since the transpose may not need to be constructed explicitly. | Implicit transpose + sparse elimination **[DEJ84]**. |
| `Rowspace(A)` | Return the rowspace of `A` over `R` (the R-space generated by the rows of `A`). Result is in dense representation; rank and number of columns must both be reasonably small. | Sparse elimination **[DEJ84]** + dense algorithm. |

### 27.11.2 Rank

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Rank(A)` | Return the rank of m × n sparse matrix `A` over `R`. | Sparse elimination via Markowitz pivoting **[DEJ84, Sec. 9.2]** to obtain a smaller dense matrix, then the dense rank algorithm. |

---

## 27.12 Determinant and Other Properties

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Determinant(A : MonteCarloSteps := -)` | Return the determinant of square sparse matrix `A` over commutative ring `R`. Parameter `MonteCarloSteps` is passed to the dense determinant algorithm for the reduced dense matrix. | Sparse elimination via Markowitz pivoting **[DEJ84, Sec. 9.2]** to obtain a smaller dense matrix, then the dense determinant algorithm. |

### 27.12.1 Elementary Divisors (Smith Form)

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ElementaryDivisors(A)` | Return the elementary divisors of m × n matrix `A` over a Euclidean ring or field `R`. Returns the sequence [e_1, ..., e_d] (e_i | e_{i+1}) of non-zero diagonal entries of the Smith form, where d = rank(A). Over a field, always returns a sequence of d ones. Smith normal form itself is not returned (it is trivially derived from the divisors, and transformation matrices would be dense). | Sparse elimination via Markowitz pivoting **[DEJ84, Sec. 9.2]** (similar to techniques in **[HHR93]**) to obtain a smaller dense matrix, then the dense Smith form algorithm (`SmithForm`). |

### 27.12.2 Verbosity

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SetVerbose("SparseMatrix", v)` | (Procedure.) Set the verbose printing level for all sparse matrix algorithms. Legal values: `false` (= 0), `true` (= 1), 0, 1, 2, 3. | — |

---

## 27.13 Linear Systems (Structured Gaussian Elimination)

The `ModularSolution` function is designed for index-calculus-type algorithms where a large sparse linear system is constructed and then a non-trivial null vector modulo M is needed. The natural equation is v·A^T = 0 mod M (so the transpose of `A` appears).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ModularSolution(A, M)` / `ModularSolution(A, L)` | Given sparse m × n integer matrix `A` (over **Z**) and a positive integer `M` (or its factorisation sequence `L`): compute a vector `v` such that v·A^T ≡ 0 (mod M); `v` is non-zero with high probability. If possible, `v` is normalised so its first entry is 1. The first form factors `M` internally; the second accepts a precomputed factorisation sequence. Parameter `Lanczos` (default `false`): if `true`, use the Lanczos algorithm instead of the default. **Memory guidance:** 100,000 × 100,000 systems require ~500 MB; 200,000 × 200,000 require ~1.5–2.0 GB. | Default: **Structured Gaussian Elimination** **[LO91b, Sec. 5]** — recursively reduces the sparse system to a smaller denser system until almost completely dense, then calls Magma's fast dense modular nullspace algorithm. If `Lanczos := true`: **Lanczos algorithm** **[LO91b, Sec. 3]** — typically 10–50× slower than the default but uses considerably less memory. |

*Worked examples: H27E3 (extended example implementing the first stage of the basic linear sieve [COS86, LO91a] for discrete logarithms in F_p; demonstrates dynamic matrix construction with `SetEntry` and solution via `ModularSolution`; applied to F_103 (toy) and F_{10^{20}+763} (≈1000 × 1000 system solved in 0.17 seconds)).*

---

## 27.14 Bibliography

| Key | Reference |
|-----|-----------|
| **[COS86]** | D. Coppersmith, A. M. Odlyzko, and R. Schroeppel. *Discrete logarithms in GF(p).* Algorithmica, 1:1–15, 1986. |
| **[DEJ84]** | I. S. Duff, A. M. Erisman, and J. K. Reid. *Direct methods for sparse matrices.* Monographs on Numerical Analysis. Oxford University Press, 1984. |
| **[HHR93]** | George Havas, Derek F. Holt, and Sarah Rees. *Recognizing badly presented Z-modules.* Linear Algebra and its Applications, 192:137–164, 1993. |
| **[LO91a]** | B. A. LaMacchia and A. M. Odlyzko. *Computation of Discrete Logarithms in Prime Fields.* In A. J. Menezes and S. Vanstone, editors, Advances in Cryptology — CRYPTO 1990, volume 537 of LNCS, pages 616–618. Springer-Verlag, 1991. |
| **[LO91b]** | B. A. LaMacchia and A. M. Odlyzko. *Solving Large Sparse Linear Systems over Finite Fields.* In A. J. Menezes and S. Vanstone, editors, Advances in Cryptology — CRYPTO 1990, volume 537 of LNCS, pages 109–133. Springer-Verlag, 1991. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Sparse Gaussian elimination via Markowitz pivoting **[DEJ84, Sec. 9.2]** | `Nullspace`, `Kernel`, `NullspaceMatrix`, `KernelMatrix`, `NullspaceOfTranspose`, `Rowspace`, `Rank`, `Determinant`, `ElementaryDivisors` |
| Dense Smith normal form (after sparse reduction) **[DEJ84, HHR93]** | `ElementaryDivisors` |
| Structured Gaussian Elimination **[LO91b, Sec. 5]** | `ModularSolution` (default) |
| Lanczos algorithm **[LO91b, Sec. 3]** | `ModularSolution(:Lanczos := true)` |
| Basic linear sieve for discrete logarithms **[COS86, LO91a]** | `ModularSolution` (illustrated in H27E3) |
| Dynamic sparse matrix construction | `SparseMatrix()`, `SetEntry` |
| Dense-sparse product (transpose-implicit) | `MultiplyByTranspose` |
