# Chapter 148 — Hadamard Matrices

**Handbook part:** XX — Combinatorics
**Handbook pages:** 4909–4915 (PDF pages 5040–5049)

---

## Scope and overview

A *Hadamard matrix* is an n × n matrix all of whose entries are ±1, such that every pair of
rows and every pair of columns differ in exactly n/2 places. Two such matrices are considered
*equivalent* if one can be transformed into the other by row swaps, column swaps, row
negations or column negations. Deciding whether two Hadamard matrices are equivalent is hard
in general.

Magma provides specialised routines for working with Hadamard matrices. Of special note is the
introduction of a **canonical form** for such matrices, based on Brendan McKay's `nauty`
program; this yields a much faster equivalence algorithm than was previously available.
Equivalence can also be tested cheaply (but only for *inequivalence*) via a 4-profile invariant,
or fully via either the `nauty` or Leon backends.

The chapter also covers: the two Hadamard 3-designs (row and column designs) associated with a
matrix; the automorphism group of a matrix (a permutation group of degree 4n); and two
built-in databases — a standard database (matrices stored in canonical form, complete for
degree ≤ 28 plus examples up to degree 256) and a skew-symmetric database (not stored in
canonical form) — together with routines for querying and updating them.

---

## 148.1 Introduction

Introductory prose only (definitions of Hadamard matrix, equivalence, and the canonical-form
approach). No intrinsics.

---

## 148.2 Equivalence Testing

Routines for normalising Hadamard matrices, computing the canonical form, computing a cheap
invariant, and testing equivalence.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsHadamard(H)` | Returns `true` if and only if the matrix `H` is a Hadamard matrix. | Direct check of the ±1 / pairwise-difference defining property. |
| `HadamardNormalize(H)` | Given a Hadamard matrix `H`, returns a normalized matrix equivalent to `H`, created by negating rows and columns so that the first row and first column consist entirely of ones. | Row/column negation to a normalized form. |
| `HadamardCanonicalForm(H)` | Given a Hadamard matrix `H`, returns a Hadamard-equivalent matrix `H'` together with transformation matrices `X` and `Y` such that `H' = XHY`. `H'` is canonical: all matrices Hadamard-equivalent to `H` (and no others) produce the same `H'`. | Canonical form based on Brendan McKay's **`nauty`** program. |
| `HadamardInvariant(H)` | Returns a sequence `S` of integers giving the 4-profile of the Hadamard matrix `H`. All Hadamard-equivalent matrices have the same 4-profile, but inequivalent ones may too. The test can establish inequivalence cheaply but cannot establish equivalence. | 4-profile invariant. |
| `IsHadamardEquivalent(H, J : Al)` | Returns `true` if and only if Hadamard matrices `H` and `J` are equivalent. Parameter `Al` (`MonStgElt`, default `"nauty"`) selects `"Leon"` or `"nauty"`. With `"nauty"`, if the matrices are equivalent the transformation matrices `X` and `Y` (with `J = XHY`) are also returned. | `nauty`-based canonical-form comparison (default) or Leon's backtrack method. |
| `HadamardMatrixToInteger(H)` | Returns an integer that encodes the entries of `H` in a more compact form. Intended to save time when repeatedly testing for equality against the same set of matrices. | Bit-packing of the ±1 entries into an integer. |
| `HadamardMatrixFromInteger(x, n)` | Returns the Hadamard matrix of degree `n` whose encoded form is the integer `x`. Inverse of `HadamardMatrixToInteger`. | Decoding of the packed integer. |

*Worked example:* H148E1 (degree-16 matrices created from compact integer form via
`HadamardMatrixFromInteger`; `HadamardInvariant` 4-profiles; `IsHadamardEquivalent` returning
transformation matrices; counting inequivalence classes via `HadamardCanonicalForm`).

---

## 148.3 Associated 3-Designs

Each row and each column of a Hadamard matrix gives rise to an associated Hadamard 3-design.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HadamardRowDesign(H, i)` | Given an n × n Hadamard matrix `H` (with n ≥ 4) and an integer `i`, `1 ≤ i ≤ n`, returns the Hadamard 3-design corresponding to the i-th row of `H`. | Standard Hadamard 3-design construction from a row. |
| `HadamardColumnDesign(H, i)` | Given an n × n Hadamard matrix `H` (with n ≥ 4) and an integer `i`, `1 ≤ i ≤ n`, returns the Hadamard 3-design corresponding to the i-th column of `H`. | Standard Hadamard 3-design construction from a column. |

*Worked example:* H148E2 (the unique 8 × 8 Hadamard class; `HadamardRowDesign(H, 3)` and
`HadamardColumnDesign(H, 8)` yielding 3-(8, 4, 1) designs with 14 blocks).

---

## 148.4 Automorphism Group

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HadamardAutomorphismGroup(H : Al)` | Given a Hadamard matrix `H` of degree `n`, returns the automorphism group of `H` as a permutation group of degree `4n`. Parameter `Al` (`MonStgElt`, default `"nauty"`) selects `"Leon"` or `"nauty"`. | Graph-automorphism computation via `nauty` (default) or Leon. |

---

## 148.5 Databases

Magma contains two databases of Hadamard matrices. The first (standard) database includes all
inequivalent matrices of degree at most 28, and examples of matrices of all degrees up to 256.
The representatives used are the canonical forms (as output by `HadamardCanonicalForm`), and
matrices of a given degree are ordered lexicographically (with 1 considered less than −1 for
this ordering). A database of skew-symmetric Hadamard matrices also exists; in this case the
matrices are **not** stored in canonical form, since canonical forms are not skew-symmetric.
With the exception of the creation routines, the intrinsics below apply to both databases.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HadamardDatabase()` | Returns the database of Hadamard matrices (the standard database). | Database accessor. |
| `SkewHadamardDatabase()` | Returns the database of skew-symmetric Hadamard matrices. | Database accessor. |
| `Matrix(D, n, k)` | Returns the k-th matrix of degree `n` in the database `D`. | Database lookup. |
| `Matrices(D, n)` | Returns the sequence of all matrices of degree `n` stored in the database `D`. | Database lookup. |
| `DegreeRange(D)` | Returns the smallest and largest degrees of matrices in the database `D`. | Database metadata. |
| `Degrees(D)` | Returns the sequence of degrees for which there is at least one matrix of that degree in the database `D`. | Database metadata. |
| `NumberOfMatrices(D, n)` | Returns the number of matrices of degree `n` in the database `D`. | Database metadata. |

*Worked example:* H148E3 (`HadamardDatabase`; `Matrix(D, 16, 3)`; `NumberOfMatrices`,
`Degrees`, and per-degree counts of the standard database).

### 148.5.1 Updating the Databases

The databases are noticeably incomplete (for example, more than 60 000 Hadamard matrices are
known for degree 32 whereas only 23 are in the database; and only matrices of degrees 36, 44 or
52 are present in the skew database). The Magma group welcomes contributions of matrices not
equivalent to those already present. The functions in this section let users create new
versions of the databases locally. They operate on a record whose format is not described and
which should only be manipulated by these functions.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HadamardDatabaseInformation(D : Canonical)` | Takes an existing Hadamard database `D` and extracts its information into an internal record form (returned) used by the other update intrinsics. Parameter `Canonical` (`BoolElt`, default `true`) indicates whether the database entries are known to be canonical (`true` for the standard database; set `false` for the skew or any non-canonical database). The value also controls whether the new record stores canonical or original forms: to extract from a canonical database but store non-canonically, create with `Canonical := false` (here or via `HadamardDatabaseInformationEmpty`) and add matrices with `UpdateHadamardDatabase` and `Canonical := true`. | Database extraction into the internal update record. |
| `HadamardDatabaseInformationEmpty(: Canonical)` | Returns the internal data corresponding to an empty database, allowing creation of a new database (or a slice of an existing one) without including all of a previous one. Parameter `Canonical` (`BoolElt`, default `true`) indicates whether entries should be written out in canonical form. | Empty update record. |
| `UpdateHadamardDatabase(~R, S : Canonical)` | Augments the database record `R` (passed by reference) with the matrices in the sequence `S`. Matrices are added only if inequivalent to those already present, which requires computing canonical forms (can be expensive). Parameter `Canonical` (`BoolElt`, default `false`): set `true` if the matrices in `S` are already known to be in canonical form. If matrices of matching degree already in `R` are not known to be canonical, their canonical forms must also be computed (troublesome for the skew database; see `WriteRawHadamardData`). | Inequivalence-filtered insertion via `HadamardCanonicalForm`. |
| `WriteHadamardDatabase(S, ~R)` | Creates the database files *name*`.dat` and *name*`.ind` from the database data `R`, where *name* is taken from the string `S`. Since canonical forms may need to be computed in the process, the data is passed by reference (`~R`) so this computation is not lost (e.g. if one writes the database, adds more matrices, and writes again). | Writes `.dat`/`.ind` files; may invoke canonical-form computation. |
| `WriteRawHadamardData(S, R)` | Saves the data in `R` to the file named by the string `S`. When loaded, the file defines a single variable `data` behaving identically to `R`. Desirable for non-canonical databases since canonical forms then need not be recomputed. Destroys the original contents, if any, of the file. | Raw serialisation of the database record. |
| `SetVerbose("HadamardDB", v)` | Procedure: sets the verbose printing level for the Hadamard database update routines. `v` should be an integer in the range 0 to 3, giving progress indications during long updates. | Verbosity control. |

*Worked example:* H148E4 (building a local database from a `matrixfile`: `HadamardDatabase`,
`HadamardDatabaseInformation`, `SetVerboseLevel("HadamardDB", 1)`, `UpdateHadamardDatabase`,
`WriteHadamardDatabase` to `hadamard.dat`/`hadamard.ind`, and reloading via `SetLibraryRoot`).

---

## 148.6 Bibliography (canonical references)

The chapter cites no formal bibliography entries. The only named external work is Brendan
McKay's **`nauty`** program (the basis of `HadamardCanonicalForm`, `IsHadamardEquivalent` and
`HadamardAutomorphismGroup`), and Leon's backtrack method (the alternative `"Leon"` backend).
Neither is given a formal bibliography key in this chapter.

| Key | Reference |
|-----|-----------|
| — | B. D. McKay. *`nauty`* — graph canonical-labelling and automorphism program. (Cited as the basis of the canonical-form and `nauty`-backend routines; no formal key in the chapter.) |
| — | J. S. Leon. Backtrack-search equivalence/automorphism method (the `"Leon"` backend; no formal key in the chapter.) |

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| `nauty` canonical form / equivalence / automorphism | `HadamardCanonicalForm`, `IsHadamardEquivalent` (default), `HadamardAutomorphismGroup` (default) |
| Leon backtrack method | `IsHadamardEquivalent(: Al := "Leon")`, `HadamardAutomorphismGroup(: Al := "Leon")` |
| 4-profile invariant (cheap inequivalence test) | `HadamardInvariant` |
| Row/column negation normalization | `HadamardNormalize` |
| Compact integer encoding of ±1 entries | `HadamardMatrixToInteger`, `HadamardMatrixFromInteger` |
| Hadamard 3-design construction | `HadamardRowDesign`, `HadamardColumnDesign` |
| Database query | `HadamardDatabase`, `SkewHadamardDatabase`, `Matrix`, `Matrices`, `DegreeRange`, `Degrees`, `NumberOfMatrices` |
| Database creation / update | `HadamardDatabaseInformation`, `HadamardDatabaseInformationEmpty`, `UpdateHadamardDatabase`, `WriteHadamardDatabase`, `WriteRawHadamardData`, `SetVerbose("HadamardDB", ·)` |
