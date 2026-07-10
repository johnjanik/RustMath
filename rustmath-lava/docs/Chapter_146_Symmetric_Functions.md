# Chapter 146 — Symmetric Functions

**Handbook part:** XX — Combinatorics
**Handbook pages:** 4847–4870 (PDF pages 4980–5004)

---

## Scope and overview

A *symmetric function* is a polynomial invariant under permutations of its indeterminates. The
symmetric functions over a commutative ring with unity, with an arbitrary number of
indeterminates, form an algebra denoted Λ. Those of fixed degree `n` form a submodule Λⁿ. There
are five standard bases of Λ, each indexed by partitions: the **Schur** (`s_λ`), **Homogeneous**
(`h_λ`), **Power Sum** (`p_λ`), **Elementary** (`e_λ`) and **Monomial** (`m_λ`) symmetric
functions. The size of any basis of Λⁿ equals the number of partitions of weight `n`. Magma can
compute in any of the five bases and convert freely between them. The main reference is
Macdonald **[Mac95]**.

The theory rests on *Young tableaux*. For a partition λ the Schur function is `s_λ = Σ_T x^T`
summed over Young tableaux of shape λ (rows weakly increasing, columns strictly increasing). The
other bases are defined from the Schur functions / monomials: `e_k = s_[1^k]`, `h_k = s_[k]`,
`p_k = m_[k]` (the `k`-th power sum), and `m_λ` is the orbit sum of `x^λ` under the symmetric
group; products over the parts of λ extend these to all partitions. A symmetric function has an
image as a symmetric polynomial in any number of indeterminates (Magma works with
finite-rank polynomial rings); normally the number of indeterminates used equals the degree.

An inner product is defined on Λ by `⟨m_λ, h_λ'⟩ = δ_{λ,λ'}` (the monomial and homogeneous
bases are dual). With respect to it the Schur functions are an orthonormal basis and the power
sum basis is orthogonal. The algebra category is `AlgSym`; its elements form category
`AlgSymElt`.

The chapter also covers transition (change-of-basis) matrices between the five bases: those
among the `s`, `h`, `m`, `e` bases are integer matrices, while any change to/from the `p` basis
is over the rationals (Macdonald **[Mac95]**, pages 54–58). Several matrices carry
representation-theoretic meaning (Kostka numbers, the symmetric-group character table).

---

## 146.1 Introduction

Introductory material defining symmetric functions, partitions, weight, Young tableaux / Ferrers
diagrams, the correspondence of a tableau `T` to a monomial `x^T`, and the five bases. No
intrinsics; see the overview above.

*Worked example:* H146E1 (correspondence between a Schur function `S.[2,1]` and its symmetric
polynomial expansion in 2, 3, 4 and 5 indeterminates).

---

## 146.2 Creation

### 146.2.1 Creation of Symmetric Function Algebras

An algebra of symmetric functions is defined by specifying its coefficient ring (a commutative
ring with unity); a separate constructor exists for each of the five standard bases, plus a
general constructor selecting the basis by a parameter.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SymmetricFunctionAlgebra(R)` / `SFA(R)` | Algebra of symmetric functions over ring `R`. Parameter `Basis` (`MonStgElt`, default `"Schur"`) selects the basis: one of `"Schur"`, `"Homogeneous"`, `"PowerSum"`, `"Elementary"`, `"Monomial"`. | — |
| `SymmetricFunctionAlgebraSchur(R)` / `SFASchur(R)` | Algebra of symmetric functions over commutative ring `R` with elements expressed in the *Schur* basis (indexed by partitions). | — |
| `SymmetricFunctionAlgebraHomogeneous(R)` / `SFAHomogeneous(R)` | As above, in the *Homogeneous* basis. | — |
| `SymmetricFunctionAlgebraPower(R)` / `SFAPower(R)` | As above, in the *Power Sum* basis. | — |
| `SymmetricFunctionAlgebraElementary(R)` / `SFAElementary(R)` | As above, in the *Elementary* basis. | — |
| `SymmetricFunctionAlgebraMonomial(R)` / `SFAMonomial(R)` | As above, in the *Monomial* basis. | — |

*Worked example:* H146E2 (a polynomial expressed in Schur and Elementary bases agree as
polynomials).

### 146.2.2 Creation of Symmetric Functions

Basis elements are indexed by partitions (weakly decreasing positive sequences); the *weight* of
a partition is its degree. General symmetric functions are linear combinations of basis elements,
built directly, by coercion from another basis, or by coercion from a (symmetric) polynomial or
scalar.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `A . P` | For a partition `P` (weakly decreasing positive sequence): the basis element of algebra `A` indexed by `P`. | — |
| `A . i` | For a positive integer `i`: the basis element of `A` indexed by the partition `[i]`. | — |
| `IsCoercible(A, f)` / `A ! f` | Coerce a multivariate polynomial `f` that is symmetric in all its indeterminates into algebra `A`, expressed in `A`'s basis. Errors (`Polynomial is not symmetric`) if `f` is not symmetric. | — |
| `A ! m` | Coerce a symmetric function `m` (possibly in a different basis) into `A`, expressed in `A`'s basis. | Change of basis (see §146.5). |
| `A ! r` | Create the scalar element `r` in the symmetric function algebra `A`. | — |

*Worked examples:* H146E3 (linear combinations of monomial basis elements); H146E4 (coercing
symmetric polynomials; a non-symmetric polynomial fails); H146E5 (a scalar element `P!3` and its
parent); H146E6 (the same symmetric function `S.[3,1]` expressed in all five bases and as a
polynomial); H146E7 (the homogeneous `h_k` as the sum of all monomial functions of weight `k`).

---

## 146.3 Structure Operations

### 146.3.1 Related Structures

The main related structure of an algebra of symmetric functions is its coefficient ring. The
algebra belongs to the Magma category `AlgSym`.

| Intrinsic | Description |
|-----------|-------------|
| `BaseRing(L)` / `CoefficientRing(L)` | The coefficient ring of the algebra of symmetric functions `L`. |
| `Category(L)` / `Parent(L)` / `PrimeRing(L)` | The category, parent and prime ring of `L`. |

### 146.3.2 Ring Predicates and Booleans

The usual ring boolean predicates are available.

| Intrinsic | Description |
|-----------|-------------|
| `IsCommutative(L)`, `IsUnitary(L)`, `IsFinite(L)`, `IsOrdered(L)`, `IsField(L)`, `IsEuclideanDomain(L)`, `IsPID(L)`, `IsUFD(L)`, `IsDivisionRing(L)`, `IsEuclideanRing(L)`, `IsDomain(L)`, `IsPrincipalIdealRing(L)` | Standard ring predicates on the algebra `L`. |
| `L eq M` / `L ne M` | Whether `L` and `M` are (not) equal. Two algebras are equal if they are over the same ring. |

### 146.3.3 Predicates on Basis Types

| Intrinsic | Description |
|-----------|-------------|
| `HasSchurBasis(A)` | `true` if `A` is an algebra with a Schur basis. |
| `HasHomogeneousBasis(A)`, `HasElementaryBasis(A)`, `HasPowerSumBasis(A)`, `HasMonomialBasis(A)` | `true` if `A` has a homogeneous, elementary, power sum or monomial basis respectively. |

---

## 146.4 Element Operations

### 146.4.1 Parent and Category

The category for elements of algebras of symmetric functions is `AlgSymElt`.

| Intrinsic | Description |
|-----------|-------------|
| `Parent(f)` / `Category(f)` | The parent algebra and category of element `f`. |

### 146.4.2 Print Styles

By default elements print using a lexicographical ordering on the indexing partitions (for
partitions of equal weight, the reverse of `Partitions(w)`; basis elements indexed by partitions
with smaller entries print first). The linear combination can be re-ordered.

| Intrinsic | Description |
|-----------|-------------|
| `A ' PrintStyle` | Retrieve or set the print style of algebra `A`. Default `"Lex"` (lexicographical, as above); other options `"Length"` (longest partition first) and `"MaximalPart"` (greatest maximal part first). |

*Worked example:* H146E8 (printing the same element under `"Lex"`, `"Length"` and
`"MaximalPart"`).

### 146.4.3 Additive Arithmetic Operators

The usual unary and binary ring operations are available; elements of different algebras may be
combined (if the coefficient rings are compatible). When operands are in different bases, the
result is written in the basis of the *second* operand.

| Intrinsic | Description |
|-----------|-------------|
| `+ a` / `- a` | Unary plus / negation. |
| `a + b` / `a - b` | Sum / difference. |
| `a +:= b` / `a -:= b` | In-place sum / difference. |

### 146.4.4 Multiplication

When operands are in different bases the result is written in the basis of the second operand;
the multiplication algorithm depends on the bases involved. The degree of the result is the sum
of the degrees of the operands.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `a * b` / `a *:= b` | Product of symmetric functions `a` and `b`. | If `b` is in the Schur basis and `a` in power sum / elementary / monomial, **Muir's rule [Mui60]** is used. If `a` is homogeneous, the **Pieri rule [Mac95]** is used; if `a` is also Schur, the method of **Schubert polynomials [LS85]** is used. Special algorithms apply when `b` is monomial and `a` is homogeneous / elementary / monomial. When both `a` and `b` are homogeneous, elementary or power sum, multiplication merges the parts of the partitions (from `f_λ = Π_i f_{λ_i}`). Otherwise `a` is coerced into the parent of `b` first. |
| `a ^ k` | The `k`-th power of `a`. | Repeated multiplication. |

*Worked example:* H146E9 (`m.[3]*s.[2,1]` in the Schur basis; merging of partitions via
`E.4*E.3*E.1 = E.[4,3,1]`).

### 146.4.5 Plethysm

Plethysm, also called composition of symmetric functions.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `a ~ b` | The plethysm (composition) of symmetric functions `a` and `b`; result given in the basis of the second operand. The degree of the result is the *product* of the degrees of the operands (may be very large). | — |

*Worked example:* H146E10 (`m.[3]~s.[2,1]` expanded in the Schur basis).

### 146.4.6 Boolean Operators

| Intrinsic | Description |
|-----------|-------------|
| `IsHomogeneous(s)` | `true` if the partitions indexing the basis elements of `s` are all of the same weight (each term has the same degree, so the polynomial expansion of `s` is homogeneous). |
| `IsZero(s)`, `IsOne(s)`, `IsMinusOne(s)` | Whether `s` is zero, one or minus one. |
| `s eq t` / `s ne t` | Whether `s` and `t` are (not) the same. |

### 146.4.7 Accessing Elements

| Intrinsic | Description |
|-----------|-------------|
| `Coefficient(s, p)` | The coefficient of the basis element `A_p` in `s` (`A` the parent of `s`, `p` a partition as a sequence); may be zero. |
| `Support(s)` | Two parallel sequences: the partitions indexing the basis elements and their coefficients in `s`. |
| `Length(s)` | The number of basis elements with non-zero coefficient in `s`, in the current basis. |
| `Degree(s)` | The degree of `s`: the maximal weight of the indexing partitions of basis elements with non-zero coefficient. |

*Worked example:* H146E11 (decomposing an element via `Support` and reconstructing it via
`Length` and the `.s[i]` accessor).

### 146.4.8 Multivariate Polynomials

A symmetric function may be viewed as a polynomial in any number of variables.

| Intrinsic | Description |
|-----------|-------------|
| `P ! s` | The polynomial expansion of symmetric function `s` in the polynomial ring `P`. |

*Worked examples:* H146E12 (`S.[3,1]` over `GF(7)` expanded in 5 variables; recovering the
elementary expansion via `IsSymmetric`); H146E13 (using polynomial expansion to change the
alphabet of a symmetric function, e.g. substituting `x_i → x_i + 1`, then coercing back via
`IsCoercible`).

### 146.4.9 Frobenius Homomorphism

The automorphism mapping the elementary symmetric function to the homogeneous symmetric function.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Frobenius(s)` | The image of `s` under the Frobenius automorphism; the image has the same parent as `s`. When power sum functions are involved it may be necessary to use a coefficient ring allowing division by an integer. | On the Schur functions, conjugates the indexing partitions. |

*Worked example:* H146E14 (Frobenius automorphism on Schur functions conjugates the indexing
partitions, checked via `ConjugatePartition`).

### 146.4.10 Inner Product

The inner product on Λ is `⟨m_λ, h_λ'⟩ = δ_{λ,λ'}` (monomial and homogeneous bases dual); this is
the inner product used by Magma.

| Intrinsic | Description |
|-----------|-------------|
| `InnerProduct(a, b)` | The inner product of symmetric functions `a` and `b`. |

*Worked example:* H146E15 (`InnerProduct(E.p, H.pc)` for a partition `p` and its conjugate `pc`
equals 1; relevant to irreducible representations of the symmetric group).

### 146.4.11 Combinatorial Objects

The Schur function is the generating function of standard tableaux, so the corresponding tableaux
can be recovered.

| Intrinsic | Description |
|-----------|-------------|
| `Tableaux(sf, m)` | For a Schur function `sf` over the integers with positive coefficients: the multiset of tableaux, with maximal entry `m`, for which `sf` is the generating function. |

### 146.4.12 Symmetric Group Character

A Schur function indexed by a single partition (a basis element) corresponds to an irreducible
character of the symmetric group.

| Intrinsic | Description |
|-----------|-------------|
| `SymmetricCharacter(sf)` | For `sf` in an algebra of symmetric functions: a linear combination of irreducible characters of the symmetric group whose coefficients are the coefficients of `sf` with respect to the Schur basis. |

*Worked example:* H146E16 (a representation-theory result on induced characters of Young
subgroups, verified with `SymmetricCharacter`, `ConjugatePartition` and `InnerProduct`).

### 146.4.13 Restrictions

Form symmetric functions whose support is a subset of that of a given symmetric function.

| Intrinsic | Description |
|-----------|-------------|
| `RestrictDegree(a, n)` | The linear combination of basis elements of `a` of degree `n` (the restriction to Λⁿ). Parameter `Exact` (`BoolElt`, default `true`); if `false`, basis elements of degree `≤ n` are kept (restriction to `⋃_{k≤n} Λᵏ`). |
| `RestrictPartitionLength(a, n)` | The linear combination of basis elements of `a` whose indexing partitions have length `n`. Parameter `Exact` (default `true`); if `false`, length `≤ n`. |
| `RestrictParts(a, n)` | The linear combination of basis elements of `a` whose indexing partitions have maximal part `n`. Parameter `Exact` (default `true`); if `false`, maximal part `≤ n`. |

---

## 146.5 Transition Matrices

Change-of-basis matrices between the five bases. The matrices among the `s`, `h`, `m`, `e` bases
are integer matrices; any change to/from the `p` basis is over the rationals (**[Mac95]**, pages
54–58). Rows/columns are indexed by partitions of weight `n` in the order given by
`Partitions(n)`.

### 146.5.1 Transition Matrices from Schur Basis

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SchurToMonomialMatrix(n)` | Matrix expanding a Schur function (partition of weight `n`) as a sum of monomial functions. This is the table of **Kostka numbers** (numbers of tableaux of each shape and content). Entries are non-negative integers; upper triangular. | — |
| `SchurToHomogeneousMatrix(n)` | Matrix expanding a Schur function as a sum of homogeneous functions. Entries are integers (positive and negative); lower triangular. It is the transpose of `MonomialToSchurMatrix(n)`. | — |
| `SchurToPowerSumMatrix(n)` | Matrix expanding a Schur function as a sum of power sum functions. Entries are rationals. | — |
| `SchurToElementaryMatrix(n)` | Matrix expanding a Schur function as a sum of elementary functions. Entries are integers (positive and negative); upper left triangular. | — |

*Worked examples:* H146E17 (`SchurToMonomialMatrix(5)`, entries as Kostka numbers, checked
against `TableauxOnShapeWithContent`); H146E18 (`SchurToPowerSumMatrix(4)` and the action of the
base-change matrix versus coercion).

### 146.5.2 Transition Matrices from Monomial Basis

| Intrinsic | Description |
|-----------|-------------|
| `MonomialToSchurMatrix(n)` | Matrix expanding a monomial function (weight `n`) as a sum of Schur functions. Entries are integers (positive and negative); upper triangular. Transpose of `SchurToHomogeneousMatrix(n)`. |
| `MonomialToHomogeneousMatrix(n)` | Matrix expanding a monomial function as a sum of homogeneous functions. Entries are positive and negative integers. |
| `MonomialToPowerSumMatrix(n)` | Matrix expanding a monomial function as a sum of power sum functions. Entries are rationals; lower triangular. |
| `MonomialToElementaryMatrix(n)` | Matrix expanding a monomial function as a sum of elementary functions. Entries are positive and negative integers; upper left triangular. |

### 146.5.3 Transition Matrices from Homogeneous Basis

| Intrinsic | Description |
|-----------|-------------|
| `HomogeneousToSchurMatrix(n)` | Matrix expanding a homogeneous function (weight `n`) as a sum of Schur functions. Entries are positive integers; lower triangular. |
| `HomogeneousToMonomialMatrix(n)` | Matrix `M` expanding a homogeneous function as a sum of monomial functions. Entries are positive integers (no zero entries). The coefficient `M_{μ,λ}` in `h_λ = Σ_μ M_{μ,λ} m_μ` is the number of non-negative integer matrices with row sums `λ_i` and column sums `μ_j` (**[Mac95]**, page 57). |
| `HomogeneousToPowerSumMatrix(n)` | Matrix expanding a homogeneous function as a sum of power sum functions. Entries are positive rationals; upper triangular. |
| `HomogeneousToElementaryMatrix(n)` | Matrix expanding a homogeneous function as a sum of elementary functions. Entries are integers; upper triangular. |

*Worked examples:* H146E19 (`SchurToMonomialMatrix(7)` equals the transpose of
`HomogeneousToSchurMatrix(7)`); H146E20 (`HomogeneousToMonomialMatrix(7)` is symmetric); H146E21
(`HomogeneousToElementaryMatrix(7)` equals `ElementaryToHomogeneousMatrix(7)`).

### 146.5.4 Transition Matrices from Power Sum Basis

| Intrinsic | Description |
|-----------|-------------|
| `PowerSumToSchurMatrix(n)` | Matrix expanding a power sum function (weight `n`) as a sum of Schur functions. Entries are positive and negative integers. This matrix is the **character table of the symmetric group**. |
| `PowerSumToMonomialMatrix(n)` | Matrix expanding a power sum function as a sum of monomial functions. Entries are positive integers; lower triangular. |
| `PowerSumToHomogeneousMatrix(n)` | Matrix expanding a power sum function as a sum of homogeneous functions. Entries are integers; upper triangular. |
| `PowerSumToElementaryMatrix(n)` | Matrix expanding a power sum function as a sum of elementary functions. Entries are integers; upper triangular. |

*Worked example:* H146E22 (`PowerSumToSchurMatrix(5)` compared to `CharacterTable(Sym(5))`: first
column ↔ unity character, last column ↔ alternating character, last row ↔ irreducible-character
dimensions).

### 146.5.5 Transition Matrices from Elementary Basis

| Intrinsic | Description |
|-----------|-------------|
| `ElementaryToSchurMatrix(n)` | Matrix expanding an elementary function (weight `n`) as a sum of Schur functions. Entries are positive integers. |
| `ElementaryToMonomialMatrix(n)` | Matrix `M` expanding an elementary function as a sum of monomial functions. Entries are positive integers. The coefficient `M_{μ,λ}` in `e_λ = Σ_μ M_{μ,λ} m_μ` is the number of 0-1 integer matrices with row sums `λ_i` and column sums `μ_j` (**[Mac95]**, page 57). |
| `ElementaryToHomogeneousMatrix(n)` | Matrix expanding an elementary function as a sum of homogeneous functions. |
| `ElementaryToPowerSumMatrix(n)` | Matrix expanding an elementary function as a sum of power sum functions. |

*Worked example:* H146E23 (`ElementaryToMonomialMatrix(7)` is symmetric).

---

## 146.6 Bibliography (canonical references)

| Key | Reference |
|-----|-----------|
| **[LS85]** | Alain Lascoux and Marcel-Paul Schützenberger. *Schubert polynomials and the Littlewood-Richardson rule.* Lett. Math. Phys., **10**(2-3):111–124, 1985. |
| **[Mac95]** | I. G. Macdonald. *Symmetric functions and Hall polynomials.* The Clarendon Press / Oxford University Press, New York, second edition, 1995. With contributions by A. Zelevinsky, Oxford Science Publications. |
| **[Mui60]** | Thomas Muir. *A treatise on the theory of determinants.* Dover Publications Inc., New York, 1960. |

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Algebra creation (five bases) | `SFASchur`, `SFAHomogeneous`, `SFAPower`, `SFAElementary`, `SFAMonomial`, `SymmetricFunctionAlgebra(:Basis)` |
| Muir's rule **[Mui60]** (multiply into Schur basis from power sum / elementary / monomial) | `a * b` |
| Pieri rule **[Mac95]** (multiply homogeneous into another basis) | `a * b` |
| Schubert polynomials **[LS85]** (multiply Schur by homogeneous) | `a * b` |
| Partition-merging multiplication (`f_λ = Π f_{λ_i}`) | `a * b` (homogeneous / elementary / power sum operands) |
| Plethysm / composition | `a ~ b` |
| Frobenius automorphism (Schur partition conjugation) | `Frobenius` |
| Inner product `⟨m_λ, h_λ'⟩ = δ` **[Mac95]** | `InnerProduct` |
| Schur ↔ tableaux generating function | `Tableaux` |
| Schur ↔ symmetric-group irreducible characters | `SymmetricCharacter`, `PowerSumToSchurMatrix` |
| Kostka numbers (Schur → monomial) | `SchurToMonomialMatrix` |
| Counting integer / 0-1 matrices by row/column sums **[Mac95]** | `HomogeneousToMonomialMatrix`, `ElementaryToMonomialMatrix` |
| Change-of-basis matrices **[Mac95]** | `*To*Matrix` family (Schur / Monomial / Homogeneous / PowerSum / Elementary) |
| Polynomial expansion / alphabet change | `P ! s`, `A ! f` (`IsCoercible`) |
