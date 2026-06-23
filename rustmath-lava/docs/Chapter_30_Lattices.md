# Chapter 30 — Lattices

**Handbook part:** V — Lattices and Quadratic Forms
**Handbook pages:** 643–715 (PDF pages 772–849)

---

## Scope and overview

A **lattice** in Magma is a free Z-module of rank m embedded in Q^n or R^n, equipped with a
positive definite inner product (v, w) = vMw^tr for a positive definite matrix M. The two pieces
of data that completely determine a lattice are the basis matrix B and the inner product matrix M;
all other invariants (Gram matrix F = BMB^tr, determinant, etc.) are derived from these.

Magma distinguishes **exact lattices** (entries of B and M in Z or Q) from **non-exact lattices**
(entries in an approximate real field). Many operations are available only for exact lattices.
Internally, a LLL-reduced basis is computed and cached whenever needed; this makes lattice
operations considerably more efficient than the Hermite-form reductions used by R-spaces.

Central algorithms in this chapter:

- **LLL basis reduction** — the Lenstra–Lenstra–Lovász algorithm **[LLL82]**, implemented as
  the provable floating-point L2 algorithm of Nguyen–Stehlé **[NS09]** and the exact integral
  method of de Weger **[dW87]**, with eight internal variants chosen automatically. The default
  is provably correct; `Proof := false` selects faster heuristic variants.
- **Vector enumeration** — the Fincke–Pohst / Kannan / Schnorr–Euchner enumeration of all
  lattice vectors within a hyperball **[FP83, Kan83, SE94]**; this underlies shortest/closest
  vector computation, kissing numbers, theta series, etc. Tree pruning **[SE94, SH95]** is
  available to trade correctness for speed.
- **Hermite–Korkine–Zolotarev (HKZ) reduction** — a stronger reduction than LLL, based on
  iteratively solving the Shortest Vector Problem in each projected sublattice.
- **Genus / spinor-genus enumeration and neighbour graphs** — p-neighbour graph exploration
  following Kneser **[Kne57]** and Schulze-Pillot **[SP91]** for enumerating isometry classes.

The chapter also covers Voronoi cells and covering radii, orthogonalisation, genera and spinor
genera, a database of lattices (Nebe–Sloane catalogue **[NS01a, NS01b]**), and Hermitian/
quaternionic lattices.

---

## 30.1 Introduction

Lattices arise in representation theory, coding theory, geometry, and algebraic number theory.
The Magma implementation centres on the fast LLL routine and a highly efficient enumeration
algorithm for short and close vectors. The LLL-reduced basis is computed on demand and cached.
The Nebe–Sloane Catalogue of Lattices is directly accessible within Magma.

*(No intrinsics in this section.)*

---

## 30.2 Presentation of Lattices

Describes the mathematical presentation: basis matrix B (m × n), inner product matrix M (n × n),
Gram matrix F = BMB^tr, coordinate lattice, exact vs. non-exact, and compatible lattices.

*(No intrinsics — mathematical background only.)*

---

## 30.3 Creation of Lattices

### 30.3.1 Elementary Creation of Lattices

Two modes: (a) supply a **generating matrix** (rows need not be independent; Magma LLL-reduces and
removes zeros to form the basis); (b) supply a **basis matrix** directly (independent rows, stored
as-is). A Gram matrix may alternatively specify a lattice directly.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Lattice(X, M)` / `Lattice(n, Q, M)` / `Lattice(S, M)` | Construct the lattice from generating matrix (given as matrix X, integer n + flat sequence Q, or R-space S) with inner product matrix M. LLL-reduces the generating matrix to obtain the basis. Parameter `CheckPositive` (default `true`) verifies M is positive definite. | LLL reduction of the generating matrix **[LLL82]**. |
| `Lattice(X)` / `Lattice(n, Q)` / `Lattice(S)` | As above but with the standard Euclidean inner product. | LLL reduction **[LLL82]**. |
| `LatticeWithBasis(B, M)` / `LatticeWithBasis(n, Q, M)` / `LatticeWithBasis(S, M)` | Construct the lattice with a given basis matrix B (independent rows) and inner product matrix M; basis is *not* LLL-reduced. Parameters `CheckIndependent` (default `true`) and `CheckPositive` (default `true`). | Direct basis specification; no reduction. |
| `LatticeWithBasis(B)` / `LatticeWithBasis(n, Q)` / `LatticeWithBasis(S)` | As above with standard Euclidean inner product. | Direct basis specification; no reduction. |
| `LatticeWithGram(F)` / `LatticeWithGram(n, Q)` | Construct a lattice with standard basis and the given Gram matrix F (also the inner product matrix). F may be given as a symmetric n × n matrix or as a sequence of length n² or C(n+1,2). Parameter `CheckPositive` (default `true`). | Direct Gram specification. |
| `StandardLattice(n)` | The standard lattice Z^n with standard Euclidean inner product. | — |
| `CoordinateLattice(L)` | The lattice with identity basis matrix but the same Gram matrix as L. The embedding gives an isometry. | — |
| `ScaledLattice(L, n)` | The coordinate lattice with Gram matrix of L scaled by integer/rational n. The embedding gives a similitude. | — |

*Worked example: H30E1 (creating a rank-2 lattice in Z³ via Lattice and LatticeWithBasis; comparing LLL-reduced vs. original basis; LatticeWithGram).*

### 30.3.2 Lattices from Linear Codes

Standard constructions A and B convert linear codes over prime fields into lattices. Structural
invariants of the codes yield estimates for the lattice invariants.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Lattice(C, "A")` | For a linear code C of length n over F_p: the full preimage of C under the map Z^n → F_p^n. | Construction A. |
| `Lattice(C, "B")` | For a linear code C of length n over F_p with all codewords having coordinate sum 0: the lattice of vectors reducing mod p to a codeword and with coordinate sum 0 mod p². Inner product matrix is the identity divided by an appropriate scalar to give an integral primitive Gram matrix. | Construction B. |

*Worked example: H30E2 (16-dimensional Barnes-Wall lattice Λ₁₆ from the first-order Reed–Muller code via Construction B).*

### 30.3.3 Lattices from Algebraic Number Fields

For a number field K of degree n with r real and t pairs of complex embeddings, the Minkowski map
K → R^n (with complex coordinates rescaled by √2) gives the T₂-norm as Euclidean norm; orders
and ideals become positive definite lattices.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `MinkowskiLattice(O)` / `Lattice(O)` | For an order O in a number field of degree n: the lattice in R^n given by the Minkowski images of the basis of O, together with the isomorphism O → L. | Minkowski embedding. |
| `MinkowskiLattice(I)` / `Lattice(I)` | For an ideal I of an order O in a number field of degree n: the lattice in R^n generated by Minkowski images of the basis of I, together with the isomorphism. | Minkowski embedding. |
| `MinkowskiSpace(K)` | For a number field K: the real inner product space V with inner product given by the T₂-norm, together with the Minkowski map K → V. | Minkowski embedding. |

*Worked example: H30E3 (lattice of the equation order of Q(∛15); T₂-norm vs. lattice norm agreement).*

### 30.3.4 Special Lattices

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Lattice(X, n)` | For family name X ∈ {"A","B","C","D","E","F","G","Kappa","Lambda"} and integer n: constructs the corresponding named lattice (root lattices A_n through G_2, Kappa-lattices K_n, laminated lattices Λ_n). Inner product matrix is chosen so the Gram matrix is integral and primitive. Supported ranges documented per family (e.g. E: 6 ≤ n ≤ 8; Lambda: 1 ≤ n ≤ 31 including Leech lattice Λ₂₄ at n=24 and Barnes-Wall Λ₁₆ at n=16). | Standard lattice constructions; inner product scaled to avoid irrational entries. |

---

## 30.4 Lattice Elements

Lattice elements are row vectors. Most R-space element operations apply.

### 30.4.1 Creation of Lattice Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `L . i` | The i-th basis element of the current basis of L. | — |
| `L ! Q` / `elt< L \| Q >` | For a sequence Q of length n = Degree(L): the lattice element with those entries. The vector must lie in L. | Membership test. |
| `CoordinatesToElement(L, C)` / `Coordelt(L, C)` | For a sequence or vector C = [c₁,…,c_m]: the lattice element c₁·b₁ + … + c_m·b_m. | Linear combination of basis vectors. |
| `L ! 0` / `Zero(L)` | The zero element of L. | — |

### 30.4.2 Operations on Lattice Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `-v` | Negation of lattice element v. | — |
| `v + w` | Sum of lattice elements v and w. | — |
| `v - w` | Difference of lattice elements v and w. | — |
| `v * s` / `s * v` | Scalar multiplication. If s is an integer, result lies in L; otherwise in the appropriate R-space. | — |
| `v / s` | Scalar division (1/s)·v; result lies in the R-space over the field of fractions of the parent of s. | — |
| `v div d` | Integer scalar division: (1/d)·v as a lattice element if the result lies in L; error otherwise. | Membership test. |
| `v +:= w` / `v -:= w` / `v *:= n` | In-place sum, difference, integer scalar product. | — |
| `v * T` | Multiply lattice element v of degree n from the right by an n × n matrix T over the base ring of L. Result must lie in L. | — |
| `InnerProduct(v, w)` / `(v, w)` | Inner product vMw^tr where M is the inner product matrix of L. | — |
| `Norm(v)` | Norm (v, v) = vMv^tr. For standard Euclidean inner product this is the square of the Euclidean length. | — |
| `Length(v, K)` / `Length(v)` | Length √(v, v) as an element of real field K (default: current default real field). | Square root. |
| `Support(v)` | Column indices at which v has non-zero entries. | — |

### 30.4.3 Predicates and Boolean Operations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `v in L` | True iff compatible element v lies in L. | Membership test. |
| `v eq w` | True iff lattice elements v and w are equal. | — |
| `v ne w` | True iff v and w are not equal. | — |
| `IsZero(v)` | True iff v is the zero element of L. | — |

### 30.4.4 Access Operations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ElementToSequence(v)` / `Eltseq(v)` | Sequence of entries of v of length Degree(L). | — |
| `Coordinates(v)` | Sequence [c₁,…,c_m] ∈ Z^m such that v = c₁·b₁ + … + c_m·b_m. | Integer linear system. |
| `Coordinates(L, v)` | As `Coordinates(v)` but for v in a (possibly different) compatible lattice L′ of the same degree, giving coordinates relative to the basis of L. | Integer linear system. |
| `CoordinateVector(v)` | The coordinate vector (c₁,…,c_m) as an element of the coordinate lattice C = CoordinateLattice(L). | — |
| `CoordinateVector(L, v)` | As above but coordinates relative to the basis of L, returned as an element of CoordinateLattice(L). | — |

*Worked example: H30E4 (arithmetic on elements of a rank-3 lattice: Coordelt, Norm, InnerProduct, matrix multiplication).*

---

## 30.5 Properties of Lattices

### 30.5.1 Associated Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AmbientSpace(L)` | The ambient rational or real vector space in which L embeds, with the embedding map as the second return value. | — |
| `CoordinateSpace(L)` | The ambient vector space of the coordinate lattice (dimension = Rank(L), inner product matrix = Gram matrix of L), with embedding map. | — |
| `Category(L)` / `Type(L)` | Returns the category `Lat` of lattices. | — |

### 30.5.2 Attributes of Lattices

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Dimension(L)` / `Rank(L)` | The rank m of L (number of basis elements). May be less than the degree n. | — |
| `Degree(L)` | The degree n of L (dimension of the ambient space R^n). | — |
| `Degree(v)` | Degree of the lattice to which element v belongs. | — |
| `Content(L)` | For an exact lattice L: the largest rational c such that (u,v) ∈ cZ for all u, v ∈ L. | — |
| `Level(L)` | For an integral lattice L: the smallest integer k such that k(v,v) ∈ 2Z for all v in the dual of L. | — |
| `Determinant(L)` | Determinant of the Gram matrix F of L. For a full-rank lattice, √Determinant(L) is the volume of a fundamental parallelotope. | — |
| `GramMatrix(L)` | The m × m Gram matrix F = BMB^tr. | — |
| `GramMatrix(X)` | For a matrix X: returns XX^tr, computed in half the time by exploiting symmetry. | — |
| `InnerProductMatrix(L)` | The n × n inner product matrix M of L. | — |
| `Basis(L)` | The basis of L as a sequence [b₁,…,b_m] of lattice elements. | — |
| `BasisMatrix(L)` | The m × n matrix whose rows are the basis elements, over the base ring of L. | — |
| `BasisDenominator(L)` | For an exact lattice L: the common denominator of entries of the current basis. | — |
| `QuadraticForm(L)` | The quadratic form of L as a multivariate polynomial. | — |

### 30.5.3 Predicates and Booleans on Lattices

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `L eq M` | True iff L and M have the same basis matrix and inner product matrix. | — |
| `L ne M` | Logical negation of `eq`. | — |
| `L subset M` | True iff L is a sublattice of M. | — |
| `IsExact(L)` | True iff L is exact (base ring Z or Q; all inner products in Q). | — |
| `IsIntegral(L)` | True iff (v,w) ∈ Z for all v, w ∈ L. | — |
| `IsEven(L)` | True iff L is integral and (v,v) ∈ 2Z for all v ∈ L. | — |

### 30.5.4 Base Ring and Base Change

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BaseRing(L)` / `CoefficientRing(L)` | The smallest ring over which the basis and inner product matrices can be represented (Z or Q for exact lattices, or an approximate real field). | — |
| `CoordinateRing(L)` | The ring of coordinate coefficients (always Z). | — |
| `ChangeRing(L, S)` / `BaseChange(L, S)` / `BaseExtend(L, S)` | Coerce basis and inner product entries into ring S; returns the new lattice and the homomorphism. Mainly useful for changing between real fields of varying precision. | — |

---

## 30.6 Construction of New Lattices

### 30.6.1 Sub- and Superlattices and Quotients

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `sub< L \| S >` | Sublattice of L generated by the elements in list S (each item may be a lattice element, sequence of elements, sublattice, or sequence of sublattices). Returns the sublattice L′ and the inclusion map L′ → L. | LLL-based sublattice computation. |
| `ext< L \| S >` | Superlattice of L generated by L together with elements in list S (any elements coercible into the ambient space V = R^n). Returns L′ and the inclusion map L → L′. | LLL-based superlattice computation. |
| `T * L` | Sublattice defined by an l × m integer transformation matrix T applied to the basis matrix from the left. Result has rank ≤ l. | — |
| `s * L` / `L * s` | Sublattice or superlattice obtained by scaling the basis matrix by scalar s. | — |
| `L / s` | Sublattice or superlattice obtained by multiplying the basis matrix by 1/s. | — |
| `quo< L \| S >` | Quotient L/L′ as an abelian group (L′ generated by elements in list S), with the natural epimorphism L → L/L′. | Abelian group quotient. |
| `L / S` | Quotient L/S as an abelian group (S a sublattice of L), with the epimorphism. | Abelian group quotient. |
| `Index(L, S)` | Index of sublattice S in L (= #(L/S)); returns 0 if infinite. | — |

*Worked example: H30E5 (sub-, ext-, and quo-constructors on a rank-3 degree-4 lattice; Index, quotient group).*

### 30.6.2 Standard Constructions of New Lattices

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Dual(L)` | The dual lattice L# = {v ∈ V : (v,l) ∈ Z ∀l ∈ L}. By default, rescales to an integral primitive lattice with LLL-reduced basis. Parameter `Rescale` (default `true`); set to `false` for the proper unscaled dual. | LLL reduction of dual basis. |
| `PartialDual(L, n)` | The n-th partial dual of an integral lattice L (n divides the exponent of the DualQuotient group). Defined by pulling back (e/n)·g for generators g of DualQuotient and intersecting with L. Parameter `Rescale` (default `true`). | Sublattice intersection. |
| `DualBasisLattice(L)` | Returns the dual L# with basis matrix F⁻¹B and inner product matrix M (Gram matrix F⁻¹). The unscaled proper dual. | Direct computation. |
| `DualQuotient(L)` | For an integral lattice L: the finite abelian group Q = L#/L of order Determinant(L), the unscaled dual L#, and the natural epimorphism φ : L# → Q. | Abelian group computation. |
| `EvenSublattice(L)` | The maximal even sublattice of integral L, together with the natural embedding into L. | — |
| `L + M` | For compatible lattices L and M: the lattice generated by their union. Basis is LLL-reduced. | LLL reduction **[LLL82]**. |
| `L meet M` | Intersection L ∩ M of compatible lattices. Basis is LLL-reduced. | LLL reduction **[LLL82]**. |
| `DirectSum(L, M)` / `OrthogonalSum(L, M)` | Orthogonal direct sum of lattices L and M; inner product is the orthogonal sum. Preserves the basis matrices. | — |
| `OrthogonalDecomposition(L)` | Sequence of indecomposable orthogonal summands of L. Additional forms may be supplied to make the decomposition orthogonal with respect to those forms as well. | Orthogonal decomposition algorithm. |
| `OrthogonalDecomposition(F)` | For a sequence of bilinear forms F (first form positive definite): returns basis matrices B₁,…,B_s of the indecomposable summands of Z^n with respect to all forms, plus per-summand form sequences. Parameter `Optimize` (default `false`): LLL-reduces each B_i. | Orthogonal decomposition; optional LLL **[LLL82]**. |
| `TensorProduct(L, M)` | Tensor product of lattices L and M; inner product given by the Kronecker product of the inner product matrices. | Kronecker product. |
| `ExteriorSquare(L)` | Exterior square of L (skew tensors in L ⊗ L); inner product inherited from the tensor square. | — |
| `SymmetricSquare(L)` | Symmetric square of L (symmetric tensors in L ⊗ L); inner product inherited from the tensor square. | — |
| `PureLattice(L)` | For L with integral or rational entries: the pure lattice P = (Q ⊗ L) ∩ Z^n; generates the same rational subspace as L with trivial elementary divisors. | — |
| `IntegralBasisLattice(L)` | For an exact lattice L: the lattice obtained by multiplying the basis by the smallest positive scalar S making the basis integral, together with S. | — |

*Worked example: H30E6 (PartialDual and DualQuotient on a 29-dimensional lattice from the database; determinant factorizations).*

---

## 30.7 Reduction of Matrices and Lattices

For each reduction algorithm there are three entry points: a basis matrix form, a Gram matrix form, and a lattice form.

### 30.7.1 LLL Reduction

The Lenstra–Lenstra–Lovász algorithm **[LLL82]** (1982) runs in polynomial time and has found
applications in cryptography, optimisation, computer algebra, and algorithmic number theory.
Magma implements the provable floating-point L2 variant of Nguyen–Stehlé **[NS09]** and the
exact integral method of de Weger **[dW87]**, with up to eight internal variants selected
automatically. The default guarantees a correct LLL-reduced output; `Proof := false` enables
faster heuristic variants. For (δ,η)-LLL-reduced bases the first basis vector satisfies
‖b₁‖ ≤ (δ−η²)^{-(d-1)/2} · min{‖v‖ : v ∈ L\{0}}.

The Gram-Schmidt step can be performed either over the Gram matrix (provable but slower) or
without (faster heuristics). Six floating-point arithmetic regimes are used: arbitrary
precision (MPFR-based), C doubles, and doubles-with-extended-exponent, each with and without
the Gram matrix; a seventh uses a factored extended exponent.

Deep insertion (Schnorr–Euchner **[SE94]**) produces shorter bases at higher cost. Early
reduction can dramatically speed up certain applications (e.g. integer-relation detection).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LLL(X)` | For a matrix X over a subring of R: returns (a) a LLL-reduced matrix Y whose rows span the same Z-lattice as the rows of X, (b) a unimodular integer matrix T with TX = Y, (c) the rank of X. Parameters: `Al` (`"New"` or `"Old"`, default `"New"`), `Proof` (default `true`), `Method` (`"FP"`, `"L2"`, or `"Integral"`, default `"FP"`), `Delta` (default 0.75), `Eta` (default 0.501), `InitialSort` (default `false`), `FinalSort` (default `false`), `StepLimit` (default 0), `TimeLimit` (default 0.0), `NormLimit`, `UseGram` (default `false`), `DeepInsertions` (default `false`), `EarlyReduction` (default `false`), `SwapCondition` (`"Lovász"` or `"Siegel"`, default `"Lovász"`), `Fast` (0, 1, or 2; default 0), `Weight` (default all zeros). Also implements the MLLL variant of M. Pohst **[Poh87]** for non-independent rows. | Floating-point LLL **[NS09]** (L2) or exact integral LLL **[dW87]**; deep insertions **[SE94]**; Siegel condition **[Akh02]**. |
| `BasisReduction(X)` / `BasisReduction(X)` | Shortcut for `LLL(X : Proof := false)`. | LLL (heuristic) **[NS09]**. |
| `LLLGram(F)` | For a symmetric matrix F = XX^tr: returns (a) LLL-reduced Gram matrix G, (b) unimodular T with G = TFT^tr, (c) rank of F. Works on singular and indefinite matrices via Simon's indefinite LLL variant **[Sim05]**. Parameter `Isotropic` (default `false`): splits off hyperbolic planes when determinant is squarefree. Same parameters as `LLL` except `UseGram` and `Weight`. | Nguyen–Stehlé L2 **[NS09]** on Gram matrix; indefinite LLL **[Sim05]**. |
| `LLLBasisMatrix(L)` | For lattice L with basis matrix B: returns the LLL basis matrix B′ (LLL-reduced with δ = 0.999 by default) and transformation matrix T with B′ = TB. The result is cached internally. Parameters same as `LLL` (limit parameters excluded). | LLL **[NS09, dW87]** with δ = 0.999 default. |
| `LLLGramMatrix(L)` | For lattice L with Gram matrix F: returns the LLL Gram matrix F′ = B′(B′)^tr and transformation T. Parameters same as `LLL` (limit parameters excluded). | Uses `LLLBasisMatrix` internally. |
| `LLL(L)` | For lattice L: returns a new lattice L′ with LLL-reduced basis B′ and transformation T with B′ = TB. Inner product used is that of L. Equivalent (ignoring T) to `LatticeWithBasis(LLLBasisMatrix(L), InnerProductMatrix(L))`. Shortcut `BasisReduction(L)` turns off proof. Parameters same as `LLL`. | LLL **[NS09, dW87]**. |
| `BasisReduction(L)` | Shortcut for `LLL(L : Proof := false)`. | LLL (heuristic) **[NS09]**. |
| `SetVerbose("LLL", v)` | Set LLL verbose level v ∈ {0,1,2,3}. Level 1: reports rank increases. Level 2: also prints norms. Level 3: adds status every 15 seconds. | — |

*Worked example: H30E7 (50-dimensional knapsack-type lattice; comparing default, Siegel condition, Delta=0.9999, and Fast=1 variants of LLL). H30E8 (extended GCD via LLL with a scale factor).*

### 30.7.2 Pair Reduction

Pairwise reduced bases satisfy 2|(v,w)| ≤ min(‖v‖, ‖w‖) for all pairs of basis vectors — a
simpler but often sufficient criterion. Can serve as LLL preprocessing or be alternated with LLL.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PairReduce(X)` | Pairwise-reduced basis matrix Y row-equivalent to X, and unimodular T with TX = Y. Rows of X need not be independent. | Pair reduction. |
| `PairReduceGram(F)` | For a symmetric positive semidefinite matrix F = XX^tr: pairwise-reduced Gram matrix G and unimodular T with G = TFT^tr. Parameter `Check` (default `false`): verify F is positive semidefinite. | Pair reduction on Gram matrix. |
| `PairReduce(L)` | Returns a new lattice L′ with pairwise-reduced basis B′ and transformation T, B′ = TB. Inner product used is that of L. | Pair reduction. |

### 30.7.3 Seysen Reduction

Seysen reduction simultaneously reduces the Gram matrix G = YY^tr and its inverse G⁻¹, useful
in representation theory for controlling the size of entries in inverse Gram matrices.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Seysen(X)` | Seysen-reduced basis matrix Y and unimodular T with TX = Y, so that both G = YY^tr and G⁻¹ have simultaneously reduced entries. Rows of X need not be independent. | Seysen simultaneous reduction. |
| `SeysenGram(F)` | For symmetric positive semidefinite F = XX^tr: Seysen-reduced Gram matrix G and unimodular T with G = TFT^tr. Parameter `Check` (default `false`). | Seysen reduction on Gram matrix. |
| `Seysen(L)` | Returns a new lattice L′ with Seysen-reduced basis B′ and T, B′ = TB. Both the basis of L′ and the dual basis are simultaneously reduced. | Seysen reduction. |

*Worked example: H30E9 (Seysen reduction of the Leech lattice Gram matrix; comparing diagonal entries of G⁻¹ for LLL, PairReduce, and Seysen — only Seysen gives all 4's).*

### 30.7.4 HKZ Reduction

A basis is **Hermite–Korkine–Zolotarev reduced** (HKZ-reduced) if: (a) all |μᵢ,ⱼ| ≤ 0.501 for
i > j; (b) b₁ is a shortest non-zero vector in the lattice; (c) the projected vectors
b₂−μ₂,₁b₁, … are recursively HKZ-reduced. HKZ-reduced bases are much harder to compute than
LLL-reduced ones but provide a better representation for subsequent enumeration.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HKZ(X)` | HKZ-reduced matrix Y and unimodular T with TX = Y. Rows of X need not be independent. The Proof option (default `true`) guarantees correctness. `Unique` (default `false`): computes a unique HKZ basis (first non-zero Gram-Schmidt coordinate positive, shortest in lexicographic order). Parameter `Prune` (default all 1.0): pruning table for the internal enumeration (one table per dimension). | Iterative shortest vector enumeration via Fincke–Pohst / Kannan / Schnorr–Euchner **[FP83, Kan83, SE94]**; guaranteed by **[PS08]**. |
| `HKZGram(F)` | For symmetric positive semidefinite F: HKZ-reduced Gram matrix G and unimodular T with G = TFT^tr. Parameters `Proof` (default `true`) and `Prune`. | As `HKZ` on Gram matrix. |
| `HKZ(L)` | Returns a new lattice L′ with HKZ-reduced basis B′ and T, B′ = TB. Parameters `Proof` (default `true`) and `Prune`. | As `HKZ`. |
| `SetVerbose("HKZ", v)` | Set HKZ verbose level v ∈ {0,1} (set "Enum" verbose for more enumeration detail). | — |
| `GaussReduce(X)` / `GaussReduceGram(F)` / `GaussReduce(L)` | Restriction of the HKZ functions to lattices of rank 2. | Gauss reduction (HKZ in dimension 2). |

*Worked example: H30E10 (60-dimensional random lattice: HKZ takes ~70× longer than LLL but ShortVectors is ~2× faster from HKZ basis; ~2.3× faster at 3/2 × minimum).*

### 30.7.5 Recovering a Short Basis from Short Lattice Vectors

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ReconstructLatticeBasis(S, B)` | Given a basis S of a finite-index sublattice of the lattice L spanned by rows of integral matrix B: returns a matrix C whose rows span L and satisfy ‖cᵢ‖ ≤ i^(1/2)‖sᵢ‖ and ‖cᵢ*‖ ≤ ‖sᵢ*‖ (Gram-Schmidt). | Algorithm of Lemma 7.1 of **[MG02]**. |

---

## 30.8 Minima and Element Enumeration

All functions in this section rest on one enumeration algorithm that finds all lattice vectors
in a hyperball **[FP83, Kan83, SE94]**. Since the Shortest Vector Problem and Closest Vector
Problem are hard **[Ajt98, vEB81]**, practical use is typically restricted to lattices of
dimension ≤ 50–60. The function `EnumerationCost` estimates the cost before committing.

A `Prune` parameter (a sequence [p₁,…,p_d] with pᵢ ∈ [0,1]) replaces the bounding condition
‖·‖² ≤ u by ‖·‖² ≤ pᵢ·u at level i. This may miss solutions but greatly speeds up the search
for probabilistically easy problems. `EnumerationCost` and `EnumerationCostArray` can estimate
both the speed gain and the probability of missing a solution under a given Prune setting.

By default, the outputs of `Minimum`, `PackingRadius`, `HermiteNumber`, `CentreDensity`,
`Density`, `KissingNumber`, `ShortestVectors`, `ShortestVectorsMatrix`, `ShortVectors`,
`ShortVectorsMatrix`, and `ThetaSeries` are guaranteed correct. Use `Proof := false` to disable.

### 30.8.1 Minimum, Density and Kissing Number

See **[JC98]** for background on minimum, density and kissing numbers.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Minimum(L)` / `Min(L)` | The minimum of L (minimal norm of a non-zero vector). Parameters `Proof` (default `true`) and `Prune` (default all 1.0). See also the `L'Minimum` attribute. | Fincke–Pohst enumeration **[FP83, Kan83, SE94]**; rigorous variant **[PS08]**. |
| `PackingRadius(L)` | Half the square root of the minimum; the packing radius. Parameters `Proof` and `Prune`. | As `Minimum`. |
| `HermiteConstant(n)` | The n-th Hermite constant γ_n raised to the power n = max over n-dim lattices of Min(L)/Determinant(L). Exact for n ≤ 8 or n = 24; upper bound otherwise. | Known values; bound for general n. |
| `HermiteNumber(L)` | Min(L)/Determinant(L)^(1/n), the Hermite number of L. Parameters `Proof` and `Prune`. | As `Minimum`. |
| `CentreDensity(L)` / `CenterDensity(L)` / `CentreDensity(L, K)` / `CenterDensity(L, K)` | The centre density √((Min(L)/4)^Rank(L)/Determinant(L)) as an element of real field K (default: current default real field). Parameters `Proof` and `Prune`. | As `Minimum`; square root computation. |
| `Density(L)` / `Density(L, K)` | The sphere-packing density of L, as an element of K. Parameters `Proof` and `Prune`. | As `CentreDensity` × volume of unit ball. |
| `KissingNumber(L)` | Number of vectors of minimal non-zero norm (twice the number of normalized shortest vectors). Parameters `Proof` and `Prune`. | Enumeration **[FP83, Kan83, SE94]**; guaranteed by **[PS08]**. |

*Worked example: H30E11 (Leech lattice Λ₂₄: minimum 4 computed in 0.020 s; kissing number 196560 in 0.180 s).*

### 30.8.2 Shortest and Closest Vectors

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ShortestVectors(L)` | Sorted sequence of the shortest non-zero vectors of L (up to sign, normalized so first non-zero entry is positive). Parameter `Max` (default ∞): limit on number of vectors. Parameters `Proof` (default `true`) and `Prune`. | Fincke–Pohst enumeration **[FP83, Kan83, SE94]**; guaranteed **[PS08]**. Solves the Shortest Vector Problem. |
| `ShortestVectorsMatrix(L)` | Shortest non-zero vectors of L as rows of a matrix (more efficient for some applications). Parameters `Max`, `Proof`, `Prune`. | As `ShortestVectors`. |
| `ClosestVectors(L, w)` | Sorted sequence Q of vectors v ∈ L closest to w, together with the minimal squared distance d. w may be in any compatible degree-n lattice or R-space. Not normalized (closest vectors are not sign-symmetric). Parameter `Max`. | Fincke–Pohst enumeration adapted for closest vector (CVP). Solves the Closest Vector Problem. |
| `ClosestVectorsMatrix(L, w)` | Closest vectors as a matrix, together with minimal squared distance d. Parameter `Max`. | CVP enumeration. |

*Worked example: H30E12 (E₈ Gosset lattice: 120 normalized shortest vectors, kissing number 240, minimum 2; then closest vectors to a hole of E₈ at squared distance 8/9).*

### 30.8.3 Short and Close Vectors

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ShortVectors(L, u)` / `ShortVectors(L, l, u)` | Sorted sequence of tuples ⟨v, r⟩ (vector and norm) for all v ∈ L with norm in (0, u] or [l, u]. Normalized (up to sign). Parameters `Max`, `Proof` (default `true`), `Prune`. | Fincke–Pohst enumeration **[FP83, Kan83, SE94]**. |
| `ShortVectorsMatrix(L, u)` / `ShortVectorsMatrix(L, l, u)` | Vectors with norm in range as rows of a matrix. Parameters `Max`, `Proof`, `Prune`. | As `ShortVectors`. |
| `CloseVectors(L, w, u)` / `CloseVectors(L, w, l, u)` | Sorted sequence of tuples ⟨v, d⟩ (vector and squared distance from w) for v ∈ L with squared distance in (0, u] or [l, u]. Not normalized. Parameters `Max`. | CVP enumeration, range-bounded. |
| `CloseVectorsMatrix(L, w, u)` / `CloseVectorsMatrix(L, w, l, u)` | Close vectors in range as rows of a matrix. Parameters `Max`. | CVP enumeration, range-bounded. |

*Worked example: H30E13 (Knapsack problem via ShortVectors: 12-element instance solved directly; 50-element instance with 1000-bit integers solved in 0.57 s LLL + 0.04 s enumeration). H30E14 (splitting homogeneous components of an integral A₅ representation using short vectors of an endomorphism-ring lattice).*

### 30.8.4 Short and Close Vector Processes

Iterator-style interface for enumerating short or close vectors one at a time.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ShortVectorsProcess(L, u)` / `ShortVectorsProcess(L, l, u)` | Create an enumeration process P for vectors v ∈ L with norm in (0, u] or [l, u]. | Initializes Fincke–Pohst enumeration **[FP83, Kan83, SE94]**. |
| `CloseVectorsProcess(L, w, u)` / `CloseVectorsProcess(L, w, l, u)` | Create a process P for vectors v ∈ L with squared distance from w in (0, u] or [l, u]. | CVP enumeration process. |
| `NextVector(P)` | Return the next vector and its norm (or squared distance for close vectors) from process P. Returns the zero vector and -1 when exhausted. Order is arbitrary (unlike sorted batch functions). | One enumeration tree step. |
| `IsEmpty(P)` | True iff the enumeration process P has found all vectors in the specified range. | — |

*Worked example: H30E15 (reimplementing ThetaSeries using ShortVectorsProcess and NextVector on E₈; comparison with built-in ThetaSeries).*

### 30.8.5 Successive Minima and Theta Series

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SuccessiveMinima(L)` / `SuccessiveMinima(L, k)` | The first k successive minima M₁ ≤ … ≤ Mₖ of lattice L (or all m if k omitted), and corresponding linearly independent vectors achieving them. L must be an exact lattice. | Enumeration-based; uses the fact that the minima satisfy Minkowski's successive minima bound. |
| `ThetaSeries(L, n)` | For an integral lattice L: the theta series Θ_L(q) as a formal power series in q to precision n (coefficient of q^k = number of vectors of norm k). Parameters `Proof` (default `true`) and `Prune`. | Enumeration of all vectors of norm ≤ n; **[FP83, Kan83, SE94]**. |
| `ThetaSeriesIntegral(L, n)` | Restriction of `ThetaSeries` to integral lattices. Parameters `Proof` and `Prune`. | As `ThetaSeries`. |

### 30.8.6 Lattice Enumeration Utilities

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SetVerbose("Enum", v)` | Set enumeration verbose level v ∈ {0,1}. Level 1: prints status every 15 seconds during minimum/shortest-vector/theta computations. | — |
| `EnumerationCost(L)` / `EnumerationCost(L, u)` | Estimate the number of tree nodes visited during enumeration for lattice L and squared radius u (default: upper bound on minimum). Based on the Gaussian heuristic **[HS07]**. Parameter `Prune`: estimate the pruned cost. Runs in polynomial time (no Prune); more expensive with Prune. | Gaussian heuristic **[HS07]**; polynomial time. |
| `EnumerationCostArray(L)` / `EnumerationCostArray(L, u)` | Estimate of node count per layer of the enumeration tree. Parameter `Prune`. | As `EnumerationCost`, layer by layer. |

*Worked example: H30E16 (50-dim lattice: cost 4×10¹⁸ before further LLL, 7×10⁷ after Delta=0.999+DeepInsertions, then 3.66 s; 65-dim with pruning: cost reduced from 2.8×10¹² to 1.75×10¹⁰, 422 s; probability of missing solution ~0.8%).*

---

## 30.9 Theta Series as Modular Forms

The theta series of an integral lattice L is a modular form of weight (1/2)·dim(L), with level
and nebentypus determined by L#/L. The algorithm computes which modular form space the theta
series belongs to (via `ThetaSeriesModularFormSpace`) and identifies the theta series as an
element of that space with the minimal number of enumerated coefficients, using linear
constraints from partial dual lattices and q-modularity. We say L is q-modular if it is
isomorphic to its q-th partial dual L_q; such modularities reduce the required enumeration.

Normalisation convention: even integral L uses Σ_{v∈L} q^{(1/2)|v|²}; odd integral L uses Σ_{v∈L} q^{|v|²}.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ThetaSeriesModularFormSpace(L)` | For an integral lattice L: the space of modular forms containing the theta series of L (with normalisation as above). | Determined by weight = (1/2)·dim(L), level, and nebentypus from L#/L. |
| `ThetaSeriesModularForm(L)` | For an integral lattice L: the theta series of L as an element of `ThetaSeriesModularFormSpace(L)`. Parameters: `KnownTheta` (known coefficients of L's theta series), `KnownDualThetas` (sequence of ⟨q, f_q⟩ tuples for partial duals), `KnownModularities` (set of q for which L is q-modular), `ComputeModularities` (boolean or set: whether to check for q-modularities). | Minimises enumeration cost via linear constraints from partial duals; uses `EnumerationCost` to balance how many coefficients to compute for each dual. |

---

## 30.10 Voronoi Cells, Holes and Covering Radius

Voronoi cell computation is exponential in dimension and is practical only for small dimensions
(up to about 10). L must be an exact lattice. See **[JC98]** for definitions.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `VoronoiCell(L)` | The Voronoi cell of L around the origin: returns (a) sequence V of vertex vectors, (b) set E of pairs {i,j} representing edges, (c) sequence P of hyperplane-defining vectors (hyperplane (x,p) = Norm(p)/2). | Convex polytope computation; exponential complexity. |
| `VoronoiGraph(L)` | A graph with the Voronoi cell vertices and edges. | From `VoronoiCell`. |
| `Holes(L)` | Sequence of vectors which are the holes of L (vertices of the Voronoi cell around the origin). | From `VoronoiCell`. |
| `DeepHoles(L)` | Sequence of vectors which are the deep holes (holes of maximum norm = points of maximum distance from all lattice points). | From `VoronoiCell`. |
| `CoveringRadius(L)` | Squared covering radius = norm of the deep holes. | From `VoronoiCell`. |
| `VoronoiRelevantVectors(L)` | The Voronoi relevant hyperplanes (same as third return of `VoronoiCell`) but computed much faster — does not compute the full Voronoi cell. | Algorithm of **[AEVZ02, Section C]**. |

*Worked example: H30E17 (perfect lattice of dimension 6: Voronoi cell has 782 vertices, 4074 edges, 104 faces; 28 deep holes at squared norm 5/2; graph diameter 8; maximal degree 20).*

---

## 30.11 Orthogonalization

These functions orthogonalize a basis over the field of fractions of the base ring (equivalent
to diagonalizing the inner product matrix of the ambient space). Contrast with
`OrthogonalDecomposition` which decomposes a lattice into orthogonal summands over the base ring.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Orthogonalize(M)` | For basis matrix M over a subring R of R: returns (a) an orthogonalized matrix N (NN^tr diagonal) row-equivalent to M over the field of fractions K of R, (b) invertible T over K with TM = N, (c) rank of M. | Gram-Schmidt orthogonalization over K. |
| `Diagonalization(F)` / `OrthogonalizeGram(F)` | For a symmetric n × n matrix F over R: returns (a) a diagonal matrix G with G = TFT^tr for invertible T over K, (b) T, (c) rank of F. F need not have full rank. | Gaussian elimination / Gram-Schmidt; symmetric. |
| `Orthogonalize(L)` | For lattice L: returns a new lattice with the same Gram matrix but embedded in a space with diagonal inner product matrix. | Uses `Orthogonalize(M)` on the basis matrix. |
| `Orthonormalize(M, K)` / `Cholesky(M, K)` / `Orthonormalize(M)` / `Cholesky(M)` | For a symmetric positive definite matrix M: a lower triangular T over real field K with M = TT^tr (Cholesky decomposition). K defaults to the current default real field. Takes a Gram matrix, not a basis matrix. | Cholesky decomposition; requires square roots. |
| `Orthonormalize(L, K)` / `Cholesky(L, K)` / `Orthonormalize(L)` / `Cholesky(L)` | For lattice L with Gram matrix F: a new lattice over K with the same Gram matrix F but standard Euclidean inner product (involves taking square roots). K defaults to the current default real field. Equivalent to `LatticeWithBasis(Orthonormalize(GramMatrix(L), K))`. | Cholesky decomposition applied to the Gram matrix. |

*Worked example: H30E18 (dual of Coxeter-Todd K₁₂: computing inner products of 756 shortest vectors; 7.12 s with original inner product, 1.30 s after Orthogonalize).*

---

## 30.12 Testing Matrices for Definiteness

Each function calls `OrthogonalizeGram` and checks the sign pattern of the resulting diagonal.
All apply to any symmetric matrix over a real subring (Z, Q, or a real field).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsPositiveDefinite(F)` | True iff vFv^tr > 0 for all non-zero v. | `OrthogonalizeGram` + sign check. |
| `IsPositiveSemiDefinite(F)` | True iff vFv^tr ≥ 0 for all non-zero v. | `OrthogonalizeGram` + sign check. |
| `IsNegativeDefinite(F)` | True iff vFv^tr < 0 for all non-zero v. | `OrthogonalizeGram` + sign check. |
| `IsNegativeSemiDefinite(F)` | True iff vFv^tr ≤ 0 for all non-zero v. | `OrthogonalizeGram` + sign check. |

---

## 30.13 Genera and Spinor Genera

A genus has type `SymGen` and holds a representative lattice and the local data defining the
genus. Each genus consists of 2^n spinor genera (n typically 0 or 1). Genus equality is fast
(comparison of canonical local data). Spinor genus equality currently calls `Representatives`.
Enumeration of isometry classes is done via p-neighbour graph closure **[Kne57, SP91]**.

### 30.13.1 Genus Constructions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Genus(L)` / `Genus(G)` | For exact lattice L or spinor genus G: the genus of L. If given a genus, returns it unchanged. | Local invariants. |
| `SpinorGenus(L)` | For exact lattice L: the spinor genus of L. | Spinor norm computation. |
| `SpinorGenera(G)` | For a genus G: the sequence of spinor genera. If G is already a spinor genus, returns [G]. | — |

### 30.13.2 Invariants of Genera and Spinor Genera

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Representative(G)` | A representative lattice for the genus or spinor genus symbol G. | — |
| `IsSpinorGenus(G)` | True iff G is a spinor genus (negation of `IsGenus`). | — |
| `IsGenus(G)` | True iff G is a genus. | — |
| `Determinant(G)` | Determinant of the genus symbol. | — |
| `LocalGenera(G)` | Sequence of p-adic genera of genus G. | — |
| `Representative(G)` | Representative lattice for genus symbol G (p-adic context). | — |
| `G1 eq G2` | True iff two genus symbols represent the same genus. Fast for genera (local data comparison); for spinor genera currently calls `Representatives`. | Local data comparison or **[Kne57]** p-neighbour graph. |
| `#G` | Number of isometry classes in genus or spinor genus G. Expensive (calls `Representatives`). | Neighbour graph enumeration. |
| `SpinorCharacters(G)` | Spinor characters of G as a sequence of Dirichlet characters; see **[JC98]** for definitions. | — |
| `SpinorGenerators(G)` | Spinor generators as a sequence of primes generating the spinor norm group. | — |
| `AutomorphousClasses(L, p)` / `AutomorphousClasses(G, p)` | Representatives of the p-adic square classes in the image of the spinor norm of lattice L or genus symbol G. | Spinor norm computation. |
| `IsSpinorNorm(G, p)` | True iff p is coprime to 2 and to the determinant, and p is the norm of an element of the spinor kernel of G. | Spinor kernel check. |

### 30.13.3 Invariants of p-adic Genera

Type `SymGenLoc`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Prime(G)` | The prime p for which G is the p-adic genus. | — |
| `Representative(G)` | Canonical representative with Gram matrix in Jordan form (diagonal for odd p). | Jordan decomposition over Z_p. |
| `Determinant(G)` | A canonical p-adic representative of the determinant (well-defined up to squares). | — |
| `Dimension(G)` | Dimension of the p-adic genus G. | — |
| `G1 eq G2` | True iff G1 and G2 have the same prime and same canonical Jordan form. | — |

### 30.13.4 Neighbour Relations and Graphs

The p-neighbour of an integral lattice L with respect to v ∈ L \ pL with (v,v) ∈ p²Z is the
lattice generated by L_v = {x ∈ L : (x,v) ∈ pZ} and p⁻¹v. See **[Kne57]** and **[SP91]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Neighbour(L, v, p)` / `Neighbor(L, v, p)` | For integral L, prime p not dividing Determinant(L), and v ∈ L \ pL with (v,v) ∈ p²Z: the p-neighbour of L with respect to v. | p-neighbour construction **[Kne57, SP91]**. |
| `Neighbours(L, p)` / `Neighbors(L, p)` | Sequence of all p-neighbours of integral lattice L. | Orbit-representative enumeration + `Neighbour`. |
| `NeighbourClosure(L, p)` / `NeighborClosure(L, p)` | Transitive closure of the p-neighbour relation from L. Only projective orbit representatives of Aut(L) on L/pL are used (isometric neighbours arise from the same orbit). Parameter `Bound` (default 2^32): error if p^Rank(L) > Bound. | p-neighbour graph exploration **[Kne57, SP91]**; automorphism group acts on L/pL. |
| `GenusRepresentatives(L)` / `SpinorRepresentatives(L)` / `Representatives(G)` | Enumerate all isometry classes in the genus or spinor genus of L (or of G) via p-neighbour closure. For the genus, sufficiently many primes are used to generate the full image modulo the spinor kernel. Parameters `Bound` (default 2^32) and `Depth`. | p-neighbour graph closure **[Kne57, SP91]** with isometry testing. |
| `AdjacencyMatrix(G, p)` | Adjacency matrix of the p-neighbour graph on representative classes for genus or spinor genus G. p must be prime and (for spinor genera) must be an automorphous number for G. | p-neighbour graph; automorphism group isometry testing. |

*Worked example: H30E19 (E₈ as the 2-neighbour of Z⁸). H30E20 (manually enumerating the genus of the Coxeter-Todd lattice K₁₂ via even 2-neighbours and isometry testing; 10 isometry classes found in 9.3 s, mass = 4649359/4213820620800).*

---

## 30.14 Attributes of Lattices

Low-level direct access to cached lattice attributes. Setting an attribute asserts a value
without full verification; incorrect values lead to unpredictable results. Test with `assigned`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `L'Minimum` | Attribute for the minimum of a rational or integer lattice L. If set, must equal the true minimum. If already set, new value must agree with the old. | — |
| `L'MinimumBound` | Attribute for an upper bound on the minimum. If already set, the new value must be ≤ the old. Magma does not verify the bound. | — |

---

## 30.15 Database of Lattices

Magma includes a database corresponding to the Nebe–Sloane Catalogue of Lattices **[NS01a,
NS01b]**. A second version (from Magma V2.16) adds lattices, more automorphism-group data, and
theta series as attributes, while removing duplicates (note: lattice numbering may differ between
versions). The database does *not* include lattices obtainable by standard creation functions or
those defined over rings other than Z or Q.

Entries may be accessed by global index i, by dimension d and index i within that dimension, or
by name N (exactly as in the catalogue, including punctuation and whitespace; use an additional
index i to disambiguate repeated names).

### 30.15.1 Creating the Database

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LatticeDatabase()` | Returns a database object D for the lattice database. | — |

### 30.15.2 Database Information

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `#D` / `NumberOfLattices(D)` | Total number of lattices in the database. | — |
| `LargestDimension(D)` | Largest dimension of any lattice in the database. | — |
| `NumberOfLattices(D, d)` | Number of lattices of dimension d in the database. | — |
| `NumberOfLattices(D, N)` | Number of lattices named N in the database. | — |
| `LatticeName(D, i)` | Name and dimension of the i-th database entry. | — |
| `LatticeName(D, d, i)` | Name and dimension of the i-th entry of dimension d. | — |
| `LatticeName(D, N)` | Name and dimension of the first entry named N. | — |
| `LatticeName(D, N, i)` | Name and dimension of the i-th entry named N. | — |

*Worked example: H30E21 (database contains 700 lattices with 673 distinct names; illustrating `NumberOfLattices`, `LatticeName`).*

### 30.15.3 Accessing the Database

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Lattice(D, i: -)` / `Lattice(D, d, i: -)` / `Lattice(D, N: -)` / `Lattice(D, N, i: -)` | Retrieve the specified entry as a lattice L. Parameter `TrustAutomorphismGroup` (default `true`): if `false`, stored automorphism group data is not loaded into L. | Database lookup. |
| `LatticeData(D, i)` / `LatticeData(D, d, i)` / `LatticeData(D, N)` / `LatticeData(D, N, i)` | Returns a record with all stored information about the specified entry (name, dim, lattice, minimum, kissing number, integrality flags, modularity, group names, group, group order, Hermitian group data, Hermitian structure). The automorphism group is returned separately from the lattice. Fields may or may not be assigned for any particular entry. | Database lookup. |

*Worked example: H30E22 (6th-dimensional entry 10 of the database: A6,1 lattice of rank 6, minimum 4, kissing number 42; automorphism group has order 96).*

### 30.15.4 Hermitian Lattices

Functions for Gram matrices over an imaginary quadratic field or a quaternion algebra. The main
application is computing automorphism groups preserving a Hermitian structure.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HermitianTranspose(M)` | For M over an imaginary quadratic field or quaternion algebra: the conjugate transpose. | — |
| `ExpandBasis(M)` | For M over an imaginary quadratic field or quaternion algebra: expand to a basis over the rationals. | — |
| `HermitianAutomorphismGroup(M)` / `QuaternionicAutomorphismGroup(M)` | For a conjugate-symmetric Gram matrix M: the automorphism group. Functions such as `CharacterTable`, `IsConjugate`, and `InvariantForms` are available via rewriting over the rationals; `QuaternionicGModule` splits G-modules over a quaternionic structure. | Rewriting over Q + automorphism group computation. |
| `InvariantForms(G)` | For a matrix group over an associative algebra or imaginary quadratic field: a basis for the forms fixed by G. | GHom restricted to elements fixed by the quaternionic structure. |
| `QuaternionicGModule(M, I, J)` | For G-module M and I, J in the endomorphism algebra that anti-commute and whose squares are scalars: write G over the quaternionic structure given by I, J. | Quaternionic G-module decomposition. |
| `MooreDeterminant(M)` | For a conjugate-symmetric matrix M over a quaternion algebra: the Moore determinant (the well-defined "normal" determinant when all diagonal elements are rational). | Moore determinant algorithm. |

*Worked example: H30E23 (Coxeter-Todd lattice over Q_{3,∞} from SU(3,3); QuaternionicGModule, InvariantForms). H30E24 (Leech lattice via 6-dim quaternionic Gram matrix; |Aut| = 503193600; Coxeter-Todd |Aut| = 12096; Eisenstein HermitianAutomorphismGroup = ShephardTodd(34)).*

---

## 30.16 Bibliography

| Key | Reference |
|-----|-----------|
| **[AEVZ02]** | E. Agrell, T. Eriksson, A. Vardy, and K. Zeger. Closest point search in lattices. *IEEE Transactions on Information Theory*, 48(8):2201–2214, 2002. |
| **[Ajt98]** | Miklós Ajtai. The Shortest Vector Problem in L₂ is NP-hard for Randomized Reductions (Extended Abstract). In *Proceedings of the 30th Symposium on the Theory of Computing (STOC 1998)*, pages 10–19. ACM, 1998. |
| **[Akh02]** | Ali Akhavi. Random lattices, threshold phenomena and efficient reduction algorithms. *Theoretical Computer Science*, 287(2):359–385, 2002. |
| **[dW87]** | Benne M.M. de Weger. Solving exponential Diophantine equations using lattice basis reduction algorithms. *J. Number Th.*, 26:325–367, 1987. |
| **[FP83]** | U. Fincke and M. Pohst. A procedure for determining algebraic integers of given norm. In *EUROCAL*, volume 162 of LNCS, pages 194–202. Springer, 1983. |
| **[HPP06]** | F. Hess, S. Pauli, and M. Pohst, editors. *ANTS VII*, volume 4076 of LNCS. Springer-Verlag, 2006. |
| **[HS07]** | Guillaume Hanrot and Damien Stehlé. Improved Analysis of Kannan's Shortest Lattice Vector Algorithm (Extended Abstract). In *Advances in cryptology — CRYPTO 2007*, volume 4622 of LNCS, pages 170–186. Springer, 2007. |
| **[JC98]** | N.J.A. Sloane and J.H. Conway. *Sphere Packings, Lattices and Groups*, volume 290 of Grundlehren der Mathematischen Wissenschaften. Springer, New York–Berlin–Heidelberg, 3rd edition, 1998. |
| **[Kan83]** | R. Kannan. Improved algorithms for integer programming and related lattice problems. In *Proceedings of the 15th Symposium on the Theory of Computing (STOC 1983)*, pages 99–108. ACM, 1983. |
| **[Kne57]** | M. Kneser. Klassenzahlen indefiniter quadratischer Formen. *Archiv Math.*, 8:241–250, 1957. |
| **[LLL82]** | Arjen K. Lenstra, Hendrik W. Lenstra, and László Lovász. Factoring polynomials with rational coefficients. *Mathematische Annalen*, 261:515–534, 1982. |
| **[MG02]** | Daniele Micciancio and Shafi Goldwasser. *Complexity of lattice problems: a cryptographic perspective*, volume 671 of The Kluwer International Series in Engineering and Computer Science. Kluwer Academic Publishers, 2002. |
| **[NS01a]** | G. Nebe and N.J.A. Sloane. The Catalogue of Lattices. URL: http://www.research.att.com/~njas/lattices/, 2001. |
| **[NS01b]** | Gabriele Nebe and Neil J.A. Sloane. A Catalogue of Lattices. URL: http://akpublic.research.att.com/~njas/lattices/index.html, 2001. |
| **[NS06]** | Phong Nguyen and Damien Stehlé. LLL on the Average. In Hess et al. **[HPP06]**, pages 238–256. |
| **[NS09]** | Phong Nguyen and Damien Stehlé. An LLL Algorithm with Quadratic Complexity. *SIAM Journal on Computing*, 39(3):874–903, 2009. |
| **[Poh87]** | Michael Pohst. A Modification of the LLL Reduction Algorithm. *J. Symbolic Comp.*, 4(1):123–127, 1987. |
| **[Pro]** | The SPACES Project. MPFR, a LGPL-library for multiple-precision floating-point computations with exact rounding. URL: http://www.mpfr.org/. |
| **[PS08]** | Xavier Pujol and Damien Stehlé. Rigorous and efficient short lattice vectors enumeration. In *Advances in Cryptology — AsiaCrypt 2008*, LNCS. Springer, 2008. |
| **[SE94]** | Claus-Peter Schnorr and Michael Euchner. Lattice Basis Reduction: Improved Practical Algorithms and Solving Subset Sum Problems. *Mathematics of Programming*, 66:181–199, 1994. |
| **[SH95]** | Claus-Peter Schnorr and Horst Helmut Hörner. Attacking the Chor-Rivest Cryptosystem by Improved Lattice Reduction. In *Advances in Cryptology — EuroCrypt 1995*, volume 921 of LNCS, pages 1–12. Springer-Verlag, 1995. |
| **[Sho]** | Victor Shoup. NTL, Number Theory C++ Library. URL: http://www.shoup.net/ntl/. |
| **[Sim05]** | Denis Simon. Solving quadratic equations using reduced unimodular quadratic forms. *Math. Comp.*, 74(251):1531–1543 (electronic), 2005. |
| **[SP91]** | Rainer Schulze-Pillot. An algorithm for computing genera of ternary and quaternary quadratic forms. In Stephen M. Watt, editor, *Proceedings ISSAC'91*, pages 134–143, Bonn, 1991. |
| **[Ste09]** | Damien Stehlé. Floating-point LLL: theoretical and practical aspects. Springer-Verlag, 2009. |
| **[vEB81]** | Peter van Emde Boas. Another NP-complete partition problem and the complexity of computing short vectors in a lattice. Technical report 81-04, Mathematisch Instituut, Universiteit van Amsterdam, 1981. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| LLL reduction **[LLL82]** (Nguyen–Stehlé L2 **[NS09]** / de Weger integral **[dW87]**) | `Lattice`, `LLL`, `BasisReduction`, `LLLGram`, `LLLBasisMatrix`, `LLLGramMatrix`, `LLL(L)`, `BasisReduction(L)` |
| MLLL for non-independent rows **[Poh87]** | `LLL(X)` |
| Deep insertions / Schnorr–Euchner **[SE94]** | `LLL(:DeepInsertions)`, `HKZ` |
| Siegel swap condition **[Akh02]** | `LLL(:SwapCondition:="Siegel")` |
| Indefinite LLL **[Sim05]** | `LLLGram` (handles indefinite Gram matrices) |
| Pair reduction | `PairReduce`, `PairReduceGram`, `PairReduce(L)` |
| Seysen simultaneous reduction | `Seysen`, `SeysenGram`, `Seysen(L)` |
| HKZ reduction (shortest vector enumeration) | `HKZ`, `HKZGram`, `HKZ(L)`, `GaussReduce` |
| Short basis from short vectors **[MG02]** | `ReconstructLatticeBasis` |
| Fincke–Pohst / Kannan / Schnorr–Euchner enumeration **[FP83, Kan83, SE94]** | `Minimum`, `Min`, `PackingRadius`, `KissingNumber`, `ShortestVectors`, `ShortestVectorsMatrix`, `ClosestVectors`, `ClosestVectorsMatrix`, `ShortVectors`, `ShortVectorsMatrix`, `CloseVectors`, `CloseVectorsMatrix`, `SuccessiveMinima`, `ThetaSeries`, `ThetaSeriesIntegral` |
| Enumeration process (iterator) **[FP83, Kan83, SE94]** | `ShortVectorsProcess`, `CloseVectorsProcess`, `NextVector`, `IsEmpty` |
| Rigorous enumeration **[PS08]** | All `Proof := true` calls to shortest/close vector functions |
| Gaussian heuristic for enumeration cost **[HS07]** | `EnumerationCost`, `EnumerationCostArray` |
| Minkowski embedding (lattices from number fields) | `MinkowskiLattice`, `MinkowskiSpace`, `Lattice(O)`, `Lattice(I)` |
| Construction A / B (lattices from codes) | `Lattice(C, "A")`, `Lattice(C, "B")` |
| Voronoi cell / covering radius | `VoronoiCell`, `VoronoiGraph`, `Holes`, `DeepHoles`, `CoveringRadius` |
| Voronoi relevant vectors **[AEVZ02]** | `VoronoiRelevantVectors` |
| Gram-Schmidt orthogonalization / Cholesky | `Orthogonalize`, `OrthogonalizeGram`, `Diagonalization`, `Orthonormalize`, `Cholesky` |
| p-neighbour relations **[Kne57, SP91]** | `Neighbour`, `Neighbours`, `NeighbourClosure`, `GenusRepresentatives`, `SpinorRepresentatives`, `Representatives`, `AdjacencyMatrix` |
| Genera and spinor genera | `Genus`, `SpinorGenus`, `SpinorGenera`, `SpinorCharacters`, `SpinorGenerators`, `AutomorphousClasses`, `IsSpinorNorm` |
| Theta series as modular forms | `ThetaSeriesModularFormSpace`, `ThetaSeriesModularForm` |
| Nebe–Sloane lattice database **[NS01a, NS01b]** | `LatticeDatabase`, `Lattice(D,…)`, `LatticeData` |
| Hermitian / quaternionic automorphism groups | `HermitianAutomorphismGroup`, `QuaternionicAutomorphismGroup`, `QuaternionicGModule`, `InvariantForms`, `MooreDeterminant` |
