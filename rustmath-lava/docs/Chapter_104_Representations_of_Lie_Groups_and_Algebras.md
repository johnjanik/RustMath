# Chapter 104 — Representations of Lie Groups and Algebras

**Handbook part:** XIII — Lie Theory
**Handbook pages:** 3139–3176 (PDF pages 3272–3309)

---

## Scope and overview

Chapter 104 provides functionality for working with direct sums of highest weight representations (modules) of semisimple Lie algebras and connected reductive algebraic groups. The class of highest weight representations includes all finite-dimensional representations when the base field is the complex numbers.

The central observation is that representations of connected reductive complex Lie groups are completely reducible, and every irreducible representation is uniquely determined by its highest weight (a dominant weight). A module therefore corresponds to a finite multiset of dominant weights with multiplicities, called a **decomposition multiset**. Many useful computations — module dimension, character multisets, tensor products, symmetric and alternating powers, restrictions and inductions — can be performed combinatorially with these weight multisets without constructing the module explicitly.

**Three levels of abstraction** are supported:

1. **Weight multisets** (decomposition, character, and dominant character multisets): purely combinatorial, ported from the LiE software package **[vLCL92]**. Algorithms for dimension, tensor product, branching, plethysm, Adams operators, etc. are all from **[vLCL92]**.
2. **Explicit representations for Lie algebras**: actual matrix representations of almost reductive structure-constant Lie algebras (Chapter 100), constructed by the algorithm of de Graaf **[dG01]**.
3. **Explicit representations for groups of Lie type**: projective representations of split groups of Lie type (Chapter 103), constructed by the algorithm of Cohen, Murray and Taylor **[CMT04]**. Over fields where the coisogeny group requires a non-trivial Kummer extension, the representation is projective rather than linear; Magma issues a warning in such cases.

Additional specialised modules cover Kazhdan–Lusztig polynomials, the maximal subgroup database for rank ≤ 8 simple Lie groups (from LiE), subalgebras of su(d) studied by Dynkin's method **[Dyn57, ZSH11]**, and Wess–Zumino–Witten fusion rules via the Kac–Walton formula **[FMS97]**.

---

## 104.1 Introduction

### 104.1.1 Highest Weight Modules

A connected reductive complex Lie group G is a homomorphic image G = ξ(G′) where G′ is a direct product of a simply connected group and a torus. Simply connected groups are themselves direct products of simple simply connected groups, identified by their Cartan name; e.g. A4C3B2T2 denotes a group of type A4×C3×B2 with a 2-dimensional torus. Most combinatorial code (ported from LiE) applies to groups of this form, identified with their root data.

Key multiset terminology:
- **Decomposition multiset**: dominant weights with multiplicities, encoding the isomorphism type of a completely reducible module.
- **Character multiset**: all weights occurring in a module, with multiplicities.
- **Dominant character multiset**: the dominant weights in the character multiset (sufficient by Weyl-group symmetry).
- **Virtual multiset**: allows negative multiplicities; proper if all multiplicities are non-negative.

### 104.1.2 Toral Elements

Several functions (notably `Spectrum`) accept finite-order torus elements in a special vector encoding: t = (a₁, …, aᵣ, n) represents the element satisfying tωᵢ = e^{2πiaᵢ/n}, where ω₁, …, ωᵣ are fundamental weights.

### 104.1.3 Other Highest Weight Representations

In positive characteristic, highest weight representations for Lie algebras and groups of Lie type are indecomposable but not necessarily irreducible. For a split group G over field k with coisogeny-group lcm r, a Kummer extension K/k (with rth roots) is needed for the projective representation; it is linear (no extension needed) when r = 1 (e.g. simply connected × torus groups, GLn) or when k is algebraically closed, or when k is finite with |k|−1 coprime to r. Rational function fields, Laurent series fields, and local fields do not admit such extensions when r > 1.

---

## 104.2 Constructing Weight Multisets

Functions to create decomposition multisets, the basic data type for combinatorial representation theory.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `TrivialLieRepresentationDecomposition(R)` / `LieRepresentationDecomposition(R)` | The decomposition multiset of the trivial representation of the weakly simply connected root datum R. | — |
| `LieRepresentationDecomposition(R, v)` | The decomposition multiset of the irreducible highest weight representation with weight v (sequence of length dim(R) or element of Z^d). R must be weakly simply connected. | — |
| `LieRepresentationDecomposition(R, Wt, Mp)` | The decomposition multiset with weights from sequence Wt and multiplicities from sequence Mp, over the weakly simply connected root datum R. | — |
| `AdjointRepresentationDecomposition(R)` | The decomposition multiset of the adjoint representation; highest weight = highest root of R, multiplicity 1. R must be weakly simply connected. | — |

*Worked example: H104E1 (adjoint representation of D4 and its highest weight).*

---

## 104.3 Constructing Representations

### 104.3.1 Lie Algebras

Functions applicable to almost reductive structure-constant Lie algebras. If L has large dimension, computing preimage information is expensive; set `ComputePreImage := false` to skip it when preimages are not required.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `TrivialRepresentation(L)` | The one-dimensional trivial representation of Lie algebra L over its base ring. | — |
| `AdjointRepresentation(L)` | The adjoint representation of L acting on itself. Parameter: `ComputePreImage` (BoolElt, default true). | — |
| `StandardRepresentation(L)` | The smallest-dimensional faithful representation of semisimple L (standard/natural representation). Requires non-degenerate Killing form. Parameter: `ComputePreImage` (BoolElt, default true). Errors if characteristic divides coisogeny order; use simply connected root datum to avoid this. | — |
| `HighestWeightRepresentation(L, w)` | Representation of L with highest weight w (vector or sequence of length rank); returns a function mapping L-elements to matrices. | de Graaf's algorithm **[dG01]** |
| `HighestWeightModule(L, w)` | The irreducible L-module with highest weight w (sequence of non-negative integers, length = rank of root datum of L); returned as a left module over L. | de Graaf's algorithm **[dG01]** |
| `TensorProduct(Q)` | Tensor product of the sequence Q of left-modules over a Lie algebra; also returns the canonical multilinear map from the Cartesian product. | — |
| `SymmetricPower(V, n)` | The nth symmetric power of left-module V over a Lie algebra (n ≥ 2), with the universal symmetric multilinear map. | — |
| `ExteriorPower(V, n)` | The nth exterior power of left-module V (2 ≤ n ≤ dim V), with the universal antisymmetric multilinear map. | — |

*Worked examples: H104E2 (StandardRepresentation of A2 in small characteristic), H104E3 (HighestWeightRepresentation of G2, 7-dimensional), H104E4 (TensorProduct and ExteriorPower of G2-modules with HighestWeightsAndVectors and DecomposeTensorProduct/DecomposeExteriorPower).*

### 104.3.2 Groups of Lie Type

Functions for constructing projective representations of split groups of Lie type. Modules are not yet implemented for groups of Lie type; only representations (homomorphisms). The optional parameter `NoWarning` suppresses the "Projective representation" warning.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `TrivialRepresentation(G)` | The one-dimensional trivial representation of the group of Lie type G over its base ring. | — |
| `StandardRepresentation(G)` | The smallest-dimensional highest weight (projective) representation of semisimple G (the natural representation for classical groups); warns if projective. | Cohen–Murray–Taylor **[CMT04]** (from Lie algebra representation) |
| `AdjointRepresentation(G)` | The adjoint (projective) representation of G over an extension of its base ring, given by the action of G on its Lie algebra; the Lie algebra is the second return value; warns if projective. | Cohen–Murray–Taylor **[CMT04]** |
| `LieAlgebra(G)` | The Lie algebra of the group of Lie type G, together with the adjoint representation; warns if projective. | — |
| `HighestWeightRepresentation(G, v)` | The highest weight (projective) representation with highest weight v of G over an extension of its base ring; warns if projective. | Cohen–Murray–Taylor **[CMT04]** |

*Worked example: H104E5 (StandardRepresentation of A2 over rationals, simply connected vs. adjoint form, projective warning, cube of toral element).*

---

## 104.4 Operations on Weight Multisets

### 104.4.1 Basic Operations

Access and arithmetic operations on decomposition multisets. Addition corresponds to direct sum; other arithmetic does not necessarily correspond to representation-theoretic operations.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RootDatum(D)` | The root datum over which the weight multiset D is defined. | — |
| `Weights(D)` / `WeightsAndMultiplicities(D)` | The sequences of weights and multiplicities in D. | — |
| `Multiset(D)` | Weights and multiplicities of D as a Magma multiset of vectors. | — |
| `Multiplicity(D, v)` | The multiplicity of weight v in D. | — |
| `D eq E` | True iff D and E have identical root data, weights, and multiplicities. | — |
| `D + E` | Direct sum (union) of weight multisets D and E (same root datum required). | — |
| `D + v` / `D +:= v` | Add weight v (sequence of length dim(RD) or element of Z^d) to multiset D; in-place variant. | — |
| `D +:= E` | Add multiset E to D in place (same root datum required). | — |
| `AddRepresentation(~D, E, c)` / `AddRepresentation(~D, E)` | Add c copies (default 1) of weight multiset E to D in place (identical root data required). | — |
| `AddRepresentation(~D, v, c)` / `AddRepresentation(~D, v)` | Add c copies (default 1) of weight v to D in place. | — |
| `D * c` / `D *:= c` | Scale all multiplicities of D by integer c; in-place variant. | — |
| `D / c` / `D /:= c` | Divide all multiplicities of D by integer c (error if not divisible); in-place variant. | — |
| `D * E` / `ProductRepresentation(D, E)` | Product of D and E viewed as polynomials (LiE convention **[vLCL92]**); root datum of result is direct sum of root data of D and E. Not the tensor product. | LiE polynomial product **[vLCL92]** |
| `ProductRepresentation(D, E, R)` | As above but the product is interpreted over the specified root datum R (error if dim(R) ≠ dim(RD) + dim(RE)). | LiE polynomial product **[vLCL92]** |
| `SubWeights(D, Q, S)` | Restriction to sub-coordinates: result over root datum S has highest weights w′ with w′ᵢ = w_{Q[i]}, same multiplicities. dim(S) must equal length of Q. | — |
| `PermuteWeights(D, pi, S)` | Permute weight components by permutation π ∈ Sym(d) and reinterpret over root datum S of the same dimension d. | — |

*Worked example: H104E6 (arithmetic with decompositions: D + v, PermuteWeights, SubWeights for A2 and A1).*

### 104.4.2 Conversion Functions

Converts between decomposition multisets, character multisets, and dominant character multisets. The user is responsible for tracking which kind of multiset is being used; passing the wrong kind yields meaningless results.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `VirtualDecomposition(C)` / `VirtualDecomposition(R, v)` | The virtual decomposition multiset of the virtual module with dominant character multiset C. Second form is shorthand for `VirtualDecomposition(LieRepresentationDecomposition(R, v))`. | — |
| `DecomposeCharacter(C)` | The decomposition multiset of the module with dominant character multiset C. Errors if C is virtual (negative multiplicities). | — |
| `DominantCharacter(D)` | The dominant character multiset with decomposition D. | Weight multiplicity algorithm **[vLCL92]** |

### 104.4.3 Calculating with Representations

Combinatorial operations on decomposition multisets, ported from LiE **[vLCL92]**. Most functions have two forms: taking a root datum R and a highest weight vector v, or taking a decomposition multiset D directly.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RepresentationDimension(D)` / `RepresentationDimension(R, v)` | Dimension of the module with decomposition D (or highest weight v over R). | Weyl dimension formula **[vLCL92]** |
| `CasimirValue(R, w)` | Value of the quadratic Casimir on the representation with highest weight w, normalised to 2 on the adjoint highest weight. (Due to Dr. Bruce Westbury, University of Warwick.) | — |
| `QuantumDimension(R, w)` | Returns two multisets Num and Den of positive integers such that ∏Num / ∏Den equals the ordinary dimension; replacing each integer by its quantum integer gives the quantum dimension. (Due to Dr. Bruce Westbury, University of Warwick.) | — |
| `Branch(FromGrp, ToGrp, v, M)` | Decomposition polynomial of the restriction of irreducible module V_v to ToGrp via restriction matrix M (dim(FromGrp)×dim(ToGrp)). Parameter: `Virtual` (BoolElt, default false) allows virtual weights in result. | LiE branching algorithm **[vLCL92]** |
| `Branch(ToGrp, D, M)` | As above with the module specified by decomposition D instead of a single highest weight. | LiE branching algorithm **[vLCL92]** |
| `Collect(R, D, M)` | Inverse of Branch: reconstruct an R-module from its restriction D using the (same) restriction matrix M used in Branch (Magma inverts M automatically, unlike LiE which requires the inverse). M must be square with dim equal to dim(RD) = dim(R). | LiE collect algorithm **[vLCL92]** |
| `TensorProduct(R, v, w)` | Decomposition multiset of the tensor product of highest weight modules V_v and V_w over R. Parameter: `Goal` (if set, returns only the multiplicity of that irreducible component, reducing memory consumption). | LiE tensor product algorithm **[vLCL92]** |
| `TensorProduct(D, E)` | Decomposition multiset of the tensor product of modules D and E. Parameter: `Goal` as above. | LiE tensor product algorithm **[vLCL92]** |
| `TensorProduct(Q)` | Decomposition multiset of the tensor product of the sequence of modules Q. Parameter: `Goal` as above. | LiE tensor product algorithm **[vLCL92]** |
| `TensorPower(R, n, v)` / `TensorPower(D, n)` | Decomposition of the nth tensor power of V_v^R or D. | LiE tensor product algorithm **[vLCL92]** |
| `AdamsOperator(R, n, v)` / `AdamsOperator(D, n)` | Decomposition polynomial of the virtual module obtained by applying the nth Adams operator to V_v^R or D. | LiE Adams operator **[vLCL92]** |
| `SymmetricPower(R, n, v)` / `SymmetricPower(D, n)` | Decomposition polynomial of S^n(V_v^R), the nth symmetric tensor power. | LiE symmetric power **[vLCL92]** |
| `AlternatingPower(R, n, v)` / `AlternatingPower(D, n)` | Decomposition polynomial of Alt^n(V_v^R), the nth alternating tensor power. | LiE alternating power **[vLCL92]** |
| `Plethysm(R, lambda, v)` / `Plethysm(D, lambda)` | Decomposition multiset of the plethysm of V_v^R corresponding to partition λ (a non-increasing sequence of positive integers summing to dim V_v^R): compose V_v^R with the GL(V_v^R)-representation indexed by λ. | Classical Frobenius formula **[And77, JK81]** |
| `Spectrum(R, v, t)` / `Spectrum(D, t)` | For toral element t = (a₁,…,aᵣ,n), returns a sequence where the ith entry is the multiplicity of the eigenvalue ζⁱ (ζ = e^{2πi/n}) in the action of t on V_v^R (or the module with decomposition D). | LiE spectrum algorithm **[vLCL92]** |
| `Demazure(R, v, w)` / `Demazure(D, w)` | Apply the Demazure operator M_{αᵢ} repeatedly, taking i from successive entries of Weyl word w. | LiE Demazure operator **[vLCL92]** |
| `Demazure(R, v)` / `Demazure(D)` | Equivalent to Demazure with w = longest Weyl word; if D is a decomposition polynomial, result is the character polynomial. Not the most efficient way to compute characters, but useful for verification. | LiE Demazure operator **[vLCL92]** |
| `LittlewoodRichardsonTensor(p, q)` | Tensor product of two highest weight SLn-modules expressed in partition coordinates (both sequences of the same length n), computed via the Littlewood-Richardson rule; returns sequences of partition coordinates P and multiplicities M. | Littlewood-Richardson rule **[vLCL92]** |
| `LittlewoodRichardsonTensor(P, M, Q, N)` | Tensor product of two modules in partition coordinates, each given as a list of partitions with multiplicities. | Littlewood-Richardson rule **[vLCL92]** |
| `LittlewoodRichardsonTensor(R, v, w)` | Tensor product of irreducible A_n representations with highest weights v and w via the Littlewood-Richardson rule (converts to partitions and back). | Littlewood-Richardson rule **[vLCL92]** |
| `LittlewoodRichardsonTensor(D, E)` | Tensor product of decompositions D and E using the Littlewood-Richardson rule. | Littlewood-Richardson rule **[vLCL92]** |
| `AlternatingDominant(D, w)` / `AlternatingDominant(R, wt, w)` | Apply successive Demazure-step simplifications for entries of Weyl word w: for each (weight v, multiplicity c) in D and simple reflection index i, if ⟨v, αᵢ⟩ ≥ 0 leave unchanged; if = −1 remove; if ≤ −2 replace by ((v + wᵢ)rᵢ − wᵢ, −c). Result has the same alternating Weyl sum as D. | LiE alternating dominant **[vLCL92]** |
| `AlternatingDominant(D)` / `AlternatingDominant(R, wt)` | Equivalent to AlternatingDominant with w = longest Weyl group element (somewhat faster); if D is interpreted as dominant weights, result contains highest weights with multiplicities. | LiE alternating dominant **[vLCL92]** |
| `AlternatingWeylSum(R, v)` / `AlternatingWeylSum(D)` | The alternating Weyl sum of V_v^R or D. Useful for demonstration; impractical for large groups due to the |W|-multiple number of terms. | LiE alternating Weyl sum **[vLCL92]** |

*Worked examples: H104E7 (RepresentationDimension and QuantumDimension of D4 adjoint), H104E8 (Branch and Collect for D4 restricted to A3T1), H104E9 (TensorPower: adjoint D4 up to 7th power, high-weight D4 up to 4th power), H104E10 (Spectrum and one-parameter subgroup restriction for A4), H104E11 (Demazure and AlternatingDominant round-trip for D4 adjoint), H104E12 (LittlewoodRichardsonTensor vs. ordinary tensor for A2 and A8; performance comparison), H104E13 (AlternatingDominant from a full character polynomial for D4 highest weight [1,5,2,1]).*

---

## 104.5 Operations on Representations

### 104.5.1 Lie Algebras

Functions for (explicit) modules and representations of almost reductive structure-constant Lie algebras.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CharacterMultiset(V)` / `CharacterMultiset(ρ)` | The character multiset of Lie algebra module V or representation ρ. | — |
| `Weights(V)` / `WeightsAndVectors(V)` | Two sequences: the weights in module V, and for each weight a basis of its weight space. | — |
| `Weights(ρ)` / `WeightsAndVectors(ρ)` | Two sequences: the weights of representation ρ, and for each weight a basis of the corresponding weight space in the underlying vector space. | — |
| `DecompositionMultiset(V)` / `DecompositionMultiset(ρ)` | The decomposition multiset of Lie algebra module V or representation ρ. | de Graaf **[dG01]** |
| `HighestWeightsAndVectors(V)` | Sequence of highest weights (weights of irreducible constituents) and corresponding highest weight vectors, whose generated submodules give a direct sum decomposition of V. | — |
| `DirectSum(U, V)` | Direct sum of Lie algebra modules U and V. | — |
| `DirectSumDecomposition(V)` / `IndecomposableSummands(V)` | Direct sum decomposition of module V as a sequence of indecomposable submodules. Over characteristic-zero semisimple Lie algebras, these are irreducible highest weight modules. | de Graaf **[dG01]** |
| `DirectSum(ρ, τ)` | Direct sum of Lie algebra representations ρ and τ. | — |
| `DirectSumDecomposition(ρ)` / `IndecomposableSummands(ρ)` | Direct sum decomposition of representation ρ as indecomposable subrepresentations; irreducible in characteristic zero over semisimple Lie algebras. | de Graaf **[dG01]** |
| `TensorProduct(Q)` | Tensor product of a sequence Q of left-modules over a Lie algebra; returns the module and the canonical map from the Cartesian product. | — |
| `SymmetricPower(V, n)` | nth symmetric power of left-module V (n ≥ 2) with universal symmetric multilinear map. | — |
| `ExteriorPower(V, n)` | nth exterior power of left-module V (2 ≤ n ≤ dim V) with universal antisymmetric multilinear map. | — |

*Worked example: H104E14 (TensorProduct and ExteriorPower of G2 modules, HighestWeightsAndVectors, DecomposeTensorProduct, DecomposeExteriorPower).*

### 104.5.2 Groups of Lie Type

Functions for projective representations of groups of Lie type. Modules are not yet implemented for groups of Lie type.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DirectSum(ρ, τ)` | Direct sum of group of Lie type representations ρ and τ. | — |
| `DirectSumDecomposition(ρ)` / `IndecomposableSummands(ρ)` | Direct sum decomposition of representation ρ as indecomposable subrepresentations; irreducible in characteristic zero. | Cohen–Murray–Taylor **[CMT04]** |
| `CharacterMultiset(V)` / `CharacterMultiset(ρ)` | Character weight multiset of group of Lie type representation ρ. | — |
| `Weights(ρ)` / `WeightsAndVectors(ρ)` | Weights of ρ together with corresponding weight vectors. | — |
| `WeightVectors(ρ)` | A basis of weight vectors of representation ρ. | — |
| `Weight(ρ, v)` | The weight corresponding to weight vector v in representation ρ. | — |
| `DecompositionMultiset(V)` / `DecompositionMultiset(ρ)` | Decomposition multiset of the group of Lie type representation ρ. | Cohen–Murray–Taylor **[CMT04]** |
| `HighestWeights(ρ)` | Highest weights of ρ with corresponding highest weight vectors. May fail for small finite fields. | — |
| `HighestWeightVectors(ρ)` | Highest weight vectors of representation ρ. | — |
| `GeneralisedRowReduction(ρ)` | Given a projective matrix representation ρ: G → GLm(k), returns its inverse. | Cohen–Murray–Taylor **[CMT04]** |

---

## 104.6 Other Functions for Representation Decompositions

### 104.6 (Main section)

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `FundamentalClosure(R, S)` | A set of fundamental roots of the minimal subsystem (not necessarily closed) of root system R containing all roots in S. Equivalent to the `fundam` function in LiE. S may contain roots or root indices; return type matches. | LiE `fundam` algorithm **[vLCL92]** |
| `Closure(R, S)` | A set of fundamental roots of the minimal closed subsystem of R containing all roots in S. Equivalent to the `closure` function in LiE. | LiE `closure` algorithm **[vLCL92]** |
| `RestrictionMatrix(R, Q)` | For a simply connected root datum R and a sequence of roots Q (as integers or vectors in root basis) forming a fundamental basis for a closed subdatum S, computes a restriction matrix for the fundamental Lie subgroup of type S. Roots in Q must be positive with mutually non-positive inner products for use in Branch/Collect. | — |
| `RestrictionMatrix(R, S)` | For a sub root datum S of R (e.g. constructed with `sub<…>`): the matrix M mapping fundamental weights of R to those of S. Not unique if rank(S) < rank(R). | — |
| `KLPolynomial(x, y)` | The Kazhdan–Lusztig polynomial P_{x,y} for Weyl group elements x, y. Parameter: `Ring` (RngUPol, default Z[X]). | Kazhdan–Lusztig recursion **[KL79]** |
| `RPolynomial(x, y)` | The R-polynomial R_{x,y} for Weyl group elements x, y. Parameter: `Ring` (RngUPol, default Z[X]). | Kazhdan–Lusztig recursion **[KL79]** |
| `Exponents(R)` | The exponents e₁ ≤ e₂ ≤ … ≤ eᵣ of root datum R, defined by the factorisation ∑_{w∈W} X^{l(w)} = ∏ᵢ ∑_{j=0}^{eᵢ} Xʲ. | — |
| `ToLiE(D)` | The LiE-syntax polynomial string equivalent of decomposition D. | — |
| `FromLiE(R, p)` | The decomposition multiset over R equivalent to LiE-syntax polynomial string p. | — |

*Worked examples: H104E15 (RestrictionMatrix for D4 and A1A1A1T1, weight-basis verification), H104E16 (KLPolynomial and RPolynomial for D4: verifying the KL recursion), H104E17 (Exponents of A3: Poincaré polynomial factorisation), H104E18 (ToLiE and FromLiE for B3).*

### 104.6.1 Operations Related to the Symmetric Group

Functions from LiE for working with the symmetric group and partition coordinates.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ConjugationClassLength(l)` | Order of the conjugacy class in S_n of permutations of cycle type l (n = sum of entries of l). | — |
| `PartitionToWeight(l)` | For a partition l with n parts, returns the corresponding weight for a group of type A_{n-1} in fundamental weight coordinates. | — |
| `WeightToPartition(v)` | For a weight v of length n (type A_n), returns n+1 partition coordinates. When v is dominant, this is a partition with n+1 parts. | — |
| `TransposePartition(l)` | The transpose (conjugate) partition of l. | — |

### 104.6.2 Fusion Rules

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `WZWFusion(R, v, w, k)` | Wess–Zumino–Witten fusion rules for weights v × w of R at level k, computed via the Kac–Walton formula. Weights may be finite (length rank(R)) or affine (length rank(R)+1). Parameter: `ReturnForm` (MonStgElt, default "Auto": match input form; or "Finite"; or "Affine"). R must be weakly simply connected. | Kac–Walton formula, **[FMS97]** Section 16.2 |
| `WZWFusion(D, E, k)` | Fusion rules for representations D and E at level k. | Kac–Walton formula, **[FMS97]** Section 16.2 |

*Worked example: H104E19 (WZWFusion for B3 at level 3, finite and affine weight forms).*

---

## 104.7 Subgroups of Small Rank

LiE's database of maximal proper subgroups of complex reductive simply connected Lie groups, for simple groups of rank at most 8, is available via the following functions.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LiEMaximalSubgroups()` | All maximal subgroups in the database, as a sequence of pairs (group name string, sequence of maximal subgroup name strings). | Database from LiE **[vLCL92]** |
| `MaximalSubgroups(G)` | Maximal subgroups of the complex reductive simply connected simple Lie group with Cartan type string G, as a sequence of strings. | Database from LiE **[vLCL92]** |
| `RestrictionMatrix(G, H)` | Restriction matrix for the maximal proper subgroup of type H of G. Parameter: `Index` (RngIntElt, default −1; must be set if multiple maximal subgroups of G have type H). | Database from LiE **[vLCL92]** |

*Worked example: H104E20 (MaximalSubgroups of E7, RestrictionMatrix for E7→A1 (Index 2), Branch of adjoint E7 to A1, dimension check).*

---

## 104.8 Subalgebras of su(d)

Functions for classifying irreducible simple subalgebras of su(d), following Dynkin **[Dyn57]**. Algorithms and implementation are due to Robert Zeier; for details and results see **[ZSH11]**. The verbose flag "SubSU" shows computation progress.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IrreducibleSimpleSubalgebrasOfSU(N)` | List of all irreducible simple subalgebras occurring in su(d) for 2 ≤ d ≤ N. | Dynkin's method **[Dyn57]**, Zeier–Schulte-Herbrüggen implementation **[ZSH11]** |
| `IrreducibleSimpleSubalgebraTreeSU(Q, d)` | The subalgebra tree for degree d as a directed graph derived from the list Q (from `IrreducibleSimpleSubalgebrasOfSU`). Vertex labels are records with fields: `algebra` (Cartan type string), `weights` (sequence of highest weight sparse vectors for irreducible representations related by outer automorphisms), and `type` (Frobenius–Schur indicator: −1 quaternionic, 1 real, 0 complex). | Dynkin's method **[Dyn57]**, **[ZSH11]** |
| `PrintTreesSU(Q, F)` | Print the subalgebra tree for sequence Q to LaTeX file F (overwritten). Parameters: `FromDegree` (RngIntElt, default 2), `ToDegree` (RngIntElt, default |Q|), `IncludeTrivial` (BoolElt, default true; if false, suppress degrees d where su(d) contains only the "obvious" subgroups). Algebras coloured: red (type −1), blue (type 1), black (type 0). | — |

*Worked example: H104E21 (IrreducibleSimpleSubalgebrasOfSU(2^10), subalgebra tree for su(12): A11→C6→A1 and A11→D6; listing all d where C6 appears; obtaining highest weight [2,0,0,0,0,0] for su(78)).*

---

## 104.9 Bibliography

| Key | Reference |
|-----|-----------|
| **[And77]** | C.M. Andersen. *Clebsch-Gordan Series for symmetrized tensor products.* J. Math. Phys, 8:988–997, 1977. |
| **[CMT04]** | Arjeh M. Cohen, Scott H. Murray, and D. E. Taylor. *Computing in groups of Lie type.* Math. Comp., 73(247):1477–1498, 2004. |
| **[dG01]** | W. A. de Graaf. *Constructing representations of split semisimple Lie algebras.* J. Pure Appl. Algebra, 164(1–2):87–107, 2001. Effective methods in algebraic geometry (Bath, 2000). |
| **[Dyn57]** | E. B. Dynkin. *Maximal Subgroups of the Classical Groups.* Amer. Math. Soc. Transl. Ser. 2, 6:245–378, 1957. |
| **[FMS97]** | Philippe Di Francesco, P. Mathieu, and D. Sénéchal. *Conformal Field Theory.* Graduate texts in contemporary physics. Springer, 1997. |
| **[JK81]** | G. James and A. Kerber. *The Representation Theory of the Symmetric Group.* Addison-Wesley, Reading MA, 1981. |
| **[KL79]** | D. Kazhdan and G. Lusztig. *Representations of Coxeter groups and Hecke algebras.* Inventiones Math., 53:165–184, 1979. |
| **[vLCL92]** | M.A.A. van Leeuwen, A.M. Cohen, and B. Lisser. *LiE, A package for Lie Group Computations.* CAN, Amsterdam, 1992. |
| **[ZSH11]** | Robert Zeier and Thomas Schulte-Herbrüggen. *Symmetry principles in quantum systems theory.* Journal of Mathematical Physics, 52(11):113510, 2011. |

---

## Algorithm-to-function quick reference

| Algorithm / Theory | Functions |
|--------------------|-----------|
| LiE combinatorial algorithms (dimension, tensor product, branching, Adams, symmetric/alternating powers, plethysm, spectrum, Demazure) **[vLCL92]** | `RepresentationDimension`, `TensorProduct`, `TensorPower`, `Branch`, `Collect`, `AdamsOperator`, `SymmetricPower`, `AlternatingPower`, `Plethysm`, `Spectrum`, `Demazure`, `AlternatingDominant`, `AlternatingWeylSum`, `DominantCharacter`, `VirtualDecomposition`, `DecomposeCharacter`, `ProductRepresentation`, `FundamentalClosure`, `Closure` |
| Littlewood-Richardson rule **[vLCL92]** | `LittlewoodRichardsonTensor` |
| Classical Frobenius formula for plethysm **[And77, JK81]** | `Plethysm` |
| de Graaf — constructing Lie algebra representations **[dG01]** | `HighestWeightRepresentation`, `HighestWeightModule`, `DirectSumDecomposition` (Lie algebras), `IndecomposableSummands` (Lie algebras), `DecompositionMultiset` (Lie algebras) |
| Cohen–Murray–Taylor — constructing group of Lie type representations **[CMT04]** | `StandardRepresentation(G)`, `AdjointRepresentation(G)`, `HighestWeightRepresentation(G, v)`, `DirectSumDecomposition(ρ)`, `DecompositionMultiset(ρ)`, `GeneralisedRowReduction(ρ)` |
| Kazhdan–Lusztig recursion **[KL79]** | `KLPolynomial`, `RPolynomial` |
| Kac–Walton formula (WZW fusion rules) **[FMS97]** | `WZWFusion` |
| LiE maximal subgroup database **[vLCL92]** | `LiEMaximalSubgroups`, `MaximalSubgroups`, `RestrictionMatrix(G, H)` |
| Dynkin's subalgebra classification **[Dyn57, ZSH11]** | `IrreducibleSimpleSubalgebrasOfSU`, `IrreducibleSimpleSubalgebraTreeSU`, `PrintTreesSU` |
| LiE–Magma format conversion | `ToLiE`, `FromLiE` |
| Symmetric group / partition utilities (LiE) **[vLCL92]** | `ConjugationClassLength`, `PartitionToWeight`, `WeightToPartition`, `TransposePartition` |
