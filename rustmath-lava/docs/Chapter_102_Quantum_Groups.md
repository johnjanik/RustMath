# Chapter 102 — Quantum Groups

**Handbook part:** XIII — Lie Theory
**Handbook pages:** 3073–3096 (PDF pages 3204–3229)

---

## Scope and overview

Chapter 102 describes Magma's functionality for quantized enveloping algebras (quantum groups). The notation and theoretical framework follow [Jan96]. The chapter first gives background sections fixing the definitions of Gaussian binomials, the quantized enveloping algebra Uq(L), its representations, PBW-type bases, the Z-form, the canonical basis of Kashiwara–Lusztig, and the Littelmann path model. Computational functions are then described for each of these objects.

In Magma, quantized enveloping algebras have type `AlgQUE` and their elements have type `AlgQUEElt`. These types inherit from `AlgPBW` and `AlgPBWElt` (general types for algebras with a PBW basis), which in turn inherit from `GenMPolB`, `Alg`, and `Rng` and their element types.

The algebra Uq(L) is constructed from a root datum R. Its generators split into three groups corresponding to the subalgebras U⁻ (spanned by divided powers F(s)k), U⁰ (spanned by K-generators and their binomials [Ki; t]), and U⁺ (spanned by divided powers E(s)k). The integral Z-form UZ over Z[q, q⁻¹] (due to Lusztig [Lus90]) underlies all computations, and specialising q → ε recovers algebras at roots of unity or, at ε = 1, the universal enveloping algebra U(L).

The canonical basis of U⁻ was independently constructed by Kashiwara and Lusztig ([Lus93], [Lus96]); algorithms for computing canonical basis elements and Kashiwara crystal operators are due to de Graaf [Gra02]. Highest-weight representations are constructed using Gröbner-basis methods [Gra04]. The crystal graph of a highest-weight module can be computed efficiently via Littelmann's path model [Lit94, Lit95, Kas96] without constructing the module itself.

---

## 102.1 Introduction

Introductory section fixing notation and types. No intrinsics.

---

## 102.2 Background

Pure background: definitions and theorems with no Magma intrinsics.

### 102.2.1 Gaussian Binomials

Defines the Gaussian integer [n]v = v^(n−1) + v^(n−3) + ··· + v^(−n+1), the Gaussian factorial [n]v!, and the Gaussian binomial C(n,k)v = [n]!/([k]![n−k]!).

### 102.2.2 Quantized Enveloping Algebras

Defines Uq(L) by generators Fα, Kα, K⁻¹α, Eα (α ∈ Δ) subject to q-Serre relations. Introduces the bar-automorphism (q ↦ q⁻¹, Kα ↦ K⁻¹α), the automorphism ω (Eα ↔ Fα, Kα ↦ K⁻¹α), the anti-automorphism τ (Eα ↦ Eα, Fα ↦ Fα, Kα ↦ K⁻¹α), and diagram automorphisms from Dynkin diagram symmetries.

### 102.2.3 Representations of Uq(L)

For each dominant weight λ there is a unique irreducible highest-weight module V(λ) with the same character as the classical module. Every finite-dimensional Uq(L)-module is a direct sum of these. Construction uses Gröbner bases [Gra04]. Also defines the Hopf algebra structure (comultiplication Δ, counit ε, antipode S) and twisted Hopf structures via automorphisms or anti-automorphisms.

### 102.2.4 PBW-type Bases

Via a reduced expression w0 = si1···sit for the longest Weyl group element, the automorphisms Tα define root vectors Fk, Ek. The divided powers F(m1)1···F(mt)t and E(n1)1···E(nt)t form PBW-type bases of U⁻ and U⁺. Product rules for PBW basis elements are due to [Gra01].

### 102.2.5 The Z-form of Uq(L)

The Lusztig integral form UZ over Z[q, q⁻¹] ([Lus90], Theorem 6.7) is the Z[q, q⁻¹]-span of basis elements F(k1)1···F(kt)t · Kδ1α1 C(Kα1, m1) ··· C(Kαl, ml) · E(n1)1···E(nt)t. Specialisation q → ε (nonzero in a field F) gives Uε; at ε = 1 modulo (Kαi − 1) one recovers U(L). The map Uq(L) → U(L) sends the integral basis to an integral basis of U(L) [Lus90].

### 102.2.6 The Canonical Basis

The canonical basis B of U⁻ (Lusztig [Lus93], Theorem 42.1.10; [Lus96], Proposition 8.2) is the unique basis whose elements are bar-invariant and, for any choice of w0, each element X = x + Σ ζixi with x the principal w0-monomial and ζi ∈ qZ[q]. The Kashiwara operators F̃α, Ẽα on B ∪ {0} are defined via the transition matrix R between two reduced expressions for w0. The canonical basis Bλ = {X·vλ | X ∈ B} \ {0} of V(λ) carries the same crystal-graph structure. Algorithms for computing Kashiwara operators and canonical basis elements are given in [Gra02].

### 102.2.7 The Path Model

Littelmann's path operators fα, eα act on piecewise-linear paths in the real weight space. Starting from the straight-line path ξλ, the set Πλ of nonzero paths obtained by applying fαi's consists precisely of Lakshmibai–Seshadri (LS) paths of shape λ. A theorem of Littelmann ([Lit95], Theorem 9.1) identifies path multiplicities with weight-space dimensions. The crystal graph Γ defined on Πλ is isomorphic to the crystal graph of V(λ) ([Kas96]).

---

## 102.3 Gauss Numbers

Standalone functions for Gaussian combinatorics, working over any ring containing a suitable element v.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `GaussNumber(n, v)` | Returns the Gauss number [n]v = v^(n−1) + v^(n−3) + ··· + v^(−n+1) for integer n and ring element v. | Direct evaluation of the defining polynomial in v. |
| `GaussianFactorial(n, v)` | Returns the Gaussian factorial [n]v! = [n]v · [n−1]v ··· [1]v. | Iterative product of Gauss numbers. |
| `GaussianBinomial(n, k, v)` | Returns the Gaussian binomial C(n,k)v = [n]v! / ([k]v! · [n−k]v!) for integers n, k and ring element v. | Ratio of Gaussian factorials. |

*Worked examples: H102E1 (computing GaussianBinomial(5, 3, q²) over Q(q)).*

---

## 102.4 Construction

Functions for constructing the quantized enveloping algebra from a root datum. The algebra is defined over Q(q), the rational function field in one variable q over Q. If the root datum has n positive roots and rank r, the algebra has 2n + r generators: F1,…,Fn (weight −βk, generators of U⁻), K1,…,Kr (generators of U⁰), E1,…,En (weight βk, generators of U⁺). Divided powers F(s)k and E(s)k are used throughout.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `QuantizedUEA(R)` / `QuantizedUEAlgebra(R)` / `QuantizedUniversalEnvelopingAlgebra(R)` | Given a root datum R, constructs the quantized enveloping algebra U over Q(q). Optional parameter `w0` (SeqEnum, default: lexicographically smallest reduced expression): a sequence of simple-root indices specifying a reduced expression for the longest Weyl group element; the PBW basis relative to that expression is used. | PBW-type basis construction **[Gra01]**; uses the integral Z-form basis **[Lus90]**. |
| `AssignNames(U, S)` | Assigns the names in sequence S to the generators of U. | — |
| `ChangeRing(U, R)` | Returns the algebra identical to U but with coefficient ring changed to R. | — |

*Worked examples: H102E2 (constructing Uq of type C3, two different w0 choices, multiplying generators).*

---

## 102.5 Related Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CoefficientRing(U)` | Returns the coefficient ring of the quantized enveloping algebra U (by default the rational function field Q(q)). | — |
| `RootDatum(U)` | Returns the root datum corresponding to U. | — |
| `PositiveRootsPerm(U)` | Returns a sequence of integers 1..n (n = number of positive roots) such that if the k-th entry is m, then the generator Fk has weight −βm and Ek has weight βm, where βm is the m-th positive root of the root datum. | — |

*Worked examples: H102E3 (D4 example showing CoefficientRing, RootDatum, PositiveRootsPerm).*

---

## 102.6 Operations on Elements

Generators are accessed as `U.i`. For 1 ≤ i ≤ n: Fi; for n+1 ≤ i ≤ n+r: Ki−n; for n+r+1 ≤ i ≤ 2n+r: Ei−n−r. Note that exponentiation of Fk or Ek uses divided powers: Fsk = [s]! · F(s)k.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `x + y`, `x - y`, `x * y`, `c * x`, `x * c`, `x ^ n` | Standard ring arithmetic on elements of U. Exponentiation scales by the appropriate divided-power factorial. | PBW multiplication rules **[Gra01]**. |
| `U ! 0` / `Zero(U)` | Returns the zero element of U. | — |
| `U ! 1` / `One(U)` | Returns the identity element of U. | — |
| `U . i` | Returns the i-th generator of U (Fi, Kj, or Ek as described above). | — |
| `U ! r` | Coerces r (from the coefficient ring or a compatible quantized enveloping algebra) as an element of U. | — |
| `KBinomial(U, i, s)` / `KBinomial(K, s)` | Returns the element [Ki; s] (the K-binomial), which is a basis element of U⁰. First form takes U, an index i (1 ≤ i ≤ rank), and positive integer s. Second form takes a generator element K = Ki directly. | Defined by the Z-form formula **[Lus90]**. |
| `Monomials(u)` | Returns the sequence of PBW monomials occurring in u; corresponds elementwise to Coefficients(u). | — |
| `Coefficients(u)` | Returns the sequence of coefficients of the PBW monomials in u; corresponds elementwise to Monomials(u). | — |
| `K ^ -1` | Given a generator K = Ki of U, returns its inverse K⁻¹i. | — |
| `Degree(u, i)` | For u ∈ U and index i corresponding to Fi (1 ≤ i ≤ n) or Ek (n+r+1 ≤ i ≤ 2n+r): returns the degree of u in that generator. | — |
| `KDegree(m, i)` | For a single monomial m and 1 ≤ i ≤ r: returns a tuple ⟨d, k⟩ where d ∈ {0,1} and k ≥ 0. If d = 0, the factor [Ki; k] occurs in m; if d = 1, the factor Ki[Ki; k] occurs. | — |

*Worked examples: H102E4 (G2 example: Monomials, Coefficients, Degree, KDegree, inverse of K).*

---

## 102.7 Representations

Functions for constructing and analysing left modules over a quantized enveloping algebra. For background on algebra modules in Magma see Chapter 89.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HighestWeightRepresentation(U, w)` | Given U (rank r) and a sequence w of r non-negative integers (highest weight), returns the irreducible representation of U with that highest weight as a function: given an element of U, it computes the corresponding matrix. | Gröbner-basis construction of V(λ) **[Gra04]**. |
| `HighestWeightModule(U, w)` | Given U and highest weight w (sequence of r non-negative integers), returns the irreducible U-module V(λ) as a left module over U. | Gröbner-basis construction **[Gra04]**. |
| `WeightsAndVectors(V)` | Returns two parallel sequences: (1) the weights occurring in V; (2) for each weight, a basis of the corresponding weight space (as a sequence of elements of V). | Weight-space decomposition. |
| `HighestWeightsAndVectors(V)` | Returns two parallel sequences: (1) the highest weights of the irreducible constituents of V; (2) for each, the corresponding highest-weight vectors. The submodules generated by these vectors give a direct sum decomposition of V. | — |
| `CanonicalBasis(V)` | Returns the canonical basis of the left module V over a quantized enveloping algebra. If V is not irreducible, returns the union of the canonical bases of its irreducible components. | Canonical/crystal basis algorithm **[Gra02]**; Kashiwara–Lusztig theory **[Lus93, Lus96]**. |
| `TensorProduct(Q)` | Given a sequence Q of left U-modules, returns their tensor product module M together with a map from the Cartesian product of the elements of Q to M (sending a tuple t to the tensor product of its entries). Uses the Hopf comultiplication on U. | Comultiplication Δ on Uq(L) (standard Hopf structure or twisted, if set). |

*Worked examples: H102E5 (G2 HighestWeightRepresentation and HighestWeightModule); H102E6 (B2 CanonicalBasis, action via ModuleWithBasis/ActionMatrix); H102E7 (B2 TensorProduct, HighestWeightsAndVectors of V1 ⊗ V2).*

---

## 102.8 Hopf Algebra Structure

Functions for the Hopf algebra structure of Uq(L) (comultiplication, counit, antipode) and for twisted variants. The default Hopf structure is as in §102.2.3; twisted structures are obtained by pre/post-composing with an automorphism or anti-automorphism f (and its inverse g).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `UseTwistedHopfStructure(U, f, g)` | Sets U to use the twisted Hopf algebra structure determined by (anti-)automorphism f and its inverse g. Must be called before any use of the Hopf structure (including creating tensor products). g is the inverse of f (not checked). | Twisted comultiplication Δf = (f ⊗ f) ∘ Δ ∘ f⁻¹; twisted antipode Sf = f ∘ S ∘ f⁻¹ (automorphism) or f ∘ S⁻¹ ∘ f⁻¹ (anti-automorphism); twisted counit εf = ε ∘ f⁻¹. |
| `HasTwistedHopfStructure(U)` | Returns true if U has been set to use a twisted Hopf structure; if true, also returns the (anti-)automorphism and its inverse. | — |
| `Counit(U)` | Returns the counit ε: U → Q(q) of the quantized enveloping algebra U. | Standard Hopf counit: ε(Eα) = ε(Fα) = 0, ε(Kα) = 1. |
| `Antipode(U)` | Returns the antipode S of U as an anti-automorphism. | Standard Hopf antipode: S(Eα) = −K⁻¹α Eα, S(Fα) = −FαKα, S(Kα) = K⁻¹α. |
| `Comultiplication(U, d)` | Returns the degree-d comultiplication of U as a map from U to the d-fold tensor power of U (d ≥ 2). Higher-degree comultiplications are obtained by iterating the degree-2 map. Elements of the d-fold tensor power are represented as a list of (d-tuple, coefficient) pairs. | Iterated Hopf comultiplication: Δ(Eα) = Eα ⊗ 1 + Kα ⊗ Eα, Δ(Fα) = Fα ⊗ K⁻¹α + 1 ⊗ Fα, Δ(Kα) = Kα ⊗ Kα. |

*Worked examples: H102E8 (A3 Comultiplication of degree 2 applied to F1).*

---

## 102.9 Automorphisms

Standard automorphisms and anti-automorphisms of Uq(L). All returned maps store their inverse, retrievable via `Inverse`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BarAutomorphism(U)` | Returns the bar-automorphism of U: the Q-algebra automorphism defined by Fα ↦ Fα, Kα ↦ K⁻¹α, Eα ↦ Eα, q ↦ q⁻¹ (§102.2.2). | Definition: bar-automorphism of Uq(L) **[Jan96]**. |
| `AutomorphismOmega(U)` | Returns the automorphism ω of U defined by ω(Fα) = Eα, ω(Eα) = Fα, ω(Kα) = K⁻¹α (§102.2.2). | Definition: ω-automorphism of Uq(L) **[Jan96]**. |
| `AntiAutomorphismTau(U)` | Returns the anti-automorphism τ of U defined by τ(Fα) = Fα, τ(Eα) = Eα, τ(Kα) = K⁻¹α (§102.2.2). | Definition: τ anti-automorphism **[Jan96]**. |
| `AutomorphismTalpha(U, k)` | For integer k between 1 and the rank of the root datum, returns the automorphism Tαk of U (the Lusztig braid group automorphism corresponding to the k-th simple root; §102.2.4). | Braid group / reflection automorphisms **[Lus93]**. |
| `DiagramAutomorphism(U, p)` / `GraphAutomorphism(U, p)` | Given a permutation p of {1,…,r} that is a diagram automorphism of the root datum (leaves the Dynkin diagram invariant), returns the induced automorphism of U (§102.2.2). | Dynkin diagram symmetry acting on generators. |

*Worked examples: H102E9 (G2 BarAutomorphism; C3 verification of T⁻¹α = τ ∘ Tα ∘ τ; D4 DiagramAutomorphism mapping canonical basis elements).*

---

## 102.10 Kashiwara Operators

Kashiwara operators F̃i and Ẽi acting on monomials in U⁻ (the negative part of a quantized enveloping algebra), as defined in §102.2.6.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Falpha(m, i)` | Given a monomial m in U⁻ (i.e., in the first n generators of U) and 1 ≤ i ≤ rank, returns the monomial F̃i(m) obtained by applying the i-th Kashiwara operator (§102.2.6). | Kashiwara crystal operator on PBW monomials via transition matrices between reduced expressions **[Gra02]**. |
| `Ealpha(m, i)` | Given a monomial m in U⁻ and 1 ≤ i ≤ rank, returns Ẽi(m) if applicable; returns the zero element of U if the first exponent of the relevant ew0-monomial is 0 (§102.2.6). | Kashiwara crystal operator **[Gra02]**. |

*Worked examples: H102E10 (F4 example of Falpha and Ealpha on a monomial in U⁻).*

---

## 102.11 The Path Model

Functions for Littelmann's path model (§102.2.7). LS-paths are represented as pairs (weight sequence, rational sequence). A special zero path is recognised; path operators cannot be applied to it but may produce it.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DominantLSPath(R, hw)` | Given a root datum R and a sequence hw of non-negative integers, returns the dominant LS-path: the straight line from the origin to hw. This is the starting path ξλ = ((λ), (0,1)). | Littelmann path model **[Lit94, Lit95]**. |
| `Falpha(p, i)` | Given a non-zero path p and 1 ≤ i ≤ rank, returns the result of applying the path operator fαi to p. | Path operator fα **[Lit94, Lit95]**. |
| `Ealpha(p, i)` | Given a non-zero path p and 1 ≤ i ≤ rank, returns the result of applying the path operator eαi to p. | Path operator eα **[Lit94, Lit95]**. |
| `WeightSequence(p)` | For a path p, returns the sequence of weights (ν1,…,νs) defining the path (§102.2.7). | — |
| `RationalSequence(p)` | For a path p, returns the sequence of rationals (a0=0, a1, …, as=1) defining the path (§102.2.7). | — |
| `EndpointWeight(p)` | Returns the weight p(1), the endpoint of the path. | — |
| `Shape(p)` | Returns the dominant weight λ that is the shape of the path p. | — |
| `WeylWord(p)` | Returns a reduced expression (as a sequence of integers 1..rank) for the shortest Weyl group element σ such that σ(λ) = ν1, where λ is the shape and ν1 is the first weight in WeightSequence(p). | Reduced expression from Bruhat order. |
| `IsZero(p)` | Returns true if p is the zero path, false otherwise. | — |
| `p1 eq p2` | Returns true if paths p1 and p2 are equal, false otherwise. | — |
| `CrystalGraph(R, hw)` | Given a root datum R and a sequence hw of non-negative integers (length = rank), returns the crystal graph G (a directed labelled digraph with labels in 1..rank) and a sequence of LS-paths. An edge from i to j with label s means fαs(pi) = pj. Isomorphic to the crystal graph of V(λ) by **[Kas96]**. | Littelmann path model **[Lit94, Lit95]**; crystal graph isomorphism **[Kas96]**. |

*Worked examples: H102E11 (B2 DominantLSPath, Falpha, Ealpha, WeightSequence, RationalSequence, WeylWord); H102E12 (G2 CrystalGraph with 14 vertices, edge/label inspection).*

---

## 102.12 Elements of the Canonical Basis

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CanonicalElements(U, w)` | Given U (rank r root datum) and a sequence w of r non-negative integers, returns the sequence of elements of the canonical basis of U⁻ of weight ν = w[1]α1 + ··· + w[r]αr. Elements are expressed as linear combinations (with coefficients in Z[q]) of PBW monomials, with the principal monomial appearing with coefficient 1 (or q-power leading term). All returned elements are bar-invariant. | Canonical basis algorithm **[Gra02]**; bar-invariance from Lusztig **[Lus93, Lus96]**. |

*Worked examples: H102E13 (F4 canonical elements of weight [1,2,1,1], bar-invariance check); H102E14 (A2 crystal graph used to predict which canonical basis element of weight [2,2] acts non-trivially on highest-weight vector of V([1,1])).*

---

## 102.13 Homomorphisms to the Universal Enveloping Algebra

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `QUAToIntegralUEAMap(U)` | Given U, returns the map from U onto the integral form of the universal enveloping algebra U(L) of the corresponding semisimple Lie algebra (obtained by specialising q → 1 and quotienting by Kαi − 1; §102.2.5). See §100.17 for universal enveloping algebras in Magma. | Specialisation map from UZ to U(L) **[Lus90]**. |

*Worked examples: H102E15 (C3 example mapping canonical basis elements of weight [1,2,1] to integral UEA elements, obtaining canonical basis of U(L)).*

---

## 102.14 Bibliography

| Key | Reference |
|-----|-----------|
| **[Gra04]** | W. A. de Graaf. Five constructions of representations of quantum groups. *Note di Matematica*, 22(1):27–48, 2003/04. |
| **[Gra01]** | W. A. de Graaf. Computing with quantized enveloping algebras: PBW-type bases, highest-weight modules, R-matrices. *J. Symbolic Comput.*, 32(5):475–490, 2001. |
| **[Gra02]** | W. A. de Graaf. Constructing canonical bases of quantized enveloping algebras. *Experimental Mathematics*, 11(2):161–170, 2002. |
| **[Jan96]** | J. C. Jantzen. *Lectures on Quantum Groups*, volume 6 of Graduate Studies in Mathematics. American Mathematical Society, 1996. |
| **[Kas96]** | M. Kashiwara. Similarity of crystal bases. In *Lie algebras and their representations (Seoul, 1995)*, pages 177–186. Amer. Math. Soc., Providence, RI, 1996. |
| **[Lit94]** | P. Littelmann. A Littlewood-Richardson rule for symmetrizable Kac-Moody algebras. *Invent. Math.*, 116(1–3):329–346, 1994. |
| **[Lit95]** | P. Littelmann. Paths and root operators in representation theory. *Ann. of Math. (2)*, 142(3):499–525, 1995. |
| **[Lus90]** | G. Lusztig. Quantum groups at roots of 1. *Geom. Dedicata*, 35(1–3):89–113, 1990. |
| **[Lus93]** | G. Lusztig. *Introduction to quantum groups*. Birkhäuser Boston Inc., Boston, MA, 1993. |
| **[Lus96]** | G. Lusztig. Braid group action and canonical bases. *Adv. Math.*, 122(2):237–261, 1996. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Gaussian combinatorics (q-analogues) | `GaussNumber`, `GaussianFactorial`, `GaussianBinomial` |
| PBW-type basis multiplication **[Gra01]** | `QuantizedUEA`, `x * y`, `x ^ n`, `Monomials`, `Coefficients` |
| Z-form / integral basis **[Lus90]** | `QuantizedUEA`, `KBinomial`, `QUAToIntegralUEAMap` |
| Gröbner-basis representation construction **[Gra04]** | `HighestWeightRepresentation`, `HighestWeightModule` |
| Canonical / crystal basis **[Lus93, Lus96, Gra02]** | `CanonicalBasis`, `CanonicalElements` |
| Kashiwara crystal operators on U⁻ **[Gra02]** | `Falpha(m, i)`, `Ealpha(m, i)` |
| Littelmann path model **[Lit94, Lit95]** | `DominantLSPath`, `Falpha(p, i)`, `Ealpha(p, i)`, `WeightSequence`, `RationalSequence`, `EndpointWeight`, `Shape`, `WeylWord`, `IsZero`, `eq` |
| Crystal graph isomorphism **[Kas96]** | `CrystalGraph` |
| Hopf algebra structure | `Counit`, `Antipode`, `Comultiplication`, `UseTwistedHopfStructure`, `HasTwistedHopfStructure`, `TensorProduct` |
| Bar / ω / τ automorphisms **[Jan96]** | `BarAutomorphism`, `AutomorphismOmega`, `AntiAutomorphismTau` |
| Braid group (Tα) automorphisms **[Lus93]** | `AutomorphismTalpha` |
| Diagram automorphisms | `DiagramAutomorphism`, `GraphAutomorphism` |
