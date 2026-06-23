# Chapter 65 — Almost Simple Groups

**Handbook part:** IX — Finite Groups
**Handbook pages:** 1879–1934 (PDF pages 2008–2067)

---

## Scope and overview

Chapter 65 describes tools for working with finite almost-simple groups (AS-groups). The
programme for non-soluble finite groups aims to reduce many problems about a non-soluble
group G to the same problem for the non-abelian simple composition factors of G.

The material falls into two main categories:

1. **Recognition** — functions that identify a particular group S known to be almost simple
   with a standard copy T. Recognition is split into *non-constructive* (asserting an
   isomorphism) and *constructive* (returning an explicit isomorphism). The constructive
   isomorphism can then be used to transfer questions from S into the well-understood
   standard group T.

2. **Properties** — functions that compute structural information about an AS-group (conjugacy
   classes, maximal subgroups, Sylow subgroups, etc.), implemented separately for each family.
   Using the recognition isomorphism, this information transfers back to the user's group S.

These functions do **not** assume that a base-and-strong-generating-set (BSGS) representation
can be constructed; hence they apply to groups of much larger order or much larger dimension
than those handled by the techniques of Chapters 58 and 59.

The techniques are described as under development and incomplete in their coverage at the
time of writing.

---

## 65.1 Introduction

### 65.1.1 Overview

*See scope and overview above.*

---

## 65.2 Creating Finite Groups of Lie Type

The construction functions return groups defined by generating matrices. As shown by Chevalley,
for each simple Lie algebra L over **C** and finite field F_q there is a matrix group L(q),
generally perfect but not simple (quotient by the centre gives the simple group). Steinberg,
Ree and others showed that when the Coxeter graph has an automorphism of order t, a twisted
version ᵗL(q) exists.

Generators for series A, C, ²A and ²B are from **[Tay87]**; for B, D and ²D from Rylands and
Taylor **[RT98]**; for exceptional groups from Howlett, Rylands and Taylor **[HRT01]**.

### 65.2.1 Generic Creation Function

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ChevalleyGroup(X, n, K: parameters)` / `ChevalleyGroup(X, n, q: parameters)` | Construct a matrix group over field K (or F_q) having the adjoint Chevalley group of Lie series X and Lie rank n as quotient modulo scalars. For series B, D and ²D the returned group is Ω(2n+1, q), Ω⁺(2n, q) or Ω⁻(2n, q) rather than the universal group. For twisted groups the first signature expects F_{q^t} while the second expects q. Supported series: "A" (SL(n+1,q)), "B" (Ω(2n+1,q)), "C" (Sp(2n,q)), "D" (Ω⁺(2n,q)), "E" (n ∈ {6,7,8}), "F" (n=4, F4(q)), "G" (n=2, G2(q); set `Irreducible := true` for the degree-6 representation), "2A" (SU(n+1,q)), "2B" (Sz(q)), "2D" (Ω⁻(2n,q)), "3D" (³D4(q)), "2E" (²E6(q)), "2F" (Ree group ²F4(q)), "2G" (Ree group ²G2(q)). Parameter: `Irreducible` (BoolElt, default false). | Generators from **[Tay87]** (series A, C, ²A, ²B), **[RT98]** (series B, D, ²D), **[HRT01]** (exceptional groups). |

### 65.2.2 The Orders of the Chevalley Groups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ChevalleyOrderPolynomial(type, n: parameters)` | Return the order polynomial in q for the universal Chevalley group X_n(q) or ᵗX_n(q). For twisted groups ²A_n, ³D4 and ²E6 the parameter q is the order of the fixed field of Frobenius. | Order formulae from the classification of finite simple groups. |
| `FactoredChevalleyGroupOrder(type, n, F: parameters)` / `FactoredChevalleyGroupOrder(type, n, q: parameters)` | Factored order of the Chevalley group of given type and rank over field F (or F_q). Default is the order of the group returned by `ChevalleyGroup`. Parameters: `Version` (MonStgElt, default "Default"; set to "Universal" or "Adjoint" for those variants), `Proof` (BoolElt, default true; passed to Magma's factorisation). | Polynomial evaluation with Magma's integer factorisation. |
| `ChevalleyGroupOrder(type, n, F: parameters)` / `ChevalleyGroupOrder(type, n, q: parameters)` | Unfactored order of the Chevalley group. Parameter: `Version` (MonStgElt, default "Default"). | As above, without factoring. |

### 65.2.3 Classical Groups

For most classical-group functions the group may be specified by: (i) degree n and field K; (ii) degree n and prime power q (unitary groups use F_{q²}); or (iii) a full vector space V = K^n.

#### 65.2.3.1 Linear Groups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `GeneralLinearGroup(n, q)` / `GeneralLinearGroup(n, K)` / `GeneralLinearGroup(V)` / `GL(n, q)` / `GL(n, K)` / `GL(V)` | General linear group GL(n, q) as a matrix group. `n` positive integer, `q` prime power, `K` finite field, `V` n-dimensional vector space over K. Abbreviated to `GL`. | Generating matrices; generators from **[Tay87]**. |
| `SpecialLinearGroup(n, q)` / `SpecialLinearGroup(n, K)` / `SpecialLinearGroup(V)` / `SL(n, q)` / `SL(n, K)` / `SL(V)` | Special linear group SL(n, q) as a matrix group. Abbreviated to `SL`. | Generating matrices; generators from **[Tay87]**. |
| `AffineGeneralLinearGroup(GrpMat, n, q)` / `AffineGeneralLinearGroup(GrpMat, n, K)` / `AffineGeneralLinearGroup(GrpMat, V)` / `AGL(GrpMat, V)` | Affine general linear group AGL(n, q) as a subgroup of GL(n+1, K). `n ≥ 2`. If the category name `GrpMat` is omitted the result is a permutation group. Abbreviated to `AGL`. | Embedding in GL(n+1, K). |
| `AffineSpecialLinearGroup(GrpMat, n, q)` / `AffineSpecialLinearGroup(GrpMat, n, K)` / `AffineSpecialLinearGroup(GrpMat, V)` / `ASL(GrpMat, V)` | Affine special linear group ASL(n, q) as a subgroup of SL(n+1, K). `n ≥ 2`. If `GrpMat` omitted the result is a permutation group. Abbreviated to `ASL`. | Embedding in SL(n+1, K). |

#### 65.2.3.2 Unitary Groups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ConformalUnitaryGroup(n, q)` / `ConformalUnitaryGroup(n, K)` / `ConformalUnitaryGroup(V)` / `CU(n, q)` / `CU(n, K)` / `CU(V)` | Conformal unitary group CU(n, q) (preserves a unitary form up to a constant). `n ≥ 2`, `K = F_{q²}`. Abbreviated to `CU`. | Generators from **[Tay87]**. |
| `GeneralUnitaryGroup(n, q)` / `GeneralUnitaryGroup(n, K)` / `GeneralUnitaryGroup(V)` / `GU(n, q)` / `GU(n, K)` / `GU(V)` | General unitary group GU(n, q). `n ≥ 2`, `K = F_{q²}`. Abbreviated to `GU`. | Generators from **[Tay87]**. |
| `SpecialUnitaryGroup(n, q)` / `SpecialUnitaryGroup(n, K)` / `SpecialUnitaryGroup(V)` / `SU(n, q)` / `SU(n, K)` / `SU(V)` | Special unitary group SU(n, q). `n ≥ 2`, `K = F_{q²}`. Abbreviated to `SU`. | Generators from **[Tay87]**. |

#### 65.2.3.3 Symplectic Groups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ConformalSymplecticGroup(n, q)` / `ConformalSymplecticGroup(n, K)` / `ConformalSymplecticGroup(V)` / `CSp(n, q)` / `CSp(n, K)` / `CSp(V)` | Conformal symplectic group CSp(n, q) (preserves a symplectic form up to a constant). `n` even, `n ≥ 4`. Abbreviated to `CSp`. | Generators from **[Tay87]**. |
| `SymplecticGroup(n, q)` / `SymplecticGroup(n, K)` / `SymplecticGroup(V)` / `Sp(n, q)` / `Sp(n, K)` / `Sp(V)` | Symplectic group Sp(n, q) in terms of two generating matrices. `n` even, `n ≥ 4`. Abbreviated to `Sp`. | Two-generator form; generators from **[Tay87]**. |

#### 65.2.3.4 Orthogonal and Spin Groups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ConformalOrthogonalGroup(n, q)` / `ConformalOrthogonalGroup(n, K)` / `ConformalOrthogonalGroup(V)` / `CO(n, q)` / `CO(n, K)` / `CO(V)` | Conformal orthogonal group CO(n, q) for odd n ≥ 3. Abbreviated to `CO`. | Generators from **[RT98]**. |
| `GeneralOrthogonalGroup(n, q)` / `GeneralOrthogonalGroup(n, K)` / `GeneralOrthogonalGroup(V)` / `GO(n, q)` / `GO(n, K)` / `GO(V)` | General orthogonal group GO(n, q) for odd n ≥ 3. Abbreviated to `GO`. | Generators from **[RT98]**. |
| `SpecialOrthogonalGroup(n, q)` / `SpecialOrthogonalGroup(n, K)` / `SpecialOrthogonalGroup(V)` / `SO(n, q)` / `SO(n, K)` / `SO(V)` | Special orthogonal group SO(n, q) for odd n ≥ 3. Abbreviated to `SO`. | Generators from **[RT98]**. |
| `ConformalOrthogonalGroupPlus(n, q)` / `ConformalOrthogonalGroupPlus(n, K)` / `ConformalOrthogonalGroupPlus(V)` / `COPlus(n, q)` / `COPlus(n, K)` / `COPlus(V)` | Conformal orthogonal group CO⁺(n, q) for even n ≥ 2. Abbreviated to `COPlus`. | Generators from **[RT98]**. |
| `GeneralOrthogonalGroupPlus(n, q)` / `GeneralOrthogonalGroupPlus(n, K)` / `GeneralOrthogonalGroupPlus(V)` / `GOPlus(n, q)` / `GOPlus(n, K)` / `GOPlus(V)` | General orthogonal group GO⁺(n, q) for even n ≥ 2. Abbreviated to `GOPlus`. | Generators from **[RT98]**. |
| `SpecialOrthogonalGroupPlus(n, q)` / `SpecialOrthogonalGroupPlus(n, K)` / `SpecialOrthogonalGroupPlus(V)` / `SOPlus(n, q)` / `SOPlus(n, K)` / `SOPlus(V)` | Special orthogonal group SO⁺(n, q) for even n ≥ 2. Abbreviated to `SOPlus`. | Generators from **[RT98]**. |
| `ConformalOrthogonalGroupMinus(n, q)` / `ConformalOrthogonalGroupMinus(n, K)` / `ConformalOrthogonalGroupMinus(V)` / `COMinus(n, q)` / `COMinus(n, K)` / `COMinus(V)` | Conformal orthogonal group CO⁻(n, q) for even n ≥ 2. Abbreviated to `COMinus`. | Generators from **[RT98]**. |
| `GeneralOrthogonalGroupMinus(n, q)` / `GeneralOrthogonalGroupMinus(n, K)` / `GeneralOrthogonalGroupMinus(V)` / `GOMinus(n, q)` / `GOMinus(n, K)` / `GOMinus(V)` | General orthogonal group GO⁻(n, q) for even n ≥ 2. Abbreviated to `GOMinus`. | Generators from **[RT98]**. |
| `SpecialOrthogonalGroupMinus(n, q)` / `SpecialOrthogonalGroupMinus(n, K)` / `SpecialOrthogonalGroupMinus(V)` / `SOMinus(n, q)` / `SOMinus(n, K)` / `SOMinus(V)` | Special orthogonal group SO⁻(n, q) for even n ≥ 2. Abbreviated to `SOMinus`. | Generators from **[RT98]**. |
| `Omega(n, q)` / `Omega(n, K)` / `Omega(V)` | Orthogonal group Ω(n, q) for odd n ≥ 3 in two generating matrices; the kernel of the spinor norm map on SO(n, K). | Generators from **[RT98]**. |
| `OmegaPlus(n, q)` / `OmegaPlus(n, K)` / `OmegaPlus(V)` | Orthogonal group Ω⁺(n, q) for even n ≥ 2 in two generating matrices; kernel of the spinor norm on SO⁺(n, K). | Generators from **[RT98]**. |
| `OmegaMinus(n, q)` / `OmegaMinus(n, K)` / `OmegaMinus(V)` | Orthogonal group Ω⁻(n, q) for even n ≥ 2 in two generating matrices; kernel of the spinor norm on SO⁻(n, K). | Generators from **[RT98]**. |
| `Spin(n, q)` / `Spin(n, K)` / `Spin(V)` | Spin group Spin(n, K) for odd n ≥ 1. | — |
| `SpinPlus(n, q)` / `SpinPlus(n, K)` / `SpinPlus(V)` | Spin group Spin⁺(n, K) for even n ≥ 2. | — |
| `SpinMinus(n, q)` / `SpinMinus(n, K)` / `SpinMinus(V)` | Spin group Spin⁻(n, K) for even n ≥ 4. | — |

*Worked examples: H65E1 (Sp(10, F_8)); H65E2 (SuzukiGroup over F_128, order and factored order).*

### 65.2.4 Exceptional Groups

#### 65.2.4.1 Suzuki Groups

Suzuki groups always have degree 4; arguments are: (i) a field K = F_{2^{2m+1}}; (ii) an integer q = 2^{2m+1}; or (iii) a vector space V = K^4.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SuzukiGroup(q)` / `SuzukiGroup(K)` / `SuzukiGroup(V)` | Simple Suzuki group Sz(q) in two generating matrices. q = 2^{2n+1}, K = F_q, V = K^4. Abbreviated to `Sz`. | Generators from **[Tay87]**. |

#### 65.2.4.2 Small Ree Groups

Small Ree groups ²G2(q) are given in an irreducible 7-dimensional matrix representation. Arguments: (i) K = F_{3^{2m+1}} with m > 0; (ii) q = 3^{2m+1} with m > 0; or (iii) V = K^7.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ReeGroup(q)` / `ReeGroup(K)` / `ReeGroup(V)` | Ree group ²G2(q) in standard generating matrices of degree 7. q = 3^{2m+1}, m > 0. Abbreviated to `Ree`. | Generators from **[HRT01]**. |

#### 65.2.4.3 Large Ree Groups

Large Ree groups ²F4(q) are given in an irreducible 26-dimensional matrix representation. Arguments: (i) K = F_{2^{2m+1}} with m > 0; (ii) q = 2^{2m+1} with m > 0; or (iii) V = K^{26}.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LargeReeGroup(q)` / `LargeReeGroup(K)` / `LargeReeGroup(V)` | Ree group ²F4(q) in standard generating matrices of degree 26. q = 2^{2m+1}, m > 0. Abbreviated to `LargeRee`. | Generators from **[HRT01]**. |

---

## 65.3 Group Recognition

### 65.3.1 Constructive Recognition of Alternating Groups

Constructive recognition of groups isomorphic to alternating or symmetric groups. The central algorithms are Beals et al. **[BLGN+03]** and Bratus–Pak **[BP00]**, implemented by Colva Roney-Dougal and Derek Holt.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RecogniseAlternatingOrSymmetric(G, n)` | Constructive recognition for G isomorphic to Alt(n) or Sym(n) with n > 11; succeeds with probability ≥ 1 − e^{−5}. Returns: success flag; symmetry flag (true = symmetric); a program that maps elements of an overgroup of G to permutations in S_n (with membership test); and the inverse program. | Beals–Leedham-Green–Niemeyer–Praeger–Seress black-box algorithm **[BLGN+03]**. |
| `RecogniseSymmetric(G, n: parameters)` | G known isomorphic to S_n for n ≥ 8. Returns true plus mutually inverse homomorphisms G ↔ S_n and G ↔ word group. If `Extension := true`, also handles 2.S_n (sixth return value distinguishes 2.S_n from 2.A_n). Parameters: `maxtries` (RngIntElt, default 100n + 5000), `Extension` (BoolElt, default false). | Bratus–Pak algorithm **[BP00]**, implemented by Derek Holt. |
| `SymmetricElementToWord(G, g)` | If G was constructively recognised as S_n (or 2.S_n), return true and an element of the word group evaluating to g; else false. Facilitates membership testing. | Uses recognition maps from `RecogniseSymmetric`. |
| `RecogniseAlternating(G, n: parameters)` | G known isomorphic to A_n for n ≥ 9. Returns true plus mutually inverse homomorphisms G ↔ A_n and G ↔ word group. If `Extension := true`, handles 2.A_n. Parameters: `maxtries` (RngIntElt, default 100n + 5000), `Extension` (BoolElt, default false). | Bratus–Pak algorithm **[BP00]**, implemented by Derek Holt. |
| `AlternatingElementToWord(G, g)` | If G was constructively recognised as A_n (or 2.A_n), return true and a word group element evaluating to g; else false. | Uses recognition maps from `RecogniseAlternating`. |
| `GuessAltsymDegree(G: parameters)` | G believed isomorphic to S_n or A_n (or extensions) for n > 6. Attempts to determine n and type by sampling element orders. Returns false if unable to decide after `maxtries` samples, or true plus type ("Symmetric"/"Alternating") and n. Warning: meaningless output if G is not of this form. Parameters: `maxtries` (RngIntElt, default 5000), `Extension` (BoolElt, default false). Written by Derek Holt. | Element-order statistics sampling (Monte Carlo). |

*Worked examples: H65E3 (RecogniseAlternatingOrSymmetric on A_13 in degree-78 action); H65E4 (GuessAltsymDegree then RecogniseAlternating on a 10-dimensional group over GF(5)).*

### 65.3.2 Determining the Type of a Finite Group of Lie Type

Given a finite quasisimple group of Lie type in any representation, probabilistic algorithms determine its defining characteristic and type.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LieCharacteristic(G : parameters)` | For a finite quasisimple permutation or matrix group G of Lie type, determine its defining characteristic. Monte Carlo — small probability of error. Parameters: `NumberRandom` (RngIntElt, default 100), `Verify` (BoolElt, default true; verifies G is perfect via `IsProbablyPerfect`). | Monte Carlo algorithm of Liebeck–O'Brien **[LO07]**. |
| `LieType(G, p : parameters)` | If G is nearly simple with non-abelian composition factor isomorphic to a group of Lie type in characteristic p, return true and its standard Chevalley name as a tuple ⟨s, n, q⟩ (Lie series, rank, field size — valid args for `ChevalleyGroup`); alternating groups give ⟨17, n, 0⟩; sporadic groups give ⟨18, n, str⟩. Monte Carlo. Parameter: `NumberRandom` (RngIntElt, default 100). | Babai–Kantor–Pálfy–Seress algorithm **[BKPS02]**; implemented by Malle and O'Brien. |
| `SimpleGroupName(G : parameters)` | If G is nearly simple, return true and a list of possible names for its non-abelian simple composition factor. Monte Carlo. Parameter: `NumberRandom` (RngIntElt, default 100). | Algorithm and implementation by Malle and O'Brien; uses `LieType` and `LieCharacteristic`. |

*Worked examples: H65E5 (LieCharacteristic on a 5-dimensional group over GF(4)); H65E6 (SimpleGroupName identifying Ω(7,5) as B3(5), M11, A5, and SL(2,5)).*

### 65.3.3 Classical Forms

Let G be an absolutely irreducible subgroup of GL(d, q). These functions compute classical forms of the underlying vector space V left invariant (or invariant modulo scalars) by G. Exit with error if G is reducible or not absolutely irreducible (unless `Scalars := true` and [G,G] is absolutely irreducible).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ClassicalForms(G: parameters)` | Find a classical form (symplectic, unitary, or orthogonal) preserved by absolutely irreducible G, or prove none exists. With `Scalars := true` finds a form preserved modulo scalars (guaranteed only when [G,G] is absolutely irreducible). Returns a record with fields: `formType`, `sign`, `bilinearForm`, `sesquilinearForm`, `quadraticForm`, `scalars`. The `formType` is one of "unknown", "linear", "symplectic", "unitary", "orthogonalcircle", "orthogonalplus", "orthogonalminus". Parameter: `Scalars` (BoolElt, default false). | Form-finding algorithm. |
| `SymplecticForm(G: parameters)` | If G preserves a symplectic form (modulo scalars with `Scalars := true`), return true and the form matrix; if known not to preserve one, return false; else exit with error. With `Scalars := true` also return scalar list. Parameter: `Scalars` (BoolElt, default false). | Specialisation of `ClassicalForms`. |
| `SymmetricBilinearForm(G: parameters)` | If G preserves an orthogonal form (modulo scalars), return true, the symmetric bilinear form matrix, and the form type (as in `ClassicalForms`). Parameter: `Scalars` (BoolElt, default false). | Specialisation of `ClassicalForms`. |
| `QuadraticForm(G)` | If G preserves a quadratic form (modulo scalars), return true, the upper-triangular form matrix, and the form type. Parameter: `Scalars` (BoolElt, default false). | Specialisation of `ClassicalForms`. |
| `UnitaryForm(G)` | If G preserves a unitary form (non-degenerate sesquilinear, modulo scalars), return true and the form matrix. Parameter: `Scalars` (BoolElt, default false). | Specialisation of `ClassicalForms`. |
| `FormType(G)` | If G preserves a classical form (modulo scalars), return its type string; otherwise return "unknown". Parameter: `Scalars` (BoolElt, default false). | Uses `ClassicalForms`. |
| `TransformForm(form, type)` | Return a matrix m such that G^m lies in the standard classical group (GU, Sp, or GO/GOPlus/GOMinus). `form` is the bilinear or sesquilinear form (quadratic form in characteristic 2 for orthogonal groups); `type` is one of "symplectic", "unitary", "orthogonalcircle", "orthogonalplus", "orthogonalminus". | Change-of-basis to standard form. |
| `TransformForm(G)` | Call `ClassicalForms` to find a form fixed by G, then return `TransformForm(form, type)`; or false if no form is found. Parameter: `Scalars` (BoolElt, default false). | Calls `ClassicalForms` then `TransformForm(form, type)`. |
| `SpinorNorm(g, form)` | Spinor norm of g with respect to an orthogonal form `form` (symmetric, nonsingular matrix). In even characteristic the spinor norm equals rank(g − I) mod 2 and `form` is ignored. | Definition of spinor norm; rank-mod-2 in char 2. |

*Worked example: H65E7 (ClassicalForms and FormType on Ω(9,11)).*

### 65.3.4 Recognizing Classical Groups in their Natural Representation

Let G be an irreducible subgroup of GL(d, q). The algorithm tests whether G contains Ω and is contained in Δ, for appropriate pairs (Ω, Δ) depending on the classical type.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RecognizeClassical(G : parameters)` | Test whether G contains the appropriate Ω and is contained in Δ. Returns true (with certainty, relying on CFSG), false (with small probability of false negative), or "Does not apply". Parameters: `Case` (MonStgElt, default "unknown"; restrict to one of "linear", "symplectic", "orthogonalplus", "orthogonalminus", "orthogonalcircle", "unitary"), `NumberOfElements` (RngIntElt, default 25), `Verbose Classical` (level up to 3). Cost O(d³ log d) bit operations for q < 2^{16}. | Niemeyer–Praeger **[NP97, NP98, NP99]** and Praeger **[Pra99]**; based on SL-recognition algorithm **[NP92]**; uses algorithms of Celler–Leedham-Green **[CLG97a, CLG97b]** and Celler et al. **[CLGM+95]**. |
| `IsLinearGroup(G)` | Test whether G ≤ GL(d, q) contains SL(d, q); small chance of false negative. | Calls `RecognizeClassical` with `Case := "linear"`. |
| `IsSymplecticGroup(G)` | Test whether G ≤ GSp(d, q) contains Sp(d, q); small chance of false negative. | Calls `RecognizeClassical` with `Case := "symplectic"`. |
| `IsOrthogonalGroup(G)` | Test whether G ≤ GO^ε(d, q) contains Ω^ε(d, q); small chance of false negative. | Calls `RecognizeClassical` with appropriate orthogonal case. |
| `IsUnitaryGroup(G)` | Test whether G ≤ GU(d, q) contains SU(d, q); small chance of false negative. | Calls `RecognizeClassical` with `Case := "unitary"`. |
| `ClassicalType(G)` | If G is known to be a classical subgroup of GL(d, q), return its type string; otherwise false. | Reads cached result from `RecognizeClassical`. |

*Worked example: H65E8 (SU(60, 9) and Sp(462, 3) recognised with RecognizeClassical).*

### 65.3.5 Constructive Recognition of Linear Groups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RecognizeSL2(G)` / `RecognizeSL2(G, q)` | If G (matrix or permutation) is isomorphic modulo scalars to (P)SL(2, q), construct homomorphisms. Returns: φ: G → (P)SL(2, q), τ: (P)SL(2, q) → G, γ: G → word group, δ: word group → G. If q is known it should be supplied; else `SL2Characteristic` is used to determine it. | Constructive recognition for SL(2, q) in natural representation: Conder–Leedham-Green–O'Brien **[CLGO06]**. Other representations: Brooksbank–O'Brien (unpublished). |
| `SL2ElementToWord(G, g)` | If G was constructively recognised as (P)SL(2, q), return true and a word group element evaluating to g; else false. | Uses recognition maps from `RecognizeSL2`. |
| `SL2Characteristic(G : parameters)` | Determine the characteristic and field size of G, assuming G has central quotient (P)SL(2, q). Monte Carlo, small probability of error. Parameters: `NumberRandom` (RngIntElt, default 100), `Verify` (BoolElt, default true). | Monte Carlo algorithm of Liebeck–O'Brien **[LO07]**. |
| `RecogniseSL3(G)` / `RecogniseSL3(G, q : parameters)` | If G ≤ GL(d, F) is isomorphic modulo scalars to (P)SL(3, q), construct homomorphisms (same four returns as `RecognizeSL2`). If q is not supplied it is computed via `LieCharacteristic` and `LieType`. Parameter: `Verify` (BoolElt, default true; if false, assumes G ≅ (P)SL(3, q)). | Lübeck–Magaard–O'Brien **[LMO07]**; current implementation (part of CompositionTree) by Bäärnhielm and O'Brien. |
| `SL3ElementToWord(G, g)` | If G was constructively recognised as (P)SL(3, q), return true and a word group element evaluating to g; else false. | Uses recognition maps from `RecogniseSL3`. |
| `RecogniseSL(G, d, q)` / `RecognizeSL(G, d, q)` | Try to find an isomorphism between black-box group G and SL(d, q) or PSL(d, q). Returns success flag, and if successful, mutually inverse homomorphisms. Warning: often returns false even for a correct G — call repeatedly until true. | Kantor–Seress algorithm (Las Vegas). |

*Worked examples: H65E9 (SL(2, 9) in natural representation); H65E10 (SL(2, 5^7) representation inside GL(6, F_{5^7})); H65E11 (SL(3, 5^4) and its symmetric square).*

### 65.3.6 Constructive Recognition of Symplectic Groups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RecogniseSpOdd(G, d, q)` / `RecognizeSpOdd(G, d, q)` | Try to find an isomorphism between black-box group G and Sp(d, q) or PSp(d, q) for odd q. Returns success flag, and if successful, mutually inverse homomorphisms (modulo scalars for PSp). Warning: often returns false — call repeatedly. | Kantor–Seress algorithm (Las Vegas). |
| `RecogniseSp4Even(G, q)` / `RecognizeSp4Even(G, q)` | Try to find an isomorphism between G and Sp(4, q) for even q. Returns success flag, and if successful, mutually inverse homomorphisms G ↔ Sp(4, q) and G ↔ word group. | Algorithm of Peter Brooksbank (Las Vegas). |

### 65.3.7 Constructive Recognition of Unitary Groups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RecogniseSU3(G, d, q)` / `RecognizeSU3(G, d, q)` | Try to find an isomorphism between black-box group G and SU(3, q) or PSU(3, q) for q > 2. Returns success flag, and if successful, mutually inverse homomorphisms (modulo scalars for PSU) and G ↔ word group maps. | Algorithm of Peter Brooksbank (Las Vegas). |
| `RecogniseSU4(G, d, q)` / `RecognizeSU4(G, d, q)` | Try to find an isomorphism between black-box group G and SU(4, q) or PSU(4, q). Returns success flag, and if successful, mutually inverse homomorphisms (modulo scalars for PSU) and G ↔ word group maps. | Algorithm of Peter Brooksbank (Las Vegas). |

### 65.3.8 Constructive Recognition of SL(d, q) in Low Degree

Let SL(d, q) ≤ H ≤ GL(d, q) with q = p^f. These functions handle the case where H acts on an irreducible module W of dimension at most d², and reconstruct a d-dimensional projective representation. Algorithms by Magaard–O'Brien–Seress **[MOAS08]**; implementations by Eamonn O'Brien.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RecogniseSymmetricSquare(G)` | G is the symmetric-square representation of H where SL(d, q) ≤ H ≤ GL(d, q), d ≥ 4. Reconstruct H; return true and H if successful, else false. | Symmetric-square reconstruction; Magaard–O'Brien–Seress **[MOAS08]**. |
| `SymmetricSquarePreimage(G, g)` | G is the symmetric-square representation of H (SL(d, q) ≤ H ≤ GL(d, q)). Return the preimage of g in H. | Inverse of symmetric-square map. |
| `RecogniseAlternatingSquare(G)` | G is the alternating-square representation of H where SL(d, q) ≤ H ≤ GL(d, q), d ≥ 3. Reconstruct H; return true and H if successful, else false. | Alternating-square reconstruction; Magaard–O'Brien–Seress **[MOAS08]**. |
| `AlternatingSquarePreimage(G, g)` | G is the alternating-square representation of H. Return the preimage of g in H. | Inverse of alternating-square map. |
| `RecogniseAdjoint(G)` | G is the adjoint representation of H where SL(d, q) ≤ H ≤ GL(d, q), d ≥ 3. Reconstruct H; return true and H if successful, else false. | Adjoint representation reconstruction; Magaard–O'Brien–Seress **[MOAS08]**. |
| `AdjointPreimage(G, g)` | G is the adjoint representation of H. Return the preimage of g in H. | Inverse of adjoint map. |
| `RecogniseDelta(G)` | G is an absolutely irreducible representation of H ⊗ H^{(p^e)}, where SL(d, q) ≤ H ≤ GL(d, q), d ≥ 4. Reconstruct H; return true and H if successful, else false. | Tensor product (delta) reconstruction; Magaard–O'Brien–Seress **[MOAS08]**. |
| `DeltaPreimage(G, g)` | G is the absolutely irreducible representation of H ⊗ H^{(p^e)}. Return the preimage of g in H. | Inverse of delta map. |

*Worked example: H65E12 (SL(4, 9) in symmetric-square representation, RecogniseSymmetricSquare and SymmetricSquarePreimage).*

### 65.3.9 Constructive Recognition of Suzuki Groups

Provides constructive recognition and membership testing for Sz(q), q = 2^{2m+1}, m > 0. Verbose flags: `SuzukiGeneral`, `SuzukiStandard`, `SuzukiConjugate`, `SuzukiTensor`, `SuzukiMembership`, `SuzukiCrossChar`, `SuzukiTrick`, `SuzukiNewTrick` (levels up to 10).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsSuzukiGroup(G)` | Non-constructive: determine if G is isomorphic to Sz(q); also return q. For degree 4 over char 2: fast Las Vegas algorithm. For other representations: uses `LieType`. | Las Vegas algorithm (degree 4, char 2) from **[Bää06a]**; otherwise Monte Carlo `LieType` **[BKPS02]**. |
| `RecogniseSz(G : parameters)` / `RecognizeSz(G : parameters)` | Constructively recognise absolutely irreducible G (defined over minimal field) as Sz(q). If G ≅ Sz(q) returns: isomorphism G → Sz(q), inverse isomorphism, map G → word group, map word group → G. Use `Function` on each component to avoid built-in membership testing. Handles 2.Sz(8). Parameters: `Verify` (BoolElt, default true; checks via `IsSuzukiGroup`, requires `FieldSize`), `FieldSize` (RngIntElt), `Optimise` (BoolElt, default false; builds optimised word group via `AddRedundantGenerators`). | Constructive recognition algorithms **[Bää06a, Bää05]**. |
| `SzElementToWord(G, g)` | If G was constructively recognised as Sz(q) and g ∈ G, return true and a GrpSLPElt from the word group evaluating to g; else false. Facilitates membership testing. | Uses constructive recognition maps. |
| `SzPresentation(q)` | For q = 2^{2m+1}, return a short presentation of Sz(q) on the standard generators (as returned by `Sz`). | Standard presentation of Suzuki groups. |
| `SatisfiesSzPresentation(G)` | G is constructively recognised as Sz(q). Verify that it satisfies a presentation for Sz(q). | Checks standard presentation. |
| `SuzukiIrreducibleRepresentation(F, twists : parameters)` | F a field of size q = 2^{2m+1}; `twists` a sequence of n distinct integers in [0 … 2m]. Return an absolutely irreducible 4n-dimensional representation of Sz(q), a tensor product of twisted powers of the standard copy. Parameter: `CheckInput` (BoolElt, default true). | Tensor product of Frobenius twists of the natural representation. |

*Worked examples: H65E13 (Sz(32) — non-constructive then constructive recognition, membership testing, SLP coercion); H65E14 (2.Sz(8) via ATLASGroup); H65E15 (Sz in 64-dimensional twisted representation); H65E16 (Sz(8) in cross-characteristic via GF(9)).*

### 65.3.10 Constructive Recognition of Small Ree Groups

Provides constructive recognition and membership testing for ²G2(q) = Ree(q), q = 3^{2m+1}, m > 0. Verbose flags: `ReeGeneral`, `ReeStandard`, `ReeConjugate`, `ReeTensor`, `ReeMembership`, `ReeCrossChar`, `ReeTrick`, `ReeInvolution`, `ReeSymSquare` (levels up to 10).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RecogniseRee(G : parameters)` / `RecognizeRee(G : parameters)` | Constructively recognise absolutely irreducible G (defined over minimal field) as Ree(q). If G ≅ Ree(q) returns: isomorphism G → Ree(q), inverse, map G → word group, map word group → G. Parameters: `Verify` (BoolElt, default true; checks via `IsReeGroup`, requires `FieldSize`), `FieldSize` (RngIntElt), `Optimise` (BoolElt, default false). | Constructive recognition algorithms of Bäärnhielm **[Bää06b]**. |
| `ReeElementToWord(G, g)` | If G was constructively recognised as Ree(q) and g ∈ G, return true and a GrpSLPElt evaluating to g; else false. | Uses recognition maps from `RecogniseRee`. |
| `IsReeGroup(G)` | Non-constructive: determine if G ≅ Ree(q); also return q. For degree 7 over char 3: fast Las Vegas algorithm. Otherwise: `LieType`. | Las Vegas (degree 7, char 3); otherwise Monte Carlo `LieType`. |
| `ReeIrreducibleRepresentation(F, twists : parameters)` | F a field of size q = 3^{2m+1}; `twists` a sequence of n distinct pairs (i, j) where i ∈ {7, 27} and j ∈ [0 … 2m]. Return an absolutely irreducible representation of Ree(q) as a tensor product of twisted powers of the 7- or 27-dimensional standard representation. Parameter: `CheckInput` (BoolElt, default true). | Tensor product of Frobenius twists. |

*Worked example: H65E17 (Ree group over F_27 — recognition, membership testing, ReeElementToWord, element outside group).*

### 65.3.11 Constructive Recognition of Large Ree Groups

Provides constructive recognition and membership testing for ²F4(q) = LargeRee(q), q = 2^{2m+1}, m > 0. Verbose flags: `LargeReeGeneral`, `LargeReeStandard`, `LargeReeConjugate`, `LargeReeRyba`, `LargeReeTrick`, `LargeReeInvolution` (levels up to 10).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RecogniseLargeRee(G : parameters)` / `RecognizeLargeRee(G : parameters)` | Constructively recognise absolutely irreducible G (defined over minimal field) as LargeRee(q). If G ≅ LargeRee(q) returns: isomorphism G → LargeRee(q), inverse, map G → word group, map word group → G. Parameters: `Verify` (BoolElt, default true; checks via `IsLargeRee`, requires `FieldSize`), `FieldSize` (RngIntElt), `Optimise` (BoolElt, default false). | — (no explicit bibliography key given in the chapter text). |
| `LargeReeElementToWord(G, g)` | If G was constructively recognised as LargeRee(q) and g ∈ G, return true and a GrpSLPElt evaluating to g; else false. | Uses recognition maps from `RecogniseLargeRee`. |
| `IsLargeReeGroup(G)` | Non-constructive: determine if G ≅ LargeRee(q); also return q. For degree 26 over char 2: fast Las Vegas algorithm. Otherwise: `LieType`. | Las Vegas (degree 26, char 2); otherwise Monte Carlo `LieType`. |

---

## 65.4 Properties of Finite Groups of Lie Type

### 65.4.1 Maximal Subgroups of the Classical Groups

Written by Derek Holt and Colva Roney-Dougal. Returns maximal subgroups of the classical quasisimple groups (SL, Sp, SU, Omega, OmegaPlus, OmegaMinus) in natural representations. Complete for dimensions up to 12 (with some omissions in Ω⁺(8, q)). Subgroups lie in nine Aschbacher categories **[Asc84]**; the first eight (geometric type) are described in **[KL90]** and returned in all dimensions; the ninth (non-geometric, dimension ≤ 12) uses **[HM01, HM02, Lü01]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ClassicalMaximals(type, d, q : parameters)` | Return conjugacy class representatives of maximal subgroups of the quasisimple classical group of the given type ("L", "S", "U", "O", "O+", "O-") and dimension d over GF(q). Parameters: `classes` (SetEnum, default {1..9}; restrict to Aschbacher categories), `all` (BoolElt, default true; if false, use full automorphism group classes → fewer subgroups), `special` (BoolElt, default true; for O/O+/O-: include normalisers in SO/SO+/SO-), `general` (BoolElt, default true; include normalisers in GL/GU/GO), `normaliser` (BoolElt, default true; include normalisers in the full normaliser in GL), `novelties` (BoolElt, default false; return intersections with novelty maximal subgroups — caution: unreliable). | Aschbacher's theorem **[Asc84]**, geometric types from **[KL90]**, non-geometric types from **[HM01, HM02, Lü01]**. |

### 65.4.2 Maximal Subgroups of the Exceptional Groups

Verbose flags: `SuzukiMaximals`, `ReeMaximals`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SuzukiMaximalSubgroups(G)` | G constructively recognised as a Suzuki group. Return a sequence of maximal subgroup representatives and sequences of GrpSLPElt generators from the word group. | Explicit construction using recognition isomorphism. |
| `SuzukiMaximalSubgroupsConjugacy(G, R, S)` | G constructively recognised as Suzuki group; R and S conjugate maximal subgroups of G. Return a conjugating element g and the corresponding GrpSLPElt. | Uses constructive membership in Sz. |
| `ReeMaximalSubgroups(G)` | G constructively recognised as a Ree group. Return maximal subgroup representatives and GrpSLPElt generators. | Explicit construction using recognition isomorphism. |
| `ReeMaximalSubgroupsConjugacy(G, R, S)` | G constructively recognised as Ree group; R and S conjugate maximal subgroups. Return conjugating element and GrpSLPElt. Not implemented if R, S are Frobenius groups. | Uses constructive membership in Ree. |

### 65.4.3 Sylow Subgroups of the Classical Groups

Written by Mark Stather. Constructs and conjugates Sylow p-subgroups for any prime p, using descriptions in **[Wei55, CF64, R.R57, Car72]** and Stather's constructive Sylow algorithms **[Sta]**. Uses Derek Holt's classical form code and Colva Roney-Dougal's form conjugation code. Conjugation uses only the Meataxe, Smash, linear algebra, and norm equations. Applicable to conjugates of GL, SL, Sp, GO, GOPlus, GOMinus, SO, SOPlus, SOMinus, Omega, OmegaPlus, OmegaMinus, GU, SU (with exception GO(2m+1, 2^e) for some intrinsics).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ClassicalSylow(G, p)` | G a classical group in natural representation (up to conjugation; not GO(2m+1, 2^e)); p a prime. Return a Sylow p-subgroup of G as a matrix group. | Constructive Sylow theorem for classical groups **[Sta]**, using descriptions in **[Wei55, CF64, R.R57, Car72]**. |
| `ClassicalSylowConjugation(G, P, S)` | G as above; P and S Sylow p-subgroups of G. Return g ∈ G with P^g = S. | Conjugation algorithm **[Sta]** using Meataxe, Smash, linear algebra, norm equations. |
| `ClassicalSylowNormaliser(G, P)` | G must be the full classical group (up to conjugation; not GO(2m+1, 2^e)); a conjugate of GL, Sp, GO, GOPlus, GOMinus, or GU. P a Sylow p-subgroup. Return the normaliser of P in G. | **[Sta]**. |
| `ClassicalSylowToPC(G, P)` | G as for `ClassicalSylow`; P a Sylow p-subgroup. Return a PC group Q ≅ P, an isomorphism P → Q, and an isomorphism Q → P. | **[Sta]**; PC group presentation of p-group. |

*Worked example: H65E18 (Sylow 7-subgroup of Sp(28, 17²), conjugation, normaliser, PC presentation).*

### 65.4.4 Sylow Subgroups of Exceptional Groups

Verbose flags: `SuzukiSylow`, `ReeSylow`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SuzukiSylow(G, p)` | G constructively recognised as Sz(q); p a prime. Return a random Sylow p-subgroup S and a list of GrpSLPElt generators from the word group. If p ∤ |G|, return the trivial subgroup. | Uses constructive recognition of Sz. |
| `SuzukiSylowConjugacy(G, R, S, p)` | G constructively recognised as Sz(q); R, S Sylow p-subgroups. Return a conjugating element g and corresponding GrpSLPElt. | Uses constructive membership in Sz. |
| `ReeSylow(G, p)` | G constructively recognised as Ree(q); p a prime. Return a random Sylow p-subgroup and GrpSLPElt generators. | Uses constructive recognition of Ree. |
| `ReeSylowConjugacy(G, R, S, p)` | G constructively recognised as Ree(q); R, S Sylow p-subgroups. Return conjugating element and GrpSLPElt. Not implemented for odd p dividing q³ + 1. | Uses constructive membership in Ree. |
| `LargeReeSylow(G, p)` | G constructively recognised as LargeRee(q); p a prime. Return a random Sylow p-subgroup and GrpSLPElt generators. Not implemented for p dividing q + 1. | Uses constructive recognition of LargeRee. |

*Worked examples: H65E19 (Sz(2^{121}) — Sylow p-subgroups and conjugation, including the Sylow 2-subgroup); H65E20 (Ree group over F_{3^{15}} — Sylow p-subgroups and conjugation).*

### 65.4.5 Conjugacy of Subgroups of the Classical Groups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsGLConjugate(H, K)` | H and K subgroups of the same GL(n, q). Return true if conjugate in GL(n, q), plus a conjugating element z; else false. | Algorithm of Roney-Dougal **[RD04]**. |

### 65.4.6 Conjugacy of Elements of the Exceptional Groups

Verbose flags: `SuzukiElements`, `ReeElements`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SzConjugacyClasses(G)` | G constructively recognised as Sz(q). Return conjugacy classes in `ConjugacyClasses` format. | Uses constructive isomorphism to standard Sz. |
| `SzClassRepresentative(G, g)` | G constructively recognised as Sz(q); g ∈ G. Return the class representative h of g (from `SzConjugacyClasses`) and a conjugating element c with g^c = h. | Uses class structure of standard Sz. |
| `SzIsConjugate(G, g, h)` | G constructively recognised as Sz(q); g, h ∈ G. Determine if g is conjugate to h; if so return true and c with g^c = h, else false. | Uses class structure of standard Sz. |
| `SzClassMap(G)` | G constructively recognised as Sz(q). Return its class map (as in the `ClassMap` intrinsic). | Uses class structure of standard Sz. |
| `ReeConjugacyClasses(G)` | G constructively recognised as Ree(q). Return conjugacy classes in `ConjugacyClasses` format. | Uses constructive isomorphism to standard Ree. |

### 65.4.7 Irreducible Subgroups of the General Linear Group

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IrreducibleSubgroups(n, q)` | Return conjugacy classes of irreducible subgroups of GL(n, q). Currently n restricted to 2; list complete for characteristic ≥ 5. | Classification of Flannery–O'Brien **[FO05]**. |
| `IrreducibleSolubleSubgroups(n, q)` | Return conjugacy classes of soluble irreducible subgroups of GL(n, q). Currently n restricted to 2 or 3; complete for characteristic ≥ 5. | Classification of Flannery–O'Brien **[FO05]**. |

*Worked example: H65E21 (IrreducibleSubgroups(2, 19^5) and IrreducibleSolubleSubgroups(2, 97²)).*

---

## 65.5 Atlas Data for the Sporadic Groups

Data derived from the Web Atlas, prepared for Magma by Michael Downward and Eamonn O'Brien (maintaining Atlas names, conventions and orderings). Functions accept matrix or permutation groups. The `GoodBasePoints` algorithm is due to O'Brien–Wilson **[OW05]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `StandardGenerators(G, str : parameters)` | Find standard generators for small quasisimple or sporadic group G with Atlas name `str`; return true plus generator sequence and corresponding SLP word sequence, else false. Works for all sporadic simple groups and quasisimple groups whose simple quotient has order ≤ 2 × 10^8. Parameters: `Projective` (BoolElt, default false; construct generators modulo centre), `AutomorphismGroup` (BoolElt, default false; treat G as automorphism group of group named `str`). | Random search for standard generators (Monte Carlo); if false, retry. |
| `IsomorphismToStandardCopy(G, str : parameters)` | Use `StandardGenerators` to construct a (possibly projective) isomorphism from G to a standard copy. First return value indicates success. Parameters as for `StandardGenerators`. | Calls `StandardGenerators`; Web Atlas data. |
| `StandardPresentation(G, str : parameters)` | Return true if the standard presentation is satisfied by generators of sporadic group G with name `str`, else false. Parameters: `Projective` (BoolElt, default false), `Generators` (SeqEnum, default []), `AutomorphismGroup` (BoolElt, default false). | Verifies presentation relations; Web Atlas data. |
| `MaximalSubgroups(G, str : parameters)` | Construct some maximal subgroups of sporadic G with name `str`. Return true and list of subgroups if standard generators found or supplied, else false. Parameters: `Projective` (BoolElt, default false), `Generators` (SeqEnum, default []), `AutomorphismGroup` (BoolElt, default false). | Web Atlas data via standard generators. |
| `Subgroups(G, str : parameters)` | Construct certain subgroups of sporadic G with name `str`. Return true and list if standard generators found or supplied, else false. Parameters: `Projective` (BoolElt, default false), `Generators` (SeqEnum, default []). | Web Atlas data via standard generators. |
| `GoodBasePoints(G, str : parameters)` | If standard generators found or supplied for sporadic G with name `str`, return true and a list of base points for G, else false. Parameters: `Projective` (BoolElt, default false), `Generators` (SeqEnum, default []). | O'Brien–Wilson algorithm **[OW05]**. |
| `SubgroupsData(str)` | Display stored subgroup data for sporadic group with name `str`. | Web Atlas data display. |
| `MaximalSubgroupsData(str : parameters)` | Display stored data for maximal subgroups of sporadic group with name `str`. Parameter: `AutomorphismGroup` (BoolElt, default false; display maximal subgroups of the automorphism group). | Web Atlas data display. |

*Worked example: H65E22 (J1 in 7-dimensional representation over GF(11) — StandardGenerators, StandardPresentation, MaximalSubgroups; resulting 7 maximal subgroups, M[4] is 19:6 of order 114 and index 1540).*

---

## 65.6 Bibliography

| Key | Reference |
|-----|-----------|
| **[Asc84]** | M. Aschbacher. On the maximal subgroups of the finite classical groups. *Invent. Math*, 76:469–514, 1984. |
| **[Bää05]** | H. Bäärnhielm. Tensor decomposition of the Suzuki groups. submitted, 2005. |
| **[Bää06a]** | H. Bäärnhielm. Recognising the Suzuki groups in their natural representations. *J. Algebra*, 300(1):171–198, 2006. |
| **[Bää06b]** | Henrik Bäärnhielm. Constructive recognition of the Ree groups. preprint, 2006. |
| **[BKPS02]** | L. Babai, W. M. Kantor, P. P. Pálfy, and Á. Seress. Black-box recognition of finite simple groups of Lie type by statistics of element orders. *J. Group Theory*, 5:383–401, 2002. |
| **[BLGN+03]** | R. Beals, C. R. Leedham-Green, A. C. Niemeyer, C. E. Praeger, and A. Seress. A black-box algorithm for recognising finite symmetric and alternating groups, I. *Trans. Amer. Math. Soc.*, 2003. To appear. |
| **[BP00]** | Sergey Bratus and Igor Pak. Fast constructive recognition of a black box group isomorphic to S_n or A_n using Goldbach's conjecture. *J. Symbolic Comp.*, 29:33–57, 2000. |
| **[Car72]** | R. Carter. *Simple Groups of Lie Type*. John Wiley & Sons, London, New York, Sydney, Toronto, 1972. |
| **[CF64]** | R. Carter and P. Fong. The Sylow 2-subgroups of the finite classical groups. *Journal of Algebra*, 1:139–151, 1964. |
| **[CLG97a]** | Frank Celler and Charles R. Leedham-Green. Calculating the Order of an Invertible Matrix. In Larry Finkelstein and William M. Kantor, editors, *Groups and Computation II*, volume 28 of DIMACS Series in Discrete Mathematics and Theoretical Computer Science, pages 55–60. AMS, 1997. |
| **[CLG97b]** | Frank Celler and C.R. Leedham-Green. A non-constructive recognition algorithm for the special linear and other classical groups. In *Groups and computation II (New Brunswick, NJ, 1995)*, volume 28 of DIMACS Ser. Discrete Math. Theoret. Comput. Sci., pages 61–67. Amer. Math. Soc., 1997. |
| **[CLGM+95]** | Frank Celler, Charles R. Leedham-Green, Scott H. Murray, Alice C. Niemeyer, and E. A. O'Brien. Generating random elements of a finite group. *Comm. Algebra*, 23(13):4931–4948, 1995. |
| **[CLGO06]** | M.D.E. Conder, C.R. Leedham-Green, and E.A. O'Brien. Constructive recognition for PSL(2, q). *Trans. Amer. Math. Soc.*, 358:1203–1221, 2006. |
| **[FO05]** | D.L. Flannery and E.A. O'Brien. Linear groups of small degree over finite fields. *Internat. J. Algebra and Comput.*, 15:467–502, 2005. |
| **[HM01]** | G. Hiß and G. Malle. Low-dimensional representations of quasi-simple groups. *LMS J. Comput. Math.*, 4:22–63, 2001. |
| **[HM02]** | G. Hiß and G. Malle. Corrigenda: Low-dimensional representations of quasi-simple groups. *LMS J. Comput. Math.*, 5:95–126, 2002. |
| **[HRT01]** | R. B. Howlett, L. J. Rylands, and D. E. Taylor. Matrix generators for exceptional groups of Lie type. *J. Symbolic Comput.*, 31(4):429–445, 2001. |
| **[KL90]** | Peter Kleidman and Martin Liebeck. *The Subgroup Structure of the Finite Classical Groups*, volume 129 of London Math. Soc. Lecture Note Ser. CUP, Cambridge, 1990. |
| **[Lü01]** | F. Lübeck. Small degree representations of finite Chevalley groups in defining characteristic. *LMS J. Comput. Math.*, 4:135–169, 2001. |
| **[LMO07]** | F. Lübeck, K. Magaard, and E.A. O'Brien. Constructive recognition of SL3(q). *J. Algebra*, 2007:617–633, 2007. |
| **[LO07]** | Martin Liebeck and E.A. O'Brien. Finding the characteristic of a group of Lie type. *J. London Math. Soc.*, 2007. |
| **[MOAS08]** | Kay Magaard, E. A. O'Brien, and Ákos Seress. Recognition of small dimensional representations of general linear groups. *J. Austral. Math. Soc.*, 85:229–250, 2008. |
| **[NP92]** | Peter M. Neumann and Cheryl E. Praeger. A Recognition Algorithm for Classical Groups. *Proc. London Math. Soc.*, 65(3):555–603, 1992. |
| **[NP97]** | Alice C. Niemeyer and Cheryl E. Praeger. Implementing a Recognition Algorithm for Classical Groups. In *Groups and computation II (New Brunswick, NJ, 1995)*, volume 28 of DIMACS Ser. Discrete Math. Theoret. Comput. Sci., pages 273–296. Amer. Math. Soc., 1997. |
| **[NP98]** | Alice C. Niemeyer and Cheryl E. Praeger. A Recognition Algorithm for Classical Groups over Finite Fields. *Proc. London Math. Soc.*, 77(3):117–169, 1998. |
| **[NP99]** | Alice C. Niemeyer and Cheryl E. Praeger. A Recognition Algorithm for Non-Generic Classical Groups over Finite Fields. *J. Austral. Math. Soc. Ser. A*, 67:223–253, 1999. |
| **[OW05]** | E.A. O'Brien and R.A. Wilson. Subgroup chains in matrix groups. preprint, 2005. |
| **[Pra99]** | Cheryl E. Praeger. Primitive prime divisor elements in finite classical groups. In *Proc. of Groups St. Andrews 1997 in Bath II*, number 261 in London Math. Soc. Lecture Notes Series, pages 605–623. Cambridge Univ. Press, 1999. |
| **[RD04]** | Colva M. Roney-Dougal. Conjugacy of subgroups of the general linear group. *Experiment. Math.*, 13:151–163, 2004. |
| **[R.R57]** | R. Ree. On some simple groups defined by Chevalley. *Trans. Am. Math. Soc.*, 84:392–400, 1957. |
| **[RT98]** | L.J. Rylands and D.E. Taylor. Matrix generators for the orthogonal groups. *J. Symbolic Comp.*, 25:351–360, 1998. |
| **[Sta]** | M. Stather. Constructive Sylow Theorems for the Classical Groups. to appear in *Journal of Algebra*. |
| **[Tay87]** | Don Taylor. Pairs of Generators for Matrix Groups. I. *Cayley Bulletin 3*, 1987. |
| **[Wei55]** | A. Weir. Sylow p-subgroups of the classical groups over finite fields with characteristic prime to p. *Proc. Am. Math. Soc*, 6:529–533, 1955. |

---

## Algorithm-to-function quick reference

| Algorithm / method | Functions |
|--------------------|-----------|
| Matrix generators from Taylor **[Tay87]** (series A, C, ²A, ²B) | `ChevalleyGroup`, `GL`/`SL`, `CU`/`GU`/`SU`, `CSp`/`Sp`, `SuzukiGroup` |
| Matrix generators from Rylands–Taylor **[RT98]** (series B, D, ²D, orthogonal) | `ChevalleyGroup`, `CO`/`GO`/`SO`, `COPlus`/`GOPlus`/`SOPlus`, `COMinus`/`GOMinus`/`SOMinus`, `Omega`/`OmegaPlus`/`OmegaMinus`, `Spin`/`SpinPlus`/`SpinMinus` |
| Matrix generators from Howlett–Rylands–Taylor **[HRT01]** (exceptional) | `ChevalleyGroup` (E, F, G, ³D, ²E, ²F, ²G), `ReeGroup`, `LargeReeGroup` |
| Beals–Leedham-Green–Niemeyer–Praeger–Seress black-box recognition **[BLGN+03]** | `RecogniseAlternatingOrSymmetric` |
| Bratus–Pak fast constructive recognition **[BP00]** | `RecogniseSymmetric`, `RecogniseAlternating` |
| Liebeck–O'Brien characteristic-finding algorithm **[LO07]** | `LieCharacteristic`, `SL2Characteristic` |
| Babai–Kantor–Pálfy–Seress Lie type by element orders **[BKPS02]** | `LieType`, `SimpleGroupName`, `IsSuzukiGroup` (fallback), `IsReeGroup` (fallback) |
| Niemeyer–Praeger classical recognition **[NP92, NP97, NP98, NP99, Pra99]** | `RecognizeClassical`, `IsLinearGroup`, `IsSymplecticGroup`, `IsOrthogonalGroup`, `IsUnitaryGroup` |
| Celler–Leedham-Green order/recognition algorithms **[CLG97a, CLG97b, CLGM+95]** | `RecognizeClassical` (internals) |
| Conder–Leedham-Green–O'Brien constructive PSL(2,q) **[CLGO06]** | `RecognizeSL2` (natural representation) |
| Lübeck–Magaard–O'Brien constructive SL(3,q) **[LMO07]** | `RecogniseSL3` |
| Kantor–Seress black-box SL(d,q) / Sp(d,q) | `RecogniseSL`, `RecogniseSpOdd` |
| Brooksbank constructive Sp(4,q) and SU(3,q)/SU(4,q) | `RecogniseSp4Even`, `RecogniseSU3`, `RecogniseSU4` |
| Magaard–O'Brien–Seress low-degree recognition **[MOAS08]** | `RecogniseSymmetricSquare`, `RecogniseAlternatingSquare`, `RecogniseAdjoint`, `RecogniseDelta` (and their Preimage variants) |
| Bäärnhielm constructive Suzuki recognition **[Bää06a, Bää05]** | `RecogniseSz`, `IsSuzukiGroup` (natural rep) |
| Bäärnhielm constructive Ree recognition **[Bää06b]** | `RecogniseRee`, `IsReeGroup` (natural rep) |
| Aschbacher's theorem + Kleidman–Liebeck **[Asc84, KL90]** | `ClassicalMaximals` |
| Hiß–Malle small representations **[HM01, HM02]** + Lübeck **[Lü01]** | `ClassicalMaximals` (Aschbacher class 9, dim ≤ 12) |
| Stather constructive Sylow theorems for classical groups **[Sta, Wei55, CF64, R.R57, Car72]** | `ClassicalSylow`, `ClassicalSylowConjugation`, `ClassicalSylowNormaliser`, `ClassicalSylowToPC` |
| Roney-Dougal GL-conjugacy **[RD04]** | `IsGLConjugate` |
| Flannery–O'Brien irreducible subgroup classification **[FO05]** | `IrreducibleSubgroups`, `IrreducibleSolubleSubgroups` |
| O'Brien–Wilson base points **[OW05]** | `GoodBasePoints` |
| Web Atlas data (Downward–O'Brien) | `StandardGenerators`, `IsomorphismToStandardCopy`, `StandardPresentation`, `MaximalSubgroups`, `Subgroups`, `SubgroupsData`, `MaximalSubgroupsData` |
