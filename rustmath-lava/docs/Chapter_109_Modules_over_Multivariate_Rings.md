# Chapter 109 — Modules over Multivariate Rings

**Handbook part:** XIV — Commutative Algebra
**Handbook pages:** 3303–3351 (PDF pages 3434–3485)

---

## Scope and overview

Chapter 109 describes R-modules over multivariate polynomial rings and related rings, where R may be:
- a **multivariate polynomial ring** (Chapters 24 and 105), whose coefficient ring may be a field or Euclidean ring;
- a **local polynomial ring** (Chapter 107, new in V2.15), whose coefficient ring must be a field;
- an **affine algebra** (Chapter 108), whose coefficient ring must be a field; or
- an **exterior algebra** (Chapter 82, new in V2.15), whose coefficient ring must be a field (technically a skew-commutative ring, treated here as a left R-module setting).

The fundamental computational tool is the construction of **Gröbner bases for modules**, since these rings are not principal ideal rings in general and standard matrix echelonisation algorithms do not apply. Module elements have type `ModMPol`, or `ModMPolGrd` for graded modules (a subtype of `ModMPol`). Homomorphisms between modules have type `ModMPolHom`.

The chapter draws a central distinction between two module paradigms:

- **Embedded modules** (created via `EModule`): explicit Gröbner-basis computations are exposed to the user; submodules and quotient modules remain embedded in the ambient. Preferred for low-level Gröbner basis work.
- **Reduced modules** (created via `RModule` / `GradedModule`): submodules and quotients are returned as new abstract ambient presentations connected by stored morphisms. Preferred for homological computations.

Key algorithmic topics include module Gröbner bases, syzygy modules, free resolutions (La Scala **[SS98]** / Faugère F4 **[Fau99]** extended algorithm; iterative algorithm), Betti numbers and Castelnuovo-Mumford regularity (**[Eis95, DL06]**), Hilbert series (**[BS92]**), Fitting ideals (**[CLO98, Eis95]**), colon and intersection modules (**[GP02]**), Hom/Ext, tensor products/Tor, and cohomology of coherent sheaves via the BGG correspondence (**[EFS03, DE02]**).

---

## 109.1 Introduction

This section introduces the scope of the chapter, the allowed base rings (multivariate polynomial rings, local polynomial rings, affine algebras, exterior algebras), the `ModMPol` / `ModMPolGrd` types, and the dependence on Gröbner basis theory.

*(No intrinsics introduced here; see §109.2 onwards.)*

---

## 109.2 Module Basics: Embedded and Reduced Modules

This section defines the two fundamental module paradigms. An **ambient** module has presentational form R^k/⟨relations⟩; a **non-ambient** module is a proper submodule of such an ambient (possible only in the embedded setting). Embedded modules mimic polynomial ideal arithmetic and expose Gröbner bases directly. Reduced modules are always ambient and use stored background morphisms to track sub/quotient relationships. The **graded** subtype `ModMPolGrd` enforces homogeneous generators throughout derived sub- and quotient modules.

*(No intrinsics introduced here; concepts are defined narratively.)*

---

## 109.3 Monomial Orders

Module monomial orders extend the underlying ring order to monomial-column pairs s[c] (a monomial s paired with a column number c). Six orders are available; they affect Gröbner basis computation difficulty and which elimination properties hold.

### 109.3.1 Term Over Position: TOP

s₁[c₁] < s₂[c₂] iff s₁ <_R s₂, or s₁ = s₂ and c₂ > c₁. Specified by argument `"top"`. Generally easiest to compute; analogous to grevlex for polynomial rings. See **[AL94, §3.5]**, **[CLO98, Def. 2.4]**.

### 109.3.2 Term Over Position (Weighted): TOPW

Given integer weight sequence W of length k: weighted degree d_i = Degree(s_i) + W[c_i]; compare first by weighted degree, then as TOP. Specified by arguments `"topw", W`. Preferred when the module has a natural column grading and elements of interest are homogeneous with respect to W.

### 109.3.3 Position Over Term: POT

s₁[c₁] < s₂[c₂] iff c₂ > c₁, or c₁ = c₂ and s₁ <_R s₂. Specified by argument `"pot"`. Gives an echelon-form-like Gröbner basis; harder to compute than TOP in general.

### 109.3.4 Position Over Term (Permutation): POTPERM

Given a permutation sequence P of [1..k]: compare columns via P, then monomials. Specified by arguments `"potperm", P`.

### 109.3.5 Block TOP-TOP: TOPTOP

Given integer k: the first k columns form block 1, the rest block 2; block comparison first, then TOP within each block. Specified by arguments `"toptop", k`. Useful for eliminating only the first k columns; easier to compute than POT.

### 109.3.6 Block TOP-POT: TOPPOT

Given integer k: block comparison first, then TOP in block 1 and POT in block 2. Specified by arguments `"toppot", k`. Similar to TOPTOP but the second block is ordered by position.

---

## 109.4 Basic Creation and Access

### 109.4.1 Creation of Ambient Embedded Modules

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `EModule(R, k)` | Create ambient embedded module R^k with default TOP order. | Extends multivariate polynomial ideal type with column tags. |
| `EModule(R, k, order)` | Create ambient embedded module R^k with specified module monomial order (see §109.3). | As above. |
| `EModule(R, W)` | Create ambient embedded module R^k where W is a sequence of k integer column weights; uses TOPW order with weights W. | As above. |
| `EModule(R, W, order)` | Create ambient embedded module R^k with column weights W and specified order. | As above. |

### 109.4.2 Creation of Reduced Modules

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RModule(R, k)` | Create free reduced module R^k with zero column weights. Always ambient; submodules/quotients returned as new ambients with stored morphisms. | — |
| `RModule(R, W)` | Create free reduced module R^k with integer column weights W. | — |
| `GradedModule(R, k)` | Create free graded reduced module R^k (type `ModMPolGrd`) with zero column weights. Submodules/quotients must be generated by homogeneous elements. | — |
| `GradedModule(R, W)` | Create free graded reduced module R^k with column weights W. | — |

### 109.4.3 Localization

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Localization(M)` | Given R-module M with R = K[x₁,…,xₙ] for a field K, return the corresponding S-module M_{⟨x₁,…,xₙ⟩} where S is the localization of R at the maximal ideal. See Chapter 107. | Passes to the local ring; Gröbner bases become local Gröbner bases. |

### 109.4.4 Basic Invariants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Ambient(M)` / `Generic(M)` | Return the ambient module A in which M is embedded. If M is reduced or is itself ambient, returns M. | — |
| `IsAmbient(M)` | Return whether M is ambient. | — |
| `IsEmbedded(M)` | Return whether M is embedded. | — |
| `IsReduced(M)` | Return whether M is reduced. | — |
| `IsRoot(M)` | Return whether M is a root module (not derived from sub/quo of another). | — |
| `CoefficientRing(M)` / `BaseRing(M)` | Return the base ring R over which M is defined. | — |
| `Degree(M)` | Return k such that the ambient of M equals R^k/⟨relations⟩. Equal to rank iff M is free and ambient. | — |
| `ColumnWeights(M)` / `Grading(M)` | Return the sequence of k integers giving the column grading of M. | — |
| `RelationModule(M)` | Return the submodule of the embedded module R^k generated by the defining relations of M. | — |
| `Relations(M)` | Return the defining relations of M as a sorted sequence of elements of R^k. | — |
| `RelationMatrix(M)` | Return the relation matrix of M (rows are the defining relations). | — |
| `Presentation(M)` | Return the presentation module P of M — a reduced module isomorphic to M, identical to M if M is already reduced. Automatic coercion between M and P is enabled. | — |
| `IsGraded(M)` / `IsHomogeneous(M)` | Return whether M is graded w.r.t. its column weights and base ring weights. True iff the Gröbner basis of M and the Gröbner basis of its relation module both consist of homogeneous elements. Always true for type `ModMPolGrd`. | Gröbner basis check. |

### 109.4.5 Creation of Module Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `M ! Q` | Given sequence Q = [a₁,…,aᵣ] coercible into R, construct the corresponding element of M. | — |
| `M ! v` | Given vector v from R-space R^r, construct the corresponding element of M. | — |
| `M ! 0` / `Zero(M)` | Create the zero element of M. | — |
| `UnitVector(M, i)` | Return the i-th unit vector of M (1 in column i, 0 elsewhere), with parent the ambient of M. Not the same as `BasisElement(M, i)`. | — |

### 109.4.6 Element Operations

#### Access

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Eltseq(f)` | Return the sequence [f₁,…,fᵣ] of r elements of R corresponding to element f of an R-module of degree r. | — |
| `Vector(f)` | Return the element of the R-space of degree r corresponding to element f. | — |
| `f[i]` | Return the i-th component of element f as an element of R. | — |

#### Arithmetic

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `f + g`, `f - g`, `-f`, `r * f`, `f * r` | Basic arithmetic on module elements; r lies in R. Elements must be compatible (same ambient). If quotient relations are present the result is reduced to unique normal form; if relations are delayed, representation may be non-unique but predicates remain correct. | — |
| `f div s` | Given scalar ring element s dividing all components of f, return the quotient of f by s. | — |
| `SPolynomial(f, g)` | Given elements f, g of M whose leading module monomials have the same column, return the S-polynomial. Result is reduced modulo quotient relations. | Standard S-polynomial construction for modules. |
| `Normalize(f)` | Return f normalised so that its leading module monomial is normalised. | — |
| `NormalForm(f, S)` | Return the normal form of element f with respect to compatible module S. Unique when R is not local. Useful when S is a non-ambient embedded module. | Module division algorithm / Gröbner normal form. |
| `Coordinates(f, M)` | Given element f ∈ S and compatible module M with f ∈ M, return coordinates of f w.r.t. the basis of M (components in R). | — |

#### Accessing the Underlying Representation

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Coefficients(f)` / `Monomials(f)` / `Terms(f)` | Access the underlying distributed polynomial representation (with columns); see §24.4.4. | — |
| `LeadingCoefficient(f)` / `LeadingMonomial(f)` / `LeadingTerm(f)` | Leading coefficient, monomial, and term with respect to the module monomial order. | — |
| `CoefficientsAndMonomials(f)` | Return coefficients and monomials together. | — |
| `Column(f)` | Given a single-term element f, return the column c of its monomial-column pair s[c]. | — |
| `Degree(f)` / `WeightedDegree(f)` | Return the weighted degree of f: maximum over all monomial-column pairs s[c] of (weighted degree of s in R) + (column weight of c in grading of M). | — |
| `IsHomogeneous(f)` | Return whether all monomial-columns of f have the same weighted degree (grading of M is significant). | — |

#### Predicates

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsZero(f)` | Return whether f is the zero element of M. May be non-trivial when relations of M are non-zero; relations are automatically computed if needed. | Module Gröbner basis computation if needed. |
| `f eq g` | Return whether f and g are equal in M. May be non-trivial (see IsZero). | — |
| `f lt g` | Return whether f < g w.r.t. the module monomial order. Also `le`, `gt`, `ge`. | — |
| `f in M` | Return whether element f of module S lies in compatible module M. | Normal form computation. |

*Worked example: H109E1 (ambient embedded modules over Q[x,y,z]; column weights; degree; leading monomial; reduced modules and graded modules).*

---

## 109.5 The Homomorphism Type

A module homomorphism f : M → N has type `ModMPolHom`. It is represented by a matrix A in either **ambient matrix** form (m × n, where m = Degree(M), n = Degree(N)) or **presentation matrix** form (using the presentation modules). When M and N are reduced these coincide.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Homomorphism(M, N, A)` | Construct homomorphism f : M → N (type `ModMPolHom`) defined by matrix A. Parameter `Presentation` (default `true`): if true, A is a presentation matrix; if false, A is an ambient matrix. | — |
| `Domain(f)` | Return the domain M of f. | — |
| `Codomain(f)` | Return the codomain N of f. | — |
| `PresentationMatrix(f)` / `Matrix(f)` | Return the presentation matrix of f (always well-defined, computed even if f was constructed via an ambient matrix). | — |
| `AmbientMatrix(f)` / `Matrix(f)` | Return the ambient matrix of f. If M and N are reduced, identical to the presentation matrix. May error if M, N are non-reduced and f was constructed via a presentation matrix. | — |
| `f(v)` / `v * f` | Return the image of element v ∈ M under f, as an element of N. | — |
| `f[i]` | Return the element of N corresponding to the i-th row of the ambient matrix of f. | — |
| `Image(f)` | Return the image of f as a submodule of N (reduced iff N is). | Module Gröbner basis. |
| `Kernel(f)` | Return the kernel of f as a submodule of M (reduced iff M is). | Syzygy computation. |
| `Cokernel(f)` | Return the cokernel of f as a quotient module of N (reduced iff N is). | — |
| `IsZero(f)` | Return whether f is the zero map (may be true even if defining matrix is non-zero). | — |
| `IsInjective(f)` | Return whether the kernel of f is the zero module. | — |
| `IsSurjective(f)` | Return whether the image of f equals N. | — |
| `IsBijective(f)` | Return whether f is injective and surjective. | — |
| `IsGraded(f)` / `IsHomogeneous(f)` | For M, N graded: return whether f is homogeneous of some degree d (every pure-degree element v maps to 0 or an element of degree Degree(v) + d). | — |
| `Degree(f)` | Return the degree of f: maximum d such that an element of degree e maps to 0 or degree e + d. | — |

*Worked example: H109E2 (inclusion homomorphism between two graded submodules of a rank-3 free module over Q[x,y]; ambient vs. presentation matrix; homogeneity check).*

---

## 109.6 Submodules and Quotient Modules

### 109.6.1 Creation

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `sub< M \| L >` | Return the submodule of M generated by the elements, sets/sequences of elements, or submodules listed in L. A morphism from the result into M is stored, mapping the i-th generator to the i-th item in L. | Module Gröbner basis computed lazily. |
| `quo< M \| L >` | Return the quotient module of M by the elements/submodules listed in L. A morphism from M onto the result is stored. | Module Gröbner basis of relations. |
| `Morphism(M, N)` | Given modules M, N related by a chain of stored sub/quo morphisms, return the resulting morphism matrix map from M to N. Errors if no such chain exists. | Composition of stored morphisms. |
| `Submodule(I)` | Given ideal I of polynomial ring R, return the submodule of R^1 generated by I. | — |
| `QuotientModule(I)` | Given ideal I of R, return the quotient module R^1/I. | — |
| `GradedModule(I)` | Given homogeneous ideal I of R, return the graded quotient module R^1/I (type `ModMPolGrd`). | — |

### 109.6.2 Module Bases

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Basis(M)` | Return the current basis of M (may or may not yet be a Gröbner basis). | — |
| `BasisElement(M, i)` | Return the i-th element of the current basis of M. Not the same as M.i. | — |
| `BasisMatrix(M)` | Return the basis matrix of M as a k × r matrix over R (k = basis length, r = degree of M). | — |
| `Groebner(M)` | (Procedure.) Explicitly force construction of a Gröbner basis for M. | Module Gröbner basis algorithm (Buchberger for modules). |

*Worked examples: H109E3 (embedded submodules, quotient modules, Gröbner bases, localization over Q[x,y,z]); H109E4 (reduced submodules and quotient modules; morphisms; RelationMatrix; coercion).*

---

## 109.7 Basic Module Constructions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `M + N` | Sum of compatible modules M, N (embedded in the same ambient): submodule generated by M and N. | Module Gröbner basis. |
| `M meet N` | Intersection of compatible modules M, N in the ambient. If the ambient has non-trivial relations, intersection is of inverse images in the free module. | Standard module intersection algorithm **[GP02, §2.8.3]**. |
| `f * M` / `M * f` | Submodule of M generated by {f·v : v ∈ M} resp. {v·f : v ∈ M} for f ∈ R. | — |
| `I * M` / `M * I` | Submodule of M generated by {f·v : f ∈ I, v ∈ M} resp. {v·f} for ideal I of R. | — |
| `M / N` | Quotient module M/(M ∩ N) for compatible M, N. Equivalent to using the `quo` constructor. | — |
| `DirectSum(M, N)` | Direct sum D = M ⊕ N; returns D and two sequences of homomorphisms giving the injections into and projections from D. | — |
| `DirectSum(S)` | Direct sum of a sequence or list S of R-modules; returns D and injection/projection homomorphism sequences. | — |
| `Twist(M, d)` | Given graded module M and integer d, return the Serre twist M(d) (column weights shifted by −d) and the isomorphism f : M → M(d) of degree −d. | — |

---

## 109.8 Predicates

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsZero(M)` | Return whether M is the zero module. | — |
| `M subset N` | Return whether M is a submodule of N (compatible modules). Involves module Gröbner basis and normal form computations. | Gröbner basis + normal form. |
| `M eq N` | Return whether M equals N (compatible modules). Checks equality of appropriate module Gröbner bases. | Gröbner basis comparison. |
| `IsFree(M)` | Return whether M is free (isomorphic to R^k for some k). Checks whether a minimised presentation of M has trivial relations. | Minimal presentation computation. |

---

## 109.9 Module Operations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `MinimalBasis(M)` | Return a minimal basis B of M. If M is graded or R is local, #B is the unique minimal number of generators (the rank). Otherwise #B satisfies only the rule that the i-th element is not in the submodule of elements 1..i−1. | In the graded/local case: eliminate basis elements whose relation matrix has a unit (or unit-leading-term) entry; repeat. |
| `MinimalBasis(S)` | Given a set or sequence S of homogeneous module elements, return a minimal basis of the submodule generated by S. | As above. |
| `Rank(M)` | Return the rank of M: the cardinality of `MinimalBasis(M)`. Unique iff M is graded or R is local. | Via `MinimalBasis`. |
| `ColonModule(M, J)` | Return the colon module M : J — the submodule of the ambient A of M consisting of all f ∈ A with f·g ∈ M for all g ∈ J. For principal J, reduces to a syzygy computation; in general, intersect colon modules for each generator of J. | Syzygy + intersection **[GP02]**. |
| `ColonIdeal(M, N)` | Return the colon ideal M : N — the ideal of R of all f ∈ R with f·N ⊂ M, for M, N submodules of a common supermodule. | Algorithm of **[GP02, §2.8.4]**. |
| `Annihilator(M)` | Return the annihilator ideal of M — the ideal of R of all f with f·M = 0. Equals the colon ideal 0_M : M. | Via `ColonIdeal`. |
| `FittingIdeal(M, i)` | Return the i-th Fitting ideal of M: ideal generated by the (r−i)-th minors of the presentation matrix of M, where r = Degree(M). See **[CLO98, p.229]**, **[Eis95, §20.2]**. | Minors of presentation matrix. |
| `FittingIdeals(M)` | Return the Fitting ideals for i = 0 to r = Degree(M) as a sequence. | Via `FittingIdeal`. |
| `SyzygyModule(M)` | Return the syzygy module S of M. If the basis B of M has length k, S has degree k and elements of S express syzygies among the k basis elements. Note: degree of result depends on the current basis. | Module Gröbner basis; syzygies are computed as a by-product. |
| `MinimalSyzygyModule(M)` | Return the syzygy module S of the **minimal** basis of M. S has degree equal to the cardinality of the minimal basis. | Module Gröbner basis of the minimal basis. |
| `SyzygyModule(Q)` | Given a sequence Q of polynomials from a multivariate polynomial ring P, return the module of syzygies of Q — a P-module of degree #Q consisting of all vectors v with Σ v[i]·Q[i] = 0. | Module Gröbner basis. |

*Worked example: H109E5 (rank of a quotient module over Q[x,y,z]; localization reduces rank from 3 to 2 since a unit appears).*

---

## 109.10 Changing Ring

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ChangeRing(M, S)` | Given R-module M (R a polynomial ring) and polynomial ring S of the same rank as R, construct the S-module obtained by coercing the coefficients of M's basis and relations into S. All old coefficient ring elements must be automatically coercible into the new coefficient ring. | — |

---

## 109.11 Hilbert Series

Hilbert series functions apply to graded or homogeneous modules; column weights are significant.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HilbertSeries(M)` | Given graded R-module M, return the Hilbert series H_M(t) as a univariate rational function over Z. The i-th coefficient is the vector-space dimension of the degree-i graded piece. | Algorithm of **[BS92]**. |
| `HilbertSeries(M, p)` | Return H_M(t) as a Laurent series to precision p. (Laurent series needed when grading has negative weights.) | Algorithm of **[BS92]**. |
| `HilbertDenominator(M)` | Return the unreduced Hilbert denominator D = ∏ᵢ(1 − t^{wᵢ}) (where wᵢ are the variable weights of R). | — |
| `HilbertNumerator(M)` | Return the unreduced Hilbert numerator N and valuation shift s, where N = D × t^s × H_M(t). s is non-zero only when M has negative column weights. | — |
| `HilbertPolynomial(I)` | Given graded R-module M, return the Hilbert polynomial H(d) ∈ Q[d] and the index of regularity (minimal k ≥ 0 such that H(d) agrees with the Hilbert function for all d ≥ k). | — |

*Worked example: H109E6 (Hilbert series, numerator, denominator, polynomial of quotient modules over Q[x,y,z]; negative grading case).*

---

## 109.12 Free Resolutions

Free resolutions are returned as chain complexes (see Chapter 56). Boundary maps have type `ModMPolHom`.

### 109.12.1 Constructing Free Resolutions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `FreeResolution(M)` | Given R-module M, return a free resolution C (a complex) and a comparison homomorphism f : C₀ → M. Parameters: `Minimal` (default `true` — construct minimal resolution; if `false`, construct non-minimal by successive syzygy modules); `Limit` (default 0 — if non-zero, compute at most l terms; set to rank(R) by default for affine/exterior algebras); `Homogenize` (default `true` — force/suppress internal homogenization trick); `Al` (default `"LaScala"` — select algorithm). | **La Scala (LS) algorithm [SS98]** extended with Faugère F4 **[Fau99]** block-normal-form techniques, used by default for homogeneous M over a finite field or Q. **Iterative algorithm** (`Al := "Iterative"`): successive syzygy modules, with optional minimization at each step. If M is non-homogeneous over a global ring, Magma may internally homogenize M to use the LS algorithm, then specialize. |
| `SetVerbose("Resolution", v)` | (Procedure.) Set verbose printing level for the free resolution algorithm and related functions to v. | — |

*Worked examples: H109E7 (minimal free resolution of R/I for the twisted cubic in P³; boundary maps; exactness check); H109E8 (non-minimal free resolution of R/I for 6 points in Q[x,y]; Hilbert-Burch theorem — 3×3 minors of boundary map regenerate I).*

### 109.12.2 Betti Numbers and Related Invariants

All functions in this section accept the same parameters as `FreeResolution`. By default the minimal free resolution is used. For graded M using the LS algorithm, computing Betti invariants may be faster than computing the full resolution.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BettiNumbers(M)` | Return Betti numbers of M: the sequence of degrees (ranks) of non-zero terms of the free resolution. Unique if M is graded or over a local ring. | Via minimal free resolution. |
| `BettiNumber(M, i, j)` | Return the graded Betti number β_{i,j}: number of generators of degree j in the i-th term F_i of the resolution. | Via minimal free resolution. |
| `MaximumBettiDegree(M, i)` | Return the maximum degree of generators in the i-th term of the resolution (maximum j with β_{i,j} ≠ 0). | Via `BettiNumber`. |
| `BettiTable(M)` | Return the Betti table of M as a sequence S of sequences and a shift s, designed so that S[1,1] ≠ 0 and S[i,j] = BettiNumber(M, i, j−i+s). | Via minimal free resolution. |
| `Regularity(M)` | For graded M or M over a local ring: return the **Castelnuovo-Mumford regularity** — the least r such that in the minimal free resolution the maximum generator degree of F_i is at most i + r. See **[Eis95, §20.5]**, **[DL06, p.167]**. | Minimal free resolution. |
| `HomologicalDimension(M)` | Return the homological dimension of M: the length (number of non-zero boundary maps) of a minimal free resolution. | Minimal free resolution. |

*Worked examples: H109E9 (Koszul complex for the ideal of n variables; Betti numbers = binomial coefficients; regularity 0); H109E10 (Koszul complex over exterior algebra; infinite resolution with Betti numbers = C(i+n−1, n−1)); H109E11 (non-homogeneous quotient module; Betti numbers non-unique; localization gives unique Betti numbers); H109E12 (Hilbert series numerator from Betti numbers via the formula H_M = (Σ (−1)^i β_{i,j} t^j)/D — verified on twisted cubic and 4×4 minor ideal); H109E13 (regularity of a weighted graded module; leading monomial ideal gives upper bound; space curve genus 11 construction via resolutions and syzygy computations).*

---

## 109.13 The Hom Module and Ext

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Hom(M, N)` | Given R-modules M and N, return H = Hom_R(M, N) as an abstract reduced module, together with a transfer map f : H → {homomorphisms M → N}. An element h ∈ H maps to an actual `ModMPolHom`; f is invertible. If M and N are graded, H is graded: the degree of h ∈ H equals the degree of the corresponding homomorphism. | Free resolution of M, then Hom applied termwise. |
| `Hom(C, N)` | Given a complex C of R-modules and R-module N, return the complex Hom_R(C, N). The i-th term is Hom_R(C_i, N); boundary maps are induced by those of C via the functor Hom_R(−, N) (arrows reversed). See **[Eis95, p.63]**. | Functorial construction. |
| `Ext(i, M, N)` | Given integer i ≥ 0 and R-modules M, N, return Ext^i(M, N): the homology at the i-th term of the complex Hom_R(C, N), where C is a free resolution of M. | Free resolution + Hom functor. |

*Worked example: H109E15 (Hom module and explicit homomorphisms between two graded quotient modules over Q[x,y,z]; degree, transfer map; Ext computation for canonical sheaf).*

---

## 109.14 Tensor Products and Tor

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `TensorProduct(M, N)` | Given R-modules M and N, return the tensor product T = M ⊗_R N as an ambient module, together with the bilinear map f : M × N → T. If M and N are graded, T is graded. | Free resolution presentation. |
| `TensorProduct(C, N)` | Given a complex C of R-modules and R-module N, return the complex C ⊗_R N. The i-th term is C_i ⊗_R N; boundary maps are induced via the functor − ⊗_R N. See **[Eis95, p.64]**. | Functorial construction. |
| `Tor(i, M, N)` | Given integer i ≥ 0 and R-modules M, N, return Tor_i(M, N): homology at the i-th term of the complex C ⊗_R N, where C is a free resolution of M. | Free resolution + tensor functor. |

*Worked example: H109E16 (tensor product and Tor modules for two graded quotient modules over Q[x,y,z]).*

---

## 109.15 Cohomology of Coherent Sheaves

Coherent sheaves on ordinary projective space P^m_k are represented by graded modules over the coordinate ring. The cohomology groups of a sheaf and its Serre twists are computed via the **Beilinson-Gelfand-Gelfand (BGG) correspondence** as described in **[EFS03]** and **[DE02]**.

The algorithm constructs (part of) the Tate resolution — a doubly infinite exact sequence of graded free modules over the exterior algebra A with m+1 generators (dim 2^{m+1} over k). The key observation is that all terms of the Tate resolution at indices ≥ reg(M) are pure graded with dimensions given by the Hilbert polynomial of M. Two consecutive terms at this level are computed from graded pieces of M and the linear multiplication maps. The resolution is then extended backwards by computing the A-projective resolution of a kernel, using non-commutative Gröbner bases for exterior algebras. Projective resolution data is cached for repeated calls.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CohomologyDimension(M, r, n)` | Given graded module M over P = k[x₀,…,x_m] (k an exact field), return dim_k H^r(P^m_k, M̃(n)) where M̃ is the coherent sheaf associated to M and M̃(n) is its n-th Serre twist. r must be a non-negative integer; n can be any integer. Parameter `Verbose Cohom` (maximum 1): verbose output level. | BGG correspondence + Tate resolution **[EFS03, DE02]**; non-commutative Gröbner bases for the exterior algebra resolution; results cached per M. |

*Worked example: H109E17 (Enriques surface of degree 9 in P⁴ over GF(17); computing H⁰, H¹, H² of the structure sheaf O_X; Ext² for the canonical sheaf K_X; verification of Serre duality dim H^r(O_X(n)) = dim H^{2−r}(K_X(−n))).*

---

## 109.16 Bibliography

| Key | Reference |
|-----|-----------|
| **[AL94]** | William Adams and Philippe Loustaunau. *An Introduction to Gröbner Bases.* Volume 3 of Graduate Studies in Mathematics. American Mathematical Society, Providence, R.I., 1994. |
| **[BS92]** | David Bayer and Michael Stillman. *Computation of Hilbert Functions.* J. Symbolic Comp., **14**(1):31–50, 1992. |
| **[CLO98]** | David Cox, John Little, and Donal O'Shea. *Using Algebraic Geometry.* Graduate Texts in Mathematics. Springer, New York–Berlin–Heidelberg, 1998. |
| **[DE02]** | Wolfram Decker and David Eisenbud. *Sheaf Algorithms Using the Exterior Algebra.* In Eisenbud et al. (eds.), *Computations in Algebraic Geometry with Macaulay2*, volume 8 of Springer Algorithms and Computation in Mathematics Series, pages 215–247. Springer-Verlag, 2002. |
| **[DL06]** | Wolfram Decker and Christoph Lossen. *Computing in Algebraic Geometry.* Volume 16 of Algorithms and Computation in Mathematics. Springer, New York–Berlin–Heidelberg, 2006. |
| **[EFS03]** | David Eisenbud, Gunnar Floystad, and Frank-Olaf Schreyer. *Sheaf Cohomology and Free Resolutions over Exterior Algebras.* Trans. Am. Math. Soc., **355**:4397–4426, 2003. |
| **[Eis95]** | David Eisenbud. *Commutative Algebra with a View Toward Algebraic Geometry.* Volume 150 of Graduate Texts in Mathematics. Springer, New York–Berlin–Heidelberg, 1995. |
| **[Fau99]** | Jean-Charles Faugère. *A New Efficient Algorithm for Computing Gröbner Bases (F4).* Journal of Pure and Applied Algebra, **139**(1–3):61–88, 1999. |
| **[GP02]** | G.-M. Greuel and G. Pfister. *A Singular Introduction to Commutative Algebra.* Springer-Verlag, Berlin–Heidelberg–New York, 2002. |
| **[SS98]** | Roberto La Scala and Michael Stillman. *Strategies for Computing Minimal Free Resolutions.* J. Symbolic Comp., **26**(4):409–431, 1998. |
| **[ST02]** | Frank-Olaf Schreyer and Fabio Tonoli. *Needles in a Haystack: Special Varieties via Small Fields.* In Eisenbud et al. (eds.), *Computations in Algebraic Geometry with Macaulay2*, volume 8 of Springer Algorithms and Computation in Mathematics Series, pages 251–277. Springer-Verlag, 2002. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Module Gröbner bases (Buchberger for modules) **[AL94, CLO98]** | `Groebner`, `sub< >`, `quo< >`, `SPolynomial`, `NormalForm`, `M subset N`, `M eq N`, `IsZero(f)`, `f in M` |
| Module intersection **[GP02, §2.8.3]** | `M meet N` |
| Colon module / colon ideal **[GP02, §2.8.4]** | `ColonModule`, `ColonIdeal`, `Annihilator` |
| Fitting ideals **[CLO98, Eis95]** | `FittingIdeal`, `FittingIdeals` |
| Syzygy modules | `SyzygyModule`, `MinimalSyzygyModule`, `Kernel` |
| Hilbert series **[BS92]** | `HilbertSeries`, `HilbertNumerator`, `HilbertDenominator`, `HilbertPolynomial` |
| La Scala free resolution algorithm **[SS98]** + Faugère F4 **[Fau99]** | `FreeResolution` (default for homogeneous modules over Q or finite field) |
| Iterative (successive syzygy) free resolution | `FreeResolution(:Al := "Iterative")` |
| Betti numbers and regularity **[Eis95, DL06]** | `BettiNumbers`, `BettiNumber`, `MaximumBettiDegree`, `BettiTable`, `Regularity`, `HomologicalDimension` |
| Hom functor / Ext **[Eis95]** | `Hom`, `Ext` |
| Tensor product functor / Tor **[Eis95]** | `TensorProduct`, `Tor` |
| BGG correspondence / Tate resolution / sheaf cohomology **[EFS03, DE02]** | `CohomologyDimension` |
| Serre twist | `Twist` |
| Localization | `Localization` |
