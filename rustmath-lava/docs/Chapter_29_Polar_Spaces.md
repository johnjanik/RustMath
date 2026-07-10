# Chapter 29 — Polar Spaces

**Handbook part:** IV — Matrices and Linear Algebra
**Handbook pages:** 609–635 (PDF pages 742–768)

---

## Scope and overview

This chapter describes Magma functions for working with quadratic, bilinear and sesquilinear forms defined on vector spaces. The emphasis is on vector spaces defined over finite fields, though in some instances the functions apply more widely. For quadratic forms defined on lattices see Chapter 32; for the interpretation of reflexive forms as algebras with involution see Chapter 87.

A σ-sesquilinear form β on a vector space V over a field K (where σ is a field automorphism) is reflexive if β(u,v) = 0 implies β(v,u) = 0 for all u, v ∈ V. By a theorem of Brauer [Bra36] (sometimes called the Birkhoff–von Neumann theorem), up to a non-zero scalar multiple there are three types of non-degenerate reflexive forms: **alternating** (σ = identity, β(u,u) = 0; isometry group is symplectic), **symmetric** (σ = identity, β(u,v) = β(v,u); isometry group is orthogonal in odd characteristic, or pseudo-alternating in characteristic 2), and **hermitian** (σ of order 2, β(u,v) = σβ(v,u); isometry group is unitary). The partially ordered set of totally isotropic subspaces with respect to a reflexive form is a polar space; this notion is extended to include vector spaces furnished with a quadratic form.

A quadratic form Q on V with polar form β satisfies Q(av) = a²Q(v) and β(u,v) = Q(u+v) − Q(u) − Q(v). If the characteristic is not 2, β determines Q; in characteristic 2 one must treat quadratic forms independently to capture the orthogonal groups. A vector space V with an attached quadratic form is a quadratic space (orthogonal geometry). General references are [Bou07, Tay92].

---

## 29.1 Introduction

Every vector space V created via `VectorSpace` (synonym `KSpace`) carries an associated bilinear form represented by a matrix, accessible via `InnerProductMatrix(V)` (attribute `ip_form`); the default is the identity matrix. To accommodate hermitian forms, a vector space of type `ModTupFld` also has an `Involution` attribute intended to hold a field automorphism of order 2. A quadratic form may additionally be attached via `QuadraticSpace`, accessed via `QuadraticFormMatrix`.

---

## 29.2 Reflexive Forms

Background theory section (no intrinsics). Defines alternating, symmetric, hermitian, skew-hermitian and pseudo-alternating forms, and the correspondence between form type and classical group type.

### 29.2.1 Quadratic Forms

Background theory section (no intrinsics). Defines quadratic forms, their polar forms, and the extension of the polar space concept to quadratic spaces.

---

## 29.3 Inner Products

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DotProduct(u, v)` | Returns β(u, v) = u J σ(vᵀ), where J is the inner product matrix of the generic space of the parent of u and v, and σ is the field automorphism from the `Involution` attribute (or the identity if unassigned). Evaluates a bilinear or sesquilinear form. | Direct matrix evaluation. |
| `DotProductMatrix(W)` | The matrix of pairwise inner products of the vectors in sequence W, computed using `DotProduct` (so the `Involution` attribute is taken into account). | Direct matrix evaluation. |
| `GramMatrix(V)` | If B is the basis matrix of V and J is the inner product matrix, returns B J Bᵀ. The `Involution` attribute is ignored. | Direct computation. |
| `InnerProductMatrix(V)` | The inner product matrix attached to the generic space of V; the attribute `V'ip_form`. | — |

*Worked examples: H29E1 (constructing a vector space with a given inner product matrix over GF(11)); H29E2 (difference between `GramMatrix` and `InnerProductMatrix` over a quadratic field); H29E3 (comparison of `DotProduct` vs `InnerProduct` with and without `Involution`).*

### 29.3.1 Orthogonality

The left orthogonal complement of X ⊆ V is ⊥X = { u ∈ V | β(u, x) = 0 for all x ∈ X }; the right orthogonal complement is X⊥ = { u ∈ V | β(x, u) = 0 for all x ∈ X }. If β is reflexive, ⊥X = X⊥.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `OrthogonalComplement(V, X : Right)` | `Right BoolElt` Default: `false`. The left orthogonal complement of X in V. Set `Right := true` to obtain the right orthogonal complement. | Linear algebra (null-space computation). |
| `Radical(V : Right)` | `Right BoolElt` Default: `false`. The left radical ⊥V of the inner product space V. Set `Right := true` for the right radical. | Null-space of the inner product matrix. |
| `IsNondegenerate(V)` | Returns `true` if the determinant of the matrix of inner products of the basis vectors of V is non-zero, otherwise `false`. Accounts for the `Involution` attribute if assigned. | Determinant check. |
| `SingularRadical(V)` | The kernel of the restriction of the quadratic form Q of the quadratic space V to the radical of V. (Over a perfect field of characteristic 2, this restriction is a semilinear functional x ↦ x².) | Kernel computation. |
| `IsNonsingular(V)` | Returns `true` if V is a non-singular quadratic space (singular radical is zero), otherwise `false`. | — |

---

## 29.4 Isotropic and Singular Vectors and Subspaces

A non-zero vector v is **isotropic** (with respect to a reflexive form β) if β(v,v) = 0; it is **singular** (with respect to a quadratic form Q) if Q(v) = 0. An ordered pair (u, v) with u, v isotropic (or singular in the quadratic case) and β(u, v) = 1 is a **hyperbolic pair**. A polar space has a **hyperbolic splitting** V = L₁ ⊥ L₂ ⊥ ··· ⊥ Lₘ ⊥ W where each Lᵢ is spanned by a hyperbolic pair; if the form is non-degenerate and not pseudo-alternating, W contains no isotropic/singular vectors, m is the **Witt index**, and W is the anisotropic component.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HasIsotropicVector(V)` | Determines whether the polar space V contains an isotropic vector; if so, the second return value is a representative. | Search. |
| `HasSingularVector(V)` | Determines whether the quadratic space V contains a singular vector; if so, the second return value is a representative. | Search. |
| `HyperbolicPair(V, u)` | Given a singular or isotropic vector u not in the radical, returns a vector v such that (u, v) is a hyperbolic pair. | Direct construction. |
| `HyperbolicSplitting(V)` | Returns a maximal list of pairwise orthogonal hyperbolic pairs together with a basis for the orthogonal complement of their span. Requires non-degenerate form; base ring must be a finite field except for symplectic spaces. | Witt-decomposition algorithm. |
| `IsTotallyIsotropic(V)` | Returns `true` if the polar space V is totally isotropic, otherwise `false`. | — |
| `IsTotallySingular(V)` | Returns `true` if the quadratic space V is totally singular, otherwise `false`. | — |
| `WittDecomposition(V)` | The Witt decomposition of V: a 4-tuple (rad(V), P, N, W) where P = ⟨e₁,…,eₘ⟩ and N = ⟨f₁,…,fₘ⟩ are totally isotropic, (eᵢ, fᵢ) are hyperbolic pairs, and W is the anisotropic component. | Hyperbolic splitting algorithm. |
| `WittIndex(V)` | The Witt index of the polar space V: half the dimension of a maximal hyperbolic subspace. | — |
| `MaximalTotallyIsotropicSubspace(V)` | A representative maximal totally isotropic subspace of the polar space V. | — |
| `MaximalTotallySingularSubspace(V)` | A representative maximal totally singular subspace of the quadratic space V. | — |

*Worked examples: H29E4 (pseudo-symplectic space over F₂: not every isotropic vector belongs to a hyperbolic pair); H29E5 (hyperbolic splitting of a symmetric bilinear form over GF(7,2)); H29E6 (handling a degenerate polar space by splitting off the radical first).*

---

## 29.5 The Standard Forms

The "standard" forms of maximal Witt index, plus the quadratic forms of non-maximal Witt index over finite fields. These are (except for the orthogonal groups) the forms preserved by Magma's classical groups over finite fields. If J is the matrix of the form, the group element X preserves it iff X J Xᵀ = J.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `StandardAlternatingForm(n, R)` / `StandardAlternatingForm(n, q)` | For n = 2m, returns the n×n matrix of the non-degenerate alternating form over ring R (or field of q elements) such that (e₁, e₂ₘ), (e₂, e₂ₘ₋₁), …, (eₘ, eₘ₊₁) are mutually orthogonal hyperbolic pairs. Isometry group is Sp(2m, R). | Standard construction. |
| `StandardPseudoAlternatingForm(n, K)` / `StandardPseudoAlternatingForm(n, q)` | The matrix of the standard pseudo-alternating form of degree n over field K (or finite field of order q), which must have characteristic 2: a symmetric form which is not alternating. | Standard construction. |
| `StandardHermitianForm(n, K)` / `StandardHermitianForm(n, q)` | First return: the n×n anti-diagonal matrix (δᵢ,ₙ₋ᵢ₊₁) over K (or the field of q² elements). Second return: the field involution x ↦ xq (finite field case) or complex conjugation. For finite fields, isometry group is GU(n, q). | Standard construction. |
| `StandardQuadraticForm(n, K : Minus, Variant)` / `StandardQuadraticForm(n, q : Minus, Variant)` | `Minus BoolElt` Default: `false`; `Variant MonStgElt` Default: `"Default"`. An n×n upper triangular matrix representing a quadratic form over K (or field of order q). Default: maximal Witt index, upper triangular with non-zero entries δᵢ,ₙ₋ᵢ₊₁. If `Minus := true` and n = 2m, returns a form of Witt index m−1 (finite fields only). `Variant` may be `"Default"`, `"Revised"`, or `"Original"` (the last returns the form preserved by `OldGOMinus(2m,q)`). | Standard construction. |
| `StandardSymmetricForm(n, K)` / `StandardSymmetricForm(n, q : Minus, Variant)` | `Minus BoolElt` Default: `false`; `Variant MonStgElt` Default: `"Default"`. In all cases returns Q + Qᵀ, where Q is the corresponding standard quadratic form. | Standard construction. |

*Worked examples: H29E7 (standard alternating form over GF(5); every non-zero vector is isotropic); H29E8 (standard quadratic form of minus type over GF(7²)); H29E9 (comparison of Default vs Revised variant for minus-type form).*

---

## 29.6 Constructing Polar Spaces

A vector space V is recognised as a polar space if: (1) there is a quadratic form attached, (2) there is a field involution and the inner product matrix is hermitian or skew-hermitian, or (3) the inner product matrix is symmetric or alternating. No non-degeneracy check is performed.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsPolarSpace(V)` | Returns `true` if V is a quadratic space or if the Gram matrix of V is a reflexive form. | Type/attribute check. |
| `PolarSpaceType(V)` | The type of the polar space V, returned as a string (e.g., `"orthogonal space"`, `"quadratic space"`, etc.). | Attribute inspection. |

*Worked example: H29E10 (standard vector space over the rationals is an orthogonal space).*

### 29.6.1 Symplectic Spaces

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SymplecticSpace(J)` | The symplectic space of dimension n defined by the n×n matrix J. Checks that J is alternating. | Attribute assignment + validation. |
| `IsSymplecticSpace(W)` | Returns `true` if the `Involution` attribute of the generic space is not assigned and the space carries an alternating form. Note: a quadratic space in characteristic 2 also satisfies these conditions. | Attribute + form check. |
| `IsPseudoSymplecticSpace(W)` | Returns `true` if the base field has characteristic 2, `Involution` is not assigned, and the form is symmetric but not alternating. | Attribute + form check. |

### 29.6.2 Unitary Spaces

A unitary space is a vector space V whose ambient generic space has the `Involution` attribute assigned, and whose inner product matrix is hermitian or skew-hermitian with respect to that involution.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `UnitarySpace(J, sigma)` | The n-dimensional unitary space over the base field K of J, where σ is an automorphism of K of order 2 and J is an n×n matrix hermitian or skew-hermitian with respect to σ. | Attribute assignment + validation. |
| `IsUnitarySpace(W)` | Returns `true` if the `Involution` attribute of the generic space of W is assigned and the form is hermitian or skew-hermitian when restricted to W. | Attribute + form check. |
| `ConjugateTranspose(M, sigma)` | The transpose of the matrix σ(M), where σ is an automorphism of the base field of M. | Direct computation. |

*Worked example: H29E11 (unitary geometry over GF(25) with standard hermitian form; `DotProduct` accounts for the involution, `InnerProduct` ignores it).*

### 29.6.3 Quadratic Spaces

A quadratic space is a vector space V with an attached quadratic form. The polar form is the inner product matrix J. If characteristic ≠ 2, the value of the quadratic form on a row vector v is ½ v J vᵀ.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `QuadraticSpace(Q)` | The quadratic space of dimension n defined by an upper triangular n×n matrix Q. Inner product matrix is Q + Qᵀ. If Q is not upper triangular, the space is constructed but correctness of other functions is not guaranteed. | Attribute assignment. |
| `QuadraticSpace(f)` | The quadratic space whose quadratic form is given by the quadratic polynomial f in n variables; for i ≤ j, the (i,j)-entry of the form matrix is the coefficient of xᵢxⱼ in f. | Polynomial parsing. |
| `SymmetricToQuadraticForm(J)` | For characteristic not 2, the upper triangular matrix representing the same quadratic form as the symmetric matrix J. | Direct construction (characteristic ≠ 2 only). |
| `QuadraticFormMatrix(V)` | The upper triangular matrix representing the quadratic form of the quadratic space V. | Attribute access. |
| `QuadraticNorm(v)` | The value Q(v), where Q is the quadratic form attached to the generic space of the parent of v. | Form evaluation. |
| `QuadraticFormPolynomial(V)` | The polynomial Σᵢ≤ⱼ qᵢⱼ xᵢ xⱼ, where Q = (qᵢⱼ) is the quadratic form matrix of V. | Direct construction. |
| `OrthogonalSum(V, W)` | The orthogonal direct sum of the quadratic spaces V and W. | Block-diagonal form construction. |
| `TotallySingularComplement(V, U, W)` | Given totally singular subspaces U and W of quadratic space V such that U⊥ ∩ W = 0, returns a totally singular subspace X with V = X ⊕ U⊥ and W ⊆ X. | Complement construction. |
| `Discriminant(V)` | For V over a finite field K with Gram matrix J: the discriminant is det(J) modulo squares (0 if det(J) is a square in K, 1 if non-square). Requires J non-degenerate. | Determinant + square test. |
| `ArfInvariant(V)` | The Arf invariant of the quadratic space V. Currently available for even-dimensional quadratic spaces over finite fields of characteristic 2 only: 0 if Witt index is m (maximal), 1 if Witt index is m−1. | Standard invariant computation. |
| `DicksonInvariant(V, f)` | The Dickson invariant of the isometry f of the quadratic space V: the rank (mod 2) of 1 − f. Defines a homomorphism O(V) → Z/2Z when the polar form is non-degenerate. | Rank computation modulo 2. |
| `SpinorNorm(V, f)` | The spinor norm of the isometry f of the quadratic space V: the discriminant of the Wall form (§29.8) of f. | Via `WallForm` + `Discriminant`. |
| `HyperbolicBasis(U, B, W)` | Given complementary totally singular subspaces U and W of a quadratic space and a basis B for U, returns a sequence of pairwise orthogonal hyperbolic pairs whose second components form a basis for W. | Gram–Schmidt-style construction. |
| `OrthogonalReflection(a)` | The reflection determined by a non-singular vector a of a quadratic space. | Standard formula. |
| `RootSequence(V, f)` | Given a matrix f representing an isometry of the quadratic space V, returns a sequence of vectors such that the product of the corresponding orthogonal reflections is f. Returns the empty sequence if f is the identity. | Factorisation into reflections. |
| `ReflectionFactors(V, f)` | Given a matrix f representing an isometry of V, returns a sequence of reflections whose product is f. Empty sequence corresponds to the identity. | Factorisation into reflections. |
| `SiegelTransformation(u, v)` | The Siegel transformation (also called Eichler transformation) ρᵤ,ᵥ defined by x ↦ x + β(x,v)u − β(x,u)v − Q(v)β(x,u)u, where u is singular, β(u,v) = 0, and the common parent is a quadratic space. | Direct formula. |

*Worked examples: H29E12 (quadratic space defined by a polynomial over the rationals); H29E13 (group of isometries generated by Siegel transformations in a quadratic space over GF(3)).*

---

## 29.7 Isometries and Similarities

A linear transformation g of V is an **isometry** if it preserves the bilinear/sesquilinear form β; it is a **similarity** if it preserves β up to a non-zero scalar multiple.

### 29.7.1 Isometries

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsIsometry(U, V, f)` | Returns `true` if the map f is an isometry from U to V with respect to their attached forms. | Form-preservation check. |
| `IsIsometry(f)` | Returns `true` if the map f is an isometry from its domain to its codomain. | Form-preservation check. |
| `IsIsometry(V, g)` | Returns `true` if the matrix g is an isometry of V with respect to the attached form. | Matrix equation check. |
| `IsIsometric(V, W)` | Determines whether the polar spaces V and W are isometric; if they are, returns an isometry as a map (second return value). | Classification of forms; congruence testing. |
| `CommonComplement(V, U, W)` | A common complement to the subspaces U and W in V (U and W must have the same dimension). Used internally by `ExtendIsometry`. | Subspace complement construction. |
| `ExtendIsometry(V, U, f)` | An extension of the isometry f : U → V to an isometry V → V, where U is a subspace of the polar space V. Implements Witt's theorem. Requires f(U ∩ rad(V)) = f(U) ∩ rad(V). If characteristic 2 and the form is symmetric, it must be alternating. | **Witt's theorem** on extension of isometries. |
| `IsometryGroup(V)` | The group of isometries of the polar space V, including degenerate polar spaces and quadratic spaces in characteristic 2. | Stabiliser computation. |

*Worked examples: H29E14 (identity form vs standard symmetric form over GF(5): similar but not isometric); H29E15 (alternating forms over GF(25): finding the congruence matrix M via `IsIsometric`); H29E16 (alternative via `TransformForm`); H29E17 (isometry group of a degenerate quadratic space over GF(4)); H29E18 (conjugating isometry groups via the congruence matrix).*

### 29.7.2 Similarities

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsSimilarity(U, V, f)` | Returns `true` if the map f is a similarity from U to V with respect to their attached forms. | Form-preservation-up-to-scalar check. |
| `IsSimilarity(f)` | Returns `true` if the map f is a similarity from its domain to its codomain. | Form-preservation-up-to-scalar check. |
| `IsSimilarity(V, g)` | Returns `true` if the matrix g is a similarity of V with respect to the attached form. | Matrix equation check. |
| `SimilarityGroup(V)` | The group of similarities of the polar space V, including degenerate polar spaces and quadratic spaces in characteristic 2. | Stabiliser computation. |

---

## 29.8 Wall Forms

Given an isometry f of a quadratic or symplectic space V with bilinear form β, the **Wall form** θ is defined on the image I of 1 − f by θ(u, v) = β(w, v) where u = w(1 − f). In general the Wall form is not reflexive.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `WallForm(V, f)` | The space of the Wall form of f and its embedding in V. | Direct construction from image of 1 − f. |
| `WallIsometry(V, I, mu)` | The inverse of `WallForm`: the isometry corresponding to the embedding µ : I → V, where V is a quadratic or symplectic space. | Inverse construction. |
| `WallDecomposition(V, f)` | For any isometry f of V: returns a Wall-regular element fᵣ and a unipotent element fᵤ such that f = fᵣ fᵤ = fᵤ fᵣ. (An isometry is Wall-regular if the restriction of 1 − f to its image is invertible.) | Decomposition algorithm. |
| `SemiOrthogonalBasis(V)` | A semi-orthogonal basis e₁, e₂, …, eₙ for V with respect to its bilinear form: β(eᵢ, eⱼ) = 0 for i < j. Requires a non-degenerate, non-alternating form; if base field is F₂ the form must be symmetric. | Triangularisation algorithm. |

---

## 29.9 Invariant Forms

Given a matrix group G acting on a vector space V over a finite field F, the space of G-invariant bilinear forms is isomorphic to HomG(V, V*). If G is irreducible and EndG(V) = D (a finite field), then HomG(V, V*) ≅ D as vector spaces; the symmetric and alternating invariant spaces are isomorphic to subfields of D, so their dimensions divide dim_F(V) or are 0.

In characteristic ≠ 2, every G-invariant form splits into a symmetric plus an alternating part. In characteristic 2, alternating forms are symmetric; if G is irreducible there is a unique G-invariant quadratic form Q with J = Q + Qᵀ for any invariant alternating J.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `InvariantBilinearForms(G)` | For a matrix group G, returns two sequences: a basis for the space of G-invariant symmetric forms and a basis for the space of G-invariant alternating forms. | Computation via HomG(V, V*). |
| `InvariantQuadraticForms(G)` | A basis for the space of quadratic forms preserved by the irreducible matrix group G. | Lifting from invariant alternating forms. |
| `SemilinearDual(M, mu)` | The semilinear dual of the G-module M with respect to the field automorphism mu. | Module construction. |
| `InvariantSesquilinearForms(G)` | A basis for the space of hermitian forms preserved by the matrix group G. Computed via HomG(V, V̄*) where V̄* is the semilinear dual. | Computation via HomG(V, semilinear dual). |
| `InvariantFormBases(G)` | Returns four sequences: bases for the spaces of symmetric, alternating, hermitian and quadratic forms preserved by G. | Combined computation. |

*Worked examples: H29E19 (reducible group with unique invariant bilinear form up to scalar); H29E20 (irreducible group: symmetric and alternating invariant spaces each have dimension 2); H29E21 (cyclic group of order 13 over GF(4): 3-dimensional space of invariant quadratic forms); H29E22 (G-invariant hermitian forms via `SemilinearDual` and `AHom`); H29E23 (irreducible group realizable over a subfield has both bilinear and sesquilinear invariant forms).*

### 29.9.1 Semi-invariant Forms

A bilinear form β is **semi-invariant** for G if for all g ∈ G there exists a scalar λ(g) such that β(ug, vg) = λ(g)β(u,v); λ : G → F× is a homomorphism. The space of semi-invariant bilinear forms is isomorphic to HomG(V, V*λ), where V*λ is the **twisted dual** with G-action ⟨v, ϕg⟩ = λ(g)⟨vg⁻¹, ϕ⟩.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `TwistedDual(M, lambda)` | The twisted dual of the G-module M with respect to the linear character lambda. | Module construction. |
| `SemiInvariantBilinearForms(G)` | A sequence of triples ⟨L, S, A⟩ where L is a sequence of field elements (one per generator) defining a homomorphism G → F×, and S, A are bases for the symmetric and alternating semi-invariant form spaces. | Computation via HomG(V, twisted dual). |
| `SemiInvariantQuadraticForms(G)` | A sequence of pairs ⟨L, Q⟩ where L defines a homomorphism G → F× and Q is a basis for the space of semi-invariant quadratic forms. | Computation via twisted dual. |
| `TwistedSemilinearDual(M, lambda, mu)` | The twisted semilinear dual of the G-module M with respect to linear character lambda and field automorphism mu. | Module construction. |
| `SemiInvariantSesquilinearForms(G)` | A sequence of pairs ⟨L, H⟩ where L defines a homomorphism G → F₀× (F₀ = fixed field of the involution, the base field being a quadratic extension of F₀) and H is a basis for the space of semi-invariant hermitian forms. | Computation via twisted semilinear dual. |

*Worked example: H29E24 (irreducible but not absolutely irreducible normal subgroup H of N; `SemiInvariantSesquilinearForms` for both H and N over GF(9)).*

---

## 29.10 Bibliography

| Key | Reference |
|-----|-----------|
| **[Bou07]** | N. Bourbaki. *Éléments de mathématique. Algèbre. Chapitre 9.* Springer-Verlag, Berlin, 2007. Reprint of the 1959 original. |
| **[Bra36]** | Richard Brauer. A characterization of null systems in projective space. *Bull. Amer. Math. Soc.*, 42(4):247–254, 1936. |
| **[Bue95]** | Francis Buekenhout. *Handbook of incidence geometry.* North-Holland, Amsterdam, 1995. |
| **[Tay92]** | Donald E. Taylor. *The geometry of the classical groups*, volume 9 of Sigma Series in Pure Mathematics. Heldermann Verlag, Berlin, 1992. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Form evaluation (bilinear/sesquilinear) | `DotProduct`, `DotProductMatrix`, `GramMatrix`, `InnerProductMatrix` |
| Orthogonal complement / radical | `OrthogonalComplement`, `Radical`, `IsNondegenerate`, `SingularRadical`, `IsNonsingular` |
| Hyperbolic decomposition / Witt theory | `HyperbolicPair`, `HyperbolicSplitting`, `WittDecomposition`, `WittIndex`, `MaximalTotallyIsotropicSubspace`, `MaximalTotallySingularSubspace` |
| Standard form construction | `StandardAlternatingForm`, `StandardPseudoAlternatingForm`, `StandardHermitianForm`, `StandardQuadraticForm`, `StandardSymmetricForm` |
| Polar space recognition and construction | `IsPolarSpace`, `PolarSpaceType`, `SymplecticSpace`, `UnitarySpace`, `QuadraticSpace`, `ConjugateTranspose`, `SymmetricToQuadraticForm` |
| Quadratic form invariants (Dickson, Arf, spinor norm, discriminant) | `Discriminant`, `ArfInvariant`, `DicksonInvariant`, `SpinorNorm` |
| Factorisation into reflections / Siegel transformations | `OrthogonalReflection`, `RootSequence`, `ReflectionFactors`, `SiegelTransformation`, `HyperbolicBasis` |
| Witt's theorem (extension of isometries) | `ExtendIsometry`, `CommonComplement` |
| Isometry and similarity testing / groups | `IsIsometry`, `IsIsometric`, `IsometryGroup`, `IsSimilarity`, `SimilarityGroup` |
| Wall form theory | `WallForm`, `WallIsometry`, `WallDecomposition`, `SemiOrthogonalBasis`, `SpinorNorm` |
| Invariant bilinear/quadratic/sesquilinear forms | `InvariantBilinearForms`, `InvariantQuadraticForms`, `InvariantSesquilinearForms`, `InvariantFormBases`, `SemilinearDual` |
| Semi-invariant forms | `SemiInvariantBilinearForms`, `SemiInvariantQuadraticForms`, `SemiInvariantSesquilinearForms`, `TwistedDual`, `TwistedSemilinearDual` |
