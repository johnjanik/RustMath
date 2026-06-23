# Chapter 116 — Algebraic Surfaces

**Handbook part:** XV — Algebraic Geometry
**Handbook pages:** 3759–3823 (PDF pages 3890–3957)

---

## Scope and overview

Chapter 116 collects Magma's specialised geometric functionality for algebraic surfaces (2-dimensional, geometrically integral schemes over a field). The general surface type `Srfc` is a subtype of `Sch`, so all scheme and coherent-sheaf machinery (Chapters 112–113) is available alongside the intrinsics here.

The chapter is divided into three major parts:

1. **General Surfaces (§116.2)** — newer functionality for surfaces in arbitrary-dimensional ordinary projective ambients. Requires at most simple (A-D-E) singularities for most intrinsics. Covers creation, fundamental invariants (geometric genus, irregularity, Chern numbers, Hodge numbers), singularity tests (normality, A-D-E type), Kodaira-Enriques classification, and computation of minimal/canonical models. A suite of random-surface constructors for non-general-type families in P⁴ (following **[DES93]**) is also provided.

2. **Surfaces in P³ (§116.3)** — older package for (singular) hypersurfaces in P³. Core is a formal desingularization package (Jung method) due to Tobias Beck **[Bec07]**, giving algebraic-power-series representations of the components of any desingularization. Built on this are: computation of adjoint linear systems and birational invariants **[BS08]**; classification and reduction of rational hypersurfaces to standard models (Schicho's algorithm **[Sch98]**); and explicit parametrization routines for rational surfaces.

3. **Del Pezzo Surfaces (§116.4)** — specialised code for Del Pezzo surfaces in their anticanonical embeddings. Covers creation, parametrization by degree (degrees 5–9 via Lie algebra method **[dG06, dGP, HS06, GSHPBS12]**; degrees 3–4 singular cases via direct special-case code), minimization and reduction of degree-3 and degree-4 Del Pezzos, point-counting and isomorphism testing for cubic surfaces over finite fields, construction via hexahedral coefficients, and classical invariant theory (Clebsch-Salmon invariants, covariants, contravariants, the pentahedron).

---

## 116.1 Introduction

The chapter introduces the `Srfc` type, distinguishes the two main bodies of functionality (general ordinary-projective surfaces with at-most-ADE singularities vs. arbitrary P³ hypersurfaces using formal desingularization), and explains the significance of allowing simple singularities: they do not affect pluri-canonical sheaf computations and arise naturally in canonical/anticanonical models.

No intrinsics in this section; it is expository.

---

## 116.2 General Surfaces

### 116.2.1 Introduction

The section covers the newer ordinary-projective-surface infrastructure. Major restrictions: singularity checks are expensive and are skipped by default (the user sets a parameter to `true`); many intrinsics require the surface to be non-singular or have only ADE singularities; ambient must be ordinary projective. Singularity-check results are cached internally.

### 116.2.2 Creation Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Surface(A, I)` / `Surface(A, f)` / `Surface(A, S)` | Returns the surface in ambient `A` defined by ideal `I`, single polynomial `f`, or sequence `S`. Parameters: `Nonsingular BoolElt` (store known singularity status, default `false`); `Check BoolElt` (verify integrality by primality test, default `true`); `Saturated BoolElt` (assert ideal already saturated, default `false`). | Primality test for integrality check; geometric integrality (stronger) is not tested. |
| `RationalRuledSurface(P, n)` | Returns a rational ruled surface (scroll) in ordinary projective space `P = Pᵐ` with parameters `n, m−1−n`. Also returns the ruling map `X → P¹`. Degenerate (cone) cases `n = 0` or `n = m−1` produce a singular apex. Non-degenerate case is the Hirzebruch surface `Xₑ` (e = |r−s|) mapped via `|C₀ + vf|` **[Har77, Ch. V §2]**. | Explicit determinantal equations of the scroll. |
| `RandomCompleteIntersection(P, ds)` | Random complete intersection surface of multi-degree `ds` in ordinary projective space `P` over a finite field or Q. Parameters: `Nonsingular BoolElt` (default `true`, check smoothness); `RndP RngIntElt` (coefficient bound, default `1`). Returns surface type `Srfc` only if dimension and (optional) smoothness checks pass. | Random coefficient generation; repeat on failure. |
| `KummerSurfaceScheme(C)` | Returns the Kummer surface of the Jacobian of a genus-2 hyperelliptic curve `C`: a quartic in P³ with 16 simple A₁ singularities. Its desingularization is a K3 surface. | Explicit defining equations of the Kummer quartic. |

*Worked examples: H116E1 (Del Pezzo degree-3, K3 quartic, degree-5 Del Pezzo from P² blown up in 4 pts, complete intersection in P⁵, rational ruled surface scroll, Abelian surface from Horrocks-Mumford bundle).*

### 116.2.3 Invariants

The following functions give standard invariants for projective surfaces with only ADE singularities (or Gorenstein condition). Due to current limitations of the cohomology machinery, most apply only to ordinary projective surfaces. Computed values are stored internally to avoid repetition.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `GeometricGenus(S)` | Geometric genus `p_g = h⁰(K)` of an ordinary projective Gorenstein surface `S`. Parameters: `CheckGor BoolElt` (check Gorensteinness, default `false`); `UseCohom BoolElt` (use `H²(Oₛ)` instead of direct global sections of `K`, default `false`). | Default: compute canonical sheaf `K` then its global sections. Alternative: coherent cohomology `H²(Oₛ)`. |
| `Plurigenus(S, n)` | The `n`-th plurigenus: dimension of global sections of `K^⊗n`. Parameter `CheckGor` as above. Returns a value larger than the desingularization plurigenus unless S has at most simple singularities. | Global sections of tensor power of the canonical sheaf. |
| `ArithmeticGenus(S)` | Arithmetic genus of `S`; calls the general scheme intrinsic. | General scheme machinery. |
| `Irregularity(S)` | Irregularity `q = dim H¹(S, Oₛ)`. If `S` is known Gorenstein or geometric genus has been computed, uses formula `q = p_g − p_a`. Parameters: `CheckGor BoolElt` (default `false`); `UseCohom BoolElt` (force cohomology, default `false`). | Cohomology or formula from `p_g` and `p_a`. |
| `ChernNumber(S, n)` | The `n`-th Chern number of the minimal desingularization `S₁` (n = 1 or 2). Requires at most ADE singularities (checked only if `CheckADE := true`). For n = 1: `K·K` on `S` (same as on `S₁` due to ADE condition). For n = 2: via `c₂ + K·K = 12(1 + p_a)`. | Intersection pairing; Noether's formula. |
| `MinimalChernNumber(S, n)` | Chern number of a minimal model `S₂` of the desingularization `S₁`. For non-rational/non-ruled surfaces, the minimal model is unique over the base field. For rational/ruled, takes the geometrically minimal model with maximal `Kₘ·Kₘ` (9 for rational, 8 for non-rational ruled). Requires ADE (checked if `CheckADE := true`). | Known values for each Kodaira dimension class. |
| `HodgeNumber(S, i, j)` | Hodge number `hⁱ·ʲ = dim Hʲ(S₁, Ωⁱ_{S₁})` for the minimal desingularization `S₁`, 0 ≤ i,j ≤ 2. Requires ADE. Computed by formula from `p_g`, `q`, and `c₁²`. | Formulae from the classification; parameter `CheckADE` (default `false`). |

*Worked examples: H116E2 (Kummer surface: verifying K3 invariants p_g = 1, q = 0, c₂ = 24, h^{i,j} matching K3 Hodge diamond).*

### 116.2.4 Singularity Properties

Intrinsics for testing singularity types; results are cached and implications between properties are exploited internally.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsNormal(S)` | Returns true if `S` is a normal variety. Checks: (1) singular subscheme has dimension ≤ 0; (2) local normality test at each singular point (depth of local ring equals 2, tested via saturation). | Serre's criterion (depth ≥ 2 at singular points). |
| `IsSimpleSurfaceSingularity(p)` | Tests whether the surface point `p` is a simple (ADE) singularity of type Aₙ (n ≥ 1), Dₙ (n ≥ 4), E₆, E₇, or E₈, as in **[BHPdV04, Ch. III §7]**. A non-singular point is classed as A₀. Returns type string ("A", "D", or "E") and index. Requires characteristic ≠ 2 (E-type analysis problematic in char. 3). Uses `IsHypersurfaceSingularity` then examines the local equation expansion. | Analytic classification of surface singularities; hypersurface singularity test. |
| `HasOnlySimpleSingularities(S)` | Determines whether all singularities of `S` are isolated and ADE. Parameter `ReturnList BoolElt` (default `false`): if `true`, also returns list of triples `(point, type, index)`. Requires char ≠ 2. | Applies `IsSimpleSurfaceSingularity` at each component of the singular subscheme. |

*Worked examples: H116E3 (degenerate degree-4 Del Pezzo with 2 conjugate A₁ singularities; cone scroll that is normal and Cohen-Macaulay but not Gorenstein, not simple-singularity).*

### 116.2.5 Kodaira-Enriques Classification

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `KodairaEnriquesType(S)` | For `S` an ordinary projective surface with at most ADE singularities. Returns: (1) Kodaira dimension κ ∈ {-1, 0, 1, 2}; (2) a subtype integer (for κ = -1: irregularity q ≥ 0; for κ = 0: -3 = Enriques, -2 = K3, -1 = torus, 2/3/4/6 = bi-elliptic of that type); (3) a descriptive string. Parameter `CheckADE BoolElt` (default `false`). Stores computed invariants (always includes `p_g` and `q`). | Computes the minimum set of invariants needed; sometimes checks the dimension of pluri-canonical maps. |
| `KodairaEnriquesDimension(S)` | Returns only the Kodaira dimension without subtype. Parameter `CheckADE BoolElt` (default `false`). Usually does as much work as `KodairaEnriquesType` unless previously computed. | As above. |

*Worked examples: H116E4 (Veronese surface → "Rational"; Kummer quartic → "K3").*

### 116.2.6 Minimal Models

Birational equivalence classes of surfaces contain infinitely many non-isomorphic projective surfaces linked by blow-ups and blow-downs. A minimal model has no exceptional (-1)-curves; for κ ≥ 0 this minimal model is unique. For rational/ruled surfaces it is not unique. Separate functions handle each Kodaira dimension; they all proceed by iterated adjunction until a termination criterion is reached.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `MinimalModelRationalSurface(S)` | `S` should be non-singular, ordinary projective, rational (κ = -1, q = 0). Non-singularity not checked by default (`CheckSing BoolElt`, default `false`). Returns a rational map `f: S → M` where `M` is a quasi-minimal model: P², Veronese embedding in P⁵, anticanonically embedded Del Pezzo, rational scroll, or conic bundle. | Iterated adjunction map until simple termination criteria are recognized. |
| `MinimalModelRuledSurface(S)` | `S` should be non-singular, ordinary projective, κ = -1 (not necessarily rational). Parameter `CheckSing` (default `false`). For rational `S`, calls `MinimalModelRationalSurface`. For non-rational ruled: adjunction to a scroll, conic bundle over non-rational curve, or non-split minimal ruled surface. Returns rational map `f: S → M`. | Iterated adjunction. |
| `MinimalModelKodairaDimensionZero(S)` | `S` non-singular, ordinary projective, κ = 0. Parameter `CheckSing` (default `false`). Computes minimal model by adjunction until first Chern number equals 0. Returns rational map `f: S → M`. | Iterated adjunction until `c₁ = 0`. |
| `MinimalModelKodairaDimensionOne(S)` | `S` non-singular, ordinary projective, κ = 1. Parameters: `CheckSing` (default `false`); `Fibration BoolElt` (default `false`; if `true`, also returns the elliptic fibration map `g: M → C`). Returns rational map `f: S → M`. Slight speed-up for positive `p_g`: uses an effective canonical divisor at each adjunction step to speed termination test. | Iterated adjunction until `c₁ = 0`; elliptic fibration from small pluri-canonical map on `M`. |
| `MinimalModelGeneralType(S)` | `S` ordinary projective, κ = 2, at most ADE singularities. Parameter `CheckADE` (default `false`). Computes a minimal model `M` as an m-canonical embedding (m = 3 generally; 4 or 5 for small invariants). Returns rational map `f: S → M` and a boolean `is_min`. | m-canonical map (m = 3, 4, or 5) that automatically factors through a non-singular minimal model. |
| `CanonicalWeightedModel(S)` | `S` as for `MinimalModelGeneralType`. Computes the full canonical model as `Proj` of the canonical coordinate ring `⊕ H⁰(S, K^⊗n)`, embedded in weighted projective space. Returns map `f: S → M` (type `MapSch`) and boolean `is_min`. Currently requires `p_g > 0`. | Riemann-Roch spaces of small multiples of an effective canonical divisor. |
| `CanonicalCoordinateIdeal(S)` | `S` as for `MinimalModelGeneralType`. Returns a homogeneous ideal `I` in a weighted polynomial ring `R` such that `R/I` is isomorphic (as a graded ring) to the canonical coordinate ring. | Calls `CanonicalWeightedModel` internally. |

*Worked examples: H116E5 (rational surface in P⁴ with `c₁ = -9` reduced to degree-5 Del Pezzo with `c₁ = 5`); H116E6 (torus blown up at one point in P⁷, unprojected to P⁸ torus); H116E7 (Enriques surface non-minimal in P⁴ reduced to minimal 10-cubic model in P⁵); H116E8 (Horikawa surface already bi-canonical; `CanonicalWeightedModel` gives sextic in P(1,1,1,2)); H116E9 (degree-5 hypersurface blown up in a point, `CanonicalWeightedModel` recovers degree-5 canonical model).*

### 116.2.7 Special Surfaces in Projective 4-space

Intrinsics for generating random surfaces from certain non-general-type families in P⁴, following the constructions of Decker, Ein and Schreyer **[DES93]**. Defining ideals are cokernels of random maps between direct sums of differential sheaf twists, giving Cohen-Macaulay (generically smooth) surfaces.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RandomRationalSurface_d10g9(P)` | Random non-minimal degree-10 rational surface with sectional genus 9 in P⁴ (Ranestad's family). Parameters: `RndP RngIntElt` (coefficient bound, default `2`); `Check BoolElt` (verify irreducibility and smoothness, default `true`). | Random map between DES-type modules **[DES93]**. |
| `RandomEnriquesSurface_d9g6(P)` | Random non-minimal degree-9 Enriques surface with sectional genus 6 in P⁴. Same parameters as above. | **[DES93]** construction. |
| `RandomAbelianSurface_d10g6(P)` | Random degree-10 abelian surface (torus) with sectional genus 6: zero section of a random element of the Horrocks-Mumford bundle. Parameters as above with `RndP` default `5`. | Horrocks-Mumford bundle section. |
| `RandomEllipticFibration_d7g6(P)` | Random minimal degree-7 elliptic surface (κ = 1), sectional genus 6, `p_g = 2`, `q = 0`. Same parameters. | **[DES93]** construction. |
| `RandomEllipticFibration_d8g7(P)` | Random minimal degree-8 elliptic surface (κ = 1), sectional genus 7, `p_g = 2`, `q = 0`. Same parameters. | **[DES93]** construction. |
| `RandomEllipticFibration_d9g7(P)` | Random minimal degree-9 elliptic surface (κ = 1), sectional genus 7, `p_g = 1`, `q = 0`. Same parameters. | **[DES93]** construction. |
| `RandomEllipticFibration_d10g10(P)` | Random non-minimal degree-10 elliptic surface (κ = 1), sectional genus 10, `p_g = 2`, `q = 0`. Same parameters. | **[DES93]** construction. |

---

## 116.3 Surfaces in P³

### 116.3.1 Introduction

This section covers packages for (hyper)surfaces in P³ over number fields (characteristic zero). The core is Tobias Beck's formal desingularization package (Jung method) **[Bec07]** giving algebraic-power-series morphisms from components of any desingularization. Adjoint maps and birational invariants follow from this data **[BS08]**. Rational surface classification uses Schicho's algorithm **[Sch98]**, and parametrization uses **[Sch00]** for scrolls/conic bundles and the Lie algebra method for Del Pezzos.

### 116.3.2 Embedded Formal Desingularization of Curves

Formal embedded desingularization of plane curves used internally by the Jung surface resolution process; also available to the user. Given a curve `C ⊂ A²_E` or `C ⊂ P²_E`, produces the collection of morphisms `Spec(Ô_{Q,p}) → P` (from completions of the desingularization `Q → P`) representing normal crossings, exceptional divisors, and original curve components.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ResolveAffineCurve(p)` | Formal embedded resolution of the affine plane curve defined by `p ∈ E[x,y]` (over a number field) using iterated point blow-ups. Returns three lists: (1) normal crossings (morphisms with two transverse components), (2) exceptional divisors, (3) original curve components, each entry `(homomorphism b, component multiplicities)`. Parameters: `Factors` (precomputed factorization), `Ps` (squarefree part), `Focus RngMPolElt` (restrict to centres vanishing on focus ideal), `ExtName MonStgElt` (name for algebraic extensions, default "alpha"), `ExtCount RngIntElt`, `Verbose`. Last return: updated `ExtCount`. | Jung method: successive blow-ups; algebraic power series for normalisation **[Bec07]**. |
| `ResolveProjectiveCurve(p)` | Same as `ResolveAffineCurve` but for a projective curve defined by homogeneous `p ∈ E[x,y,z]`. The morphisms map from bivariate rings. Parameters: `Focus`, `ExtName`, `ExtCount`, `Verbose`. | As above. |

*Worked examples: H116E10 (resolution of affine cusp-union-node curve; 7 NCs, 4 EXs, 2 DCs over Q); H116E11 (projective version of same curve).*

### 116.3.3 Formal Desingularization of Surfaces

Formal desingularization of a projective or affine hypersurface `S ⊂ P³_E` (or `A³_E`). Produces the set of morphisms `Spec(Ô_{T,pᵢ}) → S` corresponding to curve components of the exceptional divisor of a Jung resolution `T → S`. Underlying algorithm: (1) formal embedded resolution of the ramification locus (discriminant curve) by `ResolveAffineCurve`; (2) normalization of the pullback (giving a surface `T₁` with toric point singularities); (3) resolution of toric singularities by blow-ups. Fully described in **[Bec07]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ResolveAffineMonicSurface(s)` | Formal desingularization of the affine surface defined by monic squarefree `s ∈ E[x,y][z]` over a number field `E`. Returns a list of tuples `((X, Y, Z), o)` where `X, Y, Z ∈ F[[t]]` (over a degree-1 transcendence extension of `E`) satisfy `s(X, Y, Z) = 0`, and `o` is the adjoint order (negation of the order of the special differential form). Parameters: `Focus RngMPolElt` (restrict centre), `ExtName`, `ExtCount`, `Verbose`. | Jung method **[Bec07]**: curve resolution then normalization and toric blow-ups. |
| `ResolveProjectiveSurface(S)` | Principal function. `S` is a projective surface in P³ over a number field, or a defining homogeneous polynomial. Returns list of tuples `((X, Y, Z, W), o)`. Parameter `AdjComp BoolElt` (default `false`; if `true`, returns only morphisms relevant for birational invariants and adjoint spaces). Parameters: `ExtName` (default "gamma"), `ExtCount`, `Verbose`. Second return: updated `ExtCount`. | Jung method **[Bec07]**. |

*Worked examples: H116E12 (affine surface `z² − xy = 0`, 3 morphisms globally vs. 1 with focus at origin); H116E13 (projective surface `w³y²z + (xz + w²)³`, 26 morphisms; examining the first morphism's adjoint order and power series).*

### 116.3.4 Adjoint Systems and Birational Invariants

For a degree-`d` hypersurface `S ⊂ P³_E`, the sheaf of m-adjoints `F_{S,m}` is the subsheaf of `(Ω²_{E(S)/E})^⊗m` whose pullbacks are regular on any desingularization. The n-th graded piece `Γ(S, O_S(n) ⊗ F_{S,m})` is a linear subspace of degree `n + m(d-4)` polynomials in `E[x₀,x₁,x₂,x₃]`, characterised via the adjoint orders of the formal desingularization morphisms. Details in **[BS08]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HomAdjoints(m, n, S)` | For a degree-`d` surface `S ⊂ P³` over a number field and integers `m, n`: a basis for the degree-n piece of the graded module associated to `F_{S,m}` — a subspace of degree `n + m(d-4)` polynomials. Parameter `FormalDesing SeqEnum` (precomputed desingularization, default `0`). | Uses adjoint orders from the formal desingularization morphisms to impose additional linear conditions at singular places **[BS08]**. |
| `GeometricGenusOfDesingularization(S)` | Geometric genus of any desingularization of the hypersurface `S ⊂ P³`: dimension of the (1,0)-adjoint space. Parameter `FormalDesing`. | Dimension of `HomAdjoints(1, 0, S)`. |
| `PlurigenusOfDesingularization(S, m)` | The m-th plurigenus of any desingularization of `S ⊂ P³`: dimension of the (m,0)-adjoint space. Parameter `FormalDesing`. | Dimension of `HomAdjoints(m, 0, S)`. |
| `ArithmeticGenusOfDesingularization(S)` | Arithmetic genus of any desingularization of `S ⊂ P³`: computed from dimensions of (1,1)- and (1,2)-adjoints via Riemann-Roch. Parameter `FormalDesing`. | Riemann-Roch formula for surfaces. |

*Worked examples: H116E14 (computing HomAdjoints(1, n, S) and HomAdjoints(2, n, S) for the degree-8 surface `w³y²z + (xz + w²)³`).*

### 116.3.5 Classification and Parametrization of Rational Surfaces

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsRational(X)` | Returns true if ordinary projective surface `X` is geometrically rational. For `X ⊂ P³` over a number field: uses formal desingularization and Castelnuovo's criterion (arithmetic genus and second plurigenus of any desingularization are zero). For other ambients: assumes at most ADE singularities (parameter `CheckADE BoolElt`, default `false`). Parameter `FormalDesing`. | Castelnuovo's rationality criterion. |

### 116.3.6 Reduction to Special Models

Given a rational hypersurface `S ⊂ P³` over a number field, Schicho's classification **[Sch98]** identifies 10 types {0, 1, 2, 3a, 3b, 4, 5Aa, 5Ab, 5Ac, 5B} (where "0" = non-rational) using adjoint spaces `Vₙ,ₘ = HomAdjoints(m, n, S)`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ClassifyRationalSurface(S)` | For an ordinary projective surface `S ⊂ P³` over a number field: returns (1) the standard surface `Y` of a specific type, (2) list of scheme maps (birational map `S → Y`, and optionally a fibration map if `Y` is a scroll or conic bundle), (3) a type string ("P2", "Quadric surface", "Rational scroll", "Conic bundle", or "Del Pezzo of degree d"). If not rational, returns `(S, [id], "Not rational")`. Parameter `FormalDesing`. | Schicho's algorithm **[Sch98]**; uses adjoint spaces `Vₙ,ₘ` for specific (n, m) pairs determined by the type. |

*Worked examples: H116E15 (8 surfaces classified: non-rational quartic; P²; quadric; rational scroll with fibration; quartic classified as P²; conic bundle with fibration; degree-1 Del Pezzo in P(1,1,2,3); degree-6 Del Pezzo).*

### 116.3.7 Parametrization of Rational Surfaces

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ParametrizeProjectiveHypersurface(X, P2)` | `X ⊂ P³_Q`, `P2` a projective plane over Q. Returns `false` if not rational over Q; otherwise returns `true` and a birational parametrization `P2 → X`. Parameter `FormalDesing`; `Verbose`. | Calls `ClassifyRationalSurface` then special-case routines for each type. |
| `ParametrizeProjectiveSurface(X, P2)` | `X` an ordinary projective surface in `Pⁿ_Q` (n ≥ 2), `P2` over Q. Returns `false` or `true` and a birational parametrization `P2 → X`. First finds a birational projection to a hypersurface in P³, then calls `ParametrizeProjectiveHypersurface`. May be slow if the projection introduces bad singularities. | Birational projection to P³ then `ParametrizeProjectiveHypersurface`. |
| `Solve(p, F)` | `p ∈ Q[x,y,z]`, `F = Q(u,v)`. Finds birational parametrizations of the (not necessarily irreducible) affine hypersurface `p = 0` over Q. Returns a sequence of triples `(X, Y, Z) ∈ F³` with `p(X,Y,Z) = 0`, one per parametrizable irreducible component. | Calls `ParametrizeProjectiveHypersurface` on each irreducible component after homogenization. |

*Worked examples: H116E16 (all 8 surfaces from H116E15 parametrized; p7 = degree-1 Del Pezzo has parametrization with equations of degree ~365 due to the low-degree case difficulty); H116E17 (affine cubic with 3 factors; 2 parametrizable components returned).*

### 116.3.8 Parametrization of Special Surfaces

Special-case parametrization routines for the standard surface types arising from `ClassifyRationalSurface`. These can be called directly.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ParametrizeQuadric(X, P2)` | `X ⊂ P³_Q` an irreducible degree-2 hypersurface; `P2` a projective plane. Returns `false` if not parametrizable over Q; otherwise `true` and a birational parametrization. | Find a rational point on `X` (equivalent to finding an isotropic vector for the quadric form; reduced to two quadrics in three variables using lattice methods) then project. Based on **[Sch98, §3.1]**. |
| `ParametrizePencil(phi, P2)` | `X` a birationally ruled surface over Q with rational pencil `φ: X → Pⁿ` (image a rational normal curve); `P2` a projective plane over Q. Returns `false` if not parametrizable over Q; `true` and a birational parametrization otherwise. Handles rational scrolls and conic bundles. | Algorithm of **[Sch00]** for surfaces with a pencil. |
| `ParametrizeDelPezzo(X, P2)` | `X` an anticanonically embedded Del Pezzo surface (type `Sch` or `Srfc`, degrees 1–9) over Q; `P2` a projective plane over Q. Returns `false` if not parametrizable; `true` and a birational parametrization otherwise. Blows down exceptional lines to degree ≥ 5 then invokes the degree-specific intrinsics in §116.4.3. For degrees 1–4 singular cases, uses direct methods. | Blow-down of exceptional lines to high degree **[Sch98, §3.5, Man86]** then Lie algebra method for degrees 5–9. |

*Worked examples: H116E18 (quadric parametrization: 5 examples over Q, 2 not parametrizable); H116E19 (pencil parametrization of a degree-4 ruled surface).*

---

## 116.4 Del Pezzo Surfaces

### 116.4.1 Introduction

Del Pezzo surfaces in their anticanonical (weighted projective for degrees 1–2) embeddings. The specialised type `SrfDelPezzo` is a subtype of `Srfc`. Routines cover creation, parametrization by degree, minimization/reduction (degrees 3–4), point-counting and isomorphism (degree 3 over finite fields), explicit construction (degree 3, hexahedral), and invariant theory (degree 3: Clebsch-Salmon, covariants, contravariants, pentahedron).

### 116.4.2 Creation of General Del Pezzos

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DelPezzoSurface(P, L)` / `DelPezzoSurface(S)` / `DelPezzoSurface(Z)` | Del Pezzo surface of degree `9 − d` obtained by blowing up the projective plane `P` in the `d` points given as list `L`, set `S`, or length-`d` zero-dimensional scheme `Z`. Embedded by the anticanonical system. Reports error if points not in sufficiently general position. | Anti-canonical embedding of the blowup. |
| `DelPezzoSurface(f)` | Creates the degree-3 Del Pezzo surface from homogeneous cubic `f` in a 4-variable polynomial ring (grevlex order), in the P³ defined by that ring. Reports error if not smooth. | Checks smoothness; constructs as a surface in P³. |
| `IsDelPezzo(Y)` | Returns true if `Y` (in ordinary projective space) is an abstract Del Pezzo. If so, also returns the anticanonical embedding `X` and the map `Y → X`. Computationally expensive in high-dimensional ambients. | Pluri-anticanonical map computation. |

### 116.4.3 Parametrization of Del Pezzo Surfaces

Del Pezzo surfaces of degree d ≥ 3 in their anticanonical embedding as degree-d surfaces in Pᵈ. Over an algebraically closed field they are always rational; over a number field, parametrizability is an arithmetic question. The main methods use the Lie algebra of the automorphism group **[dG06, dGP, HS06, GSHPBS12]**. Degrees 5–8 also cover singular (degenerate) Del Pezzos. Degrees 3–4 singular cases have direct special-case code (from V2.17).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SetVerbose("ParamDP", v)` | Set verbose level for Del Pezzo parametrization (values: 0/false, 1/true, 2). | — |
| `ParametrizeDegree9DelPezzo(X)` | `X` a degree-9 Del Pezzo in P⁹ over Q (defined by 27 quadrics). Returns whether parametrizable over Q and, if so, a parametrization `P2 → X` given by cubic polynomials. | Lie algebra method **[dG06, HS06]**. |
| `ParametrizeDegree8DelPezzo(X)` | `X` a degree-8 Del Pezzo in P⁸ over a number field (defined by 20 quadrics). Handles three types: (1) P² blown up at one rational point (always parametrizable); (2i) product of two Galois twists of P¹ (parametrizable iff both trivial ↔ isomorphic to P¹×P¹); (2ii) Galois twist of P¹×P¹ with transitive Galois action (infinite family parametrized by Q*/Q*², isomorphic to `x₀²−ax₁² = x₂x₃`). Also handles singular (degenerate) degree-8 case. Returns boolean and (if parametrizable) a map by cubic (case 1) or degree-4 polynomials (case 2). | Lie algebra method **[dGP]**; singular case detected from Lie algebra computation. |
| `ParametrizeDegree7DelPezzo(X)` | `X` a (possibly degenerate/singular) degree-7 Del Pezzo in P⁷ over a number field. Always parametrizable; returns parametrization directly via Lie algebra without reduction to higher degree. | Lie algebra method **[dG06, HS06]**. |
| `ParametrizeDegree6DelPezzo(X)` | `X` a non-singular degree-6 Del Pezzo in P⁶ over Q or a number field K (defined by 9 quadrics). The connected component of Aut(X) is a 2-dimensional torus over K; X is parametrizable iff it is the trivial principal homogeneous space of its torus (equiv., X has a K-point; local-global principle holds). Parameter `ExistenceOnly BoolElt` (default `false`; if `true`, only checks local solubility without constructing a parametrization — much faster). Returns boolean and (if parametrizable) a parametrization of minimal degree (3, 4, or 6 depending on torus type). | Lie algebra method **[HS06]**; torus identification; norm equations over degree-3 or degree-6 extensions. |
| `Degree6DelPezzoType2_1(K, pt)` / `Degree6DelPezzoType2_2(K, pt)` / `Degree6DelPezzoType2_3(K, pt)` / `Degree6DelPezzoType3(K, pt)` / `Degree6DelPezzoType4(K, K1, pt)` / `Degree6DelPezzoType6(K, pt)` | Construct the specific degree-6 Del Pezzo surface `X` whose (connected) automorphism group is the torus `T` corresponding to field data `K` (and `K1` for Type4), containing the K-point `pt` in P⁶. The torus types classify the 6 possible 2-dimensional tori over k. Point `pt` must satisfy non-vanishing conditions specific to each type. | Explicit torus-type construction from field extensions **[HS06]**. |
| `ParametrizeDelPezzoDeg6(X)` | Variant of `ParametrizeDegree6DelPezzo` that also handles the degenerate (singular) degree-6 case. Tests for singularity first (can be slow); for singular X, projects from a singular point to P⁵ and parametrizes the resulting rational scroll. For non-singular X, calls `ParametrizeDegree6DelPezzo`. | `ParametrizeDegree6DelPezzo` for smooth case; scroll parametrization via `ParametrizePencil` for singular case. |
| `ParametrizeDegree5DelPezzo(X)` | `X` a (possibly singular) degree-5 Del Pezzo in P⁵ over a number field. Always parametrizable; returns parametrization directly without reduction to higher degree. Uses a geometric projection method (since the automorphism group is finite, the Lie algebra method is not applicable). | Geometric projection method **[GSHPBS12]**. |
| `ParametrizeSingularDegree3DelPezzo(X, P2)` | `X` a degree-3 irreducible hypersurface in P³ with finitely many ADE singularities (not checked). `P2` a projective plane over the same number field. Returns `true` and a birational parametrization with inverse if `X` is parametrizable over the base field; otherwise `false`. If a rational singular point exists, projection from it gives an immediate parametrization. Otherwise, a small number of special root subsystem configurations are handled individually (including adapted Lie algebra method for toric cases). | Projection from rational singular point; or case analysis for conjugate-singularity configurations including toric Lie algebra method. |
| `ParametrizeSingularDegree4DelPezzo(X, P2)` | As above but `X` is an irreducible complete intersection of 2 quadrics in P⁴ with finitely many ADE singularities. If a rational singular point exists, projection maps X to a line or conic bundle in P³ which is then parametrized. Otherwise, special case analysis for D₅/E₆ root subsystem configurations. | Projection from rational singular point to scroll/conic bundle; or special case methods. |

*Worked examples: H116E20 (degree-8 type 2ii anticanonic sphere `x₀² − 2x₁² = x₂x₃` lifted to P⁸, parametrized by degree-4 polynomials); H116E21 (degree-3 Del Pezzo blown down via ideal of 3 disjoint lines to degree-6 Del Pezzo, then parametrized); H116E22 (singular degree-3 Del Pezzo with 4 conjugate A₁ singularities, parametrized by special case code).*

### 116.4.4 Minimization and Reduction of Surfaces

Given a Del Pezzo surface over Z (degree 3 or 4) with large coefficients, minimization finds an isomorphic model with minimal invariants (done locally at each bad prime), and reduction finds a model with small coefficients.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `MinimizeCubicSurface(f, p)` | Local minimization of a cubic surface (homogeneous polynomial over Z) at prime `p`. Returns new equation and transformation matrix. No stability check. Verbose: `MinRedCubSurf` (max 2). | Local minimization algorithm **[Els]**. |
| `ReduceCubicSurface(f)` | Global reduction of a cubic surface (homogeneous polynomial over Z). Returns reduced polynomial and transformation matrix. Verbose: `MinRedCubSurf` (max 2). | Reduction algorithm based on **[Els]**. |
| `MinimizeReduceCubicSurface(f)` | Global minimize-and-reduce of a smooth cubic surface over Z. Returns minimized-reduced polynomial and transformation matrix (applied to `f` gives a scalar multiple). Verbose: `MinRedCubSurf` (max 2). | Algorithm **[Els]**. |
| `MinimizeDeg4delPezzo(f, p)` | Local minimization of a degree-4 Del Pezzo (pair of quadrics over Z) at prime `p`. Returns new pair of quadrics and transformation matrix. Verbose: `MinRedDeg4delPezzo` (max 1). | Local minimization; calls `ReduceQuadrics` for reduction step. |
| `MinimizeReduceDeg4delPezzo(f)` | Global minimize-and-reduce of a degree-4 Del Pezzo (pair of quadrics over Z). Returns reduced pair of quadrics and transformation matrix. Verbose: `MinRedDeg4delPezzo` (max 1). | Global minimization then `ReduceQuadrics`. |
| `MinimizeReduce(S)` | For a Del Pezzo surface `S` of degree 3 or 4: calls the appropriate minimize-reduce routine above and converts the result to a Del Pezzo surface. Returns the new surface and the transformation matrix. Verbose: `MinRedCubSurf`, `MinRedDeg4delPezzo`. | Dispatches to `MinimizeReduceCubicSurface` or `MinimizeReduceDeg4delPezzo`. |

*Worked examples: H116E23 (degree-3 Del Pezzo from 6 rational points in P², ugly model reduced to small-coefficient cubic; degree-4 Del Pezzo from 5 points similarly reduced).*

### 116.4.5 Cubic Surfaces over Finite Fields

Cubic surfaces represented by homogeneous degree-3 polynomials with finite field coefficients.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `NumberOfPointsOnCubicSurface(f)` | For a smooth cubic surface `f` over a finite field: returns the number of points and the Swinnerton-Dyer number (conjugacy class in the Weyl group W(E₆)) of the Frobenius. | Frobenius action on the 27 lines. |
| `IsIsomorphicCubicSurface(f, g)` | For smooth or singular cubic surfaces `f`, `g` over a finite field: returns true if isomorphic, plus a list of transformation matrices (one per isomorphism class over the algebraic closure). Parameter `UseLines BoolElt` (default `false`): if `false`, uses singularities of the Hessian; if `true`, uses the 135 intersection points of the 27 lines (slower, needed when Hessian degenerates). Works only over finite fields due to required large field extensions. | Analysis of a canonical finite set of points (Hessian singularities or line intersections). |

*Worked examples: H116E24 (Fermat-like cubic over large prime p, Frobenius class 13); H116E25 (Clebsch cubic has 120 automorphisms; diagonal cubic has 648; general cubic has 1).*

### 116.4.6 Construction of Cubic Surfaces

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CubicSurfaceByHexahedralCoefficients(pol)` | From a separable degree-6 polynomial `p`, constructs a cubic surface having the roots of `p` as hexahedral coefficients. The resulting surface has a Galois-invariant set of 12 lines. See **[EJ10]**. | Hexahedral construction **[EJ10]**. |
| `CoblesRadicand(p)` | For a separable degree-6 polynomial `p`: evaluates the Cobles quartic at the roots of `p`. The result is the discriminant of the hexahedral cubic surface (up to a square factor). | Cobles quartic evaluation. |

*Worked examples: H116E26 (construct cubic surface from degree-6 polynomial, compute Cobles radicand = -676, minimize-reduce the surface, compute Picard-Galois module and cohomology).*

### 116.4.7 Invariant Theory of Cubic Surfaces

Classical invariant theory of cubic surfaces; for background see **[Hun96, App. B]** and **[Sal58]**.

#### 116.4.7.1 Invariants

By Clebsch's theorem the ring of invariants is generated by invariants of degrees 8, 16, 24, 32, 40 (Salmon's explicit system). Stable cubic surfaces are isomorphic iff they define the same point in P(1,2,3,4,5).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ClebschSalmonInvariants(f)` | Computes the 5 Salmon invariants of cubic surface `f`. Second return: discriminant. | Classical invariant-theory computation **[Sal58]**. |
| `SkewInvariant100(f)` | Computes the degree-100 skew invariant I₁₀₀. Its square lies in the Clebsch ring; it vanishes iff the cubic surface has an Eckardt point. | Classical. |
| `CubicSurfaceFromClebschSalmon(inv)` | Constructs a cubic surface with prescribed Salmon invariants `inv`. Requires the last invariant to be non-zero. | Inverse problem for Clebsch-Salmon invariants. |

*Worked examples: H116E27 (construct surface from invariants [1,2,3,4,5], minimize-reduce, verify recovered invariants agree; no Eckardt points since I₁₀₀ ≠ 0).*

#### 116.4.7.2 Covariants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LinearCovariants(f)` | Salmon's 4 linear covariants of the cubic surface `f`. | Classical **[Sal58]**. |
| `ClassicalCovariantsOfCubicSurface(f)` | The 4 classical covariants of `f`: (1) Hessian, (2) T, (3) Θ, (4) degree-9 surface intersecting `f` in exactly its 27 lines. | Classical. |

#### 116.4.7.3 Contravariants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `NumericClebschTransfer(f, inv, p)` | Given a form `f`, a user-defined invariant `inv` of forms of the same degree in one fewer variable, and a point `p`: evaluates the corresponding contravariant of `f` at `p` via the Clebsch transfer principle. Used for building contravariant polynomials by interpolation. | Clebsch transfer principle (differentiation). |
| `ContravariantsOfCubicSurface(f)` | Computes 3 contravariants of cubic surface `f` (via Clebsch transfer from plane cubic invariants S, T, discriminant): (1) hyperplanes where `f ∩ hyperplane` has j-invariant 0; (2) j-invariant 1728; (3) S²−6T (degree-12 dual surface — hyperplanes where intersection is singular). | Clebsch transfer **[Sal58]**. |

*Worked examples: H116E28 (Cayley cubic, 4 A₁ singularities: dual surface factors with 4 linear factors of multiplicity 2).*

#### 116.4.7.4 Interaction of Covariants and Contravariants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ApplyContravariant(c, d)` | Applies polynomial `d` to polynomial `c` as a differential operator (xᵢ acts as ∂/∂xᵢ). In invariant theory, `d` is a contravariant and `c` is a covariant; the result is a new covariant/contravariant or an invariant. | Differential operator (Clebsch transfer action). |

*Worked examples: H116E29 (recover first Salmon invariant by applying degree-4 contravariant to the Hessian).*

### 116.4.8 The Pentahedron of a Cubic Surface

A general cubic surface can be written as a sum of 5 cubes of linear forms (uniquely up to scaling by cube roots of unity and permutation). The 5 linear forms correspond to 5 points in the dual projective space — the "faces" of the pentahedron. Algorithm described in **[RS00]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PentahedronIdeal(f)` | Computes the ideal of the 5 faces of the pentahedron of the cubic surface `f`. | Algorithm of **[RS00]** (varieties of sums of powers). |

*Worked examples: H116E30 (random cubic with rational pentahedron; diagonal cubic has degenerate pentahedron with 4 pts; degenerate cubic has only 1 pentahedron point).*

---

## 116.5 Bibliography

| Key | Reference |
|-----|-----------|
| **[Bec07]** | Tobias Beck. *Formal Desingularization of Surfaces — The Jung Method Revisited.* Technical Report 2007-31, RICAM, December 2007. URL: http://www.ricam.oeaw.ac.at/publications/reports/ |
| **[Bec08]** | Tobias Beck. *Software Documentation.* Technical Report 2008-8, RICAM, May 2008. URL: http://www.ricam.oeaw.ac.at/publications/reports/ |
| **[BHPdV04]** | W. Barth, K. Hulek, C. Peters, and A. Van de Ven. *Compact Complex Surfaces.* Ergebnisse der Mathematik und ihrer Grenzgebiete 4. Springer, second edition, 2004. |
| **[BS08]** | Tobias Beck and Josef Schicho. *Adjoint Computation for Hypersurfaces Using Formal Desingularizations.* Technical Report 2008-2, RICAM, January 2008. URL: http://www.ricam.oeaw.ac.at/publications/reports/ |
| **[DES93]** | W. Decker, L. Ein, and F.-O. Schreyer. *Construction of surfaces in P⁴.* J. Algebraic Geometry, 2:185–237, 1993. |
| **[dG06]** | W. de Graaf, M. Harrison, J. Pilnikova, and J. Schicho. *A Lie Algebra Method for Rational Parametrization of Severi-Brauer Surfaces.* J. Algebra, 303(2):514–529, 2006. |
| **[dGP]** | W. A. de Graaf and J. Pilnikova. *Parametrizing Del Pezzo surfaces of degree 8 using Lie algebras.* arXiv:math.NT/0512477. |
| **[Eis05]** | David Eisenbud. *The Geometry of Syzygies: A Second Course in Commutative Algebra and Algebraic Geometry.* Graduate Texts in Mathematics 225. Springer, New York–Berlin–Heidelberg, 2005. |
| **[EJ10]** | Andreas-Stephan Elsenhans and Jörg Jahnel. *Cubic surfaces with a Galois invariant double-six.* Cent. Eur. J. Math., 8(4):646–661, 2010. |
| **[Els]** | Andreas-Stephan Elsenhans. *Good models for cubic surfaces.* (To appear). |
| **[GSHPBS12]** | J. Gonzalez-Sanchez, M. Harrison, I. Polo-Blanco, and J. Schicho. *Algorithms for Del Pezzo Surfaces of Degree 5 (Construction, Parametrization).* 2012. |
| **[Har77]** | Robin Hartshorne. *Algebraic Geometry.* GTM 52. Springer, 1977. |
| **[Hor76]** | E. Horikawa. *Algebraic Surfaces of General Type with Small c₁². II.* Invent. Math., 37:121–155, 1976. |
| **[HS06]** | M. C. Harrison and J. Schicho. *Rational Parametrisation for Degree 6 Del Pezzo Surfaces using Lie Algebras.* In Proceedings ISSAC'06, 2006. |
| **[Hun96]** | Bruce Hunt. *The Geometry of Some Special Arithmetic Quotients.* Lecture Notes in Mathematics 1637. Springer-Verlag, Berlin, 1996. |
| **[Man86]** | Yu. I. Manin. *Cubic Forms (2nd ed.).* North-Holland Mathematical Library 4. North-Holland Publishing Co., Amsterdam, 1986. |
| **[RS00]** | Kristian Ranestad and Frank-Olaf Schreyer. *Varieties of sums of powers.* J. Reine Angew. Math., 525:147–181, 2000. |
| **[Sal58]** | George Salmon. *A Treatise on the Analytic Geometry of Three Dimensions.* Revised by R. A. P. Rogers, 7th ed., Vol. 1, ed. by C. H. Rowe. Chelsea Publishing Company, New York, 1958. |
| **[Sch98]** | Josef Schicho. *Rational parametrization of surfaces.* J. Symbolic Comput., 26(1):1–29, 1998. |
| **[Sch00]** | Josef Schicho. *Proper parametrization of surfaces with a rational pencil.* In Proceedings of ISSAC 2000 (St. Andrews), pp. 292–300 (electronic). ACM, New York, 2000. |
| **[SvdV87]** | A. J. Sommese and A. van der Ven. *On the adjunction mapping.* Math. Ann., 278:593–603, 1987. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Surface creation (integrality/smoothness check) | `Surface`, `RandomCompleteIntersection`, `KummerSurfaceScheme`, `RationalRuledSurface` |
| Kodaira-Enriques classification (invariant computation) | `KodairaEnriquesType`, `KodairaEnriquesDimension`, `GeometricGenus`, `Plurigenus`, `Irregularity`, `ChernNumber`, `MinimalChernNumber`, `HodgeNumber` |
| ADE singularity test (analytic classification) | `IsSimpleSurfaceSingularity`, `HasOnlySimpleSingularities`, `IsNormal` |
| Adjunction / minimal model (iterated adjunction maps) | `MinimalModelRationalSurface`, `MinimalModelRuledSurface`, `MinimalModelKodairaDimensionZero`, `MinimalModelKodairaDimensionOne`, `MinimalModelGeneralType` |
| Canonical/weighted model (Riemann-Roch, pluricanonical) | `CanonicalWeightedModel`, `CanonicalCoordinateIdeal` |
| Random P⁴ surfaces **[DES93]** (Beilinson spectral sequence) | `RandomRationalSurface_d10g9`, `RandomEnriquesSurface_d9g6`, `RandomAbelianSurface_d10g6`, `RandomEllipticFibration_d7g6`, `RandomEllipticFibration_d8g7`, `RandomEllipticFibration_d9g7`, `RandomEllipticFibration_d10g10` |
| Formal desingularization / Jung method **[Bec07]** | `ResolveAffineCurve`, `ResolveProjectiveCurve`, `ResolveAffineMonicSurface`, `ResolveProjectiveSurface` |
| Adjoint systems and birational invariants **[BS08]** | `HomAdjoints`, `GeometricGenusOfDesingularization`, `PlurigenusOfDesingularization`, `ArithmeticGenusOfDesingularization` |
| Schicho rational surface classification **[Sch98]** | `ClassifyRationalSurface`, `IsRational` |
| General rational surface parametrization **[Sch98, Sch00]** | `ParametrizeProjectiveHypersurface`, `ParametrizeProjectiveSurface`, `Solve`, `ParametrizeQuadric`, `ParametrizePencil`, `ParametrizeDelPezzo` |
| Lie algebra parametrization of Del Pezzos **[dG06, dGP, HS06, GSHPBS12]** | `ParametrizeDegree5DelPezzo`, `ParametrizeDegree6DelPezzo`, `ParametrizeDelPezzoDeg6`, `ParametrizeDegree7DelPezzo`, `ParametrizeDegree8DelPezzo`, `ParametrizeDegree9DelPezzo` |
| Singular degree-3/4 Del Pezzo parametrization | `ParametrizeSingularDegree3DelPezzo`, `ParametrizeSingularDegree4DelPezzo` |
| Degree-6 Del Pezzo torus construction **[HS06]** | `Degree6DelPezzoType2_1`, `Degree6DelPezzoType2_2`, `Degree6DelPezzoType2_3`, `Degree6DelPezzoType3`, `Degree6DelPezzoType4`, `Degree6DelPezzoType6` |
| Minimization and reduction **[Els]** | `MinimizeCubicSurface`, `ReduceCubicSurface`, `MinimizeReduceCubicSurface`, `MinimizeDeg4delPezzo`, `MinimizeReduceDeg4delPezzo`, `MinimizeReduce` |
| Point-counting and isomorphism over finite fields | `NumberOfPointsOnCubicSurface`, `IsIsomorphicCubicSurface` |
| Hexahedral construction **[EJ10]** | `CubicSurfaceByHexahedralCoefficients`, `CoblesRadicand` |
| Clebsch-Salmon invariant theory **[Sal58, Hun96]** | `ClebschSalmonInvariants`, `SkewInvariant100`, `CubicSurfaceFromClebschSalmon` |
| Covariants and contravariants (classical) | `LinearCovariants`, `ClassicalCovariantsOfCubicSurface`, `ContravariantsOfCubicSurface`, `NumericClebschTransfer`, `ApplyContravariant` |
| Pentahedron of cubic surface **[RS00]** | `PentahedronIdeal` |
