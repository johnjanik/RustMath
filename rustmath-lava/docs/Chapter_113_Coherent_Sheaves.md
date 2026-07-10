# Chapter 113 — Coherent Sheaves

**Handbook part:** XV — Algebraic Geometry
**Handbook pages:** 3603–3631 (PDF pages 3734–3765)

---

## Scope and overview

Chapter 113 describes the Magma functionality for working with coherent sheaves on ordinary projective schemes. The emphasis in this initial version is on invertible sheaves and on computing associated cohomological invariants and explicit divisor maps. Important examples include canonical and anticanonical maps and adjunction maps on varieties of arbitrary dimension. There is also functionality for computing an invertible sheaf corresponding to the class of an effective Cartier divisor given as a closed subscheme, as well as a basis for the Riemann-Roch space of that divisor as ambient rational functions.

The package is based on Magma's functionality for graded modules over polynomial rings and relies heavily on Gröbner basis computations. A coherent sheaf is represented by a graded module over the coordinate ring of the ambient projective space. The key difference between the category of sheaves and the category of modules is that a sheaf is not represented uniquely; however, there is a unique maximal graded module representing it. For certain algorithms — computing cohomology, for example — any module representing the sheaf may be used. For other calculations, such as explicit Riemann-Roch spaces or divisor maps, the full maximal module is often required.

A coherent sheaf S is defined by a graded module M over the polynomial ring R = k[x₀, …, xₙ] and a subscheme X of Pⁿ = Proj(R) on which M is supported (i.e. the defining ideal I ⊆ R of X annihilates M). The sheaf is just the coherent sheaf M̃ on X as described in Prop. 5.11, Section 5, Chapter II of **[Har77]**, with M considered as a graded module over the homogeneous coordinate ring of X. Sheaves are of type `ShfCoh`; homomorphisms between sheaves supported on the same scheme are of type `ShfHom`. The algorithms used in the package are based on computational commutative algebra techniques well-known to experts. A standard reference for the definition and basic properties of coherent sheaves on Noetherian schemes is Section 5, Chapter II of **[Har77]**.

The basic condition for most computations is that the support of the sheaf has irreducible components all of the same non-zero dimension. The computation of the maximal module of a sheaf from its initial defining module is one of the fundamental operations; it may be carried out in the background and its result stored by several functions.

---

## 113.1 Introduction

*(No intrinsics; see scope and overview above.)*

---

## 113.2 Creation Functions

The general creation function for sheaves takes a graded module representing the sheaf and a scheme X on which it is supported. Special constructors are provided for the structure sheaf and canonical sheaf of X (when X is locally Cohen-Macaulay and equidimensional). Serre twists of a given sheaf may also be requested.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Sheaf(M, X)` | Given an ordinary projective scheme X and a module M over the coordinate ring of the ambient of X (annihilated by the defining ideal of X), returns the coherent sheaf defined by graded module M on scheme X. | Direct construction by definition. |
| `StructureSheaf(X)` | Returns the structure sheaf OX for X, defined by the coordinate ring RX as a module. | Direct construction. |
| `StructureSheaf(X, n)` | Returns the Serre twist OX(n), whose associated graded module is RX(n) (see Section 5, Chapter II of **[Har77]**). OX(1) is the sheaf corresponding to the class of a hyperplane divisor H on X. | Direct construction. |
| `CanonicalSheaf(X)` | Returns the canonical sheaf KX for X. X must be equidimensional and locally Cohen-Macaulay (conditions are not checked). For non-singular varieties, KX is the highest alternating power of the sheaf of Kähler differentials. Acts as a dualising sheaf; see Section 7, Chapter III of **[Har77]** and Chapter 21 of **[Eis95]** for module-theoretic background. | Computed from the dual complex to the minimal free resolution of the coordinate ring of X. |
| `CanonicalSheaf(X, n)` | Returns the nth Serre twist KX(n) of the canonical sheaf. For a non-singular variety of dimension d, the map corresponding to KX(d − 1) is the important adjunction map. | As above, then twist. |
| `Twist(S, n)` | Returns the nth Serre twist of S, S(n) ≅ S ⊗_{OX} OX(n). If M is a module giving S, then M(n) gives S(n). | Direct module twist. |
| `SheafOfDifferentials(X)` | Returns the sheaf of 1-differentials Ω¹_{X/k} on X. Computes the natural representing module from the embedding of X in projective space (see Section 8, Chapter II of **[Har77]**). Parameter `Maximize` (BoolElt, default `false`): if true, the maximal module is computed and used. | Module from the embedding; Gröbner-basis-based maximisation if `Maximize := true`. |
| `TangentSheaf(X)` | Returns the sheaf of tangent vectors for X. Computes the natural representing module from the embedding in projective space (see Section 8, Chapter II of **[Har77]**). Parameter `Maximize` (BoolElt, default `false`): if true, the maximal module is computed. Combined with `IsLocallyFree`, provides an alternative method for checking non-singularity on locally Cohen-Macaulay varieties (often faster for high-codimension varieties than the Jacobian method). | Module from embedding; maximisation optional. |
| `HorrocksMumfordBundle(P)` | P must be ordinary projective 4-space P⁴ over a field. Returns the locally free rank-2 sheaf on P representing the Horrocks-Mumford bundle (see **[HM73]**). The vanishing locus of a general global section is a two-dimensional Abelian variety in P⁴. | Construction of the specific bundle **[HM73]**. |

*Worked examples: H113E1 (structure sheaf, canonical sheaf, and Twist on a smooth cubic surface x³ + y³ + z³ + t³ in P³; structure sheaf of an exceptional line using `Sheaf` and `QuotientModule`).*

---

## 113.3 Accessor Functions

The following functions provide a convenient interface to extract the basic data from a coherent sheaf.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Module(S)` | Returns the graded module that was used to define sheaf S. | Direct accessor. |
| `Scheme(S)` | Returns the ordinary projective scheme X on which sheaf S is defined. | Direct accessor. |
| `FullModule(S)` | Computes and returns the maximal module M_max giving sheaf S. The nth graded piece of M_max equals the global sections of S(n) as a finite-dimensional vector space over k; thus M_max ≅ ⊕_{n∈Z} H⁰(X, S(n)) as in **[Har77]**. Assumes the exact support of S has no irreducible components of dimension 0 and no embedded associated primes of dimension 0 (a further assumption — not checked — is equidimensionality). The module M_max is stored so that it is only computed once. | Double-dual computation over a Noether normalisation A of the supporting algebra. M is re-expressed as an A-module, M_max computed over A, then recovered as a module over k[x₀,…,xₙ] by tracking multiplication maps. Gröbner-basis-based. |
| `GlobalSectionSubmodule(S)` | Returns the submodule of M_max generated in degrees ≥ 0, that is ⊕_{n≥0} H⁰(X, S(n)). | Truncation of M_max. |
| `SaturateSheaf(~S)` | Procedure: computes and stores (but does not return) the maximal module M_max of the sheaf S. | As `FullModule`; side-effecting form. |

*Worked examples: H113E2 (structure sheaf of a non-projectively normal projection of a degree-4 rational normal curve; Hilbert series comparison of Module vs. FullModule, showing a dimension-1 discrepancy in the degree-1 graded part).*

---

## 113.4 Basic Constructions

The following functions provide basic constructions on sheaves.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `TensorProduct(S, T)` | Returns the tensor product S ⊗_{OX} T of two sheaves on the same scheme X. Parameter `Maximize` (BoolElt, default `false`): if true, the maximal module of the result is computed and used as the defining module. Note: tensor products of maximal modules can be far from maximal; setting `Maximize := true` is usually advisable. | Module tensor product; Gröbner-basis maximisation when `Maximize := true`. |
| `TensorPower(S, n)` | Returns the nth tensor power of S if n > 0, the (−n)th tensor power of the dual of S if n < 0, and OX if n = 0. Parameter `Maximize` (BoolElt, default `true`). The rank of the module presentation grows rapidly with n, so `Maximize := true` (the default) is strongly recommended. | Iterated module tensor product; maximisation by default. |
| `Dual(S)` | Returns the dual sheaf Hom_{OX}(S, OX). | Module dual via Hom computation. |
| `SheafHoms(S, T)` | Returns the sheaf H = Hom_{OX}(S, T) together with a map taking a homogeneous element of degree d in the module of H to the degree-d sheaf homomorphism it represents. The defining module M_H = Hom(M_max, N_max) is the maximal module of H. | Hom of maximal modules; Gröbner-basis-based. |
| `DirectSum(S, T)` | Returns the sheaf direct sum S ⊕ T of two sheaves on the same scheme X. | Module direct sum. |
| `Restriction(S, Y)` | Returns the restriction of sheaf S (on scheme X) to a subscheme Y of X. Parameter `Check` (BoolElt, default `true`): if true, verifies Y is a subscheme of X. | Module restriction via base change. |

*Worked examples: H113E3 (ruling L on a singular projective quadric cone X in P³; DivisorToSheaf to get OX(L); TensorProduct of OX(L) with itself; FullModule shows OX(2L) ≅ OX(1)).*

---

## 113.5 Sheaf Homomorphisms

A sheaf homomorphism is represented by a module homomorphism between representing modules (defining, maximal, or global section modules) for the two sheaves. "Homogeneous" homomorphisms of degree d uniformly shift the grading by d and are interpreted as sheaf homomorphisms from the domain to the dth Serre twist of the codomain. The type of a sheaf homomorphism is `ShfHom` (NOT a `Map` subtype).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SheafHomomorphism(S, T, h)` | Given sheaves S and T on the same scheme X and a homogeneous module homomorphism h (as returned by `IsHomogeneous`) between a representing module of S and a representing module of T: returns the sheaf homomorphism from S to T(d), where d is the degree of h. | Direct construction from module homomorphism. |
| `Domain(f)` | Returns the domain sheaf of homomorphism f. | Direct accessor. |
| `Codomain(f)` | Returns the codomain sheaf of homomorphism f. | Direct accessor. |
| `Degree(f)` | Returns the degree d of homomorphism f (the uniform grading shift). | Direct accessor. |
| `ModuleHomomorphism(f)` | Returns the underlying homogeneous graded module homomorphism of f. | Direct accessor. |
| `Kernel(f)` | Returns the kernel of f and its inclusion homomorphism into the domain of f. | Module kernel computation; Gröbner-basis-based. |
| `Image(f)` | For f of degree d with domain S and codomain T: returns the image I (a subsheaf of T(d)), the restriction g: S → I (degree d), and the inclusion h: I → T(d) (degree 0). | Module image computation; Gröbner-basis-based. |
| `Cokernel(f)` | Returns the cokernel of f and the quotient homomorphism from the codomain to it. (Here f is treated as a homomorphism from S(d) ← T rather than S ← T(d).) | Module cokernel computation; Gröbner-basis-based. |
| `Expand(hms)` | Given a sequence of sheaf homomorphisms [h1, …, hn], returns the composition h1 ∗ h2 ∗ … ∗ hn. The domain of h2 must equal the codomain of h1 etc., and the underlying module homomorphisms must be composable. | Iterated module homomorphism composition. |

---

## 113.6 Divisor Maps and Riemann-Roch Spaces

One of the main initial aims of the sheaf machinery is to provide a way of computing the rational maps associated to invertible sheaves in reasonable generality (see Section 7, Chapter 2 of **[Har77]**) and similarly for effective Cartier divisors as closed subschemes, in the form of the map or their Riemann-Roch spaces.

The `DivisorMap` function computes the rational map X → Proj(R) → Pʳ, where R is the graded k-subalgebra of ⊕_{n≥0} H⁰(X, S^⊗n) generated by H⁰(X, S) (the space of global sections) and r + 1 = dim H⁰(X, S). The map is only defined on the open subscheme where S is generated by global sections. In most cases the result is a graph map of type `MapSchGrph` (see §112.14.7); when the sheaf was constructed via `DivisorToSheaf` with `GetMax := true`, a traditional `MapSch` is returned. The major computation is the determination of the graph: an ideal defining the graph is written from the relation matrix of a minimal presentation of the global section submodule M₀ of S, then saturated with respect to an appropriate domain variable.

The `DivisorToSheaf`/`RiemannRochBasis` algorithm is based on the following. Choose r > 0 such that I contains a homogeneous polynomial G of degree r not in the ideal of X. Then there is a "complementary" divisor E of X with rH ~ D + E (H a hyperplane divisor) and L(D) ≅ L(−E)(r), where L(−E) is represented by IE/IX. If `GetMax := true`, r is chosen large enough that H¹(IX(m)) vanishes for m ≥ r, guaranteeing a maximal representing module and a full Riemann-Roch basis with denominator G.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DivisorMap(S)` | Given an invertible sheaf S on scheme X (invertibility not checked): returns the rational map from X into the projective space associated to S, and the image of the map on X. Parameter `graphmap` (BoolElt, default `false`): if true, forces return of a `MapSchGrph` even when a `MapSch` would otherwise be returned. | Graph of the map computed from the relation matrix of the minimal presentation of M₀, saturated with respect to a domain variable; Gröbner-basis-based. |
| `DivisorToSheaf(X, I)` | Given an ordinary projective scheme X and an ideal I defining an effective Cartier divisor D on X (of codimension 1, locally principal; conditions not checked): returns the invertible sheaf L(D) (see Section 6, Chapter 2 of **[Har77]**). Parameter `GetMax` (BoolElt, default `true`): if true, the maximal module of L(D) is computed and a basis for the Riemann-Roch space is stored in the attribute `rr_space` of S as a pair (sequence of numerators [G₁,…,Gₙ], denominator G). | `ColonIdeal` and `Saturation` operations; if `GetMax := true`, r is chosen so that H¹(IX(m)) vanishes for m ≥ r. Gröbner-basis-based. |
| `RiemannRochBasis(X, I)` | Carries out the same procedure as `DivisorToSheaf(X, I : GetMax := true)` and additionally returns the Riemann-Roch basis as a sequence of numerators [G₁,…,Gₙ] and the denominator G, along with the sheaf L(D). | As `DivisorToSheaf` with `GetMax := true`; Gröbner-basis-based. |

*Worked examples: H113E4 (degree-3 rational scroll in P⁴; RiemannRochBasis for a ruling line; DivisorMap gives the fibration map to P¹; extension to a maximally defined map; using graphmap parameter). H113E5 (degree-3 Del Pezzo surface; blowing down 3 disjoint lines via DivisorToSheaf and DivisorMap with a twisted sheaf H + L₁₂₃; image is a degree-6 Del Pezzo in P⁶).*

---

## 113.7 Predicates

Tests for several important properties of coherent sheaves. The isomorphism test, combined with `DivisorToSheaf`, can be used to test linear equivalence of Cartier divisors.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsLocallyFree(S)` | Returns true iff S is a locally free sheaf on X of constant rank, and if so, also returns the rank. Parameter `UseFitting` (BoolElt, default `true`): selects the algorithm (see below). | **Default (Fitting ideal method, `UseFitting := true`):** checks that the saturation of the d-th Fitting ideal of M (or M_max) is the full ring and that the (d−1)-th lies in the saturated ideal of X. Can be extremely slow. **Alternative (`UseFitting := false`, étale stratification):** assumes X equidimensional, locally Cohen-Macaulay, and connected (not checked). Uses Serre's criterion **[Ser55]** via a Noether normalisation: checks that all intermediate Ext(M_max, R₀)-modules are finite length (equivalently, equality tests for Hilbert polynomials of cokernels in the dual complex of the free resolution of M_max over R₀). Applied inductively over a chain of closed subschemes of X (the "étale stratification"). The stratification is stored with X for reuse. Note: may fail in small positive characteristic if the generic separability condition fails, potentially giving wrong results. |
| `IsIsomorphic(S, T)` | For S and T coherent sheaves on the same base scheme X: returns true iff S ≅ T, and if so, returns an isomorphism. | Hilbert polynomial check, then Betti number check on M_max and N_max (necessary conditions); then searches for an invertible matrix in the finite-dimensional space of homomorphisms between M_max and N_max. The invertible-matrix-in-a-linear-space problem is difficult in general; current implementation is described as "rather weak". |
| `IsIsomorphicWithTwist(S, T)` | Returns true iff S ≅ T(d) for some Serre twist d, and if so, returns an isomorphism and d. | As `IsIsomorphic`, with the "with twist" case also determining the possible d from Hilbert polynomial and Betti number checks. |
| `IsArithmeticallyCohenMacaulay(S)` | Returns true iff the maximal graded module M_max of S is a Cohen-Macaulay module over the coordinate ring of X. A scheme X is arithmetically Cohen-Macaulay iff its coordinate ring is Cohen-Macaulay, which holds iff the coordinate ring equals the maximal module of OX. | Freeness check if the structure of M_max over a Noether normalisation is already known; otherwise depth calculation from a minimal free resolution of M_max over the coordinate ring of the ambient. |

---

## 113.8 Miscellaneous

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CohomologyDimension(S, r, n)` | Returns the dimension over the base field of the r-th cohomology group of the n-th Serre twist of S: dim Hʳ(X, S(n)). Uses the maximal module if already computed; otherwise uses the defining module. Calling `SaturateSheaf` before cohomology computations is often faster in practice. | Calls the equivalent function for the maximal module (or defining module); local cohomology / Ext computations over the coordinate ring. |
| `DimensionOfGlobalSections(S)` | Returns the same value as `CohomologyDimension(S, 0, 0)` (dimension of the space of global sections of S), computed differently: uses straightforward linear algebra on the zero-th graded part of the maximal module given as a presentation module. Usually faster than `CohomologyDimension`. Also saturates the sheaf as a side effect. | Linear algebra on the presentation matrix of M_max. |
| `IntersectionPairing(S, T)` | For S and T invertible sheaves on a nonsingular surface X, representing divisor classes D and E: returns the surface intersection number D·E. Only minimal checks are made on input validity. | Standard computation using the Hilbert polynomials of S, T, and their tensor product. |
| `ZeroSubscheme(S, s)` | S should be a locally free sheaf on scheme X (local freeness not checked). s should be a homogeneous element of degree d of the defining, maximal, or global section module of S, representing a global section of the twisted sheaf S(d). Returns the vanishing subscheme of s: the largest subscheme of X on which s restricts to the zero section. If S ≅ L(D), then s (if non-zero) represents an effective divisor Ds in the linear system |D + dH| and the vanishing subscheme is Ds as a subscheme of X. | Local computation: for each Zariski-open U over which S(d)|U ≅ OX^n|U, the section s|U corresponds to an n-tuple (f₁,…,fₙ) and the vanishing subscheme on U is defined by ⟨f₁,…,fₙ⟩. |

---

## 113.9 Examples

Extended examples illustrating the sheaf machinery.

*Worked examples:*

- *H113E6: A rational surface X of degree 10 in P⁴ (from the family described in **[DES93]**, §2.1) with sectional genus 9. Computes the adjunction map (KX(1)), image X₁ (degree 13 in P⁸), second adjunction map to X₂ (degree-5 Del Pezzo in P⁵), verifies KX₂ ≅ OX₂(−1) via IsIsomorphicWithTwist, and checks that the composed map X → X₂ is birational using Expand.*

- *H113E7: An elliptic curve C embedded as a degree-8 subvariety of P³ over Q (arising from eight-descents). Shows that the maximal module of OC(1) is isomorphic to the normalisation of the coordinate ring of C; DivisorMap on OC(1) recovers the projectively normal embedding into P⁷ (image defined by 20 quadrics). Illustrates that saturating the sheaf before cohomology computation is faster than working with the defining module directly.*

---

## 113.10 Bibliography

| Key | Reference |
|-----|-----------|
| **[DES93]** | W. Decker, L. Ein, and F.-O. Schreyer. Construction of surfaces in P⁴. *J. Algebraic Geometry*, 2:185–237, 1993. |
| **[Eis95]** | David Eisenbud. *Commutative Algebra with a View Toward Algebraic Geometry*, volume 150 of Graduate Texts in Mathematics. Springer, New York–Berlin–Heidelberg, 1995. |
| **[Har77]** | Robin Hartshorne. *Algebraic Geometry*, GTM 52. Springer, 1977. |
| **[HM73]** | G. Horrocks and D. Mumford. A rank 2 vector bundle on P⁴ with 15,000 symmetries. *Topology*, 12:63–81, 1973. |
| **[Ser55]** | J.-P. Serre. Faisceaux Algébriques Cohérents. *Ann. Maths.*, 61:197–278, 1955. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Gröbner basis + double-dual over Noether normalisation (maximal module) | `FullModule`, `SaturateSheaf`, `GlobalSectionSubmodule` |
| Dual complex to minimal free resolution of coordinate ring | `CanonicalSheaf` |
| Serre's criterion **[Ser55]** + étale stratification | `IsLocallyFree(:UseFitting := false)` |
| Fitting ideal local freeness check | `IsLocallyFree(:UseFitting := true)` |
| Hilbert polynomial + Betti number + invertible-matrix-in-linear-space | `IsIsomorphic`, `IsIsomorphicWithTwist` |
| Depth / freeness check from minimal free resolution | `IsArithmeticallyCohenMacaulay` |
| Graph of divisor map via relation matrix saturation | `DivisorMap` |
| ColonIdeal and Saturation (Riemann-Roch / L(D) construction) | `DivisorToSheaf`, `RiemannRochBasis` |
| Hilbert polynomials of sheaves and tensor product | `IntersectionPairing` |
| Local cohomology / Ext over coordinate ring | `CohomologyDimension` |
| Linear algebra on M_max presentation | `DimensionOfGlobalSections` |
| Horrocks-Mumford bundle construction **[HM73]** | `HorrocksMumfordBundle` |
