# Chapter 129 — Small Modular Curves

**Handbook part:** XVII — Modular Arithmetic Geometry
**Handbook pages:** 4313–4334 (PDF pages 4444–4467)

---

## Scope and overview

The **Small Modular Curve** database provides simple models over the rational numbers for the
X₀(N) modular curves, together with the data needed to work with them arithmetically:
standard automorphisms, cusps and non-cuspidal rational points (in the positive-genus cases),
the various projection maps X₀(N) → X₀(M) when M divides N, and expressions for standard
functions (j(z), j(Nz)) and forms (E₄, E₆) in terms of rational functions or
k-differentials on the model. The models are called "small" because they are of low degree,
defined by reasonably sparse polynomials with small integer coefficients, and are
non-singular or have only a few simple singularities.

Because the Magma type of a model may be `CrvEll`, `CrvHyp`, `CrvPln` or just a general
`Crv`, and a type cannot extend different subtypes of `Crv`, no special `CrvMod` type was
introduced. The modular-curve component of most intrinsics' arguments is therefore a
projective curve `CN`, which should be a base change of the database model over **Q** to a
field of characteristic zero, plus the level N. For efficiency there is no initial check that
the curve really is a base change of the database model (obtainable via `SmallModularCurve`);
an incorrect curve will almost certainly cause a runtime error. The intrinsics generally
require the modular curve to be defined over a field of characteristic zero, although the
models all reduce nicely for primes not dividing the level.

**Model conventions.** Genus 0 cases use the projective line **P¹** (cusp ∞ at the point at
infinity); genus 1 cases use a minimal Weierstrass `CrvEll`; hyperelliptic (subhyperelliptic)
cases use a minimal Weierstrass model with two rational points at infinity. All other
(non-subhyperelliptic) cases use a plane model in **P²** or a non-singular model in **P³**:
canonical models for genus 3 and 4 (plane quartic / quadric∩cubic), and smallest-degree
singular birational plane (degree 6) models for genus 5 and 6. The initial database covers
all subhyperelliptic X₀(N) and all other cases of genus ≤ 6, i.e. all N < 60 plus about half
of 60 ≤ N ≤ 80 and N = 81, 121. Models were found from Dedekind eta products, weight-2
integral forms, theta series and Eisenstein series, with non-subhyperelliptic models obtained
by LLL-reduction of canonical images (cf. `ModularCurveQuotient`); see **[Hara]** for the
genus 5 and 6 degree-6 plane image construction.

---

## 129.1 Introduction

Introductory section; see the scope and overview above. No intrinsics.

---

## 129.2 Small Modular Curve Models

The models are projective models of the complete curve X₀(N) over **Q**, following the
conventions described above (see **[Ogg74]** for the hyperelliptic cases and **[Lig75]** for
the elliptic cases).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SmallModularCurve(N)` | The model for X₀(N) over the rationals, retrieved from the small modular curve database. A runtime error results if there is no database entry for level N. | Database lookup. |
| `SmallModularCurve(N, K)` | The base change of the level-N database model to `K`, which should be a characteristic-zero field. | Database lookup + base change. |
| `IsInSmallModularCurveDatabase(N)` | Whether there is data for level N in the small modular curves database. | Database lookup. |

*Worked example:* H129E1 (`IsInSmallModularCurveDatabase` for N = 79, 35; `SmallModularCurve(35)` hyperelliptic model, `SmallModularCurve(63)` plane model).

---

## 129.3 Projection Maps

The database contains information allowing reconstruction of the standard projection maps
between the level-N and level-M models for any M dividing N.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ProjectionMap(CN, N, CM, M)` | For `CN`, `CM` base changes to the same characteristic-zero field K of the database curves of levels N and M with M\|N: the natural projection map CN → CM corresponding to z ↦ z on the upper-half-plane quotient models, i.e. (E, C) ↦ (E, (N/M)C) in the moduli interpretation (non-cuspidal points ↔ isomorphism classes of an elliptic curve E with a cyclic subgroup C of order N). | Reconstruction from stored projection data. |
| `ProjectionMap(CN, N, CM, M, r)` | As above with `r` a positive integer divisor of N/M: the projection CN → CM corresponding to z ↦ rz, i.e. (E, C) ↦ (E/(N/r)C, (N/(Mr))C/(N/r)C) in the moduli interpretation. | Reconstruction from stored projection data. |

*Worked example:* H129E2 (3-projection and 7-projection of `SmallModularCurve(63)` down to levels 21 and 3).

---

## 129.4 Automorphisms

For X₀(N) there is a finite group of automorphisms B₀(N) coming from matrices acting on the
complex upper half-plane: B₀(N) ≅ Nm_{SL₂(R)}(Γ₀(N))/Γ₀(N), the normaliser of Γ₀(N) in
SL₂(R) modulo Γ₀(N). It is generated by Atkin–Lehner involutions and, if 4 or 9 divides N, a
transformation z ↦ z + (1/r) for some 1 < r \| 24 (see **[AL70]**, **[Bar08]** for the
correct structure). These are written Sᵣ, and w_d for the d-th Atkin–Lehner involution
(d\|N, (d, N/d) = 1).

By a result of Kenku–Momose **[KM88]**, completed by Elkies for N = 63 **[Elk90]**, B₀(N)
gives the *full* group of algebraic automorphisms A₀(N) of X₀(N) (in characteristic zero)
when the genus is at least 2, except for N = 37, 63 and 108 (cf. **[Harb]**), where B₀(N) has
index two in A₀(N). The w_d and Sᵣ isomorphisms are precomputed and stored. The
automorphism-group intrinsics return B₀(N) when g(N) ≤ 1 and A₀(N) when g(N) ≥ 2, as a
`GrpAutCrv`, built up from the known semidirect-product structure (one factor C₂ per prime
p > 3 dividing N, with more complex p-components for p = 2, 3). Except in the N = 108 case,
all automorphisms are defined over the cyclotomic field **Q**(μᵣ) where r is the largest
divisor of 24 with r²\|N (the w_d over **Q**, Sᵣ over **Q**(μᵣ)).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AtkinLehnerInvolution(CN, N, d)` | The Atkin–Lehner involution w_d as a scheme automorphism of `CN` (d a divisor of N with (d, N/d) = 1). If the full automorphism group G has already been computed and cached with `CN`, returns w_d as a `GrpAutCrvElt` in G. Terminates immediately when retrieving equations from the database (first case). | Retrieval of stored isomorphism. |
| `SrAutomorphism(CN, N, r, u)` | Sᵣ as a scheme automorphism of `CN` (over a characteristic-zero field K), corresponding to z ↦ z + (1/r) where `u` is a primitive r-th root of unity in K (r a divisor of 24 with r²\|N) mapping to e^{2πi/r}. If the cached full group G exists, returns Sᵣ as a `GrpAutCrvElt`. | Retrieval of stored isomorphism (and possibly a composition). |
| `ExtraAutomorphism(CN, N, u)` | For N = 37, 63 or 108: an automorphism in A₀(N) not in B₀(N) (`u` a primitive r-th root of unity, r = 1 for N = 37, r = 3 otherwise). This generates A₀(N) over B₀(N); it is the hyperelliptic involution for N = 37 and 108, and has order 4 with square w₉ for N = 63. Returned as a `GrpAutCrvElt` if the full group is cached. | Retrieval of stored extra isomorphism. |
| `AutomorphismGroupOverQ(CN, N)` | The full group of automorphisms of `CN` (B₀(N) if g(X₀(N)) ≤ 1) defined over **Q**, as a `GrpAutCrv`. Parameter: `Install` (default `true`) — install and cache the result as the (full) automorphism group of `CN`. | Builds the group from the known semidirect-product structure. |
| `AutomorphismGroupOverCyclotomicExtension(CN, N, n)` | The full automorphism group of `CN` over the cyclotomic field K obtained by adjoining the n-th roots of unity (`CN.1` should be a primitive n-th root of unity). Parameter: `Install` (default `true`). | Builds the group from the known structure. |
| `AutomorphismGroupOverExtension(CN, N, n, u)` | The full automorphism group of `CN` defined over the cyclotomic field **Q**(μₙ), with K an extension of **Q** containing the n-th roots of unity and `u` a primitive n-th root of unity in K. Parameter: `Install` (default `true`). | Builds the group from the known structure. |

*Worked example:* H129E3 (`AtkinLehnerInvolution` / `ExtraAutomorphism` for N = 37; N = 48: `AutomorphismGroupOverQ` of order 16 ≅ D₈ × C₂, `AutomorphismGroupOverCyclotomicExtension` of order 48, `SrAutomorphism` over **Q**(i)).

---

## 129.5 Cusps and Rational Points

Writing H* for the extended upper half-plane, X₀(N)(**C**) ≅ H*/Γ₀(N) and the cusps are the
images of **Q** ∪ ∞. A complete set of cusp representatives is given by points a/d, d running
over positive divisors of N and, for each d, a running over integer representatives of
(**Z**/(d, N/d)**Z**)ˣ coprime to d. With respect to the rational structure, the a/d for a
given d\|N form a set of Galois-conjugate points, each with field of definition
**Q**(μ_{(d,N/d)}); if (d, N/d) ≤ 2 the cusp 1/d is **Q**-rational (so ∞ ∼ 1/N and 0 ∼ 1/1
are always rational cusps). The cuspidal points are stored in the database (a rational point
for each cusp 1/d when (d, N/d) ≤ 2, otherwise a cluster defining the **Q**-conjugate set
when (d, N/d) > 2). The *place* over each conjugate class of cusps a/d is unique. Non-cuspidal
rational points (classes of elliptic curves over **Q** with a cyclic N-isogeny over **Q** up
to twist, as determined by Mazur and others) are also stored; there are very few on
genus > 0 curves and they are never singular on the chosen models.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Cusp(CN, N, d)` | For `CN` a characteristic-zero base change of the database curve and `d` a positive divisor of N: the point of `CN` corresponding to the cusp 1/d if (d, N/d) ≤ 2, or the reduced zero-dimensional subscheme (cluster) consisting of the φ((d, N/d)) cusps a/d if (d, N/d) > 2 (φ Euler's totient). | Retrieval of stored cuspidal data. |
| `CuspIsSingular(N, d)` | Whether the points lying under the cusps a/d for d\|N on the database model for X₀(N) are singular. For a given d, they are either all singular or all non-singular. | Database lookup. |
| `CuspPlaces(CN, N, d)` | The sequence of places of `CN` corresponding to the φ((d, N/d)) cusps a/d. If (d, N/d) ≤ 2 or K = **Q** there is only one place; over a proper extension of **Q** the **Q**-conjugate cusps may split into several Galois orbits. | Retrieval of stored place data. |
| `NonCuspidalQRationalPoints(CN, N)` | For `CN` a base change of the database curve of level N with genus > 0: the sequence of points of `CN` corresponding to non-cuspidal points in X₀(N)(**Q**). Non-empty for only a small number of N; the points are non-singular on all models. | Retrieval of stored non-cuspidal rational points. |

*Worked example:* H129E4 (cusps of N = 32 — point and degree-2 cluster — and N = 63: singular cusps, `CuspPlaces` distinguishing places over a node, clusters for d = 3, 21).

---

## 129.6 Standard Functions and Forms

Returns the j-invariant as a rational function and normalised Eisenstein forms as meromorphic
k-differentials on the database models of X₀(N). Standard variants are obtained by pulling
back via Atkin–Lehner involutions or via the projection / r-projection maps. The database
only stores precomputed expressions for E₂^{(N)}, and E₄, E₆ for prime levels N, and
reconstructs everything else from these using projection maps.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `jInvariant(CN, N)` / `jFunction(CN, N)` | The j-invariant j(z) as a rational function on `CN`. `jFunction` returns it as an element of the function field of `CN`; `jInvariant` returns it in the field of fractions of the coordinate ring of the ambient of `CN`. | Reconstruction from stored data via projection maps. |
| `jInvariant(p, N)` | The value of j(z) at a non-cuspidal point or place `p` on a base change `CN`: the value lies in L if p is a point in CN(L), or in the residue class field of p if p is a place. A point should be non-singular; if the j-function is defined at p the value is still returned. | Evaluation of the j rational function at p. |
| `jNInvariant(p, N)` | As `jInvariant` but the value of the rational function j(Nz) at the point or place — equivalently j evaluated at the image of p under the Fricke involution w_N. | Evaluation at p. |
| `E2NForm(CN, N)` | E₂^{(N)}(z) = N·E₂(Nz) − E₂(z), the weight-2 integral form for Γ₀(N) (E₂(z) = 1 − 24e^{2πiz} + … the normalised weight-2 Eisenstein series), returned as a meromorphic differential (defined over **Q**) in the function field of `CN`. | Retrieval of stored E₂^{(N)} expression. |
| `E4Form(CN, N)` | A rational function f and a differential ω in the function field of `CN` such that E₄(z) = 1 + 240e^{2πiz} + …, as a meromorphic 2-differential on `CN`, is given by f·ω². | Reconstruction from stored E₄ / projection maps. |
| `E6Form(CN, N)` | A rational function f and a differential ω such that E₆(z) = 1 − 504e^{2πiz} + …, as a meromorphic 3-differential on `CN`, is given by f·ω³. The same ω is returned by `E4Form`. | Reconstruction from stored E₆ / projection maps. |

---

## 129.7 Parametrized Structures

Functionality to explicitly compute a cyclic N-isogeny or cyclic subgroup of order N on an
elliptic curve represented by a non-cuspidal point or place on the X₀(N) model, in the usual
moduli-space interpretation (points represent equivalence classes (E, C), or cyclic
N-isogenies φ : E → F, up to isomorphism). The user may pass in a chosen base curve E for the
point's j-invariant when j(p) ≠ 0, 1728.

**Algorithm.** Using a suitable differential d on X₀(N), one forms p₁ = (N/12)(E₂^{(N)}/d)(p),
E₄ = (E₄/d²)(p), E₆ = (E₆/d³)(p), and twisted quantities Ẽ₄, Ẽ₆ from the Fricke involution.
The point is represented by (E, C) with E : y² = x³ − (E₄/48)x − (E₆/864); the monic
polynomial defining C is computed from p₁, E₄, E₆, Ẽ₄, Ẽ₆ by the same algorithm as in the
SEA Elkies variant of Schoof's algorithm. The isogeny φ : E → F with kernel C is then computed
using Vélu's formulae.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SubgroupScheme(p, N)` | For a non-cuspidal point or place `p` on `CN`: an elliptic curve E₁ and a subgroup scheme G of E₁, both over the field of definition K of p, such that G is a cyclic subgroup of order N on E₁ and (E₁, G) represents p. (A point should be non-singular.) | SEA-Elkies-style kernel-polynomial computation. |
| `SubgroupScheme(p, N, E)` | As above, with `E` an elliptic curve over a subfield of K with j-invariant j(p) (j(p) ≠ 0, 1728); E₁ is then E or its base change to K, and only G is returned. | As above. |
| `Isogeny(p, N)` | A cyclic N-isogeny of elliptic curves φ : E₁ → F₁, all over the field of definition K of `p`, representing p. (A point should be non-singular.) | Vélu's formulae on the computed kernel. |
| `Isogeny(p, N, E)` | As above, with `E` over a subfield of K with j-invariant j(p) (≠ 0, 1728); E₁ is E or its base change to K. | Vélu's formulae. |

*Worked example:* H129E5 (N = 14: non-cuspidal rational points, `jInvariant`/`jNInvariant`, `SubgroupScheme` and `Isogeny` with kernel polynomial of degree 7).

---

## 129.8 Modular Generators and q-Expansions

Although Magma's modular-symbols machinery was sometimes used when deriving the models,
explicit expressions for the modular functions t, x, y and the forms fᵢ were worked out in
terms of certain basic types, giving the models independence from the modular-symbol machinery
and allowing faster reconstruction of q-expansions of the generating functions/forms. The
basic types are: (i) Dedekind eta products η(d₁z)^{r₁}…η(dₙz)^{rₙ}; (ii) theta series of
binary quadratic forms and of quaternion algebras; (iii) weight-2 Eisenstein series of prime
level p, pE₂(pz) − E₂(z); (iv) weight-2 cusp forms of elliptic curves (`ModularForm(E)`).
Three standard operations are also used: the derivative D(f) = (1/2πi)(df/dz); the d-shift
f(dz); and the quadratic twist f ⊗ χ.

In the genus-0 cases the database model is **P¹** with uniformising parameter t = x/y (a
modular function on X₀(N), normalised with a simple zero at cusp 0 and a simple pole at cusp ∞,
q-expansion q⁻¹ + … for N > 1; for N = 1 the parameter is j). In the elliptic and
hyperelliptic cases the coordinate functions x, y of the minimal Weierstrass model are modular
functions with poles at the cusp ∞. In the non-subhyperelliptic cases the model is the image
of X₀(N) under z ↦ [f₁(z) : … : f_r(z)] for linearly-independent weight-2 cusp forms fᵢ.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `qExpansionExpressions(N)` | A procedure that prints a mini-program giving expressions for the modular functions corresponding to t (genus 0), to x, y (subhyperelliptic), or to the weight-2 cusp forms f₁,…,f_r (other cases) on X₀(N), in terms of forms/functions of basic type. The mini-program is a sequence of assignment statements v = ⟨expr⟩ built from terms — integers, q = e^{2πiz}, previously-assigned variables, eta products `e{⟨d r⟩…}`, binary-form theta functions `th0/th1/th2/th3{A B C}`, quaternion theta functions `thQ{N}`, Eisenstein series `E{p}`, elliptic-curve cusp forms `f{a₁ a₂ a₃ a₄ a₆}` — possibly modified by operators D, [d] (d-shift), ⟨d⟩ (twist). The final line is `[t,1]`, `[x,y,1]`, or `[f₁,…,f_r]`. | Prints the stored symbolic construction recipe. |
| `qExpansionsOfGenerators(N, R, r)` | q-expansions to precision `r` as Laurent series in the Laurent series ring `R` over **Q**, for the sequence of modular forms/functions of level N defining the model: `[t]` (genus 0), `[x,y]` (elliptic/hyperelliptic), or `[f₁,…,f_r]` (otherwise). t, x, y have negative power terms (poles at cusp ∞); the fᵢ are power series. | Computes q-expansions from the `qExpansionExpressions` recipe via fast basic-type routines. |

*Worked examples:* H129E6 (`qExpansionExpressions` mini-programs for genus 0, 1, hyperelliptic, genus 3, 4, 5 levels: N = 8, 15, 30, 64, 53, 63); H129E7 head (`qExpansionsOfGenerators` for genus-1 level 49).

---

## 129.9 Extended Example

A combined example using small-modular-curve functionality with Magma's curve-quotient and
elliptic-curve machinery to compute the j-invariant of a special class of elliptic curves over
**Q** up to quadratic twist (curves arising in Wiles's proof of the STW conjecture). No new
intrinsics are introduced.

*Worked example:* H129E7 (Breuil–Conrad–Diamond–Taylor–Wiles case: `SmallModularCurve(45)`,
`AtkinLehnerInvolution`/`AutomorphismGroup`/`CurveQuotient`, `Cusp`/`Representative`/`Support`,
`NonCuspidalQRationalPoints`, `ProjectionMap` 3-projection to X₀(5), `jInvariant` of images,
`ThreeTorsionType` = Dihedral, yielding j = (11/2)³ with conductor-338 minimal twist).

---

## 129.10 Bibliography (canonical references)

| Key | Reference |
|-----|-----------|
| **[AL70]** | A. O. L. Atkin and J. Lehner. *Hecke operators on Γ₀(N).* Math. Annalen **185**:134–160, 1970. |
| **[Bar08]** | F. Bars. *The group structure of the normaliser of Γ₀(N) after Atkin-Lehner.* Communications in Algebra **36**:2160–2170, 2008. |
| **[Elk90]** | N. Elkies. *The automorphism group of the modular curve X₀(63).* Compositio Mathematica **74**:203–208, 1990. |
| **[Hara]** | M. C. Harrison. *Explicit solution by radicals, gonal maps and plane models of algebraic curves of genus 5 or 6.* Preprint, arXiv:1103.4946v3 [math.AG]. |
| **[Harb]** | M. C. Harrison. *A new automorphism of X₀(108).* Preprint, arXiv:1108.5595v2 [math.NT]. |
| **[KM88]** | M. A. Kenku and F. Momose. *Automorphism groups of the modular curves X₀(N).* Compositio Mathematica **65**:51–80, 1988. |
| **[Lig75]** | G. Ligozat. *Courbes modulaires de genre 1.* Bull. Soc. Math. France (Suppl.), Mémoire 43, 1975. |
| **[Ogg74]** | A. Ogg. *Hyperelliptic modular curves.* Bull. Soc. Math. France **228**:449–462, 1974. |

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Database model lookup (small models of X₀(N)) **[Ogg74, Lig75, Hara]** | `SmallModularCurve`, `IsInSmallModularCurveDatabase` |
| Stored projection-map reconstruction | `ProjectionMap` |
| Automorphism-group construction from B₀(N)/A₀(N) structure **[AL70, KM88, Elk90, Bar08, Harb]** | `AtkinLehnerInvolution`, `SrAutomorphism`, `ExtraAutomorphism`, `AutomorphismGroupOverQ`, `AutomorphismGroupOverCyclotomicExtension`, `AutomorphismGroupOverExtension` |
| Stored cusp / rational-point data | `Cusp`, `CuspIsSingular`, `CuspPlaces`, `NonCuspidalQRationalPoints` |
| j-invariant and Eisenstein-form reconstruction (E₂^{(N)}, E₄, E₆ + projection maps) | `jInvariant`, `jFunction`, `jNInvariant`, `E2NForm`, `E4Form`, `E6Form` |
| SEA-Elkies kernel polynomial + Vélu's formulae | `SubgroupScheme`, `Isogeny` |
| Eta/theta/Eisenstein basic-type q-expansion recipes | `qExpansionExpressions`, `qExpansionsOfGenerators` |
