# Chapter 128 — Modular Curves

**Handbook part:** XVII — Modular Arithmetic Geometry
**Handbook pages:** 4293–4309 (PDF pages 4424–4443)

---

## Scope and overview

Modular curves in Magma are a special type `CrvMod` of plane curve. A modular curve `X` is
defined in terms of standard affine modular polynomials which are stored in precomputed
databases. The modular curves presently available are defined by bivariate polynomials
relating the *j*-invariant and one of several standard functions on `X_0(N)`. These give
singular models for `X_0(N)` designed for computing isogenies of elliptic curves.

Three model types are available — **"Atkin"**, **"Canonical"**, and **"Classical"** — each
giving affine models defined by the modular polynomial databases of the same name (Section
128.4). The `X_0(N)` parametrize elliptic curves together with the structure of a cyclic
isogeny (equivalently, a cyclic subgroup scheme of the *N*-torsion), and the moduli structure
permits constructing isogenies (via Vélu's formulae) and subgroup schemes from a parametrized
elliptic curve.

The chapter also covers class polynomials (Hilbert and Weber variants, invariants of CM
elliptic curves) and a separate family of intrinsics for constructing modular curves `C` over
**Q** of genus at least 2 that arise as images of `X_1(N)` (or `X_0(N)`) under a morphism
π : X_1(N) → C. These use the methods of González-Jiménez, González, Oyono and others
**[GJG03, BGJGP05, GJO10]**, identifying the Jacobian of `C` with a **Q**-rational modular
abelian subvariety of `J_1(N)` (or `J_0(N)`) and matching the cusp forms against the canonical
relations of `C`. Curves whose Jacobian pulls back into the *new* part are called *new*
modular curves; intrinsics handle the hyperelliptic case and the genus-3 non-hyperelliptic
case, with future plans for higher genus and non-new cases.

---

## 128.1 Introduction

Modular curves in Magma are a special type `CrvMod` of plane curve, defined in terms of
standard affine modular polynomials stored in precomputed databases (bivariate polynomials
relating the *j*-invariant and a standard function on `X_0(N)`), giving singular models for
`X_0(N)` for computing isogenies of elliptic curves. (No intrinsics; introductory.)

---

## 128.2 Creation Functions

Several different models for modular curves are available. The possible model types are
"Atkin", "Canonical", and "Classical", each giving affine models defined by the modular
polynomial databases of the same names (see Section 128.4).

### 128.2.1 Creation of a Modular Curve

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ModularCurve(X,t,N)` | A model of the modular curve `X_0(N)`, in an affine plane specified by `X`. The string `t` must be one of "Atkin", "Canonical", or "Classical", with `N` a level in the corresponding modular curve database. | Defined by the affine modular polynomial from the named database. |
| `ModularCurve(D, N)` | An affine model of the modular curve `X_0(N)` of level `N` from a database `D` of modular curves. | Database lookup. |

### 128.2.2 Creation of Points

Points on modular curves can be created in the same way as points on curves or schemes in
general. In addition there are constructors defined in terms of the moduli structure which
take a parametrized elliptic curve as an argument.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ModuliPoints(X,E)` | Given a modular curve `X = X_0(N)` and an elliptic curve `E`, with compatible base rings, the sequence of points over the base field of `E` corresponding to `E` with additional level structure. | Moduli interpretation of `X_0(N)`. |

*Worked example:* H128E1 (moduli interpretation: `ModuliPoints` of a curve over a finite field, then `SubgroupScheme` of the two parametrized isogenies — prime 17 splits in End(E)).

---

## 128.3 Invariants

The defining data and invariants of the curves are accessed through standard functions.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Level(X)` | The level of the modular curve `X` as an integer. | — |
| `Genus(X)` | The genus of the modular curve `X`. | — |
| `ModelType(X)` | The type of the model for `X`, presently limited to "Atkin", "Canonical", or "Classical". Used to determine the algorithm by which parametrized isogenies are computed. | — |
| `Indices(X)` | A sequence of integers `[N, M, P]` classifying the class of `X`, defining a congruence subgroup `Γ(N,M,P) = { (a b; c d) : c ≡ 0 mod N, b ≡ 0 mod P, a ≡ 1 mod M }` of PSL₂(**Z**), where `M` divides LCM(N,P). For the models presently available this is `[N,1,1]`, where `N` is the level of `X = X_0(N)`. | — |

---

## 128.4 Modular Polynomial Databases

Magma contains several databases of standard defining polynomials for modular curves, used
throughout the system for constructing isogenies of elliptic curves; these define singular
models for `X_0(N)` in terms of standard functions.

The **classical** model for `X_0(N)` is in terms of Φ_N(X,Y) with Φ_N(j(τ), j(Nτ)) = 0, where
j(τ) is the *j*-function; Φ_N is symmetric in `X` and `Y`, and the canonical involution is
(X,Y) ↦ (Y,X). For prime `N`, set s = 12/GCD(N−1,12); using that the Dedekind η-function is
holomorphic without zeros on the upper half plane **H**, the function f(τ) = N^s (η(Nτ)/η(τ))^{2s}
is invariant under Γ_0(N), and together with j(τ) generates the function field. The polynomial
Ψ_p with Ψ_p(f(τ), j(τ)) = 0 is the **canonical** modular polynomial; the Atkin–Lehner
involution sends f(τ) to N^s/f(τ), so Ψ_N(N^s/f(τ), j(Nτ)) = 0 also holds.

The **Atkin** modular polynomials Ξ_N(X,Y) satisfy Ξ_N(f(τ), j(τ)) = 0 for a modular function
f(τ) on `X_0(N)` invariant under the Atkin-Lehner involution (hence well-defined on the
quotient curve `X_0^+(N)`, also denoted `X_0^*(N)`); these are also called *star* modular
polynomials. Their construction is described by Elkies **[Elk98]** and Morain **[Mor95]**; the
Atkin database was provided by A.O.L. Atkin.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AtkinModularPolynomial(N)` | Given a prime `N` represented in the database of Atkin modular curves, the Atkin modular polynomial Ξ_N(X,Y). | Database lookup; construction per **[Elk98, Mor95]**. |
| `CanonicalModularPolynomial(N)` | Given a prime `N` represented in the database of canonical modular curves, the canonical modular polynomial Ψ_N(X,Y). | Database lookup. |
| `ClassicalModularPolynomial(N)` | Given an integer `N` represented in the database of classical modular curves, the defining classical modular polynomial Φ_N(X,Y). | Database lookup. |
| `ModularCurveDatabase(t)` / `ModularCurveDatabase(t,i)` | Given an identifier string `t` (one of "Atkin", "Canonical", "Classical"), the corresponding database object of modular curves `X_0(N)`. The Atkin database is split into objects of levels 200(i−1)+1 ≤ N < 200i for `i` in 1,…,5 (only the first two provided by default; more available from the Magma website). The canonical database contains a subset of curves for prime levels below 200; the classical database contains only levels below 60. | Database access. |
| `N in D` | Returns `true` if and only if `N` is a level represented in the modular curve database `D`. | — |
| `ExistsModularCurveDatabase(t)` / `ExistsModularCurveDatabase(t,i)` | Returns `true` if and only if the data file given by the string `t` and integer `i` exists. | — |

*Worked example:* H128E2 (comparing Atkin, Canonical, and Classical defining polynomials for `X_0(3)`, `X_0(11)`, `X_0(13)`; coefficient-size behaviour by `N` mod 12 / mod 13).

---

## 128.5 Parametrized Structures

The modular curves `X_0(N)` parametrize elliptic curves together with a cyclic isogeny
(equivalently a cyclic subgroup scheme of the *N*-torsion). Over the modular curve,
singularities of the chosen surface may obstruct construction of the corresponding isogeny.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Isogeny(E,P)` | Given an elliptic curve `E` and a point `P` on some `X_0(N)` corresponding to a cyclic level structure on `E`, returns an isogeny f : E → F corresponding to `P`. Defined so the pull-back of the invariant differential of `F` is the invariant differential on `E`. | **Vélu's formulae**. |
| `SubgroupScheme(E,P)` | Given an elliptic curve `E` and a point `P` on some `X_0(N)` corresponding to a cyclic level structure on `E`, returns the subgroup scheme of `E` parametrized by `X_0(N)`. | Moduli interpretation. |

*Worked example:* H128E3 (function field of `X_0(7)` for the canonical and Atkin models, `EllipticCurveFromjInvariant`, `ModuliPoints`, `SubgroupScheme`).

---

## 128.6 Associated Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `FunctionField(X)` | The function field of the modular curve `X`. | — |
| `jFunction(X)` | Given a modular curve `X` over a field, the *j*-invariant as a function on the curve. | — |
| `BaseCurve(X)` | Given one of the standard models `X` for `X_0(N)`, the base model curve `X(1)` and the morphism π : X_0(N) → X(1). | — |

*Worked example:* H128E4 (`BaseCurve` of `X_0(17)`; class number 1 discriminants with 17 split, moduli points mapped down to `X(1)` via `HilbertClassPolynomial` and `EllipticCurveFromjInvariant`).

---

## 128.7 Automorphisms

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CanonicalInvolution(X)` / `AtkinLehnerInvolution(X,N)` | Given a projective modular curve `X = X_0(N)`, the Atkin-Lehner involution of the modular curve as a map of schemes. Currently the only Atkin-Lehner involution returned is that for `N` equal to the level of `X`. | Defined via the canonical / Atkin-Lehner involution of the model. |

---

## 128.8 Class Polynomials

Class polynomials are invariants of elliptic curves with complex multiplication by an imaginary
quadratic order of discriminant `D`. The Hilbert class polynomials can be interpreted as
defining a subscheme or divisor on the modular curve `X(1) ≅ **P**¹`, while the Weber variants
define a subscheme of a modular curve of higher level.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HilbertClassPolynomial(D)` | Given a negative discriminant `D`, the Hilbert class polynomial, defined as the minimal polynomial of j(τ), where **Z**[τ] is an imaginary quadratic order of discriminant `D`. | Minimal polynomial of the *j*-invariant. |
| `WeberClassPolynomial(D)` | Given a negative discriminant `D` not congruent to 5 modulo 8, the Weber class polynomial, the minimal polynomial of f(τ) where f is a particular normalized Weber function generating the same class field as j(τ). The f used depends on `D` mod 8 (and `D/4` mod 8 when even) and on whether 3 divides `D`. A root f(τ) is an algebraic integer (a unit outside of 2) generating the ring class field; the *j*-root is recovered by j(τ) = F(f(τ)) with `F` a returned rational function A(Bxʳ + C)³/xʳ, r|24, A,B,C rational integers (powers of ±2). | Weber functions; cf. Yui–Zagier **[YZ97]** (odd `D`) and Schertz **[Sch76]** (even `D`). |
| `WeberToHilbertClassPolynomial(f,D)` | Given a negative discriminant `D` and the corresponding Weber class polynomial `f`, the Hilbert class polynomial for `D`. Parameter `Al` (`MonStgElt`, default "Roots"): "Roots" computes complex approximations to the roots of the Hilbert polynomial from approximations to the roots of the Weber polynomial via the rational function `F`; the other method is algebraic, using resultants and `F`. | Complex-approximation (default) or resultant-based recovery via `F`. |

*Worked example:* H128E5 (`HilbertClassPolynomial(-71)` vs `WeberClassPolynomial(-71)`, showing the much smaller Weber coefficients and unit constant term).

---

## 128.9 Modular Curves and Quotients (Canonical Embeddings)

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ModularCurveQuotient(N,A)` | Given a level `N` and a (possibly empty) sequence of integers `A` representing Atkin-Lehner involutions, computes a model for a quotient of `X_0(N)` by `A`. Returns `P¹` for a genus 0 curve, an elliptic or hyperelliptic curve, or the canonical embedding in `P^{g−1}` (coordinates = cusp forms invariant under the specified involutions). Parameters: `Raw` (`BoolElt`, default `false`; when not `true`, an initial "semi-reduction" is performed), `Reduce` (`BoolElt`, default `false`; if `true`, a complete LLL-reduction of the equations — impractical for genus greater than ~50). | Canonical embedding via invariant cusp forms; optional LLL reduction. |

*Worked example:* H128E6 (`X0NQuotient(13·29, [13,29])` — model of `X_0(13·29)` quotiented by `w_13, w_29`, defined by cubics in `P⁴`, genus 5).

---

## 128.10 Modular Curves of Given Level and Genus

The intrinsics in this section construct curves `C` over **Q** of genus at least 2 which are
images of `X_1(N)` (or `X_0(N)`) under a morphism π : X_1(N) → C defined over **Q** (modular
curves). The code uses ideas and methods from **[GJG03, BGJGP05, GJO10]**.

The morphism π induces an isogeny π\* from Jac(C) onto a **Q**-rational abelian subvariety `B`
of `J_1(N)` (or `J_0(N)`), i.e. a **Q**-rational modular abelian variety of dimension `g`, the
genus of `C`. The holomorphic differentials of Jac(C) (identified with those of `C`) pull back
under π\* to those of `B`, identified with weight-2 cusp forms associated to `B`. A
**Q**-basis f₁,…,f_g satisfies the canonical relations for a canonical embedding of `C` in the
non-hyperelliptic case. Curves `C` whose pullback π\*(Jac(C)) lies in the *new* part of
`J_1(N)` (or `J_0(N)`) are *new* modular curves. Criteria for the pullback giving precisely the
forms of a particular modular abelian variety `B` are given in **[GJG03, BGJGP05]** (provided
`B` lies in the new part) for the hyperelliptic case, and in **[GJO10]** for the genus-3
non-hyperelliptic case. Intrinsics determine all new modular hyperelliptic curves or all new
modular genus-3 non-hyperelliptic curves of `X_1(N)` / `X_0(N)` for a given level, and test
whether a given modular abelian subvariety `B` corresponds to such a curve, giving its
equations in the affirmative case.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SetVerbose("ModularCurve", v)` | Set the printing level for verbose output for the intrinsics in this section. Legal values for `v`: `true`, `false`, 0, 1, 2 (`false` = 0, `true` = 1). | — |
| `NewModularHyperellipticCurves(N, g)` | For level `N`, a list of all hyperelliptic curves of genus `g ≥ 2` which are new modular curves for `X_1(N)` if `gamma` equals 1 (default) or `X_0(N)` if `gamma` equals 0. Each curve is returned as a univariate polynomial f(x) with y² = f(x) a Weierstrass equation. Parameters: `check` (`BoolElt`, default `false`; if `true`, forms are computed to a precision slightly in excess of `prec₁` needed to guarantee that polynomial relations vanish if they vanish to precision `prec₁`), `prec` (`RngIntElt`, default 100; precision to which modular forms are expanded), `gamma` (`RngIntElt`, default 1). | Methods of **[GJG03, BGJGP05, GJO10]**: match cusp-form q-expansions of the new modular abelian variety against hyperelliptic relations. |
| `NewModularHyperellipticCurve(B)` | Given a sequence `B` of distinct modular abelian subvarieties of `J_1(N)^{new}` (presented as subspaces of modular symbols), returns `true` if the direct sum `M` corresponds exactly to a hyperelliptic curve `C` (M isogenous to Jac(C)), and if so a univariate f(x) with y² = f(x). Parameters: `check` (`BoolElt`, default `false`), `prec` (`RngIntElt`, default 100), `gamma` (`RngIntElt`, default 1; relevant only if `check` is `true`, set to 0 if all `B` are abelian subvarieties of `J_0(N)` for sharper q-expansion bounds). | As above **[GJG03, BGJGP05]**. |
| `NewModularHyperellipticCurve(F)` | Variant of the intrinsic directly above where, instead of a sequence of modular abelian subvarieties, the sequence of q-expansions of the basis of weight-2 forms for the modular abelian subvariety `M` is given. | As above. |
| `ModularHyperellipticCurve(B)` | Given a sequence `B` of distinct modular abelian subvarieties of `J_1(N)^{new}`, where the direct sum `M` need not lie in the new part, determines whether the basis of differentials of `M` satisfies the correct relations to arise from a hyperelliptic curve `C` of genus `g`; if so returns f(x) with y² = f(x). Unlike the new case, Jac(C) is not guaranteed isogenous to `M`. Parameter: `prec` (`RngIntElt`, default 100). | Relation-matching on differentials of `M`. |
| `ModularHyperellipticCurve(F)` | Variant of the intrinsic immediately above where a sequence of q-expansions of the basis of weight-2 cusp forms of `M` is given instead of the sequence of abelian varieties. | As above. |
| `NewModularNonHyperellipticCurvesGenus3(N)` | Given an integer `N`, a list of all non-hyperelliptic curves of genus 3 which are new modular curves, for `X_1(N)` if `gamma` equals 1 (default) or `X_0(N)` if `gamma` equals 0. Parameters: `check` (`BoolElt`, default `false`), `prec` (`RngIntElt`, default 100), `gamma` (`RngIntElt`, default 1). | Method of **[GJO10]**: canonical genus-3 relations on the new modular forms. |
| `NewModularNonHyperellipticCurveGenus3(B)` | Given a sequence `B` of distinct modular abelian subvarieties of `J_1(N)^{new}` whose direct sum `M` need not lie in the new part, determines whether the basis of differentials of `M` satisfies the correct relations to arise from a non-hyperelliptic curve `C` of genus 3; if so returns a defining polynomial for the canonical image of `C`. Parameters: `check` (`BoolElt`, default `false`), `prec` (`RngIntElt`, default 100), `gamma` (`RngIntElt`, default 1; relevant only if `check` is `true`, set to 0 if `M` is an abelian subvariety of `J_0(N)`). Similar to `NewModularHyperellipticCurve(B)`. | As above **[GJO10]**. |
| `NewModularNonHyperellipticCurveGenus3(F)` | Variant of `NewModularNonHyperellipticCurveGenus3(B)` where a sequence `F` of q-expansions of the basis of weight-2 cusp forms of `M` is given instead of the sequence of abelian subvarieties `B`. | As above. |
| `ModularNonHyperellipticCurveGenus3(F)` | Same as `NewModularNonHyperellipticCurveGenus3(F)` except it is not required that the modular abelian variety `M` corresponding to `F` lies in the new part `J_1(N)^{new}`. Used to search for non-new non-hyperelliptic genus-3 curves. | As above. |

*Worked examples:* H128E7 (`X_1(13)` hyperelliptic from a newform of nebentype an order-6 character; `NewModularHyperellipticCurves(80,0)` and `(80)`); H128E8 (`ModularHyperellipticCurve` for an intermediate curve `X_Δ(21)`, example from **[JK07]**); H128E9 (new modular non-hyperelliptic genus-3 curve from a subvariety of `J_0(97)`); H128E10 (`NewModularNonHyperellipticCurvesGenus3(20)`, and a non-new genus-3 curve from `J_0(178)`, denoted `C^{89A}_{178C}` in **[GJO10]**).

---

## 128.11 Bibliography

| Key | Reference |
|-----|-----------|
| **[BGJGP05]** | M. Baker, E. González-Jiménez, J. González, and B. Poonen. *Finiteness results for modular curves of genus at least 2.* Amer. J. Math., **127**:1325–1387, 2005. |
| **[Elk98]** | N. Elkies. *Elliptic and modular curves over finite fields and related computational issues.* In *Computational Perspectives on Number Theory: A conference in honor of A.O.L. Atkin*, 1998. |
| **[GJG03]** | E. González-Jiménez and J. González. *Modular curves of genus 2.* Math. Comp., **72**:397–418, 2003. |
| **[GJO10]** | E. González-Jiménez and Roger Oyono. *Non-hyperelliptic modular curves of genus 3.* J. Number Th., **130**:862–878, 2010. |
| **[JK07]** | D. Jeon and C. H. Kim. *On the arithmetic of certain modular curves.* Acta Arith., **130**:181–193, 2007. |
| **[Mor95]** | F. Morain. *Calcul du nombre de points sur une courbe elliptique dans un corps fini: aspects algorithmiques.* J. Théorie des Nombres de Bordeaux, **7**:255–282, 1995. |
| **[Sch76]** | R. Schertz. *Die singulären Werte der Weberschen Funktionen f, f₁, f₂, γ₂, γ₃.* J. reine angew. Math., **286/287**:46–74, 1976. |
| **[YZ97]** | N. Yui and D. Zagier. *On the singular values of Weber modular functions.* Mathematics of Computation, **66**(220):1645–1662, 1997. |

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Modular polynomial databases (Atkin / Canonical / Classical) **[Elk98, Mor95]** | `AtkinModularPolynomial`, `CanonicalModularPolynomial`, `ClassicalModularPolynomial`, `ModularCurve`, `ModularCurveDatabase` |
| Moduli interpretation of `X_0(N)` | `ModuliPoints`, `BaseCurve`, `jFunction`, `Indices` |
| Vélu's formulae (isogenies / subgroup schemes) | `Isogeny`, `SubgroupScheme` |
| Atkin-Lehner / canonical involutions and quotients | `CanonicalInvolution`, `AtkinLehnerInvolution`, `ModularCurveQuotient` |
| Hilbert / Weber class polynomials (CM invariants) **[YZ97, Sch76]** | `HilbertClassPolynomial`, `WeberClassPolynomial`, `WeberToHilbertClassPolynomial` |
| Modular curves from abelian subvarieties (cusp-form relation matching) **[GJG03, BGJGP05, GJO10]** | `NewModularHyperellipticCurves`, `NewModularHyperellipticCurve`, `ModularHyperellipticCurve`, `NewModularNonHyperellipticCurvesGenus3`, `NewModularNonHyperellipticCurveGenus3`, `ModularNonHyperellipticCurveGenus3` |
