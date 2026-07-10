# Chapter 114 — Algebraic Curves

**Handbook part:** XV — Algebraic Geometry
**Handbook pages:** 3639–3739 (PDF pages 3766–3873)

---

## Scope and overview

Chapter 114 describes the core functionality for constructing and studying algebraic curves in Magma. Curves are realised as a specialised type of scheme (see Chapter 112): the general type is `Crv` and plane curves have sub-type `CrvPln`. A curve is any one-dimensional scheme — it may be defined over any ring, though most functions require at least a domain and often a field, and most advanced functions additionally require the curve to be integral (reduced and irreducible).

The chapter is structured around five main themes:

1. **Local geometry** — points, singularity analysis, tangent lines and cones, resolution of singularities via blowups, log canonical thresholds, and local intersection numbers (using Fulton's algorithm **[Ful69]** and the Euclidean-algorithm method of Hilmar–Smyth **[HS10]**).
2. **Global geometry and maps** — genus, projective closure, affine patches, automorphism groups, quotient curves, morphism-induced pullbacks and pushforwards.
3. **Function fields and differentials** — the function field of a curve (type `FunFldFracSch`), together with a background algebraic function field (Chapter 42) used for deeper computation; Kähler differentials, the Cartier operator, holomorphic differentials.
4. **Divisors and Riemann–Roch** — places (valuations of the function field), the divisor group, principal divisors, linear equivalence, the class group (for curves over finite fields), Riemann–Roch spaces, canonical maps, and index calculus for discrete logarithms in the degree-0 class group **[Die06]**.
5. **Special algorithms** — random curve generation **[ST02]**, ordinary plane curves and adjoint linear systems, automorphism groups, curve quotients, regular models of arithmetic surfaces, minimization and reduction of plane quartics **[Sto11]**, and gonal maps for genus ≤ 6 curves using Lie algebra methods **[SS]** and the algorithms of Harrison **[Hara]**.

Functions for conics, elliptic curves and hyperelliptic curves are covered in Chapters 119, 120 and 122 respectively; Chapter 152 covers algebraic-geometric codes built on top of the divisor machinery here.

---

## 114.1 First Examples

This introductory section gives worked examples illustrating the main objects: ambient spaces, curve creation, projective closure, affine patches, points, coordinate changes, function fields, and divisors. No new intrinsics are introduced beyond those catalogued in later sections.

*Worked examples: H114E1 (points in affine 3-space over GF(2) and an extension); overview examples in §§114.1.1–114.1.6 showing ambient creation, singularity checking, projective closure, point creation, tangent lines, intersection numbers, function fields, and the class group of an elliptic curve.*

### 114.1.1 Ambients

Brief tour of ambient space creation. See §114.2 for the full catalogue.

### 114.1.2 Curves

Overview of curve creation from polynomials or ideals. See §114.3 for the full catalogue.

### 114.1.3 Projective Closure

Overview of `ProjectiveClosure` and `AffinePatch`. See §114.5.2 for the full catalogue.

### 114.1.4 Points

Overview of point creation via the coercion operator `!` and the role of point sets. See §114.4.1 for the full catalogue.

### 114.1.5 Choosing Coordinates

Illustrates `Translation`, `Automorphism`, `TangentCone`, and `Blowup` for local analysis.

### 114.1.6 Function Fields and Divisors

Introduces `FunctionField`, `DivisorGroup`, `Divisor`, `Decomposition`, `IsPrincipal`, and `ClassGroup`.

---

## 114.2 Ambient Spaces

Creation of ambient spaces for curves.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AffineSpace(k,n)` | Affine n-dimensional space over ring `k`. | — |
| `AffinePlane(k)` | Affine 2-dimensional space over `k`. | — |
| `ProjectiveSpace(k,n)` | Projective n-dimensional space over `k`. | — |
| `ProjectivePlane(k)` | Projective 2-dimensional space over `k`. | — |
| `DirectProduct(A,B)` | Product `A×B` of two 1-dimensional projective spaces; also returns the two projection maps. | — |
| `RuledSurface(k,n)` | Rational ruled surface over `k` with a curve of self-intersection `−n`; four variables with the first two giving the structure map to P¹. | — |
| `RuledSurface(k,a,b)` | Rational ruled surface over `k` with self-intersection `±(a−b)`. Requires `a,b ≥ 0`. | — |
| `CoordinateRing(A)` | Multivariate polynomial ring over the base ring of the ambient `A` (n variables for affine n-space, n+1 for ordinary or weighted projective n-space, 4 for a ruled surface). | — |
| `FunctionField(A)` | Field isomorphic to the field of fractions of the coordinate ring of `A`. Generators can be named with diamond-bracket notation. | — |
| `A ! [a,...]` / `A(L) ! [a,...]` | Create a point in the base-ring (resp. extension ring `L`) point set of the ambient `A`. | — |
| `Origin(A)` | The point `(0,0,…,0)` of the affine space `A`. | — |
| `Coordinates(p)` | Sequence of coordinates of the point `p`. | — |
| `p[i]` | The i-th coordinate of `p`. | — |

*Worked example: H114E1 (points in affine 3-space over GF(2) and an extension field).*

---

## 114.3 Algebraic Curves

### 114.3.1 Creation

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Curve(A,f)` | Plane curve `f = 0` in ambient plane `A`. `f` must be in the coordinate ring of `A`. Parameters: `Nonsingular`, `Reduced`, `Irreducible`, `GeometricallyIrreducible`, `Saturated` (all `BoolElt`, default `false`). | — |
| `Curve(A,I)` | Curve in ambient space `A` defined by ideal `I` of the coordinate ring of `A`. Same parameters as above. Error if not 1-dimensional. | — |
| `Curve(X,S)` | Curve in the ambient of scheme `X` defined by sequence `S` of polynomials. Special case: affine/projective 1-space with empty `S` gives the affine or projective line. Same parameters. | — |
| `IsCurve(X)` | Returns `true` iff `X` is a one-dimensional scheme. | — |
| `Curve(X)` | The smallest scheme in the inclusion chain above `X` that is a curve. If `X` is a subscheme of a curve, returns that curve. | — |
| `Line(C,p,q)` | The line through distinct points `p`, `q` on curve `C`. If the points coincide (and are smooth), returns the tangent line. | — |
| `Line(P,S)` | The line through the collinear points in set `S` in projective space `P`. | — |
| `Conic(P,S)` | The unique conic through the points of set `S` (typically 5 points in general position) in projective plane `P`. Returns a specialised type if nonsingular (see Chapter 119). | — |
| `Union(C,D)` | Union of curves `C` and `D`. Usually non-irreducible; most advanced functions will not apply. | — |

### 114.3.2 Base Change

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BaseChange(C, K)` | Base change of `C` to new base ring `K` (requires automatic coercion from the current base ring). The result lies in a newly created ambient. | — |
| `BaseChange(C, m)` | Base change of `C` by an explicit ring map `m`. Result lies in a newly created ambient. | — |
| `BaseChange(C, A)` | Base change of `C` to a curve in new ambient space `A` (same ambient type; base ring must admit coercion). | — |
| `BaseChange(C, A, m)` | Base change of `C` to ambient `A` with explicit map `m` between base rings. | — |
| `BaseChange(C, n)` | Base change of `C` to the degree-`n` extension of the finite base field of `C`. | — |

*Worked example: H114E2 (singular curve over Q with singularities defined only over Q(i); illustrates `HasSingularPointsOverExtension` and `BaseChange`).*

### 114.3.3 Basic Attributes

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AmbientSpace(C)` | The ambient space of `C`. | — |
| `BaseRing(C)` / `CoefficientRing(C)` / `BaseField(C)` | The base ring of `C` (error for `BaseField` if not a field). | — |
| `DefiningPolynomial(C)` | The defining polynomial of the plane curve `C`. | — |
| `DefiningIdeal(C)` | The defining ideal of `C` in the coordinate ring of its ambient. | — |
| `CoordinateRing(C)` | The coordinate ring of `C` (requires Gröbner basis internally). | Gröbner basis |
| `Degree(C)` | Degree of `C` in an ordinary projective ambient. | — |
| `JacobianIdeal(C)` | Ideal of partial derivatives of the defining polynomials of `C`. | — |
| `JacobianMatrix(C)` | Matrix of partial derivatives of the defining polynomials. | — |
| `HessianMatrix(C)` | Symmetric matrix of second partial derivatives of the defining polynomial of the plane curve `C`. | — |

*Worked example: H114E3 (projective Weierstrass cubic; `DefiningIdeal`, `IsPrincipal`, `HessianMatrix`, `IntersectionPoints` for flex computation).*

### 114.3.4 Basic Invariants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsReduced(C)` | `true` iff the defining ideal of `C` is reduced. | — |
| `IsIrreducible(C)` | `true` iff `C` is irreducible as a scheme. | — |
| `IsSingular(C)` | `true` iff `C` has at least one singularity over the algebraic closure of its base field. | — |
| `IsNonsingular(C)` | `true` iff `C` has no singularities over the algebraic closure. | — |

### 114.3.5 Random Curves

Implementations follow **[ST02]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RandomNodalCurve(d, g, P)` | Random plane curve of degree `d` and genus `g` in projective plane `P` with only nodes as singularities. Genus must satisfy `1 + d(d−6)/3 ≤ g ≤ (d−1)(d−2)/2`. Parameter: `RandomBound` (coefficient range for Q; default 9). Base field: finite or Q. | **[ST02]** |
| `IsNodalCurve(C)` | `true` iff the plane curve `C` is nonsingular or has only nodes as singularities. | — |
| `RandomOrdinaryPlaneCurve(d, S, P)` | Random plane curve of degree `d` with ordinary singularities specified by sequence `S = [s2,s3,…]` (sᵢ ordinary singularities of multiplicity i). Parameters: `Adjoint` (compute adjoint ideal, default `true`), `Proof` (full check, default `true`), `RandomBound` (default 9). Returns curve and optionally the adjoint ideal. | **[ST02]** |
| `RandomCurveByGenus(g, K)` | Random projective curve over field `K` of genus `g`, for `0 ≤ g ≤ 13`. For `g ≤ 10`, returns a nodal plane curve (`RandomNodalCurve` with `d = g + 2 − [g/3]`); for `11 ≤ g ≤ 13`, returns a curve in P³ via syzygy computations. `K` must be finite or Q. Parameter: `RandomBound`. | **[ST02]** |

*Worked example: H114E4 (random genus 4 curve over Q, genus 8 over GF(23), genus 12 in P³).*

### 114.3.6 Ordinary Plane Curves

A plane curve is *ordinary* if every singularity of multiplicity `m ≥ 2` has `m` distinct tangent directions over the algebraic closure (i.e. the leading term at the singularity is a product of distinct linear factors). All singularities of an ordinary curve are resolved by a single blowup. The adjoint ideal can be computed directly.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HasOnlyOrdinarySingularities(C)` | `true` iff plane curve `C` has only ordinary singularities; also returns the maximum multiplicity (1 = nonsingular) and, if `Adjoint := true` (default), the saturated adjoint ideal. | — |
| `HasOnlyOrdinarySingularitiesMonteCarlo(C)` | Monte Carlo test for ordinariness of a curve over Q: tests 5 prime reductions. Returns `true` if all pass (very likely ordinary), `false` if any fails (definitely not ordinary). Does not compute the adjoint ideal. | Probabilistic reduction mod p |
| `AdjointIdeal(C)` | Saturated adjoint ideal of the ordinary plane curve `C`. Error if not ordinary. | — |
| `AdjointIdealForNodalCurve(C)` | Adjoint ideal for a nodal curve `C` (slightly faster than the general function). | — |
| `AdjointLinearSystemForNodalCurve(C, d)` | Degree-`d` adjoint linear system for a nodal curve `C`. | — |
| `AdjointLinearSystemFromIdeal(I, d)` | Degree-`d` adjoint linear system from a pre-computed saturated adjoint ideal `I`. | — |
| `CanonicalLinearSystemFromIdeal(I, d)` | Canonical linear system for a degree-`d` plane curve with adjoint ideal `I` (same as `AdjointLinearSystemFromIdeal(I, d−3)`). Empty if genus 0. | — |
| `CanonicalLinearSystem(C)` | Canonical linear system of the plane curve `C`. If ordinary, uses the adjoint ideal; otherwise blows up all singularities. | Adjoint ideal (ordinary) or full resolution |
| `AdjointLinearSystem(C)` | General adjoint linear system of the plane curve `C`. | Adjoint ideal (ordinary) or full resolution |
| `Adjoints(C,d)` | Degree-`d` adjoint linear system of the plane curve `C`. | Adjoint ideal (ordinary) or full resolution |

*Worked example: H114E5 (random ordinary degree-7 plane curve with 3 nodes and one ordinary quadruple point; adjoint ideal for the canonical map; `AdjointLinearSystemFromIdeal`, `CanonicalImage` in P⁵).*

---

## 114.4 Local Geometry

### 114.4.1 Creation of Points on Curves

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `C ! [a,...]` | Create the point `(a,…)` with coordinates in the base ring of `C`; parent is `C(k)`. | — |
| `C(L) ! [a,...]` | Create the point `(a,…)` with coordinates in extension ring `L`; parent is `C(L)`. | — |
| `Curve(p)` | The smallest scheme in the inclusion chain above the scheme on which `p` lies that is a curve. | — |
| `Curve(P)` | The curve for which `P` is a point set. | — |
| `Coordinates(p)` | Sequence of ring elements giving the coordinates of `p`. | — |
| `p[i]` / `Coordinate(p,i)` | The i-th coordinate of `p`. | — |
| `p eq q` | `true` iff `p` and `q` lie in schemes in a common ambient, have comparable coordinates, and these are equal. | — |
| `FormalPoint(P)` | Given a nonsingular point `P ∈ C(K)`, returns a point in `C(LaurentSeriesRing(K))` specialising to `P` at 0. | — |

### 114.4.2 Operations at a Point

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `p in C` / `S in C` | `true` iff point `p` or coordinate sequence `S` satisfies the equations of `C`. | — |
| `IsNonsingular(C,p)` | `true` iff `p` is a nonsingular point of `C`. | — |
| `IsSingular(C,p)` | `true` iff `p` is a singular point of `C`. | — |
| `IsInflectionPoint(C,p)` / `IsFlex(C,p)` | `true` iff `p` is a flex of the plane curve `C`; second return is the flex order (local intersection number with the tangent line). Error if `p` is singular. | — |
| `TangentLine(p)` / `TangentLine(C,p)` | Tangent line to plane curve `C` at nonsingular point `p`, embedded as a curve in the same space. | — |
| `TangentCone(C,p)` | Tangent cone to `C` at `p`, embedded in the same ambient. | — |
| `IsTangent(C,D,p)` | `true` iff plane curves `C` and `D` are both nonsingular and tangent at `p`. | — |

### 114.4.3 Singularity Analysis

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Multiplicity(C,p)` | Multiplicity of `C` at the singular point `p`. | — |
| `IsDoublePoint(C,p)` | `true` iff `p` is a double point of `C`. | — |
| `IsOrdinarySingularity(C,p)` | `true` iff `p` is a singular point of `C` with reduced tangent cone (all tangent directions distinct). | — |
| `IsNode(C,p)` | `true` iff `p` is an ordinary double point (node) of `C`. | — |
| `IsCusp(C,p)` | `true` iff `p` is a non-ordinary double point (cusp) of `C`. | — |
| `IsAnalyticallyIrreducible(C,p)` | `true` iff the plane curve `C` has exactly one place at `p` (equivalently, the resolution map is injective above `p`). | — |

*Worked example: H114E6 (cusp `x²−y³` vs. node `x²−y³−y²` at the origin; `IsCusp`, `IsDoublePoint`, `IsReduced(TangentCone(…))`, `IsAnalyticallyIrreducible`, `IsNode`).*

### 114.4.4 Resolution of Singularities

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Blowup(C)` | Blowup of the affine plane curve `C` at the origin; returns the two affine plane curves (birational transforms) on the standard patches. Error if `C` does not contain the origin. | Standard blowup |
| `Blowup(C,M)` | Weighted blowup of `C` at the origin defined by 2×2 integer matrix `M` with `det(M) = ±1`; returns the birational transform inside the original ambient. | Weighted blowup |

*Worked example: H114E7 (resolving `y²−x⁷` by a weighted blowup with matrix M; `ResolutionGraph` showing a chain of 5 exceptional curves; confirming with `Places`).*

### 114.4.5 Log Canonical Thresholds

Let `V` be a variety with at worst log canonical singularities, `D` an effective Q-Cartier divisor. The log canonical threshold (lct) of `(V, D)` at point `P` is `lct_P(V,D) = sup{λ ∈ Q | (V,λD) is log canonical at P}`. Background: **[Kollár 1997, Singularities of pairs]**, **[Kollár 1998, Birational geometry]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LogCanonicalThreshold(C)` | Log canonical threshold of the curve `C`, computed at its singular k-points. | Resolution of singularities, discrepancy computation |
| `LogCanonicalThresholdAtOrigin(C)` | Local log canonical threshold of the affine curve `C` at the origin. | — |
| `LogCanonicalThreshold(C, P)` | Local log canonical threshold of `C` at the point `P`. | — |
| `LogCanonicalThresholdOverExtension(C)` | Log canonical threshold of `C` at all singular points, including those defined over field extensions. | — |

*Worked examples: H114E8 (cubic curves of various types; `IsNodalCurve`, `IsCusp`, `IsOrdinarySingularity`, `IsReduced`; computing lct for each); H114E9 (curve over Q with singularities over a splitting field; comparing `LogCanonicalThreshold` vs. `LogCanonicalThresholdOverExtension`).*

### 114.4.6 Local Intersection Theory

The main single-point function uses a standard Euclidean algorithm from Fulton **[Ful69]**. The multi-point variant uses the algorithm of Hilmar–Smyth **[HS10]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsIntersection(C,D,p)` | `true` iff `p` lies on both plane curves `C` and `D`. | — |
| `IsTransverse(C,D,p)` | `true` iff `p` is nonsingular on both plane curves `C`, `D` and they have distinct tangents there. | — |
| `IntersectionNumber(C,D,p)` | Local intersection number `Iₚ(C,D)` of plane curves `C` and `D` at `p`. Error if `C` or `D` have a common component at `p`. | Euclidean algorithm **[Ful69]** |
| `IntersectionNumbers(C,D)` / `IntersectionNumbers(F,G)` | List of all intersection places with multiplicities of two projective plane curves `C`, `D` (or homogeneous polynomials `F`, `G`), computed in one pass. The polynomial version represents each intersection place as type-i, ii, or iii data (projective point, irreducible factor in one variable, irreducible factor in two variables). Parameter: `Global` (use global polynomial rings; default `false`). | Algorithm of Hilmar–Smyth **[HS10]** |

*Worked examples: H114E10 (`IntersectionNumber` of `y²−x⁵` and `y−x²` at origin; comparison with product of multiplicities; `Dimension(RA/I)`); H114E11 (polynomial version of `IntersectionNumbers` following the Hilmar–Smyth paper example).*

---

## 114.5 Global Geometry

### 114.5.1 Genus and Singularities

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Genus(C)` / `GeometricGenus(C)` | The topological (geometric) genus of the integral curve `C`; the arithmetic genus of the projective normalisation `C̃`. May drop after inseparable base-field extensions over imperfect fields. | Via the algebraic function field (Chapter 42) |
| `ArithmeticGenus(C)` | Arithmetic genus of `C` (or its projective closure if affine). For a degree-`d` plane projective curve: `(d−1)(d−2)/2`. This is the genus of the scheme, not the normalisation. | Formula for plane curves |
| `NumberOfPunctures(C)` | Number of punctures of the affine plane curve `C` over the algebraic closure: the reduced degree of the polynomial of `C` at infinity. | — |
| `SingularPoints(C)` | Singular points of `C` defined over the base field. | — |
| `HasSingularPointsOverExtension(C)` | `false` iff all singularities over the algebraic closure are already defined over the base field. Requires `C` to be reduced. Uses the Jacobian algebra. | Radical of the Jacobian algebra |
| `Flexes(C)` / `InflectionPoints(C)` | Subscheme of the plane curve `C` defined by the vanishing of `det(HessianMatrix(C))`; contains the flex points (nonsingular points where the tangent meets `C` with multiplicity ≥ 3). | — |
| `C eq D` | `true` iff `C` and `D` are defined by identical ideals in the same ambient (for plane curves: comparing defining polynomials up to scalar, avoiding Gröbner bases). | — |
| `IsSubscheme(C,D)` | `true` iff `C` is contained scheme-theoretically in `D`. | — |

*Worked example: H114E12 (plane affine cubic `y²−x³−1` over GF(3): `Genus` = 0, `ArithmeticGenus` = 1; `y²−x³−t` over k(t): both genus and arithmetic genus = 1).*

### 114.5.2 Projective Closure and Affine Patches

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ProjectiveClosure(A)` | Projective closure of the affine ambient `A` (unique). | — |
| `ProjectiveClosure(C)` | Closure of the affine curve `C` in the projective closure of its ambient. | — |
| `LineAtInfinity(A)` | The line complementing the affine plane `A` in its projective closure. | — |
| `PointsAtInfinity(C)` | Rational points at infinity of the plane curve `C`. | — |
| `AffinePatch(C,i)` | The i-th affine patch of the projective curve `C`. Patch 1 is centred at `(0:0:…:0:1)`, patch 2 at `(0:…:1:0)`, etc. | — |

*Worked examples: H114E13 (`AmbientSpace` and `ProjectiveClosure` commute); H114E14 (affine patches of `x³z²−y⁵`; interesting singularity at infinity accessed via third patch).*

### 114.5.3 Special Forms of Curves

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsEllipticWeierstrass(C)` | `true` iff `C` is a nonsingular genus-1 plane curve in Weierstrass form (flex at `(0:1:0)` on `C` or its projective closure). | — |
| `IsHyperellipticWeierstrass(C)` | `true` iff `C` is a hyperelliptic curve in plane Weierstrass form: (a) first affine patch nonsingular, (b) `(0:1:0)` only point at infinity with tangent cone at the line at infinity, (c) projection away from that point has degree 2. | — |
| `EllipticCurve(C)` / `EllipticCurve(C,p)` | See Chapter 120. | — |
| `IsHyperelliptic(C)` | Whether `C` is hyperelliptic (degree-2 map to P¹) over its base field `K` (genus ≥ 2). If so and `Eqn := true` (default), returns a hyperelliptic Weierstrass model `H` over `K` and an isomorphism `C → H`. | Canonical map: check if image has arithmetic genus 0; if so map down via repeated adjunction. Function field differentials for the final equation. |
| `IsGeometricallyHyperelliptic(C)` | Whether `C` is hyperelliptic over the algebraic closure. If so and `Map := true` (default), returns a conic or P¹ (conic for odd genus, P¹ for even genus) and a degree-2 map. Verbose parameter: `IsHyp`. | Canonical map, then adjunction maps to reduce to conic or line. |

*Worked example: H114E15 (genus 4 curve in P⁵; `IsHyperelliptic` returning a Weierstrass model `y² = x⁸+x⁶+x−1` and the map).*

---

## 114.6 Maps and Curves

### 114.6.1 Elementary Maps

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IdentityAutomorphism(A)` | Identity automorphism of the affine plane `A`: `(x,y) ↦ (x,y)`. | — |
| `Translation(A,p)` | Translation by point `p = (a,b)`: `(x,y) ↦ (x−a, y−b)`. | — |
| `FlipCoordinates(A)` | Flip: `(x,y) ↦ (y,x)`. | — |
| `Automorphism(A,q)` | Automorphism `(x,y) ↦ (x+q(y),y)` where `q` is a polynomial in `y` only. | — |
| `TranslationToInfinity(C,p)` | Image of `C` under the change of coordinates translating smooth point `p` to `(0:1:0)` and making the tangent line there equal to the line at infinity. Returns the image curve and the coordinate-change map. Error if `p` is singular. | — |
| `EvaluateByPowerSeries(m, P)` | Evaluate a rational map `m : C → D` at a nonsingular base point `P` of `m`, using a power series expansion when direct evaluation fails. | Power series expansion at `P` |

*Worked examples: H114E16 (Fermat cubic: `IsFlex`, `TranslationToInfinity` to almost-Weierstrass form); H114E17 (rational map between two cubics; `EvaluateByPowerSeries` at a base point; pullback via `@@`).*

### 114.6.2 Maps Induced by Morphisms

Given a non-constant map `φ : D → C` between curves, there are induced maps on function fields, differentials, places and divisors.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Degree(m)` | Degree of a non-constant dominant map `m` between curves. | — |
| `RamificationDivisor(m)` | Ramification divisor of a non-constant dominant map `m` between irreducible curves. | — |
| `Pullback(phi, X)` | Pullback of function, differential, place or divisor `X` on `C` along `φ : D → C`. | — |
| `Pushforward(phi, X)` | Pushforward of function, place or divisor `X` on `D` along `φ : D → C`. If `φ` is undefined at the base of a place, works entirely at the places level. | — |

*Worked example: H114E18 (degree-4 Fermat curve `D` and `C`; holomorphic differential pulls back to holomorphic; `RamificationDivisor`; Riemann–Hurwitz verification; Pushforward and Divisor commute).*

---

## 114.7 Automorphism Groups of Curves

Automorphism groups of integral curves (available since Magma V2.13). The full group is computed at function-field level (algebraic function field, Chapter 42). Types: `GrpAutCrv` (group), `GrpAutCrvElt` (element, a subtype of `MapAutSch`). These are birational automorphisms of the normalisation of `C`. All functions require `C` to have a function field. Full automorphism group computation requires the base field to be perfect.

### 114.7.1 Group Creation Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AutomorphismGroup(C)` | Full automorphism group of a reduced, irreducible projective curve `C` over a perfect base field. Error if genus < 2 and base field is not finite. Result is cached. | Function field level; algebraic function field (Chapter 42) |
| `AutomorphismGroup(C,auts)` | Subgroup of the automorphism group of `C` generated by the given sequence `auts` of `MapAutSch` or `GrpAutCrvElt` elements. Same genus restrictions. | Function field level |
| `Automorphisms(C)` | Up to `Bound` (default ∞) automorphisms of `C` as a sequence of scheme maps (`MapSch`). | Function field level |
| `IsIsomorphic(C, D)` | `true` iff irreducible reduced curves `C` and `D` are isomorphic over their common base field; if so, also returns a scheme map. Restriction: not both genus 0 nor both genus 1 unless base field is finite. | Function field level |
| `Isomorphisms(C, D)` | Up to `Bound` (default ∞) isomorphisms from `C` to `D` as scheme maps. Same genus restrictions. | Function field level |

### 114.7.2 Automorphisms

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `A . i` | i-th generator of automorphism group `A` (`−n ≤ i ≤ n`; `i < 0` gives inverse; `i = 0` gives identity). | — |
| `Identity(A)` / `Id(A)` / `A ! 1` | Identity element of automorphism group `A`. | — |
| `A ! f` | Coerce scheme map `f : C → C` (or a `GrpAutCrvElt` of another automorphism group of `C`) into `A`. Error if `f ∉ A`. | — |
| `Order(f)` | Order of curve automorphism `f`. | — |
| `Inverse(f)` | Inverse of `f`. | — |
| `f * g` | Composition: first apply `f`, then `g`. Uses permutation representation for speed. | Permutation group multiplication |
| `f ^ n` | `n`-th power of `f` (`n` may be negative). Uses permutation representation. | Permutation group powering |
| `g eq h` | `true` iff `g` and `h` represent the same automorphism (they may belong to different automorphism groups of the same curve). | — |
| `g ne h` | Logical negation of `g eq h`. | — |
| `SchemeMap(f)` | Convert curve automorphism `f` to a plain `MapAutSch`. | — |

### 114.7.3 Automorphism Group Operations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Curve(A)` | The curve of which `A` is a group of automorphisms. | — |
| `Order(A)` | Order of the automorphism group `A`. | — |
| `FactoredOrder(A)` | Factored order of `A`. | — |
| `NumberOfGenerators(A)` / `Ngens(A)` | Number of generators of `A` (from the internal permutation representation). | — |
| `Generators(A)` | Small generating set of `A` as a sequence of `GrpAutCrvElt` elements. | — |
| `PermutationGroup(A)` | Abstract permutation group underlying `A`. | — |
| `PermutationRepresentation(A)` | Pair: the permutation group `G` and an invertible map `G → A`. | — |
| `MatrixRepresentation(A)` | For a curve of genus ≥ 2: matrix group `G` acting on holomorphic differentials by pullback (right multiplication on row vectors); also returns a map `A → G` and the basis of differentials used. | — |
| `a in A` | `true` iff automorphism `a` (as `GrpAutCrvElt` or `MapSch`) equals an automorphism in `A`. | — |
| `A subset B` | `true` iff `A` is a subgroup of `B` as automorphism groups of the same curve. | — |

### 114.7.4 Pullbacks and Pushforwards

For `GrpAutCrvElt` elements, pullbacks and pushforwards are handled more directly than for general maps; use these in preference to the functions of §114.6.2.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `f(X)` / `X @ f` | Image (pushforward) of point, function, differential, place, or divisor `X` on `C` under automorphism `f`. If the domain of definition of the scheme map doesn't include the point, the image is still computed (for nonsingular points). | — |
| `X @@ f` | Inverse image (pullback) of function, differential, place, or divisor `X` under `f`. | — |

*Worked examples: H114E19 (genus-3 Fermat quartic over Q and over Q(ζ₈); `Automorphisms`, `AutomorphismGroup`, `PermutationRepresentation`, pullbacks and powers of automorphisms); H114E20 (superelliptic curve over GF(7²); group of order 1344; `MatrixRepresentation`; Weierstrass place permutation representation); H114E21 (Klein quartic X(7) in two models: plane quartic and degree-6 curve in P³; `IsIsomorphic`, `IsInvertible`; then automorphism group of order 168 over GF(11³); generating subgroups and computing normalisers via `PermutationRepresentation`).*

### 114.7.5 Quotients of Curves

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CurveQuotient(G)` | For `G` a group of automorphisms of a curve `C/k` of genus ≥ 2: computes a projective nonsingular model of the quotient `C/G` and the G-invariant projection map `C → C/G`. Requires tame ramification in positive characteristic when genus(C/G) > 1. Algorithm varies by genus of the quotient: genus ≥ 2 non-hyperelliptic: canonical image; genus ≥ 2 hyperelliptic: function field + extra "y-coordinate" work; genus 0 or 1: combination of invariant theory and function fields. | Canonical image (genus ≥ 2) / invariant theory + function fields (genus 0,1) |

*Worked examples: H114E22 (Klein quartic X(7) over Q; quotient by order-3 automorphism gives genus-1 curve; `EllipticCurve`, `MinimalModel`, `Conductor`; result is X₀(49)); H114E23 (genus-4 modular curve X₀(54) with Atkin–Lehner involutions W₂, W₂₇; three quotients of genera 2, 1, 0).*

---

## 114.8 Function Fields

An integral curve `C` has a function field (type `FunFldFracSch`) as the field of fractions of its coordinate ring (affine) or the degree-0 part thereof (projective). The function fields of an affine curve and its projective closure are identified. The function field is stored on the projective curve. There is also a background algebraic function field (Chapter 42) used for place, divisor, and differential computations.

### 114.8.1 Function Fields

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `FunctionField(C)` | Function field of the integral curve `C`, isomorphic to the fraction field of the coordinate ring. Cached. | — |
| `HasFunctionField(C)` | `true` iff `C` is integral (can have a function field). | — |
| `Curve(F)` | Curve (or its projective closure) used to create the function field `F`. | — |
| `F ! r` | Coerce ring element `r` into the function field `F` (from the base ring, coordinate ring, field of fractions thereof, or any affine patch or related scheme). | — |
| `ProjectiveFunction(f)` | Return `f ∈ F` as an element of the field of fractions of the projective coordinate ring. | — |
| `p @ f` / `f(p)` / `Evaluate(f, p)` | Evaluate function `f` at nonsingular point `p`; returns infinity if `f` has a pole at `p`. | — |
| `Expand(f, p)` | Power series expansion of `f` at place `p`; also returns the uniformizing element. | Local power series expansion |
| `Completion(F, p)` | Completion of function field `F` at place `p` with a map `F → completion`. Parameter: `Precision` (default 20). | — |
| `Degree(f)` | Degree of function `f` (degree of the map to P¹; 0 if constant). | — |
| `Valuation(f, p)` | Order of vanishing of function `f` at nonsingular point `p` (negative = pole). | — |
| `Valuation(p)` | Valuation map from the function field to Z centred at point `p`. | — |
| `UniformizingParameter(p)` | A rational function of valuation 1 at the nonsingular point `p`. | — |
| `Module(S)` | Module over the base ring of `C` generated by elements `S` of the function field; also the map from the module into the function field. Parameters: `Preimages`, `IsBasis`. | — |
| `Relations(S)` / `Relations(S, m)` | Module of base-ring linear relations among elements `S` of the function field. | — |
| `Genus(C)` | Genus of `C` (from the function field). | — |
| `FieldOfGeometricIrreducibility(C)` | Algebraic closure of the base ring in the function field of `C`, with inclusion map. | — |
| `IsAbsolutelyIrreducible(C)` | `true` iff the field of geometric irreducibility of `C` equals its base ring. | — |
| `DimensionOfFieldOfGeometricIrreducibility(C)` | Degree of the field of geometric irreducibility over the base ring. | — |
| `GapNumbers(C)` | Gap numbers of the curve `C`. | — |
| `WronskianOrders(C)` | Wronskian orders of `C`. | — |
| `NumberOfPlacesOfDegreeOverExactConstantField(C, m)` / `NumberOfPlacesDegECF(C, m)` | Number of places of degree `m` of `C` over a finite field (degree taken over the field of geometric irreducibility). | — |
| `NumberOfPlacesOfDegreeOneOverExactConstantField(C)` / `NumberOfPlacesOfDegreeOneECF(C)` | Number of degree-1 places of `C` over the exact constant field. | — |
| `NumberOfPlacesOfDegreeOneOverExactConstantField(C, m)` / `NumberOfPlacesOfDegreeOneECF(C, m)` | Number of degree-1 places of `C` in the constant field extension of degree `m`. | — |
| `NumberOfPlacesOfDegreeOneECFBound(C)` / `NumberOfPlacesOfDegreeOneOverExactConstantFieldBound(C)` | Upper bound on degree-1 places (over the exact constant field). | — |
| `NumberOfPlacesOfDegreeOneECFBound(C, m)` / `NumberOfPlacesOfDegreeOneOverExactConstantFieldBound(C, m)` | Upper bound on degree-1 places in the degree-`m` constant field extension. | Serre / Ihara bound |
| `DivisorOfDegreeOne(C)` | A divisor of `C` of degree 1 over the field of geometric irreducibility. | — |
| `SerreBound(C)` / `SerreBound(C, m)` | Serre bound on the number of degree-1 places over the field of geometric irreducibility (possibly in the degree-`m` extension). Base ring must be a finite field. | Serre bound |
| `IharaBound(C)` / `IharaBound(C, m)` | Ihara bound (analogous). | Ihara bound |
| `LPolynomial(C)` / `LPolynomial(C, m)` | L-polynomial of `C` over the base finite field (or its degree-`m` extension). | — |
| `ZetaFunction(C)` / `ZetaFunction(C, m)` | Zeta function of `C` over the base finite field (or degree-`m` extension). | — |

*Worked example: H114E24 (function field construction; `Curve(F) eq C`; function field equals function field of affine patch); H114E25 (evaluating a function, computing valuations, finding `UniformizingParameter`).*

### 114.8.2 Representations of the Function Field

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AlgorithmicFunctionField(F)` | Returns the background algebraic function field `AF` (Chapter 42) and the map `F → AF` (invertible). This is the object where places, divisors, and differentials are computed. | — |
| `FunctionFieldPlace(p)` | Convert curve place `p` to a place of the algebraic function field. | — |
| `CurvePlace(C, p)` | Convert algebraic function field place `p` to a place of curve `C`. | — |
| `FunctionFieldDivisor(d)` | Convert curve divisor `d` to a divisor of the algebraic function field. | — |
| `CurveDivisor(C, d)` | Convert algebraic function field divisor `d` to a divisor of curve `C`. | — |
| `FunctionFieldDifferential(d)` | Convert curve differential `d` to a differential of the algebraic function field. | — |
| `CurveDifferential(C, d)` | Convert algebraic function field differential `d` to a differential of curve `C`. | — |

### 114.8.3 Differentials

The space of differentials is the one-dimensional vector space over the function field generated by `df` for `f` in the function field, satisfying the usual derivation conditions (Kähler differentials, see **[Har77]** II.8).

#### 114.8.3.1 Creation of Differentials

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DifferentialSpace(C)` | The space of differentials of the curve `C`. | — |
| `SpaceOfDifferentialsFirstKind(C)` / `SpaceOfHolomorphicDifferentials(C)` | A vector space `V` and a map `V → Ω(C)` with image the holomorphic (first-kind) differentials. | — |
| `BasisOfDifferentialsFirstKind(C)` / `BasisOfHolomorphicDifferentials(C)` | A basis for the space of holomorphic differentials of `C`. | — |
| `DifferentialSpace(D)` | For a divisor `D` on curve `C`: a vector space `V` and map `V → Ω(C)` with image `ωC(D)` (differentials with zeros at the positive part of `D`, poles no worse than negative part). | — |
| `DifferentialBasis(D)` | Basis of the differential space of divisor `D`. | — |
| `Differential(a)` | The exact differential `d(a)` of function field element `a`. | — |

#### 114.8.3.2 Operations on Differentials

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Identity(S)` | Identity (zero) differential of differential space `S`. | — |
| `Curve(S)` | Curve for which `S` is the space of differentials. | — |
| `Curve(a)` | Curve to which differential `a` belongs. | — |
| `f * x` / `x * f` / `x + y` / `- x` / `x - y` / `x / r` / `x / y` | Arithmetic in the space of differentials (one-dimensional vector space over the function field). Division of two differentials returns a function field element. | — |
| `S eq T` | `true` iff differential spaces `S` and `T` are identical. | — |
| `a eq b` | `true` iff differentials `a` and `b` are equal. | — |
| `a in S` | `true` iff `a` is an element of differential space `S`. | — |
| `IsExact(a)` | `true` iff `a` is already known to be exact (of the form `df`). No further attempt is made if not. | — |
| `IsZero(a)` | `true` iff `a` is the zero differential. | — |
| `Valuation(d, P)` | Valuation of differential `d` at place `P`. | — |
| `Residue(d, P)` | Residue of differential `d` at degree-1 place `P`. | — |
| `Divisor(d)` | The divisor `(f) + (dx)` of the differential `d = f dx`. | — |
| `Module(L)` | Abstract module generated by differentials in `L` plus a map into the differential space. Parameters: `IsBasis`, `PreImages`. | — |
| `Relations(L)` / `Relations(L, m)` | Module of base-ring linear relations among differentials in `L`. Argument `m` gives a generating system of "small" elements. | — |
| `Cartier(a)` / `Cartier(a, r)` | Apply the Cartier operator `CA` once (or `r` times) to differential `a`. For curve over perfect field `k`, function field `F`, separating variable `x`, and `a = g dx`: `CA(a) = (−d^(p−1)g/dx^(p−1))^(1/p) dx`. | Cartier operator |
| `CartierRepresentation(C)` / `CartierRepresentation(C, r)` | Row representation matrix `M ∈ k^(g×g)` for the Cartier operator (applied `r` times) on a basis of holomorphic differentials: `CA^r(ωᵢ) = Σₘ λᵢₘ ωₘ`. Also returns the basis `(ω₁,…,ωg)`. | Cartier operator matrix |

*Worked example: H114E26 (genus-3 Fermat quartic; `SpaceOfHolomorphicDifferentials` gives a 3-dimensional module; `Differential(a/b)`; `IsExact`).*

---

## 114.9 Divisors

Divisors are formal sums of places of the normalisation `C̃`. The chapter works implicitly on `C̃` via the algebraic function field. All calculations are ultimately done in Chapter 42's framework. The main concern is linear equivalence.

### 114.9.1 Places

A place is a point of `C̃` (a valuation of the function field).

#### 114.9.1.1 Sets of Places

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Places(C)` | The set of places of curve `C`. | — |
| `Curve(P)` (on set of places) | The (projective) curve on which the places in set `P` lie. | — |
| `P eq Q` / `P ne Q` | Equality / inequality of sets of places `P` and `Q`. | — |

#### 114.9.1.2 Individual Places

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Places(C, m)` | Sequence of all places of degree `m` on `C` over a finite field. | — |
| `HasPlace(C, m)` / `RandomPlace(C, m)` | If a place of degree `m` exists: `true` and a single such place; otherwise `false`. | — |
| `Place(p)` | Place corresponding to the nonsingular point `p`. | — |
| `Places(p)` | Sequence of places above the point `p`. | — |
| `Place(C, I)` | Place of `C` defined by the ideal `I` of the coordinate ring of `C`. | — |
| `WeierstrassPlaces(C)` | Weierstrass places of `C` (= Weierstrass places of the zero divisor). | — |
| `Place(Q)` | Place determined by a sequence `Q` of function field elements. | — |
| `Ideal(P)` | Prime ideal of the ambient coordinate ring vanishing at place `P`. | — |
| `TwoGenerators(P)` | Two function field elements determining `P` (usable as input to `Place` for reconstruction). | — |
| `Zeros(f)` / `Poles(f)` | Sequence of places at the zeros / poles of function `f`. | — |
| `Zeros(C, f)` / `Poles(C, f)` | Zeros / poles of a function `f` coercible into the function field of `C`. | — |
| `CommonZeros(L)` / `CommonZeros(C, L)` | Places that are common zeros of all functions in the sequence `L`. | — |
| `p1 + p2` / `- p1` / `p1 - p2` / `k * p` / `p div k` / `p mod k` / `Quotrem(p1, k)` | Arithmetic on places (interpreted as degree-1 divisors). | — |
| `Curve(P)` | The projective curve on which place `P` lies. | — |
| `RepresentativePoint(P)` | A representative point on the projective model corresponding to `P`. | — |
| `P eq Q` / `P ne Q` | Equality / inequality of places. | — |
| `P in S` / `P notin S` | Membership in a set of places. | — |
| `Valuation(f, P)` | Order of vanishing of function `f` at place `P` (negative = pole). | — |
| `Valuation(P)` | Valuation map from the function field centred at `P`. | — |
| `Valuation(a, P)` | Valuation of differential `a` at place `P`. | — |
| `Residue(a, P)` | Residue of differential `a` at degree-1 place `P`. | — |
| `UniformizingParameter(P)` | Function of valuation 1 at place `P`. | — |
| `IsWeierstrassPlace(P)` / `IsWeierstrassPlace(D, P)` | `true` iff degree-1 place `P` is a Weierstrass place (of divisor `D` if given). | — |
| `ResidueClassField(P)` | Residue class field of `P`. | — |
| `Evaluate(a, P)` | Evaluate function field element `a` at `P`; result in `ResidueClassField(P)`. | — |
| `Lift(a, P)` | Lift element `a ∈ ResidueClassField(P)` (including ∞) to a function on `C`. | — |
| `Degree(P)` | Degree of place `P`. | — |
| `GapNumbers(C, P)` / `GapNumbers(P)` | Gap numbers of `C` at degree-1 place `P`. | — |
| `Parametrization(C, p)` / `Parametrization(C, p, P)` | Parametrizing map for the rational curve `C` at rational point or degree-1 place `p`. If `P` (the projective line as a curve) is given, it is the domain. | — |

*Worked example: H114E27 (creating a place via `TwoGenerators`; reconstructing with `Place([…])`).*

### 114.9.2 Divisor Group

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DivisorGroup(C)` | Divisor group of the curve `C` (affine or projective). | — |
| `Curve(Div)` | Curve (or its projective model) used to create divisor group `Div`. | — |
| `Div1 eq Div2` / `Div1 ne Div2` | Equality / inequality of divisor groups. | — |

### 114.9.3 Creation of Divisors

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DivisorGroup(D)` | The divisor group containing divisor `D`. | — |
| `Curve(D)` | The projective curve on which `D` lies. | — |
| `Identity(D)` / `Id(D)` / `D ! 0` | Zero divisor of divisor group `D`. | — |
| `Div ! p` / `Divisor(p)` | Prime divisor in divisor group `Div` corresponding to place or nonsingular point `p`. | — |
| `Divisor(D, S)` / `Divisor(C, S)` / `Divisor(S)` | Divisor described by a factorization sequence `S` of `<place, integer>` tuples. | — |
| `PrincipalDivisor(C, f)` / `PrincipalDivisor(D, f)` / `PrincipalDivisor(f)` | Principal divisor of zeros and poles of function `f`. | Via algebraic function field |
| `Divisor(C, f)` / `Divisor(D, f)` / `Divisor(f)` | Aliases for `PrincipalDivisor`. | — |
| `Divisor(a)` | Divisor of differential `a`. | — |
| `Divisor(C, X)` / `Divisor(D, X)` | Divisor from the intersection of curve `C` with scheme `X`. | — |
| `Divisor(C, p, q)` / `Divisor(D, p, q)` | Principal divisor corresponding to the line through `p` and `q` (tangent line if they coincide). | — |
| `Divisor(C, I)` / `Divisor(D, I)` | Divisor of `C` defined by ideal `I` of the ambient coordinate ring. | — |
| `Decomposition(D)` | Sequence of `<place, multiplicity>` tuples characterising `D`. | — |
| `Support(D)` | Sequence of places in the support of `D`, and their sequence of multiplicities. | — |
| `CanonicalDivisor(C)` | A divisor in the canonical divisor class of `C`. | Via algebraic function field |
| `RamificationDivisor(C)` | Ramification divisor of `C`. | — |

*Worked example: H114E29 (reconstructing canonical divisor from `Support` and `TwoGenerators`); H114E30 (creating divisors from functions on an elliptic curve over GF(7)).*

### 114.9.4 Arithmetic of Divisors

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `D + E` / `- D` / `D - E` / `n * D` / `D div n` / `D mod n` | Basic formal arithmetic of divisors. | — |
| `Quotrem(D, n)` | Quotient and remainder on dividing divisor `D` by integer `n`. | — |
| `Degree(D)` | Sum of coefficients of `D` times the degrees of the corresponding place components. | — |
| `IsEffective(D)` / `IsPositive(D)` | `true` iff all coefficients of `D` are non-negative. | — |
| `Numerator(D)` / `Denominator(D)` | The effective numerator and denominator of `D` (so `D = Numerator − Denominator`). | — |
| `SignDecomposition(D)` | Minimal effective divisors `A`, `B` such that `D = A − B`. | — |
| `d in D` / `d notin D` | Membership. | — |
| `D eq E` / `D ne E` | Equality of divisors (as elements of the divisor group; not linear equivalence). | — |
| `D lt E` / `D le E` / `D gt E` / `D ge E` | Comparison (component-wise). | — |
| `IsZero(D)` | `true` iff all coefficients are zero. | — |
| `IsCanonical(D)` | `true` iff `D` is the divisor of a differential; if so, also returns a realising differential. | — |
| `GCD(D1, D2)` / `Gcd(D1, D2)` / `GreatestCommonDivisor(D1, D2)` | GCD: minimum of coefficients on common support. | — |
| `LCM(D1, D2)` / `Lcm(D1, D2)` / `LeastCommonMultiple(D1, D2)` | LCM: maximum of coefficients on the union of supports. | — |

*Worked example: H114E31 (`SignDecomposition` on a degree-0 divisor on an elliptic curve over GF(7)); H114E32 (`IsCanonical` confirms a divisor is canonical; computing its valuation at a place).*

### 114.9.5 Other Operations on Divisors

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Ideal(D)` | Ideal of the ambient coordinate ring cutting out the divisor `D`. | — |
| `Valuation(D,p)` / `Valuation(D,P)` | Coefficient of the component of `D` at point `p` or place `P`. | — |
| `ComplementaryDivisor(D,p)` / `ComplementaryDivisor(D,P)` | Divisor `D` with the component at `p` (or `P`) removed. | — |

---

## 114.10 Linear Equivalence of Divisors

### 114.10.1 Linear Equivalence and Class Group

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsPrincipal(D)` | `true` iff `D` is the divisor of a rational function; also returns such a function. | Via algebraic function field (Chapter 42) |
| `IsLinearlyEquivalent(D1,D2)` | `true` iff `D1 − D2` is principal; also returns the rational function. | — |
| `IsHypersurfaceDivisor(D)` | For an effective divisor `D` on a projective curve: `true` iff `D` is the scheme-theoretic intersection with a hypersurface; if so, returns the equation and degree of the hypersurface. | — |
| `ClassGroup(C)` | For a curve over a finite field: an abelian group isomorphic to the divisor class group (divisors mod principal divisors), a representative map from the class group to divisors, and the quotient homomorphism. The map is invertible. | Via algebraic function field (Chapter 42) |
| `ClassNumber(C)` | Order of the class group. | — |
| `GlobalUnitGroup(C)` | Multiplicative group of the field of geometric irreducibility of `C` as an abelian group, with the inclusion map into the function field. | — |
| `ClassGroupAbelianInvariants(C)` | Abelian invariants of the class group of `C`. | — |
| `ClassGroupPRank(C)` | p-rank of the class group. | — |
| `HasseWittInvariant(C)` | Hasse–Witt invariant of `C`. | — |

*Worked example: H114E33 (`IsHypersurfaceDivisor` for an effective canonical divisor on the Klein quartic); H114E34 (class group `Z/26425 + Z` for a genus-3 curve over GF(2⁵)).*

### 114.10.2 Riemann–Roch Spaces

For a divisor `D` on projective `C`, the Riemann–Roch space is `L(D) = {f ∈ k(C) | div(f) + D ≥ 0}`. The Riemann–Roch formula: `ℓ(D) − ℓ(KC − D) = deg(D) + 1 − g`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Reduction(D)` / `Reduction(D, A)` | Reduction `D̃` of a divisor `D`: computes `D̃`, `r`, `A`, `a` such that `D = D̃ + rA − (a)`, `D̃ ≥ 0`, `deg(D̃) < g + deg(A)`, and `D̃` is of minimal degree satisfying these conditions. | Via algebraic function field |
| `RiemannRochSpace(D)` | A vector space `V` and isomorphism `V → L(D)`. | Riemann–Roch via algebraic function field (Chapter 42) |
| `Basis(D)` | Basis of the Riemann–Roch space `L(D)`. | — |
| `ShortBasis(D)` | Short basis of `L(D)`. | — |
| `Dimension(D)` | Dimension `ℓ(D)` of the Riemann–Roch space. | — |
| `DifferentialSpace(D)` | Vector space `V` and map `V → Ω(C)` with image `ωC(D)`. | — |
| `DifferentialBasis(D)` | Basis of the differential space of `D`. | — |
| `IndexOfSpeciality(D)` | Index of speciality `ℓ(KC − D)` from the Riemann–Roch formula. | — |
| `IsSpecial(D)` | `true` iff `D` is special (index of speciality > 0). | — |
| `GapNumbers(D)` / `GapNumbers(D,p)` / `GapNumbers(p)` | Gap numbers of divisor `D` (at place `p` if given), or of a nonsingular point `p`. | — |
| `WeierstrassPlaces(D)` / `WeierstrassPoints(D)` | Weierstrass places (or their underlying points) of divisor `D`. | — |
| `WronskianOrders(D)` | Wronskian orders of divisor `D`. | — |
| `RamificationDivisor(D)` | Ramification divisor of `D`. | — |
| `DivisorMap(D)` / `DivisorMap(D,P)` | Map from the curve of `D` to projective space `P` (created if not given; dimension must equal `ℓ(D) − 1`). | — |
| `CanonicalMap(C)` / `CanonicalMap(C,P)` | Canonical map from `C` to projective space `P` (created if not given; dimension = `g − 1`) determined by a basis of `L(KC)`. | — |
| `CanonicalImage(C, phi)` / `CanonicalImage(C, eqns)` | Canonical image of a genus ≥ 2 projective curve, computed more efficiently than via the generic image machinery. Second return value: `true` iff `C` is hyperelliptic. Also works for genus-0 rational normal curve images. | Efficient canonical image computation |

*Worked example: H114E35 (genus-3 curve over Q; `CanonicalMap` maps to a rational curve — C is hyperelliptic; bicanonical map into P⁵ gives a scroll section).*

### 114.10.3 Index Calculus

Discrete logarithms in the degree-0 divisor class group of plane curves over finite fields. Algorithm: Diem's version of index calculus **[Die06]**, using lines through factor base points. Group order must be supplied. The algorithm requires `q ≥ d!` for a degree-`d` curve over GF(q).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IndexCalculus(D1, D2, D0, np)` / `IndexCalculus(D1, D2, D0, np, n, rr)` | Discrete logarithm of `D2 − deg(D2)D0` with base `D1 − deg(D1)D0`; group order `np` must be given. Optional: `n` (factor base size), `rr` (number of required relations). | Diem's index calculus **[Die06]** |
| `IndexCalculusMatrix(D1, D2, D0, n, rr)` | Sieving stage only: returns the sparse relation matrix `M`, positions `pos`, factor base `fb`, auxiliary divisors `Da`, `Db`, and scalars `a`, `b`. Group order not needed. | Sieving stage of Diem's index calculus **[Die06]** |
| `MultiplyDivisor(n, D, D0)` | Effective divisor `E` of minimal degree such that `E − deg(E)D0` is equivalent to `n(D − deg(D)D0)`. | — |

*Worked example: H114E36 (discrete log in degree-6 curve over GF(2¹³); L-polynomial computation; `IndexCalculus` with large prime group order).*

---

## 114.11 Advanced Examples

### 114.11.1 Trigonal Curves

Demonstration of a genus-8 trigonal curve using the canonical map, identification of the rational scroll, construction of a ruled surface model, and extraction of the g¹₃ via divisor machinery.

*Worked example: H114E37 (canonical model of `X⁸ + X⁴Y³Z + Z⁸` in P⁷; `Image(phi,C,2)` for quadrics in the canonical ideal; `RuledSurface` for the scroll; `DivisorMap` for the trigonal map to P¹).*

### 114.11.2 Algebraic Geometric Codes

Construction of the **[24, 4]** and **[24, 20]** codes over GF(8) from the Klein quartic, following van Lint–van der Geer **[vLvdG88]**.

*Worked example: H114E38 (Klein quartic over GF(2), GF(4), GF(8); `IntersectionNumber` for bitangent; degree-2 place from bitangent; `AlgebraicGeometricCode(S,G)` and its dual).*

---

## 114.12 Curves over Global Fields

### 114.12.1 Finding Rational Points

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PointsCubicModel(C, B)` | Search for rational points of naive height ≤ `B` on a plane cubic `C` over Q. Asymptotic running time `O(B)`. Parameters: `OnlyOne` (stop at first point found), `ExactBound` (discard points with height > B), `Verbose`. | Method of Elkies **[Elk00]** |

*Worked example: H114E39 (cubic `x³+9y³+73z³` over Q, search up to height 10⁴).*

### 114.12.2 Regular Models of Arithmetic Surfaces

Given an integral curve `C` over a global field `F` with ring of integers `O_F`, compute a regular model of the associated arithmetic surface at a prime `P`.

#### 114.12.2.1 Creation

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RegularModel(C, P)` | Regular model of curve `C` at prime `P` (element or ideal). `C` must have integral defining equations; the reduction mod `P` must have dimension 1. Returns a model over `O_F` whose generic fibre is isomorphic to `C`, regular on the special fibre. May replace `F` by a finite extension if needed (with a warning). Not necessarily a minimal model. Parameter: `Verbose`. | — |

#### 114.12.2.2 Using Regular Models

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IntersectionMatrix(M)` | Matrix of intersection multiplicities of reduced irreducible components of the special fibre, and a sequence of their multiplicities. | — |
| `ComponentGroup(M)` | Component group of the Néron model of the Jacobian of `C` at `P` as an abstract abelian group (computed from `IntersectionMatrix`). | — |
| `PointOnRegularModel(M, x)` | For a point `x ∈ C(F)`, lifts `x` to a point on the generic fibre of a patch of the regular model. Returns coordinates, patch equations, and the patch index. | — |

### 114.12.3 Minimization and Reduction

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ReduceCluster(X)` | Reduce a cluster `X` of n-dimensional complex vectors to a better-embedded form. Returns the transformed cluster, the transformation matrix, and its inverse. Parameters: `eps` (zero threshold, default 1e-6), `c` (acceleration factor, default 1). | Stoll's cluster reduction **[Sto11]** |
| `ReducePlaneCurve(f)` | Reduce a homogeneous polynomial `f` of degree > 2 with integer/rational coefficients. Uses `ReduceCluster` on the intersection points of `f` with its Hessian. Returns the reduced polynomial and the transformation matrix. | Stoll **[Sto11]** |
| `MinimizePlaneQuartic(f,p)` | Compute a model of the plane quartic `f` (integer coefficients) that is minimized at prime `p`. Returns the new quartic and the transformation. Parameter: `Verbose`. | — |
| `MinimizeReducePlaneQuartic(f)` | Minimize and reduce a smooth plane quartic `f` (integer coefficients). Uses `ReducePlaneCurve` for the reduction step. Returns the reduced quartic and the transformation. Parameter: `Verbose`. | Minimization + Stoll reduction **[Sto11]** |

*Worked example: H114E40 (badly embedded plane quartic over Z; `MinimizeReducePlaneQuartic` recovers a quartic with small coefficients).*

---

## 114.13 Minimal Degree Functions and Plane Models

Computation of smallest-degree covering maps from curves of general type and genus ≤ 6 to P¹ (gonal maps). Hyperelliptic (gonality 2) cases use the canonical map. Trigonal cases use the Lie algebra method of Schicho–Sevilla **[SS]**. The 4-gonal cases for genus 5 and 6 use Harrison's algorithms **[Hara]**. Curves are assumed to be defined over number fields or finite fields.

### 114.13.1 General Functions and Clifford Index One

A curve has Clifford index 1 iff it is trigonal or is a genus-6 non-singular plane quintic (gonality 4). Equivalently (for genus > 3), its canonical image is not defined by quadrics alone. The algorithm of Schicho–Sevilla applies the Lie algebra method to the rational scroll surface defined by the quadrics containing the canonical curve.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `GenusAndCanonicalMap(C)` | Returns genus `g`, a boolean (`true` iff `g ≤ 1` or `C` is hyperelliptic), and the canonical map if `g > 1`. | — |
| `CliffordIndexOne(C)` / `CliffordIndexOne(C,X)` | For a nonsingular canonical curve of Clifford index 1: returns a degree-3 map to P¹ (trigonal case) or a birational map to a smooth plane quintic. In the trigonal case, the map may be over a quadratic extension for even genus. Requires characteristic ≠ 2 in the trigonal case. The optional second argument `X` is the scroll surface (if pre-computed). | Lie algebra method of Schicho–Sevilla **[SS]**; fibration map of the scroll surface |

*Worked example: H114E41 (genus-6 canonical curve in P⁵ of gonality 3; `CliffordIndexOne` gives a degree-3 map; verification using hyperplane section degrees).*

### 114.13.2 Small Genus Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Genus2GonalMap(C)` | Degree-2 map from genus-2 curve `C` to P¹. Defined over a quadratic extension iff no map exists over the base field. | Canonical map + inverse parametrisation of its image |
| `Genus3GonalMap(C)` | Gonality (2 or 3) and a gonal map for a genus-3 curve `C`. Trigonal case: canonical image is a plane quartic Q; gonal map = canonical map composed with projection from a point on Q. Point found by height search (Q) or enumeration (finite field); otherwise a point over a ≤ 4-degree extension is used. Parameter: `IsCanonical`. | Projection from a point on the canonical quartic |
| `Genus4GonalMap(C)` | Gonality (2 or 3) and a gonal map for a genus-4 curve `C`. Trigonal case uses `CliffordIndexOne`; the fibration map of a quadric surface in P³ may require a biquadratic extension. Parameter: `IsCanonical`. | `CliffordIndexOne` **[SS]** for trigonal; canonical map for hyperelliptic |
| `Genus5GonalMap(C)` | Gonality (2, 3, or 4) and a gonal map for a genus-5 curve `C`. In the 4-gonal case, also returns a plane quintic `F` parametrising the family of gonal maps and a function `f : F(K) → {gonal maps}`. For trigonal case: fibration of the scroll from the minimal free resolution of the canonical coordinate ring (always over base field). For 4-gonal: searches for a point on `F`. Parameters: `DataOnly`, `IsCanonical`. | Trigonal: free resolution of canonical ring. 4-gonal: algorithms of Harrison **[Hara]**; Lie algebra for scroll fibration |
| `Genus6GonalMap(C)` | Gonality (2, 3, or 4), a secondary type number (1, 2, or 3), and a gonal map for a genus-6 curve `C`. Secondary type 2: `C` is a double cover of a genus-1 curve `E` (returned as fourth value). Secondary type 3: `C` is isomorphic to a plane quintic (isomorphism returned). General (type 1): finitely many gonal maps; algorithm finds one defined over a minimal extension. Parameters: `DataOnly`, `IsCanonical`. | Trigonal: `CliffordIndexOne` **[SS]**; 4-gonal (all types): algorithms of Harrison **[Hara]** |

*Worked example: H114E42 (genus-5 4-gonal curve over Q: `Genus5GonalMap` returning the gonal map and the quintic `F`; genus-6 curve which is a double cover of a genus-1 curve: `Genus6GonalMap` returning type 2 and the covering map `mpE`).*

### 114.13.3 Small Genus Plane Models

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Genus6PlaneCurveModel(C)` | For a genus-6 4-gonal curve `C` that is not a double cover of a genus-1 curve: computes a birational map to a plane curve of minimal degree (5 or 6) over a minimal extension of the base field. Returns a boolean and the map. Subtype 3 (plane quintic): uses `CliffordIndexOne`. General (subtype 1): degree-6 plane model via the same algorithm as the gonal maps. Parameter: `IsCanonical`. | `CliffordIndexOne` **[SS]** (subtype 3); Harrison **[Hara]** (general) |
| `Genus5PlaneCurveModel(C)` | For a genus-5 curve `C`: tries to compute a birational map over the base field to a plane curve of minimal degree (5 for gonality 3, 6 for 4-gonal). Returns a boolean and the map. Gonality-3: projection from a line in the scroll; always succeeds over base field. 4-gonal: needs a secant or tangent line (a rational point or degree-2 reduced divisor on the canonical model). Attempts point search (Q) or point enumeration (finite field); returns `false` if search fails. Parameter: `IsCanonical`. | Gonality-3: projection from the scroll line. 4-gonal: projection from a secant/tangent line |
| `Genus5PlaneCurveModel(C,P)` | As above, with a user-supplied rational nonsingular point `P` on `C`. Always succeeds. | — |
| `Genus5PlaneCurveModel(C,Z)` | As above, with a user-supplied reduced degree-2 subscheme `Z` of the nonsingular locus of `C`. Always succeeds. | — |

*Worked example: H114E43 (modular curve X₀(58) as genus-6 canonical model; `Genus6PlaneCurveModel` giving a degree-6 plane model; X₀(42) as a genus-5 4-gonal model; `Genus5PlaneCurveModel` with and without an explicit rational point).*

---

## 114.14 Bibliography

| Key | Reference |
|-----|-----------|
| **[Bos00]** | Wieb Bosma, editor. *ANTS IV*, volume 1838 of LNCS. Springer-Verlag, 2000. |
| **[Die06]** | Claus Diem. An Index Calculus Algorithm for Plane Curves of Small Degree. In Hess et al. **[HPP06]**, pages 543–557. |
| **[Elk00]** | N. Elkies. Rational Points Near Curves and Small Nonzero |x³ − y²| via Lattice Reduction. In Bosma **[Bos00]**, pages 33–63. |
| **[Ful69]** | William Fulton. *Algebraic Curves*. Mathematics Lecture Note Series. W. A. Benjamin, New York–Amsterdam, 1969. |
| **[Hara]** | M. C. Harrison. Explicit solution by radicals, gonal maps and plane models of algebraic curves of genus 5 or 6. Preprint: arXiv:1103.4946v3 [math.AG]. |
| **[Har77]** | Robin Hartshorne. *Algebraic Geometry*, GTM 52. Springer, 1977. |
| **[HPP06]** | F. Hess, S. Pauli, and M. Pohst, editors. *ANTS VII*, volume 4076 of LNCS. Springer-Verlag, 2006. |
| **[HS10]** | J. Hilmar and C. Smyth. Euclid Meets Bézout: Intersecting Algebraic Plane Curves with the Euclidean Algorithm. *The American Mathematical Monthly*, 117:250–260, (March) 2010. |
| **[SS]** | J. Schicho and D. Sevilla. Effective radical parametrization of trigonal curves. Preprint: arXiv:1104.2470v1 [math.AG]. |
| **[ST02]** | Frank-Olaf Schreyer and Fabio Tonoli. Needles in a Haystack: Special Varieties via Small Fields. In Eisenbud et al., editors, *Computations in Algebraic Geometry with Macaulay2*, volume 8 of Springer Algorithms and Computation in Mathematics Series, pages 251–277. Springer-Verlag, 2002. |
| **[Sto11]** | Michael Stoll. Reduction theory of point clusters in projective space. 2011. |
| **[vLvdG88]** | J. H. van Lint and G. van der Geer. *Introduction to Coding Theory and Algebraic Geometry*, volume 12 of DMV Seminar. Birkhäuser, Basel, 1988. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Random curve generation **[ST02]** | `RandomNodalCurve`, `RandomOrdinaryPlaneCurve`, `RandomCurveByGenus` |
| Adjoint ideal (ordinary plane curves) | `HasOnlyOrdinarySingularities`, `AdjointIdeal`, `AdjointIdealForNodalCurve`, `AdjointLinearSystem`, `CanonicalLinearSystem`, `Adjoints`, `AdjointLinearSystemFromIdeal`, `CanonicalLinearSystemFromIdeal` |
| Local intersection numbers — Euclidean algorithm **[Ful69]** | `IntersectionNumber` |
| Global intersection numbers — Hilmar–Smyth **[HS10]** | `IntersectionNumbers` |
| Log canonical thresholds (discrepancy computation) | `LogCanonicalThreshold`, `LogCanonicalThresholdAtOrigin`, `LogCanonicalThresholdOverExtension` |
| Blowup / resolution of singularities | `Blowup`, `ResolutionGraph` |
| Function field methods (places, divisors, Riemann–Roch) | `FunctionField`, `AlgorithmicFunctionField`, `Places`, `DivisorGroup`, `Divisor`, `PrincipalDivisor`, `RiemannRochSpace`, `Basis`, `Dimension`, `CanonicalDivisor`, `CanonicalMap`, `CanonicalImage`, `DivisorMap` |
| Class group (curves over finite fields) | `ClassGroup`, `ClassNumber`, `IsPrincipal`, `IsLinearlyEquivalent` |
| Cartier operator | `Cartier`, `CartierRepresentation` |
| Index calculus — Diem **[Die06]** | `IndexCalculus`, `IndexCalculusMatrix`, `MultiplyDivisor` |
| Automorphism groups (function field level) | `AutomorphismGroup`, `Automorphisms`, `IsIsomorphic`, `Isomorphisms` |
| Curve quotients (canonical image / invariant theory) | `CurveQuotient` |
| Rational point search on cubics — Elkies **[Elk00]** | `PointsCubicModel` |
| Regular models of arithmetic surfaces | `RegularModel`, `IntersectionMatrix`, `ComponentGroup`, `PointOnRegularModel` |
| Cluster reduction — Stoll **[Sto11]** | `ReduceCluster`, `ReducePlaneCurve`, `MinimizePlaneQuartic`, `MinimizeReducePlaneQuartic` |
| Clifford index 1 / trigonal curves — Schicho–Sevilla **[SS]** | `CliffordIndexOne`, `Genus3GonalMap`, `Genus4GonalMap`, `Genus6GonalMap` (trigonal) |
| 4-gonal maps, genus 5–6 — Harrison **[Hara]** | `Genus5GonalMap`, `Genus6GonalMap`, `Genus6PlaneCurveModel`, `Genus5PlaneCurveModel` |
