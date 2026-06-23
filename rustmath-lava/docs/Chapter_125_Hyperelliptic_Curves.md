# Chapter 125 — Hyperelliptic Curves

**Handbook part:** XVI — Arithmetic Geometry
**Handbook pages:** 4123–4221 (PDF pages 4252–4355)
**Initial development:** Michael Stoll, with members of the Magma group. Major contributed packages: invariant/twist/automorphism machinery (genus 2 and 3) by Reynald Lercier and Christophe Ritzenthaler; Igusa invariants by Everett W. Howe (based on `gp` routines of Fernando Rodriguez-Villegas); canonical heights by Jan Steffen Müller; descent/Selmer machinery drawing on work of Bruin, Stoll, Creutz, Poonen–Schaefer.

---

## Scope and overview

A hyperelliptic curve in Magma — taken to include the genus-one case — is a nonsingular
generalised Weierstrass equation

    y² + h(x)·y = f(x),

with `h(x)`, `f(x)` polynomials over a field `K`. The curve is embedded in a weighted
projective space with weights `1, g+1, 1` (on `x, y, z`), so the one or two points at infinity
are nonsingular. The category of curves is `CrvHyp`, points `PtHyp`; Jacobians are `JacHyp`
with points `JacHypPt`; the Kummer surface of a genus-2 Jacobian is `SrfKum` with points
`SrfKumPt`.

The chapter covers: creation and models (simplified, integral, minimal Weierstrass, reduced),
twisting, elementary and weighted invariants (Clebsch, Igusa–Clebsch, Igusa/J, Shioda, Maeda,
Cardona-Quer-Nart-Pujola), construction of curves from invariants, function fields, points and
point counting (zeta functions, Frobenius), isomorphisms/transformations and automorphism
groups, Jacobians (group structure, point counting over finite fields via Shanks/Pollard and
p-adic methods of Kedlaya/Vercauteren/Mestre/Lauder–Hubrechts, deformation counting), Richelot
isogenies, points on the Jacobian (Mumford representation, arithmetic, order, Weil pairing),
heights and regulators (Flynn–Smart and Arakelov-intersection methods), 2-Selmer groups and
rank bounds, two-cover descent, Chabauty's method, cyclic covers of P¹ and their descents,
Kummer surfaces, and analytic Jacobians (period matrices, isomorphisms/isogenies, endomorphism
rings, Rosenhain invariants, Voronoi cells).

Particular optimisation exists for genus-2 curves over **Q** (heights, Kummer arithmetic) and
for Jacobians over finite fields (specialised group-structure and point-counting algorithms).

---

## 125.2 Creation Functions

### 125.2.1 Creation of a Hyperelliptic Curve

A hyperelliptic curve `C : y² + h(x)·y = f(x)` is created from `h`, `f ∈ R[x]` where `R` is a
field or integral domain. If `h` is omitted it is zero; for `R` an integral domain the base
field is the field of fractions of `R`. An error results if `C` is singular.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HyperellipticCurve(f, h)` / `HyperellipticCurve(f)` / `HyperellipticCurve([f, h])` | The nonsingular curve `C : y² + h·y = f`. With one argument `h = 0`; with a sequence `[f, h]` likewise. | Direct; checks nonsingularity. |
| `HyperellipticCurve(P, f, h)` | As above using the projective space `P` (dimension 2) as the ambient. | — |
| `HyperellipticCurveOfGenus(g, f, h)` / `…(g, f)` / `…(g, [f, h])` | The nonsingular genus-`g` curve `C : y² + h·y = f`; checks numerically that the genus is `g`, raising a runtime error otherwise. | Genus verification before construction. |
| `HyperellipticCurve(E)` | The curve `C` corresponding to an elliptic curve `E`, plus the map `E → C`. | Type change from `CrvEll`. |

*Worked examples: H125E1 (creation over **Q** and over GF(7)).*

### 125.2.2 Creation Predicates

| Intrinsic | Description |
|-----------|-------------|
| `IsHyperellipticCurve([f, h])` | For `h, f ∈ R[x]`, `R` an integral domain: `true` iff `C : y² + h·y = f` is a hyperelliptic curve; returns the curve as a second value. |
| `IsHyperellipticCurveOfGenus(g, [f, h])` | `true` iff `C : y² + h·y = f` is a hyperelliptic curve of genus `g`; returns the curve as a second value. |

### 125.2.3 Changing the Base Ring

| Intrinsic | Description |
|-----------|-------------|
| `BaseChange(C, K)` / `BaseExtend(C, K)` | For `C` over `k` and extension `K/k`: the curve over `K` via the natural inclusion of `k` in `K`. |
| `BaseChange(C, j)` / `BaseExtend(C, j)` | For a ring map `j : k → K`: the curve over `K` obtained by applying `j` to the coefficients. |
| `BaseChange(C, n)` / `BaseExtend(C, n)` | For `C` over a finite field `k` and integer `n`: the curve over the degree-`n` extension `K` of `k`. |
| `ChangeRing(C, K)` | The curve over `K` obtained by the standard coercion map `k → K`. Useful when there is no ring homomorphism (e.g. `k = Q`, `K` finite). |

*Worked example: H125E2 (`ChangeRing` from **Q** to GF(101); evaluation/pullback of a point under a model isomorphism).*

### 125.2.4 Models

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SimplifiedModel(C)` | For `C` over a field of characteristic ≠ 2: an isomorphic curve `C'` of the form `y² = f(x)`, plus the isomorphism `C → C'`. | Completing the square. |
| `HasOddDegreeModel(C)` | `true` if `C` has a model `C' : y² = f(x)` with `f` of odd degree; returns `C'` and `C → C'`. | — |
| `IntegralModel(C)` | For `C` over **Q**: an isomorphic curve `C'` with integral coefficients, plus `C → C'`. Parameter `Reduce` (default `false`): if `true`, common divisors of the coefficients are removed as far as possible. | Clearing denominators. |
| `MinimalWeierstrassModel(C)` | For `C` over **Q**: a globally minimal Weierstrass model `C'`, plus `C → C'`. Parameter `Bound` (default 0): upper bound on bad primes checked (trial division, so should not exceed ≈ 10⁷). | Globally minimal Weierstrass reduction. |
| `pIntegralModel(C, p)` | For `C` over **Q** or a rational function field: a model integral at place `p` (integer/rational/polynomial/rational function/∞), plus `C → C'`. | Local integralisation. |
| `pNormalModel(C, p)` | As above, a model normal at place `p`. | Local normalisation. |
| `pMinimalWeierstrassModel(C, p)` | As above, a Weierstrass model minimal at place `p`. | Local minimisation. |
| `ReducedModel(C)` | For `C` with integral coefficients: a reduced model `C'`. Parameters: `Simple` (default `false`), `Al` (default `"Stoll"`; `"Wamelen"` requires genus 2). Stoll reduces w.r.t. the action of SL₂(**Z**) on the `(x, z)`-coordinates; the isomorphism `C → C'` is returned only for the Stoll algorithm. | Reduction under SL₂(**Z**) (Stoll) or Wamelen's genus-2 algorithm **[Wam99, Wam01]**. |
| `ReducedMinimalWeierstrassModel(C)` | For `C` over **Q**: a globally minimal integral Weierstrass model reduced under SL₂(**Z**) (Stoll's `ReducedModel`), plus `C → C'`. Parameter `Simple` (default `false`). | Combines `MinimalWeierstrassModel` and Stoll reduction. |
| `SetVerbose("CrvHypReduce", v)` | Sets the verbose level (0–3, or boolean) for the Stoll/Wamelen reduction algorithms. | — |

### 125.2.5 Predicates on Models

| Intrinsic | Description |
|-----------|-------------|
| `IsSimplifiedModel(C)` | `true` if `C` is of the form `y² = f(x)`. |
| `IsIntegral(C)` | `true` if `C` has integral coefficients. |
| `IspIntegral(C, p)` | For `C` over **Q** or a rational function field: `true` if the given model is integral at place `p`. |
| `IspNormal(C, p)` | `true` if the given model is normal at place `p`. |
| `IspMinimal(C, p)` | Decides whether the model is minimal at place `p`. Returns `false, false` (not an integral minimal model); `true, false` (integral and minimal but not the unique minimal model up to local-ring-invertible transformations); `true, false` (the unique integral minimal model up to such transformations). |

### 125.2.6 Twisting Hyperelliptic Curves

Standard quadratic twists in characteristic ≠ 2; in addition the Lercier–Ritzenthaler package
returns *all* twists of a genus-2 (or genus-3) hyperelliptic curve over a finite field of any
characteristic.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `QuadraticTwist(C, d)` | For `C` over a field `k` of characteristic ≠ 2 and `d` coercible into `k`: the quadratic twist of `C` by `d`. | — |
| `QuadraticTwist(C)` | For `C` over a finite field `k`: the standard quadratic twist over the degree-2 extension (for odd characteristic, the twist by a primitive element of `k`). | — |
| `QuadraticTwists(C)` | For `C` over a finite field of odd characteristic: a sequence of the non-isomorphic quadratic twists. | — |
| `IsQuadraticTwist(C, D)` | For `C`, `D` over a common field `k` (characteristic ≠ 2): `true` iff `C` is a quadratic twist of `D`; if so returns the twisting factor. Verbose `CrvHypIso`. | — |
| `Twists(C)` | For `C` a genus-2 or genus-3 curve over a finite field `k`: a sequence of representatives of all isomorphism classes of curves over `k` that become isomorphic to `C` over `k̄`, plus the abstract geometric automorphism group as a permutation group. Genus 2: any characteristic. Genus 3: characteristic ≥ 11 and `C` of the form `y² = f(x)`. There is also a version taking the sequence `GI` of Cardona-Quer-Nart-Pujola invariants. | Lercier–Ritzenthaler twist enumeration. |
| `HyperellipticPolynomialsFromShiodaInvariants(JI)` | All twists of a genus-3 curve, and its geometric automorphism group, corresponding to a sequence of Shioda invariants `JI` over a finite field of characteristic ≥ 11. The first return is a sequence of degree-7/8 polynomials `f(x)` with twisted curves `y² = f(x)` (`JI` may be a *singular* invariant set, giving `f` with discriminant zero). | Lercier–Ritzenthaler genus-3 package. |

*Worked examples: H125E3 (quadratic twists over GF(7)); H125E4 (twist over **Q**, isomorphism over **Q**(√7)); H125E5 (all twists of a supersingular genus-2 curve over **F₂**, geometric automorphism group of order 160).*

### 125.2.7 Type Change Predicates

| Intrinsic | Description |
|-----------|-------------|
| `IsEllipticCurve(C)` | `true` iff `C` is a genus-one hyperelliptic curve of odd degree; if so returns an isomorphic elliptic curve `E`, plus `C → E` and the inverse `E → C`. |

---

## 125.3 Operations on Curves

### 125.3.1 Elementary Invariants

| Intrinsic | Description |
|-----------|-------------|
| `HyperellipticPolynomials(C)` | The polynomials `f(x), h(x)` (in that order) defining `C : y² + h·y = f`. |
| `Degree(C)` | The degree of the hyperelliptic curve `C` (or a pointset of one). |
| `Discriminant(C)` | The discriminant of `C`. |
| `Genus(C)` | The genus of `C`. |

### 125.3.2 Igusa Invariants

The Clebsch, Igusa–Clebsch and Igusa invariants may be computed for genus-2 curves.
Rodriguez-Villegas's routines are based on Mestre **[Mes91]**, which summarises classical
Clebsch/Igusa invariant theory. Three families are computed from a quintic/sextic `f` (binary
sextic form): the **Clebsch invariants** `A, B, C, D` (Mestre p. 317); the **Igusa–Clebsch
invariants** `A', B', C', D'` (p. 319, also written `I₂, I₄, I₆, I₁₀`); and the **Igusa
invariants** (Igusa J-invariants) `J₂, J₄, J₆, J₈, J₁₀` (p. 324). The functions are
`ClebschInvariants`, `IgusaClebschInvariants`, `JInvariants` (with `IgusaInvariants` a synonym
for `JInvariants`). Igusa invariants live in weighted projective space (weights 2, 4, 6, 8, 10)
and are available for all coefficient rings; the polynomial's coefficient ring must be an
algebra over a field of characteristic ≠ 2 or 3 for the Igusa–Clebsch case (not 2, 3, 5 for
Clebsch via Überschiebungen). The **Cardona-Quer-Nart-Pujola invariants** `g₁, g₂, g₃` are
three absolute invariants giving an affine classification of genus-2 curves up to isomorphism
over `k̄`; in odd/zero characteristic they are derived from the J-invariants **[CQ05]**, in
characteristic 2 from **[CNP05]** (field `k` must be perfect).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ClebschInvariants(C)` | For `C` of genus 2 over a field of characteristic ≠ 2, 3, 5: the Clebsch invariants `A, B, C, D` (Mestre p. 317). | Überschiebungen **[Mes91]**. |
| `ClebschInvariants(f)` | For `f` of degree ≤ 6 over a field of characteristic ≠ 2, 3, 5: the Clebsch invariants. | Überschiebungen **[Mes91]**. |
| `IgusaClebschInvariants(C: parameters)` | For `C` of genus 2: the Igusa–Clebsch invariants `A', B', C', D'` (p. 319; all zero in characteristic 2). Parameter `Quick` (default `false`): if `true`, characteristic must avoid 2, 3, 5 and Überschiebungen are used; else universal formulae. | **[Mes91]**. |
| `IgusaClebschInvariants(f, h)` | For `h` of degree ≤ 3, `f` of degree ≤ 6: the Igusa–Clebsch invariants of `y² + h·y − f = 0`. | **[Mes91]**. |
| `IgusaClebschInvariants(f: parameters)` | For `f` of degree ≤ 6 over a ring in which 2 is a unit. Parameter `Quick` as above. | **[Mes91]**. |
| `IgusaInvariants(C: parameters)` / `JInvariants(C: parameters)` | For `C` of genus 2: the Igusa invariants `J₂,…,J₁₀` (p. 324). Parameter `Quick` (default `false`). | **[Mes91]**. |
| `IgusaInvariants(f, h)` / `JInvariants(f, h)` | For `h` of degree ≤ 3, `f` of degree ≤ 6: the Igusa invariants of `y² + h·y = f`. The coefficient ring `R` must have characteristic 2 or admit `ExactQuotient(n, 2)`; otherwise use `ScaledIgusaInvariants`. | **[Mes91]**. |
| `IgusaInvariants(f: parameters)` / `JInvariants(f: parameters)` | For `f` of degree ≤ 6 over a ring in which 2 is a unit. Parameter `Quick` (default `false`). | **[Mes91]**. |
| `ScaledIgusaInvariants(f, h)` | The Igusa J-invariants of `y² + h·y = f`, scaled by `[16, 16², 16³, 16⁴, 16⁵]`. | **[Mes91]**. |
| `ScaledIgusaInvariants(f)` | For `f` of degree ≤ 6, characteristic ≠ 2: the Igusa J-invariants scaled by `[16, 16², 16³, 16⁴, 16⁵]`. | **[Mes91]**. |
| `AbsoluteInvariants(C)` | For `C` of genus 2: the ten absolute invariants (Mestre p. 325). | **[Mes91]**. |
| `ClebschToIgusaClebsch(Q)` | Convert Clebsch invariants in sequence `Q` to Igusa–Clebsch invariants. | — |
| `IgusaClebschToIgusa(S)` | Convert Igusa–Clebsch invariants in sequence `S` to Clebsch invariants. | — |
| `G2Invariants(C)` | The three Cardona-Quer-Nart-Pujola invariants of genus-2 `C`. | **[CQ05, CNP05]**. |
| `G2ToIgusaInvariants(GI)` | Convert Cardona-Quer-Nart-Pujola invariants to Igusa J-invariants. | — |
| `IgusaToG2Invariants(JI)` | Convert Igusa J-invariants to Cardona-Quer-Nart-Pujola invariants. | — |

### 125.3.3 Shioda Invariants

The Shioda invariants may be computed for genus-3 curves in characteristic 0 or ≥ 11. There are
9 of them (weights 2, 3, 4, 5, 6, 7, 8, 9, 10): the first 6 are algebraically independent, the
last 3 algebraic over the field they generate. The discriminant is *not* one of them. See
**[LR12]** or **[Shi67]**. Also includes the 6 Maeda invariants (**[Mae90]**). Contributed by
Lercier and Ritzenthaler.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ShiodaInvariants(C: parameters)` | The 9 Shioda invariants of genus-3 `C` (base field characteristic 0 or ≥ 11), plus their weights `[2,…,10]`. Versions take a single `f` (degree ≤ 8, curve `y² = f`) or `[f, h]` (curve `y² + h·y = f`); if `deg f` (resp. `f − h²/4`) < 7 or the discriminant is zero the result is *singular* (binary-octic invariants, not genus-3). Parameter `normalize` (default `false`): if `true`, the sequence is scaled to a normalised point in weighted projective space (so isomorphic curves give the same normalised invariants). | **[LR12, Shi67]**. |
| `ShiodaInvariantsEqual(V1, V2)` | `true` iff sequences `V1`, `V2` of Shioda invariants represent the same point in weighted projective space (i.e. the same isomorphism class for non-singular invariants). | Comparison of normalised forms. |
| `DiscriminantFromShiodaInvariants(JI)` | The discriminant (of a binary octic) from a sequence `JI` of Shioda invariants — a polynomial in the 9 invariants, scaled if the invariants are scaled. Zero iff `JI` are *singular*. | **[LR12]**. |
| `ShiodaAlgebraicInvariants(FJI: parameters)` | Given a sequence `FJI` of 6 elements (the first 6, algebraically independent, Shioda invariants) over a field of characteristic 0 or ≥ 11: if `ratsolve := true` (default), all sequences of the full 9 invariants having `FJI` as their first 6; if `false`, the 6 polynomials in 3 variables defining the dimension-0, degree-5 system whose solutions are the possible last 3 invariants. | **[LR12]**. |
| `MaedaInvariants(C)` | The six Maeda field invariants `(I2, I3, I4, I4p, I8, I9)` of genus-3 `C` (characteristic 0 or ≥ 11; model in simplified `y² = f(x)` form). A version with argument `f` (degree ≤ 8) homogenises `f` to degree 8. | **[Mae90]**. |

*Worked example: H125E6 (`ShiodaInvariants` and `ShiodaAlgebraicInvariants` over GF(37), with and without `ratsolve`).*

### 125.3.4 Base Ring

| Intrinsic | Description |
|-----------|-------------|
| `BaseField(C)` / `BaseRing(C)` / `CoefficientRing(C)` | The base field of the hyperelliptic curve `C`. |

---

## 125.4 Creation from Invariants

Construct a genus-2 curve from a given set of Igusa–Clebsch invariants over their field of
moduli. Mestre **[Mes91]** shows this is not always possible; over a finite field or **Q** his
algorithm decides feasibility and, when possible, finds such a curve (implemented by P. Gaudry).
Mestre's algorithm fails when the Jacobian splits; Cardona–Quer **[CQ05]** then give a curve
over the field of moduli in any characteristic ≠ 2, 3, 5. The Lercier–Ritzenthaler package
produces a genus-2 curve from Cardona-Quer-Nart-Pujola invariants in any characteristic (using
**[CNP05]** for characteristic 2), and a genus-3 curve from Shioda invariants in characteristic
≠ 2, 3, 5, 7. Over characteristic zero the equations can have huge coefficients; for curves over
**Q**, Wamelen's algorithm **[Wam99, Wam01]** can reduce them.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HyperellipticCurveFromIgusaClebsch(S)` | A genus-2 curve with given Igusa–Clebsch invariants over the field `F` (rationals, a number field, or a finite field of characteristic > 5). If no curve exists over `F`, a curve over a quadratic extension is returned (when `F = Q`) or an error (number field). Parameter `Reduce` (default `false`): over **Q**, invokes Wamelen's reduction (equivalent to `ReducedModel` with `Al := "Wamelen"`). | Mestre's algorithm (and Cardona's for non-hyperelliptic involutions) **[Mes91, CQ05]**. |
| `HyperellipticCurveFromG2Invariants(S)` | A genus-2 curve with given Cardona-Quer-Nart-Pujola absolute invariants over a finite field or **Q** (all characteristics), plus the geometric automorphism group as a finitely-presented group. | **[CQ05, CNP05]**. |
| `HyperellipticCurveFromShiodaInvariants(JI)` | From 9 Shioda invariants over **Q** or a finite field `k`: a genus-3 curve over `k`, plus the abstract geometric automorphism group as a permutation group. Errors if `JI` are singular. | **[LR12]**. |
| `HyperellipticPolynomialFromShiodaInvariants(JI)` | A polynomial `f` of degree ≤ 8 with the given Shioda invariants (also works for singular invariants); if `JI` non-singular, `y² = f(x)` is a genus-3 curve with invariants `JI`. | **[LR12]**. |

*Worked example: H125E7 (`HyperellipticCurveFromIgusaClebsch` with Wamelen reduction; characteristic-2 genus-2 from G2 invariants; genus-3 from Shioda invariants).*

---

## 125.5 Function Field

### 125.5.1 Function Field and Polynomial Ring

| Intrinsic | Description |
|-----------|-------------|
| `FunctionField(C)` | The function field of the hyperelliptic curve `C`. |
| `DefiningPolynomial(C)` | A weighted homogeneous polynomial for `C`. |
| `EvaluatePolynomial(C, a, b, c)` / `EvaluatePolynomial(C, [a, b, c])` | Evaluates the homogeneous defining polynomial of `C` at the point `(a, b, c)`. |

---

## 125.6 Points

The curve is embedded in weighted projective space with weights `1, g+1, 1` on `x, y, z`, so
point triples satisfy `(x : y : z) = (μx : μ^{g+1}y : μz)`; points at infinity are normalised to
`(1 : y : 0)`.

### 125.6.1 Creation of Points

| Intrinsic | Description |
|-----------|-------------|
| `C ! [x, y]` / `C ! [x, y, z]` / `elt< PS | x, y >` / `elt< PS | x, y, z >` | The point `(x, y, z)` on `C` (or on its pointset `PS`); if `z` is omitted it is 1. |
| `C ! P` | For a point `P` on `C₁` with `C` a base extension of `C₁`: the corresponding point on `C` (e.g. reduction to finite characteristic, or the tautological coercion). |

### 125.6.2 Random Points

| Intrinsic | Description |
|-----------|-------------|
| `Points(C, x)` / `RationalPoints(C, x)` | The indexed set of all rational points on `C` with given `x`-coordinate (points at infinity have `x`-coordinate ∞). |
| `PointsAtInfinity(C)` | The points at infinity of `C` as an indexed set. |
| `IsPoint(C, S)` | `true` iff the sequence `S` specifies a point on `C`; if so returns the point. |

*Worked example: H125E8 (the point at infinity on `y² = x⁵ + 1`, weighted ambient, `IsNonSingular`).*

### 125.6.2 Random Points

| Intrinsic | Description |
|-----------|-------------|
| `Random(C)` | A random point on `C` over a finite field. If all points are already known the result is truly random; otherwise the ramification points have a slight advantage. |

### 125.6.3 Predicates on Points

| Intrinsic | Description |
|-----------|-------------|
| `P eq Q` | `true` iff `P`, `Q` on the same curve have the same coordinates. |
| `P ne Q` | `false` iff `P`, `Q` have the same coordinates. |

### 125.6.4 Access Operations

| Intrinsic | Description |
|-----------|-------------|
| `P[i]` | The `i`-th coordinate of `P` (`1 ≤ i ≤ 3`). |
| `Eltseq(P)` / `ElementToSequence(P)` | The 3-element sequence of coordinates of `P`. |

### 125.6.5 Arithmetic of Points

| Intrinsic | Description |
|-----------|-------------|
| `-P` / `Involution(P)` | The image of `P` under the hyperelliptic involution. |

### 125.6.6 Enumeration and Counting Points

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `NumberOfPointsAtInfinity(C)` | The number of points at infinity on `C`. | — |
| `PointsAtInfinity(C)` | The points at infinity as an indexed set. | — |
| `#C` | For `C` over a finite field: the number of rational points. | Naive count for small base fields; otherwise the faster p-adic methods of the `#J` section (which yield the full zeta function). Verbose `JacHypCnt`. |
| `Points(C)` / `RationalPoints(C)` | For `C` over a finite field: the indexed set of all rational points. For `C` over **Q** of the form `y² = f(x)` with integral coefficients: points whose `x`-coordinate has naive height < `Bound`. Parameters: `Bound`, `NPrimes` (default 30), `DenominatorBound` (default `Bound`). | Over a number field, a sieve method (Appendix A of **[Bru02]**); `NPrimes` controls the number of primes, `DenominatorBound` the denominator size. |
| `PointsKnown(C)` | `true` iff the points of `C` have already been computed. | — |
| `ZetaFunction(C)` | For `C` over a finite field: the zeta function, as an element of the rational function field over **Z**. | Naive count over extensions of degree `1,…,g`, or the faster p-adic methods of `#J`. Verbose `JacHypCnt`. |
| `ZetaFunction(C, K)` | For `C` over **Q** with good reduction at the characteristic of `K`: the zeta function of the base extension of `C` to `K`. | — |

*Worked example: H125E9 (Diophantus's curve `y² = x⁶ + x² + 1`, `Points` with `Bound`, Wetherell **[Wet97]**).*

### 125.6.7 Frobenius

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Frobenius(P, F)` | Applies the Frobenius `x ↦ x^{#F}` to `P`. Parameter `Check` (default `true`): verifies the curve is defined over the finite field `F`. | — |

---

## 125.7 Isomorphisms and Transformations

A hyperelliptic-curve isomorphism is given by a linear fractional transformation
`t(x : z) = (ax + bz : cx + dz)`, a scale factor `e`, and a polynomial `u(x)` of degree ≤ `g+1`,
acting by `(x : y : z) ↦ (ax + bz : ey + ũ(x, z) : cx + dz)` where `ũ` is the degree-`g+1`
homogenisation of `u`. When unspecified, `e = 1` and `u = 0`. Isomorphisms are created by
coercing a tuple `⟨[a, b, c, d], e, u⟩` into the structure of isomorphisms, or as a
transformation of a given curve. Equal isomorphisms may have different representations due to the
projective weighting.

### 125.7.1 Creation of Isomorphisms

| Intrinsic | Description |
|-----------|-------------|
| `Aut(C)` | The structure of all automorphisms of `C`. |
| `Iso(C1, C2)` | The structure of all isomorphisms between `C1` and `C2` (same genus and base field). |
| `Transformation(C, t)` / `Transformation(C, u)` / `Transformation(C, e)` / `Transformation(C, e, u)` / `Transformation(C, t, e, u)` | The codomain curve `C'` of the isomorphism from `C` specified by ring-element sequence `t`, ring element `e`, and polynomial `u`, plus the isomorphism. |

*Worked example: H125E10 (`Transformation` of `y² = x⁵ − 7`; `IsIsomorphic`).*

### 125.7.2 Arithmetic with Isomorphisms

Isomorphisms act on points as right operators; the map syntax `@` is used (`f(P)` is not
available).

| Intrinsic | Description |
|-----------|-------------|
| `f * g` | The composition of the maps `f`, `g` as right operators. |
| `Inverse(f)` | The inverse of the isomorphism `f`. |
| `f in M` | `true` iff `f` and the isomorphism structure `M` share the same domains and codomains. |
| `P @ f` / `Evaluate(f, P)` | The evaluation of `f` at `P` (also for points of the Jacobian and, in genus 2, the Kummer surface). |
| `P @@ f` / `Pullback(f, P)` | The inverse image of `f` at `P` (likewise for Jacobian/Kummer points). |
| `f eq g` | `true` iff `f`, `g` (same domain/codomain) are equal (possibly with distinct defining data). |

### 125.7.3 Invariants of Isomorphisms

| Intrinsic | Description |
|-----------|-------------|
| `Parent(f)` | The "parent structure" of `f` (all isomorphisms with the same domain and codomain). |
| `Domain(f)` | The domain curve of `f`. |
| `Codomain(f)` | The codomain (target) curve of `f`. |

### 125.7.4 Automorphism Group and Isomorphism Testing

`IsGL2Equivalent` is central to isomorphism testing. The Lercier–Ritzenthaler genus-2 package
computes the geometric automorphism group in any characteristic via Cardona-Quer-Nart-Pujola
invariants (replacing an old function restricted to odd/0 characteristic); the genus-3 package
does the same via Shioda invariants in characteristic 0 or ≥ 11.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsGL2Equivalent(f, g, n)` | `true` iff `f`, `g` are in the same GL₂(k)-orbit modulo scalars, viewed as homogeneous degree-`n` polynomials (`n ≥ 4`). Second return: the sequence of matrix entries `[a, b, c, d]` such that `g(x)` is a constant times `f((ax+b)/(cx+d))·(cx+d)ⁿ`. | GL₂-orbit test. |
| `IsIsomorphic(C1, C2)` | `true` iff `C1`, `C2` are isomorphic over their common base field; if so returns an isomorphism. Verbose `CrvHypIso`. | — |
| `AutomorphismGroup(C)` | For `C` of characteristic ≠ 2 and genus ≥ 1: a permutation group, an isomorphism to the group of automorphisms of `C` over its base ring (those commuting with the hyperelliptic involution), and the action map `C × G → C`. | — |
| `GeometricAutomorphismGroup(C)` | For `C` of genus 2 or 3: a finitely-presented group isomorphic to the geometric automorphism group (automorphisms over `k̄`). Genus 2 uses Cardona-Quer-Nart-Pujola invariants and the classification **[SV01, CQ05, CNP05]**; genus 3 uses Shioda invariants (characteristic 0 or ≥ 11). A genus-2 version takes the sequence `GI` of invariants instead of the curve. | **[SV01, CQ05, CNP05]**. |
| `GeometricAutomorphismGroupFromShiodaInvariants(JI)` | Genus-3 variant taking the Shioda-invariants sequence `JI` instead of the curve (same characteristic restrictions). | **[LR12]**. |
| `GeometricAutomorphismGroupGenus2Classification(F)` | For a finite field `F` (any characteristic): two sequences — the list of all possible geometric automorphism groups for genus-2 curves over `F` (as finitely-presented groups), and the number of `k̄`-isomorphism classes of `F`-curves with each group. | Classification of **[Car03, CNP05]**. |
| `GeometricAutomorphismGroupGenus3Classification(F)` | For a finite field `F` of characteristic ≥ 11: two sequences — the list of all possible geometric automorphism groups for genus-3 curves over `F` (as permutation groups), and the number of `k̄`-isomorphism classes with each. | Lercier–Ritzenthaler classification. |

*Worked examples: H125E11 (automorphism group of a genus-1 supersingular curve in characteristic 3; base extension to find more automorphisms); H125E12 (`GeometricAutomorphismGroup` of genus-2 and genus-3 curves; comparison with `AutomorphismGroup` over an algebraic closure); H125E13 (genus-2 classification over GF(2)).*

---

## 125.8 Jacobians

The Jacobian is implemented as the divisor class group of the curve; no equations for it ever
appear. Created for any hyperelliptic curve, but the interesting functionality is over finite
fields, or for genus 2 over number fields or **Q**.

### 125.8.1 Creation of a Jacobian

| Intrinsic | Description |
|-----------|-------------|
| `Jacobian(C)` | The Jacobian of `C`. |

### 125.8.2 Access Operations

| Intrinsic | Description |
|-----------|-------------|
| `Curve(J)` | The curve from which `J` was constructed. |
| `Dimension(J)` | The dimension of `J` (equal to the genus of the curve). |

### 125.8.3 Base Ring

| Intrinsic | Description |
|-----------|-------------|
| `BaseField(J)` / `BaseRing(J)` / `CoefficientRing(J)` | The base field of `J`. |

### 125.8.4 Changing the Base Ring

| Intrinsic | Description |
|-----------|-------------|
| `BaseChange(J, F)` / `BaseExtend(J, F)` | The base extension of `J` to the field `F`. |
| `BaseChange(J, j)` / `BaseExtend(J, j)` | The base extension by the ring homomorphism `j` (domain = base field of `C`). |
| `BaseChange(J, n)` / `BaseExtend(J, n)` | The base extension over a finite field to its degree-`n` extension. |

---

## 125.9 Richelot Isogenies

Let `k` have characteristic ≠ 2 and `C : y² = f(x)` a genus-2 curve with `f` square-free of
degree 5 or 6, `J = Jac(C)`. A *Richelot isogeny* is a polarised isogeny `Φ : J → A` of
principally polarised abelian surfaces whose kernel over `k̄` has group structure
`Z/2Z × Z/2Z`; `J[Φ] ⊂ J[2]` is maximal isotropic for the Weil pairing. Writing
`f = c·Q₁Q₂Q₃` (the `Qᵢ` of degree 2, or `Q₁` of degree 1 representing a root at ∞ when
`deg f = 5`), the kernel is `{0, [Q₁=0]−[Q₂=0], [Q₁=0]−[Q₃=0], [Q₂=0]−[Q₃=0]}`. The `Qᵢ` need
not be individually defined over `k`; one may write `f = c·Norm_{L[x]/k[x]} Q(x)` with
`L = k[t]/(h(t))` for a square-free cubic `h`. See **[Smi05]** Ch. 8 and **[BD09]**. In special
cases the codomain `A` can be a product of elliptic curves or the Weil restriction of an
elliptic curve.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RichelotIsogenousSurfaces(J)` / `RichelotIsogenousSurfaces(C)` | The Richelot isogenies defined over the base field; returns a list of codomain objects (a Jacobian, or a Cartesian product of elliptic curves, or an elliptic curve over a quadratic extension for a Weil restriction). Parameter `Kernels` (default `true`): if set, also a list of quadratic polynomials over cubic algebras describing the kernels. | **[Smi05, BD09]**. |
| `RichelotIsogenousSurface(J, kernel)` / `RichelotIsogenousSurface(C, kernel)` | The codomain for a genus-2 Jacobian and a Richelot kernel (quadratic `Q(x)` over a cubic algebra `L` with `Norm_{L[x]/k[x]} Q = c·f(x)`); kernel descriptions are those returned by `RichelotIsogenousSurfaces` with `Kernels:=true`. | **[Smi05, BD09]**. |

*Worked example: H125E14 (Richelot isogenies on `Jac(y² = x⁵ + x)` — all three codomain types realised; kernel descriptions; L-series agreement via `ExcFactors:="Ogg"`).*

---

## 125.10 Points on the Jacobian

Points on `Jac(C)` are divisors on `C`, specified by points/divisors on `C` or in Mumford
representation `⟨a(x), b(x), d⟩` (the form Magma stores). The triple specifies the degree-`d`
divisor `D` defined by `A(x, z) = 0, y = B(x, z)` (with `A`, `B` the degree-`d` and degree-`(g+1)`
homogenisations), and the point on the Jacobian is `D` minus a multiple of the divisor at
infinity. There is a uniquely determined reduced triple representing each point. Magma cannot
represent points when `g` is odd and there are 0 or 2 rational points at infinity (the "bad"
cases); for 2 points at infinity a unique default representative is chosen internally. Technical
conditions: `a` monic of degree ≤ `g`; `b` of degree ≤ `g+1` with `a | b² + h·b − f`; `d` a
positive integer with `deg(a) ≤ d ≤ g+1` and `deg(b² + h·b − f) ≤ 2g+2−d+deg(a)`.

### 125.10.1 Creation of Points

| Intrinsic | Description |
|-----------|-------------|
| `J ! 0` / `Id(J)` / `Identity(J)` | The identity element on `J`. |
| `J ! [a, b]` / `elt< J | a, b >` / `elt< J | [a, b] >` / `elt< J | a, b, d >` / `elt< J | [a, b], d >` | The point on `J` defined by `a`, `b` and the positive integer `d` (default `deg(a)`). |
| `P - Q` / `J ! [P, Q]` / `elt< J | P, Q >` | For points `P`, `Q` on the curve: the image of the divisor class `[P − Q]`. |
| `J ! [S, T]` / `elt< J | S, T >` | For sequences `S = [Pᵢ]`, `T = [Qᵢ]` of curve points (each length `n`): the image of `Σ[Pᵢ] − Σ[Qᵢ]`. |
| `JacobianPoint(J, D)` | The point on `J` associated to the divisor `D` on `C` (subtracting a suitable multiple of the divisor at infinity if `deg D ≠ 0`; `D` must have even degree when the divisor at infinity has even degree). Not implemented in characteristic 2. |
| `J ! P` | For a point `P` on a Jacobian `J'` with `J` a base extension of `J'`: the image of `P` on `J`. |
| `Points(J, a, d)` / `RationalPoints(J, a, d)` | All points on `J` with first component `a` and degree `d`. Only for genus-2 curves `y² = f(x)`. |

*Worked examples: H125E15 (points on `y² = x⁶ − 3x − 1` and images on `J`); H125E16 (the nontrivial 2-torsion point on `y² = (x²+1)(x⁶+7)`); H125E17 (a point on `J` not coming from the curve, constructed via a divisor/ideal).*

### 125.10.2 Random Points

| Intrinsic | Description |
|-----------|-------------|
| `Random(J)` | A random point on `J` over a finite field. |

### 125.10.3 Booleans and Predicates for Points

| Intrinsic | Description |
|-----------|-------------|
| `P eq Q` | `true` iff `P`, `Q` on the same Jacobian are equal. |
| `P ne Q` | `false` iff `P`, `Q` are equal. |
| `IsZero(P)` / `IsIdentity(P)` | `true` iff `P` is the zero element of the Jacobian. |

### 125.10.4 Access Operations

| Intrinsic | Description |
|-----------|-------------|
| `P[i]` | For `1 ≤ i ≤ 2`, the `i`-th defining polynomial of `P`; for `i = 3`, the degree `d` of the reduced divisor. |
| `Eltseq(P)` / `ElementToSequence(P)` | The two defining polynomials of `P`, then the degree of the divisor. |

### 125.10.5 Arithmetic of Points

| Intrinsic | Description |
|-----------|-------------|
| `-P` | The additive inverse of `P`. |
| `P + Q` / `P +:= Q` | The sum of `P` and `Q` (and in-place assignment). |
| `P - Q` / `P -:= Q` | The difference `P − Q` (and in-place assignment). |
| `n * P` / `P * n` / `P *:= n` | The `n`-th multiple of `P` (and in-place assignment). |

### 125.10.6 Order of Points on the Jacobian

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Order(P)` | The order of `P` on `J` over a finite field or the rationals, or 0 if `P` has infinite order. Computes `#J` first when `J` is over a finite field. | — |
| `Order(P, l, u)` | The order of `P` where `l`, `u` bound the order of `P` (or of `J`). Does not compute `#J`. Parameters: `Alg` (default `"Shanks"`; `"PollardRho"` uses a Pollard-rho variant **[GH00]**), `UseInversion` (default `true`; halves the search space by point negation). | Shanks baby-step/giant-step or Pollard-rho **[GH00]**. |
| `Order(P, l, u, n, m)` | As above with the group order known to be `n mod m`. | Shanks/Pollard-rho **[GH00]**. |
| `HasOrder(P, n)` | `true` iff the order of `P` is `n`. | — |

### 125.10.7 Frobenius

| Intrinsic | Description |
|-----------|-------------|
| `Frobenius(P, k)` | For `P` on `J` over `k = F_q`: the image of `P` under `x ↦ x^q`. Parameter `Check` (default `true`): verifies the Jacobian is defined over `k`. |

### 125.10.8 Weil Pairing

| Intrinsic | Description |
|-----------|-------------|
| `WeilPairing(P, Q, m)` | The Weil pairing of `m`-torsion points `P`, `Q` on a 2-dimensional Jacobian `J` over a finite field. |

*Worked example: H125E18 (Weil pairing in MOV-reduction of the discrete logarithm on a supersingular Jacobian over GF(2)).*

---

## 125.11 Rational Points and Group Structure over Finite Fields

### 125.11.1 Enumeration of Points

| Intrinsic | Description |
|-----------|-------------|
| `Points(J)` / `RationalPoints(J)` | All rational points on the Jacobian `J` over a finite field. |

### 125.11.2 Counting Points on the Jacobian

Several algorithms compute `#J` depending on size, genus and type. In genus 2 the techniques of
**[GH00]** are included. The best methods are based on p-adic liftings (the defaults when
applicable and the characteristic is not too large): Kedlaya's algorithm in odd characteristic
**[Ked01]**; Mestre's canonical lift method (adapted by Lercier–Lubicz **[LL]**) and
Vercauteren's characteristic-2 version of Kedlaya **[Ver02]** in characteristic 2. The p-adic
methods yield the Euler factor. The latest p-adic methods are Lauder's deformation algorithms
over parametrised families, implemented for hyperelliptic families by Hubrechts **[Hub06]**
(see `JacobianOrdersByDeformation` etc.).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SetVerbose("JacHypCnt", v)` | Set the verbose level for Jacobian point counting (`true`/`false`/0–4). | — |
| `#J` / `Order(J)` | For `J` over a finite field: the order of the group of rational points. Parameters (every genus): `NaiveAlg` (default `false`; counts curve points over the first `g` extensions), `ShanksLimit` (default 10¹²; switches from Shanks to Pollard), `CartierManinLimit` (default 5×10⁵; maximal characteristic for the Cartier–Manin trick), `UseSubexpAlg` (default `true`). Genus-2 odd-characteristic parameters: `UseGenus2` (default `false`; enables the genus-2 methods), `UseSchoof` (default `true`), `UseHalving` (default `true`). | p-adic (Kedlaya / Vercauteren / Mestre) when applicable; else naive counting, Cartier–Manin, the subexponential function-field algorithm, or Shanks/Pollard group-order algorithms. Mestre's method requires an *ordinary* Jacobian and a group law (single point at infinity); for genus > 4 or when Mestre fails, Vercauteren's Kedlaya is used. |

*Worked examples: H125E19 (naive counting); H125E20 (Kedlaya over GF(3²⁰)); H125E21 (Vercauteren over GF(2²⁵), non-ordinary); H125E22 (Mestre's method, genus-3 ordinary over GF(2²⁵)); H125E23 (Shanks vs Pollard via `ShanksLimit`); H125E24 (`UseSchoof`/`UseHalving` in genus 2).*

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `FactoredOrder(J)` | For `J` over a finite field: the factorisation of `#J`. | — |
| `EulerFactor(J)` | The Euler factor of `J` — the reciprocal of the characteristic polynomial of Frobenius on `H¹(J)`. (Same as `ZetaFunction(C)`.) | — |
| `EulerFactorModChar(J)` | The Euler factor of `J` modulo the characteristic (not for high characteristic, `p ≤ 10⁶`). | Cartier–Manin. |
| `EulerFactor(J, K)` | For `J` over **Q** with good reduction at the finite field `K`: the Euler factor of the base extension of `J` to `K`. | — |

### 125.11.3 Deformation Point Counting

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `JacobianOrdersByDeformation(Q, Y)` / `EulerFactorsByDeformation(Q, Y)` / `ZetaFunctionsByDeformation(Q, Y)` | Compute orders (resp. Euler factors, zeta functions) of the curves `y² = Q(x, z)` for `z` specialised to each value in `Y`, in a 1-parameter family over a finite field. `Q(x, z)` must be monic of odd degree in `x`; `Y` a sequence in a finite extension `K` of the base field `k`. Efficient for several values at once (the Frobenius-matrix part is computed only once). | Lauder–Hubrechts deformation **[Hub06]**; Kedlaya over `k` initialises the deformation. Verbose `JacHypCnt`. |

*Worked example: H125E25 (Euler factors of 4 members of a linear family of elliptic curves via `EulerFactorsByDeformation`).*

### 125.11.4 Abelian Group Structure

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Sylow(J, p)` | For `J` over a finite field and prime `p`: the Sylow `p`-subgroup of `J` as an abstract abelian group `A`, the injection `A → J`, and the generators of the `p`-Sylow subgroup. | — |
| `AbelianGroup(J)` | For `J` over a finite field `K`: the group of rational points as an abstract abelian group `A`, plus the isomorphism `A → J(K)`. Parameters: `UseGenerators` (default `false`), `Generators` (a set; if `UseGenerators`, relations are extracted from these). | — |
| `HasAdditionAlgorithm(J)` | `true` iff `J` has an addition algorithm (needed for the abelian-group computation when no generators are supplied). | — |

---

## 125.12 Jacobians over Number Fields or Q

Some functions work for general number fields (notably `TwoSelmerGroup`); many only over **Q**.

### 125.12.1 Searching For Points

| Intrinsic | Description |
|-----------|-------------|
| `Points(J)` / `RationalPoints(J)` | For `J` of a genus-2 curve over **Q** (integral model): all rational points whose naive height on the associated Kummer surface is ≤ `Bound`. Parameter `Bound` (default 0). |

### 125.12.2 Torsion

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `TwoTorsionSubgroup(J)` | For `J` of genus 2, or odd degree, over a number field `K` (curve in form `y² = f(x)`): `J(K)[2]` as an abstract group, plus a map to points on `J`. | — |
| `TorsionBound(J, n)` | For `J` over **Q**: a bound on the rational torsion subgroup, from `J(F_p)` for the first `n` good primes `p`. | Reduction modulo primes. |
| `TorsionSubgroup(J)` | For `J` of a genus-2 curve over **Q** (`y² = f(x)`, integral coefficients): the rational torsion subgroup, plus the map into `J`. | — |

*Worked example: H125E26 (full 2-torsion on `y² = (x+3)(x+2)(x+1)x(x−1)(x−2)`; a curve with torsion **Z**/24).*

### 125.12.3 Heights and Regulator

Height functions on the Mordell–Weil group of `Jac(C)` over a number field `k` (naive heights
and height constants only for genus-2 Jacobians over **Q**). For genus-2 curves over **Q** the
canonical height uses Flynn–Smart **[FS97]** with Stoll's improvements **[Sto99]** (local error
functions on the Kummer surface). Otherwise the method of Chapter 5 of **[Mül10a]** is used:
based on a theorem of Faltings–Hriljac expressing the canonical height pairing via Arakelov
intersection theory (ideas due to David Holmes **[Hol06]**). Non-archimedean intersections use
regular models and Gröbner bases over p-adic quotient rings; archimedean ones use theta functions
on the analytic Jacobian (§125.18). The main non-archimedean bottleneck is integer factorisation.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `NaiveHeight(P)` | For `P` on a genus-2 Jacobian (or its Kummer surface): the logarithmic height of the image of `P` in `P³` under `J → K → P³`. | — |
| `Height(P: parameters)` / `CanonicalHeight(P: parameters)` | The canonical height of `P` on a Jacobian over a number field or **Q**. Genus 2 over **Q**: on the associated Kummer surface (Flynn–Smart/Stoll); otherwise the self-pairing via Arakelov theory. Parameters: `lambda` (default 1), `mu` (default 0), `LocalPrecision` (default 0), `UseArakelov` (default `false`), `Precision` (default 0). | **[FS97, Sto99]** (genus 2 / **Q**); else Arakelov intersection **[Mül10a, Hol06]**. |
| `HeightConstant(J: parameters)` | For `J` of a genus-2 curve over **Q** (`y² = f(x)`, integral): a real `c` with `h(P) ≤ ĥ(P) + c` for all `P ∈ J(Q)`. Parameters: `Effort` (0, 1, or 2), `Factor` (default `false`; factor the discriminant for a better bound). Second return: a bound for `μ_∞`. | Bound on `h − ĥ`. |
| `HeightPairing(P, Q: parameters)` | The canonical height pairing `⟨P, Q⟩ = (ĥ(P+Q) − ĥ(P) − ĥ(Q))/2` for rational points on `J` over a number field. Same parameters as `Height`. | **[FS97, Sto99]** / Arakelov **[Mül10a]**. |
| `HeightPairingMatrix(S: Precision)` | For a sequence `[P₁,…,Pₙ]`: the matrix with entries `⟨Pᵢ, Pⱼ⟩`. Parameter `Precision` (default 0). | — |
| `Regulator(S: Precision)` | For a sequence `S` on `J` over a number field `k`: the determinant of the height pairing matrix (zero iff the points are dependent in the Mordell–Weil group). Parameter `Precision` (default 0). | — |
| `ReducedBasis(S: Precision)` | For a sequence of points on `J` over a number field `k`: an LLL-reduced basis for the subgroup of `J(k)/J_tors(k)` generated by the points (reduces under the height pairing form), plus the height pairing matrix. Parameter `Precision` (default 0). | LLL on the canonical-height lattice. |

*Worked examples: H125E27 (properties of heights on `Jac(y² = x⁶ + x² + 2)`, showing a generator of rank-1 free quotient; continued in H125E35); H125E28 (`ReducedBasis` and height-pairing matrix on `y² = x⁶ + x² + 1`).*

### 125.12.4 The 2-Selmer Group

The Mordell–Weil group `J(K)` over **Q** or a number field; 2-descent bounds the rank since
`J(K)/2J(K)` embeds in the 2-Selmer group. `HasSquareSha` determines the parity of the 2-rank
of Ш; `RankBounds` collects all computable information. From Magma V2.13 a simpler interface to
`TwoSelmerGroup` chooses among three internal implementations.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BadPrimes(C)` / `BadPrimes(J)` | For `C` (or its Jacobian) with integral coefficients over a number field: the primes of bad reduction of the given model. Parameter `Badness` (default 1). | — |
| `HasSquareSha(J)` / `IsEven(J)` | For `J` over **Q** or a number field, assuming Ш(`J`) finite: `true` iff `#Ш(J)` is a square (otherwise twice a square, by Poonen–Stoll **[PS99]**); square iff the number of "deficient" primes is even. | **[PS99]**. |
| `IsDeficient(C, p)` | For a genus-2 curve over **Q** or a number field: `true` iff `C` is "deficient" at `p` (no points over any odd-degree extension of **Q**_p, or over **R** for `p = 0`; equivalently no **Q**_p-rational divisor of odd degree). | — |
| `HasIndexOne(C, p)` | For `C` over **Q** or a number field: `true` iff there is an odd-degree divisor on `C` defined over the completion at `p`. (An even-genus curve is deficient at `p` iff it lacks index one at `p`.) | — |
| `HasIndexOneEverywhereLocally(C)` | `true` iff `C` has index one over all completions (including real). | — |
| `TwoSelmerGroup(J)` | The 2-Selmer group of `J` over **Q** or a number field, as a finite abelian group `S`, plus a map from `S` to an affine algebra `A` (the standard map to `A*/(A*)²` or `A*/(A*)²Q*` depending on whether `deg f` is odd or even). Parameters: `Al` (`"TwoSelmerGroupOld"`, `"TwoSelmerGroupNew"`, `"TwoSelmerGroupData"`), `Fields` (precomputed class-group fields), `ReturnFakeSelmerData` (default `false`), `ReturnRawData` (default `false`), `Verbose Selmer`. With `ReturnFakeSelmerData`: an extra tuple `⟨B₁, B₂, B₃⟩` describing the "fake Selmer group" **[Sto01]**. With `ReturnRawData`: additional `expvecs` and `factorbase` specifying the images of generators in `A*/(A*)²`. | Class group / unit calculations; speed up via `SetClassGroupBounds` or precomputed `Fields`. **[Sto01]**. |
| `RankBound(J)` / `RankBounds(J)` | An upper bound (resp. a lower and upper bound) on the rank of the Mordell–Weil group of `J` over **Q** or a number field. `RankBound` needs `TwoSelmerGroup` and `TwoTorsionSubgroup`; `RankBounds` is only over **Q**. For even-degree curves the bound is for the subgroup represented by rational divisors; a warning is printed if the curve lacks index one everywhere locally. Can be sharpened: if Ш has non-square order the bound drops by 1 (`HasSquareSha`); if `T` lies in Ш[2] and is not divisible by 2 (algorithm of **[Cre12]**) the bound drops by 2. The lower bound searches for points and tests independence via the height pairing or images in `J(Q)/2J(Q)`. | 2-descent **[Sto01, Cre12]**. |

*Worked examples: H125E29 (Mordell–Weil group of `y² = x(x+1344²)(x+10815²)(x+5406²)(x+2700²)`; full 2-torsion, `#TwoSelmerGroup`, `HasSquareSha`); H125E30 (Ш twice a square for `y² = 3(x⁶ − x² + a)`, `RankBound`); H125E31 (non-trivial Ш on a genus-2 curve; quadratic twist with high rank, `SetClassGroupBounds`); H125E32 (rank of genus-3 Jacobian with Ш[2] via `TwoSelmerSet`/`RankBounds`); H125E33 (even-degree odd-genus curve without index one everywhere locally).*

---

## 125.13 Two-Selmer Set of a Curve

Let `C : y² = f(x)` be a genus-`g` curve over a number field `k`. A *two-cover* `π_δ : D_δ → C`
is a cover isomorphic (over `k̄`) to the pullback of the embedding of `C` into its Jacobian along
multiplication by 2. The two-Selmer set classifies `k`-isomorphism classes of two-covers with
points everywhere locally. The involution acts by `ι*(π_δ) = π_δ ∘ ι`; the *fake* two-Selmer set
is `{δ : D_δ(k_v) ≠ ∅ for all places v}/ι*`. If empty, `C` has no `k`-rational points (possible
even when `C` has points everywhere locally). See **[BS09]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `TwoCoverDescent(C)` | The fake 2-Selmer set as an abstract set, plus a map giving a representation as elements in an algebra (to construct the corresponding cover explicitly). Parameters: `Bound`, `Fields`, `Raw` (as for `TwoSelmerGroup`); `PrimeBound` (default 0; restrict good primes considered to those of norm ≤ bound — may give a larger set); `PrimeCutoff` (default 0; restrict bad primes of large norm — may give a larger set). | **[BS09]**. |

*Worked example: H125E34 (genus-2 curves with empty 2-Selmer set hence no rational points; a rank-2 example combining `TwoCoverDescent`, factorisation of `f`, `PseudoMordellWeilGroup`, elliptic-curve Chabauty).*

---

## 125.14 Chabauty's Method

A method for finding rational points on a curve `C` of genus ≥ 2 when the rank of `Jac(C)` is
less than the genus, via local calculations at a prime of good reduction. Two routines for
genus-2 curves over **Q**: an older one (user supplies the prime, results usually not precise),
and a powerful new one combining Chabauty calculations with a "Mordell–Weil sieve" over many
primes to determine `C(Q)` precisely. The new routine requires a known rational point and a
generator of the (rank-1) Mordell–Weil group. `Chabauty0` handles the rank-0 case.
Under rank < genus, the closure `V` of `J(Q)` in `J(Q_p)` is a locus where certain power series
vanish; intersecting the image of `C` with `V` bounds `#C(Q)` per residue class.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Chabauty0(J)` | For `J` of a curve `C` over **Q** with `J(Q)` of rank 0: all rational points on `C` (assumption not checked; if it fails, a subset of `C(Q)` corresponding to the torsion is returned). | Rank-0 Chabauty. |
| `Chabauty(P: ptC)` | For a curve `C` of genus 2 over **Q**: the full set of rational points, via Chabauty's method with a Mordell–Weil sieve. `P` should generate the rank-1 Mordell–Weil group; needs one known rational point (supply via `ptC`, else found by searching). | Chabauty + Mordell–Weil sieve. |
| `Chabauty(P, p: Precision)` | Uses Chabauty at the prime `p` to bound the number of rational points on `C` (genus 2, `y² = f(x)`, integral). Good reduction at `p`; `P` generates a subgroup of index coprime to `p` in `J(Q)/J_tors(Q)`. Returns an indexed set of tuples `⟨x, z, v, k⟩`: at most `k` pairs of rational points have image congruent to `(x:z)` mod `p^v` (apart from Weierstrass points). Parameter `Precision` (default 5). | Chabauty at `p`. |

*Worked examples: H125E35 (both Chabauty implementations prove `y² = x⁶ + x² + 2` has only 6 rational points); H125E36 (precisely 6 rational points on `y² = x⁶ + 8`); H125E37 (rank-2 curve from H125E29: points whose image lies in a rank-1 subgroup); H125E38 (`Chabauty0` on rank-0 `y² = x⁶ + 1`).*

---

## 125.15 Cyclic Covers of P¹

A `q`-cyclic cover of P¹ is a curve `C` admitting a degree-`q` morphism to P¹ with geometrically
cyclic function-field extension. By Kummer theory it has an affine model `y^q = f(x)` (when the
base field contains the `q`-th roots of unity); this class includes hyperelliptic curves. Current
functionality: local solubility testing, point searching, descents on curves and their Jacobians
over a number field. Most functionality requires `f(x)` separable and `q` prime; descent on the
curve is available for any `q`-th-power-free `f(x)`.

### 125.15.1 Points

A point on `C : y^q = f_d x^d + … + f_0` is a triple in the appropriate weighted projective plane
(`P²(1 : d/q : 1)` when `q | d`, else `P²(q : d : 1)`).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RationalPoints(f, q)` | For `f` over a number field (or **Q**): searches for rational points on `y^q = f(x)` in a box bounded by `Bound` (must be specified). Returns triples `(x, y, z)` satisfying `y^q = F(x, z)` with `F` the homogenisation taking `x` to weight `LCM(q, deg f)/deg f`. Parameters: `Bound` (default 0), `DenominatorBound` (default 0), `NPrimes` (default 0). | Sieve method, Appendix A of **[Bru02]**. |
| `HasPoint(f, q, v)` | For `f` over a number field `k`: checks if `y^q = f(x)` has points over the completion `k_v`. Second return: a triple in the completion giving a point. | Local solubility (polynomial time in residue-field cardinality). |
| `HasPointsEverywhereLocally(f, q)` | `true` iff `y^q = f(x)` has points over all non-archimedean completions of `k`. | — |

### 125.15.2 Descent

Let `C : y^q = f(x)` be a `q`-cyclic cover over a number field `k`. A primitive `q`-th root of
unity `ζ` gives the automorphism `ι : (x, y) ↦ (x, ζy)`, inducing `φ = 1 − ζ` in the
endomorphism ring `Z[ζ]`. The `φ`-Selmer set classifies isomorphism classes of `φ`-coverings of
`C` everywhere locally soluble; the fake `φ`-Selmer set is its quotient by `ι`. If empty, `C` has
no `k`-rational points. For `q = 2`, `φ` is multiplication by 2 (the 2-Selmer set of
`TwoCoverDescent`).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `qCoverDescent(f, q)` | The fake `φ`-Selmer set of `C : y^q = f(x)` as an abstract set (a subset of the Cartesian product of number fields from the irreducible factors of `f`), plus the map into the abstract group `A(S, q)`. Requires `f` `q`-th-power-free over a number field of class number 1, `q` prime. Parameters: `PrimeBound` (default 0), `PrimeCutoff` (default 0; meanings as for `TwoCoverDescent`), `KnownPoints` (a lower bound from images of known points), `Verbose CycCov`. | Cyclic-cover descent **[Bru02, PS97]**. |

*Worked example: H125E39 (genus-15 curve `y⁷ = 2x⁷ + 6` everywhere locally soluble but with no rational points).*

### 125.15.3 Descent on the Jacobian

Let `J` be the Jacobian of `C : y^q = f(x)` over `k` containing the `q`-th roots of unity. The
`φ`-Selmer group of `J` (a finite group containing `J(k)/φ(J(k))`) bounds the rank of `J(k)`;
for `q = 2` this is the 2-Selmer group of `TwoSelmerGroup`. For `q` prime it is computed by the
Poonen–Schaefer algorithm **[PS97]** (which computes a fake Selmer group, a quotient by a
subgroup of order dividing `q`); when `k` lacks the `q`-th roots of unity one works over the
cyclotomic extension.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PhiSelmerGroup(f, q)` | For `f` over a number field `k`: the (fake) `φ`-Selmer group of `Jac(y^q = f(x))` over `k(μ_q)`, as a finite abelian group `S`, plus the map `S → A` to an affine algebra (standard map to `A*/k*(A*)^q` or `A*/(A*)^q`). `f(x)` separable, `q` prime. Parameters: `ReturnRawData` (default `false`; yields `expvecs`, `factorbase`, `selpic1` — the latter specifying the coset of the (fake) Selmer set of the Pic¹ torsor **[Cre12]**), `Verbose Selmer`. | Poonen–Schaefer **[PS97]**; Pic¹ torsor info **[Cre12]**. |
| `RankBound(f, q)` / `RankBounds(f, q)` | An upper (resp. lower and upper) bound on the rank of `Jac(y^q = f(x))`. Upper bound via the descent above, incorporating the Pic¹ torsor information; lower bound via a naive search for divisors. Parameter `ReturnGenerators` (default `false`; a third value listing univariate polynomials over P¹ that lift to divisors, generating a subgroup of rank at least the lower bound). | Descent **[PS97, Cre12]**. |

*Worked example: H125E40 (rank and 3-primary Ш of a genus-4 cyclic cover via `PhiSelmerGroup`/`RankBounds`).*

### 125.15.4 Partial Descent

The fake `φ`-Selmer set is computed via a map sending `(x, y) ↦ x − θ` (with `θ` a generic root
of `f`). More generally one specifies, for each irreducible factor `f_i` of `f`, an extension
`K_i/k` and an irreducible factor `h_i(x) ∈ K_i[x]` of `f_i`; the map `(x, y) ↦ (h₁(x),…,h_n(x))`
induces a partial-descent map. The resulting set gives information on `C(k)`; if empty, `C` has
no rational points. Geometrically a partial descent computes everywhere-locally-soluble
intermediate coverings corresponding to a Galois submodule of `J[φ]`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `qCoverPartialDescent(f, factors, q)` | A partial descent on `y^q = f(x)` with respect to the descent map given by `factors`. Returns the (cartesian-product) Selmer-type set and the map into `A(S, q)`. Requires `f` separable over a number field `k` of class number 1; for each irreducible factor `h_i ∈ k[x]`, `factors` must contain an irreducible factor of `h_i` over some extension of `k`. Parameters: `PrimeBound`, `PrimeCutoff`, `KnownPoints`, `Verbose CycCov` (as for `qCoverDescent`). | Galois-submodule partial descent. |

*Worked example: H125E41 (partial descent on `y³ = 3(x²+1)(x²+17)(x²−17)`, with elliptic-curve Chabauty determining all rational points).*

---

## 125.16 Kummer Surfaces

The Kummer surface `K` of the Jacobian `J` of a genus-2 curve is the quotient of `J` by the
inverse map — a quartic hypersurface in P³ with 16 ordinary double points (images of the 2-torsion
of `J`); resolving the singularities gives a K3 surface. Kummer surfaces in Magma are not schemes
but of type `SrfKum`; they allow arithmetic on the Jacobian without reduction of divisors, and
point searching. Implemented in arbitrary characteristic following **[Mül10b]**, extending work of
Flynn (**[CF96]** Ch. 3).

### 125.16.1 Creation of a Kummer Surface

| Intrinsic | Description |
|-----------|-------------|
| `KummerSurface(J)` | The Kummer surface of the Jacobian `J` of a genus-2 curve. |

### 125.16.2 Structure Operations

| Intrinsic | Description |
|-----------|-------------|
| `DefiningPolynomial(K)` | The defining polynomial of the Kummer surface `K`. |

### 125.16.3 Base Ring

| Intrinsic | Description |
|-----------|-------------|
| `BaseField(K)` / `BaseRing(K)` / `CoefficientRing(K)` | The base field of `K`. |

### 125.16.4 Changing the Base Ring

| Intrinsic | Description |
|-----------|-------------|
| `BaseChange(K, F)` / `BaseExtend(K, F)` | Extends the base field of `K` to `F`. |
| `BaseChange(K, j)` / `BaseExtend(K, j)` | Extends the base field by the ring map `j` (domain = base field of `C`). |
| `BaseChange(K, n)` / `BaseExtend(K, n)` | Extends a finite base field to its degree-`n` extension. |

---

## 125.17 Points on the Kummer Surface

Points are given by projective coordinates, normalised depending on the base field.

### 125.17.1 Creation of Points

| Intrinsic | Description |
|-----------|-------------|
| `K ! 0` | The image of the identity, normalised to the origin `(0 : 0 : 0 : 1)`. |
| `K ! [x1, x2, x3, x4]` | The point on `K` with the given projective coordinates. |
| `K ! P` | For `P` on the Jacobian of `K` (or on a Kummer surface for which `K` is a base extension): the point on `K`. |
| `IsPoint(K, S)` | For a 4-element sequence `S`: `true` iff `S` defines a point on `K`; if so returns the point. |
| `Points(K, [x1, x2, x3])` | The indexed set of points on `K` with given first three coordinates. |

### 125.17.2 Access Operations

| Intrinsic | Description |
|-----------|-------------|
| `P[i]` | The `i`-th coordinate of `P` (`1 ≤ i ≤ 4`). |
| `Eltseq(P)` / `ElementToSequence(P)` | The coordinates of `P` as a sequence. |

### 125.17.3 Predicates on Points

| Intrinsic | Description |
|-----------|-------------|
| `P eq Q` | `true` iff `P`, `Q` on the same Kummer surface are equal. |
| `P ne Q` | `false` iff `P`, `Q` are equal. |

### 125.17.4 Arithmetic of Points

| Intrinsic | Description |
|-----------|-------------|
| `-P` | The negation of `P` (equal to `P` itself). |
| `n * P` / `P * n` | The `n`-th multiple of `P`. |
| `Double(P)` | The double `2*P`. |
| `PseudoAdd(P1, P2, P3)` | Given Kummer images `P₁, P₂, P₃` of `P, Q, P−Q` on `J`: the image of `P + Q`. |
| `PseudoAddMultiple(P1, P2, P3, n)` | Given Kummer images `P₁, P₂, P₃` of `P, Q, P−Q`: the image of `P + n*Q`. |

### 125.17.5 Rational Points on the Kummer Surface

| Intrinsic | Description |
|-----------|-------------|
| `RationalPoints(K, Q)` | Given a sequence `Q` of three elements of the base ring `R`: the indexed set of points on `K` whose first three coordinates correspond to `Q`. |

*Worked example: H125E42 (searching for points on the Kummer surface of `y² = x⁵ − 7`).*

### 125.17.6 Pullback to the Jacobian

| Intrinsic | Description |
|-----------|-------------|
| `Points(J, P)` / `RationalPoints(J, P)` | For `P` on the Kummer surface of `J` (genus 2): the indexed set of points on `J` mapping to `P`. |

---

## 125.18 Analytic Jacobians of Hyperelliptic Curves

For a genus-`g` curve `C` over **C**, the analytic Jacobian is the abelian torus `C^g/Λ`
constructed from the period lattice: with holomorphic differentials `φ_i = x^{i-1}dx/y` and a
symplectic homology basis, the columns of the big period matrix `P = (ω₁, ω₂)` span `Λ`, and the
small period matrix `τ = ω₂⁻¹ω₁` lies in Siegel upper half-space. The Abel–Jacobi theorem gives a
bijection with the algebraic Jacobian. Following Mumford **[Mum84]** a special symplectic basis
linked to an ordering of the roots is needed; `HomologyBasis` provides it, with the roots stored
as the attribute `A`Roots`.

### 125.18.1 Creation and Access Functions

| Intrinsic | Description |
|-----------|-------------|
| `AnalyticJacobian(f)` | For `f ∈ C[x]` (complex field, precision 20 ≤ p < 2000): the analytic Jacobian of `y² = f(x)` (`deg f ≥ 3`). |
| `HyperellipticPolynomial(A)` | The polynomial defining the hyperelliptic curve whose Jacobian is `A`. |
| `SmallPeriodMatrix(A)` | The small period matrix `τ` of `A` — a symmetric `g × g` matrix with positive-definite imaginary part, `τ = ω₂⁻¹ω₁`. |
| `BigPeriodMatrix(A)` | The full `g × 2g` period matrix; `Λ = C^g/Λ` spanned by its columns (w.r.t. `φ_i = x^{i-1}dx/y` and a symplectic homology basis). |
| `HomologyBasis(A)` | The symplectic homology basis used for the period matrix. Returns `basepoints` (points in the complex plane), `loops` (a list of `2g` index lists into `basepoints`), and `S` (a matrix giving the linear combinations of loops forming the symplectic basis). For even-degree `HyperellipticPolynomial(A)` the basis is for the odd-degree curve obtained by sending `A`InfiniteRoot` to infinity via `x ↦ 1/(x−a)`. |
| `Dimension(A)` / `Genus(A)` | The dimension of `A` as a complex abelian variety (= genus of the curve). |
| `BaseField(A)` / `BaseRing(A)` / `CoefficientRing(A)` | The base field of `A`. |

### 125.18.2 Maps between Jacobians

| Intrinsic | Description |
|-----------|-------------|
| `ToAnalyticJacobian(x, y, A)` | For `A` the analytic Jacobian of `y² = f(x)`: maps the divisor `(x, y) − (a, 0)` (with `a = ∞` for odd degree, else `a = A`InfiniteRoot`) to `A`; by linearity any algebraic-Jacobian point maps in. Returns a `g × 1` matrix (an element of `C^g/Λ`, `Λ` from `BigPeriodMatrix`). |
| `FromAnalyticJacobian(z, A)` | For `A` with small period matrix `τ` and `z` a `g × 1` complex matrix (element of `C^g/Λ`): a list of `g` (or fewer) pairs `P_i = ⟨x_i, y_i⟩` with `y² = f(x)`, an element of the algebraic Jacobian (divisor `Σ P_i − g·∞`). |

*Worked example: H125E43 (moving between algebraic and analytic Jacobians; adding two rank-2 generators).*

#### 125.18.2.1 Isomorphisms, Isogenies and Endomorphism Rings of Analytic Jacobians

For `φ : A₁ → A₂` lifting to `α : C^g → C^g`, `α` is the complex representation and the integral
`2g × 2g` matrix `M` with `αP₁ = P₂M` the rational representation. For isomorphisms with Frobenius
bases, `M` is symplectic (`MJ^tM = J`) and acts on Siegel upper half-space by
`τ ↦ (aτ + b)(cτ + d)⁻¹`. Isomorphisms between abelian surfaces use a fundamental domain for
2-dimensional upper half-space (**[Got59]**); other functions rely on `LinearRelation` and need
high precision.

| Intrinsic | Description |
|-----------|-------------|
| `To2DUpperHalfSpaceFundamentalDomian(z)` | For a complex matrix `z` in 2-dimensional Siegel upper half-space: the element of the Gottschling fundamental domain (**[Got59]**), and the symplectic matrix taking `z` to it. |
| `AnalyticHomomorphisms(t1, t2)` | For small period matrices `t₁`, `t₂`: a basis for the **Z**-module of integral `2g × 2g` matrices `M` such that some complex `g × g` `α` satisfies `α(t₁, 1) = (t₂, 1)M`. |
| `IsIsomorphicSmallPeriodMatrices(t1, t2)` | For small period matrices: finds a symplectic integral `M` with `α(t₁, 1) = (t₂, 1)M`. Returns `true`/`false` and the matrix `M` (zero if none). |
| `IsIsomorphicBigPeriodMatrices(P1, P2)` | For big period matrices: finds a symplectic integral `M` with `αP₁ = P₂M`. Returns `true`/`false`, `M`, and `α`. |
| `IsIsomorphic(A1, A2)` | For analytic Jacobians with big period matrices `P₁`, `P₂`: finds a symplectic integral `M` with `αP₁ = P₂M`. Returns `true`/`false`, `M`, `α`. |
| `IsIsogenousPeriodMatrices(P1, P2)` | For period matrices (small or big): finds a nonsingular integral `M` with `α(P₁, 1) = (P₂, 1)M` (small) or `αP₁ = P₂M` (big), defining an isogeny. Returns `true`/`false`, `M`, and (big case) `α`. |
| `IsIsogenous(A1, A2)` | For analytic Jacobians with big period matrices: finds a nonsingular integral `M` with `αP₁ = P₂M`. Returns `true`/`false`, `M`, `α`. |
| `EndomorphismRing(P)` | The endomorphism ring of the analytic Jacobian of period matrix `P`, as a matrix algebra; for a big period matrix also a list of `α`-matrices with `αP = PM` for each generator `M`. |
| `EndomorphismRing(A)` | The endomorphism ring of the analytic Jacobian `A` as a matrix algebra; second return: the `α`-matrices for each generator. |

*Worked example: H125E44 (rational isogenies between genus-2 Jacobians via `AnalyticHomomorphisms`; `PowerRelation`/`LinearRelation` to recognise α-entries as algebraic numbers; isogeny class from Smart **[Sma97]**).*

### 125.18.3 From Period Matrix to Curve

| Intrinsic | Description |
|-----------|-------------|
| `RosenhainInvariants(t)` | For a small period matrix `t` of a genus-`g` analytic Jacobian `A`: a set `S` of `2g − 1` complex numbers such that `y² = x(x−1)∏_{s∈S}(x − s)` (Rosenhain normal form) has Jacobian isomorphic to `A`. |

*Worked example: H125E45 (genus-2 CM curve over **Q**(√(−2+√2)): principal polarisation via Wamelen **[Wam99]**, Frobenius basis, `EndomorphismRing`, `RosenhainInvariants`, and `IgusaClebschInvariants`/`HyperellipticCurveFromIgusaClebsch`/`ReducedWamelenModel` to recover the curve).*

### 125.18.4 Voronoi Cells

| Intrinsic | Description |
|-----------|-------------|
| `Delaunay(sites)` | The Delaunay triangulation for `sites` (a sequence of pairs of real numbers, all in the same real field). Returns, for each site `i`, the list of indices of sites to which `i` is joined. |
| `Voronoi(sites)` | The Voronoi cells for `sites`. Returns `siteedges` (as `Delaunay`), `dualsites` (triples `⟨x, y, m⟩`; `m = 0` for a finite point, else a point "at infinity" in direction `(x, y)`), and `cells` (for each site, an index list into `dualsites` describing the surrounding cell, with the first/last indices indicating infinite sides). |

---

## 125.19 Bibliography (canonical references)

| Key | Reference |
|-----|-----------|
| **[BD09]** | Nils Bruin and Kevin Doerksen. *The arithmetic of genus two curves with (4,4)-split Jacobians.* arXiv:0902.3480, 2009. |
| **[Bos00]** | Wieb Bosma, editor. *ANTS IV*, volume 1838 of *LNCS*. Springer-Verlag, 2000. |
| **[Bru02]** | N. R. Bruin. *Chabauty methods and covering techniques applied to generalized Fermat equations*, volume 133 of *CWI Tract*. Stichting Mathematisch Centrum Centrum voor Wiskunde en Informatica, Amsterdam, 2002. Dissertation, University of Leiden, 1999. |
| **[BS09]** | Nils Bruin and Michael Stoll. *Two-cover descent on hyperelliptic curves.* Math. Comp., 78:2347–2370, 2009. |
| **[Car03]** | G. Cardona. *On the number of curves of genus 2 over a finite field.* Finite Fields and Their Applications, 9(4):505–526, 2003. |
| **[CF96]** | J. W. S. Cassels and E. V. Flynn. *Prolegomena to a Middlebrow Arithmetic of Curves of Genus 2.* Cambridge University Press, Cambridge, 1996. |
| **[CNP05]** | G. Cardona, E. Nart, and J. Pujolas. *Curves of genus two over fields of even characteristic.* Mathematische Zeitschrift, 250:177–201, 2005. |
| **[CQ05]** | Gabriel Cardona and Jordi Quer. *Field of moduli and field of definition for curves of genus 2.* Lecture Notes Ser. Comput., 13:71–83, 2005. |
| **[Cre12]** | Brendan Creutz. *Explicit descent in the Picard group of a cyclic cover of the projective line.* In Everett Howe and Kiran Kedlaya, editors, *ANTS X: Proceedings of the Tenth Algorithmic Number Theory Symposium*, volume 1 of *OBS*. Mathematics Sciences Publishers, 2012. |
| **[FS97]** | E. V. Flynn and N. P. Smart. *Canonical heights on the Jacobians of curves of genus 2 and the infinite descent.* Acta Arith., 79:333–352, 1997. |
| **[GH00]** | P. Gaudry and R. Harley. *Counting Points on Hyperelliptic Curves over Finite Fields.* In Bosma **[Bos00]**, pages 313–332. |
| **[Got59]** | E. Gottschling. *Explizite Bestimmung der Randflächen des Fundamentalbereiches der Modulgruppe zweiten Grades.* Math. Annalen, 138:103–124, 1959. |
| **[Hol06]** | David Holmes. *Canonical heights on hyperelliptic curves.* Preprint, arXiv:1004.4503, 2006. |
| **[Hub06]** | Hendrik Hubrechts. *Point counting in families of hyperelliptic curves.* Preprint, arXiv:math.NT/0601438, 2006. |
| **[Ked01]** | Kiran S. Kedlaya. *Counting Points on Hyperelliptic Curves using Monsky-Washnitzer Cohomology.* J. Ramanujan Math. Soc., 16:323–338, 2001. |
| **[LL]** | R. Lercier and D. Lubicz. *A Quasi-Quadratic Time Algorithm for Hyperelliptic Curve Point Counting.* To appear. |
| **[LR12]** | R. Lercier and C. Ritzenthaler. *Hyperelliptic curves and their invariants: geometric, arithmetic and algorithmic aspects.* Journal of Algebra, 2012. doi:10.1016/j.jalgebra.2012.07.054. |
| **[Mae90]** | T. Maeda. *On the invariant fields of binary octavics.* Hiroshima Math. J., 20:619–632, 1990. |
| **[Mes91]** | J.-F. Mestre. *Construction de courbes de genre 2 à partir de leurs modules.* In T. Mora and C. Traverso, editors, *Effective methods in algebraic geometry*, volume 94 of *Progr. Math.*, pages 313–334. Birkhäuser, 1991. |
| **[Mül10a]** | Jan Steffen Müller. *Computing canonical heights on Jacobians.* PhD Thesis, Universität Bayreuth, 2010. |
| **[Mül10b]** | Jan Steffen Müller. *Explicit Kummer surface formulas for arbitrary characteristic.* LMS J. Comput. Math., 4:47–64, 2010. |
| **[Mum84]** | David Mumford. *Tata Lectures on Theta II*, volume 43 of *Progress in Mathematics*. Birkhäuser, 1984. |
| **[PS97]** | Bjorn Poonen and Ed Schaefer. *Explicit descent for Jacobians of cyclic covers of the projective line.* J. Reine Angew. Math., 488:141–188, 1997. |
| **[PS99]** | Bjorn Poonen and Michael Stoll. *The Cassels-Tate pairing on polarized abelian varieties.* Ann. of Math. (2), 150(3):1109–1149, 1999. |
| **[Shi67]** | T. Shioda. *On the Graded Ring of Invariants of Binary Octavics.* Am. Jour. Math., 89(4):1022–1046, 1967. |
| **[Sma97]** | N. P. Smart. *S-unit equations, binary forms and curves of genus 2.* Proc. London Math. Soc. (3), 75(2):271–307, 1997. |
| **[Smi05]** | Benjamin A. Smith. *Explicit endomorphisms and correspondences.* PhD thesis, University of Sydney, 2005. |
| **[Sto99]** | M. Stoll. *On the height constant for curves of genus two.* Acta Arith., 90(2):183–201, 1999. |
| **[Sto01]** | Michael Stoll. *Implementing 2-descent for Jacobians of hyperelliptic curves.* Acta Arith., 98(3):245–277, 2001. |
| **[SV01]** | Tony Shaska and Völklein. *Elliptic subfields and automorphisms of genus 2 function fields.* Springer, 2001. arXiv:math.AG/0107142. |
| **[Ver02]** | F. Vercauteren. *Computing zeta functions of hyperelliptic curves over finite fields of characteristic 2.* In *Advances in cryptology—CRYPTO 2002*, volume 2442 of *LNCS*, pages 369–384. Springer, Berlin, 2002. |
| **[Wam99]** | P. Van Wamelen. *Examples of Genus Two CM Curves Defined over the Rationals.* Mathematics of Computation, 68:307–320, 1999. |
| **[Wam01]** | P. Van Wamelen. *Computing with the Jacobian of a Genus 2 Curve.* 2001. |
| **[Wet97]** | J. L. Wetherell. *Bounding the number of rational points on certain curves of high rank.* PhD thesis, U.C. Berkeley, 1997. |

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Classical invariant theory of binary sextics (Mestre, Überschiebungen) **[Mes91]** | `ClebschInvariants`, `IgusaClebschInvariants`, `IgusaInvariants`/`JInvariants`, `AbsoluteInvariants`, `ScaledIgusaInvariants` |
| Cardona-Quer-Nart-Pujola genus-2 absolute invariants **[CQ05, CNP05]** | `G2Invariants`, `G2ToIgusaInvariants`, `IgusaToG2Invariants`, `HyperellipticCurveFromG2Invariants` |
| Shioda / Maeda binary-octic invariants (genus 3) **[LR12, Shi67, Mae90]** | `ShiodaInvariants`, `ShiodaAlgebraicInvariants`, `MaedaInvariants`, `DiscriminantFromShiodaInvariants`, `HyperellipticCurveFromShiodaInvariants` |
| Curve from Igusa–Clebsch invariants (Mestre / Cardona) **[Mes91, CQ05]** | `HyperellipticCurveFromIgusaClebsch` |
| Reduction of models (SL₂(**Z**) / Wamelen) **[Wam99, Wam01]** | `ReducedModel`, `ReducedMinimalWeierstrassModel`, `MinimalWeierstrassModel` |
| GL₂-orbit isomorphism testing / classification **[SV01, CQ05, CNP05, Car03]** | `IsGL2Equivalent`, `IsIsomorphic`, `Twists`, `GeometricAutomorphismGroup`, `GeometricAutomorphismGroupGenus2Classification`, `GeometricAutomorphismGroupGenus3Classification` |
| Richelot isogenies of genus-2 Jacobians **[Smi05, BD09]** | `RichelotIsogenousSurfaces`, `RichelotIsogenousSurface` |
| Jacobian point counting — p-adic liftings **[Ked01, Ver02, LL]** | `#J`, `Order(J)`, `EulerFactor`, `ZetaFunction` |
| Genus-2 point counting (Gaudry–Harley, Shanks/Pollard) **[GH00]** | `#J`, `Order(P, l, u)` (`PollardRho`) |
| Cartier–Manin trick | `EulerFactorModChar`, `#J` (`CartierManinLimit`) |
| Deformation point counting (Lauder–Hubrechts) **[Hub06]** | `JacobianOrdersByDeformation`, `EulerFactorsByDeformation`, `ZetaFunctionsByDeformation` |
| Canonical heights — Kummer surface (Flynn–Smart/Stoll) **[FS97, Sto99]** | `Height`/`CanonicalHeight`, `HeightPairing`, `HeightConstant`, `NaiveHeight` (genus 2 / **Q**) |
| Canonical heights — Arakelov intersection **[Mül10a, Hol06]** | `Height`/`CanonicalHeight`, `HeightPairing` (`UseArakelov`), `Regulator`, `ReducedBasis` |
| 2-descent on Jacobians **[Sto01]** | `TwoSelmerGroup`, `RankBound`, `RankBounds`, `HasSquareSha`/`IsEven` **[PS99]** |
| Two-cover descent on curves **[BS09]** | `TwoCoverDescent` |
| Cyclic-cover / φ-descent (Poonen–Schaefer) **[PS97, Cre12]** | `qCoverDescent`, `qCoverPartialDescent`, `PhiSelmerGroup`, `RankBound(f,q)`, `RankBounds(f,q)` |
| Chabauty's method + Mordell–Weil sieve | `Chabauty`, `Chabauty0` |
| Point search by sieving **[Bru02]** | `Points`/`RationalPoints` (number fields), `RationalPoints(f, q)` |
| Kummer surface arithmetic **[Mül10b, CF96]** | `KummerSurface`, `PseudoAdd`, `PseudoAddMultiple`, `Double`, `RationalPoints(K, Q)` |
| Analytic Jacobians / period matrices (Mumford, Gottschling) **[Mum84, Got59]** | `AnalyticJacobian`, `BigPeriodMatrix`, `SmallPeriodMatrix`, `HomologyBasis`, `ToAnalyticJacobian`, `FromAnalyticJacobian`, `IsIsomorphic`, `IsIsogenous`, `EndomorphismRing`, `RosenhainInvariants`, `To2DUpperHalfSpaceFundamentalDomian` |
| CM curve construction (Wamelen) **[Wam99]** | `RosenhainInvariants` (+ Igusa-invariant pipeline) |
