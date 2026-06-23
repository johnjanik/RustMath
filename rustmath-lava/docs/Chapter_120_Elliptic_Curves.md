# Chapter 120 — Elliptic Curves

**Handbook part:** XVI — Arithmetic Geometry
**Handbook pages:** 3939–3976 (PDF pages 4068–4109)

---

## Scope and overview

Chapter 120 is the first of four chapters on elliptic curves in Magma. It covers the basics applicable to curves over general fields: construction, arithmetic, and basic properties. Specialised machinery for curves over finite fields appears in Chapter 121; curves over Q and number fields (Mordell–Weil group, heights, descent, analytic theory) are treated in Chapter 122; and curves over function fields in Chapter 123.

An elliptic curve E is the projective closure of the generalised Weierstrass equation

    y² + a₁xy + a₃y = x³ + a₂x² + a₄x + a₆

specified by the coefficient sequence [a₁, a₂, a₃, a₄, a₆], or the abbreviated [a₄, a₆] when a₁ = a₂ = a₃ = 0. The base ring is currently restricted to fields; integer coefficients are coerced into Q. Categories: `CrvEll` (curves), `PtEll` (points), `SetPtEll` (point sets), `SchGrpEll` (subgroup schemes), `SymKod` (Kodaira symbols used in Chapter 122).

Elliptic curves are specialised instances of the general curve and scheme types (Chapters 112 and 114); all functions applicable to those types also apply here, though some behave differently. In particular, the parent of a point is a point set `E(K)`, not the curve `E` itself.

Algorithms for the Mordell–Weil group are heavily based on work of John Cremona **[Cre97]**. The chapter covers: creation from Weierstrass data or genus-1 curves; alternative models (integral, minimal, simplified); twists; elementary invariants (a-, b-, c-invariants, discriminant, j-invariant); division and torsion polynomials; subgroup schemes; the formal group; point sets and point arithmetic; morphisms and isogenies (using Vélu's formulae); endomorphisms; the Weil pairing.

---

## 120.1 Introduction

*No intrinsics in this section — introductory prose only (see Scope and overview above).*

---

## 120.2 Creation Functions

### 120.2.1 Creation of an Elliptic Curve

An elliptic curve E may be created by specifying Weierstrass coefficients over a field K (integers are coerced into Q). The coordinate ring K defines the base point set of E; points over extension fields must be created in the appropriate point set.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `EllipticCurve([a, b])` / `EllipticCurve([a1, a2, a3, a4, a6])` | Creates the elliptic curve E over K from a sequence of Weierstrass coefficients of length 2 (short form y² = x³ + ax + b) or 5 (general Weierstrass form). The discriminant must be nonzero. | Direct definition; discriminant check. |
| `EllipticCurve(f)` / `EllipticCurve(f, h)` | Creates the elliptic curve defined by y² + h(x)y = f(x), or y² = f(x) if h is omitted. f must be monic of degree 3; h must have degree at most 1. | Direct definition. |
| `EllipticCurveFromjInvariant(j)` / `EllipticCurveWithjInvariant(j)` | Creates an elliptic curve with given j-invariant: y² + y = x³ if j = 0 (char ≠ 3); y² = x³ + x if j = 1728 (or j = 0, char 3); otherwise y² + xy = x³ − (36/(j−1728))x − 1/(j−1728). | Explicit Weierstrass model by j-invariant formula. |
| `EllipticCurve(C)` | Given a genus-1 scheme C with an easily recognised rational point, returns an elliptic curve E and a birational map C → E. Handles hyperelliptic curves of genus 1 (f of degree 3 or 4), nonsingular plane cubics (checks for rational flex), and singular plane quartics with a unique cusp. Verbose parameter `EllModel` (max 3). | Recognises rational branch points or points at infinity; uses linear transformations or Riemann–Roch as needed. |
| `EllipticCurve(C, P)` | Given a genus-1 scheme C and a nonsingular rational point P, returns E and a birational map sending P to the origin. For degree-3 curves (char ≠ 2, 3): uses a linear transform if P is a flex; otherwise Nagell's algorithm **[Nag28, Cas91]**. For degree-4 curves with unique cusp: construction from **[Cas91]**. Otherwise: Riemann–Roch (potentially expensive). | Nagell's algorithm **[Nag28]** / Riemann–Roch. |
| `EllipticCurve(C, pl)` | Given a plane genus-1 curve C and a place pl of degree 1, returns E and a birational map via a Riemann–Roch computation (potentially expensive; differs from the point version which uses special methods for degree 3 or 4). | Riemann–Roch computation. |
| `SupersingularEllipticCurve(K)` | Given a finite field K, returns a representative supersingular elliptic curve over K. | — |

*Worked examples: H120E1 (three ways to create the curve y² = x³ + x over Q); H120E2 (Nagell's algorithm vs. Riemann–Roch for a plane cubic with rational point); H120E3 (genus-1 curve requiring Riemann–Roch).*

### 120.2.2 Creation Predicates

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsEllipticCurve([a, b])` / `IsEllipticCurve([a1, a2, a3, a4, a6])` | Returns true if the given sequence of ring elements defines an elliptic curve (discriminant nonzero); also returns the elliptic curve when true. | Discriminant check. |
| `IsEllipticCurve(C)` | Given a hyperelliptic curve C, returns true if C has degree 3; also returns the elliptic curve and isomorphisms to/from it. *Deprecated — use `Degree(C) eq 3` and `EllipticCurve(C)` instead.* | Degree check. |

*Worked example: H120E4 (checking which small primes make given coefficients define an elliptic curve over GF(p)).*

### 120.2.3 Changing the Base Ring

The following general scheme functions generate a new elliptic curve over a different field. If the result is not a valid elliptic curve, the functions may return a general curve rather than an elliptic curve.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BaseChange(E, K)` / `BaseExtend(E, K)` | Given E over k and a field K that is an extension of k, returns E′ over K using the natural inclusion of k in K to map coefficients. | Natural inclusion of coefficient rings. |
| `ChangeRing(E, K)` | Given E over k and a field K, returns E′ over K using the standard coercion (useful when no ring homomorphism k → K is available, e.g. k = Q and K a finite field). | Standard coercion. |
| `BaseChange(E, h)` / `BaseExtend(E, h)` | Given E over k and a ring map h : k → K, returns E′ over K by applying h to the coefficients. | Application of ring map h. |
| `BaseChange(E, n)` / `BaseExtend(E, n)` | Given E over a finite field K, returns the base extension of E to the degree n extension of K. | Extension of finite fields. |

*Worked example: H120E5 (base extension over GF(23) and GF(23²) using field, ring map, and degree; also shows ChangeRing in reverse direction).*

### 120.2.4 Alternative Models

Each function returns an isomorphic curve E′ in the desired model, together with the isomorphisms E → E′ and E′ → E (the second is the inverse of the first; in a future release only the first will be returned).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `WeierstrassModel(E)` | Returns an isomorphic E′ in simplified Weierstrass form y² = x³ + ax + b. Does not apply in characteristic 2 or 3. | Change of variables to short Weierstrass form. |
| `IntegralModel(E)` | For E over a number field K (including Q), returns an isomorphic E′ over K with integral coefficients. | Scaling to clear denominators. |
| `SimplifiedModel(E)` | Returns a simplified model of E. Over Q: same as MinimalModel. In characteristic ≠ 2, 3: same as WeierstrassModel. In characteristic 2 or 3: see chapter 4 of **[Con99]**. | Depends on base field; see **[Con99]** for char 2, 3. |
| `MinimalModel(E)` | For E over Q or a number field K with class number 1, returns a global minimal model E′ (integral model with minimal discriminant valuation at all non-zero prime ideals). Errors if no global minimal model exists. | Local minimisation at all primes. |
| `MinimalModel(E, p)` / `MinimalModel(E, P : -)` | For E over a number field K and a prime ideal P (or integer prime p when K = Q), returns an E′ isomorphic to E that is minimal at P. Parameter `UseGeneratorAsUniformiser` (default false): if true and P is principal, uses an ideal generator as uniformiser to keep the model integral at other primes. | Local minimisation at P using a uniformiser. |

*Worked example: H120E6 (integral model, minimal model, and Weierstrass model of a curve over Q; verifying isomorphism).*

### 120.2.5 Predicates on Curve Models

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsWeierstrassModel(E)` | Returns true iff E is in simplified Weierstrass form. | — |
| `IsIntegralModel(E)` | Returns true iff E is an integral model. | — |
| `IsSimplifiedModel(E)` | Returns true iff E is a simplified model. | — |
| `IsMinimalModel(E)` | Returns true iff E is a minimal model. | — |
| `IsIntegralModel(E, P)` | For E over a number field and P a prime ideal of an order of the coefficient ring, returns true iff the defining coefficients of E have non-negative valuation at P. | Valuation check at P. |

### 120.2.6 Twists of Elliptic Curves

All twists are returned as simplified models.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `QuadraticTwist(E, d)` | Returns the quadratic twist of E by d. Isomorphic to E iff (char 2 and Tr(d) = 0) or (char ≠ 2 and d is a square). Does not always work in characteristic 2. | Quadratic twist by d. |
| `QuadraticTwist(E)` | For E over a finite field, returns a quadratic twist (a non-isomorphic curve whose trace is the negation of Trace(E)). | — |
| `QuadraticTwists(E)` | For E over a finite field, returns the sequence of all non-isomorphic quadratic twists; the first entry is isomorphic to E. | — |
| `Twists(E)` | For E over a finite field K, returns the sequence of all non-isomorphic elliptic curves over K that are isomorphic to E over some extension field; the first entry is isomorphic to E. | Classification by j-invariant and automorphism group. |
| `IsTwist(E, F)` | Returns true iff E and F are isomorphic over an extension field (tests only j-invariants). | j-invariant comparison. |
| `IsQuadraticTwist(E, F)` | Returns true iff E and F are isomorphic over a quadratic extension; if so, also returns d such that F is isomorphic to `QuadraticTwist(E, d)`. | Quadratic twist criterion. |
| `MinimalQuadraticTwist(E)` | For a rational elliptic curve E, determines the minimal quadratic twist. Handles odd primes of bad reduction by iteratively twisting by p if that reduces the discriminant; handles p = 2 and sign by minimising the 2-adic valuation of the conductor, then choosing the twist making (−1)^(v₂(c₆)) odd(c₆) ≡ 3 (mod 4). Returns also the integer d by which E was twisted. | Local minimisation prime-by-prime; see algorithm description in text. |

*Worked examples: H120E7 (quadratic twists of a curve over GF(13)); H120E8 (all twists of curves with j-invariants 0 and 12 over GF(13)); H120E9 (MinimalQuadraticTwist of a rational curve).*

---

## 120.3 Operations on Curves

### 120.3.1 Elementary Invariants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `aInvariants(E)` / `Coefficients(E)` / `ElementToSequence(E)` / `Eltseq(E)` | Returns the sequence [a₁, a₂, a₃, a₄, a₆] of Weierstrass coefficients of E (always 5 elements, even if E was defined by a length-2 sequence). | Direct read-off of stored coefficients. |
| `bInvariants(E)` | Returns [b₂, b₄, b₆, b₈] where b₂ = a₁² + 4a₂, b₄ = a₁a₃ + 2a₄, b₆ = a₃² + 4a₆, b₈ = a₁²a₆ + 4a₂a₆ − a₁a₃a₄ + a₂a₃² − a₄². | Standard formulas. |
| `cInvariants(E)` | Returns [c₄, c₆] where c₄ = b₂² − 24b₄, c₆ = −b₂³ + 36b₂b₄ − 216b₆. | Standard formulas. |
| `Discriminant(E)` | Returns the discriminant Δ = −b₂²b₈ − 8b₄³ − 27b₆² + 9b₂b₄b₆; also satisfies 1728Δ = c₄³ − c₆². | Standard formula. |
| `jInvariant(E)` | Returns the j-invariant j = c₄³/Δ. Two curves over the same base field are isomorphic over some extension iff their j-invariants are equal. | Standard formula. |
| `HyperellipticPolynomials(E)` | Returns the polynomials x³ + a₂x² + a₄x + a₆ and a₁x + a₃ derived from the invariants of E. | Direct extraction. |

*Worked examples: H120E10 (aInvariants, Discriminant, jInvariant for a curve over Q); H120E11 (generic elliptic curve over a function field verifying all b-, c-invariant and discriminant formulas).*

### 120.3.2 Associated Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Category(E)` / `Type(E)` | Returns the category `CrvEll` of elliptic curves. | — |
| `BaseRing(E)` / `CoefficientRing(E)` | The base ring of E; the parent of its coefficients and the coefficient ring of the default point set. | — |

### 120.3.3 Predicates on Elliptic Curves

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `E eq F` | Returns true iff E and F are defined over the same ring and have the same coefficients. | Coefficient comparison. |
| `E ne F` | Logical negation of eq. | — |
| `IsIsomorphic(E, F)` | Returns true iff there exists an isomorphism E → F over the base field; if so, also returns the isomorphism. Requires the ability to take roots in the base field. | Root extraction in base field. |
| `IsIsogenous(E, F)` | For E and F over Q or a finite field, returns true iff E and F are isogenous over that field. For Q: also returns the isogeny. For finite fields: uses point counting (no isogeny returned). | Over Q: isogeny search; over finite fields: point counting. |

*Worked example: H120E12 (quadratic twists over GF(13) are non-isomorphic over base field but isomorphic over quadratic extension; j-invariants agree).*

---

## 120.4 Polynomials

The torsion (division) polynomials define the subschemes of n-torsion points on the elliptic curve.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DefiningPolynomial(E)` | Returns the homogeneous defining polynomial for E. | — |
| `DivisionPolynomial(E, n)` / `DivisionPolynomial(E, n, g)` | Returns the n-th division polynomial as a univariate polynomial over the base ring of E. For even n, has multiplicity two at the nonzero 2-torsion points; the second return value is this polynomial divided by the 2-torsion polynomial; the third return is the cofactor (equal to the 2-torsion polynomial if n even, else 1). If a polynomial g is passed as third argument, computation is done efficiently modulo g. | Standard division polynomial recursion. |
| `TwoTorsionPolynomial(E)` | Returns the multivariate 2-torsion polynomial 2y + a₁x + a₃ of E as a bivariate polynomial. | Direct definition. |

*Worked example: H120E13 (roots of 5th and 9th division polynomials over GF(101); illustrates that roots of the division polynomial may not give rational points — corresponding points may lie in a quadratic extension).*

---

## 120.5 Subgroup Schemes

A subgroup scheme G of an elliptic curve E is a subscheme defined by a univariate polynomial ψ and closed under the group law. Points of G are those points of E whose x-coordinate is a root of ψ. Elliptic curves themselves are considered subgroup schemes with defining polynomial ψ = 0.

### 120.5.1 Creation of Subgroup Schemes

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SubgroupScheme(G, f)` | Creates the subgroup scheme of G (which may be an elliptic curve) defined by univariate polynomial f. No checking that the rational points actually form a group. | — |
| `TorsionSubgroupScheme(G, n)` | Returns the subgroup scheme of n-torsion points of the subgroup scheme G (G may be an elliptic curve). | Via division polynomial. |

### 120.5.2 Associated Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Category(G)` / `Type(G)` | Returns `SchGrpEll`, the category of elliptic curve subgroup schemes. | — |
| `Curve(G)` / `Generic(G)` | Returns the elliptic curve E of which G is a subgroup scheme. | — |
| `BaseRing(G)` / `CoefficientRing(G)` | Returns the base ring of G (same as the base ring of its curve). | — |
| `DefiningSubschemePolynomial(G)` | Returns the univariate polynomial that defines G as a subscheme of its curve. | — |

### 120.5.3 Predicates on Subgroup Schemes

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `G1 eq G2` | Returns true iff G1 and G2 are subgroup schemes of the same elliptic curve and are defined by equal polynomials. | Polynomial equality. |
| `G1 ne G2` | Logical negation of eq. | — |

### 120.5.4 Points of Subgroup Schemes

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `#G` / `Order(G)` | The order of the group of rational points on the subgroup scheme G. | — |
| `FactoredOrder(G)` | The factorisation of the order of the group of rational points on G. | — |
| `Points(G)` / `RationalPoints(G)` | The indexed set of rational points of G over its base ring. | — |

*Worked example: H120E14 (subgroup scheme of E over GF(49) defined by (t−4)(t−5)(t−6); forming a further subgroup for 3-torsion; finding the same subgroup via TorsionSubgroupScheme and SubgroupScheme).*

---

## 120.6 The Formal Group

Functions applicable to elliptic curves over any exact field.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `FormalGroupLaw(E, prec)` | Returns a bivariate polynomial T₁ + T₂ + … giving the formal group law associated to addition on E, truncated at total degree prec. The formal variables T₁, T₂ are identified with −x/y on E (a local parameter near O_E). | Formal power series expansion of the addition law. |
| `FormalGroupHomomorphism(phi, prec)` | Returns the formal group homomorphism associated to the isogeny phi, as a univariate power series up to precision prec, in the parameter −x/y. | Formal expansion of isogeny map. |
| `FormalLog(E)` | Parameter `Precision` (default 10). Returns the formal logarithm as a power series f(T) and a formal point P(T) with coordinates in a Laurent series ring with generator T (corresponding to −x/y), giving a formal parametrisation of E near O_E. | Formal logarithm of the formal group. |

---

## 120.7 Operations on Point Sets

Each elliptic curve E has associated point sets E(L) for extensions L of the base ring K. Points on E lie in point sets, not in E itself. Passing E to functions expecting a point set uses the base point set E(K). These conventions also apply to subgroup schemes.

### 120.7.1 Creation of Point Sets

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `E(L)` / `PointSet(E, L)` | Given an elliptic curve E (or subgroup scheme) and an extension L of its base ring, returns the point set E(L). | — |
| `E(m)` / `PointSet(E, m)` | Given E and a map m from the base ring of E to a field L, returns the point set of E with coefficients in L; the map is retained to permit coercions between point sets. | — |

### 120.7.2 Associated Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Category(H)` / `Type(H)` | Returns `SetPtEll`, the category of point sets of elliptic curves. | — |
| `Scheme(H)` | Returns the associated scheme (elliptic curve or subgroup scheme) of which H is a point set. | — |
| `Curve(H)` | Returns the associated elliptic curve containing Scheme(H). | — |
| `Ring(H)` | Returns the ring containing coordinates of points in H. | — |

### 120.7.3 Predicates on Point Sets

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `H1 eq H2` | Returns true iff the two point sets have equal coefficient rings and equal elliptic curves (or subgroup schemes). | — |
| `H1 ne H2` | Logical negation of eq. | — |

*Worked example: H120E15 (point sets over GF(5) and GF(5²) of the same curve; point sets of a subgroup scheme vs. the curve are not equal even if the ring is the same).*

---

## 120.8 Morphisms

Four types of maps between elliptic curves may be constructed: isogenies, isomorphisms, translations, and rational maps. Isogenies are always surjective as scheme maps. There is an internal limit that the degrees of polynomials defining an isogeny cannot exceed 10⁷.

### 120.8.1 Creation Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Isomorphism(E, F, [r, s, t, u])` | Given E, F over the same field K and u ≠ 0, returns the isomorphism E → F mapping O_E ↦ O_F and (x,y) ↦ (u²x + r, u³y + su²x + t). Errors if values do not define a valid isomorphism. | Direct formula; validity check. |
| `Isomorphism(E, F)` | Computes and returns an isomorphism E → F where one exists (same map as second return value of IsIsomorphic). | Root extraction in base field. |
| `Automorphism(E, [r, s, t, u])` | Returns the automorphism E → E sending (x,y) ↦ (u²x + r, u³y + su²x + t). Errors if values do not define an automorphism. | Direct formula; validity check. |
| `IsomorphismData(I)` | Returns the sequence [r, s, t, u] of elements defining the isomorphism I. | — |
| `IsIsomorphism(I)` | Returns true iff the isogeny I has the same action as some isomorphism; if so, also returns that isomorphism. | — |
| `IsomorphismToIsogeny(I)` | Takes a map I of type isomorphism and returns an equivalent map of type isogeny. | — |
| `TranslationMap(E, P)` | Given a rational point P on E, returns the morphism t_P : E → E defined by t_P(Q) = P + Q. | Addition on E. |
| `RationalMap(i, t)` | Given an isogeny i and a translation map t_P, returns the rational map E → F obtained by composing i and t (t applied first). Any rational map E → F can be represented this way. | Composition of morphisms. |
| `TwoIsogeny(P)` | Given a 2-torsion point P of E, returns a 2-isogeny on E with P as kernel. | Vélu's formulae for 2-isogeny. |
| `IsogenyFromKernel(G)` | Given a subgroup scheme G of E, returns the curve E_f and the separable isogeny f : E → E_f with kernel G, computed using Vélu's formulae. | Vélu's formulae **[Sil86, III §4]**. |
| `IsogenyFromKernelFactored(G)` | Returns a sequence of isogenies whose composition equals `IsogenyFromKernel(G)`. Each has degree 2 or odd degree. Introduced to avoid the expense of composing isogenies. | Vélu's formulae applied in stages. |
| `IsogenyFromKernel(E, psi)` | Given E and a univariate polynomial psi defining a subgroup scheme of E, computes the isogeny using Vélu's formulae. | Vélu's formulae. |
| `IsogenyFromKernelFactored(E, psi)` | Returns a sequence of isogenies (degree 2 or odd) whose composition is `IsogenyFromKernel(E, psi)`. | Vélu's formulae applied in stages. |
| `PushThroughIsogeny(I, v)` / `PushThroughIsogeny(I, G)` | Given an isogeny I and a subgroup G (or its defining polynomial v) containing the kernel of I, finds the image of G under I. | Image computation via I. |
| `DualIsogeny(phi)` | Given an isogeny φ : E₁ → E₂, returns the dual isogeny φ* : E₂ → E₁ such that φ* ∘ φ is multiplication by deg(φ). The result is memoised: `DualIsogeny(DualIsogeny(phi))` returns phi. | Dual isogeny construction via division polynomial and Vélu's formulae. |

*Worked examples: H120E16 (2-isogeny following Silverman [Sil86] Ex. III.4.5; parametrised family and its dual); H120E17 (isomorphism routines; applying by hand using IsomorphismData); H120E18 (converting between isomorphism and isogeny types); H120E19 (generic map constructors for the doubling and negation maps); H120E20 (computing the dual isogeny via PushThroughIsogeny and Vélu's formulae; verification via DualIsogeny).*

### 120.8.2 Predicates on Isogenies

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsZero(I)` / `IsConstant(I)` | Returns true iff the image of the isogeny I is the zero element of its codomain. | — |
| `I eq J` | Returns true iff isogenies I and J are equal. | Map equality. |

### 120.8.3 Structure Operations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsogenyMapPsi(I)` | Returns the univariate polynomial ψ used in defining I; roots of ψ determine the kernel. | — |
| `IsogenyMapPsiMulti(I)` | Returns ψ as a multivariate polynomial. | — |
| `IsogenyMapPsiSquared(I)` | Returns the denominator of the rational function giving the image of the x-coordinate under I. | — |
| `IsogenyMapPhi(I)` | Returns the univariate polynomial φ used in defining I (numerator of x-image). | — |
| `IsogenyMapPhiMulti(I)` | Returns φ as a multivariate polynomial. | — |
| `IsogenyMapOmega(I)` | Returns the multivariate polynomial ω used in defining I. | — |
| `Kernel(I)` | Returns the subgroup of the domain mapping to zero under I. | — |
| `Degree(I)` | Returns the degree of the morphism I. | — |

### 120.8.4 Endomorphisms

The endomorphism ring End(E) of an elliptic curve E contains a subring isomorphic to Z; a curve with additional endomorphisms has complex multiplication. Over finite fields E always has CM, and the endomorphism ring is isomorphic to an order in a quadratic number field or quaternion algebra.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `MultiplicationByMMap(E, m)` | Returns the multiplication-by-m endomorphism [m] : E → E with [m](P) = m * P. | Division polynomial / formal group; Vélu for general m. |
| `IdentityIsogeny(E)` | Returns the identity map E → E as an isogeny. | — |
| `IdentityMap(E)` | Returns the identity map E → E as an isomorphism with defining coefficients [r, s, t, u] = [0, 0, 0, 1]. | — |
| `NegationMap(E)` | Returns the isomorphism of E mapping P to −P. | Negation on E. |
| `FrobeniusMap(E, i)` | Returns the Frobenius isogeny (x : y : 1) ↦ (x^(pⁱ) : y^(pⁱ) : 1) for E over a finite field of characteristic p. | — |
| `FrobeniusMap(E)` | Equivalent to `FrobeniusMap(E, 1)`. | — |

*Worked example: H120E21 (FrobeniusMap action: fixed points of Frobenius on E₂ over GF(23²) coincide with points of E₁ over GF(23)).*

### 120.8.5 Automorphisms

Functions dealing with automorphisms in the category of elliptic curves (automorphisms of the underlying curve that also preserve the group structure, i.e. fix O_E). Warning: for a general curve in Magma (Crv but not CrvEll) over a finite field, `AutomorphismGroup` and `Automorphisms` give curve automorphisms, not group-preserving ones.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AutomorphismGroup(E)` | For E over a general field K, returns the group A of automorphisms of E defined over K (an abelian or polycyclic group) together with a map A → actual automorphisms (scheme maps E → E). | Classification of automorphism groups of elliptic curves. |
| `Automorphisms(E)` | For E over a general field K, returns a sequence containing the elements of the AutomorphismGroup of E. | — |

---

## 120.9 Operations on Points

Points on an elliptic curve over a field are given in projective coordinates (x : y : z) (equivalence class under scaling by a nonzero field element). Normalised: z = 1 or (z = 0 and y = 1). The identity point is O = (0 : 1 : 0).

### 120.9.1 Creation of Points

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `H ! [x, y, z]` / `elt< H \| x, y, z >` / `E ! [x, y, z]` / `elt< E \| x, y, z >` | Given a point set H = E(R) and coordinates x, y, z in R satisfying the equation of E, returns the normalised point (x : y : z) in H. If z is not specified, z = 1 is assumed. | — |
| `H ! 0` / `Id(H)` / `Identity(H)` / `E ! 0` / `Id(E)` / `Identity(E)` | Returns the normalised identity point (0 : 1 : 0). | — |
| `Points(H, x)` / `Points(E, x)` | Returns the sequence of points in H on E whose x-coordinate is x. | — |
| `Points(H)` / `RationalPoints(H)` / `Points(E)` / `RationalPoints(E)` | Returns an indexed set of points on E (or in H). Over finite fields: all rational points. Over Q or number fields: requires a positive `Bound` (x-coordinate height over Q; similar box search over number fields). Additional parameters for number fields: `DenominatorBound` (default = Bound), `Denominators` (set of integral elements), `Max` (return as soon as this many points found), `NPrimes` (default 30, controls sieve primes). Algorithm: sieve method; number field case described in Appendix A of **[Bru02]**. | Sieve method; number field case: **[Bru02]** Appendix A. |
| `PointsAtInfinity(H)` / `PointsAtInfinity(E)` | Returns the indexed set containing the identity point of the point set H or E. | — |

### 120.9.2 Creation Predicates

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsPoint(H, S)` / `IsPoint(E, S)` | Returns true if the sequence S gives coordinates of a point in H or on E; also returns the point when true. | — |
| `IsPoint(H, x)` / `IsPoint(E, x)` | Returns true if x is the x-coordinate of a point in H or on E; also returns a corresponding point when true. Note: the point at infinity is never returned. | — |

### 120.9.3 Access Operations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `P[i]` | Returns the i-th coordinate of point P (1 ≤ i ≤ 3). | — |
| `ElementToSequence(P)` / `Eltseq(P)` | Returns a normalised sequence of length 3 of coordinates of P. | — |

### 120.9.4 Associated Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Category(P)` / `Type(P)` | Returns `PtEll`, the category of elliptic curve points. | — |
| `Parent(P)` | Returns the parent point set of P. | — |
| `Scheme(P)` / `Curve(P)` | Returns the corresponding scheme or elliptic curve of the parent point set of P. | — |

### 120.9.5 Arithmetic

The points on an elliptic curve over a field form an abelian group under additive notation with identity O = (0 : 1 : 0).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `-P` | Returns the additive inverse of P. | Group law on Weierstrass model. |
| `P + Q` | Returns the sum P + Q of two points on the same elliptic curve. | Group law (chord-and-tangent). |
| `P +:= Q` | Sets P equal to P + Q. | Group law. |
| `P - Q` | Returns P − Q. | Group law. |
| `P -:= Q` | Sets P equal to P − Q. | Group law. |
| `n * P` | Returns the n-th multiple of P. | Double-and-add (scalar multiplication). |
| `P *:= n` | Sets P equal to n * P. | Double-and-add. |

### 120.9.6 Division Points

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `P / n` | Given P on E and integer n, returns a point Q with P = nQ if one exists; runtime error otherwise. | Division polynomial / exhaustive search. |
| `P /:= n` | Sets P equal to a point Q with P = nQ if one exists; runtime error otherwise. | — |
| `DivisionPoints(P, n)` | Returns the sequence of all points Q on E with P = nQ; empty sequence if none. | Division polynomial roots. |
| `IsDivisibleBy(P, n)` | Returns true if P is n-divisible, and if so, an n-division point. | Division polynomial. |

*Worked examples: H120E22 (DivisionPoints for a rational curve showing 3 three-division points; 9-torsion via DivisionPoints(E!0, 9)); H120E23 (point arithmetic and reduction to a minimal generating set over Q); H120E24 (generic point on an elliptic surface; MultiplicationByMMap verification).*

### 120.9.7 Point Order

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Order(P)` | For P on E over Q or a finite field, returns the smallest positive n with n·P = O; returns 0 for infinite order. For finite fields, computes the curve order first. | Baby-step giant-step or Pohlig–Hellman; uses group order of E. |
| `FactoredOrder(P)` | For P on E over Q or a finite field, returns the factorisation of the order of P. For finite fields, repeated calls are faster since the factored group order is stored. Errors if P has infinite order over Q. | Pohlig–Hellman using stored factored group order. |

*Worked example: H120E25 (Order and FactoredOrder of points on a curve over GF(NextPrime(10¹²))).*

### 120.9.8 Predicates on Points

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsId(P)` / `IsIdentity(P)` / `IsZero(P)` | Returns true iff P is the identity point of its point set. | — |
| `P eq Q` | Returns true iff P and Q are on the same elliptic curve with the same normalised coordinates. | Coordinate comparison. |
| `P ne Q` | Logical negation of eq. | — |
| `P in H` | Returns true iff P is in the point set H: satisfies the equation of E and has coordinates in R (where H = E(R)). | — |
| `P in E` | Returns true iff P is on the elliptic curve E (satisfies the defining equation); P need not lie in the base point set of E. | — |
| `IsOrder(P, m)` | Returns true iff P has order m. Likely much faster than calling Order(P) when the order is already known or conjectured. | Verify m*P = O and check m is minimal. |
| `IsIntegral(P)` | For P on E over Q, returns true iff the coordinates of the normalisation of P are integers. | — |
| `IsSIntegral(P, S)` | For P on E over Q and S a sequence of primes, returns true iff the denominators of x(P) and y(P) are supported only by primes in S. | — |

*Worked example: H120E26 (integral and S-integral points on y² = x³ + 17 using IsIntegral and IsSIntegral with S = [2, 3]).*

### 120.9.9 Weil Pairing

Magma contains an optimised implementation of the Weil pairing on an elliptic curve, used in computing the group structure over finite fields.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `WeilPairing(P, Q, n)` | Given n-torsion points P and Q of E, computes the Weil pairing of P and Q. | Optimised Weil pairing; used internally for group structure computation over finite fields. |
| `IsLinearlyIndependent(S, n)` | Given a sequence S of points of E each with order dividing n, returns true iff the points are linearly independent over Z/nZ. | Via Weil pairing. |
| `IsLinearlyIndependent(P, Q, n)` | Returns true iff P and Q form a basis of the n-torsion points of E. | Via Weil pairing. |

*Worked example: H120E27 (Weil pairing of two 3-torsion points on y² = x³ − 3 over Q; coercion into a composite number field of degree 6; result is a primitive cube root of unity).*

---

## 120.10 Bibliography

| Key | Reference |
|-----|-----------|
| **[Bru02]** | N. R. Bruin. *Chabauty methods and covering techniques applied to generalized Fermat equations*, volume 133 of CWI Tract. Stichting Mathematisch Centrum Centrum voor Wiskunde en Informatica, Amsterdam, 2002. Dissertation, University of Leiden, Leiden, 1999. |
| **[Cas91]** | J. W. S. Cassels. *Lectures on elliptic curves*, volume 24 of London Mathematical Society Student Texts. Cambridge University Press, Cambridge, 1991. |
| **[Con99]** | I. Connell. *Elliptic Curve Handbook.* McGill University, 1999. |
| **[Cre97]** | J. E. Cremona. *Algorithms for modular elliptic curves.* Cambridge University Press, Cambridge, second edition, 1997. |
| **[Nag28]** | Trygve Nagell. Sur les propriétés arithmétiques des cubiques planes du premier genre. *Acta Math.*, 52:93–126, 1928. |
| **[Sil86]** | J. Silverman. *The arithmetic of elliptic curves*, volume 106 of Graduate Texts in Mathematics. Springer-Verlag, New York, 1986. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Weierstrass model / direct definition | `EllipticCurve`, `IsEllipticCurve`, `EllipticCurveFromjInvariant` |
| Nagell's algorithm **[Nag28, Cas91]** (plane cubic, point given) | `EllipticCurve(C, P)` (degree-3 case) |
| Riemann–Roch (general genus-1 curve) | `EllipticCurve(C)`, `EllipticCurve(C, P)` (fallback), `EllipticCurve(C, pl)` |
| Local minimisation / minimal models | `MinimalModel`, `MinimalModel(E, P)`, `IntegralModel` |
| Simplified model (char 2, 3) **[Con99]** | `SimplifiedModel` |
| Vélu's formulae (isogenies from kernel) **[Sil86]** | `IsogenyFromKernel`, `IsogenyFromKernelFactored`, `TwoIsogeny`, `DualIsogeny`, `PushThroughIsogeny` |
| Group law (chord-and-tangent) | `+`, `-`, `*`, `DivisionPoints`, `IsDivisibleBy` |
| Division polynomials (torsion subschemes) | `DivisionPolynomial`, `TwoTorsionPolynomial`, `TorsionSubgroupScheme` |
| Weil pairing | `WeilPairing`, `IsLinearlyIndependent` |
| Sieve method for rational points **[Bru02]** | `Points` / `RationalPoints` (over Q and number fields) |
| Point counting (finite fields, isogeny detection) | `IsIsogenous` (finite field case), `Order(P)`, `FactoredOrder(P)` |
| Formal group | `FormalGroupLaw`, `FormalGroupHomomorphism`, `FormalLog` |
| Cremona's Mordell–Weil algorithms **[Cre97]** | (Chapter 122; referenced in introduction) |
