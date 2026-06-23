# Chapter 112 — Schemes

**Handbook part:** XV — Algebraic Geometry
**Handbook pages:** 3475–3599 (PDF pages 3602–3733)

---

## Scope and overview

Schemes in Magma are geometric objects defined by the vanishing of polynomial equations in affine or projective space. Standard references include Hartshorne's introductory text [Har77] and Reid's student text [Rei88]. Magma does not support entirely general schemes (no a-priori affine-patch gluing), but it provides a rich working environment covering:

- **Ambient spaces** — affine spaces, projective spaces (possibly weighted), rational scrolls, and product projective spaces — all with an associated polynomial coordinate ring.
- **Schemes** — subschemes of an ambient defined by vanishing polynomials or ideals. Affine schemes correspond bijectively to ideals of their coordinate ring; projective schemes are associated to a largest *saturated* ideal (saturation is deferred until needed and cached).
- **Points and point sets** — points are elements of a *point set* X(L) for a k-algebra L, not elements of the scheme itself, allowing coordinates in extensions without base-changing the scheme.
- **Maps** — rational maps defined by sequences of polynomials or rational functions; two maps are equal if they agree on a common dense open subset.
- **Linear systems** — finite-dimensional linear spaces of hypersurfaces of a common degree, closely related to maps and to Riemann-Roch theory.
- **Divisors** — on projective varieties of dimension > 1, with Riemann-Roch spaces and a Zariski-decomposition algorithm for surfaces.

Groebner basis calculations require an exact field or Euclidean domain; resultants require a UFD; factorisation requires Z, Q, finite fields, algebraically closed fields, number fields, or function fields. Linear systems are restricted to ambient spaces over fields.

Many specialised functions (for curves, conics, elliptic curves, hyperelliptic curves, modular curves, algebraic surfaces) live in other chapters (114, 119, 125, 128, 130); this chapter documents general-scheme functionality only.

---

## 112.1 Introduction and First Examples

This section introduces the key idioms through worked examples: creation of ambient spaces, definition of subschemes, rational points and point sets, projective closure and affine patches, maps, linear systems, and an aside on the Magma type hierarchy for schemes.

### 112.1.1 Ambient Spaces

Ambient spaces (affine and projective) are the containers for schemes. The coordinate ring is a polynomial ring; names are assigned via angle-bracket notation or `AssignNames`.

*Worked examples: H112E1 (affine 3-space over GF(23)), H112E7 (renaming coordinates).*

### 112.1.2 Schemes

Schemes are defined by polynomial equations on an ambient. The projective case requires homogeneous polynomials; Magma checks homogeneity at construction. The ambient of a subscheme is always the top-level ambient, not the intermediate superscheme.

*Worked examples: H112E2 (twisted cubic via 2×3 minors).*

### 112.1.3 Rational Points

Points are elements of a point set X(L). The coercion `X ! [a,b,...]` is shorthand when coordinates lie in the base ring; otherwise `X(L) ! [...]` is required.

*Worked examples: H112E3 (parabola, coercion, `in` test), H112E9 (point comparison across different k-algebras).*

### 112.1.4 Projective Closure

Affine schemes have a unique projective closure; projective schemes have standard affine patches. Closure and patching are cached and referentially stable.

*Worked examples: H112E4 (AffinePatch, ProjectiveClosureMap), H112E20, H112E21.*

### 112.1.5 Maps

Maps between schemes are defined by polynomial or rational-function sequences. Compatibility with projective gradings is checked; map images require Groebner basis computation.

*Worked examples: H112E5 (rational map P1 → P2, Image, Pullback, PointsOverSplittingField).*

### 112.1.6 Linear Systems

Linear systems are parametrised families of hypersurfaces of a fixed degree. Sections define maps to projective space (e.g. the Veronese embedding).

*Worked examples: H112E6 (conics in P2, Veronese surface via sections), H112E17 (projection from a point, cubic scroll).*

### 112.1.7 Aside: Types of Schemes

Main type `Sch`; specialised subtypes include `Aff`, `Prj` (with scrolls, multi-graded variants), `Crv`, `Clstr`, and further curve subtypes `CrvCon`, `CrvRat`, `CrvEll`, `CrvHyp`.

---

## 112.2 Ambients

For any scheme, there is a containing ambient space whose coordinate ring is a polynomial ring. Creation functions are listed here.

### 112.2.1 Affine and Projective Spaces

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AffineSpace(k,n)` | n-dimensional affine space over ring k. | — |
| `ProjectiveSpace(k,n)` / `ProjectiveSpace(k,W)` | n-dimensional projective space over k, optionally with a weight sequence W for the coordinate functions. | — |
| `AffineSpace(R)` / `Spec(R)` | Affine space whose coordinate ring is the multivariate polynomial ring R; names inherited from R. | — |
| `ProjectiveSpace(R)` / `Proj(R)` | Projective space whose homogeneous coordinate ring is R; grading from R or standard-by-degree if none. | — |
| `AssignNames(~A,N)` | Procedure: change the print names of the first #N coordinate functions of ambient A; does not assign variables to identifiers. | — |
| `A . i` / `Name(A,i)` | The i-th coordinate function of A as an element of the coordinate ring. | — |
| `A eq B` | True iff A and B are the same instance of ambient creation (identity, not isomorphism). | — |

### 112.2.2 Scrolls and Products

Multi-graded ambient spaces for rational ruled surfaces and product projective spaces.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DirectProduct(A,B)` | Product ambient A×B, returns the ambient together with the two projection maps. | — |
| `RuledSurface(k,a,b)` | Ruled surface over k whose negative section has self-intersection ±(a−b); a,b non-negative integers. | — |
| `RuledSurface(k,n)` | Ruled surface Fn over k (negative section self-intersection −n); standard gradings [1,1,−n,0],[0,0,1,1]. | — |
| `AbsoluteRationalScroll(k,N)` | Rational scroll with base ring k and grading entries −N for a sequence N of non-negative integers. | — |
| `ProductProjectiveSpace(k,N)` | Product P^n1 × … × P^nr of projective spaces with dimensions from sequence N=[n1,…,nr] over k. | — |
| `SegreProduct(Xs)` | Embeds a sequence of schemes lying in ordinary projective spaces into ordinary projective space via the iterated Segre embedding; returns the product and the r projection maps. **[Har77 Ex. 2.14]** | Iterated Segre embedding **[Har77]**. |
| `SegreEmbedding(X)` | X lies in a product projective ambient; computes and returns the image in ordinary projective space under the iterated Segre embedding together with an isomorphism X → image. Faster than using the general map machinery. | Iterated Segre embedding **[Har77]**. |

*Worked example: H112E8 (Segre embedding of E×E).*

### 112.2.3 Functions and Homogeneity on Ambient Spaces

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CoordinateRing(A)` | Polynomial ring of rank n over BaseRing(A); the coordinate ring of ambient A. | — |
| `FunctionField(A)` | Field of fractions of the coordinate ring of A. | — |
| `HasFunctionField(A)` | Returns true if A has a function field. | — |
| `Gradings(X)` | Sequence of all gradings on projective space X (or any scheme in X); each grading is a sequence of integers of length = #coordinates. | — |
| `NumberOfGradings(X)` / `NGrad(X)` | Number of independent gradings on the projective ambient of X. | — |
| `NumberOfCoordinates(X)` / `Length(X)` | Number of coordinate functions of the ambient of X. | — |
| `Lengths(X)` | Lengths of the groups of ones in the gradings of a scroll X. | — |
| `IsHomogeneous(X,f)` | True iff polynomial f is homogeneous with respect to all gradings on X. | — |
| `Multidegree(X,f)` | Sequence of homogeneous degrees of f with respect to each grading of X. | — |

### 112.2.4 Prelude to Points

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `A ! [a,b,...]` / `A(L) ! [a,b,...]` | Create the affine point (a,b,…) or projective point (a:b:…) in the ambient A or its point set A(L). Projective points are normalised so the last nonzero coordinate is 1 (or analogously for multi-graded ambients). | — |
| `Origin(A)` | The origin of affine space A. | — |
| `Simplex(A)` | Sequence of standard simplex points of A: (1,0,…,0), …, (0,…,0,1), (1,…,1). | — |
| `Coordinates(p)` | Sequence of ring elements giving the coordinates of point p. | — |
| `p[i]` / `Coordinate(p,i)` | The i-th coordinate of point p. | — |
| `p @ f` / `f(p)` / `Evaluate(f, p)` | Evaluate a function f (from the function field of X or its ambient) at the point p lying on X. | — |

*Worked example: H112E10 (evaluating function field elements at points).*

---

## 112.3 Constructing Schemes

Schemes are defined inside an ambient (affine or projective) or inside another scheme, by vanishing of polynomials, sequences of polynomials, ideals, or quotient ring denominators. Saturation of projective ideals is deferred and cached; the `Saturated` parameter may mark the input as already saturated.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Scheme(X,f)` / `Scheme(X,F)` / `Scheme(X,I)` / `Scheme(X,Q)` | Create the subscheme of X defined by the vanishing of polynomial f, sequence F, ideal I, or quotient ring Q=R/I. Optional Boolean parameters `Nonsingular`, `Reduced`, `Irreducible`, `GeometricallyIrreducible`, `Saturated` allow asserting properties without checking. | — |
| `Cluster(X,f)` / `Cluster(X,F)` / `Cluster(X,I)` / `Cluster(X,Q)` | As Scheme, but also performs a dimension test and returns a zero-dimensional scheme (cluster). Optional parameter `Saturated`. | — |
| `Spec(R)` | The affine scheme Spec(R) associated to affine algebra R; ambient = Spec(Generic(R)). | — |
| `Proj(R)` | The projective scheme Proj(R) associated to graded algebra R; ambient = Proj(Generic(R)). | — |
| `EmptyScheme(X)` / `EmptySubscheme(X)` | The empty subscheme of X (defined by 1 in the affine case, or by the irrelevant ideal in the projective case); marked saturated. | — |
| `X meet Y` / `Intersection(X,Y)` | Intersection: concatenates defining equations without emptiness check. | — |
| `X join Y` / `Union(X,Y)` | Union: computes the intersection of defining ideals via Groebner basis; result is saturated if both inputs are. | Groebner basis. |
| `&join S` | Union of all schemes in sequence S. | Groebner basis. |
| `Difference(X, Y)` | Closure of X \ (X ∩ Y) with multiplicity; ideal is the colon ideal of ideals(X) and ideal(Y); saturated if X is. | Ideal colon. |
| `RemoveLinearRelations(X)` | Use linear relations between variables on X to eliminate variables; returns scheme Y in a lower-dimensional projective space isomorphic to the smallest linear subspace containing X, together with the (linear) isomorphism. Currently only for ordinary projective space. | Linear algebra. |
| `BlowUp(X,Y)` / `BlowUp(X,p)` | Blow up subscheme Y (or point p) of scheme X. Parameter `Ordinary` (default true) embeds the blowup in ordinary projective space via Segre; otherwise returns a product ambient. Uses `ReesIdeal`. **[Har77 §II.7]** | Rees ideal construction **[Har77]**. |
| `Saturate(~X)` | If X is projective and not already saturated, saturate its defining ideal. | Ideal saturation. |
| `AssignNames(~X,N)` | Assign strings N to the ambient coordinate functions of scheme X. | — |
| `X . i` / `Name(X,i)` | The i-th coordinate function of the ambient of X. | — |

*Worked examples: H112E11 (creating subschemes), H112E12 (Difference behaviour).*

---

## 112.4 Different Types of Scheme

Type-checking and type-change predicates. Each returns a Boolean; if true, may also return a new scheme of the specialised type.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsAffine(X)` | True iff X is an affine space. | — |
| `IsProjective(X)` | True iff X is a projective space (including scrolls). | — |
| `IsOrdinaryProjectiveSpace(X)` | True iff X is projective space with a single grading in which all variables have weight 1. | — |
| `IsAmbient(X)` | True iff X is an ambient space. | — |
| `IsCluster(X)` | True iff X is a zero-dimensional scheme (not the empty scheme). | — |
| `IsCurve(X)` | True iff X is a one-dimensional scheme. | — |
| `IsPlaneCurve(X)` | True iff X is a one-dimensional scheme defined by a single equation in a two-dimensional ambient. | — |
| `IsConic(X)` | True iff X is a nonsingular curve defined by an equation of degree 2. | — |
| `IsRationalCurve(X)` | True iff X is a curve of genus 0. | — |
| `IsHyperellipticCurve(X)` | True (and returns a CrvHyp) iff X is already of CrvHyp type or is defined by a nonsingular Weierstrass equation in correctly weighted P2. | — |
| `IsModularCurve(X)` | True iff X is of type CrvMod. | — |

---

## 112.5 Basic Attributes of Schemes

### 112.5.1 Functions of the Ambient Space

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AmbientSpace(X)` / `Ambient(X)` | The ambient space containing X. | — |
| `SuperScheme(X)` | The scheme X was created as a subscheme of. | — |
| `BaseRing(X)` / `CoefficientRing(X)` | The base ring of X. | — |
| `BaseField(X)` / `CoefficientField(X)` | The base ring of X if it is a field; error otherwise. | — |
| `IsAffine(X)` | True iff the ambient of X is affine. | — |
| `IsProjective(X)` | True iff the ambient of X is projective. | — |
| `IsOrdinaryProjective(X)` | True iff the ambient of X is ordinary projective space (coordinate ring generated in degree 1). | — |
| `IsPlanar(X)` | True iff the ambient of X is 2-dimensional. | — |
| `IsSaturated(X)` | True iff the current defining ideal of X is saturated. | — |

### 112.5.2 Functions of the Equations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DefiningPolynomials(X)` | Sequence of polynomials defining the ideal of X (no Groebner basis overhead). | — |
| `DefiningPolynomial(X)` | The defining polynomial of X if X is a hypersurface; error otherwise. | — |
| `DefiningIdeal(X)` | The ideal of X as a multivariate polynomial ring ideal. | — |
| `CoordinateRing(X)` | Quotient of the ambient coordinate ring by the ideal of X. | — |
| `Curve(X)` | The smallest scheme in the inclusion chain above X which is a curve. | — |
| `GroebnerBasis(X)` | Sequence of polynomials of a Groebner basis of the defining ideal of X; updates the ideal's stored basis. | Groebner basis. |
| `MinimalBasis(X)` | A minimal basis of the defining ideal of X (no redundant generators); most human-readable basis Magma can provide. | Groebner basis. |
| `IsHypersurface(X)` | True iff X is definable by a single polynomial; performs a GCD simplification; returns the polynomial as a second value. | GCD computation. |
| `JacobianIdeal(X)` | The ideal of partial derivatives of the defining polynomials. | — |
| `JacobianMatrix(X)` | The matrix (∂fi/∂xj) of partial derivatives. | — |
| `HessianMatrix(X)` | The Hessian matrix (∂²f/∂xi∂xj) of hypersurface X. | — |
| `X eq Y` | True iff X and Y have the same types, ambients, and ideals (projective schemes are saturated before ideal comparison). | Groebner basis. |
| `IsSubscheme(X, Y)` | True iff X is contained scheme-theoretically in Y; checks reverse ideal inclusion (X saturated first if projective). | Groebner basis. |
| `IsLinear(X)` | True iff X is defined by linear equations (possibly after computing a Groebner basis). | Groebner basis. |

*Worked example: H112E13 (subscheme inclusion and equality tests).*

---

## 112.6 Function Fields and their Elements

Function fields of irreducible varieties are birational invariants. For an affine scheme X, `FunctionField(X)` equals `FunctionField(ProjectiveClosure(X))`. Currently supported: function fields of projective/affine spaces and curves. Type: `FldFunFracSch`, elements: `FldFunFracSchElt`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Scheme(F)` | The (projective) scheme whose function field is F. | — |
| `IntegerRing(F)` / `Integers(F)` | The integer ring of F (coordinate ring of one patch of the scheme of F). | — |
| `AssignNames(~F, S)` | Assign strings S as names of the integer ring of F. | — |
| `F ! g` | Coerce element g (a function on the scheme of F) into the function field F. | — |
| `F . i` | The i-th indeterminate of the coordinate ring of the scheme of F as an element of F. | — |
| `ProjectiveFunction(f)` | Given f in a function field, return f as an element of the field of fractions of the coordinate ring of its scheme. | — |
| `ProjectiveRationalFunction(f)` | Return f as an element of the field of fractions of the coordinate ring of the ambient of the scheme, whose restriction to the scheme is f. | — |
| `RestrictionToPatch(f, Xi)` | Return f as an element of the field of fractions of the coordinate ring of the patch Xi of its scheme. | — |
| `Numerator(f)` / `Denominator(f)` | Numerator and denominator of a function field element f. | — |
| `f * g`, `f + g`, `f - g`, `-f`, `f / g`, `f ^ n` | Arithmetic on function field elements. | — |
| `f eq g` / `IsZero(f)` / `IsOne(f)` / `IsMinusOne(f)` / `IsUnit(f)` | Comparison and unit tests for function field elements. | — |
| `IntegralSplit(f, X)` | For f on projective scheme X, return the numerator and denominator of a rational function g on the ambient P restricting to f. | — |
| `Numerator(f, X)` / `Denominator(f, X)` | First/second return of `IntegralSplit(f, X)`. | — |
| `Restriction(f, Y)` | Restrict function field element f from scheme X to subscheme Y with a function field; returns Infinity if f has a pole along Y. | — |
| `GenericPoint(X)` | A point in X(FunctionField(X)) whose coordinates generate FunctionField(X). | — |

*Worked example: H112E14 (conversion of function field elements between patches and ambient).*

---

## 112.7 Rational Points and Point Sets

Points are elements of a point set X(L) (a k-algebra L); the word "point" always refers to such an object.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `X(L)` / `PointSet(X,L)` / `X(m)` / `PointSet(X,m)` | The point set of X of points with coordinates in ring L or in the codomain of ring homomorphism m. | — |
| `P eq Q` | True iff point sets P and Q were created on the same scheme with the same base-ring map. | — |
| `Scheme(P)` | The scheme associated to the point set P. | — |
| `Curve(P)` | The smallest scheme in the inclusion chain above the scheme of P which is a curve. | — |
| `Ring(P)` | The ring L of the point set P = X(L). | — |
| `RingMap(P)` | The map from BaseRing(scheme of P) to Ring(P). | — |
| `X ! Q` / `X(L) ! Q` | Point of X (or X(L)) with coordinate sequence Q; universe of Q must be coercible to BaseRing(X) or L. | — |
| `p eq q` | True iff p and q lie in a common scheme (possibly after coercion) and their coordinates are equal. | — |
| `p in X` | True iff point p lies in scheme X or is coercible into it. | — |
| `Scheme(p)` | The scheme on which point p lies. | — |
| `Curve(p)` | The smallest scheme in the inclusion chain above the scheme of p which is a curve. | — |
| `Q in X` | True iff all points of set/sequence Q lie in X or are coercible into it. | — |
| `S subset X` | True iff all points of set S lie in X or are coercible into it. | — |
| `IsCoercible(X,Q)` | True iff Q is the coordinate sequence of some point of X; also returns the point. | — |
| `RationalPoints(X)` / `RationalPoints(X,L)` / `Points(X)` / `Points(X,L)` | Indexed set of points in X(L) (default L = BaseField(X)). Works for: (i) L finite field (all points); (ii) X zero-dimensional (all points); (iii) L = Rationals() via PointSearch up to `Bound` (default 1000). | Groebner basis for dimension; finite-field enumeration; or p-adic point search. |
| `RationalPointsByFibration(X)` | Enumerate rational points over a finite field by Noether normalization fibration. Parameter `UseHypersurface` (default false) uses a two-stage variant. | Noether normalisation fibration. |
| `Random(S)` | A random point from the point set S = X(k) for X over a finite field k (not uniformly distributed). | Noether normalisation. |
| `HasNonsingularPoint(X)` / `HasNonsingularPoint(X,L)` | True iff X (over finite field, optionally over L) contains a nonsingular point; returns such a point if true. | — |

*Worked example: H112E15 (points over finite field, IsCoercible, RationalPoints, RationalPointsByFibration).*

---

## 112.8 Zero-dimensional Schemes

Clusters are zero-dimensional schemes. The word "cluster" refers to schemes known to be zero-dimensional; their ideals may be nonradical.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Cluster(p)` / `Cluster(X, p)` | Reduced scheme supported at point p, optionally as a subscheme of X. | — |
| `Cluster(S)` / `Cluster(X, S)` | Reduced scheme supported at the set of points S, optionally as a subscheme of X. | — |
| `RationalPoints(Z)` / `RationalPoints(Z,L)` | Set of rational points of cluster Z (over L if given). | — |
| `PointsOverSplittingField(Z)` | Finds a point set Z(L) containing all geometric points of cluster Z over an algebraic closure, and returns all such points. | — |
| `HasPointsOverExtension(X)` / `HasPointsOverExtension(X,L)` | False iff all geometric points of X are already defined over the current base field (or over L). | — |
| `Degree(Z)` | Degree of cluster Z; equals #support over an algebraic closure when Z is reduced. | — |

*Worked example: H112E16 (intersection of two plane curves as a cluster, PointsOverSplittingField).*

---

## 112.9 Local Geometry of Schemes

Local intrinsics accept either a point alone (using the point's parent scheme) or both a scheme and a point.

### 112.9.1 Point Conditions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsSingular(X,p)` | True iff p is a singular point of X. | Jacobian criterion. |
| `IsNonsingular(X,p)` | True iff p is a nonsingular point of X. | Jacobian criterion. |
| `IsOrdinarySingularity(X,p)` | True iff the tangent cone to X at p is reduced and X is singular at p. Currently only for hypersurfaces. | Tangent-cone computation. |

### 112.9.2 Point Computations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Multiplicity(p)` / `Multiplicity(X,p)` | Multiplicity of p as a point of X. Non-hypersurface case uses local Groebner bases. | Local Groebner basis. |
| `TangentSpace(p)` / `TangentSpace(X,p)` | Tangent space to X at p, embedded in the same ambient; error if p is singular or not rational. | Jacobian matrix. |
| `TangentCone(p)` / `TangentCone(X,p)` | Tangent cone to X at p embedded in the same ambient. Non-hypersurface case uses local Groebner bases. | Local Groebner basis. |

### 112.9.3 Analytically Hypersurface Singularities

Tests whether an isolated singular point p is analytically equivalent to a hypersurface singularity (completion of local ring ≅ k[[x1,…,xd]]/(F)) and expands F to a given precision using local Groebner bases after localising to an affine patch and translating to the origin.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsHypersurfaceSingularity(p,prec)` | Tests whether isolated singular p (in X(k) for field k) is a hypersurface singularity: tangent space dimension d+1 and local ring is a local complete intersection. If true, also returns a polynomial F1 (the analytic equation to precision prec), a sequence of rational coordinate functions, and a data record for further expansion. | Local Groebner bases after affine localisation and translation. |
| `HypersurfaceSingularityExpandFurther(dat,prec,R)` | Using data record dat from `IsHypersurfaceSingularity`, expand the analytic equation F to include all terms of degree ≤ prec; result in polynomial ring R of rank d+1. | Local Groebner bases. |
| `HypersurfaceSingularityExpandFunction(dat,f,prec,R)` | Expand a rational function f on X at the singularity to precision prec in the analytic coordinate ring; returns two polynomials a, b in R such that a/b is the finite approximation. | Local Groebner bases. |

*Worked example: H112E17 (Del Pezzo surface with conjugate singular points; IsHypersurfaceSingularity, ExpandFurther, ExpandFunction).*

---

## 112.10 Global Geometry of Schemes

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Dimension(X)` | Dimension of the highest-dimensional component of X (empty scheme returns −1); Groebner basis if not cached; projective multi-graded X is saturated first. | Groebner basis (Hilbert function). |
| `Codimension(X)` | Codimension = Dimension(Ambient) − Dimension(X). | — |
| `Degree(X)` | Degree of X. | Hilbert polynomial. |
| `ArithmeticGenus(X)` | Arithmetic genus of X; ambient must be ordinary projective space. | Hilbert polynomial. |
| `IsEmpty(X)` | True iff X has no points over any algebraic closure; tests triviality of the ideal then applies Nullstellensatz. | Nullstellensatz / Groebner basis. |
| `IsNonsingular(X)` | True iff X is nonsingular and equidimensional over an algebraic closure; tests emptiness of the scheme defined by appropriate minors of the Jacobian matrix. | Jacobian criterion + IsEmpty. |
| `IsSingular(X)` | True iff X has a singular point or fails to be equidimensional. | Jacobian criterion. |
| `SingularSubscheme(X)` | Subscheme defined by appropriate-sized minors of the Jacobian; lower-dimensional components of X are included even if nonsingular. | Jacobian minors. |
| `PrimeComponents(X)` | Irredundant prime (irreducible) components of X. | Primary decomposition. |
| `PrimaryComponents(X)` | Irredundant primary components of X. | Primary decomposition. |
| `ReducedSubscheme(X)` | Subscheme of X with reduced structure, with the map to X; uses Groebner basis to compute the radical. | Radical of ideal via Groebner basis. |
| `IsIrreducible(X)` | True iff X has a unique prime component; Groebner basis needed if not a hypersurface; projective X is saturated first. | Primary decomposition / Groebner basis. |
| `IsReduced(X)` | True iff the defining ideal equals its radical; hypersurface case uses derivatives only; otherwise Groebner basis (projective X saturated first). | Radical test / Groebner basis. |
| `IsCohenMacaulay(X)` | True iff every scheme-theoretic point of X has a Cohen-Macaulay local ring; requires X equidimensional and ordinary projective. Parameter `CheckEqui`. | Minimal free resolution. |
| `IsGorenstein(X)` | True iff X is Cohen-Macaulay and the canonical sheaf is locally free of rank 1; computationally heavy. Requires X equidimensional and ordinary projective. | Minimal free resolution + canonical sheaf check. |
| `IsArithmeticallyCohenMacaulay(X)` | True iff the coordinate ring of X satisfies the Cohen-Macaulay property; implies `IsCohenMacaulay`; ordinary projective, equidimensional. | Minimal free resolution. |
| `IsArithmeticallyGorenstein(X)` | True iff the coordinate ring of X is Gorenstein; implies `IsArithmeticallyCohenMacaulay`. | Minimal free resolution. |

*Worked example: H112E18 (dimension, reducedness, primary components, ReducedSubscheme).*

---

## 112.11 Base Change for Schemes

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BaseChange(A,K)` / `BaseExtend(A,K)` | Base change of ambient or scheme A to field K (automatic coercion from BaseRing(A) to K required). No cached data transferred. | ChangeRing on polynomial ring. |
| `BaseChange(A,m)` / `BaseExtend(A,m)` | Base change of A via ring homomorphism m whose domain is BaseRing(A). | ChangeRing via m. |
| `BaseChange(F,K)` / `BaseExtend(F,K)` / `BaseChange(F,m)` / `BaseExtend(F,m)` | Base change of a sequence F of schemes in a common ambient, to field K or via map m. | — |
| `BaseChange(X,A)` / `BaseExtend(X,A)` / `BaseChange(X,A,m)` / `BaseExtend(X,A,m)` | Base change of scheme X into ambient A of the same type and dimension; equations transferred via coercion or map m. | — |
| `BaseChange(X, n)` / `BaseExtend(X, n)` | Base change of X (over a finite field) to the degree-n extension of its base field. | — |

*Worked example: H112E19 (intersection points over quadratic extension, base change splitting curve).*

---

## 112.12 Affine Patches and Projective Closure

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ProjectiveClosure(X)` | Projective closure of affine scheme X; cached and referentially stable (always returns the identical object). For X already a projective scheme (or computed from a patch) returns the stored closure. | Homogenisation. |
| `AffinePatch(X,i)` | The i-th affine patch of projective scheme X; cached. Order: the first patch has its final coordinate nonzero. | Dehomogenisation. |
| `AffinePatch(X,p)` | A standard affine patch of X containing point p; also returns the corresponding point on the patch. | Dehomogenisation. |
| `IsStandardAffinePatch(A)` | Whether affine space A is a standard affine patch of its projective closure and, if so, which index; false if no projective closure exists. | — |
| `NumberOfAffinePatches(X)` | Number of standard affine patches of X (0 for affine X). | — |
| `HasAffinePatch(X, i)` | Whether the i-th patch of X can be created. | — |
| `HyperplaneAtInfinity(X)` | The hyperplane complement of affine scheme X in its projective closure. | — |
| `ProjectiveClosureMap(A)` / `PCMap(A)` | The map from affine space A to its projective closure. | — |
| `AffineDecomposition(P)` | A sequence of maps from affine spaces to projective space P giving the standard disjoint decomposition P^n = A^n ∪ A^{n-1} ∪ … ∪ {p}; also returns the point p = (1:0:…:0). | — |
| `CentredAffinePatch(S, p)` | An affine patch of S centred at point p (via translation of a standard patch) and the embedding into S. | — |
| `MakeProjectiveClosureMap(A, P, S)` / `MakePCMap(A, P, S)` / `MakeProjectiveClosureMap(m)` / `MakePCMap(m)` | Set a given affine-to-projective map (defined by polynomials S, or a map m) as the projective closure map of A; used when no standard relationship between A and P exists. | — |

*Worked examples: H112E4, H112E20, H112E21, H112E39 (patch/closure round-trips, RestrictionToPatch).*

---

## 112.13 Arithmetic Properties of Schemes and Points

### 112.13.1 Height

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HeightOnAmbient(P)` | Exponential height of point P in its ambient affine or projective space (possibly weighted). Parameters: `Absolute` (default false; if true returns the absolute height), `Precision` (default 30). Works for P defined over Q, a number field, or a function field. | Weil height formula. |

### 112.13.2 Restriction of Scalars

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RestrictionOfScalars(S, F)` / `WeilRestriction(S, F)` | Weil restriction of affine scheme S (base ring K) to subfield F. Returns: the restriction scheme Sres over F, a map (Sres)⊗K → S, a function Sres(R) → S(R⊗F K), and a map S(K) → Sres(F). Parameters: `SubfieldMap` (inclusion F→K if not automatic), `ExtensionBasis` (basis of K/F). | Direct substitution using basis of K/F. |

### 112.13.3 Local Solubility

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsEmpty(Xm)` | For X a scheme over a number field K, Xm = X(L) where L is a p-adic completion, tests if X(L) is empty; returns an approximation to a local point if false. Parameters: `Smooth` (test only nonsingular points, plane curves only), `AssumeIrreducible`, `AssumeNonsingular`, `Verbose LocSol`. Special cases: (i) hyperelliptic curves over large odd residue field (Bruin generalisation of [MSS96]); (ii) hyperelliptic over small/even residue field (depth-first backtrack); (iii) nonsingular curves with possibly singular planar model (backtrack with Hensel criterion); (iv) cone-plus-quadric intersections in P3; (v) general schemes (primary decomposition then backtrack). **[Bru04]** | p-adic Hensel lifting / depth-first backtracking **[Bru04]**. |
| `IsLocallySolvable(X, p)` | For projective X over a number field or Q, tests local solvability at prime ideal p (or prime number p); returns true and an approximation if locally solvable. Same optional parameters as IsEmpty. **[Bru04]** | p-adic Hensel lifting **[Bru04]**. |
| `LiftPoint(P, n)` | Lift a p-adic point P in X(L) (L a completion) to precision n using quadratic Newton iteration; `Strict` (default true) causes an error if desired precision is unattainable. | Newton's method / quadratic lifting. |
| `LiftPoint(F, d, P, n)` | Same, but requires all input data over L: F is the defining equations of a scheme of dimension d, P is coordinates of an approximate point. Parameter `Strict` (default false). | Newton's method / quadratic lifting. |

*Worked examples: H112E22 (IsEmpty on P1-bundle), H112E23 (hyperelliptic, p=32003), H112E24 (IsLocallySolvable), H112E25 (LiftPoint).*

### 112.13.4 Searching for Points

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PointSearch(S,H)` | Search for points on scheme S over Q up to roughly height H; S in affine or non-weighted projective space. Parameters: `Dimension` (skip dimension computation), `Primes` (one or two small primes), `OnlyOne` (stop at first point). | p-adic lift (find local points, lift p-adically, recognise globally via LLL lattice reduction). |

*Worked example: H112E26 (surface in P3, height 100 search).*

---

## 112.14 Maps between Schemes

Maps in Magma are *rational* maps — morphisms from a dense open subset of X to Y. Two maps are equal if they agree on a common dense open subset. The base scheme `BaseScheme(f)` is the locus where the map is "naively" undefined.

### 112.14.1 Creation of Maps

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `map< X -> Y \| F >` / `map< X -> Y \| F, G >` / `map< X -> Y \| u, F >` | Create map X → Y from sequence F of polynomials or rational functions on X (inverse from G if given; base-ring map u if given). F and G may each be a sequence of sequences for alternative defining sets. Parameters: `Check` (default true), `CheckInverse` (default true). | — |
| `iso< X -> Y \| F, G >` | Create a map X → Y (from F) with birational inverse Y → X (from G); X and Y must have the same base ring. | — |
| `IdentityMap(X)` | The identity map X → X. | — |
| `ConstantMap(X,Y,p)` / `map< X -> Y \| Q >` | The constant map sending all points of X to point p of Y (coordinates Q). | — |
| `Projection(X,Y)` | Linear projection from projective space X to projective space Y omitting the first dim(X)−dim(Y) coordinates. | — |
| `Projection(X, Q)` / `Projection(X)` / `Projection(X, p)` | Projection of scheme X away from point p (default p = (1:0:…:0)) into projective space Q (if given). | — |
| `ProjectionFromNonsingularPoint(X,p)` | Projection of X from nonsingular rational point p; returns the image, the projection map, and the image of the blowup of p. | — |
| `ProjectiveMap(L, Y)` / `ProjectiveMap(L)` | Map X → Y from list L of function field elements used as projective coordinates; Y is a projective space of dimension #L−1 (created if omitted). | — |
| `ProjectiveMap(f, Y)` / `ProjectiveMap(f)` | Shorthand for `ProjectiveMap([f,1],X,Y)`. | — |
| `Elimination(X,V)` | Affine scheme obtained by eliminating variables at indices V from the equations of affine scheme X; result lives in the affine subspace where those variables are 0. | Elimination ideal. |
| `Inverse(f)` | The inverse of map f if inverse equations are stored; error otherwise. | — |
| `IsInvertible(f)` | Tests birationality of f; if so, returns a birational inverse computed via Groebner basis on affine graph ideals. | Groebner basis on graph. |
| `HasKnownInverse(f)` | True iff f has an inverse stored. | — |
| `g * f` | Composition g ∘ f (g acts first; order matches `p @ g` then `@ f` notation); stored in factored form unless expanded. | — |
| `Components(f)` | The constituent maps composed to form f. | — |
| `Restriction(f,X,Y)` | Restriction of map f to subscheme X of the domain; codomain is Y. Parameter `Check` (default true) controls subscheme relationship checks. | — |
| `Expand(phi)` | If phi is stored in factored form, return it in expanded form (may be expensive). | Polynomial substitution. |
| `Extend(phi)` | Return an expanded map with extra alternative equations reducing the base scheme to the maximal domain of definition; computationally heavy (Groebner basis on affine graph ideals). | Groebner basis on graph. |
| `Prune(phi)` | Remove alternative equations in expanded phi that do not reduce the base scheme. | — |
| `Normalization(phi)` / `Normalisation(phi)` | Remove common factors from the defining polynomials of phi. | GCD. |

*Worked examples: H112E27 (basic map creation), H112E28 (map from function field elements), H112E29 (Frobenius map with base-ring map), H112E30 (anticanonical embedding via ProjectiveMap), H112E31 (IsInvertible), H112E32 (Hom spaces, composition), H112E33 (Expand/Extend to proper morphism).*

### 112.14.2 Basic Attributes

#### 112.14.2.1 Trivial Attributes

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Domain(f)` | Domain of map f. | — |
| `Codomain(f)` | Codomain of map f. | — |
| `DefiningPolynomials(f)` / `DefiningEquations(f)` | Sequence of functions defining map f (expands composites). | — |
| `FactoredDefiningPolynomials(f)` | If f is an unexpanded composite: sequence of sequences of defining equations of each component; otherwise `DefiningPolynomials(f)`. | — |
| `InverseDefiningPolynomials(f)` | Sequence of functions defining the inverse of f (expands composites). | — |
| `FactoredInverseDefiningPolynomials(f)` | Factored form of inverse defining equations. | — |
| `AllDefiningPolynomials(f)` | Polynomials of all alternate definitions of f. | — |
| `AllInverseDefiningPolynomials(f)` | Polynomials of all alternate inverse definitions of f. | — |
| `AlgebraMap(f)` | The underlying polynomial-ring map; if F is the sequence of defining equations and x is the first variable of the codomain, then F[1] = x under AlgebraMap(f). | — |
| `FunctionDegree(f)` | Degree of the homogeneous polynomials defining projective map f; minimum over all alternate defining sets. | — |

#### 112.14.2.2 Basic Tests

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `f eq g` | True iff f and g have the same domain, codomain, and define the same rational map. | Rational function identity. |
| `IsRegular(f)` / `IsPolynomial(f)` | True iff f is defined at all points of its domain (no denominators). | — |
| `IsIsomorphism(f)` | True iff f has (or can easily compute) inverse defining equations; returns a recognised isomorphism map as second value. | — |
| `IsDominant(f)` | True iff the closure of the image of f is the whole codomain. | Image computation. |
| `IsLinear(f)` | True iff f is a regular map defined by linear polynomials. | — |
| `IsAffineLinear(f)` | True iff f is a map between affine spaces defined by polynomials of degree ≤ 1. | — |

### 112.14.3 Maps and Points

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `f(p)` | Image of point p under map f; p must not lie in the base scheme (except for curves with a function field, where the function field machinery is used). Handles sets and sequences of points. | Function field for curve base-scheme points (from V2.17). |
| `Pullback(f, p)` | Preimage of point p; if f is an isomorphism with inverse g, returns g(p); otherwise returns Pullback(f, Cluster(p)). | — |
| `p @@ f` | Same as `Pullback(f,p)` except for isogenies between elliptic curves, where a single rational preimage point is returned. | — |
| `f(K)` / `f(m)` | The map of point sets X(K) → Y(K) induced by f; or X(m) → Y(m(u)) where u is the base-ring map. | — |

*Worked example: H112E34 (mapping individual points and via point-set map for efficiency).*

### 112.14.4 Maps and Schemes

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Pullback(f, X)` | Scheme in the domain of f given by pulling back the equations defining subscheme X of the codomain; contains the base scheme of f. | Substitution. |
| `Image(f)` / `f(X)` | Scheme-theoretic closure of f(X∩U) in the codomain (U = complement of base scheme); stored with f for reuse. Multi-graded domain: X is saturated first. | Groebner basis (elimination). |
| `Image(f,X,d)` | Basis of degree-d polynomials in the codomain containing f(X); linear algebra over the field. | Linear algebra. |
| `BaseScheme(f)` | The subscheme of the domain where f is "naively" undefined (intersection of base schemes of all alternate equation sets). | — |
| `BasePoints(f)` / `BasePoints(f,L)` | Sequence of points in the base scheme of f defined over the base ring (or over extension field L); error if base scheme is not finite. | Point enumeration. |

*Worked examples: H112E35 (quartic curve in P3 via Image), H112E36 (Image(f,C,d) for trigonal canonical curve), H112E37 (base points), H112E38 (elementary transformation of scroll).*

### 112.14.5 Maps and Closure

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ProjectiveClosure(f)` | Map between projective closures of domain and codomain; if either is already projective it remains unchanged. | Homogenisation. |
| `MakeProjectiveClosureMap(A, P, S)` / `MakePCMap(A, P, S)` / `MakeProjectiveClosureMap(m)` / `MakePCMap(m)` | Set the map (given by polynomials S or by a scheme map m) as the projective closure map of affine space A. | — |
| `RestrictionToPatch(f,j)` | Restriction of affine-to-projective map f to a rational map from its domain to the j-th standard affine patch of its codomain. | Dehomogenisation. |
| `RestrictionToPatch(f,i,j)` | Restriction of projective-to-projective map f to a rational map from the i-th patch of its domain to the j-th patch of its codomain. | Dehomogenisation. |

### 112.14.6 Automorphisms

Automorphisms of schemes over fields may be constructed. Groups of automorphisms are computable only in special cases (linear automorphisms of projective spaces over finite fields; see Chapter 114 for curves).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Automorphism(X,F)` | Automorphism of scheme X determined by polynomial sequence F; uses Groebner basis to find the inverse. | Groebner basis. |
| `IdentityAutomorphism(X)` / `IdentityMap(X)` | The identity map X → X. | — |
| `IsEndomorphism(f)` | True iff domain and range of f are equal. | — |
| `IsAutomorphism(f)` | True iff f is an automorphism; if so, returns f as a `MapAutSch` with its inverse computed. | Groebner basis (if inverse not yet known). |

#### 112.14.6.1 Affine Automorphisms

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Automorphism(A,F)` | Automorphism of affine space A from polynomial sequence F (computes inverse via Groebner basis). | Groebner basis. |
| `Automorphism(A,M)` | Linear automorphism of affine space A from matrix M (acting from the right on points). | — |
| `Translation(A,p)` | Translation of affine space A taking rational point p to the origin. | — |
| `PermutationAutomorphism(A, g)` / `Automorphism(A, g)` | Automorphism of affine space A permuting coordinates according to permutation g. | — |
| `Automorphism(A,p)` | Automorphism adding polynomial p (not involving x) to the first coordinate x of affine space A. | — |
| `AffineDecomposition(f)` | For affine linear endomorphism f (degree ≤ 1), returns a linear endomorphism ℓ and a translation t such that f = ℓ ∗ t. | — |
| `NagataAutomorphism(A)` | The Nagata automorphism (u,v,w) ↦ (−u²w³−2uv²w²−2uvw+u−v⁴w−2v³, uw²+v²w+v, w) of affine 3-space A; not known to be tame. | — |
| `Projectivity(A,M)` | Restriction to affine space A of the linear automorphism of its projective closure determined by matrix M; not regular on A in general. | — |

*Worked examples: H112E40 (hyperelliptic involution), H112E41 (Jacobian conjecture test), H112E42 (permutation automorphism), H112E43 (AffineDecomposition), H112E44 (Projectivity).*

#### 112.14.6.2 Projective Automorphisms

Regular projective automorphisms are always linear; they correspond to matrices acting on coordinates. Over finite fields, the full automorphism group can be computed.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Automorphism(P,F)` | Automorphism of projective space P from polynomial sequence F. | — |
| `Matrix(f)` | Matrix corresponding to linear automorphism f of a projective space. | — |
| `Automorphism(P,M)` | Linear automorphism of projective space P from matrix M (acting on the left of coordinates). | — |
| `Aut(P)` | Parent object (set) of all automorphisms of projective space P. | — |
| `AutomorphismGroup(P)` | Automorphism group of P (P must be over a finite field) as a general linear group, together with a map matching group elements to automorphisms. Currently returns GL not PGL. | — |
| `TranslationOfSimplex(P,Q)` | Unique automorphism of n-dim P mapping the n+2 standard simplex points to the n+2 linearly independent rational points in Q. | Linear algebra. |
| `Translation(P,Q)` | Automorphism mapping the n+1 standard coordinate points to the n+1 linearly independent rational points of Q (does not fix (1:…:1)). | Linear algebra. |
| `Translation(P,p,q)` | A choice of linear automorphism of P taking rational point p to rational point q. | Linear algebra. |
| `Translation(X,p)` | Linear automorphism of projective space P (containing X) taking (0:…:0:1) to rational point p; for affine X, takes p to the origin. | Linear algebra. |
| `QuadraticTransformation(P)` / `QuadraticTransformation(P,Q)` | Standard quadratic transformation (x:y:…) ↦ (1/x:1/y:…) of projective space P; second form conjugates with the translation mapping points of Q to standard basis vectors. | — |
| `QuadraticTransformation(X)` / `QuadraticTransformation(X,Q)` | Birational pullback of projective scheme X by the quadratic transformation (of the ambient P, or conjugated by Q); removes exceptional components in the total pullback. | Pullback + ideal computation. |

*Worked examples: H112E45 (GL(3,GF(5)) action on P2), H112E46 (Translation in affine and projective context), H112E47 (factorisation of Cremona transformation into quadratic transformations).*

### 112.14.7 Scheme Graph Maps

A variant map type `MapSchGrph` (introduced V2.16) whose defining data is the closure of the graph of a rational map in the product ambient, without explicit defining polynomials. Graph maps are automatically maximally defined; useful for divisor maps where deriving explicit polynomial equations is expensive. Currently only between ordinary projective schemes.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SchemeGraphMap(X, Y, I)` | Create a graph map from X to Y where I is a bihomogeneous ideal in a grevlex-ordered polynomial ring of n+m+2 variables defining the closure of the graph (first m+1 vars for domain, last n+1 for codomain). Parameter `Saturated` (default false); if false, I is domain-saturated internally via ColonIdeal. | ColonIdeal (domain saturation). |
| `SchemeGraphMapToSchemeMap(f)` | Convert graph map f to a usual scheme map; if f is known invertible, inverse defining polynomials are added. May produce very high-degree polynomials and a large base scheme. | Groebner basis on graph ideal. |
| `IsInvertible(f)` | Returns whether graph map f is birationally invertible; if so, returns the inverse (as a reversed graph). Also performs codomain-saturation if needed. | Groebner basis. |

*Worked example: H112E48 (elliptic curve in P3, graph map construction, invertibility, image, preimage).*

---

## 112.15 Tangent and Secant Varieties and Isomorphic Projections

### 112.15.1 Tangent Varieties

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `TangentVariety(X)` | Subscheme of the ambient whose closed points form the closure of the union of all tangent spaces to closed points of X. Parameter `PatchIndex` (default 0; if i>0, uses the i-th affine patch and takes the projective closure, which is faster). | Projection from a 2n-dimensional ambient (tangent map image). |
| `IsInTangentVariety(X,P)` | Fast test whether a given ambient point P lies in the tangent variety of projective X; much faster than computing the full tangent variety. | Point membership test. |

### 112.15.2 Secant Varieties

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SecantVariety(X)` | Subscheme of the ambient whose closed points form the closure of the union of all secant lines through distinct pairs of closed points of X. Parameter `PatchIndex` (default 0; if i>0 uses the i-th affine patch). Note: the union of secants is not necessarily closed even when X is projective. | Projection from a 2n+1 dimensional ambient. |
| `IsInSecantVariety(X,P)` | Fast test whether ambient point P lies in the union of secants of projective X (not its closure); no affine-patch restriction. | Point membership test. |

*Worked examples: H112E49 (TangentVariety of a curve in P3), H112E50 (SecantVariety of three-component curve in P4).*

### 112.15.3 Isomorphic Projection to Subspaces

For X in P^n of dimension d (assuming n > 2d+1), there exist points in P^n outside both the tangent and secant varieties; projection from such a point is an isomorphism to a hyperplane, reducing n by 1. Iterating reaches n = 2d+1. Random points are used to avoid computing the full tangent/secant variety.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsomorphicProjectionToSubspace(X)` | Isomorphically project X (assumed reduced) to a linear subspace of dimension 2d+1 (or less if possible). Parameter `Verbose IsoToSub`. Returns the image scheme and the explicit projection map. | Random point selection + IsInSecantVariety/IsInTangentVariety, iterated projection. |
| `EmbedPlaneCurveInP3(C)` | Embed a plane curve C as a nonsingular projective curve in P2 or P3 over the base field, using the function field machinery followed by IsomorphicProjectionToSubspace. Parameter `Verbose EmbCrv`. Returns image and map C → image. | Function field + isomorphic projection. |

*Worked example: H112E51 (genus-5 plane curve embedded in P3).*

---

## 112.16 Linear Systems

A linear system on projective space P is a vector space of homogeneous polynomials of a fixed degree (or multi-degree). Its sections form a basis; the coefficient space is the corresponding abstract vector space with maps to/from polynomials.

### 112.16.1 Creation of Linear Systems

#### 112.16.1.1 Explicit Creation

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LinearSystem(P,d)` | Complete linear system on affine or projective space P of degree d; sections = all monomials of that degree (all polynomials of degree ≤ d for affine P); d must be positive. | — |
| `LinearSystem(P, d)` | Complete linear system of multi-degree d (sequence of degrees, one per grading); positive entries required. | — |
| `LinearSystem(P,F)` | Linear system generated by polynomials (homogeneous of common degree for projective P, or arbitrary degree for affine P) in sequence F; linearly dependent inputs trigger recomputation of a basis. | Linear algebra. |
| `MonomialsOfWeightedDegree(X, D)` | Monomials in the coordinate ring of the ambient of X having degree D[i] with respect to the i-th grading. | — |
| `ImageSystem(f,S,d)` | Linear system on the codomain of map f consisting of degree-d hypersurfaces containing f(S); S must be in the domain of f. | Linear algebra (evaluating map on degree-d monomials). |

*Worked examples: H112E52 (two equal linear systems recognised via coefficient spaces), H112E53 (ImageSystem for canonical embedding of genus-4 curve).*

#### 112.16.1.2 Geometrical Restrictions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LinearSystem(L,p)` / `LinearSystem(L,S)` | Subsystem of L comprising hypersurfaces passing through point p or all points of sequence S. Nonrational points impose conditions as the union of their Galois conjugates. | Linear algebra (evaluate sections at p). |
| `LinearSystem(L,p,m)` | Subsystem of L comprising hypersurfaces passing through p with multiplicity ≥ m. | Linear algebra. |
| `LinearSystem(L,X)` | Subsystem of L comprising elements containing scheme X; sections = polynomials in the defining ideal of X of the same degree as L. | Ideal membership. |
| `LinearSystemTrace(L,X)` | Trace of L on scheme X: sections of L modulo the equations of X; result is still a linear system on the common ambient. | Polynomial reduction. |

*Worked examples: H112E54 (multiplicities at prescribed points, rational curve parametrisation), H112E55 (trace of cubics on a twisted cubic).*

#### 112.16.1.3 Explicit Restrictions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LinearSystem(L,F)` | Subsystem of L generated by polynomials in sequence F (must already be sections of L); linearly dependent inputs trigger recomputation of a basis. | Linear algebra. |
| `LinearSystem(L,V)` | Subsystem of L determined by subspace V of the complete coefficient space of L. | — |

### 112.16.2 Basic Algebra of Linear Systems

#### 112.16.2.1 Tests for Linear Systems

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Ambient(L)` / `AmbientSpace(L)` | The projective space on which L is defined. | — |
| `L eq K` | True iff L and K are equal as linear subsystems of some common complete linear system; error if they lie in different complete systems. | Subspace equality (linear algebra). |
| `IsComplete(L)` | True iff L is the complete linear system of polynomials of some degree. | — |
| `IsBasePointFree(L)` / `IsFree(L)` | True iff L has no base points. | — |

#### 112.16.2.2 Geometrical Properties

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Sections(L)` | Basis of sections of L (maximal linearly independent set of polynomials). | — |
| `Random(LS)` | A random section of LS (small rational coefficients for Q; zero if no random generator). | — |
| `Degree(L)` | Degree (or multi-degree) of the sections of L. | — |
| `Dimension(L)` | Projective dimension of L (= number of independent sections − 1). | — |
| `BaseScheme(L)` | Scheme defined by the sections of L; not tested for emptiness. | — |
| `BaseComponent(L)` | The hypersurface common to all elements of L (codimension-1 base locus). | GCD. |
| `Reduction(L)` | L with its codimension-1 base locus removed (common factors of sections removed). | GCD. |
| `BasePoints(L)` | Sequence of base points of L if the base locus is finite. | Point enumeration. |
| `Multiplicity(L,p)` | Generic multiplicity of hypersurfaces of L at point p. | — |

*Worked example: H112E56 (degree-6 curves, BaseComponent, Reduction, MinimalPrimeComponents of base locus).*

#### 112.16.2.3 Linear Algebra

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CoefficientSpace(L)` | The vector space whose vectors are coefficient vectors of sections of L. | — |
| `CoefficientMap(L)` | Map from the polynomial ring to the coefficient space: evaluates a polynomial f to its coefficient vector w.r.t. the basis of L. | — |
| `PolynomialMap(L)` | Map from the coefficient space of L to the polynomial ring: maps a coefficient vector to the corresponding section of L. | — |
| `Complement(L,K)` | Maximal subsystem of L not containing any hypersurface of L that lies in K. | Linear algebra. |
| `Complement(L,X)` | Maximal subsystem of L comprising hypersurfaces not containing scheme X. | Linear algebra. |
| `L meet K` / `Intersection(L,K)` | Linear system whose coefficient space is the intersection of coefficient spaces of L and K; error if they lie in different complete systems. | Linear algebra. |
| `X in L` | True iff scheme X occurs among the hypersurfaces of L. | — |
| `f in L` | True iff polynomial f is a section of L. | Linear algebra. |
| `K subset L` / `IsSubsystem(L,K)` | True iff the coefficient space of K is contained in that of L; error if not in a common system. | Linear algebra. |

*Worked example: H112E57 (coefficient spaces, PolynomialMap, LinearSystem from subspace).*

### 112.16.3 Linear Systems and Maps

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Pullback(f,L)` | The pullback system f*L on the domain of map f, where L is a linear system on the codomain; produces the system of homaloids via substitution of the map equations. | Polynomial substitution. |

---

## 112.17 Divisors

Divisors on projective varieties of dimension > 1. Integral effective divisors are stored as ideals; general divisors in partially factored form as lists of (ideal, rational multiplicity) pairs. Q-divisors are supported. Many intrinsics require X to be ordinary projective and the divisor to be Cartier.

### 112.17.1 Divisor Groups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DivisorGroup(X)` | The divisor group of variety X (type `DivSch`). | — |
| `Variety(G)` | The variety of divisor group G. | — |
| `G1 eq G2` | True iff G1 and G2 are divisor groups of the same variety. | — |

### 112.17.2 Creation of Divisors

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Divisor(X,f)` (function field element) | The divisor of a global function f: zeroes minus poles. | Factorisation of ideal. |
| `Divisor(X,f)` (field of fractions element) | Same for f in the field of fractions of the coordinate ring of the ambient. | — |
| `Divisor(X,f)` (homogeneous polynomial) | The effective divisor on X defined by f; subscheme of X with ideal generated by ideal(X) and f. | — |
| `Divisor(X,Q)` | Effective divisor defined by the ideal generated by ideal(X) and the sequence Q. Parameters: `CheckSaturated`, `CheckDimension`, `UseCodimensionOnePart`. | Ideal saturation. |
| `Divisor(X,Y)` | Effective divisor defined by subscheme Y of X. Parameters as above. | Ideal saturation. |
| `Divisor(X,I)` | Effective divisor defined by ideal I in the coordinate ring of the ambient of X (saturation should contain ideal(X)). Parameters as above. | Ideal saturation. |
| `HyperplaneSectionDivisor(X)` | Divisor given by a hyperplane section of projective variety X. | — |
| `ZeroDivisor(X)` | The zero divisor on X. | — |
| `CanonicalDivisor(X)` | A canonical divisor on X (must be ordinary projective Gorenstein); uses the canonical sheaf and `SheafToDivisor`. | Canonical sheaf computation. |
| `SheafToDivisor(S)` | For invertible (locally free, rank 1) coherent sheaf S on variety X, return an effective Cartier divisor D such that S ≅ L(D). | Sheaf computation. |
| `RoundDownDivisor(D)` | For Q-rational D in prime factorisation, return the integral divisor obtained by rounding down all rational coefficients. | — |
| `RoundUpDivisor(D)` | As above, rounding up. | — |
| `FractionalPart(D)` | D − RoundDownDivisor(D). | — |
| `IntegralMultiple(D)` | Find positive integer N such that N*D is integral; returns N*D and N. | — |

### 112.17.3 Ideals and Factorisations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Ideal(D)` | Defining ideal of an effective integral divisor D. | — |
| `Support(D)` | Subscheme of the variety of effective Q-divisor D giving its support. | — |
| `IdealOfSupport(D)` | Ideal in the coordinate ring of the ambient defining the support of effective Q-divisor D. | — |
| `SignDecomposition(D)` | Decompose D = A − B with A, B effective; returns A and B (not guaranteed relatively prime). | — |
| `IdealFactorisation(D)` | Current stored factorisation of D as a sequence of (ideal, rational multiplicity) pairs. | — |
| `CombineIdealFactorisation(~D)` | Simplify the current factorisation of D by combining terms with the same ideal. | — |
| `ComputeReducedFactorisation(~D)` / `ReducedFactorisation(D)` | Replace the factorisation of D with an equivalent one where all ideals are primary. | Primary decomposition. |
| `ComputePrimeFactorisation(~D)` / `PrimeFactorisation(D)` | Replace the factorisation with an equivalent prime factorisation. | Prime decomposition. |
| `Multiplicity(D,E)` | Multiplicity of prime divisor E in divisor D. | — |

### 112.17.4 Basic Divisor Predicates

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsZeroDivisor(D)` | True iff D is the zero divisor. | — |
| `IsIntegral(D)` | True iff D is an integral divisor; may convert to prime factorisation. | Prime factorisation. |
| `IsEffective(D)` | True iff D is effective; may convert to prime factorisation. | Prime factorisation. |
| `IsPrime(D)` | True iff D is a prime divisor. | — |
| `IsFactorisationPrime(D)` | True iff the current factorisation of D has all ideals prime. | — |
| `IsDivisible(D)` | True iff D is integral and divisible by some n > 1; if so, returns the maximum such n. | — |

### 112.17.5 Arithmetic of Divisors

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `D1 + D2`, `D1 - D2`, `-D` | Addition, subtraction, and negation of divisors; one argument may be a toric divisor for the same variety. | — |
| `n * D`, `r * D` | Multiply divisor D by integer n or rational r. | — |
| `D1 eq D2` | True iff D1 and D2 lie on the same variety and are equal. | Prime factorisation comparison. |

### 112.17.6 Further Divisor Properties

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsCanonical(D)` | True iff D is a canonical divisor; tests whether its sheaf is isomorphic to the canonical sheaf. Variety must be ordinary projective Gorenstein. | Sheaf comparison. |
| `IsAnticanonical(D)` | True iff D is anticanonical; tests whether its sheaf is isomorphic to the dual canonical sheaf. | Sheaf comparison. |
| `IsCanonicalWithTwist(D)` | True iff D = hyperplane-section divisor of degree d + canonical divisor; also returns d. | Sheaf comparison. |
| `IsPrincipal(D)` | True iff D is a principal divisor; if so, returns f in FunctionField(ambient of X) with D = div(f). Uses Riemann-Roch space of D. | Riemann-Roch. |
| `IsLinearlyEquivalent(D,E)` | True iff D and E are linearly equivalent; if so, returns f with D = E + div(f). | Riemann-Roch (via IsPrincipal). |
| `BaseLocus(D)` / `IsBasePointFree(D)` / `IsMobile(D)` | Base locus of |[D]| (round-down); whether empty; whether of codimension ≥ 2. Requires Cartier [D] and ordinary projective X. | Riemann-Roch. |
| `IntersectionNumber(D1,D2)` | Intersection pairing D1.D2 on a surface (dim X = 2); one divisor assumed Cartier. | Sheaf cohomology. |
| `SelfIntersection(D)` | Self-intersection number D.D on a surface. | Sheaf cohomology. |
| `Degree(D)` / `Degree(D,H)` | Intersection number of D with a hyperplane divisor (first form) or with H (second form, equivalent to IntersectionNumber(D,H)); dim X = 2. | Intersection number. |
| `IsNef(D)` | True iff D (a Q-Cartier effective divisor on projective surface X) has non-negative intersection with all effective divisors on X. | Intersection computation. |
| `IsNefAndBig(D)` | True iff D is nef and has positive self-intersection. | Intersection computation. |
| `NegativePrimeDivisors(D)` | Sequence of prime divisor components of D with negative intersection with D. | Intersection computation. |
| `ZariskiDecomposition(D)` | Returns Q-divisors P (nef) and N (negative-definite support) such that D = P + N; Zariski decomposition on a surface. | Intersection matrix computation. |

### 112.17.7 Riemann-Roch Spaces

All intrinsics here require X to be ordinary projective and [D] (round-down of D) to be Cartier.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Sheaf(D)` | The invertible sheaf corresponding to the divisor class of [D]; uses the coherent sheaf package. | Coherent sheaf. |
| `RiemannRochBasis(D)` | Basis of the Riemann-Roch space of [D] as a sequence of function field elements. | Coherent sheaf (or variant for non-effective D). |
| `RiemannRochSpace(D)` | Riemann-Roch space as an abstract vector space V over the base field, with a map V → FunctionField. | Coherent sheaf. |
| `RiemannRochCoordinates(f,D)` | True iff f lies in the Riemann-Roch space of D after coercion to the function field; if so, returns coordinates w.r.t. the RiemannRochBasis. | Riemann-Roch. |
| `IsLinearSystemNonEmpty(D)` | True iff there exists an effective divisor linearly equivalent to D; if so, returns such a divisor. | Riemann-Roch. |

---

## 112.18 Isolated Points on Schemes

Experimental code to find isolated Q-rational points on affine schemes defined by at least n equations in n variables (positive-dimensional components may be present but are ignored). Algorithm: (1) find local points mod a prime p where the Jacobian has maximal rank; (2) lift via Newton's method to high p-adic precision; (3) recognise over a number field via LLL. Optional preconditioning: linear elimination and resultant elimination of variables.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LinearElimination(S)` | Iteratively eliminate variables appearing strictly linearly in some equation of scheme S; parameter `EliminationOrder`. Returns a map from the reduced scheme to S with inverse. | Linear substitution. |
| `IsolatedPointsFinder(S,P)` | Given affine n-dimensional scheme over Q (defined by ≥ n equations) and a sequence of primes P, find liftable points modulo the primes. Parameters: `LinearElimination` (order for linear elimination), `ResultantElimination`, `FactorizationInResultant` (default true; apply squarefree factorisation to resultants). | p-adic search with optional resultant/linear preprocessing. |
| `IsolatedPointsLifter(S,P)` | Given a scheme S and a finite-field point P (from IsolatedPointsFinder) with maximal-rank Jacobian, lift via Newton's method and recognise over a number field via LLL. Returns (true, point) or false. Parameters: `LiftingBound` (default 10; number of Newton steps → precision p^(2^LiftingBound)), `DegreeBound` (default 32; max degree of number field to check), `DegreeList` (explicit degrees to try), `OptimizeFieldRep`. | p-adic Newton lifting + LLL lattice recognition. |
| `IsolatedPointsLiftToMinimalPolynomials(S,P)` | As IsolatedPointsLifter but returns a minimal polynomial for each coordinate separately rather than finding a common number field. Same parameters (no `OptimizeFieldRep`). | p-adic Newton lifting + LLL per coordinate. |

*Worked examples: H112E58 (Elkies construction of large integral points on elliptic curves, degrees (4,5,0,1)), H112E59 (Hall's conjecture / Belyi maps, genus-6 number field), H112E60 (generic random scheme, IsolatedPointsLiftToMinimalPolynomials vs Groebner basis), H112E61 (M23 monodromy, degree-22 system, quartic number field).*

---

## 112.19 Advanced Examples

### 112.19.1 A Pair of Twisted Cubics

Constructs the intersection Z of two twisted cubics C1, C2 in P3 (defined by 2×3 matrix minors). Demonstrates: cluster of degree 5, reducedness, splitting field points, the automorphism x↔t, and realisation of permutation elements as automorphisms.

*Worked example: H112E62.*

### 112.19.2 Curves in Space

Constructs an elliptic curve in P3 as a complete intersection of two quadrics, projects from a nonsingular rational point to a plane cubic, and uses `EllipticCurve` to find the Weierstrass model.

*Worked example: H112E63.*

---

## 112.20 Bibliography

| Key | Reference |
|-----|-----------|
| **[BC04]** | W. Bosma and J. Cannon, editors. *Discovering Mathematics with Magma.* Springer-Verlag, Heidelberg, 2004. |
| **[Bru04]** | Nils Bruin. *Some ternary Diophantine equations of signature (n,n,2).* In Bosma and Cannon [BC04]. |
| **[Har77]** | Robin Hartshorne. *Algebraic Geometry,* GTM 52. Springer, 1977. |
| **[MSS96]** | J. R. Merriman, S. Siksek, and N. P. Smart. *Explicit 4-descents on an elliptic curve.* Acta Arith. **77**(4):385–404, 1996. |
| **[Rei88]** | Miles Reid. *Undergraduate Algebraic Geometry.* CUP, Cambridge, 1988. |
| **[Rei97]** | Miles Reid. *Chapters on Algebraic Surfaces.* In J. Kollár, editor, *Complex algebraic varieties,* IAS/Park City Mathematics Series 3, pp. 1–154. AMS, Providence R.I., 1997. |
| **[vdE00]** | Arno van den Essen. *Polynomial automorphisms and the Jacobian conjecture,* vol. 190 of Progress in Mathematics. Birkhäuser, 2000. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Groebner basis (ideal computation, saturation, dimension) | `Union`, `ReducedSubscheme`, `Dimension`, `IsEmpty`, `IsIrreducible`, `IsReduced`, `GroebnerBasis`, `MinimalBasis`, `IsSubscheme`, `X eq Y`, `Image(f)`, `IsInvertible(f)`, `Extend` |
| Primary / prime decomposition | `PrimeComponents`, `PrimaryComponents`, `IsIrreducible`, `IsReduced`, `ComputePrimeFactorisation`, `ComputeReducedFactorisation` |
| Local Groebner bases (multiplicity, tangent cone, singularity analysis) | `Multiplicity`, `TangentCone`, `IsHypersurfaceSingularity`, `HypersurfaceSingularityExpandFurther`, `HypersurfaceSingularityExpandFunction` |
| Jacobian criterion (singularity, nonsingularity) | `IsSingular`, `IsNonsingular`, `SingularSubscheme`, `IsNonsingular(X)`, `TangentSpace` |
| Noether normalisation fibration (point enumeration) | `RationalPointsByFibration`, `Random` |
| p-adic Hensel lifting + backtracking **[Bru04]** (local solubility) | `IsEmpty(Xm)`, `IsLocallySolvable`, `LiftPoint` |
| Weil height | `HeightOnAmbient` |
| Weil restriction / direct substitution | `RestrictionOfScalars`, `WeilRestriction` |
| p-adic lift + LLL (point search) | `PointSearch` |
| p-adic Newton lifting + LLL (isolated points) | `IsolatedPointsFinder`, `IsolatedPointsLifter`, `IsolatedPointsLiftToMinimalPolynomials`, `LinearElimination` |
| Segre embedding **[Har77]** | `SegreProduct`, `SegreEmbedding` |
| Rees ideal / blowup **[Har77]** | `BlowUp` |
| Projection from tangent/secant (isomorphic projection) | `IsomorphicProjectionToSubspace`, `EmbedPlaneCurveInP3` |
| Coherent sheaf / Riemann-Roch | `CanonicalDivisor`, `SheafToDivisor`, `Sheaf`, `RiemannRochBasis`, `RiemannRochSpace`, `RiemannRochCoordinates`, `IsLinearSystemNonEmpty`, `IsPrincipal`, `IsLinearlyEquivalent`, `BaseLocus`, `IsCanonical`, `IsAnticanonical`, `IsCanonicalWithTwist` |
| Intersection theory on surfaces | `IntersectionNumber`, `SelfIntersection`, `Degree(D)`, `IsNef`, `IsNefAndBig`, `NegativePrimeDivisors`, `ZariskiDecomposition` |
| Minimal free resolution (Cohen-Macaulay/Gorenstein) **[Har77]** | `IsCohenMacaulay`, `IsGorenstein`, `IsArithmeticallyCohenMacaulay`, `IsArithmeticallyGorenstein` |
| Linear algebra (linear systems, coefficient spaces) | `LinearSystem`, `Sections`, `CoefficientSpace`, `CoefficientMap`, `PolynomialMap`, `Complement`, `Intersection(L,K)`, `IsComplete`, `IsBasePointFree(L)`, `Reduction`, `Pullback(f,L)`, `ImageSystem` |
| Graph map / ColonIdeal (graph maps) | `SchemeGraphMap`, `SchemeGraphMapToSchemeMap`, `IsInvertible(f)` (graph variant) |
| Quadratic transformation / Cremona group **[vdE00]** | `QuadraticTransformation`, and composition with `Translation`, `Automorphism` |
| Automorphism group of projective space over finite field | `AutomorphismGroup(P)`, `Aut(P)`, `Matrix(f)` |
