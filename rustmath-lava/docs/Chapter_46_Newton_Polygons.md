# Chapter 46 — Newton Polygons

**Handbook part:** VII — Local Arithmetic Fields
**Handbook pages:** 1235–1260 (PDF pages 1366–1393)

---

## Scope and overview

Chapter 46 introduces Newton polygons as geometric objects derived from polynomials and
from sets of rational points. A Newton polygon is interpreted here as the convex hull of
finitely many points of the rational plane (rather than as an intersection of half-spaces),
with possible inclusion of points at +∞ along the axes. Points of the plane are written as
tuples ⟨a, b⟩; faces are written ⟨a, b, c⟩ and represent the line ax + by = c.

The **standard Newton polygon** of a bivariate polynomial f(u, v) is the convex hull of the
Newton points ⟨a, b⟩ for each monomial u^a v^b with nonzero coefficient, together with the
points at +∞ on the two axes. For univariate polynomials over a local ring or a Puiseux
(fractional power series) field, the defining points are ⟨i, v(aᵢ)⟩ where aᵢ is the coefficient
of the i-th power and v is the valuation. The different "flavours" of polygon (inner, lower,
outer, all faces/vertices) reflect the different conventions used in applications.

The primary intended applications are:
- **Newton–Puiseux analysis** of singular points of plane curves and factorisation of
  polynomials over Puiseux fields.
- **Valuations of roots**: reading the slopes of the Newton polygon to determine how many
  roots a polynomial has with a given valuation.
- **Root-finding over series rings**: two algorithms are provided — Walker's algorithm
  **[Wal78]** (general, but may not terminate when char k ≤ deg f) and the faster algorithm
  of Duval **[Duv89]** (requires essentially Laurent series input and char k = 0 or
  char k > degree of each squarefree factor).

All examples in the chapter are stated to run consecutively in a single Magma session.

---

## 46.1 Introduction

Newton polygons are often used as an intersection of finitely many rational half-spaces in
the rational plane — a definition that emphasises non-compactness. In Magma they are
implemented instead as the convex hull of finitely many rational points (possibly including
some points at +∞ along the axes). The x and y coordinates of the plane are the first and
second coordinate functions. Points are written ⟨a, b⟩; faces ⟨a, b, c⟩ represent ax + by = c
and are one-dimensional boundary intersections of N with that line.

The standard Newton polygon of a bivariate polynomial keeps only those faces and vertices
that "face the origin" (lower-left part of the hull). The functions `AllFaces`, `AllVertices`,
etc. expose the full compact convex hull when required.

*Worked examples: H46E1 (Newton polygons from a Puiseux series polynomial and a local
ring polynomial; Newton polygons from explicit point sets N2 and N6 used throughout).*

---

## 46.2 Newton Polygons

All polygons are determined by a finite collection of points Pₙ in the rational plane.
The distinguishing data is the collection of lines and points considered to be faces and
vertices, which depends on the creation function used.

### 46.2.1 Creation of Newton Polygons

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `NewtonPolygon(f)` | Standard Newton polygon of a bivariate polynomial `f`. Hull of Newton points together with +∞ on each axis. Horizontal and vertical end faces are excluded. Parameter `Faces` (`"Inner"`, `"Lower"`, or `"All"`; default `"Inner"`) controls which faces are returned by `Faces`. | Convex hull construction. |
| `NewtonPolygon(f)` *(univariate over series or local ring)* | Newton polygon of a univariate polynomial `f` over a series ring or a local ring/field. `SwapAxes` (`BoolElt`, default `false`): if `true` (series ring only) plots series exponents on horizontal axis and polynomial exponents on vertical. For series rings the hull includes +∞ on each axis; for local rings it does not. Parameter `Faces` (`"All"`, `"Inner"`, or `"Lower"`): default `"Inner"` for series rings, `"Lower"` for local rings. | Convex hull from valuation points ⟨i, v(aᵢ)⟩. |
| `NewtonPolygon(f, p)` *(prime integer or prime ideal)* | Newton polygon of `f` where `p` is a prime used for p-adic valuations of the coefficients. `f` may be over Z, Q, a number field, an algebraic function field, or an order thereof; `p` may be an integer or a prime ideal. Points are ⟨i, vₚ(aᵢ)⟩; +∞ on each axis included. Parameter `Faces` (`"Inner"`, `"Lower"`, or `"All"`; default `"Inner"`). | p-adic valuation + convex hull. |
| `NewtonPolygon(f, p)` *(place of function field)* | Newton polygon of `f` where `p` is a place of an algebraic function field, used to determine coefficient valuations. Points at +∞ included. Parameter `Faces` (`"Inner"`, `"Lower"`, or `"All"`; default `"Inner"`). | Place valuation + convex hull. |
| `NewtonPolygon(C)` | Standard Newton polygon of the defining polynomial of the plane curve `C`. | Convex hull of Newton points of the defining polynomial. |
| `NewtonPolygon(V)` | Newton polygon that is the compact convex hull of the set or sequence `V` of rational points ⟨a, b⟩. Parameter `Faces` (`"All"`, `"Lower"`, or `"Inner"`; default `"All"`). | Compact convex hull of given points. |
| `DefiningPoints(N)` | The points of the rational plane used in the initial creation of `N`. Allows comparison of defining points between two polygons (no explicit equality test is provided). | — |

*Worked examples: H46E1 (NewtonPolygon from Puiseux series polynomial; from local ring polynomial; from explicit point sets).*

### 46.2.2 Vertices and Faces of Polygons

Faces are returned as tuples ⟨a, b, c⟩ (the line ax + by = c), listed anticlockwise starting
from the lowest of the leftmost points. The different intrinsics expose different subsets of
the compact convex hull depending on the convention required.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Faces(N)` | Sequence of faces of `N`, listed anticlockwise. Which faces are returned depends on the creation function (Inner, Lower, or All). | — |
| `InnerFaces(N)` | Faces of the compact convex hull of Pₙ starting at the lowest leftmost point that have strictly negative gradient. | — |
| `LowerFaces(N)` | Faces of the compact convex hull of Pₙ that bound it below in the y direction. | — |
| `OuterFaces(N)` | Union of lower faces that are not inner faces and faces bounding the compact hull above in the y direction (ignoring infinite points). | — |
| `AllFaces(N)` | All faces of the compact convex hull of Pₙ. | — |
| `Vertices(N)` | Sequence of vertices of `N`, listed anticlockwise from the lowest of the leftmost points. | — |
| `InnerVertices(N)` | Sequence of vertices that are endpoints of inner faces. | — |
| `LowerVertices(N)` | Sequence of vertices that are endpoints of lower faces. | — |
| `OuterVertices(N)` | Sequence of vertices that are endpoints of outer faces. | — |
| `AllVertices(N)` | Sequence of vertices of the compact convex hull of Pₙ. | — |
| `EndVertices(F)` | A sequence containing the two end vertices of the face `F = ⟨a, b, c⟩`. | — |
| `FacesContaining(N, p)` | Those faces of `N` (as returned by `Faces`) on which the point `p = ⟨a, b⟩` lies. | — |
| `GradientVector(F)` | The ⟨a, b⟩ values of the face `F`, where `F` is described by ax + by = c with a, b, c integers. | — |
| `GradientVectors(N)` | A sequence of gradient vectors of all faces of `N`. | — |
| `Weight(F)` | The c value of the line ax + by = c describing face `F`. | — |
| `Slopes(N)` | Slopes of the faces of `N` (as returned by `Faces`). | — |
| `InnerSlopes(N)` | Slopes of the inner faces of `N`. | — |
| `LowerSlopes(N)` | Slopes of the lower faces of `N`. | — |
| `AllSlopes(N)` | Slopes of all faces of the compact convex hull. | — |

*Worked examples: H46E2 (Faces, InnerFaces, OuterFaces, AllFaces, LowerFaces for various polygon types); H46E3 (InnerVertices, Vertices, AllVertices, OuterVertices, showing overlap); H46E4 (EndVertices and FacesContaining); H46E5 (GradientVector and Weight as component access, manual slope computation).*

### 46.2.3 Tests for Points and Faces

Whether a point lies in a polygon is tested against the faces returned by `Faces(N)`, which
are fixed at first computation. Tests against other collections (e.g. AllFaces) must be
performed explicitly.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsFace(N, F)` | Returns `true` iff the tuple `F = ⟨a, b, c⟩` describes a face coinciding with a face of `N` as returned by `Faces`. Also returns the normalised form of the face. | Comparison against `Faces(N)`. |
| `IsVertex(N, p)` | Returns `true` iff `p = ⟨a, b⟩` is a vertex of `N` as returned by `Vertices`. | Comparison against `Vertices(N)`. |
| `IsInterior(N, p)` | Returns `true` iff `p = ⟨a, b⟩` lies strictly in the interior of `N`. | Half-plane containment. |
| `IsBoundary(N, p)` | Returns `true` iff `p = ⟨a, b⟩` lies on the boundary (contained in some face) of `N`. | Half-plane/face containment. |
| `IsPoint(N, p)` | Returns `true` iff `p = ⟨a, b⟩` lies on `N` (interior or boundary). | Combination of above. |

---

## 46.3 Polynomials Associated with Newton Polygons

The polynomial used to define a polygon can be recovered, and the restrictions of that
polynomial to particular faces — the *face functions* — are key characteristic data. Most
functions in this section return an error if `N` was not defined from a polynomial.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HasPolynomial(N)` | Returns `true` iff `N` was defined as the Newton polygon of some polynomial. | — |
| `Polynomial(N)` | The polynomial used to define `N`. | — |
| `ParentRing(N)` | The parent ring of the polynomial of `N`. | — |
| `IsNewtonPolygonOf(N, f)` | Returns whether `N` is defined by the polynomial `f`. | — |
| `FaceFunction(F)` | For a bivariate polynomial: those monomial terms of `f` whose Newton point lies on face `F` (coefficients are preserved). For a univariate polynomial over a series ring: the univariate polynomial supported on `F`. | — |
| `IsDegenerate(F)` | Returns `true` if the face function along `F` is not squarefree. | Squarefreeness test of `FaceFunction(F)`. |
| `IsDegenerate(N)` | Returns `true` if any face function on any face of `N` is degenerate. | Tests all faces via `IsDegenerate(F)`. |

---

## 46.4 Finding Valuations of Roots of Polynomials from Newton Polygons

The slopes of the Newton polygon determine the valuations of the roots of the associated
polynomial (and the number of roots at each valuation), without computing the roots
themselves.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ValuationsOfRoots(f)` | Valuations of the roots of `f`, paired with the number of roots having that valuation. `f` must be a polynomial over a local ring or a series ring. | Newton polygon slope reading. |
| `ValuationsOfRoots(f, p)` | Valuations of the roots of `f` with respect to `p`, paired with multiplicity. `p` may be a prime integer, a prime ideal of a number field, or a place of a function field. | Newton polygon at prime `p`. |

---

## 46.5 Using Newton Polygons to Find Roots of Polynomials over Series Rings

Two main algorithms are implemented for finding Puiseux expansions of roots of polynomials
over series rings:

1. **Walker's algorithm** **[Wal78]** — general purpose; implemented in `PuiseuxExpansion`.
   May not terminate when the characteristic of the coefficient field is ≤ degree of f (see
   **[Gri95]**, pp. 269–272 for the precise statement). Care is needed with low-precision
   coefficients since extracting the squarefree part loses further precision.

2. **Duval's algorithm** **[Duv89]** — faster; implemented in `DuvalPuiseuxExpansion`.
   Restricted to polynomials essentially over a Laurent series ring where the coefficient ring
   has characteristic 0 or characteristic greater than the degree of each squarefree factor.
   It does not compute zero terms and defers some field extensions to the parametrization
   stage, making it substantially faster than Walker's method in most cases.

`Roots` automatically selects the appropriate algorithm: Duval's is used by default; Walker's
is used when coefficients involve fractional powers or the characteristic is less than the
degree of a squarefree factor.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SetVerbose("Newton", v)` | Set verbose output level `v` (0, 1, or 2) for `PuiseuxExpansion`, `ExpandToPrecision`, `DuvalPuiseuxExpansion`, `Roots`, and `ImplicitFunction`. Level 1: prints partial solutions that could not reach requested precision; prints polynomials used in extensions; prints which algorithm `Roots` is using (and current denominator for Walker's). Level 2: additionally prints the last polynomials from the Newton polygon stage and some evaluated polynomials in `ImplicitFunction`. | — |

### 46.5.1 Operations not associated with Duval's Algorithm

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PuiseuxExpansion(f, n)` | Returns a sequence of partial Puiseux series expansions of the roots of `f` over a series ring. Roots are returned with relative precision at least n/d, where d is the lcm of the exponent denominator of the expansion and the exponent denominators of the coefficients of `f`. With `n = 0`, returns expansions from the Newton polygon stage only. Parameters: `PreciseRoot` (`BoolElt`, default `false`) — checks if any partial expansion is an exact root and returns it with full precision; `TestSquarefree` (`BoolElt`, default `true`) — remove multiple factors before expanding (set to `false` to avoid precision loss when f is known squarefree); `NoExtensions` (`BoolElt`, default `false`) — find expansions in the original Puiseux ring only without field extensions; `LowerFaces` (`BoolElt`, default `true`) — if `false`, expansions with negative valuations are not found; `OneRoot` (`BoolElt`, default `false`) — return only representatives of conjugate roots rather than all roots. | Walker's algorithm **[Wal78]**. May not terminate for char k ≤ deg f (see **[Gri95]**, pp. 269–272). |
| `ExpandToPrecision(f, c, n)` | Given a polynomial `f` over a Puiseux series ring and a partial root `c` (found e.g. by `PuiseuxExpansion`), continues expanding `c` until it has relative precision n/d. Reduces precision of `c` to n/d if given to higher precision. Errors if `c` is not a partial expansion or is not a unique partial root. Parameters: `PreciseRoot` (`BoolElt`, default `false`); `TestSquarefree` (`BoolElt`, default `true`). | Walker's algorithm **[Wal78]** applied to extend `c`. |
| `ImplicitFunction(f, d, n)` | Returns a root of `f` over a series ring where `d` is the denominator (or a multiple thereof) of the exponent denominator of the root; root is given to absolute precision n/d. Requires f evaluated at 0 to vanish but its derivative not to. Parameter `Verbose Newton` (max 2). | Newton polygon / implicit function approach. |
| `IsPartialRoot(f, c)` | Returns `true` if the series `c` can be expanded to at least one root of `f`. | Polynomial divisibility check. |
| `IsUniquePartialRoot(f, c)` | Returns `true` if `c` can be expanded to exactly one distinct root of `f`. Parameter `TestSquarefree` (`BoolElt`, default `true`): if `false`, takes `f` as given (avoids precision loss for squarefree `f`, but may miss uniqueness of expansions of multiple roots). | Squarefree factorisation + partial root multiplicity. |
| `PuiseuxExponents(p)` | Given a series `p`, returns the sequence of exponents [a/b] of the nonzero terms up to and including the first one where b is the global denominator. | — |
| `PuiseuxExponentsCommon(p, q)` | Given two series, returns the sequence of exponents [a/b] of the nonzero initial terms of `p` and `q` that are equal, up to but not including the first unequal term. | — |

*Worked examples: H46E6 (PuiseuxExpansion then ExpandToPrecision on a degree-3 polynomial; timing comparison showing benefit of starting at low precision and expanding later); H46E7 (IsPartialRoot and IsUniquePartialRoot on (y²−x³)²−yx⁶; errors from ExpandToPrecision on non-unique and non-partial roots; full workflow); H46E8 (PuiseuxExponents and PuiseuxExponentsCommon on expansions over a finite Puiseux field; higher-degree example showing common exponent prefixes).*

### 46.5.2 Operations associated with Duval's Algorithm

The following functions implement Duval's algorithm **[Duv89]**, which is faster than
Walker's and can handle larger degree polynomials. It works with the squarefree part of f
only and requires the polynomial to be essentially over a Laurent series ring, with
coefficient ring of characteristic 0 or characteristic greater than the degree of each
squarefree factor. Rather than computing nonzero terms explicitly at each Newton polygon
stage, it returns parametrizations that can be converted to series separately.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DuvalPuiseuxExpansion(f, n)` | Returns a sequence of parametrizations of Puiseux expansions of roots of `f` (a polynomial over a series ring) as Puiseux series. Each expansion has at least `n` nonzero terms (fewer only if the expansion is finite). Coefficients of `f` must have exponent denominator 1. Parameters: `Version` (`MonStgElt`, default `"Rational"`) — set to `"Classical"` for the slower classical branch that makes all extensions during computation (similar to Walker's issues over Q); `TestSquarefree` (`BoolElt`, default `true`); `NoExtensions` (`BoolElt`, default `false`) — only find expansions in the coefficient ring Puiseux field; `LowerFaces` (`BoolElt`, default `true`); `OneRoot` (`BoolElt`, default `false`). Note: parametrizations from different parametrization tuples may lie in different Puiseux series rings. | Duval's algorithm **[Duv89]**. Faster than Walker: does not iterate through zero terms and defers field extensions to the `ParametrizationToPuiseux` stage. |
| `ParametrizationToPuiseux(T)` | Converts a parametrization tuple `T` (returned by `DuvalPuiseuxExpansion`) to the sequence of series that satisfy it. Found by evaluating `T[2]` at t where `T[1] = λtᵉ`. | Series evaluation of parametrization. |
| `PuiseuxToParametrization(S)` | Returns the simplest parametrization of the series `S`: takes the denominator out of `S` and makes it the exponent of the first entry. | — |

*Worked examples: H46E9 (DuvalPuiseuxExpansion vs PuiseuxExpansion timing on (y²−x³)²−yx⁶; parametrization conversion; example with multiple parametrizations in different Puiseux rings; handling of finite-precision vs exact-coefficient polynomials; combining DuvalPuiseuxExpansion + ExpandToPrecision and comparison with re-running Duval from scratch).*

### 46.5.3 Roots of Polynomials

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Roots(f)` | Finds the roots of `f` lying in the coefficient ring of `f`. Works for any polynomial over any ring for which Magma can compute roots; for series rings computes to at least the default precision of the ring (or the ring's precision if finite). Parameter `Verbose Newton` (max 2). | Duval's algorithm **[Duv89]** by default; switches to Walker's **[Wal78]** if coefficients involve fractional powers or char k < deg(squarefree factor). |
| `Roots(f, n)` | As `Roots(f)` but specific to series rings; `n` specifies a lower bound on the precision to which roots are returned (relative to the lcm of exponent denominators). Roots that are distinct but agree to the specified precision are returned as two distinct roots. | Same algorithm selection as `Roots(f)`. |
| `HasRoot(f)` | Returns `true` and a root if `f` has a root in its coefficient ring computable to the fixed or default precision. Returns `false` if `f` is irreducible over its coefficient ring. | Uses root-finding for the coefficient ring type. |

*Worked examples: H46E10 (Roots with verbose level 1; multiplicity handling for repeated roots; behaviour with limited precision coefficients; switching to Walker's algorithm for fractional-power coefficients).*

---

## 46.6 Bibliography

| Key | Reference |
|-----|-----------|
| **[Duv89]** | Dominique Duval. *Rational Puiseux Expansions.* Compositio Mathematica, **70**:119–154, 1989. |
| **[Gri95]** | Deryn Griffiths. *Series Expansions of Algebraic Functions.* In W. Bosma and A. van der Poorten, editors, *Computational Algebra and Number Theory*, pages 267–277. Kluwer Academic Publishers, Netherlands, 1995. |
| **[Wal78]** | Robert J. Walker. *Algebraic Curves*, pages 98–99. Springer-Verlag, 1978. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Walker's algorithm **[Wal78]** (general Puiseux expansions) | `PuiseuxExpansion`, `ExpandToPrecision`, `Roots` (fallback) |
| Duval's algorithm **[Duv89]** (fast Puiseux expansions via parametrizations) | `DuvalPuiseuxExpansion`, `ParametrizationToPuiseux`, `Roots` (default) |
| Characteristic bound on Walker's method **[Gri95]** | `PuiseuxExpansion` (limitation), `Roots` (limitation) |
| Newton polygon slope reading (valuations of roots) | `ValuationsOfRoots` |
| Convex hull construction (polygon creation) | `NewtonPolygon`, `AllFaces`, `AllVertices`, `InnerFaces`, `LowerFaces`, `OuterFaces` |
| Face function / degeneracy | `FaceFunction`, `IsDegenerate` |
| Partial root checking | `IsPartialRoot`, `IsUniquePartialRoot`, `ExpandToPrecision` |
| Puiseux exponent analysis | `PuiseuxExponents`, `PuiseuxExponentsCommon` |
