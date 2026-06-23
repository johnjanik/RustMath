# Chapter 130 — Congruence Subgroups of PSL₂(R)

**Author:** Helena Verrill
**Handbook part:** XVII — Modular Arithmetic Geometry
**Handbook pages:** 4337–4359 (PDF pages 4468–4493)

---

## Scope and overview

The group GL₂⁺(**R**) of 2×2 real matrices with positive determinant acts on the upper half
complex plane **H** = {z ∈ **C** | Im(z) > 0} by fractional linear transformation
`((a,b),(c,d)) : z ↦ (az + b)/(cz + d)`. Any subgroup Γ of GL₂⁺(**R**) also acts on **H**. To
compactify the quotient Γ\**H** one adjoins the *cusps*, the points of **P¹(Q) = Q ∪ {∞}};
one writes **H*** := **H** ∪ **P¹(Q)**.

A *fundamental domain* for the action of Γ is a region of **H*** containing a representative of
each orbit; it is most useful when it can be taken compact with finite hyperbolic area, which is
the situation for *congruence subgroups*. A congruence subgroup is any discrete subgroup Γ of
SL₂(**R**) that is commensurable with SL₂(**Z**) (i.e. Γ ∩ SL₂(**Z**) has finite index in both)
and contains Γ(N) for some N; the *level* of Γ is the greatest such N. Magma abuses notation and
refers to projectivizations (subgroups of PSL₂) by the same names.

This chapter describes how to work in Magma with **H*** and with congruence subgroups and their
action on **H***. This allows computation of generators for congruence subgroups, together with
cusps, elliptic points, genus, indices, and other invariants. **Farey symbols** (generalized
Farey sequences with edge-identification labels) are the central computational device for finding
fundamental domains; they can be computed and manipulated by the user. Procedures are provided
for producing graphical images, as PostScript files, that illustrate fundamental domains and
draw geodesics and polygons in the upper half plane.

The package was written by Helena Verrill and is partly based on a Java program for drawing
fundamental domains **[Ver00]**. The Farey sequence algorithm used is that of Kulkarni
**[Kul91]**.

---

## 130.2 Congruence Subgroups

Magma works with the standard families of congruence subgroups of SL₂(**Z**) (and their
projectivizations): Γ₀(N), Γ₁(N), Γ(N), Γ¹(N) and Γ⁰(N), defined by congruence conditions on the
matrix entries mod N.

### 130.2.1 Creation of Subgroups of PSL₂(R)

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PSL2(R)` | Returns PSL₂(R), the projective linear group over the ring `R`. | — |
| `Gamma0(N)` | The group Γ₀(N) for any positive integer `N`. | — |
| `Gamma1(N)` | The group Γ₁(N) for any positive integer `N`. | — |
| `GammaUpper0(N)` | The group Γ⁰(N) for any positive integer `N`. | — |
| `GammaUpper1(N)` | The group Γ¹(N) for any positive integer `N`. | — |
| `CongruenceSubgroup(N)` | The group Γ(N) for any positive integer `N`. | — |
| `CongruenceSubgroup(i, N)` | For a positive integer `N` and `i = 0, 1, 2, 3, 4`: the group Γ₀(N), Γ₁(N), Γ(N), Γ¹(N) or Γ⁰(N) respectively. | — |
| `CongruenceSubgroup([N, M, P])` | The congruence subgroup consisting of 2×2 integer matrices `[a,b,c,d]` with `b ≡ 0 (mod P)`, `c ≡ 0 (mod N)`, and `a ≡ d ≡ 1 (mod M)`. Requires `M` divides `NP`. | Intersection of standard congruence conditions. |
| `Intersection(G, H)` / `G meet H` | The intersection of congruence subgroups `G` and `H`. | — |

*Worked examples:* H130E1 (`CongruenceSubgroup(0,35)`, `Generators`, `CosetRepresentatives`,
`DisplayPolygons` of triangle translates and a fundamental domain); H130E2 (defining
`CongruenceSubgroup([2,3,6])` and intersections, e.g. printing as
`Gamma_0(2) intersection Gamma^1(3) intersection Gamma^0(2)`).

### 130.2.2 Relations

| Intrinsic | Description |
|-----------|-------------|
| `G eq H` | Returns `true` iff the congruence subgroups `G` and `H` are equal. |
| `H subset G` | For `G`, `H` contained in PSL₂(**Z**): returns `true` iff `H` is a subgroup of `G`. |
| `Index(G, H)` | For congruence subgroups `G`, `H`: the index of `G` in `H`, provided `G` is a subgroup of `H`. |
| `Index(G)` | For `G` a congruence subgroup in PSL₂(**Z**): the index in PSL₂(**Z**). |

### 130.2.3 Basic Attributes

| Intrinsic | Description |
|-----------|-------------|
| `Level(G)` | The level of a congruence subgroup `G`. |
| `IsCongruence(G)` | Returns `true` iff `G` is a congruence subgroup. |
| `IsGamma0(G)` | Returns `true` iff `G` equals Γ₀(N) for some integer `N`. |
| `IsGamma1(G)` | Returns `true` iff `G` equals Γ₁(N) for some integer `N`. |
| `BaseRing(G)` | The base ring over which matrices of the congruence subgroup `G` are defined. |
| `Identity(G)` | The identity matrix in the congruence subgroup `G`. |

---

## 130.3 Structure of Congruence Subgroups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CosetRepresentatives(G)` | If `G` is a subgroup of finite index in PSL₂(**Z**): a sequence of coset representatives of `G` in PSL₂(**Z**). | — |
| `Generators(G)` | A sequence of generators of the congruence subgroup `G`. | Farey symbol method (Kulkarni **[Kul91]**). |
| `FindWord(G, g)` | For a congruence subgroup `G` and `g ∈ G`: a sequence `[e₁n₁, e₂n₂, …, eₘnₘ]` (positive `nᵢ`, `eᵢ = ±1`) expressing `g = L[n₁]^{e₁} L[n₂]^{e₂} … L[nₘ]^{eₘ}` in terms of the fixed generator list `L = Generators(G)`. Since the computation is in PSL₂(**R**), the equality holds only up to multiplication by ±1. | Rewriting against the Farey-symbol generators. |
| `Genus(G)` | The genus of the upper half plane quotiented by the congruence subgroup `G`. | — |
| `FundamentalDomain(G)` | For `G` a subgroup of PSL₂(**Z**): a sequence of points in the upper half plane which are the vertices of a fundamental domain for `G`. | Farey-symbol fundamental domain. |

*Worked examples:* H130E3 (generators of Γ₀(12), coset representatives, triangle translates);
H130E4 (writing `g := G![21,4,68,13]` in terms of `Generators` via `FindWord`, verifying
`gens[8]^(-1)*gens[1]`).

### 130.3.1 Cusps and Elliptic Points of Congruence Subgroups

| Intrinsic | Description |
|-----------|-------------|
| `Cusps(G)` | A sequence of inequivalent cusps of the congruence subgroup `G`. |
| `CuspWidth(G, x)` | The width of `x` as a cusp of the congruence subgroup `G`. |
| `EllipticPoints(G)` / `EllipticPoints(G, H)` | A list of inequivalent elliptic points for the congruence subgroup `G`. A second argument specifies the upper half plane `H` containing these elliptic points. |

*Worked example:* H130E5 (representative cusps and widths of Γ₁(12) — note `&+Widths(G)` equals
`Index(G)` = 24; finding the Γ₀(N), N < 20, with most elliptic points — Γ₀(13), and listing them,
e.g. `5/13 + (1/13)*root(-1)`).

---

## 130.4 Elements of PSL₂(R)

### 130.4.1 Creation

| Intrinsic | Description |
|-----------|-------------|
| `G ! x` | If `x` is a sequence `[a,b,c,d]` of base-ring elements: returns the matrix `((a,b),(c,d))`, provided it is an element of `G`. If `x` is an integer the identity matrix is returned. If `x` is a matrix, it is coerced into `G` if possible. |
| `Random(G, m)` | A random element of the projective linear group `G`, with `m` determining the size of the coefficients. |

*Worked example:* H130E6 (`G![2,0,0,2]` coerced to identity `[1 0]/[0 1]`; `H![7,6,8,7]` in
`CongruenceSubgroup([2,3,6])`).

### 130.4.2 Membership and Equality Testing

| Intrinsic | Description |
|-----------|-------------|
| `g eq h` | For `g`, `h` in PSL₂(**Z**): returns `true` if they have compatible coefficient rings and `g = h`. Since the group is projective, returns `true` if the matrices are equal up to a nonzero scalar multiple. |
| `IsEquivalent(g, h, G)` | For `g`, `h` in PSL₂(**Z**): returns `true` if they are over the same field and `Gg = Gh`, i.e. if `gh⁻¹ ∈ G`. |
| `g in G` | For `g` in PSL₂(**Z**): returns `true` iff `g` is in the congruence subgroup `G`. |

### 130.4.3 Basic Functions

| Intrinsic | Description |
|-----------|-------------|
| `Eltseq(g)` | The sequence of four numbers which are the entries of the matrix `g`. |
| `g * h` | If `g` and `h` have the same parent, returns their product. |
| `g ^ n` | For a matrix `g` and integer `n`, returns `gⁿ`. |

---

## 130.5 The Upper Half Plane

The upper half complex plane is **H** := {z ∈ **C** | Im(z) > 0}. SL₂(**Z**) acts on **H** by
fractional linear transformations; **H**/SL₂(**Z**) is not compact and is compactified by adding
the cusps (points of **Q** together with ∞). One sets **H*** to be the upper half plane union the
cusps, so **H***/SL₂(**Z**) is compact. In **H*** there are two distinguished elliptic points,
√−1 and (1 + √−3)/2. In general, points constructed in **H*** may come from at most quadratic
extensions of **Q** (so there is a canonical embedding in **C**).

### 130.5.1 Creation

| Intrinsic | Description |
|-----------|-------------|
| `UpperHalfPlane()` | Creates a copy of the upper half complex plane, with cusps included. As a set this is all complex numbers with positive imaginary part, together with all rationals and the point at infinity. (Also available as `UpperHalfPlaneWithCusps()`.) |
| `H ! x` | Returns `x` as a point in `H`. Here `x` can be a cusp, rational, integer, an element in a quadratic extension of **Q**, or a complex number with positive imaginary part. |

*Worked example:* H130E7 (coercing a cusp `Cusps()!(1/2)`, an element `u+5` of `QuadraticField(-7)`,
and naming the two distinguished elliptic points via `H<i,rho> := UpperHalfPlaneWithCusps()` — `i`
= `root(-1)`, `rho` = `1/2 + (1/2)*root(-3)`).

### 130.5.2 Basic Attributes

| Intrinsic | Description |
|-----------|-------------|
| `Imaginary(z)` | The imaginary part of the argument, as an element of `RealField`. |
| `Real(z)` | The real part of the argument, as an element of `RealField`. |
| `IsReal(z)` | Returns `true` iff `z` lies on the real line (and is not the infinite cusp). |
| `IsCusp(z)` | Returns `true` iff `z` is a cusp. |
| `IsInfinite(z)` | Returns `true` iff `z` is the cusp at infinity. |
| `IsExact(z)` | Returns `true` iff `z` is a cusp or has an exact value defined in a quadratic extension of the rationals. |
| `ExactValue(z)` | If `z` is a cusp, returns its value as a `SetCspElt`; if `z` has an exact value in a quadratic extension, returns it as a `FldQuadElt`; otherwise returns a complex value of type `FldComElt`. |
| `ComplexValue(x)` | Returns `x` as a complex number. When `x` is the cusp at infinity, the value returned is `MaxValue + i*MaxValue`. Parameters: `Precision` (`RngIntElt`); `MaxValue` (`RngIntElt`, default 600). |
| `x eq y` | Returns `true` iff the points `x` and `y` in the upper half plane are equal. |

---

## 130.6 Action of PSL₂(R) on the Upper Half Plane

| Intrinsic | Description |
|-----------|-------------|
| `g * z` | For `z` of type `SpcHypElt`, `SetCspElt` (or `[SpcHypElt]`) and `g` an element of a projective linear group: the image of `z` under the action of `g`. The image type matches that of `z`. If `g` is a positive integer, returns `az`, equivalent to acting by the matrix `((a,0),(0,1)) ∈ PGL₂(**R**)`. |
| `FixedPoints(g, H)` | A sequence of points in `H` fixed by the action of `g`. |
| `IsEquivalent(G, a, b)` | If points `a`, `b` in the upper half plane are equivalent under the action of `G`, returns `true` and a matrix `g ∈ G` with `g·a = b`; otherwise returns `false` and the identity. |
| `EquivalentPoint(x)` | For a point `x` in the upper half plane: a point `z` in the region with −1/2 < z ≤ 1/2 and \|z\| ≥ 1, and a matrix `g ∈ PSL₂(**Z**)` with `g*x = z`. |
| `Stabilizer(a, G)` | A generator of the subgroup of `G` stabilizing `a`. |
| `FixedArc(g, H)` | If `g ∈ PSL₂(**Z**)` is an involution: the end points on the real line of the arc fixed by `g`, with mid point of the arc also fixed by `g`. For any point `b`, the arc from `b` to `g·b` is fixed by `g`. |

### 130.6.1 Arithmetic

| Intrinsic | Description |
|-----------|-------------|
| `z + a` / `z - a` | For any integer `a` and element `z` in the upper half plane: the element `z + a` (resp. `z − a`) in the same copy of the upper half plane. |
| `a * z` / `z * a` / `a * seq` / `z / a` | Given an element `z` (or a sequence of elements) in the upper half plane, and a positive rational `a`: the product(s) in the same copy of the upper half plane. |

### 130.6.2 Distances, Angles and Geodesics

| Intrinsic | Description |
|-----------|-------------|
| `Distance(z, w)` | The hyperbolic distance between `z` and `w`. Parameter `Precision` (`RngIntElt`). |
| `TangentAngle(x, y)` | The angle of the tangent at `x` of the geodesic from `x` to `y`, to the given precision. Parameter `Precision` (`RngIntElt`). |
| `Angle(e1, e2)` | Given sequences `e₁ = [z1, z2]` and `e₂ = [z1, z3]`: the angle between the geodesics at `z1`. Parameter `Precision` (`RngIntElt`). |
| `ExtendGeodesic([z1, z2], H)` | Given `z1`, `z2` in the upper half plane `H`: extends the geodesic between them to a semicircle with endpoints on the real line, and returns the two real endpoints as elements of `H`. |
| `GeodesicsIntersection(x1, x2)` | The intersection in the upper half plane of the two geodesics whose endpoints are given by the sequences `x1` and `x2`. If the geodesics intersect along a line, the empty sequence is returned. |

---

## 130.7 Farey Symbols and Fundamental Domains

One method of finding fundamental domains for congruence subgroups is the method of Farey symbols
**[Kul91]**. A *generalized Farey sequence* is a sequence of rationals
`a₁/b₁ < a₂/b₂ < … < aₙ/bₙ` such that for a consecutive pair `b/d, a/c` (in lowest terms)
`ad − bc = 1`. The rationals are extended to **Q** ∪ {−∞, ∞} with `−∞ = −1/0` and `∞ = 1/0`.

A *Farey symbol* is a Farey sequence of length `n` starting with `−1/0` and ending with `1/0`,
together with a sequence of `n − 1` labels. Labels are any elements of N>0 ∪ {−2, −3}, with each
element of N>0 appearing either exactly twice or not at all. The fractions give cusps that are
vertices of the domain, and the labels give edge identifications: for `aᵢ, aᵢ₊₁` with label `lᵢ`
not −3 the corresponding edge is the geodesic between `aᵢ` and `aᵢ₊₁`; label −3 indicates an extra
elliptic point of order 3 on the boundary (the two edges between these cusps are identified); label
−2 indicates an elliptic point of order 2 on the geodesic between `aᵢ` and `aᵢ₊₁` (the two halves
of the geodesic are identified).

| Intrinsic | Description |
|-----------|-------------|
| `FareySymbol(G)` | Computes the Farey Symbol of a congruence subgroup `G` in PSL₂(**Z**). |
| `Cusps(FS)` | The cusp sequence of the Farey symbol `FS`. Note: this is *not* a sequence of inequivalent cusps of the corresponding group. |
| `Labels(FS)` | The sequence of edge labels of a Farey symbol `FS`. |
| `Generators(FS)` | The generators of the congruence subgroup corresponding to the Farey symbol `FS`. |
| `Group(FS)` | The congruence subgroup corresponding to the Farey Symbol `FS`. |
| `Widths(FS)` | The sequence of integers giving twice the widths of the cusp list of the Farey symbol `FS`. |
| `Index(FS)` | The index of `Group(FS)` in PSL₂(**Z**). |
| `FundamentalDomain(FS)` / `FundamentalDomain(FS, H)` | The vertices in the upper half plane of the fundamental domain described by the Farey Sequence `FS`. A second argument may specify the upper half plane `H`. |
| `CosetRepresentatives(FS)` | The coset representatives of the congruence subgroup of PSL₂(**Z**) corresponding to the Farey symbol `FS`. |
| `InternalEdges(FS)` | A sequence of pairs of cusps which are cusps of the Farey Symbol `FS`, not adjacent in `FS`, but which are images of 0 and infinity under some matrix in PSL₂(**Z**). |

---

## 130.8 Points and Geodesics

Geodesics in the upper half plane are given by circles or lines which intersect the real axis at
right angles; a geodesic is defined by its two end points. Points and geodesics can be drawn using
the graphics functions of the next section.

| Intrinsic | Description |
|-----------|-------------|
| `GeodesicsIntersection(x, y)` | The intersection in the upper half plane of the two geodesics `x`, `y`, specified by their end points (which must be cusps). If the geodesics intersect along a line, the empty sequence is returned. |

*Worked example:* H130E8 (intersection points of geodesics `[H\|-1,2]`, `[H\|0,6]`, `[H\|1,5]`,
`[H\|2,Infinity()]`, displayed via `DisplayPolygons`).

---

## 130.9 Graphical Output

When working with a congruence subgroup it is often useful to produce a picture of a fundamental
domain of the group, and to draw images of geodesics in **H***. These functions produce PostScript
files.

| Intrinsic | Description |
|-----------|-------------|
| `DisplayPolygons(P, file)` | Given a sequence of polygons, each defined by a sequence of points in the upper half plane, produces the PostScript drawing and writes it to the named file. Returns a sequence of 4 real numbers `[x0, x1, h, S]` where `(x0,0)` is the lower-left corner, `(x1,h)` the upper-right, and `S` the scale in pixels per unit. A polygon of ≥3 points is drawn as a polygon; 2 points draw a geodesic; 1 point marks the point. Parameters (with defaults): `Colours` (`SeqEnum`, `[1,1,0]`; fill colour, RGB in 0–1); `PenColours` (`SeqEnum`, `[0,0,0]`; outline colour); `Outline` (`BoolElt`, `true`); `Fill` (`BoolElt`, `true`; if both `Fill` and `Outline` are false, outline is reset to true); `Show` (`BoolElt`, `false`; if true issues `System("gv file &")`); `Labels` (`SeqEnum`, `[0,1]`; `SetCspElt`/`FldRatElt`/`RngIntElt` labelled on the real axis); `Fontsize` (`RngIntElt`, 2); `Size` (`SeqEnum`, `[]`; `[x0,x1,y,S]` to set image size/scale); `Pixels` (`RngIntElt`, 300; autoscale width in pixels, min 10); `Overwrite` (`BoolElt`, `false`); `Radius` (`FldReElt`, 0.5; radius of marked points). |
| `DisplayFareySymbolDomain(FS, file)` / `DisplayFareySymbolDomain(G, file)` | Displays a fundamental domain corresponding to a Farey symbol `FS`, or a group `G` for which the Farey symbol is computed, with edge identifications and cusps labelled. Returns `[x0, x1, h, S]` as above. Parameters (with defaults): `Colour` (`SeqEnum`, `[1,1,0]`; domain colour); `Show` (`BoolElt`, `false`); `Fontsize` (`RngIntElt`, 2; cusp label font); `Labelsize` (`RngIntElt`, 3; identification-label font); `Autoscale` (`BoolElt`, `true`); `Size` (`SeqEnum`, `[]`; required if `Autoscale` is false); `Pixels` (`RngIntElt`, 300; autoscale width, min 10); `Overwrite` (`BoolElt`, `false`); `ShowInternalEdges` (`BoolElt`, `false`; if true, internal edges divide the domain into fundamental domains for Γ₀(2)). |

*Worked examples:* H130E9 (fundamental domain for Γ₀(24)/Γ₀(12) via `FareySymbol`, a procedure
`draw_fundamental_domain` colouring coset translates, Γ₀(11) tilings, and Γ⁰(2) tilings from
generators); H130E10 (`FareySymbol` of `CongruenceSubgroup(5)` with `Cusps`/`Labels`,
`DisplayFareySymbolDomain`, genus 0); H130E11 (Γ₀(37) — fundamental domain, inequivalent elliptic
points, custom colours/sizes, and inclusion of the PostScript file in LaTeX via `graphicx`).

---

## 130.10 Bibliography (canonical references)

| Key | Reference |
|-----|-----------|
| **[Kul91]** | Ravi S. Kulkarni. *An arithmetic-geometric method in the study of the subgroups of the modular group.* Amer. J. Math. **113**(6):1053–1133, 1991. |
| **[Ver00]** | H. A. Verrill. *Fundamental domain drawing program.* URL: http://hverrill.net/fundomain/, 2000. |

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Farey symbol / generalized Farey sequence method (Kulkarni) **[Kul91]** | `FareySymbol`, `Generators`, `FindWord`, `FundamentalDomain`, `CosetRepresentatives`, `Cusps`, `Labels`, `Widths`, `Index`, `Group`, `InternalEdges` |
| Congruence-subgroup construction (congruence conditions mod N) | `Gamma0`, `Gamma1`, `GammaUpper0`, `GammaUpper1`, `CongruenceSubgroup`, `Intersection`/`meet` |
| Action of PSL₂(R) on H* (fractional linear transformations) | `g * z`, `FixedPoints`, `IsEquivalent`, `EquivalentPoint`, `Stabilizer`, `FixedArc` |
| Cusp / elliptic-point / genus invariants | `Cusps`, `CuspWidth`, `EllipticPoints`, `Genus`, `Level` |
| Hyperbolic geometry of H* (distances, angles, geodesics) | `Distance`, `TangentAngle`, `Angle`, `ExtendGeodesic`, `GeodesicsIntersection` |
| PostScript graphical output (fundamental domains, geodesics) **[Ver00]** | `DisplayPolygons`, `DisplayFareySymbolDomain` |
