# Chapter 126 — Hypergeometric Motives

**Handbook part:** XVI — Arithmetic Geometry
**Handbook pages:** 4225–4241 (PDF pages 4356–4375)

---

## Scope and overview

Hypergeometric motives arise from the arithmetic study of generalised hypergeometric
differential equations. Given two n-tuples α, β ∈ Cⁿ (taken to be rational, and considered
modulo 1), the generalised hypergeometric differential equation

    z(θ + α₁)···(θ + αₙ)F(z) = (θ + β₁ − 1)···(θ + βₙ − 1)F(z),  θ = z d/dz,

has regular singularities only at 0, 1, and ∞. The monodromy representation of the
fundamental group of the thrice-punctured Riemann sphere acting on the solution space
is characterised by: eigenvalues e^{−2πiβⱼ} around 0, eigenvalues e^{2πiαⱼ} around ∞, and
a pseudo-reflection around 1 (with n−1 eigenvalues equal to 1). By a theorem of Levelt
(Amsterdam, 1961), given disjoint sets of eigenvalues (equivalently, all αᵢ − βⱼ nonintegral),
there is a unique such monodromy representation up to conjugacy.

For arithmetic applications the eigenvalues are taken to be roots of unity and the sets
Galois-invariant, so hypergeometric data H can be specified by two coprime products of
cyclotomic polynomials of equal degree. Rodriguez-Villegas conjectures the existence of a
family of pure motives over Q for each such H, with Frobenius traces given by hypergeometric
sums defined by Katz [Kat90, Kat96]. For each rational t ≠ 0, 1 there should be a motive
Hₜ whose L-function satisfies a functional equation of a prescribed type, with Euler factors
at good primes given in terms of Gauss sums via p-adic Γ-functions (the bad Euler factors
are less understood and depend on deformation theory).

The **weight** w of a hypergeometric motive is determined by how much α and β interlace
as multisets of roots of unity. Writing D(x) = #{α : α ≤ x} − #{β : β ≤ x}, one has
w + 1 = maxₓ D(x) − minₓ D(x). Completely interlacing data gives weight 0, corresponding
to Artin representations. The weight controls the size of Euler factor coefficients.

The core computation — the Euler factor Eₚ(T) at a good prime p — uses the formula

    Uq(t) = (1/(1−q)) · Σᵣ ωₚ(Mt)ʳ Qq(r),

where ωₚ is the Teichmüller character, M is the MValue, Qq(r) = (−1)^{m₀} q^{D+m₀−mᵣ} Gq(r)
involves Gauss sums from a GammaArray, and p-adic Γ-functions expedite computation.
The Euler factor is then Eₚ(T) = exp(−Σₙ Upⁿ(t) Tⁿ/n), a polynomial satisfying a local
functional equation.

---

## 126.1 Introduction

*(Introductory section — no intrinsics; see Scope and overview above.)*

---

## 126.2 Functionality

### 126.2.1 Creation Functions

Hypergeometric data can be specified in six different ways, all routing through
`HypergeometricData`. The package always normalises α and β so that β contains the
smallest rational in [0, 1).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HypergeometricData(A, B)` | First principal form: takes two sequences `A` and `B` of rationals of the same length, which must be disjoint modulo 1 and each Galois-invariant (e.g. if 1/6 appears, so must 5/6). Returns hypergeometric data with α = `A`, β = `B` (switched if necessary so β holds the smallest rational in [0, 1)). | Direct specification from α, β. |
| `HypergeometricData(F, G)` | Second principal form: takes two products `F` and `G` of cyclotomic polynomials, coprime and of the same degree. The polynomials determine α and β. α and β are switched if necessary. | Cyclotomic polynomial specification. |
| `HypergeometricData(G)` (sequence of integers) | Third form: a sequence of integers `G` with Σᵥ v·G[v] = 0. Here Pα(T)/Pβ(T) = Πᵥ(Tᵛ − 1)^{G[v]}, and α, β are recovered by Möbius inversion. Negating all entries of G corresponds to swapping α and β. | GammaArray/Möbius inversion specification. |
| `HypergeometricData(G)` (list of nonzero integers) | Fourth form: a list of nonzero integers (with repetition) corresponding to the sequence G of the previous form; negative integers indicate where G[v] is negative. The sum of all members must be 0. | GammaList specification. |
| `HypergeometricData(F, G)` (arrays of integers) | Fifth form: two arrays `F` and `G` of integers specifying which cyclotomic polynomials appear in α and β respectively. | Cyclotomic index array specification. |
| `HypergeometricData(E)` | Utility form: takes a sequence `E` of two sequences and passes them to the appropriate intrinsic above. | Dispatcher. |
| `Twist(H)` | Given hypergeometric data `H`, adds 1/2 to every element in α and β and returns new hypergeometric data. The new α and β may be switched due to twisting. | Direct shift of α, β by 1/2. |
| `PrimitiveData(H)` | Given hypergeometric data `H`, returns its primitive associated data. Most easily described via `GammaList`: divide all elements by their gcd. | GammaList gcd reduction. |
| `PossibleHypergeometricData(d)` | Generates all possible hypergeometric data of degree `d`, returned as a sequence of pairs of rational sequences of length `d`. Parameters: `Weight` (restrict to data of this weight; default: `false`), `TwistMinimal` (only return twist-minimal data; default: `false`), `CyclotomicData` (return cyclotomic data sequences rather than rationals; default: `false`), `Primitive` (if `true`, only primitive data; if a positive integer, data with that imprimitivity; default: `0`). | Enumeration over Galois-invariant multiset pairs. |

### 126.2.2 Access Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Weight(H)` | Returns the weight of the given hypergeometric data `H`. Computed as w + 1 = maxₓ D(x) − minₓ D(x) where D(x) = #{α : α ≤ x} − #{β : β ≤ x}. | Interlacing count formula. |
| `Degree(H)` | Returns the degree of the given hypergeometric data `H` (the common length n of α and β). | — |
| `DefiningPolynomials(H)` | Returns the (products of cyclotomic) polynomials corresponding to α and β for hypergeometric data `H`. | — |
| `CyclotomicData(H)` | Returns two arrays of integers specifying which cyclotomic polynomials occur for α and β. For example, Φ₃Φ₄²Φ₆ is represented by [3,4,4,6]. | — |
| `AlphaBetaData(H)` | Returns two arrays of rationals giving α and β of the hypergeometric data `H`. | — |
| `MValue(H)` | Returns the scaling parameter M of the given hypergeometric data `H`. Defined by Mₙ = Πᵈ|ₙ d^{d·μ(n/d)} for the nth cyclotomic polynomial, combined across the products for α and β and divided; equivalently M = Πᵥ v^{γᵥ}. | Product formula from cyclotomic data. |
| `GammaArray(H)` | Returns an array of integers γᵥ where Pα(T)/Pβ(T) = Πᵥ(Tᵛ − 1)^{γᵥ} and Σᵥ v·γᵥ = 0. | Möbius inversion. |
| `GammaList(H)` | Returns a list of integers: sgn(γᵥ)·v appears in the list |γᵥ| times. | Expansion of GammaArray. |
| `H1 eq H2` | True if hypergeometric data instances `H1` and `H2` have the same α and β. | Equality of multisets. |
| `H1 ne H2` | True if hypergeometric data instances `H1` and `H2` do not have the same α and β. | Equality of multisets. |
| `IsPrimitive(H)` | Returns `true` if the given hypergeometric data `H` is primitive, and also returns the index of imprimitivity (the gcd of the elements in the GammaList). | GammaList gcd. |

### 126.2.3 Functionality with L-series and Euler Factors

The central computation of the package is `EulerFactor`, which implements the Gauss-sum
and p-adic Γ-function formula for the Euler factor of a hypergeometric motive. The bad
prime theory distinguishes three regimes: **multiplicative** primes (where vₚ(t − 1) > 0,
conductor exponent 1, factor of degree d − 1 recoverable by p-adic Γ methods), primes
dividing 1/t (monodromy ∞, inertia from β roots of unity with maximal Jordan blocks),
primes dividing t (monodromy 0, inertia from α), and **wild** primes (dividing the denominator
of some α or β, not handled automatically). Tame bad primes are handled by extracting
suitable terms from the hypergeometric trace using the smallest q = pᶠ ≡ 1 (mod m).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `EulerFactor(H, t, p)` | The heart of the package. Given hypergeometric data `H`, a rational `t ≠ 0, 1`, and a prime `p`, computes the pth Euler factor of the hypergeometric motive at `t`. The prime `p` must not be wild (must not divide the denominator of any α or β). Parameters: `Precision` (number of terms to compute; 0 = full polynomial; default: 0), `Check` (if `false`, suppress use of the local functional equation; default: `false`), `Fake` (compute hypergeometric traces for t with vₚ(Mt) = 0, including wild primes, without the local functional equation; default: `false`). | p-adic Γ-function formula for Gauss sums, as indicated by Cohen. Local functional equation used to expedite computation (unless disabled). See **[Kat90]**. |
| `LSeries(H, t)` | Given hypergeometric data `H` and rational `t ≠ 0, 1`, constructs the L-series of the associated motive. Wild prime Euler factors must be supplied by the user; tame/multiplicative Euler factors and γ-factors are computed automatically. Parameters: `BadPrimes` (sequence of triples `<p, conductor_exponent, factor>` for bad primes; default: `[]`), `GAMMA` (γ-factors; default: `[]`), `Identify` (attempt to identify weight 0 motives as Artin representations and weight 1 motives as (hyper)elliptic curves; default: `true`). | Euler product from `EulerFactor`; functional equation framework from Magma's L-series machinery. |

#### 126.2.3.1 Identification of Hypergeometric Data as Other Objects

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ArtinRepresentation(H, t)` | Given hypergeometric data `H` of weight 0 and rational `t ≠ 0, 1`, attempts to determine the associated Artin representation. Implemented for all such `H` of degree ≤ 3, and for some of degree 4 and higher; requires `GammaList(H)` to have cardinality 3. Parameter: `Check` (if `true`, Euler factors at good primes up to 100 are verified; default: `true`). | Identification via Euler factor matching. |
| `EllipticCurve(H)` | Given hypergeometric data `H` of degree 2 and weight 1 (10 such families exist), returns the associated elliptic curve over a function field, as catalogued by Cohen. For imprimitive data of index r, returns generically an elliptic curve over an extension of degree r (or an array when xʳ − 1/Mt splits). | Cohen's catalogue of degree-2 weight-1 families. |
| `EllipticCurve(H, t)` | As `EllipticCurve(H)` but specialised at rational `t ≠ 0, 1`. | Cohen's catalogue; specialisation. |
| `HyperellipticCurve(H)` | Given hypergeometric data `H` of degree 4 and weight 1 where the data is known to correspond to a hyperelliptic curve, returns the associated curve over a function field. There are 18 cases giving a genus-2 curve from `CanonicalCurve` (36 with twists), plus a few others with higher genus canonical curves admitting a genus-2 quotient. Returns `false` if no such curve can be constructed. | Catalogue of degree-4 weight-1 cases. |
| `HyperellipticCurve(H, t)` | As `HyperellipticCurve(H)` but specialised at rational `t ≠ 0, 1`. | Catalogue of degree-4 weight-1 cases; specialisation. |
| `Identify(H, t)` | Given hypergeometric data `H` and rational `t ≠ 0, 1`, returns any known associated object or `false`. Currently returns: an Artin representation (weight 0); an elliptic curve over Q (weight 1, degree 2); an elliptic curve over a number field (weight 1, degree 2r with imprimitivity r), possibly multiple curves; or a hyperelliptic curve over Q (weight 1, degree 4). | Dispatches to `ArtinRepresentation`, `EllipticCurve`, or `HyperellipticCurve`. |

*Worked examples: H126E1 (constructing motives, identifying elliptic curves and Artin representations; checking Euler factors); H126E2 (twisting hypergeometric data, twist-related Artin motives and (hyper)elliptic curves); H126E5 (degree 4 weight 3 example, `LSeries` with bad primes, comparison to tensor product of L-series, Siegel modular form connection [vGvS93]); H126E6 (handling bad primes via deformation theory, `Fake` vararg for wild prime); H126E7 (quintic 3-fold, tame and wild prime handling, Grössencharacter example).*

### 126.2.4 Associated Schemes and Curves

The canonical scheme associated to hypergeometric data H is constructed from the GammaList.
For GammaList elements gᵢ⁺ (positive) and gⱼ⁻ (negative), the scheme lives in an ambient
space with variables Xᵢ (one per positive element) and Yⱼ (one per negative element), defined
by Σ Xᵢ = Σ Yⱼ = 1 and Πᵢ Xᵢ^{gᵢ⁺} / Πⱼ Yⱼ^{|gⱼ⁻|} = 1/(Mt). The canonical curve reduces
this to a plane curve when the GammaList has 4 or sometimes 6 elements.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CanonicalScheme(H)` | Given hypergeometric data `H`, constructs the canonical associated scheme over a function field (the parameter t not specialised). The scheme is determined from the GammaList, with a variable for every element; it is the intersection of Σ Xᵢ = Σ Yⱼ = 1 with the monomial equation above. | Scheme construction from GammaList monomials. |
| `CanonicalScheme(H, t)` | As `CanonicalScheme(H)` but with the rational parameter `t ≠ 0, 1` specialised. | Specialisation of canonical scheme. |
| `CanonicalCurve(H)` | Given suitable hypergeometric data `H`, attempts to construct an associated plane curve over a function field. Possible when the GammaList has 4 elements; sometimes possible with 6 elements (when the largest absolute-value element equals the negation of the sum of two others). Returns `false` if not possible. | Plane curve from GammaList Jacobi sums. |
| `CanonicalCurve(H, t)` | As `CanonicalCurve(H)` but with the rational parameter `t ≠ 0, 1` specialised. | Specialisation of canonical curve. |

*Worked examples: H126E4 (degree-4 data, canonical scheme and curve, hyperelliptic check, example with reducible curve over a constant field extension); H126E8 (degree-10 weight-1 data, genus-5 hyperelliptic curve from `CanonicalCurve`, Euler factor comparison, tame prime behaviour).*

### 126.2.5 Utility Functions

These intrinsics manage a cache of precomputed p-adic Γ-function values, which can
significantly accelerate repeated Euler factor computations at the same prime.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HypergeometricMotiveSaveLimit(n)` | Instructs the package to cache all computed p-adic Γ-function values when the prime power q is less than `n`. The qth table entry has (q − 1) elements. | Memoisation of Gauss sum pre-computations. |
| `HypergeometricMotiveClearTable()` | Clears the table of cached p-adic Γ-function values. | Cache invalidation. |

*Worked example: H126E9 (enumerating `PossibleHypergeometricData` by degree and weight; speed test showing cache effect of `HypergeometricMotiveSaveLimit`).*

---

## 126.3 Examples

*(Examples H126E1–H126E9 are embedded in the relevant subsections above. The source code is on handbook pages 4233–4240.)*

---

## 126.4 Bibliography

| Key | Reference |
|-----|-----------|
| **[Kat90]** | N. M. Katz. *Exponential Sums and Differential Equations*, volume 124. Annals of Math. Studies., 1990. |
| **[Kat96]** | N. M. Katz. *Rigid Local Systems*, volume 139. Annals of Math. Studies., 1996. |
| **[vGvS93]** | B. van Geemen and D. van Straten. The cusp forms of weight 3 on Γ₂(2, 4, 8). *Math. Comp.*, 61(204):849–872, 1993. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| p-adic Γ-function Gauss-sum formula for Euler factors **[Kat90]** | `EulerFactor` |
| Local functional equation (expediting Euler factor computation) | `EulerFactor` (suppressed by `Check:=false` or `Fake`) |
| Tame/multiplicative bad prime Euler factors (automatic) | `EulerFactor`, `LSeries` |
| L-series construction (Euler product + functional equation) | `LSeries` |
| Identification as Artin representation (weight 0) | `ArtinRepresentation`, `Identify`, `LSeries(:Identify)` |
| Cohen's catalogue of degree-2 weight-1 elliptic curve families | `EllipticCurve` |
| Catalogue of degree-4 weight-1 hyperelliptic curve cases | `HyperellipticCurve` |
| Canonical scheme from GammaList monomials | `CanonicalScheme` |
| Plane curve from GammaList Jacobi sums | `CanonicalCurve` |
| Möbius inversion (α, β from GammaArray) | `HypergeometricData(G)`, `GammaArray`, `GammaList`, `MValue` |
| Cache of p-adic Γ-function pre-computations | `HypergeometricMotiveSaveLimit`, `HypergeometricMotiveClearTable` |
| Enumeration of hypergeometric data by degree/weight/primitivity | `PossibleHypergeometricData` |
