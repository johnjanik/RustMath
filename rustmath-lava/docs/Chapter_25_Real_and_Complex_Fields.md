# Chapter 25 — Real and Complex Fields

**Handbook part:** III — Basic Rings
**Handbook pages:** 473–513 (PDF pages 602–649)

---

## Scope and overview

Chapter 25 documents Magma's facilities for computing with approximate real and complex
numbers. Because real and complex numbers cannot be stored exactly in a computer, Magma
represents them as arbitrary-precision approximations. The implementation is built primarily
on two C libraries:

- **MPFR** (multiple-precision floating-point with correct rounding): provides real arithmetic
  conforming to an extension of the ANSI/IEEE-754 double-precision standard. Magma uses MPFR
  2.4.1. Each function uses MPFR unless stated otherwise; for full algorithmic details see
  mpfr.org.
- **MPC** (multiple-precision complex arithmetic, built on MPFR): extends MPFR to complex
  numbers. Magma uses MPC 0.8.

Where MPFR/MPC do not yet provide a function, Magma falls back to **Pari**; such functions
are noted explicitly below.

Real numbers are stored internally as base-2 expansions Σ bᵢ2ⁱ. A complex number is a
pair of real numbers of identical precision. Magma maintains a global cache of real/complex
fields; any two fields of the same fixed precision are guaranteed to be identical objects.
When a binary operation involves operands of different precisions the result has the smaller
of the two precisions.

Types involved: `FldRe` / `FldReElt` (real field and elements); `FldCom` / `FldComElt`
(complex field and elements). Although called "fields" internally, a fixed-precision subset
of R or C need not be closed under arithmetic; Magma nonetheless treats them as fields.

The chapter is structured as: field creation and options (§25.1–§25.2); generic ring/field
predicates on structures and elements (§25.3–§25.4); transcendental functions (§25.5);
elliptic and modular functions (§25.6–§25.7); gamma, Bessel and special functions
(§25.8–§25.10); and numerical analysis utilities (§25.11).

---

## 25.1 Introduction

### 25.1.1 Overview of Real Numbers in Magma

Real and complex fields are arbitrary-precision numeric domains; precision is specified in
decimal digits by default (converted internally to ⌈log₂ 10ᵖ⌉ binary digits) or in binary
digits when `Bits := true`. Two operands of differing precision yield a result at the lower
precision.

*Worked examples: H25E1 (creating reals at precisions 20 and 10; observing precision
reduction in mixed arithmetic).*

### 25.1.2 Coercion

Integer and rational arguments are automatically coerced into the default real field when
passed to functions expecting a real number. For binary operations one real and one
integral/rational argument, the real's parent field is used. Elements of real quadratic and
cyclotomic fields can be coerced with `!` into any real field; any quadratic or cyclotomic
element can be coerced into any complex field. Automatic coercion for these algebraic
elements does not occur for function arguments.

### 25.1.3 Homomorphisms

The only homomorphisms from a real or complex field are coercions.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `hom< R -> S \| >` | Create a coercion homomorphism from real/complex field `R` to structure `S` (which must accept all elements of `R`, e.g. another real/complex field or a polynomial ring over one). | Coercion. |

*Worked examples: H25E2 (two equivalent ways to embed a real field into a polynomial ring
over a complex field, using `hom` and `Bang`).*

### 25.1.4 Special Options

At startup Magma creates real and complex fields of precision 30 as defaults for literal
reals.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SetDefaultRealField(R)` | Procedure: change the default parent for literal real numbers to real field `R`. Default precision is 30. | — |
| `GetDefaultRealField()` | Return the current default real field. | — |
| `AssignNames(~C, [s])` | Procedure: change the print name of √−1 in complex field `C` to string `s`. Does not bind an identifier; use angle brackets or an assignment for that. Modifies `C` in place (hence the `~`). | — |
| `Name(C, 1)` | Return the purely imaginary element √−1 of complex field `C` (the element whose print name was set by `AssignNames`). | — |

*Worked examples: H25E3 (creating `C<i> := ComplexField(20)`; using `AssignNames` to rename `i` to `k`; retrieving `k` via `Name`).*

### 25.1.5 Version Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `GetGMPVersion()` | Return the version string of GMP in use. | — |
| `GetMPFRVersion()` | Return the version string of MPFR in use. | — |
| `GetMPCVersion()` | Return the version string of MPC in use. | — |

---

## 25.2 Creation Functions

### 25.2.1 Creation of Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RealField(p)` | Create the real field R of precision `p`. Parameter `Bits` (BoolElt, default `false`): if `true`, `p` is in binary digits; if `false`, `p` is decimal digits converted to ⌈log₂ 10ᵖ⌉ bits. | MPFR. |
| `RealField()` | Return the default real field (precision 30 unless changed by `SetDefaultRealField`). | — |
| `ComplexField(p)` | Create the complex field C of precision `p`. Parameter `Bits` (BoolElt, default `false`): same semantics as for `RealField`. √−1 has print name `C.1` by default; change with `AssignNames` or angle-bracket syntax `C<i> := ComplexField(p)`. | MPC. |
| `ComplexField()` | Return the default complex field. | — |
| `ComplexField(R)` | Return the complex field whose real subfield is `R`, i.e. the complex field with the same precision as real field `R`. | — |

### 25.2.2 Creation of Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `d.eEfpg` / `d.eefpg` | Literal real: constructs `r = d.e × 10ᶠ` in the real field of precision `g`. If `g` is omitted the default real field is used; if all of `e`, `f`, `g` are omitted the result is an integer. Leading `+`/`-` signs and leading zeroes are accepted. | — |
| `elt< R \| m, n >` | Construct the real number `m × 2ⁿ` in real field `R`, where `m` is coercible into `R` and `n` is an integer. | — |
| `elt< C \| x, y >` / `C ! [x, y]` | Construct the complex number `x + yi` in complex field `C`, where `x` and `y` are coercible into the underlying real field. | — |
| `R ! a` | Coerce integer, rational, or real/quadratic/cyclotomic field element `a` into real field `R`. If `a` has precision `s ≥ p`, truncates to `p` digits; if `s < p`, pads with zeroes. Error if `a` is non-real. | — |
| `C ! a` | Coerce integer, rational, or quadratic/cyclotomic field element `a` into complex field `C`. Same precision rules as for real coercion. | — |
| `One(R)` / `Identity(R)` | Return the multiplicative identity 1 of `R`. | — |
| `Zero(R)` / `Representative(R)` | Return 0 of `R`. | — |

*Worked examples: H25E4 (creating 1.2345 in many ways: literal, scientific notation, `p`-suffix, `!` coercion, `elt<>`; observing default precision effects).*

---

## 25.3 Structure Operations

### 25.3.1 Related Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Category(R)` | The Magma category of structure `R`. | — |
| `Parent(R)` | The parent of `R`. | — |
| `PrimeField(R)` | The prime subfield of `R`. | — |

### 25.3.2 Numerical Invariants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Characteristic(R)` | Returns 0 (real and complex fields have characteristic 0). | — |

### 25.3.3 Ring Predicates and Booleans

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsCommutative(R)`, `IsUnitary(R)` | True for all real/complex fields. | — |
| `IsFinite(R)`, `IsOrdered(R)` | `IsFinite` returns false; `IsOrdered` returns true for real fields, false for complex. | — |
| `IsField(R)`, `IsEuclideanDomain(R)` | Both true. | — |
| `IsPID(R)`, `IsUFD(R)` | Both true. | — |
| `IsDivisionRing(R)`, `IsEuclideanRing(R)` | Both true. | — |
| `IsPrincipalIdealRing(R)`, `IsDomain(R)` | Both true. | — |
| `R eq S`, `R ne S` | Equality/inequality of field structures (same type and precision). | — |

### 25.3.4 Other Structure Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Precision(R)` | Return the decimal precision `p` of real or complex field `R`. | — |
| `BitPrecision(R)` | Return the internal binary precision of `R`. | — |

---

## 25.4 Element Operations

### 25.4.1 Generic Element Functions and Predicates

Predicates that test equality to an integer do so within the precision of the parent field
(e.g. `IsOne(c)` for a precision-20 complex field returns true iff the real part equals 1
and the imaginary part equals 0, both up to 20 decimal places).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Parent(r)`, `Category(r)` | Parent field and category of element `r`. | — |
| `IsZero(r)`, `IsOne(r)`, `IsMinusOne(r)` | Test if `r` equals 0, 1, or −1 within precision. | — |
| `IsUnit(r)`, `IsZeroDivisor(r)` | `IsUnit` true for nonzero elements; `IsZeroDivisor` false. | — |
| `IsIdempotent(r)`, `IsNilpotent(r)` | Standard ring predicates applied within precision. | — |
| `IsIrreducible(r)`, `IsPrime(r)` | Standard ring predicates. | — |

### 25.4.2 Comparison and Membership

Equality on reals tests up to precision; equality on complexes tests real and imaginary parts
separately. The ordering operators `gt`, `ge`, `lt`, `le` are defined only for real numbers.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `a eq b`, `a ne b` | Equality/inequality (up to precision). | — |
| `a in R`, `a notin R` | Membership test. | — |
| `a gt b`, `a ge b`, `a lt b`, `a le b` | Ordered comparison (reals only). | — |
| `Maximum(a, b)`, `Minimum(a, b)` | Binary maximum/minimum of two real numbers. | — |
| `Maximum(Q)`, `Minimum(Q)` | Maximum/minimum over a sequence `Q` of real numbers. | — |

### 25.4.3 Other Predicates

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsIntegral(c)` | True iff real or complex number `c` is a rational integer (within precision). | — |
| `IsReal(c)` | True iff complex number `c` has imaginary part zero to the precision of its parent field. | — |

### 25.4.4 Arithmetic

Automatic coercion applies for mixed integer/rational/real/complex operands.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `+ r`, `- r` | Unary plus and negation. | MPFR/MPC. |
| `r + s`, `r - s`, `r * s`, `r / s`, `r ^ k` | Standard binary arithmetic; result precision is the minimum of the two argument precisions. | MPFR/MPC. |
| `r +:= s`, `r -:= s`, `r *:= s`, `r /:= s`, `r ^:= s` | In-place assignment versions. | MPFR/MPC. |

### 25.4.5 Conversions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `MantissaExponent(r)` | For real `r`, returns mantissa `m` (with 1 ≤ m < 10) and integer exponent `e` such that `r = m × 2ᵉ`. | — |
| `ComplexToPolar(c)` | For complex `c`, returns modulus `m ≥ 0` and argument `a` (−π ≤ a ≤ π) as real numbers of the same precision as `c`. | — |
| `PolarToComplex(m, a)` | Construct complex number `m·eⁱᵃ` from real modulus `m` and argument `a`; result has the smaller of the two precisions. Integer/rational arguments allowed; both integral/rational yields default precision. | — |
| `Argument(c)` / `Arg(c)` | Argument (angle in radians, in (−π, π]) of complex number `c` as a real number of the same precision. | — |
| `Modulus(c)` | Modulus |c| of complex number `c` as a real number of the same precision. | — |
| `Real(c)` / `Re(c)` | Real part of `c = x + yi`, returned as a real number of the same precision. | — |
| `Imaginary(c)` / `Im(c)` | Imaginary part of `c = x + yi`, returned as a real number of the same precision. | — |

### 25.4.6 Rounding

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Round(r)` | Nearest integer to real `r` (away from zero when equidistant). For complex `r`, nearest Gaussian integer. | — |
| `Truncate(r)` | Round toward zero: ⌊r⌋ for r > 0, −⌊−r⌋+1 for r < 0. | — |
| `Ceiling(r)` | Smallest integer ≥ r. | — |
| `Floor(r)` | Largest integer ≤ r. | — |

### 25.4.7 Precision

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Precision(c)` | Decimal precision of element `c` (same as `Precision` of its parent field). | — |
| `BitPrecision(c)` | Internal binary precision of element `c`. | — |
| `Precision(L)` | For a sequence `L` of real or complex numbers, the precision of their parent field. | — |
| `ChangePrecision(r, n)` / `ChangePrecision(c, n)` | Coerce real number `r` (or complex `c`) into the field of precision `n`. | — |

### 25.4.8 Constants

Constants are computed to the precision of the given real or complex field `R`. If `R` is complex, the result has imaginary part zero.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Catalan(R)` | Catalan's constant G = Σₖ₌₀^∞ (−1)ᵏ(2k+1)⁻² to the precision of `R`. | MPFR, using formula (31) of Adamchik's "33 representations for Catalan's constant". |
| `EulerGamma(R)` | Euler's constant γ = limₙ→∞(1 + 1/2 + … + 1/n − log n) ≈ 0.57721566 to the precision of `R`. | MPFR. |
| `Pi(R)` | π to the precision of `R`. | MPFR. |

### 25.4.9 Simple Element Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AbsoluteValue(r)` / `Abs(r)` | Absolute value of real or complex number `r`. | MPFR/MPC. |
| `Sign(r)` | Returns integer +1, 0, or −1 according to whether real number `r` is positive, zero, or negative. | — |
| `ComplexConjugate(c)` / `Conjugate(c)` | Complex conjugate x − yi of c = x + yi. | MPC. |
| `Norm(c)` | Real norm of `c`: for complex `c = x + yi`, returns x² + y²; for real `c`, returns |c|. Result lies in the same field as the argument. | — |
| `Root(r, n)` | n-th real root of real number `r` (n must be positive; if n is even, r must be non-negative). Computed using Newton's method without divisions. | Newton's method (division-free variant). |
| `SquareRoot(c)` / `Sqrt(c)` | Square root of real or complex number `c` in the same field. | MPFR/MPC. |
| `Distance(x, L)` | Minimum distance from `x` to any element of sequence `L` of real or complex numbers; also returns the index in `L` of the achieving element. (Four overloads covering real/complex combinations.) | — |
| `Diameter(L)` | Diameter of sequence `L`: smallest distance between distinct elements. | — |

### 25.4.10 Roots

Magma's primary root-finding algorithm is based on Xavier Gourdon's implementation of
**Schönhage's algorithm** [Sch82]. The algorithm takes a polynomial p ∈ C[z] and ε > 0
and finds linear factors Lⱼ = uⱼz − vⱼ such that |p − L₁···Lₙ| < ε|p|. The key steps
are:

1. **Graeffe process**: root-squaring iterations estimate the moduli r₁ ≤ … ≤ rₙ of the
   roots, providing a **splitting circle** Γ separating some k roots from the remaining n−k.
2. **Splitting circle step**: for the k roots inside Γ, power sums sₘ = Σ uᵢᵐ are computed
   via the residue theorem as the discrete sum (1/N)Σⱼ (p′(ωʲ)/p(ωʲ))ω^{(m+1)j} (where
   ω = exp(2πi/N)). The polynomial F of the inner roots is recovered from sₘ via Newton's
   formulae; G = p/F.
3. **Quadratic refinement**: an auxiliary polynomial H is found via a contour-integral formula
   and then improved by Newton iteration Hₘ₊₁ ≡ Hₘ(2 − HₘG₀) (mod F₀), yielding
   quadratic convergence.
4. The process is applied **recursively** to F and G until only linear factors remain.

For complex polynomials, **Pari** is used instead of the MPFR-based implementation.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Roots(p)` | For univariate polynomial `p` over a real or complex field: a sequence of pairs `<r, m>` (root, multiplicity). Parameter `Al` (MonStgElt, default `"Schonhage"`): one of `"Schonhage"`, `"Laguerre"`, `"NewtonRaphson"`, `"Combination"`. With `"Schonhage"`, roots are correct to absolute error 10⁻ᵈ where `d = Digits`. Parameter `Digits` (RngIntElt, default: current precision of the free real field). Schönhage's algorithm gives correct results in all cases; others are heuristic. For complex polynomial coefficients Pari is used. | Schönhage splitting-circle algorithm **[Sch82]** (Gourdon's implementation); or Laguerre / Newton-Raphson / combination heuristics. |
| `RootsNonExact(p)` | For polynomial `p` of degree n over a real or complex field: a sequence [v₁,…,vₙ] with |p − a(z−v₁)⋯(z−vₙ)| < 10⁻ᵈ|p| (d = field precision, a = leading coefficient). Optionally returns error bounds [e₁,…,eₙ] such that any approximated polynomial with |p̂−p| < 10⁻ᵈ|p| has roots within eᵢ of vᵢ. Error bounds may not be returned if precision is insufficient. | Schönhage's algorithm (treats p as an approximation). |
| `HenselLift(f, R, k)` | Given real or complex polynomial `f` and an approximation `R` to a single zero of `f`, apply Newton iteration to improve the root approximation to precision `k`. | Newton iteration (Hensel lifting). |

*Worked examples: H25E5 (roots of (z−1.1)⁶ over ComplexField() showing floating-point
sensitivity; using exact rational input to get exact roots). H25E6 (RootsNonExact with
error bounds).*

### 25.4.11 Continued Fractions

These functions use continued-fraction expansions for Diophantine approximation; obtained
from Pari.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ContinuedFraction(r)` | For real `r`, return the sequence of partial quotients [s₁,s₂,…,sₙ] of the regular continued fraction expansion r ≈ s₁ + 1/(s₂ + 1/(s₃ + … + 1/sₙ)). Length n is determined by the precision of `r`. Parameter `Bound` (RngIntElt, default −1): if ≥ 0, limits the length to `Bound`. | Pari continued-fraction algorithm. |
| `BestApproximation(r, n)` | For real `r` and positive integer `n`, return a rational approximation to `r` with denominator ≤ n, at least as close as the best continued-fraction convergent with that bound. | Pari. |
| `Convergents(s)` | For sequence `s` of non-negative integers (partial quotients), return the 2×2 matrix [[pₙ, pₙ₋₁], [qₙ, qₙ₋₁]] of the last two convergents pₙ₋₁/qₙ₋₁ and pₙ/qₙ. | Standard convergent recurrence. |

### 25.4.12 Algebraic Dependencies

These functions find integer linear or polynomial relations among real/complex numbers.
Pari is used throughout.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LinearRelation(q: -)` / `LinearRelation(v: -)` | For a sequence `q` or vector `v` over a complex field, return an integer sequence/vector of coefficients for a small linear dependency among the entries. Parameter `Al` (MonStgElt, default `"Hastad"`): `"Hastad"` (variation of LLL due to Hastad, Lagarias, and Schnorr) or `"LLL"` (straight LLL). Superseded by `IntegerRelation`. | LLL-based (Hastad–Lagarias–Schnorr variant or standard LLL); Pari. |
| `AllLinearRelations(q, p)` | For sequence `q` over a real/complex field and integer `p`, return the lattice of all small integer linear dependencies; "small" means the coefficient digit-sum is < p and the relation is zero to within 10⁻ᵖ. | LLL; Pari. |
| `PowerRelation(r, k: -)` | For element `r` from a real/complex field and integer k > 0, return a univariate integer polynomial of degree ≤ k having `r` as an approximate root. Parameters `Al` and `Precision` as for `LinearRelation`. Superseded by `MinimalPolynomial`. | LLL-based (Hastad–Lagarias–Schnorr or LLL); Pari. |

---

## 25.5 Transcendental Functions

### 25.5.1 Exponential, Logarithmic and Polylogarithmic Functions

Power series expansions: eᶻ = Σ zⁿ/n!, ln(1+z) = Σ (−1)ⁿ⁻¹ zⁿ/n. Further information on
Dilog and Polylog in Lewin [Lew81].

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Exp(f)` | For power series `f` over a real or complex field, return the exponential power series eᶠ. | MPFR (series). |
| `Exp(c)` | For real or complex number `c` (free or fixed precision), return eᶜ in the same field. | MPFR/MPC. |
| `Log(f)` | For power series `f` over a real or complex field (valuation 0), return the natural logarithm series. | MPFR (series). |
| `Log(c)` | For nonzero real or complex `c`, return the natural logarithm (principal value, imaginary part in (−π, π]). Returns real if `c` is real and positive. | MPFR/MPC. |
| `Log(b, r)` | For non-negative reals `b`, `r`, return logb(r). Automatic coercion applied. | MPFR. |
| `Dilog(s)` | For complex `s`, return the principal branch of the dilogarithm Li₂(s) = −∫₀ˢ log(1−s)/s ds (analytic continuation of Σ sⁿ/n²). For large arguments a functional equation is used. | Pari. |
| `Polylog(m, f)` | For integer m ≥ 2 and power series `f` (positive valuation, m > 1), return the m-th polylogarithm of `f`. | MPFR (series). |
| `Polylog(m, s)` | For integer m ≥ 2 and complex `s`, return the principal branch of Limₘ(s) = ∫₀ˢ Lim₋₁(s)/s ds. For large arguments a functional equation is used. | Pari. |
| `PolylogD(m, s)`, `PolylogDold(m, s)`, `PolylogP(m, s)` | For integer m ≥ 2 and complex `s`, return modified versions D̃ₘ, Dₘ, Pₘ of Limₘ(s) that satisfy fₘ(1/s) = (−1)ᵐfₘ(s). See Zagier [Zag91] for definitions. | Pari. |

### 25.5.2 Trigonometric Functions

Available for real numbers, complex numbers, and power series over real/complex fields.
MPFR is used for real and complex arguments unless noted.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Sin(f)` / `Sin(c)` | Sine of power series `f` or real/complex number `c`. | MPFR/MPC. |
| `Cos(f)` / `Cos(c)` | Cosine of power series `f` or real/complex number `c`. | MPFR/MPC. |
| `Sincos(f)` / `Sincos(s)` | Return both sin and cos of `f` (power series) or `s` (real/complex number) simultaneously. | MPFR/MPC. |
| `Tan(f)` / `Tan(c)` | Tangent: sin(c)/cos(c). Note: `c` should not be near a zero of cos, i.e. π/2 + nπ. | MPFR/MPC. |
| `Cot(f)` / `Cot(c)` | Cotangent: cos(c)/sin(c). `f` must have valuation 0. `c` should not be near nπ. | MPFR/MPC. |
| `Sec(f)` / `Sec(c)` | Secant: 1/cos(c). `c` should not be near π/2 + nπ. | MPFR/MPC. |
| `Cosec(f)` / `Cosec(c)` | Cosecant: 1/sin(c). `f` must have valuation 0. `c` should not be near nπ. | MPFR/MPC. |

### 25.5.3 Inverse Trigonometric Functions

All available for arbitrary real or complex arguments; principal values as specified.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Arcsin(f)` / `Arcsin(r)` | Inverse sine of series `f` or real/complex `r`. Principal value with real part in [−π/2, π/2]. Returns real if `r` is real with −1 ≤ r ≤ 1. | MPFR/MPC (via arcsin(z) = (1/i) log(iz + √(1−z²))). |
| `Arccos(f)` / `Arccos(r)` | Inverse cosine of series `f` or real/complex `r`. Principal value with real part in [0, π]. Returns real if `r` is real with −1 ≤ r ≤ 1. | MPFR/MPC (via arccos(z) = (1/i) log(z + √(z²−1))). |
| `Arctan(f)` / `Arctan(r)` | Inverse tangent of series `f` or real/complex `r`. Principal value with real part in (−π/2, π/2). Returns real if `r` is real. | MPFR/MPC (via arctan(z) = (1/(2i)) log((1+iz)/(1−iz))). |
| `Arctan(x, y)` / `Arctan2(x, y)` | For real numbers `x`, `y`: arctan(y/x) in (−π, π), resolved by the signs of x and y (four-quadrant arctangent). Error if both are zero; if y = 0 and x ≠ 0, returns sign(x)·π/2. | MPFR (two-argument arctangent). |
| `Arccot(r)` | arccot of real/complex `r`; principal value real part in (−π/2, π/2). Returns real if `r` is real. | MPFR/MPC. |
| `Arcsec(r)` | arcsec of real/complex `r`; principal value real part in [0, π/2) ∪ (π/2, π]. Returns real if `r` is real. | MPFR/MPC. |
| `Arccosec(r)` | arccosec of real/complex `r`; principal value real part in [−π/2, 0) ∪ (0, π/2]. Returns real if `r` is real. | MPFR/MPC. |

### 25.5.4 Hyperbolic Functions

Defined via sinh(z) = (eᶻ − e⁻ᶻ)/2, cosh(z) = (eᶻ + e⁻ᶻ)/2.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Sinh(f)` / `Sinh(s)` | Hyperbolic sine of series `f` or real/complex `s`. | MPFR/MPC. |
| `Cosh(f)` / `Cosh(r)` | Hyperbolic cosine of series `f` or real/complex `r`. | MPFR/MPC. |
| `Tanh(f)` / `Tanh(r)` | Hyperbolic tangent sinh(r)/cosh(r). | MPFR/MPC. |
| `Coth(r)` | Hyperbolic cotangent cosh(r)/sinh(r). | MPFR/MPC. |
| `Sech(r)` | Hyperbolic secant 1/cosh(r). | MPFR/MPC. |
| `Cosech(r)` | Hyperbolic cosecant 1/sinh(r). | MPFR/MPC. |

### 25.5.5 Inverse Hyperbolic Functions

Principal values as specified; real/complex arguments throughout.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Argsinh(f)` / `Argsinh(r)` | Inverse hyperbolic sine of series `f` or real/complex `r`; principal value imaginary part in [−π/2, π/2]. Returns real if `r` is real. | MPFR/MPC. |
| `Argcosh(f)` / `Argcosh(r)` | Inverse hyperbolic cosine of series `f` or real/complex `r`; principal value imaginary part in [0, π]. Returns real if `r` is real and r ≥ 1. | MPFR/MPC. |
| `Argtanh(f)` / `Argtanh(s)` | Inverse hyperbolic tangent of series `f` or real/complex `s`; principal value imaginary part in [−π/2, π/2]. Returns real if `s` is real with −1 < s < 1. | MPFR/MPC. |
| `Argsech(s)` | Inverse hyperbolic secant of `s`; principal value imaginary part in [0, π]. Returns real if `s` is real and |s| ≥ 1. | MPFR/MPC. |
| `Argcosech(s)` | Inverse hyperbolic cosecant of `s`; principal value imaginary part in [−π/2, π/2]. Returns real if `s` is real. | MPFR/MPC. |
| `Argcoth(s)` | Inverse hyperbolic cotangent of `s`; principal value imaginary part in [−π/2, π/2]. Returns real if `s` is real and 0 < s ≤ 1 (returns free real). | MPFR/MPC. |

---

## 25.6 Elliptic and Modular Functions

General references: Chandrasekharan [Cha85] for elliptic functions; Koblitz [Kob84] for
modular functions.

### 25.6.1 Eisenstein Series

The Eisenstein series are the coefficients of the Laurent expansion of the Weierstrass
℘-function:
℘(z, L) = 1/z² + Σₖ≥₂ Gₖ(L)(2k−1)z^{2k−2}.
The normalized series E₂ₙ(z) = (1/(2ζ(2n))) G₂ₙ(z) has rational q-expansion (q = e^{2πiz}).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Eisenstein(k, z)` | For positive even integer k = 2n and complex power series `z` with positive valuation, return the q-expansion of E₂ₙ. Parameter `Precision` (RngIntElt, default: precision of the parent field of `z`). | q-expansion via the modular definition. |
| `Eisenstein(k, t)` | For positive even integer k = 2n and point `t` in the upper half-plane, return the value of E₂ₙ at `t`. | Direct evaluation. |
| `Eisenstein(k, L)` | For positive even integer k = 2n and lattice `L = [a, b]` in C, return the value of E₂ₙ relative to `L`. | Lattice evaluation. |
| `Eisenstein(k, F)` | For positive even integer k = 2n and binary quadratic form F = ax² + bxy + cy², return E₂ₙ at τ = (−b + √(b²−4ac))/(2a). | CM point evaluation. |

*Worked examples: H25E7 (q-expansion of E₄ to O(q²⁰); evaluation at z₁ = 2.5 + i both via series and via direct `Eisenstein(4, t)`).*

### 25.6.2 Weierstrass Series

The normalized Weierstrass ℘-function q-expansion: WeierstrassSeries(z, q) = (2πi)⁻² ℘(q, z/(2πi)).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `WeierstrassSeries(z, q)` | For complex power series `z` with positive valuation and power series `q`: q-expansion of the Weierstrass ℘-function. Each term is an Eisenstein series to precision `Precision` (default: precision of `q`). | Eisenstein series terms. |
| `WeierstrassSeries(z, t)` | For series `z` and upper-half-plane point `t = τ`: q-expansion at q = e^{2πiτ}. | Eisenstein terms evaluated at τ. |
| `WeierstrassSeries(z, L)` | For series `z` and lattice `L = [a, b]` in C: q-expansion relative to `L`. | Lattice Eisenstein. |
| `WeierstrassSeries(z, F)` | For series `z` and binary quadratic form F = ax² + bxy + cy²: q-expansion at τ = (−b + √(b²−4ac))/(2a). | CM point Eisenstein. |

### 25.6.3 The Jacobi θ and Dedekind η-functions

The first Jacobi theta function: θ(q, z) = (1/i) Σₙ₌₋∞^∞ (−1)ⁿ q^{(n+1/2)²} e^{(2n+1)iz}.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `JacobiTheta(q, z)` (series z) | For real or complex `q` with |q| < 1, return θ(q, z) as a power series in `z` over C. | Pari. |
| `JacobiTheta(q, z)` (numeric z) | For real or complex `q`, `z` with |q| < 1, return the value θ(q, z). | Pari. |
| `JacobiThetaNullK(q, k)` | For integer k ≥ 0, return the k-th derivative θ^{(k)}(q, 0) of θ(q, z) at z = 0. | Pari. |
| `DedekindEta(z)` (series) | For complex power series `z` with positive valuation, return the q-expansion of Dedekind's η-function (unnormalized; factor q^{1/24} not removed). See Lang [Lan87]. | q-series for η. |
| `DedekindEta(s)` (numeric) | For complex `s` with positive imaginary part, return η(s) = e^{2πis/24} (1 + Σₙ₌₁^∞ (−1)ⁿ(q^{n(3n−1)/2} + q^{n(3n+1)/2})) where q = e^{2πis}. | Pari / Euler product. |

### 25.6.4 The j-invariant and the Discriminant

The discriminant: Δ(τ) = q Π_{n=1}^∞ (1 − qⁿ)²⁴ (q = e^{2πiτ}).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `jInvariant(q)` (series) | For power series `q` over a real or complex field with positive valuation, return the q-expansion of the elliptic j-invariant: j(q) = q⁻¹ + 744 + 196884q + …, with j(q) = E₄(q)³/Δ(q). | j = E₄³/Δ. |
| `jInvariant(s)` (numeric) | For complex `s` with positive imaginary part, return the value of j(s) (weight-0 modular function, Fourier expansion j(s) = e^{−2πis} + 744 + …). | Direct evaluation. |
| `jInvariant(L)` | For lattice L = [a, b] in C, return j of τ = a/b or b/a (whichever lies in the upper half-plane). | Reduction to τ. |
| `jInvariant(F)` | For binary quadratic form F = ax² + bxy + cy² with negative discriminant, return j at τ = (−b + √(b²−4ac))/(2a). | CM point. |
| `Delta(z)` (series) | For complex power series `z`, return the q-expansion of the discriminant Δ(z). | η-product formula. |
| `Delta(t)` (upper-half-plane) | For `t` in the upper half-plane, return the q-expansion of Δ evaluated at q = e^{2πit}. | η-product. |
| `Delta(L)` | For lattice L = [a, b], return q-expansion of Δ at q = e^{2πiτ} where τ = a/b or b/a in the upper half-plane. | η-product. |

### 25.6.5 Weber's Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `WeberF(s)` | For complex `s` in the upper half-plane, return Weber's f-function, satisfying j(s) = (f(s)²⁴ − 16)³/f(s)²⁴. | η-quotient definition. |
| `WeberF2(g)` (series) | For complex power series `g` with positive valuation, return the q-expansion of Weber's f₂ function f₂(x) = √2 η(2x)/η(x), satisfying j(s) = (f₂(s)²⁴ + 16)³/f₂(s)²⁴. | η-quotient. |
| `WeberF1(s)` / `WeberF2(s)` (numeric) | For complex `s` in the upper half-plane, return Weber's f₁ and f₂, satisfying j(s) = (f_{1/2}(s)²⁴ + 16)³/f_{1/2}(s)²⁴. Here f₁(x) = f₂(−1/x) = η(x/2)/η(x). | η-quotient. |

*Worked examples: H25E8 (q-expansion of f₂(z) to O(q⁷)).*

---

## 25.7 Theta Functions

The multivariable theta function with characteristic c ∈ R^{2g}: θ[c](z, τ) = Σ_{m ∈ Zᵍ} exp(πi ᵗ(m+c′)τ(m+c′) + 2πi ᵗ(m+c′)(z+c″)), where c′ is the first g entries of c and c″ the last g.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Theta(char, z, tau)` | Multidimensional theta function with characteristic `char` (2g×1 matrix) at `z` (g×1 matrix) and symmetric g×g matrix `tau` with positive-definite imaginary part. | Direct series summation. |
| `Theta(char, z, A)` | As above but `tau` is the small period matrix of analytic Jacobian `A`. Caches theta null values (z = 0) at half-integer characteristics. | Direct summation with caching. |

---

## 25.8 Gamma, Bessel and Associated Functions

General reference: Whittaker and Watson [WW15].

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Gamma(f)` | Gamma function Γ(f) for a power series `f` over a real or complex field (valuation 0, constant term 1). | MPFR (series). |
| `Gamma(r)` | For real or complex `s` (not 0, −1, −2, …), compute Γ(s). For Re(s) > 0: Γ(s) = ∫₀^∞ u^{s−1} e^{−u} du. Extended by analytic continuation; satisfies Γ(s)Γ(1−s) = π/sin(πs) and Γ(s+1) = sΓ(s). | MPFR/MPC. |
| `Gamma(r, s)` | Incomplete gamma function γ(s, t) = ∫₀ᵗ u^{s−1} e^{−u} du for real s, t. Parameter `Complementary` (BoolElt, default `false`): if true, returns ∫ₜ^∞ u^{s−1} e^{−u} du. Parameter `Gamma` (FldReElt): if supplied, the value of Γ(s) for efficiency. | Pari. |
| `GammaD(s)` | For free real `s` (s + 1/2 not a non-positive integer), return Γ(s + 1/2) using Legendre's doubling formula Γ(s+1/2) = 2^{1−2s}√π Γ(2s)/Γ(s) for integer s (faster than `Gamma(s+1/2)`). | Pari (Legendre doubling formula). |
| `LogGamma(f)` | Log-Gamma series log(Γ(f)) for series `f` (valuation 0, constant term 1) over a real or complex field. | MPFR (series). |
| `LogGamma(r)` | Principal branch of log(Γ(s)) for real or complex `s` (not a non-positive integer). | MPFR/MPC. |
| `LogDerivative(s)` / `Psi(s)` | Digamma function Ψ(s) = d(log Γ(s))/ds = Γ′(s)/Γ(s) for real or complex `s` (not a non-positive integer); has expansion Ψ(s) = −γ − 1/s + s Σₙ₌₁^∞ 1/(n(s+n)). | Pari. |
| `BesselFunction(n, r)` | Bessel function of the first kind Jₙ(r) for small integer `n` and real `r`. J₋ₙ(r) = Jₙ(−r) = (−1)ⁿJₙ(r). Defined by contour integral; satisfies Jₙ(x) = Σₖ₌₀^∞ (−1)ᵏ z^{n+2k}/(2^{n+2k} k! Γ(n+k+1)). | MPFR/Pari. |
| `BesselFunctionSecondKind(n, r)` | Bessel function of the second kind Yₙ(r) for small integer `n` and real `r`. Y₋ₙ(r) = −(−1)ⁿ Yₙ(r); Yₙ(−r) is not real. Satisfies the Bessel differential equation. | MPFR/Pari. |
| `JBessel(n, s)` | Bessel function of the first kind with half-integral index n+1/2, i.e. J_{n+1/2}(s). | Pari. |
| `KBessel(n, s)` / `KBessel2(n, s)` | Modified Bessel function of the second kind Kₙ(s) for complex `n` and positive real `s`; Kₙ(s) = (π/2)(iⁿJ₋ₙ(is) − i⁻ⁿJₙ(s))cot(nπ). `KBessel2` is an alternative (often faster) implementation. | Pari. |

---

## 25.9 The Hypergeometric Function

Reference: Husemöller [Hus87], page 176.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HypergeometricSeries(a, b, c, z)` | Return the hypergeometric series F(a, b, c; z) = Σ_{n≥0} (a)ₙ(b)ₙ / (n! (c)ₙ) zⁿ, where (a)ₙ = a(a+1)···(a+n−1). | Direct power series summation. |
| `HypergeometricU(a, b, s)` | For positive real `s` and complex `a`, `b`: the confluent hypergeometric function U(a, b, s) = (1/Γ(a)) ∫₀^∞ e^{−su} u^{a−1} (1+u)^{b−a−1} du. | Pari. |

---

## 25.10 Other Special Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ArithmeticGeometricMean(x, y)` / `AGM(f, g)` (series) | Hyperbolic arithmetic-geometric mean of power series `f` and `g` over a field (valuations must be equal). | AGM iteration (series). |
| `ArithmeticGeometricMean(x, y)` / `AGM(x, y)` (numeric) | Arithmetic-geometric mean of real or complex numbers `x`, `y`: limit of xᵢ, yᵢ with x₀ = x, y₀ = y, xᵢ₊₁ = (xᵢ+yᵢ)/2, yᵢ₊₁ = √(xᵢyᵢ). Iterates until within precision. | AGM iteration. |
| `BernoulliNumber(n)` | For non-negative integer `n`, return the exact n-th Bernoulli number Bₙ, defined by t/(eᵗ−1) = Σₙ₌₀^∞ Bₙtⁿ/n!. | Exact integer/rational computation. |
| `BernoulliApproximation(n)` | For non-negative integer `n`, return a real approximation to Bₙ. | Floating-point approximation. |
| `DawsonIntegral(r)` | For real `r`, compute Dawson's integral e^{−x²} ∫₀ˣ e^{u²} du at x = r. | mp real package. |
| `ErrorFunction(r)` / `Erf(r)` | Error function erf(r) = √(4/π) ∫₀ˣ e^{−u²} du; odd function with erf(0) = 0. | MPFR. |
| `ComplementaryErrorFunction(r)` / `Erfc(r)` | Complementary error function erfc(r) = 1 − erf(r). | MPFR. |
| `ExponentialIntegral(r)` | Principal value of ∫_{−∞}^x eᵘ/u du at x = r. | MPFR/Pari. |
| `ExponentialIntegralE1(r)` | Principal value of ∫ₓ^∞ eᵘ/u du at x = r (E₁ function). | MPFR/Pari. |
| `LogIntegral(r)` | Logarithmic integral li(r) = principal value of ∫₀ˣ 1/log(u) du at x = r (r ≥ 0, r ≠ 1). | mp real package. |
| `ZetaFunction(s)` | For real or complex `s ≠ 1`, return the value of Riemann's ζ(s) (analytic continuation of Σ i⁻ˢ, Re(s) > 1). | MPFR algorithm of Pétermann and Rémy **[PR06]**. |
| `ZetaFunction(R, n)` | For real field `R` and integer `n ≠ 1`, return ζ(n) in `R`. | MPFR **[PR06]**. |

---

## 25.11 Numerical Functions

Numerical analysis utilities taken from Pari.

### 25.11.1 Summation of Infinite Series

A sum is specified as a map `m` from the integers to R (or C) where `m(n)` is the n-th
term; summation starts at term `i`. Result precision equals the default precision of the
real field.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `InfiniteSum(m, i)` | Approximation to m(i) + m(i+1) + m(i+2) + … Also works for maps to C. | Pari general series summation. |
| `PositiveSum(m, i)` | Approximation to Σ m(i+k) for series with all positive terms. Uses van Wijngaarden's transformation to convert to an alternating series. Terms equal to 0 cause problems and should be removed. | van Wijngaarden's trick; Pari. |
| `AlternatingSum(m, i)` | Approximation to Σ m(i+k) for alternating-sign series. Parameter `Al` (MonStgElt, default `"Villegas"`): `"Villegas"` or `"EulerVanWijngaarden"`. Terms equal to 0 should be removed. | Villegas or Euler–van Wijngaarden algorithm; Pari. |

### 25.11.2 Integration

Romberg-like methods from Pari. Precision should not be too large; singularities (including
at the boundary) are not allowed.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Interpolation(P, V, x)` | Given points `P` and values `V`, interpolate `x` under a polynomial p with p(P[i]) = V[i]; returns the interpolated value and an error estimate. | Neville's algorithm. |
| `RombergQuadrature(f, a, b: parameters)` | Approximate ∫ₐᵇ f dx using Romberg's method of order 2K. Parameters: `Precision` (FldReElt, default 1.0e−6), `MaxSteps` (RngIntElt, default 20), `K` (RngIntElt, default 5, the Romberg order). Halts after `MaxSteps` iterations if precision not achieved. | Romberg's method; Pari. |
| `SimpsonQuadrature(f, a, b, n)` | Approximate ∫ₐᵇ f dx using Simpson's rule on `n` sub-intervals. | Simpson's rule; Pari. |
| `TrapezoidalQuadrature(f, a, b, n)` | Approximate ∫ₐᵇ f dx using the trapezoidal rule on `n` sub-intervals. | Trapezoidal rule; Pari. |

### 25.11.3 Numerical Derivatives

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `NumericalDerivative(f, n, z)` | For a sufficiently smooth function `f`, compute a numerical approximation to the n-th derivative of `f` at point `z`. | Interpolation + Taylor expansion. |

*Worked examples: H25E9 (10th derivative of e^{2x} at x=1; 1st derivative of LogGamma at x=3, verified against Psi(3.0)).*

---

## 25.12 Bibliography

| Key | Reference |
|-----|-----------|
| **[Cha85]** | K. Chandrasekharan. *Elliptic Functions*, volume 281 of Grundlehren der mathematischen Wissenschaften. Springer, Berlin, 1985. |
| **[Hus87]** | Dale Husemöller. *Elliptic Curves*, volume 111 of Graduate Texts in Mathematics. Springer, New York, 1987. |
| **[Kob84]** | Neal Koblitz. *Introduction to Elliptic Curves and Modular Forms*, volume 97 of Graduate Texts in Mathematics. Springer, New York, 1984. |
| **[Lan87]** | Serge Lang. *Elliptic Functions*, volume 112 of Graduate Texts in Mathematics. Springer, New York, 1987. |
| **[Lew81]** | Leonard Lewin. *Polylogarithms and Associated Functions*. North Holland, New York, 1981. |
| **[PR06]** | Y.-F. S. Pétermann and Jean-Luc Rémy. *Arbitrary Precision Error Analysis for Computing ζ(s) with the Cohen-Olivier Algorithm: Complete Description of the Real Case and Preliminary Report on the General Case.* Research Report 5852, INRIA, 2006. URL: http://www.inria.fr/rrrt/rr-5852.html. |
| **[Sch82]** | A. Schönhage. *The fundamental theorem of algebra in terms of computational complexity.* Technical report, Univ. Tübingen, 1982. |
| **[vdGOS91]** | G. van der Geer, F. Oort, and J. Steenbrink, editors. *Arithmetic Algebraic Geometry*, volume 89 of Progress in Mathematics, Basel, 1991. Birkhäuser Verlag. |
| **[WW15]** | E. T. Whittaker and G. N. Watson. *A Course of Modern Analysis*. Cambridge University Press, Cambridge, 2nd edition, 1915. |
| **[Zag91]** | Don Zagier. *Polylogarithms, Dedekind Zeta Functions, and the Algebraic K-Theory of Fields.* In van der Geer et al. [vdGOS91], pages 377–390. |

---

## Algorithm-to-function quick reference

| Algorithm / library | Functions |
|--------------------|-----------|
| MPFR (arbitrary-precision real, correct rounding) | All real-field arithmetic, `Exp`, `Log`, `Sin`, `Cos`, `Sincos`, `Tan`, `Cot`, `Sec`, `Cosec`, inverse trig, hyperbolic and inverse hyperbolic functions, `Gamma`, `LogGamma`, `BesselFunction`, `BesselFunctionSecondKind`, `Sqrt`, `Root`, `AbsoluteValue`, `Catalan`, `EulerGamma`, `Pi`, `ErrorFunction`, `Erfc` |
| MPC (arbitrary-precision complex, built on MPFR) | All complex-field counterparts of the above, `ComplexConjugate`, `Modulus`, `Argument` |
| Pari (fallback for missing MPFR/MPC coverage) | `Dilog`, `Polylog(m, s)`, `PolylogD`, `PolylogDold`, `PolylogP`, `LinearRelation`, `AllLinearRelations`, `PowerRelation`, `BestApproximation`, `ContinuedFraction`, `GammaD`, `LogDerivative`/`Psi`, `JBessel`, `KBessel`, `KBessel2`, `HypergeometricU`, `Gamma(r, s)`, `JacobiTheta`, `JacobiThetaNullK`, `DedekindEta`, complex-polynomial `Roots`, `InfiniteSum`, `PositiveSum`, `AlternatingSum`, `RombergQuadrature`, `SimpsonQuadrature`, `TrapezoidalQuadrature`, `Interpolation` |
| Schönhage splitting-circle algorithm **[Sch82]** | `Roots` (default `Al := "Schonhage"`), `RootsNonExact` |
| Newton iteration / Hensel lifting | `HenselLift`, `Root` |
| Neville's algorithm (polynomial interpolation) | `Interpolation` |
| Romberg's method | `RombergQuadrature` |
| van Wijngaarden's transformation | `PositiveSum` |
| Villegas / Euler–van Wijngaarden | `AlternatingSum` |
| LLL / Hastad–Lagarias–Schnorr variant | `LinearRelation`, `AllLinearRelations`, `PowerRelation` |
| Pétermann–Rémy (zeta function) **[PR06]** | `ZetaFunction` |
| Legendre doubling formula | `GammaD` |
| AGM iteration | `ArithmeticGeometricMean` / `AGM` |
| q-expansion / modular forms | `Eisenstein`, `WeierstrassSeries`, `jInvariant`, `Delta`, `WeberF`, `WeberF1`, `WeberF2`, `DedekindEta` |
| Multivariable theta summation | `Theta` |
