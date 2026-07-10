# Chapter 49 — Power, Laurent and Puiseux Series

**Handbook part:** VII — Local Arithmetic Fields
**Handbook pages:** 1321–1346 (PDF pages 1452–1479)

---

## Scope and overview

Chapter 49 describes the operations on formal power, Laurent and Puiseux series available
in Magma. Internally Magma has only one kind of formal series (with general fractional
exponents), but externally three kinds are distinguished: **power series** (non-negative
integral valuation and integral exponents, ring R[[x]], category `RngSerPow`/`RngSerPowElt`),
**Laurent series** (integral valuation and integral exponents, possibly negative, ring R((x)),
category `RngSerLaur`/`RngSerLaurElt`), and **Puiseux series** (rational valuation and
rational exponents, ring R⟨⟨x⟩⟩, category `RngSerPuis`/`RngSerPuisElt`). The three kinds
exist primarily for error-checking; a user who does not want exponent restrictions may use
Puiseux series throughout with no loss of efficiency or functionality.

Formal series are stored in **truncated form** as approximations
`c_v x^v + c_{v+1} x^{v+1} + ... + O(x^p)` to a given absolute precision p ≥ v. Because
series arithmetic does not involve carries, coefficient errors do not propagate: every known
coefficient (below the precision threshold) is exact. Puiseux series are stored internally
with a single exponent denominator d per series, but different Puiseux series may have
different denominators and may be freely mixed.

Two precision models are supported. In a **free precision** ring (the default) elements
carry individual precisions; operations preserve the maximum precision the inputs allow;
the ring has a configurable default precision (20 by default) used when a result must be
truncated. In a **fixed precision** ring elements all share the same fixed precision p, giving
R[[x]] ≅ R[x]/x^p behaviour for power series, and fixed relative precision behaviour for
Laurent/Puiseux rings.

**Equality** is strict by default (`eq` returns true only for identical series). A separate
notion of **weak equality** (coefficients agree wherever both are defined) is provided
through `IsWeaklyEqual`; weak equality is not transitive. Extensions of series rings
(unramified or totally ramified) are available over finite fields and support the full
arithmetic, factorization, and precision machinery.

---

## 49.1 Introduction

### 49.1.1 Kinds of Series

Power series must have non-negative integral valuation and integral exponents. Laurent
series must have integral valuation (possibly negative) and integral exponents. Puiseux
series may have rational valuation and rational exponents. All three share the same
internal representation; the distinctions exist solely for user-level error checking.

### 49.1.2 Puiseux Series

Each Puiseux series is internally stored as a Laurent series in y = x^{1/d} for a minimal
positive integer d (the **exponent denominator**). Different Puiseux series in the same ring
may have different denominators and may be mixed freely in arithmetic.

### 49.1.3 Representation of Series

Series are stored as truncated approximations. The O(x^p) notation denotes all terms of
degree ≥ p. For a Puiseux series with valuation w/d and precision q/d, the internal form is
`c_w x^{w/d} + c_{w+1} x^{(w+1)/d} + ... + O(x^{q/d})` with d minimal.

### 49.1.4 Precision

Two precision measures: **absolute precision** p (largest known exponent) and **relative
precision** p − v (number of known coefficient slots starting from valuation v). They satisfy
p = v + (relative precision).

### 49.1.5 Free and Fixed Precision

Free precision: elements carry individual precisions; default precision 20 used when
truncation is forced by the operation. Fixed precision: all elements share the same p
(absolute for power series, relative for Laurent/Puiseux series); zero is stored as exact
zero with infinite absolute precision even in fixed-precision Laurent/Puiseux rings.

### 49.1.6 Equality

`eq` returns true only for identical series (same valuation, precision, and every
coefficient). **Weak equality**: f and g are weakly equal if f − g is weakly zero (i.e., of
the form O(x^p) for some p or exactly zero). Weak equality is not transitive.

### 49.1.7 Polynomials over Series Rings

Polynomials over series rings are discussed in Chapter 46 and Section 49.7.

---

## 49.2 Creation Functions

### 49.2.1 Creation of Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PowerSeriesRing(R)` / `PowerSeriesRing(R, p)` | Create the free power series ring R[[x]] over commutative ring R, or (with integer `p`) a fixed-precision power series ring with absolute precision p. Optional parameter `Precision` (default 20) sets the default precision of the free ring. Optional Boolean parameter `Global` (default `true`) controls whether a global (unique per R) or non-global ring is returned. Angle-bracket notation: `S<x> := PowerSeriesRing(R)`. | — |
| `LaurentSeriesRing(R)` / `LaurentSeriesRing(R, p)` | Create the free Laurent series ring R((x)) over commutative ring R, or (with integer `p`) a fixed-precision Laurent series ring. Parameters `Global` and `Precision` as above. Angle-bracket notation: `S<x> := LaurentSeriesRing(R)`. | — |
| `PuiseuxSeriesRing(R)` / `PuiseuxSeriesRing(R, p)` | Create the free Puiseux series ring R⟨⟨x⟩⟩ over commutative ring R, or (with integer `p`) a fixed-precision Puiseux series ring. Parameters `Global` and `Precision` as above. Angle-bracket notation: `S<x> := PuiseuxSeriesRing(R)`. | — |

*Worked examples: H49E1 (global vs. non-global rings; automatic coercion between rings in x, a, b over Q).*

### 49.2.2 Special Options

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AssertAttribute(S, "DefaultPrecision", n)` | Procedure: change the default precision of free series ring S to the non-negative integer n. | — |
| `HasAttribute(S, "DefaultPrecision")` | Returns a Boolean (always true for free series rings) and the current default precision (integer, default 20). | — |
| `AssignNames(~S, ["x"])` | Procedure: change the printed name of the indeterminate of series ring S to the string `x`. Does not assign to the identifier `x`. | — |
| `Name(S, 1)` / `S . 1` | Return the indeterminate (generating transcendental element) of the series ring S. | — |

### 49.2.3 Creation of Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `R . 1` / `UniformizingElement(R)` | Return the generator (indeterminate) of the series ring R. | — |
| `elt< R \| v, [a1, ..., ad], p >` | Create the series `a1*x^v + ... + ad*x^{v+d-1} + O(x^{v+p})` in ring R with valuation v, coefficients `[a1,...,ad]`, and relative precision p. If p = −1, returns the exact (infinite-precision) series. Either v or p may be omitted (v defaults to 0; p defaults to v + d). | — |
| `R ! s` | Coerce s into series ring R. If s is a sequence `[a1,...,ad]`, creates `a1 + a2*x + ... + ad*x^{d-1} + O(x^d)`. | — |
| `BigO(f)` / `O(f)` | Create the series O(x^v) where v is the valuation of f and x is the generator of the parent of f. Typical use: `O(x^n)`. | — |
| `One(Q)` / `Identity(Q)` | The multiplicative identity of the series ring Q. | — |
| `Zero(Q)` / `Representative(Q)` | The zero element (or a representative) of the series ring Q. | — |

---

## 49.3 Structure Operations

### 49.3.1 Related Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Parent(R)` / `Category(R)` | Parent and category of the series ring R. | — |
| `BaseRing(R)` / `CoefficientRing(R)` | The coefficient ring of the series ring R. | — |
| `IntegerRing(R)` / `Integers(R)` / `RingOfIntegers(R)` | Return the power series ring that is the integer ring of the Laurent series ring R. | — |
| `FieldOfFractions(R)` | Return the Laurent series ring that is the field of fractions of the series ring R. | — |
| `ChangePrecision(R, r)` / `ChangePrecision(~R, r)` | Return (or mutate in place) a series ring identical to R but with precision r. | — |
| `ChangeRing(R, C)` | Return the series ring identical to R but with coefficient ring C. | — |
| `ResidueClassField(R)` | Return the residue class field of R (same as the coefficient ring) and the map from R into it. | — |

### 49.3.2 Invariants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Characteristic(R)` | The characteristic of the series ring R. | — |
| `Precision(R)` / `GetPrecision(R)` | Return the precision of the fixed-precision series ring R. For fixed-precision power series rings this is the fixed absolute precision; for fixed-precision Laurent/Puiseux rings this is the maximum relative precision. | — |

### 49.3.3 Ring Predicates and Booleans

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsCommutative(Q)` / `IsUnitary(Q)` | Boolean predicates on the ring Q. | — |
| `IsFinite(Q)` / `IsOrdered(Q)` | Boolean predicates on the ring Q. | — |
| `IsField(Q)` / `IsEuclideanDomain(Q)` | Boolean predicates on the ring Q. | — |
| `IsPID(Q)` / `IsUFD(Q)` | Boolean predicates on the ring Q. | — |
| `IsDivisionRing(Q)` / `IsEuclideanRing(Q)` | Boolean predicates on the ring Q. | — |
| `IsPrincipalIdealRing(Q)` / `IsDomain(Q)` | Boolean predicates on the ring Q. | — |
| `R eq S` / `R ne S` | Ring equality/inequality. | — |

---

## 49.4 Basic Element Operations

### 49.4.1 Parent and Category

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Parent(r)` / `Category(r)` | Parent ring and category of the series element r. | — |

### 49.4.2 Arithmetic Operators

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `+b` / `-b` | Unary plus/minus. | — |
| `a + b` / `a - b` / `a * b` / `a ^ k` | Binary addition, subtraction, multiplication, and power. Precision of the result is the maximum compatible with the inputs. | — |
| `a div b` / `a / b` | Euclidean quotient / division. If b has infinite precision (e.g., a unit polynomial), the result precision equals the default precision. | — |

### 49.4.3 Equality and Membership

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `a eq b` / `a ne b` | Strict equality/inequality (identical valuation, precision, and all coefficients). | — |
| `a in R` / `a notin R` | Membership in series ring R. | — |

### 49.4.4 Predicates on Ring Elements

Note: `eq`, `IsZero`, `IsOne`, `IsMinusOne` all use the strict (identical) equality convention.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsZero(a)` / `IsOne(a)` / `IsMinusOne(a)` | Test for exact zero, one, or minus-one. | — |
| `IsNilpotent(x)` / `IsIdempotent(x)` | Nilpotency/idempotency predicates. | — |
| `IsUnit(a)` / `IsZeroDivisor(x)` / `IsRegular(x)` | Unit, zero-divisor, and regular-element predicates. | — |
| `IsIrreducible(x)` / `IsPrime(x)` | Irreducibility and primality predicates. | — |
| `IsWeaklyZero(f)` | True iff f is exactly zero or of the form O(x^p) for some p; i.e., all known coefficients are zero. | — |
| `IsWeaklyEqual(f, g)` | True iff f − g is weakly zero (coefficients agree wherever both are defined). Not transitive. | — |
| `IsIdentical(f, g)` | True iff f and g have exactly the same valuation, precision, and coefficients. | — |

### 49.4.5 Precision

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AbsolutePrecision(f)` | The absolute precision p stored with f: the exponent of the first unknown term, i.e., the least p such that f ∈ O(x^p). Infinite if f is exact; a non-integral rational for Puiseux series. | — |
| `RelativePrecision(f)` | Number of known coefficients starting from the first non-zero term; equals AbsolutePrecision − Valuation. Infinite if f is exact; may be a non-integral rational for Puiseux series. | — |
| `ChangePrecision(f, r)` / `ChangePrecision(~f, r)` | The (non-Puiseux) series f with absolute precision r (which may be positive infinity). | — |

### 49.4.6 Coefficients and Degree

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Coefficients(f)` / `ElementToSequence(f)` / `Eltseq(f)` | Returns the sequence Q of coefficients of f, the unscaled valuation v, and the exponent denominator d (so the true valuation is v/d). The i-th entry Q[i] is the coefficient of x^{(v+i−1)/d}. | — |
| `Coefficient(f, i)` | The coefficient of x^i in f as an element of the coefficient ring. For Puiseux series i may be a non-integral rational; for power series i must be a non-negative integer. Must satisfy i < AbsolutePrecision(f). | — |
| `LeadingCoefficient(f)` | The first non-zero coefficient of f (coefficient of x^v where v is the valuation). | — |
| `LeadingTerm(f)` | The first non-zero term of f (the monomial x^v with its coefficient). | — |
| `Truncate(f)` | The exact series obtained by truncating f after the last known non-zero coefficient. | — |
| `ExponentDenominator(f)` | The lowest common denominator of all exponents of non-zero terms of f. Always 1 for power and Laurent series. | — |
| `Degree(f)` | The exponent of the last known non-zero term (degree of the truncation). May be a non-integral rational for Puiseux series. | — |
| `Valuation(f)` | The smallest exponent v such that the coefficient of x^v is not known to be zero. Infinite for exact zero. May be non-integral rational for Puiseux series. | — |

### 49.4.7 Evaluation and Derivative

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Derivative(f)` | The derivative of f with respect to its indeterminate; precision decreases by 1 unless f has infinite precision. | — |
| `Derivative(f, n)` | The n-th derivative of f; precision decreases by n unless f has infinite precision. | — |
| `Integral(f)` | An antiderivative F of f such that Derivative(F) = f. The coefficient of x^{−1} in f must be zero. Precision of F exceeds that of f by 1 (unless f has infinite precision). | — |
| `Evaluate(f, s)` | Value of f when the indeterminate is evaluated at s ∈ S; result is an element of the common overstructure of the coefficient ring and S. | — |
| `Laplace(f)` | The Laplace transform of f: if f = Σ_{i≥0} a_i x^i, returns Σ_{i≥0} (i! a_i) x^i. Valuation of f must be integral and non-negative. | — |

### 49.4.8 Square Root

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SquareRoot(f)` / `Sqrt(f)` | Square root of f. For power or Laurent series, f must have even valuation. | — |

### 49.4.9 Composition and Reversion

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Composition(f, g)` | The composition f ∘ g = Σ_{i<p} f_i (g^i), where f = Σ_{i<p} f_i x^i, for f and g in the same series ring. | — |
| `Reversion(f)` / `Reverse(f)` | The compositional inverse g of f: the series such that f ∘ g = x to best precision. For power/Laurent series, the valuation of f must be 1. For Puiseux series, valuation must be positive; if not equal to 1, the leading coefficient of f must be 1. | — |
| `Convolution(f, g)` | The convolution f * g = Σ_{i<min(p,q)} f_i g_i x^i, where f = Σ f_i x^i + O(x^p), g = Σ g_i x^i + O(x^q). | — |

*Worked examples: H49E2 (Composition and Reversion: verifying Arcsin is the reversion of Sin; reversion of a series of valuation 3; reversion of a proper Puiseux series with fractional valuation).*

---

## 49.5 Transcendental Functions

For all functions in this section: the precision of the result approximates the precision of
the argument (or the default precision if the argument has infinite precision). The
coefficient ring of the series must be a field. If the argument has a non-zero constant
term, the coefficient ring must be a real or complex domain so that the transcendental
function can be evaluated at the constant term. See also the chapter on real and complex
fields for elliptic and modular functions defined on formal series.

### 49.5.1 Exponential and Logarithmic Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Exp(f)` | Exponential of f, defined over a field. | Asymptotically fast series arithmetic (as noted in H49E3, the Bernoulli number B_{10000} was computed using this; see `BernoulliNumber`). |
| `Log(f)` | Natural logarithm of f, defined over a field of characteristic zero. The valuation of f must be zero. | — |

*Worked examples: H49E3 (computing Bernoulli numbers via the exponential generating function E(x) = x/(e^x − 1); Laplace transform to scale coefficients; B_{500} computed exactly. Notes that B_{10000} has been computed using asymptotically fast series division).*

### 49.5.2 Trigonometric Functions and their Inverses

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Sin(f)` | Sine of f over a field of characteristic zero; valuation of f must be zero. | — |
| `Cos(f)` | Cosine of f over a field of characteristic zero; valuation of f must be zero. | — |
| `Sincos(f)` | Returns both sine and cosine of f over a field of characteristic zero; valuation of f must be zero. | — |
| `Tan(f)` | Tangent of f over a field. | — |
| `Arcsin(f)` | Inverse sine of f over a field of characteristic zero. | — |
| `Arccos(f)` | Inverse cosine of f over the real or complex field. | — |
| `Arctan(f)` | Inverse tangent of f over a field of characteristic zero. | — |

### 49.5.3 Hyperbolic Functions and their Inverses

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Sinh(f)` | Hyperbolic sine of f over a field. | — |
| `Cosh(f)` | Hyperbolic cosine of f over a field. | — |
| `Tanh(f)` | Hyperbolic tangent of f over a field. | — |
| `Argsinh(f)` | Inverse hyperbolic sine of f over a field of characteristic zero. | — |
| `Argcosh(f)` | Inverse hyperbolic cosine of f over the real or complex field. | — |
| `Argtanh(f)` | Inverse hyperbolic tangent of f over a field of characteristic zero. | — |

---

## 49.6 The Hypergeometric Series

For more information on the hypergeometric series, see **[Hus87]**, page 176.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HypergeometricSeries(a, b, c, z)` | Returns the hypergeometric series F(a, b, c; z) = Σ_{n≥0} [(a)_n (b)_n / (n! (c)_n)] z^n, where (a)_n = a(a+1)···(a+n−1) is the Pochhammer symbol. | Direct term-by-term computation using the recurrence for Pochhammer symbols; see **[Hus87]**. |

---

## 49.7 Polynomials over Series Rings

Factorization is available for polynomials over series rings defined over finite fields. It
is recommended to construct polynomials from sequences rather than by addition of terms
(especially over fields) to avoid precision loss: `Polynomial([t, 0, 0, 0, 1])` preserves more
precision than constructing `x^4 + t` term-by-term.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HenselLift(f, L)` | Given a polynomial f over a series ring (or extension thereof) over a finite field, and a factorization L of f known to precision 1, return a factorization of f known to the full precision of the coefficient ring of f. | Hensel's lemma / lifting. |
| `Factorization(f)` | Factorization of polynomial f into irreducibles over a power series ring, Laurent series field over a finite field, or extension of either. Parameters: `Certificates` (default `false`; if `true`, return a two-element irreducibility certificate per factor), `Ideals` (default `false`; if `true`, include two ideal generators per factor in the certificate), `Extensions` (default `false`; if `true`, include an extension for each factor in the certificate). | — |

*Worked examples: H49E4 (factoring x^5 + t·x^4 − t^2·x^3 + (1+t^{20})·x^2 + t·x + t^6 over GF(101)[[t]]; demonstrating `Extensions` option with certificate records including F, Rho, E, Pi, IdealGen1, IdealGen2, Extension fields).*

---

## 49.8 Extensions of Series Rings

Extensions of series rings are either **unramified** or **totally ramified**. Only series
rings defined over finite fields can be extended. Polynomials should be constructed from
sequences (not by addition of terms) to ensure full precision, since extensions require
full-precision defining polynomials.

### 49.8.1 Constructions of Extensions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `UnramifiedExtension(R, f)` | Construct the unramified extension of R (a series ring or extension thereof) defined by the inertial polynomial f: adjoin a root of f to R. | — |
| `TotallyRamifiedExtension(R, f)` | Construct the totally ramified extension of R defined by the Eisenstein polynomial f: adjoin a root of f to R. Parameter `MaxPrecision` (default: precision of R): the maximum precision to which the coefficients of f are known. Setting it higher than the precision of R allows the result precision to be increased to deg(f) × MaxPrecision. The polynomial f may be given over a higher-precision ring to allow further precision increase. | — |
| `ChangePrecision(E, r)` / `ChangePrecision(~E, r)` | Return (or mutate) the extension E with precision r. Increasing the precision of a ramified extension requires that MaxPrecision was set at construction and r ≤ MaxPrecision × ramification degree, or that the defining polynomial was given to more precision than the coefficient ring. | — |
| `FieldOfFractions(E)` | The field of fractions of the extension E of a series ring. | — |

*Worked examples: H49E5 (two-step extension: UnramifiedExtension by x^2+2 over GF(101)[[t]], then TotallyRamifiedExtension by x^2+t·x+t; ChangePrecision to 200 and 1000; FieldOfFractions of both extensions).*

### 49.8.2 Operations on Extensions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Precision(E)` / `GetPrecision(E)` | The maximum precision elements of the extension E may have. | — |
| `CoefficientRing(E)` / `BaseRing(E)` | The ring over which extension E was defined. | — |
| `DefiningPolynomial(E)` | The polynomial used to define the extension E. | — |
| `InertiaDegree(E)` | The degree of E if E is an unramified extension, 1 otherwise. | — |
| `RamificationIndex(E)` / `RamificationDegree(E)` | The degree of E if E is a totally ramified extension, 1 otherwise. | — |
| `ResidueClassField(E)` | The residue class field E/πE and the map from E to it. | — |
| `UniformizingElement(E)` | A uniformizing element (element of valuation 1) for the extension E. | — |
| `IntegerRing(E)` / `Integers(E)` / `RingOfIntegers(E)` | The ring of integers of E when E is a field (i.e., an extension of a Laurent series ring). | — |
| `E1 eq E2` | Test equality of extensions E1 and E2. | — |
| `E . i` | The primitive element of E (the root of the defining polynomial adjoined to the coefficient ring). | — |
| `AssignNames(~E, S)` | Assign the string in sequence S as the name of the primitive element of E. | — |

*Worked examples: H49E6 (two-step extension of GF(53)((t)): UnramifiedExtension by x^3+3x^2+x+4, TotallyRamifiedExtension by y^4+t; querying Precision, CoefficientRing, DefiningPolynomial, InertiaDegree, RamificationDegree, ResidueClassField, UniformizingElement, Integers, and the primitive element E.1).*

### 49.8.3 Elements of Extensions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `x * y` / `x + y` / `x - y` / `-x` / `x ^ n` / `x div y` / `x / y` | Arithmetic on elements of extension rings. | — |
| `x eq y` / `IsZero(e)` / `IsOne(e)` / `IsMinusOne(e)` / `IsUnit(e)` | Equality and predicates on elements of extension rings. | — |
| `Valuation(e)` | Valuation of e: the index of the largest power of π dividing e. | — |
| `RelativePrecision(e)` | Number of π-adic digits of e that are known. | — |
| `AbsolutePrecision(e)` | Sum of the relative precision and the valuation of e. | — |
| `Coefficients(e)` / `Eltseq(e)` / `ElementToSequence(e)` | Coefficients of e with respect to powers of the uniformizing element of the extension. | — |

*Worked examples: H49E7 (arithmetic in a two-step extension of GF(101)[[t]]; Valuation of elements coerced into different extensions; RelativePrecision and AbsolutePrecision; Coefficients of u^7 and tt^8).*

### 49.8.4 Optimized Representation

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `OptimizedRepresentation(E)` / `OptimisedRepresentation(E)` | Returns an optimized representation for the unramified extension E of a series ring: a series ring over the residue class field of E, plus the map from E to this simpler ring. Requires that the defining polynomial of E is coercible into the residue class field. | — |

*Worked examples: H49E8 (UnramifiedExtension of GF(101)[[t]] by x^2+2x+3; OptimizedRepresentation gives a power series ring over GF(101^2) with the conversion map).*

---

## 49.9 Bibliography

| Key | Reference |
|-----|-----------|
| **[Hus87]** | Dale Husemöller. *Elliptic Curves*, volume 111 of Graduate Texts in Mathematics. Springer, New York, 1987. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Truncated formal series arithmetic (exact coefficients, no error propagation) | All arithmetic operators: `+`, `-`, `*`, `^`, `div`, `/`, `Composition`, `Convolution` |
| Compositional inversion | `Reversion`, `Reverse` |
| Asymptotically fast series division | `Exp`, `Log` (implicitly; noted in H49E3 context) |
| Laplace transform (coefficient scaling by i!) | `Laplace` |
| Hypergeometric series **[Hus87]** | `HypergeometricSeries` |
| Hensel lifting | `HenselLift`, `Factorization` |
| Unramified / totally ramified extensions over finite fields | `UnramifiedExtension`, `TotallyRamifiedExtension`, `OptimizedRepresentation` |
| Precision management (free vs. fixed) | `AssertAttribute`, `HasAttribute`, `ChangePrecision`, `AbsolutePrecision`, `RelativePrecision` |
