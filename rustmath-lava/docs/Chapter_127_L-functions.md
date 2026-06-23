# Chapter 127 — L-functions

**Handbook part:** XVI — Arithmetic Geometry
**Handbook pages:** 4245–4287 (PDF pages 4376–4423)

---

## Scope and overview

Chapter 127 covers Magma's machinery for constructing and numerically evaluating L-functions
arising in number theory and arithmetic geometry. The central objects are Dirichlet series
`L(s) = Σ aₙ/nˢ` that admit a meromorphic continuation to all of **C** and satisfy a functional
equation of the standard type involving a Γ-factor, a conductor, a weight, and a sign.

The chapter falls into four natural themes:

1. **Built-in L-series** — one-line constructors for the Riemann zeta function, Dedekind zeta
   functions of number fields, L-series of Artin representations, elliptic curves (over **Q** and
   number fields), hyperelliptic curves, Dirichlet characters, Hilbert modular forms, Hecke
   Grössencharacters, and modular forms.

2. **Evaluation** — computing values `L(s₀)`, derivatives `L⁽ᴰ⁾(s₀)`, Taylor expansions, and
   values of the completed function `L*(s₀)`. The computational engine follows **Dokchitser
   [Dok04]** and the Pari package ComputeL **[Dok02]**; see also Lavrik **[Lav67]**, Tollis
   **[Tol97]**, and the exposition in Cohen **[Coh00]**, §10.3.

3. **Arithmetic** — product, quotient, tensor product, and symmetric power of L-series, allowing
   construction of motivic L-functions from simpler pieces (following Serre's formalism of
   systems of ℓ-adic representations).

4. **General user-defined L-series** — a generic constructor for any L-function whose invariants
   (weight, gamma shifts, conductor, coefficient function, sign, poles and residues) are known.
   The user supplies these data; Magma assembles the functional equation and computes values.
   `CheckFunctionalEquation` should always be called to verify the setup.

A final section (**127.10**) provides auxiliary routines for Weil polynomials (characteristic
polynomials of Frobenius on étale cohomology), used in point-counting and Picard-rank
computations.

---

## 127.1 Overview

The section motivates the chapter with a one-line example (`LSeries(E)` followed by `Evaluate`)
and lists the coverage. It refers to Manin–Panchishkin **[Sha95]** Chapter 4, Serre **[Ser65]**,
and the collected volume **[JKS94]** for background.

*(No intrinsics defined in this section.)*

---

## 127.2 Built-in L-series

Every built-in constructor returns a variable of type `LSer`. The `LSer` object retains only
its arithmetic invariants and a reference to the original object (for printing); subsequent
operations (evaluation, arithmetic) are independent of the origin.

The `Method` parameter in `LSeries(K)` and `LSeries(E,K)` can be `"Artin"`, `"Direct"`, or
`"Default"`: the Artin method factors the Dedekind zeta via Artin representations (often smaller
conductors, faster evaluation), while the Direct method counts prime ideals.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RiemannZeta()` | Returns the Riemann ζ-function as an `LSer`. Parameter: `Precision` (digits of precision; default = precision of the default real field). | Functional equation method **[Dok04, Dok02]**; this is weight 1, gamma `[0]`, conductor 1, all `aₙ = 1`, sign 1, pole at s = 1 with residue −1. |
| `LSeries(K)` | Dedekind zeta function ζ(K, s) of a number field K, defined as `Σ Norm(I)⁻ˢ` over nonzero ideals I of the maximal order. Parameters: `Method` (`"Artin"`, `"Direct"`, or `"Default"`); `ClassNumberFormula` (bool, default `false`; if `true` computes the residue at s = 1 algebraically via Signature, Regulator, ClassGroup, TorsionSubgroup); `Precision`. | Direct method: approximate residue from functional equation (faster unless discriminant is small and precision very high). Artin method: factors ζ(K, s) as a product of Artin L-series **[Dok04, Dok02]**. |
| `LSeries(A)` | L-series of an Artin representation A (see Chapter 44). Parameter: `Precision`. | Functional equation method **[Dok04, Dok02]**; coefficients from character values of the Artin representation. |
| `LSeries(E)` | L-series L(E, s) of an elliptic curve E defined over **Q** or a number field. Parameter: `Precision`. Note: over general number fields, analytic continuation and functional equation are conjectural. Computation time grows as `√conductor`; for only the leading term at s = 1 over **Q**, use `AnalyticRank` or `ConjecturalRegulator` instead. | Functional equation method **[Dok04, Dok02]**; local factors from TraceOfFrobenius and RootNumber. |
| `LSeries(E, K)` | L-series L(E/K, s) for elliptic curve E/Q and number field K. Technically the tensor product of the ℓ-adic representations of E/Q and K/Q. Parameters: `Method` (same as for `LSeries(FldNum)`; "Direct" forced when E and K have simultaneous wild ramification at 2 or 3); `Precision`. | Tensor product of ℓ-adic representations **[Dok04]**; analytic continuation and functional equation conjectural. |
| `LSeries(E, A)` | Twisted L-series of elliptic curve E/Q by Artin representation A. Not allowed when E and A are simultaneously wildly ramified at 2 or 3. Parameter: `Precision`. | Twist via tensor product of ℓ-adic representation of E with the Artin representation; functional equation method **[Dok04, Dok02]**. |
| `LSeries(C)` | L-series of a hyperelliptic curve C defined over **Q**. Parameters: `Precision`; `ExcFactors` (sequence of `⟨prime, conductor_exponent, local_factor⟩` triples for bad primes where the Euler factor is known in advance). | Functional equation method **[Dok04, Dok02]**; local Euler factors computed from Jacobian at good primes. |
| `LSeries(Chi)` | Dirichlet L-series L(χ, s) = Σ χ(n)/nˢ for a primitive Dirichlet character χ : (Z/mZ)* → C*. Character values must lie in Z, Q, or a cyclotomic field. Parameter: `Precision`. See §19.8 for Dirichlet characters. | Functional equation method **[Dok04, Dok02]**. |
| `LSeries(hmf)` | L-series of a cuspidal Hilbert modular newform. (No additional parameters documented beyond the form itself.) | Functional equation method **[Dok04, Dok02]**; coefficients from the Hilbert modular form eigenvalues. |
| `LSeries(psi)` (two signatures) | L-series of a primitive Hecke (Grössencharacter) on ideals. See §34.9 for Hecke characters. Parameter: `Precision`. | Functional equation method **[Dok04, Dok02]**. |
| `LSeries(f)` | L-series L(f, s) = Σ aₙ/nˢ for a modular form f with q-expansion Σ aₙqⁿ. It is assumed (not checked) that L(f, s) satisfies the standard functional equation. Parameters: `Embedding` (map or user function embedding the coefficient ring into C; required when f is defined over a number field); `Precision`. | Functional equation method **[Dok04, Dok02]**. The user must ensure the functional equation holds; `CheckFunctionalEquation` should be called. |

*Worked examples: H127E1 (ζ(2) vs π²/6); H127E2 (Dedekind zeta of Q(i) at s=2); H127E3 (direct vs Artin method for Q(³√3) of degree 12); H127E4 (Serre–Armitage zero at central point, Artin factorisation, odd sign); H127E5 (Artin L-series of the two characters of Gal(Q(i)/Q)); H127E6 (6-dimensional Artin L-series for A₇ polynomial); H127E7 (E of conductor 43 over Q and over Q(i)); H127E8 (LSeries(E,K) for E over Q(√5), and over cyclotomic field Q(ζ₁₁)); H127E9 (twisted L-values for 11A3 and characters of Q(ζ₅)); H127E10 (2-dimensional quaternion-group Artin twist of 11a3); H127E11 (L-series of y² = x⁵+1); H127E12 (Dirichlet L-function for a character mod 37); H127E13 (Hilbert modular form over Q(√5)); H127E14 (modular form with coefficients in Z[i], two complex embeddings, Embedding parameter).*

---

## 127.3 Computing L-values

Once an `LSer` object has been constructed (from a built-in constructor, a user-defined L-series,
or arithmetic on L-series), Magma computes values, derivatives, and Taylor expansions using the
functional equation. All evaluation uses the functional equation even in the region of absolute
convergence of the Dirichlet series, which speeds convergence.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Evaluate(L, s0)` | Compute L(s₀) for complex s₀, or the Dth derivative L⁽ᴰ⁾(s₀) if `Derivative := D > 0`. If it is known that L(s₀) = … = L⁽ᴰ⁻¹⁾(s₀) = 0, setting `Leading := true` substantially reduces computation time (useful when determining the order of vanishing). | Functional equation / approximate functional equation method **[Dok04, Dok02]**; see also **[Lav67, Tol97, Coh00]** §10.3. |
| `CentralValue(L)` | For an L-function of even weight 2k (in Magma's sense), returns L(k). | As `Evaluate(L, k)`. |
| `LStar(L, s0)` | Compute L*(s₀) or its Dth derivative L*⁽ᴰ⁾(s₀) (parameter `Derivative := D`, default 0), where L*(s) = (conductor/πᵈ)^(s/2) γ(s) L(s) is the completed L-function satisfying L*(s) = sign · L̄*(weight − s). | Completed functional equation **[Dok04]**. |
| `LTaylor(L, s0, n)` | Compute the first n+1 terms of the Taylor expansion of L about s₀: L(s₀) + L′(s₀)x + L″(s₀)x²/2! + … + L⁽ⁿ⁾(s₀)xⁿ/n! + O(xⁿ⁺¹). Parameter: `ZeroBelow := k` asserts the first k terms vanish, reducing computation time. | Repeated differentiation of the functional equation integral **[Dok04]**. |

*Worked example: H127E15 (conductor-5077 rank-3 elliptic curve: successive derivatives at s=1 until non-zero; LTaylor with and without ZeroBelow; LStar derivative vs manual computation via chain rule).*

---

## 127.4 Arithmetic with L-series

Products and quotients require the L-series to have **weakly multiplicative** coefficients
(`amn = am·an` for gcd(m,n) = 1). The tensor product requires both L-series to arise from
systems of ℓ-adic representations.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `L1 * L2` | Product L(s) = L₁(s)·L₂(s) of two L-series of the same weight with weakly multiplicative coefficients. Parameters: `Poles` (list of poles of L₁*(s)L₂*(s) with Re ≥ weight/2); `Residues` (residues at those poles); `Precision`. | Coefficient-wise product; Euler factors multiply. Functional equation derived from those of L₁ and L₂. |
| `L1 / L2` | Quotient L(s) = L₁(s)/L₂(s), assuming the quotient is a genuine L-function with finitely many poles. (Not checked.) Parameters: `Poles`, `Residues`, `Precision`. | Coefficient-wise division of Euler factors; poles at zeros of L₂*(s) must be supplied by user. |
| `TensorProduct(L1, L2, ExcFactors)` / `TensorProduct(L1, L2)` / `TensorProduct(L1, L2, ExcFactors, K)` / `TensorProduct(L1, L2, K)` | Tensor product L(V₁⊗V₂, s) for L₁ = L(V₁, s) and L₂ = L(V₂, s) associated to ℓ-adic representations V₁, V₂ (à la Serre). Both must have integer conductor, weakly multiplicative coefficients, and an underlying Hodge structure (recovered from γ-shifts). `ExcFactors` is a list of `⟨p, v⟩` or `⟨p, v, Fₚ(x)⟩` tuples specifying conductor valuation and inverse local factor at primes of simultaneous bad reduction; if omitted for a prime, Magma assumes (V₁⊗V₂)^Iₚ = V₁^Iₚ ⊗ V₂^Iₚ. The sign cannot be derived from the signs of the factors and is computed numerically from the functional equation unless supplied via `Sign`. Optional field argument `K`: tensor product over K rather than Q. Parameters: `Precision`, `Sign`. | Euler-factor-by-Euler-factor tensor product; Hodge structure combinatorics for gamma factors **[Dok04]**. For motivic L-functions the standard functional equation is expected. |

*Worked examples: H127E24 (LSeries(E,K) reproduced via TensorProduct; conductor correction at p=2 where E acquires good reduction); H127E25 (non-abelian twist of Mordell curve; BSD prediction on rank); H127E26 (tensor product of two elliptic curves over Q); H127E27 (tensor product of two level-1 modular forms; Ramanujan congruence); H127E28 (tensor product related to Siegel modular forms, see [vGvS93]); H127E29 (tensor product over a quadratic field K).*

---

## 127.5 General L-series

### 127.5.1 Terminology

The section sets up the four assumptions that Magma requires of any L-function:

1. The defining Dirichlet series converges for Re(s) sufficiently large (coefficients grow at most polynomially).
2. L(s) has a meromorphic continuation to all of **C**.
3. There exist: a real positive **weight** w, a complex **sign** of absolute value 1, a real positive **conductor** N, and a **gamma factor** γ(s) = Γ((s+λ₁)/2)·…·Γ((s+λd)/2) (rational γ-shifts λ₁,…,λd, dimension d ≥ 1), such that L*(s) = (N/πd)^(s/2) γ(s) L(s) satisfies `L*(s) = sign · L̄*(w − s)`.
4. L*(s) has finitely many simple poles.

The weight parameter here equals w (so that s → w − s is the functional equation map), which
is the motivic weight plus 1. The **motivic L-functions** — L-functions attached to cohomology
groups of varieties over number fields — are the canonical expected example.

*(No intrinsics defined in this subsection.)*

### 127.5.2 Constructing a General L-Series

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LSeries(weight, gamma, conductor, cffun)` | General L-series constructor. Compulsory arguments: real positive `weight`; `gamma` (sequence of rationals: the γ-shifts λᵢ; e.g. `[0]` for ζ(s), `[0,1]` for an elliptic curve over Q, `[0,…,0,1,…,1]` with r₁+r₂ zeros and r₂ ones for Dedekind ζ of a field with r₁ real and r₂ complex places); real positive `conductor`; `cffun` specifying the Dirichlet coefficients (a finite sequence `[a₁,…,aₙ]`; a function `f(n)` returning aₙ; a function `f(p,d)` returning the inverse local factor at p up to degree d; or `0` meaning coefficients will be set later via `LSetCoefficients`). Optional parameters: `Sign` (complex number of absolute value 1, or 0 to compute numerically, or a function `s(p)` or `s(L,p)` computing it to precision p); `Poles` (sequence of poles z of L*(s) with Re(z) ≥ weight/2; default []); `Residues` (sequence of residues at those poles, same length as `Poles`, or [] to compute numerically); `Parent` (any Magma object, used only for printing); `CoefficientGrowth` (function `f(x)` or `f(L,x)` bounding \|aₙ\| ≤ f(n)); `Precision`; `ImS` (largest Im(s) for which L(s) will be evaluated); `Asymptotics` (bool, default `true`: use asymptotic expansions as well as Taylor series). | Functional equation method **[Dok04, Dok02]**; approximate functional equation via theta functions (Lavrik **[Lav67]**, Tollis **[Tol97]**, Cohen **[Coh00]** §10.3). |
| `CheckFunctionalEquation(L)` | Tests the functional equation numerically by evaluating the two theta functions at a real point t and subtracting; should return 0 to current precision. Computes sign and residues if not yet known. If sign was 0 (unknown), returns `\|Sign\| − 1`. Optional parameter `t` (real, 1.05 < t < 1.2, default 1.2). A value significantly far from 0 indicates an incorrect parameter (conductor, sign, etc.) or insufficient coefficients. | Approximate functional equation: evaluates `Σ aₙ Γ((weight−s+λᵢ)/2, n·t/√N)` versus the conjugate sum at `weight−s` **[Dok04]**. |

*Worked example: H127E16 (step-by-step construction of the odd quadratic Dirichlet character mod 3; shows wrong γ-shifts and wrong sign before correcting; demonstrates that LCfRequired tells how many coefficients to supply).*

### 127.5.3 Setting the Coefficients

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LSetCoefficients(L, cffun)` | Set or replace the coefficient function of an existing `LSer` object L. `cffun` may be: a finite sequence `[a₁,…,aₙ]`; a function `f(n)` returning aₙ; or a function `f(p,d)` returning the inverse local factor Fₚ(x) at prime p up to degree d (as a polynomial or power series of precision O(xᵈ⁺¹)). Invoking `LSeries(…, cffun)` is equivalent to `LSeries(…, 0)` followed by `LSetCoefficients(L, cffun)`. | — |

*Worked example: H127E17 (Riemann ζ function defined with coefficients deferred; LCfRequired tells N=6; setting 6 coefficients vs 2 coefficients; using the functional equation for faster convergence even with 2 coefficients).*

### 127.5.4 Specifying the Coefficients Later

When a finite coefficient list is to be supplied, `LCfRequired(L)` (§127.6) tells the user how
many are needed. The L-series object is created with `cffun = 0`; `LSetCoefficients` is then
called with the required vector.

*(No additional intrinsics defined; see `LCfRequired` in §127.6 and the worked example H127E17.)*

### 127.5.5 Generating the Coefficients from Local Factors

For L-series with weakly multiplicative coefficients an Euler product formula holds:
`L(s) = Πₚ 1/Fₚ(p⁻ˢ)`. The coefficients are specified by supplying a function `f(p,d)` that
returns the inverse local factor Fₚ(x) at prime p either as a polynomial or as a power series
of precision O(xᵈ⁺¹). Magma expands these into Dirichlet coefficients as needed.

*(No additional intrinsics defined; see the `cffun` parameter description in §127.5.2 and examples H127E20–H127E22.)*

---

## 127.6 Accessing the Invariants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LCfRequired(L)` | Number of Dirichlet coefficients aₙ required to compute L-values at current precision. May be called before coefficients are set. | Derived from the conductor and precision via the Dokchitser bound **[Dok04]**. |
| `LGetCoefficients(L, N)` | Returns the vector `[* a₁, …, aₙ *]` of the first N Dirichlet coefficients. | Evaluates the coefficient function set on L. |
| `EulerFactor(L, p)` | The p-th Euler factor of L, as a polynomial or power series. Optional parameters: `Degree` (truncate series to that length); `Precision` (for complex-valued coefficients). | — |
| `Conductor(L)` | Conductor of the L-series (real number, usually an integer). Evaluation time is proportional to √conductor. | — |
| `Sign(L)` | Sign in the functional equation (complex number of absolute value 1, or 0 if not yet computed). Calling `CheckFunctionalEquation` or any evaluation function sets the sign. | — |
| `GammaFactors(L)` | Sequence of γ-shifts λ₁,…,λd for L(s); each represents a factor Γ((s+λᵢ)/2) in the functional equation. | — |
| `LSeriesData(L)` | Returns a tuple of length 7: `(weight, gamma_shifts, conductor, coefficient_function, sign, poles_of_L*, residues_of_L*)`. Sign = 0 means not computed; Residues = [] means not computed. The tuple contains enough data to recreate L via the general `LSeries` call. | — |
| `Factorization(L)` | If L is internally represented as a product `Π Lᵢ(s)^nᵢ`, returns the sequence `[...<Lᵢ, nᵢ>...]`. | — |

*Worked examples: H127E18 (comparing LGetCoefficients of modular form vs elliptic curve; LSeriesData for E/K showing large conductor; precision reduction and factoring to speed up computation); H127E19 (Factorization of the Riemann zeta function and of the Dedekind zeta of the splitting field of x³−2).*

---

## 127.7 Precision

The default precision is the precision of the default real field at the time `LSeries` is called.
Precision can be changed later with `LSetPrecision`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LSetPrecision(L, precision)` | Change the working precision (in decimal digits) of the L-series L to `precision`. Also accepts the same optional parameters as the general `LSeries` constructor: `CoefficientGrowth`, `ImS`, `Asymptotics`. | — |

### 127.7.1 L-series with Unusual Coefficient Growth

The `CoefficientGrowth` parameter names a function `f(x)` (or `f(L, x)`) that is an increasing
function of a positive real variable x satisfying `|aₙ| ≤ f(n)`. Default: `f(x) = 1.5·xᵖ⁻¹`
where ρ is the largest real pole of L*(s) (if poles exist), or `f(x) = 2x^((weight−1)/2)` otherwise.

### 127.7.2 Computing L(s) when Im(s) is Large (ImS Parameter)

When Im(s₀) is large, cancellation causes loss of precision. Set `ImS` to the largest imaginary
part expected; otherwise Magma prints a warning. Can be set at `LSeries` creation or later via
`LSetPrecision`.

### 127.7.3 Implementation of L-series Computations (Asymptotics Parameter)

The `Asymptotics` parameter (default `true`) controls whether the special functions in the
functional equation integral are evaluated using:
- `true` (default): Taylor series at the origin plus continued fractions of asymptotic
  expansions at infinity. Faster, but convergence not always proved.
- `false`: Taylor series at the origin only. Provably convergent, but slower.

---

## 127.8 Verbose Printing

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SetVerbose("LSeries", n)` | Set verbose level for all L-series functions. Levels: 0 = quiet (default; only prints loss-of-precision warnings); 1 = prints sign and residues when computed from functional equation; 2 = messages for new L-function construction, coefficient generation, and expansion computation; 3 = progress indicator (every 1000 coefficients, every 5000 series terms). Frequencies stored in `L'vprint_coeffs` and `L'vprint_series` attributes. | — |

---

## 127.9 Advanced Examples

These subsections give worked code illustrating the general `LSeries` constructor and arithmetic
operations. No new intrinsics are introduced except in §127.9.8.

### 127.9.1 Handmade L-series of an Elliptic Curve

Example H127E20: reproduces `LSeries(E)` using the general constructor with local factors from
`TraceOfFrobenius` and `RootNumber`.

### 127.9.2 Self-made Dedekind Zeta Function

Example H127E21: reproduces `LSeries(K)` using local factors from ideal decomposition
(`Decomposition(MaximalOrder(K), p)`) and residue at s = 1 from the class number formula.

### 127.9.3 L-series of a Genus 2 Hyperelliptic Curve

Example H127E22: constructs the L-series of H¹ of the Jacobian of a hyperelliptic curve,
computing the bad local factor at p = 13 by blowing up the singular fibre.

### 127.9.4 Experimental Mathematics for Small Conductor

Example H127E23: recovers the first 20 coefficients of the L-series of a conductor-11 elliptic
curve experimentally using `CheckFunctionalEquation` as an oracle (Stark–Mestre method).

### 127.9.5 Tensor Product of L-series Coming from ℓ-adic Representations

Example H127E24: illustrates `TensorProduct(LE, LK)` for L(E) and L(K) with K = Q(∛2),
including the correction needed at p = 2 where E acquires good reduction over K. Shows how
`LocalInformation` and `TraceOfFrobenius` supply the missing local factor.

### 127.9.6 Non-abelian Twist of an Elliptic Curve

Example H127E25: the non-abelian twist `LSeries(E,K)/LSeries(E)` for E = Mordell curve and
K = Q(∛2). BSD prediction on Selmer rank confirmed computationally; same L-function obtained
via `LSeries(E, rho)` using the Artin representation.

### 127.9.7 Other Tensor Products

Section and examples H127E26–H127E29 illustrate:
- Tensor product of two elliptic curve L-functions over Q (Example H127E26).
- Tensor product of two level-1 modular forms; Ramanujan congruence recovered (Example H127E27).
- Tensor product related to Siegel modular forms **[vGvS93]** (Example H127E28).
- Tensor product over a quadratic field K = Q(√−3) for a Hecke character (Example H127E29).

### 127.9.8 Symmetric Powers

For GL(1), the kth symmetric power of `L(ψ, s)` is `L(ψᵏ, s)` (after primitivisation). For GL(2),
the kth symmetric power of an L-function with eigenvalues α₁(p), α₂(p) at each prime has Euler
factors `Π_{i=0}^{k} (1 − α₁(p)^(k−i) α₂(p)^i / Npˢ)⁻¹`. More generally the kth symmetric power of a
degree-d L-function has degree `C(k+d−1, d−1)`. Bad Euler factors for elliptic curves over Q are
computed from the formulae of Martin–Watkins **[MW06]** and Dummigan–Martin–Watkins **[DMW09]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SymmetricPower(L, m)` | Returns the L-series corresponding to the mth symmetric power of L. Parameter: `BadEulerFactors` (sequence of `⟨p, f, E⟩` triples, where p is a prime, f the conductor exponent, and E a polynomial giving the inverse Euler factor at p). | For GL(1): primitivization and power of the underlying character. For GL(2) / elliptic curves: bad Euler factors from **[MW06, DMW09]**; general case requires user to supply bad factors; Hodge structure combinatorics for gamma factors. |

*Worked examples: H127E30 (symmetric powers of GL(1) Dirichlet character); H127E31 (symmetric powers of GL(1) Hecke character on a quadratic field); H127E32 (symmetric powers of a Hecke Grössencharacter); H127E33 (symmetric square of E = 389A, ModularDegree = 40 recovered); H127E34 (symmetric cube of E = 73A; central value vanishes to order 4, work of Buhler–Schoen–Top **[BST97]**).*

---

## 127.10 Weil Polynomials

Auxiliary routines for the characteristic polynomial of Frobenius on étale cohomology. These
make frequent use of `PowerSumToCoefficients`, `CoefficientsToElementarySymmetric`, and
`ElementarySymmetricToPowerSums`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SetVerbose("WeilPolynomials", v)` | Set verbose level for Weil polynomial routines. Maximum value 2. | — |
| `HasAllRootsOnUnitCircle(f)` | For a polynomial f with rational coefficients, checks that all complex roots have absolute value 1. Does not use floating-point approximations. | Exact algebraic method (no floating point). |
| `FrobeniusTracesToWeilPolynomials(tr, q, i, deg)` | Given Frobenius traces on the i-th étale cohomology over Fq, returns a list of possible Weil polynomials of degree `deg`. For even i, determines the sign in the functional equation from absolute values of roots when possible; otherwise returns both candidates. Contradictory data returns an empty sequence. Optional parameter: `KnownFactor` (a known polynomial factor). | Lift traces to symmetric functions; test functional equation and root-on-unit-circle conditions. |
| `WeilPolynomialToRankBound(f, q)` | Counts zeros of f (with multiplicity) that are q times a root of unity; this is an upper bound for the Picard rank of the corresponding algebraic surface (equals Picard rank conditionally on the Tate conjecture). | Root counting on the unit circle. |
| `ArtinTateFormula(f, q, h20)` | For Weil polynomial f corresponding to H² of an algebraic surface with Hodge number h²·⁰, evaluates the Artin–Tate formula. Returns: arithmetic Picard rank; absolute value of discriminant(Pic) × order(Br), conditional on the Tate conjecture. | Artin–Tate formula; see **[EJ10]**. |
| `WeilPolynomialOverFieldExtension(f, deg)` | For the characteristic polynomial f of Frobenius on an étale cohomology group, returns the characteristic polynomial of the `deg`-fold iterated Frobenius. | Polynomial substitution x → x^deg (adapting the Newton power-sum structure). |
| `CheckWeilPolynomial(f, q, h20)` | Checks conditions that the characteristic polynomial of Frobenius on H² of an algebraic surface must satisfy: valuation of roots at all places (including p and ∞), functional equation, and Artin–Tate conditions **[EJ10]**. Returns `true` or `false`. Parameters: q (base field size), h20 (Hodge number h²·⁰); optional `SurfDeg` (degree of surface; −1 = unknown). | Valuations, functional equation check, Artin–Tate conditions per **[EJ10]**. |

*Worked example: H127E35 (quartic surface over F₂; FrobeniusTracesToWeilPolynomials returns two candidates; CheckWeilPolynomial with SurfDeg:=4 selects the correct sign; WeilPolynomialToRankBound and ArtinTateFormula recover Picard rank 1 and discriminant bounds; advanced variant: bounding Picard rank with fewer point counts via cyclotomic KnownFactors).*

---

## 127.11 Bibliography

| Key | Reference |
|-----|-----------|
| **[Arm71]** | J. V. Armitage. *Zeta functions with a zero at s = 1/2.* Invent. Math., **15**(3):199–205, 1971. |
| **[BST97]** | J. Buhler, C. Schoen, and J. Top. *Cycles, L-functions and triple products of elliptic curves.* J. Reine. Angew. Math., **492**:93–133, 1997. |
| **[Coh00]** | Henri Cohen. *Advanced Topics in Computational Number Theory.* Springer, Berlin–Heidelberg–New York, 2000. |
| **[DMW09]** | N. Dummigan, P. Martin, and M. Watkins. *Euler factors and local root numbers for symmetric powers of elliptic curves.* Pure and Appl. Math. Qu., **5**(4):1311–1341, 2009. |
| **[Dok02]** | Tim Dokchitser. *ComputeL, pari package to compute motivic L-functions.* URL: http://www.maths.dur.ac.uk/~dma0td/computel/, 2002. |
| **[Dok04]** | Tim Dokchitser. *Computing special values of motivic L-functions.* Experiment. Math., **13**(2):137–149, 2004. |
| **[EJ10]** | Andreas-Stephan Elsenhans and Jörg Jahnel. *Weil polynomials of K3 surfaces.* In Algorithmic number theory, volume 6197 of Lecture Notes in Computer Science, pages 126–141, Berlin, 2010. Springer. |
| **[Fri76]** | J. B. Friedlander. *On the class numbers of certain quadratic extensions.* Acta Arith., **28**(4):391–393, 1975/76. |
| **[HPP06]** | F. Hess, S. Pauli, and M. Pohst, editors. *ANTS VII,* volume 4076 of LNCS. Springer-Verlag, 2006. |
| **[JKS94]** | Uwe Jannsen, Steven Kleiman, and Jean-Pierre Serre, editors. *Motives,* volume 55 of Proceedings of Symposia in Pure Mathematics, Providence, RI, 1994. American Mathematical Society. |
| **[Lav67]** | A. F. Lavrik. *An approximate functional equation for the Hecke zeta-function of an imaginary quadratic field.* Mat. Zametki, **2**:475–482, 1967. |
| **[MW06]** | P. Martin and M. Watkins. *Symmetric powers of elliptic curve L-functions.* In Hess et al. [HPP06], pages 377–392. |
| **[Ser65]** | Jean-Pierre Serre. *Zeta and L functions.* In Arithmetical Algebraic Geometry (Proc. Conf. Purdue Univ., 1963), pages 82–92. Harper & Row, New York, 1965. |
| **[Ser71]** | J.-P. Serre. *Conducteurs d'Artin des caractères réels.* Invent. Math., **14**(3):173–183, 1971. |
| **[Sha95]** | I. R. Shafarevich, editor. *Number theory I,* volume 49 of Encyclopaedia of Mathematical Sciences. Springer-Verlag, Berlin, 1995. |
| **[Tol97]** | Emmanuel Tollis. *Zeros of Dedekind zeta functions in the critical strip.* Math. Comp., **66**(219):1295–1321, 1997. |
| **[vGvS93]** | B. van Geemen and D. van Straten. *The cusp forms of weight 3 on Γ₂(2,4,8).* Math. Comp., **61**(204):849–872, 1993. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Dokchitser functional equation method **[Dok04, Dok02]** (core of all L-value computation) | `Evaluate`, `CentralValue`, `LStar`, `LTaylor`, `CheckFunctionalEquation`, and all `LSeries(…)` constructors |
| Approximate functional equation / theta function test **[Lav67, Tol97, Coh00 §10.3]** | `CheckFunctionalEquation` |
| Artin factorisation of Dedekind zeta | `LSeries(K)` with `Method := "Artin"` |
| Class number formula for residue at s = 1 | `LSeries(K)` with `ClassNumberFormula := true` |
| Tensor product of ℓ-adic representations (Serre) | `TensorProduct`, `LSeries(E, K)`, `LSeries(E, A)` |
| Symmetric power L-functions; bad Euler factors for elliptic curves **[MW06, DMW09]** | `SymmetricPower` |
| Weil polynomial / Frobenius trace lifting | `FrobeniusTracesToWeilPolynomials`, `WeilPolynomialOverFieldExtension` |
| Artin–Tate formula **[EJ10]** | `ArtinTateFormula`, `CheckWeilPolynomial` |
| Exact root-on-unit-circle test | `HasAllRootsOnUnitCircle` |
| Picard rank bound (Tate conjecture) | `WeilPolynomialToRankBound` |
| Stark–Mestre experimental coefficient recovery | Illustrated in H127E23 via `CheckFunctionalEquation` + `LSetCoefficients` loop |
