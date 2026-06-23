# Chapter 23 — Univariate Polynomial Rings

**Handbook part:** III — Basic Rings
**Handbook pages:** 411–439 (PDF pages 540–573)

---

## Scope and overview

Univariate polynomial rings may be defined over any ring R. The univariate polynomial ring
in indeterminate x over coefficient ring R is denoted P = R[x]. Magma supports two kinds of
polynomials: univariate polynomials, represented as vectors of coefficients, and multivariate
polynomials stored in distributive form (linear sums of coefficient–monomial pairs). This
chapter covers univariate polynomials only.

The vector representation enables fast arithmetic on univariate polynomials but requires
considerable memory for multivariate use. Only univariate polynomial rings using the vector
representation can be created directly; it is technically possible to nest them (e.g. R[x][y])
but this is not recommended. Multivariate polynomials can be stored efficiently in
distributive form, though arithmetic on single-variable polynomials stored that way may be
considerably slower.

Elements of univariate polynomial rings belong to category `RngUPolElt`; elements of
quotient rings belong to `RngUPolResElt`. The ring itself is in category `RngUPol`.

---

## 23.1 Introduction

### 23.1.1 Representation

The chapter's representation discussion (summarised above) motivates the design choice: the
vector form makes single-variable arithmetic fast, while the distributive form suits
multivariate rings created via `PolynomialRing(R, n)`. Users should use `PolynomialRing(R)`
(no second argument) for univariate work and avoid nesting univariate rings to simulate
multivariate ones.

---

## 23.2 Creation Functions

### 23.2.1 Creation of Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PolynomialAlgebra(R)` / `PolynomialRing(R)` | Create the univariate polynomial ring R[x] over the ring `R`, stored in vector form. Regarded as an R-algebra via identification of R with the constant polynomials. Use `P<x> := PolynomialRing(R)` to name the indeterminate. Parameter `Global` (BoolElt, default `true`): if `true`, returns the unique global univariate polynomial ring over R (all calls return the same object); if `false`, returns a fresh non-global ring with its own indeterminate name. | — |

*Worked example: H23E1 (global vs. non-global rings; automatic coercion between them).*

### 23.2.2 Print Options

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AssignNames(~P, s)` | Procedure to change the name of the indeterminate of polynomial ring `P` to the string in sequence `s`. Only affects printing; does not bind Magma identifiers. `P` must be passed by reference (`~P`). | — |
| `Name(P, i)` | Returns the i-th indeterminate of polynomial ring `P` as an element of `P`. | — |

### 23.2.3 Creation of Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `P . 1` | Returns the indeterminate of polynomial ring `P` as an element of `P`. | — |
| `elt< P \| a0, ..., ad >` | Given a polynomial ring P = R[x] and elements a0, …, ad coercible into R, returns the polynomial a0 + a1·x + … + ad·x^d. | — |
| `P ! s` / `elt< P \| s >` | Coerce `s` into P = R[x]. `s` may be: an element of P (returned unchanged); an element coercible into R (becomes a constant polynomial); an element of another univariate polynomial ring (coefficients are coerced); or a sequence (empty → zero polynomial; non-empty → coefficients in ascending order of degree). | — |
| `Polynomial(Q)` | Given a sequence `Q` of elements from a ring R, return the polynomial over R with those coefficients. Equivalent to `PolynomialRing(Universe(Q)) ! Q`. | — |
| `Polynomial(R, Q)` | Given ring `R` and sequence `Q`, return the polynomial over `R` with coefficients given by elements of `Q` coerced into R. Equivalent to `PolynomialRing(R) ! ChangeUniverse(Q, R)`. | — |
| `Polynomial(R, f)` | Given ring `R` and polynomial `f` over ring S, return the polynomial over `R` obtained by coercing the coefficients of `f` into R. Equivalent to `PolynomialRing(R) ! f`. | — |
| `One(P)` / `Identity(P)` | The multiplicative identity of P. | — |
| `Zero(P)` / `Representative(P)` | The zero element of P. | — |

*Worked example: H23E2 (creating x^3 + 3x + 1 via angle brackets, `elt< >`, and `!`; coercion subtleties with sequences).*

---

## 23.3 Structure Operations

### 23.3.1 Related Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BaseRing(P)` / `CoefficientRing(P)` / `CoefficientRing(f)` | Return the coefficient ring of polynomial ring `P` (or the parent ring of polynomial `f`). Univariate polynomial rings belong to category `RngUPol`. | — |
| `Category(P)` / `Parent(P)` / `PrimeRing(P)` | Standard structure-level functions. | — |

### 23.3.2 Changing Rings

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ChangeRing(P, S)` | Given polynomial ring P = R[x] and ring `S` (with automatic coercion R → S available), construct Q = S[y] and the induced homomorphism h : P → Q. Supports angle-bracket naming on the result. | — |
| `ChangeRing(P, S, f)` | Given P = R[x], ring `S`, and a map f : R → S, construct Q = S[y] and the homomorphism h : P → Q obtained by applying f to each coefficient. | — |

*Worked example: H23E3 (integer-to-rational coercion via `ChangeRing`; non-standard embedding mapping 1 → 3).*

### 23.3.3 Numerical Invariants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Rank(P)` | The rank of polynomial ring `P` (maximum number of independent indeterminates over the coefficient ring); always 1 for univariate polynomial rings. | — |
| `#P` | The cardinality of `P`; returns an integer only for finite `P` (i.e. quotients of polynomial rings over finite coefficient rings). | — |
| `Characteristic(P)` | The characteristic of `P`. | — |

### 23.3.4 Ring Predicates and Booleans

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsCommutative(P)` / `IsUnitary(P)` | Test commutativity / unitarity of P. | — |
| `IsFinite(P)` / `IsOrdered(P)` | Test finiteness / orderedness of P. | — |
| `IsField(P)` / `IsEuclideanDomain(P)` | Test whether P is a field / Euclidean domain. | — |
| `IsPID(P)` / `IsUFD(P)` | Test whether P is a PID / UFD. | — |
| `IsDivisionRing(P)` / `IsEuclideanRing(P)` | Test division ring / Euclidean ring properties. | — |
| `IsDomain(P)` / `IsPrincipalIdealRing(P)` | Test integral domain / principal ideal ring properties. | — |
| `P eq Q` / `P ne Q` / `P lt Q` / `P gt Q` / `P le Q` / `P ge Q` | Comparison operators on polynomial rings. | — |

### 23.3.5 Homomorphisms

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `hom< P -> S \| f, y >` / `hom< P -> S \| y >` | Given P = R[x], ring `S`, map f : R → S, and element y ∈ S, create the homomorphism g : P → S by g(Σ sᵢxⁱ) = Σ f(sᵢ)yⁱ. The coefficient map `f` may be omitted, in which case a unitary homomorphism (1_R ↦ 1_S) is used. The image `y` may come from a structure that coerces automatically into S. | — |

*Worked example: H23E4 (mapping Z[x] → R by sending x ↦ 1/2, using the unitary coefficient map).*

---

## 23.4 Element Operations

### 23.4.1 Parent and Category

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Parent(p)` / `Category(p)` | Return the parent ring and category of polynomial element `p`. | — |

### 23.4.2 Arithmetic Operators

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `+a` / `-a` | Unary plus and negation. | — |
| `a + b` / `a - b` / `a * b` / `a ^ k` / `a / b` / `a div b` / `a mod b` | Binary arithmetic. Division `/` is only allowed by elements of the coefficient field (when P is over a field); `div` and `mod` give polynomial quotient and remainder. Negative powers not allowed. | — |
| `a +:= b` / `a -:= b` / `a *:= b` | In-place arithmetic operators. | — |

### 23.4.3 Equality and Membership

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `a eq b` / `a ne b` | Equality and inequality of polynomial elements. | — |
| `a in R` / `a notin R` | Membership test for a polynomial element in a ring. | — |

### 23.4.4 Predicates on Ring Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsZero(a)` / `IsOne(a)` / `IsMinusOne(a)` | Test for zero, one, or minus-one polynomial. | — |
| `IsNilpotent(a)` / `IsIdempotent(a)` | Test nilpotency / idempotency. Not available for all coefficient rings. | — |
| `IsUnit(a)` / `IsZeroDivisor(a)` / `IsRegular(a)` | Test unit / zero-divisor / regular element. Not available for all coefficient rings. | — |
| `IsIrreducible(a)` / `IsPrime(a)` / `IsMonic(a)` | Test irreducibility, primality, and monicness. | — |

### 23.4.5 Coefficients and Terms

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Coefficients(p)` / `ElementToSequence(p)` / `Eltseq(p)` | Return the coefficients of p ∈ R[x] in ascending order (constant term first) as a sequence of elements of R. | — |
| `Coefficient(p, i)` | Return the coefficient of xⁱ in p ∈ R[x] as an element of R. Returns zero if i exceeds the degree of p. | — |
| `MonomialCoefficient(p, m)` | Given polynomial p and monomial m (exactly one non-zero coefficient equal to 1) in P = R[x], return the coefficient of m in p as an element of R. | — |
| `LeadingCoefficient(p)` | Coefficient of the highest occurring power of x in p, as an element of R. | — |
| `TrailingCoefficient(p)` | Coefficient of the lowest occurring power of x in p, as an element of R. | — |
| `ConstantCoefficient(p)` | The constant term (coefficient of x^0) of p, as an element of R. | — |
| `Terms(p)` | Non-zero terms of p in ascending degree order, as a sequence of elements of P. | — |
| `LeadingTerm(p)` | Term of p with the highest occurring power of x, as an element of P. | — |
| `TrailingTerm(p)` | Term of p with the lowest occurring power of x, as an element of P. | — |
| `Monomials(p)` | The monomials of univariate p (powers of the indeterminate up to the degree), matching up with `Coefficients(p)`. | — |
| `Support(p)` | Returns the positions (exponents) in p with non-zero coefficients, and the corresponding coefficients. | — |
| `Round(p)` | For p ∈ R[x] where R is Z, Q, or a real field, return the polynomial in Z[x] obtained by rounding all coefficients. | — |
| `Valuation(p)` | The valuation of p: the exponent of the largest power of x dividing p. The zero polynomial has valuation ∞. | — |

### 23.4.6 Degree

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Degree(p)` | The degree of p ∈ R[x], i.e., the exponent of the highest power of x with non-zero coefficient. The zero polynomial has degree −1. | — |

### 23.4.7 Roots

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Roots(p)` | For p over an allowed coefficient ring (complex/real fields, Z, Q, finite fields, residue class rings with prime modulus): a sorted sequence of pairs (root, multiplicity) of roots in the coefficient ring. Parameter `Max` (RngIntElt): if set to a non-negative integer m, return at most m roots. | — |
| `Roots(p, S)` | As `Roots(p)`, but finds roots in the ring `S` (the coefficients of `p` must coerce into `S`). Same coefficient ring restrictions apply. | — |
| `HasRoot(p)` | Returns `true` iff `p` has a root in the coefficient ring R; if so, also returns a root. Especially fast for finite fields. Same coefficient ring restrictions as `Roots`. | — |
| `HasRoot(p, S)` | Returns `true` iff `p` has a root in ring `S` (containing R); if so, also returns a root in S. | — |
| `SmallRoots(p, N, X)` | For monic non-zero p ∈ Z[x] and positive integers N, X: returns all x₀ with |x₀| ≤ X and p(x₀) ≡ 0 (mod N), provided X ≤ 0.5 · N^(1/d) (d = deg p). Parameters: `Bits` (BoolElt, default false; if true, X is read as 2^X); `Beta` (RngElt, default 1.0; finds roots modulo divisors N' ≥ N^β of N, provided X ≤ 0.5 · N^(β²/d)); `Exponent` (shape of lattice basis — highest power of p used); `Finalshifts` (number of final shifts of p^m); `Direct` (BoolElt, default false; reduce lattice at once rather than progressively). | Coppersmith's algorithm for small roots of a univariate modular polynomial **[Cop96]**, as described in **[May03]**; relies on LLL lattice-basis reduction **[LLL82]**. Used in cryptanalysis of public-key systems. |
| `SetVerbose("SmallRoots", v)` | Set verbose printing level for `SmallRoots`. Legal values: true, false, 0, 1, 2 (false = 0, true = 1). | — |

*Worked example: H23E5 (using `SmallRoots` to recover an RSA factor from an approximation, demonstrating the Beta parameter).*

### 23.4.8 Derivative, Integral

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Derivative(p)` | Returns the (formal) derivative of p ∈ P as an element of P. | — |
| `Derivative(p, n)` | Returns the n-th (formal) derivative of p ∈ P (n ≥ 0) as an element of P. | — |
| `Integral(p)` | Returns the formal integral of p ∈ P over a field of characteristic zero, as an element of P. | — |

### 23.4.9 Evaluation, Interpolation

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Evaluate(p, r)` | Evaluate polynomial p at element r of a ring S. If r coerces into the coefficient ring R, the result is in R; otherwise, a generic evaluation is performed and the result is in S. | — |
| `Interpolation(I, V)` | Given sequences I and V of n elements of a field K, return the unique univariate polynomial p over K of degree less than n with p(I[i]) = V[i] for all 1 ≤ i ≤ n. | Polynomial interpolation (implicitly Newton or Lagrange). |

### 23.4.10 Quotient and Remainder

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Quotrem(f, g)` | Returns polynomials q (quotient) and r (remainder) in P = R[x] such that f = q·g + r with degree of r minimal. The leading coefficient of g must be a non-zero divisor in R. If it is a unit, deg(r) < deg(g). | Polynomial division. |
| `f div g` | The polynomial quotient of f by g (first return value of `Quotrem`). | — |
| `IsDivisibleBy(a, b)` | Returns whether polynomial a is exactly divisible by polynomial b (i.e. ∃ q with a = q·b); if so, also returns the quotient q. | — |
| `ExactQuotient(f, g)` | Assuming g exactly divides f, returns the exact quotient (error if g does not divide f). | — |
| `f mod g` | Remainder of division of f by g (second return value of `Quotrem`). | — |
| `Valuation(f, g)` | The exponent of the highest power of polynomial g that divides polynomial f. | — |
| `Reductum(f)` | The reductum of f: the polynomial obtained by removing the leading term of f. | — |
| `PseudoRemainder(f, g)` | For f, g ∈ R[x] (R an integral domain): the pseudo-remainder r defined by c^d · f = q·g + r (where c is the leading coefficient of g and d = max(0, deg(f) − deg(g) + 1)), with deg(r) < deg(g). | — |
| `EuclideanNorm(p)` | The Euclidean norm of p (the degree function, which makes R[x] a Euclidean ring). | — |

### 23.4.11 Modular Arithmetic

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Modexp(f, n, g)` | Given univariate polynomials f, g ∈ K[x] over a field K and non-negative integer n, return f^n mod g. `g` may be a constant polynomial. | Modular polynomial exponentiation. |
| `ChineseRemainderTheorem(X, M)` / `CRT(X, M)` | Given sequences X and M of polynomials with elements of M pairwise coprime, find a single polynomial t solving t ≡ X[i] (mod M[i]) for all i. | Chinese Remainder Theorem for polynomials. |

### 23.4.12 Other Operations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ReciprocalPolynomial(f)` | The reciprocal of the given univariate polynomial (coefficients reversed). | — |
| `PowerPolynomial(f, n)` | The polynomial whose roots are the n-th powers of the roots of f (f should have coefficients in a field). | — |
| `f ^ M` | Transformation of univariate polynomial f under the linear fractional transformation given by the 2×2 matrix M (obtained by homogenizing f and making a linear substitution). | — |

---

## 23.5 Common Divisors and Common Multiples

Functions in this section apply to univariate polynomials over a field, over the integers, or
over a residue class ring of integers with prime modulus, or any polynomial ring over these.

### 23.5.1 Common Divisors and Common Multiples

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `GreatestCommonDivisor(f, g)` / `Gcd(f, g)` / `GCD(f, g)` | Greatest common divisor of univariate polynomials f and g over ring R (R must have a GCD algorithm for its elements). Result is normalized (unique). Zero inputs: GCD is the other polynomial; both zero gives zero. | Over finite fields: Euclidean algorithm (no intermediate coefficient blowup). Over Z or Q: combination of (1) heuristic GCDHEU algorithm **[CGG89, GCL92 §7.7]** for small/moderate dense polynomials and (2) modular algorithm similar to **[vzGG99, Algorithm 6.38]** / **[GCL92 §7.4]**. Over algebraic, quadratic, or cyclotomic number fields: fast modular algorithm (maps field to a residue class polynomial ring mod a small prime). Over algebraic function fields or polynomial quotient rings over function fields (since V2.10): fast modular algorithm of Allan Steel (unpublished), evaluating/interpolating for each base transcendental. Over other polynomial rings / function fields: polynomials are first "flattened" to a multivariate polynomial ring, then the appropriate algorithm is used. Over any other ring: generic subresultant algorithm **[Coh93 §3.3]**. |
| `ExtendedGreatestCommonDivisor(f, g)` / `Xgcd(f, g)` / `XGCD(f, g)` | Returns polynomials c, a, b with deg(a) < deg(g), deg(b) < deg(f), c the monic GCD of f and g, and c = a·f + b·g. The coefficient ring must be a field. Multipliers are unique if both inputs are non-zero. | Over Q: modular algorithm of Allan Steel (unpublished). Over other fields: basic Euclidean algorithm. |
| `LeastCommonMultiple(f, g)` / `Lcm(f, g)` / `LCM(f, g)` | Least common multiple of f and g; result is normalized. LCM with zero is zero. Effectively computed as `Normalize((f div GCD(f,g)) * g)` for non-zero inputs. Valid coefficient rings same as GCD. | Reduction to GCD. |
| `Normalize(f)` | Returns the unique normalized polynomial g associate to f (g = u·f for a unit u ∈ R). Over a field: g is monic. Over Z: leading coefficient of g is positive. Over a polynomial ring: leading coefficient is recursively normalized. | — |

### 23.5.2 Content and Primitive Part

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Content(p)` | The content of p: the GCD of the coefficients of p, as an element of the coefficient ring. | — |
| `PrimitivePart(p)` | The primitive part of p: p divided by its content. | — |
| `ContentAndPrimitivePart(p)` / `Contpp(p)` | Returns both the content and the primitive part of p. | — |

---

## 23.6 Polynomials over the Integers

Functions in this section apply to univariate polynomials over Z only.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Sign(p)` | The sign of the leading coefficient of p. | — |
| `AbsoluteValue(p)` / `Abs(p)` | Returns p or −p, whichever has non-negative leading coefficient. | — |
| `MaxNorm(p)` | The maximum of the absolute values of the coefficients of p. | — |
| `SumNorm(p)` | The sum of the coefficients of p. | — |
| `DedekindTest(p, m)` | Given a monic polynomial p (univariate or multivariate in one variable) and a prime m: returns true iff p satisfies the Dedekind criterion at m (i.e. the equation order corresponding to p is locally maximal at m **[PZ89, p. 295]**). | Dedekind criterion **[PZ89]**. |

---

## 23.7 Polynomials over Finite Fields

Functions in this section apply to univariate polynomials over finite fields only.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PrimePolynomials(R, d)` / `PrimePolynomials(R, d, n)` | A sequence of all monic prime (irreducible) polynomials of R of degree d; resp. a sequence of n such polynomials. | Enumeration. |
| `RandomPrimePolynomial(R, d)` | A random monic prime polynomial of R of degree d. | Random selection. |
| `NumberOfPrimePolynomials(q, d)` / `NumberOfPrimePolynomials(K, d)` / `NumberOfPrimePolynomials(R, d)` | The number of monic prime polynomials of degree d over the given finite field (specified by cardinality q, field K, or polynomial ring R). | Combinatorial formula. |
| `JacobiSymbol(a, b)` | The Jacobi symbol (a/b) of polynomials a, b ∈ F_q[x] (q must be odd). If b is irreducible: equals 0 if b\|a, 1 if a is a square mod b, −1 otherwise. Extends multiplicatively to all non-constant b. | Analogue of the classical Jacobi symbol for polynomial rings over finite fields. |

---

## 23.8 Factorization

### 23.8.1 Factorization and Irreducibility

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Factorization(f)` / `Factorisation(f)` | Given a univariate polynomial f over ring R, returns the factorization as a sequence of (irreducible factor, multiplicity) pairs (each factor is normalized), plus the unit u such that f = u · ∏ qᵢ^kᵢ. R must be: a finite field F_q, Z, Q, an algebraic number field Q(α), a local ring, a polynomial ring, function field, or finite-dimensional affine algebra over any of the above. Parameter `Al` (MonStgElt, default `"Default"`): selects the factorization algorithm over finite fields. Values: `"Default"` (automatic), `"BerlekampSmall"` or `"BerlekampLarge"` (Berlekamp variants), `"GKS"` (von zur Gathen/Kaltofen/Shoup). | Over small finite fields: Berlekamp algorithm **[Knu97 §4.6.2]** / **[vzGG99 §14.8]** (fast linear algebra). Over medium/large finite fields: von zur Gathen/Kaltofen/Shoup algorithm **[vzGS92, KS95, Sho95]**. Over Z or Q (since V2.8): van Hoeij's algorithm **[vH02, vH01]** — factorize mod a small prime, Hensel-lift using Shoup's tree-lifting **[vzGG99 §15.5]**, then find correct combinations via a Knapsack problem solved by LLL lattice-basis reduction **[LLL82]** (not by exhaustive search as in Berlekamp–Zassenhaus). Over algebraic number fields, algebraic function fields, and affine algebras: Trager's norm-based algorithm **[Tra76]** (substitution + resultant to reduce variable count; then van Hoeij combination in characteristic 0). Over function fields in small characteristic with inseparable extensions: algorithm of Allan Steel **[Ste05]**. |
| `HasPolynomialFactorization(R)` | Returns whether factorization of polynomials over ring R is supported in Magma. | — |
| `SetVerbose("PolyFact", v)` | Procedure: sets the verbose printing level for all polynomial factorization algorithms. Legal levels: 0, 1, 2, 3. | — |
| `FactorisationToPolynomial(f)` / `Facpol(f)` | Given a sequence of (irreducible polynomial, positive integer exponent) tuples, returns the product polynomial. | — |
| `SquarefreeFactorization(f)` | Given f over Z or any field, returns the squarefree factorization as a sequence of (factor, multiplicity) pairs; factors contain no square of any non-constant polynomial. | Computes GCD of f with its derivative repeatedly; special handling in characteristic p. |
| `DistinctDegreeFactorization(f)` | Given a squarefree f ∈ F[x] (F a finite field), returns the distinct-degree factorization as a sequence of (degree d, product of degree-d irreducible factors) pairs. Parameter `Degree` (RngIntElt, default 0): if > 0, only return factors up to that degree. | Standard distinct-degree factorization over finite fields. |
| `EqualDegreeFactorization(f, d, g)` | Given squarefree f ∈ F[x] (F a finite field) known to be a product of distinct degree-d irreducibles, integer d, and g = x^q mod f (q = #F): returns the irreducible factors of f as a sequence. Result is unpredictable if conditions are not met. | Cantor–Zassenhaus-style equal-degree splitting. |
| `IsIrreducible(f)` | Returns true iff f is irreducible over R. Conditions on R same as for `Factorization`. | Reduction to `Factorization`. |
| `IsSeparable(f)` | For f ∈ K[x] with deg(f) ≥ 1 and K a field allowing polynomial factorization: returns true iff f is separable (no repeated roots). | GCD with derivative. |
| `QMatrix(f)` | For univariate f of degree d over a finite field F: returns the Berlekamp Q-matrix (element of the degree-(d−1) matrix algebra over F). | Berlekamp Q-matrix construction **[Knu97]**. |

*Worked example: H23E6 (factoring Swinnerton-Dyer polynomials — worst-case for Berlekamp–Zassenhaus — using van Hoeij's algorithm; degree-192 product of SD6 and SD7 polynomials factored in ~17 seconds).*

### 23.8.2 Resultant and Discriminant

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Discriminant(f)` | Returns the discriminant D ∈ R of f ∈ R[x], defined as D = c_n^(2n−2) · ∏_{i≠j}(αᵢ − αⱼ) where c_n is the leading coefficient and αᵢ are the roots. R must be a domain. | Computed via resultant of f and its derivative. |
| `Resultant(f, g)` | Returns the resultant of f and g ∈ R[x] (degrees m and n), defined as the determinant of the (m+n)×(m+n) Sylvester matrix. Result is in R. R must be a domain. | Sylvester matrix determinant or subresultant sequence. |
| `CompanionMatrix(f)` | For monic univariate f of degree d over R: returns the companion matrix of f as an element of the degree-(d−1) full matrix algebra over R. The companion matrix for f = a₀ + a₁x + … + a_{d−1}x^{d−1} + x^d is the d×d matrix with 1s on the superdiagonal and −a₀, −a₁, …, −a_{d−1} in the last row. | — |

### 23.8.3 Hensel Lifting

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HenselLift(f, s, P)` | Given the squarefree factorization sequence s of integer polynomial f modulo a prime p, and P the univariate polynomial ring over Z/p^k·Z: returns the Hensel lifting, a sequence of polynomials in P satisfying f ≡ ∏ tᵢ (mod p^k). | Hensel's lemma lifting of a squarefree modular factorization. |

*Worked example: H23E7 (lifting the factors of x^5 − x^3 + 2x^2 − 2 from F_5 to Z/125Z).*

---

## 23.9 Ideals and Quotient Rings

Currently ideals and quotient rings can only be created in univariate polynomial rings over
fields. Such rings are principal ideal domains: every ideal is generated by a single element.

### 23.9.1 Creation of Ideals and Quotients

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ideal< R \| a1, ..., ar >` | Given univariate polynomial ring R over a field K, returns the ideal generated by a1, …, ar ∈ R. Equivalent to the ideal generated by GCD(a1, …, ar). Returns the ideal as a subring of R generated by a single element. | — |
| `quo< R \| I >` / `quo< R \| a1, ..., ar >` | Given an ideal I (or generators a1, …, ar) in univariate polynomial ring R over a field, returns the quotient ring R/I and the projection map h : R → R/I. Supports angle-bracket naming: `Q<q> := quo< R \| I >`. | — |

### 23.9.2 Ideal Arithmetic

Ideals of R are regarded as subrings of R (so R itself is a valid ideal).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `I + J` | Sum of ideals I and J (same polynomial ring R); generated by GCD(I.1, J.1) since R is a PID. | — |
| `I * J` | Product of ideals I and J; generated by I.1 * J.1. | — |
| `I meet J` | Intersection of ideals I and J; since R is a PID, equals the product I * J, generated by I.1 * J.1. | — |
| `a in I` | Returns true iff polynomial a is contained in ideal I. | — |
| `a notin I` | Returns true iff polynomial a is not contained in ideal I. | — |
| `I eq J` | Returns true iff ideals I and J are equal. | — |
| `I ne J` | Returns true iff ideals I and J are not equal. | — |
| `I subset J` | Returns true iff ideal I is contained in ideal J. | — |
| `I notsubset J` | Returns true iff ideal I is not contained in ideal J. | — |

### 23.9.3 Other Functions on Ideals

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `I . 1` | Given ideal I in a univariate polynomial ring R, returns the generator of I as an element of I. | — |

### 23.9.4 Other Functions on Quotients

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Modulus(Q)` | Given quotient ring Q = R[x]/I, returns the generator of I as an element of R. | — |
| `PreimageRing(Q)` | If Q = R/I for univariate polynomial ring R, returns R. | — |

---

## 23.10 Special Families of Polynomials

### 23.10.1 Orthogonal Polynomials

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ChebyshevFirst(n)` / `ChebyshevT(n)` | Constructs the Chebyshev polynomial of the first kind T_n(x), defined by T_n(x) = cos(nθ) with x = cos θ. | Classical recurrence. |
| `ChebyshevSecond(n)` / `ChebyshevU(n)` | Constructs the Chebyshev polynomial of the second kind U_n(x) of degree n−1, defined by U_n(x) = (1/n)·T_n'(x) = sin(nθ)/sin θ with x = cos θ. | Classical recurrence. |
| `LegendrePolynomial(n)` | Constructs the Legendre polynomial P_n(x) of degree n via the recurrence P_0 = 1, P_1 = x, P_n(x) = (1/n)·((2n−1)·x·P_{n−1}(x) − (n−1)·P_{n−2}(x)). | Three-term recurrence. |
| `LaguerrePolynomial(n)` / `LaguerrePolynomial(n, m)` | Constructs the generalized Laguerre polynomial L_n^m(x) of degree n with parameter m (default m = 0 if omitted), via the recurrence L_0 = 1, L_1 = 1 + m − x, L_n = (1/n)·(((2n+m−1)−x)·L_{n−1}^m − (n−1+m)·L_{n−2}^m). | Three-term recurrence. |
| `HermitePolynomial(n)` | Constructs the Hermite polynomial H_n(x) of degree n via the recurrence H_0 = 1, H_1 = 2x, H_n(x) = 2x·H_{n−1}(x) − 2n·H_{n−2}(x). | Three-term recurrence. |
| `GegenbauerPolynomial(n, m)` | Constructs the Gegenbauer polynomial C_n^m(x) of degree n with parameter m via the recurrence C_0^m = 1, C_1^m = 2mx, C_n^m = (1/n)·(2(n−1+m)·x·C_{n−1}^m − (n+2m−2)·C_{n−2}^m). | Three-term recurrence. |

### 23.10.2 Permutation Polynomials

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DicksonFirst(n, a)` | Constructs the Dickson polynomial of the first kind D_n(x, a) of degree n, defined by D_n(x, a) = Σ_{i=0}^{⌊n/2⌋} (n/(n−i)) · C(n−i, i) · (−a)^i · x^{n−2i}. | Closed-form sum. |
| `DicksonSecond(n, a)` | Constructs the Dickson polynomial of the second kind E_n(x, a) of degree n, defined by E_n(x, a) = Σ_{i=0}^{⌊n/2⌋} C(n−i, i) · (−a)^i · x^{n−2i}. | Closed-form sum. |

### 23.10.3 The Bernoulli Polynomial

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BernoulliPolynomial(n)` | Constructs the n-th Bernoulli polynomial. | Bernoulli number recurrence. |

### 23.10.4 Swinnerton-Dyer Polynomials

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SwinnertonDyerPolynomial(n)` | Constructs the n-th Swinnerton-Dyer polynomial ∏(x ± √2 ± √3 ± √5 ± … ± √p_n), where p_i is the i-th prime and the product runs over all 2^n sign combinations. The polynomial lies in Z[x], has degree 2^n, and is irreducible over Z (but has at least 2^{n−1} factors modulo any prime). See also Example H40E2 for construction details and a generalization. | Product over all sign combinations of linear factors; result is integral by symmetry. |

---

## 23.11 Bibliography

| Key | Reference |
|-----|-----------|
| **[CGG89]** | Bruce W. Char, Keith O. Geddes, and Gaston H. Gonnet. *GCDHEU: Heuristic Polynomial GCD Algorithm Based on Integer GCD Computation.* J. Symbolic Comp., 7(1):31–48, 1989. |
| **[Coh93]** | Henri Cohen. *A Course in Computational Algebraic Number Theory*, volume 138 of Graduate Texts in Mathematics. Springer, Berlin–Heidelberg–New York, 1993. |
| **[Cop96]** | Don Coppersmith. *Finding a small root of a univariate modular equation.* In Advances in Cryptology—EuroCrypt 1996, volume 1070 of LNCS, pages 155–165. Springer, 1996. |
| **[GCL92]** | Keith O. Geddes, Stephen R. Czapor, and George Labahn. *Algorithms for Computer Algebra.* Kluwer, Boston/Dordrecht/London, 1992. |
| **[Knu97]** | Donald E. Knuth. *The Art of Computer Programming*, volume 2. Addison Wesley, Reading, Massachusetts, 3rd edition, 1997. |
| **[KS95]** | Erich Kaltofen and Victor Shoup. *Subquadratic-time factoring of polynomials over finite fields.* In Proceedings of the Twenty-Seventh Annual ACM Symposium on Theory of Computing, pages 398–406. ACM, 1995. |
| **[LLL82]** | Arjen K. Lenstra, Hendrik W. Lenstra, and László Lovász. *Factoring polynomials with rational coefficients.* Mathematische Annalen, 261:515–534, 1982. |
| **[May03]** | Alexander May. *New RSA Vulnerabilities Using Lattice Reduction Methods.* Dissertation, University of Paderborn, 2003. |
| **[PZ89]** | Michael E. Pohst and Hans Zassenhaus. *Algorithmic Algebraic Number Theory.* Encyclopaedia of mathematics and its applications. Cambridge University Press, Cambridge, 1989. |
| **[Sho95]** | Victor Shoup. *A New Polynomial Factorization Algorithm and its Implementation.* J. Symbolic Comp., 20(4):363–397, 1995. |
| **[Ste05]** | Allan Steel. *Conquering Inseparability: Primary Decomposition and Multivariate Factorization over Algebraic Function Fields of Positive Characteristic.* J. Symbolic Comp., 40(3):1053–1075, 2005. |
| **[Tra76]** | Barry M. Trager. *Algebraic factoring and rational function integration.* In R.D. Jenks, editor, Proc. SYMSAC '76, pages 196–208. ACM press, 1976. |
| **[vH01]** | Mark van Hoeij. *Factoring Polynomials and 0-1 vectors.* In Proceedings of the Cryptography and Lattices Conference (CaLC 2001), Brown University, Providence, RI, USA, March 29–30, 2001, pages 142–146. Springer, 2001. |
| **[vH02]** | Mark van Hoeij. *Factoring Polynomials and the knapsack problem.* J. Number Th., 95(2):167–189, 2002. |
| **[vzGG99]** | Joachim von zur Gathen and Jürgen Gerhard. *Modern Computer Algebra.* Cambridge University Press, Cambridge, 1999. |
| **[vzGS92]** | Joachim von zur Gathen and Victor Shoup. *Computing Frobenius Maps And Factoring Polynomials.* Computational Complexity, 2:187–224, 1992. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Berlekamp algorithm (small finite fields) **[Knu97, vzGG99]** | `Factorization` (`Al := "BerlekampSmall"` / `"BerlekampLarge"`), `QMatrix` |
| von zur Gathen/Kaltofen/Shoup algorithm (medium/large finite fields) **[vzGS92, KS95, Sho95]** | `Factorization` (`Al := "GKS"`) |
| Van Hoeij knapsack/LLL factorization over Z/Q **[vH02, vH01, LLL82]** | `Factorization` |
| Trager norm-based factorization over number/function fields **[Tra76]** | `Factorization` |
| Steel inseparable-extension factorization **[Ste05]** | `Factorization` (small characteristic function fields) |
| Hensel lifting (Shoup tree lifting) **[vzGG99 §15.5]** | `HenselLift`, `Factorization` |
| GCDHEU heuristic GCD **[CGG89, GCL92]** | `GCD` / `Gcd` / `GreatestCommonDivisor` |
| Modular GCD algorithm **[vzGG99, GCL92]** | `GCD` / `Gcd` / `GreatestCommonDivisor` |
| Subresultant GCD **[Coh93]** | `GCD` (generic rings) |
| Coppersmith small-roots / LLL **[Cop96, May03, LLL82]** | `SmallRoots` |
| Dedekind criterion **[PZ89]** | `DedekindTest` |
| Chinese Remainder Theorem | `ChineseRemainderTheorem` / `CRT` |
| Three-term recurrences (orthogonal polynomials) | `ChebyshevFirst`, `ChebyshevSecond`, `LegendrePolynomial`, `LaguerrePolynomial`, `HermitePolynomial`, `GegenbauerPolynomial` |
| Dickson polynomial closed form | `DicksonFirst`, `DicksonSecond` |
| Swinnerton-Dyer product construction | `SwinnertonDyerPolynomial` |
