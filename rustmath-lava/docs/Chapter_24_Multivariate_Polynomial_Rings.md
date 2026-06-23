# Chapter 24 — Multivariate Polynomial Rings

**Handbook part:** III — Basic Rings
**Handbook pages:** 443–468 (PDF pages 574–601)

---

## Scope and overview

Chapter 24 describes multivariate polynomial rings in Magma. A multivariate polynomial ring
in any number of variables n ≥ 1 can be created over an arbitrary coefficient ring R, denoted
P = R[x₁, …, xₙ]. Only certain functions require the coefficient ring to satisfy additional
conditions (e.g. a domain for GCD, a field of characteristic zero for integration).

Polynomials are stored in **distributive form** using arrays of coefficient-monomial pairs. The
total degree of any monomial may be up to 2³⁰ − 1 = 1 073 741 823. Since V2.7 (June 2000) a
generalised monomial representation uses differing byte sizes depending on the monomials
encountered; monomial overflow is rigorously detected and the byte size is extended
automatically, so the maximum degree need not be known in advance.

Various monomial orders can be applied. The default is lexicographic order; full details of
orders and their role in Gröbner bases appear in Chapter 105 (Ideals and Gröbner Bases).
Graded/weighted polynomial rings are likewise covered in §105.3.2. Invariant rings of finite
groups acting on multivariate polynomial rings are covered in Chapter 110; affine algebras and
modules over them appear in Chapters 108–109.

This chapter covers ring creation, structure operations, arithmetic and element accessors,
GCD and content, factorization, resultants/discriminants, and integer-specific norm functions.

---

## 24.1 Introduction

### 24.1.1 Representation

Multivariate polynomials in Magma belong to the category `RngMPol`. A **monomial** (power
product) of P = R[x₁, …, xₙ] is an expression x₁^e₁ · · · xₙ^eₙ with eᵢ ≥ 0. A **term** is
a coefficient (from R) multiplied by a monomial. Polynomials are stored in distributive form
as arrays of (coefficient, monomial) pairs. The current monomial order determines how terms
are printed and ordered.

---

## 24.2 Polynomial Rings and Polynomials

### 24.2.1 Creation of Polynomial Rings

Multivariate polynomial rings are created from a coefficient ring, a number of indeterminates,
and an optional monomial order. If no order is specified, lexicographic order is used.
The angle-bracket notation `P<x, y> := PolynomialRing(R, 2)` assigns names to indeterminates
simultaneously with ring creation.

By default a non-global ring is returned. Setting `Global := true` returns the unique global
polynomial ring over R with n variables (so two calls with the same arguments return
identical objects). Explicit coercion is always allowed between polynomial rings having the
same number of variables and compatible base rings; coercion maps the i-th variable of one
ring to the i-th variable of the other.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PolynomialRing(R, n)` / `PolynomialAlgebra(R, n)` | Create a multivariate polynomial ring in n > 0 indeterminates over R with lexicographic order. Parameter `Global` (BoolElt, default `false`): if `true`, return the unique global ring. | — |
| `PolynomialRing(R, n, order)` / `PolynomialAlgebra(R, n, order)` | Create a multivariate polynomial ring in n > 0 indeterminates over R with the specified monomial order. See §105.2 for order details. | — |

*Worked examples: H24E1 (angle brackets vs. `AssignNames`; default printing of `$.1`, `$.2`); H24E2 (global vs. non-global rings; explicit coercion between non-global rings).*

### 24.2.2 Print Names

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AssignNames(~P, s)` | Procedure (takes a reference `~P`). Sets the print names of the indeterminates of polynomial ring P; the i-th indeterminate gets the name `s[i]`. The sequence `s` may be shorter than the rank of P, in which case remaining names are unchanged. Does **not** assign Magma identifiers. | — |
| `Name(P, i)` | Returns the i-th indeterminate of P as an element of P. | — |

### 24.2.3 Graded Polynomial Rings

It is possible to assign weights to the variables of a multivariate polynomial ring, giving
monomials a weighted degree. Such graded/weighted rings are covered in §105.3.2 (Gröbner
bases chapter), as the subject is intimately related to ideals.

### 24.2.4 Creation of Polynomials

The angle-bracket construction is the easiest way to create polynomials. The functions below
provide additional options.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `P . i` | Returns the i-th indeterminate (1 ≤ i ≤ n) of polynomial ring P as an element of P. | — |
| `R ! s` / `elt< R \| s >` | Coerces a scalar `s` (coercible into the coefficient ring R) into P as a constant polynomial; if `s` is already in P, it is returned unchanged. | — |
| `MultivariatePolynomial(P, f, i)` / `MultivariatePolynomial(P, f, v)` | Given a univariate polynomial `f ∈ R[x]` and a multivariate polynomial ring P = R[x₁,…,xₙ], returns the element of P corresponding to f viewed as a polynomial in the indeterminate xᵢ (specified by integer i or polynomial v = xᵢ). Inverse operation: `UnivariatePolynomial`. | — |
| `One(P)` / `Identity(P)` | The multiplicative identity of P. | — |
| `Zero(P)` / `Representative(P)` | The zero element / a representative of P. | — |

---

## 24.3 Structure Operations

### 24.3.1 Related Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BaseRing(P)` / `CoefficientRing(P)` | Returns the coefficient ring of polynomial ring P. | — |
| `Category(P)` | The Magma category of P (i.e. `RngMPol`). | — |
| `Parent(P)` | The parent structure of P. | — |
| `PrimeRing(P)` | The prime ring of P. | — |

### 24.3.2 Numerical Invariants

Note: the `#` operator returns a value only for finite (quotients of) polynomial rings.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Rank(P)` | Returns the number of indeterminates of polynomial ring P over its coefficient ring. | — |
| `Characteristic(P)` / `# P` | Characteristic of P; `#P` gives cardinality (only for finite rings). | — |

### 24.3.3 Ring Predicates and Booleans

The usual ring predicates returning Boolean values are available. Not all predicates are
meaningful for all coefficient rings.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsCommutative(P)` / `IsUnitary(P)` | Whether P is commutative / unitary. | — |
| `IsFinite(P)` / `IsOrdered(P)` | Whether P is finite / ordered. | — |
| `IsField(P)` / `IsEuclideanDomain(P)` | Whether P is a field / Euclidean domain. | — |
| `IsPID(P)` / `IsUFD(P)` | Whether P is a principal ideal domain / unique factorization domain. | — |
| `IsDivisionRing(P)` / `IsEuclideanRing(P)` | Whether P is a division ring / Euclidean ring. | — |
| `IsDomain(P)` | Whether P is an integral domain. | — |
| `IsPrincipalIdealRing(P)` | Whether P is a principal ideal ring. | — |
| `P eq Q` / `P ne Q` | Ring equality / inequality. | — |

### 24.3.4 Changing Coefficient Ring

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ChangeRing(P, S)` | Given polynomial ring P = R[x₁,…,xₙ] and ring S, constructs Q = S[x₁,…,xₙ]. Requires that all elements of R can be automatically coerced into S. | — |

### 24.3.5 Homomorphisms

A ring homomorphism from P = R[x₁,…,xₙ] requires a coefficient ring map f : R → S and
images y₁,…,yₙ ∈ S of the n indeterminates.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `hom< P -> S \| f, y1, ..., yn >` | Creates the homomorphism g : P → S sending xᵢ ↦ yᵢ and applying f on coefficients: g(r x₁^a₁…xₙ^aₙ) = f(r) y₁^a₁…yₙ^aₙ, extended by linearity. | — |
| `hom< P -> S \| y1, ..., yn >` | As above, but omitting the coefficient ring map; coefficients are mapped by the unitary homomorphism 1_R ↦ 1_S. | — |

*Worked example: H24E3 (mapping Q[x,y] into the number field Q(∛2, √5) by x ↦ ∛2, y ↦ √5).*

---

## 24.4 Element Operations

### 24.4.1 Arithmetic Operators

For polynomial rings over fields, division by elements of the coefficient field is allowed (result
stays in P). The `div` operator requires exact divisibility: if b does not divide a, an error
results (unlike the univariate case).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `+ a` / `- a` | Unary plus / negation. | — |
| `a + b` / `a - b` / `a * b` / `a ^ k` / `a / b` / `a div b` | Standard binary ring arithmetic; `/` divides by a coefficient-field element; `div` requires b | a exactly. | — |
| `a +:= b` / `a -:= b` / `a *:= b` / `a div:= b` | In-place assignment versions of the arithmetic operators. | — |

### 24.4.2 Equality and Membership

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `a eq b` / `a ne b` | Equality / inequality of polynomial elements. | — |
| `a in R` / `a notin R` | Membership test: whether element a belongs to ring R. | — |

### 24.4.3 Predicates on Ring Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsDivisibleBy(a, b)` | Returns whether a is divisible by b in P (i.e. whether q ∈ P exists with a = q·b). If true, also returns the quotient q. | — |
| `IsAlgebraicallyDependent(S)` | Returns true iff the set S of multivariate polynomials is algebraically dependent. | — |
| `IsZero(f)` / `IsOne(f)` / `IsMinusOne(f)` | Tests whether f is zero / one / minus one. | — |
| `IsNilpotent(f)` / `IsIdempotent(f)` | Tests nilpotency / idempotency of f. | — |
| `IsUnit(f)` / `IsZeroDivisor(f)` / `IsRegular(f)` | Tests whether f is a unit / zero divisor / regular element. | — |
| `IsIrreducible(f)` / `IsPrime(f)` | Tests irreducibility / primeness of f. | — |

### 24.4.4 Coefficients, Monomials and Terms

Many functions come in three forms: (1) no variable specified (returns values in the coefficient
ring), (2) variable specified by integer i, (3) variable specified by polynomial element v.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Coefficients(f)` | Returns a sequence of the base coefficients (elements of R) of f, in the same order as `Monomials(f)`. | — |
| `Coefficients(f, i)` / `Coefficients(f, v)` | Returns a sequence of coefficients of f viewed as a polynomial in xᵢ = v (ascending powers), as elements of P; xᵢ does not appear in the coefficients. | — |
| `Coefficient(f, i, k)` / `Coefficient(f, v, k)` | Returns the coefficient of xᵢᵏ in f (as an element of P, with xᵢ absent). | — |
| `LeadingCoefficient(f)` | Returns the leading coefficient of f as an element of R (coefficient of the leading monomial with respect to the monomial order). | — |
| `LeadingCoefficient(f, i)` / `LeadingCoefficient(f, v)` | Returns the coefficient (in P, without xᵢ) of the highest power of xᵢ occurring with non-zero coefficient in f. | — |
| `Length(f)` | Returns the number of terms of f. | — |
| `TrailingCoefficient(f)` | Returns the trailing coefficient of f as an element of R (coefficient of the last monomial with respect to the order). | — |
| `TrailingCoefficient(f, i)` / `TrailingCoefficient(f, v)` | Returns the coefficient (in P, without xᵢ) of the least power of xᵢ occurring with non-zero coefficient in f. | — |
| `MonomialCoefficient(f, m)` | Returns the coefficient (in R) with which monomial m occurs in f. | — |
| `Monomials(f)` | Returns a sequence of the monomials of f (as elements of P), in the same order as `Coefficients(f)`. | — |
| `CoefficientsAndMonomials(f)` | Returns parallel sequences C and M of the coefficients and monomials of f. More efficient than calling `Coefficients` and `Monomials` separately since only one scan is performed. | — |
| `LeadingMonomial(f)` | Returns the leading monomial of f (the first monomial with respect to the order). | — |
| `Terms(f)` | Returns the sequence of non-zero terms of f as elements of P, ordered by the monomial order. Each term equals Coefficients(f)[i] * Monomials(f)[i]. | — |
| `Terms(f, i)` / `Terms(f, v)` | Returns a sequence of terms of f viewed as a polynomial in xᵢ, in ascending order of powers of xᵢ. | — |
| `Term(f, i, k)` / `Term(f, v, k)` | Returns the term of f involving the k-th power of xᵢ (k ≥ 0). | — |
| `LeadingTerm(f)` | Returns the leading term of f (product of leading monomial and leading coefficient) as an element of P. | — |
| `LeadingTerm(f, i)` / `LeadingTerm(f, v)` | Returns the leading term of f when viewed as a polynomial in xᵢ (the term involving the largest power of xᵢ with non-zero coefficient). | — |
| `TrailingTerm(f)` | Returns the trailing term of f (the last monomial term with respect to the order). | — |
| `TrailingTerm(f, i)` / `TrailingTerm(f, v)` | Returns the trailing term of f when viewed as a polynomial in xᵢ (the term involving the least power of xᵢ with non-zero coefficient). | — |
| `Exponents(f)` | Given a single-term polynomial f in a ring of rank n, returns the exponents of the monomial as a sequence of n integers. (The coefficient is ignored.) | — |
| `Monomial(P, E)` | Given polynomial ring P = R[x₁,…,xₙ] and a sequence E of n non-negative integers, returns the monomial x₁^E[1]…xₙ^E[n]. Semi-inverse of `Exponents`. | — |
| `Polynomial(C, M)` | Given a length-k sequence C of coefficients in R and a length-k sequence M of monomials of a polynomial ring, returns the multivariate polynomial with coefficients C and monomials M. (For any f: `Polynomial(Coefficients(f), Monomials(f))` equals f.) | — |

*Worked examples: H24E4 (coefficient and term functions on f = (2x+y)z³ + 11xyz + x²y² in Q[x,y,z]); H24E5 (resultants, `UnivariatePolynomial`, roots, and GCDs in GF(5)[x,y]).*

### 24.4.5 Degrees

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Degree(f, i)` / `Degree(f, v)` | Returns the degree of f in the variable xᵢ = v (the largest exponent of xᵢ in any monomial of f). Returns −1 if f is zero. | — |
| `TotalDegree(f)` | Returns the total degree of f: the maximum over all monomials of the sum of exponents. (Ignores variable weights if any.) Returns −1 if f is zero. | — |
| `LeadingTotalDegree(f)` | Returns the total degree of the leading monomial of f. Returns −1 if f is zero. | — |

### 24.4.6 Univariate Polynomials

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsUnivariate(f)` | Returns whether f ∈ R[x₁,…,xₙ] is actually univariate in one of its indeterminates. If true, also returns a univariate version u ∈ R[x] and the (first) index i such that f is univariate in xᵢ. | — |
| `IsUnivariate(f, i)` / `IsUnivariate(f, v)` | Returns whether f is univariate in xᵢ specifically. If true, also returns the univariate version u ∈ R[x]. | — |
| `UnivariatePolynomial(f)` | Given f ∈ R[x₁,…,xₙ] known to be univariate in some xᵢ, returns a univariate version u ∈ R[x] with the same coefficients. | — |

### 24.4.7 Derivative, Integral

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Derivative(f, i)` / `Derivative(f, v)` | Returns the partial derivative of f with respect to xᵢ = v as an element of P. | — |
| `Derivative(f, k, i)` / `Derivative(f, k, v)` | Returns the k-th partial derivative of f with respect to xᵢ = v (k > 0), as an element of P. | — |
| `Integral(f, i)` / `Integral(f, v)` | Returns the formal integral of f with respect to xᵢ = v, as an element of P. Requires the coefficient ring to have characteristic zero. | — |
| `JacobianMatrix([f])` | Creates the Jacobian matrix whose (i, j)-th entry is the partial derivative of the i-th polynomial in the list with respect to the j-th indeterminate of its parent ring. | — |

### 24.4.8 Evaluation, Interpolation

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Evaluate(f, s)` | Evaluates f ∈ P = R[x₁,…,xₙ] at a sequence or tuple s of length n by substituting xᵢ = s[i]. If s[i] can be lifted into R, the result is in R; otherwise a generic evaluation is attempted. | — |
| `Evaluate(f, i, r)` / `Evaluate(f, v, r)` | Evaluates f by substituting only the variable xᵢ = v with ring element r. If r is coercible into the coefficient ring, the result is in P; otherwise the other variables of P must be coercible into the parent of r. | — |
| `Interpolation(I, V, i)` / `Interpolation(I, V, v)` | Given a field K, multivariate polynomial ring P = K[x₁,…,xₙ], interpolation points I (a sequence of k elements of K), and interpolation values V (a sequence of k elements of P not involving xᵢ), returns the unique f ∈ P of degree < k in xᵢ with f(I[j]) = V[j] for j = 1,…,k. | — |

*Worked example: H24E6 (interpolation in Q[x,y,z]: polynomial recovering y, z, y+z at x=1,2,3).*

### 24.4.9 Quotient and Reductum

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `f div g` / `ExactQuotient(f, g)` | Returns the exact quotient of f by g in R[x₁,…,xₙ] (R must be a domain). If q ∈ P with f = q·g exists, returns q; otherwise an error results. | — |
| `Reductum(f)` | Returns the reductum of f: the polynomial obtained by removing the leading term of f. | — |
| `Reductum(f, i)` / `Reductum(f, v)` | Returns the reductum of f obtained by removing the leading term with respect to the variable xᵢ = v. | — |

### 24.4.10 Diagonalizing a Polynomial of Degree 2

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SymmetricBilinearForm(f)` | Returns the symmetric bilinear form (as a matrix) of a multivariate polynomial of degree 2. For a non-homogeneous polynomial, a homogenizing variable is introduced, giving a 4×4 matrix. | — |
| `DiagonalForm(f)` | Returns the diagonal form of the multivariate polynomial of degree 2, plus the transformation matrix. | Gram orthogonalization. |

*Worked example: H24E7 (symmetric bilinear form and diagonal form of a non-homogeneous degree-2 polynomial in Q[x,y,z]; verification via `OrthogonalizeGram`).*

---

## 24.5 Greatest Common Divisors

The GCD functions can be applied to multivariate polynomials over any ring that itself has a
GCD algorithm.

### 24.5.1 Common Divisors and Common Multiples

For polynomials over **Z** or **Q**, a combination of three algorithms is used:
1. The heuristic **GCDHEU** evaluation algorithm **[CGG89, GCL92 §7.7]** — suitable for
   moderate-degree dense polynomials with several variables.
2. The **EEZ-GCD** algorithm of Wang **[Wan80, MY73, GCL92 §7.6]** — based on evaluation
   and sparse ideal-adic multivariate Hensel lifting **[Wan78, GCL92 §6.8]** — suitable for
   sparse polynomials.
3. A **recursive multivariate evaluation-interpolation** algorithm (cf. **[GCL92 §7.4]**) —
   works generically over Z or most fields.

For polynomials over any finite field or any characteristic-zero field besides Q, algorithm (3)
is used, exploiting any fast modular algorithm for the base univariate polynomials.

For polynomials over another polynomial ring or rational function field, the polynomials are
first "flattened" to a multivariate polynomial ring over the base coefficient ring, then the
appropriate algorithm is applied.

For polynomials over any other ring, the **generic subresultant algorithm** **[Coh93 §3.3]**
is called recursively on a subring with one fewer variable.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `GreatestCommonDivisor(f, g)` / `Gcd(f, g)` / `GCD(f, g)` | Returns the greatest common divisor of f and g in P (normalized, hence unique). If either input is zero, the other is returned; if both zero, zero is returned. | GCDHEU **[CGG89, GCL92]**, EEZ-GCD **[Wan80, MY73, GCL92]**, or recursive evaluation-interpolation **[GCL92]**, depending on the coefficient ring; subresultant **[Coh93]** for other rings. |
| `GCD(Q)` | Returns the GCD of all polynomials in the sequence Q. If Q is empty (with universe P), returns the zero element of P. | As above. |
| `LeastCommonMultiple(f, g)` / `Lcm(f, g)` / `LCM(f, g)` | Returns the least common multiple of f and g in P (normalized). LCM of zero and anything is zero. Computed as `Normalize((f div GCD(f,g)) * g)` for non-zero inputs. | Reduction to GCD. |
| `LCM(Q)` | Returns the LCM of all polynomials in sequence Q. If Q is empty (with universe P), returns the one element of P. | Reduction to GCD. |
| `Normalize(f)` | Returns the unique normalized associate of f. If R is a field, the result is monic; if R = Z, the leading coefficient is positive; if R is a polynomial ring, the leading coefficient is recursively normalized. | — |
| `ClearDenominators(f)` | For f over a field K that is the fraction field of a domain D: returns the polynomial g = L·f over D (cleared denominators) and the LCD L. | — |
| `ClearDenominators(Q)` | Returns the sequence of polynomials obtained by independently clearing denominators in each polynomial in sequence Q. | — |

### 24.5.2 Content and Primitive Part

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Content(f)` | Returns the content of f: the GCD of the coefficients of f as an element of the coefficient ring. | — |
| `PrimitivePart(f)` | Returns the primitive part of f: f divided by its content. | — |
| `ContentAndPrimitivePart(f)` / `Contpp(f)` | Returns both the content (GCD of coefficients) and the primitive part of f. | — |

---

## 24.6 Factorization and Irreducibility

Factorization is implemented for multivariate polynomials over the following coefficient rings:
a finite field F_q, the integers Z, the rationals Q, an algebraic number field Q(α), or a
polynomial ring / function field / finite-dimensional affine algebra (which is a field) over
any of the above.

- **Bivariate polynomials:** a polynomial-time algorithm in the spirit of van Hoeij's Knapsack
  factoring algorithm **[vH02]**.
- **Over Z or Q:** an algorithm based on evaluation and sparse ideal-adic multivariate Hensel
  lifting, similar to **[Wan78, GCL92 §6.8]**.
- **Over any finite field:** a similar lifting algorithm with modifications for non-zero
  characteristic (see e.g. **[BM97]**).
- **Over algebraic number fields and affine algebras:** a multivariate version of the
  norm-based algorithm of Trager **[Tra76]**, which performs a suitable substitution and
  multivariate resultant computation and then factors the resulting integral multivariate
  polynomial.
- Each algorithm reduces to univariate factorization over the base ring (see Chapter 23 for
  those algorithms).
- For polynomials over another polynomial ring or function field, the polynomials are first
  "flattened" to a multivariate polynomial ring over the base coefficient ring.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Factorization(f)` | Returns the factorization of f as a sequence of pairs `<qᵢ, kᵢ>` of normalized irreducible factors and their multiplicities, plus the normalizing unit u (so f = u · ∏ qᵢ^kᵢ). | Bivariate: Knapsack-style **[vH02]**; over Z/Q: Hensel-lifting evaluation **[Wan78, GCL92]**; over finite fields: **[BM97]**; over number fields/affine algebras: Trager norm **[Tra76]**. |
| `SquarefreeFactorization(f)` | Returns the squarefree factorization of f as a sequence of tuples `<factor, multiplicity>`. Factors contain no square of any polynomial of positive degree. Same allowable coefficient rings as `Factorization`. | — |
| `SquarefreePart(f)` | Returns the largest normalized squarefree divisor of f. | — |
| `IsIrreducible(f)` | Returns whether f is irreducible over its coefficient ring. Same allowable coefficient rings as `Factorization`. | Calls `Factorization`. |
| `SetVerbose("PolyFact", v)` | Sets the verbose printing level for all polynomial factorization algorithms. Legal levels: 0, 1, 2, 3. | — |

*Worked examples: H24E8 (product of 8 trinomials, 461 terms, total degree 15, factorized in 0.29 s); H24E9 (Vandermonde determinant of rank 6 factorized into 15 linear factors); H24E10 (square of triangle area via Heron's formula factors); H24E11 (determinant of a Frobenius matrix over GF(2)).*

---

## 24.7 Resultants and Discriminants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Resultant(f, g, i)` / `Resultant(f, g, v)` | Returns the resultant of f and g in P = R[x₁,…,xₙ] with respect to the variable xᵢ = v (the determinant of the Sylvester matrix for f and g viewed as polynomials in xᵢ). Result is an element of P. R must be a domain. | Modular interpolation method **[GCL92, pp. 412–413]**. |
| `Discriminant(f, i)` / `Discriminant(f, v)` | Returns the discriminant of f ∈ R[x₁,…,xₙ] viewed as a polynomial in xᵢ = v. Result is an element of P. R must be a domain. | Computed via `Resultant`. |

---

## 24.8 Polynomials over the Integers

These functions are available for multivariate polynomials over Z only.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Sign(f)` | Returns the sign of the leading coefficient of f. | — |
| `AbsoluteValue(f)` / `Abs(f)` | Returns f or −f, whichever has a non-negative leading coefficient. | — |
| `MaxNorm(f)` | Returns the maximum of the absolute values of the coefficients of f (the infinity-norm / max-norm). | — |
| `SumNorm(f)` | Returns the sum of the (absolute values of the) base coefficients of f (the 1-norm). | — |

---

## 24.9 Bibliography

| Key | Reference |
|-----|-----------|
| **[BM97]** | Laurent Bernardin and Michael B. Monagan. *Efficient Multivariate Factorization Over Finite Fields.* In Proceedings of AAECC, volume 1255 of LNCS, pages 15–28. Springer-Verlag, 1997. |
| **[CGG89]** | Bruce W. Char, Keith O. Geddes, and Gaston H. Gonnet. *GCDHEU: Heuristic Polynomial GCD Algorithm Based on Integer GCD Computation.* J. Symbolic Comp., 7(1):31–48, 1989. |
| **[Coh93]** | Henri Cohen. *A Course in Computational Algebraic Number Theory*, volume 138 of Graduate Texts in Mathematics. Springer, Berlin–Heidelberg–New York, 1993. |
| **[GCL92]** | Keith O. Geddes, Stephen R. Czapor, and George Labahn. *Algorithms for Computer Algebra.* Kluwer, Boston/Dordrecht/London, 1992. |
| **[MY73]** | J. Moses and D.Y.Y. Yun. *The EZ GCD algorithm.* Proc. ACM Annual Conference, 73(2):159–166, 1973. |
| **[Tra76]** | Barry M. Trager. *Algebraic factoring and rational function integration.* In R.D. Jenks, editor, Proc. SYMSAC '76, pages 196–208. ACM press, 1976. |
| **[vH02]** | Mark van Hoeij. *Factoring Polynomials and the knapsack problem.* J. Number Th., 95(2):167–189, 2002. |
| **[Wan78]** | Paul S. Wang. *An improved multivariate polynomial factoring algorithm.* Math. Comp., 32(144):1215–1231, 1978. |
| **[Wan80]** | Paul S. Wang. *The EEZ-GCD algorithm.* SIGSAM Bulletin, 14(2):50–60, 1980. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Lexicographic / monomial orders (see Ch. 105) | `PolynomialRing`, `PolynomialAlgebra` |
| GCDHEU heuristic evaluation GCD **[CGG89, GCL92]** | `GCD`, `Gcd`, `GreatestCommonDivisor` (over Z/Q, dense) |
| EEZ-GCD / sparse multivariate Hensel lifting **[Wan80, MY73, Wan78, GCL92]** | `GCD`, `Gcd`, `GreatestCommonDivisor` (over Z/Q, sparse) |
| Recursive evaluation-interpolation GCD **[GCL92]** | `GCD`, `Gcd`, `GreatestCommonDivisor` (finite fields, other fields) |
| Generic subresultant GCD **[Coh93]** | `GCD`, `Gcd`, `GreatestCommonDivisor` (other rings) |
| Modular interpolation resultant **[GCL92]** | `Resultant`, `Discriminant` |
| Knapsack-style bivariate factorization **[vH02]** | `Factorization` (bivariate) |
| Evaluation + sparse Hensel lifting factorization **[Wan78, GCL92]** | `Factorization` (over Z/Q) |
| Finite-field multivariate factorization **[BM97]** | `Factorization` (over Fq) |
| Trager norm-based factorization **[Tra76]** | `Factorization` (over number fields / affine algebras) |
| Gram orthogonalization | `DiagonalForm`, `SymmetricBilinearForm` |
