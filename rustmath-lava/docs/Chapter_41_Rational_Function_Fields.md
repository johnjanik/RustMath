# Chapter 41 — Rational Function Fields

**Handbook part:** VI — Global Arithmetic Fields
**Handbook pages:** 1059–1077 (PDF pages 1190–1211)

---

## Scope and overview

Chapter 41 covers rational function fields in Magma: fields of fractions of polynomial rings
over an integral domain admitting a GCD algorithm for polynomials. These are objects of type
`FldFunRat` with elements of type `FldFunRatElt`.

Elements are fractions P/Q whose numerator and denominator lie in the corresponding polynomial
ring over the coefficient ring R. Fractions are always stored in reduced form: numerator and
denominator are coprime, and the denominator is normalised (monic over fields, positive over
**Z**). Both univariate and multivariate cases are supported, and the representation tracks the
polynomial-ring representation of the underlying ring. The chapter notes that using **Z** as the
coefficient ring (rather than **Q**) gives substantially faster arithmetic, even though Z(t) and
Q(t) are mathematically equal.

The chapter also covers Padé-Hermite approximants — simultaneous rational approximation of
tuples of formal power series by polynomial vectors — with the underlying algorithms due to
Derksen **[Der94]** and Beckermann–Labahn **[BL94]**.

---

## 41.1 Introduction

The chapter introduces the type system and normalisation conventions for rational function
fields. No intrinsics are defined in this section.

---

## 41.2 Creation Functions

### 41.2.1 Creation of Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `FunctionField(R)` / `RationalFunctionField(R)` | Create the field F of rational functions in 1 indeterminate over the integral domain R (consisting of quotients of univariate polynomials). Parameter `Global` (BoolElt, default `true`): if `true`, returns the unique global univariate function field over R; if `false`, returns a non-global function field to which a separate indeterminate name can be assigned. Angle-bracket notation supported: `K<t> := FunctionField(IntegerRing());`. | — |
| `FunctionField(R, r)` / `RationalFunctionField(R, r)` | Create the field F of rational functions in r indeterminates over the integral domain R. Parameter `Global` (BoolElt, default `false`): if `true`, returns the unique global function field over R with r variables; if `false`, returns a non-global function field (default, so that multiple function fields with the same number of variables but different names can be created). Explicit coercion is always allowed between function fields having the same number of variables and suitable base rings, mapping the i-th variable of one to the i-th variable of the other. | — |
| `FieldOfFractions(P)` | Given a polynomial ring P, return its field of fractions F consisting of quotients f/g with f, g ∈ P. Angle-bracket notation supported: `K<t> := FieldOfFractions(P);`. Repeated calls for a fixed P return the identical function field. | — |

### 41.2.2 Names

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AssignNames(~F, s)` | Procedure to change the name of the indeterminates of the function field F. The i-th indeterminate is given the name of the i-th element of the string sequence s (for 1 ≤ i ≤ #s); a sequence shorter than the number of indeterminates leaves the remaining names unchanged. Changes only the printing representation — does not assign identifiers. Because this procedure modifies F, the reference ~F is required. | — |
| `Name(F, i)` | Returns the i-th indeterminate of the function field F as an element of F. | — |

### 41.2.3 Creation of Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `F ! [a, b]` / `elt< F \| a, b >` | Given the rational function field F (field of fractions of polynomial ring R), and polynomials a, b ∈ R with b ≠ 0, construct the rational function a/b. | — |
| `F ! a` | Given the rational function field F and a polynomial a ∈ R, create the rational function a/1 in F. | — |
| `K . i` | The i-th generator of the field of fractions K of R over the coefficient ring of R. | — |
| `One(F)` / `Identity(F)` / `Zero(F)` / `Representative(F)` | Standard element constructors: the multiplicative identity, the additive identity, and a representative element of F. | — |

*Worked examples: H41E1 (creating Q(w) as FieldOfFractions of Z[x], coercing polynomials and pairs into the field).*

---

## 41.3 Structure Operations

### 41.3.1 Related Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IntegerRing(F)` / `RingOfIntegers(F)` | Returns the polynomial ring from which the rational function field F was constructed as its field of fractions. | — |
| `BaseRing(F)` / `CoefficientRing(F)` | The coefficient ring of the (ring of integers of the) rational function field F. | — |
| `Rank(F)` | The rank (number of indeterminates) of the rational function field F. | — |
| `ValuationRing(F)` | For a rational function field F with coefficients from a field: the valuation ring of F with respect to the degree valuation — those rational functions g/h for which deg(h) ≥ deg(g). | — |
| `ValuationRing(F, f)` | For a rational function field F with coefficients from a field and an irreducible polynomial f in the ring of integers of F: the valuation ring of F with respect to the valuation associated with f — those rational functions g/h for which f divides g but not h. | — |
| `Category(R)` / `Parent(R)` / `PrimeRing(R)` | Generic ring operations: category, parent, and prime ring of R. | — |

### 41.3.2 Invariants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Characteristic(F)` | The characteristic of the rational function field F. | — |

### 41.3.3 Ring Predicates and Booleans

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsCommutative(F)` / `IsUnitary(F)` / `IsFinite(F)` / `IsOrdered(F)` / `IsField(F)` / `IsEuclideanDomain(F)` / `IsPID(F)` / `IsUFD(F)` / `IsDivisionRing(F)` / `IsEuclideanRing(F)` / `IsPrincipalIdealRing(F)` / `IsDomain(F)` | Standard ring-category predicates; return Boolean values indicating the algebraic properties of F. | — |
| `F eq G` / `F ne G` | Equality and inequality of rational function fields. | — |

### 41.3.4 Homomorphisms

A ring homomorphism with domain F = R(x₁, …, xₙ) requires n + 1 pieces of data: a map on the coefficient ring R and the images of the n indeterminates.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `hom< F -> S \| f, y1, ..., yn >` | Given a function field F = R(x₁, …, xₙ), a ring S, a map f : R → S, and elements y₁, …, yₙ ∈ S: create the homomorphism g : F → S by g(r x₁^a₁ · · · xₙ^aₙ) = f(r) y₁^a₁ · · · yₙ^aₙ on monomials, extended by linearity to polynomials and by g(n/d) = g(n)/g(d) to fractions. | — |
| `hom< F -> S \| y1, ..., yn >` | As above, but omitting the coefficient-ring map; coefficients are mapped to S by the unitary homomorphism sending 1_R to 1_S. Images yᵢ may come from a structure that allows automatic coercion into S. | — |

*Worked examples: H41E2 (homomorphism Q(x,y) → Q(∛2, √5) sending x ↦ ∛2 and y ↦ √5).*

---

## 41.4 Element Operations

### 41.4.1 Arithmetic

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `+ a` / `- a` | Unary plus and negation. | — |
| `a + b` / `a - b` / `a * b` / `a / b` / `a ^ k` | Standard binary arithmetic operations on elements of a rational function field. | — |

### 41.4.2 Equality and Membership

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `a eq b` / `a ne b` | Equality and inequality of rational function field elements. | — |
| `a in F` / `a notin F` | Membership test for elements in a rational function field. | — |

### 41.4.3 Numerator, Denominator and Degree

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Numerator(f)` | Given a rational function f = P/Q in K (field of fractions of R), return the numerator P as an element of the polynomial ring R. | — |
| `Denominator(f)` | Given a rational function f = P/Q in K (field of fractions of R), return the denominator Q as an element of the polynomial ring R. | — |
| `Degree(f)` | For a univariate rational function f: the degree of f, defined as the maximum of the degree of the numerator and the degree of the denominator. | — |
| `TotalDegree(f)` | For a multivariate rational function f: the total degree of f, defined as the total degree of the numerator minus the total degree of the denominator. | — |
| `WeightedDegree(f)` | For a multivariate rational function f: the weighted degree of f, defined as the weighted degree of the numerator minus the weighted degree of the denominator. | — |

### 41.4.4 Predicates on Ring Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsZero(a)` / `IsOne(a)` / `IsMinusOne(a)` / `IsNilpotent(a)` / `IsIdempotent(a)` / `IsUnit(a)` / `IsZeroDivisor(a)` / `IsRegular(a)` | Standard element predicates; return Boolean values indicating properties of the rational function a. | — |

### 41.4.5 Evaluation

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Evaluate(f, r)` | For a univariate rational function f in F: return the rational function obtained by substituting r for the indeterminate. r must be from (or coercible into) the coefficient ring of the integers of F. | — |
| `Evaluate(f, v, r)` | For a multivariate rational function f in F: return the rational function obtained by substituting r for the v-th variable. r must be from (or coercible into) the coefficient ring of the integers of F. | — |

### 41.4.6 Derivative

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Derivative(f)` | For a univariate rational function f: the first derivative with respect to its variable. | — |
| `Derivative(f, k)` | For a univariate rational function f: the k-th derivative with respect to its variable. k must be non-negative. | — |
| `Derivative(f, v)` | For a multivariate rational function f: the first partial derivative with respect to variable number v. | — |
| `Derivative(f, v, k)` | For a multivariate rational function f: the k-th partial derivative with respect to variable number v. k must be non-negative. | — |

### 41.4.7 Partial Fraction Decomposition

The partial fraction routines decompose a univariate rational function f ∈ K(x) into a sum
of terms nₜ/dₜ^kₜ. The result is returned as a sorted sequence Q of triples ⟨d, k, n⟩ where d
is the denominator factor, k its multiplicity, and n the corresponding numerator (with deg(n)
< deg(d)). If f is improper (deg(numerator) ≥ deg(denominator)), the first triple is ⟨1, 1, q⟩
where q is the polynomial quotient.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PartialFractionDecomposition(f)` | For a univariate rational function f in F = K(x): the unique complete partial fraction decomposition over K, with each denominator factor d being **irreducible**. Returns a sorted sequence of triples ⟨d, k, n⟩. | Euclidean algorithm / irreducible factorisation over the coefficient field. |
| `SquarefreePartialFractionDecomposition(f)` | For a univariate rational function f in F = K(x): the unique complete squarefree partial fraction decomposition, with each denominator factor d being **squarefree**. Returns a sorted sequence of triples ⟨d, k, n⟩. | Squarefree factorisation (Yun's algorithm) over the coefficient field. |

*Worked examples: H41E3 (squarefree and complete partial fraction decompositions of ((t+1)^8 − 1) / ((t³−1)(t+1)²(t²−4)²) in Q(t) and Z(t); also in a function field over a multivariate coefficient ring Z(a,b)(t)).*

---

## 41.5 Padé-Hermite Approximants

### 41.5.1 Introduction

A rational function F(z) ∈ k(z) can be identified with its power series expansion f(z) ∈ k((z))
at the place (z). The Padé-Hermite problem is the converse: given a tuple f^T = (f₁, …, fₘ)^T
of formal power series in k[[z]] and an m-tuple of non-negative integers n = (n₁, …, nₘ),
find a non-zero polynomial vector P = (P₁, …, Pₘ) ∈ k[z]^m such that

  P · f^T = O(z^N),  N = n₁ + n₂ + · · · + nₘ + m − 1.

Such a P is a Padé-Hermite approximant of f^T of type n, and a non-trivial one always exists.
The approximant lies in the subspace V_{f,N} = {Q ∈ k[z]^m : Q · f^T = O(z^N)}, which is
generated by the **minimal vector sequence** of V.

The implementation is based on **[Der94]** and **[BL94]**. Approximants are implemented as
sequences (not vectors), and the output is returned in the same ring as the entries of f.

A sequence P = [P₁, …, Pₘ] has **maximum degree** = max(deg(Pᵢ) − dᵢ) where d = [d₁, …, dₘ]
is an optional **distortion**. The **type** of P is the highest index i for which Pᵢ achieves
this maximum degree. A **minimal vector sequence** is a sequence S of m vectors in V such that
S[i] is a non-trivial polynomial vector in V of minimal degree of type i, for i = 1, …, m.

### 41.5.2 Ordering of Sequences

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `MaximumDegree(f)` | Returns the degree of a sequence of polynomials or power series: the maximum of deg(f[i]) − d[i], where d is the optional distortion (parameter `Distortion`, SeqEnum, default `[]`). Returns −∞ if f is weakly equal to the zero-sequence. | — |
| `TypeOfSequence(f)` | Returns two values: the highest index i among those f[i] whose (distorted) degree equals the maximum of all entries, and that maximum degree. Parameter `Distortion` (SeqEnum, default `[]`). Returns (m, −∞) for a zero-sequence. | — |
| `MinimalVectorSequence(f, n)` | Returns a minimal sequence of vectors Q₁, Q₂, …, Qₘ with respect to the sequence f of length m (entries are polynomials or power series), such that the order of Qᵢ · f is at least n. Parameters: `Distortion` (SeqEnum, default `[]`) — distort the degree comparison; `Power` (RngIntElt, default `1`) — consider Qᵢ(z^p) instead of Qᵢ(z). | Algorithm of **[Der94]** and **[BL94]**. |

*Worked examples: H41E4 (MaximumDegree with and without distortion); H41E5 (TypeOfSequence with and without distortion); H41E6–H41E9 (MinimalVectorSequence for sequences of length 2 and 3, with distortion and Power parameters).*

### 41.5.3 Approximants

The Padé-Hermite approximant of type d = [d₁, …, dₘ] is the element of V_{f,N} (generated by
the minimal vector sequence with distortion d) that is smallest with respect to the degree on
sequences. The implementation returns the approximant, the corresponding minimal vector
sequence basis, and the order of the inner-product term.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PadeHermiteApproximant(f, d)` | Given a sequence f of polynomials or power series and a type d = [d₁, …, dₘ], returns: (1) the Padé-Hermite form P of f with distortion d, smallest with respect to degree on sequences; (2) the corresponding minimal vector sequence; (3) the order of P · f. Parameter `Power` (RngIntElt, default `1`). | Algorithm of **[BL94]** (uniform approach for fast computation of matrix-type Padé approximants) and **[Der94]**. |
| `PadeHermiteApproximant(f, m)` | Variant where m is an integer (not a type vector): returns the Padé-Hermite form of minimal degree in the minimal vector sequence such that its inner product with f has order at least m. Returns (1) the approximant and (2) the corresponding minimal vector sequence. Parameter `Power` (RngIntElt, default `1`). Also accepts f as a sequence of vectors of polynomials or power series. | Algorithm of **[BL94]** and **[Der94]**. |

*Worked examples: H41E10 (example from p. 813 of [BL94]: f = [1, u, u/(1−u⁴)+…, u/(1+u⁴)+…], type [2,2,2,2], with Power variant); H41E11 (example from p. 815 of [BL94]: type [2,2,3,3], Power:=2 variant); H41E12 (example from p. 816 of [BL94]: type [2,2,3,3], Power:=2); H41E13 (scalar and vector f, including f = [Sin, Cos, Exp] and vectors of polynomials); H41E14 (approximants for trig/exp power series with distortion).*

---

## 41.6 Bibliography

| Key | Reference |
|-----|-----------|
| **[BL94]** | Bernhard Beckermann and George Labahn. *A uniform approach for the fast computation of matrix-type Padé approximants.* SIAM J. Matrix Anal. Appl. **15**(3):804–823, 1994. |
| **[Der94]** | Harm Derksen. *An algorithm to compute generalized Padé-Hermite Forms.* Technical Report 9403, Department of Mathematics, Catholic University Nijmegen, Jan 1994. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Euclidean / irreducible factorisation over coefficient field | `PartialFractionDecomposition` |
| Squarefree factorisation (Yun-style) | `SquarefreePartialFractionDecomposition` |
| Uniform fast Padé approximant algorithm **[BL94]** | `PadeHermiteApproximant(f,d)`, `PadeHermiteApproximant(f,m)`, `MinimalVectorSequence` |
| Generalised Padé-Hermite forms **[Der94]** | `PadeHermiteApproximant(f,d)`, `PadeHermiteApproximant(f,m)`, `MinimalVectorSequence` |
| Degree and type ordering on polynomial/power-series sequences | `MaximumDegree`, `TypeOfSequence` |
