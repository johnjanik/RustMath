# Chapter 20 — Rational Field

**Handbook part:** III — Basic Rings
**Handbook pages:** 351–360 (PDF pages 482–493)

---

## Scope and overview

Chapter 20 describes the functions relating to the field of rational numbers Q in Magma.
The rational field Q is automatically created when Magma starts, so arithmetic can be
performed without explicitly constructing the structure first. Q.1 returns 1 for
compatibility with other rings and fields.

Rational numbers are stored as numerator/denominator pairs and are always kept in reduced
form: the numerator and denominator are coprime, and the denominator is always positive.
A rational number whose denominator is 1 still has type RatFld (it is never silently
converted to an integer).

Automatic coercion between Q and any characteristic-0 ring R is supported: Magma promotes
operands to the larger of Q and R before computing, unless the result would lie in a
structure strictly larger than both (e.g. Q(x) when R = Z[x], or the number field K when
R = O_K). The chapter covers creation of the field and its elements, the full range of
structure operations inherited from Magma's number-field hierarchy, arithmetic and
comparison operators on elements, rounding/truncation, rational reconstruction, valuations,
and sequence conversions.

---

## 20.1 Introduction

The rational field Q is automatically created at Magma startup. Q.1 returns 1 for
compatibility with other rings and fields.

### 20.1.1 Representation

Rational numbers are stored as pairs of numerator and denominator. Whenever a rational
number is created it is put in reduced form (coprime numerator and denominator, positive
denominator). A rational number with denominator 1 represents a rational integer but
its type is never automatically changed to integer.

### 20.1.2 Coercion

Automatic coercion occurs between elements of Q and elements of any ring R of
characteristic 0. The result lies in the larger of Q and R (usually R, unless R is a
subring of Q such as Z). Exceptions arise when the result would lie in a structure
strictly larger than both Q and R, for example R = Z[x] (result in Q(x)) or R = O_K an
order in a number field.

*Worked examples: H20E1 (three successful automatic coercions including cyclotomic field
and real, and one failure case with Z[x]).*

### 20.1.3 Homomorphisms

Since homomorphisms from Q must be unitary, a ring homomorphism from Q to R is fully
determined by the image of 1, which must be 1 in R:

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `hom< Q -> R \| >` | Defines a (unitary) ring homomorphism from Q to R. The image is entirely determined by 1 ↦ 1. Magma permits non-proper homomorphisms via `hom` for convenience (e.g. sending r/s to rs⁻¹ mod m). | — |

*Worked examples: H20E2 (coercing rationals with denominator coprime to 11 into Z/11Z
via a hom).*

---

## 20.2 Creation Functions

### 20.2.1 Creation of Structures

The rational field is unique in Magma: multiple calls to the creation function return the
same object, not an isomorphic copy.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Rationals()` `RationalField()` | Create (or return the unique existing) field Q of rational numbers. | — |
| `MaximalOrder(Q)` `IntegerRing(Q)` `IntegerRing()` `Integers()` `RingOfIntegers(Q)` | Create (or return) the ring Z of rational integers. | — |
| `FieldOfFractions(Z)` | Returns Q when the argument is Z or Q itself. | — |
| `Completion(Q, P)` | Compute the completion of Q at the integral prime ideal P, together with the injection into the completion. Parameter: `Precision` (RngIntElt, default ∞) — specifies the working p-adic precision. | — |

### 20.2.2 Creation of Elements

Rational numbers and integers can be created as literals without defining the parent field
first, since Q and Z are loaded at Magma startup.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `a / b` | Given integers a and b ≠ 0, form the rational number a/b in reduced form. | — |
| `Q ! [a]` | Inverse of `Eltseq`; returns Q!a (the single integer a coerced into Q). | — |
| `Q ! [a, b]` `elt< Q \| a, b >` | Given integers a, b with b ≠ 0, construct the rational number a/b in reduced form. | — |
| `Q ! a` | Given integer a, create a/1 in Q. Also coerces any element from a quadratic, cyclotomic, or number field (or an order thereof) that is rational into Q. | — |
| `One(Q)` `Identity(Q)` | Returns 1 in Q. | — |
| `Zero(Q)` `Representative(Q)` | Returns 0 in Q. | — |
| `RootOfUnity(n, Q)` | Returns a primitive n-th root of unity in Q. For the rational field, n must be 1 or 2; returns 1 or −1 accordingly. | — |
| `Random(Q, m)` | Returns a random rational number with numerator in [−u..u] and denominator in [1..u], where u = |m|. | — |

---

## 20.3 Structure Operations

### 20.3.1 Related Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Category(Q)` `Parent(Q)` `PrimeField(Q)` | Generic structure-inspection functions. | — |
| `IntegralBasis(Q)` | Returns an integral basis for Q as a number field: the sequence [1]. | — |
| `MinimalField(q)` | Returns the least cyclotomic field containing the cyclotomic field element q; returns the rational field if q is rational. | — |
| `MinimalField(S)` | Returns the minimal cyclotomic field containing all elements of the enumerated set S; returns the rational field if all elements are rational. | — |
| `BaseField(Q)` | Returns the coefficient field of Q (which is Q itself), in analogy with number fields. | — |
| `Basis(Q)` `AbsoluteBasis(Q)` | A basis for Q as a Q-vector space: [1]. | — |
| `UnitGroup(Q)` | The unit group of the maximal order of Q (i.e. of Z). | — |
| `ClassGroup(Q)` | The class group of Z (which is trivial). | — |
| `AutomorphismGroup(Q)` `AutomorphismGroup(Q, Q)` | The group of Q-automorphisms of Q: a trivial finitely presented group, together with the parent structure for Q-automorphisms and a map from the group to actual field automorphisms. The only Q-automorphism is the identity. | — |
| `Algebra(Q, Q)` | Returns an associative Q-algebra isomorphic to Q and the map from the algebra to Q. | — |
| `VectorSpace(Q, Q)` | Returns a Q-vector space isomorphic to Q and the map from the vector space to Q. | — |
| `Decomposition(Q, p)` | For a prime p or `Infinity()`: compute the decomposition of Q as a number field. Returns a list of length one containing a 2-tuple (p, ramification degree = 1). | — |

### 20.3.2 Numerical Invariants

These functions are defined for Q mainly because it arises as a degenerate case of
quadratic or cyclotomic field constructions (see Chapters 35 and 36).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Characteristic(Q)` | Returns 0 (the characteristic of Q). | — |
| `Conductor(Q)` | The smallest positive integer n such that Q ⊆ Q(ζ_n). For Q this is 1. | — |
| `Degree(Q)` `AbsoluteDegree(Q)` | The degree of Q as a number field: 1. | — |
| `Discriminant(Q)` `AbsoluteDiscriminant(Q)` | The field discriminant of Q: 1. | — |
| `DefiningPolynomial(Q)` | An irreducible polynomial over Q whose root generates Q as a number field; for Q returns the linear polynomial x − 1. | — |
| `Signature(Q)` | The signature (number of real embeddings, number of pairs of complex embeddings) of Q. | — |

### 20.3.3 Ring Predicates and Booleans

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsCommutative(Q)` `IsUnitary(Q)` `IsFinite(Q)` `IsOrdered(Q)` `IsField(Q)` `IsEuclideanDomain(Q)` `IsPID(Q)` `IsUFD(Q)` `IsDivisionRing(Q)` `IsEuclideanRing(Q)` `IsPrincipalIdealRing(Q)` `IsDomain(Q)` | Standard ring predicate tests for Q. | — |
| `Q eq R` `Q ne R` | Equality and inequality of ring structures. | — |

---

## 20.4 Element Operations

A variety of operations are provided for rational elements: arithmetic, comparison,
predicates, rounding, rational reconstruction, valuation, and sequence conversion.

### 20.4.1 Parent and Category

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Parent(r)` `Category(r)` | Return the parent structure and category of a rational element r. | — |

### 20.4.2 Arithmetic Operators

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `+ a` `- a` | Unary plus and negation. | — |
| `a + b` `a - b` `a * b` `a ^ k` `a / b` | Standard binary arithmetic operators on rational numbers. | — |
| `a +:= b` `a -:= b` `a *:= b` `a /:= b` `a ^:= k` | In-place assignment variants of the arithmetic operators. | — |

### 20.4.3 Numerator and Denominator

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Numerator(q)` | The integer numerator of rational q in reduced form. | — |
| `Denominator(q)` | The integer denominator of rational q in reduced form; always a positive integer. | — |

*Worked examples: H20E3 (demonstration that rationals are immediately reduced: Numerator(10/-4) = -5, Denominator(10/-4) = 2).*

### 20.4.4 Equality and Membership

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `a eq b` `a ne b` | Equality and inequality of rational numbers. | — |
| `a in R` `a notin R` | Membership test for a rational element in a ring R. | — |

### 20.4.5 Predicates on Ring Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsIntegral(q)` | Returns true if the rational number q lies in Z, false otherwise. | — |
| `IsZero(a)` `IsOne(a)` `IsMinusOne(a)` `IsNilpotent(a)` `IsIdempotent(a)` | Standard element predicates. | — |
| `IsUnit(a)` `IsZeroDivisor(a)` `IsRegular(a)` `IsIrreducible(a)` `IsPrime(a)` | Further standard element predicates. | — |

### 20.4.6 Comparison

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `a gt b` `a ge b` `a lt b` `a le b` | Ordered comparison of rational numbers. | — |
| `Maximum(a, b)` `Maximum(Q)` | Maximum of two rationals, or maximum of the sequence Q. | — |
| `Minimum(a, b)` `Minimum(Q)` | Minimum of two rationals, or minimum of the sequence Q. | — |

### 20.4.7 Conjugates, Norm and Trace

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ComplexConjugate(q)` | The complex conjugate of q, which is q itself (Q embeds in R). | — |
| `Conjugate(q)` | The conjugate of q, which is q itself. | — |
| `Norm(q)` | The norm of q in Q, which is q itself. | — |
| `Trace(q)` | The trace of q in Q, which is q itself. | — |
| `MinimalPolynomial(q)` | The minimal polynomial of the rational number q: the monic linear polynomial with constant coefficient −q over Q. (Returns $.1 − q if the polynomial ring indeterminate has no name.) | — |

### 20.4.8 Absolute Value and Sign

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AbsoluteValue(q)` `Abs(q)` | The absolute value |q| of a rational number q. | — |
| `Sign(q)` | Returns the sign of q: one of the integers −1, 0, 1 corresponding to q < 0, q = 0, q > 0. | — |
| `Height(q)` | The height of q = r/s with r, s coprime: max(|r|, |s|). | — |

### 20.4.9 Rounding and Truncating

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Ceiling(q)` | The ceiling of q: the least integer ≥ q. | — |
| `Floor(q)` | The floor of q: the largest integer ≤ q. | — |
| `Round(q)` | The integer nearest to q. In case of a tie (half-integer), rounds away from zero (i + 1/2 rounds to i + 1 for non-negative i; i − 1/2 rounds to i − 1 for non-positive i). | — |
| `Truncate(q)` | The integer truncation of q: the integral part, i.e. rounding towards 0. | — |
| `Qround(q, M)` | Finds a rational approximation d of q such that the denominator of d is bounded by M. Parameter: `ContFrac` (BoolElt, default false) — if true, computes an optimal approximation via the continued fraction process; by default a faster rounding procedure is used (gives less accurate results). | Continued fraction algorithm (when `ContFrac := true`); otherwise a rounding heuristic. |

### 20.4.10 Rational Reconstruction

Under certain circumstances a partial inverse of the map ψ_m : Q → Z/mZ (taking residues
mod m, defined when the denominator is coprime to m) is useful. For s ∈ Z/mZ, the value
ψ_m⁻¹(s) is the rational r with ψ_m(r) = s and |numerator(r)|, |denominator(r)| ≤ √(m/2);
such r need not exist but is unique when it does.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RationalReconstruction(s)` | Given an element s of a ring S of m elements (either `Integers(m)` or a prime finite field `FiniteField(p)` with p = m), returns a boolean indicating whether a rational r = n/d in minimal terms exists satisfying n·d⁻¹ ≡ s mod m, |n| ≤ √(m/2), 0 < d ≤ √(m/2). If true, also returns r. Also accepts s as a matrix over a prime finite field, in which case a matrix rational reconstruction is attempted. | Partial inverse of reduction mod m via the extended Euclidean algorithm / continued fractions. |

### 20.4.11 Valuation

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Valuation(x, p)` `Valuation(x, I)` | The valuation v of the rational number x at the prime p (or prime ideal I): the difference of the valuations of the numerator and denominator of x. The optional second return value is the rational unit u such that x = p^v · u. | — |

### 20.4.12 Sequence Conversions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ElementToSequence(a)` `Eltseq(a)` | Returns the sequence [a], for compatibility with other field types. | — |

---

## Algorithm-to-function quick reference

| Algorithm / concept | Functions |
|--------------------|-----------|
| Literal rational construction | `a / b`, `Q ! [a, b]`, `elt< Q \| a, b >`, `Q ! a` |
| p-adic completion | `Completion(Q, P)` |
| Rounding (floor/ceiling/truncate) | `Floor`, `Ceiling`, `Round`, `Truncate` |
| Best rational approximation (continued fractions) | `Qround(:ContFrac)` |
| Rational reconstruction (partial inverse of mod-m reduction) | `RationalReconstruction` |
| p-adic / prime-ideal valuation | `Valuation(x, p)`, `Valuation(x, I)` |
| Number-field hierarchy (norm, trace, minimal polynomial, signature, etc.) | `Norm`, `Trace`, `MinimalPolynomial`, `Signature`, `Degree`, `Discriminant`, `Conductor`, `IntegralBasis`, `DefiningPolynomial` |
| Decomposition as number field at a prime | `Decomposition(Q, p)` |
