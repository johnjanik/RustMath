# Chapter 35 — Quadratic Fields

**Handbook part:** VI — Global Arithmetic Fields
**Handbook pages:** 835–845 (PDF pages 966–979)

---

## Scope and overview

Quadratic fields in Magma are created as a subtype of the number fields `FldNum`. The
special quadratic field type provides faster, dedicated algorithms compared to the general
number field machinery; the general functions described in Chapter 34 also apply.

The categories involved are `FldQuad` for fields, `RngQuad` for their orders, and
`FldQuadElt` / `RngQuadElt` for their elements.

**Representation.** For every squarefree integer d (not 0 or 1) there is a unique quadratic
field Q(√d). Given any integer m, `QuadraticField` creates Q(√d) where d is the squarefree
kernel of m. Magma maintains a list of quadratic fields currently present so that two fields
with the same d are identical objects. The discriminant D of Q(√d) is D = d if d ≡ 1 mod 4
and D = 4d if d ≡ 2, 3 mod 4.

Elements of Q(√d) are represented by a common positive denominator b and two integer
coefficients: α = (1/b)(x + y√d). The ring of integers O_F = Z + ε_d Z where ε_d = √d
if d ≡ 2, 3 mod 4, and ε_d = (1 + √d)/2 if d ≡ 1 mod 4. For any positive integer f there
is a suborder of conductor f in O_F whose elements are of the form x + yf ε_d. The equation
order E_F = Z + √d Z admits suborders whose elements are of the form x + yf√d.

---

## 35.2 Creation of Structures

Squarefree integers determine quadratic fields. Associated with any quadratic field is its
ring of integers (maximal order) and an equation order; for every positive integer f there
exists an order of conductor f inside the maximal order. For information on creating
elements see Section 34.2.3.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `QuadraticField(m)` | Given a non-square integer m, creates Q(√d) where d is the squarefree part of m. A name for √d may be assigned using angle brackets: `R<s> := QuadraticField(m)`. Returns an existing object if Q(√d) has been created before. | — |
| `EquationOrder(F)` | Creates the equation order Z[√d] in the quadratic field F = Q(√d), with d squarefree. | — |
| `MaximalOrder(F)` / `IntegerRing(F)` / `RingOfIntegers(F)` | Given a quadratic field F = Q(√d) with d squarefree, creates its maximal order: Z[√d] if d ≡ 2, 3 mod 4; Z[(1+√d)/2] if d ≡ 1 mod 4. | — |
| `NumberField(O)` | Given a quadratic order O, returns the quadratic field of which it is an order. | — |
| `sub< O \| f >` | Creates the sub-order of index f in the order O of a quadratic field. If O is maximal, this is the unique order of conductor f. | — |
| `IsQuadratic(K)` / `IsQuadratic(O)` | Returns true if the field K or order O can be created as a quadratic field or order, and the quadratic field or order if so. | — |

*Worked examples: H35E1 (creating Q(√5) and a conductor-7 suborder; coercion between order and field elements); H35E2 (defining an injection Q(√5) → Q(ζ₅) via factorization over a cyclotomic field).*

---

## 35.3 Operations on Structures

The majority of functions for quadratic fields and orders are shared with general number
fields and orders. The functions listed here either exist only for quadratic fields/orders or
deserve special mention.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AssignNames(~F, [s])` / `AssignNames(~O, [s])` | Procedure to change the name of the generator of a quadratic field F or order O to the string s. Elements print as `1/b*(x + y*s)` for the field or `x + y*s` for an order. Does not assign to identifier s; use angle brackets at creation time for that. Modifies F or O in place (reference ~ required). | — |
| `Name(F, 1)` / `Name(O, 1)` | Returns the named element: √d in the field, fε_d in a suborder of the maximal order, or f√d in a suborder of the equation order. | — |
| `FundamentalUnit(K)` / `FundamentalUnit(O)` | A generator for the unit group of the order O or the maximal order of the quadratic field K. | — |
| `Discriminant(K)` | The discriminant of the field K (defined only up to squares; will be the discriminant of the polynomial or better). | — |
| `Conductor(K)` | The conductor of the field K: the order of the smallest cyclotomic field containing K, plus a sequence of the ramified real places. | — |
| `Conductor(O)` | The conductor of the order O, equal to the index of O in the maximal order. | — |

### 35.3.1 Ideal Class Group

The function `ClassGroup` is available for number fields and orders in general, but a
different and faster algorithm is used by default for quadratic fields. All algorithms except
the sieving method of **[Jac99]** (which uses the multiple polynomial quadratic sieve, MPQS)
are based on binary quadratic forms; see `ClassGroup` (Chapter 33) for further details.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ClassGroup(K)` / `ClassGroup(O)` | Class group of a maximal order O or the maximal order of the quadratic field K, as an abelian group, plus a map from the group to the power structure of ideals. Parameters: `FactorBasisBound` (FldReElt, default 0.1), `ProofBound` (FldReElt, default 6), `ExtraRelations` (RngIntElt, default 1), `Al` (MonStgElt, default `"Automatic"`; set to `"Sieve"` or `"NoSieve"` to control sieving; sieving used when discriminant > 10²⁰ by default), `Verbose ClassGroupSieve` (maximum 5). | Binary quadratic forms; sieve method **[Jac99]** (MPQS) when discriminant is large. |
| `ClassNumber(K)` / `ClassNumber(O)` | The class number of the maximal order O or the maximal order of the quadratic field K. Parameters: `FactorBasisBound` (default 0.1), `ProofBound` (default 6), `ExtraRelations` (default 1), `Al` (default `"Automatic"`). | Same as `ClassGroup`. |
| `PicardGroup(O)` / `PicardNumber(O)` | The Picard group (group of invertible ideals of O modulo principal ones) of the order O, or the size of this group. `PicardGroup` also returns a map from the group to the ideals of O. Parameters: `FactorBasisBound` (default 0.1), `ProofBound` (default 6), `ExtraRelations` (default 1), `Al` (default `"Automatic"`). | Same as `ClassGroup`. |
| `QuadraticClassGroupTwoPart(K)` / `QuadraticClassGroupTwoPart(O)` / `QuadraticClassGroupTwoPart(d)` | Computes the 2-part of the class group of a quadratic order. Returns: an array of forms generating the 2-part, and an array giving the orders of the respective elements. Optional parameter `Factorization` (RngIntEltFact, default `[]`) to supply the factorization of the given discriminant. | Bosma–Stevenhagen algorithm. |

*Worked examples: H35E3 (class group computations using sieving for imaginary and real quadratic fields; use of the ideal–group map); H35E4 (`QuadraticClassGroupTwoPart` for a large discriminant).*

### 35.3.2 Norm Equations

For imaginary quadratic fields Q(√m) with m < 0, `NormEquation` finds integral elements
of a given norm using a constructive method (Cornacchia's algorithm, **[Coh93]** §1.5.2).
For real quadratic fields, the same algorithm as for general number fields is used (conics;
see Section 119.5.1). A version with integer arguments d and m also exists (Section 18.12.2).

If d ≡ 1 mod 4 (squarefree), the function searches for a solution in integers to
x² + y²d = 4m and returns α = (x + y√d)/2; for d ≡ 2, 3 mod 4, it searches for
α = x + y√d with x² + y²d = m. In an order of conductor f, d is replaced by f²d.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `NormEquation(F, m)` / `NormEquation(F, m: parameters)` / `NormEquation(O, m)` / `NormEquation(O, m: parameters)` | Given a quadratic field F (or suborder O) and a non-negative integer m: returns true if there exists an element α in the ring of integers of F (or in O) with norm m, false otherwise. For imaginary quadratic fields also returns a solution [x] as a second value. Parameters: `Factorization` ([<RngIntElt, RngIntElt>]) to speed up computation when factorization of m is known; `All` (BoolElt, default true); `Solutions` (RngIntElt, default All); `Exact` (BoolElt, default false); `Ineq` (BoolElt, default false); `Verbose NormEquation` (maximum 1). | Imaginary quadratic: Cornacchia's algorithm **[Coh93]** §1.5.2. Real quadratic: general number field algorithm (conics, Section 119.5.1). |

*Worked example: H35E5 (NormEquation on a conductor-6 suborder of a large imaginary quadratic field).*

---

## 35.4 Special Element Operations

A number of functions are available only for elements of certain maximal orders.
`Conjugate` returns a quadratic element (instead of a real) as well as `ComplexConjugate`.
The `mod` operator and GCD/LCM functions are restricted to maximal orders Q(√d) for d in
the set {−1, −2, −3, −7, −11, 2, 3, 5, 13}.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `a mod b` | Remainder on dividing a by b where a and b lie in the maximal order of Q(√d) for d ∈ {−1, −2, −3, −7, −11, 2, 3, 5, 13}. `div` is provided for order elements in general, but will fail if division is not exact for discriminants outside this list. | — |

### 35.4.1 Greatest Common Divisors

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Gcd(a, b)` / `GCD(a, b)` / `GreatestCommonDivisor(a, b)` | Greatest common divisor of a and b in the maximal order of Q(√d), where d ∈ {−1, −2, −3, −7, −11, 2, 3, 5, 13}. | — |
| `Lcm(a, b)` / `LCM(a, b)` / `LeastCommonMultiple(a, b)` | Least common multiple of a and b in the maximal order of Q(√d), where d ∈ {−1, −2, −3, −7, −11, 2, 3, 5, 13}. | — |

### 35.4.2 Modular Arithmetic

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Modexp(a, e, n)` | Computes aᵉ mod n in the maximal order of Q(√d), where d ∈ {−1, −2, −3, −7, −11, 2, 3, 5, 13}. | — |

### 35.4.3 Factorization

Factorization in maximal orders of quadratic number fields is based on factoring the norm
in the integers. Comments about the `Factorization` command for integers also apply here.
Since the factorization may be off by a unit power, that power is also returned (the unit
being −1, √−1, or (1 + √−3)/2).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Factorization(n)` / `Factorisation(n)` | Factorization of n in the maximal order of Q(√d) for d ∈ {−1, −2, −3, −7, −11}. Returns the factorization together with the appropriate power of a unit (the unit being −1, √−1, or (1 + √−3)/2). | Norm factorization in Z. |
| `TrialDivision(n, B)` | Trial division of n by primes of relative norm ≤ B in the maximal order of Q(√d) for d ∈ {−1, −2, −3, −7, −11}. Returns the factored part, the unfactored part, and the power of the unit that the factorization is off by (unit being −1, √−1, or (1 + √−3)/2). | Trial division by bounded-norm primes. |

### 35.4.4 Conjugates

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ComplexConjugate(a)` | The complex conjugate of quadratic field element a: returns a itself in a real quadratic field; returns ā = x − y√d if a = x + y√d in an imaginary quadratic field Q(√d). | — |
| `Conjugate(a)` | The conjugate x − y√d of a = x + y√d in the quadratic field Q(√d). | — |

### 35.4.5 Other Element Functions

For the ring of integers of Q(i), the biquadratic residue symbol (generalizing the Legendre
symbol) is available.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BiquadraticResidueSymbol(a, b)` | Given a Gaussian integer a and a primary, non-unit Gaussian integer b with gcd(a, b) = 1: returns the value of the biquadratic character (a/b)₄. The value is iᵏ for some k ∈ {0, 1, 2, 3}. Returns 0 if a and b have a common factor; returns an error if b is not primary or is a unit. | Biquadratic reciprocity (Gaussian integers). |
| `Primary(a)` | Returns the unique associate ā of the Gaussian integer a satisfying ā ≡ 1 mod (1 + i)³, or 0 if a is divisible by 1 + i. | — |

*Worked example: H35E6 (verifying Euler's conjecture / Gauss's theorem: z⁴ ≡ 2 mod p has a solution ⟺ p = x² + 64y² for primes p ≡ 1 mod 4 with 65 ≤ p ≤ 1000, using `NormEquation`, `BiquadraticResidueSymbol`, and `Primary`).*

---

## 35.5 Special Functions for Ideals

Ideals of orders of quadratic fields inherit from ideals of orders of number fields (see
Section 37.9 for the general list). There is also a correspondence between quadratic ideals
and binary quadratic forms (see Chapter 33); the following functions exploit that
correspondence.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Content(I)` | The content of the ideal. | Quadratic form correspondence. |
| `Conjugate(I)` | The conjugate of the ideal. | Quadratic form correspondence. |
| `Discriminant(I)` | The discriminant of the quadratic form associated with I. | Quadratic form correspondence. |
| `QuadraticForm(I)` | The binary quadratic form associated with the ideal I. | Quadratic form correspondence. |
| `Ideal(f)` | The quadratic ideal with associated quadratic form f. | Quadratic form correspondence. |
| `Reduction(I)` | The quadratic ideal whose associated quadratic form is a reduction of the quadratic form associated with I. | Quadratic form reduction (Chapter 33). |

---

## 35.6 Bibliography

| Key | Reference |
|-----|-----------|
| **[Coh93]** | Henri Cohen. *A Course in Computational Algebraic Number Theory*, volume 138 of Graduate Texts in Mathematics. Springer, Berlin–Heidelberg–New York, 1993. |
| **[Jac99]** | M. J. Jacobson, Jr. Applying sieving to the computation of quadratic class groups. *Math. Comp.*, 68(226):859–867, 1999. |

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Squarefree kernel / quadratic field creation | `QuadraticField`, `EquationOrder`, `MaximalOrder`, `IntegerRing`, `RingOfIntegers`, `sub<>`, `IsQuadratic` |
| Binary quadratic forms | `ClassGroup`, `ClassNumber`, `PicardGroup`, `PicardNumber`, `Content`, `Conjugate` (ideal), `Discriminant` (ideal), `QuadraticForm`, `Ideal`, `Reduction` |
| MPQS sieve (class group) **[Jac99]** | `ClassGroup(:Al:="Sieve")`, `ClassNumber(:Al:="Sieve")`, `PicardGroup(:Al:="Sieve")` |
| Bosma–Stevenhagen 2-part algorithm | `QuadraticClassGroupTwoPart` |
| Cornacchia's algorithm **[Coh93]** §1.5.2 | `NormEquation` (imaginary quadratic fields) |
| Norm factorization in Z | `Factorization`, `Factorisation`, `TrialDivision` |
| Biquadratic reciprocity (Gaussian integers) | `BiquadraticResidueSymbol`, `Primary` |
| GCD / mod arithmetic (Euclidean rings) | `Gcd`, `GCD`, `GreatestCommonDivisor`, `Lcm`, `LCM`, `LeastCommonMultiple`, `Modexp`, `mod` |
