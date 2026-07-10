# Chapter 33 — Binary Quadratic Forms

**Handbook part:** V — Lattices and Quadratic Forms
**Handbook pages:** 753–763 (PDF pages 884–899)

---

## Scope and overview

A binary quadratic form is an integral form ax² + bxy + cy² represented in Magma by a tuple ⟨a, b, c⟩. Binary quadratic forms play a central role in the ideal theory of quadratic fields, the classical theory of complex multiplication, and the theory of modular forms. Algorithms for binary quadratic forms provide efficient means of computing in the ideal class group of orders in a quadratic field.

The structures of quadratic forms of a given discriminant D correspond to ordered bases of ideals in an order in a quadratic number field, defined up to scaling by the rationals. A form is primitive if the coefficients a, b, and c are coprime. For negative discriminants the primitive reduced forms in this structure are in bijection with the class group of projective or invertible ideals. For positive discriminants, the reduced orbits of forms are used for this purpose.

Magma holds efficient algorithms for composition, enumeration of reduced forms, class group computations, and discrete logarithms. A significant novel feature is the treatment of nonfundamental discriminants, corresponding to nonmaximal orders, and the collections of homomorphisms between different class groups coming from the inclusions of these orders.

The functionality is rounded out with various functions for applying modular and elliptic functions to forms, and for class polynomials associated to class groups of definite forms. By using the explicit relation of definite quadratic forms with lattices with nontrivial endomorphism ring in the complex plane, one can apply modular and elliptic functions to forms and exploit the analytic theory of complex multiplication.

---

## 33.2 Creation Functions

### 33.2.1 Creation of Structures

For any integer D congruent to 0 or 1 modulo 4, it is possible to create the parent structure of binary quadratic forms of discriminant D.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BinaryQuadraticForms(D)` / `QuadraticForms(D)` | Create the structure of integral binary quadratic forms of discriminant D. | — |

### 33.2.2 Creation of Forms

Binary quadratic forms may be created by coercing a triple [a, b, c] of integer coefficients into the parent structure of forms of discriminant D = b² − 4ac. Other constructors are provided for constructing the group identity, prime forms, or allowing the omission of the third element c.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Identity(Q)` / `Q ! 1` | Create the principal form in the structure Q of binary quadratic forms of discriminant D. The principal form is X² − D/4 · Y² if D ≡ 0 (mod 4), or X² + XY + (D−1)/4 · Y² if D ≡ 1 (mod 4). This form is a reduced form representing the identity element of the class group of Q. | — |
| `Q ! [a, b, c]` / `elt< Q \| a, b, c >` / `elt< Q \| a, b >` | Returns the binary quadratic form aX² + bXY + cY² in the magma of forms Q of discriminant D. Here c is determined by the solution of D = b² − 4ac; if no integer c satisfying this exists, an error occurs. | — |
| `PrimeForm(Q, p)` | If p is a split prime or a ramified prime not dividing the conductor of the magma of quadratic forms Q, returns a quadratic form pX² + bXY + cY² in Q. | — |

---

## 33.3 Basic Invariants

Structures of binary quadratic forms are defined in terms of a discriminant, and membership in a structure is determined by this invariant. Additional elementary functions are provided to test integer inputs to determine if they define valid discriminants.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Discriminant(f)` | The discriminant b² − 4ac of a quadratic form f = aX² + bXY + cY². | — |
| `Discriminant(Q)` | The discriminant of the quadratic forms belonging to the magma of quadratic forms Q. | — |
| `IsDiscriminant(D)` | Returns true if the integer D is the discriminant of some quadratic form; false otherwise. | — |
| `FundamentalDiscriminant(D)` | The fundamental discriminant corresponding to the integer D. | — |
| `IsFundamental(D)` / `IsFundamentalDiscriminant(D)` | Returns true if D is an integer other than 0 or 1, congruent to 0 or 1 modulo 4, which is not of the form m²D_K for m > 1 and any other such integer D_K. | — |
| `Conductor(Q)` | The conductor of quadratic forms whose discriminant is that of the magma of quadratic forms Q. | — |

---

## 33.4 Operations on Forms

### 33.4.1 Arithmetic

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Conjugate(f)` | Given a form f = ax² + bxy + cy², returns the conjugate form ax² − bxy + cy². | — |
| `f * g` | Composition of two binary quadratic forms f and g. Returns a reduced representative of the product using a fast composition algorithm of Shanks. | Fast composition algorithm of Shanks. |
| `Composition(f, g)` | Composition of two binary quadratic forms f and g. Parameters: `Al` (MonStgElt, default `"Gauss"`) selects the algorithm of Gauss or Shanks; `Reduction` (BoolElt, default `false`) whether to return a reduced representative. When `Reduction := false`, one works in the group of forms rather than the class group representatives. The combination `Reduction := false` and `Al := "Shanks"` is incompatible and returns a runtime error (Shanks performs partial intermediate reductions). | Algorithm of Gauss or Shanks (selectable). |
| `f ^ n` | Returns a reduced representative of the n-th power of a form f, using the fast composition algorithm of Shanks. | Fast composition algorithm of Shanks. |
| `Power(f, n)` | Returns the n-th power of a form f. Parameters: `Al` (MonStgElt, default `"Gauss"`); `Reduction` (BoolElt, default `false`). The combination `Reduction := false` and `Al := "Shanks"` is incompatible. | Algorithm of Gauss or Shanks (selectable). |
| `Reduction(f)` / `ReducedForm(f)` | Returns a reduced quadratic form equivalent to f, and the transformation matrix. | Form reduction. |
| `ReductionStep(f)` | Returns the result of applying one reduction step to the quadratic form f. | Single reduction step. |
| `ReductionOrbit(f)` | The cycle of reduced forms equivalent to f (and each other) where f has positive discriminant. | Reduction orbit enumeration. |
| `Order(f)` | For a binary quadratic form f, returns its order as an element of the class group Cl(Q) where Q is the parent of f. | — |

### 33.4.2 Attribute Access

The coefficient sequence can be accessed as a sequence of integers, providing the inverse operation to the forms coercion constructor.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `f[i]` | The i-th coefficient of f, where 1 ≤ i ≤ 3. | — |
| `Eltseq(f)` / `ElementToSequence(f)` | The sequence [a, b, c] where f is the form ax² + bxy + cy². | — |

### 33.4.3 Boolean Operations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `f in Q` | Returns true if and only if f is in Q, that is f and Q have the same discriminant. | — |
| `f eq g` | Returns true if the quadratic forms f and g are equal and false otherwise. | — |
| `IsIdentity(f)` | Returns true if and only if f is the principal form in its parent structure. | — |
| `IsReduced(f)` | Returns true if the quadratic form f is reduced; false otherwise. | — |
| `IsEquivalent(f, g)` | Returns true if the quadratic forms f and g reduce to the same form and false otherwise. If true and the discriminant is negative, the transformation matrix is also returned. An error is returned if the forms are not of the same discriminant. | Reduction to canonical representative. |

### 33.4.4 Related Structures

In addition to the Parent and Category structures of binary quadratic forms, the quadratic forms map to the ideals of a fixed order of discriminant D in a quadratic number field.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Parent(f)` | The parent structure (magma of quadratic forms) of f. | — |
| `Category(Q)` | The category of the structure Q. | — |
| `QuadraticOrder(Q)` | Given a structure of quadratic forms of discriminant D, returns the associated order of discriminant D in a quadratic field. | — |
| `Ideal(f)` | Given a quadratic form f = ax² + bxy + cy², returns the ideal (a, (−b + √D)/2) in the quadratic order Z[(t + √D)/2], where t equals 0 or 1. | — |

---

## 33.5 Class Group

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ReducedForms(Q)` | Given the structure of quadratic forms of negative discriminant D, returns the sequence of all primitive reduced forms of discriminant D. | Enumeration of reduced forms. |
| `ReducedOrbits(Q)` | Given the structure of quadratic forms of positive discriminant D, returns the sequence of all reduced orbits of primitive forms of discriminant D, as an indexed set. | Reduction orbit enumeration. |
| `ClassNumber(Q: parameters)` / `ClassNumber(D: parameters)` | The class number of binary quadratic forms Q of discriminant D. Parameter `Al` (MonStgElt, default `"Automatic"`) selects the method: `"ReducedForms"` (enumerating all reduced forms), `"Shanks"` (Shanks-based algorithm), `"Sieve"` or `"NoSieve"`. The default uses reduced form enumeration for small discriminants, Shanks for the middle range, index-calculus for large discriminants, and the sieve method **[Jac99]** for very large discriminants. Parameters `FactorBasisBound` (FldReElt, default 0.1), `ProofBound` (FldReElt, default 6), `ExtraRelations` (RngIntElt, default 1) apply to index-calculus only. | Reduced form enumeration / Shanks / index-calculus **[CDO93, HM89, Coh93]** / MPQS sieve **[Jac99]**. |
| `ClassGroup(Q: parameters)` | The class group of the binary quadratic forms Q of discriminant D. Also returns a map from the abelian group to the structure of quadratic forms. Parameters: `FactorBasisBound` (FldReElt, default 0.1), `ProofBound` (FldReElt, default 6), `ExtraRelations` (RngIntElt, default 1), `Al` (MonStgElt, default `"Automatic"`). The index-calculus method builds a factor basis of prime forms of norm < B1 := FactorBasisBound · log² |D|, finds relations generating the full lattice (Smith form gives group structure), then verifies all prime forms of norm < B2 := ProofBound · log² |D| lie in the generated group. Result is checked against the Euler product over the first 30,000 primes. Setting `Al := "Sieve"` or |D| > 10²⁰ triggers the MPQS sieve **[Jac99]** (proven under GRH). | Shanks-based method **[Tes98, BJT97]** / index-calculus **[CDO93, HM89, Coh93]** / MPQS sieve **[Jac99]**. |
| `ClassGroupStructure(Q: parameters)` | The structure of the class group of the binary quadratic forms Q of discriminant D returned as a sequence of integers giving the abelian invariants. Accepts the same parameters as `ClassGroup`. | As `ClassGroup`. |
| `AmbiguousForms(Q)` | Enumerates the ambiguous forms of negative discriminant D, where D is the discriminant of the magma of binary quadratic forms Q. | — |
| `TwoTorsionSubgroup(Q)` | The subgroup of 2-torsion elements in the class group of Q. | — |

*Worked example: H33E1 (class group of a magma of quadratic forms of discriminant 7537543; generating class group elements as forms; computing the reduction orbit via `ReductionStep`; determining orders of classes by testing whether the identity lies in the orbit cycle).*

---

## 33.6 Class Group Coercions

The class group of a nonmaximal quadratic order R of discriminant m²D_K is related to the class group of the maximal order O_K of fundamental discriminant D_K by an exact sequence:

1 → (O_K/mO_K)* / O_K* (Z/mZ)* → Cl(O) → Cl(O_K) → 1

Similar maps exist between quadratic orders O₁ and O₂ in a field K, with conductors m₁ and m₂ respectively, such that m₁ | m₂. The homomorphism is returned as a map object or can be called directly via the coercion operator.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `FundamentalQuotient(Q)` | The quotient homomorphism from the class group of Q to the class group of fundamental discriminant. | Norm-conductor exact sequence. |
| `QuotientMap(Q1, Q2)` | Given two structures of quadratic forms Q1 and Q2 such that the discriminant of Q2 equals a square times the discriminant of Q1, the quotient homomorphism from Q1 to Q2 is returned as a map object. | Norm-conductor exact sequence. |
| `Q ! f` | The ! operator applies the quotient homomorphism for automatic coercion of forms f of discriminant m²D into the structure Q of forms of discriminant D. | — |

---

## 33.7 Discrete Logarithms

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Log(b, x)` | The discrete logarithm of binary quadratic form x with respect to base b, or −1 if Magma can determine no solution exists. Exists only for negative discriminant forms. If the user is unsure whether a solution exists, it is safest to use `Log` with a time limit to prevent an infinite loop. | Pohlig–Hellman algorithm with a collision search subroutine (variant of Pollard's rho method). |
| `Log(b, x, t)` | Searches for up to t seconds for the discrete logarithm of binary quadratic form x with respect to base b. Exists only for negative discriminant forms. Returns −1 if no solution exists, −2 if no solution found within the time frame. | Pohlig–Hellman algorithm with a collision search subroutine (variant of Pollard's rho method). |

---

## 33.8 Elliptic and Modular Invariants

Binary quadratic forms of negative discriminant describe positive definite lattices in the complex plane, with integral-valued inner product. As such, it is possible to apply modular and elliptic functions to the form, interpreting it as an element of the upper half plane.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Lattice(f)` | Given a binary quadratic form f = ax² + bxy + cy² of negative discriminant, returns the rank two lattice of f having Gram matrix [[a, b/2], [b/2, c]]. The lattice L is the half-integral lattice such that integral representations f(x, y) = n are in bijection with vectors (x, y) of norm n (a rational number). | — |
| `GramMatrix(f)` | Returns the Gram matrix of the binary quadratic form f, which need not be of negative discriminant. The matrix will be half-integral and defined over the rationals. | — |
| `ThetaSeries(f, n)` | The integral theta series of the binary quadratic form f to precision n. | — |
| `RepresentationNumber(f, n)` | The n-th representation number of the form f of negative discriminant. | — |
| `jInvariant(f)` | For a binary quadratic form f = ax² + bxy + cy² with negative discriminant, returns the j-invariant of f, equal to the j-invariant of τ = (−b + √(b² − 4ac)) / 2a. | Analytic theory of complex multiplication. |
| `Eisenstein(k, f)` | Given a positive even integer k = 2n and a binary quadratic form f = ax² + bxy + cy², returns the value of the Eisenstein series E_k(L) at the complex lattice L = ⟨a, (−b + √(b² − 4ac))/2⟩. | Eisenstein series evaluation at a CM lattice. |
| `WeierstrassSeries(z, f)` | Given a complex power series z with positive valuation and a binary quadratic form f = ax² + bxy + cy², returns the q-expansion of the Weierstrass ℘-function at the complex lattice L = ⟨a, (−b + √(b² − 4ac))/2⟩. | Weierstrass ℘-function expansion at a CM lattice. |

*Worked example: H33E2 (discriminant −163; `PrimeForm` at p = 41; computing `WeierstrassSeries` and `Eisenstein` series, then verifying the Weierstrass equation y² = x³ + Ax + B to numerical precision).*

---

## 33.9 Class Invariants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HilbertClassPolynomial(D)` | Given a negative discriminant D, returns the Hilbert class polynomial, defined as the minimal polynomial of j(τ), where Z[τ] is an imaginary quadratic order of discriminant D. | Analytic theory of complex multiplication; evaluation of j(τ) at CM points. |
| `WeberClassPolynomial(D)` | Given a negative discriminant D congruent to 1 modulo 8, returns the Weber class polynomial, defined as the minimal polynomial of f(τ), where Z[τ] is an imaginary quadratic order of discriminant D and f is a particular normalized Weber function generating the same class field as j(τ). A root f(τ) of the Weber class polynomial is an integral unit generating the ring class field. The relation to the Hilbert class polynomial is: j(τ) = (f(τ)²⁴ − 16)³ / f(τ)²⁴ when GCD(D, 3) = 1, and j(τ) = (f(τ)⁸ − 16)³ / f(τ)⁸ if 3 divides D. See Yui and Zagier **[YZ97]**. | Weber modular function evaluation at CM points **[YZ97]**. |

---

## 33.10 Matrix Action on Forms

A matrix in SL(2, Z) acts on the right on quadratic forms by the rule:

f(x, y) · [[r, s], [t, u]] = f(rx + sy, tx + uy)

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `f * M` | The action of a matrix M ∈ SL(2, Z) on a binary quadratic form f (right action). | SL(2, Z) right action on forms. |

---

## 33.11 Bibliography

| Key | Reference |
|-----|-----------|
| **[BJT97]** | J. Buchmann, M. J. Jacobson, Jr., and E. Teske. *On Some Computational Problems in Finite Abelian Groups.* Mathematics of Computation, 66:1663–1687, 1997. |
| **[CDO93]** | H. Cohen, F. Diaz y Diaz, and M. Olivier. *Calculs de nombres de classes et de régulateurs de corps quadratiques en temps sous-exponentiel.* In Séminaire de Théorie des Nombres, Paris, 1990–91, volume 108 of Progr. Math., pages 35–46. Birkhäuser Boston, Boston, MA, 1993. |
| **[Coh93]** | Henri Cohen. *A Course in Computational Algebraic Number Theory*, volume 138 of Graduate Texts in Mathematics. Springer, Berlin–Heidelberg–New York, 1993. |
| **[HM89]** | J. Hafner and K. McCurley. *A rigorous subexponential algorithm for computation of class groups.* Journal American Math. Soc., 2:837–850, 1989. |
| **[Jac99]** | M. J. Jacobson, Jr. *Applying sieving to the computation of quadratic class groups.* Math. Comp., 68(226):859–867, 1999. |
| **[Tes98]** | E. Teske. *A Space Efficient Algorithm for Group Structure Computation.* Mathematics of Computation, 67:1637–1663, 1998. |
| **[YZ97]** | N. Yui and D. Zagier. *On the singular values of Weber modular functions.* Mathematics of Computation, 66(220):1645–1662, 1997. |

---

## Algorithm-to-function quick reference

| Algorithm / Theory | Functions |
|--------------------|-----------|
| Form reduction | `Reduction`, `ReducedForm`, `ReductionStep`, `ReductionOrbit`, `ReducedForms`, `ReducedOrbits`, `IsReduced`, `IsEquivalent` |
| Composition algorithm of Gauss | `Composition(:Al:="Gauss")`, `Power(:Al:="Gauss")` |
| Composition algorithm of Shanks (fast, partial reductions) | `f * g`, `f ^ n`, `Composition(:Al:="Shanks")`, `Power(:Al:="Shanks")` |
| Shanks-based class group **[Tes98, BJT97]** | `ClassNumber(:Al:="Shanks")`, `ClassGroup` (middle range) |
| Index-calculus class group **[CDO93, HM89, Coh93]** | `ClassNumber`, `ClassGroup`, `ClassGroupStructure` (large discriminants) |
| MPQS sieve class group **[Jac99]** | `ClassNumber(:Al:="Sieve")`, `ClassGroup(:Al:="Sieve")` (very large discriminants) |
| Pohlig–Hellman / Pollard rho | `Log(b,x)`, `Log(b,x,t)` |
| Norm-conductor exact sequence (class group coercions) | `FundamentalQuotient`, `QuotientMap`, `Q ! f` |
| Analytic complex multiplication (elliptic/modular) | `jInvariant`, `Eisenstein`, `WeierstrassSeries`, `Lattice`, `GramMatrix`, `ThetaSeries`, `RepresentationNumber` |
| Hilbert class polynomial (CM theory) | `HilbertClassPolynomial` |
| Weber modular functions **[YZ97]** | `WeberClassPolynomial` |
| SL(2, Z) right action on forms | `f * M` |
