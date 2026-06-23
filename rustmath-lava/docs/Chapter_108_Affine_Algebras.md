# Chapter 108 — Affine Algebras

**Handbook part:** XIV — Commutative Algebra
**Handbook pages:** 3287–3299 (PDF pages 3418–3433)

---

## Scope and overview

An affine algebra in Magma is the quotient ring of a multivariate polynomial ring
P = R[x₁, …, xₙ] by an ideal J of P. Such rings arise commonly in commutative algebra
and algebraic geometry, and can be viewed as generalizations of number fields and algebraic
function fields when R is a field.

Elements of affine algebras are multivariate polynomials always kept reduced to normal form
modulo the defining ideal J of "relations". Practically all operations applicable to
multivariate polynomials are also applicable to elements of affine algebras (when
meaningful). The base ring R may currently be a field or a Euclidean ring.

If the ideal J is maximal and R is a field, then the affine algebra A = R[x₁,…,xₙ]/J is
itself a field usable with any Magma algorithm that works over fields, including
factorization of polynomials (in any characteristic, since V2.10). If the affine algebra
defined over a field has finite dimension as a vector space over the coefficient field, a
further set of special operations is available on its elements.

An affine algebra has type `RngMPolRes` and its elements type `RngMPolResElt`. Rings of
fractions of affine algebras have type `RngFunFrac` and their elements `RngFunFracElt`.

---

## 108.1 Introduction

*(Introductory section; no intrinsics.)*

---

## 108.2 Creation of Affine Algebras

An affine algebra can be created by forming the quotient of a multivariate polynomial ring
by an ideal (via the `quo` constructor or the `/` operator), or via the dedicated
`AffineAlgebra` constructor which avoids the need to create the base polynomial ring
separately.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `quo< P \| J >` / `quo< P \| a1, ..., ar >` | Given a multivariate polynomial ring `P` and an ideal `J` of `P` (or generators `a1, …, ar` of `J`), return the quotient ring `P/J`. Angle bracket notation assigns names to indeterminates. | Quotient ring construction via Gröbner basis reduction. |
| `P / J` | Given a multivariate polynomial ring `P` and an ideal `J` of `P`, return the quotient affine algebra `P/J`. | As above. |
| `AffineAlgebra< R, X \| L >` | Given a ring `R`, a list `X` of `n` identifiers, and a list `L` of polynomial relations in the variables `X`, create and return `R[X]/⟨L⟩`. Angle bracket notation assigns names to indeterminates. | Quotient ring construction via Gröbner basis reduction. |

*Worked examples: H108E1 (relative extension of a number field as an affine algebra; algebraic function field over a rational function field).*

---

## 108.3 Operations on Affine Algebras

Most operations on ideals of affine algebras are performed by mapping the computation into
the preimage polynomial ring, performing the corresponding operation there, and mapping
the result back into the affine algebra.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Q . i` | Return the `i`-th indeterminate of affine algebra `Q` as an element of `Q`. | — |
| `CoefficientRing(Q)` | Return the coefficient ring of the affine algebra `Q`. | — |
| `Rank(Q)` | Return the rank of `Q` (number of indeterminates). | — |
| `DivisorIdeal(I)` | Given an ideal `I` of the affine algebra `Q = P/J`, return the defining ideal `J` of `Q` as an ideal of the polynomial ring `P`. | — |
| `PreimageIdeal(I)` | Given an ideal `I` of `Q = P/J`, return the ideal `I′` of `P` such that the image of `I′` under the natural epimorphism `P → Q` is `I`. | — |
| `PreimageRing(Q)` | Given `Q = P/J`, return the polynomial ring `P`. | — |
| `OriginalRing(Q)` | Return the generic polynomial ring `P` such that `Q = P/J` for some ideal `J`. | — |
| `I eq J` | Return `true` if and only if ideals `I` and `J` of the same affine algebra are equal. | Equality via preimage ideals and Gröbner bases. |
| `I subset J` | Return `true` if and only if ideal `I` is contained in ideal `J`. | Membership via Gröbner basis. |
| `I + J` | Return the sum `I + J` of two ideals of the same affine algebra. | Lifted to preimage ring. |
| `I * J` | Return the product `I * J` of two ideals of the same affine algebra. | Lifted to preimage ring. |
| `I ^ n` | Return the power `Iⁿ` of ideal `I` (integer `n`). | Lifted to preimage ring. |
| `I meet J` | Return the intersection `I ∩ J` of two ideals of the same affine algebra. | Lifted to preimage ring. |
| `IsProper(I)` | Return whether ideal `I` of affine algebra `Q` is proper (strictly contained in `Q`). | — |
| `IsZero(I)` | Return whether ideal `I` is the zero ideal (equivalently, whether the preimage ideal of `I` equals the divisor ideal of `Q`). | — |
| `IsPrime(I)` | Return whether ideal `I` of affine algebra `Q` is prime. | Via preimage ideal primality test. |
| `IsPrimary(I)` | Return whether ideal `I` of affine algebra `Q` is primary. | Via preimage ideal. |
| `IsRadical(I)` | Return whether ideal `I` of affine algebra `Q` is radical. | Via preimage ideal. |
| `PrimaryDecomposition(I)` | Return the primary decomposition of ideal `I`, together with the associated prime ideals. | Mapped through preimage polynomial ring; Gröbner-basis primary decomposition algorithms. |
| `RadicalDecomposition(I)` | Return the (prime) decomposition of the radical of ideal `I`. | Mapped through preimage polynomial ring; radical decomposition algorithms. |

*Worked examples: H108E2 (primary decomposition of an ideal in an affine algebra of rank 3).*

---

## 108.4 Maps between Affine Algebras

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AffineAlgebraMapKernel(phi)` | Return the kernel of the homomorphism `φ` of affine algebras. | Kernel computed via preimage ideal methods. |

---

## 108.5 Finite Dimensional Affine Algebras

If an affine algebra is defined over a field and has finite dimension as a vector space over
the coefficient field, the following special operations are available on the algebra and its
elements. Analogous operations for affine algebras over general Euclidean rings are planned
for future support.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HasFiniteDimension(Q)` | Given an affine algebra `Q` defined over a field, return whether `Q` has finite dimension as a vector space. | Dimension of ideal (Gröbner basis methods). |
| `Dimension(Q)` | Given a finite dimensional affine algebra `Q` over a field, return the vector-space dimension of `Q`. | Monomial basis count from Gröbner basis. |
| `VectorSpace(Q)` | Given a finite dimensional affine algebra `Q` over a field, construct the vector space `V` isomorphic to `Q` and return `V` together with the isomorphism `f : Q → V`. | Basis from reduced monomials modulo `J`. |
| `MonomialBasis(Q)` | Given a finite dimensional affine algebra `Q` over a field, return the basis `B` of monomials of `Q` as a sequence of length `d = Dimension(Q)`, such that `f(B[i]) = V.i` where `V` and `f` are the returns of `VectorSpace`. | Reduced monomials modulo Gröbner basis of `J`. |
| `MatrixAlgebra(Q)` | Given a finite dimensional affine algebra `Q` over a field, construct the matrix algebra `A` isomorphic to `Q` and return `A` together with the isomorphism `f : Q → A`. | Representation by multiplication matrices. |
| `RepresentationMatrix(f)` | Given an element `f` of a finite dimensional affine algebra `Q` over a field, return the `d × d` representation matrix of `f` over the coefficient field (where `d = Dimension(Q)`). | Multiplication-by-`f` matrix in the monomial basis. |
| `IsUnit(f)` | Given an element `f` of a finite dimensional affine algebra `Q` over a field, return whether `f` is a unit. | Via representation matrix: invertible iff `det ≠ 0`. |
| `IsNilpotent(f)` | Given an element `f` of a finite dimensional affine algebra `Q` over a field, return whether `f` is nilpotent, and if so the smallest `q` such that `fq = 0`. | Via minimal polynomial (root at 0). |
| `MinimalPolynomial(f)` | Given an element `f` of a finite dimensional affine algebra `Q` over a field, return the minimal polynomial of `f` as a univariate polynomial over the coefficient field. | Minimal polynomial of the representation matrix. |

*Worked examples: H108E3 (minimal polynomial of √2 + ∛5 computed in Q[x,y]/(x²−2, y³−5)).*

---

## 108.6 Affine Algebras which are Fields

If the ideal J of relations defining A = K[x₁,…,xₙ]/J (K a field) is maximal, then A is
itself a field and can be used with any Magma algorithm that works over fields. Polynomial
factorization over such affine algebra fields is supported in any characteristic (since
V2.10). An affine algebra that is a field also has finite dimension over its coefficient
field, so all operations from Section 108.5 apply.

*(No additional intrinsics beyond those in Sections 108.2–108.5; this section is
illustrative.)*

*Worked examples: H108E4 (generic elliptic curve over Q(a,b,x)[y]/(y²−(x³+ax+b)); multiples of a generic point); H108E5 (factorization of polynomials over the same affine algebra); H108E6 (affine algebra isomorphic to a number field; factorization of the minimal polynomial).*

---

## 108.7 Rings and Fields of Fractions of Affine Algebras

Given any affine algebra Q = K[x₁,…,xₙ]/J (K a field), one may create the ring of
fractions R of Q: the set of fractions a/b where a, b ∈ Q and b is invertible. The
defining ideal J need not be zero-dimensional. Internally, the ring of fractions is
represented as an affine algebra over an appropriate rational function field, but the user
sees fractions with accessible numerators and denominators.

If J is prime, then R is the field of fractions of Q and may be used with any Magma
algorithm that works over fields (including polynomial factorization in any
characteristic). The construction proceeds via an extension/contraction step: if the ideal
has Krull dimension d, a sequence L of d maximally independent variables is found; the
ideal of relations over the resulting rational function field (in d variables) becomes
zero-dimensional, and appropriate maps are set up.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RingOfFractions(Q)` / `FieldOfFractions(Q)` | Given an affine algebra `Q` over a field `K`, return the ring of fractions of `Q`. `FieldOfFractions` additionally requires the defining ideal to be prime. | Extension/contraction: find `d` maximally independent variables (Krull dimension), extend to a rational function field in `d` variables so the ideal becomes zero-dimensional, then set up the fraction field. |
| `Numerator(a)` | Given an element `a` from the ring of fractions of affine algebra `Q`, return the numerator of `a` as an element of `Q`. | — |
| `Denominator(a)` | Given an element `a` from the ring of fractions of affine algebra `Q`, return the denominator of `a` as an element of `Q`. | — |

*Worked examples: H108E7 (field of fractions of Q[x,y]/(y²−x³−1); basic arithmetic and polynomial factorization over the fraction field); H108E8 (internal mechanics of the extension/contraction construction; explicit maps between the affine algebra and its fraction field).*

---

## 108.8 Bibliography

*(Chapter 108 contains no bibliography. The algorithms for ideal operations, Gröbner bases,
primary decomposition, and related computations are general Buchberger/Gröbner-basis
methods documented in the polynomial ring chapters of the Magma Handbook.)*

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Quotient ring / Gröbner basis reduction | `quo< >`, `P / J`, `AffineAlgebra< >` |
| Preimage-ring reduction (ideal arithmetic) | `DivisorIdeal`, `PreimageIdeal`, `PreimageRing`, `OriginalRing`, `I + J`, `I * J`, `I ^ n`, `I meet J`, `I eq J`, `I subset J` |
| Primality / primary / radical testing | `IsPrime`, `IsPrimary`, `IsRadical` |
| Primary decomposition | `PrimaryDecomposition` |
| Radical decomposition | `RadicalDecomposition` |
| Finite-dimensional representation (multiplication matrices) | `VectorSpace`, `MonomialBasis`, `MatrixAlgebra`, `RepresentationMatrix`, `HasFiniteDimension`, `Dimension` |
| Minimal polynomial of a representation matrix | `MinimalPolynomial`, `IsUnit`, `IsNilpotent` |
| Extension/contraction (fraction field via rational function field) | `RingOfFractions`, `FieldOfFractions`, `Numerator`, `Denominator` |
| Kernel of affine algebra map | `AffineAlgebraMapKernel` |
