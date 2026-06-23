# Chapter 107 — Local Polynomial Rings

**Handbook part:** XIV — Commutative Algebra
**Handbook pages:** 3273–3284 (PDF pages 3404–3417)

---

## Scope and overview

This chapter describes local polynomial rings in Magma. Given a field K, the local polynomial
ring K[x₁,…,xₙ]⟨x₁,…,xₙ⟩ is the collection of all rational functions f/g with g(0,…,0) ≠ 0;
it is a local ring (unique maximal ideal) and is called the **localization** of the global
multivariate ring K[x₁,…,xₙ] at the origin. Magma restricts elements to strict polynomials
(non-trivial denominators are not supported); units are automatically removed from standard
bases, so this restriction is harmless in practice.

The fundamental object for computation is the **standard basis** of an ideal — the counterpart
to a Gröbner basis in the global setting. Magma uses the **Mora normal form and standard basis
algorithm with homogenization** (see [CLO98, Sec. 4.4]), which reduces to global Gröbner basis
algorithms. Standard bases are currently only supported over fields. Unlike the global case,
standard bases are not unique in general (lower-order terms may differ), but leading monomials
are always sorted and are unique.

Elements are ordered with respect to a **local monomial order**, which is the negation of a
global order: the monomial 1 is smallest, making polynomials resemble formal power series.
All global polynomial arithmetic carries over automatically; see Chapter 24. Computations with
R-modules over a local polynomial ring R are also fully supported (see Chapter 109).

The reader should be familiar with multivariate polynomial rings and their ideals (Chapters 24
and 105) before using this chapter. References: [CLO98, Chapter 4] and [GP02, Chapter 1] for
theory; [DL06] for further background.

---

## 107.2 Elements and Local Monomial Orders

A local monomial order on R of rank n is a total order < on the monomials of R such that
s ≤ 1 for all monomials s, s ≤ t implies su ≤ tu for all s, t, u, and < is a well-ordering.
Each local order is the negation of a corresponding global order. The three available orders
are listed below (further orders will be added in future versions). In all definitions s and t
are monomials from a ring of rank n; the quoted string is the argument to pass to intrinsics
expecting a monomial order.

### 107.2.1 Local Lexicographical: llex

s < t iff there exists 1 ≤ i ≤ n such that the j-th exponents of s and t are equal for
i < j ≤ n, but the i-th exponent of s is **greater** than that of t. Specified by `"llex"`.
This is the negation of the global lexicographical order (with variables in reverse: the
first variable is the greatest).

### 107.2.2 Local Graded Lexicographical: lglex

s < t iff the total degree of s is greater than that of t, or they have equal total degree
and s > t under the (global) lexicographical order. Specified by `"lglex"`.
This is the negation of the global glex order.

### 107.2.3 Local Graded Reverse Lexicographical: lgrevlex

s < t iff the total degree of s is greater than that of t, or they have equal total degree
and s < t under the global lexicographical order applied to the exponents in **reverse** order.
Specified by `"lgrevlex"`. This is the negation of the global grevlex order.

---

## 107.3 Local Polynomial Rings and Ideals

### 107.3.1 Creation of Local Polynomial Rings and Accessing their Monomial Orders

Local polynomial rings are created from a coefficient field, the number of variables, and an
optional monomial order. The default order is the local lexicographical order (`"llex"`).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LocalPolynomialRing(K, n)` | Create a local polynomial ring in n > 0 variables over field K. Default order: local lexicographical (`"llex"`). | — |
| `LocalPolynomialRing(K, n, order)` / `LocalPolynomialAlgebra(K, n, order)` | Create a local polynomial ring in n > 0 variables over ring K with the given monomial order string `order` (see §107.2). | — |
| `LocalPolynomialRing(K, n, T)` | Create a local polynomial ring in n > 0 variables over field K with the order given by tuple T. T must have components matching the valid order arguments from §107.2; such a tuple is also returned by `MonomialOrder`. | — |
| `MonomialOrder(R)` | Given a local polynomial ring R (or an ideal thereof), return a description of the monomial order of R as a tuple (matching the relevant arguments in §107.2). The tuple may be passed directly as the third argument to `LocalPolynomialRing`. | — |
| `MonomialOrderWeightVectors(R)` | Given a polynomial ring R of rank n (or an ideal thereof), return the weight vectors of the underlying monomial order as a sequence of n sequences of n rationals (see [CLO98, p. 153]). | — |
| `Localization(R)` | Given a (global) multivariate polynomial ring R = K[x₁,…,xₙ] (or an ideal I of such R), return the localization K[x₁,…,xₙ]⟨x₁,…,xₙ⟩ of R (or the corresponding ideal of the localization). Variable print names are carried over. | — |

*Worked examples: H107E1 (constructing local polynomial rings with llex and lgrevlex orders; inspecting `MonomialOrder` and `MonomialOrderWeightVectors`; observing monomial sorting in elements).*

### 107.3.2 Creation of Ideals and Accessing their Bases

Within the local polynomial ring context, "basis" means an ordered sequence of polynomials
generating an ideal (may contain duplicates and zeros; not a vector-space basis).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ideal< R \| L >` | Given a local polynomial ring R, return the ideal generated by the elements in list L. Each term of L may be: (a) an element of R; (b) a set or sequence of elements of R; (c) an ideal of R; (d) a set or sequence of ideals of R. | — |
| `Ideal(B)` | Given a set or sequence B of polynomials from a local polynomial ring R, return the ideal of R generated by B, with basis B. Equivalent to the `ideal< >` constructor but more convenient for a sequence. | — |
| `Ideal(f)` | Given a polynomial f from a local polynomial ring R, return the principal ideal of R generated by f. | — |
| `Basis(I)` | Given an ideal I, return the current basis. Returns the standard basis if it has been computed; otherwise the original basis. | — |
| `BasisElement(I, i)` | Given an ideal I and integer i, return the i-th element of the current basis of I. Equivalent to `Basis(I)[i]`. | — |

---

## 107.4 Standard Bases

Computation in ideals of local polynomial rings relies on standard bases — the local analogue
of Gröbner bases. Magma uses the **Mora normal form algorithm with the homogenization technique**
([CLO98, Sec. 4.4]), which reduces the standard basis computation to global Gröbner basis
algorithms. The verbose flags are shared with the global Gröbner basis computation; see
§105.4.6 for details. A standard basis is automatically generated whenever needed; the
functions below allow explicit control.

### 107.4.1 Construction of Standard Bases

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `StandardBasis(I)` | Given an ideal I of a local polynomial ring, force computation of the standard basis and return it. | Mora normal form + standard basis algorithm with homogenization **[CLO98, Sec. 4.4]**. |
| `StandardBasis(S)` | Given a set or sequence S of polynomials of a local polynomial ring R, return a standard basis of the ideal generated by S as a sorted sequence. | Mora normal form + standard basis algorithm with homogenization **[CLO98, Sec. 4.4]**. |

*Worked examples: H107E2 (standard basis of the ideal from [CLO98, p. 167]; factorisation showing no unit factors in the basis elements). H107E3 (comparison: the standard basis of a localized ideal can be far smaller than the Gröbner basis of the global ideal; `QuotientDimension` comparison 12 vs. 4).*

---

## 107.5 Operations on Ideals

The ring R itself is treated as a valid ideal (the ideal containing 1).

### 107.5.1 Basic Operations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `I + J` | Sum of ideals I and J of the same polynomial ring P: the ideal generated by the union of their generators. | — |
| `I * J` | Product of ideals I and J of the same polynomial ring P: the ideal generated by all pairwise products of generators. | — |
| `I ^ k` | k-th power of ideal I of polynomial ring P, for integer k. | — |
| `QuotientDimension(I)` | Given an ideal I of a local polynomial ring R over a field K, return the dimension of R/I as a K-vector space. (Distinct from `Dimension`, which returns the Krull dimension.) | — |
| `Generic(I)` | Given an ideal I of a local polynomial ring R, return R. | — |
| `LeadingMonomialIdeal(I)` | Given an ideal I, return the leading monomial ideal of I: the ideal generated by all leading monomials of I. | — |
| `I meet J` | Intersection of ideals I and J of the same polynomial ring P. | — |
| `&meet S` | Given a set or sequence S of ideals of the same local polynomial ring R, return the intersection of all ideals in S. | — |

### 107.5.2 Ideal Predicates

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `I eq J` | Return whether ideals I and J of the same polynomial ring P are equal. | — |
| `I ne J` | Return whether ideals I and J of the same polynomial ring P are not equal. | — |
| `I notsubset J` | Return whether I is not contained in J (ideals in the same polynomial ring P). | — |
| `I subset J` | Return whether I is contained in J (ideals in the same polynomial ring P). | — |
| `IsZero(I)` | Return whether ideal I of the local polynomial ring R is the zero ideal (contains only zero). | — |
| `IsProper(I)` | Return whether ideal I of the local polynomial ring R is proper (strictly contained in R, i.e. the standard basis does not contain 1). | — |
| `IsZeroDimensional(I)` | Return whether ideal I is zero-dimensional: whether R/I has non-zero finite dimension as a vector space over the coefficient field. (Note: R itself has dimension −1 and is not zero-dimensional.) | — |

*Worked examples: H107E4 (constructing ideals in Q[x,y,z]; computing product A = I*J and intersection M = I meet J; testing A eq M and A subset M). H107E5 (element membership and normal form in ideals of the localization; `NormalForm` returns 0 iff the element is in the ideal).*

### 107.5.3 Operations on Elements of Ideals

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `f in I` | Given a polynomial f from a local polynomial ring R and an ideal I of R, return whether f is in I. | Membership tested via `NormalForm`. |
| `NormalForm(f, I)` | Given a polynomial f from a local polynomial ring R and an ideal I of R, return a normal form of f with respect to (the standard basis of) I. The normal form is zero if and only if f is in I. | Mora normal form **[CLO98, Sec. 4.4]**. |
| `f notin I` | Given a polynomial f from a local polynomial ring R and an ideal I of R, return whether f is not in I. | — |

---

## 107.6 Changing Coefficient Ring

The `ChangeRing` function allows changing the coefficient ring of a local polynomial ring or
ideal. If K and L are fields with K a known subfield of L and the current basis of I is a
standard basis, then the basis of the result is automatically marked as a standard basis.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ChangeRing(I, L)` | Given an ideal I of local polynomial ring R = K[x₁,…,xₙ] and a field L, construct the ideal J of S = L[x₁,…,xₙ] by coercing the coefficients of the basis of I into L. Requires that all elements of K can be automatically coerced into L. | — |

---

## 107.7 Changing Monomial Order

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ChangeOrder(I, Q)` | Given an ideal I of local polynomial ring R = K[x₁,…,xₙ] and a local polynomial ring S of rank n (possibly with a different order), return the ideal J of S corresponding to I and the isomorphism f: R → S mapping R.i to S.i for each i. | — |
| `ChangeOrder(I, order)` | Given an ideal I of polynomial ring P = R[x₁,…,xₙ] and a monomial order string `order` (see §107.2), construct Q = R[x₁,…,xₙ] with that order, and return the ideal J of Q corresponding to I and the isomorphism f: P → Q (mapping P.i to Q.i). | — |

---

## 107.8 Dimension of Ideals

The dimension of an ideal I of K[x₁,…,xₙ]⟨x₁,…,xₙ⟩ is defined as the maximum cardinality
of all independent sets modulo I (analogous to the global polynomial ring case; see §107.8 for
details). The full local polynomial ring R has dimension −1 by convention.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Dimension(I)` | Given an ideal I of a local polynomial ring R defined over a field, return the Krull dimension d of I, together with a sorted sequence U of integers of length d such that the variables of P corresponding to U form a maximally independent set modulo I. If I is the full local ring R, the dimension is −1 and the second return value is not set. | Algorithm of **[BW93, p. 449]**. |

---

## 107.9 Bibliography

| Key | Reference |
|-----|-----------|
| **[BW93]** | Thomas Becker and Volker Weispfenning. *Gröbner Bases.* Graduate Texts in Mathematics. Springer, New York–Berlin–Heidelberg, 1993. |
| **[CLO98]** | David Cox, John Little, and Donal O'Shea. *Using Algebraic Geometry.* Graduate Texts in Mathematics. Springer, New York–Berlin–Heidelberg, 1998. |
| **[DL06]** | Wolfram Decker and Christoph Lossen. *Computing in Algebraic Geometry,* volume 16 of Algorithms and Computation in Mathematics. Springer, New York–Berlin–Heidelberg, 2006. |
| **[GP02]** | G.-M. Greuel and G. Pfister. *A Singular Introduction to Commutative Algebra.* Springer-Verlag, Berlin–Heidelberg–New York, 2002. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Mora normal form + standard basis algorithm with homogenization **[CLO98, Sec. 4.4]** | `StandardBasis(I)`, `StandardBasis(S)`, `NormalForm(f, I)`, `f in I`, `f notin I` |
| Dimension algorithm (independent sets modulo I) **[BW93, p. 449]** | `Dimension(I)` |
| Local monomial order theory **[CLO98, Sec. 4.3], [DL06, Sec. 9.1], [GP02, Sec. 1.2]** | `LocalPolynomialRing`, `MonomialOrder`, `MonomialOrderWeightVectors`, `ChangeOrder` |
| Ideal arithmetic (sum, product, power, intersection) | `+`, `*`, `^`, `meet`, `&meet` |
| Ideal membership and normal form | `in`, `notin`, `NormalForm`, `QuotientDimension`, `LeadingMonomialIdeal` |
| Ring and order change | `ChangeRing`, `ChangeOrder`, `Localization` |
