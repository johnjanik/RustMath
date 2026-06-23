# Chapter 79 — Algebras

**Handbook part:** XI — Algebras
**Handbook pages:** 2421–2430 (PDF pages 2552–2563)

---

## Scope and overview

Chapter 79 is the introductory/overview chapter for Part XI of the Magma Handbook on Algebras.
Algebras are treated as free modules over a unital ring R together with an additional
(bilinear) multiplication. No a priori conditions are imposed on R beyond unitality, but some
functions require that an echelonization algorithm is available for modules over R; others
additionally require that R is a field. In particular, quotient algebras can only be constructed
over fields, because the quotient module is not necessarily free over a general R.

The most general representation is via structure constants, but for special types Magma uses
more efficient dedicated representations. The chapter surveys the seven main algebra categories
available in Magma, then gives the generic constructors and operations applicable across all or
most of them. Detailed treatments of each category appear in subsequent chapters.

Seven main categories are recognized:

1. **AlgGen** — general algebras defined by structure constants (top of the hierarchy).
2. **AlgAss** — associative algebras defined by structure constants (inherits from AlgGen).
3. **AlgQuat** — quaternion algebras as a special type of associative algebra (inherits from AlgAss).
4. **AlgLie** — Lie algebras defined by structure constants (inherits from AlgGen).
5. **AlgGrp** / **AlgGrpSub** — group algebras and their subalgebras (inherits from AlgAss).
6. **AlgMat** — matrix algebras (inherits from AlgAss).
7. **AlgFP** — finitely presented algebras (independent of the other categories).

---

## 79.1 Introduction

### 79.1.1 The Categories of Algebras

The hierarchy of algebra categories is: `AlgGen` at the top; `AlgAss` and `AlgLie` on the
next level inheriting from `AlgGen`; `AlgQuat`, `AlgGrp`, and `AlgMat` on a third level
inheriting from `AlgAss`. Finitely presented algebras (`AlgFP`) are independent of the
others.

---

## 79.2 Construction of General Algebras and their Elements

### 79.2.1 Construction of a General Algebra

Construction depends on the algebra category; individual chapters cover each type in detail.
The generic constructors listed here give an overview. A general algebra of dimension n over R
is specified by n³ structure constants: ei \* ej = Σ a^k_{ij} e_k.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Algebra< R, n \| Q >` | Creates a general algebra A of dimension n over ring R. Q is a sequence of n³ elements of R giving the structure constants: position (i−1)·n² + (j−1)·n + k of Q holds the coefficient a^k_{ij} of e_k in ei \* ej. | Structure constant algebra; no associativity check. |
| `AssociativeAlgebra< R, n \| Q >` | Creates the associative structure constant algebra returned by `Algebra< R, n \| Q >`. Parameter `Check` (BoolElt, default `true`) controls whether associativity is verified. Returns an algebra of type AlgAss. | Structure constants; optional associativity check. |
| `QuaternionAlgebra< K \| a, b >` | Creates the quaternion algebra A over field K on generators x and y with relations x² = a, y² = b, xy = −yx. | Quaternion algebra construction; category AlgQuat. |
| `LieAlgebra< R, n \| Q >` | Creates the Lie structure constant algebra as returned by `Algebra< R, n \| Q >`. Parameter `Check` (BoolElt, default `true`) controls whether the Lie axioms are verified. Returns type AlgLie. | Structure constants; optional Lie-algebra check. |
| `LieAlgebra(A)` | Given an associative algebra A, creates the Lie algebra on the same underlying space using the induced Lie bracket (x, y) → x \* y − y \* x. | Induced Lie product from associative multiplication. |
| `GroupAlgebra(R, G)` | Given a ring R and a group G, constructs the group algebra R[G] of dimension |G| over R. | Group algebra construction; category AlgGrp. |
| `MatrixAlgebra(R, n)` | Given a positive integer n and a ring R, creates the full matrix algebra M_n(R) of dimension n² over R. | Full matrix algebra; category AlgMat. |

### 79.2.2 Construction of an Element of a General Algebra

Construction of generic elements varies by algebra type and is described in the corresponding
chapters. The following generic constructors are available for all algebra categories.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Zero(A)` / `A ! 0` | Creates the zero element of algebra A. | — |
| `One(A)` / `A ! 1` | Creates the identity element of algebra A; an error occurs if no identity exists. | — |
| `Random(A)` | Returns a random element of algebra A, which must be defined over a finite ring. | — |

---

## 79.3 Construction of Subalgebras, Ideals and Quotient Algebras

### 79.3.1 Subalgebras and Ideals

When the coefficient ring R of an algebra A is a Euclidean domain, submodules and ideals of A
may be constructed. The `sub`, `lideal`, `rideal`, and `ideal` constructors all accept a list L
of any combination of: elements of A; sets or sequences of elements of A; subalgebras or ideals
of A; or sets or sequences of subalgebras or ideals of A. Each constructor returns the algebra
object together with the inclusion homomorphism f : result → A. For group algebras, the result
is of type AlgAss or the special type AlgGrpSub rather than the original algebra type.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `sub< A \| L >` | Creates the subalgebra S of A generated by the elements defined by list L. Returns S and the inclusion homomorphism f : S → A. | Module-theoretic subalgebra construction over a Euclidean domain. |
| `lideal< A \| L >` | Creates the left ideal I of A generated by the elements defined by L. Returns I and the inclusion homomorphism f : I → A. | Left ideal generation over a Euclidean domain. |
| `rideal< A \| L >` | Creates the right ideal I of A generated by the elements defined by L. Returns I and the inclusion homomorphism f : I → A. | Right ideal generation over a Euclidean domain. |
| `ideal< A \| L >` | Creates the two-sided ideal I of A generated by the elements defined by L. Returns I and the inclusion homomorphism f : I → A. | Two-sided ideal generation over a Euclidean domain. |

### 79.3.2 Quotient Algebras

Quotient algebras can only be constructed when the coefficient ring R is a field (so that the
quotient module is free).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `quo< A \| L >` | Creates the quotient algebra Q = A/I, where I is the two-sided ideal generated by the elements of A specified by list L. Returns Q (a structure constant algebra of type AlgAss if A is known associative, AlgGen otherwise) and the natural homomorphism f : A → Q. | Quotient module construction over a field; requires R a field. |
| `A / S` | The quotient of algebra A by the two-sided ideal closure of its subalgebra S. | As above. |

---

## 79.4 Operations on Algebras and Subalgebras

### 79.4.1 Invariants of an Algebra

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CoefficientRing(A)` / `BaseRing(A)` | Returns the coefficient ring (base ring) over which algebra A is defined. | — |
| `Dimension(A)` | Returns the dimension of algebra A. | — |
| `#A` | Returns the cardinality of algebra A when both R and the dimension of A are finite. Cannot be computed if the dimension is too large. | — |

### 79.4.2 Changing Rings

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ChangeRing(A, S)` | Given algebra A with base ring R and a ring S, constructs the algebra B with base ring S by coercing coefficients of A into S, together with the homomorphism A → B. Cannot be applied to algebras of type AlgGrpSub. | Coefficient coercion. |
| `ChangeRing(A, S, f)` | As above but uses an explicit map f : R → S to map coefficients. Cannot be applied to AlgGrpSub. | Coefficient mapping via f. |

### 79.4.3 Bases

Every algebra comes with a basis corresponding to its underlying module structure, with the
sole exception of group algebras in the "Terms" representation (where the dimension may be too
large to create vectors of that degree).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BasisElement(A, i)` / `A . i` | Returns the i-th basis element of algebra A. | — |
| `Basis(A)` | Returns the basis of A as a sequence of elements. For AlgGrpSub, the returned elements belong to the full group algebra of which A is a subalgebra. | — |
| `IsIndependent(Q)` | Given a sequence Q of elements of the R-algebra A, returns true if they are linearly independent over R; otherwise false. | Linear algebra over R. |
| `ExtendBasis(S, A)` / `ExtendBasis(Q, A)` | Given algebra A and either a subalgebra S of dimension m or a sequence Q of m linearly independent elements, returns a basis of A whose first m elements are the basis of S (resp. the elements of Q). | Basis extension over R. |

### 79.4.4 Decomposition of an Algebra

An algebra A can be regarded as a left or right module for itself. When A is defined over a
finite field, module-decomposition machinery over finite fields can be used to investigate the
structure of A.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CompositionSeries(A)` | Computes a composition series for algebra A. Returns: (a) a sequence of subalgebras forming an ascending chain whose successive quotients are irreducible A-modules; (b) the composition factors as structure constant algebras; (c) a transformation matrix to a basis compatible with the composition series. | Module decomposition over a finite field. |
| `CompositionFactors(A)` | Computes the composition factors of a composition series for A (same as the second return value of `CompositionSeries`), but often much faster. | Module decomposition over a finite field. |
| `MinimalLeftIdeals(A : -)` | Returns the minimal left ideals of A in non-decreasing size. Parameter `Limit` (RngIntElt, default ∞): if set to n, at most n ideals are computed, and the second return value indicates whether all were found. | — |
| `MinimalRightIdeals(A : -)` | Returns the minimal right ideals of A in non-decreasing size. Same `Limit` parameter. | — |
| `MinimalIdeals(A : -)` | Returns the minimal two-sided ideals of A in non-decreasing size. Same `Limit` parameter. | — |
| `MaximalLeftIdeals(A : -)` | Returns the maximal left ideals of A in non-decreasing size. Parameter `Limit` (RngIntElt, default ∞). | — |
| `MaximalRightIdeals(A : -)` | Returns the maximal right ideals of A in non-decreasing size. Same `Limit` parameter. | — |
| `MaximalIdeals(A : -)` | Returns the maximal two-sided ideals of A in non-decreasing size. Same `Limit` parameter. | — |
| `JacobsonRadical(A)` | Constructs the Jacobson (nilpotent) radical of A: the intersection of the maximal (left, right, or two-sided) ideals of A. | Ideal-intersection / radical computation. |
| `IsSemisimple(A)` | Returns true if the Jacobson radical of A is trivial; otherwise false. | Via `JacobsonRadical`. |
| `IsSimple(A)` | Returns true if A has no non-trivial composition factor; otherwise false. | Via `CompositionSeries`. |

*Worked example: H79E1 — constructs a division algebra of dimension 4 over Q as a matrix algebra; verifies integrality of (1+i+j+k)/2; builds the maximal order as a structure constant algebra over Z using `ChangeRing` and `AssociativeAlgebra`; checks simplicity at odd primes using `IsSimple` and `ChangeRing` to GF(p); shows the composition series at p = 2 via `CompositionSeries`.*

### 79.4.5 Operations on Subalgebras

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsZero(A)` | Returns true if algebra A is trivial; otherwise false. | — |
| `A eq B` | Returns true if subalgebras A and B (with a common superalgebra) are equal; otherwise false. | — |
| `A ne B` | Returns true if A and B are not equal; otherwise false. | — |
| `A subset B` | Returns true if A is a subalgebra of B; otherwise false. | — |
| `A notsubset B` | Returns true if A is not a subalgebra of B; otherwise false. | — |
| `A meet B` | The intersection of algebras A and B, which must have a common superalgebra. | Module intersection. |
| `A * B` | The algebra product A \* B of algebras A and B, which must have a common superalgebra. | Product of subalgebras. |
| `A ^ n` | The left-normed n-th power of algebra A: ((…(A \* A) \* …) \* A). | Iterated algebra product. |
| `Morphism(A, B)` | Returns the morphism from A to B: the embedding A → B if A is a subalgebra of B, or the natural epimorphism A → B if B is a quotient algebra of A. | — |

---

## 79.5 Operations on Elements of an Algebra

### 79.5.1 Operations on Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `a + b` | Sum of elements a and b of an algebra A. | — |
| `-a` | Negation of algebra element a. | — |
| `a - b` | Difference of elements a and b. | — |
| `a * b` | Product of elements a and b of algebra A. | — |
| `a * r` / `r * a` | Product of algebra element a and ring element r ∈ R (the coefficient ring). | Scalar multiplication. |
| `a / r` | Product of a and the inverse 1/r ∈ R; requires R to be a field. | Scalar division. |
| `a ^ n` | n-th power of element a. If n > 0: left-normed product ((…(a\*a)\*a…)\*a). If n = 0 and A has an identity: returns the identity. If n < 0 and a has an inverse a⁻¹: the (−n)-th power of a⁻¹. | Repeated multiplication. |
| `MinimalPolynomial(a)` | Returns the minimal polynomial of algebra element a, when R is a field or the integer ring. | Linear algebra / Cayley–Hamilton. |
| `Parent(a)` | For an algebra element a, returns the algebra A to which a belongs. | — |

### 79.5.2 Comparisons and Membership

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `a eq b` | Returns true if elements a and b of an algebra A are equal; otherwise false. | — |
| `a ne b` | Returns true if a and b are not equal; otherwise false. | — |
| `a in A` | Returns true if a is an element of algebra A; otherwise false. | — |
| `a notin A` | Returns true if a is not an element of algebra A; otherwise false. | — |

### 79.5.3 Predicates on Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsZero(a)` | Returns true if algebra element a is zero; otherwise false. | — |
| `IsOne(a)` | Returns true if a is the identity element; otherwise false. | — |
| `IsMinusOne(a)` | Returns true if a is the negative of the identity element; otherwise false. | — |
| `IsUnit(a)` | Returns true if a is a unit and returns the inverse; otherwise false. | — |
| `IsRegular(a)` | Returns true if a is regular (not a zero divisor); otherwise false. | — |
| `IsZeroDivisor(a)` | Returns true if a is a zero divisor; otherwise false. | — |
| `IsIdempotent(a)` | Returns true if a is idempotent (a² = a); otherwise false. | — |
| `IsNilpotent(a)` | Returns true if a is nilpotent (aⁿ = 0 for some n ≥ 0); otherwise false. If true, also returns the minimal such n. | — |

---

## 79.6 Bibliography

This chapter carries no separate bibliography of its own. It is a general overview chapter;
all attribution and references appear in the chapters for the individual algebra categories
(Chapters 80–86 and subsequent chapters in Part XI).

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Structure constant algebra construction | `Algebra< >`, `AssociativeAlgebra< >`, `LieAlgebra< >` |
| Quaternion algebra construction | `QuaternionAlgebra< >` |
| Group algebra construction | `GroupAlgebra` |
| Full matrix algebra construction | `MatrixAlgebra` |
| Subalgebra / ideal generation over a Euclidean domain | `sub< >`, `lideal< >`, `rideal< >`, `ideal< >` |
| Quotient algebra over a field | `quo< >`, `A / S` |
| Module decomposition over a finite field (composition series) | `CompositionSeries`, `CompositionFactors` |
| Jacobson radical / semisimplicity / simplicity | `JacobsonRadical`, `IsSemisimple`, `IsSimple` |
| Minimal / maximal ideal enumeration | `MinimalLeftIdeals`, `MinimalRightIdeals`, `MinimalIdeals`, `MaximalLeftIdeals`, `MaximalRightIdeals`, `MaximalIdeals` |
| Basis extension and linear independence | `ExtendBasis`, `IsIndependent` |
| Ring change (coefficient coercion or map) | `ChangeRing` |
| Minimal polynomial of an element | `MinimalPolynomial` |
| Element predicates (units, idempotents, nilpotents, zero divisors) | `IsUnit`, `IsRegular`, `IsZeroDivisor`, `IsIdempotent`, `IsNilpotent`, `IsZero`, `IsOne`, `IsMinusOne` |
