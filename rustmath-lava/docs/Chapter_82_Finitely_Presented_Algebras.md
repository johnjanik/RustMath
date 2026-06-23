# Chapter 82 — Finitely Presented Algebras

**Handbook part:** XI — Algebras
**Handbook pages:** 2469–2503 (PDF pages 2600–2637)

---

## Scope and overview

This chapter describes finitely presented algebras (FPAs) in Magma. An FPA is a quotient
of a free associative algebra by an ideal of relations. To compute with these ideals, one
constructs noncommutative Gröbner bases (GBs), which have many parallels with the standard
commutative GBs discussed in Chapter 105. At the heart of the theory is a noncommutative
version of the Buchberger algorithm that computes a GB of an ideal of an algebra starting
from an arbitrary basis (generating set) of the ideal. One significant difference with the
commutative case is that a noncommutative GB may not be finite for a finitely-generated ideal.
For overviews of the theory and the basic algorithms see **[Mor94, Li02]**.

Magma also contains an implementation of a noncommutative generalization of the Faugère F4
algorithm (due to Allan Steel), based on sparse linear algebra techniques, which usually
performs dramatically better than the Buchberger algorithm and so is used by default.

The chapter also covers **exterior algebras**, which are skew-commutative quotients of free
algebras by the relations x²_i = 0 and x_i x_j = −x_j x_i (i ≠ j). Because elements can be
written in terms of commutative monomials (via a collection algorithm), the associated
algorithms are much more efficient than in the general noncommutative case, and a Gröbner
basis of an ideal of an exterior algebra is always finite (the whole exterior algebra has
dimension 2^n as a vector space).

A final major section covers **vector enumeration**, an algorithm (based on the
Haselgrove–Leech–Trotter variant of the Todd–Coxeter algorithm) for converting a finitely
presented module for an FP-algebra into a concrete vector space with explicit matrix action.
This is used to compute matrix representations of group algebras of fp-groups and, more
generally, Hecke algebras and quotients of polynomial rings.

---

## 82.1 Introduction

*(Prose section — no intrinsics. See Scope and overview above.)*

---

## 82.2 Representation and Monomial Orders

Let A = K⟨x₁, …, xₙ⟩ be the free algebra of rank n over a field K. Monomials are
associative products of variables (monoid words). Elements of A, called noncommutative
polynomials, are finite sums of terms (coefficient × monomial). Terms are sorted by an
admissible order satisfying: (a) p < q ⇒ pr < qr and sp < sq; (b) p = qr ⇒ p > q and
p > r. Currently Magma supports only the noncommutative **graded-lexicographical order
(glex)**, which compares degrees first, then breaks ties by left-lexicographic comparison.
(There is no admissible lexicographic order in the noncommutative case.)

*(Prose section — no intrinsics.)*

---

## 82.3 Exterior Algebras

Available since V2.15 (December 2008). Exterior algebras may be constructed with
`ExteriorAlgebra` (§82.4.1) and all operations applicable to general FP-algebras apply to
them as well. Modules over exterior algebras are covered in Chapter 109.

*(Prose section — no intrinsics beyond the constructor listed in §82.4.1.)*

---

## 82.4 Creation of Free Algebras and Elements

### 82.4.1 Creation of Free Algebras

Free algebras are objects of type `AlgFr` with elements of type `AlgFrElt`. Algebras may
only be created over fields.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `FreeAlgebra(K, n)` | Create a free algebra in n > 0 variables over the field K. Angle-bracket notation assigns names to indeterminates, e.g. `F<a,b,c> := FreeAlgebra(GF(2), 3)`. | — |
| `ExteriorAlgebra(K, n)` | Create an exterior algebra in n > 0 variables over the field K. The algebra is the quotient of K⟨x₁,…,xₙ⟩ by x²_i = 0 and x_i x_j = −x_j x_i (i ≠ j). | Collection algorithm (skew-commutativity). GB always finite; dimension 2^n. |

### 82.4.2 Print Names

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AssignNames(~F, s)` | Procedure: change the names of the indeterminates of free algebra F. The i-th indeterminate is given the name of the i-th element of the string sequence s; shorter sequences leave remaining names unchanged. Does not assign Magma identifiers — use angle-bracket notation at creation for that. | — |
| `Name(F, i)` | Return the i-th indeterminate of free algebra F as an element of F. | — |

### 82.4.3 Creation of Polynomials

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `F . i` | Return the i-th indeterminate (1 ≤ i ≤ n) of free algebra F as an element of F. | — |
| `elt< R \| a >` / `R ! s` / `elt< R \| s >` | Element constructor: given a free algebra F = R⟨x₁,…,xₙ⟩ and an element a coercible into the coefficient ring R, returns the constant polynomial a; if a is already in F it is returned unchanged. Only useful for trivial (scalar) construction. | — |
| `One(F)` / `Identity(F)` | Return the multiplicative identity of F. | — |
| `Zero(F)` / `Representative(F)` | Return the zero element / a representative element of F. | — |

---

## 82.5 Structure Operations

### 82.5.1 Related Structures

Multivariate free algebras belong to the Magma category `AlgFr`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BaseRing(F)` / `CoefficientRing(F)` | Return the coefficient ring of the free algebra F. | — |
| `Category(F)` / `Parent(F)` / `PrimeRing(F)` | Standard generic ring operations. | — |

### 82.5.2 Numerical Invariants

Note: the `#` operator returns a value only for finite (quotients of) free algebras.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Rank(F)` | Return the number of indeterminates of free algebra F over its coefficient ring. | — |
| `Characteristic(F)` / `# F` | Return the characteristic / cardinality of F. | — |

### 82.5.3 Homomorphisms

A homomorphism from K⟨x₁,…,xₙ⟩ requires n + 1 pieces of information: a map on the
coefficient ring K and images of the n indeterminates. The coefficient ring map is optional
(defaults to coercion).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `hom< F -> S \| f, y1, ..., yn >` / `hom< F -> S \| y1, ..., yn >` | Given free algebra F = K⟨x₁,…,xₙ⟩, a ring or associative algebra S (including another FP-algebra or a matrix algebra), a map f : K → S (optional), and n elements y₁,…,yₙ ∈ S: create the homomorphism g : F → S by g(r·x₁^a₁·…·xₙ^aₙ) = f(r)·y₁^a₁·…·yₙ^aₙ, extended by linearity. No check that the map is a genuine homomorphism is performed. | — |

*Worked example: H82E1 (mapping F = Q⟨x,y,z⟩ into itself by x↦xy, y↦yx, z↦zx; then into a 2×2 matrix algebra).*

---

## 82.6 Element Operations

### 82.6.1 Arithmetic Operators

Multiplication is associative but noncommutative.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `+a` / `-a` | Unary plus / negation. | — |
| `a + b` / `a - b` / `a * b` / `a ^ k` / `a / b` / `a div b` | Standard ring arithmetic. `*` is noncommutative. `^` is integer power. `/` and `div` divide by a scalar or a unit. | — |
| `a +:= b` / `a -:= b` / `a *:= b` / `a div:= b` | In-place assignment variants. | — |

### 82.6.2 Equality and Membership

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `a eq b` / `a ne b` | Test equality / inequality of noncommutative polynomials. | — |
| `a in R` / `a notin R` | Test membership of element a in ring / algebra R. | — |

### 82.6.3 Predicates on Algebra Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsZero(f)` / `IsOne(f)` / `IsMinusOne(f)` | Test whether f is zero / the identity / the negation of the identity. | — |
| `IsNilpotent(f)` / `IsIdempotent(f)` | Test nilpotency / idempotency of f. | — |
| `IsUnit(f)` / `IsZeroDivisor(f)` / `IsRegular(f)` | Test whether f is a unit / zero divisor / regular element. | — |
| `IsIrreducible(f)` / `IsPrime(f)` | Test irreducibility / primality of f. | — |

### 82.6.4 Coefficients, Monomials, Terms and Degree

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Coefficients(f)` | Given a noncommutative polynomial f with coefficients in R, return a sequence of elements of R occurring as coefficients of the monomials in f, in the same order as `Monomials(f)`. | — |
| `LeadingCoefficient(f)` | Return the leading coefficient of f (coefficient of the leading monomial with respect to the monomial order). | — |
| `TrailingCoefficient(f)` | Return the trailing coefficient of f (coefficient of the last monomial with respect to the monomial order). | — |
| `MonomialCoefficient(f, m)` | Return the coefficient of monomial m in f as an element of R. | — |
| `Monomials(f)` | Return the sequence of monomials (monoid words) occurring in f, ordered by the monomial order. Corresponds exactly to `Coefficients(f)`. | — |
| `LeadingMonomial(f)` | Return the leading monomial of f (first monomial with respect to the ordering). | — |
| `Terms(f)` | Return the sequence of non-zero terms of f as elements of F, ordered by the monomial order. The i-th term equals the i-th coefficient times the i-th monomial. | — |
| `LeadingTerm(f)` | Return the leading term of f (product of leading monomial and leading coefficient). | — |
| `TrailingTerm(f)` | Return the trailing term of f (last term with respect to the monomial order). | — |
| `Length(m)` | Given a noncommutative monomial (word) m, return its length (number of letters). Note: differs from the commutative case, where the number of terms in a polynomial is returned. | — |
| `m[i]` | Given a noncommutative monomial m of length l and 1 ≤ i ≤ l, return the i-th letter of m. | — |
| `TotalDegree(f)` | Return the total degree of f: the maximum of the lengths of all monomials in f. Returns −1 for the zero polynomial. | — |
| `LeadingTotalDegree(f)` | Return the length of the leading monomial of f. | — |

*Worked example: H82E2 (illustrating Coefficients, Monomials, Terms, MonomialCoefficient, LeadingTerm, Length, m[i] for f = (3xy − 2yz)(4x − 7zy) + 23xyz).*

### 82.6.5 Evaluation

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Evaluate(f, s)` | Given f ∈ F = R⟨x₁,…,xₙ⟩ and a sequence or tuple s of length n, return f evaluated by substituting xᵢ = s[i]. If elements of s can be lifted into R, the result is in R; otherwise generic evaluation is performed and the result has the same type as the elements of s. Behaves as the hom constructor. | — |

*Worked example: H82E3 (evaluating g = xy + yz at [1,2,3] giving 8 ∈ Q; and at [y,x,z] giving xz + yx ∈ F).*

---

## 82.7 Ideals and Gröbner Bases

Magma supports left-sided, right-sided, and two-sided ideals of free algebras. Quotients
are supported only for two-sided ideals. The "basis" of an ideal is an ordered sequence of
polynomials generating it (may contain duplicates and zeros — not a vector-space basis).

A Gröbner basis may not be finite; the Buchberger or F4 algorithms may therefore run
indefinitely. One can interrupt with Ctrl-C, or use `GroebnerBasis(S, d)` to compute a
truncated degree-d basis. Magma always produces the unique sorted minimal reduced GB.

### 82.7.1 Creation of Ideals

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ideal< A \| L >` | Given free algebra A over a field K, return the two-sided ideal of A generated by the elements specified in list L. L may contain: elements of A, sets/sequences of elements, ideals of A, or sets/sequences of ideals. | — |
| `lideal< A \| L >` | As above but returns the left-sided ideal. | — |
| `rideal< A \| L >` | As above but returns the right-sided ideal. | — |
| `Basis(I)` | Return the current basis of ideal I. If a Gröbner basis has been computed, that is returned instead of the original generators. | — |
| `BasisElement(I, i)` | Return the i-th element of the current basis of I. Equivalent to `Basis(I)[i]`. | — |

### 82.7.2 Gröbner Bases

GBs may be computed for left-, right-, or two-sided ideals, but for single-sided ideals the GBs are generally weak (rarely differing from the original generators).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Groebner(I: Faugere)` | (Procedure.) Explicitly force a Gröbner basis for ideal I to be constructed. Parameter `Faugere` (BoolElt, default true): if true and the field is finite or Q, uses the noncommutative F4 algorithm; otherwise uses the Buchberger algorithm. Normally unnecessary since Magma computes the GB automatically when needed. | (1) Noncommutative generalization of Faugère F4 **[Fau99]** (Allan Steel, sparse linear algebra, two-sided ideals over finite fields or Q); (2) Noncommutative Buchberger algorithm **[CLO96, Chap. 2, §7]** (any field). F4 is the default as it is usually dramatically faster. |
| `GroebnerBasis(I: Faugere)` | Force the GB of ideal I to be computed and return it. Same parameters as `Groebner`. | Same as `Groebner`. |
| `GroebnerBasis(S: Faugere)` | Given a set or sequence S of polynomials, return the unique GB of the two-sided ideal generated by S as a sorted sequence. Useful for computing GBs without constructing an ideal object. Same parameters as `Groebner`. | Same as `Groebner`. |
| `GroebnerBasis(S, d: Faugere)` | Given a set or sequence S of polynomials, return the degree-d (truncated) Gröbner basis: all S-polynomial pairs with total degree > d are ignored. For homogeneous ideals the result equals all GB elements of degree ≤ d, and membership is decidable by normal-form zero test; for non-homogeneous ideals these properties may not hold. Same parameters as `Groebner`. | Same as `Groebner`, truncated to degree d. |

### 82.7.3 Verbosity

Separate verbose flags for each algorithm; the all-encompassing `"Groebner"` flag subsumes all others. `false` ≡ level 0 (silent); `true` ≡ level 1 (minimal). Each `SetVerbose` has a corresponding `GetVerbose`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SetVerbose("Groebner", v)` | (Procedure.) Set verbose level for all Gröbner basis algorithms to v (levels 0–4). Level 1 gives minimal output; higher levels can also be tuned per sub-algorithm. | — |
| `SetVerbose("Buchberger", v)` | (Procedure.) Set verbose level for the Buchberger algorithm (levels 0–4). Effective level is max(v, Groebner flag). | — |
| `SetVerbose("Faugere", v)` | (Procedure.) Set verbose level for the Faugère algorithm (levels 0–3). Effective level is max(v, Groebner flag). | — |

### 82.7.4 Related Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `MarkGroebner(I)` | (Procedure.) Mark the current basis of ideal I as its Gröbner basis without recomputing. The basis must exactly equal the unique (reverse-)sorted minimal reduced GB; results are unpredictable otherwise. Useful when the GB was computed externally. | — |
| `Reduce(S)` | Given a set or sequence S of polynomials, return the reduction of S: each element is reduced to normal form with respect to the others, zero elements are removed, and the result is sorted. Useful for simplifying a generating set that is not a GB. All GBs returned by Magma are automatically reduced. | Interreduction (normal form with respect to the remaining elements). |

*Worked example: H82E4 (left-, right-, two-sided ideals of Q⟨x,y,z⟩ generated by B = [x²−yz, xy−yz, yx−z², y³−xz]; GBs in left/right cases are unchanged from B; two-sided GB has 18 elements; truncated GBs at degrees 2, 3, 4, 5).*

---

## 82.8 Basic Operations on Ideals

Note: the free algebra F itself is a valid ideal (the ideal containing 1).

### 82.8.1 Construction of New Ideals

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `I + J` | Given ideals I and J of the same algebra F, return their sum (ideal generated by the union of generators of I and J). | — |
| `I * J` | Given ideals I and J of the same algebra A, return their product (ideal generated by products of generators of I with those of J). | — |
| `F / J` | Given algebra F over a field and ideal J of F, return the fp-algebra F/J. | — |
| `Generic(I)` | Given an ideal I of a generic algebra A, return A. | — |

### 82.8.2 Ideal Predicates

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `I eq J` | Return whether ideals I and J of the same algebra F are equal. | Gröbner basis comparison. |
| `I ne J` | Return whether ideals I and J are not equal. | Gröbner basis comparison. |
| `I notsubset J` | Return whether I is not contained in J. | Normal form reduction. |
| `I subset J` | Return whether I is contained in J. | Normal form reduction. |
| `IsZero(I)` | Return whether I is the zero ideal (contains zero alone). | — |

### 82.8.3 Operations on Elements of Ideals

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `f in I` | Given polynomial f from algebra F and ideal I of F, return whether f is in I. | Normal form: f ∈ I iff NormalForm(f, I) = 0. |
| `NormalForm(f, I)` | Return the unique normal form of f with respect to the Gröbner basis of I. The result is zero iff f ∈ I. | Noncommutative reduction modulo the GB of I. |
| `NormalForm(f, S)` | Return a normal form of f with respect to a set or sequence S of polynomials. Not unique in general; zero result implies f ∈ ⟨S⟩ but the converse fails unless S is a GB. | Noncommutative reduction. |
| `f notin I` | Return whether f is not in I. | Normal form. |

*Worked example: H82E5 (membership and normal form for elements of ideal ⟨(x+y)³, (y−z)², y²z+z⟩ in Q⟨x,y,z⟩).*

---

## 82.9 Changing Coefficient Ring

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ChangeRing(I, S)` | Given an ideal I of algebra F = R⟨x₁,…,xₙ⟩ with coefficient ring R, and a ring S, construct the ideal J of Q = S⟨x₁,…,xₙ⟩ by coercing coefficients into S. Requires automatic coercion R → S. If R and S are fields with R ⊆ S and the basis of I is a GB, the basis of J is automatically marked as a GB. | — |

---

## 82.10 Finitely Presented Algebras

*(Prose section.)* An fp-algebra in Magma is the quotient ring F/J of a free algebra
F = R⟨x₁,…,xₙ⟩ by an ideal J. It is an object of type `AlgFP` with elements of type
`AlgFPElt`. Elements are noncommutative polynomials always kept reduced to normal form
modulo J. When the fp-algebra has finite dimension as a vector space over its coefficient
field, additional special operations are available (§82.13).

---

## 82.11 Creation of FP-Algebras

**Note:** When an fp-algebra is created, the ideal of relations is left unchanged. The
Gröbner basis of the underlying ideal is computed automatically as soon as any non-trivial
operation (e.g. printing or multiplying elements) is performed.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `quo< F \| J >` / `quo< F \| a1, ..., ar >` | Given a free algebra F and two-sided ideal J (or generators a₁,…,aᵣ ∈ F): return the fp-algebra (quotient algebra) F/J. Angle-bracket notation assigns names to indeterminates. | GB computed automatically on first use (noncommutative F4 **[Fau99]** or Buchberger **[CLO96]**). |
| `F / J` | Given free algebra F and ideal J of F, return the fp-algebra F/J. | As above. |
| `FPAlgebra< K, X \| L >` | Given a field K, a list X of n identifiers, and a list L of noncommutative polynomials (relations) in the n variables X: create the fp-algebra of rank n with base ring K and quotient relations ⟨L⟩, i.e. K⟨X⟩/⟨L⟩. Angle-bracket notation assigns names to indeterminates. | As above. |

*Worked example: H82E6 (equivalent constructions of A = Q⟨x,y⟩/(x²y−yx, xy³−yx) using FPAlgebra and quo; showing that operating on elements triggers automatic GB computation expanding the relations list from 2 to 11 elements).*

---

## 82.12 Operations on FP-Algebras

Most operations parallel those for free algebras; computation is mapped to the preimage ideal and results are mapped back.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `A . i` | Return the i-th indeterminate of fp-algebra A as an element of A. | — |
| `CoefficientRing(A)` | Return the coefficient ring of the fp-algebra A. | — |
| `Rank(A)` | Return the rank of fp-algebra A (number of indeterminates). | — |
| `DivisorIdeal(I)` | Given ideal I of fp-algebra A = F/J, return the defining ideal J of A in F. | — |
| `PreimageIdeal(I)` | Given ideal I of fp-algebra A = F/J, return the ideal I′ of F whose image under the natural epimorphism F → A is I. | — |
| `PreimageRing(A)` | Given fp-algebra A = F/J, return the free algebra F. | — |
| `OriginalRing(A)` | Return the generic free algebra F such that A = F/J. | — |
| `IsCommutative(A)` | Return whether the algebra A is commutative. | — |
| `I eq J` | Return true iff ideals I and J of the same fp-algebra A are equal. | — |
| `I subset J` | Return true iff ideal I is contained in ideal J of the same fp-algebra A. | — |
| `I + J` | Return the sum of ideals I and J of the same fp-algebra A. | — |
| `I * J` | Return the product of ideals I and J of the same fp-algebra A. | — |
| `IsProper(I)` | Return whether ideal I of fp-algebra A is proper (strictly contained in A). | — |
| `IsZero(I)` | Return whether ideal I of fp-algebra A is the zero ideal. Equivalent to testing whether the preimage ideal of I equals the divisor ideal of A. | — |

---

## 82.13 Finite Dimensional FP-Algebras

When an fp-algebra A has finite dimension as a vector space over its coefficient field,
the following additional operations are available.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Dimension(A)` | Return the dimension of finite dimensional fp-algebra A. | Determined from the Gröbner basis of the defining ideal. |
| `VectorSpace(A)` | Construct the vector space V isomorphic to A; return V together with the isomorphism f : A → V. | Basis of normal forms modulo the GB. |
| `MatrixAlgebra(A)` | Construct the matrix algebra M isomorphic to A; return M together with the isomorphism f : A → M. | Left-multiplication matrices on normal-form basis. |
| `Algebra(A)` | Construct the associative structure-constant algebra S isomorphic to A; return S and the isomorphism f : A → S. | Structure constants from normal-form multiplication table. |
| `RepresentationMatrix(f)` | Given element f of a finite dimensional fp-algebra A, return the d × d representation matrix of f over the coefficient field (d = dim A). | Left-multiplication by f on the normal-form basis. |
| `IsUnit(f)` | Given element f of a finite dimensional fp-algebra A over a field, return whether f is a unit. | Invertibility of the representation matrix. |
| `IsNilpotent(f)` | Given element f of a finite dimensional fp-algebra A over a field, return whether f is nilpotent; if so, also return the smallest q with fq = 0. | Nilpotency of the representation matrix. |
| `MinimalPolynomial(f)` | Given element f of a finite dimensional fp-algebra A over a field, return the minimal polynomial of f as a univariate polynomial over the coefficient field. | Minimal polynomial of the representation matrix. |

*Worked example: H82E7 (fp-algebra A = Q⟨x,y,z⟩/(x²−yzy+z, y²−yxy+1, z²−yxy−xzx) of dimension 18; MinimalPolynomial of x (degree 8) and y (degree 16); VectorSpace isomorphism showing normal-form basis of 18 elements; MatrixAlgebra of degree 18; RepresentationMatrix; Algebra conversion and Centre (dimension 15), JacobsonRadical (dimension 4), IsNilpotent).*

---

## 82.14 Vector Enumeration

Vector enumeration (originally called module enumeration) is an algorithm for converting a
finitely presented module for a finitely presented algebra into a concrete vector space on
which the algebra has explicit matrix action. The algebra may be the group algebra of an
fp-group (yielding a matrix representation of the group) or a more general fp-algebra such
as a Hecke algebra or a quotient of a polynomial ring.

### 82.14.1 Finitely Presented Modules

For a ring R, let M be an R-module generated by s elements {m₁,…,mₛ}. There is an
R-module epimorphism ψ : Rˢ → M given by (r₁,…,r₢) ↦ m₁r₁ + ⋯ + mₛrₛ, so
M ≅ Aˢ/ker ψ. If ker ψ is generated as an R-module by a finite set L, M is said to be
presented by s generators and relators L.

*(Prose section — no intrinsics.)*

### 82.14.2 S-Algebras

If S is another ring equipped with a central ring homomorphism φ : S → R (so R is an
S-algebra), any R-module is an S-module on which R acts as a ring of S-module
endomorphisms. When S is a field k, any finite-dimensional R-module V of k-dimension n
is characterised by its dimension, and R acts on it as a subring of M_n(k).

*(Prose section — no intrinsics.)*

### 82.14.3 Finitely Presented Algebras

Given a finite set X and ring S, the free S-algebra A generated by X and a finite set R ⊂ A
of relators defines the fp-algebra P = A/⟨ARA⟩ = ⟨X | R⟩. The monoid algebra of any
fp-monoid (or group) is finitely presented, and any quotient of an fp-algebra by a
finitely-generated two-sided ideal is finitely presented.

*(Prose section — no intrinsics.)*

### 82.14.4 Vector Enumeration

The vector enumeration algorithm reconciles the two descriptions of an R-module (where R
is a finitely presented k-algebra for a field k, and M is a finitely presented R-module of
finite k-dimension), computing the k-dimension of M and the matrices giving the action of
generators of R on M. If M has infinite k-dimension the algorithm fails to terminate.

*(Prose section — no intrinsics; see Examples H82E8 for abstract illustrations.)*

### 82.14.5 The Isomorphism

The algorithm computes the k-vector space isomorphic to M as an R-module, giving images
in the vector space for the s standard generators of Rˢ, and pre-images in Rˢ of the basis
vectors of the vector space.

*(Prose section — no intrinsics.)*

### 82.14.6 Sketch of the Algorithm

The algorithm is based on the **Haselgrove–Leech–Trotter (HLT) variant of the Todd–Coxeter
algorithm**, adapted from permutation coset enumeration to vector-space enumeration. It
maintains a table (columns = algebra generators, rows = current basis vectors) of the partial
action of the free algebra on the vector space, extended with a "defining" mode that adds
new rows when needed. The algorithm exploits:

1. The action must be complete.
2. Relators of P annihilate every vector.
3. Images in the space of relators of M are zero (giving linear dependencies — coincidences).
4. The space must contain images of the R-module generators of M.

Coincidences reduce the basis; the algorithm terminates if and only if M is finite-dimensional.
Completeness is enforced by adding relators x − x for each generator x ∈ X.

*(Prose section — no intrinsics.)*

### 82.14.7 Weights

Each relator r is assigned a user-specified weight wᵣ; each basis vector e has a weight wₑ.
A current weight w increases as the computation progresses. At weight w, all (basis vector,
relator) pairs (b, r) with w_b + w_r ≤ w are processed. New basis vectors defined during
processing inherit weight w; initial basis vectors and those from submodule generator
processing get weight 1.

*(Prose section — no intrinsics.)*

### 82.14.8 Setup Functions

For V2.11, these functions create the representation of fp-algebras required by the Vector
Enumeration algorithm, for compatibility with older versions.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `FreeAlgebra(R, M)` / `FreeAlgebra(R, G)` | Construct the special fp-algebra over ring R and monoid M or group G, for use as input to the Vector Enumeration algorithm. | — |

### 82.14.9 The Quotient Module Function

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `QuotientModule(A, S)` | Given an fp-k-algebra A (for a field k) with r generators, and a submodule N of the free A-module of rank s specified by S: construct an A-module isomorphic to the quotient Aˢ/N together with the isomorphism. Returns three values: (1) M — a sequence of r n×n matrices over k giving the action of generators of A; (2) I — a sequence of s vectors of length n (images of the s standard generators of Aˢ); (3) P — a sequence of n elements of Aˢ (pre-images of the n basis vectors of kⁿ). S may be: a finitely generated right ideal of A (s=1, N treated as submodule of A); or a sequence of elements of Aˢ (N = submodule generated by them). Supports extensive optional parameters for weights, limits, logging, and control (see §82.14.11–82.14.15). | HLT variant of the Todd–Coxeter algorithm, adapted to vector spaces over k. Uses a more efficient technique for the relators of an fp-group underlying A (exploiting invertibility of group generators). |

### 82.14.10 Structuring Presentations

Relations come from three sources: (1) relations of the fp-group or fp-monoid underlying A;
(2) relations of A itself; (3) generators of N (treated as subgroup generators, as in
Todd–Coxeter). When the underlying monoid is an fp-group, the algorithm uses a more
efficient technique for (1) exploiting invertibility of generators. Users should ensure:
(a) fp-group monoids are presented as such; (b) monomial-equality relations appear as
monoid relations rather than algebra relations.

*(Prose section — no additional intrinsics.)*

### 82.14.11 Options and Controls

`QuotientModule` supports a large selection of optional arguments, detailed in the following
subsections.

### 82.14.12 Weights

`QuotientModule` weight parameters (defaults: all relations get weight 3 except fp-group
relators, which get weight = half the length of the relator; lookahead uses the same
defaults unless overridden):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `MonomialWeights` / `MonWts` | `[RngIntElt]` | — | Weights for relations from the underlying monoid of A. Shorter sequences use defaults for the remainder; extra entries are silently discarded. Also used in lookahead mode unless `MonomialLookaheadWeights` is given. |
| `MonomialLookaheadWeights` / `MonLWts` | `[RngIntElt]` | — | Weights for monoid relations in lookahead mode only. |
| `AlgebraWeights` / `AlgWts` | `[RngIntElt]` | — | Weights for the explicit algebra relations of A. |
| `AlgebraLookaheadWeights` / `AlgLWts` | `[RngIntElt]` | — | Weights for algebra relations in lookahead mode only. |

### 82.14.13 Limits

If a limit is exceeded, `undef` is returned unless `ErrorOnFail` is set.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `MaximumDimension` / `MaxDim` | `RngIntElt` | ∞ | Limit on the dimension of the vector space constructed (and intermediate spaces). |
| `MaximumTime` / `MaxTime` | `FldReElt` | ∞ | CPU time limit in seconds. Only checked at certain points, so overruns are possible. |
| `MaximumWeight` / `MaxWt` | `RngIntElt` | 100 | Limit on the maximum weight of (basis vector, relation) pairs processed. The weight of a pair = weight of the basis vector + weight of the relation. |

### 82.14.14 Logging

When multiple contradictory logging options are given, the first takes precedence.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `NoLogging` / `NoLog` / `Silent` | `BoolElt` | false | Suppress all informational messages from the vector enumerator. |
| `MaximumLogging` / `MaxLog` | `BoolElt` | false | Highest possible logging detail (for debugging only). |
| `LogActions` / `LogAct` | `RngIntElt` | 0 | Level of messages about computation of the algebra action on the vector space. Level 0: none; levels ≥ 1: copious output (levels > 2 are equivalent). |
| `LogCoincidences` / `LogCoin` | `RngIntElt` | 0 | Level for messages about coincidences. Level 0: none; level 1: every coincidence and deduction; level ≥ 2: also logs the operation of finding the undeleted image of a vector. |
| `LogInitialization` / `LogInitialisation` / `LogInit` | `RngIntElt` | 0 | Level for messages about initialization of new basis vectors. Level 0: none; levels ≥ 1: message for each new basis vector defined. |
| `LogPacking` / `LogPack` | `RngIntElt` | 1 | Level for messages about reclamation of free space. Level 0: none; level 1 (default): message each time the pack routine is called; level ≥ 2: records the exact renaming. |
| `LogPushes` / `LogPush` | `RngIntElt` | 0 | Level for messages about pushing (tracing) of (basis vector, relation) pairs. Level 0: none; level 1: message for each push started; level ≥ 2: also records the outcome. |
| `LogProgress` / `LogStages` | `RngIntElt` | 0 | Level for messages about overall algorithm progress. Level 0: none; level 1: major stages; level 2: relations printed as read in, final action printed; level ≥ 3: action printed after submodule generator processing. |
| `LogWeightChanges` / `LogWt` | `RngIntElt` | 1 | Level for messages about changes in current weight. Level 0: none; level ≥ 1 (default): prints new weight and current dimension at each weight change. |

### 82.14.15 Miscellaneous

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Lookahead` | `BoolElt` or `RngIntElt` | true | Controls lookahead. false: no lookahead; true: default lookahead (two weights); positive integer n: lookahead by n weights. Sufficiently large n is equivalent to complete lookahead. Lookahead is commenced approximately every time the dimension doubles. |
| `EarlyClosing` / `Early` | `BoolElt` | false | If true, allow the algorithm to stop as soon as the table represents a complete action, without checking all relations. Usually correct in practice; default is to continue and verify all relations. |
| `EarlyClosingMinimum` / `ECMin` | `RngIntElt` | — | Minimum dimension at which the algorithm may stop without checking all relators. Implies `EarlyClosing`. |
| `EarlyClosingMaximum` / `ECMax` | `RngIntElt` | — | Maximum dimension at which the algorithm may stop without checking all relators. Implies `EarlyClosing`. |
| `ConstructMorphism` / `Morphism` | `BoolElt` | true | Controls whether the third return value P is computed. When false, P is not computed (saving time and space). |
| `ErrorOnFail` / `ErrFail` | `BoolElt` | — | If present, a run-time error is generated when insufficient time or space prevents completion; otherwise `undef` is returned. |

*Worked example: H82E9 (permutation action of D₈ = ⟨a,b | a⁴=b²=(ab)²=1⟩ over Q: FreeAlgebra(q,d8), rideal⟨b−1⟩, QuotientModule giving 4×4 matrices, image (1,0,0,0), pre-images 1, a⁻¹, a⁻¹b, a⁻²).*

*Worked example: H82E10 (quotient of the D₈ permutation module by the all-ones vector: rideal⟨b−1, 1+a³+a³b+a²⟩, QuotientModule giving 3-dimensional quotient with 3×3 matrices).*

---

## 82.15 Bibliography

| Key | Reference |
|-----|-----------|
| **[CLO96]** | David Cox, John Little, and Donal O'Shea. *Ideals, Varieties and Algorithms.* Undergraduate Texts in Mathematics. Springer, New York–Berlin–Heidelberg, 2nd edition, 1996. |
| **[Fau99]** | Jean-Charles Faugère. A new efficient algorithm for computing Gröbner bases (F4). *Journal of Pure and Applied Algebra*, 139(1–3):61–88, 1999. |
| **[Li02]** | Huishi Li. *Noncommutative Gröbner Bases and Filtered-Graded Transfer.* Volume 1795 of Lecture Notes in Math. Springer-Verlag, Berlin–Heidelberg–New York, 2002. |
| **[Mor94]** | Teo Mora. An introduction to commutative and noncommutative Gröbner bases. *Theoretical Computer Science*, 134:134–173, 1994. |

---

## Algorithm-to-function quick reference

| Algorithm / Theory | Functions |
|--------------------|-----------|
| Noncommutative Buchberger algorithm **[CLO96, Chap. 2, §7]** | `Groebner(:Faugere:=false)`, `GroebnerBasis(:Faugere:=false)` |
| Noncommutative Faugère F4 (Allan Steel) **[Fau99]** | `Groebner(:Faugere:=true)` (default), `GroebnerBasis` (default) |
| Noncommutative GB theory / overview **[Mor94, Li02]** | All GB and FP-algebra operations |
| Normal form reduction modulo GB | `NormalForm`, `f in I`, `f notin I`, `Reduce` |
| Gröbner basis (truncated, degree-d) | `GroebnerBasis(S, d)` |
| Left-/right-multiplication matrices (finite-dimensional FP-algebras) | `MatrixAlgebra`, `RepresentationMatrix`, `VectorSpace`, `Algebra` |
| Minimal polynomial (via representation matrix) | `MinimalPolynomial`, `IsUnit`, `IsNilpotent` |
| HLT Todd–Coxeter variant (vector enumeration) | `QuotientModule` |
| Vector enumeration — weight control | `QuotientModule(:MonomialWeights, AlgebraWeights, …)` |
| Vector enumeration — limits | `QuotientModule(:MaximumDimension, MaximumTime, MaximumWeight)` |
| Vector enumeration — logging | `QuotientModule(:LogActions, LogCoincidences, LogProgress, …)` |
| Skew-commutativity collection (exterior algebras) | `ExteriorAlgebra`, all ideal/GB operations on exterior algebras |
