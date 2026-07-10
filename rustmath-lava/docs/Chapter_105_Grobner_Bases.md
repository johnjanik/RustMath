# Chapter 105 — Gröbner Bases

**Handbook part:** XIV — Commutative Algebra
**Handbook pages:** 3181–3222 (PDF pages 3312–3355)

---

## Scope and overview

Chapter 105 describes the configuration and use of Magma's Gröbner basis machinery, which
underlies computations with ideals and modules over multivariate polynomial rings. Gröbner
bases were introduced by Bruno Buchberger **[Buc65]**; the Buchberger algorithm computes a
Gröbner basis starting from an arbitrary generating set. The books *Ideals, Varieties and
Algorithms* **[CLO96]** and *Gröbner Bases* **[BW93]** have strongly influenced Magma's design.

Since V2.11 (May 2004), Magma also contains a highly optimised implementation of the
Faugère F4 algorithm **[Fau99]**, based on sparse linear algebra, which typically performs
dramatically better than the Buchberger algorithm. For ideals over Euclidean rings (Z, Z/mZ,
K[x], Galois rings, p-adic quotient rings, valuation rings), Magma uses an extension of F4
due to Allan Steel (unpublished) that computes the unique minimal reduced strong Gröbner
basis, incorporating the advanced pair-elimination criteria of Möller **[Möl88]**.

Order-conversion algorithms are also available: the **FGLM algorithm** **[FGLM93]** (for
zero-dimensional ideals) and the **Gröbner Walk algorithm** **[CKM97]** (general case). A
**Hilbert-driven Buchberger algorithm** **[Tra96]** is provided for homogeneous ideals whose
Hilbert series is known. For systems over F₂, a special boolean polynomial ring type with a
compact bit-vector monomial representation is available, and an interface to the MiniSat SAT
solver is included.

Chapter 24 covers the basics of multivariate polynomial rings and their elements. Related
chapters: invariant rings of finite groups (Chapter 110), affine algebras (Chapter 108),
modules over affine algebras (Chapter 109), and algebraically closed fields (Chapter 40).

---

## 105.1 Introduction

No intrinsics are defined in this purely introductory section. See the scope and overview
above for the algorithmic summary drawn from this section.

---

## 105.2 Representation and Monomial Orders

A monomial order on the set M of monomials of P = R[x₁,…,xₙ] is a total well-ordering <
on M satisfying 1 ≤ s for all s ∈ M, and s ≤ t ⟹ su ≤ tu. Any such order can be
specified by n linearly-independent weight vectors from Qⁿ. All orders may be passed as
argument sequences (or tuples) to `PolynomialRing`, `ChangeOrder`, and `VariableExtension`.

### 105.2.1 Lexicographical: lex

`s < t` iff there exists 1 ≤ i ≤ n with equal first i−1 exponents and the i-th exponent of
s less than that of t. Specified by `("lex")`. Yields the most information about an ideal
but is the hardest order to compute a Gröbner basis for.

### 105.2.2 Graded Lexicographical: glex

`s < t` iff deg(s) < deg(t), or deg(s) = deg(t) and s < t in lex order. Specified by
`("glex")`. Rarely preferred over grevlex.

### 105.2.3 Graded Reverse Lexicographical: grevlex

`s < t` iff deg(s) < deg(t), or deg(s) = deg(t) and s > t in lex order applied in reverse.
Specified by `("grevlex")`. Usually the easiest order for computing any Gröbner basis;
recommended whenever any Gröbner basis suffices.

### 105.2.4 Graded Reverse Lexicographical (Weighted): grevlexw

As grevlex but using weighted degree w.r.t. a sequence W of n positive integer weights.
Specified by `("grevlexw", W)`. Reduces to grevlex when W = [1,…,1]. Useful if the ideal
is homogeneous w.r.t. the grading given by W.

### 105.2.5 Elimination (k): elim

Block-grevlex order: first k variables eliminated (block-grevlex on first k, then
block-grevlex on remaining n−k). Specified by `("elim", k)`. G ∩ K[x_{k+1},…,xₙ] is a
Gröbner basis of the k-th elimination ideal.

### 105.2.6 Elimination List: elim

Generalisation: given disjoint sequences U, V partitioning {1,…,n}, compare by grevlex on
U first, then V. Specified by `("elim", U, V)` or `("elim", U)` (V computed automatically).

### 105.2.7 Inverse Block: invblock

Same as elimination list order but with the roles of U and V swapped: grevlex on V first,
then U. Specified by `("invblock", U, V)` or `("invblock", U)`. See **[BW93, p. 390]**.

### 105.2.8 Univariate: univ

`s < t` iff s is greater than t after eliminating all variables but the i-th. Specified by
`("univ", i)`. A Gröbner basis of a zero-dimensional ideal with this order contains the
unique monic generator of the elimination ideal in the i-th variable alone.

### 105.2.9 Weight: weight

Given n linearly-independent weight vectors W₁,…,Wₙ ∈ Qⁿ: `s < t` iff there exists
1 ≤ i ≤ n with s·Wⱼ = t·Wⱼ for j < i and s·Wᵢ < t·Wᵢ. Specified by
`("weight", Q)` where Q is a sequence of n² rationals in row-major order. Subsumes all
other orders; the specialised orders are much faster in practice. See also **[CLO98, p. 153]**.

---

## 105.3 Polynomial Rings and Ideals

### 105.3.1 Creation of Polynomial Rings and Accessing their Monomial Orders

If no order is specified, lexicographical order is used by default. Note that the lex
Gröbner basis is often the hardest to compute; grevlex is usually preferable when any
Gröbner basis will do.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PolynomialRing(R, n)` / `PolynomialAlgebra(R, n)` | Create a multivariate polynomial ring in n > 0 variables over R with the default lexicographical order. Parameter `Global` (default `false`): if `true`, returns the unique global polynomial ring over R with n variables. | — |
| `PolynomialRing(R, n, order)` / `PolynomialAlgebra(R, n, order)` | Create a multivariate polynomial ring in n > 0 variables over R with the specified monomial order (given as a string and optional extra arguments; see §105.2). | — |
| `PolynomialRing(R, n, T)` / `PolynomialAlgebra(R, n, T)` | Create a multivariate polynomial ring in n > 0 variables over R with the monomial order given by tuple T (whose components match §105.2 arguments, or a tuple returned by `MonomialOrder`). | — |
| `MonomialOrder(P)` | Given a polynomial ring P (or an ideal thereof), return a tuple describing the monomial order, suitable for passing as the third argument to `PolynomialRing`. | — |
| `MonomialOrderWeightVectors(P)` | Given a polynomial ring P of rank n (or an ideal thereof), return the weight vectors of the underlying monomial order as a sequence of n sequences of n rationals. See **[CLO98, p. 153]**. | — |

*Worked examples: H105E1 (constructing polynomial rings with lex, grevlex, block-elim, and weight orders; examining weight vectors).*

### 105.3.2 Creation of Graded Polynomial Rings

A polynomial ring is graded if weights d₁,…,dₙ are assigned to the variables; the weighted
degree of xₑ₁₁·…·xₑₙₙ is Σ eᵢdᵢ. As of V2.15, `PolynomialRing(R, Q)` defaults to the
grevlexw order with weights Q, since Gröbner bases of homogeneous ideals tend to be smaller
under this order.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PolynomialRing(R, Q)` / `PolynomialAlgebra(R, Q)` | Given a ring R and a non-empty sequence Q of positive integers, create a graded polynomial ring in n = #Q variables over R with the i-th variable having weighted degree Q[i]. Default monomial order is grevlexw with weights Q (since V2.15). | — |
| `Grading(P)` / `VariableWeights(P)` | Given a graded polynomial ring P (or an ideal thereof), return the variable weights as a sequence of n integers. Returns n copies of 1 if P has trivial grading. | — |

### 105.3.3 Element Operations Using the Grading

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Degree(f)` / `WeightedDegree(f)` | Given a polynomial f of a graded polynomial ring P, return the weighted degree of f (the maximum weighted degree over all monomials of f). Different from the natural total degree `TotalDegree(f)`. | — |
| `LeadingWeightedDegree(f)` | Given a polynomial f of a graded polynomial ring P, return the weighted degree of the leading monomial of f. | — |
| `IsHomogeneous(f)` | Given a polynomial f (or an ideal I) of a graded polynomial ring P, return whether f (resp. every generator of I) is homogeneous w.r.t. the variable weights (i.e. all monomials have equal weighted degree). | — |
| `HomogeneousComponent(f, d)` | Given a polynomial f of a graded polynomial ring P and a non-negative integer d, return the weighted degree-d homogeneous component of f (sum of all terms of weighted degree d). Returns 0 if no such terms exist. | — |
| `HomogeneousComponents(f)` | Given a polynomial f of a graded polynomial ring P, return the sequence of all weighted homogeneous components of f. | — |
| `MonomialsOfDegree(P, d)` | Given a polynomial ring P and a non-negative integer d, return an indexed set of all monomials of P with total degree d (grading is ignored even if P is graded). | — |
| `MonomialsOfWeightedDegree(P, d)` | Given a graded polynomial ring P and a non-negative integer d, return an indexed set of all monomials of P with weighted degree d. Equivalent to `MonomialsOfDegree` if P has trivial grading. | — |

*Worked examples: H105E2 (graded ring with weights [1,2,4]; `Degree`, `TotalDegree`, `IsHomogeneous`, `MonomialsOfDegree`, `MonomialsOfWeightedDegree` on various elements).*

### 105.3.4 Creation of Ideals and Accessing their Bases

Ideals are created from a polynomial ring and a generating set. An ideal with a *fixed basis*
(created via `IdealWithFixedBasis`) stores extra information so that polynomials can be
expressed in terms of the original generators, but this makes the Gröbner basis computation
substantially more expensive.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ideal< P \| L >` | Given a polynomial ring P, return the ideal of P generated by the elements of the list L. Each term of L may be an element of P, a set/sequence of elements of P, an ideal of P, or a set/sequence of ideals of P. | — |
| `Ideal(B)` | Given a set or sequence B of polynomials from a polynomial ring P, return the ideal of P generated by the elements of B. Equivalent to the ideal constructor, but convenient when a set/sequence is already available. | — |
| `Ideal(f)` | Given a polynomial f from a polynomial ring P, return the principal ideal of P generated by f. | — |
| `IdealWithFixedBasis(B)` | Given a sequence B of polynomials from a polynomial ring P, return the ideal of P generated by B with B stored as a fixed basis. When `Coordinates` is called, results are expressed w.r.t. B. WARNING: Gröbner basis computation is very expensive; avoid unless coordinates w.r.t. B are needed. | — |
| `Basis(I)` | Given an ideal I, return the current basis of I (the fixed basis if one exists, otherwise the current generating basis whether or not it is yet a Gröbner basis). | — |
| `BasisElement(I, i)` | Given an ideal I and integer i, return the i-th element of the current basis of I. Equivalent to `Basis(I)[i]`. | — |

---

## 105.4 Gröbner Bases

For ideals over fields, the unique sorted minimal reduced Gröbner basis is computed
automatically when needed **[CLO96, Chap. 2, §7, Prop. 7]**. For ideals over Euclidean
rings, Magma computes a unique minimal reduced strong Gröbner basis (a D-Gröbner basis
in the sense of **[BW93, Def. 10.4]**) via Steel's extension of the F4 algorithm **[Fau99]**,
using unique echelon forms of sparse matrices over Euclidean rings.

### 105.4.1 Gröbner Bases over Fields

Over fields a basis is *minimal* if each polynomial is monic and not in the ideal generated
by the others **[CLO96, Chap. 2, §7, Def. 4]**. A basis is *reduced* if each polynomial is
monic and no monomial of any basis element is divisible by the leading monomial of another
**[CLO96, Chap. 2, §7, Def. 5]**. Every ideal over a field has a unique sorted minimal
reduced Gröbner basis for a fixed monomial order.

### 105.4.2 Gröbner Bases over Euclidean Rings

Magma uses the notion of a *strong* Gröbner basis: G ⊆ I is a Gröbner basis if for every
f ∈ I there exists g ∈ G such that the leading term of g (coefficient times monomial)
divides the leading term of f. Weak Gröbner bases are not used. Magma computes a unique
reduced strong Gröbner basis, even for Euclidean rings with zero divisors. The advanced
pair-elimination criteria of Möller **[Möl88]** are also extended to Euclidean rings. Current
supported Euclidean rings: Z, Z/mZ, K[x] over any field K, Galois rings, p-adic quotient
rings, and valuation rings.

### 105.4.3 Construction of Gröbner Bases

Two direct algorithms are available over fields (since V2.11): the Faugère F4 algorithm
**[Fau99]** (for finite fields or Q, using sparse linear algebra) and the Buchberger algorithm
**[CLO96, Chap. 2, §7]** (for any field). Both use Möller's advanced pair-elimination
criteria **[Möl88]**. Two order-conversion algorithms are also available: the FGLM algorithm
**[FGLM93]** (zero-dimensional ideals only) and the Gröbner Walk **[CKM97]** (general).
Since V2.12, if the input is not homogeneous, Magma attempts to find a homogeneous weight
vector W and uses the grevlexw order internally. Since V2.18, the F4 algorithm over Fp
(prime p < 2^23.5) supports multi-threading via `Nthreads`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Groebner(I: parameters)` | (Procedure.) Explicitly force a Gröbner basis for ideal I to be constructed. Normally not necessary (computed automatically when needed), but allows control via parameters. Parameters: `Al` (`"Default"`, `"Direct"`, `"FGLM"`, `"Walk"`; default `"Default"` — FGLM for zero-dimensional ideals over finite field or Q, Walk otherwise); `Faugere` (BoolElt, default `true` — use F4 if field is finite or Q, else Buchberger); `HomogeneousWeights` (BoolElt, default `true` — search for homogeneous weight vector); `Homogenize` (BoolElt, default `true` over Q, `false` elsewhere — homogenize before computing); `DegreeStart` (RngIntElt — ignore S-polynomial pairs of degree < this value); `AllPairs` (BoolElt, default `false` — F4: include all pairs in queue at each step, not just current degree); `PairsLimit` (RngIntElt, default 0 — F4: max pairs per step to limit matrix size); `ReversePairs` (BoolElt, default `false` — F4: reverse pairs list when `PairsLimit` is active); `HFE` (BoolElt, default `false` — F4: apply HFE optimisations for F₂ systems with secret degree ≤ 127); `Boolean` (BoolElt, default `false`); `Nthreads` (RngIntElt, default 1 — F4 multi-threading over Fp); `ReduceInitial` (BoolElt, default `true` — Buchberger: reduce basis before S-pairs); `RemoveRedundant` (BoolElt, default `true` — Buchberger: remove redundant input polynomials first); `ReduceByNew` (BoolElt, default `true` — Buchberger: reduce current basis by each newly inserted polynomial); `SigmaEpsilon` (FldRatElt, default 1/2 — Walk: perturbation factor for initial weight vector σ); `TauEpsilon` (FldRatElt, default 1/n — Walk: perturbation factor for final weight vector τ); `SigmaVectors` (RngIntElt, default n — Walk: number of weight vectors of initial order used for σ perturbation); `TauVectors` (RngIntElt, default ⌈n/2⌉ — Walk: number of weight vectors of final order used for τ perturbation). Over Euclidean rings, only `Homogenize` applies. | F4 **[Fau99]** / Buchberger **[CLO96]**, both with pair elimination **[Möl88]**; order conversion via FGLM **[FGLM93]** or Gröbner Walk **[CKM97]**. Over Euclidean rings: Steel's extension of F4 **[Fau99]** (unpublished). |
| `GroebnerBasis(I: parameters)` | Given an ideal I, force the Gröbner basis to be computed and return it. Parameters are the same as for `Groebner`. | F4 **[Fau99]** / Buchberger **[CLO96]** / FGLM **[FGLM93]** / Gröbner Walk **[CKM97]**. |
| `GroebnerBasis(S: parameters)` | Given a set or sequence S of polynomials, return the unique Gröbner basis of the ideal generated by S as a sorted sequence. Useful when construction of an ideal object is not desired. Parameters are the same as for `Groebner`. | F4 **[Fau99]** / Buchberger **[CLO96]** / FGLM **[FGLM93]** / Gröbner Walk **[CKM97]**. |
| `GroebnerBasisUnreduced(S: parameters)` | Given a set or sequence S of polynomials, return an *unreduced* Gröbner basis of the ideal generated by S as a sorted sequence. Parameters: `Homogenize` (BoolElt, default `true`), `ReduceInitial` (BoolElt, default `true`), `ReduceByNew` (BoolElt, default `true`). Useful when reduction is very expensive. | Buchberger **[CLO96]** with controlled reduction. |
| `GroebnerBasis(S, d: parameters)` | Given a set or sequence S of polynomials, return the degree-d Gröbner basis: the truncated basis obtained by ignoring S-polynomial pairs of total degree > d. For homogeneous ideals, the result equals all GB elements of degree ≤ d, and degree-≤-d membership is certified by zero normal form. For non-homogeneous ideals, results may differ. See also **[BW93, §10.2]**. Parameters are the same as for `Groebner`. | Truncated F4 **[Fau99]** / Buchberger **[CLO96]**. |

*Worked examples: H105E3 (Cyclic-6 ideal over Q with lex order; Gröbner basis of 17 polynomials, factorisation of the univariate element); H105E4 (Runge-Kutta 2 system over rational function field; lex GB yields a unique solution).*

### 105.4.4 Related Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HasGroebnerBasis(I)` | Given an ideal I, return whether the Gröbner basis of I can be computed (requires base ring to be a field or Euclidean ring). | — |
| `EasyIdeal(I)` | Given an ideal I, return the ideal E equal to I but with Gröbner basis w.r.t. an "easy" order (usually grevlex or grevlexw with suitable weights), together with an isomorphism from I to E. The easy basis is used extensively internally by Magma. | — |
| `EasyBasis(I)` | Given an ideal I, return the Gröbner basis of the easy ideal of I. | — |
| `SmallBasis(I)` | Given an ideal I, return the shortest-length basis of I currently known (either the original basis or a Gröbner basis, always in the same monomial order as I). | — |
| `MarkGroebner(I)` | (Procedure.) Mark the current basis of I as its Gröbner basis w.r.t. the monomial order of I. The basis must exactly equal the unique sorted minimal reduced Gröbner basis. Useful when a Gröbner basis is known from a previous computation. Unpredictable results if the basis is not the unique Gröbner basis. | — |
| `IsGroebner(S)` | Given a set or sequence S of polynomials, return whether S is a (not necessarily minimal or reduced) Gröbner basis of the ideal it generates. | — |
| `Coordinates(I, f)` | Given an ideal I with basis b₁,…,bₖ and a polynomial f ∈ I, return a sequence [g₁,…,gₖ] with f = Σ gᵢ·bᵢ. If I was created by `IdealWithFixedBasis`, the fixed basis is used; otherwise the Gröbner basis is used. Result is not necessarily unique. | — |
| `CoordinateMatrix(I)` | Given an ideal I with a fixed basis (created via `IdealWithFixedBasis`), return the coordinate matrix C: the i-th row gives the coordinates of the i-th Gröbner basis element w.r.t. the fixed basis. Computes the Gröbner basis first if not already done. | — |
| `NormalForm(f, I)` | Given a polynomial f from polynomial ring P and an ideal I of P, return the unique normal form of f w.r.t. the Gröbner basis of I. The normal form is 0 iff f ∈ I. | Division algorithm using GB. |
| `NormalForm(f, S)` | Given a polynomial f and a set or sequence S of polynomials, return a normal form g of f w.r.t. S (not unique in general; zero iff f ∈ Ideal(S) only when S is a GB). If S is a sequence, a second return value C gives the coordinates: g = f − Σ C[i]·S[i]. | Multivariate division algorithm. |
| `SPolynomial(f, g)` | Given polynomials f and g from a polynomial ring P, return the S-polynomial of f and g. | Standard S-polynomial construction. |
| `Reduce(S)` | Given a set or sequence S of polynomials, return the reduced sequence: each element reduced to normal form w.r.t. the others, zero elements removed, result sorted. Normally used only for simplifying non-GB sets; all Gröbner bases returned by Magma are automatically reduced. | Interreduction. |
| `ReduceGroebnerBasis(S)` | Given a set or sequence S assumed to be a (not necessarily minimal or reduced) Gröbner basis, return its reduction: first remove polynomials whose leading terms are multiples of another's, then reduce remaining polynomials as in `Reduce`. Intended for non-reduced Gröbner bases obtained from external sources. | Interreduction of a known GB. |

*Worked examples: H105E6 (Gröbner bases over Z and Z/4Z; NormalForm peculiarities over rings; ChangeRing to GF(2), GF(3)); H105E7 (finding primes modulo which a system has solutions via integer Gröbner basis; Factorization of the integer constant); H105E8 (Gröbner bases over Z[√−5] modelled in Z with an extra variable S; Coordinates and IdealWithFixedBasis; comparison to Adams–Loustaunau weak GB **[AL94]**); H105E9 (IdealWithFixedBasis over Q and Z; Coordinates of 1 and 2).*

### 105.4.5 Gröbner Bases of Boolean Polynomial Rings

Since V2.15, a special boolean polynomial ring type is available: the quotient
F₂[x₁,…,xₙ]/⟨x₁²+x₁,…,xₙ²+xₙ⟩, with bit-vector monomial representation. This is the
natural setting for solving polynomial systems over F₂ (algebraic attacks on cryptosystems).
Magma uses the same optimised F4 variant automatically when field relations x²ᵢ+xᵢ are
present in a standard F₂ ideal, but using the boolean ring type from the outset saves memory
and avoids conversion costs.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BooleanPolynomialRing(n)` | Create the boolean polynomial ring in n variables over F₂ with default lexicographical order. | — |
| `BooleanPolynomialRing(n, order)` | Create the boolean polynomial ring in n variables over F₂ with the specified order. Currently `order` must be one of `"lex"`, `"grevlex"`, `"glex"`. | — |
| `BooleanPolynomialRing(B, Q)` | Given a boolean polynomial ring B of rank n and a sequence Q of integers (each in [0, 2ⁿ−1]), create the boolean polynomial in B whose monomials are given by the binary expansions of the entries of Q. Entries are sorted w.r.t. the monomial order of B; duplicates are added. Provided for compact storage and reading back of boolean polynomials. | — |

*Worked examples: H105E5 (solving a 5-variable F₂ system by adding field polynomials to a standard ring; then the same via `BooleanPolynomialRing`; GB of a single polynomial in a 10-variable boolean ring has 38 elements).*

### 105.4.6 Verbosity

Verbose flags for Gröbner basis algorithms. The `"Groebner"` flag encompasses all
sub-algorithm flags. `false` ≡ level 0 (nothing), `true` ≡ level 1 (minimal). Each
`SetVerbose` procedure has a corresponding `GetVerbose` function.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SetVerbose("Groebner", v)` | (Procedure.) Set verbose level for all Gröbner basis algorithms. Legal levels: 0–4. Implicitly includes all sub-algorithm flags. | — |
| `SetVerbose("Buchberger", v)` | (Procedure.) Set verbose level for the Buchberger algorithm. Legal levels: 0–4. If the `"Groebner"` flag value w > v, then w is used. | — |
| `SetVerbose("Faugere", v)` | (Procedure.) Set verbose level for the Faugère F4 algorithm. Legal levels: 0–3. | — |
| `SetVerbose("FGLM", v)` | (Procedure.) Set verbose level for the FGLM order-change algorithm. Legal levels: 0–3. | — |
| `SetVerbose("GroebnerWalk", v)` | (Procedure.) Set verbose level for the Gröbner Walk order-change algorithm. Legal levels: 0–3. | — |

### 105.4.7 Degree-d Gröbner Bases

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `GroebnerBasis(S, d: parameters)` | Given a set or sequence S of polynomials from a *graded* polynomial ring P, return the weighted degree-d Gröbner basis: the truncated GB obtained by ignoring S-polynomial pairs of weighted degree (w.r.t. the grading on P) greater than d. For homogeneous ideals, the result equals the set of all full-GB polynomials of weighted degree ≤ d, and weighted-degree-≤-d membership is certified by zero normal form. Parameters are the same as for `Groebner`. Base ring may be a field or Euclidean ring. See **[BW93, §10.2]**. | Truncated F4 **[Fau99]** / Buchberger **[CLO96]** over graded ring. |

*Worked examples: H105E11 (graded ring with weights [4,3,2,1]; 4 homogeneous polynomials; degree-d GB for d = 1 to 10; verification that degree-≤-D polynomials of full GB are contained in truncated GB).*

---

## 105.5 Changing Coefficient Ring

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ChangeRing(I, S)` | Given an ideal I of P = R[x₁,…,xₙ] and a ring S, construct the corresponding ideal J of Q = S[x₁,…,xₙ] by coercing all basis coefficients from R to S. If R and S are fields with R a known subfield of S, and the current basis of I is a Gröbner basis, the basis of J is automatically marked as a Gröbner basis. | Coefficient coercion. |

*Worked examples: H105E12 (Cyclic-5 ideal over Q; Gröbner basis computed once; `ChangeRing` to cyclotomic field K; `Variety(J)` yields 70 points); H105E6 (ChangeRing to GF(2), GF(3), IntegerRing(4)).*

---

## 105.6 Changing Monomial Order

Three variants of `ChangeOrder` are available, differing in how the target order is
specified. When the Gröbner basis of the source ideal is known, Magma's order-conversion
algorithms (FGLM **[FGLM93]** or Gröbner Walk **[CKM97]**) make the conversion much more
efficient than recomputing from scratch.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ChangeOrder(I, Q)` | Given an ideal I of P = R[x₁,…,xₙ] and a polynomial ring Q of rank n (possibly with a different order), return the ideal J of Q corresponding to I and the isomorphism f: P → Q (mapping P.i → Q.i). When the Gröbner basis of J is needed, a conversion algorithm starting from a Gröbner basis of I is used. | FGLM **[FGLM93]** or Gröbner Walk **[CKM97]**, via `Groebner` parameters. |
| `ChangeOrder(I, order)` | Given an ideal I of P = R[x₁,…,xₙ] and a monomial order specification (see §105.2), construct Q = R[x₁,…,xₙ] with that order and return the ideal J of Q and the isomorphism f: P → Q. | FGLM **[FGLM93]** or Gröbner Walk **[CKM97]**. |
| `ChangeOrder(I, T)` | Given an ideal I of P = R[x₁,…,xₙ] and a tuple T (matching the §105.2 argument format, or a tuple returned by `MonomialOrder`), construct Q = R[x₁,…,xₙ] with the monomial order given by T and return the ideal J of Q and the isomorphism f: P → Q. | FGLM **[FGLM93]** or Gröbner Walk **[CKM97]**. |

*Worked examples: H105E13 (function `univgen` implementing `UnivariateEliminationIdealGenerator` via `ChangeOrder` to `"univ"` order; univariate elimination ideal generators for a 3-variable ideal).*

---

## 105.7 Hilbert-driven Gröbner Basis Construction

Magma implements the Hilbert-driven Buchberger algorithm **[Tra96]** for homogeneous ideals
whose Hilbert series is known. Knowledge of the Hilbert series eliminates many unnecessary
S-polynomial reductions, making the algorithm often much more efficient than standard
Buchberger. It can also serve as an alternative to the Gröbner Walk for order conversion
(compute the Hilbert series w.r.t. an easy order, then use it to drive computation w.r.t.
the desired order). It is used extensively in Magma's invariant theory algorithms
(Chapter 110). The function returns `false` if the supplied Hilbert data is provably
incorrect (the claimed coefficient exceeds the computed one); otherwise it returns `true`
and either the correct Gröbner basis or a partial basis.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HilbertGroebnerBasis(S, H)` / `HilbertGroebnerBasis(S, N)` | Let S be a set or sequence of homogeneous polynomials in P = K[x₁,…,xₙ] (K a field), I = Ideal(S). Let H be the Hilbert series H_{P/I}(t) (as a rational function in Z(t)) or N ∈ Z[t] be the weighted numerator of the Hilbert series (H_{P/I}(t) multiplied by ∏(1−t^{dᵢ}) where dᵢ is the weighted degree of the i-th variable). Attempts to construct the reduced Gröbner basis of I using this Hilbert data. Returns: `true`, Gröbner basis (if Hilbert data is correct or an under-estimate of the true numerator coefficient-wise) or `false` (if data is provably wrong, i.e. claimed coefficient exceeds the true coefficient at some degree). | Hilbert-driven Buchberger algorithm **[Tra96]**. |
| `SetVerbose("HilbertGroebner", v)` | (Procedure.) Set verbose level for the Hilbert-driven Buchberger algorithm. Legal values: `true`, `false`, 0, 1. | — |

*Worked examples: H105E14 (testing primary invariants of cyclic group of order 4 over F₂; `HilbertGroebnerBasis` with weighted numerator N = ∏(1−t^{deg(fᵢ)}); returns `true` and the Gröbner basis instantly).*

---

## 105.8 SAT Solver

Since V2.16, Magma provides an interface to the MiniSat satisfiability solver. Given a
boolean polynomial system, `SAT` transforms it into conjunctive normal form and calls the
external MiniSat program. MiniSat must be separately installed and available in the
executable path. To install: download minisat2-070721.zip from minisat.se, unzip, run
`make` in `minisat/core`, and copy the `minisat` executable into the path.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SAT(B)` | Given a sequence B of boolean polynomials in a rank-n boolean polynomial ring (or a rank-n polynomial ring over F₂), call MiniSat on the associated boolean system and return: (1) whether the system is satisfiable, and (2) if so, a solution S as a length-n sequence of elements of F₂. Parameters: `Exclude` ([RngMPolElt], default [] — sequences of length-n F₂-vectors to exclude as solutions, implemented by adding exclusion relations to the system); `Verbose` (BoolElt, default `true` — controls MiniSat's own verbose output). | MiniSat SAT solver (external). Boolean polynomial to CNF transformation. |

*Worked examples: H105E15 (same 5-variable F₂ system as H105E5; SAT finds both solutions iteratively by excluding each found solution; third call returns `false` confirming no further solutions).*

---

## 105.9 Bibliography

| Key | Reference |
|-----|-----------|
| **[AL94]** | William Adams and Philippe Loustaunau. *An introduction to Gröbner bases*, volume 3 of Graduate studies in mathematics. American Mathematical Society, Providence, R.I., 1994. |
| **[Buc65]** | Bruno Buchberger. *Ein Algorithmus zum Auffinden der Basiselemente des Restklassenringes nach einem nulldimensionalen Polynomideal.* PhD thesis, University of Innsbruck, Austria, 1965. |
| **[BW93]** | Thomas Becker and Volker Weispfenning. *Gröbner Bases.* Graduate Texts in Mathematics. Springer, New York–Berlin–Heidelberg, 1993. |
| **[CKM97]** | Stephane Collart, Michael Kalkbrener, and Daniel Mall. *Converting Bases with the Gröbner Walk.* J. Symbolic Comp., 24(3):465–469, 1997. |
| **[CLO96]** | David Cox, John Little, and Donal O'Shea. *Ideals, Varieties and Algorithms.* Undergraduate Texts in Mathematics. Springer, New York–Berlin–Heidelberg, 2nd edition, 1996. |
| **[CLO98]** | David Cox, John Little, and Donal O'Shea. *Using Algebraic Geometry.* Graduate Texts in Mathematics. Springer, New York–Berlin–Heidelberg, 1998. |
| **[Fau99]** | Jean-Charles Faugère. *A new efficient algorithm for computing Gröbner bases (F4).* Journal of Pure and Applied Algebra, 139(1–3):61–88, 1999. |
| **[FGLM93]** | Jean-Charles Faugère, Patrizia Gianni, Daniel Lazard, and Teo Mora. *Efficient computations of zero-dimensional Gröbner bases by change of ordering.* J. Symbolic Comp., 16:329–344, 1993. |
| **[Möl88]** | H. M. Möller. *On the construction of Gröbner bases using syzygies.* J. Symbolic Comp., 6:345–359, 1988. |
| **[Ste04]** | Allan Steel. *Gröbner Basis Timings Page.* URL: http://magma.maths.usyd.edu.au/users/allan/gb/, 2004. |
| **[Tra96]** | Carlo Traverso. *Hilbert Functions and the Buchberger Algorithm.* J. Symbolic Comp., 22(4):355–376, 1996. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Buchberger algorithm **[Buc65, CLO96]** with pair elimination **[Möl88]** | `Groebner(:Faugere:=false)`, `GroebnerBasis(:Faugere:=false)`, `GroebnerBasisUnreduced`, `HilbertGroebnerBasis` |
| Faugère F4 algorithm **[Fau99]** with pair elimination **[Möl88]** | `Groebner(:Faugere:=true)`, `GroebnerBasis(:Faugere:=true)`, `GroebnerBasis(S)`, `GroebnerBasis(S,d)` |
| Steel's extension of F4 to Euclidean rings **[Fau99]** (unpublished) | `Groebner(I)`, `GroebnerBasis(I)` when base ring is Euclidean |
| FGLM order-change algorithm **[FGLM93]** (zero-dimensional ideals) | `Groebner(:Al:="FGLM")`, `ChangeOrder` |
| Gröbner Walk order-change algorithm **[CKM97]** | `Groebner(:Al:="Walk")`, `ChangeOrder` |
| Hilbert-driven Buchberger algorithm **[Tra96]** | `HilbertGroebnerBasis` |
| Monomial order weight vector representation **[CLO98]** | `MonomialOrderWeightVectors`, `PolynomialRing(R,n,"weight",Q)` |
| Boolean polynomial / F₂ field-relation optimisation | `BooleanPolynomialRing`, `GroebnerBasis` (auto-detected), `SAT` |
| MiniSat SAT solver interface | `SAT` |
| Normal form / division algorithm | `NormalForm`, `Reduce`, `ReduceGroebnerBasis` |
| S-polynomial construction | `SPolynomial` |
| Coordinate expressions w.r.t. fixed basis | `IdealWithFixedBasis`, `Coordinates`, `CoordinateMatrix` |
| Coefficient ring change | `ChangeRing` |
| Graded rings and weighted-degree operations | `PolynomialRing(R,Q)`, `Degree`/`WeightedDegree`, `IsHomogeneous`, `HomogeneousComponent`, `MonomialsOfWeightedDegree` |
