# Chapter 40 — Algebraically Closed Fields

**Handbook part:** VI — Global Arithmetic Fields
**Handbook pages:** 1037–1055 (PDF pages 1168–1189)

---

## Scope and overview

Chapter 40 documents Magma's system for computing with algebraically closed fields **[Ste02,
Ste10]**. Because no explicit algebraic closure can be constructed in full, the system works
by *incrementally and lazily* building larger and larger algebraic extensions of an original
base field as roots are demanded during a computation, thereby giving the illusion of
computing inside a true algebraically closed field.

The design supersedes the earlier D5 system **[DDD85]**, which had difficulty with the
parallelism that arises when computing with several conjugates of a root of a reducible
polynomial (some evaluations are invertible, others are not). Magma's system handles this
transparently: all standard algorithms that work over generic fields — including all Gröbner
basis algorithms — operate without modification. Concretely, this enables computing the
variety of any zero-dimensional multivariate polynomial ideal over the algebraic closure of
its base field, and computing Puiseux expansions of polynomials.

**Representation.** An algebraically closed field is backed by an *affine algebra* (a
quotient ring of a multivariate polynomial ring by an ideal of "relation" polynomials). The
defining polynomials need not be irreducible; the system avoids factorisation over algebraic
number fields when possible and automatically splits the defining polynomials whenever
factors emerge during arithmetic. The most expensive single operation is zero-testing: to
decide whether an element `a` is zero, Magma computes the recursive GCD of `a` (viewed as
a polynomial in its highest variable) with the appropriate defining polynomial. A non-trivial
GCD forces a splitting of the defining polynomial, reduces all field elements, and may resolve
the zero question. Despite this internal splitting, zero-testing always returns an *invariable*
result — this is the key property that sustains the illusion of a true field.

**Terminology.** Return values are called *invariable* if their mathematical value is
guaranteed not to change despite subsequent simplifications of the field (they may print
differently but are constant with respect to `eq`). Return values are called *variable* if
they may change as the field evolves (e.g. `Degree(A, v)`).

Currently the base field of an algebraic closure may be **Q**, a finite field, or a rational
function field over a finite field or **Q**.

---

## 40.1 Introduction

*See Scope and overview above.*

---

## 40.2 Representation

*See Scope and overview above.* Details of the internal design, including the recursive-GCD
zero-test and technical optimisations, are documented in **[Ste10]**.

Care must be taken with the interpretation of roots: roots are only defined algebraically,
and the system makes an arbitrary (unpredictable) choice whenever two conjugate roots are
related by an arithmetic expression. For unrelated problems, separate algebraic closure
objects should be created.

---

## 40.3 Creation of Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AlgebraicClosure(K)` | Create the algebraic closure `A` of the field `K`. Currently `K` may be **Q**, a finite field, or a rational function field over a finite field or **Q**. | Incremental/lazy algebraic closure construction **[Ste02, Ste10]**. |
| `AlgebraicClosure()` | Create the algebraic closure `A` of the rational field **Q**. | Same as above. |
| `AssignNamePrefix(A, S)` | (Procedure.) Given an algebraically closed field `A` and a string `S`, reassign the string prefix of the names of `A` to `S`. Default prefix is `"r"`. New variables introduced by root-taking are named by this prefix with the variable number appended. | — |

---

## 40.4 Creation of Elements

The primary ways to create elements of an algebraically closed field `A` are coercion from
the base field and construction of roots of polynomials over `A`.

### 40.4.1 Coercion

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `A ! a` | Coerce `a` into `A`. `a` may be (i) an element of `A`, or (ii) an integer or rational. | — |
| `One(A)` / `Identity(A)` | Return `A!1`. | — |
| `Zero(A)` / `Representative(A)` | Return `A!0`. | — |

### 40.4.2 Roots

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Roots(f)` / `Roots(f, A)` | Given a polynomial `f` over an algebraically closed field `A` (or a polynomial over a subring of `A` together with `A`): return a sorted sequence of tuples `(root, multiplicity)` for all roots of `f` in `A`. Since `A` is algebraically closed, `f` always splits completely. Parameter `Max` (non-negative integer, default unset): if set to `m`, return at most `m` roots. Using `Max := 1` when only one root is needed avoids introducing unnecessary variables that make full simplification harder. `Factorization(f)` is also supported and returns the corresponding linear factors with multiplicities. | Incremental/lazy algebraic closure construction **[Ste02, Ste10]**; zero-testing via recursive GCD. |
| `RootOfUnity(n, A)` | Return a primitive `n`-th root of unity `ω` in `A`: `ωⁿ = 1` and `ωⁱ ≠ 1` for `1 ≤ i < n`. Return value is invariable. Equivalent to `Roots(CyclotomicPolynomial(n), A: Max := 1)[1,1]`. | — |
| `SquareRoot(a)` / `Sqrt(a)` | Return a square root `y` of `a` in `A` (i.e. `y² = a`). Always exists; return value is invariable. | Incremental/lazy algebraic closure **[Ste02, Ste10]**. |
| `IsSquare(a)` | Return `true` and a square root `y` of `a` such that `y² = a`. Always returns `true`; return value is invariable. | — |
| `Root(a, n)` | Return an `n`-th root `y` of `a` in `A` (i.e. `yⁿ = a`). Always exists; return value is invariable. | Incremental/lazy algebraic closure **[Ste02, Ste10]**. |
| `IsPower(a, n)` | Return `true` and an `n`-th root `y` of `a` such that `yⁿ = a`. Always returns `true`; return value is invariable. | — |

### 40.4.3 Variables

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `A . i` | Return the `i`-th variable of `A`. `i` must be between `1` and `Rank(A)`. Initially `A` has no variables; new variables are created only by root-taking functions such as `Roots` or `Sqrt`. As long as `Prune` or `Absolutize` are not called (which shift variable numbers), the return value is invariable: `A.i` for fixed `i` always returns the same mathematical object despite any simplifications. New roots are always assigned higher generator numbers. | — |

*Worked examples: H40E1 (creating roots via `Roots` and via `Sqrt`/`Root`, using `Max := 1`); H40E2 (Swinnerton-Dyer polynomials and the generalised-SD construction `GSD`, computing a degree-128 product and factoring via van Hoeij's algorithm); H40E3 (Puiseux expansions over an algebraic closure using `PuiseuxExpansion` and `PuiseuxSeriesRing`).*

---

## 40.5 Related Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Category(A)` / `Parent(A)` / `Centre(A)` | Standard generic ring functions. | — |
| `PrimeRing(A)` / `PrimeField(A)` | The prime ring / prime field of `A`. | — |
| `FieldOfFractions(A)` | The field of fractions of `A`. | — |

---

## 40.6 Properties

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BaseField(A)` | Return the base field over which `A` is defined. Return value is invariable. | — |
| `Rank(A)` | Return the current rank of `A`: the number of variables currently defining `A`. This can increase (new roots constructed) or decrease (after `Prune`), so the return value is **variable**. | — |
| `Degree(A, v)` | Given `A` of rank `r` and integer `v` with `1 ≤ v ≤ r`, return the current degree of the defining polynomial for variable `v`. **Variable** return value: simplifications of `A` may reduce the degree of the defining polynomial for `v`. | — |
| `Degree(A)` | Return the current absolute degree of `A` over its base field. Forces a call to `Simplify` (see §40.9), so may be very time-consuming. Return value is invariable after simplification until new roots are computed. | Calls `Simplify(A)` internally. |
| `AffineAlgebra(A)` / `QuotientRing(A)` | Return the affine algebra (multivariate quotient ring) `R` that currently represents `A`. Coercion between `A` and `R` is possible, but variable numbers are inverted (A.1 is the *smallest* variable in `R` with respect to lexicographic order, so that reductions modulo the Gröbner basis are in correct form). If `A` changes (simplification or pruning), the returned `R` stays fixed and may no longer match `A`. | — |
| `Ideal(A)` | Return the ideal of defining polynomials currently defining `A`. Equivalent to `DivisorIdeal(AffineAlgebra(A))`. | — |

---

## 40.7 Ring Predicates and Properties

Standard predicates and characteristic function for the ring `A`. All return invariable values.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsCommutative(A)` / `IsUnitary(A)` | Standard ring predicates. | — |
| `IsFinite(A)` / `IsOrdered(A)` | Standard ring predicates. | — |
| `IsField(A)` / `IsEuclideanDomain(A)` | Standard ring predicates. | — |
| `IsPID(A)` / `IsUFD(A)` | Standard ring predicates. | — |
| `IsDivisionRing(A)` / `IsEuclideanRing(A)` | Standard ring predicates. | — |
| `IsPrincipalIdealRing(A)` / `IsDomain(A)` | Standard ring predicates. | — |
| `A eq B` / `A ne B` | Equality / inequality of algebraically closed fields. | — |
| `Characteristic(A)` | Return the characteristic of `A`. | — |

---

## 40.8 Element Operations

### 40.8.1 Arithmetic Operators

Elements are always kept in normal form with respect to the defining relations of the field.
Computing the inverse of an element may cause a simplification of the field.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `+ a` / `- a` | Unary plus and negation. | — |
| `a + b` / `a - b` / `a * b` / `a / b` / `a ^ k` | Binary arithmetic operators. Division may trigger field simplification. | Reduction modulo the Gröbner basis of defining relations; zero-test via recursive GCD **[Ste02, Ste10]**. |
| `a +:= b` / `a -:= b` / `a *:= b` | In-place arithmetic assignment operators. | — |

### 40.8.2 Equality and Membership

Equality testing (`a eq b`) is performed by testing whether `a − b` is zero via the `IsZero`
algorithm (see §40.8.4). Membership (`a in A`, `a notin A`) is the standard generic operation.

### 40.8.3 Parent and Category

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Parent(a)` / `Category(a)` | Return the parent field and the category of element `a`. | — |

### 40.8.4 Predicates on Ring Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsZero(a)` | Return whether `a` is the zero element of its field. This is the most difficult arithmetic function for algebraically closed fields. Magma computes the recursive GCD of `a` (as a polynomial in its highest variable) with the appropriate defining polynomial. A non-trivial GCD forces a splitting of the defining polynomial; all elements are reduced; the original element may then be zero. Despite possible simplifications, the return value is invariable — this is the central property that sustains the illusion of a true field. | Recursive GCD zero-test **[Ste02, Ste10]**. |
| `IsOne(a)` | Return whether `a` equals one; determined by testing `IsZero(a − 1)`. A simplification may occur; return value is invariable. | Reduces to `IsZero`. |
| `IsMinusOne(a)` | Return whether `a` equals minus one; determined by testing `IsZero(a + 1)`. A simplification may occur; return value is invariable. | Reduces to `IsZero`. |
| `a eq b` | Return whether `a = b`; determined by testing `IsZero(a − b)`. A simplification may occur; return value is invariable. | Reduces to `IsZero`. |
| `a ne b` | Inequality; negation of `a eq b`. | — |
| `a in A` / `a notin A` | Membership test. | — |
| `IsNilpotent(a)` / `IsIdempotent(a)` | Standard ring-element predicates. | — |
| `IsUnit(a)` / `IsZeroDivisor(a)` / `IsRegular(a)` | Standard ring-element predicates. | — |
| `IsIrreducible(a)` / `IsPrime(a)` | Standard ring-element predicates. | — |

### 40.8.5 Minimal Polynomial, Norm and Trace

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `MinimalPolynomial(a)` | Return the minimal polynomial of `a` relative to the base field of `A`: the unique irreducible monic polynomial of minimal degree with base-field coefficients having `a` as a root. Internally: compute the minimal polynomial `M` of `a` in the corresponding affine algebra; factor `M`; for each irreducible factor `F`, evaluate `F(a)` and test `IsZero` — exactly one evaluation is zero and its factor is the answer. Evaluations of non-zero factors cause simplifications of `A`. Return value is invariable. | Factor `M` over the base field; select the factor via the `IsZero` algorithm **[Ste02, Ste10]**. |
| `Norm(a)` | Return the absolute norm of `a` to the base field of `A`. Computed as `(−1)^deg(M)` times the constant coefficient of `M`, where `M = MinimalPolynomial(a)`. A simplification may occur; return value is invariable. | Via `MinimalPolynomial`. |
| `Trace(a)` | Return the absolute trace of `a` to the base field of `A`. Computed as the negation of the coefficient of `x^{n−2}` in `M = MinimalPolynomial(a)`. A simplification may occur; return value is invariable. | Via `MinimalPolynomial`. |
| `Conjugates(a)` | Return the conjugates of `a` as a sequence of elements of `A`. The conjugates are the roots of `MinimalPolynomial(a)` in `A`; `a` itself is always included. Equivalent to `[t[1] : t in Roots(MinimalPolynomial(a), A)]`. No multiplicities are returned (minimal polynomial is always squarefree). A simplification may occur; return value is invariable. | Via `MinimalPolynomial` and `Roots`. |

*Worked example: H40E4 (creating two conjugate elements `x = Sqrt(A!2) + Sqrt(A!-3)` and `y = Sqrt(A!(-1 + 2*Sqrt(A!-6)))`, testing `x eq y`, computing `Conjugates(x)`, and verifying they share the same minimal polynomial `z^4 + 2*z^2 + 25`).*

---

## 40.9 Simplification

The following procedures allow one to simplify an algebraically closed field so that it
represents a true field (i.e. its affine algebra is a field, meaning the defining ideal is
maximal).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Simplify(A)` | (Procedure.) Simplify the algebraically closed field `A` in place so that the affine algebra representing it is a true field (i.e. the multivariate polynomial ideal of defining polynomials is maximal). First performs a partial simplification by calling `MinimalPolynomial` on all variables and on all sums of two variables of `A` (forcing the corresponding minimal polynomials to be irreducible, causing many simplifications). Then performs all remaining simplifications by successively computing absolute representations and factorising the absolute polynomials that arise. This final step can be very expensive if the absolute degree exceeds about 20. Parameter `Partial` (boolean, default `false`): if `true`, perform only the partial simplification (usually fast and often sufficient). | Partial step: `MinimalPolynomial`-based splitting **[Ste02, Ste10]**. Full step: successive absolute field computation and factorisation. |
| `Prune(A)` | (Procedure.) Remove useless variables from `A` in place. For each variable `v` whose defining polynomial is linear, remove `v` and its defining polynomial from `A` and shift higher variable numbers down. Since elements of `A` are kept in normal form, any such `v` cannot appear in any element of `A`, making removal safe. | — |

---

## 40.10 Absolute Field

One may construct an absolute field isomorphic to the current subfield represented by an
algebraically closed field. This may be very expensive (it involves factoring polynomials over
successive subfields), and in practice the absolute degree is often large enough that an
absolute presentation is impractical, while the non-absolute multivariate presentation
remains effective for computation.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AbsoluteAffineAlgebra(A)` / `AbsoluteQuotientRing(A)` | Fully simplify `A` (see `Simplify`) and return: (1) a univariate affine algebra `R` isomorphic to the current algebraic field represented by `A`; (2) the isomorphism `f : A → R`. | Full simplification via `Simplify(A)`, then absolute representation **[Ste02, Ste10]**. |
| `AbsolutePolynomial(A)` | Fully simplify `A` and return the defining polynomial `f` of the absolute field, i.e. a polynomial `f` such that `K[x]/<f>` is isomorphic to the current state of `A`. | Full simplification and absolute polynomial computation **[Ste02, Ste10]**. |
| `Absolutize(A)` | (Procedure.) Modify `A` in place so that `A` has an absolute presentation: compute an isomorphic absolute field with defining polynomial `f` (as in `AbsolutePolynomial`) and modify `A` and all its elements in place so that `A` now has exactly one variable `v` with defining polynomial `f(v)`, and existing elements correspond via the isomorphism to their old representation. | Full simplification and isomorphism rewriting **[Ste02, Ste10]**. |

*Worked examples: H40E5 (the Cyclic-6 ideal: computing a 156-point variety over the algebraic closure of **Q**, simplifying from 28 variables to 3, then computing `AbsolutePolynomial` giving `x^8 + 4*x^6 − 6*x^4 + 4*x^2 + 1` and finally `Absolutize` to a single degree-8 variable); H40E6 (splitting field of a degree-8 polynomial with Galois group of order 16: computing roots, simplifying to a 2-generator field with polynomials of degrees 2 and 8, then `AbsolutePolynomial` yields the degree-16 splitting-field polynomial).*

---

## 40.11 Bibliography

| Key | Reference |
|-----|-----------|
| **[DDD85]** | J. Della Dora, C. Dicrescenzo, and D. Duval. About a new method for computing in algebraic number fields. In B. F. Caviness, editor, *Proc. EUROCAL '85*, volume 204 of LNCS, pages 289–290, Linz, 1985. Springer. |
| **[FK02]** | Claus Fieker and David R. Kohel, editors. *ANTS V*, volume 2369 of LNCS. Springer-Verlag, 2002. |
| **[Ste02]** | Allan Steel. A new scheme for computing with algebraically closed fields. In Fieker and Kohel [FK02], pages 491–505. |
| **[Ste10]** | Allan K. Steel. Computing with algebraically closed fields. *J. Symb. Comput.*, 45(3):342–372, March 2010. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Incremental/lazy algebraic closure construction **[Ste02, Ste10]** | `AlgebraicClosure`, `Roots`, `RootOfUnity`, `SquareRoot`/`Sqrt`, `Root`, `IsSquare`, `IsPower` |
| Recursive GCD zero-test **[Ste02, Ste10]** | `IsZero`, `IsOne`, `IsMinusOne`, `eq` (element), arithmetic operators |
| Minimal polynomial via factor-selection with IsZero **[Ste02, Ste10]** | `MinimalPolynomial`, `Norm`, `Trace`, `Conjugates` |
| Partial simplification (MinimalPolynomial-based splitting) **[Ste02, Ste10]** | `Simplify(:Partial := true)`, `Degree(A)` |
| Full simplification (successive absolute field + factorisation) **[Ste02, Ste10]** | `Simplify`, `AbsoluteAffineAlgebra`/`AbsoluteQuotientRing`, `AbsolutePolynomial`, `Absolutize` |
| Linear-variable pruning | `Prune` |
