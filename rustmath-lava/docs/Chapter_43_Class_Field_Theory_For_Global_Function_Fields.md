# Chapter 43 — Class Field Theory For Global Function Fields

**Handbook part:** VI — Global Arithmetic Fields
**Handbook pages:** 1191–1214 (PDF pages 1322–1347)

---

## Scope and overview

Global function fields admit a class field theory in the same way as number fields do (Chapter 39). From a computational point of view the main difference is the use of divisors rather than ideals and the availability in general of analytical methods (see Section 43.6). Class field theory deals with the abelian extensions of a given field. In the number field case, all abelian extensions can be parameterized using more general class groups; in the case of global function fields, the same is achieved using the divisor class group and extensions of it.

The chapter is organized into eight sections: ray class groups (§43.1), creation of class fields using Artin-Schreier-Witt theory for p-extensions (§43.2), properties of class fields computable directly from norm groups without defining equations (§43.3), Witt vectors of finite length parametrizing cyclic p-power extensions (§43.4), the ring of twisted polynomials central to the analytic theory (§43.5), the analytic theory via Drinfeld modules and Carlitz modules (§43.6), related auxiliary functions (§43.7), and place enumeration utilities (§43.8).

A key feature is that many invariants of an abelian extension (conductor, discriminant divisor, genus, decomposition types, number of rational places) can be read off directly from the norm group data, enabling computation over very large fields without constructing defining equations.

---

## 43.1 Ray Class Groups

The ray divisor class group Cl_m modulo a divisor m of a global function field K is defined via the exact sequence:

```
1 → k× → O×_m → Cl_m → Cl → 1
```

where O×_m is the group of units in the "residue ring" mod m, k is the exact constant field of K, and Cl is the divisor class group of K. This follows the methods outlined in **[HPP97]**. Note that in contrast to the number field case, the ray class group of a function field is infinite. For large examples it may be necessary to precompute the class group using `ClassGroup` directly.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RayResidueRing(D)` | Let D = Σ nᵢPᵢ be an effective divisor (nᵢ > 0). Returns the ray residue ring R = O×_m := ∏ (O_{Pᵢ}/Pᵢ^{nᵢ})×, together with a map from R into the function field that admits a pointwise inverse for discrete logarithm computations. | Follows **[HPP97]**; product of local unit groups. |
| `RayClassGroup(D)` | For an effective (positive) divisor D, returns the ray class group modulo D (quotient of divisors coprime to D modulo certain principal divisors) computed using the exact sequence 1 → k× → O×_m → Cl_m → Cl → 1. Second return value is the map from the ray class group into the group of divisors (admits a pointwise inverse). | **[HPP97]**. |
| `RayClassGroupDiscLog(y, D)` | Returns the discrete log of the place or divisor y in the ray class group modulo the divisor D. This is a cached version of the pointwise inverse of the map returned by `RayClassGroup`: repeated decompositions of the same place or divisor are instantaneous, but large numbers of distinct decompositions may waste memory. Two overloads: y a place, or y a divisor. | Cached discrete logarithm using the ray class group map. |

*Worked example: H43E1 (creation of ray residue rings and ray class groups over GF(4); verification of the exact sequence 1 → k× → O×_m → Cl_m → Cl → 1).*

---

## 43.2 Creation of Class Fields

The method used to compute defining equations for class fields of global function fields is essentially the same as for number fields. The main differences are the treatment of p-extensions in characteristic p using Artin-Schreier-Witt theory, and the fact that the divisor class group is infinite. No defining equations are computed at the time an abelian extension object is created.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AbelianExtension(D, U)` | Given an effective divisor D and a subgroup U of the ray class group Cl_D (see `RayClassGroup`) such that the quotient Cl_D/U is finite, creates the abelian extension defined by this data. No defining equations are computed at this point. | Class field theory correspondence; deferred equation computation. |
| `MaximalAbelianSubfield(K)` | For a relative extension K/k of global function fields, computes the maximal abelian subfield K/A/k of K/k as an abelian extension of k. In particular, computes the norm group of K/k as a subgroup of a suitable ray class group. | Norm group computation. |
| `HilbertClassField(K, p)` | For a global function field K and a place p of K, computes the Hilbert class field of K as an abelian extension of K. This field is characterized as the maximal abelian unramified extension of K where p is totally split. | Class field theory (unramified abelian extensions). |
| `FunctionField(A)` | Given an abelian extension A of function fields (as created by `AbelianExtension`), computes defining equations for the corresponding (ray) class field. Decomposes Cl/U = ∏ Cl/Uᵢ into cyclic prime-power quotients and computes a defining equation for each Uᵢ; A is represented as the compositum. Parameters: `WithAut` (BoolElt, default false) — if true, the second return is a sequence of generating automorphisms for each component field; if false, only a single function field in non-simple representation is returned. `Verbose ClassField` (maximum 3). | Artin-Schreier-Witt theory for p-extensions; Kummer theory for prime-to-p extensions. |
| `MaximalOrderFinite(A)` / `MaximalOrderInfinite(A)` | Compute the finite or infinite maximal order, respectively, of the function field of the abelian extension A. | Maximal order algorithms for function fields. |

*Worked example: H43E2 (construction of a class field of exponent 5 with prescribed splitting at infinity; enumeration of 768 degree-4 subgroups with genus/rational-places ratio maximized).*

---

## 43.3 Properties of Class Fields

The main existence theorem of class field theory asserts that there is exactly one function field corresponding to the quotient Cl_D/U whose Galois group is isomorphic to Cl_D/U canonically. Since the field is uniquely defined this way, so are its invariants. The functions in this section compute invariants directly from the group data; none computes defining equations, so they are usable even for very large fields.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Conductor(m)` | For an effective divisor m, computes the conductor of Cl_m: the smallest divisor f such that the projection Cl_m → Cl_f is surjective. | Divisor-class-group theory. |
| `Conductor(m, U)` | For an effective divisor m and a subgroup U of the ray class group of m, computes the conductor of Cl_m/U: the smallest divisor f such that the projection π: Cl_m/U → Cl_f/π(U) is an isomorphism. | Divisor-class-group theory. |
| `Conductor(A)` | For an abelian extension A of global function fields, computes its conductor, i.e. the conductor of the norm group of A. | Norm group. |
| `DiscriminantDivisor(m, U)` | For an effective divisor m and a subgroup U of the ray class group such that Cl_m/U is finite, computes the discriminant divisor defined as the norm of the different divisor. | Conductor-discriminant formula for function fields. |
| `DiscriminantDivisor(A)` | For an abelian extension A of a global function field, computes its discriminant divisor (norm of the different divisor) from the norm group — no defining equation is derived. | Norm group. |
| `DegreeOfExactConstantField(m)` | For an effective divisor m: since the ray class field modulo m is always an infinite field extension containing the algebraic closure of the constant field, returns ∞. | — |
| `DegreeOfExactConstantField(m, U)` | For an effective divisor m and a subgroup U of the ray class group modulo m, computes the degree of the algebraic closure of the constant field in the class field corresponding to Cl_m/U. May be infinite. | Norm group theory. |
| `DegreeOfExactConstantField(A)` | The degree of the exact constant field of the abelian extension A (degree of the algebraic closure of the constant field of the base field in A). Computed from the norm group; no defining equation derived. | Norm group. |
| `Genus(m, U)` | For an effective divisor m and a subgroup U of the ray class group modulo m, computes the genus of the class field corresponding to Cl_m/U. | Riemann-Hurwitz formula applied to norm group data. |
| `Genus(A)` | The genus of the abelian extension A of a global function field. | Norm group. |
| `DecompositionType(m, U, p)` | For an effective divisor m, a subgroup U of the ray class group such that Cl_m/U is finite, and a place p, determines the decomposition type of p in the extension defined by Cl_m/U: returns a sequence of pairs ⟨f, e⟩ giving the inertia degree and ramification index for all places above p. | Class field theory decomposition law. |
| `DecompositionType(A, p)` | For an abelian extension A of a global function field k and a place p of k, computes the degree and ramification index of all places P lying above p. | Norm group. |
| `NumberOfPlacesOfDegreeOne(m, U)` | For an effective divisor m and a subgroup U of the ray class group such that Cl_m/U is finite, computes the number of degree-1 places of the corresponding class field. | Frobenius density / class field counting. |
| `NumberOfPlacesOfDegreeOne(A)` | For an abelian extension A of global function fields, computes the number of places of A that are of degree one over the constant field of the base field. | Norm group. |
| `Degree(A)` | For an abelian extension A of global function fields, returns the degree of A over its base ring. | — |
| `BaseField(A)` | For an abelian extension A of global function fields, returns the base field k over which A was created as an extension. | — |
| `A eq B` | For two abelian extensions of the same base field, decides if they describe the same field, i.e. if the norm groups pulled back into a common overgroup agree. | Norm group comparison. |
| `A subset B` | For two abelian extensions of the same base field, tests if the first is contained in the second by comparing norm groups in a common overgroup; no defining equations computed. | Norm group inclusion. |
| `A meet B` | Computes the intersection of two abelian extensions of the same base field as an abelian extension. | Norm group join. |
| `A * B` | Computes the compositum of two abelian extensions of the same base field as an abelian extension. Both fields are normal, so the compositum is well-defined and computed from the norm groups alone. | Norm group meet. |

---

## 43.4 The Ring of Witt Vectors of Finite Length

The ring of Witt vectors of length n (type `RngWitt`) over a global function field K parametrizes the cyclic extensions of K of degree pⁿ, where p is the characteristic of K. Witt vectors (type `RngWittElt`) can be defined over any ring of positive characteristic p, and the ring of Witt vectors of length n always has characteristic pⁿ. Over finite fields, Witt rings are isomorphic to finite quotients of unramified p-adic rings. The functionality here is mainly motivated by class field theory for short-length vectors. The Witt ring implementation is based on code developed by David Kohel.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `WittRing(F, n)` | Creates the ring of Witt vectors of length n where F must be a field of positive characteristic. | Standard Witt vector construction. |
| `W ! a` | Constructs a Witt vector of the ring W from a, where a may be: an element of the same ring, an integer, an element of the base ring, or a sequence of length `Length(W)` whose universe is coercible to the base ring of W. | — |
| `BaseRing(W)` / `BaseField(W)` | The field of coefficients of the Witt ring W. | — |
| `Length(W)` | The length (dimension) of the elements of the Witt ring W. | — |
| `Eltseq(a)` | The list of coefficients of the Witt vector a. | — |
| `Unity(W)` / `Zero(W)` | The one and zero, respectively, of the Witt ring W. | — |
| `W . 1` | The first non-trivial basis element of the Witt ring. | — |
| `x in W`, `a eq b`, `a - b`, `- a`, `a + b`, `a * b`, `a ^ n` | Standard membership test and arithmetic operations on Witt vectors. | Witt vector addition/multiplication formulas. |
| `FrobeniusMap(W)` | The Frobenius map on the Witt ring W: the map sending vectors to vectors where every coefficient is raised to the pth power. | — |
| `FrobeniusImage(e)` | Computes the image of the Witt vector e under the Frobenius map. | — |
| `VerschiebungMap(W)` | The Verschiebung map of the Witt ring W: shifts all coefficients one position to the right and pads with a zero in front. | — |
| `VerschiebungImage(e)` | Computes the image of the Witt vector e under the Verschiebung map. | — |
| `Random(W)` | For finite Witt rings (defined over finite fields), returns a random element. | — |
| `Random(W, n)` | For Witt rings where the base field admits a random function with size restriction n, returns a random element with that restriction. | — |
| `TeichmuellerSystem(R)` | A Teichmüller system for the local ring R: a system of representatives for the residue class field of R that is closed under multiplication. | Teichmüller lift. |
| `LocalRing(W)` | Any Witt ring W of finite length over a finite field is isomorphic to some unramified local ring. Creates the corresponding local ring and the embedding into it. | — |
| `ArtinSchreierMap(W)` | Returns the map x ↦ F(x) − x where F is the Frobenius map of the Witt ring W (the Artin-Schreier-Witt operator). | Artin-Schreier-Witt theory. |
| `ArtinSchreierImage(e)` | Computes the image of the Witt vector e under the Artin-Schreier map. | — |
| `FunctionField(e)` | For a Witt vector e = (e₁, …, eₙ) of length n over k with e₁ not in the image of the Artin-Schreier map (e₁ ∉ {xᵖ − x : x ∈ k}), computes the cyclic extension K/k of degree pⁿ defined by e. Parameters: `WithAut` (BoolElt, default true) — if true, also returns a generating automorphism; `Abs` (BoolElt, default false) — if true, constructs K as a single step extension; if false, constructs K as a series of n Artin-Schreier extensions; `Check` (BoolElt, default false) — if true, verifies the extension has degree pⁿ and tests the restriction on e₁. | Artin-Schreier-Witt theory. |

---

## 43.5 The Ring of Twisted Polynomials

The ring of twisted polynomials plays a core role in the analytic side of class field theory for global function fields. Twisted polynomials can be viewed as additive polynomials where multiplication is composition, or equivalently as polynomials in the Frobenius automorphism F as indeterminate, with multiplication defined by rF = Frq for all r in the base ring. Magma uses the Frobenius representation (lower degrees) but the additive polynomial representation is always available via `Polynomial`. Every endomorphism in positive characteristic can be represented by an additive polynomial or additive power series, so this section also makes the endomorphism ring accessible. In Magma the ring of twisted polynomials is of type `RngUPolTwst` and elements are of type `RngUPolTwstElt`. They can be created over any ring of characteristic p > 0. The ring is left Euclidean and therefore a left PIR (but not in general a right PIR).

### 43.5.1 Creation of Twisted Polynomial Rings

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `TwistedPolynomials(R)` | Given a ring R of characteristic p > 0, creates the ring of twisted polynomials over R. Parameter: `q` (RngIntElt, default false — defaults to p): if given, must be a power of p. Multiplication is defined by rF = Frq for all r ∈ R; elements are represented as polynomials in F. | — |

### 43.5.2 Operations with the Ring of Twisted Polynomials

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Unity(R)` | For a ring R of twisted polynomials, returns the polynomial representing 1. | — |
| `Zero(R)` | For a ring R of twisted polynomials, returns the polynomial representing 0. | — |
| `R eq S` | For two rings of twisted polynomials R and S, tests equality: whether the underlying polynomial rings coincide (i.e. the base rings of R and S coincide). | — |
| `BaseRing(R)` | The coefficient ring of the ring R of twisted polynomials. | — |
| `R . i` | For a ring of twisted polynomials R and integer i (should be 1), returns the transcendental element of R. | — |

### 43.5.3 Creation of Twisted Polynomials

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AdditivePolynomialFromRoots(x, P)` | Given a place P and a ring element x, computes the additive polynomial with roots L(InfBound · P) evaluated at x (the module M is a Riemann-Roch space). Parameters: `InfBound` (RngIntElt, default 5) — controls the Riemann-Roch space bound; `Map` (Map, default id) — applied to elements of the Riemann-Roch space before polynomial computation, allowing computation over completions; `Limit` (RngIntElt, default ∞) — reduces the polynomial modulo x^Limit; `Class` (DivFunElt, default 0) — if non-zero, uses L(nP + Class) instead of L(nP); `Scale` (RngElt, default false) — scales the module for normalization. | Riemann-Roch space construction; product formula for additive polynomials from roots. |
| `Random(F, n)` | For F the ring of twisted polynomials over a finite ring, returns a twisted polynomial of degree n−1 with randomly chosen coefficients. | — |

*Worked example: H43E3 (construction and arithmetic in TwistedPolynomials over GF(4); `AdditivePolynomialFromRoots` with InfBound and analytic Map parameters; verification that Riemann-Roch elements are roots of the resulting additive polynomial).*

### 43.5.4 Operations with Twisted Polynomials

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `A + B`, `A - B`, `- A`, `A * B`, `A ^ n` | Standard arithmetic on twisted polynomials (where multiplication is composition of additive polynomials). | Non-commutative ring arithmetic. |
| `A eq B`, `IsZero(A)` | Equality test and zero check for twisted polynomials. | — |
| `LeadingCoefficient(F)` | For a twisted polynomial F, returns the leading coefficient as an element of the coefficient ring. | — |
| `ConstantCoefficient(F)` | For a twisted polynomial F, returns the constant coefficient as an element of the coefficient ring. | — |
| `Degree(F)` | For a twisted polynomial F, returns its degree. Note: the degree of the underlying additive polynomial is q times the degree of the twisted polynomial. | — |
| `Quotrem(F, G)` | For twisted polynomials F and G in the same ring, performs right division with remainder: computes Q and R such that F = Q * G + R and deg(R) < deg(G). In general, unless the coefficient ring is algebraically closed or perfect, there is no left quotient; the ring is a left-PID but not a right-PID. | Left Euclidean division. |
| `GCD(F, G)` | For twisted polynomials F and G in the same ring, computes the greatest common right divisor: a twisted polynomial H (monic, maximal degree) such that F = f₁H and G = f₂H for some twisted polynomials f₁, f₂. | Left-PID GCD algorithm. |
| `BaseRing(F)` | For a twisted polynomial F, returns the coefficient ring (the ring where all coefficients of F are from). | — |
| `Polynomial(G)` | For a twisted polynomial G, returns the corresponding additive polynomial by replacing F^i by T^{qᵢ} for i = 0, …, degree of G. | — |
| `SpecialEvaluate(F, x)` | For a twisted polynomial F, returns the result of evaluating the corresponding additive polynomial at x. Second overload: for a univariate polynomial F, evaluates F at x, optimized for sparse polynomials (e.g. additive polynomials) and imprecise coefficients; in the general case `Evaluate` is faster. | Optimized sparse polynomial evaluation. |
| `Eltseq(F)` | For a twisted polynomial F, returns a sequence containing its coefficients. | — |

---

## 43.6 Analytic Theory

Probably the most significant difference between the class field theories for number fields and function fields is that function fields allow an analytic description of abelian extensions in general, whereas number fields (currently) only admit the analytical view for extensions of the rationals (cyclotomic fields) and imaginary quadratic fields (CM-theory). The analytic description is based on Drinfeld modules of rank 1 (or, in the case of the rational function field, the Carlitz module). Informally, a Drinfeld module is a representation of some (infinite) maximal order of a function field into the ring of additive polynomials over some appropriate ring containing the original field. Similar to CM-theory, abelian extensions are generated by adjoining torsion points under this action.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CarlitzModule(R, x)` | For a rational function field k = Fq(t) and a polynomial f in k[t], computes the image of f under the Carlitz module as an element in the ring of twisted polynomials R. The Carlitz module is the representation induced by sending t to F + t (where F is the Frobenius of k); since Fq[t] is freely generated by t, this defines a ring homomorphism Fq[t] → R. | Carlitz module: the canonical rank-1 Drinfeld module for Fq[t]. |
| `AnalyticDrinfeldModule(F, p)` | For F a global function field and p a place, computes an algebraic description of "the" Drinfeld module of rank 1 defined for the ring R of functions integral outside p. Internally: changes the representation of F so that p is the only infinite place (making R the finite maximal order), then constructs the rank-1 lattice Λ in the completion C, and computes the corresponding exponential functions to define the Drinfeld module map R → End(C). Returns a non-constant element A of R (chosen with maximal valuation at p, sign 1) and its image as a twisted polynomial over the Hilbert class field of F. If the place has degree 1, the Drinfeld module is sign-normalized. | Rank-1 Drinfeld module over the Hilbert class field; sign-normalization when deg(p) = 1. |
| `Extend(D, x, p)` | Given D (the image of a non-constant element under a Drinfeld module for the ring R of functions integral outside p) and an element x in R, computes the image of x under the Drinfeld module. | Extends a Drinfeld module to arbitrary elements of R using the ring homomorphism property. |
| `Exp(x, p)` | Computes an approximation to the exponential of the "standard lattice" associated to the place p. Let R be the ring of functions integral outside p, C the completion at p, and Λ ⊂ C the 1-dimensional lattice (in Drinfeld's sense). The exponential is exp: z ↦ z ∏' (1 − z/l) over non-zero lattice points l. Approximation via Riemann-Roch spaces: fn(z) = z ∏'_{l ∈ L(np)\{0}} (1 − z/l) → exp as n → ∞; this function computes fn for n = InfBound. Parameters: `InfBound` (RngIntElt, default 5); `Limit` (RngIntElt, default ∞) — truncates fn at that term; `Class` (DivFunElt, default 0) — uses L(np + d) to approximate an exponential from a non-isomorphic lattice; `Map` (Map, default id) — applied to each element of L(np) for analytic approximations; `Scale` (RngElt, default false) — scales elements of L(np) for lattice scaling. The exponential is evaluated at x (typically the transcendental element of a polynomial ring, twisted polynomial ring, or power series ring). | Drinfeld exponential via Riemann-Roch product formula. |
| `AnalyticModule(x, p)` | Let Λ be the lattice as in `Exp`. By Drinfeld's theory the exponential functions of Λ and xΛ are related through some polynomial. Computes that polynomial for x — i.e. "the" image of x under the Drinfeld module defined by Λ. Parameters as for `Exp`. | Drinfeld module via ratio of exponentials; same parameters as `Exp`. |
| `CanNormalize(F)` | For a twisted polynomial F (typically over a completion), tries to conjugate F so that the coefficients are integral with small valuations. On success, returns true, the new polynomial, and the element used to normalize F. | Integral normalization heuristic. |
| `CanSignNormalize(F)` | For a twisted polynomial F (typically over a completion), tries to conjugate F so that the highest coefficient is an element in the residue class field. On success, returns true, the new polynomial, and the normalizing element. | Sign normalization heuristic. |
| `AlgebraicToAnalytic(F, p)` | Given a non-trivial image F under a Drinfeld module with "infinite place" p, computes a basis for a submodule of the lattice underlying F. Parameter `RelPrec` limits the number of coefficients of the exponential reconstructed, thus also limiting the dimension of the submodule. | Lattice reconstruction from the Drinfeld module image. |

*Worked example: H43E4 (Carlitz module for p = t²+t+w over GF(4); construction of the ray-class field modulo p; verification of automorphism group isomorphism to (Fq[t]/p)×; comparison with algebraic class field machinery via `NormGroup` and `AbelianExtension`).*

*Worked example: H43E5 (analytic Drinfeld module on an elliptic curve of genus 1 over GF(4); `AnalyticDrinfeldModule`, `Extend`, `GCD` of twisted polynomials; construction of ray-class field extension; verification via `NormGroup` and `FunctionField`).*

---

## 43.7 Related Functions

This section lists related functions that are either useful in the context of class fields for function fields or are necessary for their computation.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `StrongApproximation(m, S)` | Given an effective divisor m and a sequence S of pairs (Qᵢ, eᵢ) of places and elements, finds an element a and a place Q₀ such that v_{Qᵢ}(a − eᵢ) ≥ v_{Qᵢ}(m) and a is integral everywhere outside Qᵢ (0 ≤ i ≤ n). Parameters: `Exception` (DivFunElt, default false) — if not false, specifies the place Q₀; `Strict` (BoolElt, default false) — if true, enforces v_{Qᵢ}(a − eᵢ) = v_{Qᵢ}(m) (doubles running time); `Raw` (BoolElt, default false) — returns technical internal values used internally. | Strong approximation theorem for function fields. |
| `NonSpecialDivisor(m)` | Given an effective divisor m, finds a place P coprime to m and an integer r ≥ 0 such that rP − m is a non-special divisor; returns r and P. Parameter: `Exception` (DivFunElt, default not set) — if specified, must be an effective divisor n coprime to m; finds r > 0 such that rn − m is non-special and returns r and n. | Riemann-Roch / divisor theory. |
| `NormGroup(F)` | Given a global function field F, tries to compute its norm group (the group generated by norms of unramified divisors). Provided F is abelian, computes a divisor m and a subgroup U of the ray class group modulo m such that F is isomorphic to the ray class field thus defined. Parameters: `Cond` (DivFunElt, default not set) — if given, an effective divisor used as the potential conductor (if too small, the result may be wrong; the discriminant divisor is the default starting point but is in general far too large); `AS` (RngWittElt, default not set) — if given, a Witt vector e such that F is the corresponding function field, enabling a much better initial guess for the conductor; `Extra` (RngIntElt, default 5) — terminates after the norm group quotient has size ≤ degree for this many additional places (heuristic). | Heuristic norm group computation (uses unramified Frobenius elements). |
| `Sign(a, p)` | Given a function a in a global function field and a place p such that a is integral at p (non-negative valuation), returns the sign of a: the first non-zero coefficient in the expansion of a at p. The sign function is chosen when the residue class field map is created. | Residue class field expansion. |
| `ChangeModel(F, p)` | Given a global function field F and a place p, returns a new function field G that is Fq-isomorphic to F and has p as the only infinite place. | Change of function field model. |

*Worked example: H43E6 (strong approximation on a function field over GF(4): finding an element x with prescribed valuations at a sequence of places; verification that v_{lp[i]}(x − e[i]) ≥ i).*

---

## 43.8 Enumeration of Places

In several situations one needs to loop over the places of a function field until a place with special properties is found or until they generate a certain group. The functions here support this.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PlaceEnumInit(K)` | Initialises an enumeration process for places of the function field K: loops over all irreducible polynomials of the underlying finite field and for each polynomial over all primes lying above it. Parameters: `Coprime` (Any, default not set) — a set of places or a divisor; if a divisor, only places coprime to it are returned; `All` (BoolElt, default false) — if false, infinite places are not considered. | Ordered place enumeration by norm. |
| `PlaceEnumInit(P)` | Constructs an enumeration process for places starting at the place P. Parameter: `Coprime` (Any, default not set) — as above. | — |
| `PlaceEnumInit(K, Pos)` | Constructs an enumeration environment that starts at the place of the function field K indexed by Pos, as returned from `PlaceEnumPosition`. Parameter: `Coprime` (Any, default not set) — as above. | — |
| `PlaceEnumCopy(R)` | Copies the environment and current state of the enumeration process R. | — |
| `PlaceEnumPosition(R)` | Returns a list of integers acting as an index to the places as enumerated by the environment R. | — |
| `PlaceEnumNext(R)` | Returns the "next" place of the process R. | — |
| `PlaceEnumCurrent(R)` | Returns the current place pointed to by the environment R (the last place returned by `PlaceEnumNext`). | — |

---

## 43.9 Bibliography

| Key | Reference |
|-----|-----------|
| **[HPP97]** | Florian Heß, Sebastian Pauli, and Michael E. Pohst. *On the computation of the multiplicative group of residue class rings.* Math. Comp., 1997. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Ray class group (exact sequence, multiplicative group of residue rings) **[HPP97]** | `RayResidueRing`, `RayClassGroup`, `RayClassGroupDiscLog` |
| Artin-Schreier-Witt theory (p-power cyclic extensions) | `FunctionField(A)`, `FunctionField(e)`, `ArtinSchreierMap`, `ArtinSchreierImage` |
| Class field theory (norm group / abelian extension correspondence) | `AbelianExtension`, `MaximalAbelianSubfield`, `HilbertClassField`, `MaximalOrderFinite`, `MaximalOrderInfinite`, `NormGroup` |
| Invariants from norm group (no defining equations) | `Conductor`, `DiscriminantDivisor`, `DegreeOfExactConstantField`, `Genus`, `DecompositionType`, `NumberOfPlacesOfDegreeOne`, `Degree`, `BaseField`, `eq`, `subset`, `meet`, `*` |
| Witt vector ring (parametrizes cyclic pⁿ extensions) | `WittRing`, `FrobeniusMap`, `FrobeniusImage`, `VerschiebungMap`, `VerschiebungImage`, `TeichmuellerSystem`, `LocalRing` |
| Twisted polynomial ring (endomorphism ring / additive polynomials) | `TwistedPolynomials`, `AdditivePolynomialFromRoots`, `Quotrem`, `GCD`, `Polynomial`, `SpecialEvaluate` |
| Carlitz module (rank-1 Drinfeld module for rational function field) | `CarlitzModule` |
| Drinfeld module of rank 1 (analytic class field theory for function fields) | `AnalyticDrinfeldModule`, `Extend`, `Exp`, `AnalyticModule`, `CanNormalize`, `CanSignNormalize`, `AlgebraicToAnalytic` |
| Strong approximation theorem | `StrongApproximation` |
| Place enumeration | `PlaceEnumInit`, `PlaceEnumCopy`, `PlaceEnumPosition`, `PlaceEnumNext`, `PlaceEnumCurrent` |
