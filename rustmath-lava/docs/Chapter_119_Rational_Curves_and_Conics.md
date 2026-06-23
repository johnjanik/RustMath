# Chapter 119 — Rational Curves and Conics

**Handbook part:** XVI — Arithmetic Geometry
**Handbook pages:** 3913–3934 (PDF pages 4044–4067)

---

## Scope and overview

Chapter 119 describes the specialised Magma categories for nonsingular plane curves of genus zero: **conics** (`CrvCon`, degree 2) and **rational curves** (`CrvRat`, degree 1). The unifying theme is the birational classification and parametrisation of genus zero curves.

The central theoretical fact is that for a genus zero curve C, the canonical divisor K_C has degree −2, so the Riemann–Roch space of the effective divisor −K_C has dimension 3 and gives an anti-canonical embedding of C in P² as a **conic**. If C has a rational point then the Riemann–Roch space of that point's divisor gives a birational isomorphism with the projective line, i.e. a **parametrisation**.

For conics over **Q** the chapter is organised around two complementary algorithms of **D. Simon [Sim05]**: a minimisation stage that reduces the discriminant of the defining matrix prime-by-prime, followed by an indefinite LLL step that reduces it to a unimodular diagonal matrix equivalent to x² + y² − z² = 0, whose Pythagorean parametrisation is then pulled back to the original conic. Over number fields a variant of **Lagrange's method** (lattice-reduction in two copies of the base field) is used instead. Over rational function fields the algorithm is due to **Cremona and van Hoeij [CR06]**. Over finite fields a simple random-x search is used.

The obstruction to having a rational point — and hence to parametrising by P¹ — is measured by the **bad (ramified) primes** and encoded in an isomorphism class of **quaternion algebras**. This quaternion algebra connection underpins the automorphism-group and isomorphism-classification algorithms in the final sections of the chapter. **Point reduction** uses a variant of Mordell's reduction due to Cremona [CR03] to produce a point satisfying **Holzer's bounds**.

---

## 119.1 Introduction

The chapter introduces the two specialised curve types (`CrvCon`, `CrvRat`), explains the Riemann–Roch basis for the anti-canonical embedding, describes the role of the quaternion algebra associated to a conic (used for automorphisms and isomorphism classification), and identifies the chapter's main algorithmic tools: Simon's algorithm [Sim05] for point-finding and parametrisation over Q, and associated reduction algorithms.

*No intrinsics are introduced in this section.*

---

## 119.2 Rational Curves and Conics

The general curve tools from Chapter 114 are inherited; this section adds the specialised constructors and type-testing functions for `CrvCon` and `CrvRat`, and documents the main parametrisation entry points.

### 119.2.1 Rational Curve and Conic Creation

Nonsingularity is equivalent to absolute irreducibility for conics and imposes no condition on a linear equation in the plane. Both `Conic` and `RationalCurve` accept an optional `Ambient` parameter to specify the ambient projective space; otherwise a new ambient is created.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Conic(coeffs)` | Creates a conic from a sequence of 3 or 6 coefficients. `[a, b, c]` → aX² + bY² + cZ²; `[a, b, c, d, e, f]` → aX² + bY² + cZ² + dXY + eYZ + fXZ. Optional parameter `Ambient` specifies the ambient projective space. | Direct construction. |
| `Conic(M)` | Creates a conic from a symmetric 3×3 matrix M; the equation is [X, Y, Z] M [X, Y, Z]ᵀ. Optional parameter `Ambient`. | Direct construction from matrix. |
| `Conic(X, f)` | Returns the conic defined by the polynomial f in the projective plane X. | — |
| `IsConic(S)` | Returns `true` iff the scheme S is a nonsingular plane curve of degree 2; if so, also returns a curve of type `CrvCon` with the same defining polynomial. | Degree/nonsingularity check. |
| `RationalCurve(X, f)` | Returns the rational curve defined by the linear polynomial f in the projective plane X. | — |
| `IsRationalCurve(S)` | Returns `true` iff S is defined by a linear polynomial in some P²; if so, returns a curve of type `CrvRat` with the same defining polynomial. | Linearity/projective-plane check. |

*Worked examples: H119E1 (creating a conic from a degree-2 curve via `IsConic`; illustrating `Type`, `AmbientSpace`, `DefiningIdeal`).*

### 119.2.2 Access Functions

The basic access functions are inherited from the general machinery for plane curves and hypersurface schemes.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DefiningPolynomial(C)` | Returns the defining polynomial of the conic or rational curve C. | Inherited from plane curves. |
| `DefiningIdeal(C)` | Returns the defining ideal of the conic or rational curve C. | Inherited from plane curves. |
| `BaseRing(C)` / `BaseField(C)` | Returns the base ring of the curve C. | — |
| `Category(C)` / `Type(C)` | Returns the category `CrvRat` or `CrvCon`; both are special subtypes of `CrvPln`, which are subtypes of `Crv`. | — |

### 119.2.3 Rational Curve and Conic Examples

*Worked examples: H119E2 (highly singular geometric-genus-zero curve over GF(71); `ArithmeticGenus`, `Genus`, `SingularSubscheme`, `IrreducibleComponents`, `Parametrization` from a nonsingular point); H119E3 (nonsingular conic over Q with large coefficients; `HasRationalPoint`, `RationalPoint`, `Parametrization`, `RationalPoints` naive search with `Bound`); H119E4 (diagonal models via `LegendrePolynomial`, `ReducedLegendrePolynomial`; `BaseExtend` to a number field).*

---

## 119.3 Conics

### 119.3.1 Elementary Invariants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Discriminant(C)` | Returns the discriminant of conic C. For equation a₁₁x² + a₁₂xy + a₁₃xz + a₂₂y² + a₂₃yz + a₃₃z² = 0, the discriminant is 4a₁₁a₂₂a₃₃ − a₁₁a²₂₃ − a²₁₂a₃₃ + a₁₂a₁₃a₂₃ − a²₁₃a₂₂. Over rings where 2 is invertible this equals 1/2 times the determinant of the associated symmetric matrix. | Direct formula. |

### 119.3.2 Alternative Defining Polynomials

Curves over Q compute and store a diagonalised Legendre model whose defining polynomial is accessible here.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LegendrePolynomial(C)` | Returns the Legendre polynomial of C — a diagonalised defining polynomial of the form ax² + by² + cz². Once computed, it is stored as an attribute. The transformation matrix defining the isomorphism from C to the Legendre model is returned as the second value. | Diagonalisation by completing the square / congruence reduction. |
| `ReducedLegendrePolynomial(C)` | Returns the reduced Legendre polynomial of C (C must be over Q or Z) — a diagonalised integral polynomial with pairwise coprime, square-free coefficients. The transformation matrix is returned as the second value. | Reduction of Legendre form by removing squares and cross-prime factors. |

### 119.3.3 Alternative Models

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LegendreModel(C)` | Returns the Legendre model of C — an isomorphic curve of the form ax² + by² + cz² = 0 — together with an isomorphism to this model. | — |
| `ReducedLegendreModel(C)` | Returns the reduced Legendre model of C (C must be over Q or Z) — a curve in diagonal form ax² + by² + cz² = 0 with pairwise coprime, square-free coefficients — together with the isomorphism from C to this model. | — |

### 119.3.4 Other Functions on Conics

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `MinimalModel(C)` | Returns a conic whose defining-polynomial matrix has smaller discriminant than that of C (where possible), plus a map from the new conic to C. | Minimisation stage of Simon's algorithm **[Sim05]**, as used in `HasRationalPoint`. |

*Worked examples: H119E5 (reducing a conic at 13 via `MinimalModel`; `BadPrimes`, `Discriminant`, `Factorization`).*

---

## 119.4 Local-Global Correspondence

The **Hasse–Minkowski principle** implies a conic over a number field has a point over the number field if and only if it has a point over every completion (finite and infinite prime). Only the finitely many primes dividing the discriminant need to be checked; Hensel's lemma reduces this to a finite computation. The algorithms implemented currently treat only **Q**.

### 119.4.1 Local Conditions for Conics

A prime p is **ramified (bad)** for a conic C if there is no p-integral model with nonsingular reduction. Every such prime divides the Legendre polynomial's coefficients. The parity of the number of bad finite primes determines ramification at infinity (the total count including infinity must be even).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BadPrimes(C)` | Given a conic C over Q, returns the sequence of finite ramified primes — those at which C has intrinsic locally singular reduction. The parity of the sequence length encodes ramification at the infinite prime (not included in the returned sequence). | Quaternion algebra ramification. |

### 119.4.2 Norm Residue Symbol

Hilbert's norm residue symbol gives a precise condition for a quadratic form to represent zero over Q_p; this is equivalent to the condition that a conic has a local point. The theory is treated in Cassels [Cas78] and Lam [Lam73].

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `NormResidueSymbol(a, b, p)` | Given rational numbers or integers a, b and a prime p, returns 1 if the quadratic form ax² + by² − z² represents zero over Q_p and −1 otherwise. | Classical norm residue symbol computation; see **[Cas78]**. |
| `HilbertSymbol(a, b, p : parameters)` | Computes the Hilbert symbol (a, b)_p, where a, b are elements of a number field and p is either a prime number (if a, b ∈ Q) or a prime ideal. Parameter `Al` (`MonStgElt`, default `"NormResidueSymbol"`): by default uses `NormResidueSymbol`; set to `"Evaluate"` to use the number-field algorithm instead. | `"NormResidueSymbol"`: see **[Cas78]**, **[Lam73]**; `"Evaluate"`: general number-field algorithm. |

*Worked examples: H119E6 (using `NormResidueSymbol` to identify bad primes of a diagonal conic; confirming against `BadPrimes`).*

---

## 119.5 Rational Points on Conics

Functions for deciding solubility and finding points on conics over: **Q** (or Z), **finite fields**, **number fields**, and **rational function fields in odd characteristic**. When a point is found it is cached for later use.

**Over Q/Z** — Simon's algorithm [Sim05]: works with the symmetric matrix of the conic, computes transformations reducing the determinant prime-by-prime (dividing the discriminant), then applies an indefinite LLL to produce a unimodular integral diagonal matrix equivalent to x² + y² − z² = 0, whose Pythagorean parametrisation is pulled back.

**Over number fields** — a variant of Lagrange's method: reduce to diagonal form, then iterate a lattice-reduction step (short vector in a lattice defined by local congruence conditions inside two copies of the base field). Often a solution is found by easy search after reduction; otherwise `NormEquation` is called on the reduced conic. The class group may assist further reduction.

**Over rational function fields** — algorithm of Cremona and van Hoeij **[CR06]**, contributed by John Cremona and David Roberts.

**Over finite fields** — random-x search for a point (x : y : 1).

### 119.5.1 Finding Points

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HasRationalPoint(C)` | C must be defined over Z, Q, a finite field, a number field, or a rational function field over a finite field of odd characteristic. Returns `true` iff a point on C exists, and if so also returns one such point. | Over Q/Z: Simon **[Sim05]**; over number fields: Lagrange-variant lattice reduction; over function fields: **[CR06]**; over finite fields: random-x search. |
| `RationalPoint(C)` | Same base field requirements as `HasRationalPoint`. Returns a rational point on C over its base ring; errors if no such point exists. | Same as `HasRationalPoint`. |
| `Random(C : parameters)` | Returns a randomly selected rational point of C. Parameter `Bound` (`RngIntElt`, default 10⁹): upper bound on the random integers fed to the parametrisation. Parameter `Reduce` (`BoolElt`, default `false`): if `true`, applies point reduction (§119.5.2) to the result. | Parametrisation evaluation at a random parameter value. |
| `Points(C : parameters)` / `RationalPoints(C : parameters)` | For C over Q or a finite field: returns an indexed set of rational points. Over Q, parameter `Bound` (`RngIntElt`) must be given; returns points whose integral coordinates on the reduced Legendre model are bounded by `Bound`. | Naive bounded search on the reduced Legendre model (over Q); enumeration (over finite fields). |

*Worked examples: H119E7 (three conics in Legendre form with same discriminant primes; naive `RationalPoints` search; `BadPrimes` proving no-point; discussion of parametrisation alternative); H119E9 (full workflow: `HasRationalPoint`, `RationalPoint`, `Random` with `Reduce`, reduction testing, parametrisation and reduction composition).*

### 119.5.2 Point Reduction

**Holzer's theorem**: if a Legendre-form conic ax² + by² + cz² = 0 over Q has a point, then there exists an integer point (x : y : z) with |x| ≤ √|bc|, |y| ≤ √|ac|, |z| ≤ √|ab| (equivalently, max(|ax²|, |by²|, |cz²|) ≤ |abc|). Such a point is called **Holzer-reduced**. The Magma implementation uses a variant of Mordell's reduction due to **Cremona [CR03]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsReduced(p)` | Returns `true` iff the projective point p on a conic satisfies Holzer's bounds on the reduced Legendre model. If C is not already a reduced Legendre model, the test is done after passing to that model. | Holzer bound check; see **[CR03]**. |
| `Reduction(p)` | Returns a Holzer-reduced point derived from p. | Mordell-variant reduction algorithm of Cremona **[CR03]**. |

*Worked examples: H119E8 (small reduction example; `ReducedLegendreModel`, `IsReduced`, `Reduction`, pullback via `@@`); H119E9 (reduction in a fuller workflow; see §119.5.1).*

---

## 119.6 Isomorphisms

### 119.6.1 Isomorphisms with Standard Models

This section covers isomorphisms between heterogeneous types — `Crv`, `CrvCon`, `CrvRat` — and parametrisations by a projective line.

The key idea: the 2-uple embedding φ: P¹ → P² via (u : v) ↦ (u² : uv : v²) gives an isomorphism of P¹ with the standard conic C₀: y² = xz. An isomorphism of any conic C₁ with P¹ is determined by a change of variables mapping C₀ onto C₁; this **parametrisation matrix** is stored with C₁ once a rational point is found.

`Parametrization` requires its domain P to be given (or created) as a **curve** rather than as an ambient space, to enable direct pullback/push-forward functionality.

`ParametrizeRationalNormalCurve` applies to non-singular rational curves of degree d in ordinary d-dimensional projective space (d ≥ 1): if d is odd it returns an isomorphism from P¹ to C; if d is even it returns an isomorphism from a plane conic to C. It uses no function field machinery.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Conic(C)` | Given a curve C of genus zero, returns a conic determined by the anti-canonical embedding (Riemann–Roch space of −K_C, dimension 3, degree 2 functions). | Anti-canonical embedding via Riemann–Roch; see **[Sim05]** for the Q case. |
| `ParametrizationMatrix(C)` | Optimised routine for C defined over Z or Q. Returns a 3×3 matrix M such that for a point (x₀ : y₀ : z₀) on C, the point (x₁ : y₁ : z₁) = (x₀ : y₀ : z₀)M satisfies y₁² = x₁z₁ (i.e. lies on the standard conic). Action is on the right, consistently with Magma's scheme-map convention. | Parametrisation via 2-uple embedding and change of variables; Simon **[Sim05]**. |
| `Parametrization(C)` / `Parametrization(C, P)` / `Parametrization(C, p)` / `Parametrization(C, p, P)` | Returns an isomorphism of schemes P → C (P a projective line, optionally specified; a rational point or place p may be specified to pin the parametrisation). If no p is given, the base field must be one of the types accepted by `HasRationalPoint`. Errors if C has no rational points. | Conic parametrisation algorithm: 2-uple embedding composed with parametrisation matrix; over Q uses Simon **[Sim05]**. |
| `ParametrizeOrdinaryCurve(C)` / `ParametrizeOrdinaryCurve(C, p)` / `ParametrizeOrdinaryCurve(C, p, I)` | As `Parametrization`, but for plane curves with only ordinary singularities (see §114.3.6); uses a more specialised procedure that is faster and can produce nicer parametrisations. The optional argument I is the adjoint ideal of C (avoids recomputation if already known). | Adjoint ideal / ordinary-singularity method (faster than general function field machinery). |
| `ParametrizeRationalNormalCurve(C)` | For non-singular rational curves of degree d in ordinary d-dimensional projective space (d ≥ 1): returns an isomorphism from P¹ (d odd) or a plane conic (d even) to C. Irreducibility of C is not checked. | Adjoint maps; no function field machinery. |

*Worked examples: H119E10 (`Conic` applied to the singular genus-zero curve of H119E2 over GF(71); anti-canonical model); H119E11 (`ParametrizationMatrix` verification — explicit change of variables to y² = xz, right-action convention); H119E12 (parametrising a degree-7 plane curve over GF(101) with `Parametrization` from a nonsingular point; `Image`, `DefiningIdeal` equality).*

---

## 119.7 Automorphisms

### 119.7.1 Automorphisms of Rational Curves

Automorphisms of P¹ are 3-transitive: for any three distinct points p₀, p₁, p∞ there is a unique automorphism sending 0 = (0:1), 1 = (1:1), ∞ = (1:0) to p₀, p₁, p∞. This characterisation defines automorphisms of rational curves.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Automorphism(C, S, T)` | Given a rational curve C and two indexed sets S = {p₀, p₁, p∞} and T = {q₀, q₁, q∞} of distinct points over the base ring, returns the unique automorphism of C taking pᵢ to qᵢ. | 3-transitivity of Aut(P¹); cross-ratio / Möbius transformation. |

### 119.7.2 Automorphisms of Conics

The automorphism group of a conic C/K for a Legendre equation ax² + by² + cz² = 0 (abc ≠ 0) is isomorphic to the projective unit group A*/K* of the **quaternion algebra** A over K with generators i, j, k = c⁻¹ij satisfying i² = −bc, j² = −ac, k² = −ab, acting on the trace-zero part A⁰_L by conjugation. The isomorphism of two conics is equivalent to the isomorphism of their quaternion algebras; a rational parametrisation is equivalent to an isomorphism A ≅ M₂(K). The current implementation requires characteristic ≠ 2 (Legendre model does not exist in characteristic 2). See Lam [Lam73] and Vignéras [Vig80].

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `QuaternionAlgebra(C)` | Returns the quaternion algebra in which automorphisms of the conic C can be represented. | Quaternion algebra construction from Legendre coefficients; see **[Lam73]**, **[Vig80]**. |
| `Automorphism(C, a)` | Given a conic C and a unit a of the quaternion algebra associated to C, returns the automorphism of C corresponding to a (conjugation action on the trace-zero part). | Quaternion conjugation on A⁰; see **[Lam73]**, **[Vig80]**. |

*Worked examples: H119E13 (conic with no rational Q-point; `QuaternionAlgebra`, `RamifiedPrimes`, `MaximalOrder`, `Basis`; constructing three automorphisms; pure-quaternion elements give involutions; generating points over Q(√74) via automorphism action on a known quadratic-extension point).*

---

## 119.8 Bibliography

| Key | Reference |
|-----|-----------|
| **[Cas78]** | J. W. S. Cassels. *Rational Quadratic Forms.* Academic Press, London–New York–San Francisco, 1978. |
| **[CR03]** | J. E. Cremona and D. Rusin. Efficient solution of rational conics. *Mathematics of Computation*, 72(243):1417–1441, 2003. |
| **[CR06]** | J. E. Cremona and D. Rusin. Solving conics over function fields. *Journal de Théorie des Nombres de Bordeaux*, 18:595–606, 2006. |
| **[Lam73]** | T. Y. Lam. *The Algebraic Theory of Quadratic Forms.* W. A. Benjamin, Inc., Reading, MA, 1973. |
| **[Sim05]** | Denis Simon. Solving quadratic equations using reduced unimodular quadratic forms. *Math. Comp.*, 74(251):1531–1543 (electronic), 2005. |
| **[Vig80]** | M.-F. Vignéras. *Arithmétique des Algèbres de Quaternions*, volume 800 of Lecture Notes in Mathematics. Springer-Verlag, Berlin, 1980. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Simon's algorithm (point-finding, minimisation, parametrisation over Q/Z) **[Sim05]** | `HasRationalPoint`, `RationalPoint`, `MinimalModel`, `ParametrizationMatrix`, `Parametrization`, `Conic(C)` |
| Lagrange-variant lattice reduction (points over number fields) | `HasRationalPoint`, `RationalPoint` |
| Cremona–van Hoeij (conics over function fields) **[CR06]** | `HasRationalPoint`, `RationalPoint` |
| Holzer / Mordell–Cremona point reduction **[CR03]** | `IsReduced`, `Reduction`, `Random(:Reduce)` |
| Hasse–Minkowski / norm residue symbol **[Cas78, Lam73]** | `NormResidueSymbol`, `HilbertSymbol`, `BadPrimes` |
| Diagonalisation / Legendre models | `LegendrePolynomial`, `ReducedLegendrePolynomial`, `LegendreModel`, `ReducedLegendreModel` |
| Anti-canonical embedding (Riemann–Roch) | `Conic(C)` |
| 2-uple embedding / conic parametrisation matrix | `ParametrizationMatrix`, `Parametrization` |
| Adjoint ideal / ordinary singularity parametrisation | `ParametrizeOrdinaryCurve` |
| Adjoint maps (rational normal curves) | `ParametrizeRationalNormalCurve` |
| Quaternion algebra / automorphism group **[Lam73, Vig80]** | `QuaternionAlgebra`, `Automorphism(C, a)`, `BadPrimes` |
| 3-transitivity of Aut(P¹) | `Automorphism(C, S, T)` |
