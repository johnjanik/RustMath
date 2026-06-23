# Chapter 124 — Models of Genus One Curves

**Handbook part:** XVI — Arithmetic Geometry
**Handbook pages:** 4103–4117 (PDF pages 4234–4251)

---

## Scope and overview

This chapter deals with curves of genus one given by equations in particular normal forms.
The primary type is `ModelG1`, which is not a subtype of any curve or scheme type in
Magma. Functionality covers invariant theory of these models and applications to arithmetic
problems concerning genus one curves over number fields.

Geometrically (over an algebraically closed field), a genus one model of degree n is an
elliptic curve embedded in P^(n-1) via the linear system |n.O|. Over a general field, a
genus one model of degree n is a principal homogeneous space for an elliptic curve (of
order n in the Weil–Chatelet group) embedded in P^(n-1) analogously. Such models are
sometimes called genus one normal curves. Not every element of order n in the
Weil–Chatelet group admits such an embedding, though it does if it is everywhere locally
soluble.

The degree n may be 2, 3, 4, or 5. The defining data is:

- **Degree 2:** a binary quartic g(x, z) (model without cross terms), or more generally
  y^2 + f(x,z)y − g(x,z) with f, g homogeneous of degrees 2 and 4 (weights 1, 1, 2).
- **Degree 3:** a cubic form in 3 variables (projective plane cubic).
- **Degree 4:** a pair of homogeneous degree-2 polynomials in 4 variables (intersection
  of two quadrics in P^3). This is the standard form returned by `FourDescent`.
- **Degree 5:** a 5×5 alternating matrix of linear forms in 5 variables; the associated
  subscheme of P^4 is cut out by the 4×4 Pfaffians. Every genus one normal curve of
  degree 5 arises in this way.

Degenerate cases are allowed: the scheme associated to a genus one model is not always
a smooth curve of genus 1.

---

## 124.1 Introduction

*(Introductory section; no intrinsics. See Scope and overview above.)*

---

## 124.2 Related Functionality

*(Cross-reference section; no intrinsics.)*

Section 122.2.11 (three descent on elliptic curves) is relevant for rational points on plane
cubics. A very efficient point search for schemes (Section 112.13.4) finds all rational points
up to absolute height H in time O(H^(2/d)), where d is the dimension of the ambient
projective space.

---

## 124.3 Creation of Genus One Models

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `GenusOneModel(n, seq)` / `GenusOneModel(seq)` / `GenusOneModel(n, str)` | Genus one model of degree n (2, 3, 4, or 5) from the given coefficient sequence or string. Length-5 seq → degree 2 binary quartic; length-8 → degree 2 with cross terms; length-10 → degree 3 cubic; lengths 20 or 50 → degrees 4 or 5 respectively. Coefficients may belong to any ring. | Direct coefficient assignment; recover via `Eltseq`. |
| `GenusOneModel(C)` | Genus one model representing the given curve C. For degree 2, C should be a subscheme of weighted projective space P(1,1,2) or a hyperelliptic curve. For n = 3, 4, 5, C should be a genus one normal curve of the corresponding degree. | — |
| `GenusOneModel(f)` / `GenusOneModel(seq)` | Genus one model given by the polynomial f or sequence of equations seq. | — |
| `GenusOneModel(n, E)` | Genus one model of degree n (2, 3, 4, or 5) representing elliptic curve E embedded in P^(n-1) via |n.O|. Also returns the image curve C together with maps E → C and C → E. | Linear system embedding. |
| `GenusOneModel(mat)` | Genus one model of degree 5 from the given 5×5 matrix. | — |
| `GenusOneModel(mats)` | Genus one model of degree 4 from a pair of 4×4 symmetric matrices in the sequence mats. (Recover matrices via `ModelToMatrices`.) | — |
| `CompleteTheSquare(model)` | Given a degree 2 genus one model, returns a simplified degree 2 model without cross terms by completing the square on the defining polynomial. | Completing the square. |
| `RandomGenusOneModel(n)` / `RandomModel(n)` | A random genus one model of degree n (2, 3, 4, or 5). Parameter: `Size` (RngIntElt, default unspecified) controls coefficient size. | Random generation. |
| `GenericModel(n)` | The generic genus one model of degree n (2, 3, 4, or 5), with indeterminate coefficients in a suitable polynomial ring. | — |
| `ChangeRing(model, B)` | Genus one model defined over ring B, obtained by coercing the coefficients of the given model into B. | — |
| `CubicFromPoint(E, P)` | The 3-covering corresponding to the rational point P on elliptic curve E. Returns the projective plane cubic equation, the covering map, and a point mapping to P under the covering. | — |
| `HesseModel(n, seq)` | Genus one model of degree n invariant under the standard representation of the Heisenberg group. Second argument is a sequence of two ring elements. | — |
| `DiagonalModel(n, seq)` | Genus one model of degree n invariant under the diagonal action of µ_n. Second argument is a sequence of n ring elements. | — |

*Worked example: H124E1 — Constructs the degree 5 genus one model of the generic elliptic curve E_{a,b} : y^2 = x^3 + ax + b over Q(a,b) via `GenusOneModel(5, Eab)`, displaying the alternating matrix, computing `Equations` (the 4×4 Pfaffians), and verifying that the model's invariants c4, c6, Δ match those of E_{a,b}.*

---

## 124.4 Predicates on Genus One Models

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsGenusOneModel(f)` / `IsGenusOneModel(seq)` / `IsGenusOneModel(mat)` | Returns `true` iff the given polynomial, sequence of polynomials, or matrix determines a genus one model in the sense of this chapter. When true, also returns the model. Note: does not imply the associated scheme is a smooth curve of genus 1, as degenerate cases are allowed. | Structural check on polynomial/matrix form. |
| `IsEquivalent(model1, model2)` / `IsEquivalent(cubic1, cubic2)` | Returns `true` iff the two cubics (or genus one models of degree 3) are equivalent as genus one models, i.e. there exists a linear transformation of the ambient P^2_Q taking one to the other up to scaling. When true, also returns the transformation tuple. | Algorithm of **[Fis06]**. |

---

## 124.5 Access Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Degree(model)` | The degree (2, 3, 4, or 5) of the given model. | — |
| `DefiningEquations(model)` | Sequence of defining equations for a model of degree 2, 3, or 4. | — |
| `Equations(model)` | Sequence of equations for the scheme associated to the given model (any degree). For degrees 2, 3, 4 this agrees with `DefiningEquations`. For degree 5 these are the 4×4 Pfaffians of the matrix. | — |
| `Matrix(model)` | The defining 5×5 matrix of a degree 5 genus one model. | — |
| `Curve(model)` / `HyperellipticCurve(model)` / `QuadricIntersection(model)` | The curve associated to the genus one model. For degree 2: `Curve` returns the curve in weighted projective space (y^2 + f(x,z)y − g(x,z) = 0); `HyperellipticCurve` creates it explicitly as a hyperelliptic curve. For degree 4: the intersection of two quadrics in P^3. Errors in degenerate cases. | — |
| `Matrices(model)` | For a degree 4 model: a sequence containing two 4×4 symmetric matrices representing the two quadrics. | — |
| `BaseRing(model)` | The coefficient ring of the given model. | — |
| `PolynomialRing(model)` | The polynomial ring used to define the model. | — |
| `Eltseq(model)` / `ModelToString(model)` | A sequence (resp. string) containing the coefficients of the defining data (polynomial, pair of polynomials, or matrix). The model may be recovered via `GenusOneModel`. | — |

---

## 124.6 Minimisation and Reduction

Minimisation and reduction compute simpler integral global models of genus one curves.
Throughout, "model" means a global integral model.

- **Minimisation**: computes a model isomorphic to the original over its ground field but
  possibly not integrally equivalent. A model is minimal if it is locally minimal at all
  primes; locally minimal means the valuation of its discriminant is as small as possible
  among all integral models. For locally solvable models this minimal discriminant equals
  that of the associated elliptic curve.
- **Reduction**: computes a model integrally equivalent to the original (transformations in
  both directions over the ring of integers). Informally, a reduced model has small-height
  coefficients. One precise formulation attaches to each model an invariant in a symmetric
  space; the model is reduced iff its invariant lies in a fixed fundamental domain. This
  also provides the algorithmic basis for reduction.

Algorithms for degrees 2, 3, 4 over Q are described in **[CFS10]**. Degree 5 uses
additional techniques by Fisher. Degree 2 over number fields uses further techniques by
Donnelly and Fisher.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Minimise(model : parameters)` | Given a genus one model of degree 2, 3, 4, or 5 over Q, returns a minimal model. Also implemented for degree 2 without cross terms (returns minimal among cross-term-free models, or nearly minimal with a small extra discriminant factor when class group obstructions arise; primes for the extra factor specifiable via `ClassGroupPrimes`). Returns the minimal model, the transformation taking original to minimal (unless `Transformation := false`), and the set of primes of positive level (primes where the model is not soluble over Q^nr_p, except p=2 for degree 2 when `CrossTerms := false`). The degree 5 routine is not yet proven to work in all cases. Parameters: `Transformation` (BoolElt, default `true`), `CrossTerms` (BoolElt, default `false`), `UsePrimes` (SeqEnum, default `[]`), `ClassGroupPrimes` (SeqEnum, default `[]`), `Verbose Minimise` (maximum 3). | **[CFS10]** for degrees 2–4; Fisher's techniques for degree 5; Donnelly–Fisher for degree 2 over number fields. |
| `Minimise(f)` | Given the equation f of a nonsingular projective plane cubic, returns the minimal model equation and a transformation tuple. | **[CFS10]**. |
| `pMinimise(f, p)` | Given the equation f of a nonsingular projective plane cubic, returns a model minimal at prime p, plus a matrix M giving the transformation (the minimised cubic is the original evaluated at M[x,y,z]^tr, up to scaling). | Local minimisation at p; **[CFS10]**. |
| `Reduce(model)` / `Reduce(f)` | Given a genus one model of degree 2, 3, or 4 over Q (or the appropriate polynomial f, either homogeneous for degrees 2–3, or univariate for degree 2 without cross terms), returns a reduced model and the transformation taking original to reduced. `Verbose Reduce` (maximum 3). | Symmetric-space fundamental-domain method; **[CFS10]**. |
| `ReduceQuadrics(seq)` | Computes a reduced basis for the space spanned by the given quadrics (sequence of homogeneous quadratic forms in x1,…,xn). Returns the reduced forms, a matrix S (change of homogeneous coordinates: substitute [x1,…,xn].S for [x1,…,xn]), and a matrix T (change of basis of forms, acting from the left). The current implementation is not optimal. `Verbose Reduce` (maximum 3). | LLL-style coordinate and basis reduction (heuristic). |

---

## 124.7 Genus One Models as Coverings

The curve defined by a genus one model of degree n is a principal homogeneous space for
its Jacobian (an elliptic curve). The Jacobian and the degree n^2 covering map can be read
from the invariants and covariants of the model. Two models with the same Jacobian can
be added as elements of the Weil–Chatelet group.

Related: `ThreeSelmerElement` (Section 122.2.11) for degree 3; `AssociatedEllipticCurve`
and `AssociatedHyperellipticCurve` from the four descent package (Section 122.2.9) for
degree 4.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Jacobian(model)` / `Jacobian(C)` | The Jacobian, as an elliptic curve, of the given genus one model or of the curve C corresponding to a genus one model. | Invariant-theoretic formula. |
| `nCovering(model : parameters)` | The covering map from the given model to its Jacobian. Returns: the degree-n curve C, its Jacobian as an elliptic curve E, and a map of schemes C → E. Parameter: `E` (CrvEll, default unspecified) — if supplied, must be isomorphic to the Jacobian, and will be used as the target. | Covariant formula for the covering map. |
| `AddCubics(cubic1, cubic2 : parameters)` / `AddCubics(model1, model2 : parameters)` / `model1 + model2` | Given two ternary cubic polynomials (or two degree 3 genus one models) with the same invariants, returns the sum of the corresponding elements of H^1(Q, E[3]). Errors if the two cubics do not belong to the same elliptic curve E. Parameters: `E` (CrvEll), `ReturnBoth` (BoolElt, default `false`). See also Section 122.2.11. | Group law in the Weil–Chatelet group via invariant theory. |
| `DoubleGenusOneModel(model)` | Given a genus one model of degree 4 or 5, computes twice the associated element in the Weil–Chatelet group and returns it as a genus one model (degree 2 or 5 respectively). | Doubling in the Weil–Chatelet group. |
| `FourToTwoCovering(model : parameters)` / `FourToTwoCovering(C : parameters)` | Given a degree 4 genus one model (or its associated curve C), returns three values: the curve C4 in P^3, a plane quartic C2 representing twice the model in the Weil–Chatelet group, and the map of schemes C4 → C2. Parameter: `C2` (Crv, default unspecified). Equivalent to calling `AssociatedHyperellipticCurve(Curve(model))`. | Doubling map; degree-4 specialisation. |

---

## 124.8 Families of Elliptic Curves with Prescribed n-Torsion

Rubin and Silverberg **[RS95]** explicitly construct families of elliptic curves over Q with
the same Galois representation on the n-torsion subgroup as a given elliptic curve.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RubinSilverbergPolynomials(n, J : parameters)` | For n = 2, 3, 4, or 5, and E : y^2 = x^3 + ax + b with j-invariant 1728J, returns polynomials α(t) and β(t) such that every nonsingular member F_t of the family F : y^2 = x^3 + aα(t)x + bβ(t) satisfies F_t[n] ≅ E[n] as Z[G]-modules (with G the absolute Galois group of Q), and the isomorphisms preserve the Weil pairing. For n = 3, 4, 5, all n-congruent curves belong to the same family. Parameter: `Parameter` (RngElt, default unspecified) — a specific parameter value. | Construction of **[RS95]**; closely related to `HessePolynomials` for r ≡ 1 (mod n). |

---

## 124.9 Transformations between Genus One Models

A transformation between two genus one models of degree n is a tuple of two elements.
The second element is an n×n matrix acting on projective coordinates from the right. The
first element is a rescaling (matrix or scalar):

- **Degree 2 (no cross terms) / Degree 3:** tuple ⟨k, S⟩ with k a ring element. Apply
  by substituting coordinates via S and multiplying the equation by k.
- **Degree 2 (with cross terms):** tuple ⟨k, [A, B, C], S⟩; additionally substitutes
  y + Ax^2 + Bxz + Cz^2 for y.
- **Degree 4:** first element is a 2×2 matrix acting on the two quadric equations from
  the left; second element is the 4×4 coordinate matrix.
- **Degree 5:** tuple ⟨T, S⟩ with T and S both 5×5 matrices; the transformed model is
  TMS^T^tr where MS is obtained from M by substituting coordinates via S.

Equivalent models have the same invariants up to scaling by the 4th and 6th powers of
ScalingFactor.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsTransformation(n, g)` | Returns `true` iff the tuple g represents a valid transformation between genus one models of degree n. | Structural check. |
| `RandomTransformation(n : parameters)` | A random transformation between genus one models of degree n. Parameters: `Size` (RngIntElt, default 5, passed to `RandomSL` or `RandomGL`), `Unimodular` (BoolElt, default `false`; when true the transformation is integrally invertible), `CrossTerms` (BoolElt, default `false`; when false in degree 2, preserves the set of cross-term-free models). | Random matrix generation. |
| `g * model` / `ApplyTransformation(g, model)` | Applies the transformation g to the genus one model. | Substitution and rescaling. |
| `g1 * g2` / `ComposeTransformations(g1, g2)` | Composition g1 ∗ g2 of two genus one model transformations. Transformations act on the left: (g1 ∗ g2) ∗ f = g1 ∗ (g2 ∗ f). | Left-composition. |
| `ScalingFactor(g)` | The scaling factor λ of a transformation g: if the original model has invariants c4 and c6, the transformed model has invariants λ^4 c4 and λ^6 c6. | — |

---

## 124.10 Invariants for Genus One Models

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `aInvariants(model)` | The invariants [a1, a2, a3, a4, a6] of a genus one model of degree 2, 3, or 4. Degree 3 formulae from **[ARVT05]**. | Classical Weierstrass invariant formulae; **[ARVT05]** for degree 3. |
| `bInvariants(model)` | The invariants [b2, b4, b6, b8] for a model of degree 2, 3, or 4. Computed from `aInvariants` in the standard way (as for elliptic curves). | Standard transformation from a-invariants. |
| `cInvariants(model)` | The invariants [c4, c6] of the given genus one model. For n = 2, 3, 4: the classical invariants as in **[AKM+01]**. For n = 5: algorithm of **[Fis08]**. | **[AKM+01]** (n = 2, 3, 4); **[Fis08]** (n = 5). |
| `Invariants(model)` | The triple (c4, c6, Δ) — the two classical invariants and the discriminant. | Derived from `cInvariants` and `Discriminant`. |
| `Discriminant(model)` | The discriminant Δ of the given genus one model. | Invariant-theoretic formula. |
| `SL4Invariants(model)` | The SL4-invariants of a genus one model of degree 4. | SL4 invariant theory. |

---

## 124.11 Covariants and Contravariants for Genus One Models

Functions in this section implement the invariant theory developed in **[Fis]**.

Let X_n denote the (affine) space of genus one models of degree n, and X*_n its dual.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Hessian(model)` | The module of covariants X_n → X_n is a free module of rank 2 over the ring of invariants, generated by the identity map and a second covariant called the Hessian. For n = 2 or 3 this is the determinant of the matrix of second partial derivatives. Returns the Hessian of the given model. | Invariant theory of **[Fis]**; standard Hessian for n = 2, 3. |
| `CoveringCovariants(model)` | The covariants defining the covering map from the given model to its Jacobian (same as the defining equations of the `nCovering`). | Covariant formula; **[Fis]**. |
| `Contravariants(model)` | The module of contravariants X_n → X*_n is a free module of rank 2 over the ring of invariants; evaluates the generators P and Q at the given model. | Invariant theory of **[Fis]**. |
| `HesseCovariants(model, r)` | Evaluates a pair of covariants (depending on integer r) at a genus one model of degree prime to r. The pencil spanned by these models is a family of genus one curves invariant under the same representation of the Heisenberg group (universal family above a twist of X(n)). For r ≡ 1 (mod n): identity and Hessian. For r ≡ −1 (mod n): the contravariants. For n = 5, r ≡ 2 (mod 5): values in ∧^2W ⊗ V*; r ≡ 3 (mod 5): values in ∧^2W* ⊗ V (identifying X5 = ∧^2V ⊗ W). | Invariant theory of **[Fis]**. |
| `HessePolynomials(n, r, invariants : parameters)` | The Hesse polynomials D(x,y), c4(x,y), c6(x,y) giving the invariants for the pencil computed by `HesseCovariants`. `RubinSilverbergPolynomials` are closely related for r ≡ 1 (mod n). Parameter: `Variables` ([RngMPolElt], default unspecified). | Invariant theory of **[Fis]**; **[RS95]** for the Rubin–Silverberg connection. |

---

## 124.12 Examples

*Worked example H124E1:* Genus one model of degree 5 for the generic elliptic curve; computes the defining Pfaffians and verifies agreement of invariants c4, c6, Δ.

*Worked example H124E2:* Finding a cubic counterexample to the Hasse principle via "visibility" of Tate–Shafarevich elements **[CM00]**. Starting from a rank-0 curve E (Cremona 4343b1) and a rank-1 auxiliary curve F, uses `ThreeDescent`, `Hessian`, `HessePolynomials`, `Minimise`, `Reduce`, `Jacobian`, `nCovering`, and `Pullback` to produce a locally solvable plane cubic with Jacobian E but no rational points.

---

## 124.13 Bibliography

| Key | Reference |
|-----|-----------|
| **[AKM+01]** | Sang Yook An, Seog Young Kim, David C. Marshall, Susan H. Marshall, William G. McCallum, and Alexander R. Perlis. *Jacobians of genus one curves.* J. Number Theory, 90(2):304–315, 2001. |
| **[ARVT05]** | Michael Artin, Fernando Rodriguez-Villegas, and John Tate. *On the Jacobians of plane cubics.* Adv. Math., 198(1):366–382, 2005. |
| **[CFS10]** | J.E. Cremona, T.A. Fisher, and M. Stoll. *Minimisation and reduction of 2-, 3- and 4-coverings of elliptic curves.* Algebra & Number Theory, 4(6):763–820, 2010. |
| **[CM00]** | John E. Cremona and Barry Mazur. *Visualizing elements in the Shafarevich–Tate group.* Experiment. Math., 9(1):13–28, 2000. |
| **[Fis]** | Tom Fisher. *The Hessian of a genus one curve.* (Preprint/unpublished.) |
| **[Fis06]** | T.A. Fisher. *Testing equivalence of ternary cubics.* In S. Pauli, F. Hess, and M. Pohst, editors, ANTS VII, volume 4076 of LNCS, pages 333–345. Springer-Verlag, 2006. |
| **[Fis08]** | T.A. Fisher. *The invariants of a genus one curve.* Proc. Lond. Math. Soc., 97(3):753–782, 2008. |
| **[RS95]** | K. Rubin and A. Silverberg. *Families of elliptic curves with constant mod p representations.* In Elliptic curves, modular forms, & Fermat's last theorem (Hong Kong, 1993), Ser. Number Theory, I, pages 148–161. Internat. Press, Cambridge, MA, 1995. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Genus one normal curve construction / linear system embedding | `GenusOneModel(n, E)`, `GenericModel`, `HesseModel`, `DiagonalModel` |
| Equivalence testing of ternary cubics **[Fis06]** | `IsEquivalent` |
| Minimisation of 2-, 3-, 4-coverings **[CFS10]** | `Minimise`, `Minimise(f)`, `pMinimise` |
| Reduction (symmetric-space fundamental domain) **[CFS10]** | `Reduce`, `ReduceQuadrics` |
| Invariants (classical, degrees 2–4) **[AKM+01]** | `aInvariants`, `bInvariants`, `cInvariants`, `Invariants`, `Discriminant` |
| Invariants (degree 5) **[Fis08]** | `cInvariants`, `Invariants`, `Discriminant`, `SL4Invariants` |
| Jacobians of plane cubics **[ARVT05]** | `aInvariants` (degree 3), `Jacobian` |
| Covering map / Weil–Chatelet group law | `nCovering`, `AddCubics`, `DoubleGenusOneModel`, `FourToTwoCovering`, `Jacobian` |
| Hessian and covariant/contravariant invariant theory **[Fis]** | `Hessian`, `CoveringCovariants`, `Contravariants`, `HesseCovariants`, `HessePolynomials` |
| Rubin–Silverberg n-torsion families **[RS95]** | `RubinSilverbergPolynomials` |
| Visibility of Tate–Shafarevich elements **[CM00]** | `AddCubics`, `nCovering` (see H124E2) |
