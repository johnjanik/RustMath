# Chapter 137 — Hilbert Modular Forms

**Handbook part:** XVII — Modular Arithmetic Geometry
**Handbook pages:** 4653–4668 (PDF pages 4784–4801)

---

## Scope and overview

Hilbert modular forms generalise classical modular forms by replacing the modular group with a
subgroup of GL₂(**Z**_F), where **Z**_F is the ring of integers of a totally real number field
F. The Magma package (first released in V2.15, December 2008, and under continued development)
computes spaces of **Hilbert cusp forms** of weight k and level Γ₀(N): it efficiently computes
Hecke operators on these spaces, decomposes spaces into newforms, and produces large numbers of
eigenvalues for newforms (at least those of small degree). Atkin–Lehner operators and degeneracy
maps are also provided.

The primary focus is on **parallel weight 2**. Higher-weight spaces (including non-parallel
weight) are handled, but some features are weight-2 only and the main routines are best optimised
there. All levels Γ₀(N) are allowed, and some spaces with nontrivial character are handled. In the
current implementation only cusp forms are supported (full spaces and Eisenstein series are
planned). Standard references for the theory are the books by Freitag **[Fre90]** and Garrett
**[Gar90]**.

**Algorithms (via the Jacquet–Langlands correspondence).** Both implemented algorithms rely on the
Jacquet–Langlands correspondence: the Hecke action on a space of Hilbert cusp forms equals the
Hecke action on a space of automorphic forms on some order in a suitable quaternion algebra (see
**[Hid06]** for the definitions and the correspondence). Let F have degree n over **Q**.

- **Algorithm I** uses a *definite* quaternion algebra over F (ramified at all n infinite places);
  it is an efficient formulation of the Brandt-module approach **[Dem07, DD08]**. Its key advantage
  is that the most expensive steps occur in precomputation depending only on the quaternion algebra,
  so forms of many different levels and weights can be computed from one precomputation.
- **Algorithm II** uses the Shimura curve associated to a quaternion algebra ramified at exactly
  n − 1 infinite places **[GV11]**; it computes the homology of the Shimura curve (closer to the
  classical modular-symbols algorithm over **Q**) via Voight's algorithm **[Voi09]** for the
  fundamental domain of a Fuchsian group. Algorithm II is implemented only for parallel weight 2.

By default the algorithm and order are selected automatically; the essential requirement is that
the quaternion order be ramifiable only at primes p for which the space is p-new. Algorithm I is
generally more optimised and preferred; for spaces over odd-degree fields with level not divisible
by small primes, Algorithm II may be preferable. An exposition of both algorithms is given in
**[DV12]**. The Magma category for spaces is `ModFrmHil`; elements have type `ModFrmHilElt`.

**Verbose output.** `SetVerbose("ModFrmHil", n)` with n = 0 (silent, default), 1 (concise), 2 or 3
(possibly bulky) prints information during computation.

---

## 137.1 Introduction

Introductory material only (definitions, background, algorithm sketches, categories and verbose
output) — see the overview above. No intrinsics are defined in this section other than the verbose
flag `SetVerbose("ModFrmHil", n)`.

---

## 137.2 Creation of Full Cuspidal Spaces

In the current implementation only cusp forms are supported. Computations in the space are done by
realising it as a space of automorphic forms on an order in a suitable quaternion algebra.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HilbertCuspForms(F, N, k)` / `HilbertCuspForms(F, N)` | The space of Hilbert modular forms over field `F` (a number field or the rationals) on Γ₀(N) with weight `k`. The level `N` should be an ideal in the maximal order of `F`; `k` should be a sequence of deg(F/**Q**) integers, all at least 2 and all of the same parity. If `k` is not specified, the weight is taken to be parallel weight 2, i.e. `[2, 2, ..., 2]`. Parameter `QuaternionOrder` (`AlgAssVOrd`) lets the user specify the order used; otherwise it is chosen automatically (and hidden). The quaternion algebra may be definite (ramified at all infinite places of `F` ⇒ Algorithm I) or indefinite (ramified at all infinite places except one ⇒ Algorithm II). Indefinite algebras may only be used for parallel weight 2, and the algebra must be unramified at all finite primes (for full cuspidal spaces). In the definite case the order must be maximal; in the indefinite case it must be an Eichler order of discriminant equal to the level `N`. | Realises the space as automorphic forms on a quaternion order via the Jacquet–Langlands correspondence; definite ⇒ Brandt-module Algorithm I **[Dem07, DD08]**, indefinite ⇒ Shimura-curve Algorithm II **[GV11, Voi09]**. |

*Worked examples:* H137E1 (spaces over **Q**(√85): level 1 and a split prime over 3, recycling the
quaternion order via `QuaternionOrder`).

---

## 137.3 Basic Properties

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BaseField(M)` | The field on which the space `M` was defined. | — |
| `Weight(M)` | The weight of the space `M`. | — |
| `CentralCharacter(M)` | The central character of the weight representation defining `M`. Significant only for higher-weight spaces (not parallel weight 2). | — |
| `Level(M)` | The level of the space `M`. | — |
| `DirichletCharacter(M)` | The nebentypus of the space `M`. | — |
| `IsCuspidal(M)` | True if `M` was created as (a subspace of) a space of cusp forms. In the current implementation always `true`. | — |
| `IsNew(M)` | True if `M` was created as (a subspace of) a new space (e.g. via `NewSubspace` or `NewformDecomposition`), or if `M` is known to satisfy `NewLevel(M) = Level(M)`. | — |
| `NewLevel(M)` | The level at which `M` is known to be new (see `NewSubspace`). | — |
| `Dimension(M)` | The dimension of the space `M`. Parameter `UseFormula` (`BoolElt`, default `true`): determines the dimension either by "dimension formulae" or by explicitly constructing the space; by default formulae are used when available and cheap, and can be forced on/off via `UseFormula`. | Dimension formulae are implemented for parallel weight 2 (sums over certain cyclotomic extensions of the base field of `F`). Formula results are guaranteed only under GRH (they may involve conditionally computed class numbers); if the space is later explicitly computed, the dimension is verified unconditionally. |
| `QuaternionOrder(M)` | The quaternion order used internally to compute the space `M`. | — |
| `IsDefinite(M)` | Indicates which of the two algorithms is used to compute `M`. Equivalent to `IsDefinite(Algebra(QuaternionOrder(M)))`. Calling this causes a `QuaternionOrder` for `M` to be chosen (if not already set). | — |

*Worked examples:* H137E2 (continuation of H137E1 over **Q**(√85): `Dimension`, `IsDefinite`
showing Algorithm I, and speedup from recycling the quaternion order).

---

## 137.4 Elements

The current implementation does not provide functionality for manipulating elements of spaces of
Hilbert modular forms. Unlike classical modular forms in Magma, Hilbert modular forms may be
defined over extensions of the base field of their parent space.

| Intrinsic | Description |
|-----------|-------------|
| `Parent(f)` | The space of Hilbert modular forms containing `f`. |
| `BaseField(f)` | The field over which the Hilbert modular form `f` is defined. This is either equal to, or an extension of, the base field of `Parent(f)`. |

---

## 137.5 Operators

Operators on spaces of Hilbert modular forms are returned as matrices with respect to a basis of
`M`. For parallel weight 2, operators are matrices over **Q** and the basis is permanently fixed.
For all other weights, two finite extensions of **Q** may arise: (i) the *raw* field used for the
raw computations, in which operators are originally computed; and (ii) the minimal field F for
which there exists a basis of `M` with operator entries in F — a **rational basis** of `M` (for all
parallel weights this minimal field is **Q**).

Because changing the basis can be expensive, for some spaces a rational basis is not computed by
default; there Hecke operators are returned over the raw field until `SetRationalBasis(M)` is
invoked, after which the basis (and hence the operators) permanently change. A space `M` is
guaranteed to have a permanently-fixed rational basis when: (i) `M` has parallel weight; (ii) `M`
was constructed via `NewSubspace` (unless `RationalBasis` set to `false`) or `NewformDecomposition`;
or (iii) `SetRationalBasis(M)` has been invoked.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HeckeOperator(M, P)` | A matrix representing the Hecke operator T_P on the space `M`. | Double-coset Hecke action via the quaternion order (Algorithm I/II). |
| `AtkinLehnerOperator(M, P)` | A matrix representing the Atkin–Lehner operator w_P on the space `M`. | Double-coset operator. |
| `DegeneracyOperator(M, P, Q)` | Degeneracy maps in the "downward" direction, as maps from `M` to itself. Here `M` has level `N`, `P` is a prime dividing `N`, and `Q` equals either `P` or the unit ideal `(1)`. Returns a matrix representing a map from `M` to `M` whose image equals a copy of the space of level `N/P`. When `Q = (1)`, this is the double-coset operator defined by cosets of an element of determinant 1 (a "norm" map); when `Q = P`, the double-coset operator defined by cosets of an element of determinant `Norm(P)`. | Double-coset degeneracy operator. |
| `DeleteHeckePrecomputation(O)` / `DeleteHeckePrecomputation(O, P)` | Procedures that delete data obtained during the precomputation phase of the "definite" algorithm (Algorithm I). This data is used to compute Hecke (and other) operators for given primes and is re-usable for all spaces computed with the same quaternion order `O` (often the same for all weights and levels over a given number field). The data is stored by default since it is the most expensive part of the Hecke computation, but is very memory-intensive; these procedures reclaim that memory. | — |

*Worked examples:* H137E3 (over **Q**(√2): `HeckeOperator` on a dimension-1 space of weight `[2,4]`
reading off eigenvalues; a dimension-3 level-5 space with Hecke matrices over an extension `K`).

---

## 137.6 Creation of Subspaces

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `NewSubspace(M)` / `NewSubspace(M, I)` | Given a cuspidal space `M` of level `N` and an ideal `I` dividing `N`, the subspace of `M` consisting of forms new at the ideal `I` (or new at `N`, if `I` is not given) — the complement of the space generated by all images under degeneracy maps of spaces of level `N/P` for primes `P` dividing `I`. `I` must be squarefree and coprime to `N/I`. The new subspace is not necessarily an explicit subspace of `M`: in many cases it is obtained explicitly by computing degeneracy maps, otherwise it is computed independently of `M` using an automatically chosen quaternion order. Parameter `QuaternionOrder` (`AlgAssVOrd`) overrides the automatic choice (allowable orders as for `HilbertCuspForms`, but here the quaternion algebra may be ramified at finite primes dividing `I`; when indefinite, the ramified finite primes must be precisely those dividing `I`). In the non-parallel weight case, set `RationalBasis := false` to defer computing a rational basis (later set it with `SetRationalBasis`). | New subspace via degeneracy maps, or computed independently on a separate quaternion order; a different algorithm and order may be chosen for the new subspace than for the containing space. |
| `SetRationalBasis(M)` | Procedure that changes the basis of `M` to a rational basis (full explanation in §137.5). If the basis of `M` is already known to be rational, nothing is done; in particular this has no effect on parallel-weight-2 spaces. After invocation, the basis of `M` is never modified again. | — |

*Worked examples:* H137E4 (over **Q**(√10): new/old forms with level dividing 3, confirming
degeneracy-map images are independent); H137E5 (over the degree-3 real subfield of **Q**(ζ₇):
new subspaces computed independently with `IsDefinite` showing different algorithms; timing
comparison of `HeckeOperator` on `Mnew` (Algorithm I) versus `M`).

---

## 137.7 Eigenspace Decomposition and Eigenforms

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HeckeEigenvalueBound(M, P)` | A bound on the absolute value of the Hecke eigenvalue at the prime `P` that must hold for all newforms in the space `M`. | Analytic eigenvalue bound. |
| `NewformDecomposition(M)` | Given a space `M` created as a `NewSubspace`, decomposes `M` into subspaces that are irreducible modules under the Hecke action. | Simultaneous diagonalisation under the Hecke action. |
| `NewformsOfDegree1(M)` | The list of new eigenforms in `M` with rational eigenvalues, i.e. the 1-dimensional components of `NewformDecomposition`. `M` need not be a new space. Avoids constructing the new subspace of `M` and uses bounds on the eigenvalues. | Targeted search for rational (degree-1) eigenforms using eigenvalue bounds. |
| `Eigenform(M)` | An eigenform contained in the space `M`, which should be an irreducible module under the Hecke action (e.g. a space obtained from `NewformDecomposition`). | — |
| `Eigenforms(M)` | A list containing an eigenform from each space in `NewformDecomposition(M)`. | — |
| `HeckeEigenvalueField(M)` | Given a space `M` constructed using `NewformDecomposition`, the number field over which the `Eigenform` of `M` is defined. | — |
| `HeckeEigenvalue(f, P)` | The eigenvalue of the Hecke operator T_P acting on the eigenform `f` (which should be a Hilbert modular form constructed using `Eigenform`). | — |

*Worked examples:* H137E6 (over **Q**(√2): newforms corresponding to elliptic curves of conductor
11; `NewformDecomposition`, `Eigenform`, `HeckeEigenvalue` agreeing with the classical conductor-11
form via `Newforms`/`CuspForms(11)`; a 5-dimensional piece with totally real
`HeckeEigenvalueField`).

---

## 137.8 Further Examples

*Worked examples:* H137E7 (over **Q**(√15): a weight-`[2,4]` level-1 space, `HeckeOperator` over an
extension of F, then `IsNew`/`SetRationalBasis` giving a **Z**-rational basis via rational canonical
form, and `NewformDecomposition`); H137E8 (computing classical modular forms of level 14 three
independent ways: Algorithm I via a definite quaternion algebra over `RationalsAsNumberField`
ramified at 2 and ∞; Algorithm II via an indefinite quaternion algebra over **Q** ramified at 2 and
7; and the standard modular-forms package `CuspForms(14)` — all eigenvalue lists agree).

---

## 137.9 Bibliography (canonical references)

| Key | Reference |
|-----|-----------|
| **[DD08]** | L. Dembele and S. Donnelly. *Computing Hilbert Modular Forms Over Fields With Nontrivial Class Group.* In S. Pauli, F. Hess and M. Pohst, editors, *ANTS VIII*, volume 5011 of *LNCS*. Springer-Verlag, 2008. |
| **[Dem07]** | L. Dembélé. *Quaternionic Manin symbols, Brandt matrices, and Hilbert modular forms.* Math. Comp., **76**(258):1039–1057 (electronic), 2007. |
| **[DV12]** | L. Dembélé and J. Voight. *Explicit methods for Hilbert modular forms.* To appear, *Elliptic curves, Hilbert modular forms and Galois deformations*, 2012. |
| **[Fre90]** | E. Freitag. *Hilbert modular forms.* Springer-Verlag, Berlin, 1990. |
| **[Gar90]** | Paul B. Garrett. *Holomorphic Hilbert modular forms.* The Wadsworth & Brooks/Cole Mathematics Series. Wadsworth & Brooks/Cole Advanced Books & Software, Pacific Grove, CA, 1990. |
| **[GV11]** | M. Greenberg and J. Voight. *Computing systems of eigenvalues associated to Hilbert modular forms.* Math. Comp., **80**:1071–1092, 2011. |
| **[Hid06]** | Haruzo Hida. *Hilbert modular forms and Iwasawa theory.* Oxford Mathematical Monographs. The Clarendon Press Oxford University Press, Oxford, 2006. |
| **[Voi09]** | J. Voight. *Computing fundamental domains for cofinite Fuchsian groups.* J. Théor. Nombres Bordeaux, **21**(2):469–491, 2009. |

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Jacquet–Langlands correspondence (space realised on a quaternion order) **[Hid06]** | `HilbertCuspForms`, `NewSubspace` |
| Algorithm I — definite quaternion orders / Brandt modules **[Dem07, DD08]** | `HilbertCuspForms` (definite case), `DeleteHeckePrecomputation` |
| Algorithm II — indefinite quaternion orders / Shimura-curve homology **[GV11, Voi09]** | `HilbertCuspForms` (indefinite case), `NewSubspace` (indefinite) |
| Exposition of both algorithms **[DV12]** | (whole package) |
| Dimension formulae (parallel weight 2, GRH-conditional) | `Dimension(:UseFormula)` |
| Double-coset Hecke / Atkin–Lehner / degeneracy operators | `HeckeOperator`, `AtkinLehnerOperator`, `DegeneracyOperator` |
| Newform decomposition and eigenforms | `NewformDecomposition`, `NewformsOfDegree1`, `Eigenform`, `Eigenforms`, `HeckeEigenvalue`, `HeckeEigenvalueField`, `HeckeEigenvalueBound` |
| Rational-basis selection (rational canonical form) | `SetRationalBasis`, `NewSubspace(:RationalBasis)` |
| Standard references for the theory **[Fre90, Gar90]** | (background) |
