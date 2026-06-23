# Chapter 93 — Mod P Galois Representations

**Handbook part:** XII — Representation Theory
**Handbook pages:** 2789–2794 (PDF pages 2920–2927)

---

## Scope and overview

Chapter 93 provides tools to work with φ-modules over k((u)), where k is a finite field of
characteristic p, and representations of the absolute Galois group of k((u)) with coefficients
in a finite field. The central algorithmic task is computing the **semisimplification** of a
given φ-module and the semisimplification of the Galois representation naturally attached to
it via the Katz equivalence of categories.

A φ-module over K = k((u)) is a finite-dimensional K-vector space D equipped with a
semilinear endomorphism φ: D → D (semilinear with respect to a Frobenius σ on K defined
by σ(Σ aᵢuⁱ) = Σ aᵢ^(p^s) u^(bi) for integers s ≥ 0, b ≥ 2). A φ-module is **étale** if φ
is injective, equivalently if the Frobenius matrix is invertible. A theorem of Katz provides
an equivalence of categories between étale φ-modules over K (with σ = classical Frobenius
x ↦ xᵖ) and Fₚ-representations of G_K = Gal(Kˢᵉᵖ/K).

The simple objects in the category of étale φ-modules over the maximal unramified extension
Kᵘʳ of K are the modules D(d, h) (when σ ≠ id) or D(d, h, λ) (when σ = id), where D(d, h)
is the φ-module of dimension d whose Frobenius matrix is the companion matrix of T^d − u^h.
The **slope** of a simple module isomorphic to D(d, h, λ) is the rational number h/(b^d − 1)
(up to a natural equivalence relation). The slopes of a φ-module encode the **tame inertia
weights** of the corresponding Galois representation.

The package is motivated by the problem of understanding the Brauer-Nesbitt semisimplification
(T/pT)^ss of the mod-p reduction of a Zₚ-lattice T in a Qₚ-representation of G_K. This
semisimplification is independent of the choice of T. Computing the slopes of the associated
φ-module gives the tame inertia weights of the Galois representation. The implementation
covers: creation and manipulation of φ-modules, computation of Jordan-Hölder sequences
(semisimple decompositions), slope computation, and representation of semisimple Galois
representations of absolute Galois groups.

---

## 93.1 Introduction

This section provides the mathematical background: motivation from p-adic Galois representations,
the definition of φ-modules and the Frobenius σ, classification of simple étale φ-modules via
D(d, h, λ) and their slopes, and the Katz equivalence connecting étale φ-modules to Fₚ-representations
of G_K.

No intrinsics are defined in this introductory section.

---

## 93.2 φ-modules and Galois Representations in Magma

Overview of the package functionality: φ-modules have Magma type **PhiMod**; elements have
type **PhiModElt**. Semisimple Galois representations have type **SSGalRep**. All creation,
attribute, and operation functions are described in the subsections below.

### 93.2.1 φ-modules

#### 93.2.1.1 Category

φ-modules have type `PhiMod`; elements of φ-modules have type `PhiModElt`.

#### 93.2.1.2 Creation functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PhiModule(M)` | Create the φ-module whose Frobenius matrix (in some basis) is given by `M`. Optional parameter `F := [s, b]` (default `[1, p]`, the absolute Frobenius) specifies the Frobenius action on coefficients: φ sends a ↦ a^(p^s) on the residue field and maps u ↦ u^b. | — |
| `ElementaryPhiModule(S, d, h)` | Create the elementary φ-module D(d, s) of dimension d whose Frobenius matrix is the companion matrix of T^d − u^s (note: the parameter labelled `h` in the handbook corresponds to the exponent of u). Optional parameter `F` as above (default `[1, p]`). | — |
| `PhiModuleElement(x, D)` | Create the element of the φ-module D whose coordinates are given by the vector `x`. | — |

#### 93.2.1.3 Attributes of φ-modules

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Dimension(D)` | The dimension of the φ-module D. | — |
| `CoefficientRing(D)` | The coefficient ring (Laurent series field) of D. | — |
| `FrobeniusMatrix(D)` | Return the matrix of the action of φ on D in the current basis. | — |

#### 93.2.1.4 Basic operations and properties of φ-modules

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsEtale(D)` | Return true if the action of φ on D is injective (i.e. the Frobenius matrix is invertible). Only checkable up to the precision of the coefficient ring of D. | — |
| `ChangePrecision(~D, prec)` | Change the precision of the coefficient ring of D to `prec` (in-place). | — |
| `DirectSum(D1, D2)` | The direct sum of two φ-modules. The coefficient rings and Frobenius action on coefficients must be the same for both modules. | — |
| `BaseChange(~D, P)` | Change the basis of D in-place. `P` is the base-change matrix: if G is the current Frobenius matrix, the new matrix is P⁻¹ G φ(P). | — |
| `RandomBaseChange(~D)` | Randomly change the basis of D (in-place). | — |
| `Phi(D, x)` | Compute the image of element x ∈ D under the action of φ. | — |

#### 93.2.1.5 Reduction of φ-modules and Galois Representations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SemisimpleDecomposition(D)` | Compute a Jordan-Hölder sequence for the φ-module D. Returns `G, P, sl, pol`: `G` is the Frobenius matrix in a basis where it is block upper-triangular with diagonal blocks corresponding to simple φ-modules; `P` is the corresponding basis-change matrix; `sl` is the list of slopes of D; `pol` is a list of polynomials determining the isomorphism class of each simple diagonal block (together with the corresponding slope). | Slope-based block decomposition; slopes computed over the maximal unramified extension Kᵘʳ. |
| `Slopes(D)` | Compute the list of slopes of D (with multiplicities), as rational numbers up to the equivalence relation x ~ y ⟺ ∃ m,n ∈ ℕ with b^m x − b^n y ∈ ℤ. Each slope h/(b^d − 1) corresponds to a simple constituent D(d, h). | Reduction to K^ur; classification of simple étale φ-modules. |
| `SSGaloisRepresentation(D)` | Compute the semisimplification of the Galois representation (of type SSGalRep) corresponding to the φ-module D via the Katz equivalence. | Katz equivalence of categories; invokes `SemisimpleDecomposition`. |

*Worked examples: H93E1 (create D1 = ElementaryPhiModule of dimension 3 and D2 from an explicit matrix; take their DirectSum; compute Slopes and SSGaloisRepresentation).*

### 93.2.2 Semisimple Galois Representations

Semisimple representations of absolute Galois groups G_K (K a field of Laurent series k((u)),
k finite) with coefficients in a finite field, represented by their tame inertia weights and
polynomials giving the action of Frobenius on the unramified part. Magma type: **SSGalRep**.

#### 93.2.2.1 Category

Semisimple Galois representations have type `SSGalRep`.

#### 93.2.2.2 Creation functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SSGaloisRepresentation(E, K, w, P)` | Create the semisimple representation of the absolute Galois group of `K` (a Laurent series field k((u))) with coefficients in finite field `E`, tame inertia weights given by `w`, and Frobenius action on the unramified part described by the elements of list `P`. | Direct construction from data (tame inertia weights + Frobenius polynomials). |

#### 93.2.2.3 Basic operations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CoefficientRing(V)` | The coefficient ring (finite field) of the semisimple Galois representation V. | — |
| `FixedField(V)` | The fixed field of the absolute Galois group of which V is a representation (i.e. the Laurent series field K = k((u))). | — |
| `Weights(V)` | The tame inertia weights of V. | — |

#### 93.2.2.4 Representation associated to a φ-module

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SSGaloisRepresentation(D)` | If D is a φ-module over a Laurent series field K = k((u)), return the semisimplification of the Galois representation (SSGalRep) associated to D via the Katz equivalence. | Katz equivalence; semisimple decomposition of D; tame inertia weights extracted from slopes. |

---

## 93.3 Examples

The chapter contains one worked example (H93E1) illustrating the core workflow:

1. Construct a finite field `k = GF(3,2)` and Laurent series ring `S<u>` over k with precision 20.
2. Create `D1 = ElementaryPhiModule(S, 3, 2)` (dimension 3, companion matrix of T³ − u²).
3. Create `D2 = PhiModule(M)` from an explicit 2×2 matrix with a scalar multiple of u as entry.
4. Form `D = DirectSum(D1, D2)`.
5. Compute `Slopes(D)`, yielding slopes `[2, 1]` and `[3, 2]` (i.e. slopes 2/(3−1) = 1 and 2/(3²−1) = 2/8, displayed with numerators and dimension denominators).
6. Compute `SSGaloisRepresentation(D)`, which returns a semisimple representation of the absolute Galois group of the Laurent series field over GF(3²) with coefficients in GF(3) and components `[[3, 18], [2, 3]]`.

---

## 93.4 Bibliography

Chapter 93 does not list an explicit bibliography section in the handbook text. The mathematical
foundations rely on the following works cited in the chapter prose:

| Key | Reference |
|-----|-----------|
| **[Katz]** | N. M. Katz. *Local-to-global extensions of representations of fundamental groups.* Ann. Inst. Fourier (Grenoble) **36**(4):69–106, 1986. (Theorem establishing the equivalence of categories between étale φ-modules and Fₚ-representations of G_K, cited in §93.1.4.) |
| **[BN]** | R. Brauer and C. Nesbitt. *On the modular characters of groups.* Ann. of Math. **42**:556–590, 1941. (Brauer–Nesbitt theorem: semisimplification (T/pT)^ss is independent of the choice of lattice T, cited in §93.1.1.) |
| **[FL]** | J.-M. Fontaine and G. Laffaille. *Construction de représentations p-adiques.* Ann. Sci. École Norm. Sup. (4) **15**(4):547–608, 1982. (Fontaine–Laffaille theory, cited in §93.1.1.) |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Katz equivalence (étale φ-modules ↔ Fₚ-representations of G_K) | `SSGaloisRepresentation(D)`, `SemisimpleDecomposition(D)` |
| Jordan-Hölder / semisimple decomposition of φ-modules (slope theory over K^ur) | `SemisimpleDecomposition`, `Slopes` |
| Construction of elementary φ-modules D(d, h) | `ElementaryPhiModule` |
| Direct-sum formation and basis change | `DirectSum`, `BaseChange`, `RandomBaseChange` |
| Tame inertia weight extraction from slopes | `Slopes`, `SSGaloisRepresentation` |
| Semisimple Galois representation construction from weights/Frobenius data | `SSGaloisRepresentation(E, K, w, P)`, `CoefficientRing`, `FixedField`, `Weights` |
