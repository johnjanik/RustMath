# Chapter 138 — Modular Forms over Imaginary Quadratic Fields

**Handbook part:** XVII — Modular Arithmetic Geometry
**Handbook pages:** 4669–4676 (PDF pages 4803–4810)

---

## Scope and overview

This package deals with **cuspidal spaces of weight-2 modular forms on Γ₀(N)** over an arbitrary
imaginary quadratic field. In the current version it can compute Hecke operators for principal
ideals on these spaces and determine the newforms; the package is to be developed further.

Modular forms over imaginary quadratic fields are referred to as **Bianchi modular forms**. They
are defined analogously to classical modular forms, but the modular group `SL(2, Z)` is replaced by
`SL(2, O_F)`, where `O_F` is the ring of integers of an imaginary quadratic field `F`. This group
acts on **H₃**, 3-dimensional hyperbolic space, and Bianchi modular forms (of weight `k ≥ 2`) are
functions on **H₃** satisfying a natural automorphy relation under this action. For an ideal `N` of
`O_F`, the congruence subgroup `Γ₀(N)` of `GL(2, O_F)` is the subgroup of matrices that are upper
triangular modulo `N`; the space of Bianchi modular forms on `F` of level `N` consists of functions
satisfying the automorphy relation on `Γ₀(N)`. For precise definitions see **[EGM98]**.

Several previous implementations of weight-2 Bianchi modular forms exist, each for specific fields:
**[Cre84]** (and references therein) for Euclidean fields, **[Whi90]** for fields of class number
one, **[Byg99]** for **Q**(√−5), and **[Lin05]** for **Q**(√−23) and **Q**(√−31).

**Algorithmic approach.** A theorem of Franke gives `H*(Γ; E) ≃ H*(g, K; A(Γ, G) ⊗ E)`, so the
cohomology `H*(Γ; E)` is a concrete realization of certain automorphic forms. Ash, Gunnells and
Lee-Szczarba **[LS78, Ash94, Gun00]** define a homology complex `S*(Γ)`, the *sharbly complex*, and a
theorem of Borel–Serre **[BS73]** gives `H^{ν−k}(Γ; C) ≃ H_k(S*(Γ))`, where `ν = vcd(Γ)` is the
virtual cohomological dimension of Γ. There is a natural Hecke action on this complex agreeing with
the Hecke action on automorphic forms. Positive-definite binary Hermitian forms over `F` form an
open cone in a real vector space, with a natural decomposition into polyhedral cones corresponding to
the facets of the Voronoi polyhedron **[Gun99, Koe60, Ash77]**; these facets are in 1–1 correspondence
with perfect forms over `F` and give rise to ideal polytopes in **H₃**. The polytope structure yields
a finite (modulo Γ) spanning set for the sharbly complex — the analogue of unimodular symbols in the
classical case — so the modular-symbol algorithm for the Hecke action is replaced by a 0-sharbly
reduction algorithm. Unlike the usual modular-symbols algorithm, this does **not** require the number
field to be Euclidean.

For a given imaginary quadratic field `F`, Magma computes the Voronoi polyhedron from a complete set
of `GL₂(O)`-class representatives of perfect forms; given a level `n ⊆ O` it uses the polyhedron to
compute `H²(Γ₀(n))`, and given a prime ideal `p ⊆ O` the 0-sharbly reduction algorithm computes the
action of the Hecke operator `T_p` on `H²(Γ₀(n))`. The efficient Voronoi-polyhedron algorithm
of **[Gun99]** replaces the modular-symbol algorithm. A key advantage is that the most expensive steps
occur in a **precomputation phase that depends only on the imaginary quadratic field**, so forms of
many different levels over the same field can be computed from the same precomputation. The
cohomology, as a module for the Hecke algebra, decomposes into an Eisenstein piece and a cuspidal
piece; Magma strips away the Eisenstein part (recognisable from the Hecke eigenvalues) and returns the
**cuspidal subspace only**.

---

## 138.1 Introduction

Introductory material (algorithms, categories, verbose output). See the overview above for the
algorithmic background.

### 138.1.1 Algorithms

Described above: Franke's theorem realizes automorphic forms in group cohomology; the sharbly complex
and Borel–Serre duality reduce the computation to homology; Voronoi reduction of binary Hermitian
forms provides a spanning set; and a 0-sharbly reduction algorithm computes the Hecke action without
requiring the field to be Euclidean. References: **[LS78, Ash94, Gun00, BS73, Gun99, Koe60, Ash77]**.

### 138.1.2 Categories

Spaces of Bianchi modular forms in Magma are objects of type `ModFrmBianchi`. This is functionally the
same as the type `ModFrmHil` for Hilbert modular forms (see Chapter 137): functions for Hilbert
modular forms can also be applied to Bianchi spaces, for example `NewformDecomposition`.

### 138.1.3 Verbose Output

| Intrinsic | Description |
|-----------|-------------|
| `SetVerbose("Bianchi", n)` | Print information during computation about what the program is doing. `n` is 0 (silent, the default), 1 (concise information), or 2, 3, 4 (which may display bulky data). |

---

## 138.2 Creation

In the current implementation only cusp forms are supported.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BianchiCuspForms(F, N)` | Creates the cuspidal subspace of the space of Bianchi modular forms over the imaginary quadratic field `F` on Γ₀(N) with weight 2. The level `N` should be an ideal in the maximal order of `F`. Parameter `VorData` (record; default unset): set equal to `VoronoiData(M)` for a previously computed space `M` of forms over the same field, to avoid repeating the time-consuming precomputations that depend only on the field. | Voronoi-reduction / 0-sharbly cohomology construction **[Gun99]**. |

*Worked example:* H138E1 (spaces of modular forms over **Q**(√−14) for various levels: norm-1 level has dimension 0; level the square of a split prime above 3 has dimension 1; reuse of `VoronoiData` to skip precomputation).

---

## 138.3 Attributes

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BaseField(M)` | The field on which the space `M` of Bianchi modular forms was defined. | — |
| `Level(M)` | The level of the space `M`. | — |
| `Dimension(M)` | The dimension of the space `M`. Dimension formulas are not available, so the dimension is computed by explicit construction of the space. | Explicit construction. |
| `VoronoiData(M)` | A record containing technical data computed in the precomputation phase of the algorithm. This depends only on the base field of `M`, and the data can be reused when computing spaces of different levels over the same field. | Voronoi precomputation **[Gun99]**. |

---

## 138.4 Hecke Operators

The computations are done essentially on the cohomology of `Γ\H₃`, so for a non-principal ideal `P`
the Hecke operator `T_P` does not act on this space directly. Sometimes the Hecke action can still be
deduced from the action of principal Hecke operators (see for instance **[Lin05]**): if `p` is a prime
ideal coprime to the level, there exists an ideal `a`, coprime to `p` and the level, such that `a²p`
is principal, and then the composition `T_{a,a} T_p` acts on the cohomology.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HeckeOperator(M, P)` | A matrix representing a certain Hecke action `T` on the space `M` of Bianchi modular forms, with respect to the fixed basis of `M`. The ideal `P` must be principal (not necessarily prime), or prime and a square in the class group. If `P` is principal, `T` is the Hecke operator `T_P`. When `P` is prime and a square in the class group, `T` is the composition `T_{a,a} T_P` for a suitably chosen ideal `a`. | 0-sharbly reduction of the principal Hecke action **[Gun99, Lin05]**. |

*Worked example:* H138E2 (continues H138E1: `HeckeOperator` at the two primes above 23 and at `2*OF` on the dimension-1 space over **Q**(√−14), reading off the single eigenform's eigenvalues).

---

## 138.5 Newforms

The functions `NewSubspace` and `NewformDecomposition` may be applied to spaces of Bianchi modular
forms. (See Chapter 137.)

| Intrinsic | Description |
|-----------|-------------|
| `NewSubspace(M)` | The new subspace of a space `M` of Bianchi modular forms (Hilbert modular forms function, applicable to Bianchi spaces; see Chapter 137). |
| `NewformDecomposition(M)` | The decomposition of `M` into newforms (Hilbert modular forms function, applicable to Bianchi spaces; see Chapter 137). |

---

## 138.6 Bibliography (canonical references)

| Key | Reference |
|-----|-----------|
| **[Ash77]** | Avner Ash. *Deformation retracts with lowest possible dimension of arithmetic quotients of self-adjoint homogeneous cones.* Math. Ann. **225**(1):69–76, 1977. |
| **[Ash94]** | Avner Ash. *Unstable cohomology of SL(n, O).* J. Algebra **167**(2):330–342, 1994. |
| **[BS73]** | A. Borel and J.-P. Serre. *Corners and arithmetic groups.* Comment. Math. Helv. **48**:436–491, 1973. Avec un appendice: Arrondissement des variétés à coins, par A. Douady et L. Hérault. |
| **[Byg99]** | Jeremy Bygott. *Modular forms and modular symbols over imaginary quadratic fields.* PhD thesis, University of Exeter, 1999. |
| **[Cre84]** | J. E. Cremona. *Hyperbolic tessellations, modular symbols, and elliptic curves over complex quadratic fields.* Compositio Math. **51**(3):275–324, 1984. |
| **[EGM98]** | J. Elstrodt, F. Grunewald, and J. Mennicke. *Groups acting on hyperbolic space.* Springer Monographs in Mathematics. Springer-Verlag, Berlin, 1998. Harmonic analysis and number theory. |
| **[Gun99]** | Paul E. Gunnells. *Modular symbols for **Q**-rank one groups and Voronoi reduction.* J. Number Theory **75**(2):198–219, 1999. |
| **[Gun00]** | Paul E. Gunnells. *Computing Hecke eigenvalues below the cohomological dimension.* Experiment. Math. **9**(3):351–367, 2000. |
| **[Koe60]** | Max Koecher. *Beiträge zu einer Reduktionstheorie in Positivitätsbereichen. I.* Math. Ann. **141**:384–432, 1960. |
| **[Lin05]** | Mark Lingham. *Modular forms and elliptic curves over imaginary quadratic fields.* PhD thesis, University of Nottingham, 2005. URL: http://etheses.nottingham.ac.uk/138/. |
| **[LS78]** | Ronnie Lee and R. H. Szczarba. *On the torsion in K₄(**Z**) and K₅(**Z**).* Duke Math. J. **45**(1):101–129, 1978. |
| **[Whi90]** | Elise Whitley. *Modular symbols and elliptic curves over imaginary quadratic fields.* PhD thesis, University of Exeter, 1990. |

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Voronoi-polyhedron precomputation (perfect binary Hermitian forms) **[Gun99, Koe60, Ash77]** | `BianchiCuspForms` (`:VorData`), `VoronoiData` |
| Sharbly complex / Borel–Serre duality **[LS78, Ash94, Gun00, BS73]** | `BianchiCuspForms`, `Dimension` |
| 0-sharbly reduction for the Hecke action **[Gun99]** | `HeckeOperator`, `Dimension` |
| Deduction of non-principal Hecke action via `T_{a,a}T_p` **[Lin05]** | `HeckeOperator` |
| Shared Hilbert/Bianchi (`ModFrmHil`) machinery | `NewSubspace`, `NewformDecomposition` |
