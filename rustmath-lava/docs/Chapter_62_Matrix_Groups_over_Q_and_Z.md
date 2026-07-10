# Chapter 62 — Matrix Groups over Q and Z

**Handbook part:** IX — Finite Groups
**Handbook pages:** 1781–1787 (PDF pages 1912–1921)

---

## Scope and overview

Chapter 62 covers specialised functionality for finite matrix groups over **Q** and **Z** that
goes beyond what is available for generic finite matrix groups (Chapter 61). Specifically, it
addresses:

1. **Invariant bilinear forms** — computing bases for the space of G-invariant symmetric and
   antisymmetric forms for a finite matrix group G < GL(n, Q), including fast modular methods
   for counting forms without constructing them.

2. **Endomorphism rings** — computing the full endomorphism ring (commuting algebra) of G, its
   centre, and retrieving independent endomorphisms or central endomorphisms.

3. **New groups from others** — constructing the Bravais group of a finite integral matrix
   group and converting a rational matrix group to an equivalent integral one.

4. **Perfect forms and normalizers/centralizers** — computing G-perfect forms and, from them,
   the normalizer or centralizer of a finite integral matrix group G in GL(n, Z). The
   underlying algorithms are explained in **[OPS98, Opg01]** and are based on the sublattice
   machinery and enumeration of G-perfect forms. They perform well when the space of
   G-invariant symmetric forms has small dimension (typically less than 15) and the index of G
   in its Bravais group is not too large.

5. **Conjugacy** — deciding GL(n, Q)- and GL(n, Z)-conjugacy for finite integral or rational
   matrix groups, and splitting GL(n, Q)-conjugacy classes into GL(n, Z)-classes via orbits of
   G-invariant lattices under the normalizer.

6. **Conjugacy tests for matrices** — deciding GL(n, Z)- and SL(n, Z)-conjugacy for individual
   rational or integral matrices, and computing centralizers in GL(n, Z). Currently limited to
   matrices of finite order or the 2×2 case.

The databases of maximal finite irreducible rational, integral, symplectic, and quaternionic
matrix groups are documented separately in Chapter 66.

---

## 62.2 Invariant Forms

Let G be a finite matrix group G < GL(n, Q). A matrix F ∈ M(n, Q) is G-invariant if
g F g^tr = F for all g ∈ G.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PositiveDefiniteForm(G)` | For a finite integral or rational matrix group G, return a positive definite symmetric G-invariant form. | — |
| `InvariantForms(G)` / `SymmetricForms(G)` / `AntisymmetricForms(G)` | For an integral or rational matrix group G, return a basis for the space of G-linear forms, or for the subspace of symmetric or antisymmetric forms respectively. The first form returned by `InvariantForms` and `SymmetricForms` will be positive definite. | — |
| `InvariantForms(G, n)` / `SymmetricForms(G, n)` / `AntisymmetricForms(G, n)` | For an integral or rational matrix group G, return a sequence consisting of n ≥ 0 G-invariant (symmetric or antisymmetric) bilinear forms for G. | — |
| `NumberOfInvariantForms(G)` / `NumberOfSymmetricForms(G)` / `NumberOfAntisymmetricForms(G)` | For an integral or rational matrix group G or a G-lattice L, return the dimension of the space of (symmetric or antisymmetric) invariant bilinear forms for G. | Modular method — much faster than explicitly computing the forms. |

---

## 62.3 Endomorphisms

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `EndomorphismRing(G)` | For an integral or rational matrix group G, return the endomorphism ring (i.e., the commuting algebra) of G as a subalgebra of M(n, Z) or M(n, Q) respectively. | — |
| `CentreOfEndomorphismRing(G)` / `CentreOfEndomorphismAlgebra(G)` | For an integral or rational matrix group G, return the centre of the endomorphism ring (commuting algebra) of G as a subalgebra of M(n, Z) or M(n, Q) respectively. | — |
| `DimensionOfEndomorphismRing(G)` | Return the dimension of the endomorphism ring of an integral or rational matrix group G. | Modular method. |
| `DimensionOfCentreOfEndomorphismRing(G)` | Return the dimension of the centre of the endomorphism ring of an integral or rational matrix group G. | Modular method. |
| `Endomorphisms(G, n)` | For an integral or rational matrix group G, return a sequence containing n independent endomorphisms of G. n must lie in [0..d] where d is the dimension of the endomorphism ring of G. | — |
| `CentralEndomorphisms(G, n)` | For an integral or rational matrix group G, return a sequence containing n independent central endomorphisms of G. n must lie in [0..d] where d is the dimension of the centre of the endomorphism ring of G. | — |

---

## 62.4 New Groups From Others

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BravaisGroup(G)` | For a finite integral matrix group G, compute its Bravais group: the integral group fixing all symmetric bilinear forms fixed by G. | — |
| `IntegralGroup(G)` | Return the action of the finite rational matrix group G on an invariant lattice as an integral matrix group H, together with the transformation matrix T from the standard lattice to the invariant lattice, so that H = T · G · T⁻¹. | — |

---

## 62.5 Perfect Forms and Normalizers

A positive definite symmetric G-invariant form F is called G-perfect if for every nonzero
symmetric G-invariant form F′ there exists some shortest vector x of F such that F′ x^tr x has
nonzero trace. The normalizer of the Bravais group of G in GL(n, Z) acts on the set of
integral G-perfect forms whose entries have GCD 1, and the number of orbits is finite.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PerfectForms(G)` | Return a sequence of representatives of the orbits of integral G-perfect forms (with GCD 1) under the normalizer of the Bravais group of G in GL(n, Z). Parameter: `Limit` (RngIntElt, default ∞) — if set to a positive integer m, the algorithm stops after m orbits have been enumerated. | Enumeration of G-perfect form orbits **[OPS98, Opg01]**. |
| `NormalizerGLZ(G)` / `CentralizerGLZ(G)` | Given a finite subgroup G of GL(n, Z), return the normalizer or centralizer of G in GL(n, Z). Parameter: `IsBravais` (BoolElt, default false) — set to true if G is known to equal its Bravais group, to speed up the computation. | Variation of Opgenorth's normalizer algorithm **[Opg01]**. |

---

## 62.6 Conjugacy

The GL(n, Q)-conjugacy class of a finite integral or rational matrix group G splits into
finitely many GL(n, Z)-conjugacy classes. Representatives of these classes are constructed as
the action of G on G-invariant sublattices; the GL(n, Z)-conjugacy classes are in bijection
with the orbits of G-invariant lattices under the normalizer N of G in GL(n, Q).

A G-lattice L′ belongs to a G-lattice L if L = Σᵢ L′ eᵢ where e₁, …, eᵣ are the central
idempotents of the endomorphism ring of G. L is called homogeneously decomposable if L belongs
to itself.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ZClasses(G)` | Given a finite integral or rational matrix group G, return (1) a sequence of integral matrix groups describing the action of G on the G-invariant lattices (each corresponding to a GL(n, Z)-conjugacy class) and (2) a sequence of sequences of basis matrices of the lattices. Parameter: `Homogeneously` (BoolElt, default false) — if true, only compute the homogeneously decomposable lattices and their corresponding matrix groups (much faster for reducible G, but will not yield all conjugacy classes). | Orbit enumeration of G-invariant sublattices under the normalizer N of G in GL(n, Q) **[OPS98, Opg01]**. |
| `IsGLZConjugate(G, H)` | Test whether the finite integral matrix groups G and H are conjugate in GL(n, Z). If so, also return a matrix x such that G^x = H. | — |
| `IsBravaisEquivalent(G, H)` | Given two finite integral matrix groups G and H, test whether their Bravais groups B(G) and B(H) are conjugate in GL(n, Z). If so, also return a matrix x such that B(G)^x = B(H). Does not require computing the Bravais groups explicitly, and is faster than calling `IsGLZConjugate` on the Bravais groups directly. | — |
| `IsGLQConjugate(G, H)` | Test whether the finite rational matrix groups G and H are conjugate in GL(n, Q). If so, also return a matrix x such that G^x = H. Parameter: `Al` (MonStgElt) — `"Aut"` uses the GModule machinery together with the outer automorphism group of H; `"ZClasses"` splits the GL(n, Q)-conjugacy class of H into GL(n, Z)-classes then decides whether an integral copy of G lies in one of them via calls to `IsGLZConjugate`. If not provided, a sensible choice is made automatically. | Two algorithms: GModule + outer automorphism group (`"Aut"`), or Z-class splitting (`"ZClasses"`). |

*Worked examples: H62E1 (splitting the GL₃(Q)-conjugacy class of a dihedral-12 representation into GL₃(Z)-classes, verifying lattice membership via central idempotents of the endomorphism ring). H62E2 (automorphism groups of the lattices B₈ and D₈ are GL₈(Q)-conjugate but not GL₈(Z)-conjugate).*

---

## 62.7 Conjugacy Tests for Matrices

Given two n × n matrices A and B with rational or integral entries, Magma can test whether A
is conjugate to B in GL(n, Z) or SL(n, Z). The current implementation is limited to the cases
where A and B have finite order, or where n = 2. This limitation is expected to be removed in
future versions.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsGLZConjugate(A, B)` / `IsSLZConjugate(A, B)` | Test whether two rational or integral matrices A and B are conjugate in GL(n, Z) or SL(n, Z) respectively. If so, also return a matrix x such that A^x = B. Currently limited to matrices of finite order or the 2×2 case. | — |
| `CentralizerGLZ(A)` | Given a rational or integral matrix A, return its centralizer in GL(n, Z). Currently limited to matrices of finite order or 2×2 matrices. | — |

*Worked examples: H62E2 (GL₈(Q) vs GL₈(Z) conjugacy of B₈ and D₈ automorphism groups). H62E3 (companion matrix of the 5th cyclotomic polynomial; finding a unimodular matrix inducing the automorphism C → C², confirming it cannot be realised by a matrix of determinant 1 via `IsSLZConjugate` and `CentralizerGLZ`).*

---

## 62.9 Bibliography

| Key | Reference |
|-----|-----------|
| **[Opg01]** | J. Opgenorth. *Dual Cones and the Voronoi Algorithm.* Exp. Math., **10**(4):599–608, 2001. |
| **[OPS98]** | J. Opgenorth, W. Plesken, and T. Schulz. *Crystallographic Algorithms and Tables.* Acta Crystallographica, **A54**:517–531, 1998. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| G-invariant form computation (modular method for counting) **[OPS98]** | `NumberOfInvariantForms`, `NumberOfSymmetricForms`, `NumberOfAntisymmetricForms`, `DimensionOfEndomorphismRing`, `DimensionOfCentreOfEndomorphismRing` |
| Bravais group and integral group construction | `BravaisGroup`, `IntegralGroup` |
| G-perfect form enumeration **[OPS98, Opg01]** | `PerfectForms` |
| Opgenorth's normalizer algorithm **[Opg01]** | `NormalizerGLZ`, `CentralizerGLZ` (group variant) |
| GL(n, Z)-class splitting via G-invariant lattice orbits **[OPS98, Opg01]** | `ZClasses`, `IsGLZConjugate` (group variant), `IsBravaisEquivalent` |
| GL(n, Q)-conjugacy (GModule + outer automorphisms, or Z-class splitting) | `IsGLQConjugate` |
| GL(n, Z) / SL(n, Z) conjugacy and centralizer for matrices | `IsGLZConjugate` (matrix variant), `IsSLZConjugate`, `CentralizerGLZ` (matrix variant) |
