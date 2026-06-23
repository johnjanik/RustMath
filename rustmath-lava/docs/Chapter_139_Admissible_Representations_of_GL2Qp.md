# Chapter 139 — Admissible Representations of GL₂(Q_p)

**Handbook part:** XVII — Modular Arithmetic Geometry
**Handbook pages:** 4679–4688 (PDF pages 4810–4823)

---

## Scope and overview

This package lets one start from a classical cuspidal newform and study the local component
at a prime `p` of the associated automorphic representation. Concretely, beginning with a
cuspidal eigenform (given as a space of modular symbols containing a single Galois conjugacy
class of newforms), one defines the **admissible representation** of GL₂(Q_p) that is its
local component, and determines its key invariants (central character, conductor, whether it
is principal series, Steinberg, or supercuspidal). Furthermore, via the **local Langlands
correspondence**, there is a related two-dimensional representation of the absolute Galois
group of Q_p; one may compute (the restriction to inertia of) that Galois / Weil
representation. The algorithms implemented are those described in **[LW]**.

**Mathematical setting.** For a local non-archimedean field `F` and a reductive group `G`
over `F`, the representation theory of `G` connects (often conjecturally) with that of the
absolute Galois group of `F`. This package treats admissible irreducible representations in
the case `G = GL₂` and `F = Q_p`; such objects correspond canonically to two-dimensional
representations of the absolute Galois group of `F`. The first systematic study of admissible
representations is **[JL70]**; an accessible introduction is **[BH06]**.

An *admissible* representation of the locally compact group `G = GL₂(Q_p)` on a complex vector
space `V` is a homomorphism π : G → Aut V such that (i) every vector `v ∈ V` is fixed by a
compact open subgroup of `G`, and (ii) for every compact open subgroup `K ⊂ G`, the fixed
space `V^K` is finite-dimensional. Each irreducible admissible π has a unique **central
character** ε : Q_p^× → C^× with π(g) acting as ε(g) on the center. The **conductor** measures
how small a compact open subgroup must be before nonzero invariant vectors appear (see
**[Cas73]**): using the filtration `K_0(p^n)` of GL₂(Z_p) by matrices with `c ≡ 0 (mod p^n)`,
π is *spherical* / *unramified principal series* (conductor 1) if it has a nonzero
`K_0(1) = GL₂(Z_p)`-fixed vector, and otherwise has conductor `p^n` for the minimal `n ≥ 1`
admitting a vector transforming by ε. That distinguished vector (unique up to scaling) is the
**new vector**. A twist `π ⊗ χ` by a character χ of Q_p^× sends `g ↦ χ(g)π(g)`; π is **minimal**
if it has minimal conductor among all its twists.

Although admissible representations are generally infinite-dimensional, Magma presents them
through its infrastructure for representations of finite groups and Dirichlet characters.

**Classification.** The **principal series** π(χ₁, χ₂) := Ind_B^G χ is built by inducing a
character of the Borel subgroup B of upper-triangular matrices from two characters χ₁, χ₂ of
Q_p^×; it is irreducible unless χ₁χ₂⁻¹ = |.|^{±1}, in which case it has length two, yielding a
1-dimensional factor and the irreducible infinite-dimensional **Steinberg representation**
St_G (trivial central character, conductor `p`). A *supercuspidal* representation is one not
in the principal series; it has conductor `p^c` with `c ≥ 2`, and by **[BH06]**, Ch. 15, is
induced (π = Ind_K^G Ξ) from a representation Ξ of an open compact-mod-center subgroup `K`:
`K = Q_p^× GL₂(Z_p)` when `c` is even, and `K` the normalizer of the Iwahori subgroup when `c`
is odd. The pair `(K, Ξ)` is a **cuspidal inducing datum**. The **local Langlands
correspondence** is the canonical bijection π ↦ σ(π) between irreducible admissible
representations of `G` and local 2-dimensional Galois representations; the conductor of π
equals the Artin conductor of σ(π), and π is principal series / Steinberg / supercuspidal iff
σ(π) is a sum of two characters / reducible-indecomposable / irreducible. The correspondence
for GL₂ was laid down in **[JL70]** and completed in **[Kut80]**, **[Kut84]**. For `p ≠ 2`,
an irreducible σ is induced from a character χ of a quadratic extension `E/Q_p`, and `(E, χ)`
is an **admissible pair** (**[BH06]**, Ch. 18).

**Connection with modular forms.** A cuspidal newform `f` for Γ₀(N) with Dirichlet character ε
gives a cuspidal automorphic representation Π_f (**[Gel75]**), a restricted tensor product of
local admissible representations π_{f,p} of GL₂(Q_p). By **[Car83]**, σ(π_{f,p}) is the
restriction of Deligne's Galois representation ρ_f to the decomposition group at `p`. These are
almost always unramified principal series; the only challenge is computing π_{f,p} when `p`
divides `N`.

**Category and verbosity.** Admissible representations are objects of Magma type `RepLoc`. Set
`SetVerbose("RepLoc", 1)` to see information about computations in progress.

---

## 139.2 Creation of Admissible Representations

One starts with a classical cuspidal eigenform, given as a space of modular symbols.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LocalComponent(M, p)` | The admissible representation of GL₂(Q_p) associated to the cuspidal eigenform specified by the space of modular symbols `M`. `M` must be cuspidal and contain only a single Galois conjugacy class of newforms (such spaces are created with `NewformDecomposition`). | Local-component computation of **[LW]**. |

*Worked example:* H139E1 (`LocalComponent` of the level-11 weight-2 newform; recognised as the Steinberg representation of GL(2, Q_11)).

---

## 139.3 Attributes of Admissible Representations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CentralCharacter(pi)` | The central character of π (an admissible representation on GL(Q_p)): a Dirichlet character of `p`-power conductor. | Direct. |
| `Conductor(pi)` | The conductor of π, written multiplicatively. | Direct. |
| `DefiningModularSymbolsSpace(pi)` | The space of modular symbols from which π was created. | Direct. |
| `IsMinimal(pi)` | Returns `true` if the conductor of π cannot be lowered by twisting by a character of Q_p^×. If π is not minimal, also returns a minimal representation π′ and a Dirichlet character χ with π = π′ twisted by χ. Equivalent to `IsMinimalTwist(DefiningModularSymbolsSpace(pi))` being true. | Reduction to `IsMinimalTwist` of the defining modular symbols space. |

*Worked example:* H139E2 (continues H139E1: `DefiningModularSymbolsSpace`, `Conductor` = 11, trivial central character of the level-11 Steinberg representation).

---

## 139.4 Structure of Admissible Representations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsPrincipalSeries(pi)` | `true` iff the admissible representation π belongs to the principal series. | Direct. |
| `IsSupercuspidal(pi)` | `true` iff the admissible representation π is supercuspidal. | Direct. |
| `PrincipalSeriesParameters(pi)` | For a principal series representation π of GL₂(Q_p): two Dirichlet characters of `p`-power conductor representing the restriction to Z_p^× × Z_p^× of the character of the split torus of GL₂(Q_p) associated to π. | Restriction of the split-torus character (see §139.1.3). |
| `CuspidalInducingDatum(pi)` | For a minimal supercuspidal representation π of GL₂(Q_p): a cuspidal inducing datum `(K, Ξ)` giving rise to π. Since Ξ factors through a finite quotient `K/K₁` of `K`, the function returns a representation of `K/K₁`, from which one deduces the representation on `K`, hence π. | Construction of the inducing datum (see §139.1.4), **[BH06]** Ch. 15. |

---

## 139.5 Local Galois Representations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `GaloisRepresentation(pi)` / `WeilRepresentation(pi)` | For a minimal representation π of GL₂(Q_p): the representation of the Weil group associated to π under the local Langlands correspondence (see §139.1.5). Returns four objects `G, α, L, ρ`: `L/Q_p` is a finite Galois extension through which the representation factors, `G` an abstract group, α a bijective map identifying `G` with Gal(L/Q_p), and ρ a `G`-module describing the Weil representation on Gal(L/Q_p). Parameter `Precision` (`RngIntElt`, default 10). See Chapter 90 for group modules. | Local Langlands correspondence **[JL70, Kut80, Kut84]**. |
| `AdmissiblePair(pi)` | For an ordinary minimal supercuspidal representation π of GL₂(Q_p): the associated admissible pair `(E, χ)` (see §139.1.5). Returns a quadratic field extension `E/Q_p` and a character χ of the unit group of `E` (χ can only be evaluated on units of `E`). | Admissible-pair construction **[BH06]** Ch. 18. |

---

## 139.6 Examples

*Worked examples:* H139E3 (weight-5 level-7 newform whose local representation at 7 is a ramified principal series; `PrincipalSeriesParameters` gives the trivial character and the character of order 2 on Z/7Z; `WeilRepresentation` yields an abelian Galois group over a totally ramified degree-6 extension, with ρ the sum of two characters of Gal(Q_7) via local class field theory); H139E4 (supercuspidal representation of conductor 121 from a weight-2 level-121 newform; `CuspidalInducingDatum` returns a dimension-10 G-module over Q with group GL₂(Z/11Z), and `WeilRepresentation` gives a group isomorphic to the dihedral group of order 6 over a tame extension of Q_11); H139E5 (supercuspidal of conductor 3³ from a weight-4 level-27 newform; `CuspidalInducingDatum` gives a 2-dimensional G-module over the Iwahori subgroup of GL₂(Z_3); `AdmissiblePair` returns `E = Q_3(√3)` with χ(1+E.1) = zeta_3; `WeilRepresentation` over a degree-6 totally ramified extension of Q_3).

---

## 139.7 Bibliography (canonical references)

| Key | Reference |
|-----|-----------|
| **[BH06]** | Colin J. Bushnell and Guy Henniart. *The local Langlands conjecture for GL(2)*, volume 335 of Grundlehren der Mathematischen Wissenschaften [Fundamental Principles of Mathematical Sciences]. Springer-Verlag, Berlin, 2006. |
| **[Car83]** | Henri Carayol. *Sur les Représentations ℓ-adiques attachees aux formes modulaires de Hilbert.* C. R. Acad. Sci. Paris., 296(15):629–632, 1983. |
| **[Cas73]** | W. Casselman. *The restriction of a representation of GL₂(k) to GL₂(O).* Mathematischen Annalen, 206(4), 1973. |
| **[Gel75]** | S. Gelbart. *Automorphic forms on adele groups.* Princeton University Press, 1975. |
| **[JL70]** | H. Jacquet and R. P. Langlands. *Automorphic forms on GL(2).* Lecture Notes in Mathematics, Vol. 114. Springer-Verlag, Berlin, 1970. |
| **[Kut80]** | Philip Kutzko. *The Langlands conjecture for Gl₂ of a local field.* Ann. of Math. (2), 112(2):381–412, 1980. |
| **[Kut84]** | P. C. Kutzko. *The exceptional representations of Gl₂.* Compositio Math., 51(1):3–14, 1984. |
| **[LW]** | David Loeffler and Jared Weinstein. *On the computation of local components of a newform.* preprint. |
| **[Tat79]** | J. Tate. *Number Theoretic Background.* Proc. Symp. Pure Math., 33, part 2:3–26, 1979. |

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Local-component computation of a newform **[LW]** | `LocalComponent` |
| Conductor / new-vector theory **[Cas73]** | `Conductor`, `CentralCharacter`, `IsMinimal` |
| Principal series structure (induction from the Borel) | `IsPrincipalSeries`, `PrincipalSeriesParameters` |
| Supercuspidal classification / cuspidal inducing data **[BH06]** | `IsSupercuspidal`, `CuspidalInducingDatum` |
| Local Langlands correspondence (Weil representation) **[JL70, Kut80, Kut84]** | `GaloisRepresentation`, `WeilRepresentation` |
| Admissible pairs `(E, χ)` for `p ≠ 2` **[BH06]** | `AdmissiblePair` |
