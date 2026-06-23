# Chapter 87 — Algebras With Involution

**Handbook part:** XI — Algebras
**Handbook pages:** 2665–2677 (PDF pages 2796–2811)

---

## Scope and overview

This chapter describes techniques for computing with *-algebras: algebras equipped with an
anti-automorphism x → x* of order at most 2 (an involution or star). For further theoretical
background see [Alb61] and [KMRT98].

The principal application is to isometry groups of systems of reflexive forms and to
intersections of classical groups. Group algebras of moderate dimension may also be treated
with these tools.

To any set of reflexive forms defined on a common vector space (a system of forms) one may
associate a matrix *-algebra called the adjoint algebra of the system. The group of units of
this adjoint algebra contains a natural subgroup of unitary elements — those x satisfying
x* = x⁻¹ — which coincides with the group of isometries of the system and with the
intersection of the general classical groups associated with those forms.

The StarAlgebras package underpins the chapter. The core algorithms are due to Peter
Brooksbank and James Wilson [BW11a, BW11b].

---

## 87.1 Introduction

Introductory section; no intrinsics. See the scope and overview above.

---

## 87.2 Algebras with Involution

Two general constructions produce *-algebras:

1. The algebra of adjoints of a system of reflexive (alternating, symmetric, or Hermitian)
   forms [φ₁, …, φₑ] defined on a common K-vector space V.
2. The group algebra K[G], where K is any ring and G is a finite group.

A constructor for simple *-algebras is also provided.

### 87.2.1 Reflexive Forms

A reflexive form on a K-vector space V is a bilinear function φ : V × V → K such that
φ(u,v) = 0 implies φ(v,u) = 0. By a theorem of Birkhoff and von Neumann there are three
similarity classes: alternating, symmetric, and Hermitian. Each form φ is represented by a
matrix F and a field automorphism α, so that φ(u,v) = u^α F v^tr. A matrix g ∈ GL(d,K)
is an isometry if g^α F g^tr = F, and a similarity if g^α F g^tr = aF for some scalar a.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsometryGroup(F : parameters)` | Returns the group of isometries of the (possibly degenerate) reflexive form represented by matrix F over a finite field F_{p^e}. Parameter `Auto` (RngIntElt, default 0): the Frobenius exponent f in x → x^{p^f}; default treats F as bilinear over its base ring. | — |
| `SimilarityGroup(F : parameters)` | Returns the group of similarities of the (possibly degenerate) reflexive form represented by matrix F over a finite field F_{p^e}. Parameter `Auto` (RngIntElt, default 0): same convention as above. | — |

### 87.2.2 Systems of Reflexive Forms

A system of forms is a sequence [φ₁, …, φₑ] where each φᵢ is a reflexive form on a common
K-vector space V. Systems arise naturally from sets of classical groups sharing a defining
module, and also from p-groups via the commutator map V × V → W (where V = G/N,
W = γ₂(G)/γ₃(G), and N = ⟨Φ(G), ζ_{n-1}(G)⟩).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PGroupToForms(G)` | Returns a system of forms associated to the p-group G. For matrix groups the input must be a class-2 p-group; PC-group input is preferred when readily available. | — |

*Worked examples: H87E1 (system of forms for a Sylow 7-subgroup of GL(3,7), via both PCGroup and matrix p-group).*

### 87.2.3 Basic Attributes of *-Algebras

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsStarAlgebra(A)` | Returns true if and only if A has an assigned involution. | — |
| `Star(A)` | Returns the involution map associated to the *-algebra A. | — |

### 87.2.4 Adjoint Algebras

For a nondegenerate system of forms S = [φ₁, …, φₑ] on a K-vector space V, the adjoint
algebra is

  Adj(S) = { (x, y) ∈ R × R^op : φᵢ(ux, v) = φᵢ(u, yv) for all u, v ∈ V, all i }

where R = End_K(V). Because the forms are reflexive, (x,y) ∈ Adj(S) iff (y,x) ∈ Adj(S).
Nondegeneracy forces y to be uniquely determined by x, so Adj(S) is identified with its
projection onto R, and x* := y defines an involution. The computation implements
[BW11a, Proposition 5.1] and [BW11b, Section 5].

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AdjointAlgebra(S : parameters)` | Given a sequence S containing a nondegenerate system of reflexive forms over a finite field, returns the *-algebra of adjoints. Individual forms in S may be degenerate. Parameter `Autos` (SeqEnum, default [0,…,0]): list of Frobenius exponents, one per form; default treats all forms as bilinear over their common base ring. | **[BW11a, Proposition 5.1]**; **[BW11b, Section 5]** |

*Worked examples: H87E2 (adjoint algebra of a pair of forms over GF(5²); accessing the involution via `Star(A)`).*

### 87.2.5 Group Algebras

If G is a finite group and R is any ring, the group algebra A = R[G] carries a natural
involution: for a = Σ_{g∈G} α_g g, define a* = Σ_{g∈G} α_g g⁻¹.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `StarOnGroupAlgebra(A)` | Attaches the natural involution (induced by inversion in the underlying group) to the group algebra A and returns it. | — |
| `GroupAlgebraAsStarAlgebra(R, G)` | Constructs the group algebra R[G] already equipped with the natural involution afforded by inversion in G. | — |

*Worked examples: H87E3 (ℤ[S₃] as a *-algebra; construction via `GroupAlgebraAsStarAlgebra` and via `GroupAlgebra` + `StarOnGroupAlgebra`).*

### 87.2.6 Simple *-Algebras

Artinian simple *-algebras (no proper *-invariant ideals) were classified by Albert [Alb61].
They fall into two flavours: classical (simple as algebras, arising as adjoints of nondegenerate
reflexive forms) and exchange (direct sum of two isomorphic simple algebras, involution
interchanging the factors). Following the Magma convention for classical forms, six names
are used:

- `"symplectic"` — defined by an alternating form
- `"orthogonalcircle"` — symmetric form in odd dimension
- `"orthogonalplus"` — symmetric form of maximal Witt index
- `"orthogonalminus"` — symmetric form of non-maximal Witt index
- `"unitary"` — Hermitian form
- `"exchange"` — exchange type

Involutions of `"unitary"` and `"exchange"` type are "of the second kind" (non-trivial on the
centre); the remaining four are "of the first kind" [KMRT98].

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SimpleStarAlgebra(name, d, K)` | Constructs the standard copy of the simple *-algebra of the given type name, defined naturally on a K-vector space of dimension d. | Classification of [Alb61] |

*Worked examples: H87E4 (standard exchange *-algebra of dimension 8 over GF(16); applying `Star`).*

---

## 87.3 Decompositions of *-Algebras

Every finite-dimensional K-algebra A has a Wedderburn decomposition A = J ⊕ W, where J is
the Jacobson radical and W is a semisimple subring. The Wedderburn procedure is adapted from
a Magma function by W. de Graaf for structure-constant algebras.

When A is a *-algebra and char(K) ≠ 2, a result of Taft [Taf57] guarantees a Wedderburn
decomposition A = J ⊕ T in which T is *-invariant (a Taft decomposition). The computation
follows Taft's original proof as described in [BW11a, Proposition 4.3].

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `WedderburnDecomposition(A)` | Returns the Jacobson radical J and a semisimple complement W to J in the *-algebra A. A may be a matrix algebra or a group algebra over any field. | Adapted from W. de Graaf's structure-constant algebra procedure |
| `TaftDecomposition(A)` | Returns the Jacobson radical J and a *-invariant Wedderburn complement T to J in the *-algebra A. Requires char(base ring of A) ≠ 2. A may be a matrix *-algebra or a group algebra. | Taft's theorem [Taf57]; **[BW11a, Proposition 4.3]** |

*Worked examples: H87E5 (Wedderburn and Taft decompositions of GF(5)[A₅]; verifying *-invariance of the Taft complement T but not of the general Wedderburn complement W).*

---

## 87.4 Recognition of *-Algebras

All functions in this section require that the base ring of the given algebra is a finite field
of odd order. They implement the methods described in [BW11a, Sections 4.2 and 4.3].

### 87.4.1 Recognition of Simple *-Algebras

Constructive recognition of a simple *-algebra A finds explicit inverse isomorphisms between A
and the standard copy returned by `SimpleStarAlgebra`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RecogniseClassicalSSA(A)` | Decides whether the matrix *-algebra A is a simple *-algebra of classical type. If so, returns the standard *-algebra T, a *-isomorphism A → T, and its inverse T → A. | **[BW11a, Sections 4.2–4.3]** |
| `RecogniseExchangeSSA(A)` | Decides whether the matrix *-algebra A is a simple *-algebra of exchange type. If so, returns the standard *-algebra T, a *-isomorphism A → T, and its inverse T → A. | **[BW11a, Sections 4.2–4.3]** |

*Worked examples: H87E6 (constructive recognition of a symplectic *-algebra from an adjoint algebra over GF(7); verification of the *-isomorphism).*

### 87.4.2 Recognition of Arbitrary *-Algebras

Constructive recognition of an arbitrary *-algebra A proceeds in three steps:
1. Find a Taft decomposition A = J ⊕ T (J = Jacobson radical, T *-invariant semisimple).
2. Decompose T = I₁ ⊕ … ⊕ Iₜ into minimal *-ideals.
3. Constructively recognise each simple *-algebra Iⱼ.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RecogniseStarAlgebra(A)` | Constructively recognises the *-algebra A (matrix *-algebra or group algebra). Also initiates recognition if not yet done, and is called implicitly by the access functions below. | **[BW11a, Sections 4.2–4.3]** |
| `IsSimpleStarAlgebra(A)` | Returns true if and only if A is a simple *-algebra. Initiates recognition if needed. | — |
| `SimpleParameters(A)` | Returns the parameters (as a sequence of tuples) that determine, up to *-isomorphism, the minimal *-ideals of the semisimple quotient A/J. Initiates recognition if needed. | — |
| `NormGroup(A)` | Returns the group of unitary elements of A: all units x ∈ A satisfying x* = x⁻¹. | **[BW11a, Section 5]** |

*Worked examples: H87E7 (distinguishing GF(5)[D₈] from GF(5)[Q₈] as *-algebras via `SimpleParameters`; both semisimple with four 1-dimensional and one 4-dimensional *-ideal, but differing in type "orthogonalplus" vs "symplectic"). H87E8 (distinguishing two p-groups of class 2 and order 43⁶ via `SimpleParameters` of their adjoint algebras).*

---

## 87.5 Intersections of Classical Groups

The principal application of the *-algebra machinery is computing the isometry group of a
system of forms, equivalently the intersection of classical groups defined on a common vector
space. The main functions implement the algorithms of [BW11a, Theorem 1.2] and
[BW11b, Theorem 1.1].

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsometryGroup(S : parameters)` | Given a sequence S of reflexive forms (a system of forms), returns the group of isometries of the system. Handles degenerate individual forms and degenerate systems. Parameter `Autos` (SeqEnum, default [0,…,0]): Frobenius exponents for each form. Parameter `DisplayStructure` (BoolElt, default false): if true, prints the structure of the isometry group. | **[BW11a, Theorem 1.2]**; **[BW11b, Theorem 1.1]** |
| `ClassicalIntersection(S)` | Given a sequence S of classical groups, each preserving (up to similarity) a unique nondegenerate reflexive form on a common finite vector space V, returns the intersection of the groups. It is not required that each group in S be the full isometry group. | **[BW11a, Theorem 1.2]**; **[BW11b, Theorem 1.1]** |

*Worked examples: H87E9 (isometry group of a system of forms for a Sylow 5-subgroup of Sp(4,5²), with `DisplayStructure`). H87E10 (intersection of Sp(F₁) and Ω⁻(F₂) over GF(3) via `ClassicalIntersection`).*

---

## 87.6 Bibliography

| Key | Reference |
|-----|-----------|
| **[Alb61]** | A. Adrian Albert. *Structure of Algebras.* American Mathematical Society, Providence, RI, 1961. Revised printing. |
| **[BW11a]** | Peter A. Brooksbank and James B. Wilson. Computing isometry groups of Hermitian maps. *Transactions of the American Mathematical Society*, 2011. To appear. |
| **[BW11b]** | Peter A. Brooksbank and James B. Wilson. Intersecting two classical groups. Preprint, 2011. |
| **[KMRT98]** | Max-Albert Knus, Alexander Merkurjev, Markus Rost, and Jean-Pierre Tignol. *The Book of Involutions.* American Mathematical Society, Providence, RI, 1998. Preface by Jacques Tits. |
| **[Taf57]** | E. J. Taft. Invariant Wedderburn factors. *Illinois Journal of Mathematics*, 1:565–573, 1957. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Isometry / similarity groups of a single reflexive form | `IsometryGroup(F)`, `SimilarityGroup(F)` |
| p-group commutator system of forms **[—]** | `PGroupToForms` |
| Adjoint algebra construction **[BW11a, Prop. 5.1; BW11b, §5]** | `AdjointAlgebra` |
| Natural involution on group algebras | `StarOnGroupAlgebra`, `GroupAlgebraAsStarAlgebra` |
| Albert's classification of simple *-algebras **[Alb61]** | `SimpleStarAlgebra` |
| Wedderburn decomposition (de Graaf's method) | `WedderburnDecomposition` |
| Taft's *-invariant complement theorem **[Taf57; BW11a, Prop. 4.3]** | `TaftDecomposition` |
| Constructive recognition of simple *-algebras **[BW11a, §§4.2–4.3]** | `RecogniseClassicalSSA`, `RecogniseExchangeSSA` |
| Constructive recognition of arbitrary *-algebras **[BW11a, §§4.2–4.3]** | `RecogniseStarAlgebra`, `IsSimpleStarAlgebra`, `SimpleParameters` |
| Unitary / norm group computation **[BW11a, §5]** | `NormGroup` |
| Isometry group of a system of forms **[BW11a, Thm. 1.2; BW11b, Thm. 1.1]** | `IsometryGroup(S)` |
| Intersection of classical groups **[BW11a, Thm. 1.2; BW11b, Thm. 1.1]** | `ClassicalIntersection` |
