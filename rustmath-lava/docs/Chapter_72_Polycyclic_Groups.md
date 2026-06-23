# Chapter 72 — Polycyclic Groups

**Handbook part:** X — Finitely-Presented Groups
**Handbook pages:** 2251–2288 (PDF pages 2382–2421)

---

## Scope and overview

Chapter 72 covers the Magma category `GrpGPC` (general polycyclic groups): possibly infinite
polycyclic groups defined by a consistent polycyclic presentation. The chapter distinguishes
this class from the finite solvable groups of `GrpPC` (Chapter 63) and explains when and why
to prefer `GrpGPC`.

A **polycyclic group** is a group G admitting a subnormal series G = G₁ ▷ G₂ ▷ … ▷ Gₙ₊₁ = 1
with each factor Gᵢ/Gᵢ₊₁ cyclic. The associated **polycyclic presentation** specifies power
relations aᵢᵐⁱ = wᵢ,ᵢ (for i in an index set I of generators with finite order) and conjugate
relations aⱼᵃⁱ = wᵢ,ⱼ (for 1 ≤ i < j ≤ n). A presentation is **consistent** if the
prescribed orders equal the actual factor orders. Every element has a unique **normal form**
a₁ᵉ¹ … aₙᵉⁿ, recovered by the collection algorithm.

The collection algorithm is by Volker Gebhardt **[Geb02]**; its cost grows logarithmically in
a bound on the absolute values of the exponents that arise during collection. Algorithms for
subgroup intersection, normalisers, centralisers, and conjugacy in nilpotent groups are based
on **[Lo98]**. The Fitting subgroup algorithm is due to Eick **[Eic01]**. A survey of basic
polycyclic-group algorithms is **[Sim94, ch. 9]**, which underpins most of the remaining
implementations. Infinite polycyclic groups are described as a comparatively new topic in
computational group theory; for finite polycyclic groups the `GrpPC` type (Chapter 63) is
preferred.

---

## 72.1 Introduction

Introductory section establishing the Magma category `GrpGPC`, the distinction from `GrpPC`,
and the relevant algorithmic references. No intrinsics are listed here.

---

## 72.2 Polycyclic Groups and Polycyclic Presentations

### 72.2.1 Introduction

Defines the theoretical basis: polycyclic series, normal form, consistency condition, and
Gebhardt's collection algorithm **[Geb02]**.

### 72.2.2 Specification of Elements

Elements of `GrpGPC` are words built from generators by product, power, conjugate, and
commutator operations.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `G ! Q` | Given polycyclic group `G` and an integer sequence `Q = [e₁, …, eₙ]` (with 0 ≤ eᵢ < mᵢ for i ∈ I), construct the element a₁ᵉ¹ … aₙᵉⁿ of `G`. | Direct normal-form construction. |
| `Identity(G)` / `Id(G)` / `G ! 1` | The identity element of the polycyclic group `G`. | — |

### 72.2.3 Access Functions for Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ElementToSequence(x)` / `Eltseq(x)` | For element `x = a₁ᵉ¹ … aₙᵉⁿ` in normal form, returns the sequence `[e₁, …, eₙ]`. | Collection to normal form **[Geb02]**. |
| `LeadingTerm(x)` | Returns `aᵢᵉⁱ` where i is the smallest index with eᵢ > 0; returns the identity if `x` is trivial. | Normal-form access. |
| `LeadingGenerator(x)` | Returns the generator `aᵢ` at the leading position (smallest i with eᵢ > 0); returns the identity if `x` is trivial. | Normal-form access. |
| `LeadingExponent(x)` | Returns the exponent `eᵢ` at the leading position; returns 0 if `x` is trivial. | Normal-form access. |
| `Depth(x)` | Returns the smallest i such that eᵢ > 0 (equivalently: the maximal i such that `x ∈ Gᵢ`); returns n+1 if `x` is trivial. | Normal-form access. |

*Worked examples: H72E1 (infinite dihedral group — element construction, `LeadingGenerator`, `LeadingExponent`, `Depth`, `ElementToSequence`).*

### 72.2.4 Arithmetic Operations on Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `g * h` | Product of elements `g` and `h` belonging to a common subgroup of a polycyclic group. Result lies in the smallest known common supergroup. | Collection to normal form **[Geb02]**. |
| `g *:= h` | Replaces `g` with `g * h`. | Collection to normal form **[Geb02]**. |
| `g ^ n` | The n-th power of element `g` (n a positive or negative integer). | Collection to normal form **[Geb02]**. |
| `g ^:= n` | Replaces `g` with `g ^ n`. | Collection to normal form **[Geb02]**. |
| `g / h` | Quotient `g * h⁻¹`; `g` and `h` must belong to a common subgroup. | Collection to normal form **[Geb02]**. |
| `g /:= h` | Replaces `g` with `g * h⁻¹`. | Collection to normal form **[Geb02]**. |
| `g ^ h` | Conjugate `h⁻¹ * g * h`; `g` and `h` must belong to a common subgroup. | Collection to normal form **[Geb02]**. |
| `g ^:= h` | Replaces `g` with its conjugate by `h`. | Collection to normal form **[Geb02]**. |
| `(g1, ..., gn)` | Left-normed commutator of `g₁, …, gₙ`, defined inductively: `(g₁, g₂) = g₁⁻¹ * g₂⁻¹ * g₁ * g₂` and `(g₁, …, gₙ) = ((g₁, …, gₙ₋₁), gₙ)`. | Collection to normal form **[Geb02]**. |

### 72.2.5 Operators for Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Order(x)` | The order of the element `x`. | Derived from the polycyclic presentation. |
| `Parent(x)` | The parent group `G` of element `x`. | — |

### 72.2.6 Comparison Operators for Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `g eq h` | True if elements `g` and `h` in a common polycyclic group are equal (i.e. have identical normal form). | Normal-form comparison. |
| `g ne h` | True if elements `g` and `h` are distinct. | Normal-form comparison. |
| `IsIdentity(g)` / `IsId(g)` | True if `g` is the identity element. | Normal-form test. |

### 72.2.7 Specification of a Polycyclic Presentation

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `quo< GrpGPC : F \| R : parameters >` | Given a free group `F` of rank n and a collection `R` of polycyclic relations, construct the polycyclic group `G` defined by `⟨X\|R⟩`. Relations may be power relations `aᵢᵐⁱ = wᵢ,ᵢ`, conjugate relations `aⱼᵃⁱᵉ = wᵉ·ᵢ,ⱼ`, or bare powers treated as trivial right-hand-sides. Returns `G` and the natural homomorphism `F → G`. Parameter `Check` (default `true`): verify consistency of the presentation. | Consistency check via collection **[Geb02]**. |
| `PolycyclicGroup< x1, ..., xn \| R : parameters >` | Construct polycyclic group `G` from a consistent polycyclic presentation with named generators. Returns `G` and a map from the free group on the `xᵢ` to `G`. Parameter `Check` (default `true`): verify consistency. Parameter `Class` (default `""`): set to `"GrpGPC"` to force `GrpGPC` when the presentation is also a valid power-conjugate presentation for a finite soluble group (otherwise `GrpPC` is returned). | Consistency check via collection **[Geb02]**. |

*Worked examples: H72E1 (infinite polycyclic group `⟨a,b,c | bᵃ = bc, (a,c), (b,c)⟩`, infinite dihedral group via `quo<>`); H72E2 (`PolycyclicGroup<>` with `Class := "GrpGPC"` for dihedral groups of orders 10 and 8).*

### 72.2.8 Properties of a Polycyclic Presentation

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsConsistent(G)` | True if the stored presentation for `G` is consistent, false otherwise. Useful after constructing with `Check := false`. | Consistency check via collection **[Geb02]**. |
| `IsIdenticalPresentation(G, H)` | True if the polycyclic presentations for `G` and `H` are syntactically identical. | Direct comparison. |
| `PresentationIsSmall(G)` | True if only small integers appear in the presentation of `G`. Required for `FPGroup` and `PCGroup` category transfers. | — |

---

## 72.3 Subgroups, Quotient Groups, Homomorphisms and Extensions

### 72.3.1 Construction of Subgroups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `sub< G \| L >` | The subgroup `H` of polycyclic group `G` generated by the elements/sequences/subgroups specified by `L`. Repetitions and the identity are removed. Returns `H` and the inclusion map `H → G`. | Polycyclic subgroup construction; see **[Sim94, ch. 9]**. |
| `ncl< G \| L >` | The normal closure `N` of the subgroup generated by `L` in `G`. Returns `N` and the inclusion map `N → G`. | Normal-closure algorithm; see **[Sim94, ch. 9]**. |

### 72.3.2 Coercions Between Groups and Subgroups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `G ! g` | Rewrite element `g ∈ H` (a subgroup of `G`) as an element of `G`. | Collection to normal form **[Geb02]**. |
| `H ! g` | Rewrite element `g ∈ G` as an element of subgroup `H ≤ G` (where `H` contains `g`). | Collection to normal form **[Geb02]**. |
| `K ! g` | Rewrite element `g ∈ H` as an element of `K`, where `H` and `K` are subgroups of a common group, both containing `g`. | Collection to normal form **[Geb02]**. |
| `InclusionMap(G, H)` | The inclusion map from subgroup `H` to `G`. | — |

*Worked example: H72E3 (computing the subnormal chain G₁ ▷ … ▷ G₄ for `⟨a,b,c | bᵃ = bc, …⟩` using `PCGenerators` and `Depth`).*

### 72.3.3 Construction of Quotient Groups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `quo< G \| L >` | The quotient `Q` of polycyclic group `G` by the normal closure of elements in `L`. Returns `Q` and the natural epimorphism `G → Q`. | Polycyclic quotient algorithm; see **[Sim94, ch. 9]**. |
| `G / N` | Given a normal subgroup `N` of `G`, the quotient group `G/N`. | Polycyclic quotient algorithm; see **[Sim94, ch. 9]**. |

### 72.3.4 Homomorphisms

For homomorphisms from a `GrpGPC` domain, the kernel can be computed when the codomain is of type `GrpGPC`, `GrpPC`, `GrpAb`, `GrpPerm`, `ModAlg`, `ModGrp`, or (if the image is finite) `GrpMat`. For `GrpFP` codomains the kernel may be computable when the domain is nilpotent.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `hom< P -> G \| S : parameters >` | The homomorphism from polycyclic group `P` to group `G` defined by assignment `S`. `S` may be: (i) a list of images of the polycyclic generators `P.1, …, P.n` (order matters); or (ii) a list/set of tuples `⟨xᵢ, yᵢ⟩` or arrow-pairs `xᵢ -> yᵢ` for a generating set `{x₁, …, xᵣ}` of `P` (order irrelevant). By default (`Check := true`) the homomorphism is verified by checking that images satisfy the defining relations. | Checked via collection **[Geb02]**; `Check := false` disables verification. |

*Worked example: H72E4 (dihedral group D₄ mapped onto Z₂ × Z₂; kernel computation).*

### 72.3.5 Construction of Extensions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DirectProduct(G, H)` | The direct product `K` of polycyclic groups `G` and `H`. Returns `K`, a sequence of inclusion maps `[Iᴳ: G → K, Iᴴ: H → K]`, and a sequence of projection maps `[Pᴳ: K → G, Pᴴ: K → H]`. | Standard polycyclic direct product construction. |

*Worked example: H72E5 (D₃ × D∞ as `GrpGPC`, subgroup and quotient construction using inclusion maps).*

### 72.3.6 Construction of Standard Groups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AbelianGroup(GrpGPC, Q)` | The abelian group Z_{n₁} × … × Z_{nᵣ} as a polycyclic group. Entries nᵢ = 0 give infinite cyclic factors; nᵢ > 1 give cyclic groups of order nᵢ. | Standard polycyclic presentation. |
| `CyclicGroup(GrpGPC, n)` | Cyclic group of order n for n > 0, or the infinite cyclic group for n = 0. | Standard polycyclic presentation. |
| `DihedralGroup(GrpGPC, n)` | Dihedral group of order 2n for n ≥ 3, or the infinite dihedral group for n = 0. | Standard polycyclic presentation. |
| `ElementaryAbelianGroup(GrpGPC, p, n)` | Elementary abelian group of order pⁿ (prime p, positive integer n). | Standard polycyclic presentation. |
| `ExtraSpecialGroup(GrpGPC, p, n : parameters)` | Extra-special group of order p^{2n+1}. Parameter `Type` (default `"+"`): `"+"` gives (for p=2) the central product of n copies of D₈, or (for p>2) the unique extra-special group of exponent p; `"-"` gives (for p=2) the central product of Q₈ and n−1 copies of D₈, or (for p>2) the unique extra-special group of exponent p². | Standard polycyclic presentation. |
| `FreeAbelianGroup(GrpGPC, n)` | Free abelian group of rank n as a polycyclic group. | Standard polycyclic presentation. |
| `FreeNilpotentGroup(r, e)` | Free nilpotent group of rank r and class e as a polycyclic group. | Standard polycyclic presentation. |

---

## 72.4 Conversion between Categories

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AbelianGroup(G)` | Converts the abelian polycyclic group `G` to a `GrpAb` representation `A`. Returns `A` and the isomorphism `G → A`. | — |
| `FPGroup(G)` | Converts `G` to a `GrpFP` representation `F`. Returns `F` and the isomorphism `F → G`. Requires `PresentationIsSmall(G)`. | — |
| `PCGroup(G)` | Converts the finite polycyclic group `G` to a `GrpPC` representation `F`. Returns `F` and the isomorphism `G → F`. Requires `PresentationIsSmall(G)`. | — |
| `GPCGroup(G)` | Converts a solvable group `G` (of type `GrpPerm`, `GrpMat` (finite only), `GrpAb`, or `GrpPC`) to a `GrpGPC` representation `P`. Returns `P` and the isomorphism `G → P`. | — |

*Worked example: H72E6 (finite solvable matrix group converted to `GrpGPC` via `GPCGroup`, combined with infinite dihedral group in a direct product, normal closure and quotient computed, result converted back to `GrpPC` via `PCGroup`).*

---

## 72.5 Access Functions for Groups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `G . i` | The i-th polycyclic generator of `G` for i > 0; the inverse of the \|i\|-th generator for i < 0; the identity for i = 0. | — |
| `Generators(G)` / `PCGenerators(G)` | An indexed set of the polycyclic generators of `G`. | — |
| `Generators(H, G)` / `PCGenerators(H, G)` | An indexed set of the polycyclic generators of subgroup `H` expressed as elements of `G`. | — |
| `NumberOfGenerators(G)` / `Ngens(G)` / `NumberOfPCGenerators(G)` / `NPCgens(G)` | The number of polycyclic generators of `G`. | — |
| `PCExponents(G)` | Sequence Q where `Q[i] = mᵢ = |Gᵢ/Gᵢ₊₁|` for finite factors (i ∈ I), and `Q[i] = 0` for infinite factors. | — |
| `HirschNumber(G)` | The Hirsch number of `G`: the number of infinite cyclic factors, equal to n − \|I\|. `G` is finite if and only if its Hirsch number is 0. | — |

---

## 72.6 Set-Theoretic Operations in a Group

### 72.6.1 Functions Relating to Group Order

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `FactoredIndex(G, H)` | The factored index of `H` in `G` (where `H` has finite index). | — |
| `FactoredOrder(G)` | The factored order of finite `G`. | — |
| `Index(G, H)` | The index of subgroup `H` in `G`, as an ordinary integer. | — |
| `Order(G)` / `#G` | The order of `G`, as an ordinary integer. | — |

### 72.6.2 Membership and Equality

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `g in G` | True if element `g` belongs to group `G`. | Normal-form membership test. |
| `g notin G` | True if element `g` does not belong to `G`. | Normal-form membership test. |
| `S subset G` | True if set `S` of elements (from a group `H` sharing a covering group with `G`) is a subset of `G`. | Normal-form membership test. |
| `S notsubset G` | True if `S` is not a subset of `G`. | Normal-form membership test. |
| `H subset G` | True if subgroup `H` (sharing a covering group with `G`) is a subgroup of `G`. | Normal-form membership test. |
| `H notsubset G` | True if `H` is not a subgroup of `G`. | Normal-form membership test. |
| `G eq H` | True if groups `G` and `H` (sharing a covering group) are equal. | Normal-form comparison. |
| `G ne H` | True if groups `G` and `H` are distinct. | Normal-form comparison. |

### 72.6.3 Set Operations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Representative(G)` / `Rep(G)` | A representative element of `G`. | — |
| `RandomProcess(G)` | Creates a process for generating pseudo-random elements of `G`. Uses an expansion procedure storing N elements (N = max(n+1, Slots)) that form a generating set; `Scramble` (default 100) iterations are run before returning. Parameters: `Slots` (default 10), `Scramble` (default 100). Caution: quality may be poor for infinite polycyclic groups. | Expansion/product-replacement method **[CLGM+95]**. |
| `Random(P)` | Returns a pseudo-random element of `G` from process `P` (created by `RandomProcess(G)`). | Product-replacement step **[CLGM+95]**. |
| `Random(G)` / `Random(G, max)` | A pseudo-random element of `G` chosen by selecting a random exponent vector in normal form. Exponents for generators without a power relation have absolute value ≤ max (default 10). Distribution is uniform only for finite `G`. | Random exponent vector. |

---

## 72.7 Coset Spaces

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CosetTable(G, H)` | The right coset table for `G` over subgroup `H` of finite index: a map `{1, …, \|G:H\|} × G → {1, …, \|G:H\|}` describing the right-multiplication action of `G` on right cosets. The enumeration matches that of `Transversal`/`RightTransversal`. | Polycyclic coset enumeration. |
| `Transversal(G, H)` / `RightTransversal(G, H)` | Returns: (a) an indexed set `T` forming a right transversal of `G` over `H` (matching `CosetTable`'s enumeration); (b) the transversal map φ: G → T where φ(g) = tᵢ with g ∈ H * tᵢ. | Polycyclic transversal algorithm. |
| `CosetAction(G, H)` | Given `H ≤ G` of finite index, the permutation representation induced by the right-multiplication action of `G` on right cosets of `H`. Returns: (a) the homomorphism `f: G → L ≤ Sym(|G:H|)`; (b) the epimorphic image `L`; (c) the kernel `K`. | Polycyclic coset action. |
| `CosetImage(G, H)` | The epimorphic image `L` of the permutation representation of `G` acting on right cosets of `H` (finite index). | Polycyclic coset action. |
| `CosetKernel(G, H)` | The kernel `K` of the permutation representation `f: G → L ≤ Sym(|G:H|)` induced by right-coset action. | Polycyclic coset action. |

*Worked examples: H72E7 (right transversal of a subgroup of the infinite dihedral group, constructing the coset table and coset action function); H72E8 (non-faithful permutation representation of `⟨a,b,c | bᵃ = bc, …⟩` via `CosetAction`, kernel expressed in terms of generators).*

---

## 72.8 The Subgroup Structure

### 72.8.1 General Subgroup Constructions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `H ^ g` / `Conjugate(H, g)` | The conjugate `g⁻¹ * H * g` of subgroup `H` by element `g`, where `H` and `g` belong to a common group. | Polycyclic conjugation. |
| `H ^ G` / `ncl< G \| H >` / `NormalClosure(G, H)` | The normal closure of subgroup `H` in `G`. | Normal-closure algorithm; see **[Sim94, ch. 9]**. |
| `CommutatorSubgroup(G, H, K)` / `CommutatorSubgroup(H, K)` | The commutator subgroup `[H, K]`, where `H` and `K` are subgroups of a common group `G`. | Polycyclic commutator subgroup construction. |

### 72.8.2 Subgroup Constructions Requiring a Nilpotent Covering Group

These functions require that the groups involved are contained in a common nilpotent group. Based on algorithms from **[Lo98]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `H meet K` | Intersection of subgroups `H` and `K` contained in a common nilpotent group `G`. | Intersection in finitely generated nilpotent groups **[Lo98]**. |
| `H meet:= K` | Replaces `H` with the intersection of `H` and `K` (same nilpotency requirement). | **[Lo98]**. |
| `Centraliser(G, g)` / `Centralizer(G, g)` | The subgroup of `G` centralising element `g`; `g` and `G` must be in a common nilpotent group. | Centraliser in nilpotent groups **[Lo98]**. |
| `Centraliser(G, H)` / `Centralizer(G, H)` | The subgroup of `G` centralising subgroup `H`; both must be in a common nilpotent group. | Centraliser in nilpotent groups **[Lo98]**. |
| `Core(G, H)` | The maximal normal subgroup of the nilpotent group `G` contained in subgroup `H`. | Core in nilpotent groups **[Lo98]**. |
| `Normaliser(G, H)` / `Normalizer(G, H)` | The subgroup of `G` normalising `H`; both must be in a common nilpotent group. | Normaliser in nilpotent groups **[Lo98]**. |

*Worked examples: H72E9 (nilpotent group on 5 generators, commutator subgroups, intersection via `meet`); H72E10 (D₁₆ ≀ 2 — normalisers and centralisers).*

---

## 72.9 General Group Properties

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsAbelian(G)` | True if `G` is abelian. | Derived from the polycyclic presentation. |
| `IsCyclic(G)` | True if `G` is cyclic. | Derived from the polycyclic presentation. |
| `IsElementaryAbelian(G)` | True if `G` is an abelian p-group of exponent p for some prime p. | Derived from the polycyclic presentation. |
| `IsFinite(G)` | True if `G` is finite (equivalently, Hirsch number is 0). | Hirsch number test. |
| `IsNilpotent(G)` | True if `G` is nilpotent. | Algorithm from **[Lo98]**. |
| `IsPerfect(G)` | True if `G` is perfect; every polycyclic group is perfect if and only if it is trivial. | Derived subgroup test. |
| `IsSimple(G)` | True if `G` is simple; a polycyclic group is simple iff it is cyclic of prime order. | Order and cyclic test. |
| `IsSoluble(G)` / `IsSolvable(G)` | True (always, since every polycyclic group is solvable). | — |

### 72.9.1 General Properties of Subgroups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsCentral(G, H)` | True if subgroup `H` of `G` lies in the centre of `G`. | Normal-form test. |
| `IsNormal(G, H)` | True if `H` is normal in `G`. | Normal-form test. |

### 72.9.2 Properties of Subgroups Requiring a Nilpotent Covering Group

Based on algorithms from **[Lo98]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsConjugate(G, H, K)` | For groups `G`, `H`, `K` with a common nilpotent covering group: true if there exists `c ∈ G` with `H^c = K`; returns the conjugating element `c` as second value. | Conjugacy in nilpotent groups **[Lo98]**. |
| `IsSelfNormalising(G, H)` / `IsSelfNormalizing(G, H)` | True if subgroup `H` of the nilpotent group `G` satisfies `N_G(H) = H`. | Normaliser computation **[Lo98]**. |

*Worked examples: H72E9 (subgroup properties in a nilpotent group; `IsCyclic`, `IsCentral`, intersection via `meet`); H72E10 (conjugacy tests in D₁₆ ≀ 2).*

---

## 72.10 Normal Structure and Characteristic Subgroups

### 72.10.1 Characteristic Subgroups and Subgroup Series

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Centre(G)` / `Center(G)` | The centre of `G`. For nilpotent `G`: centraliser algorithm **[Lo98]**. Otherwise: simultaneous fixed-point space of the generator action on the centre of the Fitting subgroup **[Eic01]**. | **[Lo98]** (nilpotent), **[Eic01]** (general). |
| `DerivedLength(G)` | The derived length of `G`. | Iterated derived subgroup. |
| `DerivedSeries(G)` | The derived series of `G` as a sequence of subgroups. | Iterated derived subgroup. |
| `DerivedSubgroup(G)` / `DerivedGroup(G)` | The derived subgroup `G'` of `G`. | Polycyclic commutator subgroup construction. |
| `EFASeries(G)` | A normal series of `G` whose factors are either elementary abelian p-groups or free abelian groups. | Polycyclic series refinement. |
| `FittingLength(G)` | The Fitting length of `G`: the smallest k such that Fₖ = G, where F₀ = 1 and Fᵢ/Fᵢ₋₁ = Fit(G/Fᵢ₋₁). Every polycyclic group has a finite Fitting length. | **[Eic01]**. |
| `FittingSeries(G)` | The Fitting series `[F₀, …, Fₖ]` as a sequence of subgroups of `G`. | **[Eic01]**. |
| `FittingSubgroup(G)` / `FittingGroup(G)` | The Fitting subgroup of `G`: the maximal nilpotent normal subgroup. | Algorithm from **[Eic01]**. |
| `HasComputableLCS(G)` | True if the lower central series of `G` is computable (i.e. terminates); use to avoid runtime errors in loops. | Termination check **[Lo98]**. |
| `LowerCentralSeries(G)` | The lower central series of `G` as a sequence of subgroups. May fail for infinite polycyclic groups not satisfying the descending chain condition; use `HasComputableLCS` first. | Algorithm from **[Lo98]**. |
| `NilpotencyClass(G)` | The nilpotency class of `G`; returns −1 if `G` is not nilpotent. | Via lower central series **[Lo98]**. |
| `NilpotentPresentation(G)` | For a nilpotent polycyclic group `G`: returns a group `N` isomorphic to `G` given by a nilpotent polycyclic presentation (where each `Gᵢ₊₁` is normal in `G` and `Gᵢ/Gᵢ₊₁` is central in `G/Gᵢ₊₁`), together with the isomorphism `G → N`. The nilpotent series is obtained by refining the lower central series. | Refinement of lower central series **[Lo98]**. |
| `SemisimpleEFASeries(G)` | A normal series of `G` whose factors are either elementary abelian p-groups semisimple as Fₚ[G]-modules or free abelian groups semisimple as Q[G]-modules. A refinement of `EFASeries`. | Polycyclic series refinement. |
| `UpperCentralSeries(G)` | The upper central series `[Z₀, …, Zₖ]` of `G`, where Z₀ = 1 and Zᵢ/Zᵢ₋₁ = Z(G/Zᵢ₋₁). Every polycyclic group has a finite upper central series (ascending chain condition). | Iterated centre computation. |

*Worked example: H72E11 (dihedral group D₃₂ — `IsNilpotent`, `NilpotencyClass`, `LowerCentralSeries` with generators expressed in D₃₂, `NilpotentPresentation`; also symmetric group S₃ showing `HasComputableLCS` / `LowerCentralSeries` for a non-nilpotent group).*

### 72.10.2 The Abelian Quotient Structure of a Group

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AbelianQuotient(G)` | The maximal abelian quotient `G/G'` as a `GrpAb`. Returns the quotient and the natural epimorphism. | Polycyclic abelianisation. |
| `AbelianQuotientInvariants(G)` / `AQInvariants(G)` | Sequence of invariants of `G/G'`; infinite cyclic factors are represented by 0. | Polycyclic abelianisation. |
| `ElementaryAbelianQuotient(G, p)` | The maximal p-elementary abelian quotient of `G` as a `GrpAb`. Returns the quotient and the natural epimorphism. | Polycyclic p-abelianisation. |
| `FreeAbelianQuotient(G)` | The maximal free abelian quotient of `G` as a `GrpAb`. Returns the quotient and the natural epimorphism. | Polycyclic free abelianisation. |

---

## 72.11 Conjugacy

All functions in this section require a nilpotent covering group. Based on **[Lo98]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsConjugate(G, g, h)` | For elements `g`, `h` and group `G` contained in a common nilpotent group: true if there exists `c ∈ G` with `g^c = h`; returns the conjugating element as second value. | Conjugacy of elements in nilpotent groups **[Lo98]**. |
| `IsConjugate(G, H, K)` | For groups `G`, `H`, `K` with a common nilpotent covering group: true if there exists `c ∈ G` with `H^c = K`; returns the conjugating element as second value. | Conjugacy of subgroups in nilpotent groups **[Lo98]**. |

*Worked example: H72E12 (D₁₆ ≀ 2 — conjugacy of subgroups D₁, D₂ and elements b, d⁻¹; negative results in the subgroup D₃).*

---

## 72.12 Representation Theory

Functions for creating R[G]-modules from a polycyclic group G acting by conjugation on factors of normal series. For the full module API see Chapter 89; `GModuleAction` extracts the associated matrix representation.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `EFAModuleMaps(G)` | Returns a sequence `[f₁, …, fᵣ]` of natural epimorphisms `fᵢ: Nᵢ → Mᵢ` where `G = N₁ ▷ … ▷ Nᵣ₊₁ = 1` is the EFA-series and each Mᵢ is a Z[G]-module (if free abelian) or Fₚ[G]-module (if p-elementary abelian). Kernels of fᵢ are computable, giving preimages of submodules as normal subgroups of G. | EFA-series construction. |
| `EFAModules(G)` | Returns the sequence `[M₁, …, Mᵣ]` of Rᵢ[G]-modules (Rᵢ ∈ {Fₚ, Z}) for the EFA-series of `G`. | EFA-series construction. |
| `GModule(G, A, p)` / `GModule(G, A)` | For `A` a normal subgroup of `G`: the Fₚ[G]-module (p prime) or Z[G]-module (p = 0, or omitted when A/A' is free abelian or p-elementary abelian) corresponding to conjugation action of `G` on the maximal p-elementary abelian (or free abelian) quotient of `A`. Returns the module `M` and epimorphism `π: A → M`; kernel of π is computable. | Conjugation action on abelian quotient. |
| `GModule(G, A, B, p)` / `GModule(G, A, B)` | As `GModule(G, A, p)` but for the quotient `A/B` with `B < A` both normal in `G`. Returns the module `M` and epimorphism `π: A → M`; kernel of π gives normal subgroups of G between A and B. | Conjugation action on abelian quotient of A/B. |
| `GModulePrimes(G, A)` | For `A` normal in `G`: determines those primes p for which the Fₚ[G]-module Mₚ (maximal p-elementary abelian quotient of A) is nontrivial. Returns a multiset S: if 0 ∉ S the maximal abelian quotient of A is finite and the multiplicity of p equals dim(Mₚ); if 0 ∈ S with multiplicity m there are m copies of Z in A/A'. | Abelian quotient analysis. |
| `GModulePrimes(G, A, B)` | As `GModulePrimes(G, A)` but for the quotient `A/B`, with `B < A` both normal in `G`. | Abelian quotient analysis. |
| `SemisimpleEFAModuleMaps(G)` | Returns a sequence `[f₁, …, fᵣ]` of epimorphisms onto the semisimple EFA-series modules (factors either semisimple Q[G]-modules or semisimple Fₚ[G]-modules). A refinement of `EFAModuleMaps`. | Semisimple EFA-series construction. |
| `SemisimpleEFAModules(G)` | Returns the sequence `[M₁, …, Mᵣ]` of semisimple Rᵢ[G]-modules for the semisimple EFA-series of `G`. | Semisimple EFA-series construction. |

*Worked examples: H72E13 (`GModulePrimes` and `GModule` for a normal subgroup H, submodules via `Submodules`, preimages as normal subgroups); H72E14 (`GModulePrimes` / `GModule` loop over all relevant primes, checking decomposability); H72E15 (Fitting subgroup as intersection of kernels of semisimple EFA module actions via `SemisimpleEFAModules` and `GModuleAction`); H72E16 (`EFAModuleMaps`, extracting a random submodule, its preimage as a normal subgroup).*

---

## 72.13 Power Groups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Parent(G)` | The `PowerStructure` of category `GrpGPC`. | — |
| `PowerGroup(G)` | The set of all subgroups of `G`. Specifying this as the universe of a set of subgroups avoids ambiguity when the automatic universe would be the parent structure of `G`. | — |

---

## 72.14 Bibliography

| Key | Reference |
|-----|-----------|
| **[CLGM+95]** | Frank Celler, Charles R. Leedham-Green, Scott H. Murray, Alice C. Niemeyer, and E. A. O'Brien. *Generating random elements of a finite group.* Comm. Algebra, **23**(13):4931–4948, 1995. |
| **[Eic01]** | Bettina Eick. *On the Fitting subgroup of a polycyclic-by-finite group and its applications.* J. Algebra, **242**(1):176–187, 2001. |
| **[Geb02]** | Volker Gebhardt. *Efficient collection in infinite polycyclic groups.* J. Symbolic Comput., **34**(3):213–228, 2002. |
| **[Lo98]** | Eddie H. Lo. *Finding intersections and normalizers in finitely generated nilpotent groups.* J. Symbolic Comput., **25**(1):45–59, 1998. |
| **[Sim94]** | Charles C. Sims. *Computation with finitely presented groups.* Cambridge University Press, Cambridge, 1994. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Collection to normal form (Gebhardt) **[Geb02]** | All element arithmetic (`*`, `/`, `^`, `meet:=`, coercions), `ElementToSequence`, `Eltseq`, `IsConsistent`, `quo<>`, `PolycyclicGroup<>` |
| Polycyclic presentation construction and consistency **[Sim94, ch. 9]** | `quo<>`, `PolycyclicGroup<>`, `IsConsistent`, `sub<>`, `ncl<>`, `quo<G\|L>`, `G/N` |
| Intersection and normaliser in nilpotent groups **[Lo98]** | `H meet K`, `H meet:= K`, `Centraliser`, `Centralizer`, `Core`, `Normaliser`, `Normalizer`, `IsNilpotent`, `IsConjugate`, `IsSelfNormalising`, `LowerCentralSeries`, `HasComputableLCS`, `NilpotentPresentation`, `UpperCentralSeries`, `Centre`/`Center` (nilpotent case) |
| Fitting subgroup of polycyclic-by-finite groups **[Eic01]** | `FittingSubgroup`, `FittingGroup`, `FittingLength`, `FittingSeries`, `Centre`/`Center` (non-nilpotent case) |
| Product-replacement (random elements) **[CLGM+95]** | `RandomProcess`, `Random(P)` |
| EFA-series and module representations | `EFASeries`, `EFAModules`, `EFAModuleMaps`, `SemisimpleEFASeries`, `SemisimpleEFAModules`, `SemisimpleEFAModuleMaps`, `GModule`, `GModulePrimes` |
| Category transfers | `AbelianGroup`, `FPGroup`, `PCGroup`, `GPCGroup` |
