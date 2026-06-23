# Chapter 63 — Finite Soluble Groups

**Handbook part:** IX — Finite Groups
**Handbook pages:** 1793–1867 (PDF pages 1922–2001)

---

## Scope and overview

Any finite soluble group possesses a subnormal series with cyclic factors, giving rise to
**polycyclic presentations**. Magma uses the specific form called a **power-conjugate
presentation** (pc-presentation): generators a1, …, an with power relations ajpj = wjj
and conjugate relations ajai = wij, where wij are words in later generators. The Magma
category for groups represented in this way is **GrpPC** (pc-groups).

The **word problem** in a consistent pc-presentation is solved algorithmically by the
**collection algorithm**: given any word in the pc-generators, collection reduces it to the
unique normal form a1α1 · · · anαn. Group multiplication is implemented via collection,
making pc-groups the preferred representation for intensive computation with finite soluble
and p-groups.

Over roughly two decades a substantial body of efficient algorithms has been developed for
pc-groups, covering conjugacy, subgroups, automorphism groups, character theory, p-group
generation, and cohomology. While most functions apply to any soluble group, some are
restricted to p-groups and are noted in the text. Magma recommends using GrpPC whenever
intensive calculation with a finite soluble group is required.

Three types of pc-presentation coexist internally: the **user presentation** (for input/output),
the **conditioned presentation** (used in computation, guarantees elementary abelian factors),
and the **special presentation** (exhibits a Sylow system and LG-series refinement, required
by several algorithms). A fourth format, the **compact presentation**, stores the group as a
compact integer sequence for efficient library storage and retrieval.

---

## 63.1 Introduction

### 63.1.1 Power-Conjugate Presentations

A **power-conjugate presentation** (pc-presentation) for a finite soluble group G is a
presentation of the form

```
< a1, …, an | ajpj = wjj  (1 ≤ j ≤ n),  ajai = wij  (1 ≤ i < j ≤ n) >
```

where pj is the least prime such that ajpj ∈ ⟨aj+1, …, an⟩, and wij are words in generators
ai+1, …, an. When the presentation satisfies the **consistency condition**, every element of G
has a unique normal form a1α1 · · · anαn, 0 ≤ αi < pi.

The **collection algorithm** reduces any word in the pc-generators to its normal form,
implementing group multiplication. The category GrpPC stores groups internally as
pc-presentations and always displays the pc-presentation when a group is printed.

---

## 63.2 Creation of a Group

A GrpPC group can be created via built-in construction functions, directly from a
pc-presentation, or by converting an existing group in another category.

### 63.2.1 Construction Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CyclicGroup(GrpPC, n)` | Cyclic group of order n as a pc-group. | Direct construction. |
| `AbelianGroup(GrpPC, Q)` | Abelian group Cn1 × Cn2 × … × Cnr where Q = [n1, …, nr], as a pc-group. | Direct product of cyclic groups. |
| `DihedralGroup(GrpPC, n)` | Dihedral group of order 2n as a pc-group. | Direct construction. |
| `ExtraSpecialGroup(GrpPC, p, n : Type)` | Extra-special group of order p2n+1 as a pc-group. `Type := "+"` (default): for p = 2, central product of n copies of D8; for p > 2, exponent p. `Type := "-"`: for p = 2, central product of Q8 and n−1 copies of D8; for p > 2, exponent p2. | Direct construction by type. |

*Worked examples: H63E1 (cyclic group C12, properties), H63E2 (group of order 80 via PolycyclicGroup and quo).*

### 63.2.2 Definition by Presentation

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PolycyclicGroup< x1, …, xn \| R : Check, ExponentLimit, Class >` | Construct the soluble group G defined by the pc-presentation ⟨x1, …, xn \| R⟩. R may contain power relations ajpj = wjj, conjugate relations ajai = wij, or commutator relations (aj, ai) = wij (but not mixed). Returns G and a map from the free group of rank n to G. `Check` (default `true`): verify consistency. `ExponentLimit` (default 20): precompute products to speed collection. `Class` (default empty): setting `"GrpPC"` causes an invalid presentation to raise a runtime error rather than fall through to GrpGPC. | Collection algorithm (consistency check); falls back to GrpGPC if invalid and Class not set. |
| `quo< GrpPC : F \| R : Check, ExponentLimit >` | Given free group F and pc-relations R, construct the quotient pc-group. Same parameters as PolycyclicGroup. Returns G and the natural map F → G. | Same as PolycyclicGroup. |

### 63.2.3 Possibly Inconsistent Presentations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsConsistent(G)` | Returns `true` if G has a consistent pc-presentation, `false` otherwise. Intended for use with `Check := false` in the constructors when testing families of presentations. | Consistency check for pc-presentations. |

*Worked example: H63E3 (testing consistency over a family of presentations).*

---

## 63.3 Basic Group Properties

### 63.3.1 Infrastructure

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `G . i` | The i-th pc-generator of G. Negative i gives the inverse; `G.0` gives the identity. | — |
| `Generators(G)` | Set of defining generators. For p-groups, guaranteed to be a minimal generating set; for non-p-groups, the full pc-generator set. | — |
| `NumberOfGenerators(G)` / `Ngens(G)` | Number of defining generators. | — |
| `PCGenerators(G)` | Indexed set of all pc-generators of G. | — |
| `NumberOfPCGenerators(G)` / `NPCGenerators(G)` / `NPCgens(G)` | Number of pc-generators of G. | — |
| `PCPrimes(G)` | Sequence [p1, …, pn] of primes associated with each pc-generator. | — |

### 63.3.2 Numerical Invariants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Order(G)` / `#G` | Order of G as an ordinary integer. | Read from the pc-presentation (product of primes). |
| `FactoredOrder(G)` | Factored order of G as a sequence of prime-power pairs. | — |
| `Exponent(G)` | Exponent of G. | — |

### 63.3.3 Predicates

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsAbelian(G)` | True if G is abelian. | From pc-presentation relations. |
| `IsCyclic(G)` | True if G is cyclic. | — |
| `IsElementaryAbelian(G)` | True if G is elementary abelian. | — |
| `IsNilpotent(G)` | True if G is nilpotent. | — |
| `IsPerfect(G)` | True if G is perfect (always false for non-trivial pc-groups). | — |
| `IsSimple(G)` | True if G is simple. | — |
| `IsSoluble(G)` / `IsSolvable(G)` | True if G is soluble; always `true` for pc-groups. | — |
| `IsTrivial(G)` | True if G has order 1. | — |
| `IsSpecial(G)` | For a p-group G: true if G is special. | — |
| `IsExtraSpecial(G)` | For a p-group G: true if G is extra-special. | — |

*Worked example: H63E4 (extra-special 3-group of exponent 9, FactoredOrder, Exponent, IsNilpotent).*

---

## 63.4 Homomorphisms

Arbitrary homomorphisms between pc-groups are defined with the `hom<>` constructor,
which can take images of any generating set (not just pc-generators). Magma verifies that
the images define a valid homomorphism and supports computation of kernels and inverse
images.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `hom< G -> H \| L : Check >` | Homomorphism φ : G → H defined by the list L. L may be: (a) list/set/sequence of 2-tuples ⟨gi, hi⟩; (b) arrow pairs gi → hi; or (c) ordered sequence of n images for the n pc-generators. `Check` (default `true`): verify the map is a homomorphism. | — |
| `IsHomomorphism(G, H, L)` | Conditional form: returns `true` and the map if L defines a homomorphism; otherwise `false`. | — |
| `IdentityHomomorphism(G)` | Identity map from G to G. | — |
| `Kernel(f)` | Kernel of a homomorphism f between pc-groups; returned as a pc-group subgroup of the domain. | — |
| `Homomorphisms(G, H)` | For finite abelian G and H: all elements of Hom(G, H) as a sequence of Magma maps. | Uses the GrpAb Hom machinery. |

*Worked example: H63E5 (projection homomorphism from S4 to a complement, kernel computation).*

---

## 63.5 New Groups from Existing

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DirectProduct(G, H)` | Direct product K = G × H. Returns K, a sequence of inclusion maps [IG, IH], and a sequence of projection maps [PG, PH]. First pc-generators of K correspond to G, remaining to H. | Direct pc-presentation construction. |
| `DirectProduct(Q)` | Direct product of pc-groups in the non-empty sequence Q, with inclusion and projection maps. | — |
| `Extension(G, H, f)` | Split extension K of the pc-group G by pc-group H, with action of H on G given by the sequence of maps f (f[i] defines the action of the i-th pc-generator of H). K has a normal subgroup G˜ ≅ G with K/G˜ ≅ H. | Polycyclic extension construction. |
| `Extension(M, H)` | Split extension K of G-module M (d-dimensional over GF(p)) by pc-group H, using the H-module action. | — |
| `Extension(G, H, f, t)` | Non-split extension K of G by H; action via f, tails t given as a sequence [x11, x21, x22, …, xss] or as a set of tuples ⟨i, j, xij⟩ for non-trivial tails. | Polycyclic extension with non-trivial tails. |
| `Extension(M, H, t)` | Non-split extension of G-module M by H with tails t. | — |
| `IsExtension(G, H, f)` / `IsExtension(M, H)` / `IsExtension(G, H, f, t)` / `IsExtension(M, H, t)` | Conditional forms of the Extension functions: return a boolean (and the extension group if successful) rather than raising a runtime error on failure. | — |
| `WreathProduct(G, H)` | Wreath product of pc-groups G and H using the regular permutation representation of H. | — |
| `WreathProduct(G, H, f)` | Wreath product where the action of H is given by f (a homomorphism H → GrpPerm or a sequence of permutations). | — |

*Worked examples: H63E6 (split and non-split extensions of C4 on an elementary abelian 3-group); H63E7 (Cossey–Hawkes example [CH00], conjugacy class sizes in a derived-length-3 group).*

---

## 63.6 Elements

Elements of a pc-group are always stored and printed in normal form a1α1 · · · anαn.

### 63.6.1 Definition of Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `G ! Q` | Construct the element x = a1α1 · · · anαn from the exponent sequence Q = [α1, …, αn], 0 ≤ αi < pi. | — |
| `ElementToSequence(x)` / `Eltseq(x)` | Return the exponent sequence [α1, …, αn] for the element x = a1α1 · · · anαn. | — |
| `Identity(G)` / `Id(G)` / `G ! 1` | The identity element of G. | — |

### 63.6.2 Arithmetic Operations on Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `g * h` / `g *:= h` | Product of elements g and h (in a common covering pc-group). | Collection algorithm. |
| `g ^ n` / `g ^:= n` | n-th power of g (n a positive or negative integer). | Repeated collection. |
| `g / h` / `g /:= h` | Quotient g · h−1. | Collection algorithm. |
| `g ^ h` / `g ^:= h` | Conjugate h−1 · g · h of g by h. | Collection algorithm. |
| `(g1, …, gn)` | Left-normed commutator of g1, …, gn. | Collection algorithm. |

### 63.6.3 Properties of Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Order(x)` | Order of element x. | — |
| `Parent(x)` | The parent group G of element x. | — |

### 63.6.4 Predicates for Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `g eq h` | True if g and h are the same element. | Normal form comparison. |
| `g ne h` | True if g and h are distinct elements. | — |
| `IsIdentity(g)` / `IsId(g)` | True if g is the identity. | — |
| `IsConjugate(G, g, h)` | True if g and h are conjugate in G; also returns a conjugating element z with gz = h. | Conjugacy algorithm for elements (see §63.7). |

### 63.6.5 Set Operations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `NumberingMap(G)` | Bijection G → {1…|G|} (depending on the current presentation). | Enumeration via exponent vectors. |
| `Random(G)` | Uniformly random element of G, chosen by generating a random exponent vector. Recommended method for random elements of pc-groups. | Uniform random via exponent vector sampling. |
| `RandomProcess(G : Slots, Scramble)` | Creates a process P for generating random elements. Maintains N = max(Slots, Ngens(G)+1) stored elements, expanded by repeated products. `Slots` (default 10); `Scramble` (default 20) initial mixing operations. | Product-replacement style random process. |
| `Random(P)` | Next random element from process P. | Product replacement. |
| `Representative(G)` / `Rep(G)` | A representative element; for a pc-group, always the identity. | — |

*Worked examples: H63E8 (element arithmetic, coercion, commutators); H63E9 (IsConjugate, commutator test); H63E10 (NumberingMap, Random, histograms); H63E11 (set products and element sets).*

---

## 63.7 Conjugacy

For non-p-groups, conjugacy classes are computed using the **homomorphism principle**
down a series with elementary abelian factors, together with orbit-stabiliser at each quotient
**[MN89]**. For p-groups, an algorithm based on linear algebra developed by Charles
Leedham-Green is used.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Class(H, g)` / `Conjugates(H, g)` / `g ^ H` | Set of conjugates of g under the action of H (if H = parent, this is the full conjugacy class). | — |
| `ConjugacyClasses(G)` / `Classes(G)` | Sequence of tuples (order, class length, representative) for all conjugacy classes of G. | Non-p-groups: homomorphism principle + orbit-stabiliser **[MN89]**; p-groups: linear algebra (Leedham-Green). |
| `ClassMap(G)` | Map M : G → {1…n} sending each element to its class number. | — |
| `ClassRepresentative(G, x)` | Designated representative for the conjugacy class of G containing x. | — |
| `IsConjugate(G, g, h)` | True if g and h are conjugate in G; also returns a conjugating element z. | Conjugacy test for elements **[MN89]** (non-p-groups), linear algebra (p-groups). |
| `NumberOfClasses(G)` / `Nclasses(G)` | Number of conjugacy classes of G. | — |
| `PowerMap(G)` | Map M : {1…n} × Z → {1…n}: for class c and integer n, returns the class where class c maps under the n-th power. | From conjugacy class data. |

*Worked example: H63E12 (conjugacy classes of SL(2,3), class map, structure constants).*

---

## 63.8 Subgroups

Subgroups of pc-groups are independent pc-groups with their own pc-presentation and a
maintained subgroup relationship. Magma tracks subgroup relationships in internal tables,
supporting automatic coercion between presentations.

### 63.8.1 Definition of Subgroups by Generators

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `sub< G \| L >` | Subgroup H of pc-group G generated by the elements/subgroups in generator list L. Computes pc-generators for H (and a minimal generating set if H is a p-group). Returns H and the inclusion map H → G. | Collection and Gaussian elimination on exponent vectors. |
| `ncl< G \| L >` | Normal closure of the subgroup generated by elements in L, as a subgroup of G. Returns H and inclusion map. | Iterated conjugation and closure. |

### 63.8.2 Membership and Coercion

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `g in G` | True if g is an element of G (both in a common covering group). | Membership test via normal form. |
| `g notin G` | True if g is not an element of G. | — |
| `G ! g` | Rewrite g (in subgroup H ≤ G) as an element of G. | Coercion via inclusion map. |
| `H ! g` | Rewrite g (in G) as an element of H ≤ G (assuming g ∈ H). | — |
| `K ! g` | Rewrite g from H to K, where H and K are subgroups of a common covering group. | — |

### 63.8.3 Inclusion and Equality

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `S subset G` | True if set S (of elements of H) is a subset of G. | — |
| `S notsubset G` | True if S is not a subset of G. | — |
| `H subset G` | True if H is a subgroup of G. | — |
| `H notsubset G` | True if H is not a subgroup of G. | — |
| `G eq H` | True if G and H are the same group. | Normal form / generating set comparison. |
| `G ne H` | True if G and H are distinct. | — |
| `InclusionMap(G, H)` | The inclusion map from H ≤ G to G. | — |

### 63.8.4 Standard Subgroup Constructions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `H ^ g` / `Conjugate(H, g)` | Conjugate g−1Hg of H. | — |
| `H meet K` / `H meet:= K` | Intersection of H and K. | For non-p-groups: **[GS90]**. |
| `CommutatorSubgroup(G, H, K)` / `CommutatorSubgroup(H, K)` | Commutator subgroup [H, K] where H, K ≤ G. | — |
| `Centralizer(G, g)` / `Centraliser(G, g)` | Centraliser of element g in G. | — |
| `Centralizer(G, H)` / `Centraliser(G, H)` | Centraliser of subgroup H in G. | — |
| `Core(G, H)` | Largest normal subgroup of G contained in H. | — |
| `H ^ G` / `NormalClosure(G, H)` | Normal closure of H in G. | Iterated conjugation. |
| `Normalizer(G, H)` / `Normaliser(G, H)` | Normaliser of H in G. | For non-p-groups: **[GS90]**. |

### 63.8.5 Properties of Subgroups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Index(G, H)` | Index [G : H] as an ordinary integer. | — |
| `FactoredIndex(G, H)` | Factored index [G : H]. | — |

### 63.8.6 Predicates for Subgroups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsCentral(G, H)` | True if H lies in the centre of G. | — |
| `IsConjugate(G, H, K)` | True if H and K are conjugate in G; also returns a conjugating element z. | — |
| `IsMaximal(G, H)` | True if H is a maximal subgroup of G. | — |
| `IsNormal(G, H)` | True if H is normal in G. | — |
| `IsSelfNormalizing(G, H)` | True if H = NG(H). | — |
| `IsSubnormal(G, H)` | True if H is subnormal in G. | — |

*Worked examples: H63E13 (Z5 wr Z3, sub vs ncl); H63E14 (coercion between subgroups); H63E15 (centralizer, meet, join in C6 × D5); H63E16 (normalizer chain, NormalClosure, IsConjugate on S4 pc-presentation).*

### 63.8.7 Hall π-Subgroups and Sylow Systems

All functions assume G is a soluble group of order p1e1 · · · pkek.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ComplementBasis(G)` | A complement basis: sequence of k subgroups where the i-th has order |G|/piei (complement of the i-th Sylow subgroup). | From Sylow basis. |
| `HallSubgroup(G, S)` | Hall π-subgroup of G where π is defined by set S (a set of primes, a single prime, or −p for the p′-Hall subgroup). | Hall's theorem applied to soluble groups. |
| `pCore(G, S)` | Core of the Hall π-subgroup Oπ(G); S as for HallSubgroup. If S = −p, returns Op′(G). | — |
| `SylowBasis(G)` | A Sylow basis: sequence of k subgroups of orders p1e1, …, pkek. | — |
| `SylowSubgroup(G, p)` / `Sylow(G, p)` | A Sylow p-subgroup of G. | Hall's theorem for soluble groups. |
| `SystemNormalizer(G)` / `SystemNormaliser(G)` | System normalizer: intersection of normalizers of all members of a complement basis. | Direct definition: N(Σ) = ∩NG(Hi). |

*Worked example: H63E17 (Hall 2-subgroup and 2′-subgroup of D3 wr D5).*

### 63.8.8 Conjugacy Classes of Subgroups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SubgroupClasses(G)` / `Subgroups(G)` | Conjugacy class representatives for all subgroups of G. | Algorithm of M. Slattery, essentially **[Hul99]** without automorphism action. |
| `AbelianSubgroups(G)` | Conjugacy class representatives for all abelian subgroups. | **[Hul99]** with abelian filter. |
| `CyclicSubgroups(G)` | Conjugacy class representatives for all cyclic subgroups. | **[Hul99]** with cyclic filter. |
| `ElementaryAbelianSubgroups(G)` | Conjugacy class representatives for all elementary abelian subgroups. | **[Hul99]** with elementary abelian filter. |
| `NilpotentSubgroups(G)` | Conjugacy class representatives for all nilpotent subgroups. | **[Hul99]** with nilpotent filter. |
| `MaximalSubgroups(G)` | Conjugacy class representatives for the maximal subgroups of G. | Algorithm of Charles Leedham-Green, using a special presentation for G. |
| `SubgroupLattice(G)` | Lattice of conjugacy classes of subgroups of G. | — |
| `BurnsideMatrix(G)` | The Burnside matrix corresponding to the subgroup lattice. | From SubgroupLattice. |
| `DisplayBurnsideMatrix(G)` | Pretty-print the Burnside matrix. | — |

*Worked example: H63E18 (C3 × D3: normal subgroups, Complements, non-normal subgroup classes).*

---

## 63.9 Quotient Groups

### 63.9.1 Construction of Quotient Groups

One strength of pc-presentations is that arbitrary quotient groups can be computed from
a normal subgroup description. The pQuotient function (see §63.9.2 and §63.16.1) handles
quotients of fp-groups; the functions here handle quotients of existing pc-groups.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `quo< G \| L >` | Quotient Q = G/N where N is the smallest normal subgroup containing the elements in L. Returns Q and the natural map f : G → Q. | Polycyclic quotient via collection. |
| `G / N` | Quotient of pc-group G by its normal subgroup N. | — |

*Worked example: H63E19 (computing O3′,3(G) for S4).*

### 63.9.2 Abelian and p-Quotients

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AbelianQuotient(G)` | Maximal abelian quotient G/G′ as GrpAb, with natural epimorphism π : G → G/G′. | — |
| `AbelianQuotientInvariants(G)` / `AQInvariants(G)` | Abelian invariants of the maximal abelian quotient as a sequence of integers. | — |
| `ElementaryAbelianQuotient(G, p)` | Maximal p-elementary abelian quotient Q as GrpAb, with natural epimorphism. | — |
| `pQuotient(G, p, c : Workspace, Metabelian, Exponent, Print)` | For a pc-group G, prime p, and positive integer c: largest p-quotient P of G with lower exponent-p class ≤ c (c = 0 sets limit 127). Returns P (by pc-presentation), the natural map π : G → P, a sequence S describing pc-generator definitions, and a flag indicating if P is maximal. `Workspace` (default 5000000), `Metabelian` (default `false`), `Exponent` (default 0), `Print` (default 0). | p-quotient algorithm; collection. |

---

## 63.10 Normal Subgroups and Subgroup Series

### 63.10.1 Characteristic Subgroups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Centre(G)` / `Center(G)` | The centre Z(G). | — |
| `CommutatorSubgroup(G)` / `DerivedSubgroup(G)` / `DerivedGroup(G)` | The derived subgroup G′ = [G, G]. | — |
| `FittingSubgroup(G)` / `FittingGroup(G)` | The Fitting subgroup F(G). | — |
| `FrattiniSubgroup(G)` | The Frattini subgroup Φ(G). | — |
| `Hypercentre(G)` / `Hypercenter(G)` | The hypercentre (stationary term of the upper central series). | — |
| `MinimalNormalSubgroups(G)` | Sequence of all minimal normal subgroups of G. | — |
| `pCore(G, S)` | Maximal normal π-subgroup Oπ(G); S a set of primes, a prime, or −p for Op′(G). | — |
| `Socle(G)` | The socle of G (product of all minimal normal subgroups). | — |

### 63.10.2 Subgroup Series

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AbelianBasis(G)` | For abelian G: sequences B and I with ⟨B⟩ = G, Order(B[i]) = I[i], giving the type of each p-primary component. | — |
| `AbelianInvariants(G)` / `Invariants(G)` | Abelian invariants of the abelian group G as a sequence of integers. | — |
| `ChiefSeries(G)` | A chief series for G, as a sequence of subgroups. | — |
| `CompositionSeries(G)` | A composition series for G, as a sequence of subgroups. The i-th term is presented by generators G.i through G.NPCgens(G). | — |
| `CompositionFactors(G)` | Sequence of integer tuples describing the composition factors (each cyclic of prime order q, represented as ⟨19, 0, q⟩). | — |
| `CompositionSeries(G, i)` | The (i+1)-th entry of the composition series (presented by generators G.(i+1) through G.m). | — |
| `DerivedSeries(G)` | The derived series of G, as a sequence of subgroups. | — |
| `DerivedLength(G)` | The derived length of G. | — |
| `ElementaryAbelianSeries(G)` | A series of normal subgroups of G with elementary abelian successive quotients. | — |
| `ElementaryAbelianSeriesCanonical(G)` | A canonical (isomorphism-type-dependent) elementary abelian series of characteristic subgroups. Slower than ElementaryAbelianSeries. | — |
| `LowerCentralSeries(G)` | The lower central series, as a sequence of subgroups. | — |
| `NilpotencyClass(G)` | Nilpotency class of G if nilpotent; otherwise −1. | — |
| `pCentralSeries(G, p)` | The p-central series P1 ▷ P2 ▷ … where P1 = G, Pi+1 = (G, Pi)Pip. Returned as a sequence of subgroups. | — |
| `SubnormalSeries(G, H)` | A subnormal series from G to H (each term normal in the previous), or empty if H is not subnormal in G. | — |
| `UpperCentralSeries(G)` | The upper central series, as a sequence of subgroups. | — |

*Worked example: H63E20 (elementary abelian series of D3 wr D5).*

### 63.10.3 Series for p-groups

These functions are defined only for pc-groups that are p-groups.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Agemo(G, i)` | Characteristic subgroup generated by {xpi : x ∈ G} for positive integer i. | — |
| `Omega(G, i)` | Characteristic subgroup generated by elements of order dividing pi. | — |
| `JenningsSeries(G)` | Jennings series J1 ▷ J2 ▷ … where J1 = G, Ji+1 = ⟨(Ji, G), Jpk⟩ with k = ⌈(i+1)/p⌉, as a sequence of subgroups. | — |
| `pClass(G)` | Lower exponent-p class of the p-group G. | — |
| `pRanks(G)` | Sequence whose i-th entry is the number of pc-generators for the lower exponent-p class i quotient. | — |

### 63.10.4 Normal Subgroups and Complements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `NormalSubgroups(G)` | All normal subgroups of G as a sequence. | — |
| `NormalLattice(G)` | Lattice of normal subgroups of G. | — |
| `MinimalNormalSubgroup(G)` | An elementary abelian minimal normal subgroup of the soluble group G. | — |
| `MinimalNormalSubgroup(G, N)` | An elementary abelian minimal normal subgroup of G contained in the non-trivial normal subgroup N. | — |
| `Complements(G, N)` | Conjugacy class representatives of all complements of the normal subgroup N in G. | First cohomology computation **[CNW90]**. |
| `NormalComplements(G, N)` | All normal complements of N in G. | First cohomology **[CNW90]**. |
| `NormalComplements(G, H, N)` | All complements of N in H (where N ⊴ G, H ⊴ G, N ≤ H) which are normal in G. | First cohomology **[CNW90]**. |

*Worked example: H63E21 (extraspecial × D3: complements in Sylow 3-subgroup, NormalComplements).*

---

## 63.11 Cosets

### 63.11.1 Coset Tables and Transversals

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Transversal(G, H)` / `RightTransversal(G, H)` | Indexed set T forming a right transversal of G over H, and the transversal mapping φ : G → T with φ(g) = ti where g ∈ H·ti. | — |
| `CosetTable(G, H)` | For H of index r in G: mapping M : {1…r} × G → {1…r} describing the action of G on right cosets of H. | — |
| `Transversal(G, H, K)` | Indexed set of representatives for the double cosets HuK in G, and the transversal mapping. | **[Sla01]** (double cosets in soluble groups). |
| `ShortCosets(p, H, G)` | Representatives for cosets of G mod H containing p; does not compute a full transversal, usable even for very large index. | — |

### 63.11.2 Action on a Coset Space

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CosetAction(G, H)` | Permutation representation of G on right cosets of H. Returns: natural map f : G → L; induced group L (permutation group); kernel K. | — |
| `CosetImage(G, H)` | Image L of G acting on right cosets of H, as a permutation group. | — |
| `CosetKernel(G, H)` | Kernel of the action of G on right cosets of H. | — |

---

## 63.12 Automorphism Group

### 63.12.1 General Soluble Group

Two algorithms are available for soluble non-p-groups:

1. **Lifting algorithm (default)** — developed by M. Smith **[Smi94]** and extended by Smith and Slattery using second cohomology. Computes Aut(G/G2) ≅ GL(d, p) and lifts through the elementary abelian layers G/G3, …, G/Gk−1 of a characteristic series.

2. **Sylow-based algorithm** — developed by D. Howden **[How12]**. Determines a Sylow p-subgroup P of G, uses the automorphism group of P (via the p-group algorithm), and constructs Aut(G) from it. When Aut(G) is soluble the algorithm automatically provides a pc-representation; otherwise it may produce a permutation representation.

A `GrpAuto` object is returned. Two special attributes, `GenWeights` and `WeightSubgroupOrders`, describe the weight filtration of the automorphism group: an automorphism has weight 2i+1 if it acts non-trivially on Gi/Gi+1 (where Gi are characteristic series terms) and weight 2i+2 if it acts trivially on Gi/Gi+1.

#### 63.12.1.1 Lifting Algorithm

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AutomorphismGroup(G)` | For a soluble pc-group G: the automorphism group as GrpAuto. The attribute `CharacteristicSeries` gives the characteristic series used; `GenWeights` and `WeightSubgroupOrders` are always set. | Lifting through elementary abelian layers of a characteristic series **[Smi94]**, extended by Smith–Slattery using second cohomology. |
| `HasAttribute(A, "GenWeights")` | If set, returns `true` and the sequence of integers indicating where each generator lies in the weight series. Short form: `A'GenWeights`. | — |
| `HasAttribute(A, "WeightSubgroupOrders")` | If set, returns `true` and the sequence of orders of the weight subgroups of A. Short form: `A'WeightSubgroupOrders`. | — |

*Worked example: H63E22 (AGL(1,8) constructed from GF(8): AutomorphismGroup, extension by automorphism, GenWeights, WeightSubgroupOrders).*

#### 63.12.1.2 Lifting from Automorphism Group of a Sylow p-subgroup

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AutomorphismGroupSolubleGroup(G : p)` | For a soluble pc-group G: automorphism group as GrpAuto. `p` (default 1 = largest Sylow prime) selects the Sylow subgroup used; the p-core of G must be non-trivial. Verbose flag: `AutomorphismGroupSolubleGroup`. | **[How12]**: constructs Aut(G) from Aut(P) for a Sylow p-subgroup P; returns pc-representation if Aut(G) is soluble. |
| `IsIsomorphicSolubleGroup(G, H : p)` | For soluble pc-groups G, H: returns `true` if isomorphic (and an isomorphism G → H). `p` selects the Sylow prime; p-cores must be non-trivial. | **[How12]**: tests isomorphism via Sylow p-subgroup isomorphism and extension. |

*Worked example: H63E23 (group from solgps library, AutomorphismGroupSolubleGroup, PCGroup of automorphism group).*

### 63.12.2 p-group

For p-groups, the automorphism group is computed by the algorithm of **[ELGO02]**. The
algorithm exploits characteristic structure and is most difficult for p-groups of large Frattini
rank (> ~6) and p-class 2.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AutomorphismGroup(G : CharacteristicSubgroups)` | For a p-group G: automorphism group as GrpAuto. `CharacteristicSubgroups` (default `[]`): supply known characteristic subgroups to improve efficiency (not verified by the algorithm). Verbose flag: `AutomorphismGroup`. | **[ELGO02]**: p-group automorphism group via characteristic structure and linear algebra. |
| `OrderAutomorphismGroupAbelianPGroup(A)` | Order of the automorphism group of the abelian p-group G = Ca1 × Ca2 × … where A = [a1, a2, …]. | Direct formula from the structure of Aut(G) for abelian p-groups. |

*Worked examples: H63E24 (SmallGroup(64, 78), AutomorphismGroup, generator orders); H63E25 (NumberOfSubgroupsAbelianPGroup, OrderAutomorphismGroupAbelianPGroup for C4 × C8 × C64).*

### 63.12.3 Isomorphism and Standard Presentations

A **standard presentation** (or canonical presentation) of a p-group is the pc-presentation
produced by the default p-group generation algorithm. Two p-groups in the same
isomorphism class have identical standard presentations. Algorithm: **[O'B94]**. The
difficulty of isomorphism testing is governed by Frattini rank; most practical for rank ≤ 5.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `StandardPresentation(G)` / `StandardPresentation(G : StartClass)` | For a p-group G (arbitrary pc-presentation): returns the group H defined by its standard presentation, and a map G → H. `StartClass` (default 1): use pQuotient up to class k−1 then standardize from class k. Verbose flag: `Standard`. | Canonical pc-presentation via the p-group generation algorithm **[O'B94]**. |
| `IsIdenticalPresentation(G, H)` | True if G and H have identical presentations. Intended for comparing standard presentations for efficient isomorphism batching. | Direct comparison of pc-relations. |
| `IsIsomorphic(G, H)` | True if p-groups G and H are isomorphic; if so, returns an isomorphism G → H. Constructs standard presentations class by class and checks equality. | **[O'B94]**: standard presentation comparison. |

*Worked examples: H63E26 (pQuotient of fp-groups: IsIsomorphic on class-3 2-quotients; explicit isomorphism between 5-groups; batch isomorphism testing using IsIdenticalPresentation).*

---

## 63.13 Generating p-groups

The **p-group generation algorithm** **[New77, O'B90]** constructs (immediate) descendants
of a p-group. A **descendant** of G (Frattini rank d, p-class c) is a group H with Frattini
rank d such that H/Pc(H) ≅ G; an **immediate descendant** has class c+1.

The p-central series of G: P0(G) = G, Pi(G) = [Pi−1(G), G]Pi−1(G)p. If Pc(G) = 1, G has p-class c.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `GeneratepGroups(p, d, c : Exponent, OrderBound, StepSizes, All)` | Generate all d-generator p-groups of p-class ≤ c. `Exponent` (default 0): groups satisfy the given exponent. `OrderBound` (default 0 = no bound): restrict to order ≤ pn. `StepSizes` (default `[]`): only construct descendants of order p(n+s) for s in StepSizes. `All` (default `true`): if false, return only capable groups. Verbose flag: `GeneratepGroups`. | p-group generation algorithm **[New77, O'B90]**. |
| `Descendants(G : ...)` / `Descendants(G, c : ...)` | Construct descendants of G with p-class ≤ c (default: p-class one larger than G). Supports same parameters as GeneratepGroups. | p-group generation algorithm **[New77, O'B90]**. |
| `ClassTwo(p, d : Exponent)` / `ClassTwo(p, d, Step : Exponent)` / `ClassTwo(p, d, s : Exponent)` | Count d-generator p-groups of p-class 2. Without s/Step: sequence of length C(d,2), m-th entry = count of groups of order p(d+m). With Step: count for m ∈ Step. With s: count of groups of order p(d+s). `Exponent`: count those of exponent p. Verbose flag: `ClassTwo`. | **[EO99]**: enumeration of p-class 2 groups. |

*Worked examples: H63E27 (Descendants of D16 up to p-class 8); H63E28 (2-generator exponent-4 groups, derived lengths); H63E29 (2-generator 3-groups of abundance zero up to order 35).*

---

## 63.14 Representation Theory

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CharacterDegrees(G)` | For a finite pc-group G (non-p-group version): sequence [⟨d1,c1⟩, …] where ci = number of irreducible characters of degree di. | **[Con90b]** (Conlon's algorithm for soluble groups). |
| `CharacterDegrees(G, z, p)` | For G with central element z and prime (or zero) p: number of absolutely irreducible characters of G in characteristic p lying over a faithful linear character of ⟨z⟩. | **[Con90b]**. |
| `CharacterDegrees(G)` | For a finite p-group G: sequence [⟨d1,c1⟩, …]. | **[Sla86]**. |
| `CharacterDegreesPGroup(G)` | For a p-group G: sequence [C0, C1, …] where Ci = number of irreducible characters of degree pi. | **[Sla86]**. |
| `CharacterTable(G : Al, DSSizeLimit)` | Ordinary irreducible character table of G. `Al`: `"Default"` (Dixon-Schneider for |G| ≤ 5000, Unger's algorithm otherwise), `"DS"` (force Dixon-Schneider), `"IR"` (force Unger **[Ung06]**). `DSSizeLimit` (default 10^4): before switching to Unger, use Dixon-Schneider for classes of size ≤ this bound. | Dixon-Schneider (small groups) or Unger's induction/reduction **[Ung06]**. |
| `CharacterTableConlon(G)` | Character table of a finite p-group G. | Conlon's algorithm **[Con90a]**. |
| `GModule(G, M)` | G-module for the action of G on the vector space defined by matrix ring M. | — |
| `GModule(G, A)` | KG-module M for the action of G on the elementary abelian subgroup A of G; also returns the map A → underlying vector space. | — |
| `GModule(G, A, B)` | KG-module M for the action of G on the elementary abelian section A/B of G; also returns the map A → underlying vector space. | — |
| `AbsolutelyIrreducibleRepresentationsSchur(G, k : Process, GaloisAction, MaxDimension, ExactDimension)` / `AbsolutelyIrreducibleModulesSchur(G, k : ...)` | Absolutely irreducible representations (resp. modules) of G over extensions/subfields of k (finite field, rationals, or cyclotomic field). For finite k: Glasby-Howlett algorithm for minimal field. `Process` (default `true`): return list of (index, representation) pairs for selective extension. `GaloisAction`: `"Yes"` (default at intermediate levels; collapse Galois orbits), `"No"` (all inequivalent), `"Relative"` (use Gal(K/k)). `MaxDimension`, `ExactDimension`: dimension filters. | Schur's method climbing the composition series defined by the pc-presentation; Glasby-Howlett for minimal finite field. |
| `IrreducibleRepresentationsSchur(G, k : ...)` / `IrreducibleModulesSchur(G, k : ...)` | Irreducible representations (resp. modules) over k. First computes absolutely irreducible representations, then rewrites over k. All parameters as above. | Schur's method then field descent. |

*Worked example: H63E31 (dihedral group of order 20: modular and rational representations via IrreducibleModulesSchur and AbsolutelyIrreducibleRepresentationsSchur).*

---

## 63.15 Central Extensions

This section provides functions to compute H2(G, U) for finite soluble G and finite abelian
U (trivial G-module), and to construct central extensions of U by G. The group H2(G, U)
factors as I × T where I is the image of Ext(G/G′, U) under inflation and T is the image
of Hom(H2(G), U) under transgression. Cocycles are represented as "cocyclic matrices"
with entries in U. For details see **[FO00]**. Verbose flag: `Cocycle`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ExtGenerators(G, U)` | For soluble pc-group G and abelian pc-group U: sequence of tuples (cocyclic matrix representative, order in H2(G,U)) for generators of Ext(G/G′, U). | **[FO00]**: second cohomology decomposition. |
| `HomGenerators(G, U)` | Sequence of tuples (cocyclic matrix representative, order in H2(G,U)) for generators of Hom(H2(G), U). | **[FO00]**. |
| `ElementSequence(G)` | Indexed set of elements of G in the order used by ExtGenerators and HomGenerators. | — |
| `RepresentativeCocycles(G, U, Ext, Hom)` | Complete and irredundant set of representatives for H2(G, U) as cocyclic matrices, given Ext and Hom from ExtGenerators and HomGenerators. | **[FO00]**. |
| `CentralExtension(G, U, A)` | Central extension of U by G determined by the cocyclic matrix A (from RepresentativeCocycles). | **[FO00]**. |
| `CentralExtensions(G, U, Q)` | Sequence of central extensions of U by G determined by the sequence of cocyclic matrices Q. Note: extensions need not be mutually non-isomorphic. | **[FO00]**. |
| `CentralExtensionProcess(G, U)` | Creates a process P for iterating over all central extensions of U by G (covering all isomorphism types; need not be mutually non-isomorphic). | **[FO00]**. |
| `NextExtension(~P)` | Advance the process P and construct the next central extension. | — |
| `IsEmpty(P)` | True if all central extensions from process P have been constructed. | — |

*Worked example: H63E32 (H2(D4, C2): ExtGenerators, HomGenerators, AbelianInvariants, RepresentativeCocycles, CentralExtension, CentralExtensions; central extension process on SmallGroup(12,5) × C2 × C3).*

---

## 63.16 Transfer Between Group Categories

### 63.16.1 Transfer to GrpPC

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PCGroup(G)` | GrpPC representation of G (for G in GrpPerm, GrpMat, etc.) and the isomorphism φ : G → P. | Polycyclic presentation computation from a permutation/matrix group. |
| `pQuotient(F, p, c : Workspace, Metabelian, Exponent, Print)` | For finitely presented group F, prime p, positive integer c: largest p-quotient H of F with lower exponent-p class ≤ c (c = 0 sets limit 127). Returns H (pc-presentation) and the homomorphism F → H. `Workspace` (default 1000000), `Metabelian` (default `false`), `Exponent` (default 0), `Print` (default 0). | p-quotient algorithm; see Chapter 71 for full details. |
| `SolubleQuotient(G)` / `SolvableQuotient(G)` | GrpPC representation P of the largest soluble quotient of G, and the natural homomorphism φ : G → P. | Soluble quotient algorithm; see Chapter 71. |

*Worked example: H63E33 (PCGroup of a Sylow 3-subgroup of GL(4, GF(3))).*

### 63.16.2 Transfer from GrpPC

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AbelianGroup(G)` | For abelian pc-group G: a GrpAb group H ≅ G and an isomorphism φ : G → H. | — |
| `FPGroup(G)` | GrpFP representation F of G and the isomorphism G → F. | — |
| `GPCGroup(G)` | GrpGPC representation F of G and the isomorphism G → F. | — |

*Worked example: H63E34 (SmallGroup(576, 4123): minimum-degree permutation representation via Subgroups, Core, CosetAction).*

---

## 63.17 More About Presentations

Magma maintains up to three pc-presentations for a group simultaneously:
- **User presentation**: for input/output; specified in the constructor.
- **Conditioned presentation**: for internal computation; composition series refines a normal series with elementary abelian factors; for p-groups, guarantees a central series with the first d generators giving the Frattini quotient.
- **Special presentation**: exhibits a Sylow system (Hall subgroups), the LG-series (refining the nilpotent series, within each nilpotent section refining by Frattini factors), and head splittings (complements in G/F for each Frattini factor). Required by several algorithms.

The **compact presentation** is an efficient serialization format (integer sequence) for library storage.

### 63.17.1 Conditioned Presentations

#### 63.17.1.1 Structure Operations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ConditionedGroup(G)` | The internally used conditioned presentation of G, recorded as a subgroup of G so that coercion works between presentations. | — |
| `IsConditioned(G)` | True if G's user presentation is also the internal conditioned presentation. | — |

#### 63.17.1.2 Element Operations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LeadingTerm(x)` | For x = a1α1 · · · anαn in a conditioned presentation: returns aiαi for the smallest i with αi > 0; returns identity if x = 1. | — |
| `LeadingGenerator(x)` | Returns ai for the smallest i with αi > 0 in the normal form of x. | — |
| `LeadingExponent(x)` | Returns αi for the smallest i with αi > 0; returns 0 for the identity. | — |
| `Depth(x)` | Returns the smallest i with αi > 0; returns 0 for the identity. | — |
| `PCClass(x)` / `WeightClass(x)` | Weight class of x: k if x ∈ Gδk−1 but x ∉ Gδk; returns n+1 for the identity. | — |

### 63.17.2 Special Presentations

A special presentation (C. R. Leedham-Green) satisfies: (1) the composition series refines the LG-series; (2) it exhibits a Sylow system; (3) it exhibits head splittings (complements in G/F for each Frattini factor). Magma computes special presentations automatically when needed; users can also request one explicitly.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SpecialPresentation(G)` | Returns a new group H (with special presentation) that is a subgroup of G equal to G, so coercion of elements and subgroups works between the presentations. | LG-series and Sylow system computation. |
| `SpecialWeights(G)` | Sequence of triples (nilpotent section number, square-free exponent abelian section number, elementary abelian p-layer number) for each pc-generator of a special presentation. | — |
| `NilpotentLength(G)` | Number of nilpotent factors in the nilpotent series. | — |
| `NilpotentBoundary(G, i)` | Subscript of the last generator in the i-th nilpotent section (1 ≤ i ≤ NilpotentLength(G)). | — |
| `MinorLength(G, i)` | Number of minor (Frattini factor) sections in the i-th nilpotent section. | — |
| `MinorBoundary(G, i, j)` | Subscript of the last generator in the j-th minor section of the i-th nilpotent section. | — |
| `LayerLength(G, i, j)` | Number of elementary abelian p-layers in the j-th minor section of the i-th nilpotent section. | — |
| `LayerBoundary(G, i, j, k)` | Subscript of the last generator in the k-th elementary abelian p-layer of the j-th minor section of the i-th nilpotent section. | — |

*Worked example: H63E35 (PolycyclicGroup wreath product vs SpecialPresentation for Z3^4 and C2 wr C6; coercion between presentations; SpecialWeights, MinorBoundary).*

### 63.17.3 Compact Presentation

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CompactPresentation(G)` | Returns a sequence of integers encoding the pc-group's presentation for compact storage (e.g., in libraries). | Serialization of pc-relations. |
| `PCGroup(Q : Check, ExponentLimit)` | Reconstruct a GrpPC group from the integer sequence Q produced by CompactPresentation. `Check` (default `false`): verify consistency (should only be skipped if Q is known valid). `ExponentLimit` (default 20): precompute products ai∗bj for i, j in [1, e]. Fast construction with very low parser overhead. | Deserialization; collection precomputation. |

*Worked example: H63E36 (CompactPresentation of S4 pc-group, reconstruction via PCGroup with literal sequence).*

---

## 63.18 Optimizing Magma Code

### 63.18.1 PowerGroup

When working with enumerated sets of subgroups of a common over-group G, define the set
universe as `PowerGroup(G)`. This enables canonical generator forms for subgroups of G
and provides a high-quality hash function for subgroup sets, resulting in dramatically faster
set membership operations.

*Worked example: H63E37 (ExtraSpecialGroup(GrpPC, 3, 3): enumeration of random subgroups with and without PowerGroup; ~8x speedup observed).*

---

## 63.19 Bibliography

| Key | Reference |
|-----|-----------|
| **[CH00]** | John Cossey and Trevor Hawkes. *On the largest conjugacy class size in a finite group.* Rend. Sem. Mat. Univ. Padova, 103:171–179, 2000. |
| **[CNW90]** | F. Celler, J. Neubüser, and C. R. B. Wright. *Some remarks on the computation of complements and normalizers in soluble groups.* Acta Appl. Math., 21:57–76, 1990. |
| **[Con90a]** | S. B. Conlon. *Calculating characters of p-groups.* J. Symbolic Comp., 9:535–550, 1990. |
| **[Con90b]** | S. B. Conlon. *Computing modular and projective character degrees of soluble groups.* J. Symbolic Comp., 9:551–570, 1990. |
| **[ELGO02]** | Bettina Eick, C. R. Leedham-Green, and E. A. O'Brien. *Constructing automorphism groups of p-groups.* Comm. Algebra, 30:2271–2295, 2002. |
| **[EO99]** | Bettina Eick and E. A. O'Brien. *Enumerating p-groups.* J. Austral. Math. Soc., 67:191–205, 1999. |
| **[FO00]** | D. L. Flannery and E. A. O'Brien. *Computing 2-cocycles for central extensions and relative difference sets.* Comm. Algebra, 28:1935–1955, 2000. |
| **[GS90]** | S. P. Glasby and Michael C. Slattery. *Computing intersections and normalizers in soluble groups.* J. Symbolic Comp., 9:637–651, 1990. (Computational group theory, Part 1.) |
| **[How12]** | David J. A. Howden. *Computing automorphism groups and isomorphism testing in finite groups.* PhD thesis, University of Warwick, 2012. |
| **[Hul99]** | Alexander Hulpke. *Computing subgroups invariant under a set of automorphisms.* J. Symbolic Comp., 27:415–427, 1999. |
| **[MN89]** | M. Mecky and J. Neubüser. *Some remarks on the computation of conjugacy classes of soluble groups.* Bull. Austral. Math. Soc., 40(2):281–292, 1989. |
| **[New77]** | M. F. Newman. *Determination of groups of prime-power order.* In Group Theory (Canberra, 1975), volume 573 of Lecture Notes in Mathematics, pages 73–84. Springer-Verlag, Berlin–Heidelberg–New York, 1977. |
| **[O'B90]** | E. A. O'Brien. *The p-group generation algorithm.* J. Symbolic Comput., 9:677–698, 1990. |
| **[O'B94]** | E. A. O'Brien. *Isomorphism testing for p-groups.* J. Symbolic Comp., 17:133–147, 1994. |
| **[Sla86]** | Michael C. Slattery. *Computing character degrees in p-groups.* J. Symbolic Comp., 2:51–58, 1986. |
| **[Sla01]** | Michael C. Slattery. *Computing double cosets in soluble groups.* J. Symbolic Comp., 31:179–192, 2001. (Computational algebra and number theory, Milwaukee, WI, 1996.) |
| **[Smi94]** | Michael J. Smith. *Computing automorphisms of finite soluble groups.* PhD thesis, Australian National University, 1994. |
| **[Ung06]** | W. R. Unger. *Computing the character table of a finite group.* J. Symbolic Comp., 41(8):847–862, 2006. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Collection algorithm (pc-presentation word problem) | All group construction, element arithmetic, subgroup operations in GrpPC |
| p-group generation **[New77, O'B90]** | `GeneratepGroups`, `Descendants` |
| p-class 2 enumeration **[EO99]** | `ClassTwo` |
| Standard presentation / isomorphism testing for p-groups **[O'B94]** | `StandardPresentation`, `IsIsomorphic`, `IsIdenticalPresentation` |
| Automorphism group of p-group **[ELGO02]** | `AutomorphismGroup` (p-group variant) |
| Lifting automorphism algorithm for soluble groups **[Smi94]** | `AutomorphismGroup` (non-p-group default) |
| Sylow-based automorphism / isomorphism **[How12]** | `AutomorphismGroupSolubleGroup`, `IsIsomorphicSolubleGroup` |
| Conjugacy classes: homomorphism principle + orbit-stabiliser **[MN89]** | `ConjugacyClasses`, `Classes`, `IsConjugate` (elements, non-p-groups) |
| Intersections and normalisers in soluble groups **[GS90]** | `H meet K`, `Normalizer` (non-p-groups) |
| First cohomology / complements **[CNW90]** | `Complements`, `NormalComplements` |
| Second cohomology / central extensions **[FO00]** | `ExtGenerators`, `HomGenerators`, `RepresentativeCocycles`, `CentralExtension`, `CentralExtensions`, `CentralExtensionProcess` |
| Subgroup enumeration **[Hul99]** | `SubgroupClasses`, `Subgroups`, `AbelianSubgroups`, `CyclicSubgroups`, `ElementaryAbelianSubgroups`, `NilpotentSubgroups` |
| Double cosets in soluble groups **[Sla01]** | `Transversal(G, H, K)` |
| Character degrees (soluble groups) **[Con90b]** | `CharacterDegrees` (non-p-group) |
| Character degrees (p-groups) **[Sla86]** | `CharacterDegrees`, `CharacterDegreesPGroup` (p-group) |
| Character table (p-groups, Conlon) **[Con90a]** | `CharacterTableConlon` |
| Character table (Unger induction/reduction) **[Ung06]** | `CharacterTable(:Al := "IR")` |
| Representation theory (Schur climbing) | `AbsolutelyIrreducibleRepresentationsSchur`, `AbsolutelyIrreducibleModulesSchur`, `IrreducibleRepresentationsSchur`, `IrreducibleModulesSchur` |
