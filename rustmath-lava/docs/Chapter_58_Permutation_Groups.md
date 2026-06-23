# Chapter 58 — Permutation Groups

**Handbook part:** IX — Finite Groups
**Handbook pages:** 1521–1635 (PDF pages 1648–1769)

---

## Scope and overview

Chapter 58 is the primary reference for permutation groups in Magma (category `GrpPerm`). A permutation group G is a group of bijections X → X for a finite set X of cardinality n ≤ 2³⁰. Magma represents the element set of G by a **base and strong generating set (BSGS)** — a fundamental algorithmic concept that makes order computation, membership testing, orbit computation, and structural analysis efficient. The great majority of structural functions require a BSGS; if one is not stored, Magma will attempt to compute one automatically.

The chapter covers the complete lifecycle of a permutation group:

1. **Construction** — creating symmetric groups, permutation elements, and general subgroups; standard groups (alternating, cyclic, dihedral, etc.); direct and wreath products.
2. **Elementary properties** — degree, order, abstract structure properties, and homomorphisms.
3. **Actions** — G-sets, orbits, stabilizers, block systems, coset spaces, and derived representations (orbit action, blocks action, coset action, reduced/primitive quotients, Jellyfish algorithm).
4. **Subgroup structure** — arbitrary subgroups (all, maximal, normal, low-index), conjugacy classes of subgroups, complements, supplements, normal series, socle, radical.
5. **Structural algorithms** — conjugacy classes of elements (random, action, inductive, extension algorithms), composition series (Kantor tabular algorithm), chief series, automorphism groups, isomorphism testing.
6. **BSGS algorithms** — Schreier-Sims, random Schreier, Todd-Coxeter-Schreier, soluble Schreier, verification.
7. **Identification** — recognizing alternating/symmetric groups (Beals et al., Bratus-Pak), identification of 2-transitive groups (Cameron-Cannon).
8. **Supplementary topics** — presentations (Todd-Coxeter-Schreier-Sims), word representations, cohomology, representation theory (character tables, modules), permutation representations of classical linear groups, databases, and ordered partition stacks.

---

## 58.1 Introduction

### 58.1.1 Terminology

A permutation group G acts on a finite set X of cardinality n. The **natural G-set** is X; other sets with a legitimate G-action are G-sets. The **fixed-point set** of G on Y is the set of points fixed by all elements; the **support** of G is the set of points moved by some element. The **degree** of G equals |X|; the degree of an element g is the size of its support. Magma is limited to degree less than 2³⁰.

### 58.1.2 The Category of Permutation Groups

All permutation groups of finite degree form the category `GrpPerm`. Morphisms are group homomorphisms.

### 58.1.3 The Construction of a Permutation Group

Every permutation group is created as a subgroup of Sym(X). Construction is a two-step process: create Sym(X), then define G as a subgroup. The `PermutationGroup< >` constructor combines both steps.

---

## 58.2 Creation of a Permutation Group

### 58.2.1 Construction of the Symmetric Group

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Sym(n)` / `SymmetricGroup(n)` | Create the generic symmetric group on {1, 2, ..., n}. Initially only a structure table is created; generators are built dynamically if needed. | — |
| `Sym(X)` / `SymmetricGroup(X)` | Given a finite set X of cardinality n, create the symmetric group Sym(X). Internally represented on {1,...,n}; translation via `Labelling`. | — |
| `StandardGroup(G)` | Return a group H isomorphic to G but acting on {1,...,n}. Also returns the isomorphism G → H. | — |

*Worked examples: H58E1 (symmetric group on string set; on {0..9}).*

### 58.2.2 Construction of a Permutation

Throughout this subsection G acts on X = {x₁,...,xₙ}. Note: the `elt<>` and `!` constructors may trigger BSGS construction for membership testing.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `elt< G \| L >` | Construct the permutation g of G defined by xᵢ → aᵢ from list L of images. Tests membership in G. | Membership via BSGS strip. |
| `G ! Q` | Coerce sequence Q (of images) into permutation of G. Fails if g ∉ G. | BSGS membership test. |
| `G ! (...)(...)...(...)` | Coerce product-of-cycles notation into permutation of G. | — |
| `G ! \(...)(...)...(...)` | Construct permutation from literal integer cycles (recommended for large permutations to avoid large parse trees). | — |
| `G ! Q` (Q = indexed set of cycles) | Construct permutation from disjoint cycles given as indexed sets. | — |
| `ElementToSequence(g)` / `Eltseq(g)` | Sequence of images of the G-set of g; satisfies `Parent(g)!Eltseq(g) eq g`. | — |
| `Identity(G)` / `Id(G)` / `G ! 1` | Construct the identity permutation in G. | — |

*Worked examples: H58E2 (three constructions of (2,3)(4,5,6)).*

### 58.2.3 Construction of a General Permutation Group

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PermutationGroup< X \| L >` | Construct the permutation group G acting on set X, generated by permutations in list L. Equivalent to `sub< Sym(X) \| L >`. | — |
| `PermutationGroup< n \| L >` | Construct the permutation group G acting on {1,...,n}, generated by L. | — |

*Worked examples: H58E3 (Hessian group of order 216 on 9 points).*

---

## 58.3 Elementary Properties of a Group

### 58.3.1 Accessing Group Information

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `G . i` | The i-th defining generator of G. Negative subscript gives inverse; `G.0` gives identity. | — |
| `Degree(G)` | The degree of the permutation group G (cardinality of its natural G-set). | — |
| `Generators(G)` | A set of elements of G that generate G. | — |
| `GeneratorsSequence(G)` | The sequence of defining generators, preserving duplicates and identity elements. Equals `[G.i : i in [1..Ngens(G)]]`. | — |
| `NumberOfGenerators(G)` / `Ngens(G)` | The number of defining generators. | — |
| `FewGenerators(G)` | A typically short sequence of random elements generating G (especially useful when G is a subgroup). | Random element generation. |
| `Generic(G)` | The generic group containing G (i.e. the symmetric group in which G is naturally embedded). | — |
| `Parent(g)` | The parent group G for permutation g. | — |
| `GSet(G)` | The natural G-set for G. | — |

*Worked examples: H58E4 (group of order 648 on 12 points; degree, GSet, Generic, Generators, Ngens, Parent).*

### 58.3.2 Group Order

Order computation requires a BSGS if not already known.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Order(G)` / `#G` | The order of G as an integer. Constructs BSGS if order is not known. | BSGS / Schreier-Sims **[Leo80]** |
| `FactoredOrder(G)` | The order of G as a factored integer `[<p₁,e₁>,...,<pₙ,eₙ>]`. Computes order if needed. | BSGS / Schreier-Sims **[Leo80]** |

### 58.3.3 Abstract Properties of a Group

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsAbelian(G)` | True if G is abelian. | — |
| `IsCyclic(G)` | True if G is cyclic. | — |
| `IsElementaryAbelian(G)` | True if G is elementary abelian. | — |
| `IsSpecial(G)` | True if G (a p-group) is special. | — |
| `IsExtraSpecial(G)` | True if G (a p-group) is extra-special. | — |
| `IsNilpotent(G)` | True if G is nilpotent. | — |
| `IsSoluble(G)` / `IsSolvable(G)` | True if G is soluble. | Algorithm of Sims **[Sim90]**. |
| `IsPerfect(G)` | True if G is perfect (G = G'). | — |
| `IsSimple(G)` | True if G is simple. | — |
| `IsWreathProduct(G)` | True if G ≅ A ≀ B (B transitive). If true, also returns subgroups A, B, C such that G ≅ WreathProduct(A, CosetImage(B, C)). | — |

*Worked examples: H58E5 (perfect non-simple subgroups of M24 via PerfectSubgroups).*

---

## 58.4 Homomorphisms

Magma provides extensive facilities for group homomorphisms. Many useful homomorphisms are returned by constructors and intrinsic functions (quo, sub, OrbitAction, BlocksAction, FPGroup, RadicalQuotient, etc.).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `hom< G -> H \| L >` | Construct the homomorphism f : G → H given by generator images in L. L may be a list of elements of H, or a list of pairs (generator, image). | — |
| `Domain(f)` | The domain of homomorphism f. | — |
| `Codomain(f)` | The codomain of homomorphism f. | — |
| `Image(f)` | The image (range) of f as a subgroup of the codomain. | Simultaneous image/kernel computation **[LGPS91]**. |
| `Kernel(f)` | The kernel of f as a normal subgroup of the domain. | Simultaneous image/kernel computation **[LGPS91]**. |
| `IsHomomorphism(G, H, Q)` | Returns true if the sequence Q (of length Ngens(G), containing elements of H) defines a valid homomorphism G → H. If so, also returns the homomorphism. | Algorithm of **[LGPS91]**. |

*Worked examples: H58E6 (conjugation action of a group of order 648; image and kernel; preimage of O₂(H)).*

---

## 58.5 Building Permutation Groups

### 58.5.1 Some Standard Permutation Groups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AbelianGroup(GrpPerm, Q)` | Construct the abelian group Z(n₁) × Z(n₂) × ··· × Z(nᵣ) for Q = [n₁,...,nᵣ]. | — |
| `AlternatingGroup(GrpPerm, n)` / `AlternatingGroup(n)` / `Alt(n)` | Alternating group of degree n on standard generators ((3,4,...,n) and (1,2,3) if n odd; (1,2)(3,...,n) and (1,2,3) if n even). | — |
| `CyclicGroup(GrpPerm, n)` / `CyclicGroup(n)` | Cyclic group of order n with generator (1,2,...,n). | — |
| `DihedralGroup(GrpPerm, n)` / `DihedralGroup(n)` | Dihedral group of degree n and order 2n with generators (1,2,...,n) and (1,n)(2,n-1)···. | — |
| `Sym(GrpPerm, n)` / `SymmetricGroup(GrpPerm, n)` / `Sym(n)` / `SymmetricGroup(n)` | Symmetric group of degree n with generators (1,2,...,n) and (1,2). | — |
| `ExtraSpecialGroup(GrpPerm, p, n : parameters)` / `ExtraSpecialGroup(p, n : parameters)` | Extra-special group of order p^{2n+1} in GrpPerm. `Type := "+"` (default): for p=2 central product of n copies of D₈; for p>2 exponent-p type. `Type := "-"`: for p=2 central product of Q₈ and (n-1) copies of D₈; for p>2 exponent-p² type. | — |
| `YoungSubgroup(L)` | Young subgroup parameterized by L: direct product of symmetric groups Sym(L[i]). Optional parameter `Full` sets the ambient symmetric group degree. | — |

*Worked examples: H58E7 (Z₂ × Z₂ × Z₄, Alt(12), Z₂₄, D₁₂, Sym(8)).*

### 58.5.2 Direct Products and Wreath Products

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DirectProduct(G, H)` | Direct product of G and H as an intransitive group of degree deg(G)+deg(H). Also returns sequences of inclusions I and projections P. | — |
| `DirectProduct(Q)` | Direct product of the sequence Q of permutation groups. Returns D, inclusions I, projections P. | — |
| `PrimitiveWreathProduct(G, H)` | Wreath product G ≀ H with **product action**. | — |
| `PrimitiveWreathProduct(Q)` | Iterated wreath product (...(Q[1] ≀ Q[2]) ≀ ... ≀ Q[n]) with product action. | — |
| `WreathProduct(G, H)` | Wreath product W = G ≀ H with **imprimitive action**. Returns W, sequence of inclusions of G into W, inclusion of H into W, projection of W onto H. | — |
| `WreathProduct(Q)` | Iterated wreath product with imprimitive action. | — |
| `WreathProduct(B)` | Wreath product corresponding to block system B of some permutation group G. | — |
| `WreathProduct(G, B)` | Smallest wreath product W to block system B of G such that G ⊆ W. Also returns complement as subgroup of W. | — |

*Worked examples: H58E8 (DirectProduct, PrimitiveWreathProduct, WreathProduct of Sym(4) and DihedralGroup(3)).*

---

## 58.6 Permutations

### 58.6.1 Coercion

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `G ! g` | Given g ∈ Sym(X) with g ∈ G, embed g in G (changes parent to G). | BSGS membership. |
| `G !! H` | Embed group H as a subgroup of G, where H's G-set X is a subset of G's G-set Y. Fails if image of H is not a subgroup of G. | — |

### 58.6.2 Arithmetic with Permutations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `g * h` | Product of permutations g, h in the same generic group U. Result parent determined by the least subgroup containing both. | — |
| `g ^ n` | n-th power of permutation g (n ∈ Z). | — |
| `g / h` | g * h⁻¹. Same parent rules as multiplication. | — |
| `g ^ h` | Conjugate h⁻¹gh. | — |
| `(g, h)` | Commutator g⁻¹h⁻¹gh. | — |
| `(g1, ..., gr)` | Left-normed commutator of r permutations. | — |

### 58.6.3 Properties of Permutations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CycleStructure(g)` | Partition of n as sequence of pairs `<l, count>` where l is cycle length. | — |
| `Degree(g)` | Number of points moved by g. | — |
| `IsEven(g)` | True if g is an even permutation. | — |
| `Sign(g)` | 1 if g is even, -1 if g is odd. | — |
| `Order(g)` | Order of the permutation g. | — |

### 58.6.4 Predicates for Permutations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `g eq h` | True if g and h are the same element of their common generic group. | — |
| `g ne h` | True if g and h are distinct elements. | — |
| `IsId(g)` / `IsIdentity(g)` | True if g is the identity permutation. | — |

*Worked examples: H58E9 (operations in Sym(9): product, inverse, power, quotient, conjugate, commutator, CycleStructure, Degree, Order).*

### 58.6.5 Set Operations

The BSGS imposes a lexicographic order on elements of G by base images, enabling a compact representation.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `G * H` | Set product {g*h | g ∈ G, h ∈ H} as a set of group elements. | — |
| `ElementSet(G, H)` | Elements of subgroup H as a set of elements of G. Only for small groups. | — |
| `NumberingMap(G)` | Bijection G → {1,...,|G|} based on the BSGS ordering. | BSGS lexicographic order. |
| `RandomProcess(G : parameters)` | Create a process to generate random elements via the product-replacement algorithm **[CLGM+95]** with an accumulator. Parameters: `Slots` (default 10), `Scramble` (default 20). | Product-replacement / accumulator method **[CLGM+95]**. |
| `Random(G : parameters)` | A randomly chosen element of G. Uniform if BSGS is known; otherwise biased towards shorter words. Parameter `Short := true` uses short word. | Uniform (BSGS known) or random word. |
| `Random(P)` | Next random element from process P (use for large-degree or BSGS-unknown groups). | Product-replacement **[CLGM+95]**. |
| `Representative(G)` / `Rep(G)` | An element chosen from G. | — |

*Worked examples: H58E10 (NumberingMap for multiplication table of D₁₂), H58E11 (Random elements and CycleStructure in WreathProduct(Sym(4), CyclicGroup(6))).*

---

## 58.7 Conjugacy

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Class(H, x)` / `Conjugates(H, x)` | Set of conjugates of x under H. If H = K (parent of x), returns the conjugacy class. | — |
| `ConjugacyClasses(G : parameters)` / `Classes(G : parameters)` | Representatives of conjugacy classes as sequence of triples `<order, length, representative>`. Parameters: `Reps` (supply known reps or rep/length pairs), `Al` (algorithm choice), `WeakLimit` (default 500), `StrongLimit` (default 5000), `Centralisers`, `PowerMap`. | Multiple algorithms (see below). |
| `ClassRepresentative(G, x)` | The stored representative of the conjugacy class of x in G (classes must be known). | — |
| `ClassCentraliser(G, i)` | Centraliser of the representative of the i-th conjugacy class. Stored for future calls. | — |
| `ClassMap(G : parameters)` | Conjugacy classes and the class map f: for any x ∈ G, f(x) is the class index. `Orbits := true` computes via conjugation orbits (fast map, small groups only). | Random class algorithm (default) or orbit algorithm. |
| `IsConjugate(G, g, h : parameters)` | True if g and h are conjugate in G; if so also returns a conjugating element. Parameters: `Centralizer` ("Default"/"Left"/"Right"/"Both"/"None"), `LeftSubgroup`, `RightSubgroup` (known subgroups of centralizers to speed search). | Backtrack search of Leon **[Leo97]**. |
| `IsConjugate(G, H, K : parameters)` | True if subgroups H and K are conjugate in G; if so also returns a conjugating element. Parameters: `Compute` ("Default"/"Left"/"Right"/"Both"/"None" controlling normalizer precomputation), `LeftSubgroup`, `RightSubgroup`. | Backtrack search of Leon **[Leo97]**. |
| `Exponent(G)` | Exponent of G, computed as product of exponents of Sylow subgroups. | — |
| `NumberOfClasses(G)` / `Nclasses(G)` | Number of conjugacy classes. | — |
| `PowerMap(G)` | The power map f: C × Z → C where f(i,j) is the index of the class containing xᵢʲ. | — |
| `AssertAttribute(G, "Classes", Q)` | Assert conjugacy class representatives; acts like Classes with `Reps := Q`. | — |

**Conjugacy class algorithms (Al parameter):**
- `"Action"`: Orbits under conjugation; feasible only for small groups.
- `"Random"`: Probabilistic search using weak conjugacy (cycle structure) then ordinary conjugacy tests **[CB92]**.
- `"Inductive"`: Butler's inductive method **[But94]**.
- `"Extend"`: Compute classes in G/R (R = solvable radical), then lift through elementary abelian series using affine action on each layer (Mecky–Neubüser **[MN89]**). Holt's extension of the Cannon-Souvignier fusion method **[CS97]** is used for the TF-group.
- `"Default"`: If IsAltsym(G): from partitions of Degree(G). If solvable: via pc-group. If |G| ≤ 5000: action algorithm; otherwise: extension algorithm.

*Worked examples: H58E12 (classes of M₁₁), H58E13 (Higman-Sims group with WeakLimit/StrongLimit tuning; power map tabulation).*

---

## 58.8 Subgroups

### 58.8.1 Construction of a Subgroup

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `sub< G \| L >` | Subgroup H of G generated by elements in list L (sequences, elements of G, subgroups of G, or sets/sequences thereof). | — |
| `ncl< G \| L >` | Normal closure in G of the subgroup generated by L. | Algorithm of Butler-Cannon **[BC82]**. |

*Worked examples: H58E14 (four ways to define PGL(2,7)), H58E15 (action on unordered pairs in Alt(7)), H58E16 (derived subgroup of Hessian group via ncl).*

### 58.8.2 Membership and Equality

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `g in G` | True if permutation g is an element of G. | BSGS strip. |
| `g notin G` | True if g is not an element of G. | BSGS strip. |
| `S subset G` | True if set S (of permutations of a group H in the same generic group) is a subset of G. | — |
| `S notsubset G` | True if S is not a subset of G. | — |
| `H subset G` | True if H is a subgroup of G (both in the same generic group). | — |
| `H notsubset G` | True if H is not a subgroup of G. | — |
| `H eq G` | True if H and G are the same subgroup. | — |
| `H ne G` | True if H and G are distinct subgroups. | — |

### 58.8.3 Elementary Properties of a Subgroup

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Index(G, H)` | Index [G:H] as an integer. Computes orders if needed. | — |
| `FactoredIndex(G, H)` | Index [G:H] as a factored integer. | — |
| `IsCentral(G, H)` | True if H lies in the centre of G. | — |
| `IsNormal(G, H)` | True if H is normal in G. | — |
| `IsSelfNormalizing(G, H)` / `IsSelfNormalising(G, H)` | True if H is self-normalizing in G. | — |
| `IsSubnormal(G, H)` | True if H is subnormal in G. | — |

### 58.8.4 Standard Subgroups

All these functions may create a BSGS if order is not known.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `H ^ g` / `Conjugate(H, g)` | Conjugate g⁻¹Hg. H and g must be in the same symmetric group. | — |
| `H meet K` | Intersection of subgroups H and K (same symmetric group). | Backtrack search of Leon **[Leo97]**. |
| `IntersectionWithNormalSubgroup(G, N)` | Intersection of G and N, where G normalises N. Parameter `Check` (default true). | Algorithm of Cooperman-Finkelstein-Luks **[CFL89]** (degree-doubled permutation representation). |
| `CommutatorSubgroup(G, H, K)` / `CommutatorSubgroup(H, K)` | Commutator subgroup [H, K] in G. G may be omitted if K ≤ H. | Algorithm of Butler-Cannon **[BC82]**. |
| `Centralizer(G, g : parameters)` / `Centraliser(G, g : parameters)` | Centralizer C_G(g). Parameter `Subgroup` (known subgroup of centralizer). | Backtrack search **[Leo97]**. |
| `Centralizer(G, H)` / `Centraliser(G, H)` | Centralizer C_G(H). | Backtrack search **[Leo97]**. |
| `CentralizerOfNormalSubgroup(G, H)` | Centralizer of normal subgroup H in G in polynomial time. | Polynomial-time reduction of Beals **[Bea93]**. |
| `SectionCentraliser(G, H, K)` / `SectionCentralizer(G, H, K)` | Full preimage in G of the centralizer in G/K of H/K (K ◁ G, K ≤ H ◁ G). | Algorithm of Luks **[Luk93]** (core of subgroup in a degree-doubled group). |
| `Core(G, H)` | Maximal normal subgroup of G contained in H. | Repeated conjugation/intersection via backtrack **[Leo97]**. |
| `H ^ G` / `NormalClosure(G, H)` | Normal closure of H in G. | — |
| `Normalizer(G, H : parameters)` / `Normaliser(G, H : parameters)` | Normalizer N_G(H). Parameters: `Subgroup` (known subgroup of normalizer, default H), `Bound` (terminate once normalizer has order ≥ Bound). | Backtrack search **[Leo97]**. |
| `SymmetricNormalizer(G)` / `SymmetricNormaliser(G)` | Normalizer of G in the full symmetric group on G's G-set. Same parameters as Normalizer. | Backtrack search **[Leo97]**. |
| `SylowSubgroup(G, p)` / `Sylow(G, p)` | A Sylow p-subgroup of G. | Algorithm of Cannon-Cox-Holt **[CCH97]**. |

*Worked examples: H58E17 (Sylow 2-subgroup and normalizer in a group of degree 30).*

### 58.8.5 Maximal Subgroups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsMaximal(G, H : parameters)` | True if H is maximal in G. Parameter `Al` ("Subgroups" or "CosetImage"): default computes maximal subgroups and tests conjugacy if [G:H] > 1000; otherwise tests primitivity of coset representation. | `Subgroups` algorithm **[CH04]** or coset primitivity test. |
| `IsProbablyMaximal(G, H : parameters)` | Probabilistic maximality test: adjoin random elements to H, check if result is G. Parameter `Tries` (default 20). | Probabilistic. |
| `MaximalSubgroups(G : parameters)` | Sequence of records for maximal subgroup classes of G. Same parameters as Subgroups. | **[CH04]**. |

*Worked examples: H58E18 (46 maximal subgroups of the 4×4×4 Rubik's cube group of order ~1.7×10⁵⁵).*

### 58.8.6 Conjugacy Classes of Subgroups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SubgroupClasses(G : parameters)` / `Subgroups(G : parameters)` | Representatives of conjugacy classes of subgroups as sequence of records `(subgroup, order, length[, presentation])`. Parameters: `Al` ("All"/"Maximal"/"Normal"), `LayerSizes`, `Series`, `Presentation`, `OrderEqual`, `OrderDividing`, `OrderMultipleOf`, `IndexLimit`, `IsElementaryAbelian`, `IsCyclic`, `IsAbelian`, `IsNilpotent`, `IsSolvable`, `IsNotSolvable`, `IsPerfect`, `IsRegular`, `IsTransitive`. | Elementary abelian series + radical quotient lookup + database; algorithm of Cannon-Cox-Holt **[CCH01]**. Database covers TF-groups of order ≤ 216 000, perfect groups of order ≤ 10⁶, and special families. |
| `SubgroupsLift(G, A, B, Q : parameters)` | Lift subgroups (given as records Q for G/A) through the elementary abelian layer A/B to G/B. | One step of the extension process of **[CCH01]**. |
| `LowIndexSubgroups(G, n : parameters)` / `LowIndexSubgroups(G, t : parameters)` | Subgroups of index ≤ n (or in interval [a,b] if t = `<a,b>`); one per conjugacy class. Parameters: `Presentation`, `Print`, `Algorithm`. | Cannon-Holt-Slattery-Steel **[CHSS03]**. |

*Worked examples: H58E19 (all subgroups of 2-fold cover of M₁₂; 293 classes; restriction to elementary abelian 2-subgroups), H58E20 (SubgroupLattice of PSL(2,9)).*

### 58.8.7 Classes of Subgroups Satisfying a Condition

Convenience wrappers for Subgroups with specific filter parameters:

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `NormalSubgroups(G : parameters)` | Equivalent to `Subgroups(G: Al := "Normal")`. | **[CCH01]** |
| `ElementaryAbelianSubgroups(G : parameters)` | Equivalent to `Subgroups(G: IsElementaryAbelian := true)`. | **[CCH01]** |
| `CyclicSubgroups(G : parameters)` | Equivalent to `Subgroups(G: IsCyclic := true)`. | **[CCH01]** |
| `AbelianSubgroups(G : parameters)` | Equivalent to `Subgroups(G: IsAbelian := true)`. | **[CCH01]** |
| `NilpotentSubgroups(G : parameters)` | Equivalent to `Subgroups(G: IsNilpotent := true)`. | **[CCH01]** |
| `SolvableSubgroups(G : parameters)` | Equivalent to `Subgroups(G: IsSolvable := true)`. | **[CCH01]** |
| `PerfectSubgroups(G : parameters)` | Equivalent to `Subgroups(G: IsNotSolvable := true)`. | **[CCH01]** |
| `NonsolvableSubgroups(G : parameters)` | Equivalent to `Subgroups(G: IsNotSolvable := true)`. | **[CCH01]** |
| `SimpleSubgroups(G : parameters)` | Equivalent to `Subgroups(G: Al := "Simple")`; returns non-abelian simple subgroup classes. | **[CCH01]** |

---

## 58.9 Quotient Groups

### 58.9.1 Construction of Quotient Groups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `quo< G \| L >` | Construct quotient Q = G/N where N is the normal closure of the subgroup generated by L. Returns Q and the natural epimorphism f: G → Q. Constructed via regular representation; restricted to small index. | Regular representation + degree reduction. |
| `G / N` | Quotient of G by normal subgroup N, constructed via regular representation. Restricted to small index. | Regular representation + degree reduction. |

*Worked examples: H58E21 (Sym(4) modulo Klein 4-group).*

### 58.9.2 Abelian, Nilpotent and Soluble Quotients

These methods first construct an fp-group presentation and apply the appropriate fp-group algorithm.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AbelianQuotient(G)` | Maximal abelian quotient G/G' as GrpAb, plus the natural epimorphism. | fp-group abelianization. |
| `ElementaryAbelianQuotient(G, p)` | Maximal p-elementary abelian quotient as GrpAb, plus epimorphism. | fp-group methods. |
| `pQuotient(G, p, c)` | Largest p-quotient of G with lower exponent-p class ≤ c (c=0 means class ≤ 127). Returns the pc-group P, natural homomorphism π, sequence S describing pc-generator definitions, and a flag indicating if P is the maximal p-quotient. | p-quotient algorithm. |
| `NilpotentQuotient(G, c)` | Class-c nilpotent quotient of G and the epimorphism π. | Nilpotent quotient algorithm. |
| `SolvableQuotient(G)` / `SolubleQuotient(G)` | Largest soluble quotient S of G and the epimorphism π: G → S. | Soluble quotient algorithm. |

*Worked examples: H58E22 (soluble quotient of WreathProduct(Sym(6), DihedralGroup(6))).*

---

## 58.10 Permutation Group Actions

### 58.10.1 G-Sets

A G-set is a pair (Y, f) where f: Y × G → Y satisfies f(f(y,g),h) = f(y,gh) and f(y,1) = y. Magma supports three types: the natural G-set X, derived sets of X (subsets, k-subsets, k-sequences, ordered partitions, Cartesian products), and general G-sets with a user-defined action map.

### 58.10.2 Creating a G-Set

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `GSetFromIndexed(G, Y)` | G-set corresponding to the bijection between Labelling(G) and indexed set Y. | — |
| `GSet(G, X, Y)` / `GSet(G, Y)` | Smallest derived G-set containing Y as a subset under the action of G on X (default: natural action). | Orbit closure. |
| `GSet(G)` | The G-set for the natural action of G. | — |
| `GSet(G, Y, f)` | Smallest G-set containing Y with user-defined action f: Y × G → Y. | Orbit closure with custom action. |
| `Action(Y)` | The action map of the G-set Y. | — |
| `Group(Y)` | The group associated with G-set Y. | — |
| `Labelling(G)` | Indexed set giving the internal mapping of the natural G-set onto {1,...,n}. | — |
| `Degree(g, Y)` / `Degree(g)` | Number of points of Y moved by g. If Y omitted, natural G-set assumed. | — |
| `Degree(G, Y)` / `Degree(G)` | Cardinality of G-set Y. | — |
| `Support(g, Y)` / `Support(g)` | Subset of Y moved by g. | — |
| `Support(G, Y)` / `Support(G)` | Subset of Y moved by some element of G. | — |

*Worked examples: H58E23 (G-set of irreducible characters of a normal subgroup with conjugation action; inertia group computation).*

### 58.10.3 Images, Orbits and Stabilizers

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `x ^ g` | Image of derived G-set element x under permutation g. | — |
| `Image(g, Y, y)` / `Image(g, y)` | Image of y ∈ Y under g. Y may be omitted for derived G-set elements. | — |
| `Fix(g, Y)` / `Fix(g)` | Fixed-point set of g on Y (or natural G-set). | — |
| `Fix(G, Y)` / `Fix(G)` | Fixed-point set of G on Y (or natural G-set). | — |
| `x ^ G` | Orbit of derived G-set element x under G (returned as a G-set). | — |
| `Cycle(e, x)` | Cycle of x under e: indexed set of images of x under repeated application of e, with x first. | — |
| `CycleDecomposition(e)` | Full disjoint cycle decomposition of e as a sequence of indexed sets partitioning the G-set X. | — |
| `Orbit(G, Y, y)` / `Orbit(G, y)` | Orbit of y under G on Y (returned as G-set). Y may be omitted. | — |
| `Orbits(G, Y)` / `Orbits(G)` | All orbits of G on Y (sequence of G-sets). | — |
| `OrbitRepresentatives(G)` | Orbits of G on its natural G-set as sequence of tuples `<length, representative>`. Space-efficient (no full orbit stored). | — |
| `OrbitClosure(G, Y, S)` / `OrbitClosure(G, S)` | Smallest G-invariant subset of Y containing S. | — |
| `IsConjugate(G, Y, y, z)` / `IsConjugate(G, y, z)` | True if ∃ g ∈ G with y^g = z; returns conjugating element if so. Applies to elements, sets, multisets, sequences, ordered partitions. | — |
| `Stabilizer(G, Y, y)` / `Stabiliser(G, Y, y)` / `Stabilizer(G, y)` / `Stabiliser(G, y)` | Stabilizer of y (element, sequence, set, ordered partition, or tuple over Y) as a subgroup of G. | — |
| `IsPrimitive(G, Y)` / `IsPrimitive(G)` | True if G acts primitively on Y. | — |
| `IsTransitive(G, Y)` / `IsTransitive(G)` | True if G acts transitively on Y. | — |
| `IsTransitive(G, Y, k)` / `IsTransitive(G, k)` | True if G acts k-transitively on Y. | — |
| `IsSharplyTransitive(G, Y, k)` / `IsSharplyTransitive(G, k)` | True if G acts sharply k-transitively on Y. | — |
| `Transitivity(G, Y)` / `Transitivity(G)` | Degree of transitivity of G on Y. | — |
| `IsRegular(G, Y)` / `IsRegular(G)` | True if G acts regularly on Y. | Algorithm of Sims **[CB92]**. |
| `IsSemiregular(G, Y)` / `IsSemiregular(G)` | True if G acts semiregularly on Y. | Variation of Sims' regularity test **[CB92]**. |
| `IsSemiregular(G, Y, S)` / `IsSemiregular(G, S)` | True if G acts semiregularly on S (a union of orbits for G in its action on Y). | Variation of Sims' regularity test **[CB92]**. |
| `IsFrobenius(G)` | True if G is a Frobenius group: transitive but non-regular, with trivial pointwise stabilizer of any two distinct points. | — |

*Worked examples: H58E24 (M₂₄: images, stabilizer of a point and of a sequence, Steiner system block orbit).*

### 58.10.4 Action on a G-Space

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Action(G, Y)` | Homomorphism φ: G → L (action on G-set Y), induced group L, and kernel. | — |
| `ActionImage(G, Y)` | Permutation group L giving the action of G on G-set Y. | — |
| `ActionKernel(G, Y)` | Kernel of the action homomorphism G → L. | — |
| `IsFaithful(G, Y)` | True if the action of G on Y is faithful. | — |

*Worked examples: H58E25 (PSL(3,4) on flags; coset stabilizer).*

### 58.10.5 Action on Orbits

Efficient algorithms for G-sets that are G-invariant subsets of the natural G-set; see **[But85]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `OrbitAction(G, T)` | Homomorphism f: G → L induced by action of G on G-invariant subset T (union of orbits). | Efficient orbit-based method **[But85]**. |
| `OrbitImage(G, T)` | Group L from action of G on G-invariant subset T. | **[But85]** |
| `OrbitKernel(G, T)` | Kernel of f: G → L for action on G-invariant T. | **[But85]** |
| `IsOrbit(G, S)` | True if subset S of Support(G) is G-invariant. | — |

*Worked examples: H58E26 (intransitive group of degree 36; restriction to orbit of size 12; elementary abelian kernel), H58E28 (Rubik's cube group decomposition into primitive components via OrbitAction and BlocksAction).*

### 58.10.6 Action on a G-invariant Partition

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsBlock(G, S)` | True if subset S of the natural G-set X is a block for the transitive group G. | — |
| `IsPrimitive(G)` | True if transitive G is primitive (no non-trivial block systems). | — |
| `MaximalPartition(G)` | A maximal G-invariant partition P of X (no block of P is properly contained in any block of another G-invariant partition). Returns the trivial partition {X} if G is primitive. | — |
| `MinimalPartition(G : parameters)` | A non-trivial minimal G-invariant partition P (no block of P properly contains a block of another G-invariant partition). Parameter `Block := S` restricts to partitions with a block containing S; if S is non-empty, uses algorithm of Atkinson **[Atk75]**. Returns empty set if G is primitive or no valid partition found. | Schönert-Seress **[SS94]** (general); Atkinson **[Atk75]** (with Block parameter). |
| `MinimalPartitions(G : parameters)` | All non-trivial minimal G-invariant partitions. Parameter `Limit := n` returns at most n block systems. | Schönert-Seress **[SS94]**. |
| `AllPartitions(G)` | All non-trivial G-invariant partitions; returned as a set of blocks containing the first element of Labelling(G). | — |
| `BlocksAction(G, P)` (four overloads) | Action of G on blocks of G-invariant partition P; P may be the full partition, a single block, etc. Returns natural homomorphism f: G → L, induced group, kernel. | — |
| `BlocksImage(G, P)` (four overloads) | Group induced by action of G on blocks of P. | — |
| `BlocksKernel(G, P)` (four overloads) | Kernel of action on blocks of P. | — |

*Worked examples: H58E27 (Capel's degree-100 imprimitive group: MaximalPartition, MinimalPartition, MinimalPartitions, BlocksAction), H58E28 (Rubik's cube analysis via OrbitAction and BlocksAction).*

### 58.10.7 Action on a Coset Space

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CosetAction(G, H : parameters)` | Permutation representation of G on the right cosets of H. Returns homomorphism f: G → L, induced group L, kernel K. Parameter `Al` ("Wang"/"Canonical"/"Default"): Wang's algorithm (stabilizer chain-based induction/blocks, default for trivial H) **[Wang]**; Canonical algorithm of Richardson **[Ric73]** (canonical minimal-base-image coset representatives). | Wang's algorithm or Richardson canonical algorithm **[Ric73]**. |
| `CosetImage(G, H : parameters)` | Image L of G in the coset representation. | As CosetAction. |
| `CosetKernel(G, H)` | Kernel of the coset representation. | — |

### 58.10.8 Reduced Permutation Actions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `TransitiveQuotient(G)` | Transitive constituent of G on its longest orbit, plus action homomorphism and kernel. Returns G, identity, trivial subgroup if G is transitive. | OrbitAction on longest orbit. |
| `PrimitiveQuotient(G)` | Blocks image of transitive G on a maximal block system, plus action homomorphism and kernel. Returns G, identity, trivial subgroup if G is primitive. | BlocksAction on MaximalPartition. |
| `DegreeReduction(G)` | Attempt to find a faithful permutation representation of G of lower degree using orbit images and blocks images. Returns G and identity map if no reduction found. | Combination of OrbitAction and BlocksAction. |

### 58.10.9 The Jellyfish Algorithm

The Jellyfish algorithm **[LNPS06]** finds faithful low-degree permutation representations for large-base primitive permutation groups in the target family: subgroups G of W = Sₙ ≀ Sᵣ (product action on r-tuples of k-subsets of an n-set) with M = Aₙʳ ≤ G, conjugation action transitive on the r copies of Aₙ, n > 2rk², rk > 1. Target degree is n^r·C(n,k)^r; image degree is nk. Groups not in the target family are alternating, symmetric, or short-base. Algorithm is one-sided Monte-Carlo (success guarantees faithfulness).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `JellyfishConstruction(G : parameters)` | Attempt Jellyfish construction for G. If successful, stores data (T1 and T5 of **[LNPS06]**) with G and returns true. Parameter `Limit` controls random element attempts (default: max(15, 2⌊log₂(deg)⌋)). | Jellyfish algorithm **[LNPS06]**. |
| `JellyfishImage(G)` | Faithful image of G found by Jellyfish (applies JellyfishConstruction if needed; errors on failure). | **[LNPS06]** |
| `JellyfishImage(G, x)` | Image of element x (may fail if x ∉ G, proving non-membership; may succeed for x ∉ G). Parent is a symmetric group. | **[LNPS06]** |
| `JellyfishPreimage(G, x)` | Preimage of Jellyfish-image element x (nearly linear in the large-degree). May fail or succeed spuriously; parent is a symmetric group. | Extension of **[LNPS06]** data structure. |

---

## 58.11 Normal and Subnormal Subgroups

### 58.11.1 Characteristic Subgroups and Normal Series

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DerivedSeries(G)` | Derived series of G as a sequence of subgroups. | Algorithm of Butler-Cannon **[BC82]**. |
| `CompositionSeries(G)` | A composition series (descending chain of normal subgroups, each quotient simple) as a sequence. | — |
| `CommutatorSubgroup(G)` / `DerivedSubgroup(G)` / `DerivedGroup(G)` | Derived subgroup G' = [G, G]. | — |
| `SolubleResidual(G)` / `SolvableResidual(G)` | Last term of the derived series (solvable residual). | — |
| `DerivedLength(G)` | Derived length of G; for non-solvable G, length of series terminating at the solvable residual. | — |
| `LowerCentralSeries(G)` | Lower central series as a sequence (first element is G). | Algorithm of Butler-Cannon **[BC82]**. |
| `NilpotencyClass(G)` | Nilpotency class; returns -1 if G is not nilpotent. | — |
| `UpperCentralSeries(G)` | Upper central series (sequence from trivial subgroup up). Uses centre and section centralisers via Luks **[Luk93]** up the chain; more restricted range than DerivedSeries. | Centre computation + SectionCentraliser (Luks **[Luk93]**). |
| `Centre(G)` / `Center(G)` | Centre of G, via CentralizerOfNormalSubgroup(G, G). | Beals **[Bea93]** (polynomial-time). |
| `Hypercentre(G)` / `Hypercenter(G)` | Stationary term of the upper central series. | — |
| `pCore(G, p)` | Maximal normal p-subgroup of G. | Algorithm of Unger **[Ung06b]**. |
| `pCoreQuotient(G, p)` | Quotient G/pCore(G,p) as permutation group of the same degree, quotient map, and pCore. | — |
| `FittingSubgroup(G)` | Fitting subgroup: product of p-cores of the radical. | — |
| `FrattiniSubgroup(G)` | Frattini subgroup. For p-groups: derived group with p-th powers. For solvable: via GrpPC. For non-solvable: intersection of maximal subgroups (same restrictions as MaximalSubgroups). | — |
| `JenningsSeries(G)` | Jennings series of p-group G as sequence of subgroups. | — |
| `pCentralSeries(G, p)` | Lower p-central series of solvable G and prime p as sequence of subgroups. | — |
| `SubnormalSeries(G, H)` | A subnormal series from G to subnormal subgroup H (each term normal in preceding); returns empty sequence if H is not subnormal in G. | — |

*Worked examples: H58E29 (derived series, DerivedLength, LowerCentralSeries, NilpotencyClass, Centre, pCentralSeries of WreathProduct(Sym(4), DihedralGroup(4))).*

### 58.11.2 Maximal and Minimal Normal Subgroups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `MaximalNormalSubgroup(G)` | A maximal normal subgroup of G (trivial if G is simple). Uses homomorphic reductions to a primitive group and O'Nan-Scott considerations. | O'Nan-Scott type analysis. |
| `MinimalNormalSubgroups(G)` | The minimal normal subgroups of G (computed via the socle and splitting normal factors). | — |

### 58.11.3 Lattice of Normal Subgroups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `NormalSubgroups(G)` | All normal subgroups of G. | Method of Cannon-Souvignier **[CS]**. |
| `NormalLattice(G)` | Normal subgroup lattice with inclusion relations, using the same method. | Cannon-Souvignier **[CS]** + inclusion determination. |

*Worked examples: H58E30 (29 normal subgroups of WreathProduct(Sym(8), DihedralGroup(4)) of order ~2×10¹⁹).*

### 58.11.4 Composition and Chief Series

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ChiefFactors(G)` | Sequence of isomorphism types `<f, d, q, m>` of chief factors (m copies of the simple group described by `<f, d, q>`; see Tables 1 and 2). | Algorithm of Unger **[Ung]**. |
| `ChiefSeries(G)` | Chief series and corresponding isomorphism type sequence. Organised to include the solvable radical, and (if insoluble) the socle of the radical quotient. | Algorithm of Unger **[Ung]**. |
| `CompositionFactors(G)` | Sequence of triples `<f, d, q>` for composition factors of G ordered by some composition series. Family encodings (f=1: A(d,q), f=2: B(d,q), ..., f=17: Alternating(d), f=18: Sporadic (see Table 2), f=19: Cyclic(q)). | Tabular algorithm of Kantor **[Kan91]**, valid for groups of degree ≤ 10⁷. |

*Worked examples: H58E31 (composition factors of the Rubik's cube group).*

### 58.11.5 The Socle

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Socle(G)` | Socle of G (product of minimal normal subgroups). | Cannon-Holt **[CH97]**. |
| `SocleFactor(G)` | A simple factor of the socle. | — |
| `SocleFactors(G)` | Simple factors of the socle as a sequence; index corresponds to points of SocleAction image. | — |
| `SocleSeries(G)` | Chain S₁, S₁×S₂, ..., S₁×...×Sᵣ (Sᵢ simple factors of socle of primitive G). | — |
| `EARNS(G)` | Elementary abelian regular normal subgroup (EARNS) of primitive G; trivial subgroup if none exists. | Algorithm of Neumann **[Neu86]**. |
| `IsAffine(G)` | True if G is of primitive affine type; also returns EARNS if so. Combines IsTransitive, IsPrimitive, EARNS. | EARNS **[Neu86]**. |
| `AffineAction(G)` | Representation of G (with non-trivial EARNS A) by action on elements of A; image is a point stabilizer; kernel is A. Returns homomorphism, image, kernel. | — |
| `AffineImage(G)` | Image of G from the affine action. | — |
| `AffineKernel(G)` | Kernel of the affine action (equals A). | — |
| `SocleAction(G)` | For G with trivial Fitting subgroup: action of G on simple factors of the socle N. Returns homomorphism, image, kernel. | — |
| `SocleImage(G)` | Permutation group L induced by action on socle factors. | — |
| `SocleKernel(G)` | Kernel of the action on socle factors. | — |
| `SocleQuotient(G)` | Permutation representation of G/N; degree bounded by Σ|Out(Uᵢ)|. Returns G/N, quotient homomorphism, kernel (= socle). | Cannon-Souvignier **[CS]**. |
| `RefineSection(G, M, N)` | For N ◁ G, M ◁ G, N < M: a sequence N = L₀ < L₁ < ... < Lᵣ = M of G-normal subgroups such that each Lᵢ₊₁/Lᵢ is elementary abelian or a direct product of non-abelian simple groups. | — |

*Worked examples: H58E32 (PrimitiveWreathProduct(Sym(5),Sym(3)): EARNS, DerivedSeries, Socle, SocleFactors, SocleSeries, SocleQuotient).*

### 58.11.6 The Soluble Radical and its Quotient

The key idea: compute solvable radical R, solve for G/R (which has a faithful representation of degree ≤ deg(G) by Holt's theorem), then lift solutions down the elementary abelian series using the Lifting Strategy.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Radical(G)` / `SolubleRadical(G)` / `SolvableRadical(G)` | Maximal normal solvable subgroup of G. | Algorithm of Unger **[Ung06b]**. |
| `RadicalQuotient(G)` | Quotient G/R as a permutation group of the same degree, with homomorphism φ: G → Q and R (the kernel). Proceeds by repeatedly applying AbelianNormalQuotient up the derived series of R. | Lifting strategy via AbelianNormalQuotient. |
| `ElementaryAbelianSeries(G : parameters)` / `ElementaryAbelianSeries(G, N : parameters)` | Elementary abelian series R = N₁ > N₂ > ... > Nᵣ = {1} (or down to N). Each consecutive quotient is elementary abelian; top is the solvable radical. Parameter `LayerSizes` controls layer refinement. | — |
| `ElementaryAbelianSeriesCanonical(G)` | Elementary abelian series consisting of characteristic subgroups of G (depends only on isomorphism type of the radical). May be slower. | — |

*Worked examples: H58E33 (Radical, RadicalQuotient, ElementaryAbelianSeries of a group of degree 16).*

### 58.11.7 Complements and Supplements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Complements(G, M)` | Representatives of conjugacy classes of complements of normal subgroup M in G. | — |
| `Complements(G, M, N)` | Representatives of conjugacy classes of complements of M/N in G/N as subgroups of G (N ◁ G, N ⊊ M). | — |
| `HasComplement(G, M)` | True if M (normal in G) has a complement; if so, returns one. | — |
| `Supplements(G, M)` | Representatives of conjugacy classes of minimal supplements for solvable normal M in G. | — |
| `Supplements(G, M, N)` | Representatives for classes of minimal supplements of M/N in G/N (N ◁ G, N ⊊ M, M/N solvable). | — |
| `HasSupplement(G, M)` | True if solvable normal M has a proper supplement; if so, returns one. | — |

*Worked examples: H58E34 (Complements of normal subgroup of order 6912 in group of order 165888).*

### 58.11.8 Abelian Normal Subgroups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AbelianNormalSubgroup(G)` | An abelian normal subgroup of G (trivial if none exists). | — |
| `AbelianNormalQuotient(G, H)` | Quotient of G by an abelian normal subgroup containing H. Returns quotient (same degree), epimorphism, kernel K (with K ⊇ H; #K and #H have the same prime divisors; if H is elementary abelian, so is K). | — |
| `SolubleNormalQuotient(G, H)` | Quotient of G by a solvable normal subgroup containing H. Returns quotient, epimorphism, kernel K (K ⊇ H; #K and #H have same prime divisors). | — |
| `ElementaryAbelianNormalSubgroup(G)` | An elementary abelian normal subgroup of G (last non-trivial group in elementary abelian series of radical); trivial if none. | — |
| `pElementaryAbelianNormalSubgroup(G, p)` | An elementary abelian normal p-subgroup (last non-trivial group in elementary abelian series for pCore(G,p)); trivial if none. | — |
| `MEANS(G)` | A minimal elementary abelian normal subgroup of G. | — |
| `MEANS(G, N)` | A minimal elementary abelian normal subgroup of G contained in the elementary abelian normal subgroup N. | — |

---

## 58.12 Cosets and Transversals

### 58.12.1 Cosets

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `H * g` | Right coset H*g of subgroup H of G (g ∈ G). | — |
| `DoubleCoset(G, H, g, K)` | Double coset H*g*K of subgroups H, K of G (g ∈ G). | — |
| `DoubleCosetRepresentatives(G, H, K)` | Representatives of H-K-double cosets in G (first is always identity); also sizes. Refines double cosets down a subgroup chain from G. | Chain-of-subgroups refinement. |
| `ProcessLadder(L, G, U)` | Compute data for Lₙ-U-double cosets in G given a ladder L (L₁=G). Parameter `Verbose`, `DoubleCosets` (maximum 3). | Ladder method of Schmalz **[Sch90]**. |
| `GetRep(p, R)` | Canonical double coset representative for permutation p ∈ G, given data R from ProcessLadder. | **[Sch90]** |
| `DeleteData(R)` | Free the data structure computed by ProcessLadder. | — |
| `YoungSubgroupLadder(L)` | Ladder from Sym(n) down to YoungSubgroup(L) for double coset enumeration. Parameter `Full` sets ambient symmetric group degree. | — |
| `StabilizerLadder(G, d)` | Ladder from Sym(n) to the stabilizer of monomial d, for processing by ProcessLadder. | — |
| `x in C` | True if element x of G lies in coset C. | — |
| `x notin C` | True if x does not lie in coset C. | — |
| `C1 eq C2` | True if cosets C1 and C2 are equal. | — |
| `C1 ne C2` | True if C1 ≠ C2. | — |
| `#C` | Cardinality of coset C. | — |
| `CosetTable(G, H)` | Right coset table for G over H relative to defining generators. | Todd-Coxeter. |
| `#CosetTable(G, f)` | Coset table for G corresponding to permutation representation f (f a homomorphism from G onto a transitive permutation group). | — |

### 58.12.2 Transversals

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Transversal(G, H)` / `RightTransversal(G, H)` | Indexed set T forming a right transversal for G over H, and the transversal mapping φ: G → T (φ(g) = tᵢ where g ∈ H*tᵢ). | — |
| `TransversalProcess(G, H)` | Process to enumerate a left transversal of H in G via backtrack search for canonical coset representatives (use when index is too large for full transversal). | Backtrack canonical representative. |
| `TransversalProcessRemaining(P)` | Number of coset representatives the process P has yet to produce. | — |
| `TransversalProcessNext(P)` | Advance to next coset representative (first call always gives identity). | — |
| `ShortCosets(p, H, G)` | Representatives of all cosets of G mod H that contain element p, without computing the full transversal. | — |

---

## 58.13 Presentations

### 58.13.1 Generators and Relations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `FPGroup(G)` | Presentation of G on its defining generators as an fp-group F isomorphic to G. Also returns homomorphism φ: F → G. Uses regular representation then Todd-Coxeter-Schreier to find presentation on strong generators (which are identical to defining generators in this context). | Todd-Coxeter-Schreier algorithm. |
| `FPQuotient(G, N)` | fp-group representation F of G/N and homomorphism φ: G → F. | — |
| `FPGroupStrong(G : parameters)` | Presentation of G on strong generators as fp-group F isomorphic to G; also the isomorphism φ: F → G. Uses Schreier-Todd-Coxeter-Sims combined with Brownie-Cannon-Sims verification. Parameters: `Random` (default true, enables random Schreier pre-processing), `Run` (default 20). | Schreier-Todd-Coxeter-Sims + Brownie-Cannon-Sims verification **[Leo80, Geb00]**. |

### 58.13.2 Permutations as Words

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `WordGroup(G)` | Given G on d generators: returns (a) a free group W on d generators (SLP group type), and (b) the homomorphism φ: W → G with W.i → G.i. | — |
| `InverseWordMap(G)` | Inverse mapping ρ: G → W (word group); g@ρ gives a preimage of g under φ. Creates word group if not already present. | BSGS-based word decomposition. |
| `ActingWord(G, x, y)` | Given x, y in the same G-orbit, returns a word w in the word group W such that x^φ(w) = y. | BSGS-based. |

---

## 58.14 Automorphism Groups

Subject to the same restrictions as MaximalSubgroups (non-abelian composition factors must appear in the database).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AutomorphismGroup(G : parameters)` | Full automorphism group of the permutation group G. | Methods of Cannon-Holt **[CH03]**. |
| `IsIsomorphic(G, H : parameters)` | True if G and H are isomorphic as abstract groups; if so, also returns an isomorphism. | Methods of Cannon-Holt **[CH03]**. |

*Worked examples: H58E35 (isomorphism tests among groups of order 120; automorphism group of a perfect group of order 120).*

---

## 58.15 Cohomology

G is a finite permutation group, p a prime, K = F_p. F is an fp-group with the same number of generators as G and an epimorphism onto G. Algorithms of Holt **[Hol84, Hol85a, Hol85b]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `pMultiplicator(G, p)` | Invariant factors of the p-part of the Schur multiplicator of G. | Holt **[Hol84]**. |
| `pCover(G, F, p)` | Presentation for the p-cover of G as an extension of the p-multiplier by F. | Holt **[Hol85a]**. |
| `CohomologicalDimension(G, M, i)` | Dimension of the i-th cohomology group H^i(G, M) (i = 1 or 2), where M is a K[G]-module. | Holt **[Hol85b]**. |
| `ExtensionProcess(G, M, F)` | Create an extension process for G by module M. | Holt **[Hol85b]**. |
| `Extension(P, Q)` | Construct the extension corresponding to vector Q = [a₁,...,aₗ] ∈ H²(G,M), returned as an fp-group. | Holt **[Hol85b]**. |
| `#NextExtension(P)` | Return the next extension defined by process P. | — |
| `SplitExtension(G, M, F)` | The split extension of module M by group G. | — |

*Worked examples: H58E36 (6-fold cover of A₆), H58E37 (two extensions of A₅ by F₂[A₅]-module of dimension 5).*

---

## 58.16 Representation Theory

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CharacterTable(G : parameters)` | Table of ordinary irreducible characters for G. Parameter `Al` ("Default"/"DS"/"IR"); `DSSizeLimit` controls Dixon-Schneider pre-processing threshold. Default: Dixon-Schneider for |G| ≤ 5000; Unger's induction/reduction algorithm **[Ung06a]** for larger groups. | Dixon-Schneider (small groups) or Unger's induction/reduction **[Ung06a]**. |
| `PermutationCharacter(G)` | Character afforded by the natural permutation representation of G. | — |
| `PermutationCharacter(G, H)` | Character afforded by the permutation representation of G on the cosets of H. | — |
| `GModule(G, S)` | K[G]-module M from G (r generators) and subalgebra S of Mₙ(R) (r matrices); i-th generator of G acts by i-th matrix of S. | — |
| `GModule(G, A, B)` | K[G]-module from the action of G on elementary abelian section A/B (K = F_p, |A/B| = p^n). Returns module M and homomorphism φ: A/B → M. B may be omitted if trivial. | — |
| `PermutationModule(G, H, R)` | R[G]-module for the permutation action of G on cosets of H over ring R. | — |
| `PermutationModule(G, R)` | Natural permutation module for G over ring R. | — |

*Worked examples: H58E38 (refining an elementary abelian normal subgroup of order 8 via GModule and Submodules).*

---

## 58.17 Identification

### 58.17.1 Identification as an Abstract Group

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `NameSimple(G)` | For simple G: returns isomorphism type as triple `<f, d, q>` (same encoding as CompositionFactors). | Database lookup / recognition. |

### 58.17.2 Identification as a Permutation Group

Based on the 'Detect Alternating' algorithm of **[CB92]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsAlternating(G)` | True if G = Alt(X) in its natural action on X. | Detect Alternating **[CB92]**. |
| `IsSymmetric(G)` | True if G = Sym(X) in its natural action on X. | Detect Alternating **[CB92]**. |
| `IsAltsym(G)` | True if G ⊇ Alt(X). | Detect Alternating **[CB92]**. |
| `TwoTransitiveGroupIdentification(G)` | For 2-transitive G: a tuple giving the abstract isomorphism type. | Method of Cameron-Cannon **[CC91]**. |
| `RecogniseAlternatingOrSymmetric(G, n)` | Constructive recognition of G ≅ Aₙ or Sₙ (n > 11). Returns: success flag, is-symmetric flag, two programs defining mutually inverse isomorphisms (black-box ↔ natural action). Probability of success ≥ 1 - e⁻⁵. | Black-box algorithm of Beals et al. **[BLGN+03]** (implemented by Roney-Dougal). |
| `IsEven(G)` | True if G is even (contained in the alternating group). | — |
| `RecogniseSymmetric(G, n : parameters)` | G ≅ Sₙ (n ≥ 8): construct isomorphism using Bratus-Pak algorithm **[BP00]** (implemented by Holt). Returns true, φ: G → Sₙ, φ⁻¹, word map G → word group, word-to-G map. Parameter `Extension` (also handles 2.Sₙ), `maxtries` (default 100n+5000). | Bratus-Pak **[BP00]**. |
| `SymmetricElementToWord(G, g)` | If G has been constructively recognised as Sₙ (or 2.Sₙ): returns true and word group element evaluating to g; otherwise false. Facilitates membership testing. | Bratus-Pak inverse map **[BP00]**. |
| `RecogniseAlternating(G, n : parameters)` | G ≅ Aₙ (n ≥ 9): construct isomorphism using Bratus-Pak. Same returns and parameters as RecogniseSymmetric. | Bratus-Pak **[BP00]**. |
| `AlternatingElementToWord(G, g)` | If G recognised as Aₙ (or 2.Aₙ): returns true and word; otherwise false. | Bratus-Pak inverse map **[BP00]**. |
| `GuessAltsymDegree(G : parameters)` | Guess n and symmetry/alternating type for G believed isomorphic to Sₙ or Aₙ (n > 6) by sampling element orders. Returns false (after `maxtries` attempts, default 5000) or true, type, n. No guarantee if G is not Sₙ/Aₙ. Parameter `Extension` (for 2.Sₙ/2.Aₙ). | Element-order sampling (Derek Holt). |

*Worked examples: H58E39 (RecogniseAlternatingOrSymmetric on degree-78 coset image of Alt(13)), H58E40 (GuessAltsymDegree + RecogniseAlternating for A₁₀ in GL(10,5)).*

---

## 58.18 Base and Strong Generating Set

A **base** B = [b₁,...,bₖ] for G ≤ Sym(Ω) is a sequence of distinct points with trivial pointwise stabilizer. A **strong generating set** S (w.r.t. B) satisfies G^(i) = ⟨S ∩ G^(i)⟩ for the stabilizer chain G^(1) = G > G^(2) > ... > G^(k+1) = {1}. A BSGS enables immediate order computation: |G| = Π|Δᵢ| (basic orbit lengths). Brownie-Cannon-Sims (1991) showed BSGS construction is practical for short-base groups up to degree 10⁷.

### 58.18.1 Construction of a Base and Strong Generating Set

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BSGS(G)` | General BSGS construction using default algorithm choices. | Default algorithm selection. |
| `SimsSchreier(G : parameters)` | Schreier-Sims algorithm for G. Parameter `SV` (default true): if true, transversals stored as Schreier vectors; if false, also as permutation lists. If base attribute previously defined, uses base-image-optimized variant. | Schreier-Sims algorithm **[Leo80]**. |
| `RandomSchreier(G : parameters)` | Probable BSGS from random elements. Terminates when BSGS defines a group of the asserted order (if Order attribute is set) or after Max random elements with Run consecutive elements already in the BSGS. Parameters: `Max` (default 100), `Run` (default 20). Most efficient for groups with known order or large degree. | Random Schreier-Sims. |
| `ToddCoxeterSchreier(G : parameters)` | BSGS via Todd-Coxeter-Schreier algorithm. | Todd-Coxeter-Schreier (TCSS). |
| `SolubleSchreier(G : parameters)` / `SolvableSchreier(G : parameters)` | BSGS for solvable G via algorithm of Sims **[Sim90]** (works by recursively constructing derived series terms). Parameter `Depth` (default ⌈1.6 log₂ deg(G)⌉, based on Dixon's bound on derived series length). Does not terminate for non-solvable G. Significantly faster than general Schreier-Sims for solvable groups. | Sims' solvable algorithm **[Sim90]**. |
| `Verify(G : parameters)` | Given G with a probable BSGS, verify and complete it. Parameters: `Levels` (levels verified by TCSS before switching to Brownie-Cannon-Sims (BCS)), `OrbitLimit` (default 4000; orbit length threshold for TCSS vs BCS). | Brownie-Cannon-Sims (BCS) verification. |

*Worked examples: H58E41 (ToddCoxeterSchreier for Higman-Sims group on 100 letters), H58E42 (RandomSchreier + Verify for Rudvalis group Ru on 4060 letters), H58E43 (AssertAttribute Order + RandomSchreier for WreathProduct(Sym(42),Alt(8))).*

### 58.18.2 Defining Values for Attributes

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AssertAttribute(G, "Order", n)` | Define the order attribute for G (as integer). | — |
| `AssertAttribute(G, "Order", Q)` | Define the factored order attribute for G. | — |
| `#AssertAttribute(G, "BSGS", S)` | Define the BSGS structure S for G (advanced use). | — |

### 58.18.3 Accessing the Base and Strong Generating Set

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Base(G)` | Base for G as a sequence of points. Computes BSGS if not known. | — |
| `BasePoint(G, i)` | The i-th base point. BSGS must be known. | — |
| `BasicOrbit(G, i)` | Basic orbit at level i (BSGS must be known). | — |
| `BasicOrbits(G)` | All basic orbits as a sequence of indexed sets (BSGS must be known). | — |
| `BasicOrbitLength(G, i)` | Length of basic orbit at level i (BSGS must be known). | — |
| `BasicOrbitLengths(G)` | Lengths of all basic orbits as a sequence of integers (BSGS must be known). | — |
| `BasicStabilizer(G, i)` / `BasicStabiliser(G, i)` | Subgroup G^(i): fixes first i-1 base points. BSGS must be known. | — |
| `BasicStabilizerChain(G)` / `BasicStabiliserChain(G)` | Stabilizer chain as sequence of subgroups. Constructs BSGS if not known. | — |
| `IsMemberBasicOrbit(G, i, a)` | True if point a lies in the basic orbit at level i. BSGS must be known. | BSGS lookup. |
| `NumberOfStrongGenerators(G)` / `Nsgens(G)` | Number of elements in the current strong generating set. | — |
| `NumberOfStrongGenerators(G, i)` / `Nsgens(G, i)` | Number of strong generators for the i-th stabilizer chain term. | — |
| `SchreierVectors(G)` | Schreier vectors for the current BSGS as a sequence of integer sequences. | — |
| `SchreierVector(G, i)` | Schreier vector for the i-th stabilizer chain term. | — |
| `StrongGenerators(G)` | Set of strong generators; computed if not available. | — |
| `StrongGenerators(G, i)` | Strong generators for the i-th stabilizer chain term. BSGS must be known. | — |

### 58.18.4 Working with a Base and Strong Generating Set

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BaseImage(x)` | Base image of permutation x ∈ G (BSGS known for G). | — |
| `Permutation(G, Q)` | Given base-image sequence Q (distinct points defining an element of G), return the permutation. BSGS must be known. | BSGS coset decomposition. |
| `SVPermutation(G, i, a)` | Permutation defined by the Schreier vector at level i, taking point a to the level-i base point. BSGS must be known. | Schreier vector traversal. |
| `SVWord(G, i, a)` | Word group element corresponding to the Schreier vector path at level i from a to base point. BSGS must be known. | Schreier vector traversal. |
| `Strip(H, x)` | Strip x through the BSGS of H. Returns: membership boolean, residual permutation y, first level i where y ∉ H^(i). | Schreier-Sims strip. |
| `WordStrip(H, x)` | Strip x through the BSGS of H. Returns: membership boolean, residual word w (in word group of G), first level i of failure. | Schreier-Sims word strip. |
| `BaseImageWordStrip(H, x)` | Strip x using base images. Returns: whether base image strip succeeded at all levels, residual word w, first level of failure. | Base-image word strip. |
| `WordInStrongGenerators(H, x)` | For x ∈ H (BSGS known): a word in strong generators equal to x (inverse of second return of BaseImageWordStrip). | Base-image decomposition. |

### 58.18.5 Modifying a Base and Strong Generating Set

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ChangeBase(~G, Q)` | Change base of G so that points in Q form an initial segment of the new base. | Base change algorithm. |
| `AddNormalizingGenerator(~H, x)` | Extend BSGS of H to a BSGS for ⟨H, x⟩, where x normalizes H. | BSGS extension. |
| `ReduceGenerators(~G)` | Remove redundant strong generators from the BSGS of G. | — |

---

## 58.19 Permutation Representations of Linear Groups

Each function returns: (a) a permutation group G, (b) an indexed set giving the correspondence between the geometric points and the G-set. Parameters are degree n and field (as integer q, finite field K, or vector space V).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AffineGeneralLinearGroup(arguments)` / `AGL(arguments)` | AGL(n,q) = GL(n,q) acting on affine points of Fq^n. | — |
| `AffineSpecialLinearGroup(arguments)` / `ASL(arguments)` | ASL(n,q) = SL(n,q) acting on affine points. | — |
| `AffineGammaLinearGroup(arguments)` / `AGammaL(arguments)` | AΓL(n,q) = Aut(GL(n,q)) acting on affine points. | — |
| `AffineSigmaLinearGroup(arguments)` / `ASigmaL(arguments)` | AΣL(n,q) = Aut(SL(n,q)) acting on affine points. | — |
| `ProjectiveGeneralLinearGroup(arguments)` / `PGL(arguments)` | PGL(n,q) = GL(n,q) acting on projective points of Fq^n (n ≥ 2). | — |
| `ProjectiveSpecialLinearGroup(arguments)` / `PSL(arguments)` | PSL(n,q) = SL(n,q) acting on projective points (n ≥ 2). | — |
| `ProjectiveGammaLinearGroup(arguments)` / `PGammaL(arguments)` | PΓL(n,q) = PGL(n,q) extended by field automorphisms of Fq (n ≥ 2). | — |
| `ProjectiveSigmaLinearGroup(arguments)` / `PSigmaL(arguments)` | PΣL(n,q) = PSL(n,q) extended by field automorphisms (n ≥ 2). | — |
| `ProjectiveGeneralUnitaryGroup(arguments)` / `PGU(arguments)` | PGU(n,q) over Fq² (n ≥ 2). | — |
| `ProjectiveSpecialUnitaryGroup(arguments)` / `PSU(arguments)` | PSU(n,q) over Fq² (n ≥ 2). | — |
| `ProjectiveGammaUnitaryGroup(arguments)` / `PGammaU(arguments)` | PΓU(n,q) = PGU(n,q) extended by field automorphisms of Fq² (n ≥ 2). | — |
| `ProjectiveSigmaUnitaryGroup(arguments)` / `PSigmaU(arguments)` | PΣU(n,q) = PSU(n,q) extended by field automorphisms of Fq² (n ≥ 2). | — |
| `ProjectiveSymplecticGroup(arguments)` / `PSp(arguments)` | PSp(n,q), n even ≥ 4, over Fq. | — |
| `ProjectiveSigmaSymplecticGroup(arguments)` / `PSigmaSp(arguments)` | PΣSp(n,q) = PSp(n,q) extended by field automorphisms (n even ≥ 4). | — |
| `ProjectiveGeneralOrthogonalGroup(arguments)` / `PGO(arguments)` | PGO(n,q), n odd ≥ 3, over Fq. | — |
| `ProjectiveGeneralOrthogonalGroupPlus(arguments)` / `PGOPlus(arguments)` | PGO⁺(n,q), n even ≥ 2, over Fq. | — |
| `ProjectiveGeneralOrthogonalGroupMinus(arguments)` / `PGOMinus(arguments)` | PGO⁻(n,q), n even ≥ 2, over Fq. | — |
| `ProjectiveSpecialOrthogonalGroup(arguments)` / `PSO(arguments)` | PSO(n,q), n odd ≥ 3, over Fq. | — |
| `ProjectiveSpecialOrthogonalGroupPlus(arguments)` / `PSOPlus(arguments)` | PSO⁺(n,q), n even ≥ 2, over Fq. | — |
| `ProjectiveSpecialOrthogonalGroupMinus(arguments)` / `PSOMinus(arguments)` | PSO⁻(n,q), n even ≥ 2, over Fq. | — |
| `ProjectiveOmega(arguments)` / `POmega(arguments)` | PΩ(n,q), n odd ≥ 3, over Fq. | — |
| `ProjectiveOmegaPlus(arguments)` / `POmegaPlus(arguments)` | PΩ⁺(n,q), n even ≥ 2, over Fq. | — |
| `ProjectiveOmegaMinus(arguments)` / `POmegaMinus(arguments)` | PΩ⁻(n,q), n even ≥ 2, over Fq. | — |
| `ProjectiveSuzukiGroup(arguments)` / `PSz(arguments)` | PSz(q), Suzuki simple group Sz(q) (q = 2^{2n+1}), degree 4 over Fq. | — |
| `AffineGroup(M)` | For matrix group M of degree d over F: the semidirect product V:M where V = F^d. Returns G (degree |F|^d) and correspondence V → G-set. | — |

---

## 58.20 Permutation Group Databases

Magma includes databases of all transitive permutation groups of degree ≤ 32 and all primitive permutation groups of degree ≤ 4095. These are described in Chapter 66.

---

## 58.21 Ordered Partition Stacks

Ordered partition stacks implement the data structure of Leon **[Leo97]** (section 2), useful for implementing backtrack searches. The domain is always {1..n} (degree n). "Ordered" refers to cells being in fixed order; point order within a cell is not significant. The push operation refines a partition per Definition 2 of **[Leo97]**.

### 58.21.1 Construction of Ordered Partition Stacks

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `OrderedPartitionStack(n)` | Create a complete ordered partition stack of degree n (initially one partition with a single block). | — |
| `OrderedPartitionStackZero(n, h)` | Create a zero-based ordered partition stack of degree n with height limit h (initially height 0, single block). | — |

### 58.21.2 Properties of Ordered Partition Stacks

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Degree(P)` | Degree of the ordered partition stack P. | — |
| `Height(P)` | Height of P (for a complete stack = number of cells of the finest partition). | — |
| `NumberOfCells(P, h)` | Number of cells in the partition at height h (default: height of P). | — |
| `CellNumber(P, h, x)` | Number of the cell at height h containing element x (default h: height of P). | — |
| `CellSize(P, h, i)` | Size of cell i at height h (default h: height of P). | — |
| `Cell(P, h, i)` | Contents of cell i at height h as a sequence of integers (order may vary; default h: height of P). | — |
| `Random(P, i)` | A random element of cell i of the finest partition on P. | — |
| `Representative(P, i)` / `Rep(P, i)` | An element of cell i of the finest partition on P. | — |
| `ParentCell(P, i)` | Number of the cell that was split to create cell i. | — |

### 58.21.3 Operations on Ordered Partition Stacks

Splitting pushes a finer partition; the new cell gets number k+1 (where k = current number of cells), and the residue retains the original cell number. This follows Leon **[Leo97]** Definition 2, differing from Seress **[Ser03]** Chapter 9.2 and McKay **[McK81]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SplitCell(P, i, x)` / `SplitCell(P, i, Q)` | Refine top partition by splitting cell i into {x} (or intersection of Q with cell i) and its complement in cell i. Returns true if P changed. | Definition 6 of Leon **[Leo97]**. |
| `SplitAllByValues(P, V)` | Refine top partition by splitting all possible cells using values V. Returns true/false and a hash value suitable as an indicator function (per Definition 2-16 of McKay **[McK81]**). | Definition 15 of Leon **[Leo97]**. |
| `SplitCellsByValues(P, C, V)` / `SplitCellsByValues(P, i, V)` | Refine by splitting cells given in C (or cell i) using values V in the given order; terminates and returns false if any cell in C does not split. | Leon **[Leo97]**. |
| `Pop(P)` / `Pop(P, h)` | Reduce height of P by 1 (or to height h). | "Retract" algorithm of Leon **[Leo97]**, Fig. 7. |
| `Advance(X, L, P, h)` | X: zero-based stack of degree d; L: length-n sequence in {1..d} (unordered partition of {1..n} into d blocks); P: complete stack of degree n; h: positive integer ≤ height(P). Fundamental operation in Leon's unordered partition stabilizer algorithm. | "Advance" algorithm of Leon **[Leo97]**, Fig. 7. |

*Worked examples: H58E44 (degree-12 stack: SplitCell, SplitAllByValues, Pop).*

---

## 58.22 Bibliography

| Key | Reference |
|-----|-----------|
| **[Atk75]** | M. D. Atkinson. An algorithm for finding the blocks of a permutation group. *Math. Comp.*, 29:911–913, 1975. |
| **[BC82]** | G. Butler and J. J. Cannon. Computing with permutation and matrix groups I: Normal closure, commutator subgroups, series. *Math. Comp.*, 39:671–680, 1982. |
| **[Bea93]** | G. Beals. Algorithms for finite groups. PhD thesis, University of Chicago, 1993. |
| **[BLGN+03]** | R. Beals, C. R. Leedham-Green, A. C. Niemeyer, C. E. Praeger, and A. Seress. A black-box algorithm for recognising finite symmetric and alternating groups, I. *Trans. Amer. Math. Soc.*, 2003. To appear. |
| **[BP00]** | Sergey Bratus and Igor Pak. Fast constructive recognition of a black box group isomorphic to Sn or An using Goldbach's conjecture. *J. Symbolic Comp.*, 29:33–57, 2000. |
| **[But85]** | Gregory Butler. Effective computation with group homomorphisms. *J. Symbolic Comp.*, 1:143–157, 1985. |
| **[But94]** | Greg Butler. An inductive schema for computing conjugacy classes in permutation groups. *Mathematics of Computation*, 62(205):363–383, 1994. |
| **[CB92]** | J. J. Cannon and W. Bosma. Structural computation in finite permutation groups. *CWI Quarterly*, 5(2):127–160, 1992. |
| **[CC91]** | P. J. Cameron and J. J. Cannon. Recognizing doubly transitive groups. *J. Symb. Comp.*, 12(4/5):459–474, 1991. |
| **[CCH97]** | J. J. Cannon, B. Cox, and D. F. Holt. Computing Sylow subgroups in permutation groups. *J. Symb. Comp.*, 24(3/4):303–316, 1997. |
| **[CCH01]** | J. J. Cannon, B. Cox, and D. F. Holt. Computing the subgroups of a permutation group. *J. Symb. Comp.*, 31:149–161, 2001. |
| **[CFL89]** | G. Cooperman, L. Finkelstein, and E. M. Luks. Reduction of group constructions to point stabilizers. In *Proc. of International Symposium on Symbolic and Algebraic Computation ISSAC '89*, pages 351–356. ACM, 1989. |
| **[CH97]** | J. J. Cannon and D. F. Holt. Computing chief series, composition series and socles in large permutation groups. *J. Symb. Comp.*, 24(3/4):285–301, 1997. |
| **[CH03]** | J. J. Cannon and D. F. Holt. Automorphism group computation and isomorphism testing in finite groups. *J. Symbolic Comp.*, 35(3):241–267, 2003. |
| **[CH04]** | J. J. Cannon and D. F. Holt. Computing maximal subgroups of finite groups. *J. Symbolic Comp.*, 37(5):589–609, 2004. |
| **[CHSS03]** | J. J. Cannon, D. F. Holt, M. Slattery, and A. K. Steel. Computing subgroups of low index in a finite group. 2003. |
| **[CLGM+95]** | Frank Celler, Charles R. Leedham-Green, Scott H. Murray, Alice C. Niemeyer, and E. A. O'Brien. Generating random elements of a finite group. *Comm. Algebra*, 23(13):4931–4948, 1995. |
| **[CS]** | J. J. Cannon and B. Souvignier. On the computation of normal subgroups in permutation groups. To appear, *International Journal of Algebra and Computation*. |
| **[CS97]** | J. J. Cannon and B. Souvignier. On the computation of conjugacy classes in permutation groups. In *Proceedings of the 1997 International Symposium on Symbolic and Algebraic Computation*, pages 392–399. ACM, 1997. |
| **[Geb00]** | Volker Gebhardt. Constructing a short defining set of relations for a finite group. *J. Algebra*, 233:526–542, 2000. |
| **[Hol84]** | D. F. Holt. The calculation of the Schur multiplier of a permutation group. In *Computational group theory (Durham, 1982)*, pages 307–319. Academic Press, London, 1984. |
| **[Hol85a]** | D. F. Holt. A computer program for the calculation of a covering group of a finite group. *J. Pure Appl. Algebra*, 35(3):287–295, 1985. |
| **[Hol85b]** | D. F. Holt. The mechanical computation of first and second cohomology groups. *J. Symbolic Comp.*, 1(4):351–361, 1985. |
| **[Kan91]** | William M. Kantor. Finding composition factors of permutation groups of degree n ≤ 10⁶. *J. Symbolic Comp.*, 12(4/5):517–526, 1991. |
| **[Leo80]** | Jeffrey S. Leon. On an algorithm for finding a base and a strong generating set for a group given by generating permutations. *Math. Comp.*, 35(151):941–974, 1980. |
| **[Leo97]** | Jeffrey S. Leon. Partitions, refinements, and permutation group computation. In Larry Finkelstein and William M. Kantor, editors, *Groups and Computation II*, volume 28 of DIMACS series in Discrete Mathematics and Computer Science, pages 123–158. Amer. Math. Soc., 1997. |
| **[LGPS91]** | C. R. Leedham-Green, C. E. Praeger, and L. H. Soicher. Computing with group homomorphisms. *J. Symbolic Comp.*, 12(4/5):527–532, 1991. |
| **[LNPS06]** | M. Law, A. C. Niemeyer, C. E. Praeger, and A. Seress. A reduction algorithm for large-base primitive permutation groups. *LMS J. Comput. Math.*, 9:159–173, 2006. |
| **[Luk93]** | E. M. Luks. Permutation groups and polynomial-time computation. In *Groups and computation (New Brunswick, NJ, 1991)*, volume 11 of DIMACS Ser. Discrete Math. Theoret. Comput. Sci., pages 139–175. Amer. Math. Soc., 1993. |
| **[McK81]** | B. D. McKay. Practical Graph Isomorphism. *Congressus Numerantium*, 30:45–87, 1981. |
| **[MN89]** | M. Mecky and J. Neubüser. Some remarks on the computation of conjugacy classes of soluble groups. *Bull. Austral. Math. Soc.*, 40(2):281–292, 1989. |
| **[Neu86]** | P. M. Neumann. Some algorithms for computing with finite permutation groups. In C. M. Campbell and E. F. Robertson, editors, *Groups — St. Andrews 1985*, number 121 in London Math. Soc. Lecture Notes Series, 1986. |
| **[Ric73]** | J. S. Richardson. Group: a computer system for group-theoretic calculations. Master's thesis, Department of Pure Mathematics, University of Sydney, September 1973. |
| **[Sch90]** | Bernd Schmalz. Verwendung von Untergruppenleitern zur Bestimmung von Doppelnebenklassen. *Bayreuther Mathematische Schriften*, 31:109–143, 1990. |
| **[Ser03]** | Ákos Seress. *Permutation Group Algorithms*, volume 152 of Cambridge Tracts in Mathematics. Cambridge University Press, Cambridge, 2003. |
| **[Sim90]** | Charles C. Sims. Computing the order of a solvable permutation group. *J. Symb. Comp.*, 9(5/6):699–705, 1990. |
| **[SS94]** | M. Schönert and A. Seress. Finding blocks of imprimitivity in small-base groups in nearly linear time. In *Proc. 1994 ACM-SIGSAM Inter. Symp. on Symbolic and Algebraic Comp.*, pages 154–157, 1994. |
| **[Ung]** | W. R. Unger. Computing chief series of a large permutation group. In preparation. |
| **[Ung06a]** | W. R. Unger. Computing the character table of a finite group. *J. Symbolic Comp.*, 41(8):847–862, 2006. |
| **[Ung06b]** | W. R. Unger. Computing the solvable radical of a permutation group. *J. Algebra*, 300(1):305–315, 2006. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Base and strong generating set (BSGS) / Schreier-Sims **[Leo80]** | `Order`, `#G`, `FactoredOrder`, `BSGS`, `SimsSchreier` |
| Random Schreier-Sims | `RandomSchreier` |
| Todd-Coxeter-Schreier (TCSS) | `ToddCoxeterSchreier`, `FPGroup`, `FPGroupStrong` |
| Solvable Schreier-Sims (Sims **[Sim90]**) | `SolubleSchreier`, `IsSoluble` |
| Brownie-Cannon-Sims verification | `Verify` |
| BSGS data access and word decomposition | `Base`, `BasicOrbit(s)`, `SchreierVector(s)`, `StrongGenerators`, `Strip`, `WordStrip`, `BaseImageWordStrip`, `WordInStrongGenerators`, `SVPermutation`, `SVWord` |
| Backtrack search (Leon **[Leo97]**) | `IsConjugate` (elements), `IsConjugate` (subgroups), `Centralizer`, `Normalizer`, `Core`, `H meet K`, `TransversalProcess` |
| Subgroup computation (Cannon-Cox-Holt **[CCH01]**) | `SubgroupClasses`, `Subgroups`, `NormalSubgroups`, `ElementaryAbelianSubgroups`, `CyclicSubgroups`, `AbelianSubgroups`, `NilpotentSubgroups`, `SolvableSubgroups`, `PerfectSubgroups`, `NonsolvableSubgroups`, `SimpleSubgroups`, `SubgroupsLift` |
| Low-index subgroups (Cannon-Holt-Slattery-Steel **[CHSS03]**) | `LowIndexSubgroups` |
| Maximal subgroups **[CH04]** | `MaximalSubgroups`, `IsMaximal` |
| Sylow subgroups (Cannon-Cox-Holt **[CCH97]**) | `SylowSubgroup` |
| Conjugacy classes: random algorithm **[CB92]** | `ConjugacyClasses`, `Classes`, `ClassMap` |
| Conjugacy classes: Butler inductive **[But94]** | `ConjugacyClasses` (Al := "Inductive") |
| Conjugacy classes: Cannon-Souvignier/Holt fusion **[CS97]** | `ConjugacyClasses` (Al := "Extend", TFAl default) |
| Conjugacy classes: Mecky-Neubüser lifting **[MN89]** | `ConjugacyClasses` (Al := "Extend") |
| Composition factors: Kantor tabular **[Kan91]** | `CompositionFactors` |
| Chief series / socle (Cannon-Holt **[CH97]**, Unger **[Ung]**) | `ChiefFactors`, `ChiefSeries`, `Socle`, `SocleFactor(s)`, `SocleSeries` |
| EARNS / affine type (Neumann **[Neu86]**) | `EARNS`, `IsAffine`, `AffineAction`, `AffineImage`, `AffineKernel` |
| Solvable radical (Unger **[Ung06b]**) | `Radical`, `SolubleRadical`, `pCore` |
| Normal subgroups / socle quotient (Cannon-Souvignier **[CS]**) | `NormalSubgroups`, `NormalLattice`, `SocleQuotient` |
| Normal closure, derived/lower central series (Butler-Cannon **[BC82]**) | `ncl<>`, `DerivedSeries`, `LowerCentralSeries`, `CommutatorSubgroup` |
| Upper central series / section centralizer (Luks **[Luk93]**) | `UpperCentralSeries`, `SectionCentraliser` |
| CentralizerOfNormalSubgroup (polynomial time, Beals **[Bea93]**) | `CentralizerOfNormalSubgroup`, `Centre` |
| CFL intersection with normal subgroup **[CFL89]** | `IntersectionWithNormalSubgroup` |
| Automorphism group / isomorphism (Cannon-Holt **[CH03]**) | `AutomorphismGroup`, `IsIsomorphic` |
| Cohomology (Holt **[Hol84, Hol85a, Hol85b]**) | `pMultiplicator`, `pCover`, `CohomologicalDimension`, `ExtensionProcess`, `Extension`, `SplitExtension` |
| Character table: Dixon-Schneider / Unger **[Ung06a]** | `CharacterTable` |
| Presentations (Leon **[Leo80]**, Gebhardt **[Geb00]**) | `FPGroupStrong` |
| Image/kernel of homomorphisms (Leedham-Green-Praeger-Soicher **[LGPS91]**) | `Image(f)`, `Kernel(f)`, `IsHomomorphism` |
| Product-replacement random elements (Celler et al. **[CLGM+95]**) | `RandomProcess`, `Random(P)` |
| Orbit actions on unions of orbits (Butler **[But85]**) | `OrbitAction`, `OrbitImage`, `OrbitKernel` |
| Minimal block systems (Schönert-Seress **[SS94]**, Atkinson **[Atk75]**) | `MinimalPartition`, `MinimalPartitions` |
| Coset action: Richardson canonical **[Ric73]** | `CosetAction` (Al := "Canonical") |
| Double coset ladders (Schmalz **[Sch90]**) | `ProcessLadder`, `GetRep`, `YoungSubgroupLadder`, `StabilizerLadder` |
| Jellyfish reduction for large-base primitives **[LNPS06]** | `JellyfishConstruction`, `JellyfishImage`, `JellyfishPreimage` |
| Alternating/symmetric recognition (Beals et al. **[BLGN+03]**) | `RecogniseAlternatingOrSymmetric`, `IsAlternating`, `IsSymmetric`, `IsAltsym` |
| Constructive Sn/An recognition (Bratus-Pak **[BP00]**) | `RecogniseSymmetric`, `RecogniseAlternating`, `SymmetricElementToWord`, `AlternatingElementToWord`, `GuessAltsymDegree` |
| 2-transitive identification (Cameron-Cannon **[CC91]**) | `TwoTransitiveGroupIdentification` |
| Ordered partition stacks (Leon **[Leo97]**, McKay **[McK81]**) | `OrderedPartitionStack`, `SplitCell`, `SplitAllByValues`, `SplitCellsByValues`, `Pop`, `Advance` |
