# Chapter 57 — Groups

**Handbook part:** IX — Finite Groups
**Handbook pages:** 1463–1513 (PDF pages 1592–1647)

---

## Scope and overview

Chapter 57 is the introductory overview chapter for Part IX (Finite Groups). It presents the
functions that are provided for groups *collectively* — across all categories — noting which
functions apply only to finite groups and which apply generally. Category-specific functions are
described in the chapters devoted to each individual category.

Groups arise in several different categories in Magma. The category of permutation groups
(GrpPerm) and the category of soluble groups defined by a power-conjugate presentation
(GrpPC) contain only finite groups. The finitely-presented group category (GrpFP), the
polycyclic group category (GrpGPC), the abelian group category (GrpAb) and the matrix group
category (GrpMat) contain both finite and infinite groups. For GrpAb and GrpMat, many
functions are available only for finite groups.

The chapter uses **GrpFin** to refer collectively to GrpPerm, GrpPC, and the finite subcategories
of GrpMat, GrpAb, and GrpGPC. The name **Grp** is used when an operation does not depend
on finiteness.

---

## 57.1 Introduction

### 57.1.1 The Categories of Finite Groups

Magma contains five main categories of finite groups:

| Category | Description |
|----------|-------------|
| `GrpPerm` | Permutation groups (finite only) |
| `GrpMat` | Finite matrix groups (also contains infinite groups) |
| `GrpPC` | Finite solvable groups given by a power-conjugate presentation (finite only) |
| `GrpAb` | Finite abelian groups (also contains infinite groups) |
| `GrpGPC` | Finite polycyclic groups (also contains infinite groups) |

---

## 57.2 Construction of Elements

### 57.2.1 Construction of an Element

Throughout this subsection, the carrier set of the group G is assumed to be a subset of some
ambient set S (e.g. Sym(X) for a permutation group on X).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `elt< G \| L >` | Given group G and a list L of objects defining an element of the ambient set S, construct the element g of S, test membership in G, and return g with parent G. Fails if g ∉ G. | — |
| `G ! Q` | Given group G and sequence Q = [a1, …, an] defining an element of S, construct g, test membership, and return g with parent G. Fails if g ∉ G. | — |
| `Identity(G)` / `Id(G)` | Construct the identity element of G. | — |

### 57.2.2 Coercion

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `G ! g` | Given groups G and H that are subgroups of a common overgroup, and element g ∈ H with g ∈ G, embed g in G (changes the parent to G). May fail for groups in GrpFP. | — |

### 57.2.3 Homomorphisms

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `hom< G -> H \| L >` | Return the group homomorphism φ: G → H defined by extending the generator map given by list L. L may be: (a) a list of n 2-tuples `<gi, hi>`; (b) a list of n arrow-pairs `gi -> hi`; or (c) an ordered list h1, …, hn. Magma assumes (but does not verify) the map is a homomorphism. | — |
| `hom< G -> H \| x :-> e(x) >` | Return the group homomorphism φ: G → H defined by the rule φ(x) = e(x), where x is a general element of G and e(x) is an expression in x. Magma assumes (but does not verify) the expression defines a homomorphism. | — |
| `IdentityHomomorphism(G)` | Return the identity homomorphism φ: G → G, x ↦ x. | — |

*Worked examples: H57E1 (isomorphism from cyclic group of order 15 to Z/15Z), H57E2 (endomorphism of a cyclic group via expression rule).*

### 57.2.4 Arithmetic with Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `g * h` | Product of elements g and h in the same generic group U. If g, h ∈ G ≤ U, the result has parent G. In abelian groups, written as `g + h`. | — |
| `g ^ n` | n-th power of element g (n a positive, negative, or zero integer). In abelian groups, written `n * g`. | — |
| `g / h` | Product g·h⁻¹; g and h must belong to the same generic group. In abelian groups, written `g - h`. | — |
| `g ^ h` | Conjugate h⁻¹gh of g by h; g and h must belong to the same generic group. Not defined for abelian groups. | — |
| `(g, h)` | Commutator g⁻¹h⁻¹gh of elements g and h in the same generic group. | — |
| `(g1, ..., gr)` | Left-normed commutator of r elements in a common group (evaluated left to right). | — |
| `g eq h` | True if elements g and h in the same generic group are equal. | — |
| `g ne h` | True if elements g and h in the same generic group are distinct. | — |
| `IsId(g)` / `IsIdentity(g)` | Returns true if the group element g is the identity element. | — |
| `Order(g)` | The order of the group element g. | — |

*Worked example: H57E3 (arithmetic operations on elements of Sym(9)).*

---

## 57.3 Construction of a General Group

### 57.3.1 The General Group Constructors

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PermutationGroup< X \| L >` | Construct a permutation group G acting on the set X, generated by the elements/subgroups given by list L. Returns G and the inclusion homomorphism G → Sym(X). Shorthand for `sub< Sym(X) \| L >`. | — |
| `PermutationGroup< n \| L >` | Construct a permutation group G acting on {1, …, n}, generated by list L. Returns G and the inclusion homomorphism G → Sym(n). | — |
| `MatrixGroup< n, R \| L >` | Construct a matrix group G of degree n over ring R, generated by list L. Returns G and the inclusion homomorphism G → GL(n, R). Shorthand for `sub< GL(n,R) \| L >`. | — |
| `Group< X \| R >` | Construct the finitely presented group (GrpFP) on generators X with relations R. Returns G and the natural homomorphism from the free group to G. | — |
| `PolycyclicGroup< X \| R >` | Construct a finite soluble group (GrpPC) or general polycyclic group (GrpGPC) from a power-conjugate or consistent polycyclic presentation R. Returns G and the natural homomorphism from the free group. GrpPC is returned if R is a valid power-conjugate presentation (unless `Class := "GrpGPC"`); otherwise GrpGPC. | — |
| `AbelianGroup< X \| R >` | Construct an abelian group (GrpAb) on generators X with relations R. Returns G and the natural homomorphism. | — |

*Worked examples: H57E4 (permutation group of degree 8; matrix group over GF(9); finitely presented group Q; soluble GrpPC group; finite abelian group on 4 generators), H57E5 (PolycyclicGroup with Class parameter; infinite dihedral group).*

### 57.3.2 Construction of Subgroups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `sub< G \| L >` | Construct the subgroup H of G generated by the elements/subgroups given by list L. Repetitions and occurrences of the identity are removed (unless H is trivial). | — |
| `ncl< G \| L >` | Construct the subgroup of G that is the normal closure of the subgroup generated by the elements specified by list L. | — |

*Worked example: H57E6 (subgroup of the finitely presented group Q generated by ts² and u⁴).*

### 57.3.3 Construction of Quotient Groups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `quo< G \| L >` | Construct the quotient group Q = G/N, where N is the normal closure of the subgroup generated by the elements in list L. Returns the quotient Q and the natural epimorphism f: G → Q. For GrpPerm and GrpMat, restricted to index of N less than 2³⁰ (constructed via regular representation). | — |
| `G / N` | Given a (normal) subgroup N of G, construct the quotient G/N. For GrpFP, N is not checked to be normal (result is G modulo the normal closure of N). For GrpPerm and GrpMat, restricted to index at most one million; result degree may be reduced. | — |

*Worked example: H57E7 (quotient of an abelian group with use of the natural homomorphism).*

---

## 57.4 Standard Groups and Extensions

### 57.4.1 Construction of a Standard Group

A number of functions construct standard groups on standard generators. The result category
may be specified as an argument.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AbelianGroup(C, Q)` / `AbelianGroup(Q)` | Construct the abelian group Z_n1 × Z_n2 × … × Z_nr for sequence Q = [n1, …, nr] of positive integers (ni = 0 denotes the infinite cyclic group in some categories). Without category C, constructed in GrpAb; with C, category may be GrpAb, GrpFP, GrpGPC, GrpPC, or GrpPerm. | — |
| `AlternatingGroup(C, n)` / `AlternatingGroup(n)` / `Alt(C, n)` / `Alt(n)` | Construct the alternating group on n letters. Default category GrpPerm; with C, may be GrpFP or GrpPerm. | — |
| `CyclicGroup(C, n)` / `CyclicGroup(n)` | Construct the cyclic group of order n. Default category GrpPerm; with C, may be GrpAb, GrpFP, GrpGPC, GrpPC, or GrpPerm. | — |
| `DihedralGroup(C, n)` / `DihedralGroup(n)` | Construct the dihedral group of order 2n. Default category GrpPerm; with C, may be GrpFP, GrpGPC, GrpPC, or GrpPerm. | — |
| `DicyclicGroup(n)` | Construct the dicyclic group of order 4n. | — |
| `DicyclicGroup(A, a)` | Given an abelian group A and an element a of order 2, construct the associated dicyclic group generated by A and an element x with x² = a and aˣ = a⁻¹ for all a ∈ A. | — |
| `SymmetricGroup(C, n)` / `SymmetricGroup(n)` / `Sym(GrpFin, n)` / `Sym(n)` | Construct the symmetric group on n letters. Default category GrpPerm; with C, may be GrpFP or GrpPerm. | — |
| `ExtraSpecialGroup(C, p, n : parameters)` / `ExtraSpecialGroup(p, n : parameters)` | Given prime p and positive integer n, construct an extra-special group of order p^(2n+1). Parameter `Type` (default `"+"`) selects isomorphism type: `"+"` gives the central product of n copies of the dihedral group of order 8 (p=2) or the unique extra-special group of exponent p (p>2); `"-"` gives the central product involving a quaternion group (p=2) or exponent p² (p>2). Default category GrpPerm; with C, may be GrpFP, GrpGPC, GrpPC, or GrpPerm. | — |

*Worked example: H57E8 (abelian group Z6×Z2×Z7; alternating group A6; dihedral group of order 8 as GrpPC; symmetric group S7 as GrpFP).*

### 57.4.2 Construction of Extensions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DirectProduct(G, H)` | Given two groups G and H in the same category C, construct the direct product G × H in category C. | — |
| `DirectProduct(Q)` | Given a sequence Q of n groups in the same category C, construct Q[1] × Q[2] × … × Q[n] in category C. | — |
| `SemidirectProduct(K, H, f: parameters)` | Given groups K and H and a homomorphism f: H → Aut(K), construct the semidirect product. Returns the semidirect product and maps embedding H and K into it. Parameters: `MaxDeg` (RngIntElt, default 1000000 — maximum degree permutation representation); `UseRegular` (BoolElt, default false — force use of regular representations). | — |

*Worked example: H57E9 (direct product of S4 and DihedralGroup(3)).*

---

## 57.5 Transfer Functions Between Group Categories

Since certain group computations are possible or feasible only for particular group
representations, Magma provides functions that transfer a group from one category to
another (or to some related group in another category).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `pQuotient(F, p, c: parameters)` | Given GrpFP group F, prime p and positive integer c (or 0 for class ≤ 127), construct the largest p-quotient of F with lower exponent-p class at most c as a GrpPC group. Also returns the homomorphism F → G. Parameters: `Exponent` (RngIntElt, default 0 — enforce exponent law xᵐ = 1); `Metabelian` (BoolElt, default false — compute largest metabelian p-quotient); `Print` (RngIntElt, default 0 — verbosity 0–3); `Workspace` (RngIntElt, default 5000000 — memory for computation). | p-quotient algorithm |
| `CosetAction(G, H)` | Given subgroup H of G, construct the permutation representation of G on the right cosets of H. Returns: (a) natural homomorphism f: G → L; (b) induced permutation group L; (c) kernel K (may be undefined for GrpFP). Uses the Todd-Coxeter procedure; G may be infinite as long as [G:H] is finite. | Todd-Coxeter procedure |
| `CosetImage(G, H)` | Returns the image L of G in its action on the right cosets of H (the second return value of `CosetAction(G, H)`). | Todd-Coxeter procedure |
| `CosetKernel(G, H)` | Returns the kernel of G in its action on the right cosets of H (the third return value of `CosetAction(G, H)`). May fail for GrpFP; only available when [G:H] is very small. | Todd-Coxeter procedure |
| `GPCGroup(G)` | Given a soluble group G in GrpPerm, GrpMat, GrpAb, or GrpPC (finite if GrpMat), construct an isomorphic polycyclic group P (GrpGPC). Returns P and an isomorphism φ: G → P. | — |
| `PCGroup(G)` | Given a finite soluble group G in GrpPerm, GrpMat, GrpAb, or GrpGPC, construct an isomorphic group S given by a power-conjugate presentation (GrpPC). Returns S and an isomorphism φ: G → S. | — |
| `FPGroup(G: parameters)` | Given G in GrpPerm, GrpMat, GrpGPC, or GrpPC, construct an isomorphic finitely presented group F on the given generators (or strong generators if `StrongGenerators := true`). Returns F and isomorphism φ: F → G. For GrpPerm and GrpMat, uses the Todd-Coxeter Schreier algorithm. Parameters: `StrongGenerators` (BoolElt, default false); `Random` (BoolElt, default true); `Max` (RngIntElt, default 100); `Run` (RngIntElt, default 20). | Todd-Coxeter Schreier algorithm |

*Worked examples: H57E10 (coset action of GrpFP group of order 168), H57E11 (permutation representation of Sp(2,4)), H57E12 (FP group isomorphic to PSU(3,3)).*

---

## 57.6 Basic Operations

### 57.6.1 Accessing Group Information

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `G . i` | The i-th defining generator of G (if i > 0); the inverse of the (−i)-th generator if i < 0; identity if i = 0. | — |
| `Generators(G)` | A set containing the defining generators of G. | — |
| `NumberOfGenerators(G)` / `Ngens(G)` | The number of defining generators of G. | — |
| `Generic(G)` | For G in GrpPerm or GrpMat, return the largest group in which G is naturally embedded (Sym(X) or GL(n,R)). | — |
| `Parent(g)` | The parent group G for the group element g. | — |
| `Orbit(G, M, x)` | For a finitely generated group G acting on the parent structure of x via map M, compute the orbit of x under G. Will exhaust memory if the orbit is infinite. | — |
| `OrbitClosure(G, M, S)` | For G acting on the universe of S via map M, compute the smallest G-invariant subset containing S. Will exhaust memory if the orbit closure is infinite. | — |

*Worked example: H57E13 (generators and generic group of the Suzuki group Sz(8) as a matrix group over GF(8)).*

---

## 57.7 Operations on the Set of Elements

### 57.7.1 Order and Index Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Order(G)` / `#G` | The order of G as an integer. Computed if not already known; may fail for GrpFP. | — |
| `FactoredOrder(G)` | The order of finite G as a factored integer — a sequence of pairs `<p, e>`. May fail for GrpFP. | — |
| `Index(G, H)` | The index [G:H] as an integer. May fail for GrpFP. | — |
| `FactoredIndex(G, H)` | The index [G:H] as a factored integer (same format as FactoredOrder). H must have finite index. May fail for GrpFP. | — |

*Worked example: H57E14 (order and index of a GrpFP group and subgroup).*

### 57.7.2 Membership and Equality

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `g in G` | True if element g belongs to G. | — |
| `g notin G` | True if element g does not belong to G. | — |
| `S subset G` | True if the set S of group elements (from some subgroup H of the same generic group as G) is a subset of G. | — |
| `S notsubset G` | True if S is not a subset of G. | — |
| `H subset G` | True if group H is a subgroup of G (both in the same generic group). | — |
| `H notsubset G` | True if H is not a subgroup of G. | — |
| `H eq G` | True if groups G and H (in the same generic group) are equal. | — |
| `H ne G` | True if groups G and H are distinct. | — |

### 57.7.3 Set Operations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `NumberingMap(G)` | For finite G in GrpPerm, GrpMat, GrpPC, or GrpAb, return a bijective map G → {1, …, |G|}. The precise mapping depends on the representation. | — |
| `Representative(G)` / `Rep(G)` | An element chosen from G. | — |

*Worked example: H57E15 (multiplication table for the dihedral group of order 12 via NumberingMap).*

### 57.7.4 Random Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Random(G: parameters)` | A randomly chosen element of G. If the carrier set representation is known, the element is genuinely random; otherwise selected as a random word in the generators (biased). Parameter `Short` (BoolElt, default false) causes a short word to be used. | — |
| `RandomProcess(G)` | Create a product-replacement process for generating random elements of G. Parameters: `Slots` (RngIntElt, default 10 — size of generating set in process); `Scramble` (RngIntElt, default 50 — initial scramble steps); `WordGroup` (GrpSLP). Uses a variant of the Rattle product-replacement method **[LGM02]**. Not recommended for GrpFP. | Product-replacement method **[LGM02]** |
| `RandomProcessWithWords(G)` | As RandomProcess, but each call to `Random(P)` also returns a GrpSLPElt describing the random element as a straight-line program. | Product-replacement method **[LGM02]** |
| `RandomProcessWithValues(G, Q)` | As RandomProcess, but each call to `Random(P)` also returns the result of parallel computation with the values Q (images of generators under a homomorphism). | Product-replacement method **[LGM02]** |
| `RandomProcessWithWordsAndValues(G, Q)` | Combines both: each call to `Random(P)` returns three values — random element, straight-line program, and value. | Product-replacement method **[LGM02]** |
| `Random(P)` | Given a process P created by `RandomProcess(G)`, produce a random element of G (plus SLP / value if applicable). For large groups without a known BSGS, preferred over `Random(G)`. | Product-replacement method **[LGM02]** |
| `InitialiseProspector(G: parameters)` | Initialise a product-replacement prospector for G that searches for an element x satisfying some predicate, aiming for a short straight-line program. Uses statistical tests (cycle structure for GrpPerm, characteristic polynomial factors for GrpMat) to assess randomness. | Extended product-replacement with statistical tests |
| `Prospector(G, f: parameters)` | Run an initialised prospector to find x ∈ G such that f(x) is true. Returns success flag; if true, also returns x and its straight-line program. Parameter `MaxTries` limits the number of random selections. | Extended product-replacement |

*Worked examples: H57E16 (Random elements of the wreath product Sym(4) ≀ C6), H57E17 (RandomProcessWithWords and Prospector to find a 300-cycle in Sym(300)).*

### 57.7.5 Action on a Coset Space

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CosetTable(G, H)` | The right coset table for G over subgroup H, relative to the defining generators. | Todd-Coxeter |
| `#CosetTable(G, f)` | The coset table for G corresponding to the permutation representation f (a homomorphism from G to a transitive permutation group). | — |
| `Transversal(G, H)` / `RightTransversal(G, H)` | Returns: (a) an indexed set T forming a right transversal for G over H; (b) the transversal mapping φ: G → T where φ(g) = t_i if g ∈ H·t_i. | — |
| `CosetAction(G, H)` | Permutation representation of G on the right cosets of H. Returns: (a) natural homomorphism f: G → L; (b) induced group L; (c) kernel K. G may be any type; K may be undefined for GrpFP. | Todd-Coxeter |
| `CosetImage(G, H)` | Image L of G in its action on the right cosets of H. | Todd-Coxeter |
| `CosetKernel(G, H)` | Kernel of the action of G on the right cosets of H. | Todd-Coxeter |

---

## 57.8 Standard Subgroup Constructions

Some functions in this section may not exist or may have restrictions for some categories.
Details are in the chapters on individual categories.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `H ^ g` / `Conjugate(H, g)` | Construct the conjugate g⁻¹Hg of group H by element g. H and g must belong to the same generic group. | — |
| `H meet K` | Intersection of groups H and K (both subgroups of the same symmetric group). | — |
| `CommutatorSubgroup(G, H, K)` / `CommutatorSubgroup(H, K)` | Commutator subgroup [H, K] of subgroups H, K of G. If K ≤ H, the ambient G may be omitted. | — |
| `Centralizer(G, g)` / `Centraliser(G, g)` | Centralizer of element g in G. | — |
| `Centralizer(G, H)` / `Centraliser(G, H)` | Centralizer of subgroup H in G. | — |
| `Core(G, H)` | The maximal normal subgroup of G contained in H. | — |
| `H ^ G` / `NormalClosure(G, H)` | The normal closure of H in G. | — |
| `Normalizer(G, H)` / `Normaliser(G, H)` | The normalizer of H in G. | — |
| `pCore(G, p)` | The maximal normal p-subgroup of G. | — |
| `SylowSubgroup(G, p)` / `Sylow(G, p)` | A Sylow p-subgroup of G. | — |

### 57.8.1 Abstract Group Predicates

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsAbelian(G)` | True if G is abelian. | — |
| `IsCyclic(G)` | True if G is cyclic. | — |
| `IsElementaryAbelian(G)` | True if G is elementary abelian. | — |
| `IsCentral(G, H)` | True if subgroup H lies in the centre of G. | — |
| `IsConjugate(G, g, h)` | True if elements g, h ∈ G are conjugate in G; if so, also returns a conjugating element k. | — |
| `IsConjugate(G, H, K)` | True if subgroups H, K ≤ G are conjugate in G; if so, also returns a conjugating element z. | — |
| `IsExtraSpecial(G)` | For a p-group G, true if G is extra-special. | — |
| `IsMaximal(G, H)` | True if H is a maximal subgroup of G. Evaluated by constructing the permutation representation on cosets of H and testing primitivity. Avoid if [G:H] > 100000. | Permutation representation + primitivity test |
| `IsNilpotent(G)` | True if G is nilpotent. | — |
| `IsNormal(G, H)` | True if H is a normal subgroup of G. | — |
| `IsPerfect(G)` | True if G is perfect. | — |
| `IsSelfNormalizing(G, H)` / `IsSelfNormalising(G, H)` | True if H is self-normalizing in G. | — |
| `IsSimple(G)` | True if G is simple. | — |
| `IsSoluble(G)` / `IsSolvable(G)` | True if G is soluble. | — |
| `IsSpecial(G)` | For a p-group G, true if G is special. | — |
| `IsSubnormal(G, H)` | True if H is subnormal in G. | — |
| `IsTrivial(G)` | True if G is trivial. | — |

---

## 57.9 Characteristic Subgroups and Normal Structure

### 57.9.1 Characteristic Subgroups and Subgroup Series

Some functions may not exist or may have restrictions for some categories; see individual
category chapters.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Centre(G)` / `Center(G)` | The centre of G. | — |
| `Hypercentre(G)` / `Hypercenter(G)` | The hypercentre of G (stationary term of the upper central series). | — |
| `DerivedLength(G)` | The derived length of G. For non-soluble G, returns the number of terms in the series terminating at the soluble residual. | — |
| `DerivedSeries(G)` | The derived series of G, as a sequence of subgroups. | — |
| `DerivedSubgroup(G)` / `DerivedGroup(G)` | The derived subgroup [G, G] of G. | — |
| `FittingSubgroup(G)` | The Fitting subgroup of G. | — |
| `FrattiniSubgroup(G)` | For a p-group G, the Frattini subgroup. | — |
| `JenningsSeries(G)` | For a p-group G, the Jennings series, returned as a sequence of subgroups. | — |
| `LowerCentralSeries(G)` | The lower central series of G, as a sequence of subgroups. | — |
| `NilpotencyClass(G)` | The nilpotency class of G; returns −1 if G is not nilpotent. | — |
| `H ^ G` / `NormalClosure(G, H)` | The normal closure of H in G. | — |
| `NormalLattice(G)` | The normal subgroups of G arranged as a lattice. | — |
| `NormalSubgroups(G)` | The normal subgroups of G. | — |
| `pCentralSeries(G, p)` | For a soluble group G and prime p dividing |G|, the lower p-central series, as a sequence of subgroups. | — |
| `Radical(G)` | The maximal normal solvable subgroup of G. | — |
| `SolubleResidual(G)` / `SolvableResidual(G)` | The solvable residual of G. | — |
| `SubnormalSeries(G, H)` | For G and a subnormal subgroup H, a sequence G = G0 ≥ G1 ≥ … ≥ H where each is normal in the previous. Returns the empty sequence if H is not subnormal. | — |
| `UpperCentralSeries(G)` | The upper central series of G, as a sequence commencing with the trivial subgroup. Requires conjugacy classes of G; more restricted than `DerivedSeries`/`LowerCentralSeries`. | Requires conjugacy classes |

### 57.9.2 The Abstract Structure of a Group

Composition factors are encoded as triples `<f, d, q>` where f identifies the family (see Tables 1 and 2 in the chapter) and d, q are family parameters. Family numbers include: 1=A(d,q), 2=B(d,q), …, 11=2B(2,q), 17=Alternating(d), 18=sporadic, 19=Cyclic(q).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CompositionFactors(G)` | For finite G in GrpPerm, GrpMat, GrpPC, or GrpAb, return a sequence of triples `<f, d, q>` representing the composition factors of G according to some composition series. | — |
| `AbelianInvariants(G)` / `Invariants(G)` | For abelian G in GrpPerm, GrpMat, GrpPC, or GrpAb, return a sequence containing the types of each p-primary component of G. | — |
| `AbelianBasis(G)` | For abelian G in GrpPerm, GrpPC, or GrpAb, return sequences B and I where I contains the types of each p-primary component and B contains corresponding elements of G of those orders that generate G. | — |

---

## 57.10 Conjugacy Classes of Elements

The algorithms used depend on the category of G:
- **GrpPerm / GrpMat:** conjugacy determined by backtrack search over base-images.
- **GrpPC:** testing conjugacy by transforming elements into a standard representative via an orbit-stabilizer process working down a sequence of quotients.
- **GrpGPC:** only possible for nilpotent groups, using an algorithm by E. Lo **[Lo98]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Class(H, x)` / `Conjugates(H, x)` | Given group H and element x in a group K (H, K subgroups of a common symmetric group), return the set of conjugates of x under H. If H = K, returns the conjugacy class of x in H. | — |
| `ClassMap(G: parameters)` | Construct the conjugacy classes and the class map f for G. For any x ∈ G, f(x) is the index of x's conjugacy class. For GrpPerm: parameter `Orbits` (BoolElt) — if true, classes computed as orbits of elements; `WeakLimit` and `StrongLimit` control the random classes algorithm. | Backtrack search (GrpPerm/GrpMat); orbit-stabilizer (GrpPC) |
| `ConjugacyClasses(G: parameters)` / `Classes(G: parameters)` | Construct representatives of the conjugacy classes of G. Returns a sequence of triples `(order, class length, representative)`. Parameters for GrpPerm/GrpMat: `Reps` (supply known representatives); `Al := "Action"` (orbits of elements under conjugation, small groups only); `Al := "Random"` (random search, controlled by `WeakLimit` (default 200) and `StrongLimit` (default 500)); `Al := "Extend"` (extend from quotient G/R where R is the radical, currently GrpPerm only). | Backtrack search; random classes; extension algorithm |
| `ClassRepresentative(G, x)` | For G with known conjugacy classes and x ∈ G, return the designated representative for the conjugacy class of x. | — |
| `IsConjugate(G, g, h)` | True if g, h ∈ G are conjugate in G; if so, also returns a conjugating element k. | Backtrack search (GrpPerm/GrpMat); orbit-stabilizer (GrpPC); **[Lo98]** (nilpotent GrpGPC) |
| `IsConjugate(G, H, K)` | True if subgroups H, K ≤ G are conjugate in G; if so, also returns a conjugating element z. | As above |
| `Exponent(G)` | The exponent of G. | — |
| `NumberOfClasses(G)` / `Nclasses(G)` | The number of conjugacy classes of elements of G. | — |
| `PowerMap(G)` | Construct the power map f: C × Z → C for G (where C = {1, …, r} indexes the conjugacy classes), defined by f(i, j) = class index of xᵢʲ. | — |

*Worked example: H57E18 (conjugacy classes of the Mathieu group M11).*

---

## 57.11 Conjugacy Classes of Subgroups

Magma uses an algorithm that first constructs representatives of conjugacy classes of subgroups
of Q = G/R (where R is the maximal normal soluble subgroup of G), then successively extends
these to larger quotients until G itself is reached.

- If G is soluble: Q is trivial; subgroups are known.
- If G is non-soluble: Q is located in a database of groups with trivial Fitting subgroup.
  This database contains all such groups of order up to 216,000 and all perfect groups of
  order up to 1,000,000. If not found, maximal subgroups of Q are found using a method of
  Derek Holt (requires all simple factors of the socle of Q to be in a second database,
  currently containing all simple groups of order < 1.6 × 10⁷, as well as M24, HS, J3, McL,
  Sz(32), L6(2), and by special routines: An for n ≤ 999, L2(q), L3(q), L4(q), L5(q) for all q,
  S4(q), U3(q), U4(q) for all q, Ld(2) for d ≤ 14, L6(3), L7(3), U6(2), S8(2), S10(2),
  O±8(2), O±10(2), S6(3), O7(3), O⁻8(3), G2(4), G2(5), 3D4(2), 2F4(2)', Co2, Co3, He, Fi22).

### 57.11.1 Conjugacy Classes of Subgroups

Most features in this section are currently only available for groups in GrpPerm, GrpMat, or GrpPC.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SubgroupClasses(G: parameters)` / `Subgroups(G: parameters)` | Representatives of the conjugacy classes of subgroups of G. Returns a sequence of records with fields: `subgroup` (representative), `order`, `length`, and optionally `presentation`. Parameters: `Presentation` (BoolElt, default false); `OrderEqual` (RngIntElt — only subgroups of this order); `OrderDividing` (RngIntElt — only subgroups whose order divides n); `IsNormal` (BoolElt, default false); `IsRegular` (BoolElt, default false, GrpPerm only); `LayerSizes` (SeqEnum — control elementary abelian series splitting). | Extension algorithm from radical quotient |
| `ElementaryAbelianSubgroups(G: parameters)` | Representatives of conjugacy classes of elementary abelian subgroups. Same record format and parameters as `Subgroups`. | As above |
| `AbelianSubgroups(G: parameters)` | Representatives of conjugacy classes of abelian subgroups. Same record format and parameters. | As above |
| `CyclicSubgroups(G: parameters)` | Representatives of conjugacy classes of cyclic subgroups. Same record format and parameters. | As above |
| `NilpotentSubgroups(G: parameters)` | Representatives of conjugacy classes of nilpotent subgroups. Same record format and parameters. | As above |
| `SolubleSubgroups(G: parameters)` / `SolvableSubgroups(G: parameters)` | Representatives of conjugacy classes of solvable subgroups. Same record format and parameters. | As above |
| `NonsolvableSubgroups(G: parameters)` | Representatives of conjugacy classes of nonsolvable subgroups. Same record format and parameters. | As above |
| `PerfectSubgroups(G: parameters)` | Representatives of conjugacy classes of perfect subgroups. Same record format and parameters. | As above |
| `SimpleSubgroups(G: parameters)` | Representatives of conjugacy classes of non-abelian simple subgroups. Same record format and parameters. | As above |
| `RegularSubgroups(G: parameters)` | Representatives of conjugacy classes of regular subgroups of the permutation group G. Same record format and parameters. | As above |
| `SetVerbose("SubgroupLattice", i)` | Enable verbose printing for the subgroup algorithm at level i (1 = moderate, 2 = maximal). Reports extensions of each subgroup at each level of the elementary abelian series. | — |
| `Class(G, H)` / `Conjugates(G, H)` | The G-conjugacy class of subgroups containing H. | — |

*Worked example: H57E19 (conjugacy classes of subgroups of the dihedral group of order 12).*

### 57.11.2 The Poset of Subgroup Classes

Magma allows construction of the poset L of subgroup classes, where elements correspond to
conjugacy classes of subgroups and two classes are joined by an edge if some subgroup of one
class is maximal in some subgroup of the other. Currently only available for GrpPerm or GrpPC.

#### 57.11.2.1 Creating the Poset of Subgroup Classes

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SubgroupLattice(G)` | Create the poset L of subgroup classes of G. Parameters: `Properties` (BoolElt, default false — record abstract type of each class: elementary abelian, abelian, nilpotent, soluble, simple, perfect); `Centralizers` (BoolElt, default false — record centralizer class for each class); `Normalizers` (BoolElt, default false — record normalizer class for each class). | Extension algorithm |

*Worked examples: H57E20 (subgroup poset of ASL(2,3) with Properties, Normalizers, Centralizers), H57E21 (subgroup lattice of AΓL(1,8), locating the Fitting subgroup and constructing a chain bottom to top).*

#### 57.11.2.2 Operations on Subgroup Class Posets

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `#L` | The cardinality of L (number of conjugacy classes of subgroups of G). | — |
| `L ! i` | Create the i-th element of L (sorted by number of prime factors of order, then by order). | — |
| `L ! H` | Create the element of L corresponding to subgroup H of G. | — |
| `Bottom(L)` | The bottom of L (element corresponding to the trivial subgroup, or the smallest class if L was created with restrictions). | — |
| `Top(L)` | The top of L (element corresponding to G). | — |
| `Random(L)` | A random element of L. | — |

#### 57.11.2.3 Operations on Poset Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IntegerRing() ! e` | The integer index corresponding to poset element e. | — |
| `e eq f` | True if poset elements e and f are equal. | — |
| `e ge f` | True if poset element e contains f. | — |
| `e le f` / `e subset f` | True if poset element e is contained in f. | — |
| `e lt f` | True if e is strictly contained in f. | — |

#### 57.11.2.4 Class Information from a Conjugacy Class Poset

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Group(e)` | The subgroup of G that is the chosen class representative for poset element e. | — |
| `Centraliser(e, f)` / `Centralizer(e, f)` | The poset element corresponding to the class of centralizers of subgroups of class f taken in a subgroup of class e. Fails if no subgroup of class f lies in class e. | — |
| `Normaliser(e, f)` / `Normalizer(e, f)` | The poset element corresponding to the class of normalizers of subgroups of class f taken in a subgroup of class e. Fails if no subgroup of class f lies in class e. | — |
| `Length(e)` | The number of subgroups in the conjugacy class corresponding to e. | — |
| `Order(e)` | The order of the subgroup corresponding to e. | — |
| `MaximalSubgroups(e)` | The maximal subgroups of e, as a set of poset elements. | — |
| `MinimalOvergroups(e)` | The minimal overgroups of e, as a set of poset elements. | — |
| `NumberOfInclusions(e, f)` | The number of subgroups of conjugacy class e that lie in a fixed representative of conjugacy class f. | — |

---

## 57.12 Cohomology

Here G is a group in category GrpPerm, p is a prime, K = GF(p), F is a finitely presented group
with the same number of generators as G such that the mapping F.i ↦ G.i is an epimorphism.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `pMultiplicator(G, p)` | Return the invariant factors of the p-part of the Schur multiplicator of G. | Schur multiplicator |
| `pCover(G, F, p)` | Given G and the finitely presented group F (an epimorphic preimage of G), return a presentation for the p-cover of G constructed as an extension of the p-multiplier by F. | p-cover construction |
| `CohomologicalDimension(G, M, i)` | For G, the K[G]-module M, and integer i ∈ {1, 2}, return the dimension of the i-th cohomology group H^i(G, M). | Cohomology computation |
| `ExtensionProcess(G, M, F)` | Create an extension process for G by module M (to be used with `Extension` and `NextExtension`). The possible extensions of M by G correspond to elements of H²(G, M). | Cohomological extension |
| `Extension(P, Q)` | Given extension process P and a sequence Q = [a1, …, al] of integers representing an element of H²(G, M), return the corresponding extension of M by G as a finitely presented group. | — |
| `#NextExtension(P)` | Return the next extension of G defined by process P (iterate over all p^l extensions). | — |
| `SplitExtension(G, M, F)` | The split extension of module M by G. | — |

---

## 57.13 Characters and Representations

Full character theory is in Chapter 91; full representation theory is in Chapter 89. This section
provides basic functions for creating characters and representations.

### 57.13.1 Character Theory

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CharacterDegrees(G)` | For a finite pc-group G, return the sequence `[<d1, c1>, <d2, c2>, …]` where ci is the number of irreducible characters of degree di. | Algorithm of Conlon **[Con90]** |
| `CharacterTable(G)` | Construct the table of irreducible characters of G. | — |
| `PermutationCharacter(G)` | For G a permutation group, the character afforded by the defining permutation representation. | — |
| `PermutationCharacter(G, H)` | The ordinary character of G afforded by the permutation representation given by the action of G on the coset space of H in G. | — |

### 57.13.2 Representation Theory

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `GModule(G, S)` | Let G be defined on r generators and S a subalgebra of Mn(R) defined by r non-singular matrices. Assuming the map G.i ↦ S.i extends to a group homomorphism, give the natural module M for S the structure of an S[G]-module with the action of the i-th generator of G given by S.i. | — |
| `GModule(G, A, B)` | For finite G with normal subgroups B ≤ A where A/B is elementary abelian of order p^n, create the K[G]-module M (K = Fp) corresponding to the action of G on A/B. Returns M and the homomorphism φ: A/B → M. B may be omitted if trivial. | — |
| `PermutationModule(G, H, R)` | For finite G and ring R, create the R[G]-module corresponding to the permutation action of G on the cosets of H. | — |
| `PermutationModule(G, R)` | For finite permutation group G and ring R, create the natural permutation module for G over R. | — |

*Worked examples: H57E22 (permutation module for M10 over GF(2)), H57E23 (GModule for a split extension of elementary abelian group of order 16 by Alt(6)).*

---

## 57.14 Databases of Groups

Magma contains several built-in databases of groups. For full descriptions, see Chapter 66.

| Database | Contents |
|----------|----------|
| Small Groups | All groups of order up to 1000 (excluding orders 512 and 768). |
| Perfect Groups | All perfect groups up to order 50000; many classes up to order 1,000,000. Each defined by a finite presentation, with information for constructing permutation representations. |
| Rational Maximal Matrix Groups | Rational maximal finite matrix groups and their invariant forms, for dimensions up to 31. Accessible as matrix groups or lattices. |
| Quaternionic Matrix Groups | Finite absolutely irreducible subgroups of GL_n(D) for D a definite quaternion algebra with centre of degree d over Q and nd ≤ 10. Accessible as matrix groups or lattices. |
| Transitive Permutation Groups | All transitive permutation groups of degree up to 22. |
| Primitive Permutation Groups | All primitive permutation groups of degree up to 50. |

---

## 57.15 Bibliography

| Key | Reference |
|-----|-----------|
| **[Con90]** | S. B. Conlon. *Computing modular and projective character degrees of soluble groups.* J. Symbolic Comp., **9**:551–570, 1990. |
| **[LGM02]** | C. R. Leedham-Green and Scott H. Murray. *Variants of product replacement.* Contemp. Math., **298**:97–104, 2002. |
| **[Lo98]** | Eddie H. Lo. *Finding intersections and normalizers in finitely generated nilpotent groups.* J. Symbolic Comput., **25**(1):45–59, 1998. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Todd-Coxeter procedure (coset enumeration) | `CosetAction`, `CosetImage`, `CosetKernel`, `CosetTable`, `FPGroup` |
| Todd-Coxeter Schreier algorithm (FP presentation) | `FPGroup` |
| p-quotient algorithm | `pQuotient` |
| Product-replacement / Rattle method **[LGM02]** | `RandomProcess`, `RandomProcessWithWords`, `RandomProcessWithValues`, `RandomProcessWithWordsAndValues`, `Random(P)` |
| Extended product-replacement prospector | `InitialiseProspector`, `Prospector` |
| Backtrack search over base-images | `IsConjugate` (GrpPerm/GrpMat), `Classes` (GrpPerm/GrpMat) |
| Orbit-stabilizer (conjugacy in GrpPC) | `IsConjugate` (GrpPC), `Classes` (GrpPC) |
| Lo's algorithm (conjugacy in nilpotent GrpGPC) **[Lo98]** | `IsConjugate` (nilpotent GrpGPC) |
| Extension algorithm from radical quotient (subgroup classes) | `Subgroups`, `SubgroupClasses`, `ElementaryAbelianSubgroups`, `AbelianSubgroups`, `CyclicSubgroups`, `NilpotentSubgroups`, `SolubleSubgroups`, `PerfectSubgroups`, `SimpleSubgroups`, `RegularSubgroups`, `SubgroupLattice` |
| Conlon's algorithm (character degrees of pc-groups) **[Con90]** | `CharacterDegrees` |
| Schur multiplicator / cohomological extension | `pMultiplicator`, `pCover`, `CohomologicalDimension`, `ExtensionProcess`, `Extension`, `SplitExtension` |
| Primitivity test (maximality of subgroup) | `IsMaximal` |
