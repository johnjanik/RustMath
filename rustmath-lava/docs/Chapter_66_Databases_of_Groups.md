# Chapter 66 — Databases of Groups

**Handbook part:** IX — Finite Groups
**Handbook pages:** 1939–1990 (PDF pages 2068–2125)

---

## Scope and overview

Chapter 66 documents Magma's collection of group databases, covering fifteen distinct
libraries of finite groups. The databases span small groups (all groups of order up to 2000),
p-groups, metacyclic p-groups, perfect groups, almost-simple groups, transitive and primitive
permutation groups, various classes of maximal finite matrix groups (rational, integral,
quaternionic, symplectic, irreducible, quasisimple, and soluble irreducible), a database of
ATLAS representations, and a database of fundamental groups of 3-manifolds.

Each database is accessed through a common pattern: a database-open function returns a
handle, intrinsics retrieve individual entries by index or by structured keys, predicate-based
search functions return the first or all matching groups, and process-based iteration allows
memory-efficient traversal. Group identification functions provide the inverse mapping from
a concrete group to its database label.

Groups extracted from the Small Groups database are returned as `GrpPC` (polycyclic
presentation) when soluble and as `GrpPerm` when insoluble. Most databases store exactly
one representative per isomorphism class (or conjugacy class, for matrix groups), and the
chapter references the original enumerations that established these classifications.

---

## 66.1 Introduction

The chapter opens with a summary table of all available databases and their provenance.
The available databases are:

- **Small Groups** — all groups of order ≤ 2000 (except order 1024), groups whose order
  is a product of at most 3 primes, groups of order dividing p⁶ (p prime), groups of order
  qⁿp (qⁿ dividing 2⁸, 3⁶, 5⁵ or 7⁴, p ≠ q prime), and groups of square-free order.
  Constructed by Besche, Eick and O'Brien **[BE99a, BEO01, BE99b, O'B90, BE01, O'B91,
  MNVL04, OVL05, DE05]**.
- **p-groups** — all p-groups of order pⁿ, n ≤ 7. Data from Besche, Eick, O'Brien,
  Newman and Vaughan-Lee **[BE99a, BEO01, BE99b, O'B90, BE01, O'B91, MNVL04, OVL05]**.
- **Metacyclic p-groups** — all metacyclic groups of order pⁿ. Functions by Newman,
  O'Brien and Vaughan-Lee.
- **Perfect Groups** — all perfect groups up to order 50000, and many classes up to
  order one million. Constructed by Holt and Plesken **[HP89]**.
- **Almost Simple Groups** — groups G with S ≤ G ≤ Aut(S) for simple S of order less
  than 16000000, plus M₂₄, HS, J₃, McL, Sz(32) and L₆(2). Originally by Holt, extended
  by Gebhardt; implementation by Cox.
- **Transitive Permutation Groups** — all transitive groups of degree up to 32. Degree
  ≤ 15 by Butler–McKay; degree 16–30 by Hulpke **[Hul05]**; degree 32 by Cannon–Holt
  **[CH08]**.
- **Primitive Permutation Groups** — all primitive groups of degree < 4096. Degree ≤ 50
  by Sims **[Sim70]**; degree < 1000 by Roney-Dougal–Unger **[RDU03]**; degree < 2500
  by Roney-Dougal **[RD05]**; degree < 4096 by Coutts–Quick–Roney-Dougal **[CQRD11]**.
- **Rational Maximal Matrix Groups** — rational maximal finite matrix groups and their
  invariant forms, dimensions up to 31. By Nebe and Plesken **[NP95, Neb96]**.
- **Quaternionic Matrix Groups** — finite absolutely irreducible subgroups of GLₙ(D)
  for D a definite quaternion algebra, nd ≤ 10. By Nebe **[Neb98]**.
- **Irreducible Matrix Groups** — irreducible subgroups of GLₖ(p), pᵏ < 2500.
  Same provenance as affine primitive groups.
- **Soluble Irreducible Groups** — one representative per conjugacy class of irreducible
  soluble subgroups of GL(n, p), pⁿ < 256. By Short **[Sho92]**.
- **ATLAS Groups** — representations of nearly simple groups from the Birmingham ATLAS
  of Finite Group Representations. Data supplied by Rob Wilson.
- **Fundamental Groups of 3-Manifolds** — the 10,986 small-volume closed hyperbolic
  3-manifolds (Hodgson–Weeks census); presentations by Weeks (SnapPea); homology data
  by Dunfield–Thurston **[DT03]**.

---

## 66.2 Database of Small Groups

The Small Groups Library (Besche, Eick, O'Brien) uses the same internal data format and
group numbering as the GAP implementation. Soluble groups are returned as `GrpPC`;
insoluble groups as `GrpPerm`. Most functions accept an optional database handle `D`
(opened by `SmallGroupDatabase()`) as first argument to reduce file operations during
extended searches. The parameter `Search` (default `"All"`) can restrict results to
`"Soluble"` or `"Insoluble"` groups.

### 66.2.1 Basic Small Group Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SmallGroupDatabase()` / `OpenSmallGroupDatabase()` | Open the small groups database and return a handle `D`. Pass `D` as first argument to other functions to reduce file operations. | — |
| `delete D` | Close database handle `D` and free associated resources. | — |
| `SmallGroupDatabaseLimit()` / `SmallGroupDatabaseLimit(D)` | Return the limiting order up to which all groups (except order 1024) are stored; currently 2000. | — |
| `IsInSmallGroupDatabase(o)` / `IsInSmallGroupDatabase(D, o)` | Return `true` if groups of order `o` are in the database, `false` otherwise. Useful for guard checks in loops. | — |
| `NumberOfSmallGroups(o)` / `NumberOfSmallGroups(D, o)` | Return the number of groups of order `o` in the database (0 if not present). | — |
| `SmallGroup(o, n)` / `SmallGroup(D, o, n)` / `Group(D, o, n)` | Return the `n`-th group of order `o`. Error if `o` not in database or `n` out of range. | Database lookup **[BE99a, BEO01, BE99b, O'B90, BE01, O'B91, MNVL04]**. |
| `SmallGroup(o: Search)` / `SmallGroup(D, o: Search)` | Return the first group of order `o` meeting the search criterion (`"All"`, `"Soluble"`, or `"Insoluble"`). | — |
| `SmallGroup(o, f: Search)` / `SmallGroup(D, o, f: Search)` / `SmallGroup(S, f: Search)` / `SmallGroup(D, S, f: Search)` | Return the first group of order `o` (or order in sequence `S`) meeting the search criterion and satisfying predicate `f`. | Sequential scan with predicate filter. |
| `IsSoluble(D, o, n)` / `IsSolvable(D, o, n)` / `SmallGroupIsSoluble(o, n)` / `SmallGroupIsSoluble(D, o, n)` / `SmallGroupIsSolvable(o, n)` / `SmallGroupIsSolvable(D, o, n)` | Return `true` iff `SmallGroup(o, n)` is soluble, without loading the group. | Stored metadata lookup. |
| `SmallGroupIsInsoluble(o, n)` / `SmallGroupIsInsoluble(D, o, n)` / `SmallGroupIsInsolvable(o, n)` / `SmallGroupIsInsolvable(D, o, n)` | Return `true` iff `SmallGroup(o, n)` is insoluble, without loading the group. | Stored metadata lookup. |
| `SmallGroups(o: Search, Warning)` / `SmallGroups(D, o: Search, Warning)` / `SmallGroups(S: Search, Warning)` / `SmallGroups(D, S: Search, Warning)` | Return all groups of order `o` (or orders in sequence `S`) meeting the search criterion. Parameter `Warning` (default `true`) prints a warning for very large result sets. | Sequential scan. |
| `SmallGroups(o, f: Search)` / `SmallGroups(D, o, f: Search)` / `SmallGroups(S, f: Search)` / `SmallGroups(D, S, f: Search)` | Return all groups of order `o` or orders in `S` meeting the search criterion and satisfying predicate `f`. | Sequential scan with predicate filter. |

*Worked examples: H66E1 (non-abelian groups of order 27; first group with derived length > 2; insoluble groups of order 240; counting groups of order 2432; groups of order 7⁶ with cyclic centre of order 7²).*

### 66.2.2 Processes

A small group process allows iteration over groups of specified orders satisfying a given
predicate without storing all groups simultaneously. Each process opens the database for an
extended search automatically; process variants with a database handle `D` as first argument
are therefore not provided. The `Search` parameter (`"All"`, `"Soluble"`, `"Insoluble"`)
applies as for the retrieval functions.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SmallGroupProcess(o: Search)` | Return a process iterating over all groups of order `o` meeting the search criterion. | — |
| `SmallGroupProcess(S: Search)` | Return a process iterating over all groups with order in sequence `S` meeting the search criterion. | — |
| `SmallGroupProcess(o, f: Search)` | Return a process iterating over groups of order `o` meeting the search criterion and satisfying predicate `f`. | — |
| `SmallGroupProcess(S, f: Search)` | Return a process iterating over groups with order in `S` meeting the search criterion and satisfying `f`. | — |
| `IsEmpty(p)` | Return `true` if process `p` has passed its last group. | — |
| `Current(p)` | Return the current group of process `p`. | — |
| `CurrentLabel(p)` | Return `o` and `n` such that the current group is `SmallGroup(o, n)`. | — |
| `Advance(∼p)` | Advance process `p` to its next group. | — |

*Worked example: H66E2 (nilpotency class distribution over all 2328 groups of order 128).*

### 66.2.3 Small Group Identification

The identification functions invert the small group retrieval functions: given a concrete
group G isomorphic to some database entry, they return the pair ⟨o, n⟩. Identification of
finitely presented groups requires constructing a permutation representation, which may
fail.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IdentifyGroup(G)` | Return ⟨o, n⟩ such that `SmallGroup(o, n)` is isomorphic to `G`. Error if not in database or identification of groups of order `|G|` is not supported. | Hash/invariant-based identification; see also Chapter 70. |
| `CanIdentifyGroup(o)` | Return `true` if identification of groups of order `o` is supported. | — |

*Worked example: H66E3 (identifying `DihedralGroup(10)` as `SmallGroup(20, 4)`).*

### 66.2.4 Accessing Internal Data

These functions expose the internal encoding of the Small Groups Library; intended for
expert use only.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Data(D, o, n)` | Return the raw internal data from which group number `n` of order `o` is constructed. Format is order-dependent. | — |
| `SmallGroupEncoding(G)` | For a finite soluble group `G` in `GrpPC`: return two integers `c` and `o` encoding its power-conjugate presentation; `o` is the order. | Internal encoding of PC relations as integers. |
| `SmallGroupDecoding(c, o)` | Given integers `c` and `o` encoding a power-conjugate presentation of order `o`: return the corresponding `GrpPC`. | Inverse of `SmallGroupEncoding`. |

*Worked example: H66E4 (encoding of `SmallGroup(D, 1053, 51)`; decoding `Data(D, 525, 2)` to recover `SmallGroup(D, 525, 2)`).*

---

## 66.3 The p-groups of Order Dividing p⁷

Magma can construct all p-groups of order pⁿ for n ≤ 7 (or p = 2, n ≤ 9). The
underlying data was supplied by Besche, Eick, O'Brien, Newman and Vaughan-Lee
**[BE99a, BEO01, BE99b, O'B90, BE01, O'B91, MNVL04, OVL05]**. The groups of order
p⁷ were contributed by O'Brien and Vaughan-Lee. This section provides search and count
access to these constructions, including groups not in the Small Groups Library (order p⁷).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SearchPGroups(p, n: Rank, Class, Select, Limit)` | Return a sequence of groups of order pⁿ (n ≤ 7, or p = 2 and n ≤ 9) satisfying the parameter conditions. `Rank` (default {1,…,n}): restrict to specified Frattini quotient ranks. `Class` (default {1,…,n}): restrict to specified p-classes. `Select`: a predicate program `G ↦ true/false` applied after rank/class filtering. `Limit` (default 0): if positive, stop after finding at least that many groups. | Systematic construction from stored classification data **[BE99a, BEO01, O'B90, O'B91, MNVL04, OVL05]**. |
| `CountPGroups(p, n: Rank, Class, Select)` | Count the groups satisfying the parameter conditions. Same parameters as `SearchPGroups` except `Limit` is ignored. | As above. |

*Worked example: H66E5 (searching 19⁷ groups for rank 5, class 3, prime exponent; using `Limit`; counting).*

---

## 66.4 Metacyclic p-groups

Magma provides a complete enumeration and constructive recognition of metacyclic p-groups
of order pⁿ. Functions were developed by Newman, O'Brien, and Vaughan-Lee.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `MetacyclicPGroups(p, n: PCGroups)` | Return a list of all metacyclic groups of order pⁿ. Parameter `PCGroups` (default `true`): if `true` returns `GrpPC` groups; if `false` returns `GrpFP` groups (faster for large class). | Systematic construction of metacyclic p-groups by Newman–O'Brien–Vaughan-Lee. |
| `IsMetacyclicPGroup(P)` | For a p-group `P` (pc, matrix, or permutation): return `true` if `P` is metacyclic, `false` otherwise. | Structural test via invariants. |
| `InvariantsMetacyclicPGroup(P)` | For a metacyclic p-group `P`: return a tuple ⟨r, s, t, n⟩ (plus additional invariants for p = 2) that uniquely identifies `P` among metacyclic p-groups of the same order. Meaning: order pⁿ⁺ˢ; derived quotient Cₚᵣ × Cₚˢ; derived group cyclic of order pⁿ⁻ʳ; exponent pⁿ⁺ˢ⁻ᵗ. For p = 2 additional invariants encode abelian invariants of the centre and (for maximal-class cases) whether the group is dihedral, quaternion, or semidihedral. | Classification invariants. |
| `StandardMetacyclicPGroup(P)` | For a metacyclic p-group `P`: return an isomorphic group with a canonical pc-presentation. Two metacyclic p-groups are isomorphic iff their canonical presentations are identical. | Canonical form algorithm. |
| `NumberOfMetacyclicPGroups(p, n)` | Return the number of metacyclic groups of order pⁿ. | Count from classification. |
| `HasAllPQuotientsMetacyclic(G)` / `HasAllPQuotientsMetacyclic(G, p)` | For a finitely presented group `G`: return `true` if all p-quotients (for all primes p, or for the specified prime p) are metacyclic. If `false`, also return a description of the primes with non-metacyclic p-quotient. | p-quotient enumeration and metacyclicity test. |

*Worked example: H66E6 (listing metacyclic groups of order 3⁶; identifying `SmallGroup(729, 59)` as metacyclic; finding its canonical form in the list; checking a finitely presented group for metacyclic p-quotients).*

---

## 66.5 Database of Perfect Groups

Magma includes a database of finite perfect groups containing all perfect groups up to
order 50000 and many classes up to order one million. Each group is defined by a finite
presentation; additional data enables construction of permutation representations. The
database was constructed by Derek Holt and Willem Plesken **[HP89]**.

### 66.5.1 Specifying an Entry of the Database

Entries may be specified in three ways:
1. A single integer `i` (the i-th entry; no particular ordering).
2. A pair `o, i` (the i-th entry of order `o`).
3. The notation from Chapter 5.3 of **[HP89]**: a base group `Q` (given as a string from
   `TopQuotients(D)`), a prime `p`, and integers `r ≥ 0`, `n ≥ 0` specifying the class
   Q#p⟨r,n⟩. A fifth key `Variant` (default 1) is needed in three exceptional cases
   for compatibility with Holt–Plesken tables.

### 66.5.2 Creating the Database

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PerfectGroupDatabase()` | Return a database object `D` required as first argument to all other perfect-group functions. | — |

### 66.5.3 Accessing the Database

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Group(D, i)` / `Group(D, o, i)` / `Group(D, Q)` / `Group(D, Q, p, r, n: Variant)` | Return the specified entry as a finitely presented group, together with a sequence of pairs ⟨[i₁,…,iₙ], [H₁,…,Hₙ]⟩ each affording an isomorphism onto a permutation group of degree Σiⱼ. `Variant` (default 1) selects among stored variants for the three exceptional entries. | Finite presentation from **[HP89]** data. |
| `IdentificationNumber(D, i)` / `IdentificationNumber(D, o, i)` / `IdentificationNumber(D, Q)` / `IdentificationNumber(D, Q, p, r, n: Variant)` | Return the integer index (method 1) for the specified entry. | — |
| `NumberOfRepresentations(D, i)` / `NumberOfRepresentations(D, o, i)` / `NumberOfRepresentations(D, Q)` / `NumberOfRepresentations(D, Q, p, r, n: Variant)` | Return the number of stored methods for building a permutation representation of the specified entry. | — |
| `PermutationRepresentation(D, i: Variant, Representation)` / `PermutationRepresentation(D, o, i: Variant, Representation)` / `PermutationRepresentation(D, Q: Variant, Representation)` / `PermutationRepresentation(D, Q, p, r, n: Variant, Representation)` | Return an isomorphism from the finitely presented group to a permutation representation, and both groups. `Representation` (default 1) selects which stored construction method to use. | Coset enumeration on stored subgroups. |
| `PermutationGroup(D, i: Variant, Representation)` / `PermutationGroup(D, o, i: Variant, Representation)` / `PermutationGroup(D, Q: Variant, Representation)` / `PermutationGroup(D, Q, p, r, n: Variant, Representation)` | Return the specified entry directly as a permutation group. `Representation` (default 1) selects the construction method. | Coset enumeration. |

### 66.5.4 Finding Legal Keys

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `#D` / `NumberOfGroups(D)` | Total number of entries in the database. | — |
| `NumberOfGroups(D, o)` | Number of entries of order `o`. | — |
| `TopQuotients(D)` | The set of strings denoting all base perfect groups `Q` stored. | — |
| `ExtensionPrimes(D, Q)` | The set of primes `p` for which a non-trivial p-extension of the group named `Q` is in the database. | — |
| `ExtensionExponents(D, Q, p)` | The set of exponents `r` such that a non-trivial extension of `Q` by pʳ is in the database. | — |
| `ExtensionNumbers(D, Q, p, r)` | The set of integers `n` such that Q#p⟨r,n⟩ is in the database. | — |
| `ExtensionClasses(D, Q)` | The set of triples ⟨p, r, n⟩ such that Q#p⟨r,n⟩ is in the database. | — |

*Worked example: H66E7 (navigating keys to find extensions of L(3,4); constructing a group of order 80640 as FP and permutation group; verifying chief factors and radical).*

---

## 66.6 Database of Almost-Simple Groups

Magma includes a database of almost-simple groups G with S ≤ G ≤ Aut(S) for simple
groups S of order less than 16000000, plus M₂₄, HS, J₃, McL, Sz(32), and L₆(2).
Originally designed by Derek Holt, with a major extension by Volker Gebhardt and sporadic
additions by Bill Unger; implementation by Bruce Cox. The primary use is computing
maximal subgroups and automorphism groups.

### 66.6.1 The Record Fields

The `GroupData` function returns a record with the following fields:

- `resname`: String name of the socle S (its soluble residual).
- `resorder`: Order of S.
- `geninfo`: Sequence of two tuples, each giving generator order, conjugacy class length, and probability of a random element having the right order/class; used to find standard generators x, y of S.
- `rels`: Relations in x, y that (with the generator orders from `geninfo`) present S.
- `permrep`: Permutation representation of the full automorphism group Aut(S); first two generators are x, y, followed by outer generators t, u, v.
- `outimages`: Images of x, y under the outer automorphism generators.
- `order`: Order of G.
- `inv`: Invariant to separate non-isomorphic (|S|, |G|) possibilities — the sum over conjugacy classes of G of the element order in each class.
- `name`: Name of G.
- `conjelts`: Words in t, u, v giving coset representatives of the normaliser of G in Aut(S) (when G is not normal in Aut(S)).
- `subgens`: Words in t, u, v that together with x, y generate G.
- `subpres`: Presentation of G/S on `subgens`.
- `normgens`: Words in t, u, v generating the outer automorphism group of G.
- `normpres`: Presentation of the outer automorphism group of G on `normgens`.
- `maxsubints`: Records describing intersections of maximal subgroups of G (not containing S) with S: each record gives order, class length, generators as words in x, y, and a presentation.

### 66.6.2 Creating the Database

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AlmostSimpleGroupDatabase()` | Return a database object `D` for use with the almost-simple group functions. | — |

### 66.6.3 Accessing the Database

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `#D` | Number of entries in the database. | — |
| `GroupData(D, i)` / `GroupData(D, o1, o2, k)` | Return the i-th entry of the database as a record, or the entry for an almost-simple group of order `o2` with socle of order `o1` and invariant `k` (sum of element orders over conjugacy classes of G). | Record lookup. |
| `ExistsGroupData(D, o1, o2)` / `ExistsGroupData(D, o1, o2, i)` | Return whether a record exists for simple group of order `o1` and supergroup of order `o2` (with optional invariant `i`). When `true`, also return the record. | — |
| `NumberOfGroups(D, o1, o2)` | Return the number of records for simple group of order `o1` and supergroup of order `o2`, and the index of the first such record. | — |
| `IdentifyAlmostSimpleGroup(G)` | Construct a monomorphism f from the almost-simple group G into the stored permutation representation A of Aut(S) for S = socle of G. Also handles alternating and symmetric groups of degree up to 50 (not in database proper). | Uses database record data; for alternating/symmetric groups, algorithm of Holt from **[BP00]**. |

*Worked example: H66E8 (querying for an almost-simple group of order 720; using k to resolve three groups with the same order pair; constructing G inside the full Aut representation; using `IdentifyAlmostSimpleGroup`).*

---

## 66.7 Database of Transitive Groups

Magma contains all transitive permutation groups of degree up to 32. Degree ≤ 15 by
Butler and McKay; degree 16–30 by Hulpke **[Hul05]**; degree 32 by Cannon and Holt
**[CH08]**. Groups are stored by degree; the numbering matches the standard transitive-groups
database. Within a degree, groups are ordered by the library's fixed ordering.

### 66.7.1 Accessing the Databases

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `TransitiveGroupDatabaseLimit()` | The limiting degree of the transitive groups database. | — |
| `NumberOfTransitiveGroups(d)` | Number of transitive groups of degree `d`. | — |
| `TransitiveGroup(d, n)` | The n-th transitive group of degree `d`, plus a description string. | Database lookup **[Hul05, CH08]**. |
| `TransitiveGroupDescription(d, n)` | A description string for the n-th transitive group of degree `d`. | — |
| `TransitiveGroupDescription(G)` | A description string for the transitive group `G`. | — |
| `TransitiveGroup(d)` | The first transitive group of degree `d` plus its description. | — |
| `TransitiveGroup(d, f)` | The first transitive group of degree `d` satisfying predicate `f`, plus its description. | Sequential scan with predicate. |
| `TransitiveGroup(S, f)` | The first transitive group with degree in sequence `S` satisfying predicate `f`, plus its description. | Sequential scan with predicate. |
| `TransitiveGroups(d: Warning)` | Sequence of all transitive groups of degree `d`. `Warning` (default `true`) prints a warning for large result sets. | — |
| `TransitiveGroups(S: Warning)` | Sequence of all transitive groups with degree in `S`. | — |
| `TransitiveGroups(d, f)` | Sequence of all transitive groups of degree `d` satisfying predicate `f`. | Sequential scan with predicate. |
| `TransitiveGroups(S, f)` | Sequence of all transitive groups with degree in `S` satisfying predicate `f`. | Sequential scan with predicate. |

*Worked example: H66E9 (50 transitive groups of degree 8; the 7 primitive ones).*

### 66.7.2 Processes

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `TransitiveGroupProcess(d)` | Process iterating over all transitive groups of degree `d`. | — |
| `TransitiveGroupProcess(S)` | Process iterating over all transitive groups with degree in `S`. | — |
| `TransitiveGroupProcess(d, f)` | Process iterating over transitive groups of degree `d` satisfying predicate `f`. | — |
| `TransitiveGroupProcess(S, f)` | Process iterating over transitive groups with degree in `S` satisfying predicate `f`. | — |
| `IsEmpty(p)` | `true` if process `p` has passed its last group. | — |
| `Current(p)` | Current group plus a description string. | — |
| `CurrentLabel(p)` | Return `d` and `n` such that current group is `TransitiveGroup(d, n)`. | — |
| `Advance(∼p)` | Advance to the next group. | — |

*Worked example: H66E10 (listing orders of all 5 transitive groups of degree 5 via process).*

### 66.7.3 Transitive Group Identification

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `TransitiveGroupIdentification(G: Raw)` | Return the number and degree of the database entry isomorphic to the transitive group `G` (degree ≤ 30 for proven identification). If `Raw := false` (default `true`), also return a permutation conjugating `G` to the library copy. | Invariant-based identification. |

*Worked example: H66E11 (identifying an insoluble transitive group of degree 16 from the small groups database as group 715).*

---

## 66.8 Database of Primitive Groups

All primitive permutation groups of degree < 4096. Within each degree, groups are stored
by O'Nan-Scott class in the order: soluble affine, insoluble affine, diagonal, product
action, almost simple. Within each class, groups are ordered by increasing size.
Provenance: Sims (degree ≤ 50, **[Sim70]**); Roney-Dougal–Unger (degree < 1000,
**[RDU03]**); Roney-Dougal (degree < 2500, **[RD05]**); Coutts–Quick–Roney-Dougal
(degree < 4096, **[CQRD11]**). The `Filter` parameter (default `"All"`) restricts to O'Nan-Scott
classes: `"Soluble"`, `"Affine"`, `"Diagonal"`, `"Product"`, `"AlmostSimple"`, `"Simple"`,
`"SimpleNA"`.

### 66.8.1 Accessing the Databases

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PrimitiveGroupDatabaseLimit()` | The limiting degree of the primitive groups database. | — |
| `NumberOfPrimitiveGroups(d)` | Number of primitive groups of degree `d`. | — |
| `NumberOfPrimitiveSolubleGroups(d)` | Number of soluble primitive groups of degree `d`. | — |
| `NumberOfPrimitiveAffineGroups(d)` | Number of primitive affine groups of degree `d`. | — |
| `NumberOfPrimitiveDiagonalGroups(d)` | Number of primitive diagonal-action groups of degree `d`. | — |
| `NumberOfPrimitiveProductGroups(d)` | Number of primitive product-action groups of degree `d`. | — |
| `NumberOfPrimitiveAlmostSimpleGroups(d)` | Number of primitive almost-simple groups of degree `d`. | — |
| `PrimitiveGroup(d, n)` | The n-th primitive group of degree `d`, plus a description string and its O'Nan-Scott type string. | Database lookup **[Sim70, RDU03, RD05, CQRD11]**. |
| `PrimitiveGroupDescription(d, n)` | A description string for the n-th primitive group of degree `d`. | — |
| `PrimitiveGroup(d)` | The first primitive group of degree `d` plus description and O'Nan-Scott type. | — |
| `PrimitiveGroup(d, f)` | First primitive group of degree `d` satisfying predicate `f`. | Sequential scan. |
| `PrimitiveGroup(S, f)` | First primitive group with degree in `S` satisfying predicate `f`. | Sequential scan. |
| `PrimitiveGroups(d: Filter)` / `PrimitiveGroups(S: Filter)` / `PrimitiveGroups(: Filter)` | Sequence of all primitive groups of degree `d`, degrees in `S`, or all legal degrees; filtered by O'Nan-Scott class via `Filter`. Omitting degree implies all legal degrees. | — |
| `PrimitiveGroups(d, f: Filter)` / `PrimitiveGroups(S, f)` / `PrimitiveGroups(f)` | Sequence of primitive groups passing the `Filter` and satisfying predicate `f`. Filter is applied before predicate for efficiency. Degree may be a sequence or omitted. | Sequential scan, filter then predicate. |

*Worked example: H66E12 (counts and access for degree 625; 698 total, 647 affine, 509 soluble).*

### 66.8.2 Processes

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PrimitiveGroupProcess(d: Filter)` / `PrimitiveGroupProcess(S: Filter)` / `PrimitiveGroupProcess(: Filter)` | Process iterating over primitive groups of degree `d` (or degrees in `S`, or all legal degrees) passing the filter. | — |
| `PrimitiveGroupProcess(d, f: Filter)` / `PrimitiveGroupProcess(S, f: Filter)` / `PrimitiveGroupProcess(f: Filter)` | Process iterating over primitive groups passing the filter and satisfying predicate `f`. | — |
| `IsEmpty(p)` | `true` if process `p` has passed its last group. | — |
| `Current(p)` | Current group plus its description string. | — |
| `CurrentLabel(p)` | Return `d` and `n` such that the current group is `PrimitiveGroup(d, n)`. | — |
| `Advance(∼p)` | Advance to the next group. | — |

*Worked example: H66E13 (iterating diagonal-type primitive groups of degree 60; orbit structures of their Sylow 2-subgroups).*

### 66.8.3 Primitive Group Identification

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PrimitiveGroupIdentification(G)` | Return the number and degree of the database entry permutation-isomorphic to primitive group `G` (degree ≤ 2499). | Invariant-based identification (permutation isomorphism). |

*Worked example: H66E14 (constructing a degree-625 affine primitive group via `Getvecs`/`Semidir`; identifying it as group 595 of degree 625).*

---

## 66.9 Database of Rational Maximal Finite Matrix Groups

The database contains rational maximal finite matrix groups and their invariant forms, for
dimensions up to 31, as determined by Nebe and Plesken **[NP95, Neb96]**. Each entry is
accessible as either a matrix group (order and base set on return) or as a lattice (with
automorphism group set).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RationalMatrixGroupDatabase()` | Return a database object `D`. | — |
| `LargestDimension(D)` | The largest dimension stored; it is an error to reference larger dimensions. | — |
| `#D` / `NumberOfGroups(D)` / `NumberOfLattices(D)` | Total number of entries in the database. | — |
| `NumberOfGroups(D, d)` / `NumberOfLattices(D, d)` | Number of entries of dimension `d`. | — |
| `Group(D, i)` | The i-th entry as a matrix group. | Database lookup **[NP95, Neb96]**. |
| `Lattice(D, i)` | The i-th entry as a lattice. | — |
| `Group(D, d, i)` | The i-th entry of dimension `d` as a matrix group. | — |
| `Lattice(D, d, i)` | The i-th entry of dimension `d` as a lattice. | — |

*Worked example: H66E15 (354 total entries up to dimension 31; dimension 24 has the most, 65 entries; finding the smallest-order group in dimension 24).*

---

## 66.10 Database of Integral Maximal Finite Matrix Groups

Representatives of GL(n, Z)-conjugacy classes of irreducible maximal finite subgroups of
GL(n, Z) for n ≤ 11 and n ∈ {13, 17, 19, 23}. Sources: n < 10 from **[PP77, PP80]**;
dimension 10 from **[Sou94]**; prime dimensions n > 10 from **[Ple85]**. Each entry is
accessible as a matrix group or as a lattice with its invariant forms; `Construction` gives
the lattice name or form coefficients.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IntegralMatrixGroupDatabase()` | Return a database object `D`. | — |
| `LargestDimension(D)` | The largest dimension stored. | — |
| `#D` / `NumberOfGroups(D)` / `NumberOfLattices(D)` | Total number of entries. | — |
| `NumberOfGroups(D, d)` / `NumberOfLattices(D, d)` | Number of entries of dimension `d`. | — |
| `Group(D, i)` | The i-th entry as a matrix group. | Database lookup **[PP77, PP80, Sou94, Ple85]**. |
| `Lattice(D, i)` | A lattice L and sequence of additional invariant forms F for the i-th entry. | — |
| `Construction(D, i)` | A string S describing the construction of the i-th group. If the G-invariant lattice is well known, S is its name; if the degree d is prime, S is the coefficient vector [a₀, a₁, …] of the invariant form; otherwise S gives the isomorphism type. Also returns the indices of all rational matrix groups of degree d that contain a GL(d,Q)-conjugate copy of G. | — |
| `Group(D, d, i)` | The i-th entry of dimension `d` as a matrix group. | — |
| `Lattice(D, d, i)` | Lattice L and forms F for the i-th entry of dimension `d`. | — |
| `Construction(D, d, i)` | Construction string and integer for the i-th entry of dimension `d`. | — |

*Worked example: H66E16 (222 entries total; group A8* of dimension 8 embedded into rational matrix group database entry 3 of dimension 8 via lattice isometry).*

---

## 66.11 Database of Finite Quaternionic Matrix Groups

Finite absolutely irreducible subgroups of GLₙ(D) where D is a definite quaternion algebra
whose centre has degree d over Q with nd ≤ 10. Due to Gabriele Nebe **[Neb98]**. Entries
are accessible as matrix groups or as lattices with associated forms.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `QuaternionicMatrixGroupDatabase()` | Return a database object `D`. | — |
| `LargestDimension(D)` | The largest dimension stored. | — |
| `#D` / `NumberOfGroups(D)` / `NumberOfLattices(D)` | Total number of entries. | — |
| `NumberOfGroups(D, d)` / `NumberOfLattices(D, d)` | Number of entries of dimension `d`. | — |
| `Group(D, i)` | The i-th entry as a matrix group (order and base set). | Database lookup **[Neb98]**. |
| `Lattice(D, i)` | A lattice L and sequence of forms F for the i-th entry. | — |
| `Construction(D, i)` | A string and integer describing the construction of the i-th entry. | — |
| `Group(D, d, i)` | The i-th entry of dimension `d` as a matrix group. | — |
| `Lattice(D, d, i)` | Lattice L and forms F for the i-th entry of dimension `d`. | — |
| `Construction(D, d, i)` | Construction string and integer for the i-th entry of dimension `d`. | — |

*Worked example: H66E17 (largest dimension 40; 10 groups in dimension 36; group of order 2¹⁰·3⁵·5²·7; determinant of associated lattice).*

---

## 66.12 Database of Finite Symplectic Matrix Groups

Maximal finite irreducible subgroups of Sp₂ₙ(Q) for 1 ≤ n ≤ 11, up to conjugacy in
GL₂ₙ(Q). Due to Markus Kirschmer **[Kir09]**. Stored groups do not fix the standard
skew-symmetric form; the example shows how to conjugate to the standard form. Entries
are accessible as matrix groups or as lattices with a pair of forms (Gram matrix and
skew-symmetric form).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SymplecticMatrixGroupDatabase()` | Return a database object `D`. | — |
| `LargestDimension(D)` | The largest dimension stored. | — |
| `#D` / `NumberOfGroups(D)` / `NumberOfLattices(D)` | Total number of entries. | — |
| `NumberOfGroups(D, d)` / `NumberOfLattices(D, d)` | Number of entries of dimension `d`. | — |
| `Group(D, i)` | The i-th entry as a matrix group. | Database lookup **[Kir09]**. |
| `Lattice(D, i)` | A lattice L and a sequence S of two integral forms: the Gram matrix of L and a skew-symmetric form. The sequence is normalized as described in the appendix of **[Kir09]**. | — |
| `Construction(D, i)` | A string describing the construction of the i-th entry. | — |
| `Group(D, d, i)` | The i-th entry of dimension `d` as a matrix group. | — |
| `Lattice(D, d, i)` | Lattice and forms for the i-th entry of dimension `d`. | — |
| `Construction(D, d, i)` | Construction string for the i-th entry of dimension `d`. | — |

*Worked example: H66E18 (91 groups of dimension 16; conjugating to standard symplectic form via `TransformForm`).*

---

## 66.13 Database of Irreducible Matrix Groups

All irreducible subgroups of GLₖ(p) (p prime, k ≥ 1, pᵏ < 2500); one representative
per conjugacy class. The data is the same as that used for the affine primitive permutation
groups (see §66.8 for provenance). Within each pᵏ, soluble groups come first, then
insoluble, each ordered by increasing order (GLₖ(p) is last in each list).

### 66.13.1 Accessing the Database

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `NumberOfIrreducibleMatrixGroups(k, p)` | Number of subgroups of GLₖ(p) stored (pᵏ < 2500). | — |
| `NumberOfSolubleIrreducibleMatrixGroups(k, p)` | Number of soluble subgroups of GLₖ(p) stored. | — |
| `IrreducibleMatrixGroup(k, p, n)` | The n-th subgroup of GLₖ(p) stored. | Database lookup **[RDU03, RD05]**. |

*Worked example: H66E19 (647 irreducible subgroups of GL₄(5); 509 soluble; group 511 with chief factors C₂, A₅, C₂; checking irreducibility and absolute irreducibility).*

---

## 66.14 Database of Quasisimple Matrix Groups

A database of characteristic-0 representations of finite quasisimple groups.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `QuasisimpleMatrixGroup(N, d, p: OverZ, Automorphisms, RepNo)` | Return an absolutely irreducible matrix group in characteristic `p` (prime or 0) of dimension `d` for the quasisimple group named `N`. The group is derived from reduction mod `p` of a characteristic-0 representation. Parameters: `OverZ` (default `true` iff `p = 0`): use the integral form of the representation; `Automorphisms` (default `false`): include extra generators for outer automorphisms stabilising the representation (may introduce infinite scalar subgroup when `p = 0`); `RepNo` (default 1): select among multiple representations of the same (N, d). If reduction mod `p` is reducible, a random irreducible constituent is returned. ATLAS names are used for groups in §66.16. | Reduction modulo `p` of stored integral or number-field representation. |
| `QuasisimpleMatrixGroups()` | Return a list of tuples `(name, dimension, number_of_representations)` specifying the contents of the database. | — |

---

## 66.15 Database of Soluble Irreducible Groups

One representative of each conjugacy class of irreducible soluble subgroups of GL(n, p)
(p prime) for n > 1 and pⁿ < 256. Constructed by Mark Short **[Sho92]**. Groups are
labelled by triples (d, p, i): degree d ≥ 2, prime p, index i within that degree/field class.

### 66.15.1 Basic Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsolGroupDatabase()` | Open the database and return a handle `D`. | — |
| `IsolGroup(n, p, i)` / `Group(D, n, p, i)` | Return the i-th irreducible soluble subgroup of GL(n, p) from the database. | Database lookup **[Sho92]**. |
| `IsolNumberOfDegreeField(n, p)` | The number of groups of degree `n` over F_p stored. | — |
| `IsolInfo(n, p, i)` | A string giving order and primitivity information about the group with label (n, p, i). | — |
| `IsolOrder(n, p, i)` | The order of the group with label (n, p, i). | — |
| `IsolMinBlockSize(n, p, i)` | The minimal block size of the group (0 if primitive). | — |
| `IsolIsPrimitive(n, p, i)` | Whether the group with label (n, p, i) is primitive. | — |
| `IsolGuardian(n, p, i)` | The "guardian" of the group — the maximal subgroup of GL(n, p) containing it. | — |

*Worked example: H66E20 (22 groups of degree 3 over F₅; group 10 of order 62; its guardian of order 372).*

### 66.15.2 Searching with Predicates

Predicates may be: a function `f` taking a matrix group and returning a boolean; a
1-tuple `⟨g⟩` where `g` takes a label; or a 2-tuple `⟨g, f⟩` applying `g` (on label) first to
avoid expanding the group unnecessarily.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsolGroupSatisfying(f)` | Return the first database group satisfying predicate `f` (across all degrees and fields). Error if none found. | Sequential scan. |
| `IsolGroupOfDegreeSatisfying(d, f)` | First group of degree `d` satisfying `f`. | Sequential scan. |
| `IsolGroupOfDegreeFieldSatisfying(d, p, f)` | First group of degree `d` over F_p satisfying `f`. | Sequential scan. |
| `IsolGroupsSatisfying(f)` | Sequence of all database groups satisfying `f`. | Sequential scan. |
| `IsolGroupsOfDegreeSatisfying(d, f)` | Sequence of all groups of degree `d` satisfying `f`. | Sequential scan. |
| `IsolGroupsOfDegreeFieldSatisfying(d, p, f)` | Sequence of all groups of degree `d` over F_p satisfying `f`. | Sequential scan. |

### 66.15.3 Associated Functions

These functions assist in constructing the semidirect product of a finite vector space with
an irreducible matrix group, yielding a soluble affine permutation group. Note that
`Getvecs` depends only on n and p, so it need only be called once when `Semidir` is
called repeatedly for the same GL(n, p).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Getvecs(G)` | For a matrix group G over a finite prime field: return a sequence Q of all vectors of G's natural module, in an ordering that depends only on the module, not on G. | Enumeration of natural module. |
| `Semidir(G, Q)` | For irreducible matrix group G of degree n over F_p and the sequence Q from `Getvecs`: return the permutation group H of degree pⁿ that is the semidirect product of G with its natural module, acting on {1, …, pⁿ}. G is isomorphic to each point stabiliser; H is always primitive; every primitive group with soluble socle arises this way. | Semidirect product construction; well-known result that the construction is primitive. |

*Worked example: H66E14 uses `Getvecs`/`Semidir` to build a primitive group and identify it in §66.8.*

### 66.15.4 Processes

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsolProcess()` | Process iterating over all groups in the database. | — |
| `IsolProcessOfDegree(d)` | Process iterating over all groups of degree `d`. A degree specifier may be a single value or a tuple ⟨l, h⟩ meaning all degrees in [l, h]. | — |
| `IsolProcessOfField(p)` | Process iterating over all groups over the specified prime field. A field specifier may be a single prime or a tuple ⟨l, h⟩. | — |
| `IsolProcessOfDegreeField(d, p)` | Process iterating over groups with degree specifier `d` and field specifier `p` (principal key: degree). | — |
| `IsEmpty(p)` | `true` if process `p` has passed its last group. | — |
| `Current(p)` | Current group. | — |
| `CurrentLabel(p)` | Return `d`, `n`, `i` such that the current group is `IsolGroup(d, n, i)`. | — |
| `Advance(∼p)` | Advance to the next group. | — |

*Worked example: H66E21 (collecting orders of all groups of degree 3 via `IsolProcessOfDegree`).*

---

## 66.16 Database of ATLAS Groups

Magma includes representations of nearly simple groups from the Birmingham ATLAS of
Finite Group Representations (http://web.mat.bham.ac.uk/atlas/v2.0). Data supplied
by Rob Wilson. Groups are accessed by name; names follow ATLAS conventions for simple
groups with modifications: "T" for twisted Lie-type groups; a leading number for a central
extension; "d" to separate a simple group name from an automorphism; "i" for isoclinic
variants.

### 66.16.1 Accessing the Database

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ATLASGroupNames()` | Return the indexed set of names of groups with representations in the database. | — |
| `ATLASGroup(N)` | Return the ATLAS group object (type `GrpAtlas`) for the group named `N`. | — |

### 66.16.2 Accessing the ATLAS Groups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Order(A)` / `#A` | The order of the ATLAS group `A`. | — |
| `Multiplier(A)` | The order of the Schur multiplier of `A`, when `A` is simple. | — |
| `MatRepKeys(A)` | Sequence of keys to all stored matrix representations of `A`. Empty if none stored. | — |
| `MatRepDegrees(A)` | Set of degrees of stored matrix representations of `A`. | — |
| `MatRepFieldSizes(A)` | Set of field sizes for which a matrix representation of `A` is available. | — |
| `MatRepCharacteristics(A)` | Set of characteristics for which a matrix representation of `A` is available. | — |
| `PermRepKeys(A)` | Sequence of keys to stored permutation representations of `A`. Empty if none. | — |
| `PermRepDegrees(A)` | Set of degrees of stored permutation representations of `A`. | — |

### 66.16.3 Representations of the ATLAS Groups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `MatrixGroup(K)` | Construct and return the matrix group corresponding to database key `K`. | Construction from ATLAS generator data. |
| `MatRep(K)` | The generators of the matrix group designated by key `K`. | — |
| `PermutationGroup(K)` | Construct and return the permutation group corresponding to database key `K`. | — |
| `PermRep(K)` | The generators of the permutation group designated by key `K`. | — |

*Worked example: H66E22 (list of ATLAS group names in V2.11). H66E23 (accessing 2.J2.2; no permutation representations; three matrix representations of degrees 12, 6, 12; composition factors of the degree-12 representation over GF(3)).*

---

## 66.17 Fundamental Groups of 3-Manifolds

The database consists of fundamental groups of the 10,986 small-volume closed hyperbolic
3-manifolds in the Hodgson–Weeks census. Presentations generated by Jeffrey Weeks'
SnapPea program. Information about finite-index subgroups with homology generated by
Dunfield and Thurston **[DT03]**.

### 66.17.1 Basic Functions

The `Manifold` function returns a record with the following fields:

- `Name`: String name of the manifold M.
- `Volume`: Volume of M as a floating-point number.
- `Homology`: Sequence of integers describing the first homology group of M.
- `Group`: Fundamental group of M as a finitely presented group.
- `GoodCoverImage`: Possibly empty sequence of permutations (or 1 for the identity) defining a homomorphism from the fundamental group to Sₙ whose kernel has infinite abelianization.
- `GoodCover`: List describing the construction of the good cover.
- `Degree`: Degree of the `GoodCoverImage` permutation representation (equals 1 when the manifold itself has positive Betti number).
- `KnownPosBettiCover`: Boolean, always `true` in the current database.
- `KnownWeakPosBettiCover`: Boolean, always `true` in the current database.
- `Reason`: String: `"AbelianInvariants"`, `"RationalReconstruction"`, or `"MAGMA"`.
- `Rank`: Positive integer.
- `GoodCoverImageU`: Possibly empty sequence of permutations.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ManifoldDatabase()` | Open the database and return a reference `D`. | — |
| `Manifold(D, i)` | Extract the i-th record (1 ≤ i ≤ 11126) as a record with the fields above. | — |

### 66.17.2 Accessing the Data

The database object returned by `ManifoldDatabase()` supports direct iteration (`for r in D`) in addition to indexed access via `Manifold`. The 132 manifolds with positive Betti number are those records where `Degree` equals 1.

*Worked examples: H66E24 (extracting record 100, name "m019(1,4)", homology [2,31], fundamental group, GoodCoverImage; finding the 132 Dunfield–Thurston positive-Betti-number manifolds; locating "s527(-5,1)" by name and verifying infinite abelianization).*

---

## 66.18 Bibliography

| Key | Reference |
|-----|-----------|
| **[BE99a]** | Hans Ulrich Besche and Bettina Eick. Construction of finite groups. *J. Symbolic Comput.*, 27(4):387–404, 1999. |
| **[BE99b]** | Hans Ulrich Besche and Bettina Eick. The groups of order at most 1000 except 512 and 768. *J. Symbolic Comput.*, 27(4):405–413, 1999. |
| **[BE01]** | Hans Ulrich Besche and Bettina Eick. The groups of order qⁿ·p. *Comm. Algebra*, 29(4):1759–1772, 2001. |
| **[BEO01]** | Hans Ulrich Besche, Bettina Eick, and E. A. O'Brien. The groups of order at most 2000. *Electron. Res. Announc. Amer. Math. Soc.*, 7:1–4 (electronic), 2001. |
| **[BP00]** | Sergey Bratus and Igor Pak. Fast constructive recognition of a black box group isomorphic to Sₙ or Aₙ using Goldbach's conjecture. *J. Symbolic Comp.*, 29:33–57, 2000. |
| **[CH08]** | J. J. Cannon and D. F. Holt. The transitive permutation groups of degree 32. *Experiment. Math.*, 17:307–314, 2008. |
| **[CQRD11]** | Hannah J. Coutts, Martyn Quick, and Colva M. Roney-Dougal. The primitive permutation groups of degree less than 4096. *Communications in Algebra*, 39:10:3526–3546, 2011. |
| **[DE05]** | Heiko Dietrich and Bettina Eick. On the groups of cubefree order. *J. Algebra*, 292:122–137, 2005. |
| **[DT03]** | Nathan M. Dunfield and William P. Thurston. The virtual Haken conjecture; experiments and examples. *Geometry & Topology*, 7:399–441, 2003. |
| **[HP89]** | D. F. Holt and W. Plesken. *Perfect Groups*. Oxford University Press, 1989. |
| **[Hul05]** | Alexander Hulpke. Constructing transitive permutation groups. *J. Symbolic Comput.*, 39(1):1–30, 2005. |
| **[Kir09]** | M. Kirschmer. Finite symplectic matrix groups. Dissertation, RWTH Aachen, 2009. Available at http://www.math.rwth-aachen.de/Markus.Kirschmer/symplectic/thesis.pdf. |
| **[MNVL04]** | M. F. Newman, E. A. O'Brien, and M. R. Vaughan-Lee. Groups and nilpotent Lie rings whose order is the sixth power of a prime. *J. Algebra*, 278:383–401, 2004. |
| **[Neb96]** | G. Nebe. Finite subgroups of GL_n(Q) for 25 ≤ n ≤ 31. *Comm. Algebra*, 24(7):2341–2397, 1996. |
| **[Neb98]** | G. Nebe. Finite quaternionic matrix groups. *Represent. Theory*, 2:106–223, 1998. |
| **[NP95]** | G. Nebe and W. Plesken. Finite rational matrix groups. *Mem. Amer. Math. Soc.*, 116(556), 1995. |
| **[O'B90]** | E. A. O'Brien. The p-group generation algorithm. *J. Symbolic Comput.*, 9:677–698, 1990. |
| **[O'B91]** | E. A. O'Brien. The Groups of Order 256. *J. Algebra*, 143:219–235, 1991. |
| **[OVL05]** | E. A. O'Brien and M. R. Vaughan-Lee. The groups with order p⁷ for odd prime p. *J. Algebra*, 2005. |
| **[Ple85]** | Wilhelm Plesken. Finite unimodular groups of prime degree and circulants. *J. Algebra*, 97:286–312, 1985. |
| **[PP77]** | Wilhelm Plesken and Michael Pohst. On maximal finite irreducible subgroups of GL(n,Z). Parts I and II. *Math. Comp.*, 31:536–576, 1977. |
| **[PP80]** | Wilhelm Plesken and Michael Pohst. On maximal finite irreducible subgroups of GL(n,Z). Parts III–V. *Math. Comp.*, 34(149):245–301, 1980. |
| **[RD05]** | Colva M. Roney-Dougal. The primitive permutation groups of degree less than 2500. *J. Algebra*, 292(1):154–183, 2005. |
| **[RDU03]** | Colva M. Roney-Dougal and William R. Unger. The affine primitive permutation groups of degree less than 1000. *J. Symbolic Comp.*, 35:421–439, 2003. |
| **[Sho92]** | Mark W. Short. *The Primitive Soluble Permutation Groups of Degree less than 256*, volume 1519 of Lecture Notes in Math. Springer, Berlin and Heidelberg, 1992. |
| **[Sim70]** | C. C. Sims. Computational methods in the study of permutation groups. In J. Leech, editor, *Computational problems in abstract algebra*, pages 169–183. Oxford – Pergamon, 1970. |
| **[Sou94]** | Bernd Souvignier. Irreducible finite integral matrix groups of degree 8 and 10. *Math. Comp.*, 63:335–350, 1994. |

---

## Algorithm-to-function quick reference

| Database / Algorithm / Source | Key Functions |
|-------------------------------|---------------|
| Small Groups Library **[BE99a, BEO01, BE99b, O'B90, BE01, O'B91, MNVL04, OVL05, DE05]** | `SmallGroupDatabase`, `SmallGroup`, `SmallGroups`, `SmallGroupProcess`, `IdentifyGroup`, `CanIdentifyGroup`, `IsSoluble`/`IsInsoluble` variants, `Data`, `SmallGroupEncoding`, `SmallGroupDecoding` |
| p-group enumeration (n ≤ 7) **[BE99a, BEO01, O'B90, O'B91, MNVL04, OVL05]** | `SearchPGroups`, `CountPGroups` |
| Metacyclic p-groups (Newman–O'Brien–Vaughan-Lee) | `MetacyclicPGroups`, `IsMetacyclicPGroup`, `InvariantsMetacyclicPGroup`, `StandardMetacyclicPGroup`, `NumberOfMetacyclicPGroups`, `HasAllPQuotientsMetacyclic` |
| Perfect groups **[HP89]** | `PerfectGroupDatabase`, `Group`, `PermutationGroup`, `PermutationRepresentation`, `IdentificationNumber`, `NumberOfRepresentations`, `NumberOfGroups`, `TopQuotients`, `ExtensionPrimes`, `ExtensionExponents`, `ExtensionNumbers`, `ExtensionClasses` |
| Almost-simple groups (Holt–Gebhardt–Unger); alternating/symmetric identification **[BP00]** | `AlmostSimpleGroupDatabase`, `GroupData`, `ExistsGroupData`, `NumberOfGroups`, `IdentifyAlmostSimpleGroup` |
| Transitive groups (Butler–McKay; Hulpke **[Hul05]**; Cannon–Holt **[CH08]**) | `TransitiveGroupDatabaseLimit`, `NumberOfTransitiveGroups`, `TransitiveGroup`, `TransitiveGroupDescription`, `TransitiveGroups`, `TransitiveGroupProcess`, `TransitiveGroupIdentification` |
| Primitive groups (Sims **[Sim70]**; Roney-Dougal–Unger **[RDU03]**; Roney-Dougal **[RD05]**; Coutts–Quick–Roney-Dougal **[CQRD11]**) | `PrimitiveGroupDatabaseLimit`, `NumberOfPrimitiveGroups` (and class variants), `PrimitiveGroup`, `PrimitiveGroupDescription`, `PrimitiveGroups`, `PrimitiveGroupProcess`, `PrimitiveGroupIdentification` |
| Rational maximal matrix groups **[NP95, Neb96]** | `RationalMatrixGroupDatabase`, `LargestDimension`, `NumberOfGroups`, `NumberOfLattices`, `Group`, `Lattice` |
| Integral maximal matrix groups **[PP77, PP80, Sou94, Ple85]** | `IntegralMatrixGroupDatabase`, `LargestDimension`, `NumberOfGroups`, `NumberOfLattices`, `Group`, `Lattice`, `Construction` |
| Quaternionic matrix groups **[Neb98]** | `QuaternionicMatrixGroupDatabase`, `LargestDimension`, `NumberOfGroups`, `NumberOfLattices`, `Group`, `Lattice`, `Construction` |
| Symplectic matrix groups **[Kir09]** | `SymplecticMatrixGroupDatabase`, `LargestDimension`, `NumberOfGroups`, `NumberOfLattices`, `Group`, `Lattice`, `Construction` |
| Irreducible matrix groups **[RDU03, RD05]** | `NumberOfIrreducibleMatrixGroups`, `NumberOfSolubleIrreducibleMatrixGroups`, `IrreducibleMatrixGroup` |
| Quasisimple matrix groups (characteristic-0 to char-p reduction) | `QuasisimpleMatrixGroup`, `QuasisimpleMatrixGroups` |
| Soluble irreducible groups **[Sho92]** | `IsolGroupDatabase`, `IsolGroup`, `IsolNumberOfDegreeField`, `IsolInfo`, `IsolOrder`, `IsolMinBlockSize`, `IsolIsPrimitive`, `IsolGuardian`, `IsolGroupSatisfying`, `IsolGroupOfDegreeSatisfying`, `IsolGroupOfDegreeFieldSatisfying`, `IsolGroupsSatisfying`, `IsolGroupsOfDegreeSatisfying`, `IsolGroupsOfDegreeFieldSatisfying`, `IsolProcess`, `IsolProcessOfDegree`, `IsolProcessOfField`, `IsolProcessOfDegreeField`, `Getvecs`, `Semidir` |
| ATLAS representations (Wilson) | `ATLASGroupNames`, `ATLASGroup`, `Order`, `Multiplier`, `MatRepKeys`, `MatRepDegrees`, `MatRepFieldSizes`, `MatRepCharacteristics`, `PermRepKeys`, `PermRepDegrees`, `MatrixGroup`, `MatRep`, `PermutationGroup`, `PermRep` |
| 3-manifold fundamental groups (Hodgson–Weeks census; Dunfield–Thurston **[DT03]**) | `ManifoldDatabase`, `Manifold` |
