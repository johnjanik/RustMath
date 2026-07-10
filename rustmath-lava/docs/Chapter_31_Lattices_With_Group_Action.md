# Chapter 31 — Lattices With Group Action

**Handbook part:** V — Lattices and Quadratic Forms
**Handbook pages:** 719–741 (PDF pages 850–875)

---

## Scope and overview

A G-lattice in Magma is a lattice upon which a finite integral matrix group G acts by right
multiplication. The chapter covers three main areas:

1. **Automorphism groups and isometry testing** — computing the full automorphism group of
   a lattice (the largest matrix group preserving the inner product) and deciding whether two
   lattices are isometric. The implementation uses a backtrack search designed by Bill Unger,
   based on the **Plesken–Souvignier algorithm [PS97]** together with ordered partition methods.
   Optionally, orthogonal decomposition code of Gabi Nebe can be invoked first. A parallel
   subsection treats definite bilinear forms over Fq[t] using a canonical reduction algorithm
   **[Ger03, Kir12]**.

2. **Lattices from matrix groups (G-lattices)** — creating G-lattices from a finite integral
   matrix group G, querying the group action, computing invariant bilinear forms, and computing
   the endomorphism ring and its centre by an averaging-operator approximation.

3. **G-invariant sublattices** — enumerating all G-invariant sublattices of a given G-lattice
   and organising them as a lattice of sublattices (type `LatLat`). The algorithm uses
   **Plesken's centering algorithm [Ple74]**: maximal G-invariant sublattices are constructed
   as kernels of FpG-epimorphisms L/pL → S for simple FpG-modules S, iterated to cover all
   prime-power-index sublattices, then combined by intersection for coprime primes.

The functions for automorphism groups and isometry require exact lattices (over Z or Q).
For the sublattice enumeration, the set of G-invariant sublattices at a given prime p is
finite if and only if Qp ⊗ L is irreducible as a QpG-module.

---

## 31.2 Automorphism Group and Isometry Testing

The automorphism group of a lattice L is the group of Z-module automorphisms of L that
preserve the inner product. The backtrack algorithm searches for images of basis vectors,
restricting candidates to those with the correct inner products with previously placed images.
Additional invariants (Bacher polynomials, orthogonal decomposition) can accelerate the
search for difficult cases. The lattices must be exact (over Z or Q).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AutomorphismGroup(L)` | Computes the automorphism group G of lattice L (automorphisms of the underlying Z-module preserving the inner product). Returns an integral matrix group of degree equal to the rank of L, acting on coordinate vectors. Parameters: `Stabilizer` (compute only the point stabilizer of that many basis vectors; default 0), `BacherDepth` (depth for Bacher polynomial invariants; default −1 means auto; 0 forbids use), `Generators` (known subgroup generators to seed the search), `NaturalAction` (if rank = degree, return the group acting on lattice vectors rather than coordinate vectors; default false), `Decomposition` (attempt orthogonal decomposition via G. Nebe's method first; default false), `Vectors` (matrix whose rows give the search domain, replacing the default set of short vectors; rows together with negations must be closed under the full automorphism group and must span L). | Backtrack search **[PS97]** with ordered partition methods. When `Decomposition := true`, uses G. Nebe's method (automorphism groups of components combined). Bacher polynomials are a combinatorial invariant computed by counting vector pairs with given scalar product. |
| `AutomorphismGroup(L, F)` | Computes the subgroup of the automorphism group of L that also fixes each form in the set or sequence F (matrices not required to be positive definite or symmetric). Useful for automorphism groups of lattices over algebraic number fields. Same parameters as above (except `Decomposition` is absent). | Same backtrack search with additional form-fixing constraint **[PS97]**. |
| `AutomorphismGroup(F)` | Computes the matrix group fixing all forms in sequence F. The first form in F must be symmetric and positive definite; the others are arbitrary. Equivalent to `AutomorphismGroup(LatticeWithGram(F[1]), [F[i] : i in [2..#F]])`. Can be used to compute the Bravais group of a matrix group G (the full automorphism group of all G-fixed forms). | Same as above **[PS97]**. |
| `IsIsometric(L, M)` / `IsIsomorphic(L, M)` | Determines whether lattices L and M are isometric. If so, returns a transformation matrix T as the second value such that F2 = T F1 Tᵀ (F1, F2 the Gram matrices of L, M). Parameters: `BacherDepth`, `LeftGenerators` / `RightGenerators` (known automorphism groups of L / M), `LeftVectors` / `RightVectors` (search-domain matrices; either both or neither must be set; any isometry must map left vectors into the union of right vectors and their negations). | Backtrack search analogous to automorphism group computation **[PS97]**; cost for isometric lattices is roughly the cost of finding one automorphism. |
| `IsIsometric(L, F1, M, F2)` / `IsIsomorphic(L, F1, M, F2)` | Determines whether L and M are isometric with an isometry that also respects the additional bilinear forms given by sequences of Gram matrices F1 and F2. Same return values and parameters as above. | Backtrack search **[PS97]** with additional form constraints. |
| `IsIsometric(F1, F2)` / `IsIsomorphic(F1, F2)` | For two sequences of Gram matrices F1 and F2, determines whether a simultaneous isometry exists (a matrix T with T F1[i] Tᵀ = F2[i] for all i). The first form in each sequence must be positive definite. Same return values and parameters. | Backtrack search **[PS97]** applied simultaneously to all forms; first forms must be positive definite. |

*Worked examples: H31E1 (automorphism group of E8, transforming coordinate action to natural lattice vector action, verifying via `NaturalAction`); H31E2 (using `Stabilizer` to quickly find a large subgroup of Aut(Kappa_13), then extending by combining stabilizers with pair-reduced basis); H31E3 (automorphism group of Lambda_19, analysis of derived series and lower central series); H31E4 (proving that two constructions of the 16-dimensional Barnes-Wall lattice are isometric, and that L is 2-modular).*

### 31.2.1 Automorphism Group and Isometry Testing over Fq[t]

Let q be a power of an odd prime. A bilinear form b over Fq[t] is called **definite** if the
corresponding quadratic form is anisotropic over the completion of Fq(t) at the infinite
place (1/t). The functions below compute automorphism groups and isometries of definite
bilinear forms over Fq[t] using a canonical reduction to dominant diagonal form **[Ger03,
Kir12]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DominantDiagonalForm(X)` | For a symmetric n×n matrix X of rank n over K[t] (K a field of characteristic ≠ 2): returns a symmetric matrix G and T ∈ GL(n, K[t]) such that G = TXTᵀ has dominant diagonal (diagonal degrees ascending; each off-diagonal degree strictly less than the degrees of the corresponding diagonal entries). If K is a finite field, X represents a definite form, and `Canonical := true`: G is unique, the third return value is the automorphism group of G, and the fourth return value is a finite field E that must be supplied as `ExtensionField` in subsequent compatible runs. Parameters: `Canonical` (default false), `ExtensionField`. | Dominant diagonal reduction **[Ger03]**; canonical form algorithm **[Kir12]**. |
| `AutomorphismGroup(G)` | Computes the automorphism group of the definite bilinear form given by symmetric matrix G over Fq[t]. Second return value is a finite field (as in `DominantDiagonalForm`) which may be supplied as `ExtensionField` to speed up later calls over the same Fq. Parameter: `ExtensionField`. | Canonical form reduction **[Kir12]**. |
| `IsIsometric(G1, G2)` | Tests whether two definite bilinear forms over Fq[t] are isometric. If so, second return value is T ∈ GL(n, q) with TG1Tᵀ = G2. Third return value is a finite field (as above). Parameter: `ExtensionField`. | Canonical form comparison **[Kir12]**. |
| `ShortestVectors(G)` | Returns a sequence of tuples `<v, r>` where v is a shortest non-zero vector and r is its norm with respect to the definite bilinear form G over Fq[t] (q odd). | Direct shortest-vector enumeration over Fq[t]. |
| `ShortVectors(G, B)` | Returns all vectors in Fq[t]ⁿ whose norm with respect to definite bilinear form G over Fq[t] (q odd) is at most B, as a sequence of tuples `<v, r>`. | Bounded vector enumeration over Fq[t]. |

*Worked example: H31E5 (testing isometry of two definite forms over F5[t] via `DominantDiagonalForm` with `Canonical`, using `ExtensionField` to ensure compatible reductions).*

---

## 31.3 Lattices from Matrix Groups

In Magma a G-lattice L is a lattice upon which a finite integral matrix group G acts by right
multiplication. Each G-lattice carries references to both the original ("natural") group G
acting on the ambient standard lattice and the reduced group acting on the basis of L. The
group G must be a finite integral matrix group for all functions in this section.

### 31.3.1 Creation of G-Lattices

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Lattice(G)` | Given a finite integral matrix group G, returns the standard G-lattice with standard basis and rank equal to the degree of G. | — |
| `LatticeWithBasis(G, B)` | Given G and a non-singular basis matrix B whose row space is invariant under G (i.e., Bg = TgB for each g ∈ G, Tg a unimodular integral matrix), returns the G-lattice with basis matrix B. The number of columns of B must equal the degree of G. | — |
| `LatticeWithBasis(G, B, M)` | Given G, non-singular B (invariant as above), and a positive definite matrix M invariant under G (i.e., gMgᵀ = M for all g ∈ G), returns the G-lattice with basis matrix B and inner product matrix M. Number of columns of B and both dimensions of M must equal the degree of G. | — |
| `LatticeWithGram(G, F)` | Given G and a positive definite matrix F invariant under G (gFgᵀ = F for all g), returns the G-lattice with standard basis and inner product matrix F (so the Gram matrix equals F). Both dimensions of F must equal the degree of G. | — |

### 31.3.2 Operations on G-Lattices

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsGLattice(L)` | Returns true iff L is a G-lattice (i.e., has an associated group). | — |
| `Group(L)` | Given a G-lattice L, returns the matrix group of the reduced action of G on L (acting on the coordinate lattice of L, as the automorphism group does). | — |
| `NumberOfActionGenerators(L)` / `Nagens(L)` | Returns the number of generators of G. | — |
| `ActionGenerator(L, i)` | Returns the i-th generator of the reduced action of G on L (the reduced representation of the i-th generator of the original G; may be the identity). | — |
| `NaturalGroup(L)` | Returns the matrix group of the natural action of G on L (acts on L naturally, not on the coordinate lattice). | — |
| `NaturalActionGenerator(L, i)` | Returns the i-th generator of the natural action of G on L (simply the i-th generator of the original G). | — |

### 31.3.3 Invariant Forms

The functions in this section compute invariant bilinear forms for G-lattices, represented by
their Gram matrices.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `InvariantForms(L)` | Returns a basis for the space of all (symmetric and antisymmetric) invariant bilinear forms for G as a sequence of matrices. The first entry is a positive definite symmetric form. | — |
| `InvariantForms(L, n)` | Returns a sequence of n ≥ 0 invariant bilinear forms for G (a partial basis). | — |
| `SymmetricForms(L)` | Returns a basis for the space of symmetric invariant bilinear forms for G. The first entry is a positive definite symmetric form. | — |
| `SymmetricForms(L, n)` | Returns a sequence of n ≥ 0 independent symmetric invariant bilinear forms for G. The first entry (if n > 0) is positive definite. | — |
| `AntisymmetricForms(L)` | Returns a basis for the space of antisymmetric invariant bilinear forms for G. | — |
| `AntisymmetricForms(L, n)` | Returns a sequence of n ≥ 0 independent antisymmetric invariant bilinear forms for G. | — |
| `NumberOfInvariantForms(L)` | Returns the dimension of the space of (symmetric and antisymmetric) invariant bilinear forms for G, using a modular method that is always correct and faster than computing the forms directly. | Modular method. |
| `NumberOfSymmetricForms(L)` | Returns the dimension of the space of symmetric invariant bilinear forms for G. | Modular method. |
| `NumberOfAntisymmetricForms(L)` | Returns the dimension of the space of antisymmetric invariant bilinear forms for G. | Modular method. |
| `PositiveDefiniteForm(L)` | Returns a positive definite symmetric matrix F such that gFgᵀ = F for all g ∈ G (a positive definite G-invariant form). | — |

### 31.3.4 Endomorphisms

Endomorphisms of G-lattices are computed by approximating the averaging operator over
the group and applying it to random elements.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `EndomorphismRing(L)` | Returns the endomorphism ring of the G-lattice L as a matrix algebra over Q. | Averaging operator approximation. |
| `Endomorphisms(L, n)` | Returns a sequence of n independent endomorphisms of L as elements of the endomorphism matrix algebra over Q. n must be in [0..d] where d is the dimension of the endomorphism ring. Useful for splitting reducible lattices without computing the full algebra. | Averaging operator approximation. |
| `DimensionOfEndomorphismRing(L)` | Returns the dimension of the endomorphism algebra of the G-lattice L by a modular method (always correct). | Modular method. |
| `CentreOfEndomorphismRing(L)` | Returns the centre of the endomorphism ring of L as a matrix algebra over Q. Can be used to split a reducible lattice into its homogeneous components. | Averaging operator approximation. |
| `CentralEndomorphisms(L, n)` | Returns a sequence of n independent central endomorphisms of L as elements of the corresponding matrix algebra over Q. n must be in [0..d] where d is the dimension of the centre. | Averaging operator approximation. |
| `DimensionOfCentreOfEndomorphismRing(L)` | Returns the dimension of the centre of the endomorphism algebra of the G-lattice L by a modular method (always correct). | Modular method. |

### 31.3.5 G-invariant Sublattices

The functions in this section compute G-invariant sublattices of a given G-lattice L. For a
fixed prime p, the algorithm constructs maximal G-invariant sublattices as kernels of
FpG-epimorphisms L/pL → S for simple FpG-modules S (**Plesken's centering algorithm
[Ple74]**). Iterating yields all G-invariant sublattices of p-power index; intersecting
sublattices of coprime index yields all G-invariant sublattices. The set of sublattices at
prime p is finite if and only if Qp ⊗ L is irreducible as a QpG-module.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Sublattices(G, Q)` / `Sublattices(L, Q)` | Given an integral matrix group G (or G-lattice L) and a set or sequence Q of primes: returns (as a sequence) all G-invariant sublattices not contained in pL for any p ∈ Q and whose index in L is a product of elements of Q. Second return value indicates whether the result is complete. Parameters: `Limit` (stop after this many sublattices; default ∞), `Levels` (only compute sublattices M with at most n composition factors in L/M; default ∞), `Projections` (sequence of projection matrices; only sublattices with the same images under these projections as L are returned). | Plesken's centering algorithm **[Ple74]**. |
| `Sublattices(G, p)` / `Sublattices(L, p)` | Same as above with Q = {p} (a single prime). Same parameters. | Plesken's centering algorithm **[Ple74]**. |
| `Sublattices(G)` / `Sublattices(L)` | Same as above with Q taken to be the set of prime divisors of the order of G. Same parameters. | Plesken's centering algorithm **[Ple74]**. |
| `SublatticeClasses(G)` | For an integral matrix group G, returns representatives for the isomorphism classes of G-invariant lattices (orbits under the unit group of the endomorphism ring E of G). If `MaximalOrders := true` (default false), only sublattice classes invariant under some maximal order of E are considered. Currently requires E to be a field. | Plesken's centering algorithm **[Ple74]** combined with endomorphism ring structure. |

*Worked examples: H31E6 (sublattices of the standard G-lattice for GL(2,3) × S3; computing `PositiveDefiniteForm`, 18 sublattices reduced to 4 isomorphism classes, identified as tensor products of root lattices); H31E7 (using the `Projections` parameter with central idempotents of the endomorphism ring to restrict to finitely many sublattices when G fixes infinitely many).*

### 31.3.6 Lattice of Sublattices

Magma can construct the lattice V (type `LatLat`) of all G-invariant sublattices of the
standard lattice L = Zⁿ. Only primitive sublattices (not contained in kL for any k > 1)
are stored; elements of V have type `LatLatElt` and are numbered from 1 to n. When the
number of primitive sublattices is infinite (if G fixes infinitely many), a limit must be
imposed and operations such as coercions, intersections, and sums assume the result is a
scalar multiple of some element already stored in V.

#### 31.3.6.1 Creating the Lattice of Sublattices

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SublatticeLattice(G, Q)` | Given an integral matrix group G of degree n (or a sequence of integral matrices generating a Z-order in Qⁿˣⁿ) and a set or sequence Q of primes: computes the G-invariant sublattices of Zⁿ not contained in pZⁿ for any p ∈ Q and of index in Zⁿ that is a product of elements of Q. Second return value indicates whether all G-invariant lattices have been constructed. Optional parameters: `Limit`, `Levels`, `Projections` (same as for `Sublattices`). | Plesken's centering algorithm **[Ple74]**. |
| `SublatticeLattice(G, p)` | Same as above with Q = {p}. Same optional parameters. | Plesken's centering algorithm **[Ple74]**. |
| `SublatticeLattice(G)` | Same as above with Q taken to be the prime divisors of the order of G. Same optional parameters. | Plesken's centering algorithm **[Ple74]**. |

#### 31.3.6.2 Operations on the Lattice of Sublattices

In the following, V is a lattice of G-invariant lattices and Q is the set of primes used to
create V.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `#V` | The number of (primitive) lattices stored in V. | — |
| `V ! i` | The i-th element of V with respect to the internal labeling. | — |
| `V ! M` | Given a basis matrix of some G-invariant lattice M, creates the corresponding element of V. | — |
| `NumberOfLevels(V)` | The number of distinct levels (layers) stored in V. Levels are counted starting from 0. | — |
| `Level(V, i)` | The primitive lattices stored at the i-th level. Levels are counted from 0. | — |
| `Levels(v)` | Returns a sequence where the i-th entry is a sequence of primitive lattice elements at the (i−1)-th level. | — |
| `Primes(V)` | The primes that were used to create V. | — |
| `Constituents(V)` | A sequence of the simple FpG-modules (constituents) used during construction of the G-lattices in V. | — |
| `IntegerRing() ! e` | The integer corresponding to lattice element e. | — |
| `e + f` | The sum of lattice elements e and f. | — |
| `e meet f` | The intersection of lattice elements e and f. | — |
| `e eq f` | Tests whether e and f are equal. | — |
| `MaximalSublattices(e)` | Returns the sequence S of maximal sublattices of e having index p for some p ∈ Q, and a list C of integers such that S[i]/e is isomorphic to the C[i]-th constituent of V (in the ordering of `Constituents`). | — |
| `MinimalSuperlattices(e)` | Returns the sequence S of minimal superlattices of e in which e has index p for some p ∈ Q, and a list C of integers such that e/S[i] is isomorphic to the C[i]-th constituent of V. | — |
| `Lattice(e)` | The G-lattice corresponding to lattice element e. | — |
| `BasisMatrix(e)` / `Morphism(e)` | The basis matrix of the G-lattice corresponding to e. | — |

*Worked examples: H31E8 (creating a lattice of sublattices for the cyclic group of order 4 acting on Z²); H31E9 (automorphism group of root lattice A5 as an absolutely irreducible G; exploring G-invariant sublattices at primes 2 and 3; duality of sublattices with respect to a G-invariant form); H31E10 (8-dimensional rational representation of SL(2,7) with endomorphism ring Q(√−7); finding all G-invariant lattices invariant under the maximal order M; classifying finite extensions of G in GL(8,Q) up to conjugacy).*

---

## 31.4 Bibliography

| Key | Reference |
|-----|-----------|
| **[Ger03]** | Larry J. Gerstein. Definite quadratic forms over Fq[X]. *J. Algebra*, 268(1):252–263, 2003. |
| **[Kir12]** | M. Kirschmer. A normal form for definite quadratic forms over Fq[t]. *Math. Comp.*, 81:1619–1634, 2012. |
| **[Ple74]** | Wilhelm Plesken. Beiträge zur Bestimmung der endlichen irreduziblen Untergruppen von GL(n,Z) und ihrer ganzzahligen Darstellungen. PhD thesis, RWTH Aachen, 1974. |
| **[PS97]** | Wilhelm Plesken and Bernd Souvignier. Computing Isometries of Lattices. *J. Symbolic Comp.*, 24(3):327–334, 1997. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Plesken–Souvignier backtrack search with ordered partition methods **[PS97]** | `AutomorphismGroup(L)`, `AutomorphismGroup(L, F)`, `AutomorphismGroup(F)`, `IsIsometric(L, M)`, `IsIsometric(L, F1, M, F2)`, `IsIsometric(F1, F2)` |
| Orthogonal decomposition (Nebe) combined with **[PS97]** | `AutomorphismGroup(L : Decomposition := true)` |
| Dominant diagonal / canonical form over Fq[t] **[Ger03, Kir12]** | `DominantDiagonalForm`, `AutomorphismGroup(G)` (Fq[t] form), `IsIsometric(G1, G2)` (Fq[t] forms) |
| Shortest/short vectors over Fq[t] | `ShortestVectors(G)`, `ShortVectors(G, B)` |
| Plesken's centering algorithm **[Ple74]** | `Sublattices`, `SublatticeClasses`, `SublatticeLattice` |
| Modular dimension counting | `NumberOfInvariantForms`, `NumberOfSymmetricForms`, `NumberOfAntisymmetricForms`, `DimensionOfEndomorphismRing`, `DimensionOfCentreOfEndomorphismRing` |
| Group averaging operator (endomorphism approximation) | `EndomorphismRing`, `Endomorphisms`, `CentreOfEndomorphismRing`, `CentralEndomorphisms` |
