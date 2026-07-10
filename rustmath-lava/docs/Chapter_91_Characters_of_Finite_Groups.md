# Chapter 91 — Characters of Finite Groups

**Handbook part:** XII — Representation Theory
**Handbook pages:** 2758–2778 (PDF pages 2890–2911)

---

## Scope and Overview

Chapter 91 covers the ring of class functions on a finite group G and the computation of ordinary
irreducible characters, the Schur index, Brauer characters, and related operations.

The mathematical objects are **class functions** on G (complex-valued functions constant on
conjugacy classes), represented as vectors of k cyclotomic-field values indexed by the k conjugacy
classes.  They form a C-algebra, the **character ring** (Magma type `AlgChtr`; elements
`AlgChtrElt`).

**Algorithms for character tables.** Three independent algorithms are implemented:

1. **Unger's Induce/Reduce algorithm [Ung06]** — the default for groups of order > 5000 that are
   not p-groups.  Elementary subgroups (direct product of cyclic and p-group) are constructed and
   their characters (computed by Conlon's algorithm for the p-part [Con90]) are induced to G; LLL
   reduction maintains a manageable basis of the character space; arithmetic exploits a finite-field
   representation of generalised characters (as in Dixon's work [Dix67]) but using a prime up to
   twice as large.

2. **Dixon-Schneider algorithm [Dix67, Sch90]** — the previous default; still available on demand
   and used as a preprocessing pass inside Induce/Reduce.  Default for groups of order ≤ 5000.

3. **Conlon's algorithm [Con90]** — for p-groups; also the default for larger p-groups.

**Schur indices.** Functions for computing the Schur index of an ordinary irreducible character
over Q, arbitrary number fields, and their completions.  The algorithm (Nebe–Unger, number-field
extension by Fieker) works with characters and groups/fields only, without computing representations.

**Brauer characters.** Partial support: a Brauer character modulo p is represented as a class
function that is zero on p-singular elements, making standard operations (addition, multiplication,
induction, restriction) directly applicable.

---

## 91.1 Creation Functions

### 91.1.1 Structure Creation

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ClassFunctionSpace(G)` | Create the ring of complex-valued class functions of the finite group G.  Triggers computation of conjugacy classes if not yet known.  Irreducible character information is stored in the ring when computed. | — |
| `CharacterRing(G)` | Synonym for `ClassFunctionSpace(G)`. | — |

### 91.1.2 Element Creation

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `elt< R \| a1, ..., ak : parameters >` | Create a class function on G (with k conjugacy classes) whose value on the i-th class is ai, where all ai lie in a common cyclotomic field.  Parameter: `Character` (BoolElt, default `false`): if `true`, the result is flagged as a proper character. | — |
| `R ! [ a1, ..., ak ]` | Alternate coercion syntax for the same operation. | — |
| `R ! a` | Define a constant class function with value a (integer, rational, or cyclotomic field element) on every class. | — |
| `Id(R)` / `Identity(R)` / `One(R)` | The principal character (value 1 on every element of G). | — |
| `PrincipalCharacter(G)` | The principal character of G (value 1 everywhere).  Accepts G or its character ring R. | — |
| `Zero(R)` | The zero element of the character ring R (value 0 on every element). | — |

### 91.1.3 The Table of Irreducible Characters

The default algorithm is Unger's Induce/Reduce **[Ung06]** for non-p-groups of order > 5000,
Conlon's algorithm **[Con90]** for larger p-groups, and Dixon-Schneider **[Dix67, Sch90]** for
groups of order ≤ 5000.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `KnownIrreducibles(G)` | Return the table of irreducible characters currently stored for G (or its class function space R).  New irreducibles deducible from stored information (e.g. norm = 1) are inserted automatically. | — |
| `CharacterTable(G : parameters)` | Compute the complete table of ordinary irreducible characters of G.  Parameters: `Al` (MonStgElt, default `"Default"`): `"DS"` = Dixon-Schneider, `"IR"` = Unger Induce/Reduce **[Ung06]**, `"Conlon"` = Conlon (p-groups only); `DSSizeLimit` (RngIntElt, default 0): when positive, run Dixon-Schneider preprocessing using class matrices for classes of size ≤ this value before switching to Induce/Reduce. | Unger Induce/Reduce **[Ung06]** (default large groups); Dixon-Schneider **[Dix67, Sch90]** (small groups / preprocessing); Conlon **[Con90]** (p-groups). |
| `CharacterTableDS(G : parameters)` | Compute the character table using the Dixon-Schneider algorithm **[Dix67, Sch90]** directly.  Parameters: `ClassMatrices` (SeqEnum, default []): preferred order for class matrices; `ClassMatrixLimit` (RngIntElt, default ∞): max number of class matrices; `ClassSizeLimit` (RngIntElt, default ∞): restrict to classes of size ≤ this; `MinChars` (RngIntElt, default ∞): stop after finding this many additional characters; `Modulus` (RngIntElt, default 0): prime for internal arithmetic (must be ≡ 1 mod exponent and > 2√|G|).  Second return value: sequence of unsplit character spaces remaining. | Dixon-Schneider **[Dix67, Sch90]** |
| `Basis(R)` | Compute the basis of the character ring R consisting of irreducible characters (same computation as `CharacterTable`). | As `CharacterTable`. |
| `CharacterTableConlon(G)` | Compute the character table using Conlon's algorithm.  G must be a p-group. | Conlon's algorithm **[Con90]** |
| `LinearCharacters(G)` | Determine the (partial) character table containing only the linear (degree-1) characters of G. | — |
| `CharacterDegrees(G)` | Degrees of the ordinary irreducible characters of G as a sequence of pairs `<degree, count>`.  For p-groups uses Slattery's counting algorithm; for other soluble groups uses Conlon's counting algorithm; for insoluble groups computes the character table. | Slattery (p-groups); Conlon counting (soluble); full table (insoluble). |
| `CharacterDegrees(G, z, p)` | Degrees of the absolutely irreducible characters of G lying over a faithful linear character of ⟨z⟩, where z is a central element of G and p is zero or a prime. | — |
| `CharacterDegreesPGroup(G)` | Degrees of the ordinary irreducible characters of a p-group G as a sequence [c0, c1, c2, …] where ci = number of characters of degree pⁱ. | Slattery's counting algorithm |
| `RationalCharacterTable(G)` | Returns a sequence of minimal rational characters of G (the sums over Galois orbits on the character table). | — |

*Worked examples: H91E1 (character table of Alt(5), print format, symbolic values); H91E2 (character table of PΓU(5,4) of order 2²² · 3² · 5⁵ · 13 · 17 · 41, 160 characters).*

---

## 91.2 Character Ring Operations

### 91.2.1 Related Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Parent(R)` / `Category(R)` | Standard structure operations on the ring R. | — |
| `Group(R)` | Given the ring of class functions R on G, return G. | — |
| `Centre(x)` | The centre of character x of G: the subgroup of G consisting of classes C where \|x(g)\|, g ∈ C, equals the degree of x. | — |
| `CoefficientField(x)` | The minimal coefficient field Q(ζm) of the class function x. | — |
| `Kernel(x)` | The kernel of character x: the normal subgroup of G consisting of elements g for which x(g) = x(1). | — |

---

## 91.3 Element Operations

### 91.3.1 Arithmetic

In the list below, x and y denote class functions in the same ring, a denotes a scalar (any element
coercible into a cyclotomic field), and j denotes an integer.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `+y` / `-y` / `x + y` / `x - y` / `x * y` / `a * x` / `x ^ j` | Standard arithmetic on class functions. | — |

### 91.3.2 Predicates and Booleans

Note: all functions except `in`, `notin`, `IsReal`, and `IsFaithful` use the table of irreducible
characters, creating it if necessary.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `x in y` | Returns true if the inner product of class functions x and y is non-zero.  If x is irreducible and y is a character, tests whether x is a constituent of y. | — |
| `x notin y` | Returns true if the inner product is zero (x is not a constituent of y). | — |
| `a in F` / `a notin F` | Membership in a field. | — |
| `x eq y` / `x ne y` | Equality and inequality of class functions. | — |
| `IsCharacter(x)` | True if x is a character (all inner products with irreducible characters are non-negative integers). | — |
| `IsGeneralizedCharacter(x)` | True if x is a generalised character (all inner products with irreducible characters are integers). | — |
| `IsIrreducible(x)` | True if x is an irreducible character. | — |
| `IsLinear(x)` | True if x is a linear (degree-1) character. | — |
| `IsFaithful(x)` | True if x is faithful (trivial kernel). | — |
| `IsReal(x)` | True if x takes real values on all classes of G. | — |
| `IsOne(x)` / `IsMinusOne(x)` / `IsZero(x)` | Test whether x equals 1, −1, or 0 as a class function. | — |

### 91.3.3 Accessing Class Functions

T is a character table (enumerated sequence of characters with special printing); x is any class
function.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `T[i]` | The i-th character from the character table T (i in 1…k). | — |
| `T[i][j]` | Value of the i-th irreducible character on the j-th conjugacy class of G. | — |
| `#T` | Number of entries in the character table (or any sequence of characters). | — |
| `x(g)` / `g @ x` | Value of the class function x on element g of G. | — |
| `x[i]` | Value of the class function x on the i-th conjugacy class of G. | — |
| `#x` | Length of x (equals the number of conjugacy classes of G). | — |

### 91.3.4 Conjugation of Class Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `x ^ g` | Given a class function x on a normal subgroup N of G and g ∈ G, returns the conjugate class function x^g where x^g(n) = x(g⁻¹ng) for all n ∈ N. | — |
| `x ^ H` | Given a class function x on a normal subgroup N of G and a subgroup H of G, returns the sequence of all conjugates of x under the action of H. | — |
| `GaloisConjugate(x, j)` | Returns the Galois conjugate x^j of x under the element of Gal(Q(x)/Q) determined by integer j (which must be coprime to the exponent m of G; Q(x) is the subfield of Q(ζm) generated by the values of x). | — |
| `GaloisOrbit(x)` | Returns the sequence of all Galois conjugates of x under Gal(Q(x)/Q). | — |
| `ClassPowerCharacter(x, j)` | For class function x on G and positive integer j, returns the class function x_j where x_j(g) = x(g^j). | — |

### 91.3.5 Functions Returning a Scalar

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Degree(x)` | The degree of class function x: the value of x on the identity element of G. | — |
| `InnerProduct(x, y)` | The inner product of class functions x and y (must belong to the same character ring). | — |
| `Order(x)` | For a linear character x of G, the order of x as an element of the group of linear characters of G. | — |
| `Norm(x)` | Norm of class function x (inner product with itself). | — |
| `Schur(x, k)` | Generalised Frobenius-Schur indicator: given class function x and positive integer k, return the coefficient a_x in the expansion T_k = Σ_{χ ∈ Irr(G)} a_χ χ, where T_k(g) = \|{h ∈ G : h^k = g}\|. | — |
| `Indicator(x)` | Equivalent to `Schur(x, 2)` (the classical Frobenius-Schur indicator). | — |
| `StructureConstant(G, i, j, k)` | The structure constant a_{i,j,k} of the centre of the group algebra of G: if K_i is the formal sum of all elements in the i-th conjugacy class, then K_i * K_j = Σ_k a_{i,j,k} * K_k. | — |

### 91.3.6 The Schur Index

Magma computes the Schur index of an ordinary irreducible character over Q, absolute number fields,
and their completions.  The algorithm (Nebe–Unger; number-field extension by Fieker) uses
characters, groups, and fields without computing representations.  All routines are based on
`SchurIndices(x)`, which computes Schur indices over all completions of Q.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SchurIndex(x)` | Schur index of the complex irreducible character x over Q. | Nebe–Unger algorithm (character/group/field methods, no representations). |
| `SchurIndex(x, F)` | Schur index of x over the absolute number field F. | Nebe–Unger with Fieker's number-field extension. |
| `SchurIndices(x)` | Sequence of pairs `<completion, Schur index>` over all completions of Q where the Schur index exceeds 1.  For Q: completion 0 = reals, prime p = Q_p.  Empty sequence if Schur index is 1 everywhere. | Nebe–Unger algorithm. |
| `SchurIndices(x, F)` | As `SchurIndices(x)` but over completions of the absolute number field F; completions are PlcNumElt objects. | Nebe–Unger with Fieker extension. |
| `SchurIndices(C, s, F)` | Given character field C, output s of `SchurIndices(x)`, and a number field F: compute Schur indices of x over F without repeating group/character computations. Useful when considering several fields for one character. | Nebe–Unger with Fieker extension. |
| `SchurIndexGroup(n : parameters)` | Return a group having a faithful character with Schur index n over Q, using the metacyclic construction of [Tur01].  Parameter: `Prime` (RngIntElt): supply the prime p explicitly (p = kn+1 with k, n coprime); default uses least such prime. | Metacyclic construction **[Tur01]** |
| `CharacterWithSchurIndex(n : parameters)` | Return a character with Schur index n over Q (second return value: the group, equal to `SchurIndexGroup(n)`).  Same `Prime` parameter as above. | Metacyclic construction **[Tur01]** |

*Worked examples: H91E3 (Schur index of the faithful 2-dimensional character of D_8 vs. Q_8; Schur indices over cyclotomic fields and a non-normal cubic field); H91E4 (construction of a character with Schur index 6 via the metacyclic group with n=6, p=7, a=3).*

### 91.3.7 Attribute

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AssertAttribute(x, "IsCharacter", b)` | Procedure: given class function x and Boolean b, store with x that `IsCharacter(x)` equals b. | — |

### 91.3.8 Induction, Restriction and Lifting

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Induction(x, G)` | Given a class function x on a subgroup H of G, return the induced class function on G.  If x is a character of H, the result is a character of G.  Also accepts a sequence of characters (e.g. a character table) to induce all at once. | Frobenius induction formula. |
| `LiftCharacter(c, f, G)` | Given a class function c of the quotient group Q of G and the natural homomorphism f : G → Q, lift c to a class function of G. | — |
| `LiftCharacters(T, f, G)` | Given a sequence T of class functions (or a character table) of the quotient group Q of G and the natural homomorphism f : G → Q, lift T to a sequence of class functions of G. | — |
| `Restriction(x, H)` | Given a class function x on G and a subgroup H of G, return the restriction of x to H.  If x is a character of G, the result is a character of H. | — |

### 91.3.9 Symmetrization

See **[Mur58]** or **[Fra82]** for mathematical background.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Symmetrization(x, p)` | Given a class function x and a partition p of n (2 ≤ n ≤ 6, specified as a sequence of positive integers summing to n), return the symmetrised character with respect to p. | Symmetrization via partition **[Mur58, Fra82]** |
| `OrthogonalComponent(x, p)` | Given class function x (not a linear character, with Frobenius-Schur indicator 1) and a partition p of n (2 ≤ n ≤ 6), return the Murnaghan component of the orthogonal symmetrization of x with respect to p. | Murnaghan orthogonal symmetrization **[Mur58, Fra82]** |
| `SymplecticComponent(x, p)` | Given class function x (not a linear character, with Frobenius-Schur indicator −1) and a partition p of n (2 ≤ n ≤ 6), return the Murnaghan component of the symplectic symmetrization of x with respect to p. | Murnaghan symplectic symmetrization **[Mur58, Fra82]** |
| `SymmetricComponents(x, n)` | Given class function x and integer n, return the set of symmetrizations of x by all partitions of m with 2 < m ≤ n ≤ 5. | Symmetrization **[Mur58, Fra82]** |
| `OrthogonalComponents(x, n)` | Given class function x (Frobenius-Schur indicator 1, not linear) and integer n, return the set of Murnaghan components for orthogonal symmetrizations by all partitions of m with 2 < m ≤ n ≤ 6. | Murnaghan **[Mur58, Fra82]** |
| `SymplecticComponents(x, n)` | Given class function x (Frobenius-Schur indicator −1, not linear) and integer n, return the set of Murnaghan components for symplectic symmetrizations by all partitions of m with 2 < m ≤ n ≤ 5. | Murnaghan **[Mur58, Fra82]** |

### 91.3.10 Permutation Character

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PermutationCharacter(G)` | Given G as a permutation group, construct the character afforded by the defining permutation representation of G. | — |
| `PermutationCharacter(G, H)` | Given a group G and subgroup H, construct the character of G afforded by the permutation representation of G acting on right cosets of H in G. | — |

### 91.3.11 Composition and Decomposition

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Composition(T, q)` | Given a sequence or table of characters T for G and a sequence q of k elements of Q(ζm) (possibly Q), create the class function q₁·T₁ + ··· + qk·Tk. | — |
| `Decomposition(T, y)` | Given a sequence or table of class functions T for G (length l) and a class function y on G, attempt to express y as a linear combination of elements of T.  Returns a sequence q = [q₁, …, q_l] (where qi = (y,Ti)/(Ti,Ti)) and the residual class function z = y − Σ qi·Ti.  If T is the complete irreducible character table, z = 0 iff y is a linear combination of the Ti. | Inner-product decomposition. |

### 91.3.12 Finding Irreducibles

A common approach is to generate new characters via `SymmetricComponents`, `OrthogonalComponents`,
or `SymplecticComponents`, then use norms and inner products to identify irreducibles.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RemoveIrreducibles(I, C)` | Remove occurrences of irreducible characters in sequence I from the characters in sequence C and search for norm-1 characters among the reduced characters.  Returns a sequence of new irreducibles found and the sequence of reduced characters. | Subtraction and norm test. |
| `ReduceCharacters(I, C)` | Reduce norms of characters in C by computing differences of appropriate pairs.  Returns a sequence of new irreducibles found and a sequence of reduced characters. | Norm reduction by differencing. |

*Worked example: H91E5 (constructing the full character table of A5 from scratch using only characters on subgroups: principal character, permutation character, induction of a linear character from a stabiliser and from a cyclic subgroup of order 5, then Galois conjugation and decomposition; compared to the standard character table).*

### 91.3.13 Brauer Characters

Magma provides partial support for Brauer characters.  A Brauer character modulo p is represented
as a class function that is zero on p-singular elements.  Standard operations (addition,
multiplication, induction, restriction) apply directly in this representation.  Note: issues related
to the choice of lifting from finite fields to complex roots of unity have not yet been addressed.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BrauerCharacter(x, p)` | The Brauer character modulo the prime p obtained from x by setting values on p-singular elements to zero. | — |
| `Blocks(T, p)` | Given the full ordinary character table T of a group and a prime p, return the partition of T into p-blocks as a sequence of sets of character indices, together with the corresponding sequence of block defects.  Blocks are ordered first by decreasing defect, then by first character in block. | — |

*Worked example: H91E6 (3-modular characters of the Higman-Sims simple group: `Blocks(T,3)` gives 6 blocks of defects 2,2,1,0,0,0; the defect-1 block {T[8], T[13], T[16]} is examined via `BrauerCharacter` and projective indecomposable characters).*

---

## 91.4 Bibliography

| Key | Reference |
|-----|-----------|
| **[Con90]** | S. B. Conlon. Calculating characters of p-groups. *J. Symbolic Comp.*, 9:535–550, 1990. |
| **[Dix67]** | J. D. Dixon. High-speed computation of group characters. *Numerische Mathematik*, 10:446–450, 1967. |
| **[Fra82]** | J. S. Frame. Recursive computation of tensor power components. *Bayreuth. Math. Schr.*, (10):153–159, 1982. |
| **[Mur58]** | F. D. Murnaghan. The orthogonal and symplectic groups. *Comm. Dublin Inst. Adv. Studies. Ser. A*, no. 13:146, 1958. |
| **[Sch90]** | G. J. A. Schneider. Dixon's Character Table Algorithm Revisited. *J. Symbolic Computation*, 9:601–606, 1990. |
| **[Tur01]** | A. Turull. Schur indices of perfect groups. *Proc. Amer. Math. Soc.*, 130(2):367–370, 2001. |
| **[Ung06]** | W. R. Unger. Computing the character table of a finite group. *J. Symbolic Comp.*, 41(8):847–862, 2006. |

---

## Algorithm-to-Function Quick Reference

| Algorithm / Theory | Functions |
|--------------------|-----------|
| Unger Induce/Reduce **[Ung06]** | `CharacterTable(:Al="IR")`, `CharacterTable` (default, large non-p-groups) |
| Dixon-Schneider **[Dix67, Sch90]** | `CharacterTableDS`, `CharacterTable(:Al="DS")`, `CharacterTable` (default, order ≤ 5000) |
| Conlon's algorithm (p-groups) **[Con90]** | `CharacterTableConlon`, `CharacterTable(:Al="Conlon")`, `CharacterTable` (default p-groups), `CharacterDegrees` |
| Slattery's counting algorithm (p-groups) | `CharacterDegreesPGroup`, `CharacterDegrees` (p-groups) |
| Frobenius induction / Brauer's theorem | `Induction`, `Restriction`, `LiftCharacter`, `LiftCharacters` |
| Schur index (Nebe–Unger, Fieker extension) | `SchurIndex`, `SchurIndices` |
| Metacyclic construction for Schur index **[Tur01]** | `SchurIndexGroup`, `CharacterWithSchurIndex` |
| Murnaghan symmetrization **[Mur58, Fra82]** | `Symmetrization`, `OrthogonalComponent`, `SymplecticComponent`, `SymmetricComponents`, `OrthogonalComponents`, `SymplecticComponents` |
| Galois action on character values | `GaloisConjugate`, `GaloisOrbit`, `ClassPowerCharacter` |
| Inner-product decomposition / norm reduction | `Decomposition`, `Composition`, `RemoveIrreducibles`, `ReduceCharacters`, `InnerProduct`, `Norm` |
| Permutation character construction | `PermutationCharacter` |
| Brauer characters / p-blocks | `BrauerCharacter`, `Blocks` |
