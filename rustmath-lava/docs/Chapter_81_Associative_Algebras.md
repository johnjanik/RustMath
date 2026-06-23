# Chapter 81 — Associative Algebras

**Handbook part:** XI — Algebras
**Handbook pages:** 2443–2466 (PDF pages 2574–2599)

---

## Scope and overview

Chapter 81 covers associative algebras defined by structure constants, the operations and
decompositions available on them, and the theory of orders inside such algebras.

Defining an algebra by structure constants gives a very general setup, but many structural
concepts are restricted to associative algebras. Magma therefore provides a dedicated type
(`AlgAss`) for structure constant algebras that are known to be associative, distinct from
the general type `AlgGen`.

The chapter covers four main areas:

1. **Construction** — building an associative structure constant algebra from scratch (three
   formats for specifying structure constants: nested sequences, flat sequences, or sparse
   quadruples) or deriving one from a group algebra, matrix algebra, or field extension.

2. **Structure and decomposition** — centre, centralizer, idealizer, Lie bracket, commutator
   module/ideal, annihilators, representations, Jacobson radical, and direct sum decomposition
   into indecomposable summands and central idempotents.

3. **Orders** — the type `AlgAssVOrd` for associative orders in algebras over number fields,
   with full arithmetic, bases (including pseudobases over number rings), predicates, and
   ideal theory. Most non-trivial functionality is currently targeted at quaternion algebras
   (see Chapter 86 for the specialised quaternionic intrinsics). A key limitation: the
   rationals `FldRat` are not a number field in Magma, so many order functions require the
   algebra to be over `FldNum`; the recommended workaround is to create Q as a number field
   via `RationalsAsNumberField()`.

4. **Ideals of orders** — creation of one- and two-sided ideals, their attributes (pseudobasis,
   norm, multiplicator ring, colon ideal), arithmetic, and predicates.

---

## 81.2 Construction of Associative Algebras

### 81.2.1 Construction of an Associative Structure Constant Algebra

An associative structure constant algebra is constructed identically to a general structure
constant algebra, except that an additional parameter may avoid checking associativity. Three
internal representations are available: "Dense" (n² vectors of length n, best when most
constants are non-zero), "Sparse" (positions and values of non-zero constants only), and
"Partial" (vectors with recorded non-zero positions, intermediate).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AssociativeAlgebra<R, n \| Q : parameters>` / `AssociativeAlgebra<M \| Q : parameters>` | Creates the associative structure constant algebra over the free module `Rⁿ` with basis e₁,…,eₙ and structure constants given by the sequence `Q`. `Q` may be: (i) n sequences of n sequences of length n; (ii) n² sequences of length n or n² elements of M; (iii) a flat sequence of n³ ring elements. Parameter `Check` (BoolElt, default `true`) verifies associativity; `Rep` (MonStgElt, default `"Dense"`) selects internal representation. | Structure constant checking; three input-format variants. |
| `AssociativeAlgebra<R, n \| T : parameters>` | Creates an associative structure constant algebra from sparse input: `T` is a sequence of quadruples `<i, j, k, aᵏᵢⱼ>` giving only the non-zero structure constants; all others are 0. Parameters `Check` and `Rep` (default `"Sparse"`). | Sparse structure constant construction. |
| `AssociativeAlgebra(A)` | Given a general structure constant algebra `A` of type `AlgGen`, construct an isomorphic associative algebra of type `AlgAss`. Associativity is checked if not already known. Elements can be coerced between `A` and the result. | Isomorphism from `AlgGen` to `AlgAss`. |
| `ChangeBasis(A, B)` | Create a new associative structure constant algebra `A'` isomorphic to `A` by recomputing structure constants in basis `B`. `B` may be a set/sequence of elements of `A`, vectors, or a matrix. Returns `A'` and the isomorphism `A → A'`. Parameter `Rep` (default `"Dense"`). | Basis change via structure constant recomputation. |

### 81.2.2 Associative Structure Constant Algebras from Other Algebras

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Algebra(A)` | If `A` is a group algebra (`AlgGrp`, vector representation) or a matrix algebra (`AlgMat`): construct the isomorphic associative structure constant algebra `B` together with the isomorphism `A → B`. | Conversion from group/matrix algebra. |
| `Algebra(F, E)` | For finite fields or algebraic number fields `E ⊆ F`: returns the associative algebra `A` of dimension `[F:E]` over `E` isomorphic to `F`, together with the isomorphism `F → A` mapping the `(i−1)`-th power of the generator of `F/E` to the `i`-th basis vector of `A`. | Field-as-algebra construction. |
| `AlgebraOverCenter(A)` | For a simple algebra `A` of type `AlgMat` or `AlgAss` with center `K`: returns a `K`-algebra `B` which is `K`-isomorphic to `A`, plus the isomorphism `A → B`. | Center extraction and re-presentation. |

---

## 81.3 Operations on Algebras and their Elements

### 81.3.1 Operations on Algebras

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Centre(A)` | The centre of the associative algebra `A`. | — |
| `Centralizer(A, S)` / `Centraliser(A, S)` | The centralizer of the subalgebra `S` of `A`: the subalgebra of `A` commuting elementwise with `S`. | — |
| `Idealizer(A, B : parameters)` / `Idealiser(A, B : parameters)` | The largest subalgebra of `A` in which `B` is an ideal. Parameter `Side` (MonStgElt, default `"Both"`): `"Left"`, `"Right"`, or `"Both"` selects one-sided or two-sided idealizer. | — |
| `LieAlgebra(A)` | For an associative structure constant algebra `A`: returns the structure constant algebra `L` with product `(a, b) ↦ a*b − b*a`, plus the map identifying elements of `A` and `L`. | Lie bracket construction. |
| `CommutatorModule(A, B)` | For subalgebras `A`, `B` of an associative algebra with underlying module `M`: the submodule of `M` spanned by commutators `[a, b] = a*b − b*a`, `a ∈ A`, `b ∈ B`. | — |
| `CommutatorIdeal(A, B)` | For subalgebras `A`, `B` of an associative algebra: the ideal generated by all commutators `[a, b] = a*b − b*a`. | — |
| `LeftAnnihilator(A, B)` | Subalgebra of `A` consisting of all elements `a` with `a*b = 0` for all `b ∈ B`. | — |
| `RightAnnihilator(A, B)` | Subalgebra of `A` consisting of all elements `a` with `b*a = 0` for all `b ∈ B`. | — |

*Worked example: H81E1 (constructing `sl₃(Q)` as a structure constant algebra via `LieAlgebra(Algebra(MatrixRing(Rationals(), 3)))`, identifying the Cartan subalgebra, and working out the root system).*

### 81.3.2 Operations on Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Centralizer(A, s)` / `Centraliser(A, s)` | The centralizer of element `s` in associative algebra `A`: the subalgebra of `A` commuting with `s`. | — |
| `LieBracket(a, b)` / `(a, b)` | The Lie bracket `a*b − b*a` of elements `a`, `b` in an associative algebra. | — |
| `IsScalar(a)` | Returns `true` (and the coerced element) if `a` belongs to the base ring `F` of its parent algebra. | — |
| `RepresentationMatrix(a, M : parameters)` | Returns the matrix representation of Side-multiplication by element `a` in associative algebra `A` (with 1) on the `A`-module `M`. Parameter `Side` (MonStgElt, default `"Right"`). | — |

### 81.3.3 Representations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `MatrixAlgebra(A)` | For an associative algebra `A` of dimension `n`: returns an isomorphic matrix algebra. Degree is `n` if `A` has an identity element, otherwise `n + 1`. | — |
| `MatrixAlgebra(A, M : parameters)` | For a finite-dimensional `R`-algebra `A` and a Side `A`-module `M` (both free as `R`-modules): returns the matrix algebra of `A`-endomorphisms of `M` and the `R`-algebra homomorphism `A → End(M)`. Parameter `Side` (MonStgElt, default `"Right"`). | — |
| `RegularRepresentation(A : parameters)` | For an associative algebra `A` of dimension `n` over `R`: the regular representation as an `n × n` matrix algebra over `R`, plus the homomorphism. Basis element `eᵢ` maps to the matrix whose `i`-th row is the coordinates of `eᵢ*a` w.r.t. the stored basis. Parameter `Side` (MonStgElt, default `"Right"`): `"Left"` gives the left-regular representation (rows contain `a*eᵢ`). | — |

### 81.3.4 Decomposition of an Algebra

Functions for understanding the structure of a finite-dimensional associative algebra.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `JacobsonRadical(A)` | The largest nilpotent ideal of `A`. Works for finite-dimensional associative algebras over a field of characteristic 0 or over a finite field. Parameter `Al` (MonStgElt, default `"Default"`): setting `Al := "Meataxe"` uses the meataxe algorithm. | Default: algorithm of **[CIW97]**; alternative: Meataxe. |
| `DirectSumDecomposition(A)` / `IndecomposableSummands(A)` | Returns the direct sum decomposition of `A` as a sequence of indecomposable ideal summands (with no further decomposition as a direct sum), plus the corresponding primitive central idempotents. | Algorithm of **[EG96]**. |
| `CentralIdempotents(A)` | Let `Z` be the centre of `A` and `J(Z)` its Jacobson radical. Returns: (1) a sequence of primitive orthogonal idempotents in `Z` whose images in `Z/J(Z)` span `Z/J(Z)` (each generates a two-sided ideal in `A`); (2) the sequence of those ideals. If `A` is semisimple, the idempotents span `Z` and the ideals are simple algebras summing to `A`. | Algorithm of **[EG96]**. |

*Worked example: H81E2 (Jacobson radical of the group algebra `GF(3)[SmallGroup(27,5)]` equals the augmentation ideal, dimension 26).*

*Worked example: H81E3 (direct sum decomposition of `Q[SmallGroup(10,2)]` into two 1-dimensional and two 4-dimensional ideals via `CentralIdempotents`).*

---

## 81.4 Orders

Let `F` be a number field with ring of integers `R`, and `A` an associative algebra over `F`
(finite-dimensional, with 1). An associative order `O` of `A` is a subring `O ⊆ A` which is
a projective `R`-module such that `O·F = A`.

In Magma, associative orders have type `AlgAssVOrd` and may be defined for any associative
algebra of type `AlgAssV`, namely `AlgAss`, `AlgMat`, `AlgQuat`, and `AlgGrp`. Orders have
ideals of type `AlgAssVOrdIdl` and elements of type `AlgAssVOrdElt`. In the special case
where `A` is a quaternion algebra over Q, `A` has type `AlgQuat` and orders have type
`AlgQuatOrd`.

Orders are represented by a pseudobasis (see Section 55.10). Only basic arithmetic is
currently available for general associative orders; most non-trivial functionality targets
quaternion algebras (Chapter 86).

**Important:** In Magma, `FldRat` is not a subtype of `FldNum`. Much functionality here is
designed for algebras over `FldNum`. For algebras over Q, use `RationalsAsNumberField()` to
create Q as a number field.

### 81.4.1 Creation of Orders

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Order(R, S)` | For ring `R` (a number ring or Z) and sequence `S` of elements of an associative algebra `A`: the order of `A` generated freely over `R` by `S`. | — |
| `Order(S)` | For a sequence `S` of elements of an associative algebra `A` over a number field `F`: the order of `A` generated by `S`. | — |
| `Order(S, I)` | For a sequence `S` of elements of `A` (over number field `F` with integers `R`) and sequence `I` of ideals of `R`: the order generated by `S` with coefficient ideals `I`. | — |
| `Order(A, m, I)` | For algebra `A` (over number field `F` with integers `R`), matrix `m`, and sequence `I` of ideals of `R`: the order whose elements are the rows of `m` in the basis of `A`, with coefficient ideals `I`. | — |
| `Order(A, pm)` | For algebra `A` (over number field with integers `R`, the base ring of `pm`) and pseudomatrix `pm`: the order of `A` specified by `pm`; rows give elements in the basis of `A`. | — |
| `MaximalOrder(A)` | Computes a maximal Z-order in the semisimple associative algebra `A`, which must be defined over the rational numbers. | Algorithm of **[Fri00]**, §3.5; closely related to **[IR93]**. |

*Worked example: H81E4 (three methods for creating a quaternion order over a cubic number field: via `Order(A, M, I)`, via `Order(A, P)` with a pseudomatrix, and via `Order([alpha, beta])`).*

*Worked example: H81E5 (orders in an FP-algebra of dimension 9 over a cyclotomic field, and in a group algebra `F[DihedralGroup(6)]`).*

*Worked example: H81E6 (maximal order of a 9-dimensional matrix algebra over Q; Jacobson radical is zero; discriminant equals 1; multiplication table entries shown).*

### 81.4.2 Attributes

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BaseRing(O)` / `CoefficientRing(O)` | The base ring of the associative order `O`. | — |
| `Algebra(O)` | The container algebra of the associative order `O`. | — |
| `Degree(O)` / `Dimension(O)` | The dimension of the order `O`, equivalently the dimension of its parent algebra as a vector space over its ground field. | — |
| `Discriminant(O)` | The discriminant of `O`. For a quaternion order, returns the reduced discriminant (square root of the usual discriminant). | — |
| `FactoredDiscriminant(O)` | The factorization of the discriminant (or reduced discriminant for quaternion orders). | — |
| `MultiplicationTable(O)` | The three-dimensional multiplication table of structure constants for the maximal order `O`. `T[i][j]` is a sequence of integers giving the coefficients of the product of the `i`-th and `j`-th basis elements. | — |
| `Module(O)` | The pseudomatrix describing the basis of the associative order `O` over a number ring. | — |
| `TraceZeroSubspace(O)` | For an order `O` in a quaternion algebra: the submodule of elements with trace 0. Returns a basis or pseudo-basis according to whether the base field is Q or a number field. | — |

### 81.4.3 Bases of Orders

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Basis(O)` | A basis of the order `O`. All elements of `O` are integral linear combinations of this basis. Note: basis elements live in the parent algebra `A` and may not be elements of `O` itself (due to coefficient ideals). | — |
| `PseudoBasis(O)` | The pseudobasis of the associative order `O` over a number ring: a sequence of tuples (coefficient ideal, basis element). | — |
| `PseudoMatrix(O)` | The pseudomatrix describing the pseudobasis of `O` over a number ring. | — |
| `ZBasis(O)` | A Z-basis for the order `O`. | — |
| `Generators(O)` | A sequence of generators of `O` as a module over its base ring. | — |

*Worked example: H81E7 (demonstrating the difference between `Basis`, `PseudoBasis`, `PseudoMatrix`, and `ZBasis` for a quaternion order with non-trivial coefficient ideals).*

### 81.4.4 Predicates

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `O1 eq O2` | `true` if and only if orders `O1` and `O2` are equal as subrings of the same algebra. | — |
| `x in O` | `true` if element `x` of an associative algebra is in the order `O`. | — |
| `x notin O` | `true` if element `x` is not in the order `O`. | — |

### 81.4.5 Operations with Orders

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Adjoin(O, x)` / `Adjoin(O, x, I)` | Returns the order obtained by adjoining element `x` to order `O`, optionally with coefficient ideal `I`. | — |
| `O1 + O2` | The sum of orders `O1` and `O2`. | — |
| `O1 meet O2` | The intersection of orders `O1` and `O2`. | — |
| `O ^ x` | The conjugate order `x⁻¹Ox`. | — |

*Worked example: H81E8 (forming orders in a quaternion algebra over a cubic number field, computing discriminants, using `Adjoin` and `+` to build a larger order).*

---

## 81.5 Elements of Orders

### 81.5.1 Creation of Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `O ! 0` / `Zero(O)` | The zero element of the associative order `O`. | — |
| `O ! 1` / `One(O)` | The identity element of the associative order `O`. | — |
| `O . i` | The `i`-th basis element of `O` as an element of the parent algebra (may not itself be an element of `O` due to coefficient ideals). | — |
| `O ! x` | The element of `O` described by `x`, where `x` may be a sequence, an element of an associative order, or something coercible into the coefficient ring or algebra of `O`. | — |
| `Random(O)` | A random element of the associative order `O` with small coefficients. | — |

### 81.5.2 Arithmetic of Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `x + y` | Sum of elements `x` and `y` of an order. | — |
| `x - y` | Difference of elements `x` and `y` of an order. | — |
| `-x` | Negation of element `x`. | — |
| `x * y` | Product of elements `x` and `y`. | — |
| `u * c` / `c * u` | Product of order element `u` with scalar `c`. | — |
| `x / y` | Quotient of `x` by unit `y` in the parent algebra. | — |
| `x div y` | Exact division of `x` by `y` in the order containing them. | — |
| `x ^ n` | `x` multiplied with itself `n` times. | — |

### 81.5.3 Predicates on Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `x eq y` | `true` if elements `x` and `y` are equal. | — |
| `x ne y` | `true` if elements `x` and `y` are not equal. | — |
| `IsZero(x)` | `true` if `x` is the zero element of its associative order. | — |
| `IsUnit(a)` | `true` if element `a` is a unit in its associative order. | — |
| `IsScalar(x)` | `true` if `x` is an element of the base ring of the order, and if so returns the coerced element. | — |

### 81.5.4 Other Operations with Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ElementToSequence(x)` / `Eltseq(x)` | The sequence of coordinates of element `x` in terms of the basis of the order `O`. | — |
| `Norm(x)` | Norm of element `x` of an order in its parent algebra. | — |
| `Trace(x)` | Trace of element `x` of an order in its parent algebra. | — |
| `LeftRepresentationMatrix(e)` | Matrix describing left multiplication by element `e` of an associative order. | — |
| `RightRepresentationMatrix(e)` | Matrix describing right multiplication by element `e` of an associative order. | — |
| `RepresentationMatrix(a)` | Representation matrix of element `a` of an associative order; describes left multiplication unless parameter `Side` (MonStgElt, default `"Left"`) is set to `"Right"`. | — |
| `CharacteristicPolynomial(x)` | Characteristic polynomial of element `x` of an order in its parent algebra. | — |
| `MinimalPolynomial(x)` | Minimal polynomial of element `x` of an order in its parent algebra. | — |

---

## 81.6 Ideals of Orders

### 81.6.1 Creation of Ideals

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `lideal< O \| E >` / `rideal< O \| E >` / `ideal< O \| E >` | Construct the left, right, or two-sided `O`-ideal generated by the elements in sequence `E` (coercible into `O`). | — |
| `lideal< O \| M >` / `rideal< O \| M >` / `ideal< O \| M >` | Construct a left, right, or two-sided ideal of order `O` whose basis is given by matrix or pseudomatrix `M`. | — |
| `O * e` | Principal left ideal of order `O` generated by element `e`. | — |
| `e * O` | Principal right ideal of order `O` generated by element `e`. | — |
| `RandomRightIdeal(O)` | A random right ideal of order `O` generated by elements with small coefficients. | — |

### 81.6.2 Attributes of Ideals

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Algebra(I)` | The container algebra of ideal `I`. | — |
| `Order(I)` | The order that ideal `I` was created as an ideal of. | — |
| `LeftOrder(I)` | The order which maps ideal `I` to itself under left multiplication. | — |
| `RightOrder(I)` | The order which maps ideal `I` to itself under right multiplication. | — |
| `Basis(I)` / `Basis(I, R)` | Basis of ideal `I`. Elements are returned in `R` (an order or algebra) if given, otherwise in the algebra of `I`. | — |
| `BasisMatrix(I)` / `BasisMatrix(I, R)` | Basis matrix of `I` with respect to the basis of `R` (or of the order `I` was created from). | — |
| `PseudoBasis(I)` / `PseudoBasis(I, R)` | Sequence of tuples (coefficient ideal, basis element) for `I`. Basis elements live in `R` if given, otherwise in the algebra of `I`. | — |
| `PseudoMatrix(I)` / `PseudoMatrix(I, R)` | Pseudomatrix describing the basis of `I`; basis matrix is with respect to the basis of `R` if given, otherwise of the order `I` was created from. | — |
| `ZBasis(I)` | A Z-basis for ideal `I`. | — |
| `Generators(I)` | Generators of ideal `I` as a module over its base ring. | — |
| `Denominator(I)` | The minimal element `d` of the coefficient ring of `O` such that `d*I ⊆ O`, where `O` is the order `I` was created as an ideal of. | — |

### 81.6.3 Arithmetic for Ideals

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `I + J` | Sum of ideals `I` and `J` (must share a side in equal orders). | — |
| `I * J` | Product of right ideal `I` and left ideal `J` of the same order `O`. | — |
| `a * I` / `I * a` | Product of element `a` with ideal `I`. | — |
| `Colon(J, I)` | For left (or right) ideals `I`, `J`: the colon ideal `(J : I) = { x ∈ A : xI ⊂ J }`. | — |
| `MultiplicatorRing(I)` | The colon ideal `(I : I)`: all elements that multiply `I` into itself. | — |

### 81.6.4 Predicates on Ideals

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsLeftIdeal(I)` | `true` if `I` is a left ideal. | — |
| `IsRightIdeal(I)` | `true` if `I` is a right ideal. | — |
| `IsTwoSidedIdeal(I)` | `true` if `I` is a two-sided ideal. | — |
| `I eq J` | `true` if ideals `I` and `J` are equal. | — |
| `I subset J` | `true` if ideal `I` is contained in ideal `J`. | — |
| `a in I` | `true` if element `a` of an associative algebra is in ideal `I`. | — |
| `a notin I` | `true` if element `a` is not in ideal `I`. | — |

### 81.6.5 Other Operations on Ideals

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Norm(I)` | The norm of ideal `I`: the ideal of the base number ring generated by the norms of all elements in `I`. | — |

*Worked example: H81E9 (right ideal in a 9-dimensional FP-algebra over a cyclotomic field; `IsLeftIdeal`/`IsRightIdeal`/`IsTwoSidedIdeal`; `MultiplicatorRing`; `PseudoBasis`; `ZBasis`; `Norm`; `Denominator`; `Colon`).*

---

## 81.7 Quaternionic Orders

The following intrinsics take an argument of type `AlgAssVOrd`, `AlgAssVOrdElt`, or
`AlgAssVOrdIdl` but apply **only** to orders in quaternion algebras. They are documented in
Chapter 86.

| Intrinsic | Description |
|-----------|-------------|
| `MaximalOrder(O)` | Maximal order (quaternionic). |
| `pMaximalOrder(O, p)` | Maximal order at prime `p`. |
| `IsMaximal(O)` | True if `O` is maximal. |
| `IspMaximal(O, p)` | True if `O` is maximal at `p`. |
| `pMatrixRing(O, p)` | Matrix ring at prime `p`. |
| `Embed(Oc, O)` | Embed order `Oc` into `O`. |
| `LeftIdealClasses(S)` | Left ideal class set. |
| `RightIdealClasses(S)` | Right ideal class set. |
| `TwoSidedIdealClasses(S)` | Two-sided ideal classes. |
| `TwoSidedIdealClassGroup(S)` | Two-sided ideal class group. |
| `OptimizedRepresentation(O)` / `OptimisedRepresentation(O)` | Optimized representation of `O`. |
| `Units(S)` / `MultiplicativeGroup(S)` / `UnitGroup(S)` | Unit group. |
| `Conjugate(x)` | Quaternion conjugate of element `x`. |
| `Enumerate(O, A, B)` / `Enumerate(O, B)` / `Enumerate(I, B)` | Enumerate elements of bounded norm. |
| `ReducedBasis(O)` / `ReducedBasis(I)` | Reduced basis of order or ideal. |
| `IsIsomorphic(I, J)` | True if ideals `I` and `J` are isomorphic. |
| `IsLeftIsomorphic(I, J)` | True if `I` and `J` are left-isomorphic. |
| `IsRightIsomorphic(I, J)` | True if `I` and `J` are right-isomorphic. |
| `IsPrincipal(I)` | True if ideal `I` is principal. |

---

## 81.8 Bibliography (canonical references)

| Key | Reference |
|-----|-----------|
| **[CIW97]** | Arjeh M. Cohen, Gábor Ivanyos, and David B. Wales. *Finding the radical of an algebra of linear transformations.* J. Pure Appl. Algebra, 117/118:177–193, 1997. Algorithms for algebra (Eindhoven, 1996). |
| **[EG96]** | W. Eberly and M. Giesbrecht. *Efficient decomposition of associative algebras.* In Y. N. Lakshman, editor, Proceedings of the 1996 International Symposium on Symbolic and Algebraic Computation: ISSAC'96, pages 170–178, New York, 1996. ACM. |
| **[Fri00]** | Carsten Friedrichs. *Berechnung von Maximalordnungen über Dedekindringen.* Dissertation, Technische Universität Berlin, 2000. URL: http://www.math.tu-berlin.de/~kant/publications/diss/diss_fried.pdf.gz. |
| **[IR93]** | Gábor Ivanyos and Lajos Rónyai. *Finding maximal orders in semisimple algebras over Q.* Comput. Complexity, 3(3):245–261, 1993. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Structure constant algebra construction (dense/sparse/partial) | `AssociativeAlgebra< >`, `ChangeBasis` |
| Jacobson radical — Cohen–Ivanyos–Wales algorithm **[CIW97]** | `JacobsonRadical` |
| Jacobson radical — Meataxe algorithm | `JacobsonRadical(:Al := "Meataxe")` |
| Direct sum decomposition / central idempotents — Eberly–Giesbrecht **[EG96]** | `DirectSumDecomposition`, `IndecomposableSummands`, `CentralIdempotents` |
| Maximal order computation — Friedrichs **[Fri00]**; cf. Ivanyos–Rónyai **[IR93]** | `MaximalOrder` |
| Lie algebra from associative algebra (Lie bracket) | `LieAlgebra`, `LieBracket`, `CommutatorModule`, `CommutatorIdeal` |
| Pseudobasis / pseudomatrix representation (order theory) | `Order`, `Basis`, `PseudoBasis`, `PseudoMatrix`, `ZBasis` |
| Regular representation | `RegularRepresentation`, `MatrixAlgebra`, `RepresentationMatrix` |
| Ideal arithmetic and colon ideal | `Colon`, `MultiplicatorRing`, `Norm` |
| Quaternionic specialisations (see Chapter 86) | `MaximalOrder`, `pMaximalOrder`, `LeftIdealClasses`, `RightIdealClasses`, `Enumerate`, `IsIsomorphic`, `IsPrincipal`, etc. |
