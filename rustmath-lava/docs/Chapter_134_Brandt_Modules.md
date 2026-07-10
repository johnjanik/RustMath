# Chapter 134 — Brandt Modules

**Author:** David Kohel (implementation); Markus Kirschmer (F_q[t] quadratic-forms techniques)
**Handbook part:** XVII — Modular Arithmetic Geometry
**Handbook pages:** 4485–4495 (PDF pages 4616–4629)

---

## Scope and overview

Brandt modules provide a representation, in terms of quaternion ideals, of certain cohomology
subgroups associated to Shimura curves X₀ᴰ(N), which generalize the classical modular curves
X₀(N). The Brandt module datatype is that of a **Hecke module** — a free module of finite rank
equipped with the action of a ring of Hecke operators, a canonical basis (identified with left
quaternion ideal classes), and an inner product that is adjoint with respect to the Hecke
operators. The machinery of modular symbols, Brandt modules, and (in a future release) a module
of singular elliptic curves form the computational machinery underlying modular forms in Magma.

Brandt modules were implemented by David Kohel, motivated by the article of Mestre and Oesterlé
**[Mes86]** on the method of graphs for supersingular elliptic curves, by Pizer's article
**[Piz80]** on computing spaces of modular forms using quaternion arithmetic, and by the author's
thesis **[Koh96]** on endomorphism-ring structure of elliptic curves over finite fields. The
Brandt module machinery is described in **[Koh01]** and has been used, together with modular
symbols, in the computation of component groups of quotients of the Jacobians J₀(N) of classical
modular curves **[KS00]**.

A Brandt module is defined with respect to a definite quaternion order A in a quaternion algebra
H over Q. The *level* of M is the reduced discriminant of A; the *discriminant* is the
discriminant of H (the product of the ramified primes); and the *conductor* is the index of A in
any maximal order containing it. The product of conductor and discriminant equals the reduced
discriminant of A. Two algorithmic routes exist: the classical computation of h × h Gram matrices
of the quaternion ideal norm forms (`BrandtModule(D, m)`), and a newer algorithm
(`BrandtModule(M, N)`) that avoids explicitly constructing the ideals and is preferable when the
level is not very small. The final section treats the analogous theory over F_q[t].

---

## 134.2 Brandt Module Creation

The various constructors for Brandt modules and their elements.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BrandtModule(D)` / `BrandtModule(D, m)` | Given a product `D` of an odd number of primes and an integer `m` with valuation at most 1 at each prime divisor `p` of `D`, return the Brandt module of level `(D, m)` over the integers. If unspecified, the conductor `m` is taken to be 1. Parameter `ComputeGrams` (default `true`): when `false`, the `h × h` array of reduced Gram matrices of the quaternion ideal norm forms (`h` = left class number of level `(D, m)`) is not computed; instead the basis of quaternion ideals is stored and the degree-`p` ideal homomorphisms are computed to find the Hecke operator `T_p` for each prime `p`. Setting `false` is more space-efficient; computing the Gram matrices is preferable for moderate levels where many Hecke operators are wanted. | Classical Brandt-matrix construction via reduced Gram matrices of quaternion ideal norm forms, with Hecke operators from theta series; or, with `ComputeGrams := false`, ideal-homomorphism enumeration **[Mes86]**. |
| `BrandtModule(A)` / `BrandtModule(A, R)` | Given a definite order `A` in a quaternion algebra over **Q**, returns the Brandt module on the left ideal classes for `A`, as a module over `R`. If unspecified, `R` is taken to be the integers. Parameter `ComputeGrams` as above. | As above, over the left ideal classes of `A`. |
| `BaseExtend(M, R)` | Forms the Brandt module with coefficient ring base-extended to `R`. | Coefficient base extension. |
| `BrandtModule(M, N)` | An alternative to `BrandtModule(D, N)` using a *different algorithm*, preferable when `N` is not very small. Constructs the Brandt module attached to an Eichler order of level `N` inside the maximal order `M`; the algorithm avoids explicitly working with the Eichler order. | New ideal-free algorithm: works inside the maximal order, avoiding explicit construction of the Eichler order and its ideals. |

*Worked examples:* H134E1 (Brandt module of level 101 over **F**₇ via `QuaternionOrder`/`FiniteField`/`BrandtModule(A,FF)`, `Decomposition`; comparison with `DimensionCuspFormsGamma0`, `BrandtModuleDimension`, and the `ComputeGrams := false` route); H134E2 (`AmbientModule` recovering the original module); H134E3 (verbose ideal-enumeration output with `SetVerbose("Quaternion",2)` for level (37,1)).

### 134.2.1 Creation of Elements

| Intrinsic | Description |
|-----------|-------------|
| `M ! x` | Given a sequence or module element `x` compatible with the Brandt module `M`, forms the corresponding element in `M`. |
| `M . i` | For a Brandt module `M` and integer `i`, returns the `i`-th basis element. |

### 134.2.2 Operations on Elements

Brandt module elements support standard operations.

| Intrinsic | Description |
|-----------|-------------|
| `a * x` / `x * a` | Scalar multiplication of a Brandt module element `x` by an element `a` in the base ring. |
| `x * T` | Given a Brandt module element `x` and an element `T` of the algebra of Hecke operators of degree compatible with the parent of `x` or its ambient module, returns the image of `x` under `T`. |
| `x + y` | The sum of two Brandt module elements. |
| `x - y` | The difference of two Brandt module elements. |
| `x eq y` | Returns `true` if `x` and `y` are equal elements of the same Brandt module. |
| `Eltseq(x)` | Returns the sequence of coefficients of the Brandt module element `x`. |
| `InnerProduct(x, y)` | Returns the inner product of `x` and `y` with respect to the canonical pairing on their common parent. |
| `Norm(x)` | Returns the inner product of the Brandt module element `x` with itself. |

### 134.2.3 Categories and Parent

Brandt modules belong to the category `ModBrdt`, with elements of type `ModBrdtElt`. The `Parent`
of an element is the space to which it belongs.

| Intrinsic | Description |
|-----------|-------------|
| `Category(M)` / `Type(M)` / `Category(x)` / `Type(x)` | The category, `ModBrdt` or `ModBrdtElt`, of the Brandt module `M` or of the Brandt module element `x`. |
| `Parent(x)` | The parent module `M` of a Brandt module element `x`. |
| `x in M` | Returns `true` if `M` is the parent of `x`. |

### 134.2.4 Elementary Invariants

The elementary invariants are defined with respect to a definite quaternion order `A` in a
quaternion algebra **H** over **Q**: the *level* of `M` is the reduced discriminant of `A`; the
*discriminant* is the discriminant of **H** (product of its ramified primes); the *conductor* is
the index of `A` in any maximal order of **H** containing it. The product of conductor and
discriminant is the reduced discriminant of `A`.

| Intrinsic | Description |
|-----------|-------------|
| `Level(M)` | The level of the Brandt module — the product of the discriminant and the conductor, equal to the reduced discriminant of its defining quaternion order. |
| `Discriminant(M)` | The discriminant of the quaternion algebra **H** with respect to which `M` is defined (the product of the primes ramifying in **H**). |
| `Conductor(M)` | The conductor, i.e. the index of the defining quaternion order of `M` in a maximal order of its quaternion algebra. |
| `BaseRing(M)` | The ring over which the Brandt module `M` is defined. |
| `Basis(M)` | The basis of the Brandt module `M`. |

### 134.2.5 Associated Structures

The `AmbientModule` is the full module containing a given Brandt module, whose basis corresponds
to the left quaternion ideals. Elements of every submodule of the ambient module are displayed
with respect to the basis of the ambient module.

| Intrinsic | Description |
|-----------|-------------|
| `AmbientModule(M)` | The full module of level `(D, m)` containing a given module of this level. |
| `IsAmbient(M)` | Returns `true` if and only if the Brandt module `M` is its own ambient module. |
| `Dimension(M)` / `Rank(M)` | The rank of the Brandt module `M` over its base ring. |
| `Degree(M)` | The degree of `M`, defined to be the dimension of its ambient module. |
| `GramMatrix(M)` | The matrix `(⟨uᵢ, uⱼ⟩)` defined with respect to the basis `{uᵢ}` of the Brandt module `M`. |
| `InnerProductMatrix(M)` | The Gram matrix of the ambient module of the Brandt module `M`. |
| `Ideals(M)` | Constructs the quaternionic ideals corresponding to the basis of `M`. Only implemented when `M` was constructed using `BrandtModule(M, N)` (the new algorithm, which avoids constructing these ideals explicitly). |

### 134.2.6 Verbose Output

The verbose level for Brandt modules is set with `SetVerbose("Brandt", n)`. Since construction
requires intensive quaternion-algebra machinery for ideal enumeration, the `Quaternion` verbose
flag is also relevant. In both cases `n` may be 0 (silent), 1 (verbose), or 2 (very verbose).

---

## 134.3 Subspaces and Decomposition

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `EisensteinSubspace(M)` | The Eisenstein subspace of `M`. When the level of `M` is square-free this is the submodule generated by a vector of the form `(w/w₁, …, w/wₙ)`, if it exists in `M`, where `wᵢ` is the number of automorphisms of the `i`-th basis ideal and `w = LCM({wᵢ})`. | Submodule from the automorphism-count vector. |
| `CuspidalSubspace(M)` | The cuspidal subspace, defined to be the orthogonal complement of the Eisenstein subspace of `M`. If the discriminant of `M` is coprime to the conductor, the cuspidal subspace consists of the vectors `(a₁, …, aₙ)` with `Σᵢ aᵢ = 0`. | Orthogonal complement of the Eisenstein subspace. |
| `OrthogonalComplement(M)` | The Brandt module orthogonal to `M` in the ambient module of `M`. | Orthogonal complement w.r.t. the canonical inner product. |
| `M meet N` | The intersection of the Brandt modules `M` and `N`. | Module intersection. |
| `Decomposition(M, B)` | A decomposition of `M` with respect to the Atkin–Lehner operators and Hecke operators up to the bound `B`. Parameter `Sort` (default `true`): when `true`, returns a sequence sorted under the operator `lt`. | Simultaneous diagonalisation under Atkin–Lehner and Hecke operators `Tₙ`, `n ≤ B`. |
| `SortDecomposition(D)` | Sort the sequence `D` of spaces of Brandt modules with respect to the `lt` comparison operator. | Sort under `lt`. |

*Worked example:* H134E4 (`Decomposition(BrandtModule(2*3*17), 11 : Sort := true)` and `IsEisenstein`).

### 134.3.1 Boolean Tests on Subspaces

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsEisenstein(M)` | Returns `true` if and only if `M` is contained in the Eisenstein subspace of the ambient module. | — |
| `IsCuspidal(M)` | Returns `true` if and only if `M` is contained in the cuspidal subspace of the ambient module. | — |
| `IsIndecomposable(M, B)` | Returns `true` if and only if `M` does not decompose into complementary Hecke-invariant submodules under the Atkin–Lehner operators, nor under the Hecke operators `Tₙ` for `n ≤ B`. | Indecomposability test up to bound `B`. |
| `M1 subset M2` | Returns `true` if and only if `M1` is contained in the module `M2`. | — |
| `M1 lt M2` | For two indecomposable subspaces, returns `true` if and only if `M1 < M2` under the ordering: (1) by dimension (smaller is less); (2) an Eisenstein subspace is less than a cuspidal subspace of the same dimension; (3) by Atkin–Lehner eigenvalues, starting with the *smallest* prime dividing the level and with '+' less than '−'; (4) by `\|Tr(T_{pⁱ}(Mⱼ))\|`, `p` not dividing the level and `1 ≤ i ≤ g` with `g = Dimension(M1)`, the positive one being smaller in the event of equality. Parameter `Bound` (default 101): returns `false` if all primes up to `Bound` fail to differentiate the arguments. | Lexicographic ordering on dimension / Eisenstein-vs-cuspidal / Atkin–Lehner / Hecke-trace data. Condition (4) differs from the modular-symbols version but permits comparison of arbitrary Brandt modules. |
| `M1 gt M2` | The complement of `lt` for Brandt modules `M1` and `M2`. Parameter `Bound` (default 101). | Complement of `lt`. |

---

## 134.4 Hecke Operators

A Brandt module `M` is equipped with a family of linear Hecke operators acting on it, returned
as matrices acting on the right with respect to the basis `Basis(M)`. Hecke operators of the
ambient module may also be applied to elements of submodules. The system of Hecke operators is
computed by default using the theta series that define the classical Brandt matrices. If a module
is created with `ComputeGrams := false`, the Hecke operators are determined by enumeration of
ideals in a *p*-neighbouring operation, analogous to the method-of-graphs approach of Mestre and
Oesterlé **[Mes86]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HeckeOperator(M, n)` | A matrix representing the `n`-th Hecke operator `Tₙ` with respect to `Basis(M)` for the Brandt module `M`. | Theta series of the Brandt/Gram matrices (default), or `p`-neighbour ideal enumeration **[Mes86]** when `ComputeGrams := false`. |
| `AtkinLehnerOperator(M, p)` | The Atkin–Lehner operator on `M`, where `p` is a prime dividing the discriminant of `M`. | Atkin–Lehner involution at `p`. |

---

## 134.5 q-Expansions

A theta series can be associated to any pair of elements of a Brandt module, giving embeddings
(with respect to any fixed module element) of the Brandt module into a space of weight-2 modular
forms.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ThetaSeries(x, y, prec)` | The theta series associated to the pair `(x, y)` of elements of a Brandt module, as an element of a power series ring. | Theta series of the pairing of `x` and `y`. |
| `qExpansionBasis(M, prec)` | A sequence of power series elements, to precision `prec`, spanning the image of the theta functions associated to pairs in the Brandt module `M`. | Spans the theta-function image of pairs in `M`. |

*Worked example:* H134E5 (`EisensteinSubspace`/`CuspidalSubspace`/`Basis` and `qExpansionBasis` of `BrandtModule(7,7)`).

---

## 134.6 Dimensions of Spaces

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BrandtModuleDimension(D, N)` | The dimension of the Brandt module of level `(D, N)`. | Standard (Eichler-type) dimension formulas. |

*Worked example:* H134E6 (`BrandtModuleDimension(2*5*7, 3^i)` for `i` in `[0..10]`, the Eichler orders of index 3ⁱ in a maximal order of the quaternion algebra of discriminant 2·3·7).

---

## 134.7 Brandt Modules Over F_q[t]

This section concerns quaternion orders whose base ring is **F**_q[t]. The definitions and
constructions are similar to the case of quaternion orders over the integers. The implementation
follows the new implementation over the integers (avoiding explicit work with ideals in Eichler
orders) and uses techniques for quadratic forms over **F**_q[t] (developed by Markus Kirschmer).
Where no description is given below, the arguments and return values are similar to the
corresponding intrinsics over the integers.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `BrandtModuleDimension(D, N)` | The dimension of the Brandt module of level `(D, N)` over **F**_q[t]. | Standard formulas. |
| `BrandtModuleDimensionOfNewSubspace(D, N)` | The dimension of the new subspace of the Brandt module of level `(D, N)`. | Standard formulas. |
| `BrandtModule(M, N)` | Constructs the Brandt module attached to an Eichler order of level `N` in the maximal order `M`. | Ideal-free construction with **F**_q[t] quadratic-form techniques (Kirschmer). |
| `QuaternionOrder(M)` | The defining quaternion order of the Brandt module `M`. | — |
| `Level(M)` / `Discriminant(M)` / `Conductor(M)` | Level, discriminant, conductor (as over the integers). | — |
| `Ideals(M)` | The quaternionic ideals corresponding to the basis of `M`. | — |
| `InnerProductMatrix(M)` | The Gram matrix of the ambient module of `M`. | — |
| `HeckeOperator(M, n)` | A matrix representing the `n`-th Hecke operator `Tₙ` with respect to `Basis(M)`. | — |
| `HeckeEigenvectors(M)` | The common eigenvectors for the Hecke operators on the Brandt module `M`, as elements of `M`. | Simultaneous Hecke eigenvector computation. |
| `HeckeEigenvalue(f, p)` | For a Hecke eigenform `f` in a Brandt module, the eigenvalue for the Hecke operator at the prime `p`. | — |

---

## 134.8 Bibliography (canonical references)

| Key | Reference |
|-----|-----------|
| **[Bos00]** | Wieb Bosma, editor. *ANTS IV*, volume 1838 of *LNCS*. Springer-Verlag, 2000. |
| **[Koh96]** | D. Kohel. *Endomorphism rings of elliptic curves over finite fields.* PhD thesis, University of California, Berkeley, 1996. |
| **[Koh01]** | D. Kohel. *Hecke module structure of quaternions.* In K. Miyake, editor, *Class Field Theory — its Centenary and Prospect*, 2001. |
| **[KS00]** | D. Kohel and W. Stein. *Component groups of quotients of J₀(N).* In Bosma **[Bos00]**. |
| **[Mes86]** | J.-F. Mestre and J. Oesterlé. *Method of graphs for supersingular elliptic curves.* (Cited in §134.1 and §134.4; not listed in the chapter's own bibliography.) |
| **[Piz80]** | A. Pizer. *Computing spaces of modular forms using quaternion arithmetic.* (Cited in §134.1; not listed in the chapter's own bibliography.) |

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Classical Brandt-matrix construction (Gram matrices of ideal norm forms, theta series) | `BrandtModule(D)`, `BrandtModule(D, m)`, `BrandtModule(A)`, `BrandtModule(A, R)`, `HeckeOperator`, `GramMatrix` |
| Method of graphs / p-neighbour ideal enumeration **[Mes86]** | `BrandtModule(... : ComputeGrams := false)`, `HeckeOperator` |
| New ideal-free algorithm (Eichler order inside maximal order) | `BrandtModule(M, N)`, `Ideals` |
| Theta series of element pairs | `ThetaSeries`, `qExpansionBasis` |
| Eisenstein / cuspidal decomposition via inner product | `EisensteinSubspace`, `CuspidalSubspace`, `OrthogonalComplement` |
| Atkin–Lehner / Hecke simultaneous decomposition | `Decomposition`, `SortDecomposition`, `AtkinLehnerOperator`, `IsIndecomposable` |
| Standard (Eichler) dimension formulas | `BrandtModuleDimension`, `BrandtModuleDimensionOfNewSubspace` |
| Quaternion quadratic forms over F_q[t] (Kirschmer) | `BrandtModule(M, N)`, `HeckeEigenvectors`, `HeckeEigenvalue` (F_q[t] section) |
