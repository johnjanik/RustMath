# Chapter 37 — Orders and Algebraic Fields

**Handbook part:** VI — Global Arithmetic Fields
**Handbook pages:** 861–960 (PDF pages 988–1093)

---

## Scope and overview

Chapter 37 is the central reference chapter for Magma's number field module, which is based on the Kant/Kash system (Kant-V4) **[KAN97, KAN00]** developed by the group of M. Pohst in Berlin.

The three primary parent structures are: `FldNum` (number fields, including subtypes `FldCyc` and `FldQuad`), `RngOrd` (orders in number fields), and `FldOrd` (fields of fractions of orders). The umbrella type `FldAlg` matches all of these. Number fields are formally algebraic extensions of finite degree over **Q** or another number field; they are constructed as quotients `K = k[t]/(f(t))` or as multivariate quotients `K = k[s₁,…,sₙ]/(f₁(s₁),…,fₙ(sₙ))`. An order `O` over a maximal order `m` is represented using a (pseudo) `m`-basis, and every order carries a unique field of fractions `FieldOfFractions(O)`.

Key algorithmic content includes:
- **Maximal orders** computed via Round-2 and Round-4 algorithms **[Coh93, Bai96, Poh93, PZ89]** for absolute extensions, with a Round-2 variant for relative extensions **[Coh00, Fri97]**, and a special pseudo-basis method for radical (pure) extensions **[Sut12]**.
- **Class groups** computed by a relation/factor-basis method (index calculus) **[Heß96, Coh93]**, with an optional lattice-sieve algorithm for discriminants above 10³⁰ **[Bia]**.
- **Unit groups** by Dirichlet's method, continued fractions (real quadratic), and the relation method, following **[PZ89]** (pp. 343–344) and **[Poh93]**.
- **Norm equations** (Diophantine: class-group-based + lattice enumeration **[Fin84, PZ89]**; field-theoretic: S-unit approach) and specialist solvers for Thue equations (Bilu–Hanrot reduction **[BH96]**), unit equations (Wildanger's method **[Wil97, Wil00]**), and index form equations **[GPP93, GPP96, GS89, Wil97, Wil00]**.

---

## 37.1 Introduction

The introduction establishes the three-way structure (`FldNum` / `RngOrd` / `FldOrd`), the type hierarchy, the distinction between absolute and relative extensions, and the practical rules about when arithmetic is cheapest and which operations require an absolute representation (class groups, unit groups, subfields).

---

## 37.2 Creation Functions

### 37.2.1 Creation of General Algebraic Fields

Algebraic number fields are created as absolute extensions of **Q** or relative extensions over an algebraic field, by specifying one or more irreducible polynomials.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `NumberField(f)` | Create the number field `L = K(α)` by adjoining a root `α` of the irreducible polynomial `f` of degree `n ≥ 1` over `K = Q` or a number field `K`. Parameters: `Check` (default `true`, verify irreducibility), `DoLinearExtension` (default `false`), `Global` (default `false`). Angle-bracket notation `L<y> := NumberField(f)` binds `y = α`. | Quotient polynomial ring `K[t]/(f(t))`; no special algorithm. |
| `RationalsAsNumberField()` / `QNF()` | Returns a number field isomorphic to **Q** (type `FldNum`, not `FldRat`), allowing all number-field functions. Equivalent to `NumberField(x-1 : DoLinearExtension)`. Arithmetic slower than `Rationals()`. | Linear extension; no structural computation. |
| `NumberField(s)` | Given a sequence `s = [s₁,…,sₘ]` of nonconstant irreducible polynomials over (possibly trivial) algebraic extension `K` of **Q**, create the number field `L = K(α₁,…,αₘ)` adjoining one root of each. Parameters: `Check`, `DoLinearExtension`, `Abs`. | Tower construction; each step a quotient ring extension. |
| `ext< F \| s1, ..., sn >` / `ext< F \| s >` | Create the algebraic field extending `F` (which may be **Q** or a field of fractions) by the polynomials `sᵢ` or a sequence `s`. Returns a field of fractions if `F` is one, otherwise a number field. Parameters: `Check`, `Global`, `Abs`, `DoLinearExtension`. | Same tower construction as `NumberField(s)`. |
| `RadicalExtension(F, d, a)` | Given algebraic field `F` and integral element `a ∈ F` that is not an `n`-th power for any `n \| d`, adjoin the `d`-th root of `a` to `F`. Parameter: `Check`. | Creates extension by irreducible polynomial `t^d - a`. |
| `SplittingField(F)` | Given an algebraic field `F`, return the splitting field of its defining polynomial, together with the roots. Parameters: `Abs` (default `true`; return absolute extension), `Opt` (default `true`; try `OptimizedRepresentation`, may be expensive). | Iterative field extensions; maximal orders computed at each step if `Opt := true`. |
| `SplittingField(f)` | Given an irreducible polynomial `f` over **Z**, return its splitting field. | As above with default `Abs := true`. |
| `SplittingField(L)` | Given a sequence `L` of polynomials over a number field or **Q**, compute a common splitting field such that every polynomial in `L` splits into linear factors; also returns the roots. Parameters: `Abs` (default `false`), `Opt` (default `false`). | Iterated field extensions; primitive element theorem for `Abs := true`. |
| `sub< F \| e1, ..., en >` | Given algebraic field `F` with ground field `G` and elements `eᵢ ∈ F`, return the subfield `H = G(e₁,…,eₙ)` together with the embedding `H → F`. | Subfield construction via minimal polynomial. |
| `MergeFields(F, L)` / `CompositeFields(F, L)` | Given absolute algebraic fields `F` and `L`, return a sequence of fields `[M₁,…,Mᵣ]` each containing a root of the defining polynomial of `F` and of `L`. Factorises the defining polynomial of the larger field over the smaller, constructs extensions, then converts to absolute. | Polynomial factorisation over `F` followed by absolute-field conversion. |
| `Compositum(K, L)` | For absolute number fields `K` and `L`, at least one normal, find a smallest common overfield. Result essentially unique since one field is normal. | Normal field embedding via Galois action. |
| `Compositum(K, A)` | For normal number field `K` and abelian extension `A` of a subfield of `K`, find the smallest common overfield. | Compositum using abelian/Galois structure. |
| `OptimizedRepresentation(F)` / `OptimisedRepresentation(F)` / `OptimizedRepresentation(F, d)` / `OptimisedRepresentation(F, d)` | Given algebraic field `F` over **Q**, attempt to find an isomorphic field `L` with a better (smaller absolute discriminant) defining polynomial. If `d` is given, require additionally that `d` does not divide the index `[O_L : E_L]`. Returns `F` if no better polynomial is found. **Note:** computes the maximal order first. | LLL-based polynomial search; maximal order computation triggered. |

*Worked example: H37E1 (OptimizedRepresentation of `x⁴ − 420x² + 40000`, showing discriminant comparison and index computation).*

### 37.2.2 Creation of Orders and Fields from Orders

Orders are subrings of finite index in the ring of integers. The equation order `E_K = Z[α]` of `K = Q(α)` has the same power basis as `K`. Once an order exists, further orders can be derived from it.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `EquationOrder(f)` | Given an irreducible monic integral polynomial `f ∈ R[X]`, return the equation order `E = R[X]/f(X)`. Parameter: `Check` (default `true`). | Direct quotient construction; no structural work. |
| `EquationOrder(K)` | Return the equation order corresponding to the defining polynomial of number field `K` (which must have been defined by a monic integral polynomial). | Extension of the equation order of the ground field. |
| `SubOrder(O)` | If `O` is not an equation order (i.e. is a transformation of some order `O'`), return `O'`. | Traverses the transformation chain. |
| `EquationOrder(O)` | The suborder of `O` defined by a polynomial; equals the final order of iterating `SubOrder`. Requires `O` to have a monic defining polynomial. | — |
| `Integers(O)` / `RingOfIntegers(O)` / `IntegerRing(O)` | Returns `O` itself (the ring of integers of `O`). | Trivial (identity). |
| `sub< O \| a1, ..., ar >` | Create the suborder of `O` generated by `a₁,…,aᵣ ∈ O` as a **Z**-algebra, i.e. `Z[a₁,…,aᵣ]`. Error if the algebra has less than full rank in `O`. Note: `1` need not be among the generators. | Saturation/span computation in the order lattice. |
| `ext< O \| a1, ..., ar >` | Extend order `O` by elements `aᵢ` lying in the maximal order of `O`, forming `O[a₁,…,aᵣ]`. Does not trigger maximal-order computation. | Lattice extension. |
| `ext< O \| f >` | Given order `O` and a polynomial `f` of degree `n` over `O` (irreducible over `O`), create the extension `E = O[α] ≅ O[X]/(f)`, a free `O`-module of rank `n`. | Quotient ring extension. |
| `FieldOfFractions(O)` | Return the field of fractions of `O`. Angle-bracket notation may be used to name basis elements. | Identity map on the underlying field. |
| `Order(F)` | The order of which `F` was created as its field of fractions. Inverse of `FieldOfFractions`. | Structural look-up. |
| `NumberField(O)` | Recursively defined: the number field of `Z` is **Q**; the number field of `O` is the number field of the coefficient ring of `O` with a root of the defining polynomial adjoined. | Recursive unwinding of the tower. |
| `NumberField(F)` | The number field of `Order(F)` for a field of fractions `F`. | As above. |
| `OptimizedRepresentation(O)` / `OptimisedRepresentation(O)` / `OptimizedRepresentation(O, d)` / `OptimisedRepresentation(O, d)` | Given an order `O` over **Z**, attempt to find an isomorphic maximal order `M` with a better defining polynomial (smaller absolute discriminant; or if `d` given, additionally `d ∤ [M : E_M]`). Returns `O` if no improvement found. | LLL-based polynomial search; maximal order of `O` required first. |
| `O + P` | The smallest common overorder of orders `O` and `P` with the same equation order. | Lattice join. |
| `O meet P` | The intersection of orders `O` and `P` with the same equation order. | Lattice meet. |
| `AsExtensionOf(O, P)` | Return `O` as a transformation of `P`, where `O` and `P` have the same coefficient ring. | Transformation matrix computation. |
| `Order(O, T, d)` | Given absolute order `O` with basis `b₁,…,bₙ`, matrix `T ∈ GL(n,Q) ∩ Mat(n,Z)`, and `d ∈ N`, create the order with basis `(1/d ∑ⱼ Tᵢⱼ bⱼ)ᵢ`. Parameter: `Check` (default `true`). | Direct basis transformation. |
| `Order(O, M)` | Given order `O` with pseudo-basis and an `oₖ`-module `M = ∑ᵢ Aᵢ αᵢ`, create the order `∑ᵢ Aᵢ cᵢ` where `cᵢ = ∑ⱼ αᵢⱼ bⱼ`. Parameter: `Check`. | Pseudo-basis construction from module. |
| `Order( [ e1, ..., en ] )` | Given elements `e₁,…,eₙ` in an algebraic extension field `F` over **Q**, create the minimal order of `F` containing all `eᵢ`. Parameters: `Verify` (default `true`, checks integrality), `Order` (default `false`; if `true`, treats inputs as a basis, skips closure test). | Multiplicative closure of generators; saturation. |

*Worked examples: H37E2 (equation order vs maximal order, index 29·53), H37E3 (relationship between order basis, field-of-fractions basis, and number-field power basis).*

### 37.2.3 Maximal Orders

The maximal order `O_K` is the ring of integers of an algebraic field, consisting of all elements whose minimal polynomial over **Z** is monic. The primary algorithm is a combination of **Round-2 and Round-4** methods **[Coh93, Bai96, Poh93, PZ89]** for absolute extensions, and a Round-2 variant for relative extensions **[Coh00, Fri97]**. Radical (Kummer) extensions use a direct pseudo-basis method based only on the valuation of the constant term of the defining polynomial **[Sut12]**. A method due to Pauli is available for equation orders in simple relative extensions (factorisation of the defining polynomial over the completion). An algorithm based on **[Bj94]** (Theorems 1.2 and 7.6) is used when the discriminant or ramification is supplied.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `MaximalOrder(O)` / `MaximalOrder(F)` / `IntegerRing(F)` / `Integers(F)` / `RingOfIntegers(F)` | Return the maximal order (ring of integers) of the number field `F`, or the largest overorder of the order `O` in its number field. Parameters: `Al` (default `"Auto"`; or `"Round2"`, `"Round4"`, `"Pauli"`), `Discriminant` (known discriminant — uses algorithm of **[Bj94]**), `Ramification` (sequence of prime factors of discriminant), `Verbose MaximalOrder` (max 5). | Round-2/Round-4 **[Coh93, Bai96, Poh93, PZ89]**; relative variant **[Coh00, Fri97]**; radical extension method **[Sut12]**; known-discriminant method **[Bj94]**. |
| `MaximalOrder(f)` | Equivalent to `MaximalOrder(NumberField(f))`. Parameter: `Check` (default `true`), plus same parameters as above. | As above. |
| `pMaximalOrder(O, p)` | The `p`-maximal overorder of `O`: the largest overorder `P` such that `(P : O)` is a power of the prime `p` in the coefficient ring of `O`. For Kummer extensions, uses a direct generator method. Parameter: `Al` (same options as `MaximalOrder`). | Round-2/Round-4 at `p` **[Coh93, Bai96, Poh93, PZ89]**; Kummer special case **[Sut12]**. |
| `pRadical(O, p)` | The `p`-radical of order `O`: the ideal `{x ∈ O : xᵏ ∈ pO for some k}`. If `p` is not prime but `p > deg(O)`, computes the `p`-trace-radical `{x ∈ F : Tr(xO) ⊆ pZ}`. Together with `MultiplicatorRing`, can compute maximal orders without factoring the discriminant **[Bj94, Fri00]**. | Radical ideal computation; trace-radical if `p` squarefree. |
| `MultiplicatorRing(I)` | The multiplicator ring (right order) of the ideal `I`: the subring `M = {x ∈ F : xI ⊆ I}` of the field of fractions `F` of the order of `I`. Core subroutine of Round-2. | Direct computation in the ambient field. |

*Worked examples: H37E4 (advantage of `Ramification` parameter for degree-14 and degree-15 fields), H37E5 (implementation of Round-2 via `pRadical` + `MultiplicatorRing`; quartic field with index 64000).*

### 37.2.4 Creation of Elements

Elements of algebraic fields are printed as linear combinations (with rational coefficients) of basis elements. Order elements are printed as sequences of integer coefficients w.r.t. the order basis.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `F ! a` / `elt< F \| a >` | Coerce element `a` (integer, rational, element of a subfield or order in such) into the field `F`. | Coercion. |
| `F ! [a0, ..., am-1]` / `elt< F \| [a0, ..., am-1] >` / `elt< F \| a0, ..., am-1 >` | Given field `F` of degree `m` over its ground field `G` and elements `aᵢ ∈ G`, construct `a₀α₀ + a₁α₁ + ⋯ + aₘ₋₁αₘ₋₁` where `αᵢ` are the basis elements of `F`. | Linear combination construction. |
| `O ! a` / `elt< O \| a >` | Coerce `a` (integer, integral element of an associated algebraic field, or element of a quotient order) into the order `O`. | Coercion with integrality check. |
| `O ! [a0, ..., am-1]` / `elt< O \| [a0, ..., am-1] >` / `elt< O \| a0, ..., am-1 >` | Construct `a₀α₀ + ⋯ + aₘ₋₁αₘ₋₁ ∈ O`, where `α₀,…,αₘ₋₁` is the basis for `O` and each `aᵢ` lies in the ground order of `O`. | Linear combination construction. |
| `Random(F, m)` / `Random(O, m)` | A random element of the algebraic field `F` or order `O`. Maximum coefficient size determined by `m`. | Random generation. |
| `Random(I, m)` | A random element of the ideal `I` (as an element of the field of fractions of the associated order). Maximum coefficient size w.r.t. ideal basis determined by `m`. | Random generation. |
| `One(K)` / `One(O)` / `Identity(K)` / `Identity(O)` / `Zero(K)` / `Zero(O)` / `Representative(K)` / `Representative(O)` | Standard generic element constructors. | — |

*Worked example: H37E6 (three ways to create the same integral element in the maximal order).*

### 37.2.5 Creation of Homomorphisms

Homomorphisms from algebraic fields or orders are specified by the image of the primitive element (and optionally a map on the ground field/ring).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `hom< F -> R \| r >` / `hom< F -> R \| h, r >` | Build the homomorphism `φ : F → R` sending the defining primitive element `α` of `F` to `r ∈ R`. If `F` is a field of fractions, `r` is the image of the primitive element of its equation order. Optional `h : G → R` maps the ground field `G` (required if `G ≠ Q` and `R` doesn't cover `G`). | Ring homomorphism construction. |
| `hom< O -> R \| r >` / `hom< O -> R \| h, r >` | Construct a homomorphism `φ : O → R` sending the primitive element of the equation order of `O` to `r`. Optional `h` maps the coefficient ring of `O`. | Ring homomorphism construction. |
| `hom< O -> R \| b1, ..., bn >` / `hom< O -> R \| m, b1, ..., bn >` | Return the map from order `O` to ring `R` sending basis elements to `b₁,…,bₙ`. Optional `m` maps the coefficient ring of `O` into `R`. | Basis-image homomorphism construction. |
| `IsRingHomomorphism(m)` | Return whether the vector space homomorphism `m` is also a ring homomorphism. | Multiplicativity check. |

*Worked examples: H37E7 (embedding `Q(√2)` into `Q(√2+√3)`; homomorphism to a finite field).*

---

## 37.3 Special Options

Options controlling verbose output (from the underlying KANT package), output style, and the precision of internal real computations.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SetVerbose(s, n)` | Set verbose flag `s` (e.g. `"MaximalOrder"`, `"ClassGroup"`, `"UnitGroup"`) to level `n`. Verbose output comes from the KANT subsystem and can be technical at higher levels. | — |
| `SetKantPrinting(f)` | If `f` is `true`, activate Kant-style printing (integers/rationals as integers/rationals, more readable in relative extensions, but output cannot be pasted back). If `false`, deactivate. | — |
| `SetKantPrecision(n)` / `SetKantPrecision(O, n)` / `SetKantPrecision(O, n, m)` / `SetKantPrecision(F, n)` / `SetKantPrecision(F, n, m)` | Set the internal real precision (in decimal digits) for the field/order `F`/`O` (or globally if no `F`/`O` given). Default is `max(P, 20, 4·deg)` with `P = 52`. Unit computations use a second ring at twice the ordinary precision (controlled by the third argument). A third independent precision governs root computation. Functions generally auto-increase precision as needed. | — |

---

## 37.4 Structure Operations

### 37.4.1 General Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Category(F)` / `Parent(F)` / `Category(O)` / `Parent(O)` | Return the Magma category or notional parent power-structure of the field/order. | — |
| `AssignNames(~K, s)` | Procedure: change the names of generating elements in number field `K` to the strings in sequence `s`. The `i`-th element names the generator of the `(i-1)`-st subfield down from `K`. Modifies `K` in place (requires `~K`). | — |
| `Name(K, i)` / `K . i` | Return the element of `K` with the `i`-th name; the generator of the `(i-1)`-st subfield, or the root of the `i`-th polynomial (for multi-polynomial absolute extensions). `1 ≤ i ≤ m`. | — |
| `AssignNames(~F, s)` | Assign strings in `s` to the names of the basis elements of the field of fractions `F`. | — |
| `F . i` / `Name(F, i)` | Return the `i`-th basis element of field of fractions `F`. | — |
| `O . i` | Return the `i`-th basis element of order `O`. | — |

### 37.4.2 Related Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `GroundField(F)` / `BaseField(F)` / `CoefficientField(F)` / `CoefficientRing(F)` | Return the algebraic field over which `F` was defined; returns **Q** for an absolute number field. | — |
| `BaseRing(O)` / `CoefficientRing(O)` | Return the order over which `O` was defined; returns **Z** for an absolute order. | — |
| `AbsoluteField(F)` | Return an isomorphic number field defined as an absolute extension over **Q**. | Constructive primitive element theorem. |
| `AbsoluteOrder(O)` | Return an isomorphic order defined as an absolute order over **Z**. | As above. |
| `SimpleExtension(F)` / `SimpleExtension(O)` | Return an isomorphic field defined as a simple (single polynomial) absolute extension, or the corresponding isomorphic order. | Primitive element theorem (non-destructive). |
| `RelativeField(F, L)` | Given algebraic fields `L` and `F` where Magma knows `F` is a subfield of `L`, return an isomorphic field `M` defined as an extension over `F`. | Relative polynomial extraction. |
| `Simplify(O)` | Given an order `O` obtained by a chain of transformations from an equation order `E`, return an order given by a single transformation over `E`. | Composition of transformation matrices. |
| `LLL(O)` | Given order `O`, return an order `O'` obtained by an LLL-reduced transformation matrix `T` (also returned as second value). `O'` has an LLL-reduced basis. | **LLL lattice basis reduction**. |
| `PrimeRing(F)` / `PrimeField(F)` / `PrimeRing(O)` | Generic ring functions. | — |
| `Centre(F)` / `Centre(O)` | The centre of the field or order (equals the field/order itself as these are commutative). | — |
| `Embed(F, L, a)` | Install the embedding of simple field `F` in `L` sending the primitive element of `F` to element `a ∈ L`. Used for coercion. | — |
| `Embed(F, L, a)` | Install the embedding of non-simple field `F` in `L` where the sequence `a` gives images of the generating elements. | — |
| `EmbeddingMap(F, L)` | Return the embedding map of `F` in `L` if an embedding is known. | — |
| `Lattice(O)` / `MinkowskiLattice(O)` | Given an absolute order `O`, return the lattice determined by the real and complex embeddings of `O`. | Minkowski embedding. |
| `MinkowskiSpace(F)` | The Minkowski vector space `V` of the absolute field `F` as a real vector space with inner product given by the T₂-norm, plus the embedding `F → V`. | Minkowski embedding. |
| `Completion(K, P)` / `Completion(O, P)` / `comp< K\|P >` / `comp< O\|P >` | For absolute extension `K`/`O`, compute the completion at prime ideal `P` (maximal order prime or unramified). Returns a local field/ring with precision `Precision` (default 50) or `e·Precision` for ramification degree `e`. Also returns the canonical injection map (with pointwise inverse). | p-adic completion. |
| `Completion(K, P)` | For absolute extension `K` over **Q** and a finite place `P`, compute the completion at `P`. Parameter: `Precision` (default 50). | p-adic completion. |
| `LocalRing(P, prec)` | The completion of `Order(P)` at the prime ideal `P` up to precision `prec`. | p-adic completion. |

*Worked examples: H37E8 (constructing composite fields `Q(√2,√3,√5)` via `AbsoluteField`, `SimpleExtension`), H37E9 (LLL-reduced basis of a quartic maximal order; effect of `Simplify`).*

### 37.4.3 Representing Fields as Vector Spaces

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Algebra(K, J)` / `Algebra(K, J, S)` | Return the associative structure constant algebra isomorphic to algebraic field `K` as an algebra over `J`, and the isomorphism `K → Algebra`. If sequence `S` is given, it is used as a basis of `K` over `J`. | Structure constant algebra from basis. |
| `VectorSpace(K, J)` / `KSpace(K, J)` / `VectorSpace(K, J, S)` / `KSpace(K, J, S)` | Return the vector space isomorphic to `K` over `J` and the isomorphism `K → VectorSpace`. If `S` is given, it is used as a basis. | Vector space from basis. |

*Worked example: H37E11 (using `Algebra` to compute the minimal polynomial of an element of a relative field over a non-tower subfield).*

### 37.4.4 Invariants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Characteristic(F)` / `Characteristic(O)` | Always returns 0 for number fields and orders. | — |
| `Degree(O)` / `Degree(F)` | The relative degree `[F : G]` (resp. `[O : ground order]`). | — |
| `AbsoluteDegree(O)` / `AbsoluteDegree(F)` | The absolute degree over **Z** (resp. **Q**). | — |
| `Discriminant(O)` / `Discriminant(F)` | The discriminant of `F` (discriminant of the equation order or field-of-fractions order). For absolute orders/fields: an integer. For relative: an ideal of the base ring. Can only be computed when the coefficient ring is maximal or `O` has a power basis. | Trace-matrix determinant. |
| `AbsoluteDiscriminant(O)` / `AbsoluteDiscriminant(K)` | Absolute value of the discriminant of `O` or `K` regarded over **Z** / **Q**. | As above. |
| `ReducedDiscriminant(O)` / `ReducedDiscriminant(F)` | The maximal elementary divisor (elementary ideal) of the torsion module `O#/O` where `O#` is the dual w.r.t. the trace form. For absolute extensions: largest entry of the Smith normal form of `TraceMatrix`. For extensions with a power basis: a generator of the inverse of the ideal generated by cofactors `X, Y` of `Xf + Yf' = 1`. | Smith normal form / extended gcd of `f` and `f'`. |
| `Regulator(O: parameters)` / `Regulator(K)` | Return the regulator of `K` or `O` as a real number. Triggers maximal order and unit group computation if not known. Only for absolute extensions. Parameter: `Current` (default `false`; if `true` and independent units are known, return their regulator without seeking fundamental units). | Unit group computation; then regulator as determinant of log-embedding matrix. |
| `RegulatorLowerBound(O)` / `RegulatorLowerBound(K)` | A lower bound on the regulator of `O` or `K`. Only for absolute extensions. | — |
| `Signature(O)` / `Signature(F)` | Returns `(r₁, r₂)`: number of real embeddings and number of pairs of complex embeddings. | Root analysis of defining polynomial. |
| `UnitRank(O)` / `UnitRank(K)` | The unit rank `r₁ + r₂ - 1` (Dirichlet's theorem). | From signature. |
| `Index(O, S)` | The module index of order `S` in order `O` (for `S ⊆ O` with the same equation order). | Determinant of transformation matrix. |
| `DefiningPolynomial(F)` / `DefiningPolynomial(O)` | The polynomial defining `F` over its ground field (or `O` over its coefficient ring). For non-simple extensions: a list of polynomials. | — |
| `Zeroes(O, n)` / `Zeros(O, n)` / `Zeroes(F, n)` / `Zeros(F, n)` | Zeros of the defining polynomial of `F` (or `O`) to exactly `n` decimal digits of precision, as a sequence of length `deg(F)` (real zeros first). | Numerical root finding. |
| `Different(O)` | The different of the maximal order `O ⊆ K`: the inverse ideal of `{x ∈ K : Tr(xO) ⊆ O}`. | Inverse of the codifferent (trace form dual). |
| `Conductor(O)` | The conductor of order `O`: the largest ideal of the maximal order `M` still contained in `O`, i.e. `{x ∈ M : xM ⊆ O}`. | — |

*Worked example: H37E12 (`Zeros` and `DefiningPolynomial` for fields defined by `x⁶+108` and `x³-2`).*

### 37.4.5 Basis Representation

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Basis(O)` / `Basis(O, R)` / `Basis(F)` / `Basis(F, R)` | Return the basis of `O` or `F` over its ground ring, as elements of its field of fractions (or of `R` if given). | — |
| `IntegralBasis(F)` / `IntegralBasis(F, R)` | An integral basis for `F` (the basis for the maximal order). Triggers maximal order computation if needed. | Computes and stores `MaximalOrder(F)`. |
| `AbsoluteBasis(K)` | An absolute basis for the algebraic field `K` as a **Q**-vector space, using products of basis elements of intermediate fields (depth-first). | Tensor product of intermediate bases. |
| `BasisMatrix(O)` | For order `O` in number field `K` of degree `n`, returns an `n × n` matrix whose `i`-th row gives the rational coefficients of the `i`-th basis element of `O` w.r.t. the power basis of `K`. | Direct computation. |
| `TransformationMatrix(O, P)` | Returns the `n × n` integer transformation matrix `T` (and a common integer denominator) expressing the basis elements of `O` in terms of the basis elements of `P` (for orders with common equation order). | Matrix computation from bases. |
| `CoefficientIdeals(O)` | The coefficient ideals `{Aᵢ}` of order `O` in a relative extension, such that every element `e ∈ O` satisfies `e = ∑ aᵢ bᵢ` with `aᵢ ∈ Aᵢ` and `{bᵢ}` the basis. | From pseudo-basis. |
| `MultiplicationTable(O)` | For order `O` of degree `n`, returns a sequence of `n` matrices of size `n × n`; the `i`-th matrix has `j`-th row = basis representation of `bᵢbⱼ`. | Direct basis arithmetic. |
| `TraceMatrix(O)` / `TraceMatrix(F)` | The trace matrix with `(i,j)` entry `Tr(ωᵢωⱼ)` for the basis `{ωᵢ}` of `O` or `F`. | Direct trace computation. |

*Worked examples: H37E13 (basis expressed in different rings), H37E14 (`BasisMatrix`, `IntegralBasis` for degree-4 field; transformation between bases), H37E15 (`MultiplicationTable`, `TraceMatrix` for the same field).*

### 37.4.6 Ring Predicates

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `N eq O` | Two orders are equal iff the transformation matrix between them is integral with determinant ±1 (same number field required). | Transformation matrix check. |
| `F eq L` | Two algebraic fields are equal iff they are the same Magma object (independently created fields are never equal). | Identity check. |
| `IsCommutative(R)` / `IsUnitary(R)` / `IsFinite(R)` / `IsOrdered(R)` / `IsField(R)` / `IsNumberField(R)` / `IsAlgebraicField(R)` | Generic ring predicates. | — |
| `IsEuclideanDomain(F)` | Always returns an error (not a check for Euclidean number fields). | — |
| `IsSimple(F)` / `IsSimple(O)` | Returns `true` if `F` or `O` is defined as a simple extension over its base ring. | — |
| `IsPID(F)` / `IsUFD(F)` / `IsPrincipalIdealRing(F)` | Always `true` for fields. | — |
| `IsPID(O)` / `IsUFD(O)` / `IsPrincipalIdealRing(O)` | Always `false` for orders (even class number 1 orders are not treated as PIDs). | — |
| `IsDomain(R)` / `F ne L` / `O ne N` / `O subset P` / `K subset L` | Generic predicates. | — |
| `HasComplexConjugate(K)` | Returns `true` if there is an automorphism of `K` acting as complex conjugation. | — |
| `ComplexConjugate(x)` | For element `x` of a field `K` where `HasComplexConjugate` returns `true` (includes totally real, cyclotomic, quadratic, and CM fields): the conjugate of `x`. | Evaluation of conjugation automorphism. |

### 37.4.7 Order Predicates

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsEquationOrder(O)` | Returns `true` if the basis of `O` is an integral power basis. | — |
| `IsMaximal(O)` | Returns `true` if `O` is the maximal order of its field. **Warning:** may trigger maximal order computation. | Comparison with `MaximalOrder`; potentially Round-2/4. |
| `IsAbsoluteOrder(O)` | Returns `true` iff `O` is constructed as an absolute extension of **Z**. | — |
| `IsWildlyRamified(O)` | Returns `true` iff there exists a prime ideal `P` of `O` whose residue characteristic divides its ramification index. | Discriminant/prime ideal analysis. |
| `IsTamelyRamified(O)` | Returns `true` iff no prime ideal of `O` is wildly ramified. | As above. |
| `IsUnramified(O)` | Returns `true` iff `O` is unramified at all finite places. | As above. |

### 37.4.8 Field Predicates

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsIsomorphic(F, L)` | Returns `true` and an isomorphism `F → L` if `F` and `L` are isomorphic; `false` otherwise. | Field isomorphism test. |
| `IsSubfield(F, L)` | Returns `true` and an embedding `F ↪ L` if `F` is a subfield of `L`; `false` otherwise. | Subfield test. |
| `IsNormal(F)` | Returns `true` iff `F` is a normal extension. Only for absolute or simple relative extensions; in the relative case uses Galois group computation. | Galois group computation (relative case). |
| `IsAbelian(F)` | Returns `true` iff `F` is a normal extension with abelian Galois group. Absolute or simple relative. Relative case via Galois group computation. | Galois group computation. |
| `IsCyclic(F)` | Returns `true` iff `F` is normal with cyclic Galois group. Relative case via Galois and automorphism group computation. | Galois/automorphism group computation. |
| `IsAbsoluteField(K)` | Returns `true` iff `K` is constructed as an absolute extension of **Q**. | — |
| `IsWildlyRamified(K)` / `IsTamelyRamified(K)` / `IsUnramified(K)` | Field-level ramification predicates, applied to the maximal order. | Discriminant/prime ideal analysis. |
| `IsQuadratic(K)` | If `K` is quadratic, returns `true` and an isomorphic quadratic field. | Degree check. |
| `IsTotallyReal(K)` | Returns `true` iff all infinite places are real (i.e. the defining polynomial has only real roots for absolute fields). | Root analysis. |

### 37.4.9 Setting Properties of Orders

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SetOrderMaximal(O, b)` | Mark `O` as maximal (`b = true`) or known non-maximal (`b = false`). | Sets an internal flag. |
| `SetOrderTorsionUnit(O, e, r)` | Set the torsion unit of `O` to element `e` with order `r`. | Sets an internal attribute. |
| `SetOrderUnitsAreFundamental(O)` | Mark the currently known units of `O` as fundamental. | Sets an internal flag. |

---

## 37.5 Element Operations

### 37.5.1 Parent and Category

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Parent(a)` / `Parent(w)` / `Category(a)` / `Category(w)` | Generic element functions returning the parent structure or Magma category of an element. | — |

### 37.5.2 Arithmetic

Binary operations apply to both field elements and order elements. Division `a / b` of two order elements returns an element in the field of fractions. Negative exponents yield field-of-fractions elements.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `+ a` / `- a` / `a + b` / `a - b` / `a * b` / `a / b` / `a ^ k` | Standard arithmetic on algebraic field and order elements. Division or negative exponent yields a field-of-fractions element. | Field arithmetic in the quotient ring. |
| `w div v` | Exact quotient of order elements `w` and `v` (same order); `v` must divide `w` exactly. | Division in the order. |
| `Modexp(a, n, m)` | Modular power: `aⁿ mod m` of the order element `a`, for non-negative integer `n` and integer `m > 1`. | Repeated squaring mod `m`. |
| `Sqrt(a)` / `SquareRoot(a)` | Square root of element `a` if it exists in the containing order or field. | Root extraction. |
| `Root(a, n)` | The `n`-th root of element `a` if it exists in the containing order or field. | Root extraction. |
| `IsPower(a, k)` / `IsSquare(a)` | Returns `true` if `a` is a `k`-th power (resp. square) and returns the root if so. | Root extraction with existence check. |
| `Denominator(a)` | The least common multiple of the denominators of the coefficients of element `a`. | Coefficient analysis. |
| `Numerator(a)` | The element `a` multiplied by its denominator. | Scaling. |
| `Qround(E, M)` | Find an approximation to the field element `E` with denominator bounded by integer `M`. Parameter: `ContFrac` (default `true`; uses continued fraction algorithm on the rational coefficients). | Continued fraction / rational approximation. |

### 37.5.3 Equality and Membership

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `a eq b` / `a ne b` | Equality and inequality of elements. | Coefficient comparison. |
| `a in F` | Test whether element `a` is in field `F`. | Coercion check. |

### 37.5.4 Predicates on Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsIntegral(a)` | Returns `true` if the algebraic field element `a` is integral (lies in the ring of integers). Uses the minimal polynomial (does not trigger maximal order computation). Also returns a denominator `d` such that `d·a` is integral. Vacuously `true` for order elements. | Minimal polynomial integrality check. |
| `IsPrimitive(a)` | Returns `true` if element `a` of algebraic field `F` (or an order) generates `F`. | Minimal polynomial degree check. |
| `IsTorsionUnit(w)` | Returns `true` iff the order element `w` is a unit of finite order. | — |
| `IsPower(w, n)` | For `w ∈ O` and integer `n > 1`, returns `true` iff there exists `v ∈ O` with `w = vⁿ`; if true, also returns `v`. | Root computation in the order. |
| `IsTotallyPositive(a)` / `IsTotallyPositive(a)` | Returns `true` iff all real embeddings of element `a` are positive. | Conjugate sign check. |
| `IsZero(a)` / `IsOne(a)` / `IsMinusOne(a)` / `IsUnit(a)` / `IsNilpotent(a)` / `IsIdempotent(a)` / `IsZeroDivisor(a)` / `IsRegular(a)` / `IsIrreducible(a)` / `IsPrime(a)` | Generic element predicates. | — |

### 37.5.5 Finding Special Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `K . 1` | The image `α` of `x` in `G[x]/f`, the first generator of `K`. Primitive element for simple extensions. | Structural look-up. |
| `PrimitiveElement(K)` / `PrimitiveElement(F)` | Returns a primitive element of the simple algebraic field (= `K.1` for number fields, `F!K.1` for fields of fractions). For non-simple fields, a random element is returned. | — |
| `Generators(K)` | List of generators of `K` over its coefficient field: roots of each defining polynomial. | — |
| `Generators(K, k)` | List of generators of `K` over `k`: roots of each defining polynomial for `K` and all intermediate fields down to `k`. | — |
| `PrimitiveElement(O)` | A primitive element for the field of fractions of order `O` (element whose minimal polynomial has the same degree as the field). | — |

### 37.5.6 Real and Complex Valued Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AbsoluteValues(a)` | Sequence of length `r₁ + r₂` of real absolute values of the conjugates of element `a`. First `r₁` are absolute values of real embeddings; next `r₂` are lengths `√(xᵢ² + yᵢ²)/2` of complex pairs. | Numerical complex evaluation. |
| `AbsoluteLogarithmicHeight(a)` | The absolute logarithmic height `h(α) = (1/n) log(a₀ ∏ⱼ max(1, |αⱼ|))` where `n = deg(MinPoly)`, `a₀` is the leading coefficient, and `αⱼ` are the roots. | Logarithmic Mahler measure. |
| `Conjugates(a)` | The `n` real and complex conjugates of the algebraic number `a`, as a sequence of complex numbers. Real conjugates appear first, followed by `r₂` pairs. Ordering consistent across elements of the same field. | Numerical root evaluation. |
| `Conjugate(a, k)` | Equivalent to `Conjugates(a)[k]`. | As above. |
| `Conjugate(a, l)` | For a tower `Q ⊆ K₁ ⊆ … ⊆ Kₙ = K` and a sequence `l = [l₁,…,lₙ]`, compute the image of `a` under the embedding obtained by extending the `lᵢ`-th embedding at each level. | Iterated conjugate evaluation. |
| `Length(a)` | The T₂-norm of element `a`: the sum of the complex norms of all conjugates. | Sum of `|conjugates|²`. |
| `Logs(a)` | Sequence of length `r₁ + r₂` of logarithms of absolute values of the conjugates of `a ≠ 0`. | Logarithm of `AbsoluteValues`. |
| `CoefficientHeight(E)` / `CoefficientHeight(E)` | The coefficient height: for absolute-field elements, the maximum of the denominator and the largest coefficient w.r.t. the basis; for relative extensions, the maximum height of coefficient heights. | Coefficient analysis. |
| `CoefficientLength(E)` / `CoefficientLength(E)` | The coefficient length: for absolute-field elements, the sum of the denominator and absolute values of all coefficients; for relative, the sum of the lengths of coefficients. | Coefficient analysis. |

*Worked example: H37E16 (alternative discriminant function via `Conjugates`; alternative T₂-norm via `Norm(Conjugates(a)[i])`).*

### 37.5.7 Norm, Trace, and Minimal Polynomial

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Norm(a)` / `Norm(a, R)` | The relative norm `N_{L/F}(a)` of element `a` of `L` over `F` (the ground field or order). If `R` is given, compute the norm over `R` (must appear somewhere in the tower under `L`). | Product of conjugates (multiplication of companion matrix eigenvalues). |
| `AbsoluteNorm(a)` / `NormAbs(a)` | The absolute norm `N_{L/Q}(a)`. | As above, over **Q**. |
| `Trace(a)` / `Trace(a, R)` | The relative trace `Tr_{L/F}(a)`. Optional `R` as for `Norm`. | Sum of conjugates / trace of multiplication matrix. |
| `AbsoluteTrace(a)` / `TraceAbs(a)` | The absolute trace `Tr_{L/Q}(a)`. | As above, over **Q**. |
| `CharacteristicPolynomial(a)` / `CharacteristicPolynomial(a, R)` | The characteristic polynomial of element `a` over `F` (or `R`). | Determinant of `xI - M(a)` where `M(a)` is the multiplication matrix. |
| `AbsoluteCharacteristicPolynomial(a)` | The characteristic polynomial with rational (resp. integer) coefficients for field (resp. order) elements. | Over **Q** / **Z**. |
| `MinimalPolynomial(a)` / `MinimalPolynomial(a, R)` | The minimal polynomial of `a` over `F` (or `R`). | GCD of characteristic polynomial and its derivatives. |
| `AbsoluteMinimalPolynomial(a)` | The minimal polynomial with coefficients in **Q** (field elements) or **Z** (order elements). | Over **Q** / **Z**. |
| `RepresentationMatrix(a)` / `RepresentationMatrix(a, R)` | The matrix of the multiplication-by-`a` map w.r.t. the basis (of the order or field). The `i`-th row gives the coefficients of `a·wᵢ` w.r.t. the basis `w₁,…,wₙ`. Optional `R` for a different base ring. | Direct multiplication in the basis. |
| `AbsoluteRepresentationMatrix(a)` | The representation matrix w.r.t. the **Q**-basis of the relative number field (products of basis elements of the different levels). | Tensor product basis representation. |

*Worked example: H37E17 (norm, trace, minimal polynomial, representation matrix for `y/2` in a quartic field).*

### 37.5.8 Other Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ElementToSequence(a)` / `Eltseq(a)` | For algebraic field element `a`, returns the coefficient sequence w.r.t. the basis (universe is always a field). For order element, the coefficients w.r.t. the order basis. | Coefficient extraction. |
| `Eltseq(E, k)` | For `E ∈ K` and ring `k` in the tower for `K`, return the list of coefficients of `E` over `k` (iterate `Eltseq` until over `k`). | Iterative `Eltseq`. |
| `Flat(e)` | The coefficients of the algebraic field element `e` w.r.t. the canonical **Q**-basis (iterate `Eltseq` until coefficients are rational). Matches `AbsoluteBasis` for number field elements. | Iterative `Eltseq`. |
| `a[i]` | The coefficient of the `i`-th basis element in the algebraic field or order element `a`. | Indexing. |
| `ProductRepresentation(a)` | Return sequences `P` and `E` such that `∏ P[i]^E[i] = a`. | Factorisation-based representation. |
| `ProductRepresentation(P, E)` / `PowerProduct(P, E)` | Given sequences `P` of elements and `E` of integers, compute `∏ P[i]^E[i]`. | Product with exponents. |
| `Valuation(w, I)` | The valuation `v_I(w)` of element `w` (of an order or algebraic field) at a prime ideal `I`. Non-negative integer. | p-adic valuation. |
| `Decomposition(a)` / `Decomposition(a)` | The factorisation of order/algebraic-field element `a` into prime ideals. | Ideal factorisation. |
| `Divisors(a)` | For `a` in a maximal order, a sequence of elements that divide `a` (generators of all principal ideals returned by `Divisors(Parent(a)*a)`), up to units. | Ideal divisor enumeration. |
| `Index(a)` | The index `[Z[a] : O]` in the order `O` containing `a`, over **Z**. Infinite if `a` is not primitive. | Determinant of companion matrix. |
| `Different(a)` | The different of element `a` of an order of a number field. | Different of the suborder generated by `a`. |

---

## 37.6 Ideal Class Groups

The class group of the maximal order `O` of an absolute number field is computed by the **relation/factor-basis method (index calculus)** **[Heß96, Coh93]**:
1. Generate all prime ideals of norm below a chosen bound (factor basis).
2. Search for elements in those prime ideals whose principal ideals factor completely over the factor basis (relations).
3. Derive generators of the class group via matrix echelonisation.
4. Verify correct orders of generators.

By default, Magma uses the **Minkowski bound** and fully verifies the result. The Bach bound (conditional on GRH) is available for faster computation. For large discriminants (> 10³⁰) a **lattice-sieve / number-field-sieve method** **[Bia]** is used.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DegreeOnePrimeIdeals(O, B)` | Return all prime ideals in `O` of norm a rational prime ≤ `B`. | Decomposition below the bound `B`. |
| `ClassGroup(O: parameters)` / `ClassGroup(K: parameters)` | Return the ideal class group of `O` (or the maximal order of `K`) as an abstract abelian group, plus a map (admitting inverses) from the group to the set of ideals. Parameters: `Bound` (default `MinkowskiBound`), `Proof` (`"Full"` \| `"GRH"` \| `"Bound"` \| `"Subgroup"` \| `"Current"`), `Enum` (default `true`; if `false`, use random linear combinations for large fields), `Al` (`"Automatic"` \| `"Sieve"` \| `"NoSieve"`), `Verbose ClassGroup` (max 5), `Verbose ClassGroupSieve` (max 5). | Index calculus (relation method) **[Heß96, Coh93]**; lattice sieve **[Bia]** for `disc > 10³⁰` or `Al := "Sieve"`. |
| `RingClassGroup(O)` / `PicardGroup(O)` | For a (possibly non-maximal) order `O`, compute the ring class group (Picard group): the group of invertible ideals modulo principal ideals. | Algorithm of Klüners and Pauli **[PK05]**. |
| `ConditionalClassGroup(O)` / `ConditionalClassGroup(K)` | The class group assuming GRH (uses the Bach bound). | Index calculus with Bach bound. |
| `ClassGroupPrimeRepresentatives(O, I)` | For maximal order `O` of an absolute number field and ideal `I`, compute a set of prime ideals coprime to `I` representing all ideal classes; returns the map from class group to primes. | Class group discrete logarithm. |
| `ClassNumber(O: parameters)` / `ClassNumber(K: parameters)` | Return the class number. Same parameters as `ClassGroup`. | Order of the group returned by `ClassGroup`. |
| `BachBound(K)` / `BachBound(O)` | An integral GRH-conditional upper bound for norms of class-group generators. | Analytic number theory (GRH). |
| `MinkowskiBound(K)` / `MinkowskiBound(O)` | An unconditional integral upper bound for norms of class-group generators. | Minkowski's theorem. |
| `FactorBasis(K, B)` / `FactorBasis(O, B)` | Return a sequence of prime ideals of norm < `B` (factor basis for bound `B`). | Decomposition below `B`. |
| `FactorBasis(O)` | Return the factor basis used in the last class group computation of `O`, plus the effective upper bound used. | Cached from prior `ClassGroup` call. |
| `RelationMatrix(K, B)` / `RelationMatrix(O, B)` | Generate relations for each prime ideal in the factor basis with bound `B`; return as rows of a matrix. Stops early if relations generate the trivial group. | Relation search over the factor basis. |
| `RelationMatrix(O)` | Return the relation matrix from the last class group computation of `O`. | Cached result. |
| `Relations(O)` | Return the vector of order elements used to compute the class group (the elements whose principal ideals gave the relations). | Cached result. |
| `ClassGroupCyclicFactorGenerators(O)` | Let `aᵢ` be the generators for the cyclic factors of the class group of `O` with orders `cᵢ`. Return the generators `aᵢ^cᵢ` (i.e. generators of the principal part of each cyclic factor). | From class group structure. |
| `FactorBasisCreate(O, B)` | Create a class group process by computing a factor basis containing all ideals of norm ≤ `B` in `O`. | Factor basis initialisation. |
| `EulerProduct(O, B)` | Compute an approximation to the Euler product for `O` using prime ideals of norm ≤ `B`. | Analytic number theory. |
| `AddRelation(E)` | Add a relation (order element `E`) to the class group process of the parent of `E`. Returns `true` iff the element factors over the factor basis and the new relation matrix has full rank. | Relation matrix update. |
| `EvaluateClassGroup(O)` | Finalise the class group process for `O`: compute the class group structure from the current relation matrix and unit group (null space). Returns `true` when determined. Requires prior `FactorBasisCreate` and sufficient relations. | Smith normal form of relation matrix. |
| `CompleteClassGroup(O)` | Complete an already-started class group process for `O` by automatically seeking relations until the class group is determined. | Automated relation search then `EvaluateClassGroup`. |
| `FactorBasisVerify(O, L, U)` | Verify "completeness" of the current factor basis: for all prime ideals of norm between `L` and `U`, attempt to find a relation; success means the new prime does not add to the class group. Does not return if unsuccessful. | Relation search in the given norm range. |
| `ClassGroupSetUseMemory(O, f)` | For `O` with class group computed, decide (by flag `f`) whether discrete logarithm results are cached. Helps when `O` is used repeatedly as a coefficient ring (e.g. for ray class groups). Disabled by default. | Memory/time trade-off control. |
| `ClassGroupGetUseMemory(O)` | Return whether discrete logarithm values are being cached for `O`. | Query of the memory flag. |

*Worked examples: H37E18 (class group of `Q(√10)`; factor basis, relations, `IsPrincipal`; Minkowski vs Bach bounds), H37E19 (sieve algorithm for large discriminant fields), H37E20 (custom bounds via `SetClassGroupBoundMaps`).*

### 37.6.1 Setting the Class Group Bounds Globally

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SetClassGroupBounds(n)` / `SetClassGroupBounds(string)` | Set both class-group bounds (factor-basis size and checking bound) for all subsequent `ClassGroup` calls. Argument: integer `n` (both bounds set to `n`), or string `"GRH"` (GRH-correct bounds) or `"PARI"` (roughly PARI's default rigor). | Global parameter setting. |
| `SetClassGroupBoundMaps(f1, f2)` | Set the two bounds via maps `f₁, f₂ : PowerStructure(RngOrd) → Z`. `f₁` controls the factor-basis size; `f₂` controls the checking bound. For fields, the bounds for the maximal order are used. | Global parameter setting via maps. |

---

## 37.7 Unit Groups

The unit group is computed by the **relation method** (same infrastructure as class groups), **Dirichlet's method**, mixed methods, and continued fractions for real quadratic fields, following **[PZ89]** (pp. 343–344) and **[Poh93]**. These methods work only for absolute extensions.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `UnitGroup(O)` / `MultiplicativeGroup(O)` / `UnitGroup(K)` / `MultiplicativeGroup(K)` | Return an abstract abelian group `U` and a bijection `m : U → units of O` (or the maximal order of `K`). The group has a torsion part generated by `m(U.1)` and a free part generated by `m(U.i)` for `2 ≤ i ≤ r₁ + r₂`. Parameter: `Al` (`"Automatic"` \| `"ClassGroup"` \| `"Dirichlet"` \| `"Mixed"` \| `"Relation"` \| `"Short"` \| `"ContFrac"` for real quadratic), `Verbose UnitGroup` (max 6). | Dirichlet's method / relation method / continued fractions **[PZ89, Poh93]**. |
| `UnitGroupAsSubgroup(O)` | For a (possibly non-maximal) order `O`, return the unit group of `O` as a subgroup of the unit group of the maximal order. | Algorithm of Klüners and Pauli **[PK05]**. |
| `TorsionUnitGroup(O)` / `TorsionUnitGroup(K)` | The torsion subgroup of the unit group of `O` (or the maximal order of `K`) as an abelian group `T`, with map `m : T → O`. The group is cyclic, generated by `m(T.1)`. | Roots of unity computation. |
| `IndependentUnits(O)` / `IndependentUnits(K)` | Return a sequence of independent units generating a subgroup of finite index in the full unit group, as an abelian group plus a homomorphism to `O`. Parameters: `Al` (same as `UnitGroup`), `Verbose UnitGroup` (max 6). | Dirichlet / relation / mixed **[PZ89, Poh93]**. |
| `pFundamentalUnits(O, p)` / `pFundamentalUnits(K, p)` | Return an abstract abelian group `U` and map `m : U → O` such that `U` is a subgroup of the unit group `G` with `p ∤ (G : U)`, where `p` is the given prime. Parameters: `Al`, `Verbose UnitGroup` (max 6). | Unit group computation avoiding `p` in the index. |
| `MergeUnits(K, a)` / `MergeUnits(O, a)` | For a unit `a ∈ O`, add it to the already-known unit subgroup stored in `O`. Returns `true` iff the rank of the known unit group increases. Parameter: `Verbose UnitGroup` (max 6). | Unit subgroup augmentation. |
| `UnitRank(O)` / `UnitRank(K)` | Return the unit rank `r₁ + r₂ - 1` of the ring of integers `O` or of `K`. | Dirichlet's theorem from signature. |
| `IsExceptionalUnit(u)` | Returns `true` iff `u` is an exceptional unit of its order (both `u` and `u - 1` are units). | Unit and divisibility test. |
| `ExceptionalUnitOrbit(u)` | For an exceptional unit `u`, return the orbit `Ω(u) = {u, 1/u, 1-u, 1/(1-u), (u-1)/u, u/(u-1)}` (usually 6 elements). | Direct evaluation of 6 transformations. |
| `ExceptionalUnits(O)` | Return a sequence `S` such that every exceptional unit of `O` either belongs to `S` or to the orbit of some element of `S`. Parameter: `Verbose UnitEq` (max 5). | Unit group enumeration. |

*Worked example: H37E21 (class group, unit group, and torsion unit group of `Q(α)` where `α⁴ - 420α² + 40000 = 0`).*

---

## 37.8 Solving Equations

### 37.8.1 Norm Equations

Magma distinguishes two types: **Diophantine** (solution in a specific order; finite number of solutions up to equivalence) and **field-theoretic** (any element of the field with the required norm; no finiteness guarantee without S-unit structure). Different algorithms are used for each.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `NormEquation(O, m)` | **Diophantine norm equation.** Given order `O` and element `m` of the ground ring, find elements `α ∈ O` with `N_{F/L}(α) = m`. Returns a boolean and a sequence of ≤ `Solutions` solutions. Parameters: `All` (default `true`), `Solutions` (max number), `Exact` (default `false`; if `true`, also requires correct sign of norm), `Ineq` (if `true` and absolute, find all `x` with `|N(x)| ≤ m` via Fincke's ellipsoid method **[Fin84, PZ89]**). For absolute maximal orders: ideal enumeration + principality test (class group). For non-maximal absolute orders: solve in maximal order then adjust by unit cosets. For imaginary quadratic fields: positive definite quadratic form methods. For relative maximal orders (1 solution): S-unit approach then simplex. For relative (all solutions): lattice methods **[Fie97, Jur93, FJP97]**. | Ideal enumeration + principality test; Fincke's ellipsoid **[Fin84, PZ89]**; S-unit approach **[Fie97, Sim02, Gar80]**; lattice methods **[Fie97, Jur93, FJP97]**. |
| `NormEquation(F, m)` | **Field-theoretic norm equation.** Given field `F` and element `m` of the base field, determine whether `α ∈ F` exists with `N_{F/L}(α) = m`; return boolean and a length-1 sequence if true. Parameters: `Primes` (sequence of prime ideals supplementing S), `Nice` (default `true`; if `true`, apply LLL to find a smaller solution). Uses S-units: determine primes S, compute S-unit basis, compute norm map on S-unit group, solve as preimage under norm map. | S-unit method **[Coh00 §7.5, Fie97, Sim02, Gar80]**; LLL size reduction if `Nice := true`. |
| `NormEquation(m, N)` | Given map `N` on the multiplicative group of a number field and element `m`, find a preimage under `N` as an S-unit. Parameters: `Raw` (return unevaluated power product), `Primes`. Main use: Galois-theoretical constructions where `N` is the product over a fixed Galois subgroup. | S-unit preimage computation; preimage in abstract abelian group if `Raw := true`. |
| `IntegralNormEquation(a, N, O)` | For integer or unit `a`, multiplicative function `N` on the field of fractions of order `O`, find a unit in `O` that is a preimage of `a` under `N`. `N\|_O` must be an endomorphism of `O`. Parameter: `Nice` (default `true`; size-reduce solution). More efficient than lattice-based approach when the unit index `(Z_k)* : O*` is large. | S-unit preimage in the order. |
| `SimNEQ(K, e, f)` | Simultaneous norm equations `N_{K/k₁}(x) = e` and `N_{K/k₂}(x) = f` for `e ∈ k₁` and `f ∈ k₂`. Finds a likely S (including ramified primes, support of `e` and `f`, and class-group generators of `K`). Parameters: `S` (additional prime ideals), `HasSolution` (default `false`; if `true`, add primes until a solution is found). For normal `K/Q` the initial S suffices if a solution exists. | S-unit intersection for simultaneous equations. |

*Worked example: H37E22 (norm equation in a relative quadratic extension; the map-based approach via `AutomorphismGroup`; effect of `Raw` parameter).*

### 37.8.2 Thue Equations

Thue equations have the form `f(x, y) = k` with `f` a homogeneous polynomial in two variables. A `Thue` object stores the necessary data for the solver.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Thue(f)` | Create a Thue object from a polynomial `f` of degree ≥ 2 over **Z**. The Thue object is printed as the homogeneous form of `f`. | Stores polynomial data. |
| `Thue(O)` | Create the Thue object corresponding to the defining polynomial of the order `O` (coefficient ring **Z**). | Extracts defining polynomial. |
| `Evaluate(t, a, b)` / `Evaluate(t, S)` | Evaluate the homogeneous polynomial `f` involved in Thue object `t` at `(a, b)` (or `S = [a, b]`). | Polynomial evaluation. |
| `Solutions(t, a)` | Return all integer solutions `[x, y]` of `f(x, y) = a` for Thue object `t`. Parameter: `Exact` (default `true`; if `false`, also find solutions to `f(x,y) = −a`), `Verbose ThueEq` (max 5). | Bilu–Hanrot reduction **[BH96]**. |

*Worked example: H37E23 (`Thue` and `Solutions` for `f = x³ + xy² + y³`, finding all solutions to `f = 47`).*

### 37.8.3 Unit Equations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `UnitEquation(a, b, c)` | Return all 1×2 matrices `(e₁, e₂)` such that `a·e₁ + b·e₂ = c` for number field elements `a, b, c`, where `e₁, e₂` are units in the maximal order. Parameter: `Verbose UnitEq` (max 5). | Wildanger's method **[Wil97, Wil00]**. |

*Worked example: H37E24 (`UnitEquation` for elements of a degree-7 number field).*

### 37.8.4 Index Form Equations

Index form equations have the form `(O : Z[α]) = k` for a given positive integer `k`. For `k = 1` this finds all integral power bases (up to equivalence: `α ~ β` iff `α = ±β + r` for integer `r`). For degree > 4 the field must be normal and an integral power basis must already be known.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IndexFormEquation(O, k)` | For an absolute order `O`, find all (up to equivalence) solutions `α ∈ O` to `(O : Z[α]) = k`. Parameter: `Verbose IndexFormEquation` (max 5). | Wildanger's method **[Wil97, Wil00]** for degree > 4; Gaál–Pethő–Pohst quartic method **[GPP93, GPP96]** for degree 4; Gaál–Schulte cubic method **[GS89]** for degree 3. |

*Worked example: H37E25 (all integral power bases of `x⁴ − 14x³ + 14x² − 14x + 14`).*

---

## 37.9 Ideals and Quotients

Ideals of orders are of type `RngOrdFracIdl` (fractional) or `RngOrdIdl` (integral subtype). Ideals can be taken of orders over **Z** and orders over a maximal order (some functions not available for the latter). Returned elements are in the field of fractions (fractional ideal) or order (integral ideal).

### 37.9.1 Creation of Ideals in Orders

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `x * O` / `O * x` | Create the ideal `xO` for element `x` and order `O`. Returns with integral type if the ideal is integral. | Direct product. |
| `F !! I` / `O !! I` | Convert ideal `I` to fractional type (first form, `F` a compatible field of fractions) or integral type (second form, `O` a compatible order). | Type coercion. |
| `ideal< O \| a1, a2, ..., am >` / `ideal< O \| x >` / `ideal< O \| M, d >` / `ideal< O \| M, I1, ..., In >` | Construct the ideal of order `O` generated by the given elements/matrix/module. A single integer gives a principal ideal. A matrix `M` (or module over an order) can be supplied as a basis; optional denominator `d` or coefficient ideals. Returns with integral type if the result is integral. | Ideal module construction. |

*Worked example: H37E26 (creation of an ideal from an order element; conversion between integral and fractional type).*

### 37.9.2 Invariants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Order(I)` | The order of which the ideal `I` is an ideal. | Structural look-up. |
| `Denominator(I)` | The smallest positive integer `d` such that `d·I` is an integral ideal. | From the ideal basis. |
| `PrimitiveElement(I)` / `UniformizingElement(P)` | A primitive element of ideal `I` (an element in `I` but not in `I²`). For prime ideal `P`, `UniformizingElement` returns the primitive element. | Basis search. |
| `Index(O, I)` | The index of the integral ideal `I` as a submodule of order `O`: `|O/I|`. | Determinant of ideal basis matrix. |
| `Norm(I)` | The norm of fractional ideal `I`: equals the index if integral; extends multiplicatively to fractional ideals (norm of `I⁻¹` = reciprocal). | Index computation. |
| `MinimalInteger(I)` | The least positive integer contained in the ideal `I`. | — |
| `Minimum(I)` | The least positive integer `m` (if integral) or least positive rational `r` (if fractional) contained in `I`. | — |
| `AbsoluteNorm(I)` | The absolute norm of the fractional ideal `I`. Extends multiplicatively. | As `Norm` but absolute. |
| `CoefficientHeight(I)` / `CoefficientHeight(I)` | For ideal `I`: if given via two elements, the max coefficient height of those generators; otherwise the max entry of the basis matrix. | Coefficient analysis. |
| `CoefficientLength(I)` / `CoefficientLength(I)` | For ideal `I`: if given via two elements, the sum of coefficient lengths of generators; otherwise the sum of entries of the basis matrix. | Coefficient analysis. |
| `RamificationIndex(I, p)` / `RamificationDegree(I, p)` | For prime ideal `I` with `p ∈ I`: the max exponent `e` such that `Iᵉ \| pO`. If `p` omitted, taken to be the minimal integer of `I`. | p-adic valuation. |
| `RamificationDegree(I)` / `RamificationIndex(I)` | The relative ramification index of prime ideal `I` over its coefficient ring: the max `e` such that `Iᵉ \| pO` where `p = I ∩ o`. | p-adic valuation. |
| `ResidueClassField(O, I)` / `ResidueClassField(I)` | For prime ideal `I` of `O`, return the finite field `F ≅ O/I` and the map `O → F`. | Quotient ring construction. |
| `Degree(I)` / `InertiaDegree(I)` | For prime ideal `I`: the relative inertia degree `f(I\|p)` — the degree of `O/I` over `o/p` where `p = o ∩ I`. | Residue field degree. |
| `Valuation(I, p)` | The valuation `v_p(I)` of ideal `I` at prime ideal `p`. May be negative for fractional ideals. | Prime factorisation. |
| `Content(I)` | The content of ideal `I`: the maximal ideal of the base ring dividing `I`. | GCD of basis entries. |

*Worked example: H37E27 (properties of a fractional ideal over a relative extension; `Order`, `Denominator`, `PrimitiveElement`, `Norm`, `Minimum`, `Valuation`).*

### 37.9.3 Basis Representation

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Basis(I)` / `Basis(I, R)` | Return a basis for ideal `I` of order `O` in algebraic field `F`, as elements of `F` (fractional) or `O` (integral), or of ring `R` if given. | From the pseudo-basis of the ideal. |
| `BasisMatrix(I)` | The basis matrix for ideal `I`: elements of an ideal basis as rows of rational coefficients w.r.t. the basis of `O` (integers for integral ideals over **Z**, otherwise elements of the coefficient field). | From the ideal basis matrix. |
| `TransformationMatrix(I)` | The transformation matrix for ideal `I` (and a denominator): basis elements of `I` as rows of coefficients w.r.t. the order basis. | From the ideal basis. |
| `CoefficientIdeals(I)` | The coefficient ideals `{Aᵢ}` of ideal `I` in a relative extension, such that every `e ∈ I` satisfies `e = ∑ aᵢ bᵢ` with `aᵢ ∈ Aᵢ` and `{bᵢ}` the basis of `I`. | From the pseudo-basis. |
| `Module(I)` | For ideal `I` in a relative extension, return a Dedekind module over the coefficient ring with the same basis. | Module representation. |

*Worked example: H37E28 (`Basis`, `BasisMatrix`, `TransformationMatrix` for a fractional ideal in a relative extension).*

### 37.9.4 Two-Element Presentations

All ideals of maximal orders can be generated by at most two elements.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Generators(I)` | For fractional ideal `I` of `O`, return a sequence of two elements generating `I`. Elements in the order if integral, otherwise in the field of fractions. | Two-element generation. |
| `TwoElement(I)` | Return two elements of (the field of fractions of) `O` generating the fractional ideal `I` as an ideal. | Two-element normal form. |
| `TwoElementNormal(I)` | For an integral ideal `I` of a maximal order `O` over **Z**, return two elements of `O` forming a two-element normal presentation, plus an integer `g` such that `I` is `g`-normal. | Normal two-element presentation. |

*Worked example: H37E29 (comparing `Generators` and `TwoElement` for an ideal).*

### 37.9.5 Predicates on Ideals

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `I eq J` / `I ne J` / `x in I` / `x notin I` / `I subset J` | Standard set predicates. | Ideal comparison / membership. |
| `IsIntegral(I)` | Returns `true` iff the fractional ideal `I` is integral (denominator = 1). | Denominator check. |
| `IsZero(I)` | Returns `true` iff `I` is the zero ideal. | — |
| `IsOne(I)` | Returns `true` iff `I = 1·O`. | — |
| `IsPrime(I)` | Returns `true` iff `I` is a prime ideal. If integral and not prime, also returns a proper ideal divisor. Currently slow for large-norm ideals. | Primality test via norm and factorisation. |
| `IsPrincipal(I)` | Returns `true` iff the fractional ideal `I` is principal; if so, also returns a generator. May trigger class group computation. Parameter: `Verbose ClassGroup` (max 5). | Class group membership; discrete logarithm. |
| `IsRamified(P)` / `IsRamified(P, O)` | Returns `true` iff `P` (or any prime ideal above `P` in `O`) has ramification index > 1. | Ramification index check. |
| `IsTotallyRamified(P)` / `IsTotallyRamified(P, O)` / `IsTotallyRamified(K)` / `IsTotallyRamified(O)` | Various total-ramification predicates at the ideal, field, or order level. | Ramification index vs field degree. |
| `IsWildlyRamified(P)` / `IsWildlyRamified(P, O)` / `IsTamelyRamified(P)` / `IsTamelyRamified(P, O)` | Wild vs tame ramification predicates. | Ramification index vs residue characteristic. |
| `IsUnramified(P)` / `IsUnramified(P, O)` | Returns `true` iff the ramification index is 1. | — |
| `IsInert(P)` / `IsInert(P, O)` | Returns `true` iff the inertia degree equals the field degree (prime is inert). | Inertia degree check. |
| `IsSplit(P)` / `IsSplit(P, O)` | Returns `true` iff more than one prime ideal lies above `P ∩ base`. | Decomposition count. |
| `IsTotallySplit(P)` / `IsTotallySplit(P, O)` | Returns `true` iff the number of primes above equals the field degree. | Decomposition count vs degree. |

### 37.9.6 Ideal Arithmetic

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `I * J` | Product of (fractional) ideals `I` and `J`. | Module product. |
| `x * I` / `I * x` | Product of ideal `I` and principal ideal `(x)`. | Scaling. |
| `&*L` | Product of all ideals in sequence/set `L`. | Iterated product. |
| `I / J` / `I div J` | Quotient: fractional ideal `K` with `J·K = I`. For integral `I`, `J` dividing `I`: the integral quotient `I/J`. | Inverse times product. |
| `I / x` | Fractional ideal `I/x`. | Scaling by `1/x`. |
| `I + J` | Sum of (fractional) ideals `I` and `J` (generated by sums of elements). | Module sum = GCD for invertible ideals. |
| `I ^ k` | The `k`-th power of ideal `I` (integer `k`). Negative `k` gives fractional type. | Repeated multiplication. |
| `I eq J` | Tests equality of ideals. | Ideal comparison. |
| `I subset J` | Tests containment; for invertible ideals, equivalent to `J \| I`. | Containment check. |
| `E in I` | Tests if element `E` is in ideal `I`. | Membership test. |
| `LCM(I, J)` / `Lcm(I, J)` / `LeastCommonMultiple(I, J)` | The least common multiple of ideals `I` and `J` (both of the same maximal order). Equals `I meet J` for invertible ideals. | Intersection for invertible ideals. |
| `GCD(I, J)` / `Gcd(I, J)` / `GreatestCommonDivisor(I, J)` | The greatest common divisor of ideals `I` and `J` (same maximal order). | Sum of ideals for invertible ideals. |
| `Content(M)` | For a matrix `M` with entries in number field `k`, the GCD of all elements as principal ideals in the maximal order. | GCD of principal ideals. |
| `I meet J` | Intersection of (fractional) ideals `I` and `J`. For maximal order ideals, equals the LCM. | Intersection / LCM. |
| `&meetS` | Intersection of all ideals in sequence/set `S`. | Iterated intersection. |
| `I meet R` / `R meet I` | Intersection of ideal `I` with compatible ring `R`. Returns an ideal of `R`. | Ideal pull-back. |
| `a mod I` | A representative of the element `a` in the quotient `O/I`. | Reduction modulo the ideal. |
| `InverseMod(E, M)` / `Modinv(E, M)` | Find element `y` such that `y·E ≡ 1 mod M` where `M` is an integral ideal or integer. | Extended GCD / CRT. |
| `ColonIdeal(I, J)` / `IdealQuotient(I, J)` | The colon ideal `[I : J] = {x ∈ F : xJ ⊆ I}`. For invertible ideals, equals `I/J`; for general ideals, only `J·ColonIdeal(I,J) ⊆ O` guaranteed. | Module quotient computation. |
| `IntegralSplit(I)` | Given ideal `I`, return an integral ideal `J` and the minimal positive integer `d` such that `I = J/d`. | Denominator extraction. |

### 37.9.7 Roots of Ideals

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Root(I, k)` | Find the `k`-th root of ideal `I` if it exists. | Root via prime factorisation. |
| `IsPower(I, k)` | Returns `true` iff `I` is a `k`-th power; also returns the root. | Prime factorisation check. |
| `SquareRoot(I)` / `Sqrt(I)` | Return the square root of ideal `I` if `I` is a perfect square. | Root via prime factorisation. |
| `IsSquare(I)` | Returns `true` iff `I` is a square; also returns the square root. | Prime factorisation check. |

### 37.9.8 Factorization and Primes

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Decomposition(O, p)` | Decomposition of rational prime `p` (or prime ideal `p` of the coefficient ring) in order `O`: a sequence of tuples `(prime ideal, exponent)`. Parameter: `Verbose IdealDecompose` (max 5). | Factorisation over the residue field. |
| `DecompositionType(O, p)` | Decomposition type of `p` in `O`: a sequence of tuples `(degree of prime ideal, ramification index)`. Parameter: `Verbose IdealDecompose` (max 5). | Factorisation over the residue field. |
| `Factorization(I)` / `Factorisation(I)` | Prime ideal factorisation of `I` in order `O`, as a sequence of 2-tuples. Parameter: `Verbose IdealDecompose` (max 5). | Factorisation via `Decomposition` + aggregation. |
| `Divisors(I)` | Return all ideals dividing `I` (must be of a maximal order). | Enumeration from prime factorisation. |
| `Support(I)` | The set of prime ideals dividing a non-zero ideal `I` of some maximal order. | From prime factorisation. |
| `Support(L)` | For a sequence `L` of ideals in a maximal order (or number-field elements representing principal ideals), return the set of prime ideals dividing at least one. Parameters: `CoprimeOnly` (return a coprime basis rather than prime ideals), `GaloisStable` (if the field is normal, close the result under the Galois action), `UseBernstein` (use Dan Bernstein's asymptotically fast algorithm **[Ber05]** running essentially linearly in `#L`). | Naive or Bernstein's algorithm **[Ber05]**. |
| `CoprimeBasis(L)` | Given a sequence `L` of ideals in a maximal order, construct a coprime basis `C` (every ideal in `L` is a power product of elements of `C`; `C` closed under GCD; elements pairwise coprime). Parameters: `GaloisStable`, `UseBernstein` **[Ber05]**. | Naive (quadratic) or Bernstein **[Ber05]**. |
| `CoprimeBasisInsert(~L, I)` | Enlarge an existing coprime basis in `L` to accommodate ideal `I`. Parameters: `GaloisStable`, `UseBernstein` **[Ber05]**. | Bernstein or naive insertion. |
| `PowerProduct(B, E)` | Given sequence `B` of ideals and sequence `E` of integers, compute `∏ B[i]^E[i]`. | Direct product. |

### 37.9.9 Other Ideal Operations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ChineseRemainderTheorem(I1, I2, e1, e2)` / `ChineseRemainderTheorem(X, M)` / `CRT(I1, I2, e1, e2)` / `CRT(X, M)` | Return element `e ∈ O` such that `e₁ - e ∈ I₁` and `e₂ - e ∈ I₂` (or simultaneously `X[i] - e ∈ M[i]`). | Chinese Remainder Theorem. |
| `CRT(I1, L1, e1, L2)` / `ChineseRemainderTheorem(I1, L1, e1, L2)` | Return element `e ∈ O` such that `e₁ - e ∈ I₁` and the signs of conjugates `L₁[i]` of `e` match those in `L₂`. (`L₁` is a sorted sequence of indices `0 < lᵢ ≤ r₁` of real places.) | CRT with real-embedding sign conditions. |
| `Idempotents(I, J)` | For coprime integral ideals `I` and `J`: return `true` and elements `i ∈ I`, `j ∈ J` with `i + j = 1`. | Extended Euclidean in the order. |
| `CoprimeRepresentative(I, J)` / `MakeCoprime(I, J)` | Given integral ideals `I` and `J` in the same maximal order, find `q` in the field of fractions such that `qI` is coprime to `J`. | Valuation adjustment. |
| `ClassRepresentative(I)` | Return the representative ideal for the ideal class of `I`, given that the class group of the absolute maximal order `O` has been computed. | Class group discrete logarithm. |
| `Lattice(I)` / `MinkowskiLattice(I)` | Given an ideal `I` of an absolute order, return the lattice determined by the real and complex embeddings of `I`. | Minkowski embedding. |
| `Different(I)` | The different of the (possibly fractional) ideal `I` of an order of an algebraic number field. | Trace-form dual. |
| `Codifferent(I)` | The codifferent of ideal `I`. Equals the inverse of the different for ideals of maximal orders. | Inverse of different. |
| `SUnitGroup(I)` / `SUnitGroup(S)` | For prime ideals given as a product ideal `I` or sequence `S`, return the group of S-units (elements `μ` with `v_p(μ) = 0` for all `p ∉ S`) and a map from the group to the field of fractions. Parameter: `Raw` (if `true`, return a sequence `L` of order elements together with the S-unit group as an abstract abelian group `A` mapping into `R^n` via exponent vectors, such that the S-unit group equals `{∏ L[i]^E[i] : E ∈ Image(A)}`), `Verbose ClassGroup` (max 5). | Unit group + prime ideal enlargement; class group computation for S. |
| `SUnitAction(SU, Act, S)` | Given S-unit group description `SU` (map from `SUnitGroup`) and a (multiplicative) map `Act` on the number field, compute the induced endomorphism on the abstract abelian group. Parameter: `Base` (third return value of `SUnitGroup(...:Raw)` if `SU` is the Raw map). | Induced map on abstract group. |
| `SUnitAction(SU, Act, S)` | As above, but `Act` is a sequence of maps. Returns a sequence of endomorphisms. | Induced maps on abstract group. |
| `SUnitDiscLog(SU, x, S)` / `SUnitDiscLog(SU, L, S)` | Solve the discrete logarithm problem in the S-unit group for algebraic number `x` (or list `L`): find the abstract group element corresponding to `x`. Parameter: `Base`. | Discrete log in the finitely generated abelian S-unit group. |

*Worked examples: H37E30 (S-units of `Q(√10)`; `Raw` representation for large fields; combining `SUnitAction` and `SUnitDiscLog` to solve a norm equation), H37E31 (same operations with automorphism action; S-unit discrete log for norm computation).*

### 37.9.10 Quotient Rings

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `quo< O \| I >` / `quo< O \| M >` / `quo< O \| S >` | Create the quotient ring `Q = O/I` of order `O` by integral ideal `I`. The right-hand side may be an ideal, matrix, or sequence. | Quotient ring construction. |
| `UnitGroup(OQ)` / `MultiplicativeGroup(OQ)` | For a quotient `OQ` of an absolute maximal order, return the unit group of the quotient ring as an abelian group and a map. | CRT-based unit group construction. |
| `Modulus(OQ)` | Return the denominator ideal `I` of the quotient ring `OQ = O/I`. | — |
| `OQ ! a` | Coerce element `a` (coercible into `O`) into the quotient `OQ`. | Reduction modulo `I`. |
| `a mod I` | Canonical representative of `a ∈ O` in the quotient `O/I`. | Reduction. |
| `a * b` / `a + b` / `a - b` / `a / b` / `- a` / `a ^ n` | Arithmetic in the quotient ring. | Arithmetic modulo `I`. |
| `a eq b` / `a ne b` / `IsZero(a)` / `IsOne(a)` / `IsMinusOne(a)` / `IsUnit(a)` | Equality and element predicates in the quotient ring. | — |
| `Eltseq(a)` / `ElementToSequence(a)` | Coefficients of the quotient ring element `a` in the field of fractions of the coefficient ring. | Coefficient extraction. |
| `ReconstructionEnvironment(p, k)` / `ReconstructionEnvironment(p, k)` | Initialize a reconstruction environment for ideal `I = pᵏ` (prime ideal `p`, exponent `k`). The returned object of type `RngOrdRecoEnv` is used to reconstruct elements from approximations modulo `pᵏ`. | LLL-based initialisation **[FF00]**. |
| `Reconstruct(x, R)` / `Reconstruct(x, R)` | Given an order element `x` as an approximation modulo `pᵏ` (stored in reconstruction environment `R`), return the unique minimal `f` in the same order with `x - f ∈ pᵏ`. Parameter: `UseDenominator` (default `false`; if `true`, compute a field element rather than a ring element). | LLL-based reconstruction **[FF00]**. |
| `ChangePrecision(~R, k)` | Change the ideal stored in reconstruction environment `R` from `pˡ` to `pᵏ`. | In-place update. |

*Worked examples: H37E32 (creation of a quotient ring from an integral ideal), H37E33 (using `ReconstructionEnvironment` and `Reconstruct` to find roots of a polynomial over a number field via p-adic lifting).*

---

## 37.10 Places and Divisors

Places of a number field `K` are equivalence classes of absolute values. By Ostrowski's theorem they are either **finite** (one-to-one correspondence with nonzero prime ideals of the maximal order) or **infinite** (corresponding to real or pairs of complex embeddings of `K`). The divisor group is the free group generated by the finite places tensored with the **R**-span of the infinite ones. See also Section 34.8.

### 37.10.1 Creation of Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Places(K)` | The set of places of the number field `K`. | — |
| `DivisorGroup(K)` | The group of divisors of number field `K`. | — |

### 37.10.2 Operations on Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `d1 eq d2` / `p1 eq p2` | Equality of divisors / places. | — |
| `NumberField(P)` / `NumberField(D)` | The number field of which `P` is the set of places or `D` is the group of divisors. | — |

### 37.10.3 Creation of Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Place(I)` | The place corresponding to prime ideal `I`. | Finite place from prime ideal. |
| `Decomposition(K, p)` / `Decomposition(K, I)` | A sequence of tuples of places and multiplicities: decomposition of a finite prime `p` (or infinite prime) in the maximal order of `K`. | Prime factorisation. |
| `Decomposition(K, p)` | For number field `K` and place `p` of its coefficient field, compute all places of `K` extending `p` and their multiplicities (ramification indices for finite places; `1` for complex or real-to-real infinite places, `2` for real-to-complex). | Place lifting. |
| `Decomposition(m, p)` / `Decomposition(m, p)` | For extension `K/k` given by embedding map `m : k → K`, decompose the place `p` of `k` in `K`. For `k = Q`: `p` is a prime or `0` for the infinite place. Returns pairs `(place above p via m, ramification index)`. | Place decomposition via embedding. |
| `InfinitePlaces(K)` / `InfinitePlaces(O)` | All infinite places of `K` or `O`. | Root analysis of defining polynomial. |
| `Divisor(pl)` | The divisor `1·pl` for place `pl`. | Trivial. |
| `Divisor(I)` | The divisor corresponding to ideal `I`: a linear combination of the places in the factorisation of `I` with the factorisation exponents. | Ideal factorisation. |
| `Divisor(x)` | The principal divisor `x·O` for element `x` of the maximal order (finite divisor). | Ideal factorisation of `(x)`. |
| `RealPlaces(K)` | All real (infinite) places of `K`: the embeddings into **R** coming from real roots of the defining polynomial. | Root analysis. |

### 37.10.4 Arithmetic with Places and Divisors

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `d1 + d2` / `- d` / `d1 - d2` / `d * k` / `d div k` | Addition, negation, subtraction, and integer scaling of divisors. | Formal linear combination arithmetic. |

### 37.10.5 Other Functions for Places and Divisors

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Valuation(a, p)` | Valuation of element `a` of a number field or order at place `p`. | p-adic valuation at the prime ideal underlying `p`. |
| `Valuation(I, p)` | Valuation of ideal `I` at finite place `p`. | Ideal factorisation. |
| `Support(D)` | The support of divisor `D`: sequences of places and corresponding exponents. | From divisor representation. |
| `Ideal(D)` | The ideal corresponding to the finite part of divisor `D`. | Reconstruction from finite places. |
| `Evaluate(x, p)` / `Evaluate(x, p)` / `Evaluate(x, p)` | For `x` a number-field element and `p` a place: for finite `p`, the image in the residue class field; for infinite `p`, the corresponding conjugate (real or complex number). | Residue class map or conjugate evaluation. |
| `RealEmbeddings(a)` / `RealEmbeddings(a)` | All real embeddings of algebraic number `a` (evaluations at all real places). | Conjugate evaluation. |
| `RealSigns(a)` / `RealSigns(a)` | Sequence of ±1 giving the sign of `a` at each real place. | Sign of real conjugates. |
| `IsReal(p)` | Returns `true` iff infinite place `p` corresponds to a real embedding. | — |
| `IsComplex(p)` | Returns `true` iff infinite place `p` corresponds to a complex embedding. | — |
| `IsFinite(p)` | Returns `true` iff place `p` corresponds to a prime ideal. | — |
| `IsInfinite(p)` | Returns `true` iff place `p` corresponds to an embedding; also returns the embedding index. | — |
| `Extends(P, p)` | For place `P` of `K` and place `p` of `k` (where `K` is an extension of `k`): check whether `P` extends `p`. For finite places: the prime ideal of `P` divides that of `p` in `O_K`. For infinite: `Evaluate` at `P` and `p` agree on `k`. | Divisibility / embedding comparison. |
| `InertiaDegree(P)` / `Degree(P)` | For a place `P`: the inertia degree (degree of the residue class field over its prime field for finite places; always 1 for infinite places). | Residue field extension degree. |
| `Degree(D)` | For divisor `D`: weighted sum of the degrees of the supporting places, with multiplicities as weights. | Linear combination of `Degree(P)`. |
| `NumberField(P)` | The number field underlying place `P` or divisor `D`. | Structural look-up. |
| `ResidueClassField(P)` | For finite place `P`: the residue class field (a finite field). For infinite place: the field of real or complex numbers. | Residue class field of the underlying prime ideal. |
| `UniformizingElement(P)` | For a finite place `P`: an element of valuation 1 (also the uniformizing element of the underlying prime ideal). | — |
| `LocalDegree(P)` | The degree of the completion at place `P`: inertia degree × ramification index. | — |
| `RamificationIndex(P)` | The ramification index of place `P`. 1 for real infinite places, 2 for complex infinite places. | — |
| `DecompositionGroup(P)` | For a place `P` of a normal number field: the decomposition group as a subgroup of the (abstract) automorphism group. | Galois theory. |

---

## 37.11 Bibliography

| Key | Reference |
|-----|-----------|
| **[Bai96]** | Georg Baier. *Zum Round 4 Algorithmus.* Diplomarbeit, Technische Universität Berlin, 1996. URL: http://www.math.tu-berlin.de/~kant/publications/diplom/baier.ps.gz. |
| **[Ber05]** | Daniel J. Bernstein. *Factoring into coprimes in essentially linear time.* J. of Algorithms, **54**(1):1–30, 2005. |
| **[BH96]** | Yuri Bilu and Guillaume Hanrot. *Solving Thue Equations of High Degree.* J. Number Th., **60**:373–392, 1996. |
| **[Bia]** | J.-F. Biasse. *Number field sieve to compute Class groups.* (Preprint/unpublished.) |
| **[Bj94]** | Johannes A. Buchmann and Hendrik W. Lenstra jr. *Approximating rings of integers in number fields.* J. Théor. Nombres Bordx., **6**(2):221–260, 1994. |
| **[Bos00]** | Wieb Bosma, editor. *ANTS IV*, volume 1838 of LNCS. Springer-Verlag, 2000. |
| **[Coh93]** | Henri Cohen. *A Course in Computational Algebraic Number Theory*, volume 138 of Graduate Texts in Mathematics. Springer, Berlin–Heidelberg–New York, 1993. |
| **[Coh00]** | Henri Cohen. *Advanced Topics in Computational Number Theory.* Springer, Berlin–Heidelberg–New York, 2000. |
| **[FF00]** | Claus Fieker and Carsten Friedrichs. *On reconstruction of algebraic numbers.* In Bosma [Bos00], pages 285–296. |
| **[Fie97]** | Claus Fieker. *Über relative Normgleichungen in algebraischen Zahlkörpern.* Dissertation, Technische Universität Berlin, 1997. URL: http://www.math.tu-berlin.de/~kant/publications/diss/diss_CF.ps.gz. |
| **[Fin84]** | Ulrich Fincke. *Ein Ellipsoidverfahren zur Lösung von Normgleichungen in algebraischen Zahlkörpern.* Dissertation, Heinrich-Heine-Universität Düsseldorf, 1984. |
| **[FJP97]** | C. Fieker, A. Jurk, and M. Pohst. *On solving relative norm equations in algebraic number fields.* Math. Comput., **66**(217):399–410, 1997. |
| **[Fri97]** | Carsten Friedrichs. *Berechnung relativer Ganzheitsbasen mit dem Round-2-Algorithmus.* Diplomarbeit, Technische Universität Berlin, 1997. URL: http://www.math.tu-berlin.de/~kant/publications/diplom/friedrichs.ps.gz. |
| **[Fri00]** | Carsten Friedrichs. *Berechnung von Maximalordnungen über Dedekindringen.* Dissertation, Technische Universität Berlin, 2000. URL: http://www.math.tu-berlin.de/~kant/publications/diss/diss_fried.pdf.gz. |
| **[Gar80]** | Dennis A. Garbanati. *An Algorithm for finding an algebraic number whose norm is a given rational number.* J. reine angew. Math., **316**:1–13, 1980. |
| **[GPP93]** | István Gaál, Attila Pethő, and Michael E. Pohst. *On the resolution of index form equations in quartic number fields.* J. Symbolic Comp., **16**:563–584, 1993. |
| **[GPP96]** | István Gaál, Attila Pethő, and Michael E. Pohst. *Simultaneous representation of integers by a pair of ternary quadratic forms — With an application to index form equations in quartic number fields.* J. Number Th., **57**:90–104, 1996. |
| **[GS89]** | István Gaál and Nicole Schulte. *Computing all power integral bases of cubic fields.* Math. Comp., **53**:689–696, 1989. |
| **[Heß96]** | Florian Heß. *Zur Klassengruppenberechnung in algebraischen Zahlkörpern.* Diplomarbeit, Technische Universität Berlin, 1996. URL: http://www.math.tu-berlin.de/~kant/publications/diplom/hess.ps.gz. |
| **[Jur93]** | Andreas Jurk. *Über die Berechnung von Lösungen relativer Normgleichungen in algebraischen Zahlkörpern.* Dissertation, Heinrich-Heine-Universität Düsseldorf, 1993. |
| **[KAN97]** | KANT Group. *KANT V4.* J. Symbolic Comp., **24**(3–4):267–383, 1997. |
| **[KAN00]** | KANT Group. *The Number Theory Package KANT/KASH.* URL: http://www.math.tu-berlin.de/~kant, 2000. |
| **[PK05]** | Sebastian Pauli and Jüren Klüners. *Computing residue class rings and Picard groups of orders.* J. of Algebra, **292**:47–64, 2005. |
| **[Poh93]** | M. Pohst. *Computational Algebraic Number Theory.* DMV Seminar Band 21. Birkhäuser Verlag, Basel–Boston–Berlin, 1993. |
| **[PZ89]** | Michael E. Pohst and Hans Zassenhaus. *Algorithmic Algebraic Number Theory.* Encyclopaedia of mathematics and its applications. Cambridge University Press, Cambridge, 1989. |
| **[Sim02]** | Denis Simon. *Solving norm equations in relative number fields using S-units.* Math. Comput., **71**(239):1287–1305, 2002. |
| **[Sut12]** | Nicole Sutherland. *Efficient Computation of Maximal Orders of Radical (including Kummer) Extensions.* Journal of Symbolic Computation, **47**(5):552–567, 2012. |
| **[Wil97]** | Klaus Wildanger. *Über das Lösen von Einheiten- und Indexformgleichungen in algebraischen Zahlkörpern mit einer Anwendung auf die Bestimmung aller ganzen Punkte einer Mordellschen Kurve.* Dissertation, Technische Universität Berlin, 1997. URL: http://www.math.tu-berlin.de/~kant/publications/diss/KW_diss.ps.gz. |
| **[Wil00]** | Klaus Wildanger. *Über das Lösen von Einheiten- und Indexformgleichungen in algebraischen Zahlkörpern. (On the solution of units and index form equations in algebraic number fields).* J. Number Th., **2**(82):188–224, 2000. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Round-2 maximal order **[Coh93, Poh93, PZ89]** | `MaximalOrder`, `pMaximalOrder`, `pRadical`, `MultiplicatorRing` |
| Round-4 maximal order **[Bai96]** | `MaximalOrder(:Al:="Round4")`, `pMaximalOrder(:Al:="Round4")` |
| Pauli (relative, completion-based) maximal order | `MaximalOrder(:Al:="Pauli")`, `pMaximalOrder(:Al:="Pauli")` |
| Radical/Kummer extension maximal order **[Sut12]** | `MaximalOrder`, `pMaximalOrder` (auto-selected for radical extensions) |
| Known-discriminant maximal order **[Bj94]** | `MaximalOrder(:Discriminant:=…)`, `MaximalOrder(:Ramification:=…)` |
| Relative Round-2 **[Coh00, Fri97]** | `MaximalOrder` (for relative extensions) |
| LLL basis reduction | `LLL(O)`, `OptimizedRepresentation` |
| Index calculus / relation method (class group) **[Heß96, Coh93]** | `ClassGroup`, `ClassNumber`, `ConditionalClassGroup`, `FactorBasisCreate`, `AddRelation`, `EvaluateClassGroup`, `CompleteClassGroup` |
| Lattice sieve (class group, large discriminant) **[Bia]** | `ClassGroup(:Al:="Sieve")` |
| Picard/ring class group of non-maximal orders **[PK05]** | `RingClassGroup`, `PicardGroup`, `UnitGroupAsSubgroup` |
| Unit group: Dirichlet / mixed / relation method **[PZ89, Poh93]** | `UnitGroup`, `IndependentUnits`, `pFundamentalUnits`, `MergeUnits` |
| Continued fraction (real quadratic units) | `UnitGroup(:Al:="ContFrac")`, `IndependentUnits(:Al:="ContFrac")` |
| Norm equation: class group + Fincke ellipsoid **[Fin84, PZ89]** | `NormEquation(O, m)` (Diophantine, absolute) |
| Norm equation: S-units **[Fie97, Sim02, Gar80, Coh00]** | `NormEquation(F, m)` (field-theoretic), `NormEquation(m, N)`, `IntegralNormEquation` |
| Norm equation: lattice methods (relative) **[Fie97, Jur93, FJP97]** | `NormEquation(O, m)` (relative, all solutions) |
| Simultaneous norm equations | `SimNEQ` |
| Thue equations: Bilu–Hanrot reduction **[BH96]** | `Thue`, `Solutions` |
| Unit equations: Wildanger's method **[Wil97, Wil00]** | `UnitEquation` |
| Index form equations: Wildanger **[Wil97, Wil00]** / Gaál–Pethő–Pohst **[GPP93, GPP96]** / Gaál–Schulte **[GS89]** | `IndexFormEquation` |
| S-unit group, S-unit action, discrete log | `SUnitGroup`, `SUnitAction`, `SUnitDiscLog` |
| LLL-based reconstruction **[FF00]** | `ReconstructionEnvironment`, `Reconstruct` |
| Coprime basis: Bernstein's algorithm **[Ber05]** | `Support(:UseBernstein)`, `CoprimeBasis(:UseBernstein)`, `CoprimeBasisInsert(:UseBernstein)` |
| Minkowski/Bach bounds | `MinkowskiBound`, `BachBound` |
| p-adic completion | `Completion`, `LocalRing` |
| Minkowski lattice / space | `Lattice(O)`, `MinkowskiLattice(O)`, `MinkowskiSpace(F)` |
