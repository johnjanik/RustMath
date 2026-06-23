# Chapter 21 — Finite Fields

**Handbook part:** III — Basic Rings
**Handbook pages:** 363–387 (PDF pages 494–521)

---

## Scope and overview

Chapter 21 covers Magma's environment for computing with lattices of finite fields. Complete freedom in the manner in which fields are constructed is allowed, while assuring compatibility. Finite fields of various kinds are supported, with optimized representations for each kind. The overall scheme for compatibly embedded finite fields is described in **[BCS97]**.

**Representation.** Arithmetic in small non-prime finite fields is carried out using tables of Zech logarithms, ensuring fast arithmetic for fields of small cardinality. Larger finite fields are internally represented as polynomial rings over a small finite field. The user may specify his own irreducible polynomial, although internally an alternative representation may be used. The scheme in **[BCS97]** guarantees that all embeddings between subfields are compatible (diagrams commute).

**Conway polynomials.** To avoid ambiguities when specifying small finite fields, Conway polynomials have been defined and tabulated by R. Parker. The Conway polynomial C_{p,n} is the lexicographically first monic irreducible primitive polynomial of degree n over F_p consistent with all C_{p,m} for m dividing n (consistency: for a root α of C_{p,n}, β = α^{(p^n−1)/(p^m−1)} is a root of C_{p,m}). Conway polynomials are used as default defining polynomials for F_{p^n} when available, but their special consistency properties are not exploited internally. To compute C_{p,n} one must know all C_{p,m} for m | n; no essentially better method is known than enumerating and testing primitive polynomials of degree n in lexicographical order.

**Ground field and prime field.** The prime field of F is the unique field of cardinality p = char(F); all prime fields of the same cardinality are identical in Magma. The ground field of F is the field over which F was explicitly constructed as an extension; if F was not constructed via `ext`, the ground field is the prime field. Printing of elements is relative to the ground field (polynomial in F.1), except for Zech-logarithm fields which default to power printing. Since V2.13, a database of low-term irreducible polynomials over F_2 is available for all degrees up to 90000, enabling rapid construction of F_{2^k} for k in that range.

---

## 21.1 Introduction

*(Narrative only; see Scope and overview above. No intrinsics are defined in this section.)*

### 21.1.1 Representation of Finite Fields

*(Described in Scope and overview.)*

### 21.1.2 Conway Polynomials

*(Described in Scope and overview.)*

### 21.1.3 Ground Field and Relationships

*(Described in Scope and overview.)*

---

## 21.2 Creation Functions

Since V2.13 a database of low-term irreducible polynomials over F_2 is available for degrees up to 90000 (via `IrreducibleLowTermGF2Polynomial`). For sparse trinomial/pentanomial polynomials (V2.11 and earlier default for GF(2^k) beyond the Conway range), the parameter `Sparse` selects these polynomials.

### 21.2.1 Creation of Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `FiniteField(q)` / `GaloisField(q)` / `GF(q)` | Create the finite field F_q, where q = p^n with p prime. Magma first factors q to find p and n. Parameters: `Optimize` (default `true`; if `false`, no optimised Zech/multi-step representation is built — fast creation but slower arithmetic); `Sparse` (default `false`; if `true` and q = 2^k beyond the Conway range, uses a sparse trinomial/pentanomial instead of a low-term polynomial). When n > 1, a Conway polynomial is used if available; otherwise a low-term polynomial (for char 2) or a general irreducible. | Conway polynomial lookup (tabulated by R. Parker) or `IrreducibleLowTermGF2Polynomial`; see **[BCS97]** for the compatible-embedding scheme. |
| `FiniteField(p, n)` / `GaloisField(p, n)` / `GF(p, n)` | Create F_{p^n} given a prime p and exponent n ≥ 1. Parameters: `Check` (default `true`; if `false`, no primality check on p — useful for very large p); `Optimize`, `Sparse` as above. | Same as above. |
| `ext< F \| n >` | Create an extension G of degree n of the finite field F, together with the embedding map φ : F → G. If F is a default field, G is also a default field (ground field = prime field); otherwise ground field of G is F. Parameters: `Optimize`, `Sparse`. | Conway polynomial or low-term polynomial; **[BCS97]**. |
| `ext< F \| P >` | Create the extension G = F[α] of degree n of F defined by the irreducible polynomial P of degree n over F, together with the natural embedding φ : F → G; α is a root of P. The defining polynomial of G over F is P. Ground field of G is F. Parameter: `Optimize`. | User-supplied irreducible polynomial; **[BCS97]**. |
| `ExtensionField< F, x \| P >` | Create the extension G = F[x] of degree n of F, where P is an irreducible polynomial over F expressed in the literal identifier x. Returns G and the embedding φ : F → G; x is a root of P. Parameter: `Optimize`. | User-supplied irreducible polynomial; **[BCS97]**. |
| `RandomExtension(F, n)` | Return the extension of F by a random degree-n irreducible polynomial over F. | Random irreducible polynomial selection. |
| `SplittingField(P)` | Given a univariate polynomial P over a finite field F, return the minimal splitting field of P: the smallest-degree extension G of F over which P factors into linears. | Factorisation over successively larger extensions. |
| `SplittingField(S)` | Given a set S of univariate polynomials over F, return the minimal extension field G of F over which every polynomial in S splits into linears. | As above, applied to each polynomial in S. |
| `sub< F \| d >` | Given a finite field F of cardinality p^n and a positive divisor d of n, create the subfield E of F of degree d, together with the embedding map φ : E → F. Parameters: `Optimize`, `Sparse`. | Subfield construction; **[BCS97]**. |
| `sub< F \| f >` | Given a finite field F and an element f ∈ F, create the subfield E of F generated by f (so φ(E.1) = f), together with the embedding map φ : E → F. Parameters: `Optimize`, `Sparse`. | Subfield construction; **[BCS97]**. |
| `GroundField(F)` / `BaseField(F)` | Return the ground field of F: the field over which F was explicitly constructed as an extension, or the prime field if F was not explicitly extended. | — |
| `PrimeField(F)` | The subfield of F of prime cardinality (the unique field of order char(F)). | — |
| `IsPrimeField(F)` | Returns `true` iff F is a prime field. | — |
| `F meet G` | Given finite fields F and G of the same characteristic p, return the finite field F ∩ G. | — |
| `CommonOverfield(K, L)` | Given finite fields K and L of characteristic p, return the smallest field containing both. | — |

*Worked example: H21E1 (constructing GF(7^4) by Conway polynomial, ext with internal polynomial, ext with user-supplied polynomial, and as a tower extension of GF(7^2)).*

### 21.2.2 Creating Relations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Embed(E, F)` | Given finite fields E and F of cardinality p^d and p^n with d | n, assert (set up) the embedding relation between E and F. An isomorphism between E and the unique subfield of F of cardinality p^d is chosen and registered for subsequent coercion. If both E and F were defined with Conway polynomials, the isomorphism maps the generator β of F to α^{(p^n−1)/(p^d−1)}, where α is the generator of F. See **[BCS97]** for details. | Compatible embedding scheme **[BCS97]**. |
| `Embed(E, F, x)` | As above, but also specifies that the generator of E maps to the element x ∈ F. x must be a root of the polynomial defining E over the prime field. | Compatible embedding scheme **[BCS97]** with explicit generator image. |

### 21.2.3 Special Options

For small finite fields for which the complete Zech logarithm table is stored, elements can be printed either as powers of the primitive element or as polynomials in the generating element.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AssertAttribute(FldFin, "PowerPrinting", l)` | Set the global default printing style for all (small) finite fields created thereafter. If l is `true`, elements of Zech-logarithm fields are printed as powers of the primitive element; if `false`, elements are printed as polynomials in F.1 over the ground field. | — |
| `SetPowerPrinting(F, l)` / `AssertAttribute(F, "PowerPrinting", l)` | Set the printing style for the specific finite field F (must be small enough for Zech logarithm table storage). `true` → power printing; `false` → polynomial printing. | — |
| `HasAttribute(FldFin, "PowerPrinting", l)` | Return `true` and the current global default value of the `"PowerPrinting"` attribute. | — |
| `HasAttribute(F, "PowerPrinting")` | For a Zech-logarithm field F: return `true` if the `"PowerPrinting"` attribute is defined for F, together with its current value; otherwise `false`. | — |
| `AssignNames(~F, [f])` | Procedure. Change the name of the generating element of F to the string f. Affects printing only; does not assign the generator to an identifier named f. Since this modifies F, a reference ~F is required. | — |
| `Name(F, 1)` | Return the element F.1 (the named generator) of F. | — |

### 21.2.4 Homomorphisms

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `hom< F -> G \| x >` | Create a ring homomorphism from the finite field F to G. If F is a prime field, the right-hand side must be empty (the unique unitary map is used). If F is not of prime cardinality, the homomorphism is specified by one element x ∈ G, which is the image of the generator of F over its prime field. Correctness of the map is the user's responsibility. | — |

### 21.2.5 Creation of Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `F . 1` | The generator of F as an algebra over its ground field. If F = E[X]/(P(X)), this is the image of X. For a prime field, returns 1_F. | — |
| `elt< F \| a >` / `F ! a` | Create the element of F specified by a. a may be: (i) an element of F; (ii) an element of a subfield of F; (iii) an element of an overfield of F that lies in F; (iv) an integer (taken mod char(F)); (v) a sequence [a_0, ..., a_{n−1}] of ground-field elements, giving a_0 + a_1·w + ··· + a_{n−1}·w^{n−1} where w = F.1. | Coercion. |
| `elt< F \| a0, ..., an−1 >` | Given the generator w = F.1 of degree n over the ground field E, create a_0 + a_1·w + ··· + a_{n−1}·w^{n−1} ∈ F, where a_i ∈ E (integers or elements of a subfield of E are coerced). | Coercion. |
| `One(F)` / `Identity(F)` | The multiplicative identity 1 in F. | — |
| `Zero(F)` / `Representative(F)` | The additive identity 0 in F. | — |
| `Random(F)` | A pseudorandom element of F. | — |

### 21.2.6 Special Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `F . 1` / `Generator(F)` | The element of F that generates F over its ground field E (same as F.1; F = E[f]). | — |
| `Generator(F, E)` | An element f of F generating F over the subfield E (F = E[f]). May differ from F.1; F.1 is returned if it works. | — |
| `PrimitiveElement(F)` | A primitive element for F, i.e., a generator of the multiplicative group F*. May differ from F.1. Returns the same element on repeated calls; this is the base used by `Log(x)`. | — |
| `SetPrimitiveElement(F, x)` | (Procedure.) Set the internal primitive element of F to x. If it has already been computed or set, x must equal it. Allows fixing the base used by `Log(x)`. | — |
| `NormalElement(F)` | Return a normal element α ∈ F over the ground field G of F = F_{p^n}; that is, α, α^q, ..., α^{q^{n−1}} form a basis for F over G (q = #G). Different calls may return different elements. | — |
| `NormalElement(F, E)` | Return a normal element α ∈ F_{q^n} over the subfield E = F_q: α, α^q, ..., α^{q^{n−1}} form a basis for F over E. | — |

### 21.2.7 Sequence Conversions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SequenceToElement(s, F)` / `Seqelt(s, F)` | Given a sequence s = [s_0, ..., s_{n−1}] of elements of a subfield E of F (with n = [F:E]), construct the element s_0 + s_1·w + ··· + s_{n−1}·w^{n−1} ∈ F, where w = F.1. | — |
| `ElementToSequence(a)` / `Eltseq(a)` | Given an element a ∈ F, return the sequence [a_0, ..., a_{n−1}] of coefficients over the ground field E such that a = a_0 + a_1·w + ··· + a_{n−1}·w^{n−1}, n = [F:E]. | — |
| `ElementToSequence(a, E)` / `Eltseq(a, E)` | As above, but with coefficients in the specified subfield E of F. | — |

---

## 21.3 Structure Operations

### 21.3.1 Related Structures

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Category(F)` / `Parent(F)` / `Centre(F)` | Generic structure functions (see Chapter 17). | — |
| `PrimeRing(F)` / `PrimeField(F)` | Return the prime ring / prime field of F. | — |
| `FieldOfFractions(F)` | Return the field of fractions of F (which is F itself, since F is already a field). | — |
| `AdditiveGroup(F)` | For F = F_q (q = p^r), create the finite additive abelian group A ≅ (Z/pZ)^r of order q, together with an isomorphism A → F. | — |
| `MultiplicativeGroup(F)` / `UnitGroup(F)` | For F = F_q, create the multiplicative group of F as an abelian group: the cyclic group A of order q − 1, together with a map A → F \ {0} sending 1 to a primitive element of F. | — |
| `Set(F)` | Create the enumerated set of all elements of F. | — |
| `VectorSpace(F, E)` | For F an extension of degree n of E, return (a) the vector space V ≅ E^n and (b) the isomorphism φ : F → V, where the basis of V corresponds to the power basis {1, α, ..., α^{n−1}} with α = Generator(F, E). | — |
| `VectorSpace(F, E, B)` | As above but with the user-specified basis B = β_1, ..., β_n for F over E. Returns V and φ : F → V with φ(β_i) = e_i. | — |
| `MatrixAlgebra(F, E)` | Let F be an extension of degree n of E. Return (a) a matrix algebra A of degree n (subalgebra of M_n(E) generated by the companion matrix C of the defining polynomial of F over E), isomorphic to F, and (b) the isomorphism φ : F → A sending Generator(F, E) to C. | — |
| `MatrixAlgebra(A, E)` | Let A be a matrix algebra over a finite field F with E a subfield of F. Return (a) a matrix algebra N over E isomorphic to A, obtained by expanding each component into its block matrix over E, and (b) the E-isomorphism φ : A → N. | — |
| `GaloisGroup(K, k)` | Compute the Galois group of the extension K/k (cyclic) as a permutation group, together with the roots of the defining polynomial of K/k in a compatible ordering. | Frobenius automorphism generates the cyclic Galois group. |
| `AutomorphismGroup(K, k)` | Compute the cyclic group of k-automorphisms of K as a permutation group, together with a sequence of all automorphisms and a map from the abstract group to explicit automorphisms. | Frobenius generator. |

*Worked example: H21E2 (constructing vector spaces of dimension 2 and 4 from GF(7^4) over GF(7^2) and GF(7)).*

### 21.3.2 Numerical Invariants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Characteristic(F)` / `# F` | The characteristic p of F / the cardinality of F. | — |
| `Degree(F)` | The absolute degree of F over its prime subfield. | — |
| `Degree(F, E)` | The degree of F over the subfield E (F must have been constructed as an extension of E). | — |

### 21.3.3 Defining Polynomial

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DefiningPolynomial(F)` | The polynomial with coefficients in the ground field E used to define F as an extension of E; this is the minimal polynomial of F.1 over E. | — |
| `DefiningPolynomial(F, E)` | The polynomial with coefficients in the subfield E used to define F as an extension of E; equals the minimal polynomial of Generator(F, E) over E. | — |

### 21.3.4 Ring Predicates and Booleans

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsConway(F)` | Return `true` iff F is defined over its prime field using a Conway polynomial. | — |
| `IsDefault(F)` | Return `true` iff F is a default field. | — |
| `IsCommutative(F)` / `IsUnitary(F)` / `IsFinite(F)` / `IsOrdered(F)` / `IsField(F)` / `IsEuclideanDomain(F)` / `IsPID(F)` / `IsUFD(F)` / `IsDivisionRing(F)` / `IsEuclideanRing(F)` / `IsPrincipalIdealRing(F)` / `IsDomain(F)` | Standard ring predicate functions (all return `true` for finite fields). | — |
| `F eq G` / `F ne G` | Equality / inequality of finite fields. | — |

### 21.3.5 Roots

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Roots(f)` | Given a polynomial f over a finite field F, find all roots of f in F and return a sorted sequence of pairs (root, multiplicity). | Root-finding over F. |
| `RootsInSplittingField(f)` | Given a univariate polynomial f over a finite field K, compute the minimal splitting field S of f as an extension of K, and return the roots of f in S together with S. Faster than computing the splitting field first and then finding roots. | Splitting field construction followed by root extraction. |
| `FactorizationOverSplittingField(f)` | Given a univariate polynomial f over a finite field K, compute the minimal splitting field S of f and return the factorization of f into linears over S, together with S. Faster than first computing S and then factoring. | Splitting field construction followed by factorization into linears. |
| `RootOfUnity(n, K)` | Return a primitive n-th root of unity in the smallest possible extension field of K. | Extension field construction. |

*Worked example: H21E3 (computing roots of a degree-20 polynomial over GF(2) in its splitting field of degree 72).*

---

## 21.4 Element Operations

See also Section 17.5 of the handbook.

### 21.4.1 Arithmetic Operators

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `+ a` / `- a` | Unary plus / negation. | Zech logarithm tables (small fields) or polynomial arithmetic over the prime/ground field. |
| `a + b` / `a - b` / `a * b` / `a / b` / `a ^ k` | Binary addition, subtraction, multiplication, division, and exponentiation. | As above. |
| `a +:= b` / `a -:= b` / `a *:= b` | In-place addition, subtraction, multiplication. | As above. |

### 21.4.2 Equality and Membership

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `a eq b` / `a ne b` | Equality / inequality of field elements. | — |
| `a in F` / `a notin F` | Membership / non-membership of element a in field F. | — |

### 21.4.3 Parent and Category

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Parent(a)` / `Category(a)` | Return the parent field / category of the element a. | — |

### 21.4.4 Predicates on Ring Elements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsZero(a)` / `IsOne(a)` / `IsMinusOne(a)` | Return `true` iff a = 0 / 1 / −1. | — |
| `IsNilpotent(a)` / `IsIdempotent(a)` | Standard ring-element predicates. | — |
| `IsUnit(a)` / `IsZeroDivisor(a)` / `IsRegular(a)` | Standard ring-element predicates (in a field, every non-zero element is a unit). | — |
| `IsIrreducible(a)` / `IsPrime(a)` | Standard ring-element predicates. | — |
| `IsPrimitive(a)` | Return `true` iff the finite field element a is a primitive element, i.e., its multiplicative order is #F − 1. | Order computation via factorisation of #F − 1. |
| `IsPrimitive(f)` | Given a univariate polynomial f ∈ F[x] of degree ≥ 1, return `true` iff f defines a primitive extension G = F[x]/f of F (i.e., x is primitive in G). | — |
| `IsNormal(a)` | Return `true` iff a generates a normal basis for F over the ground field G = F_q, i.e., iff a, a^q, ..., a^{q^{n−1}} form a basis for F over G. | Normal basis test. |
| `IsNormal(a, E)` | Return `true` iff a ∈ F_{q^n} generates a normal basis for F over the subfield E = F_q. | Normal basis test. |
| `IsSquare(a)` | Return `true` and a square root b (with b^2 = a) if a is a square in F, otherwise `false`. | Square root computation. |

### 21.4.5 Minimal and Characteristic Polynomial

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `MinimalPolynomial(a)` | The minimal polynomial of a ∈ F relative to the ground field of F: the unique monic polynomial of minimum degree with ground-field coefficients having a as a root. | — |
| `MinimalPolynomial(a, E)` | The minimal polynomial of a ∈ F relative to the subfield E: the unique monic polynomial of minimum degree with coefficients in E having a as a root. | — |
| `CharacteristicPolynomial(a)` | The characteristic polynomial of a ∈ F with respect to the ground field (the characteristic polynomial of the companion matrix of a written as a polynomial over the ground field; a power of the minimal polynomial). | Companion matrix. |
| `CharacteristicPolynomial(a, E)` | The characteristic polynomial of a ∈ F with respect to the subfield E (a power of the minimal polynomial over E). | Companion matrix. |

### 21.4.6 Norm, Trace and Frobenius

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Norm(a)` | The norm of a from F to the ground field of F. | Product of conjugates. |
| `Norm(a, E)` | The relative norm of a from F to the subfield E; result is in E. | Product of conjugates. |
| `AbsoluteNorm(a)` / `NormAbs(a)` | The absolute norm of a: the norm from F to its prime subfield. | Product of all conjugates. |
| `Trace(a)` | The trace of a from F to the ground field of F. | Sum of conjugates. |
| `Trace(a, E)` | The relative trace of a from F to the subfield E; result is in E. | Sum of conjugates. |
| `AbsoluteTrace(a)` / `TraceAbs(a)` | The absolute trace of a: the trace to the prime subfield of F. | Sum of all conjugates. |
| `Frobenius(a)` | The Frobenius image of a with respect to the ground field G of the parent of a: a^{#G}. | Exponentiation. |
| `Frobenius(a, r)` | The r-th Frobenius image of a with respect to the ground field G: a^{(#G)^r}. | Exponentiation. |
| `Frobenius(a, E)` | The Frobenius image of a with respect to the subfield E: a^{#E}. | Exponentiation. |
| `Frobenius(a, E, r)` | The r-th Frobenius image of a with respect to E: a^{(#E)^r}. | Exponentiation. |
| `NormEquation(K, y)` | Given a finite field K and an element y in a subfield S of K, determine whether an element x ∈ K exists with Norm(x, S) = y, and return such x if so. | — |
| `Hilbert90(a, q)` | Given a ∈ k and a power q of char(k), return a solution x of the multiplicative Hilbert 90 equation x^q · x^{−1} = a (solution may lie in a finite-degree extension of k). | Hilbert's Theorem 90 (multiplicative). |
| `AdditiveHilbert90(a, q)` | Given a ∈ k and a power q of char(k), return a solution x of the additive Hilbert 90 equation x^q − x = a (solution may lie in a finite-degree extension of k). | Hilbert's Theorem 90 (additive / Artin–Schreier). |

*Worked example: H21E4 (Root, Trace, Norm, MinimalPolynomial over GF(7^4)/GF(7^2)/GF(7); demonstration of NormEquation).*

### 21.4.7 Order and Roots

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Order(a)` | The multiplicative order of the non-zero element a ∈ F. | Factorisation of #F − 1, then order-from-factorisation. |
| `FactoredOrder(a)` | The multiplicative order of a ∈ F returned as a factorisation sequence. | As above. |
| `SquareRoot(a)` / `Sqrt(a)` | The square root of the non-zero element a ∈ F: an element y with y^2 = a. Errors if a is not a square. | Tonelli–Shanks or Cipolla's algorithm (characteristic-dependent). |
| `Root(a, n)` | The n-th root of the non-zero element a ∈ F: an element y with y^n = a. Errors if no such root exists in F. | — |
| `IsPower(a, n)` | Given a ∈ F and integer n > 0, return `true` and b with b^n = a if such b exists in F, otherwise `false`. | — |
| `AllRoots(a, n)` | Given a ∈ F and integer n > 0, return a sequence of all n-th roots of a lying in F. | — |

---

## 21.5 Polynomials for Finite Fields

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IrreduciblePolynomial(F, n)` | Given a finite field F and n > 1, return an irreducible polynomial of degree n over F. Returns a Conway polynomial or stored sparse polynomial if available. | Conway polynomial table or `IrreducibleLowTermGF2Polynomial` / `IrreducibleSparseGF2Polynomial`; otherwise random search with irreducibility test. |
| `RandomIrreduciblePolynomial(F, n)` | Return a random irreducible polynomial of degree n over F. Generally dense (Conway or sparse tables are not used). | Random polynomial with irreducibility testing. |
| `IrreducibleLowTermGF2Polynomial(n)` | Given 1 ≤ n ≤ 100000, return the irreducible polynomial f = x^n + g over F_2 where deg(g) is minimal and g is the lexicographically first such polynomial. Uses a database constructed by Allan Steel in 2004. | Database lookup (Steel 2004). |
| `IrreducibleSparseGF2Polynomial(n)` | Given 4 ≤ n ≤ 12800, return the irreducible polynomial f = x^n + g over F_2 where g has 2 non-zero terms if possible, 4 otherwise; g lexicographically first. Uses a database constructed by Allan Steel in 1998. | Database lookup (Steel 1998). |
| `PrimitivePolynomial(F, m)` | Given a finite field F and m > 1, construct an irreducible polynomial f of degree m over F such that a root of f is a primitive element of the degree-m extension of F. | Search among irreducible polynomials for one with a primitive root. |
| `AllIrreduciblePolynomials(F, m)` | Given a finite field F and m > 1, return the set of all monic irreducible polynomials of degree m over F. | Enumeration and irreducibility testing. |
| `ConwayPolynomial(p, n)` | Given a prime p and n ≥ 1, return the Conway polynomial of degree n over F_p (read from a table; available only for a limited range of p, n). | Table lookup (R. Parker's database). |
| `ExistsConwayPolynomial(p, n)` | Given a prime p and n > 1, return `true` and the Conway polynomial if it is known for F_{p^n}; otherwise `false`. | Table lookup. |

---

## 21.6 Discrete Logarithms

Let K = F_q with q = p^k, p prime. Magma implements several advanced algorithms for computing discrete logarithms. The algorithm used depends on the type of field:

- **(a) Small fields (any characteristic):** If the largest prime l dividing q − 1 is reasonably small (typically < 2^{36}), the **Pohlig–Hellman algorithm [PH78]** is used, combined with Shanks baby-step/giant-step (for very small l) or the Pollard-ρ algorithm.
- **(b) Large prime fields (q = p):** If p has between 4 and 400 bits, p − 1 is not a square, and one of {−1, −2, −3, −7, −11} is a quadratic residue mod p, the **Gaussian integer sieve [COS86, LO91a]** is used. Otherwise, if p ≤ 300 bits, the **linear sieve [COS86, LO91a]** is used. The precomputation stage (factor base logarithms) is expensive and shared across subsequent `Log` calls; the first call may be much slower than later ones. The linear sieve requires significantly more time and memory than the Gaussian method for the precomputation stage.
- **(c) Small characteristic, non-prime (since V2.19):** For characteristic p < 2^{30}, an implementation by Allan Steel of **Coppersmith's index-calculus algorithm [Cop84, GM93, Tho01]** is used (generalised beyond the p = 2 case). Auxiliary external tables pre-compute logarithms of factor bases for many common fields, so for those fields individual logarithms can be computed immediately.
- **(d) Large characteristic, non-prime:** The **Pohlig–Hellman algorithm [PH78]** is used.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Log(x)` | The discrete logarithm of the non-zero element x ∈ F: the unique integer k with x = w^k and 0 ≤ k < #F − 1, where w = PrimitiveElement(F). Default parameters are chosen automatically for index-calculus methods. | See above: Pohlig–Hellman **[PH78]**, Gaussian/linear sieve **[COS86, LO91a]**, or Coppersmith index-calculus **[Cop84, GM93, Tho01]**. |
| `Log(b, x)` | The discrete logarithm of x to the base b: the unique integer k with x = b^k, 0 ≤ k < #F − 1. If b is not primitive, the algorithm may take longer than normal. | As above. |
| `ZechLog(K, n)` | The Zech logarithm Z(n) for field K: the integer Z(n) such that w^{Z(n)} = w^n + 1 (where w is the primitive element); returns −1 if w^n = −1. | Lookup in Zech logarithm table. |
| `Sieve(K)` | (Procedure.) Run the Gaussian integer sieve on the prime finite field K, or the linear sieve if the Gaussian sieve is inapplicable. Parameter: `Lanczos` (default `false`; if `true`, use the Lanczos algorithm **[LO91b, Sec. 3]** for the linear algebra phase — much slower but uses far less memory). | Gaussian integer sieve or linear sieve **[COS86, LO91a]**; Lanczos sparse linear algebra **[LO91b]**. |
| `SetVerbose("FFLog", v)` | (Procedure.) Set the verbose printing level for the finite field logarithm algorithm. Legal values: 0 (silent), 1 (prints when a logarithm is computed, unless the field is very small), 2 (very verbose). | — |

*Worked example: H21E5 (Log and Log(b, x) over GF(7^4); Log using Coppersmith's algorithm over GF(2^{73}), demonstrating precomputation caching).*

---

## 21.7 Permutation Polynomials

A polynomial representing a bijection of a finite field K into itself is a **permutation polynomial**. The Dickson polynomials of the first and second kind are permutation polynomials under certain conditions. By a theorem of Nöbauer, the Dickson polynomial of the first kind D_n(x, a) is a permutation polynomial for F_q iff gcd(n, q^2 − 1) = 1.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DicksonFirst(n, a)` | Construct the Dickson polynomial of the first kind D_n(x, a) of degree n, defined by D_n(x, a) = Σ_{i=0}^{⌊n/2⌋} (n/(n−i)) · C(n−i, i) · (−a)^i · x^{n−2i}. | Direct formula evaluation. |
| `DicksonSecond(n, a)` | Construct the Dickson polynomial of the second kind E_n(x, a) of degree n, defined by E_n(x, a) = Σ_{i=0}^{⌊n/2⌋} C(n−i, i) · (−a)^i · x^{n−2i}. | Direct formula evaluation. |
| `IsProbablyPermutationPolynomial(p)` | Probabilistic test whether the polynomial p over a finite field K defines a bijection on K. Returns `true` if the test succeeds for each of n attempts (default n = 100). Parameter: `NumAttempts` (default 100). | Random evaluation test (probabilistic). |

*Worked example: H21E6 (Dickson first-kind polynomial D_n(x, a) for K = F_{16}; checking (n, q^2 − 1) = 1 for permutation property; IsProbablyPermutationPolynomial).*

---

## 21.8 Bibliography

| Key | Reference |
|-----|-----------|
| **[BCS97]** | Wieb Bosma, John Cannon, and Allan Steel. *Lattices of Compatibly Embedded Finite Fields.* J. Symbolic Comp. **24**(3):351–369, 1997. |
| **[Cop84]** | D. Coppersmith. *Fast evaluation of logarithms in fields of characteristic two.* IEEE Trans. Inform. Theory, IT–30(4):587–594, July 1984. |
| **[COS86]** | D. Coppersmith, A. M. Odlyzko, and R. Schroeppel. *Discrete logarithms in GF(p).* Algorithmica, 1:1–15, 1986. |
| **[GM93]** | D. M. Gordon and K. S. McCurley. *Massively parallel computation of discrete logarithms.* In Ernest F. Brickell, editor, Advances in Cryptology — CRYPTO 1992, volume 740 of LNCS, pages 312–323. Springer-Verlag, 1993. Proc. 12th Annual International Cryptology Conference, Santa Barbara, CA, USA, August 16–20, 1992. |
| **[LO91a]** | B. A. LaMacchia and A. M. Odlyzko. *Computation of Discrete Logarithms in Prime Fields.* In A. J. Menezes and S. Vanstone, editors, Advances in Cryptology — CRYPTO 1990, volume 537 of LNCS, pages 616–618. Springer-Verlag, 1991. |
| **[LO91b]** | B. A. LaMacchia and A. M. Odlyzko. *Solving Large Sparse Linear Systems over Finite Fields.* In A. J. Menezes and S. Vanstone, editors, Advances in Cryptology — CRYPTO 1990, volume 537 of LNCS, pages 109–133. Springer-Verlag, 1991. |
| **[PH78]** | S. C. Pohlig and M. E. Hellman. *An Improved Algorithm for Computing Logarithms over GF(p) and Its Cryptographic Significance.* IEEE Trans. Inform. Theory, **24**:106–110, 1978. |
| **[Tho01]** | Emmanuel Thomé. *Computation of discrete logarithms in F_{2^{607}}.* In Colin Boyd and Ed Dawson, editors, Advances in Cryptology — AsiaCrypt 2001, volume 2248 of LNCS, pages 107–124. Springer-Verlag, 2001. Proc. 7th International Conference on the Theory and Applications of Cryptology and Information Security, Dec. 9–13, 2001, Gold Coast, Queensland, Australia. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Compatible lattice embedding scheme **[BCS97]** | `FiniteField`, `GF`, `GaloisField`, `ext`, `ExtensionField`, `sub`, `Embed`, `RandomExtension` |
| Conway polynomial construction (R. Parker) | `FiniteField(q)`, `FiniteField(p,n)`, `ext<F\|n>`, `ConwayPolynomial`, `ExistsConwayPolynomial`, `IrreduciblePolynomial`, `IsConway` |
| Low-term GF(2) polynomial database (Steel 2004) | `IrreducibleLowTermGF2Polynomial`, `FiniteField`/`ext` (char 2, large n) |
| Sparse GF(2) polynomial database (Steel 1998) | `IrreducibleSparseGF2Polynomial` |
| Zech logarithm tables | `Log`, `ZechLog`, arithmetic operators, `SetPowerPrinting`/`AssertAttribute("PowerPrinting",...)` |
| Pohlig–Hellman (+ baby-step/giant-step or Pollard-ρ) **[PH78]** | `Log`, `Log(b,x)` (small fields and large characteristic non-prime) |
| Gaussian integer sieve **[COS86, LO91a]** | `Log`, `Sieve` (large prime fields, preferred method) |
| Linear sieve **[COS86, LO91a]** | `Log`, `Sieve` (large prime fields, fallback) |
| Lanczos sparse linear algebra **[LO91b]** | `Sieve(:Lanczos)` |
| Coppersmith index-calculus (+ generalisations) **[Cop84, GM93, Tho01]** | `Log`, `Log(b,x)` (small characteristic, non-prime) |
| Dickson polynomials (permutation criterion: Nöbauer's theorem) | `DicksonFirst`, `DicksonSecond`, `IsProbablyPermutationPolynomial` |
| Hilbert 90 (multiplicative and additive) | `Hilbert90`, `AdditiveHilbert90` |
| Normal basis | `NormalElement`, `IsNormal` |
