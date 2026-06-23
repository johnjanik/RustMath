# Chapter 136 — Modular Abelian Varieties

**Authors:** William A. Stein (with this chapter's "Building blocks" section contributed by Jordi Quer)
**Handbook part:** XVII — Modular Arithmetic Geometry
**Handbook pages:** 4519–4650 (PDF pages 4646–4784)

---

## Scope and overview

This chapter is the reference for the modular abelian varieties package in Magma. A
*modular abelian variety* is an abelian variety that is a quotient of the modular Jacobian
`J₁(N)`, for some integer `N`. The package provides extensive functionality for computing
with such abelian varieties: enumerating and decomposing them, isomorphism testing,
computing exact endomorphism and homomorphism rings, doing arithmetic with finite
subgroups, and computing torsion subgroups, special values of `L`-functions, and Tamagawa
numbers.

Essentially **none** of the algorithms use explicit defining equations for varieties, so
they work in great generality; much of the functionality also makes sense for Grothendieck
motives attached to modular forms, and that is included where meaningful (such objects are
"motives" rather than abelian varieties unless base-extended to `C`).

**Representation.** Magma views an abelian subvariety `A ⊂ J₀(N)` over `Q`, with the map
`i : A → J₀(N)`, as completely determined by the image of `H₁(A, Q)` in the vector space
`H₁(X₀(N), Q)`. Rather than computing floating-point approximate lattices, the package uses
**modular symbols** to compute `H₁(X₀(N), Z)` as an abstract abelian group, and leverages the
theory of modular forms. Even though one works with homology (associated to complex tori),
the abelian variety `A` over `Q` is still determined by the defining data (a subgroup of
`H₁(X₀(N), Z)`), which the algorithms exploit.

**Limitations stated.** Magma V2.11 was the first release. The major drawback of the version
documented is that complete decomposition into simples is only implemented over the rational
numbers; the interesting behaviour over number fields (extra inner twists, `Q`-curves and
related questions) is largely unavailable, with some fundamental theoretical obstructions.
Computations can be carried out in the `+1` or `−1` quotient of homology for efficiency,
though certain results will then be off by powers of 2.

The category is `ModAbVar`; elements are `ModAbVarElt`; homomorphisms (sometimes only up to
isogeny) form `MapModAbVar`; spaces of homomorphisms form `HomModAbVar`; finitely generated
subgroups form `ModAbVarSubGrp`; homology lies in `ModAbVarHomol`; `L`-series lie in
`ModAbVarLSer`. Verbosity is controlled with `SetVerbose("ModAbVar", n)` for `n` in `0..4`
(levels 3–4 append to `ModAbVar-verbose.log`).

*Worked examples: H136E1 (one object of each category); H136E2 (verbose levels).*

---

## 136.2 Creation and Basic Functions

The functions here create modular abelian varieties, combine them, and obtain basic
information. Modular abelian varieties are much less restricted than spaces of modular
symbols, since one can take arbitrary finite direct sums.

### 136.2.1 Creating the Modular Jacobian J₀(N)

`JZero` creates the Jacobian `J₀(N)` of the modular curve `X₀(N)` (parameterizing pairs of
elliptic curves with a cyclic subgroup of order `N`). Higher-weight motivic analogues can be
created. Computations can be done in the `+1` or `−1` quotient of homology.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `JZero(N : -)` / `JZero(N, k : -)` | The modular abelian variety `J₀(N)` of level `N` and weight 2 (or weight `k ≥ 2`), i.e. the Jacobian of `X₀(N)`. Parameter `Sign` (`RngIntElt`, default 0) selects the `+1` or `−1` quotient of homology. | Modular symbols for `Γ₀(N)`. |

*Worked example: H136E3 (`JZero(23)`, weight-`k` motives, `Sign`).*

### 136.2.2 Creating the Modular Jacobians J₁(N) and J_H(N)

`JOne` creates `J₁(N)` (Jacobian of `X₁(N)`). `Js` creates a variety `Q`-isogenous to
`J₁(N)` — the direct sum of varieties `J_ε(N)` over Nebentypus characters — and is much
faster to create than `J₁(N)`. `JH` creates `J_H(N)`, isogenous to the Jacobian of `X_H(N)`,
the quotient of `X₁(N)` by a subgroup `H` of `(Z/NZ)*`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `JOne(N : -)` / `JOne(N, k : -)` | The modular abelian variety `J₁(N)` of level `N` and weight `k` (or 2), i.e. the Jacobian of `X₁(N)`. Parameter `Sign` (default 0). Note: finding the integral structure on the isogenous `J_s(N)` is much faster; computing with `J₁(N)` itself may be expensive. | Modular symbols for `Γ₁(N)`. |
| `Js(N : -)` / `Js(N, k : -)` | A modular abelian variety `Q`-isogenous to the weight-`k` (or 2) version of `J₁(N)`: the direct sum of the varieties attached to the modular symbols spaces with Nebentypus. Parameter `Sign` (default 0). | Direct sum over characters; avoids the costly integral structure of `J₁(N)`. |
| `JH(N, d : -)` / `JH(N, k, d : -)` | For `H` a (cyclic) subgroup of `G = (Z/NZ)*` with `G/H` of order `d`: the variety `J_H(N)` of level `N` and weight `k` (or 2), isogenous to the Jacobian of `X_H(N)`. It is the product of `J(ε)` over Dirichlet characters `ε` trivial on `H`. Parameter `Sign` (default 0). | Product of modular-symbols varieties `J(ε)`. |
| `JH(N, gens : -)` / `JH(N, k, gens : -)` | As above, but `H` is the subgroup of `(Z/NZ)*` generated by the sequence of integers `gens`. Parameter `Sign` (default 0). | Product of modular-symbols varieties `J(ε)`. |

*Worked example: H136E4 (`JOne`, `JH`, `Js`, `IsIsogenous(JOne(17),Js(17))`).*

### 136.2.3 Abelian Varieties Attached to Modular Forms

Commands creating abelian varieties attached to spaces of modular forms, sequences, newforms
and characters. A non-cuspidal input space is automatically replaced by its cuspidal
subspace.

| Intrinsic | Description |
|-----------|-------------|
| `ModularAbelianVariety(M : -)` | The abelian variety attached to the modular forms space `M`. Parameter `Sign` (default 0). |
| `ModularAbelianVariety(X : -)` | The abelian variety attached to the sequence `X` of modular forms spaces: the direct sum of the spaces attached to each element. Parameter `Sign` (default 0). |
| `ModularAbelianVariety(eps : -)` / `ModularAbelianVariety(eps, k : -)` | The abelian variety associated to the Dirichlet character `ε` (weight `k`): corresponds to the space of weight-`k` forms with character any Galois conjugate of `ε`, so as to be defined over `Q`. Parameter `Sign` (default 0). |
| `ModularAbelianVariety(f)` | The abelian variety `A_f` attached to the newform `f`. |
| `Newform(A)` | A newform `f` such that `A` is isogenous to the newform abelian variety `A_f`. Errors if `A` is not attached to a newform. |

*Worked examples: H136E5 (`S₂(Γ₀(11))⊕S₂(Γ₁(13))`, `IsIsomorphic`); H136E6 (`ModularAbelianVariety(eps,k)`, newform in `S₂(Γ₁(25))`, recovering newform); H136E7 (`Newform` of a `Decomposition` factor).*

### 136.2.4 Abelian Varieties Attached to Modular Symbols

Associates modular abelian varieties to spaces (or sequences of spaces) of modular symbols
and conversely. Non-cuspidal input is replaced by its cuspidal subspace.

| Intrinsic | Description |
|-----------|-------------|
| `ModularAbelianVariety(M)` | The abelian variety attached to the modular symbols space `M`. |
| `ModularAbelianVariety(X)` | The abelian variety attached to the sequence `X` of modular symbols spaces. |
| `ModularSymbols(A)` | A sequence of spaces of modular symbols associated to the abelian variety `A`. (A sequence, because arbitrary finite products are allowed; these are the spaces used internally in computations on `A`.) |

*Worked example: H136E8 (`ModularSymbols(37,2)`, `Gamma1(17)`, sign, and a sequence; `ModularSymbols(JOne(13))` etc.).*

### 136.2.5 Creation of Abelian Subvarieties

For `A` an abelian variety and `V` a vector subspace of `H₁(A, Q)`, one can decide whether
`V` is the rational homology of an abelian subvariety, and if so compute it.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DefinesAbelianSubvariety(A, V)` | Returns `true` iff the subspace `V` of rational homology defines an abelian subvariety of `A`; if so, also returns that subvariety. | Relies on a complete decomposition of `A` into simples (so currently restricted to varieties — e.g. modular abelian varieties over `Q` — for which such a decomposition can be computed). |
| `ZeroModularAbelianVariety()` / `ZeroModularAbelianVariety(k)` | The zero-dimensional abelian variety (of weight `k`). | — |
| `ZeroSubvariety(A)` | The zero subvariety of the abelian variety `A`. | — |

*Worked example: H136E9 (Atkin–Lehner kernel subspace of `J₀(33)`; zero-dimensional varieties).*

### 136.2.6 Creation Using a Label

A short string can create a modular abelian variety. A single integer `N` gives `J₀(N)`;
`"<level>k<weight>"` gives the (possibly motivic) `J₀(N)` of that weight;
`"<level>k<weight><isogeny code>"` gives `JZero(N,k)(iso)`, where the isogeny code `"A"…"Z"`,
`"AA"…"ZZ"`, … maps to `iso = 1, 2, …` The convention matches modular symbols and extends
Cremona's elliptic-curve labels (with the caveat that Cremona's database has random
scrambling for levels 56–450). Omitting the weight defaults to 2.

| Intrinsic | Description |
|-----------|-------------|
| `ModularAbelianVariety(s : -)` | The abelian variety defined by the string `s`. Parameter `Sign` (default 0); parameter `Cremona` (`BoolElt`, default `false`) — if `true`, returns the optimal quotient of `J₀(N)` with Cremona label `s`. |

*Worked example: H136E10 (labels `"37"`, `"37A"`, `"11k4A"`, `"65C"`; `Cremona := true`).*

### 136.2.7 Invariants

Commands to retrieve the base ring, dimension, character of the defining modular form, a
field of definition, the level, the sign, the weights, and a short name.

| Intrinsic | Description |
|-----------|-------------|
| `BaseRing(A)` | The ring `A` is defined over. |
| `Dimension(A)` | The dimension of `A`. |
| `DirichletCharacter(A)` | For `A = A_f` attached to a newform, the Nebentypus character of `f`. Well-defined only up to `Gal(Q̄/Q)` conjugacy. |
| `DirichletCharacters(A)` | List of all Dirichlet characters of the spaces of modular symbols parameterizing `A`. |
| `FieldOfDefinition(A)` | The best known field of definition of `A`. |
| `Level(A)` | An integer `N` such that `A` is a quotient of a power of `J₁(N)`. Need not be minimal; determined by how `A` is represented. |
| `Sign(A)` | The sign of `A` (one of 0, `−1`, `+1`). If `±1`, only the corresponding conjugation eigenspace of homology is computed, so some results are off by a factor of 2. |
| `Weights(A)` | The set of weights of `A`. Need not be a singleton, since direct sums of differently-weighted spaces are allowed. |

*Worked examples: H136E11 (all commands for `J₀(23)`); H136E12 (nontrivial character of `JOne(23)`); H136E13 (`Weights`); H136E14 (`FieldOfDefinition` of base-extended / changed-ring varieties); H136E15 (`FieldOfDefinition` after quotient by 5-torsion gives `Q̄`).*

### 136.2.8 Conductor

For `A` over `Q`, the conductor is computed by factoring `A` into newform varieties `A_f`,
whose conductor is `Nᵈ` (`N` the level of `f`, `d = dim A_f`).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Conductor(A)` | The conductor of `A` (requires `A` over `Q`). When `A = A_f` is attached to a newform of level `N`, the conductor is `Nᵈ`, `d = dim A`. | Factor into newform varieties; multiply conductors `Nᵈ`. |

*Worked example: H136E16 (conductors of `JZero(33)`, `JZero(11)⁵`, `OldSubvariety(JZero(46))`, `JOne(25)`).*

### 136.2.9 Number of Points

For `A` over a field `K`, a divisor and a multiple of `#A(K)` can be computed. When finite,
the multiple is computed using reduction mod primes up to 100. The lower bound is
currently nontrivial only when `A` is a quotient of `J₀(N)`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `NumberOfRationalPoints(A)` | A divisor and a multiple of `#A(K)`, `A` over a field `K`. If `K` is an abelian number field, the Birch–Swinnerton-Dyer conjecture is assumed. | Reduction mod primes ≤ 100 for the multiple. |
| `#A` | The cardinality of `A(K)` when an exact value is known. | — |

*Worked example: H136E17 (`#JZero(11)`, `NumberOfRationalPoints` of `JZero(37)`, `JOne(13)`, `JOne(23)`, `"43B"`).*

### 136.2.10 Inner Twists and Complex Multiplication

For a newform `f`, an *inner twist* is a Dirichlet character `χ` such that the twist of `f`
by `χ` equals a Galois conjugate of `f` (at Fourier coefficients coprime to a fixed integer).
A *CM twist* is a nontrivial `χ` such that `f` twisted by `χ` equals `f`. The parameter
`Proof` (default `false`) controls rigour: `true` uses many more terms of `q`-expansions
(slower) but is still not provably correct — Magma only checks each twist to precision
`10⁻⁵`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CMTwists(A : -)` | A sequence of the CM inner-twist characters of `A = A_f` defined over `BaseRing(A)`. To get all CM twists, base extend to `AlgebraicClosure(RationalField())` first. Parameter `Proof` (default `false`). | Compare `q`-expansions of `f` and its twists to precision `10⁻⁵`. |
| `InnerTwists(A : -)` | A sequence of the inner-twist characters of `A = A_f` defined over `BaseRing(A)`. To get all, base extend to `AlgebraicClosure(RationalField())`. Parameter `Proof` (default `false`). | As above. |

*Worked examples: H136E18 (inner twists of `J₁(13)`, factor of `J₁(23)`, CM of `J₀(32)`, non-CM twist of `J₀(81)`); H136E19 (4-dimensional `A_f` in `J₀(512)` with four inner twists, none CM).*

### 136.2.11 Predicates

Most predicates work in full generality; `IsIsomorphic`, `IsQuaternionic`, `IsSelfDual` are
somewhat limited in domain.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CanDetermineIsomorphism(A, B)` | `true` if isomorphism of `A`, `B` can be determined; if so also `true`/`false` for isomorphism plus an explicit isomorphism (or the reason as a string if undeterminable). | For `A`, `B` simple over `Q` there is an implemented algorithm (most but not all cases). Simple factors of multiplicity one: possible in principle but not programmed. |
| `HasMultiplicityOne(A)` | `true` if the simple factors of `A` appear with multiplicity one. | — |
| `IsAbelianVariety(A)` | `true` if `A` is an abelian variety, i.e. defined over a characteristic-0 ring in which the conductor is invertible, or a finite field whose characteristic does not divide the conductor. (Over `Z` with positive dimension, Raynaud's theorem implies it is not.) | Raynaud's theorem / conductor invertibility. |
| `IsAttachedToModularSymbols(A)` | `true` if the underlying homology is computed using a space of modular symbols (e.g. `J₀(N)`, newform varieties). | — |
| `IsAttachedToNewform(A)` | `true` if `A` is isogenous to a newform variety `A_f`; also returns `A_f` and an explicit isogeny `A_f → A`. | — |
| `IsIsogenous(A, B)` | `true` if `A`, `B` are isogenous (errors if undeterminable). Always determinable when both are over `Q`. | — |
| `IsIsomorphic(A, B)` | `true` if `A`, `B` are isomorphic (with explicit isomorphism). Works when over `Q` with multiplicity-one simple factors; may work otherwise, but can error — use `CanDetermineIsomorphism` to avoid errors. | — |
| `IsOnlyMotivic(A)` | `true` if any modular form attached to `A` has weight > 2. | — |
| `IsQuaternionic(A)` | `true` iff some simple factor of `A` over the base ring has quaternionic multiplication. | — |
| `IsSelfDual(A)` | `true` if `A` is known to be isomorphic to its dual (errors if undecidable). | — |
| `IsSimple(A)` | `true` iff `A` has no proper abelian subvarieties over `BaseRing(A)`. | — |

*Worked examples: H136E20 (`IsAbelianVariety` over `GF(11)`, `Z`, `pAdicRing(3)`); H136E21 (motive vs abelian variety, `IsOnlyMotivic`); H136E22 (`IsAttachedToModularSymbols`, `IsAttachedToNewform`); H136E23 (`IsIsogenous`, `CanDetermineIsomorphism`, `IsIsomorphic`); H136E24 (`HasMultiplicityOne`); H136E25 (`IsQuaternionic`); H136E26 (`IsSelfDual`, `ModularPolarization`); H136E27 (`IsSimple`).*

### 136.2.12 Equality and Inclusion Testing

| Intrinsic | Description |
|-----------|-------------|
| `A eq B` | `true` if `A` and `B` are equal. |
| `A subset B` | `true` if `A` is a subset of `B`. |

*Worked examples: H136E28 (direct product not commutative under equality, but isomorphic); H136E29 (`JZero(11) subset JZero(22)` is `false` though there is an injective map).*

### 136.2.13 Modular Embedding and Parameterization

Every modular abelian variety `A` carries a *modular parameterization* (a surjection from a
modular symbols variety such as `J₀(N)`) and a *modular embedding* (a homomorphism to a
modular symbols variety, guaranteed injective only in the category up to isogeny — i.e.
finite kernel). These two maps completely define `A`.

| Intrinsic | Description |
|-----------|-------------|
| `CommonModularStructure(X)` | Finds modular abelian varieties `J_e`, `J_p` associated to modular symbols, and returns a list of finite-kernel maps from the varieties in sequence `X` to `J_e`, and a list of modular parameterizations from `J_p` to the varieties in `X`. |
| `ModularEmbedding(A)` | A finite-kernel morphism from `A` to a modular symbols variety (an embedding only up to isogeny). |
| `ModularParameterization(A)` | A surjective morphism to `A` from a modular symbols variety. |

*Worked examples: H136E30 (`CommonModularStructure`); H136E31 (modular "embedding" need not be injective; parameterization surjective but not optimal).*

### 136.2.14 Coercion

Coercion creates points on modular abelian varieties from homology basis vectors, from
elements of other varieties, or from modular symbols. Subtleties arise because an abelian
variety is a vector space modulo a lattice embedded arbitrarily in `Qⁿ`.

| Intrinsic | Description |
|-----------|-------------|
| `A ! x` | Coerce `x` into `A`. `x` may be: an element of a modular abelian variety; the integer 0; a sequence (an `Eltseq`, i.e. a linear combination of *integral* homology); a vector on the basis for *rational* homology; or a tuple `<P(X,Y),[u,v]>` defining a modular symbol. |

*Worked examples: H136E32 (sequence vs cusps vs extended reals; higher weight coercion; coercing subvariety elements; creating 0); H136E33 (vector-vs-sequence subtlety when the lattice is `(1/10)Z×Z`).*

### 136.2.15 Modular Symbols to Homology

Modular symbols determine elements of rational homology via the modular parameterization. The
commands below convert modular symbols (variously represented) to vectors on the basis of
rational or integral homology.

| Intrinsic | Description |
|-----------|-------------|
| `ModularSymbolToIntegralHomology(A, x)` | The element of integral homology of `A` associated to the modular symbol `x = P(X,Y){α,β}` (`α,β ∈ P¹(Q)`, `P` homogeneous of degree 2). Returned on the integral homology basis. `x` may be a sequence `[α,β]` or a tuple `<P(X,Y),[α,β]>` with `α,β` in `Cusps()`. |
| `ModularSymbolToRationalHomology(A, x)` | As above, but returned on the rational homology basis. `x` may be a modular symbol, a sequence `[α,β]`, or a tuple `<P(X,Y),[α,β]>`. |

*Worked examples: H136E34 (`ModularSymbolToIntegralHomology` for `J₀(11)`, `J₀(47)`, orders of points); H136E35 (weight 4, default `x^{k−2}`; rational vs integral; coercion as torsion).*

### 136.2.16 Embeddings

`Embeddings` returns a list of embeddings (up to isogeny) from `A` to other abelian
varieties; embeddings at the front have highest priority and are used for intersections,
sums, etc. `AssertEmbedding` prepends an embedding.

| Intrinsic | Description |
|-----------|-------------|
| `Embeddings(A)` | A list of morphisms from `A` into abelian varieties, used for intersections/sums; earlier entries take precedence. (Maps may not really be injective; the modular embedding, injective only on homology, is last.) |
| `AssertEmbedding(~A, phi)` | Prepend the homomorphism `φ` (which must have finite kernel) to `Embeddings(A)`. |

*Worked example: H136E36 (`Embeddings`, `AssertEmbedding`, then a meaningful intersection in `J₀(74)`).*

### 136.2.17 Base Change

`BaseExtend` and `ChangeRing` change the base ring; `BaseExtend` is more restrictive. Abelian
varieties over finite fields support very little (number-of-points is implemented; creating
points/homomorphisms is not).

| Intrinsic | Description |
|-----------|-------------|
| `CanChangeRing(A, R)` | `true` if the base ring of `A` can be changed to `R`, plus the changed variety when possible. |
| `ChangeRing(A, R)` | The variety obtained from `A` with base ring `R`. |
| `BaseExtend(A, R)` | Extend the base ring of `A` to `R` if possible (a more restrictive `ChangeRing`). |

*Worked example: H136E37 (`J₁(13)` over `Q(ζ₇)`, `Q̄`, `R`, `C`, `GF(3)`, `GF(13)`, `Z`, polynomial ring).*

### 136.2.18 Additional Examples

*Worked examples: H136E38 (`ZeroModularAbelianVariety`, `JZero(22)`, higher-weight motives, `JOne(22)`, `+1` quotient); H136E39 (`ModularAbelianVariety(eps)` over `Q`); H136E40 (`JH(N,d)`, diamond operators, `Js(N) = JH(N,φ(N))`); H136E41/H136E42/H136E43 (varieties from `ModularForms`, `ModularSymbols`, cusp forms on `Γ₁(25)`, newforms, and a label `"43B"`).*

---

## 136.3 Homology

The homology `H₁(A, R)` of an abelian variety `A` is a free `R`-module of rank twice
`dim A`. Magma views abelian varieties as complex tori `V/Λ`, giving a canonical
`Λ ≅ H₁(A, Z)`. (If `Sign(A) = ±1`, the homology command gives a `Z`-module of rank `dim A`.)

### 136.3.1 Creation

| Intrinsic | Description |
|-----------|-------------|
| `Homology(A)` | The first integral homology of `A`, of type `ModAbVarHomol`. (If `Sign(A) = ±1`, a `Z`-module of rank `dim A`.) |

*Worked example: H136E44 (homology of `J₀(14)`; `+1` quotient halves the dimension).*

### 136.3.2 Invariants

| Intrinsic | Description |
|-----------|-------------|
| `Dimension(H)` | The dimension of the homology space `H` (twice `dim A`, or `dim A` for sign `±1`). |

*Worked example: H136E45 (`Dimension(Homology(JZero(100)))` with various signs).*

### 136.3.3 Functors to Categories of Lattices and Vector Spaces

Functors from homology/varieties to lattices and vector spaces, defined via `H₁` of the
underlying complex manifold.

| Intrinsic | Description |
|-----------|-------------|
| `IntegralHomology(A)` | The lattice underlying the homology of `A`. (When only `H₁(A,Q)` is computed, the returned lattice is `H₁(A,Z)` written w.r.t. a basis of `H₁(A,Q)` — an integral structure on `H₁(A,Q)`.) |
| `Lattice(H)` | The underlying lattice of the homology space `H` — a free `Z`-module of rank `dim H`. |
| `RationalHomology(A)` | A `Q`-vector space, `H₁(A) ⊗ Q`. |
| `RealHomology(A)` | An `R`-vector space, `H₁(A) ⊗ R`. |
| `RealVectorSpace(H)` | The `R`-vector space of dimension `dim H`. |
| `VectorSpace(H)` | The `Q`-vector space of dimension `dim H`. |

*Worked examples: H136E46 (integral/rational/real homology of `J₀(22)`); H136E47 (level-37 quadratic-character surface); H136E48 (lattice as integral structure on `H₁(A,Q)`).*

### 136.3.4 Modular Structure

If `H` is the homology of a variety attached to modular symbols, it remembers that space.

| Intrinsic | Description |
|-----------|-------------|
| `IsAttachedToModularSymbols(H)` | `true` if `H` is presented as attached to a sequence of spaces of modular symbols. |
| `ModularSymbols(H)` | If so, that sequence of spaces of modular symbols (else an error). |

*Worked examples: H136E49 (`J₀(23)×J₀(11)`); H136E50 (homology of `J₀(23)`, its type `ModAbVarHomol`); H136E51 (a factor of `J₀(37)` not attached to modular symbols; differing lattices).*

---

## 136.4 Homomorphisms

### 136.4.1 Creation

Commands to create the multiplication-by-`n` map; other homomorphism constructions
(Hecke/Atkin–Lehner operators, endomorphism/homomorphism rings) appear in later sections.

| Intrinsic | Description |
|-----------|-------------|
| `IdentityMap(A)` | The identity homomorphism `A → A`. |
| `ZeroMap(A)` | The zero homomorphism `A → A`. |
| `nIsogeny(A, n)` | The multiplication-by-`n` isogeny on `A`, `n` a rational number or integer. |

*Worked example: H136E52 (`IdentityMap`, `ZeroMap`, `nIsogeny(A,3)`, `nIsogeny(A,1/3)` up to isogeny).*

### 136.4.2 Restriction, Evaluation, and Other Manipulations

`Restriction` restricts a homomorphism to a subvariety; `RestrictEndomorphism` gives the
induced endomorphism; `Evaluate(f, φ)` computes `f(φ)` (which may only be up to isogeny if `f`
has denominators), useful with `Kernel` for cutting out subvarieties.

| Intrinsic | Description |
|-----------|-------------|
| `Restriction(phi, B)` | The restriction of `φ` to a morphism from the subvariety `B` to the codomain of `φ` (when it obviously makes sense). |
| `RestrictEndomorphism(phi, B)` | The restriction of `φ` to an endomorphism of `B` (when it makes sense; errors if `B` not left invariant). |
| `RestrictEndomorphism(phi, i)` | For `φ` an endomorphism of `A` and `i : B → A` injective with `i(B)` invariant under `φ`: the induced endomorphism `ψ` of `B`. |
| `RestrictionToImage(phi, i)` | For `i : A → D`, `φ : D → B`: the restriction of `φ` to the image of `i`, an endomorphism of that image. |
| `Evaluate(f, phi)` | The endomorphism `f(φ)` of `A`, `f` a univariate polynomial, `φ` an endomorphism. (Only up to isogeny if `f` has denominators.) |
| `DivideOutIntegers(phi)` | For `φ : A → B`, the largest integer `n` with `ψ = (1/n)·φ` still a homomorphism, returning `ψ` and `n`. |
| `SurjectivePart(phi)` | The surjective homomorphism `π : A → φ(A)` induced by `φ`. |
| `UniversalPropertyOfCokernel(pi, f)` | Given `π : B → C` the cokernel of a morphism (surjective with kernel `K`) and `f : B → D` whose kernel contains `K`, the unique `ψ : C → D` with `π∗ψ = f`. (If only the identity component of `ker π` is contained in `ker f`, then `ψ` is only a morphism up to isogeny — has a denominator.) |

*Worked examples: H136E53 (`Evaluate`/`Kernel` cut a 2-dim subvariety of `J₀(65)`); H136E54 (universal property of the cokernel); H136E55 (`DivideOutIntegers` of `10·T₃` on `J₀(23)`); H136E56 (restriction commands on factors of `J₀(65)`, `SurjectivePart`).*

### 136.4.3 Kernels

The category of abelian varieties is not abelian: kernels of `φ : A → B` are usually
disconnected, fitting in `0 → C → ker(φ) → G → 0` with `C` an abelian variety and `G` a
finite group, both over the field of `φ`. `ConnectedKernel` returns `C`;
`ComponentGroupOfKernel` returns `G` as a subgroup of `A/C`; `Kernel` returns a finite
subgroup of `A` mapping onto `G` (non-canonical when `C ≠ 0`).

| Intrinsic | Description |
|-----------|-------------|
| `ComponentGroupOfKernel(phi)` | The component group `G` of `ker(φ)`. |
| `ConnectedKernel(phi)` | The connected component `C` of `ker(φ)` and a morphism `C → domain(φ)`. |
| `Kernel(phi)` | A finite subgroup `G` of `A`, an abelian variety `C` with `ker(φ) = f(C) + G`, and an injective map `f : C → A`. If `C = 0`, `G` has the same field of definition as `φ`; otherwise `G` is only known over the algebraic closure. |

*Worked example: H136E57 (kernel of `T₂² − 3` on `J₀(65)`: extension of a surface by `(Z/2Z)⁴`; `ComponentGroupOfKernel` vs `Kernel`, `AmbientVariety`).*

### 136.4.4 Images

| Intrinsic | Description |
|-----------|-------------|
| `A @ phi` / `phi(A)` | The image of the abelian variety `A` under `φ` (when it makes sense: `A` the domain, or `dim A = 0`, or an embedding of `A` has codomain `domain(φ)`). |
| `G @ phi` / `phi(G)` | The image of the subgroup `G` under `φ` (when it makes sense). |
| `Image(phi)` | The image `C` of `φ` (an abelian subvariety of the codomain), a morphism `C → codomain(φ)`, and a surjection `domain(φ) → C`. |
| `G @@ phi` | A finite group whose image under `φ` equals the subgroup `G`. If `φ` has finite kernel this is the exact inverse image; otherwise a (non-canonical) group from a torsion inverse-image choice per generator. |

*Worked example: H136E58 (image of `J₀(37)` under `T₂` is an elliptic curve; `@`, `@@`, 2-torsion behaviour).*

### 136.4.5 Cokernels

| Intrinsic | Description |
|-----------|-------------|
| `Cokernel(phi)` | The cokernel of `φ` and a morphism from `codomain(φ)` to the cokernel. |

*Worked example: H136E59 (2-dim quotient of `J₀(33)` via `T₂ + 2` and `Cokernel`).*

### 136.4.6 Matrix Structure

Homomorphisms are stored via the linear maps they induce on homology.

| Intrinsic | Description |
|-----------|-------------|
| `Matrix(phi)` | The matrix on the chosen basis of rational homology defining `φ`. |
| `Eltseq(phi)` | The `Eltseq` of the underlying matrix (a sequence of integers or rationals). |
| `Ncols(phi)` | The number of columns (dimension of homology of the codomain). |
| `Nrows(phi)` | The number of rows (dimension of homology of the domain). |
| `Rows(phi)` | The sequence of rows of the matrix. |
| `IntegralMatrix(phi)` / `IntegralMatrixOverQ(phi)` | The matrix defining `φ` w.r.t. integral homology (over `Z`, resp. over `Q`). |
| `RealMatrix(phi)` | The matrix defining `φ` w.r.t. real homology. |

*Worked example: H136E60 (each command for `T₂` on `J₀(23)`).*

### 136.4.7 Arithmetic

Composition, addition, subtraction, exponentiation of homomorphisms.

| Intrinsic | Description |
|-----------|-------------|
| `Inverse(phi)` | The inverse of `φ` and an integer `d` with `d∗φ⁻¹` a morphism. (For `φ` an isogeny, the inverse up to isogeny; the actual inverse if `deg φ = 1`.) |
| `phi * psi` | Composition of `φ` and `ψ`. |
| `a * phi` / `phi * a` | Product of the rational/integer `a` with `φ` (may be a homomorphism only up to isogeny). |
| `phi * psi` / `psi * phi` (matrices) | The product of the defining matrix of `φ` and a matrix `ψ` (in either order). |
| `phi ^ n` | The `n`-fold composition of the endomorphism `φ`. `n = −1` gives the inverse (must be an isogeny). |
| `phi + psi` / `phi - psi` | Sum / difference of homomorphisms `φ` and `ψ`. |
| `n + phi` / `phi + n` / `n - phi` / `phi - n` | Sum / difference of multiplication-by-`n` and the endomorphism `φ` (`n` integer or rational). |
| `phi + psi` / `psi + phi` / `phi - psi` / `psi - phi` (matrices) | Sum / difference of the defining matrix of `φ` and a matrix `ψ`. |

*Worked example: H136E61 (arithmetic of `T₂`, `T₃` on `J₀(23)`, including `Inverse`, `1/3*φ`, `φ^4`, `φ^(-4)`).*

### 136.4.8 Polynomials

Characteristic and minimal polynomials of endomorphisms. Each requires `φ` to be a genuine
homomorphism (not just up to isogeny). Special cases (e.g. construction info, Deligne bounds)
can speed up the computation versus working with the matrix directly.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CharacteristicPolynomial(phi)` | The characteristic polynomial of the endomorphism `φ`. | Sometimes uses extra info about `φ` (construction, Deligne bounds) for speed. |
| `FactoredCharacteristicPolynomial(phi)` | The factorization of the characteristic polynomial of `φ` (cached). | As above. |
| `MinimalPolynomial(phi)` | The minimal polynomial of the endomorphism `φ`. | As above. |

*Worked example: H136E62 (each command for `T₂` on `J₀(66)`).*

### 136.4.9 Invariants

| Intrinsic | Description |
|-----------|-------------|
| `Domain(phi)` / `Codomain(phi)` | The domain / codomain of `φ`. |
| `Degree(phi)` | The degree of `φ`, i.e. the cardinality of `ker(φ)`; 0 if the kernel is infinite. |
| `Denominator(phi)` | The smallest positive integer `n` with `n∗φ` a homomorphism (the denominator of the matrix). |
| `ClearDenominator(phi)` | The morphism `n∗φ` where `n` is smallest positive making it a genuine homomorphism. |
| `FieldOfDefinition(phi)` | A field over which `φ` is defined (not guaranteed minimal). |
| `Nullity(phi)` / `Rank(phi)` | The dimension of the kernel of `φ`. |
| `Trace(phi)` | The trace of any matrix representing `φ` on integral homology (e.g. `2n·dim(A)` for multiplication by `n`). |

*Worked example: H136E63 (`NaturalMap(JZero(11),JZero(33))`; domain, codomain, degree, denominator, field of definition, nullity, rank, traces).*

### 136.4.10 Predicates

| Intrinsic | Description |
|-----------|-------------|
| `IsMorphism(phi)` | `true` iff `φ` is a morphism (not just up to isogeny). |
| `OnlyUpToIsogeny(phi)` | `true` if `φ` is a morphism only up to isogeny (some `n∗φ` is a morphism, but `φ` is not). |
| `HasFiniteKernel(phi)` | `true` if `ker(φ)` is finite. |
| `IsInjective(phi)` | `true` if `φ` is an injective homomorphism. |
| `IsSurjective(phi)` | `true` if `φ` is surjective. |
| `IsEndomorphism(phi)` | `true` if domain and codomain of `φ` are equal. |
| `IsInteger(phi)` | `true` if `φ` is multiplication by an integer `n` (returns `n`). |
| `IsIsogeny(phi)` | `true` if `φ` is a surjective homomorphism with finite kernel. (Agrees with Milne, not Silverman; an equivalence relation.) |
| `IsIsomorphism(phi)` | `true` if `φ` is an isomorphism. |
| `IsOptimal(phi)` | `true` if `φ` is an optimal quotient map (surjective with connected kernel). |
| `IsHeckeOperator(phi)` | `true` if `φ` was computed via `HeckeOperator`, also returning the index `n`. |
| `IsZero(phi)` | `true` if `φ` is the zero morphism. |
| `phi eq psi` | `true` if `φ` and `ψ` are equal. |
| `n eq phi` / `phi eq n` | `true` if `φ` equals multiplication by the integer `n`. |
| `phi in X` | `true` if `φ` is in the list `X` of homomorphisms. |

*Worked example: H136E64 (`T₂−1` on `J₀(65)`: all the predicates and equalities).*

---

## 136.5 Endomorphism Algebras and Hom Spaces

### 136.5.1 Creation

For `A`, `B` modular abelian varieties one can form the finite-rank free abelian group
`Hom(A,B)`, or the `Q`-vector space of homomorphisms up to isogeny, or the endomorphism
algebra. Creation is lazy — `Hom(A,B)` is not computed until rank/basis is requested.

| Intrinsic | Description |
|-----------|-------------|
| `Hom(A, B)` / `Hom(A, B, oQ)` | The group of homomorphisms `A → B`. If `oQ = true`, the vector space generated by them (homomorphisms up to isogeny). |
| `End(A)` / `End(A, oQ)` | The endomorphism ring of `A`. If `oQ = true`, the endomorphism algebra. |
| `BaseExtend(H, R)` | The space `H ⊗ R`, `H` a group of homomorphisms, `R` the rationals or integers. For `R = Q`, the homomorphisms up to isogeny. |
| `HeckeAlgebra(A)` | The Hecke algebra of `A`: a commutative subring of `End(A)` generated by Hecke operators. For a general `A` with parameterization `π : J → A` and embedding `e : A → J`, it is `e∗T∗π` (`T` the Hecke ring of `J`); may differ from the naive expectation by a finite index (e.g. for `J₁(N)` as a quotient of `J₀(N)`). |

*Worked example: H136E65 (`Hom`, `End`, `BaseExtend`, `HeckeAlgebra` for `J₀(11)`, `J₀(33)`).*

### 136.5.2 Subgroups and Subrings

Subgroups of `Hom(A,B)` and subrings of `End(A)` can be formed; Magma computes saturations.

| Intrinsic | Description |
|-----------|-------------|
| `Subgroup(X)` | The group of homomorphisms `A → B` generated by the homomorphisms in the sequence `X`. |
| `Subgroup(X, oQ : -)` | As above; parameter `IsBasis` (`BoolElt`, default `false`) assumes the elements of `X` are a basis. If `oQ = true`, the vector space generated. |
| `Subring(X)` / `Subring(X, oQ)` | The ring of endomorphisms generated by the sequence `X` (need not contain unity). If `oQ = true`, the algebra generated. |
| `Subring(phi)` | The ring of endomorphisms generated by `φ` (need not contain unity). |
| `Saturation(H)` | The saturation of the homomorphism group `H`: the subgroup `H'` of `Hom(A,B)` containing `H` with finite index such that `Hom(A,B)/H'` is torsion free. |
| `RingGeneratedBy(H)` | The ring of endomorphisms generated by the homomorphisms in the group `H`. |

*Worked examples: H136E66 (`Saturation`: levels `N ≤ 60` where the Hecke algebra of `J₀(N)` is not saturated in the full endomorphism ring); H136E67 (`Subgroup`); H136E68 (`Subring(T₂)` on `J₀(100)`); H136E69 (`T₂` and the main Atkin–Lehner involution generate a rank-10 ring).*

### 136.5.3 Pullback and Pushforward of Hom Spaces

A homomorphism induces a map between hom spaces.

| Intrinsic | Description |
|-----------|-------------|
| `Pullback(H, phi)` | For `H ⊆ Hom(A,B)` and `φ : B → C`: the image of `H` in `Hom(A,C)` under `f ↦ f∗φ`. |
| `Pullback(phi, H)` | For `H ⊆ Hom(A,B)` and `φ : C → A`: the image of `H` in `Hom(C,B)` under `f ↦ φ∗f`. |
| `Pullback(phi, H, psi)` | For `H ⊆ Hom(A,B)`, `φ : C → A`, `ψ : B → D`: the ring of homomorphisms `φ∗f∗ψ`, `f ∈ H`. |

*Worked example: H136E70 (`Pullback` along natural maps among `J₀(11)`, `J₀(22)`, `J₀(33)`).*

### 136.5.4 Arithmetic

| Intrinsic | Description |
|-----------|-------------|
| `H1 + H2` | The subgroup of `Hom(A,B)` generated by `H₁` and `H₂`. |
| `H1 meet H2` | The intersection of `H₁` and `H₂`. |

### 136.5.5 Quotients

| Intrinsic | Description |
|-----------|-------------|
| `Index(H2, H1)` | The index of `H₁` in `H₂` (both subgroups of `Hom(A,B)`). If `H₁ ⊆ H₂`, the cardinality of `H₂/H₁` (0 if infinite); otherwise the generalized lattice index `[H₁+H₂ : H₁]/[H₁+H₂ : H₂]` (errors if `H₂` lacks finite index in `H₁+H₂`). |
| `Quotient(H2, H1)` / `H2 / H1` | The abelian-group quotient `H₂/H₁`, a map `H₂ → H₂/H₁`, and a lifting map back. |

*Worked examples: H136E71 (`Dimension`, `meet`, `+` for hom spaces of `J₀(33)`); H136E72 (subgroup of `End(J₀(54))` of infinite index; `Index`, `Quotient`, `Saturation`).*

### 136.5.6 Invariants

Discriminants of hom spaces are computed with respect to the trace pairing (the trace of
endomorphisms acting on homology). Discriminants of Hecke algebras are notable because they
relate to congruences between eigenforms.

| Intrinsic | Description |
|-----------|-------------|
| `Domain(H)` / `Codomain(H)` | The domain / codomain of the homomorphisms in `H`. |
| `FieldOfDefinition(H)` | A field over which all homomorphisms in `H` are defined (not guaranteed minimal). |
| `Discriminant(H)` | The discriminant of `H` w.r.t. the trace pairing (trace of endomorphisms on homology, not left multiplication; so the Hecke-algebra discriminant is `2ᵈ` times larger when sign is 0, `d = dim A`). Over `Q`, returns the discriminant of the lattice of homomorphisms in `H`. |

*Worked example: H136E73 (`Discriminant` of `Hom(J₀(11),J₀(33))` and `End(J₀(11))` over `C`; the prime `389` dividing the Hecke-algebra discriminant of `J₀(389)`).*

### 136.5.7 Structural Invariants

| Intrinsic | Description |
|-----------|-------------|
| `Basis(H)` / `Generators(H)` | A basis for `H`. |
| `Dimension(H)` / `Rank(H)` | The rank of `H` as a `Z`-module or `Q`-vector space. |
| `Ngens(H)` | The number of generators of `H`. |
| `H . i` | The `i`-th generator of `H`. |

*Worked example: H136E74 (each command for `Hom(J₀(11), J₀(33))`).*

### 136.5.8 Matrix and Module Structure

Lattices, vector spaces, matrix algebras and matrix spaces from subspaces `H ⊆ Hom(A,B)`.

| Intrinsic | Description |
|-----------|-------------|
| `Lattice(H)` / `VectorSpace(H)` | A lattice or vector space with basis from the components (`Eltseq`s) of the matrices of a basis of `H`. |
| `MatrixAlgebra(H)` | The matrix algebra generated by the underlying matrices of all elements of `H` acting on homology. |
| `RMatrixSpace(H)` | The matrix space whose basis is the generators of `H`. |
| `RModuleWithAction(H)` | A module over the ring `R` generated by `H` with the action of `H` (where `H` must be a ring of endomorphisms). |
| `RModuleWithAction(H, p)` | A module over `H ⊗ F_p` with the action of `H ⊗ F_p` (`H` a ring of endomorphisms not yet tensored with `Q`). |

*Worked examples: H136E75 (`Lattice`, `RMatrixSpace`, `VectorSpace` for `Hom(J₀(11),J₀(33))`; `MatrixAlgebra`, `RModuleWithAction` for `End(J₀(22))`); H136E76 (`MatrixAlgebra` computes the algebra generated by `H`).*

### 136.5.9 Predicates

| Intrinsic | Description |
|-----------|-------------|
| `IsRing(H)` | `true` if `H` is a ring (need not contain unity). |
| `IsField(H)` | `true` if `H` is a field (returns the field, a map to `H`, and a map from `H`). |
| `IsCommutative(H)` | `true` iff `H` is a commutative ring. |
| `IsHeckeAlgebra(H)` | `true` if `H` was constructed via `HeckeAlgebra`. |
| `IsOverQ(H)` | `true` if `H` is a `Q`-vector space (homomorphisms up to isogeny) rather than just a `Z`-module. |
| `IsSaturated(H)` | `true` if `H` equals its saturation (`Hom(A,B)/H` is torsion free). |
| `H1 eq H2` | `true` if `H₁` and `H₂` are equal. |
| `H1 subset H2` | `true` if `H₁`, `H₂` are subgroups of a common `Hom(A,B)` and `H₁ ⊆ H₂`. |

*Worked example: H136E77 (endomorphism ring of `J₀(33)`: ring/field/commutative/Hecke/over-Q/saturated; comparison with the Hecke algebra; `J₀(23)` Hecke algebra is a field).*

### 136.5.10 Elements

| Intrinsic | Description |
|-----------|-------------|
| `H ! x` | Coerce `x` into the hom space `H`. `x` must be a homomorphism, an integer/rational, or a matrix coercible into the matrix space of `H` base-extended to `Q`. |

*Worked example: H136E78 (coercing a matrix into `End(J₀(22))`).*

---

## 136.6 Arithmetic of Abelian Varieties

### 136.6.1 Direct Sum

Arbitrary finite direct sums are allowed. `A+B` is **not** the direct sum (it denotes the sum
inside a common ambient); use `DirectSum`/`*`.

| Intrinsic | Description |
|-----------|-------------|
| `DirectSum(A, B)` / `DirectProduct(A, B)` / `A * B` | The direct sum `D` of `A`, `B`, with the embeddings `A → D`, `B → D` and projections `D → A`, `D → B`. (Cannot direct-sum varieties of different signs.) |
| `DirectSum(X)` / `DirectProduct(X)` | The direct sum `D` of the sequence `X`, with lists of embeddings into `D` and projections from `D`. (Different signs not allowed.) |
| `A ^ n` | The direct sum of `n` copies of `A`. `n = 0` gives the zero subvariety; `n < 0` gives the `(−n)`-th power of the dual of `A`. |

*Worked example: H136E79 (products of factors of `J₀(65)`, product with the weight-4 motive `J₁(11,4)`; common over-ring for differing base rings).*

### 136.6.2 Sum in an Ambient Variety

`A+B` is the sum of `A` and `B` inside a common ambient (direct only if `A ∩ B = 0`).

| Intrinsic | Description |
|-----------|-------------|
| `A + B` | The sum of the images of `A`, `B` in a common ambient. |
| `SumOf(X)` | The sum of the varieties in the sequence `X`. |
| `SumOfImages(phi, psi)` | The sum `D` of the images of `φ`, `ψ` in their common codomain, a morphism `D → codomain`, and morphisms from each domain to `D`. (If codomains differ, morphisms are replaced by maps into a direct sum.) |
| `SumOfMorphismImages(X)` | As `SumOfImages` but for a list `X` of morphisms. |
| `FindCommonEmbeddings(X)` | Returns `true` and a list of embeddings into a common variety if found using `Embeddings(A)` for all `A` in `X`. |

### 136.6.3 Intersections

Intersection requires choosing embeddings of both varieties into a common ambient. The
algorithm computes the kernel of a suitable homomorphism (e.g. `ker(f − g)` for injections
`f, g` into a common `C`). Intersections, like kernels, are often not abelian varieties
(extension of an abelian variety by a finite component group).

| Intrinsic | Description |
|-----------|-------------|
| `A meet B` / `Intersection(X)` | A finite lift `G` of the component group, the connected component `C`, and a map from the variety containing `C` to that containing `G` (relevant intersection `C+G`). Elements of `X` are replaced by their modular-embedding images; all must be embedded in the same variety. |
| `IntersectionOfImages(X)` | For a sequence `X` of morphisms into a common variety: a finite lift `G` of the component group, the connected component `C`, and the connecting map. The morphisms need not be injective. |
| `ComponentGroupOfIntersection(A, B)` / `ComponentGroupOfIntersection(X)` | The component group of the intersection of `A`, `B` (or the varieties in `X`). |

*Worked examples: H136E80 (intersection of the three newform subvarieties of `J₀(65)` is `Z/2Z`; non-finite intersections, `IntersectionOfImages`); H136E81 (failure of multiplicity one for `J₀(431)` — `[Kil02]`; congruent eigenforms with trivial intersection).*

### 136.6.4 Quotients

| Intrinsic | Description |
|-----------|-------------|
| `A / B` | The quotient of `A` by a natural image `B'` of `B` (image of `B` under modular embedding composed with the parameterization to `A`). |
| `Cokernel(phi)` | The cokernel of `φ` and a morphism from `codomain(φ)` to the cokernel. |

*Worked example: H136E82 (2-dim quotient of `J₀(33)` via `T₂`).*

---

## 136.7 Decomposing and Factoring Abelian Varieties

By the Poincaré reducibility theorem, every abelian variety is isogenous to a product of
simple subvarieties. Over `Q`, a modular abelian variety is isogenous to a product of simple
varieties `A_f` attached to newforms.

### 136.7.1 Decomposition

| Intrinsic | Description |
|-----------|-------------|
| `Decomposition(A)` | A sequence `[Bᵢ]` of simple modular abelian varieties whose product is isogenous to `A`, each equipped with an embedding into `A` whose images sum to `A`. (The embedding is the first element of `Embeddings(Bᵢ)`.) |
| `A(n)` | The `n`-th factor in `Decomposition(A)`. |

*Worked example: H136E83 (decompose `J₀(37)×J₀(22)`, find an `J₀(11)`-isogenous factor's embedding).*

### 136.7.2 Factorization

| Intrinsic | Description |
|-----------|-------------|
| `Factorisation(A)` / `Factorization(A)` | Pairwise non-isogenous simple newform varieties `A_f` whose product (with multiplicities) is isomorphic to `A`. Returns a list of pairs `<B, [φ, …]>`, `B` an isogeny-simple variety and `[φ, …]` maps `B → A` (length = multiplicity), with the product of images isogenous to `A` and the `B` pairwise non-isogenous. (For canonical embeddings use `Decomposition`.) |

*Worked example: H136E84 (factorization of `J₀(37)×J₀(22)` with explicit maps).*

### 136.7.3 Decomposition with respect to an Endomorphism or a Commutative Ring

Uses elements of a commutative subring of endomorphisms to decompose `A` via kernels
(analogous to generalized eigenspaces).

| Intrinsic | Description |
|-----------|-------------|
| `DecomposeUsing(R)` | Decompose `A` using the commutative ring of endomorphisms generated by the hom space `R`. |
| `DecomposeUsing(phi)` | Decompose `A` using the endomorphism `φ`. |

*Worked example: H136E85 (`DecomposeUsing(T₂)` and `DecomposeUsing(W)` on `J₀(100)`).*

### 136.7.4 Additional Examples

*Worked example: H136E86 (decomposition and `Factorization` of `J₀(46)` as `E × A × B`, with explicit embeddings into `J₀(46)`).*

---

## 136.8 Building blocks

The variety `A_f/Q` attached to a newform `f` is isogenous over `Q̄` to a power `B_f^r` of a
simple variety `B_f`, the *building block*. This section (contributed by Jordi Quer; see
**[Que09]** for theory and tables) computes the field of definition of `B_f` and its
endomorphism algebra. Its functions take spaces of modular symbols which **must have sign +1**
(e.g. `ModularSymbols(N,2,1)`), and the space is expected to be cuspidal, new and irreducible
over `Q`.

### 136.8.1 Background and Notation

For `f = Σ aₙqⁿ ∈ S₂ⁿᵉʷ(N, ε)` of weight 2, let `E` be the field generated by the `aₙ` and
`F` the field generated by `μ_p := a_p²/ε(p)` (`p ∤ N`). Then `dim A_f = [E:Q]` and
`End_Q(A_f) ⊗ Q ≅ E`. CM means a nontrivial (order-2) `χ` with `a_p = χ(p)a_p`; then `B_f` is
a CM elliptic curve. Assuming no CM: `F` is totally real, `E/F` abelian; `End(B_f) ⊗ Q` is a
division algebra with centre `F`, either `F` itself (then `dim B_f = [F:Q]`, `r = [E:F]`) or a
quaternion algebra over `F` (then `dim B_f = 2[F:Q]`, `r = [E:F]/2`). The *inner twists* `χ_s`
(for `s ∈ G_F`) satisfy `a_p^s = χ_s(p)a_p`. The *quadratic degree characters* `ψ_s` are
related by `χ_s(p) = ψ_s(p)√(ε(p))^s/√(ε(p))`. `K_P/Q` is the field fixed by the kernel of
`δ : G_Q → F*/F*²` (`Frob_p ↦ μ_p`), returned by `DegreeMap`. The Brauer class `γ_ε ∈ Br(Q)[2]`
governs `End⁰(B_f)`, computed by `BrauerClass`; the obstruction to descending `B_f` to `K_P`
lies in `Br(K_P)[2]`, computed by `ObstructionDescentBuildingBlock`.

| Intrinsic | Description |
|-----------|-------------|
| `BoundedFSubspace(epsilon, k, degrees)` | A sequence of the irreducible subspaces of the weight-`k`, Nebentypus-`ε` modular symbols corresponding to non-CM newforms for which `[F:Q]` is in the sequence `degrees`. |
| `HasCM(M : -)` / `IsCM(M : -)` | `true` iff the variety attached to the modular symbols space `M` has complex multiplication. Parameter `Proof` (`BoolElt`, default `false`); for level > 100 an unproven bound is used unless `Proof := true`. |
| `InnerTwists(A : -)` / `InnerTwists(M : -)` | The inner twists of the newform corresponding to the space (Dirichlet characters with `χ_s(p) = a_p^s/a_p`). Space must be irreducible over `Q`, sign +1, non-CM. Parameter `Proof` (default `false`); non-rigorous bound for level > 100 unless `Proof := true`; twists only checked to precision `10⁻⁵`. |
| `DegreeMap(M : -)` | The homomorphism `δ : G_Q → F*/F*²` attached to `M` (new, irreducible over `Q`, sign +1). Returns `F`; a sequence of tuples `<tᵢ, fᵢ>` (`tᵢ ∈ Q` quadratic discriminants giving a basis `σᵢ` of `Gal(K_P/Q)`, with `δ(σᵢ) = fᵢ`); and `δ`. Parameter `Proof` (default `false`). |
| `BrauerClass(M)` | For `M` corresponding to a newform `f`: the Brauer class of `End⁰(A_f)` (or `M_f`). If the endomorphism algebra is a number field `F`, an empty sequence; if a quaternion algebra over `F`, the sequence of places of `F` that ramify. `M` irreducible over `Q`, sign +1, non-CM. |
| `ObstructionDescentBuildingBlock(M)` | The obstruction to descending the building block `B_f` to `K_P`, an element of `Br(K_P)[2]`, given as the sequence of places of `K_P` where the class is not locally trivial. `M` irreducible over `Q`, sign +1, non-CM. |

*Worked example: H136E87 (level 28, character of order 6: `BoundedFSubspace`, `DegreeMap`, `BrauerClass = [2,3]`, `ObstructionDescentBuildingBlock`; finds `K_P = Q(√−7)`).*

---

## 136.9 Orthogonal Complements

### 136.9.1 Complements

Finds a complement of an abelian subvariety (existence guaranteed by Poincaré reducibility);
Magma uses the module-theoretic structure of the ambient.

| Intrinsic | Description |
|-----------|-------------|
| `Complement(A : -)` | The complement of the image of `A` under its first embedding (the first map of `Embeddings(A)`). Parameter `IntPairing` (`BoolElt`, default `false`) — if `true`, use the intersection pairing to compute the homology complement. |
| `ComplementOfImage(phi : -)` | For `φ : A → B`: a choice of complement `C` with `φ(A) + C = B` and `φ(A) ∩ C` finite, plus an embedding `C → B`. Parameter `IntPairing` (default `false`). |

*Worked example: H136E88 (complement of a factor of `J₀(33)`; complement of the image of a map `J₀(11) → J₀(33)`).*

### 136.9.2 Dual Abelian Variety

For `A` with injective modular map `A → J` (`J` self-dual, e.g. `J₀(N)`), the dual is computed
by finding a complement `B` of `A` in `J` whose homology is orthogonal under the intersection
pairing (often done, e.g. for newforms, via Hecke-module structure instead); then `J/B` is the
dual.

| Intrinsic | Description |
|-----------|-------------|
| `IsDualComputable(A)` | `true` if the dual of `A` can be computed, plus the dual; otherwise `false` and a message. |
| `Dual(A)` | The dual abelian variety of `A` (the modular map to a modular symbols variety must be injective). |
| `ModularPolarization(A)` | The polarization on `A` induced by pullback of the theta divisor. |

*Worked example: H136E89 (dual of a level-43 surface is isomorphic to itself; level-69 surface is not — natural map to dual is a degree-484 polarization).*

### 136.9.3 Intersection Pairing

The intersection-pairing matrix on homology w.r.t. the fixed rational/integral basis. If `A`
is not a modular symbols variety, the pairing is *pulled back from the codomain of the modular
embedding* (which may be unexpected). Currently implemented only for weight 2.

| Intrinsic | Description |
|-----------|-------------|
| `IntersectionPairing(H)` | The intersection pairing matrix on the basis of the homology `H`. |
| `IntersectionPairing(A)` | The intersection pairing matrix on the rational homology basis of `A`, pulled back via the modular embedding. |
| `IntersectionPairingIntegral(A)` | As above, on the integral homology basis. |

*Worked example: H136E90 (intersection pairings of `J₀(11)`, `J₀(33)`; `33A` pulled back has determinant 9 rather than 1).*

### 136.9.4 Projections

For `φ : A → B`, computes a projection `π` in `End(B) ⊗ Q` onto `φ(A)` with `π² = π`
(non-canonical unless required to respect the intersection pairing).

| Intrinsic | Description |
|-----------|-------------|
| `ProjectionOnto(A : -)` / `ProjectionOntoImage(phi : -)` | For `φ : A → B`, a projection onto `φ(A)` as an element of `E ⊗ Q` (`E = End(B)`). Parameter `IntPairing` (`BoolElt`, default `false`) — if `true`, `π` is required to respect the intersection pairing (uniquely determining it). |

*Worked examples: H136E91 (`ProjectionOnto("33A")`, `π² = π`); H136E92 (`ProjectionOntoImage` for `J₀(11) → J₀(44)`).*

### 136.9.5 Left and Right Inverses

Left/right inverses in the category up to isogeny, or scaled by a minimal integer to be a
genuine homomorphism. A right inverse of finite-degree `φ` is computed via the projection onto
its image; a left inverse of surjective `φ : A → B` via the complement of `ker(φ)`.

| Intrinsic | Description |
|-----------|-------------|
| `LeftInverse(phi : -)` | For surjective `φ : A → B`, a minimal-degree `ψ : B → A` (up to isogeny) with `ψ∗φ = id_B`, plus an integer `d` with `d∗ψ` a homomorphism. Parameter `IntPairing` (default `false`). |
| `LeftInverseMorphism(phi : -)` | For surjective `φ : A → B`, a minimal-degree `ψ : B → A` with `ψ∗φ` = multiplication by an integer. Parameter `IntPairing` (default `false`). |
| `RightInverse(phi : -)` | For `φ : A → B` with finite kernel, a map `ψ : B → A` (up to isogeny) with `φ∗ψ = id_A`. Parameter `IntPairing` (default `false`). |
| `RightInverseMorphism(phi : -)` | For `φ : A → B` with finite kernel, a minimal-degree `ψ : B → A` with `φ∗ψ` = multiplication by an integer. Parameter `IntPairing` (default `false`). |

*Worked example: H136E93 (difference of degeneracy maps `J₀(11)→J₀(33)`, Shimura subgroup; right and left inverses).*

### 136.9.6 Congruence Computations

The congruence modulus and modular degree each measure congruences between an abelian variety
and others. If a prime divides the modular degree it divides the congruence modulus (converse
need not hold).

| Intrinsic | Description |
|-----------|-------------|
| `CongruenceModulus(A)` | For `A = A_f` attached to a newform `f`, the congruence modulus of `f` in `S₂(N, ε)`: the order of `S_k(N,ε;Z)/(W + W⊥)`, where `W` is the intersection of `S_k(N,ε;Z)` with the span of the Galois conjugates of `f`. Measures congruences between `f` and non-conjugate forms in the Petersson complement. |
| `ModularDegree(A)` | The modular degree of `A`: the square root of the degree of the map `A → A'` (the dual). (When no algorithm for `A'` exists, prints a message and computes the square of the degree of the embedding-composed-with-parameterization. When some weight > 2, the square root is not taken.) |

*Worked example: H136E94 (modular degree ≠ congruence modulus for level-54 elliptic curve — `[AS04]`; both equal 4 for a level-65 surface).*

---

## 136.10 New and Old Subvarieties and Natural Maps

### 136.10.1 Natural Maps

For `M | N`, natural maps in both directions between `J₀(N)` and `J₀(M)` (and `J₁`, etc.)
exist for each divisor `t = N/M`, corresponding to `f(q) ↦ f(qᵗ)` and duals. Reduces to
defining natural maps between modular symbols varieties (since each `A` has `A → J_e`,
`J_p → A`).

| Intrinsic | Description |
|-----------|-------------|
| `NaturalMap(A, B, d)` | The natural map `A → B` induced (in a possibly complicated way) from `f(q) ↦ f(qᵈ)`. (Defined as the zero map when the modular forms of `A` and `B` are unrelated.) |
| `NaturalMap(A, B)` | The natural map `A → B` induced by the identity on modular forms (or the zero map if none). |
| `NaturalMaps(A, B)` | A sequence of the natural maps `A → B` for all divisors `d` of `level(A)/level(B)` or `level(B)/level(A)`. |

*Worked example: H136E95 (natural maps among `J₀(11)`, `J₀(22)`, `J₀(33)` and powers; `NaturalMaps` returns one map per divisor of the level quotient).*

### 136.10.2 New Subvarieties and Quotients

The `r`-new subvariety is the intersection of kernels of all natural maps to varieties of
level `N/r`; the new subvariety intersects these over prime divisors `r`. The `r`-new quotient
is the quotient by the sum of images of natural maps from level `N/r`.

| Intrinsic | Description |
|-----------|-------------|
| `NewSubvariety(A, r)` | The `r`-new subvariety of `A`. |
| `NewSubvariety(A)` | The new subvariety of `A`. |
| `NewQuotient(A, r)` | The `r`-new quotient of `A`. |
| `NewQuotient(A)` | The new quotient of `A`. |

*Worked example: H136E96 (new/old subvariety dimensions of `J₀(33)`).*

### 136.10.3 Old Subvarieties and Quotients

The `r`-old subvariety is the sum of images of all natural maps from level `N/r`; the old
subvariety sums these over divisors. The `r`-old quotient is `A` by its `r`-new subvariety.

| Intrinsic | Description |
|-----------|-------------|
| `OldSubvariety(A, r)` | The `r`-old subvariety of `A`. |
| `OldSubvariety(A)` | The old subvariety of `A`. |
| `OldQuotient(A, r)` | The `r`-old quotient of `A`. |
| `OldQuotient(A)` | The old quotient of `A`. |

*Worked example: H136E97 (old subvariety and old quotient of `J₀(100)` both dimension 6; new subvariety/quotient intersect in `Z/12Z × Z/12Z`).*

---

## 136.11 Elements of Modular Abelian Varieties

Torsion points are represented as follows: for `A` over `C`, `A(C) ≅ H₁(A,R)/H₁(A,Z)` with
torsion `≅ H₁(A,Q)/H₁(A,Z)`; a torsion element is a representative of `H₁(A,Q)`. Elements of
`H₁(A,R)` (floating-point) represent certain infinite-order points but are not exactly known.
Elements are created independently only by coercion (Section 136.2.14).

### 136.11.1 Arithmetic

| Intrinsic | Description |
|-----------|-------------|
| `a * x` / `x * a` | Product of the integer/rational/real `a` by the element `x`. |
| `x + y` | The sum of elements `x` and `y`. |
| `x - y` | The difference `x − y`. |

*Worked example: H136E98 (`ker(T₃ − 5)` of order 400 in `J₀(23)`; arithmetic on its elements).*

### 136.11.2 Invariants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Order(x)` | The order of `x`, if known exactly (errors otherwise). | — |
| `ApproximateOrder(x)` | The exact order if `x` is known exactly as torsion; otherwise the order of a torsion-point approximation of `x` obtained via continued fractions. | Continued-fraction approximation. |
| `Degree(x)` | The dimension of the homology of the parent of `x`. | — |
| `FieldOfDefinition(x)` | A field over which `x` is defined (not necessarily minimal). | — |

*Worked example: H136E99 (2-torsion point on `J₀(11)`; approximate orders, degree, field of definition).*

### 136.11.3 Predicates

| Intrinsic | Description |
|-----------|-------------|
| `x eq y` | `true` if `x` and `y` are equal. |
| `x in X` | `true` if `x` is in the list `X`. |
| `IsExact(x)` | `true` if `x` is known exactly (defined by an element of rational homology). |
| `IsZero(x)` | `true` if `x` is known exactly and equals 0; if not exact, `true` if a real homology vector representing `x` is "very close" (within `1/10ⁿ`, `n = M\`point_precision`) to integral homology. |

*Worked example: H136E100 (2-torsion of the two conductor-37 elliptic curves; `eq`, `in`, `IsZero`, `IsExact`, closeness threshold, `point_precision`).*

### 136.11.4 Homomorphisms

| Intrinsic | Description |
|-----------|-------------|
| `x @ phi` / `phi(x)` | The image of `x` under the homomorphism `φ`. |
| `x @@ phi` | An inverse image of `x` under `φ`. |

*Worked example: H136E101 (`T₃ − 5` on `J₀(23)`: applying to a kernel element gives 0; inverse images via `@@`).*

### 136.11.5 Representation of Torsion Points

An exact torsion representation is found via continued fractions (good rational
approximations to each coordinate of a real homology class). The homology representative can
also be retrieved.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ApproximateByTorsionPoint(x : -)` | If `x` is defined by `z ∈ H₁(A,R)`, an element of `H₁(A,Q)` approximating `z`, returned as the corresponding point. Parameter `Cutoff` (`RngIntElt`, default `10³`). | Continued fractions. |
| `Element(x)` | The vector in homology representing `x`. | — |
| `LatticeCoordinates(x)` | A rational/real vector representing `x` w.r.t. the basis for integral homology of the parent. | — |
| `Eltseq(x)` | The `Eltseq` of `LatticeCoordinates(x)`. | — |

*Worked example: H136E102 (3-torsion point in `J₀(33)`; `Element`, `Eltseq`, `LatticeCoordinates`; difference of `Element` and `LatticeCoordinates` for weight > 2).*

---

## 136.12 Subgroups of Modular Abelian Varieties

### 136.12.1 Creation

Subgroups are created from generating sequences of elements, as `n`-torsion subgroups `A[n]`,
or as kernels/images of homomorphisms. If a subgroup contains inexact (floating-point)
elements, an approximating torsion group can be found.

| Intrinsic | Description |
|-----------|-------------|
| `Subgroup(X)` | The subgroup of `A` generated by the nonempty sequence `X` of elements. |
| `ZeroSubgroup(A)` | The zero subgroup of `A`. |
| `nTorsionSubgroup(A, n)` | The kernel `A[n]` of multiplication-by-`n` on `A`. |
| `nTorsionSubgroup(G, n)` | The kernel `G[n]` of multiplication-by-`n` on the subgroup `G`. |
| `ApproximateByTorsionGroup(G : -)` | The subgroup generated by torsion approximations of a set of generators of `G`. Parameter `Cutoff` (`RngIntElt`, default `10³`). |

*Worked example: H136E103 (2-torsion of `"100A"`; `ZeroSubgroup`, `nTorsionSubgroup` on subgroups; approximation command).*

### 136.12.2 Elements

| Intrinsic | Description |
|-----------|-------------|
| `Elements(G)` | A sequence of all elements of the finite subgroup `G`. |
| `Generators(G)` | A sequence of generators of `G` (generators of the underlying abelian group). |
| `Ngens(G)` | The number of generators of `G`. |
| `G . i` | The `i`-th generator of `G`. |

*Worked example: H136E104 (kernel of `T₃` on `J₀(67)`; `Elements`, `Generators`, `Ngens`, `G.i`).*

### 136.12.3 Arithmetic

Quotients by finite subgroups, intersections, and sums. For several operations, finite groups
or varieties are replaced by their image in a common variety (from `FindCommonEmbeddings`); the
"embedding" is only guaranteed up to isogeny.

| Intrinsic | Description |
|-----------|-------------|
| `Quotient(A, G)` | The quotient `A/G` by the finite subgroup `G`, the isogeny `A → A/G`, and an isogeny `A/G → A`, whose composition is multiplication by the exponent of `G`. |
| `Quotient(G)` | The quotient `A/G` where `A` is the ambient variety of `G`, the isogeny `A → A/G`, and an isogeny back. |
| `A / G` | The quotient `A/G`, the isogeny `A → A/G`, and an isogeny `A/G → A` (`A` need not be the ambient of `G`). |
| `A meet G` / `G meet A` | The intersection of the finite subgroup `G` (of a variety `B`) with the variety `A` (replaced by images in a common variety if `A ≠ B`). |
| `G1 + G2` | The sum of subgroups `G₁`, `G₂` (replaced by images in a common variety if ambients differ). |
| `G1 meet G2` | The intersection of `G₁`, `G₂` (replaced by images in a common variety if ambients differ). |

*Worked example: H136E105 (2-torsion of `J₀(67)`; quotients via `Quotient`/`/`; sum of 2-torsion of simple factors smaller than full `J₀(67)[2]`).*

### 136.12.4 Underlying Abelian Group and Lattice

| Intrinsic | Description |
|-----------|-------------|
| `AbelianGroup(G)` | An abstract abelian group `H ≅ G` together with isomorphisms in both directions. |
| `Lattice(G)` | For `G` a finite torsion subgroup with exactly-known elements (generated by `H₁(A,Q)/H₁(A,Z)`): the lattice `L` in rational homology generated by `H₁(A,Z)` and all such `x`, so `G ≅ L/H₁(A,Z)`. |

*Worked example: H136E106 (3-torsion of `J₀(11)`; `AbelianGroup`, `Lattice` is `1/3` times integral homology).*

### 136.12.5 Invariants

| Intrinsic | Description |
|-----------|-------------|
| `AmbientVariety(G)` | The abelian variety whose elements were used to create `G`. |
| `Exponent(G)` | The smallest positive `e` with `eG = 0` (assumes `G` finite). |
| `Invariants(G)` | The invariants of an abstract abelian group isomorphic to `G`. |
| `Order(G)` / `#G` | The number of elements of `G` (errors if not known finite). |
| `FieldOfDefinition(G)` | A field `K` such that any automorphism fixing `K` fixes `G` (not guaranteed minimal). |

*Worked example: H136E107 (kernel of `T₃` on `J₀(67)`; all invariants; field of definition `Q` for the whole group but `Q̄` for a one-generator subgroup).*

### 136.12.6 Predicates and Comparisons

A subgroup is finite exactly when every element is known exactly. Equality/inclusion are
liberal: if ambient varieties differ, Magma seeks a common embedding and compares there.

| Intrinsic | Description |
|-----------|-------------|
| `IsFinite(G)` | `true` if `G` is known finite (generated by torsion elements); `false` if `G` has inexact (floating-point) elements. |
| `G1 subset G2` | `true` if `G₁ ⊆ G₂`. |
| `G subset A` | `true` if the subgroup `G` is a subset of the variety `A` (mapped to a common ambient if needed). |
| `A subset G` | `true` if the variety `A` is a subset of `G` (only when `A` is a point, i.e. 0-dimensional). |
| `G1 eq G2` | `true` if `G₁` and `G₂` are equal. |

*Worked example: H136E108 (`J₀(389)` in the `+1` quotient; 5-torsion subgroups of factors `A`, `B`; `IsFinite`, `subset`, `eq` including cross-ambient comparisons).*

---

## 136.13 Rational Torsion Subgroups

### 136.13.1 Cuspidal Subgroup

For `A` with parameterization `π : J₀(N) → A`, the cuspidal subgroup of `J₀(N)` is generated
by differences of cusps on `X₀(N)`; that of `A(Q̄)` is its image under `π`. The *rational
cuspidal subgroup* is generated by differences of cusps defined over `Q`, giving a lower
bound on the cardinality/structure of `A(Q)_tors` (important for Birch–Swinnerton-Dyer).

| Intrinsic | Description |
|-----------|-------------|
| `CuspidalSubgroup(A)` | The subgroup of `A` generated by all differences of cusps (viewing `A` as a quotient of a modular symbols variety). Need not be defined over `Q`. |
| `RationalCuspidalSubgroup(A)` | The subgroup of `A` generated by all differences of `Q`-rational cusps. |

*Worked example: H136E109 (cuspidal and rational cuspidal subgroups of `J₀(100)`; for the optimal conductor-100 curve, torsion multiple 2 means part of the cuspidal subgroup is not over `Q`).*

### 136.13.2 Upper and Lower Bounds

For `A` over a number field `K`, Magma computes a multiple and a divisor of `#A(K)_tors`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `TorsionLowerBound(A)` | A divisor of `#A(K)_tors`. Currently requires `A` to be the base extension of `B` over `Q`; the bound is the cardinality of the rational cuspidal subgroup of `B(Q)`. | Rational cuspidal subgroup. |
| `TorsionMultiple(A)` / `TorsionMultiple(A, n)` | A multiple of `#A(K)_tors`. `n` (default 50): for each good prime `p ≤ n` with `[K:Q]+1 < p` and `p ∤ level(A)`, compute `#A(k)` over residue fields `k` of characteristic `p`; the gcd of these is a multiple. | `#A(k)` via the characteristic polynomial of Frobenius on a Tate module (Hecke operators). See **[AS05]**. |

*Worked example: H136E110 (`J₀(100)`: `TorsionLowerBound = #RationalCuspidalSubgroup = 1350`, `TorsionMultiple = 16200`; over `Q(√2)`).*

### 136.13.3 Torsion Subgroup

| Intrinsic | Description |
|-----------|-------------|
| `TorsionSubgroup(A)` | Attempt to compute `A(K)_tors`. Returns `false` and a subgroup of the torsion (when not provably complete), or `true` and the exact torsion subgroup over the base field. |

*Worked example: H136E111 (`TorsionSubgroup` of `J₀(11)`, `J₀(33)`, base extensions, `J₀(100)`, `J₀(125)`).*

---

## 136.14 Hecke and Atkin-Lehner Operators

### 136.14.1 Creation

Endomorphisms induced by Atkin–Lehner and Hecke operators. The Atkin–Lehner involution `W_q`
is defined for each `q` exactly dividing the level (and divisible by the conductor of any
relevant character).

| Intrinsic | Description |
|-----------|-------------|
| `AtkinLehnerOperator(A, q)` | The Atkin–Lehner operator `W_q` of index `q` on `A`. In general `W_q` is only a morphism up to isogeny, so also returns an integer `d` with `d∗W_q` an endomorphism (and `d = 0` when `W_q` does not leave `A` invariant). Errors unless `r | q` for any character of conductor `r` in the ambient. |
| `AtkinLehnerOperator(A)` | The (main) Atkin–Lehner morphism (or morphism tensor `Q`) on `A`. |
| `HeckeOperator(A, n)` | The Hecke operator `T_n` of index `n` on `A`. In general `T_n` need not be a morphism; if `A ⊆ J₀(N)` and `T_n` does not leave `A` invariant, `T_n` is composed with a map back to `A`. (For exact Hecke operators from the action on `J₀(N)`, use `RestrictEndomorphism`.) |

*Worked examples: H136E112 (`W₂₃`, `T₂` on `J₀(23)`; `w₄·w₂₅ = w₁₀₀` on `J₀(100)`); the same example continues with `W₂₅` on `J₁(25)` and Hecke operators on a quotient of a factor of `J₀(65)`.*

### 136.14.2 Invariants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HeckePolynomial(A, n)` | The characteristic polynomial of `T_n` acting on `A`. | — |
| `FactoredHeckePolynomial(A, n)` | The factored characteristic polynomial of `T_n` on `A`. | Uses the decomposition of `A` to avoid factoring, often faster than computing `T_n` then its characteristic polynomial. |
| `MinimalHeckePolynomial(A, n)` | The minimal polynomial of `T_n` on `A`. | — |

*Worked example: H136E113 (`FactoredHeckePolynomial`, `HeckePolynomial`, `MinimalHeckePolynomial` of `T₂` on `J₀(65)`).*

---

## 136.15 L-series

### 136.15.1 Creation

`LSeries` creates `L(A,s)` for `A` over `Q` or a cyclotomic field. No computation is performed
at creation.

| Intrinsic | Description |
|-----------|-------------|
| `LSeries(A)` | The `L`-series associated to `A`. |

*Worked example: H136E114 (`LSeries` of `J₀(23)`, `"65B"`, and over `Q(ζ₅)`).*

### 136.15.2 Invariants

| Intrinsic | Description |
|-----------|-------------|
| `CriticalStrip(L)` | Integers `x`, `y` such that the critical strip of `L` is `{Re ∈ (x,y)}`. Returns `0` and `Max(W)`, `W` the set of weights of newforms giving factors of `A`. |
| `ModularAbelianVariety(L)` | The abelian variety that `L` is associated to. |

*Worked example: H136E115 (critical strips of `L`-series of `J₀(37)`, `J₀(37,6)`, `J₁(11,3)`, the weight-12 level-1 motive `Δ`).*

### 136.15.3 Characteristic Polynomials of Frobenius Elements

Characteristic polynomials of Frobenius on the `ℓ`-adic Tate modules define the local
`L`-factors of `L(A,s)`.

| Intrinsic | Description |
|-----------|-------------|
| `FrobeniusPolynomial(A : -)` | The characteristic polynomial of Frobenius on `A` over a finite field. Parameter `Factored` (`BoolElt`, default `false`) returns a factorization. |
| `FrobeniusPolynomial(A, p : -)` | The characteristic polynomial of `Frob_p` on any `ℓ`-adic Tate module of `A` over a number field (`p, ℓ ∤ level(A)`). For base ring of degree > 1, a sequence (one per prime over `p`, sorted by degree). Parameter `Factored` (default `false`). |
| `FrobeniusPolynomial(A, P)` | The characteristic polynomial of Frobenius at a nonzero prime ideal `P` (a prime of good reduction for `A`, `A` over a field containing `P`). |

*Worked example: H136E116 (`FrobeniusPolynomial` of `J₀(23)`, a product, over `Q(ζ₂₂)`; used for point counting over finite fields).*

### 136.15.4 Values at Integers in the Critical Strip

Evaluation of `L`-series at integers in the critical strip. (Algorithms for arbitrary
complex `s` exist but are not implemented in Magma.)

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `L(s)` / `Evaluate(L, s)` / `Evaluate(L, s, prec)` | The value of `L` at integer `s` in the critical strip, using `prec` terms (default 100) of `q`-expansions of modular forms corresponding to differentials on `A`. (The relation between `prec` and output precision is not known a priori; increase `prec` and observe.) | `q`-expansion power series. |
| `LRatio(A, s)` / `LRatio(L, s)` | For `A` over `Q` attached to a newform (or its `L`-series `L`) and critical integer `s`: the ratio `L(A,s)·(s−1)!/((2π)^{s−1}·Ω_s)`, `Ω_s` the integral (Néron) volume of the real points of the optimal quotient `A'` (odd `s`) or the volume of the `−1` conjugation eigenspace (even `s`). | — |
| `IsZeroAt(L, s)` | For integer `s` in the critical strip, `true` iff `L(A,s) = 0`. Provably correct (unlike `Evaluate`). | — |

*Worked examples: H136E117 (`L(1)`, `Evaluate`, `LRatio` of `J₀(23)`; `Δ` motive `LRatio(L,1)=11340/691`; `J₁(N)` factors); H136E118/H136E119 (`LeadingCoefficient`, see below).*

### 136.15.5 Leading Coefficient

| Intrinsic | Description |
|-----------|-------------|
| `LeadingCoefficient(L, s, prec)` | For `L` of `A` and integer `s` in the critical strip, the leading coefficient of the Taylor expansion of `L` about `s` and the order of vanishing of `L` at `s`. (Requires `A` of weight 2 and trivial character, so `s = 1`; need not be a newform; `prec` = number of power-series terms.) |

*Worked examples: H136E118 (`LeadingCoefficient` for `J₀(37)`, order of vanishing 4 for `J₀(3⁵)`); H136E119 (`"389A"`, factors of `J₀(65)`).*

---

## 136.16 Complex Period Lattice

### 136.16.1 Period Map

| Intrinsic | Description |
|-----------|-------------|
| `PeriodMapping(A, prec)` | The complex period mapping from `H₁(A,Q)` to `Cᵈ`, `d = dim A`, using `prec` terms of `q`-expansions. |

### 136.16.2 Period Lattice

| Intrinsic | Description |
|-----------|-------------|
| `Periods(A, n)` | Generators for the complex period lattice of `A`, using `n` terms of `q`-expansions. (Uses the map from `A` to a modular symbols variety, which must be injective.) |

---

## 136.17 Tamagawa Numbers and Component Groups of Neron Models

### 136.17.1 Component Groups

For `A` a newform variety over `Q` of level `N` and `p` exactly dividing `N`, the order of the
component group over `F̄_p` can be computed (**[CS01]**, **[KS00]**). Computing the structure
or the order under more general hypotheses is open.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ComponentGroupOrder(A, p)` | The order of the component group of the special fibre of the Néron model of `A` over `F̄_p`. `A` must be attached to a newform. | **[CS01, KS00]**. |

*Worked example: H136E120 (`ComponentGroupOrder` of a factor of `J₀(65)` at 13 and 5).*

### 136.17.2 Tamagawa Numbers

For `A` over `Q` attached to a newform, a divisor and a multiple (some power) of the Tamagawa
number at a prime `p`. When `p² | N` (additive reduction), the Lenstra–Oort bound **[LO85]**
is used.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `TamagawaNumber(A, p)` | A divisor of the Tamagawa number at `p` and an integer some power of which is a multiple; also `true` if the divisor is provably equal to the Tamagawa number. `A` must be attached to a newform. | When `p² | N`: Lenstra–Oort bound **[LO85]**. |
| `TamagawaNumber(A)` | Let `c` be the product of Tamagawa numbers at bad primes. Returns a divisor of `c`, an integer some power of which is a multiple of `c`, and `true` if the divisor is provably equal to `c`. `A` over `Q`, attached to a newform. | As above, over all bad primes. |

*Worked example: H136E121 (`TamagawaNumber` of factors of `J₀(65)` at 5 and 13; `J₀(5²·7)`).*

---

## 136.18 Elliptic Curves

### 136.18.1 Creation

Modular abelian varieties of dimension 1 are elliptic curves. From `A` of dimension 1, an
isogenous elliptic curve over `Q` can be computed; from an elliptic curve `E` over `Q`, an
isogenous modular abelian variety can be built.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `EllipticCurve(A)` | An elliptic curve over `Q` isogenous to the dimension-1 variety `A`, if one exists. (For weight > 2 use `EllipticInvariants`.) | — |
| `ModularAbelianVariety(E : -)` | A modular abelian variety over `Q` isogenous to the elliptic curve `E`. Parameter `Sign` (`RngIntElt`, default 0). (Can be slow: small-coefficient curves may have large conductor, hence a large `J₀(N)` quotient.) | — |

*Worked example: H136E122 (`EllipticCurve(J₀(49))`, round-trip via `ModularAbelianVariety`, `IsIsomorphic`).*

### 136.18.2 Invariants

For `A` over `Q` of dimension 1, standard iterative algorithms (Cremona's book) compute
`c₄, c₆, j` and period-lattice generators of the optimal quotient of `J₀(N)` associated to `A`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `EllipticInvariants(A, n)` | The invariants `c₄, c₆, j` and an elliptic curve of the dimension-1 variety `A`, using `n` terms of `q`-expansion. | Cremona's iterative algorithms. |
| `EllipticPeriods(A, n)` | The elliptic periods `w₁, w₂` of the `J₀(N)`-optimal elliptic curve associated to `A`, using `n` terms (`w₁/w₂` has positive imaginary part). | Cremona's iterative algorithms. |

*Worked example: H136E123 (`EllipticInvariants` and `EllipticPeriods` of `"100A"`).*

---

## 136.19 Bibliography (canonical references)

| Key | Reference |
|-----|-----------|
| **[AS04]** | A. Agashe and W. A. Stein. *The Manin Constant, Congruence Primes, and the Modular Degree.* Preprint, URL: http://modular.fas.harvard.edu/papers/manin-agashe/, 2004. |
| **[AS05]** | A. Agashe and W. Stein. *Visible evidence for the Birch and Swinnerton-Dyer conjecture for modular abelian varieties of analytic rank zero.* Math. Comp. **74**(249):455–484 (electronic), 2005. With an appendix by J. Cremona and B. Mazur. |
| **[Bos00]** | W. Bosma, editor. *ANTS IV*, volume 1838 of LNCS. Springer-Verlag, 2000. |
| **[CS01]** | B. Conrad and W. A. Stein. *Component groups of purely toric quotients.* Math. Res. Lett. **8**(5–6):745–766, 2001. |
| **[Kil02]** | L. J. P. Kilford. *Some non-Gorenstein Hecke algebras attached to spaces of modular forms.* J. Number Theory **97**(1):157–164, 2002. |
| **[KS00]** | D. R. Kohel and W. A. Stein. *Component Groups of Quotients of J₀(N).* In Bosma [Bos00]. |
| **[LO85]** | H. W. Lenstra, Jr. and F. Oort. *Abelian varieties having purely additive reduction.* J. Pure Appl. Algebra **36**(3):281–298, 1985. |
| **[Que09]** | J. Quer. *Fields of definition of building blocks.* Math. Comp. **78**(265):537–554, 2009. |

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Modular symbols representation of homology | `JZero`, `JOne`, `Js`, `JH`, `ModularAbelianVariety`, `ModularSymbols`, `Homology`, `IntegralHomology` |
| Decomposition into simple newform varieties (Poincaré reducibility) | `Decomposition`, `Factorization`, `DecomposeUsing`, `DefinesAbelianSubvariety`, `IsSimple` |
| Inner/CM twists; building blocks **[Que09]** | `InnerTwists`, `CMTwists`, `BoundedFSubspace`, `DegreeMap`, `BrauerClass`, `ObstructionDescentBuildingBlock`, `HasCM`/`IsCM` |
| Kernel/cokernel/image (non-abelian category, finite component groups) | `Kernel`, `ConnectedKernel`, `ComponentGroupOfKernel`, `Cokernel`, `Image`, `Quotient` |
| Endomorphism/Hom rings, saturation, trace pairing | `Hom`, `End`, `HeckeAlgebra`, `Subring`, `Saturation`, `Discriminant`, `Index` |
| Orthogonal complements & duals (intersection pairing) | `Complement`, `ComplementOfImage`, `Dual`, `ModularPolarization`, `IntersectionPairing`, `ProjectionOnto`, `LeftInverse`, `RightInverse` |
| Hecke / Atkin–Lehner operators | `HeckeOperator`, `AtkinLehnerOperator`, `HeckePolynomial`, `FactoredHeckePolynomial`, `MinimalHeckePolynomial` |
| Congruences / modular degree **[AS04]** | `CongruenceModulus`, `ModularDegree` |
| Cuspidal & rational torsion; BSD lower bounds | `CuspidalSubgroup`, `RationalCuspidalSubgroup`, `TorsionSubgroup`, `TorsionLowerBound` |
| Torsion bounds via Frobenius on Tate modules **[AS05]** | `TorsionMultiple`, `FrobeniusPolynomial`, `NumberOfRationalPoints` |
| Component groups of Néron models **[CS01, KS00]** | `ComponentGroupOrder` |
| Tamagawa numbers (Lenstra–Oort bound **[LO85]**) | `TamagawaNumber` |
| `L`-series values, ratios, leading coefficients | `LSeries`, `Evaluate`/`L(s)`, `LRatio`, `IsZeroAt`, `LeadingCoefficient`, `CriticalStrip` |
| Complex period lattices / elliptic invariants (Cremona) | `PeriodMapping`, `Periods`, `EllipticInvariants`, `EllipticPeriods`, `EllipticCurve` |
| New/old subvarieties & natural degeneracy maps | `NaturalMap`, `NaturalMaps`, `NewSubvariety`, `NewQuotient`, `OldSubvariety`, `OldQuotient` |
