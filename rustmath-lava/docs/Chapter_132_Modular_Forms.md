# Chapter 132 — Modular Forms

**Handbook part:** XVII — Modular Arithmetic Geometry
**Handbook pages:** 4385–4426 (PDF pages 4518–4559)

---

## Scope and overview

This chapter documents Magma's package for spaces of classical modular forms. Fixing positive
integers `N` (level) and `k` (weight), the central object is the finite-dimensional **C**-vector
space `M_k(Γ_1(N))` of holomorphic functions `f : H → C` on the upper half plane satisfying the
modular transformation law for `Γ_1(N)` and holomorphy at the cusps. The diamond-bracket
operators `⟨d⟩` decompose `M_k(Γ_1(N))` into character eigenspaces `M_k(Γ_1(N))(ε)` indexed by
Dirichlet characters `ε : (Z/NZ)* → C*`; `M_k([ε])` denotes the direct sum over the
Gal(**Q̄**/**Q**)-conjugates of `ε`. Every modular form has a `q`-expansion (Fourier expansion)
`f = a_0 + a_1 q + a_2 q² + …` with `q = exp(2πiz)`.

Magma computes a basis of `q`-expansions for any space of modular forms of weight `k ≥ 2`. Any
space realised in Magma has the form `M_Z ⊗_Z R` for a base ring `R` and a **Z**-defined space
`M_Z`; the canonical **Z**-basis is in Hermite normal form, and bases over **Z** require
computing `q`-expansions up to a Sturm bound (`PrecisionBound`). Each space carries a commuting
family of Hecke operators `T_n`; an **eigenform** is a simultaneous eigenvector of the Hecke
algebra, and a **newform** is a normalised eigenform not arising from lower level.

**About the package.** The modular forms package is in many ways an interface to the modular
symbols machinery (Chapter on Modular Symbols), with additional independently-implemented
features such as Eisenstein series and Hecke operators. Spaces of modular forms must be
Galois-stable over the rationals (unlike modular symbols). From Magma version 2.14, weight-one
and half-integral-weight forms were added, though some functions (e.g. `Newforms`) are not
implemented for those weights. Categories: spaces are of type `ModFrm`, elements of type
`ModFrmElt`. Verbose output: `SetVerbose("ModularForms", n)` with `n ∈ {0, 1, 2}`.

*Worked examples:* H132E1 (categories and verbosity); H132E2 (an illustrative overview:
level-1 forms, the `Gamma1(13)` conjugate newforms, the exceptional case of Serre's conjecture
at level 13, fast dimension computations).

---

## 132.2 Creation Functions

### 132.2.1 Ambient Spaces

The following create spaces of modular forms. For Dirichlet characters, see the Dirichlet
characters section (19.8).

| Intrinsic | Description |
|-----------|-------------|
| `ModularForms(N)` | The space `M_2(Γ_0(N), Z)` of weight-2 modular forms on `Γ_0(N)` over **Z** (same as `ModularForms(N, 2)`). |
| `ModularForms(N, k)` | The space `M_k(Γ_0(N), Z)` of weight-`k` modular forms on `Γ_0(N)` over **Z**. |
| `ModularForms(eps, k)` | For a Dirichlet character `eps` and integer `k`: a space of weight-`k` modular forms over **Z** which, under base extension, becomes the direct sum of the spaces `M_k(Γ_1(N), eps1)` of weight `k` and nebentypus `eps1`, where `eps1` runs over all Galois conjugates of `eps`. |
| `ModularForms(chars, k)` | The weight-`k` space over **Z** formed as the direct sum of `ModularForms(eps, k)` over all `eps` in the sequence `chars` of Dirichlet characters. |
| `ModularForms(G)` | Same as `ModularForms(G, 2)`. |
| `ModularForms(G, k)` | The space `M_k(G, Z)` for a congruence subgroup `G`. The groups `Γ_0(N)` and `Γ_1(N)` are supported, created via `Gamma0(N)` and `Gamma1(N)`. |
| `CuspForms(x)` / `CuspForms(x, y)` | Shortcut returning the `CuspidalSubspace` of the corresponding full space of modular forms. |

*Worked example:* H132E3 (constructing `M_2(Γ_0(65))`, `M_4(Γ_0(8))`, a level-20 character space `M_3(N, ε)`, direct sums over mod-20 characters, and `M_k(Γ_1(20))` plus its cusp forms).

#### 132.2.1.1 Half-integral Weight Forms

Spaces of modular forms of half-integral weight can be constructed; for these spaces
`CuspidalSubspace` and `qExpansionBasis` are available, plus basic element arithmetic. The
algorithm for the `q`-expansion basis involves computing those of related integral-weight
spaces (of weight one half smaller or one half larger, with appropriate level and character).

| Intrinsic | Description |
|-----------|-------------|
| `HalfIntegralWeightForms(N, w)` | The space of half-integral weight forms on `Gamma0(N)` of weight `w`. Here `N` should be a multiple of 4 and `w` a positive element of `Z + 1/2`. |
| `HalfIntegralWeightForms(chi, w)` | The space of half-integral weight forms on `Gamma1(N)` with character `chi` and weight `w`. The modulus of `chi` should be a multiple of 4, and `w` a positive element of `Z + 1/2`. |
| `HalfIntegralWeightForms(G, w)` | The space of half-integral weight forms on the congruence subgroup `G` and weight `w`. Here `G` must be contained in `Gamma0(4)`, and `w` is a positive element of `Z + 1/2`. |

### 132.2.2 Base Extension

If `M` is created with one of the constructors in 132.2.1, its base ring is **Z**, so `M` can
be base extended to any ring `R`.

| Intrinsic | Description |
|-----------|-------------|
| `BaseExtend(M, R)` | The base extension of `M` to the ring `R`, plus the induced map `M → BaseExtend(M, R)`. The only requirement on `R` is a natural coercion map from `BaseRing(M)` to `R`; when `BaseRing(M)` is the integers, any `R` is allowed. |
| `BaseExtend(M, phi)` | The base extension of `M` to `R` using the map `φ : BaseRing(M) → R`, plus the induced map `M → BaseExtend(M, R)`. |

*Worked example:* H132E4 (an Eisenstein series `E_12` in `M_12(1)` congruent to 1 mod 3; base extension to **Q**, to `GF(3)`, and to a polynomial ring over `GF(17)`).

### 132.2.3 Elements

| Intrinsic | Description |
|-----------|-------------|
| `M . i` | The `i`-th basis vector of the space of modular forms `M`. |
| `M ! f` | Coercion of `f` into the space of modular forms `M`. Here `f` can be a modular form, a power series with absolute precision, or something coercible into `RSpace(M)`. |
| `ModularForm(E)` | The modular form associated to the elliptic curve `E` over **Q** (see 132.18). |

*Worked example:* H132E5 (coercing power series and a form for `Gamma0(11)` into `M_2(Γ_0(22))`; the elliptic curve `EllipticCurve([0,-1,1,-10,-20])` of conductor 11 as an element of `M_2(Γ_0(11))` via `ModularForm`).

---

## 132.3 Bases

Any space of modular forms in Magma is of the form `M_Z ⊗_Z R`. The basis of `M` is the image
in `M` of `Basis(M_Z)`, with `Basis(M_Z)` in Hermite normal form. Determining this basis over
**Z** requires computing `q`-expansions up to a Sturm bound for `M` (see `PrecisionBound`); the
internal precision is at least as large. Lower-precision `q`-expansions can be obtained by
working directly with spaces of modular symbols.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Basis(M)` | The canonical basis of the space of modular forms or half-integral weight forms `M`. | Hermite normal form over **Z**, then base change. |
| `Basis(M, prec)` / `qExpansionBasis(M, prec)` | A sequence containing the `q`-expansions (to precision `prec`) of the elements of `Basis(M)`. | — |
| `PrecisionBound(M : Exact)` | An integer `b` such that `f + O(q^b)` determines any modular form `f` in `M`. With `Exact := true` (default `false`), or if a `q`-expansion basis has already been computed, the result is best-possible (the smallest such `b`); otherwise it is a Sturm bound similar to (and sometimes sharper than) the bounds in section 9.4 of **[Ste07]**. | Sturm bound **[Ste07]**; `Exact := true` gives the smallest determining precision. |
| `RModule(M)` / `RSpace(M)` / `VectorSpace(M)` | An abstract free module isomorphic to `M` over the same base ring (unless `Ring` is specified), plus a map to/from `M`. Needed to use linear-algebra functions on `M` (a space of modular forms is not a subtype of vector space in Magma). Parameter: `Ring`. | — |

*Worked examples:* H132E6 (`NewSubspace(CuspidalSubspace(M))` for `Gamma1(16)`, weight 3, showing integral vs. rational reduced forms via `SetPrecision`/`BaseExtend`); H132E8 (`PrecisionBound` related to Weierstrass points on `X_0(N)`, citing **[Atk67]**).

---

## 132.4 q-Expansions

These intrinsics give the `q`-expansion (about the cusp `∞`) of a modular form. By default
`q`-expansions are printed only to precision `O(q^12)`; adjust printing with `SetPrecision`, and
control internally-computed precision with `qExpansion`/`qExpansionBasis`.

| Intrinsic | Description |
|-----------|-------------|
| `qExpansion(f)` / `qExpansion(f, prec)` / `PowerSeries(f)` / `PowerSeries(f, prec)` | The `q`-expansion (at the cusp `∞`) of the modular form (or half-integral weight form) `f` to absolute precision `prec`. An element of the power series ring over the base ring of the parent of `f`. |
| `Coefficient(f, n)` | The `n`-th coefficient of the `q`-expansion of the modular form `f`. |
| `Precision(M)` | The default printing precision for elements of the space `M` of modular forms (hard-coded default value 12). |
| `SetPrecision(M, prec)` | Set the default printing precision for elements of the space `M` of modular forms (hard-coded default value 12). |

*Worked example:* H132E7 (computing the `q`-expansion of `f ∈ M_3(Γ_1(11))` in several ways; `Coefficient` has infinite precision, big-oh via addition of a form and a power series, `SetPrecision`).

---

## 132.5 Arithmetic

| Intrinsic | Description |
|-----------|-------------|
| `f + g` | The sum of the modular forms `f` and `g`; or the sum of a modular form `f` and a power series `g` (the `q`-expansion of `f` must be coercible into the parent of `g`). `g + f`, `f − g`, `g − f` also defined. |
| `f - g` | The difference of the modular forms `f` and `g`. |
| `a * f` | The product of the scalar `a` and the modular form `f`. |
| `f / a` | The product of the scalar `1/a` and the modular form `f`. |
| `f ^ n` | The power `f^n` of the modular form `f`, where `n ≥ 1` is an integer. |
| `f * g` | The product of the modular forms `f` and `g`. The only condition is that the base fields of `f` and `g` be the same; the weight of `f*g` is the sum of the weights of `f` and `g`. |

*Worked example:* H132E9 (sums, scalar multiples, squares and products of forms in `M_2(Γ_0(11))`, with weights adding under multiplication).

---

## 132.6 Predicates

| Intrinsic | Description |
|-----------|-------------|
| `IsAmbientSpace(M)` | Returns `true` iff `M` is an ambient space (those constructed in 132.2.1). |
| `IsCuspidal(M)` | Returns `true` if `M` is contained in the cuspidal subspace of the ambient space. |
| `IsEisenstein(M)` | Returns `true` if `M` is contained in the Eisenstein subspace of the ambient space. |
| `IsEisensteinSeries(f)` | Returns `true` if `f` is an Eisenstein newform or was computed using the intrinsic `EisensteinSeries` (see 132.10). |
| `IsGamma0(M)` | Returns `true` if `M` is a space of modular forms for `Γ_0(N)`. |
| `IsGamma1(M)` | Returns `true` if `M` was created explicitly as a space of modular forms for `Γ_1(N)`, or if the `AmbientSpace` of `M` is such a space. (Returns `false` for any `ModularForms(chars, k)`, even if `chars` consists of all mod-`N` Dirichlet characters.) |
| `IsNew(M)` | Returns `true` if `M` is contained in the new subspace of its `AmbientSpace`. |
| `IsNewform(f)` | Returns `true` if `f` was created using `Newforms`. (Sometimes `true` in other cases where `f` is obviously a newform. In number theory, "newform" means "normalized eigenform that lies in the new subspace".) |
| `IsRingOfAllModularForms(M)` | Returns `true` iff `M` is the ring of all modular forms over a given ring. |

*Worked example:* H132E10 (each predicate illustrated in `M_3(Γ_1(11))`).

---

## 132.7 Properties

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AmbientSpace(M)` | The full space of modular forms, in which `M` was created as a subspace. | — |
| `BaseRing(M)` / `CoefficientRing(M)` | The ring over which `M` was defined. | — |
| `Degree(f)` | The number of Galois-conjugates of the modular form `f` over the prime subfield of (the fraction field of) the base ring of `f`. | — |
| `Dimension(M)` | The dimension of the space `M` of modular forms or half-integral weight forms. | For spaces from the `ModularForms` constructors, counts the relevant Eisenstein series and applies a dimension formula for the cusp forms. |
| `DimensionByFormula(M)` | The dimension of `M` (which must be a full space or the cuspidal subspace of a full space), as given by the formulas in Cohen and Oesterlé ('Modular Forms in One Variable, VI', Lecture Notes in Math. 627). | Cohen–Oesterlé dimension formulas. |
| `DimensionByFormula(N, k)` / `DimensionByFormula(chi, k)` / `DimensionByFormula(N, chi, k)` | The dimension of the full space of modular forms (or half-integral weight forms) with level `N`, character `chi` (trivial if not specified) and weight `k`, by the Cohen–Oesterlé formulas. Parameter: `Cuspidal` (default `false`); if `true`, the dimension of the space of cusp forms is returned. | Cohen–Oesterlé dimension formulas. |
| `DirichletCharacters(M)` | A sequence containing exactly one representative from each Galois-conjugacy class of Dirichlet characters associated to `M`. | — |
| `DirichletCharacter(f)` | For `f` a newform created with `Newform`: a Dirichlet character that is, up to Galois conjugacy, the Nebentypus character of `f`. | — |
| `Eltseq(f)` | The sequence `[a_1, …, a_n]` such that `f = a_1 g_1 + … + a_n g_n`, where `g_1, …, g_n` is the basis of the parent of `f`. | — |
| `Level(f)` / `Level(M)` | The level of the modular form `f` (resp. the space `M`). | — |
| `Weight(f)` | The weight of the modular form `f`, if it is defined. | — |
| `Weight(M)` | The weight of the space `M` of modular forms. | — |
| `WeightOneHalfData(H)` | A list of tuples describing a basis of the given space of forms of weight 1/2. Each tuple is a pair `<f, t>`, where `t` is an integer and `f` a Dirichlet character; the tuple designates the sum over all integers `n` of `f(n) q^(t n²)`. | Serre–Stark explicit description (see 132.11). |

*Worked example:* H132E11 (each property illustrated in `M_3(Γ_1(11))`).

---

## 132.8 Subspaces

These functions compute the cuspidal, Eisenstein, and new subspaces.

| Intrinsic | Description |
|-----------|-------------|
| `ZeroSubspace(M)` | The trivial subspace of the space of modular forms `M`. |
| `CuspidalSubspace(M)` | The subspace of forms `f` in `M` such that the constant term of the Fourier expansion of `f` at every cusp is 0. |
| `EisensteinSubspace(M)` | The Eisenstein subspace of the space of modular forms `M`. |
| `EisensteinProjection(f)` / `CuspidalProjection(f)` | The projection of a given modular form to the `EisensteinSubspace` (resp. the `CuspidalSubspace`). The sum of the two projections equals the original form (after coercion). The base ring of the given form must contain the rationals. |
| `NewSubspace(M)` | The new subspace of the space of modular forms `M`. |
| `DihedralSubspace(M)` | For a space `M` of weight-1 forms, the subspace spanned by the cusp forms attached to dihedral Galois representations. |

*Worked example:* H132E12 (cuspidal, Eisenstein, new and zero subspaces of `M_2(Γ_0(33))`; `CuspidalProjection`/`EisensteinProjection` summing to the original form).

---

## 132.9 Operators

Each space `M` comes with a commuting family `T_1, T_2, T_3, …` of Hecke operators. Computation
of Hecke and other operators on spaces with nontrivial character is not yet implemented, though
computation of characteristic polynomials of Hecke operators is supported.

| Intrinsic | Description |
|-----------|-------------|
| `HeckeOperator(M, n)` | The matrix representing the `n`-th Hecke operator `T_n` with respect to `Basis(M)`. (Currently `M` must be a space with trivial character and integral weight `≥ 2`.) |
| `HeckeOperator(n, f)` | The image under the Hecke operator `T_n` of the given modular form `f`. |
| `HeckePolynomial(M, n : Proof)` | The characteristic polynomial of the `n`-th Hecke operator `T_n`. In some situations more efficient than `CharacteristicPolynomial(HeckeOperator(M, n))`. `M` can be an arbitrary space of modular forms. Parameter: `Proof` (default `true`). |
| `AtkinLehnerOperator(M, q)` | The matrix representing the `q`-th Atkin–Lehner involution `W_q` on `M` with respect to `Basis(M)`. (Currently `M` must be a cuspidal space with trivial character and integral weight `≥ 2`.) |
| `AtkinLehnerOperator(q, f)` | The image under the involution `w_q` of the given modular form `f`. |

*Worked example:* H132E13 (`HeckePolynomial` on `S_2(Γ_1(13))` over **Z** and `F_2`; a Hecke operator matrix on `M_4(Γ_0(14))`; the Atkin–Lehner involution `W_3` on `S_2(Γ_0(33))` showing Hecke and Atkin–Lehner operators need not commute).

---

## 132.10 Eisenstein Series

The intrinsics below require that the base ring of `M` has characteristic 0. To compute mod `p`
eigenforms, use the `Reduction` intrinsic (see 132.14).

| Intrinsic | Description |
|-----------|-------------|
| `EisensteinSeries(M)` | List of the Eisenstein series associated to the modular forms space `M` (i.e. lying in `M ⊗ C`). |
| `IsEisensteinSeries(f)` | Returns `true` if the modular form `f` was created using `EisensteinSeries`. |
| `EisensteinData(f)` | The data `<χ, ψ, t, χ', ψ'>` that defines the Eisenstein series (modular form) `f`. Here `χ` is a primitive character of conductor `S`, `ψ` is primitive of conductor `M`, and `MSt` divides `N` (the level of `f`). The series associated to `(χ, ψ, t)` has `q`-expansion `c_0 + Σ_{m≥1} (Σ_{n|m} ψ(n) n^{k−1} χ(m/n)) q^{mt}`, where `c_0 = 0` if `S > 1` and `c_0 = L(1−k, ψ)/2` if `S = 1`. |

*Worked example:* H132E14 (Eisenstein series in `M_3(Γ_1(12))`; `EisensteinData`, characters, parents).

---

## 132.11 Weight Half Forms

Modular forms of weight 1/2 are constructed directly as `q`-expansions following the explicit
description of Serre and Stark **[SS77]**.

| Intrinsic | Description |
|-----------|-------------|
| `WeightOneHalfData(M)` | For a space `M` of modular forms of weight 1/2, returns the basis of the space as described by Serre and Stark. A list of tuples is returned; each tuple contains a character `ψ` and an integer `t`, and the corresponding modular form is the theta series defined as the sum over all integers `n` of `ψ(n) q^(t n²)`. |

---

## 132.12 Weight One Forms

Modular forms of weight 1 can be defined using the usual constructors. For these spaces,
`Dimension`, `CuspidalSubspace`, `EisensteinSubspace`, `EisensteinSeries`, a `qExpansionBasis`,
and Hecke operators are available, plus element arithmetic.

The algorithm: Eisenstein series are constructed directly as `q`-expansions. The cuspidal
eigenforms correspond to Galois representations; those corresponding to dihedral Galois
representations are obtained explicitly (from characters on ray class groups of quadratic
fields). If the dihedral forms span the full space of cusp forms, this is proved by comparing
with suitable spaces of integral-weight forms; if not, a `q`-expansion basis for the cuspidal
space is obtained using the integral-weight spaces (the most time-consuming part).

| Intrinsic | Description |
|-----------|-------------|
| `DihedralForms(M)` | For a space `M` of weight 1, the cuspidal eigenforms in `M` corresponding to dihedral Galois representations, broken up according to character. A list of tuples is returned; each tuple contains an element of the `DirichletCharacters` of `M`, followed by a list of eigenforms. |

---

## 132.13 Newforms

This section describes how to compute both cuspidal and Eisenstein newforms. The intrinsics
below require that the base ring of `M` has characteristic 0; to compute mod `p` eigenforms, use
the `Reduction` intrinsic (see 132.14).

| Intrinsic | Description |
|-----------|-------------|
| `NumberOfNewformClasses(M : Proof)` | The number of Galois conjugacy-classes of newforms associated to `M`, which must have base ring **Z** or **Q** (i.e. newforms lying in `M ⊗ C`). Parameter: `Proof` (default `true`). |
| `Newform(M, i, j : Proof)` | The `j`-th Galois-conjugate newform in the `i`-th Galois-orbit of newforms in `M` (base ring **Z** or **Q**). Parameter: `Proof` (default `true`). |
| `Newform(M, i : Proof)` | The first Galois-conjugate newform in the `i`-th orbit in `M` (base ring **Z** or **Q**). Parameter: `Proof` (default `true`). |
| `Newforms(M : Proof)` | Sorted list of the newforms associated to `M`, divided into Galois orbits. Parameter: `Proof` (default `true`). |
| `Newforms(I, M)` | The newforms associated to `M` with prespecified eigenvalues. Here `I` is a sequence `[⟨p_1, f_1(x)⟩, …, ⟨p_n, f_n(x)⟩]` of pairs, each a prime not dividing the level of `M` and a polynomial. Returns the newforms `Σ a_n q^n` in `M` such that `f_n(a_{p_n}) = 0`. (Only works when `M` is cuspidal and defined over **Q** or **Z**.) |

*Worked example:* H132E15 (newforms in `M_5(Γ_1(8))`; `NumberOfNewformClasses`, `Newform`, `IsEisensteinSeries`; picking out a newform in `S_2(Γ_0(65))` with prespecified eigenvalues).

### 132.13.1 Labels

A Galois-conjugacy class of newforms can be obtained by giving a descriptive label to
`Newforms`. The label format is `[G0N or G1N][Level]k[Weight][Isogeny Class]`. Example labels:
`"G0N11k2A"`, `"G0N1k12A"`, `"G1N17k2B"`, `"G1N9k3B"`. If `"G0N"` or `"G1N"` is omitted, the
default is `"G0N"` (so `"11k2A"`, `"1k12A"`, `"37k4A"` are valid). If `k[Weight]` is omitted, the
default is weight 2 (so `"11A"`, `"37A"`, `"65B"` refer to weight-2 forms on some `Γ_0(N)`). The
isogeny class possibilities are `A, B, C, …, Z, AA, BB, …, ZZ, AAA, …`. This is essentially the
notation of **[Cre97]** (though for levels `≤ 450` the ordering sometimes differs). If `s` is a
valid label and `M` the space containing `ModularForm(s)`, then `ModularForm(s)` is by
definition `Newforms(M)[i]` where the isogeny class in `s` is the `i`-th isogeny class (`C` is
the 3rd, `BB` the 28th).

| Intrinsic | Description |
|-----------|-------------|
| `Newforms(label)` | The Galois-conjugacy class(es) of newforms described by the string `label`. |

*Worked example:* H132E16 (many examples of constructing newforms from labels: `"11A"`, `"G0N11k2A"`, `"G1N17k2B"`, `"G1N9k3B"`, `"37k4A"`, `"37k2"`, etc.).

---

## 132.14 Reductions and Embeddings

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Reductions(f, p)` | The mod `p` reductions of the modular form `f`, where `p ∈ Z` is prime and `f` is a modular form over a number field (or the rationals or integers). Because of denominators, the list of reductions might be empty. (The current algorithm is not close to optimal for `f` over a field of large degree.) | Reduction of newform `q`-expansions modulo primes above `p`. |
| `pAdicEmbeddings(f, p)` | The `p`-adic embeddings of the modular form `f`. | — |
| `ComplexEmbeddings(f)` | The complex embeddings of the modular form `f`. | — |

*Worked example:* H132E17 (a degree-4 newform in `S_2(Γ_0(47))`; `Reductions`, `pAdicEmbeddings` with increased `p`-adic precision via `pAdicField`, `ComplexEmbeddings`).

---

## 132.15 Congruences

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CongruenceGroup(M1, M2, prec)` | A group `C` measuring all possible congruences (to precision `prec`) between some modular form in `M1` and some modular form in `M2`. With `W_1 = q-exp(M1) ∩ Z[[q]]` and `W_2 = q-exp(M2) ∩ Z[[q]]`, and `V` the saturation of `W_1 + W_2` in `Z[[q]]`, then `C = V/(W_1 + W_2)`. | Saturation of the sum of `q`-expansion lattices. |
| `CongruenceGroupAnemic(M1, M2, prec)` | Analogous to `CongruenceGroup`, but considering congruences that hold for all `q`-expansion coefficients `a_n` with `n` coprime to the levels of both `M1` and `M2`. | As above, restricted to anemic coefficients. |

*Worked example:* H132E18 (the rank-2 elliptic-curve newform congruent mod 5 to a Galois-conjugate newform of the winding quotient of `J_0(389)`; `CongruenceGroup`, verified via `Reductions` mod 5).

---

## 132.16 Overconvergent Modular Forms

These routines compute characteristic series of operators on overconvergent modular forms.
While these are `p`-adic modular forms, the result also gives information about classical spaces:
it determines the characteristic series up to a congruence. The big advantage is that extremely
large weights can be handled (the method works by indirectly computing a small part of the large
space). The algorithm has running time linear in `log(k)`; the level-1 implementation is well
optimized. The algorithm is given in Algorithms 1 and 2 of **[Lau11]**; suggestions of David
Loeffler and John Voight are used in generating certain spaces of classical modular forms for
level `N ≥ 2`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `OverconvergentHeckeSeriesDegreeBound(p, N, k, m)` | A bound on the degree of the characteristic series modulo `p^m` of the Atkin `U_p` operator on the space of overconvergent `p`-adic modular forms of (even) weight `k` and level `Γ_0(N)`. This bound is due to Daqing Wan and depends only on `k` modulo `p−1` rather than `k` itself. | Daqing Wan's degree bound **[Lau11]**. |
| `OverconvergentHeckeSeries(p, N, k, m)` | The characteristic series `P(t)` modulo `p^m` of the Atkin `U_p` operator on overconvergent `p`-adic modular forms of level `Γ_0(N)` and weight `k`; `p ≥ 5` prime not dividing `N`. When `m ≤ k−1`, by Coleman's theorem `P(t)` is also the reverse characteristic polynomial mod `p^m` of `U_p` on classical forms of level `Γ_0(Np)` and weight `k`; and when `m ≤ (k−2)/2`, the reverse characteristic polynomial mod `p^m` of `T_p` on classical forms of level `Γ_0(N)` and weight `k`. Parameter: `WeightBound` (even integer, default 6) bounding the weight of generators chosen for certain classical spaces (level `N ≥ 2`); the algorithm may fail to terminate for some small levels unless `WeightBound` is large enough, but the output is provably correct whenever it terminates. | Algorithms 1 and 2 of **[Lau11]**; Coleman's theorem links to classical spaces. |
| `OverconvergentHeckeSeries(p, N, kseq, m)` | A sequence containing `OverconvergentHeckeSeries(p, N, k, m)` for each weight `k` in `kseq`; these weights must be congruent to each other modulo `p−1`. More efficient than computing them separately, since much of the work is shared. Parameter: `WeightBound` (default 6). | As above **[Lau11]**. |

*Worked example:* H132E19 (`OverconvergentHeckeSeries(5, 11, 10000, 5)` with and without `WeightBound := 4`; sequences of congruent weights; `OverconvergentHeckeSeriesDegreeBound`; large weight at level 1).

---

## 132.17 Algebraic Relations

| Intrinsic | Description |
|-----------|-------------|
| `Relations(M, d, prec)` | The relations of degree `d` satisfied by the `q`-expansions of the forms in the space `M` of modular forms; `q`-expansions are computed to precision `prec`. If `prec` is too small, the intrinsic might return relations not really satisfied. To be sure, `prec` must be at least `PrecisionBound(M2)`, where `M2` has the same level as `M` and weight `d` times the weight of `M`. |

*Worked example:* H132E20 (canonical embedding of `X_0(34)` via `Relations(S, 4, 20)`; canonical embedding of `X_0(75)`; the connection to models for modular curves is discussed in Steven Galbraith's Oxford Ph.D. thesis).

---

## 132.18 Elliptic Curves

Little has been implemented so far.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ModularForm(E)` / `Newform(E)` | The modular form associated to the elliptic curve `E` (which must be defined over the rationals). | — |
| `Eigenform(E, prec)` / `qEigenform(E, prec)` | The `q`-expansion of the newform associated to `E`, to the specified precision. (Exactly the same as `qExpansion(ModularForm(E), prec)`.) | — |
| `EllipticCurve(f)` | An elliptic curve `E` with associated modular form `f`, when `f` is a weight-2 newform on `Γ_0(N)` with rational Fourier coefficients. | The Cremona database is used to identify the isogeny class. A from-scratch routine is available via `EllipticCurve(M : Database := false)` for the relevant space of modular symbols (not optimized for large level). |

*Worked example:* H132E21 (the rank-2 newform in `M_2(Γ_0(389))`; `EllipticCurve(f)` recovering `y² + y = x³ + x² − 2x` of conductor 389, faster `PowerSeries` because the curve is known).

---

## 132.19 Modular Symbols

| Intrinsic | Description |
|-----------|-------------|
| `ModularSymbols(M)` | The sequence of characteristic-0 spaces of modular symbols associated to the space `M` of modular forms, when this makes sense. |
| `ModularSymbols(M, sign)` | The sequence of characteristic-0 spaces of modular symbols with given `sign` associated to `M`, when this makes sense. |

*Worked example:* H132E22 (`ModularSymbols(M, +1)` / `(M, −1)` for `Gamma0(389)`; the multi-character decomposition for `Gamma1(13)`).

---

## 132.20 Bibliography

| Key | Reference |
|-----|-----------|
| **[Atk67]** | A. O. L. Atkin. *Weierstrass points at cusps Γ_o(N).* Ann. of Math. (2), **85**:42–45, 1967. |
| **[Cre97]** | J. E. Cremona. *Algorithms for modular elliptic curves.* Cambridge University Press, Cambridge, second edition, 1997. |
| **[DI95]** | Fred Diamond and Ju Im. *Modular forms and modular curves.* In *Seminar on Fermat's Last Theorem*, pages 39–133. Amer. Math. Soc., Providence, RI, 1995. |
| **[Lau11]** | A. Lauder. *Computations with classical and p-adic modular forms.* LMS J. Comput. Math., **14**:214–231, 2011. |
| **[SS77]** | J.-P. Serre and H. M. Stark. *Modular forms of weight 1/2.* In *Modular functions of one variable, VI (Proc. Second Internat. Conf., Univ. Bonn, Bonn, 1976)*, pages 27–67. Lecture Notes in Math., Vol. 627. Springer, Berlin, 1977. |
| **[Ste07]** | William A. Stein. *Modular forms: a computational approach*, volume 79 of *Graduate Studies in Mathematics*. Amer. Math. Soc., Providence, Rhode Island, 2007. |

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Hermite-normal-form `q`-expansion bases over **Z** | `Basis`, `qExpansionBasis`, `RModule`/`RSpace`/`VectorSpace` |
| Sturm bound for determining precision **[Ste07]** | `PrecisionBound` |
| Cohen–Oesterlé dimension formulas | `Dimension`, `DimensionByFormula` |
| Eisenstein series construction (explicit `q`-expansions) | `EisensteinSeries`, `EisensteinData`, `EisensteinSubspace`, `EisensteinProjection` |
| Serre–Stark weight-1/2 theta-series description **[SS77]** | `WeightOneHalfData`, `HalfIntegralWeightForms` |
| Dihedral Galois representations / ray class groups (weight 1) | `DihedralForms`, `DihedralSubspace` |
| Hecke / Atkin–Lehner operators and characteristic polynomials | `HeckeOperator`, `HeckePolynomial`, `AtkinLehnerOperator` |
| Newform decomposition into Galois orbits | `Newforms`, `Newform`, `NumberOfNewformClasses`, `IsNewform` |
| Cremona-style labels **[Cre97]** | `Newforms(label)` |
| Reductions and `p`-adic / complex embeddings of newforms | `Reductions`, `pAdicEmbeddings`, `ComplexEmbeddings` |
| Congruence groups via saturation of `q`-expansion lattices | `CongruenceGroup`, `CongruenceGroupAnemic` |
| Overconvergent `U_p` characteristic series (Wan/Coleman) **[Lau11]** | `OverconvergentHeckeSeries`, `OverconvergentHeckeSeriesDegreeBound` |
| Algebraic relations among `q`-expansions (modular-curve models) | `Relations` |
| Cremona database lookup for elliptic curves | `EllipticCurve`, `ModularForm(E)`, `Eigenform` |
| Modular-symbols interface | `ModularSymbols` |
