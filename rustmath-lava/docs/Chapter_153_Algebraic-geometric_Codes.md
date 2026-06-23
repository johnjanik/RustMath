# Chapter 153 — Algebraic-geometric Codes

**Handbook part:** XXI — Coding Theory
**Handbook pages:** 5147–5153 (PDF pages 5278–5287)

---

## Scope and overview

Algebraic–geometric codes (AG–codes) are a family of linear codes introduced by Goppa
**[Gop81a, Gop81b]**. Let `X` be an irreducible projective plane curve of genus `g`, defined
by an absolutely irreducible homogeneous polynomial `H(X, Y, Z)` over a finite field
`K = F_q`. A *place* of `X` is the maximal ideal of a discrete valuation subring of the
function field; `v_P` denotes the valuation at place `P`, and the *degree* of `P` is the degree
of its residue class field over `K`. A *divisor* `D = Σ n_P · P` (with all but finitely many
`n_P ∈ Z` zero) is an element of the free abelian group over the places of `X`; the *support*
of `D`, `Supp D`, is the set of places of nonzero multiplicity, and divisors carry the natural
partial order `D ≤ D'` iff `n_P ≤ n'_P` for all `P`. A divisor is *defined* over `K` if it is
stable under the natural action of `Gal(K̄/K)`.

Given a tuple of degree-1 places `(P_1, …, P_n)` and a divisor `D` defined over `K` with
support disjoint from `S = {P_1, …, P_n}`, the Riemann–Roch space
`L(D) = { f ∈ K(X)* | (f) + D ≥ 0 } ∪ {0}` (a `K`-vector space of dimension `k`) defines the
`[n, k]_q` algebraic–geometric code `C = C(S, D) = { (f(P_1), …, f(P_n)) : f ∈ L(D) }`. Its
generator matrix is `G = (f_i(P_j))` for a basis `f_1, …, f_k` of `L_K(D)`. Standard references
are **[Sti93]** and **[TV91]**.

There are two implementations of AG–code construction in Magma. The first, implemented by
Lancelot Pecquet, is based on the work of Haché **[HLB95, Hac96]**. The second exploits the
divisor machinery for function fields implemented by Florian Hess. As of Magma V2.8, only the
second implementation is exported. The chapter also covers dual (differential) codes,
Hermitian codes, AG-code property tests, specialized decoding of differential codes, and toric
codes.

---

## 153.2 Creation of an Algebraic Geometric Code

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AlgebraicGeometricCode(S, D)` / `AGCode(S, D)` | For `X` an irreducible plane curve, `S` a sequence of degree-1 places of `X`, and `D` a divisor whose support is disjoint from the support of `S`: the (weakly) algebraic–geometric code obtained by evaluating functions of the Riemann–Roch space of `D` at the points of `S`. The degree of `D` need not be bounded by the cardinality of `S`. | Evaluation of an `L(D)` basis at the places of `S` (Hess function-field divisor machinery) **[Sti93, TV91]**. |
| `AlgebraicGeometricDualCode(S, D)` / `AGDualCode(S, D)` | The dual of the algebraic geometric code constructed from the sequence of places `S` and the divisor `D`; corresponds to a differential code. To exploit the algebraic geometric structure, the dual must be constructed this way and **not** by directly calling `Dual`. | Differential / residue construction of the dual AG code. |
| `HermitianCode(q, r)` | For prime power `q` and positive integer `r`: a Hermitian code `C` with respect to the Hermitian curve `X = x^(q+1) + y^(q+1) + z^(q+1)` defined over `F_{q²}`. The support consists of all degree-1 places of `X` over `F_{q²}`, except the place over `P = (1 : 1 : 0)`; the divisor used to define the Riemann–Roch space is `r * P`. | AG-code construction on the Hermitian curve. |

*Worked examples:* H153E1 (a `[25, 9, 16]` code over `F_16` from the genus-1 curve
`x³ + x²z + y³ + y²z + z³`, choosing `D` from a place of degree `k + g − 1`); H153E2 (a
`[44, 12, 29]` code over `F_16` from a genus-4 curve, divisor `15 * P1` with `P1` a degree-1
place removed from the support).

---

## 153.3 Properties of AG–Codes

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsWeaklyAG(C)` | Returns `true` iff `C` is a weakly algebraic–geometric code, i.e. `C` was constructed as an algebraic–geometric code with respect to a divisor of any degree. | Attribute test on construction. |
| `IsWeaklyAGDual(C)` | Returns `true` iff `C` was constructed as the dual of a weakly algebraic–geometric code. | Attribute test on construction. |
| `IsAlgebraicGeometric(C)` | Returns `true` iff `C` is of algebraic–geometric construction of length `n`, built from a divisor `D` with `deg(D) < n`. | Attribute test plus degree condition. |
| `IsStronglyAG(C)` | Returns `true` iff `C` is an algebraic–geometric code of length `n` constructed from a divisor `D` satisfying `2g − 2 < deg(D) < n`, where `g` is the genus of the curve. | Attribute test plus the strong (Riemann–Roch) degree condition. |

---

## 153.4 Access Functions

At the time an AG–Code is constructed a number of attributes describing its construction are
stored along with the code. These functions give access to those attributes.

| Intrinsic | Description |
|-----------|-------------|
| `Curve(C)` | The curve from which `C` was defined. |
| `GeometricSupport(C)` | The sequence of places which forms the support for `C`. |
| `Divisor(C)` | The divisor from which `C` was constructed. |
| `GoppaDesignedDistance(C)` | For `C` constructed from a divisor `D`: the Goppa designed distance `n − deg(D)`. |

---

## 153.5 Decoding AG Codes

Specialized decoding algorithms exist for differential codes, those which are the duals of the
standard algebraic-geometric codes. These algorithms generally require as input another divisor
on the curve whose support is disjoint from the divisor defining the code.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AGDecode(C, v, Fd)` | Decode the received vector `v` of the dual algebraic geometric code `C` using the divisor `Fd`. | Specialized differential-code (basic algorithm) decoding using an auxiliary divisor. |

*Worked example:* H153E3 (an AG code over `GF(8)` with Goppa designed distance 3 used to
correct one error: `AGDualCode` of `11*plc` on `x³y + y³z + xz³`, decoded with `AGDecode(C, v, 4*plc)`).

---

## 153.6 Toric Codes

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ToricCode(P, q)` | The linear code `C` over `F_q` associated with the lattice points of the polygon `P`. After a translation so that the lattice points of `P` lie in the first quadrant as close to the origin as possible, the points must lie in the box `[0, q−2] × [0, q−2]`. The code is the monomial evaluation code where each point `(a, b)` corresponds to the monomial `x^a y^b`, evaluated at the points of the torus `(F_q*)²`. | Monomial evaluation over the torus `(F_q*)²`. |
| `ToricCode(S, q)` | The linear code `C` over `F_q` associated with the lattice points in `S` (a set or sequence). As usual, the points are translated to lie within a box at the origin of the first quadrant. | Monomial evaluation over the torus `(F_q*)²`. |

*Worked example:* H153E4 (a `[36, 19, 12]` toric code over `F_7` from the polygon with vertices
`(3,0), (5,0), (3,3), (1,5), (0,3), (0,1)`, compared against `BKLCLowerBound`).

---

## 153.7 Bibliography (canonical references)

| Key | Reference |
|-----|-----------|
| **[Gop81a]** | V. D. Goppa. *Codes on algebraic curves.* Dokl. Akad. Nauk SSSR, **259**(6):1289–1290, 1981. |
| **[Gop81b]** | V. D. Goppa. *Codes on algebraic curves.* Soviet Math. Dokl., **24**(1):170–172, 1981. |
| **[Hac96]** | Gaétan Haché. *Construction effective des codes géométriques.* PhD thesis, l'Université Paris 6, 1996. |
| **[HLB95]** | Gaétan Haché and Dominique Le Brigand. *Effective construction of algebraic geometry codes.* IEEE Trans. Inform. Theory, **41**(6, part 1):1615–1628, 1995. Special issue on algebraic geometry codes. |
| **[Sti93]** | Henning Stichtenoth. *Algebraic function fields and codes.* Springer-Verlag, Berlin, 1993. |
| **[TV91]** | M. A. Tsfasman and S. G. Vlăduţ. *Algebraic-geometric codes.* Kluwer Academic Publishers Group, Dordrecht, 1991. Translated from the Russian by the authors. |

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Goppa AG-code construction (Riemann–Roch evaluation) **[Gop81a, Gop81b, Sti93, TV91]** | `AlgebraicGeometricCode` / `AGCode`, `HermitianCode` |
| Differential (dual) AG-code construction | `AlgebraicGeometricDualCode` / `AGDualCode` |
| Effective AG-code construction (Pecquet/Haché, unexported as of V2.8) **[HLB95, Hac96]** | (historical first implementation) |
| AG-code property / attribute tests | `IsWeaklyAG`, `IsWeaklyAGDual`, `IsAlgebraicGeometric`, `IsStronglyAG` |
| AG-code attribute access | `Curve`, `GeometricSupport`, `Divisor`, `GoppaDesignedDistance` |
| Differential-code (basic algorithm) decoding | `AGDecode` |
| Toric monomial-evaluation codes | `ToricCode` |
</content>
</invoke>
