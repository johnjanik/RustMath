# Chapter 122 — Elliptic Curves over Q and Number Fields

**Handbook part:** XVI — Arithmetic Geometry
**Handbook pages:** 4005–4081 (PDF pages 4134–4215)

---

## Scope and overview

Chapter 122 covers functionality specific to elliptic curves defined over the rationals **Q** or over number fields. As a general goal the functions are developed in parallel for both settings, but the functionality available over **Q** is far ahead of that available for general number fields, both in range and efficiency. Functions declared for number fields may also be used over **Q** by constructing **Q** as `RationalsAsNumberField()` rather than `Rationals()`.

The chapter is organised into four major parts:

1. **Curves over Q (§122.2)** — the largest section, covering local invariants (conductor, Tamagawa numbers, Frobenius traces, Kodaira symbols), the Mordell–Weil group and rank (via 2-descent, analytic rank, Heegner points), height theory (Néron–Tate canonical heights, height pairing, regulators, p-adic heights), descent machinery (2-, 3-, 4-, 5/7-isogeny, 6-, 8-, 9-, 12-descent, Cassels–Tate pairing), analytic information (periods, L-function, root number, modular degree, BSD-conjectural data), integral and S-integral points, and the Cremona database interface.

2. **Curves over number fields (§122.3)** — Tate's algorithm for local information, Mordell–Weil group (torsion, rank bound), heights, 2-descent and Selmer groups, Cassels–Tate pairing, elliptic curve Chabauty, auxiliary étale algebra machinery, analytic information, and conductor search routines.

3. **Curves over p-adic fields (§122.4)** — a thin interface to the same Tate algorithm code used for number fields, returning local invariants and the root number.

4. **Bibliography (§122.5)** — 24 references covering descent algorithms, height theory, Heegner points, modular degree, and integral points.

Key algorithmic methods documented include: 2-descent / 2-isogeny descent (**[MSS96], [Cre01]**), 4-descent (**[MSS96], [Wom03]**), 3-descent (**[SS04], [CFO+08], [CFO+09], [CFO+]**), p-isogeny descent for p ∈ {5,7} (**[Fis00], [Fis01]**), 8-descent (Tom Fisher), 9-descent / second p-descent (**[Cre10], [CM12]**), 6- and 12-descent (**[Fis08]**), Heegner points and Gross–Zagier (**[GZ86]**), canonical heights (AGM/Mestre, σ-function of Cohen **[Coh93]**), p-adic heights (**[MT91], [MST06], [Har08]**), modular degree (**[Wat02]**), and integral/S-integral points (linear forms in elliptic logarithms). The Mordell–Weil rank is **not** guaranteed to be determined — the chapter warns explicitly that no unconditional rank algorithm is implemented.

---

## 122.1 Introduction

This chapter deals with functionality specific to elliptic curves defined over **Q** or over number fields. Functions declared for number fields may also be used over **Q** by constructing **Q** as `RationalsAsNumberField()` rather than `Rationals()`.

*(No intrinsics in this section.)*

---

## 122.2 Curves over the Rationals

### 122.2.1 Local Invariants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Conductor(E)` | The conductor of the elliptic curve `E` defined over **Q**. | Tate's algorithm at each bad prime. |
| `BadPrimes(E)` | Sequence of primes dividing the minimal discriminant of `E`; these are the primes of bad reduction. | From the minimal model discriminant. |
| `TamagawaNumber(E, p)` | Local Tamagawa number of `E` at prime `p`: the index `[E(Q_p) : E0(Q_p)]`. Returns 1 at good primes. | Tate's algorithm. |
| `TamagawaNumbers(E)` | Sequence of Tamagawa numbers at each bad prime of `E`. | Tate's algorithm at each bad prime. |
| `LocalInformation(E, p)` | Local data at prime `p`: returns tuple `<p, v_p(disc), f_p, c_p, KodairaSymbol, split>` and a local minimal model. The boolean `split` is false iff the curve has non-split multiplicative reduction. | Tate's algorithm. |
| `LocalInformation(E)` | Sequence of the above tuples for all primes dividing the discriminant of `E`. | Tate's algorithm. |
| `ReductionType(E, p)` | String describing reduction type at `p`: `"Good"`, `"Additive"`, `"Split multiplicative"`, or `"Nonsplit multiplicative"`. | From Tate's algorithm (needed because Kodaira symbols do not distinguish split/non-split multiplicative reduction). |
| `FrobeniusTraceDirect(E, p)` | Trace of Frobenius a_p(E) computed directly and efficiently (without creating GF(p)). Argument `p` is not checked for primality. | Direct point-count method. |
| `TracesOfFrobenius(E, B)` | Sequence of Frobenius traces a_p(E) for all primes p ≤ B. Very carefully optimised. | Highly optimised sieve/counting method. |

*Worked example: H122E1 (computing `TracesOfFrobenius` up to 10^6).*

### 122.2.2 Kodaira Symbols

Kodaira symbols have their own type `SymKod`. The generic types "In" and "In\*" match any specific type of that form for comparison purposes.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `KodairaSymbol(E, p)` | Kodaira symbol (reduction type) of `E` modulo `p`. | Tate's algorithm. |
| `KodairaSymbols(E)` | Sequence of Kodaira symbols at the bad primes of `E`. | Tate's algorithm. |
| `KodairaSymbol(s)` | Creates a Kodaira symbol from a string `s` (e.g. `"I0"`, `"III*"`, `"In"` for the generic type). | — |
| `h eq k` | Equality of Kodaira symbols; generic types (e.g. `"In"`) compare equal to any specific type of the same family. | — |
| `h ne k` | Logical negation of `eq`. | — |

*Worked example: H122E2 (searching a family for curves with Kodaira symbol I0\*).*

### 122.2.3 Complex Multiplication

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HasComplexMultiplication(E)` | For `E` over **Q** (or a number field): determines whether `E` has CM, returning true/false and, if CM, the discriminant of the CM quadratic order. Not suited to very high degree j-invariants or discriminants beyond a few thousand. | Analytic methods on the j-invariant. |

### 122.2.4 Isogenous Curves

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsogenousCurves(E)` | Set of curves Q-isogenous to rational `E`, plus the largest degree of a cyclic isogeny in the class. Applies Mazur's theorem to restrict candidates to p-isogenies for p = 2, 3, 5, 7, 13; finds rational roots of division polynomials (p=2,3) or fibres of X0(p) (p=5,7,13); uses tree-based methods to extend isogeny trees. First curve returned has minimal Faltings height. | Mazur's theorem + division polynomial roots + X0(p) fibres; tree-based extension. |
| `FaltingsHeight(E)` | Faltings height of rational `E`: −½ log Vol(E) where Vol(E) is the volume of the fundamental parallelogram. Parameter: `Precision` (integer). | AGM-trick due to Mestre. |

*Worked example: H122E3 (`IsogenousCurves` and `IsogenyFromKernel` for a curve with 3-isogenies).*

### 122.2.5 Mordell–Weil Group

The Mordell–Weil theorem states that E(Q) is finitely generated. **Warning:** no algorithm is guaranteed to determine the rank. `Rank` and `MordellWeilGroup` may return a lower bound; `RankBounds` is preferred to avoid confusion. The recommended comprehensive routine is `MordellWeilShaInformation`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `MordellWeilShaInformation(E: parameters)` / `DescentInformation(E: parameters)` | Uses all available Magma machinery (2-descent, 4-descent, Cassels–Tate pairings, 3-descent, analytic routines) to obtain as much information as possible about E(Q) and Sha(E). Returns: rank bounds, independent generators found, and Sha information tuples. Parameters: `RankOnly` (default false), `ShaInfo` (default false), `Silent` (default false). | Orchestration of 2/3/4-descent, Cassels–Tate, analytic rank, Heegner points. |
| `TorsionSubgroup(E)` | Torsion subgroup of E(Q) as an abstract abelian group A and map A → E. By Mazur's theorem A is C_k (k ∈ {1..10,12}) or C_2 × C_{2k} (k ∈ {1..4}). | Mazur's theorem; torsion point search. |
| `Rank(E: parameters)` / `MordellWeilRank(E: parameters)` | Rank of E(Q). May return only a lower bound; prints a warning the first time this happens. Parameter: `Bound` (default 150, bound on numerator/denominator of x-coordinates in 2-cover search). | 2-descent (mwrank-style); similar to Cremona's mwrank **[Cre01]**. |
| `RankBounds(E: parameters)` / `MordellWeilRankBounds(E: parameters)` | Lower and upper bounds on the rank. Does not warn if bounds differ. Parameter: `Bound` (default 150). | 2-descent. |
| `MordellWeilGroup(E: parameters)` / `AbelianGroup(E: parameters)` | Mordell–Weil group as abstract group A and map A → E. May issue rank warning. Parameters: `Bound` (default 150), `HeightBound` (default 15). | 2-descent + point search up to `HeightBound`. |
| `Generators(E)` | Sequence of generators: torsion generators first (in order of `TorsionSubgroup`), then free generators. | From `MordellWeilGroup`. |
| `NumberOfGenerators(E)` / `Ngens(E)` | Number of generators (length of `Generators(E)`). | — |
| `Saturation(points, n)` | Given points on E/Q and integer n, returns p-saturated sequence for all primes p ≤ n. Parameters: `TorsionFree` (default false), `OmitPrimes` (default []), `Check` (default true). | p-saturation via height pairing. |

*Worked examples: H122E4 (rank 2 curve, rank 0 curve with nontrivial Sha), H122E5 (MordellWeilGroup with rank 2, generators), H122E6 (RankBounds for moderately large rank).*

### 122.2.6 Heights and Height Pairing

These functions require that the elliptic curve has integral coefficients.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `NaiveHeight(P)` / `WeilHeight(P)` | Naive (Weil) height h(P) = log max{|a|, |b|} for P = (a/b, c/d, 1) with integral coefficients. | Direct computation. |
| `Height(P: parameters)` / `CanonicalHeight(P: parameters)` | Canonical (Néron–Tate) height ĥ(P) = lim_{n→∞} 4^{−n} h(2^n P). Computed as sum of local heights; uses minimal model internally. Parameter: `Precision`. | Sum of local heights; archimedean part uses AGM-trick of Mestre, based on σ-function methods of **[Coh93]**. |
| `LocalHeight(P, p)` | Local height ĥ_p(P) at prime `p` (or p=0 for archimedean). Parameters: `Precision`, `Check` (default false), `Renormalization` (default false). | Local height formula; archimedean via AGM (Mestre). |
| `HeightPairing(P, Q: parameters)` | Height pairing ĥ(P,Q) = (ĥ(P+Q) − ĥ(P) − ĥ(Q)) / 2. Parameter: `Precision`. | From canonical height. |
| `HeightPairingMatrix(S: parameters)` / `HeightPairingMatrix(E: parameters)` | Height pairing matrix for a sequence of points, or for the Mordell–Weil generators of E. Parameter: `Precision`. | From canonical heights. |
| `Regulator(S)` / `Regulator(E)` | Determinant of the Néron–Tate height pairing matrix: for a sequence S, or for the free quotient generators of E(Q). Parameter: `Precision`. | Linear algebra over height matrix. |
| `SilvermanBound(E)` | Silverman bound B: for all P on E, h(P) − ĥ(P) ≤ B. | **[Sil86]**-style bound from model coefficients. |
| `SiksekBound(E: parameters)` | Siksek bound B (requires minimal model): h(P) − ĥ(P) ≤ B, generally much better than Silverman. Parameter: `Torsion` (default false; if true returns improved B_Tor such that h(P+T) − ĥ(P) ≤ B_Tor for some torsion T). | Algorithm of **[Sik95]**. |
| `IsLinearlyIndependent(P, Q)` | True iff P and Q are linearly independent in E(Q) modulo torsion. If false, returns vector v with vP + sQ torsion. | Height pairing (vanishing implies dependence). |
| `IsLinearlyIndependent(S)` | True iff points in sequence S are linearly independent modulo torsion. If false, returns kernel vector of height pairing matrix. | Height pairing matrix kernel. |
| `ReducedBasis(S)` | Returns a subset of S generating the same free subgroup of E(Q)/E_tors(Q), with non-degenerate height pairing. | LLL-reduction of height pairing matrix. |
| `pAdicHeight(P, p)` | p-adic height of P on E/Q for prime p ≥ 5 of good ordinary reduction. Parameters: `Precision` (default 0), `E2` (Eisenstein series value, default 0). | Algorithm of **[MT91]** with improvements of **[MST06]** and **[Har08]**; normalisation of **[Har08]** (2p or −2p times other papers). |
| `pAdicRegulator(S, p)` | p-adic regulator (height pairing matrix determinant) for set of points S at ordinary prime p ≥ 5. Parameters: `Precision`, `E2`. | From p-adic heights **[MT91, MST06, Har08]**; normalisation divides out a power of p. |
| `EisensteinTwo(E, p)` | Value of the Eisenstein series E_2 for E at ordinary prime p ≥ 5, via Monsky–Washnitzer / Kedlaya. Parameter: `Precision`. | Monsky–Washnitzer cohomology via Kedlaya's algorithm **[MT91, MST06, Har08]**. |

*Worked examples: H122E7 (canonical heights and local heights), H122E8 (Silverman vs Siksek bound comparison), H122E9 (linear independence test on 8-torsion + rank 4 curve), H122E10 (p-adic heights and regulators).*

### 122.2.7 Two-Descent and Two-Coverings

Two-descent determines the locally soluble two-coverings of E/Q as hyperelliptic curves C : y² = f(x) with f of degree 4. The algorithms over Q have been revisited and are faster than the general number field machinery.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `TwoDescent(E: parameters)` | 2-descent on E/Q: returns a sequence of hyperelliptic curves (quartics) representing elements of the 2-Selmer group. Parameters: `RemoveTorsion` (default false), `RemoveGens` (default {}), `WithMaps` (default true), `Verbose TwoDescent` (max 1). | 2-descent (Selmer group via quartics) **[Cre01]**. |
| `AssociatedEllipticCurve(f)` / `AssociatedEllipticCurve(C)` | Minimal model of the elliptic curve associated to a two-covering polynomial or hyperelliptic curve, plus a map from cover points to the curve. Optional: `E` (isomorphic to Jacobian). | Classical association from quartic/hyperelliptic curve. |

*Worked example: H122E11 (TwoDescent on a rank 2 curve, mapping points back to E via AssociatedEllipticCurve).*

#### 122.2.7.1 Two Descent Using Isogenies

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `TwoIsogenyDescent(E : parameters)` | For E/Q admitting a 2-isogeny φ: E′ → E, computes 2-coverings for both φ and dual φ′. Returns: seq of covers of E for φ, maps C→E, covers C′ of E′ for φ′, maps C′→E′, isogenies φ and φ′. Parameters: `Isogeny`, `TwoTorsionPoint`. | 2-isogeny descent (degree-2 covering maps rather than degree-4). |
| `LiftDescendant(C)` | Performs higher descent on curves from `TwoIsogenyDescent(E)`: returns 2-coverings D of E such that D→E factors through C, plus maps D→C and C→E. Works entirely over the base field. | Lifting from 2-isogeny level to full 2-descent level. |

#### 122.2.7.2 Invariants of Quartic Forms

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `QuarticIInvariant(q)` / `QuarticJInvariant(q)` | Classical I and J invariants of quartic q; generate the ring of integer invariants, satisfy J² − 4I³ = 27Δ(f). | Classical invariant theory **[Cre01]**. |
| `QuarticG4Covariant(q)` / `QuarticG6Covariant(q)` | Degree-4 and degree-6 covariants of quartic q. | Classical invariant theory **[Cre01]**. |
| `QuarticHSeminvariant(q)` / `QuarticPSeminvariant(q)` / `QuarticQSeminvariant(q)` / `QuarticRSeminvariant(q)` | Seminvariants of quartic q. H and P are related by H = −P. | Classical invariant theory **[Cre01]**. |
| `QuarticNumberOfRealRoots(q)` | Number of real roots of a real quartic q, computed via invariant theory. | Invariant-theory based. |
| `QuarticMinimise(q)` | Minimal model of quartic q over **Q** or a univariate rational function field. Returns: minimised quartic, transformation matrix, scaling factor. | Algorithm of **[CFS10]**. |
| `QuarticReduce(q)` | Reduced quartic and the reduction matrix. | Algorithm of **[Cre99]**. |
| `IsEquivalent(f, g)` | Determines whether quartics f and g are equivalent. | — |

### 122.2.8 The Cassels–Tate Pairing

The Tate–Shafarevich group admits an alternating bilinear form with values in Q/Z (the Cassels–Tate pairing); non-degenerate when Sha is finite (conjectured). Restricted to 2-torsion: a non-degenerate alternating form on Sha(E)[2]/2Sha(E)[4] with values in Z/2Z. Over **Q** a pairing value of 1 on 2-coverings C, D proves both are order-2 elements of Sha (no locally solvable 4-coverings above them). The pairing between 2-coverings is implemented over **Q**, number fields, and F(t) for F finite of odd characteristic; the pairing between 2- and 4-coverings over **Q**. A highly efficient implementation was released in Magma V2.15. Verbose: `SetVerbose("CasselsTate", n)` with n=1 or 2.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CasselsTatePairing(C, D)` (two 2-coverings) | Cassels–Tate pairing on 2-coverings of E over **Q**, a number field, or F(t). C and D must be y²=q(x) with q quartic, locally solvable. Returns value in Z/2Z. Verbose: `CasselsTate` (max 2). | Algorithms of Steve Donnelly (forthcoming paper); over **Q** reduces to solving a conic. |
| `CasselsTatePairing(C, D)` (4-cover, 2-cover) | Pairing between a 4-covering C (intersection of two quadrics in P³) and a 2-covering D, over **Q**. Returns Z/2Z. | Donnelly; key step is solving a conic over a degree-4 field. |

*Worked example: H122E12 (CasselsTatePairing proving nontrivial 2-torsion in Sha for 571a1).*

### 122.2.9 Four-Descent

Four-descent is performed on a two-cover (hyperelliptic curve y²=f(x), degree-4 f, no rational root). Introduced by **[MSS96]**; see also **[Wom03]**. A four-covering is a pair of symmetric 4×4 matrices defining an intersection of two quadrics in P³. FourDescent returns 2^{s−1} four-coverings where s is the Selmer 2-rank of the input.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `FourDescent(f : parameters)` / `FourDescent(C : parameters)` | 4-descent on y²=f(x) or a hyperelliptic curve C. Returns a set of 4-coverings (intersections of quadrics). Parameters: `RemoveTorsion` (default false), `IgnoreRealSolubility` (default false), `RemoveGensEC`, `RemoveGensHC`, verbose flags `FourDescent`/`LocalQuartic`/`MinimiseFD`/`QISearch`/`ReduceFD`/`QuotientFD`. | Merriman–Siksek–Smart 4-descent algorithm **[MSS96]**; see **[Wom03]**. |
| `AssociatedEllipticCurve(qi)` / `AssociatedHyperellipticCurve(qi)` | Associated elliptic and hyperelliptic curves for a quadric intersection `qi`, with maps to them. Optional: `E`. | — |
| `QuadricIntersection(F)` / `QuadricIntersection(P, F)` | Quadric intersection in P³ from a pair of symmetric 4×4 matrices F. | — |
| `QuadricIntersection(E)` / `QuadricIntersection(C)` | Write an elliptic curve E or hyperelliptic curve C as a quadric intersection. | — |
| `IsQuadricIntersection(C)` | Determines whether C is a quadric intersection (in P³ with two quadric equations), also returning the pair of matrices. | — |
| `PointsQI(C, B : parameters)` | Search for points on quadric intersection C of naive height ≤ B; asymptotic time O(B^{2/3}). Parameters: `OnlyOne` (default false), `ExactBound` (default false), verbose `QISearch`. | Efficient method of Elkies **[Elk00]**. |
| `TwoCoverPullback(H, pt)` / `TwoCoverPullback(f, pt)` | Pre-images on a two-covering (hyperelliptic curve or quartic) of a point on the elliptic curve. Faster than generic machinery. | — |
| `FourCoverPullback(C, pt)` | Pre-images on a four-covering (quadric intersection) of a point on the associated elliptic or hyperelliptic curve. | — |

*Worked example: H122E13 (4-descent proving rank 0 and Sha[2] = (Z/2)² for 571a1), H122E14 (4-descent, PointsQI with height 10^4, mapping back to E).*

### 122.2.10 Eight-Descent

8-descent is a further 2-descent on 4-coverings (intersections of two quadrics in P³) of E/Q. Determines whether such a curve has locally soluble 2-coverings. The 8-coverings are genus one normal curves of degree 8 in P⁸, minimised and reduced. Algorithm due to Tom Fisher (Magma 2.17), partly replacing an earlier implementation by Sebastian Stamminger.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `EightDescent(C : parameters)` | Further 2-descent on a curve C from `FourDescent`. Returns sequence of 8-descendant curves D and maps D→C. Parameters: `BadPrimesHypothesis` (default false), `DontTestLocalSolvabilityAt` (default {}), `StopWhenFoundPoint` (default false), verbose `EightDescent`/`LegendresMethod`. Requires class group / unit computations in degree 4 (and sometimes degree 8) number fields. | Tom Fisher's 8-descent algorithm (Magma 2.17); partly replaces Stamminger. |

### 122.2.11 Three-Descent

Three-descent computes the 3-Selmer group of E/Q and represents its elements as plane cubics. Full 3-descent requires class/S-unit computations in degree 8 number fields. For curves with a Q-rational 3-isogeny, descent by 3-isogenies is often more efficient (only quadratic field computations needed). Algorithm for the first step (3-Selmer group): **[SS04]**; second step (cubics): **[CFO+08], [CFO+09], [CFO+]**. Bulk of code by Michael Stoll and Tom Fisher.

Verbose flags: `Selmer`, `ThreeDescent`, `CSAMaximalOrder`, `Minimise`, `Reduce` (levels 0–3).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ThreeDescent(E : parameters)` | 3-Selmer group elements as plane cubics with covering maps. Returns one cubic per inverse pair of nontrivial elements, with covering maps. Parameters: `Method` (default `"HessePencil"`), verbose `Selmer`/`ThreeDescent`. | Calls `ThreeSelmerGroup` then `ThreeDescentCubic` on each element **[SS04, CFO+08, CFO+09, CFO+]**. |
| `ThreeSelmerGroup(E : parameters)` | 3-Selmer group of E/Q as an abelian group with map to the natural affine algebra. Parameters: `ThreeTorsPts`, `MethodForFinalStep` (default `"UseSUnits"`; alternatives `"Heuristic"`, `"FindCubeRoots"`), `CompareMethods` (default false), verbose `Selmer`. | **[SS04]**; class group and unit computations. |
| `ThreeDescentCubic(E, α : parameters)` | For E/Q and element α in the 3-Selmer group: returns a plane cubic C and map C→E. Parameters: `ThreeTorsPts`, `Method` (default `"HessePencil"`; alternatives `"FlexAlgebra"`, `"SegreEmbedding"`), verbose `ThreeDescent`. | **[CFO+08], [CFO+09], [CFO+]**. |
| `ThreeIsogenyDescent(E : parameters)` | Descent by 3-isogenies for E/Q admitting Q-rational 3-isogeny E→E1. Returns: cubics for Sel(E→E1), maps to E1, cubics for dual Sel(E1→E), maps to E, the isogeny. Optional: `Isog`, verbose. | Calls `ThreeIsogenySelmerGroups` then `ThreeIsogenyDescentCubic`; only quadratic field computations. |
| `ThreeIsogenySelmerGroups(E : parameters)` | Selmer groups for E→E1 and dual E1→E isogenies. Returns: group and map for E→E1, group and map for E1→E, the isogeny. Optional: `Isog`. | Selmer group computation in quadratic fields. |
| `ThreeIsogenyDescentCubic(φ, α)` | For a degree-3 isogeny φ between elliptic curves over Q and element α ∈ H¹(Q, E[φ]): returns a plane cubic C and covering map C→E of degree 3. Verbose `ThreeDescent`. | **[CFO+08], [CFO+09], [CFO+]**. |
| `ThreeDescentByIsogeny(E)` | Full 3-descent by first and second 3-isogeny descents; restricts to cubic extensions (sextic fields not needed). Verbose `Selmer`. | Composition of two 3-isogeny descents **[SS04, CFO+08, CFO+09, CFO+]**. |
| `Jacobian(C)` | Jacobian of a nonsingular projective plane cubic C over **Q** as an elliptic curve. | — |
| `ThreeSelmerElement(E, C)` / `ThreeSelmerElement(C)` | For E/Q and plane cubic C with same invariants as E: returns element α in the algebra A associated to the 3-Selmer group representing the same class. α is only determined up to inverse. | — |
| `AddCubics(cubic1, cubic2 : parameters)` | Sum in H¹(Q, E[3]) of two plane cubics with the same invariants (same Jacobian). Returns another plane cubic. Parameters: `E`, `ReturnBoth` (default false; if true returns both possible cubics). | Via `ThreeSelmerElement`, add cocycles, convert back. |
| `ThreeTorsionType(E)` | Classifies the Galois action on E[3]. Possible values: `"Generic"`, `"2Sylow"`, `"Dihedral"`, `"Generic3Isogeny"`, `"Z/3Z-nonsplit"`, `"mu3-nonsplit"`, `"Diagonal"`, `"mu3+Z/3Z"`. | Group-theoretic classification of the mod-3 representation. |
| `ThreeTorsionPoints(E : parameters)` | Tuple of one representative from each Galois orbit in E[3] \ O, defined over the appropriate field. Parameter: `OptimisedRep` (default true). | Enumeration of 3-torsion Galois orbits. |
| `ThreeTorsionMatrices(E, C)` | For plane cubic C with same invariants as E: tuple of matrices M_i corresponding to `ThreeTorsionPoints`, describing the action-by-translation on C in PGL3. | — |

*Worked examples: H122E15 (ThreeDescent for Selmer's 3x³+4y³+5z³=0, nontrivial 3-Sha of Jacobian), H122E16 (ThreeDescentByIsogeny vs ThreeDescent for large example).*

#### 122.2.11.1 Six and Twelve Descent

Combining 3-descent results with 2-descent or 4-descent gives degree-6 or degree-12 coverings (genus one normal curves of degree 6 in P⁵ and degree 12 in P¹¹). Algorithm: **[Fis08]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SixDescent(C2, C3)` / `SixDescent(model2, model3)` | From a 2-covering and a 3-covering of E (curves or genus one models): the 6-covering representing their sum in the 6-Selmer group, plus covering map C6→C3. | **[Fis08]**. |
| `TwelveDescent(C3, C4)` / `TwelveDescent(model3, model4)` | From a 3-covering and a 4-covering of E: two 12-coverings representing their sum and difference in the 12-Selmer group, plus covering maps C12→C4. | **[Fis08]**. |

### 122.2.12 Nine-Descent

A 9-descent is performed on an everywhere locally solvable plane cubic C (a class in the 3-Selmer group of its Jacobian up to sign). It computes the 3-Selmer set of C: the everywhere locally solvable 3-coverings of C, given as intersections of 27 quadrics in P⁸. Algorithm developed in the PhD thesis of Brendan Creutz **[Cre10]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `NineDescent(C : parameters)` | For plane cubic C over Q: returns sequence of curves in P⁸ (defined by 27 quadrics) and degree-9 maps making them 3-coverings of C (the 3-Selmer set of C). Parameters: `ExtraReduction` (default 10), verbose `Selmer`/`ComputeL`. Requires Galois to act transitively on flex points; otherwise use `pIsogenyDescent`. | Creutz's second 3-descent algorithm **[Cre10]**; class group/unit computations in the degree-9 étale algebra of flex points. |
| `NineSelmerSet(C)` | Computes the 3-rank of the 3-Selmer set of plane cubic C over Q. Returns −1 if the set is empty (C(Q) = ∅ and class is not 3-divisible in Sha). | **[Cre10]**. |

### 122.2.13 p-Isogeny Descent

Given an isogeny φ : E1 → E2, the φ-Selmer group / φ-Selmer set can be computed. For 2- and 3-isogenies use `TwoIsogenyDescent` / `ThreeIsogenyDescent`. For p ∈ {5, 7} with kernel generated by a Q-rational p-torsion point, use Fisher's algorithm **[Fis00, Fis01]**. Second isogeny descents for p = 3 or p ∈ {5, 7} (with flex points on coordinate hyperplanes) use Creutz's algorithm **[Cre10, CM12]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `pIsogenyDescent(E, P)` / `pIsogenyDescent(E, p)` / `pIsogenyDescent(lambda, p)` | φ-descent on E/Q for p ∈ {5,7} using Fisher's algorithm. Kernel generated by Q-rational p-torsion point P, or the point is chosen by specifying p, or by giving a Q-rational point λ on X₁(p). Returns: ranks of φ- and φ∨-Selmer groups, sequence of genus-one normal curves of degree p representing inverse pairs of nontrivial elements of φ∨-Selmer group, and the isogenous curve (plus E if λ was given). | Algorithm of Fisher **[Fis00, Fis01]**. |
| `pIsogenyDescent(C, phi)` / `pIsogenyDescent(C, E1, E2)` / `pIsogenyDescent(C, P)` | Isogeny descent on genus-one normal curve C of prime degree p ∈ {3,5}. C must be a plane cubic (p=3) or intersection of 5 quadrics in P⁴ with flex points on coordinate hyperplanes. Verbose `Selmer`. Returns: sequence of genus-one normal curves representing the φ-Selmer set, and covering maps. | Second isogeny descent, algorithm of Creutz **[Cre10, CM12]**. |
| `FakeIsogenySelmerSet(C, phi)` / `FakeIsogenySelmerSet(C, E1, E2)` / `FakeIsogenySelmerSet(C, P)` | Determines the F_p-dimension of the "fake" φ-Selmer set of genus-one normal curve C of degree p ∈ {3,5,7}. Returns −1 if empty. Verbose `Selmer`. For p=7, only the fake Selmer set is computable. | Creutz **[Cre10, CM12]**; feasible for p=7 (no coverings produced). |

*Worked examples: H122E17 (5-isogeny descent for 570l3, rank 0, nontrivial 5-Sha), H122E18 (full 5-descent to find large generator), H122E19 (fake Selmer set not surjected onto by genuine Selmer set).*

### 122.2.14 Heegner Points

For a rank-1 rational elliptic curve, the generator can be computed analytically: the elliptic logarithm of a multiple nP is the sum of the modular parametrization at CM points in the upper half-plane for a suitable quadratic field. The method implements Gross–Zagier **[GZ86]**; key contributions by Elkies (1994), Cremona, Womack, and Watkins. The calculation proceeds in three stages: choosing Q(√−d), evaluating the modular parametrization, and recovering the generator. The computation is O(h·N) terms to O(h) digits precision.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HeegnerPoint(E : parameters)` | Attempts to find a point on rank-1 rational E using Heegner points. Returns true and the point, or false. Parameters: `NaiveSearch` (default 1000), `Discriminant`, `Cover` (2- or 4-cover), `DescentPossible` (default true), `IsogenyPossible` (default true), `Traces`, verbose `Heegner`. Assumes Manin constant = 1. | Gross–Zagier method **[GZ86]**; tricks due to Watkins. |
| `HeegnerPoint(C : parameters)` / `HeegnerPoint(f : parameters)` | Utility versions: input is a hyperelliptic curve (quartic), quartic polynomial, or quadric intersection. Calls `HeegnerPoint` on the underlying elliptic curve then maps back to the covering. Parameters: `NaiveSearch` (default 10000), `Discriminant`, `Traces`, verbose. | **[GZ86]**. |
| `ModularParametrization(E, z, B : parameters)` / `ModularParametrization(E, z : parameters)` | Modular parametrization ∫_z^∞ f_E(τ)dτ for a point z in the upper half-plane. With bound B, uses first B terms of q-expansion. Optional `Traces`. | Complex analytic computation using q-expansion. |
| `ModularParametrization(E, Z, B : parameters)` / `ModularParametrization(E, Z : parameters)` | Same as above for an array Z of complex points. | — |
| `ModularParametrization(E, f, B : parameters)` / `ModularParametrization(E, f : parameters)` / `ModularParametrization(E, F, B : parameters)` / `ModularParametrization(E, F : parameters)` | Modular parametrization at a positive definite binary quadratic form f (or array F). Optional `Precision`, `Traces`. | — |
| `HeegnerDiscriminants(E, lo, hi)` | Negative fundamental discriminants in range [lo, hi] satisfying the Heegner hypothesis for E. Parameters: `Fundamental` (default false), `Strong` (default false). | Direct check of Heegner hypothesis. |
| `HeegnerForms(E, D : parameters)` | For E of conductor N and negative discriminant D satisfying the Heegner hypothesis: representatives for CM points on X₀(N). Returns sequence of 3-tuples (Q, m, T) where Q is a quadratic form, m a multiplicity, T a torsion point; the sum of m(φ(Q)+T) gives a Heegner point. Parameters: `UsePairing`, `UseAtkinLehner` (default true), `Use_wQ` (default true), `IgnoreTorsion` (default false). Requires ManinConstant = 1. | Heegner point theory; Atkin–Lehner involutions. |
| `HeegnerForms(N, D : parameters)` | For level N and discriminant D: sequence of binary quadratic forms for Heegner points. Parameter: `AtkinLehner` (default []). | — |
| `ManinConstant(E)` | (Conjectural) Manin constant of rational E; in most cases 1. | Conjectural, analytic/modular. |
| `HeegnerTorsionElement(E)` | For integer Q with gcd(Q, N/Q)=1: torsion point corresponding to the period ∫_{i∞}^{w_Q(i∞)} f_E(τ)dτ. | Modular parametrization. |
| `HeegnerPoints(E, D : parameters)` | For E/Q and suitable discriminant D: images on E of CM points on X₀(N) associated to D. Returns tuple ⟨p_D, m⟩ where p_D is an irreducible polynomial (x-coordinates lie in the ring class field) and m is the multiplicity. Also returns one conjugate point. Parameters: `ReturnPoint` (default false), `Precision` (default 100), verbose. Not rigorous (involves complex recognition). | **[GZ86]**; heuristic check by reduction mod small primes. |

*Worked examples: H122E20 (HeegnerPoint with point search and analytic method), H122E21 (163-isogenous curve trick), H122E22 (descent + 4-cover to speed Heegner computation), H122E23 (manual HeegnerForms and ModularParametrization for 43A with D=−327), H122E24 (HeegnerPoints over class field, GaloisGroup check).*

### 122.2.15 Analytic Information

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Periods(E: parameters)` | Sequence of periods of the Weierstrass ℘-function for rational E; first element is the real period. Accepts non-minimal model. Parameter: `Precision`. | AGM-trick due to Mestre. |
| `EllipticCurveFromPeriods(om: parameters)` | From two complex numbers ω₁, ω₂ (with ω₂/ω₁ in upper half-plane, corresponding to an integral model), returns the minimal model over Q. Parameter: `Epsilon` (default 0.001; proximity to integers for −27c₄ and −54c₆). | Classical Eisenstein series. |
| `RealPeriod(E: parameters)` | Real period of the Weierstrass ℘-function for E to `Precision` digits. | AGM-trick (Mestre). |
| `EllipticExponential(E, z)` | For rational E and complex z: pair [℘(z), ℘′(z)]. For small precision uses algorithm from **[Coh93]**; for high precision uses Newton iteration on EllipticLogarithm. Returns sequence (not a point). | **[Coh93]** / Newton iteration. |
| `EllipticExponential(E, S)` | For rational E and sequence S = [p, q]: elliptic exponential of p·RealPeriod + q·imaginary period. | — |
| `EllipticLogarithm(P: parameters)` | Elliptic logarithm φ(P) of point P (as complex number), with −ω₁/2 ≤ Re(φ) < ω₁/2 and −ω₂/2 ≤ Im(φ) < ω₂/2. Accepts non-minimal model. Parameter: `Precision`. | AGM-trick due to Mestre. |
| `EllipticLogarithm(E, S)` | For E and sequence S = [z₁, z₂] approximating a point P: elliptic logarithm φ(P). Parameters: `Precision`, `Check` (default true). | AGM-trick (Mestre). |
| `pAdicEllipticLogarithm(P, p: parameters)` | p-adic elliptic logarithm of P at prime p to `Precision` (default 50) digits. Order of P must not be a power of p. | p-adic logarithm. |
| `RootNumber(E)` | Global root number of E/Q: sign in the functional equation of L(E, s). Product of local root numbers at bad primes. | Local root numbers via Halberstadt's method; product formula. |
| `RootNumber(E, p)` | Local root number at prime p. For p > 3: straightforward from valuation of discriminant. For p = 2, 3: careful analysis of reduction type. | Halberstadt's method. |
| `AnalyticRank(E)` | Analytic rank of rational E: heuristic, computes derivatives of L(E,s) at s=1 until one is nonzero; returns rank and first nonzero L^{(r)}(1)/r!. Parameter: `Precision` (default 5; time increases sharply above 50 digits). Verbose `AnalyticRank`. | L-function derivatives via local power series; heuristic. |
| `ConjecturalRegulator(E)` | Assuming BSD, approximation to Reg(E) · |Sha(E)|. Returns value and (assumed) analytic rank. Parameter: `Precision`. | `AnalyticRank` + BSD formula. |
| `ConjecturalRegulator(E, v)` | Same but taking the precomputed first nonzero L^{(r)}(1)/r! = v. | BSD formula directly. |
| `ModularDegree(E)` | Modular degree deg(φ) of rational E via L(Sym²E, 2). Formula: deg(φ) = L(Sym²E,2)/(2π·Ω)·(Nc²)·∏ E_p(2). Assumes Manin constant = 1 (except cases noted in **[SW02]**). Verbose `ModularDegree`. | Algorithm of **[Wat02]**: real-number approximations converging to an integer. |

*Worked examples: H122E25 (EllipticExponential and EllipticLogarithm as inverses; adding points via Napier's method), H122E26 (AnalyticRank of a rank-6 curve, ConjecturalRegulator vs actual Regulator), H122E27 (ConjecturalRegulator at various precisions), H122E28 (ModularDegree via modular symbols and via the L-function formula).*

### 122.2.16 Integral and S-integral Points

Let S = {p₁, …, p_{s-1}, ∞}; finitely many S-integral points exist (Siegel's theorem). The algorithms use linear forms in complex and p-adic elliptic logarithms.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IntegralPoints(E)` | All integral points on E/Q modulo negation, plus representations in terms of a Mordell–Weil basis. Parameters: `FBasis` (precomputed free generators), `SafetyFactor` (extends the final search). | Linear forms in elliptic logarithms; `MordellWeilShaInformation` for generators. |
| `SIntegralPoints(E, S)` | All S-integral points on E/Q modulo negation. Same parameters as `IntegralPoints`. | Linear forms in elliptic logarithms + p-adic logarithms for p ∈ S. |
| `IntegralQuarticPoints(Q)` | All integral points (modulo negation) on y² = ax⁴+bx³+cx²+dx+e where e is a square. Input: sequence [a,b,c,d,e]. | Via `IntegralPoints` on the associated elliptic curve. |
| `IntegralQuarticPoints(Q, P)` | All integral points on the quartic y² = ax⁴+bx³+cx²+dx+e given a rational point P = [x,y]. | As above with the given point as starting generator. |
| `SIntegralQuarticPoints(Q, S)` | S-integral points on y² = ax⁴+bx³+cx²+dx+e (a must be a square) for set S of primes. | Linear forms. |
| `SIntegralLjunggrenPoints(Q, S)` | S-integral points on C : ay² = bx⁴+cx²+d for set S; requires C nonsingular. Input: [a,b,c,d]. | Linear forms. |
| `SIntegralDesbovesPoints(Q, S)` | S-integral points on C : ay³+bx³+cxy+d = 0 for set S; requires C nonsingular. Input: [a,b,c,d]. | Linear forms in elliptic logarithms for genus-1 cubic. |

*Worked examples: H122E29 (IntegralPoints on y²=x³+17), H122E30 (SIntegralPoints for S={2,3,5,7}, rank 2 curve), H122E31 (IntegralQuarticPoints), H122E32 (SIntegralDesbovesPoints).*

### 122.2.17 Elliptic Curve Database

Magma includes Cremona's database of all elliptic curves over Q up to conductor 200,000 (as of December 2011; example shows 130,000 loaded). All stored curves are global minimal models. Isogeny classes are indexed by integers.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `EllipticCurveDatabase(: parameters)` / `CremonaDatabase(: parameters)` | Returns a database object. Parameter: `BufferSize` (default 10000). | Cremona's tables **[Cre01]** extended. |
| `SetBufferSize(D, n)` | Sets the internal disk-read buffer to n bytes. | — |
| `LargestConductor(D)` | Largest conductor stored in D. | — |
| `ConductorRange(D)` | Smallest and largest conductors stored. | — |
| `#D` / `NumberOfCurves(D)` | Total number of curves in D. | — |
| `NumberOfCurves(D, N)` | Number of curves with conductor N. | — |
| `NumberOfCurves(D, N, i)` | Number of curves in the i-th isogeny class for conductor N. | — |
| `NumberOfIsogenyClasses(D, N)` | Number of isogeny classes for conductor N. | — |
| `EllipticCurve(D, N, I, J)` / `EllipticCurve(D, N, S, J)` | J-th curve of isogeny class I (integer or label string) for conductor N. | — |
| `EllipticCurve(D, S)` / `EllipticCurve(S)` | Curve with Cremona label S (e.g. `"101a"` or `"101a1"`). | — |
| `Random(D)` | A random curve from D. | — |
| `CremonaReference(D, E)` / `CremonaReference(E)` | Cremona label of E (e.g. `"101a1"`). E must be over Q with conductor in range. | — |
| `EllipticCurves(D, N, I)` / `EllipticCurves(D, N, S)` | Sequence of curves in isogeny class I (or label S) for conductor N. | — |
| `EllipticCurves(D, N)` | All curves for conductor N. | — |
| `EllipticCurves(D, S)` | Curves with label S (conductor or conductor+class). | — |
| `EllipticCurves(D)` | All curves in D (slow; prefer iteration). | — |

*Worked examples: H122E33 (database statistics, rank-3 curve 5077a1), H122E34 (iteration through database, random curve and isogeny class).*

---

## 122.3 Curves over Number Fields

Functions in this section are for elliptic curves over number fields K. The main items are: Tate's algorithm, 2-descent and descent by 2-isogenies, Cassels–Tate pairing, height machinery, and analytic tools.

### 122.3.1 Local Invariants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Conductor(E)` | Conductor of E defined over a number field (computed by `LocalInformation`). | Tate's algorithm. |
| `BadPlaces(E)` | Places of K of bad reduction for E (dividing discriminant of E). | — |
| `BadPlaces(E, L)` | Places of L of bad reduction for E/K where K ⊆ L. | — |
| `LocalInformation(E, P)` | Tate's algorithm at prime ideal P: returns tuple ⟨P, v_P(d), f_P, c_P, KodairaSymbol, s⟩ and a local minimal model E_min. Optional: `UseGeneratorAsUniformiser` (default false). | Tate's algorithm **[Sil86]**. |
| `LocalInformation(E)` | Sequence of `LocalInformation(E, P)` tuples for all P dividing discriminant of E. | Tate's algorithm. |
| `Reduction(E, p)` | Reduction of E (integral at p, good reduction at p) to an elliptic curve over the residue field of p, plus the reduction map. | — |

### 122.3.2 Complex Multiplication

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `HasComplexMultiplication(E)` | For E over a number field: whether E has CM, plus discriminant of the CM order if yes. Not suited to very high-degree j-invariants or discriminants beyond a few thousand. | Analytic methods on j-invariant. |

### 122.3.3 Mordell–Weil Groups

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `TorsionBound(E, n)` | Bound on the torsion subgroup size by looking at the first n non-inert primes of low ramification degree with good reduction. | Reduction mod primes. |
| `pPowerTorsion(E, p)` | p-power torsion of E over its base field as an abelian group and map. Optional: `Bound` (default −1). | — |
| `TorsionSubgroup(E)` | Torsion subgroup of E over a number field; uses `TorsionBound`. | Reduction + `TorsionBound`. |
| `MordellWeilShaInformation(E: parameters)` / `DescentInformation(E: parameters)` | Uses all relevant Magma machinery (2-descent, Cassels–Tate pairing, analytic routines when conductor has small norm) to determine Mordell–Weil group and Sha. Same arguments and returns as for E/Q (§122.2.5). Replaces the obsolete `PseudoMordellWeilGroup`. | 2-descent + Cassels–Tate + analytic rank. |
| `RankBound(E)` | Upper bound on rank of E(K) from 2-descent or 2-isogeny descent. Optional: `Isogeny`. | Selmer group computation. |

### 122.3.4 Heights

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `NaiveHeight(P)` | Absolute logarithmic height of the x-coordinate of P ∈ E(K). | — |
| `Height(P : parameters)` | Néron–Tate height ĥ(P) for P on E/K. Parameters: `Precision` (default 27), `Extra` (default 8; remedies precision loss when 2ⁿP near point at infinity). | Sum of local heights; Precision parameter controls output precision. |
| `HeightPairingMatrix(P : parameters)` | Height pairing matrix for an array of points. Same parameters as `Height`. | From Néron–Tate heights. |
| `LocalHeight(P, Pl : parameters)` | Local height λ_{Pl}(P) at a place Pl (finite or infinite) of K. Parameters: `Precision` (default 0), `Extra` (default 8). | Local height formula; archimedean: Extra remedies precision near ∞. |

### 122.3.5 Two Descent

The 2011 implementation is the first complete 2-descent over number fields, providing nice models of 2-coverings suitable for point searching. Key ingredients: `HasRationalPoint` for conics over number fields; minimisation and reduction techniques of Donnelly and Fisher.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `TwoDescent(E: parameters)` | Same arguments and returns as over Q; additionally returns a map from an abstract group (either `TwoSelmerGroup(E)` or a quotient) to the sequence of 2-coverings. Parameter: `MinRed` (controls minimisation/reduction, may be expensive over fields with large discriminant), plus RemoveTorsion, RemoveGens, WithMaps, verbose. | 2-descent over number fields (Donnelly–Fisher minimisation/reduction). |
| `TwoCover(e)` | Obtains a single 2-cover for element e in a cubic extension A/F. | From DescentMaps construction. |

### 122.3.6 Selmer Groups

Theory: for isogeny φ : E′ → E over K, the φ-Selmer group fits in the exact sequence 0 → S^(φ)(E/K) → H¹(K, E′[φ]) → ∏_p H¹(K_p, E′). Currently only 2-isogenies and multiplication-by-2 are implemented. For φ a 2-isogeny: H¹(K, E′[φ]) ≅ K*/K*²; for multiplication-by-2 with E : y²=f(x): A=K[x]/(f(x)) and H¹(K,E[2]) ≅ {δ ∈ A*/A*² | N_{A/K}(δ) ∈ K*²}. See **[Sil86]** for theory, **[Cas66]** for algorithms.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `DescentMaps(phi)` / `CasselsMap(phi)` | For isogeny φ : E → E1 over K (2-isogeny or multiplication-by-2): returns the connecting homomorphism µ : E1(K) → H¹(K, E[φ]) and the map τ : H¹(K, E[φ]) → homogeneous spaces. Elements of H¹ are in A*/A*². Parameter: `Fields` (for multiplication-by-2; used in `AbsoluteAlgebra`). | Cassels map construction **[Cas66, Sil86]**. |
| `SelmerGroup(phi)` | φ-Selmer group Sel^(φ)(E/K) as a finite abelian group S, with map AtoS : A → S. Standard map E1(K) → Sel(φ) is µ composed with AtoS. Parameters: `Hints` (x-coordinates to try first), `Raw` (default false; if true returns also toVec, FB, Hints), `Bound` (default −1; if positive passed to ClassGroup), verbose `Selmer`. | Class group and unit group computations **[Cas66]**; conditional on precomputed class group data if available. |
| `TwoSelmerGroup(E)` | 2-Selmer group of E over Q or a number field. Calls `SelmerGroup` for multiplication-by-2. Same options and returns. Verbose `EllSelmer`. | `SelmerGroup` for ×2 **[Cas66]**. |

*Worked examples: H122E35 (SelmerGroup for y²=x³+9x²−10x+1, rank bound 3), H122E36 (2-isogeny descent in three ways for y²=d·x(x+1)(x+3)), H122E37 (classic Kramer **[Kra81]** example, quadratic twist, base change), H122E38 (TwoDescent, TwoSelmerGroup, TwoCover for rank-3 curve).*

### 122.3.7 The Cassels–Tate Pairing

The pairing between elements of the 2-Selmer group over number fields is implemented in the same way as for curves over Q (see §122.2.8).

*(No additional intrinsics beyond those described in §122.2.8.)*

### 122.3.8 Elliptic Curve Chabauty

Developed by Nils Bruin **[Bru03, Bru04]**: for a curve admitting a suitable map to E over an extension field, finds Q-rational images under a given map E → P¹.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Chabauty(MWmap, Ecov)` | For E/K: determines the subset of E(K) whose images under Ecov : E → P¹ are Q-rational. Arguments: MWmap : A → E (e.g. from PseudoMordellWeilGroup), Ecov : E → P¹. Returns V (found points), R (index bound). Parameters: `InertiaDegreeBound` (default 20), `SmoothBound` (default 50), `PrimeBound` (default 30), `IndexBound` (default −1), `InitialPrimes` (default 50), verbose `EllChab`. | Mordell–Weil sieve strategy **[Bru03, BS10]**. |
| `Chabauty(MWmap, Ecov, p)` | p-adic version: bounds the set of points in E(K) whose Ecov-images are Q-rational. Returns N (upper bound), V (found points), R (index bound), L (coset collection). Parameters: `Cosets`, `Aux`, `Precision`, `Bound`, verbose `EllChab`. | p-adic Chabauty method of **[Bru03, Bru02]**; see **[Bru04]** for examples. |

*Worked example: H122E39 (elliptic curve over Q(ζ₁₀), u : E → P¹, both Chabauty variants).*

### 122.3.9 Auxiliary Functions for Étale Algebras

Machinery for number fields and étale algebras (algebras Q[x]/p(x)) for computing Selmer groups. The "p-Selmer group of K" (or an étale algebra) relative to a finite set S of K-primes means K(S,p) ⊆ K*/(K*)^p.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AbsoluteAlgebra(A)` | For a separable commutative algebra over Q (or finite field): returns the isomorphic direct sum of absolute fields as a Cartesian product, plus isomorphisms. Parameter: `Fields` (suggest existing field representations). Result is cached (recomputed if Fields given). Verbose `EtaleAlg`. | Direct factorisation/splitting. |
| `pSelmerGroup(A, p, S)` | p-Selmer group of semi-simple algebra A: direct sum of p-Selmer groups of irreducible summands, with embedding in A*. S must be prime ideals of the underlying number field. Verbose `EtaleAlg`. | Class group/unit group computations. |
| `LocalTwoSelmerMap(P)` | For prime ideal P of a number field K: map K* → K*_P / K*²_P, codomain as finite abelian group. | Local completion computation. |
| `LocalTwoSelmerMap(A, P)` | For commutative algebra A over K and prime P of K: map A* → A*/A*² ⊗ K_P. Also returns sequence of records (one per number field in AbsoluteAlgebra(A)) with fields i, p, map, vmap. | Direct sum of `LocalTwoSelmerMap(Q)` over extensions Q of P. |

*Worked example: H122E40 (AbsoluteAlgebra of Q[x]/(x³−1) = Q × Q(ζ₃)).*

### 122.3.10 Analytic Information

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RootNumber(E, P)` | Local root number of E/K at prime ideal P. Formulae due to Rohrlich, Halberstadt, Kobayashi, and the Dokchitser brothers. | Local epsilon factors. |
| `RootNumber(E)` | Global root number of E/K: product of local root numbers over all places (−1 from each infinite place). Conjectural sign in functional equation L(E/K,s) ↔ L(E/K,2−s). | Product of local root numbers. |
| `AnalyticRank(E)` | Analytic rank of E/K: heuristic, computes L^{(r)}(1)/r! derivatives until nonzero. Also returns first nonzero value. Assumes analytic continuation and functional equation. Parameter: `Precision` (default 6). | L-function derivatives; heuristic. |
| `ConjecturalRegulator(E)` | Assuming BSD: approximation to Reg(E/K) · |Sha(E/K)|. Returns value and analytic rank. Parameter: `Precision` (default 10). | `AnalyticRank` + BSD formula. |
| `ConjecturalSha(E, Pts)` | For E/K and a sequence Pts of purported Mordell–Weil basis points: conjectural order of Sha(E/K) from BSD. Returns 0 if points are dependent/insufficient rank; returns n²·|Sha| if generators form a subgroup of index n. Parameter: `Precision` (default 6). | `ConjecturalRegulator` / det(HeightPairingMatrix). |

### 122.3.11 Elliptic Curves of Given Conductor

Search routines for elliptic curves with given conductor or good reduction outside a given set of primes over number fields (also usable over Q via `RationalsAsNumberField()`). Not a provable enumeration; aim is efficient search. Particularly effective when Frobenius traces are specified (e.g. to match a known modular form).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `EllipticCurveSearch(N, Effort)` / `EllipticCurveWithGoodReductionSearch(S, Effort)` | Search for elliptic curves with conductor equal to ideal N (first form) or with conductor supported only by primes in S (second form). Returns sequence of non-isomorphic curves found (possibly isogenous). `Effort` (integer) controls effort roughly linearly; Effort=400 uses all techniques. Parameters: `Full` (default false), `Max`, `Primes`, `Traces`, verbose `ECSearch`. | Multi-technique search; effort incremented exponentially when early stopping specified. |

---

## 122.4 Curves over p-adic Fields

These functions provide an interface to the same Tate algorithm code used for number fields, for elliptic curves over p-adic fields.

### 122.4.1 Local Invariants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Conductor(E)` | Conductor of E defined over a p-adic field. | Tate's algorithm. |
| `LocalInformation(E)` | Tate's algorithm for E over a p-adic field: returns ⟨P, v_P(d), f_P, c_P, KodairaSymbol, s⟩ and an integral minimal model E_min. P is the uniformizer of the ground field. | Tate's algorithm. |
| `RootNumber(E)` | Local root number of E defined over a p-adic field. | Local epsilon factor. |

---

## 122.5 Bibliography

| Key | Reference |
|-----|-----------|
| **[BC04]** | W. Bosma and J. Cannon, editors. *Discovering Mathematics with Magma.* Springer-Verlag, Heidelberg, 2004. |
| **[Bos00]** | Wieb Bosma, editor. *ANTS IV,* volume 1838 of LNCS. Springer-Verlag, 2000. |
| **[Bru02]** | N. R. Bruin. *Chabauty methods and covering techniques applied to generalized Fermat equations,* volume 133 of CWI Tract. Stichting Mathematisch Centrum, Amsterdam, 2002. Dissertation, University of Leiden, 1999. |
| **[Bru03]** | Nils Bruin. Chabauty methods using elliptic curves. *J. reine angew. Math.,* 562:27–49, 2003. |
| **[Bru04]** | Nils Bruin. Some ternary Diophantine equations of signature (n, n, 2). In Bosma and Cannon **[BC04]**. |
| **[BS10]** | Nils Bruin and Michael Stoll. The Mordell–Weil sieve: Proving non-existence of rational points on curves. *LMS J. Comput. Math.,* 13:272–306, 2010. |
| **[Cas66]** | J. W. S. Cassels. Diophantine equations with special reference to elliptic curves. *J. London Math. Soc.,* 41:150–158, 1966. |
| **[CFO+]** | J. E. Cremona, T. A. Fisher, C. O'Neil, D. Simon, and M. Stoll. Explicit n-Descent on Elliptic Curves, III. Algorithms. ArXiv preprint. URL: http://arxiv.org/abs/1107.3516. |
| **[CFO+08]** | J. E. Cremona, T. A. Fisher, C. O'Neil, D. Simon, and M. Stoll. Explicit n-descent on elliptic curves. I. Algebra. *J. Reine Angew. Math.,* 615:121–155, 2008. |
| **[CFO+09]** | J. E. Cremona, T. A. Fisher, C. O'Neil, D. Simon, and M. Stoll. Explicit n-Descent on Elliptic Curves, II. Geometry. *J. reine angew. Math.,* 632:63–84, 2009. |
| **[CFS10]** | J. E. Cremona, T. A. Fisher, and M. Stoll. Minimisation and reduction of 2-, 3- and 4-coverings of elliptic curves. *Algebra & Number Theory,* 4(6):763–820, 2010. |
| **[CM12]** | B. Creutz and R. L. Miller. Second isogeny descents and the Birch and Swinnerton-Dyer conjectural formula. *J. Algebra,* 372:673–701, 2012. |
| **[Coh93]** | Henri Cohen. *A Course in Computational Algebraic Number Theory,* volume 138 of Graduate Texts in Mathematics. Springer, Berlin–Heidelberg–New York, 1993. |
| **[Cre99]** | John Cremona. Reduction of binary cubic and quartic forms. *LMS JCM,* 2:62–92, 1999. |
| **[Cre01]** | John Cremona. Classical invariants and 2-descent on elliptic curves. *J. Symbolic Comp.,* 31:71–87, 2001. |
| **[Cre10]** | Brendan Creutz. Explicit second p-descents on elliptic curves. PhD Thesis, Jacobs University Bremen, 2010. |
| **[Elk00]** | N. Elkies. Rational Points Near Curves and Small Nonzero |x³−y²| via Lattice Reduction. In Bosma **[Bos00]**, pages 33–63. |
| **[Fis00]** | Tom Fisher. On 5 and 7 descents for elliptic curves. PhD thesis, University of Cambridge, 2000. |
| **[Fis01]** | Tom Fisher. Some examples of 5 and 7 descent for elliptic curves over Q. *J. Eur. Math. Soc.,* 3(Issue 2):169–201, 2001. |
| **[Fis08]** | Tom Fisher. Finding rational points on elliptic curves using 6-descent and 12-descent. *J. Algebra,* 320(2):853–884, 2008. |
| **[FK02]** | Claus Fieker and David R. Kohel, editors. *ANTS V,* volume 2369 of LNCS. Springer-Verlag, 2002. |
| **[GZ86]** | B. Gross and D. Zagier. Heegner Points and Derivatives of L-series. *Invent. Math.,* 84:225–320, 1986. |
| **[Har08]** | D. Harvey. Efficient computation of p-adic heights. *LMS J. Comput. Math.,* 11:40–59, 2008. |
| **[Kra81]** | K. Kramer. Arithmetic of elliptic curves upon quadratic extension. *Trans. Amer. Math. Soc.,* 264(1):121–135, 1981. |
| **[MSS96]** | J. R. Merriman, S. Siksek, and N. P. Smart. Explicit 4-descents on an elliptic curve. *Acta Arith.,* 77(4):385–404, 1996. |
| **[MST06]** | B. Mazur, W. Stein, and J. Tate. Computation of p-adic heights and log convergence. *Documenta Mathematica,* Extra:577–614, 2006. |
| **[MT91]** | B. Mazur and J. Tate. The p-adic sigma function. *Duke Math. Journal,* 62(3):663–688, 1991. |
| **[Sik95]** | Samir Siksek. Infinite descent on elliptic curves. *Rocky Mountain J. Math.,* 25(4):1501–1538, 1995. |
| **[Sil86]** | J. Silverman. *The arithmetic of elliptic curves,* volume 106 of Graduate Texts in Mathematics. Springer-Verlag, New York, 1986. |
| **[SS04]** | Edward F. Schaefer and Michael Stoll. How to do a p-descent on an elliptic curve. *Trans. Amer. Math. Soc.,* 356(3):1209–1231 (electronic), 2004. |
| **[SW02]** | W. A. Stein and M. Watkins. A New Database of Elliptic Curves — First Report. In Fieker and Kohel **[FK02]**. |
| **[Wat02]** | M. Watkins. Computing the modular degree of an elliptic curve. *Experimental Mathematics,* 11(4):487–502, 2002. |
| **[Wom03]** | T. Womack. Explicit descent on elliptic curves. PhD thesis, University of Nottingham, 2003. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Tate's algorithm (local invariants, Kodaira symbols, Tamagawa numbers) | `Conductor`, `BadPrimes`, `TamagawaNumber(s)`, `LocalInformation`, `ReductionType`, `KodairaSymbol(s)`, `Reduction` |
| Mordell–Weil theorem / 2-descent (mwrank-style) **[Cre01]** | `Rank`, `MordellWeilGroup`, `RankBounds`, `Generators`, `TorsionSubgroup`, `Saturation` |
| Orchestrated descent + analytic rank | `MordellWeilShaInformation`, `DescentInformation` |
| Canonical height (AGM, Mestre; Cohen **[Coh93]**) | `Height`, `CanonicalHeight`, `LocalHeight`, `HeightPairing`, `HeightPairingMatrix`, `Regulator` |
| Silverman height bound **[Sil86]** | `SilvermanBound` |
| Siksek height bound **[Sik95]** | `SiksekBound` |
| p-adic heights **[MT91, MST06, Har08]** | `pAdicHeight`, `pAdicRegulator`, `EisensteinTwo` |
| 2-descent (Selmer group / quartics) **[Cre01]** | `TwoDescent`, `TwoSelmerGroup`, `SelmerGroup`, `DescentMaps`, `AssociatedEllipticCurve`, `TwoCover` |
| 2-isogeny descent | `TwoIsogenyDescent`, `LiftDescendant` |
| Quartic invariant theory **[Cre99, Cre01, CFS10]** | `QuarticIInvariant`, `QuarticJInvariant`, `QuarticG4Covariant`, `QuarticG6Covariant`, `QuarticH/P/Q/RSeminvariant`, `QuarticMinimise`, `QuarticReduce`, `IsEquivalent` |
| Cassels–Tate pairing (Donnelly) | `CasselsTatePairing` |
| 4-descent **[MSS96, Wom03]** | `FourDescent`, `AssociatedEllipticCurve(qi)`, `AssociatedHyperellipticCurve(qi)`, `QuadricIntersection`, `PointsQI`, `TwoCoverPullback`, `FourCoverPullback` |
| Elkies lattice search **[Elk00]** | `PointsQI` |
| 8-descent (Tom Fisher) | `EightDescent` |
| 3-descent **[SS04, CFO+08, CFO+09, CFO+]** | `ThreeDescent`, `ThreeSelmerGroup`, `ThreeDescentCubic`, `ThreeIsogenyDescent`, `ThreeIsogenySelmerGroups`, `ThreeIsogenyDescentCubic`, `ThreeDescentByIsogeny`, `Jacobian`, `ThreeSelmerElement`, `AddCubics`, `ThreeTorsionType`, `ThreeTorsionPoints`, `ThreeTorsionMatrices` |
| 6- and 12-descent **[Fis08]** | `SixDescent`, `TwelveDescent` |
| 9-descent (Creutz) **[Cre10]** | `NineDescent`, `NineSelmerSet` |
| p-isogeny descent for p=5,7 (Fisher) **[Fis00, Fis01]** | `pIsogenyDescent(E,P)`, `pIsogenyDescent(E,p)`, `pIsogenyDescent(lambda,p)` |
| Second isogeny descent (Creutz) **[Cre10, CM12]** | `pIsogenyDescent(C,phi)`, `FakeIsogenySelmerSet` |
| Heegner points / Gross–Zagier **[GZ86]** | `HeegnerPoint`, `ModularParametrization`, `HeegnerDiscriminants`, `HeegnerForms`, `ManinConstant`, `HeegnerTorsionElement`, `HeegnerPoints` |
| Periods / elliptic exponential / logarithm (AGM, Mestre; **[Coh93]**) | `Periods`, `EllipticCurveFromPeriods`, `RealPeriod`, `EllipticExponential`, `EllipticLogarithm`, `pAdicEllipticLogarithm` |
| Root number (Halberstadt; Rohrlich, Kobayashi, Dokchitser brothers) | `RootNumber` |
| Analytic rank (L-function derivatives) | `AnalyticRank` |
| BSD conjecture / conjectural data | `ConjecturalRegulator`, `ConjecturalSha` |
| Modular degree **[Wat02, SW02]** | `ModularDegree` |
| Integral/S-integral points (linear forms in elliptic logarithms) | `IntegralPoints`, `SIntegralPoints`, `IntegralQuarticPoints`, `SIntegralQuarticPoints`, `SIntegralLjunggrenPoints`, `SIntegralDesbovesPoints` |
| Isogenous curves (Mazur's theorem + X0(p) fibres) | `IsogenousCurves`, `FaltingsHeight` |
| Cremona database | `EllipticCurveDatabase`, `CremonaDatabase`, `EllipticCurve(D,…)`, `EllipticCurves(D,…)`, `CremonaReference` |
| Étale algebra / local Selmer maps | `AbsoluteAlgebra`, `pSelmerGroup`, `LocalTwoSelmerMap` |
| Elliptic curve Chabauty **[Bru02, Bru03, Bru04, BS10]** | `Chabauty` |
| Conductor search (number fields) | `EllipticCurveSearch`, `EllipticCurveWithGoodReductionSearch` |
