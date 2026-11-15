# RustMath handoff: the M23 Golay computations

Date: 2026-11-14. Written for a RustMath session that starts cold. Read `docs/HANDOFF.md` and
`CLAUDE.md` first, as usual; this file is the brief for one workstream.

The reference data, the Python reference implementation and the paper live in the public
repository `johnjanik/M23`, branch `claude/blissful-knuth-qla7ym`, pinned here at commit
`d05475b`. In this file every path that does not start with `rustmath-`, `docs/` or
`PROJECT_TRACKER.md` is relative to a checkout of that repository. Fetch it outside the RustMath
tree:

```bash
git clone --branch claude/blissful-knuth-qla7ym https://github.com/johnjanik/M23 ../M23
```

Nothing from it is vendored into RustMath yet. When a work package starts, copy the JSON files it
tests against into that crate's `tests/data/` with a note of the source commit. The same file is
kept in the M23 repository as `docs/RUSTMATH_HANDOFF.md`; this copy is the one to edit.

Three things are wanted from RustMath. Each has reference values computed independently
(Python/FLINT, cross-checked with PARI and GAP), so each is a test of RustMath as much as a
contribution to the paper `paper/golay_m23.pdf`.

| Work package | Goal | Reference data exists | New mathematics |
|---|---|---|---|
| WP-F | Reproduce and extend the Frobenius table of the genus 4 curve X | yes, 18 primes | no |
| WP-P | Reproduce and extend the table of three-point class vectors of M23 (Phase 6 of the research note) | yes, all 57 class vectors | the covers themselves are open |
| WP-A | Find an explicit polynomial for the A8 field cut out by the Golay Lagrangian (Question 10.1 of the paper) | acceptance tests only | yes, the field is unknown |

Do them in that order. WP-F and WP-P have exact expected outputs. WP-A does not: it has
predictions that the answer must satisfy, and a designed pipeline that has not been run.

## 1. Background in one page

Read `paper/golay_m23.pdf` Sections 1, 2 and 4 for the full statements. The minimum:

- **The curve.** Huang, Jackson, Lee, Poonen, Pries and Zhang (HJLPPZ, arXiv:2608.08538v2)
  give `F(T,V)` in `Z[T,V]` of bidegree (4,23). Its closure in P^1 x P^1 has exactly 62 nodes;
  the normalization `X` has genus 4. The map `t: X -> P^1` has degree 23, monodromy group M23,
  branch points `0, +sqrt(-23), -sqrt(-23)` with classes `(2A, 23A, 23B)`. The function `v` has
  degree 4.
- **The Golay Lagrangian.** The 23 sheets carry a Steiner system S(4,7,23). The even Golay
  code `G0` (dimension 11) is a sub-local-system of the permutation sheaf mod 2 and
  `W = H^1(P^1, j_* G0)` is a Galois-stable Lagrangian subgroup of `Jac(X)[2]`, of order 2^4
  (paper, Theorem 4.1).
- **What is known about W.** Galois acts on `W = F_2^4` through all of `GL_4(F_2) = A_8`
  (paper, Theorem 4.8). The kernel field `K_W` is an A8 extension of Q unramified outside
  `{2,3,5,23}`. Complex conjugation acts as a transvection. No polynomial for `K_W` is known.
- **Conventions.** Permutations act on the right: `x^(gh) = (x^g)^h`. Products are read left
  to right, as in MAGMA and GAP. Published cycles are 1-based; the Python library is 0-based.
  `23A`, `23B` are the classes of the published `g2`, `g3`. Class names of other split pairs
  (`7A/7B` and so on) are fixed only up to swapping the pair.

### Where the data is

Everything below is in the M23 repository unless a URL is given.

| File | Contents |
|---|---|
| `m23golay/` | Python reference implementation (permutations, Golay codes, M23, Nielsen classes, mod 2 cohomology, p-adic labelling, point counting) |
| `data/group_facts.json` | class table, cycle types on duads and heptads, Molien series, Nielsen class with real structure |
| `data/zeta_X.json` | point counts and L-polynomials of X for p <= 79 |
| `data/zeta_L.json` | the same for the six conjugate covers at degree-one primes of the sextic field |
| `data/h62.json` | `h62(T)`: `disc_V F = 2^1652 3^626 23^228 131^506 T^8 (T^2+23)^22 h62^2`; its roots are the T-coordinates of the 62 nodes |
| `data/golay_resolvents.json` | J5, J7, T5 and the degree 253 heptad and duad resolvents of the published polynomials f1, f2 |
| `data/J5_family.json` | J5 of the 23 roots of `F(t,V)` as `n(t)/c23(t)` |
| `data/lifts_2_11_M23.json` | fifteen generating triples of `2^11:M23` in S_46 (WP-A, route B) |
| `data/pg32_a8_model.json` | PG(3,2) in the cyclic model with its 56 spreads and the 8 letter families (WP-A, step A5) |
| `data/m23_three_point_passports.json` | 57 rational class vectors with structure constants (GAP character table) |
| `data/m23_three_point_nielsen.json` | exact sizes of the 57 Nielsen classes and numbers of real points |
| `data/passports_C_23A_23B.json` | the six class vectors `(C,23A,23B)` in detail: representatives, ovals |
| `scripts/` | the scripts that produced each file; the README has a command table |

External data, not vendored:

| Source | What | Licence note |
|---|---|---|
| `https://github.com/shaowuz/m23isgalois` | `polynomial_F.m` (the equation), `polynomial_calF.m` (model over the sextic field), f1, f2, seven triples | no licence file in the repository; fetch at test time, do not copy into RustMath |
| `https://github.com/research-reflexivity/frontier-math-m23-cover-investigation` | Le Lay's data: canonical models of all seven curves, marked points, the canonical quadric over Q, a second plane model | `data/` is CC0-1.0 and may be vendored with attribution; the paper and code are CC BY-NC-SA 4.0 and must not be copied into GPL code |

`scripts/fetch_external.sh` fetches the first. The parser for `polynomial_F.m` is five lines:
terms have the form `<int>*T^<i>*V^<j>` (see `m23golay/data.py`).

## 2. What RustMath can and cannot be trusted for

From `PROJECT_TRACKER.md` at RustMath commit `02d1bbf`. Check the tracker again before relying
on any row.

Usable as is:

- exact kernel: integers, rationals, `GF(p^n)` with Cantor-Zassenhaus factorization
  (`rustmath-finitefields/src/poly_factor.rs`), Galois rings `GR(p^a, d)` with Frobenius and
  Newton inversion (`galois_ring.rs`), which is exactly the ring `(Z/p^N)[x]/(f)` needed below;
- Zassenhaus factorization over Z, Hensel lifting, resultants, discriminants, LLL;
- number fields: round 2 maximal order, prime decomposition, polred, heuristic recognition
  (`rustmath-numberfields/src/recognize.rs`);
- permutation groups: deterministic Schreier-Sims (`rustmath-groups/src/bsgs.rs`), tested on
  M23 and M24 from generators;
- Belyi engine: triangle groups, coset graphs, high-precision modular-forms route
  (`rustmath-curves/src/belyi/`), used so far for a genus 0 cover.

Do not call (facade, broken or missing at that commit):

- `rustmath-curves`: genus, singularities, differentials, Riemann-Roch are facades or wrong.
  WP-F and WP-A need their own small exact routines, specified below.
- `rustmath-functionfields/src/genus.rs` is wrong outside its three tests.
- character tables and class-function inner products in `rustmath-groups` are broken (T-215);
  conjugacy classes exist only by brute force on a full element list.
- no zeta functions of curves, no subfields, no factorization over number fields, no period
  matrices or theta functions, interval arithmetic only on f64.
- `rustmath-galois`: exact engine up to degree 5; Stauduhar descent is an honest stub. The
  Galois groups needed here (A8 in degree 8 and 15) are certified by cycle types, see WP-A.

RustMath rules that apply: a bounded search that finds nothing never decides; certifying
routines return `Decided` or `Unresolved`; reference values come from data files, not from
memory; one crate per pull request.

## 3. WP-F: the Frobenius table

### 3.1 Specification

Input: the coefficients `F[i][j]` of `T^i V^j` (0 <= i <= 4, 0 <= j <= 23) and a prime p.
Output: `N_k = #X(F_{p^k})` for k = 1..4, the Frobenius polynomial
`P_p(T) = T^8 + a1 T^7 + a2 T^6 + a3 T^5 + a4 T^4 + p a3 T^3 + p^2 a2 T^2 + p^3 a1 T + p^4`,
and three derived checks.

Algorithm (reference: `m23golay/curve.py`, `count_points` with `fibre="V"`):

1. Fibre the plane model over the `V`-line. For each `v` in `P^1(F_q)` the fibre polynomial
   `F(T, v)` is a quartic in `T`. Count its distinct roots in `P^1(F_q)` (a drop of degree
   counts as a root at infinity).
2. For every repeated root `(t, v)`, decide whether it is a singular point of the model (both
   partial derivatives vanish). If it is smooth, it is a ramification point of the projection
   and counts once. If it is singular, compute the Hessian discriminant
   `F_TV^2 - F_TT F_VV`: nonzero square means a node with two rational branches (count 2),
   nonzero nonsquare means a node with conjugate branches (count 0), zero means the singular
   point is not a node (see 3.3).
3. Use Frobenius orbits of the base point: process only the orbit representative with the
   smallest code and weight by the orbit length. The reference implementation is stateless per
   chunk so that it parallelizes and bounds memory.
4. From `s_k = p^k + 1 - N_k` compute the elementary symmetric functions by Newton's
   identities, check that they are integers, and fill the upper half by the functional equation.
5. Weil check. The reference uses numerical roots. RustMath should do it exactly: write
   `P_p(T) = T^4 R(T + p/T)` with `R` a monic integer quartic and verify with a Sturm sequence
   that `R` has four real roots in `[-2 sqrt(p), 2 sqrt(p)]`.
6. Derived checks, all of which hold in the reference data:
   - `P_p(chi(p)) = P_p(chi(p) p) = 0 mod 23`, with `chi` the quadratic character of
     `Q(sqrt(-23))` (the 23-torsion class of the two totally ramified points);
   - `P_p mod 2 = Q Q^rev` for a quartic `Q` over `F_2` (the Golay Lagrangian);
   - both fibrations (over `V` with quartic fibres, over `T` with degree 23 fibres) give the
     same counts.

### 3.2 Reference values

`T` = model stays nodal mod p; `D` = the plane model has a non-nodal singular point mod p and the
count treats it as unibranch (accepted because it passes the Weil test, not proved).

| p | N1 | N2 | N3 | N4 | (a1, a2, a3, a4) | P_p mod 2 | model |
|---:|---:|---:|---:|---:|---|---|---|
| 5 | | | | | not obtained: not a Weil polynomial | | |
| 7 | 11 | 69 | 383 | 2433 | (3, 14, 46, 138) | `(x+1)^2 (x^3+x+1) (x^3+x^2+1)` | T |
| 11 | 15 | 133 | 1368 | 14905 | (3, 10, 33, 145) | `(x^4+x+1) (x^4+x^3+1)` | T |
| 13 | 11 | 161 | 2135 | 29089 | (-3, 0, -12, 188) | `(x+1)^2 (x^3+x+1) (x^3+x^2+1)` | D |
| 17 | 21 | 285 | 4923 | 82985 | (3, 2, 0, -130) | `(x+1)^2 (x^3+x+1) (x^3+x^2+1)` | T |
| 19 | 17 | 405 | 7265 | 130913 | (-3, 26, 66, 74) | `(x+1)^2 (x^3+x+1) (x^3+x^2+1)` | T |
| 29 | 35 | 789 | 24773 | 708785 | (5, -14, 16, 1060) | `(x+1)^2 (x^3+x+1) (x^3+x^2+1)` | T |
| 31 | 33 | 993 | 29640 | 926713 | (1, 16, -35, 875) | `(x^4+x+1) (x^4+x^3+1)` | D |
| 37 | 43 | 1433 | 50881 | 1881609 | (5, 44, 254, 3156) | `(x+1)^2 (x^3+x+1) (x^3+x^2+1)` | T |
| 41 | 42 | 1780 | 69408 | 2827728 | (0, 49, 162, 1692) | `(x+1)^4 (x^2+x+1)^2` | D |
| 43 | 44 | 1996 | 80192 | 3422024 | (0, 73, 228, 3470) | `(x+1)^4 (x^2+x+1)^2` | T |
| 47 | 47 | 2349 | 104321 | 4882281 | (-1, 70, 96, 2934) | `(x+1)^2 (x^3+x+1) (x^3+x^2+1)` | T |
| 53 | 54 | 2804 | 148704 | 7881808 | (0, -3, -58, -2164) | `(x+1)^4 (x^2+x+1)^2` | T |
| 59 | 72 | 3560 | 205056 | 12120780 | (12, 111, 648, 3991) | `(x^4+x^3+x^2+x+1)^2` | T |
| 61 | 64 | 3740 | 226852 | 13855856 | (2, 11, -24, 2476) | `(x+1)^4 (x^2+x+1)^2` | T |
| 67 | 80 | 4460 | 302192 | 20163724 | (12, 57, 584, 8759) | `(x^4+x^3+x^2+x+1)^2` | T |
| 71 | 68 | 5048 | 358256 | 25432076 | (-4, 11, 92, 4679) | `(x^4+x^3+x^2+x+1)^2` | T |
| 73 | 83 | 5541 | 390089 | 28393065 | (9, 146, 1428, 12030) | `(x+1)^2 (x^3+x+1) (x^3+x^2+1)` | D |
| 79 | 77 | 6369 | 491636 | 38964585 | (-3, 68, -663, 7335) | `(x^4+x+1) (x^4+x^3+1)` | T |

Certificate for good reduction at 7 and 11 (the two primes the proof uses):
`scripts/verify_curve_model.py` checks the discriminant identity over Z at 190 integers, that
`h62` stays squarefree of degree 62 and prime to `T(T^2+23)` mod p, and that the model has a node
above each of its 62 roots.

Covers over the sextic field `L = Q[s]/(s^6 - s^5 - 3s^4 - 3s^3 + s^2 + 5s + 4)`, model
`polynomial_calF.m`, reduced at the degree-one prime `(p, s - s0)`:

| p | s0 | N1 | N2 | N3 | N4 | (a1, a2, a3, a4) | Q up to reversal |
|---:|---:|---:|---:|---:|---:|---|---|
| 5 | 3 | | | | | model degenerate at this prime | |
| 7 | 5 | 7 | 45 | 343 | 2545 | (-1, -2, 2, 38) | `x^4+x^3+x^2+1` |
| 11 | 2 | | | | | model degenerate at this prime | |
| 19 | 3 | 15 | 413 | 6849 | 130169 | (-5, 38, -152, 650) | `x^4+x^3+x^2+1` |
| 31 | 12 | 44 | 1014 | 30266 | 923726 | (12, 98, 758, 5021) | `x^4+x^2+1` |
| 41 | 8 | 40 | 1790 | 68866 | 2828954 | (-2, 56, -128, 2402) | `x^4+1` |
| 43 | 15 | 43 | 1829 | 78799 | 3422769 | (-1, -10, -226, 1278) | `x^4+x^3+x^2+1` |

Published anchor: at 53 the three degree-one primes give 45, 57, 52 points over F_53; the values
45 and 52 are in HJLPPZ Proposition 3.14. This is in `tests/test_core.py`.

### 3.3 Extensions wanted

1. **Remove the D flags.** At p = 13, 31, 41, 73 recount on a second model with different
   singular points and reconcile. Candidates: Le Lay's bidegree (8,23) model
   (`data/Fint_coefficients_Z.json` in his repository, 84 nodes) or a plane projection of the
   canonical model. Expected: the same `N_k`.
2. **Decide p = 3 and p = 5.** The reference count at 5 fails the Weil test, which says only that
   this plane model is too degenerate there. Find a model that is nodal mod p with the right
   number of nodes, or show that the canonical model has singular reduction. This matters for
   WP-A: if X has good reduction at 3 and 5, the A8 field is unramified outside `{2,23}`
   or `{2,3,23}`.
3. **More primes.** p up to about 127. The cost is dominated by k = 4 (`q = p^4` base points,
   one quartic root count each).
4. **Resolve `Q` versus `Q^rev`.** The table gives the characteristic polynomial of Frobenius
   on `W` only up to reversal. It is decided by WP-A (the degree 15 polynomial distinguishes
   points from hyperplanes).
5. **Order of Frobenius on all of `Jac(X)[2]`.** Needed to choose the working prime in WP-A.
   When `P_p mod 2` is squarefree (p = 11, 31, 79) the order is 15.

Suggested home: a new module `rustmath-curves/src/nodal_biprojective.rs` (point counts on a
nodal curve in P^1 x P^1 given by a bihomogeneous form) and `rustmath-curves/src/weil.rs`
(L-polynomial from counts, exact Weil test, mod l factorization). MAGMA reference: Handbook
chapter 114 (`ZetaFunction`, `LPolynomial`, `NumberOfPointsOnCurve`) and chapter 42 for the
function-field versions. The method here is naive counting, which is all genus 4 needs.

Acceptance: exact equality with every row of the two tables, the three derived checks, and the
value 45, 52, 57 at 53.

## 4. WP-P: three-point class vectors of M23

### 4.1 What "Phase 6" asked for and what now exists

The research note's Phase 6: enumerate the class vectors of M23 that can carry a cover defined
over Q, use braid-orbit and real-structure data to shortlist, then compute. For three branch
points the group-theoretic part is done and is the reference for RustMath.

A three-point class vector can carry a cover over Q in two ways (branch cycle lemma):

- `rational`: three rational classes at three rational points. The rational classes of M23 are
  `2A, 3A, 4A, 5A, 6A, 8A`.
- `conjugate_pair`: a rational class `R` at a rational point and a pair `D, D^-1` of irrational
  classes at the two conjugate points of the quadratic field of `D`. The pairs are `7AB`
  (`Q(sqrt(-7))`), `11AB` (`Q(sqrt(-11))`), `14AB` (`Q(sqrt(-7))`), `15AB` (`Q(sqrt(-15))`),
  `23AB` (`Q(sqrt(-23))`). HJLPPZ is `(2A, 23A, 23B)`.

Not yet treated: a repeated rational class placed at two conjugate points of an arbitrary
quadratic field (a twist of the `rational` rows that have a repeated class by the swap of the two
points), and four or more branch points. See 4.4.

For each of the 57 class vectors with nonnegative genus, `data/m23_three_point_nielsen.json`
gives:

- `structure_constant` n: from the character table,
  `n = |C1||C2||C3| / |G|^2 * sum over chi of chi(c1) chi(c2) chi(c3) / chi(1)`;
- `size`: the number of inner classes of generating triples, which is the degree over Q of the
  Hurwitz scheme;
- `real`: the number of classes fixed by complex conjugation, which is the number of real points
  of the Hurwitz scheme;
- `genus23`: the genus of the degree 23 cover.

| type | class vector | genus | n | size | real |
|---|---|---:|---:|---:|---:|
| conjugate_pair | (2A, 7A, 7B) | 0 | 43/2 | 0 | 0 |
| conjugate_pair | (2A, 23A, 23B) | 4 | 7 | 7 | 3 |
| conjugate_pair | (2A, 14A, 14B) | 2 | 43/2 | 17 | 3 |
| conjugate_pair | (2A, 11A, 11B) | 2 | 31 | 22 | 4 |
| conjugate_pair | (2A, 15A, 15B) | 2 | 27 | 24 | 2 |
| conjugate_pair | (3A, 7A, 7B) | 2 | 349 | 100 | 8 |
| conjugate_pair | (3A, 23A, 23B) | 6 | 134 | 134 | 16 |
| conjugate_pair | (3A, 15A, 15B) | 4 | 830/3 | 264 | 20 |
| conjugate_pair | (3A, 14A, 14B) | 4 | 349 | 303 | 15 |
| conjugate_pair | (3A, 11A, 11B) | 4 | 466 | 344 | 18 |
| conjugate_pair | (4A, 23A, 23B) | 7 | 660 | 660 | 42 |
| conjugate_pair | (4A, 7A, 7B) | 3 | 1951 | 995 | 21 |
| conjugate_pair | (5A, 23A, 23B) | 8 | 1422 | 1422 | 32 |
| conjugate_pair | (4A, 15A, 15B) | 5 | 1496 | 1480 | 54 |
| conjugate_pair | (6A, 23A, 23B) | 8 | 1632 | 1632 | 72 |
| conjugate_pair | (4A, 14A, 14B) | 5 | 1951 | 1879 | 55 |
| conjugate_pair | (5A, 7A, 7B) | 4 | 3756 | 1994 | 22 |
| conjugate_pair | (8A, 23A, 23B) | 9 | 2272 | 2272 | 0 |
| conjugate_pair | (4A, 11A, 11B) | 5 | 2772 | 2390 | 58 |
| conjugate_pair | (5A, 15A, 15B) | 6 | 16982/5 | 3381 | 43 |
| conjugate_pair | (6A, 7A, 7B) | 4 | 4371 | 3470 | 40 |
| conjugate_pair | (5A, 14A, 14B) | 6 | 3756 | 3627 | 37 |
| conjugate_pair | (6A, 15A, 15B) | 6 | 11668/3 | 3859 | 103 |
| conjugate_pair | (6A, 14A, 14B) | 6 | 4371 | 4328 | 102 |
| conjugate_pair | (8A, 7A, 7B) | 5 | 5908 | 4848 | 0 |
| conjugate_pair | (5A, 11A, 11B) | 6 | 5658 | 4874 | 54 |
| conjugate_pair | (8A, 15A, 15B) | 7 | 5424 | 5424 | 0 |
| conjugate_pair | (8A, 14A, 14B) | 7 | 5908 | 5888 | 0 |
| conjugate_pair | (6A, 11A, 11B) | 6 | 7056 | 6750 | 96 |
| conjugate_pair | (8A, 11A, 11B) | 7 | 10080 | 9648 | 0 |
| rational | (2A, 8A, 8A) | 0 | 491/8 | 28 | 0 |
| rational | (3A, 5A, 5A) | 0 | 1310/3 | 30 | 0 |
| rational | (3A, 4A, 8A) | 0 | 441/2 | 54 | 0 |
| rational | (3A, 5A, 6A) | 0 | 863/3 | 70 | 4 |
| rational | (4A, 4A, 5A) | 0 | 909 | 104 | 2 |
| rational | (3A, 6A, 6A) | 0 | 1757/4 | 120 | 16 |
| rational | (3A, 5A, 8A) | 1 | 476 | 170 | 0 |
| rational | (4A, 4A, 6A) | 0 | 1713/2 | 192 | 14 |
| rational | (3A, 6A, 8A) | 1 | 634 | 350 | 0 |
| rational | (4A, 5A, 5A) | 1 | 1936 | 396 | 4 |
| rational | (3A, 8A, 8A) | 2 | 889 | 568 | 0 |
| rational | (4A, 4A, 8A) | 1 | 1170 | 568 | 0 |
| rational | (4A, 5A, 6A) | 1 | 1757 | 776 | 22 |
| rational | (5A, 5A, 5A) | 2 | 61058/15 | 972 | 0 |
| rational | (4A, 6A, 6A) | 1 | 8691/4 | 1220 | 56 |
| rational | (4A, 5A, 8A) | 2 | 2712 | 1484 | 0 |
| rational | (5A, 5A, 6A) | 2 | 3836 | 2178 | 24 |
| rational | (4A, 6A, 8A) | 2 | 6615/2 | 2534 | 0 |
| rational | (5A, 5A, 8A) | 3 | 5504 | 3288 | 0 |
| rational | (5A, 6A, 6A) | 2 | 14417/3 | 3416 | 90 |
| rational | (4A, 8A, 8A) | 3 | 10305/2 | 4208 | 0 |
| rational | (6A, 6A, 6A) | 2 | 69737/12 | 4860 | 114 |
| rational | (5A, 6A, 8A) | 3 | 7140 | 5666 | 0 |
| rational | (6A, 6A, 8A) | 3 | 8590 | 7940 | 0 |
| rational | (5A, 8A, 8A) | 4 | 10680 | 8816 | 0 |
| rational | (6A, 8A, 8A) | 4 | 13335 | 12584 | 0 |
| rational | (8A, 8A, 8A) | 5 | 18720 | 17760 | 0 |

Of the 57 class vectors, 1 has no generating triple, 22 have covers but no real point, and 34 have real points.

Consequences already visible:

- Every class vector containing `8A` has no real point, hence no rational point. Reason: the
  eight elements of M23 that invert an element of order 8 all have order 4, and a real structure
  needs an inverting involution.
- `(2A, 7A, 7B)` has no generating triple.
- `(2A, 8A, 8A)`, `(3A, 4A, 8A)`, `(3A, 5A, 5A)` have genus 0 and no real point.
- The smallest live targets are `(2A, 14A, 14B)` (17 covers of genus 2, 3 real, branch points
  `0, +-sqrt(-7)`), `(2A, 11A, 11B)` (22 covers, 4 real, `+-sqrt(-11)`), `(2A, 15A, 15B)`
  (24 covers, 2 real, `+-sqrt(-15)`), then `(3A, 5A, 6A)` (70 covers of genus 0, 4 real) and
  `(4A, 4A, 5A)` (104 covers of genus 0, 2 real).
- Odd size forces a real point. The class vectors of odd size are: `(2A,23A,23B)` (7), `(2A,14A,14B)` (17), `(3A,14A,14B)` (303), `(4A,7A,7B)` (995), `(4A,14A,14B)` (1879), `(5A,15A,15B)` (3381), `(5A,14A,14B)` (3627), `(6A,15A,15B)` (3859).

Whether any of these Hurwitz schemes other than `(2A,23A,23B)` has a rational point is not known
to us. A rational point would be a regular M23 extension of Q(t) different from HJLPPZ's. The
literature should be checked before any claim: Haefner, arXiv:2202.08222 (braid orbits for M23),
Koenig's dissertation (Hurwitz spaces for M23), Malle and Matzat, Inverse Galois Theory.

### 4.2 How the reference was computed (and what RustMath must reproduce)

Reference: `scripts/passports_all.py`, six minutes in Python with numpy.

1. **Elements.** Enumerate all 10,200,960 elements from a stabilizer chain (base of 6 points,
   orbit lengths 23, 22, 21, 20, 3, 16 for the published generators) as rows of a
   `u8` array. RustMath: do the same; 235 MB.
2. **Cycle types.** The triple `(Fix g, Fix g^2, Fix g^3)` separates the 12 cycle types.
3. **Split classes.** For a pair `D, D^-1`:
   - orders 7, 14, 15, 23: the trace of `g` on the even Golay code over `F_2` differs on `D` and
     `D^-1`. With a systematic basis `v_k` (pivot coordinates `i_k`),
     `tr(g) = sum over k of v_k[g^-1(i_k)]`.
   - order 11: the trace does not separate. Label each 11-cycle of `g` by `Z/11` through `g`
     and record the positions of the heptads on it; this family of subsets is the invariant.
     It is preserved by `g -> g^k` exactly when `k` is a square mod 11.
   - order 23 has a second, independent invariant: label the points by `Z/23` through `g`; the
     Steiner system becomes one of the two cyclic systems (generator polynomial
     `z^11+z^9+z^7+z^6+z^5+z+1` or its reciprocal).
4. **Counting.** Fix `b` in the class with the smallest centralizer. Survivors are the `a` in
   `C1` with `a b` in the right class. Then `#survivors = n |C(b)|` exactly; the reference asserts
   this for all 57 rows, which checks the class separation and the character table against each
   other.
5. **Generation.** A subgroup of M23 containing an element of order not in `{1, 11, 23}` is M23
   if and only if it is transitive on the 23 points, because every maximal subgroup other than
   `23:11` is intransitive. So `size = #(transitive survivors) / |C(b)|`.
6. **Real points.**
   - `conjugate_pair`, triple `(g1, g2, g3)` with `g2 = b`: complex conjugation sends it to
     `(g1^-1, g3^-1, g2^-1)`. Real classes correspond to involutions `c` with
     `a := c b c b^-1` in `R` and `<a, b> = M23`; divide the count by `|C(b)|`.
   - `rational`: real classes correspond to pairs `(a, b)` for which some involution inverts
     both. Enumerate the involutions `c` inverting `b` and `a = c u` with `u` an involution.
7. **Ovals** (for `(C, 23A, 23B)` only, `m23golay/nielsen.py`, `real_components_general`):
   `A = Fix(c0)`, `B = Fix(c0 g1)`; join `x` in `A` to `x^(g2^-1)` in `B`; on each `g1`-cycle
   join the points of `A` and `B` (pattern (1,1) for odd length, (2,0) or (0,2) for even length).
   The number of cycles of this graph is the number of components of `X(R)`. Check:
   `dim Jac(X)(R)[2] = g + r - 1`, computed separately from the action of conjugation on
   parabolic cocycles.

RustMath needs for this: element enumeration from a BSGS, class identification as above, and a
correct character table for the cross-check. Either fix the class-function layer (tracker T-215)
and store the ATLAS table of M23 as data, or compute `n` by the counting in step 4 and compare
with `data/m23_three_point_passports.json`. MAGMA reference: chapter 58 (permutation groups,
classes), chapter 91 (characters), chapter 68 for the cohomology in 4.3.

Acceptance: all 57 rows of `data/m23_three_point_nielsen.json`; the seven published triples,
the seven rows of Janik's table and Le Lay's seven representatives each map bijectively onto the
seven classes of `(2A,23A,23B)` (dictionary in `data/group_facts.json`, key `nielsen`).

### 4.3 The Golay data per cover

For every triple of every class vector with a degree 23 cover of genus g, the parabolic
cohomology `H^1(P^1, j_* G0)` has dimension g and sits in `H^1(P^1, j_* P0)` of dimension 2g
(reference: `m23golay/cohomology.py`). Cocycles are pairs `(c1, c2)` in the module, with
`c3 = c1 g2 g3 + c2 g3`; coboundaries are `(a(1+g1), a(1+g2))`; parabolic means
`c_i in M(1+g_i)`. This is linear algebra over `F_2` with 11 and 22 dimensional modules.
RustMath should reproduce `dim W = g` for all seven triples of `(2A,23A,23B)` and all
`(C,23A,23B)`, and the dimensions of the fixed spaces of conjugation in
`data/group_facts.json` and `data/passports_C_23A_23B.json`.

### 4.4 Next steps after the table (the actual Phase 6)

In order of cost:

1. **Twists.** For `rational` rows with a repeated class `(R, D, D)`, compute the action of the
   swap of the last two entries on the Nielsen class. The Hurwitz scheme of
   `(R; D at sqrt(d), D at -sqrt(d))` is the twist by `Q(sqrt(d))`. Report the numbers of orbits
   and of fixed classes, and the real points for `d < 0` (same formula as `conjugate_pair` with
   `D^-1 = D`).
2. **Compute the small Hurwitz schemes.** For `(2A,14A,14B)`, `(2A,11A,11B)`, `(2A,15A,15B)`
   the covers have degree 23 and genus 2 with three branch points. Compute all covers in the
   class numerically, then the etale algebra of the Hurwitz scheme (as HJLPPZ and Le Lay did for
   seven covers: an invariant of the cover, its minimal polynomial over the quadratic field,
   factor degrees). Any degree-one factor is a candidate rational point. The existing Belyi
   engine is genus 0; genus 2 needs the modular-forms route with a basis of weight 2 forms, or
   a different method. The genus 0 rows `(3A,5A,6A)`, `(4A,4A,5A)`, `(3A,6A,6A)`, `(4A,4A,6A)`
   fit the existing genus 0 engine directly and are the natural first run: 70, 104, 120 and 192
   Belyi maps of degree 23.
3. **Four branch points.** Braid orbits on Nielsen classes of length 4, genera of the reduced
   Hurwitz curves. Not started here.

Certification standard before any claim of a new M23 realization: exact Belyi identity, branch
cycles by certified continuation, monodromy group M23, field of moduli Q (the centre of M23 is
trivial, so the cover is then defined over Q).

## 5. WP-A: a polynomial for the A8 field

### 5.1 The structure to exploit (proved in the paper, Proposition 4.5)

Let `Theta` be the set of 256 theta characteristics of X, a torsor under `J[2] = Jac(X)[2]`.
For `theta` in `Theta`, `q_theta(a) = h^0(theta + a) - h^0(theta) mod 2` is a quadratic form on
`J[2]` whose polar form is the Weil pairing. Because `W` is Lagrangian:

- the theta characteristics with `q_theta | W = 0` form one coset `Theta_0` of `W`: 16 even
  theta characteristics, stable under Galois;
- each of the other 15 cosets contains 8 odd and 8 even theta characteristics, and the cosets
  correspond to the nonzero linear forms on `W`, compatibly with Galois.

The canonical model of X is the intersection of a quadric and a cubic in P^3. The quadric is
smooth (its discriminant is 4873 = 11 * 443 up to squares, Le Lay, Proposition 8.22), so the
120 odd theta characteristics are exactly the 120 tritangent planes.

**So the 120 tritangent planes fall into 15 octads, and Galois permutes the octads as A8 = GL_4(2)
permutes the 15 hyperplanes of W.** A symmetric function of each octad is a root of a degree 15
polynomial over Q with Galois group A8 whose splitting field is `K_W`. The octic follows by a
resolvent (5.5).

This was checked numerically only at the level of linear algebra (the coset statement, script
`scripts/verify_group_theory.py`). The tritangent planes themselves have not been computed.

### 5.2 Acceptance tests for the answer

Whatever route is used, the result is a monic `g8` in `Z[x]` of degree 8 (and `g15` of degree 15)
such that:

1. `g8` is irreducible with square discriminant, and its factorization mod p has the cycle type
   in the last column of the table below for every p listed. Cycle types `1 7` and `3 5` at two
   primes, together with the square discriminant, prove that the group is A8 (the only maximal
   subgroup of A8 containing an element of order 15 is `(A5 x 3):2`, of order 360).
2. `g8` has no real root (complex conjugation is a transvection, cycle type `2^4`).
3. The field discriminant of `g8` is divisible only by primes in `{2, 3, 5, 23}`.
4. `g15 mod p` has the factorization type of the matching element of `GL_4(2)` on the 15
   hyperplanes of `F_2^4`.

Frobenius on `W`: characteristic polynomial `Q_p` up to reversal, and cycle type on 8 letters.

| p | Q_p (one of the two) | order of the semisimple part | cycle type on 8 letters |
|---:|---|---:|---|
| 7 | `x^4+x^2+x+1` | 7 | `1 7` |
| 11 | `x^4+x+1` | 15 | `3 5` |
| 13 | `x^4+x^2+x+1` | 7 | `1 7` |
| 17 | `x^4+x^2+x+1` | 7 | `1 7` |
| 19 | `x^4+x^2+x+1` | 7 | `1 7` |
| 29 | `x^4+x^2+x+1` | 7 | `1 7` |
| 31 | `x^4+x+1` | 15 | `3 5` |
| 37 | `x^4+x^2+x+1` | 7 | `1 7` |
| 41 | `x^4+x^3+x+1` | 3 | `1^2 3^2 or 2 6` |
| 43 | `x^4+x^3+x+1` | 3 | `1^2 3^2 or 2 6` |
| 47 | `x^4+x^2+x+1` | 7 | `1 7` |
| 53 | `x^4+x^3+x+1` | 3 | `1^2 3^2 or 2 6` |
| 59 | `x^4+x^3+x^2+x+1` | 5 | `1^3 5` |
| 61 | `x^4+x^3+x+1` | 3 | `1^2 3^2 or 2 6` |
| 67 | `x^4+x^3+x^2+x+1` | 5 | `1^3 5` |
| 71 | `x^4+x^3+x^2+x+1` | 5 | `1^3 5` |
| 73 | `x^4+x^2+x+1` | 7 | `1 7` |
| 79 | `x^4+x+1` | 15 | `3 5` |

How the last column follows from `Q_p`: irreducible quartic of order 15 gives `3 5`;
`(x+1)(x^3+x+1)` or its reverse gives `1 7`; `x^4+x^3+x^2+x+1` gives `1^3 5`;
`(x+1)^2(x^2+x+1)` gives `1^2 3^2` or `2 6`.

A rigorous link between a candidate `g15` and `K_W` needs more than these tests; see 5.6.

### 5.3 Route A (recommended): tritangent planes, exact and p-adic

Everything here uses layers RustMath already has: `GF(p^k)`, Galois rings, Hensel lifting,
linear algebra, rational reconstruction.

**A0. Canonical model of X over Q.** Two sources; use one and test with the other.

- From `F`: the canonical differentials are `A(T,V) dT / F_V` where `A` has bidegree at most
  (2, 21) and vanishes at the 62 nodes. That is 66 unknowns and 62 conditions, so a 4 dimensional
  space. The nodes are the points `(tau, s(tau))` with `h62(tau) = 0`, and `s` is found from the
  gcd of `F(tau, V)` and `F_V(tau, V)`. Do it modulo many primes (each condition expands over
  the residue field), take the reduced row echelon basis and reconstruct by CRT. Then find the
  quadric (the unique relation among the 10 products) and the cubic (the relation among the 20
  cubic monomials modulo multiples of the quadric), again modularly.
- From Le Lay's data (CC0): `data/canonical_quadric_Q.json` and
  `data/hurwitz_canonical_models_candidate.json` (the degree-one component, over
  `Q(sqrt(-23))`), with the explicit descent to Q in his Remark 5.10. Not validated here.

Tests for A0: the quadric is unique and its discriminant is `4873` times a square; the model is
smooth; the number of points over `F_p` (enumerate `P^3(F_p)`) equals `N_1` of the table in 3.2
for several p; the two totally ramified points have canonical vanishing sequence (0,1,2,3).

**A1. Tritangent planes over a finite field.** Choose p with Frobenius of order 15 on `J[2]`
and good behaviour of the quadric: p = 79 first (the plane model is nodal there, so the
reference counts are certified), then p = 31. Do not start with p = 11: the discriminant of the
canonical quadric has odd valuation at 11 and at 443, so the quadric is singular modulo those
primes (the two rulings coincide) and the P^1 x P^1 description fails there.

A plane `H` is tritangent when `H . X = 2(P1 + P2 + P3)`. One formulation: over a field where the
quadric has rulings, X is a curve of bidegree (3,3) on P^1 x P^1 with equation `G(s,u)`; a
nondegenerate plane section is the graph of a Moebius map `u = mu(s)`; `H` is tritangent exactly
when the binary sextic `(gamma s + delta)^3 G(s, mu(s))` is a constant times the square of a
cubic. With the four coefficients of `mu` and the four coefficients of the cubic as unknowns this
is 7 equations, zero dimensional after removing degenerate `mu`, with 240 solutions (each plane
twice). Solve by a Groebner basis over `F_p` (the field engine is the sound one) or by any other
exact method, then factor the eliminant with Cantor-Zassenhaus.

Predictions that test A1 before anything else is built:

- there are exactly 120 tritangent planes over the algebraic closure;
- at p = 31 and p = 79 they form 8 Frobenius orbits of length 15, so the eliminant has 8
  irreducible factors of degree 15;
- there is exactly one theta characteristic defined over `F_p`, and it is even.

(Reason: Frobenius acts on `J[2]` semisimply with characteristic polynomial
`(x^4+x+1)(x^4+x^3+1)`, without eigenvalue 1, so it has one fixed point on `Theta` and 17 orbits
of length 15 on the rest; 120 = 8 * 15 and 135 = 9 * 15.)

**A2. The octads.** Let `phi` be Frobenius, `O_1, ..., O_8` the eight orbits. `J[2]` has exactly
two `phi`-stable subspaces of dimension 4, `U = ker(phi^4 + phi + 1)` and
`U' = ker(phi^4 + phi^3 + 1)`, and `W` is one of them. A coset of `W` that is not `Theta_0`
contains one plane from each orbit, and `phi` permutes the 15 cosets cyclically. So a block
system is determined by seven phases: fix `x` in `O_1`; for each `i >= 2` there is exactly one
`y` in `O_i` with `x - y` in `U`, and exactly one with `x - y` in `U'`.

Test for `x - y` in `U`: `(1 + phi + phi^4)(x - y) = 0`, which says that
`x + phi(x) + phi^4(x)` and `y + phi(y) + phi^4(y)` are linearly equivalent divisor classes of
degree 9, which says that the function
`l_x l_phi(x) l_phi^4(x) / (l_y l_phi(y) l_phi^4(y))` is a square in the function field of X,
where `l_z` is the linear form of the plane `z`. For `U'` use `phi^3` in place of `phi`.

Deciding squareness without divisor arithmetic: a function `g` defined over `F_q` that is a
square over the algebraic closure is `c h^2` with `c` constant, so `g(P)` lies in one square class
for all `P` in `X(F_q)`. For each of the 15 candidates `y` in `O_i`, evaluate at rational points;
two values in different square classes refute the candidate. Exactly one candidate survives for
`U` and one for `U'`, and it is then proved by elimination. Work over `F_q` with `q = p^15`,
where all planes are defined, and use the points of `X(F_p)` (33 at p = 31, 77 at p = 79) or of
a small extension.

**A3. Lift.** Hensel-lift the 120 planes to the Galois ring `GR(p^N, 15)` by Newton iteration on
the tritangency system (the Jacobian is invertible at simple solutions). The block systems are
determined mod p.

**A4. The degree 15 polynomial.** Fix two rational points `P0, P1` of P^3. For an octad `B` put
`tau_B = sum over planes H in B of l_H(P1) / l_H(P0)` and
`g15(z) = product over the 15 octads of (z - tau_B)`. Its coefficients are fixed by Frobenius, so
they lie in `Z/p^N`.

Which of `U`, `U'` is `W`: exactly one of the two block systems is stable under all of Galois
(or both, if `J[2]` splits as a Galois module, which is not known). For the right one the
coefficients are rational numbers; for the wrong one they are not. This is the same dichotomy,
with the same kind of proof, as the Steiner labelling of roots (paper, Proposition 5.1):
reconstruct rationals from `Z/p^N`, increase `N` until stable, and confirm at a second prime
(79 if the first was 31), where the computation is independent. Then reduce `g15` with polred.

**A5. The octic.** The roots of `g15` are labelled by the nonzero linear forms on `W`, and the
`F_2`-structure is visible in the cyclic model: `lambda_j = lambda_0 phi^-j`, so
`lambda_i + lambda_j = lambda_k` exactly when `x^i + x^j + x^k = 0` in `F_2[x]/(Q^rev)`, where
`Q` is the characteristic polynomial of `phi` on `W`. In `PG(3,2)` there are 35 lines and 56
spreads (sets of 5 disjoint lines). Call two spreads adjacent when they share exactly one line.
This graph is the Johnson graph J(8,3): the spreads are the 3-subsets of a set of 8 letters, the
28 maximal cliques of size 6 are the pairs of letters, and a letter is a set of 7 such cliques
that pairwise share a spread. The union of those 7 cliques is a family of 21 spreads. This is
done and verified in `scripts/pg32_a8_model.py`; the lines, spreads and the 8 families for both
primitive quartics are in `data/pg32_a8_model.json`, with the check that `GL_4(2)` acts on the
8 letters with order 20160, the Singer cycle with type `3 5` and a transvection with type `2^4`.
Set
`theta_m = sum over spreads S in family m of product over lines L in S of (sum of tau over the points of L)`,
`g8(z) = product of (z - theta_m)`. If two `theta_m` collide, change `P0, P1`.

**A6. Certificates.** See 5.6.

### 5.4 Route B: the fifteen covers of degree 46

By the paper (Corollary 4.4), the nonzero elements of `W` are the fifteen lifts of the HJLPPZ
triple to `2^11:M23` with inertia in the classes of `(g_i, 0)`. As permutations of 46 points they
are in `data/lifts_2_11_M23.json` (cycle types `1^14 2^16`, `23^2`, `23^2`, group order
`2^11 * 10200960`, genus 7; verified). Each is the unramified double cover `Z_w -> X`.

Computing these fifteen three-point covers numerically and recognizing a common invariant gives
`g15` directly (the action on points of `P(W)`; route A gives the action on hyperplanes; the two
degree 15 fields are arithmetically equivalent and not isomorphic, and both have Galois closure
`K_W`). The triangle group is `Delta(2,23,23)`, the subgroup has index 46, and the space of
weight 2 forms has dimension 7: 4 from X and 3 Prym differentials with the quadratic character.
This needs the modular-forms route beyond genus 0, so it is a stress test of the Belyi engine and
an independent confirmation, not the first choice.

### 5.5 Route C: periods and theta functions

Compute a period matrix of X to several hundred digits, identify `W` inside `H_1(X, F_2)` from
the monodromy (the cocycle description in 4.3 evaluates on lifted loops), and get the tritangent
planes from gradients of theta functions with odd characteristics. RustMath has no period
matrices and no theta functions (in the handbook digests they appear only in chapter 125, for
hyperelliptic curves), so
this is the most expensive route. It is listed for completeness.

### 5.6 What would make the result a theorem

Tests 1 to 4 of 5.2 show that `g8` defines an A8 field with the right local behaviour at 17
primes. That is strong evidence, not a proof that the field is `K_W`. For a proof:

- A0 must be exact (it is: linear algebra over Q, verified by substitution in the function field
  `Q(T)[V]/(F)`);
- the tritangency system must be shown to have exactly the 120 lifted solutions (degree count
  plus simplicity mod p);
- the statement "`g15` vanishes on `tau`" must hold exactly, not to precision `p^N`. Use the
  argument of the paper's Proposition 5.1: `g15(tau_B)` is an algebraic number of bounded degree
  with archimedean absolute values bounded by certified complex approximations of the 120 planes;
  if it is nonzero its norm has bounded p-adic valuation, so vanishing to a large explicit
  precision proves vanishing. This needs certified complex bounds for the solutions, hence real
  interval or ball arithmetic beyond f64. Until that exists, report the result as
  `Unresolved (verified at two primes and 17 Frobenius classes)`.

### 5.7 Risks and unknowns

- The pipeline in 5.3 is designed, not executed. Its internal predictions (120 planes, 8 orbits
  of length 15, exactly one surviving phase per orbit, rational coefficients for exactly one of
  the two block systems) are there so that a wrong step fails loudly.
- The Groebner computation in A1 is the main cost. If it is too slow over `F_p`, first project:
  eliminate to a single coordinate of the dual space and factor a degree 240 polynomial.
- If `J[2]` is a split extension, both block systems are rational. That would be a new fact about
  X (a second rational Lagrangian) and both give the same field `K_W`.
- The squareness test assumes the planes are in general position with respect to the evaluation
  points; skip points where a linear form vanishes.

## 6. Shared infrastructure, by crate

| Need | Used by | Suggested home | MAGMA chapter |
|---|---|---|---|
| Point counts on a nodal curve in P^1 x P^1, L-polynomial, exact Weil test | WP-F | `rustmath-curves` (new modules, independent of the facade layer) | 114, 42 |
| Element enumeration from a BSGS; class identification for M23 including split classes | WP-P | `rustmath-groups` | 58 |
| Class structure constants; a correct character table of M23 as data | WP-P | `rustmath-groups` (fix T-215) | 91 |
| Nielsen classes of length 3, complex conjugation, ovals | WP-P | `rustmath-curves/src/belyi/` or a new `rustmath-groups/src/nielsen.rs` | 58, 131 |
| Binary Golay code, S(4,7,23), modules `G0`, `P0`, `P0/G0`, parabolic cohomology over `F_2` | WP-P, WP-A | `rustmath-coding` and `rustmath-groups` | 152, 147, 68 |
| Adjoint computation: canonical model of a nodal curve in P^1 x P^1 | WP-A | `rustmath-curves` | 114 |
| Zero-dimensional solving over `F_p`, Newton lifting in `GR(p^N, k)` | WP-A | `rustmath-polynomials`, `rustmath-finitefields` | 105, 47, 48 |
| Rational reconstruction with stability check, polred | WP-A | `rustmath-numberfields` (exists) | 34 |
| Certified complex enclosures (ball arithmetic) | WP-A certificate | `rustmath-reals` | 25 |

The Steiner labelling of roots (paper, Section 5; `m23golay/arith.py`) is a good first exercise
for the Galois-ring layer, with exact expected output: at p = 29 for `f2`, `T5 = -1059840`,
`J5 = 61916`, `J7 = -83030`; for `f1`, `T5 = 92747040`, `J5 = 524492`, `J7 = -2281968`; the
rejected candidate for `T5` has more than 40 digits. The degree 253 resolvents in
`data/golay_resolvents.json` are the larger version of the same test.

## 7. Facts found while checking the sources

These came out of reading Le Lay's paper and Zarhin's two papers against our draft. They matter
for anyone extending the work.

- Le Lay's preprint is now titled "Arithmetic and geometry of an M23 Hurwitz scheme". HJLPPZ v2
  cites it under an earlier title. The bidegree (4,23) equation of HJLPPZ v2 is the minimal
  equation constructed there.
- The lemma that a 23-cycle preserves exactly two Steiner systems, with the same generator
  polynomial, is in Le Lay, Section 5.4. He uses the heptad invariant `J7` to choose between
  them for wild covers in characteristic 23. Our p-adic labelling is the number-field analogue.
- The heptad of unramified points above `t = 0`, its Fano plane, the affine structure on the
  eight ramified points and the real points of that fibre are in Le Lay, Section 8. Our paper
  now attributes them.
- Le Lay's class numbers 1..7 map to ours as `[0, 1, 2, 4, 6, 5, 3]`; his real classes 3, 6, 7
  are our 2, 5, 3 and his distinguished class 6 is our rational class 5. Three independent
  computations of the real classes agree.
- Zarhin's theorems cover `S_n`, `A_n` (2000) and `M11`, `M12`, `L_m(q)` (2002). They need the
  module of even subsets of the roots to be very simple. For M23 in degree 23 that module
  contains the even Golay code, so the method does not apply; the papers do not discuss M23.
- 11 divides the discriminant class 4873 of the canonical quadric, so the canonical quadric of
  the reduction of X at 11 is singular and X mod 11 should have a vanishing even theta-null.
  This follows from the square class and good reduction; it was not checked by computation.
  It is why route A avoids p = 11.

## 8. Order of work and definition of done

1. WP-F core (3.1, 3.2): done when both tables and the three derived checks are reproduced in
   Rust tests that read `data/zeta_X.json` and `data/zeta_L.json`.
2. Steiner labelling exercise (end of Section 6): done when the six integers and one degree 253
   resolvent are reproduced.
3. WP-P table (4.2, 4.3): done when all 57 rows are reproduced and the three dictionaries hold.
4. WP-F extensions 1 and 2 (the D flags, p = 3 and 5).
5. WP-A A0 and A1 with their predictions. Stop and report if a prediction fails.
6. WP-A A2 to A5, then the tests of 5.2 against the table.
7. WP-P 4.4, item 2: the genus 0 class vectors with the existing Belyi engine.

Report each result with its status: reproduced from reference, new and verified at two primes,
or new and unverified. Do not report the A8 polynomial as a theorem until 5.6 is done.
