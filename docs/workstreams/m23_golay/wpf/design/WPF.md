# WP-F design: point counts and Frobenius polynomials of the M23 curve

Date: 2026-11-15. Worktree `/home/user/RustMath-wpf`, branch `claude/m23-wpf`, base `origin/main`
`0d277f0`. Brief: `docs/workstreams/m23_golay/HANDOFF.md`, Sections 1, 2, 3, 6, 7, 8 (Section 2 as
refreshed at RustMath `6e03987`). Prototype: `scratchpad/m23/design/WPF/proto` (Rust, depends on the
worktree crates by path, own target directory), with PARI/GP scripts in `scratchpad/m23/design/WPF/work`
and `scratchpad/m23/design/WPF/gp`.

## 0. Summary

What the prototype does and finds (all exact, all reproduced in Rust on the worktree crates):

1. **Both reference tables reproduce.** All 18 rows of `zeta_X.json` (p = 7..79, k = 1..4), the
   refusal at p = 5 ("not a Weil polynomial"), all 7 rows of `zeta_L.json` (5 and 11 refused as
   degenerate), and the anchor 45, 52, 57 at 53 come out equal, value for value, in the
   reference-compatible mode (non-nodal singular points counted once).
2. **Node certificate.** A new exact criterion decides good reduction: every singular point of the
   reduction lies over a root of `D(V) = disc_T F` mod p; each such fibre is analysed in its residue
   field. If all singular points are nodes and there are exactly `p_a - g = 66 - 4 = 62` of them, X has
   good reduction at p and the counts are those of X mod p (Theorem 4.5). X is certified nodal at
   every prime 7 <= p <= 131 except 23 (not run), 5, 13, 31, 41, 73 and 107 (one non-nodal point
   each, item 3 and 4); p = 3 is refused (degenerate model). At 7 and 131 this needs the right
   model (Section 3). The covers over L are certified nodal at all their degree-one primes up to
   127 except 5 and 11 (refused) and 31 (s0 = 12), 79 (s0 = 15), 103 (s0 = 93) (non-nodal points).
3. **The D flags go away.** The non-nodal point at 13, 31, 41, 73 is an ordinary cusp (A2,
   delta 1) on the fibre `T = 0`; with 61 nodes the delta-sum is 62, so the same theorem proves good
   reduction, and counting the cusp once is exact. The reference values at these primes are right
   and become certified (status per decision D1, Section 13).
4. **p = 5 is decided: X has good reduction at 5.** The model has 60 nodes and a tacnode (A3, delta
   2) with conjugate branches. The reference counted the tacnode once and got a non-Weil polynomial.
   Counting its branches correctly gives `N = 7, 37, 169, 561`,
   `P_5(T) = T^8 + T^7 + 6T^6 + 20T^5 + 16T^4 + 100T^3 + 150T^2 + 125T + 625`, which is Weil (exact
   test), satisfies `P(chi) = P(chi p) = 0 mod 23` and splits mod 2 as `Q Q^rev`.
5. **The reference row of `zeta_L.json` at p = 31, s0 = 12 is wrong.** That model has 57 nodes, one
   ordinary triple point (one rational and two conjugate tangents) and one tacnode with conjugate
   branches. Counting both once (the reference rule) gives the published `44, 1014, 30266, 923726`,
   which passes the numerical Weil test but fails the mod 23 check (`P(1) = 15`, `P(31) = 7`
   mod 23). The exact branch counts give `43, 1017, 30265, 923729`, `a = (11, 88, 682, 4438)`,
   which pass all checks; `Q` is `x^4+x^3+x^2+1` or its reverse, not `x^4+x^2+1`.
6. **New primes.** X at 83..127 and the L covers at their degree-one primes up to 127: Section 7.
   p = 3 is not reachable with these models (F mod 3 = `(V-1)^23 T (T-1)^3`); it goes to the
   follow-up list with the reason.
7. **The equation question is settled.** Le Lay's CC0 file `data/optimal_23_4_Z.json` (sha256
   `cfb4185b...`, matching his SOURCE_NOTES) is HJLPPZ's F in another coordinate:
   `F(T, (967W - 3)/(81 - 2267W)) (81 - 2267W)^23 = c G(T, W)` with `c > 0` (checked exactly in
   PARI/GP). Vendor G. It reduces like F except at the primes dividing the determinant
   `71526 = 2 3 7 13 131`. Two further integral models of the same curve, `G(T, 7V + 3)` and
   `G(T, (11V + 130)/(V + 13))` (made primitive), cover 7 and 13. F and calF are read at test time
   from `$M23ISGALOIS` only; without it, the L table, the anchor at 53 and the F/G identity test do
   not run (they are `#[ignore]` and panic with a clear message if run without the variable).
8. **Cost.** A word-size `GF(p^k)` kernel is needed: the existing `FiniteField`/`FqPoly` stack
   costs 4.0 ms per fibre against 5.0 us for the kernel (one thread, p = 31, k = 4). Measured
   release times on 4 threads (shared machine): X for p <= 79 and k <= 4 in 60 s; X for p <= 127
   in 8.4 min; the L covers for p <= 127 in 8.3 min. The debug test suite proposed in Section 9
   runs in about 2 s (prototype equivalents, one thread; test harness and compile time not included).

## 1. Sources and conventions

Mathematics:

- **[HJLPPZ]** X. Huang, B. Jackson, K.-H. Lee, B. Poonen, R. Pries, S. Zhang, *The Mathieu group
  M23 is a Galois group over Q*, arXiv:2608.08538v2. The curve: F of bidegree (4,23), 62 nodes,
  genus 4 (Prop. 3.10); the covers over the sextic field L (Section 3.7); the counts 45 and 52 at
  53 (Prop. 3.14).
- **[LeLay]** Le Lay, *Arithmetic and geometry of an M23 Hurwitz scheme* (repository
  `research-reflexivity/frontier-math-m23-cover-investigation`, data CC0-1.0). Only `data/` is used.
- **[Weil]** A. Weil, *Sur les courbes algebriques et les varietes qui s'en deduisent* (1948): the
  zeta function of a smooth projective curve of genus g over F_q is `L(T)/((1-T)(1-qT))` with
  `L` of degree 2g, functional equation, `|alpha| = sqrt q`.
- **Newton's identities** for the power sums `s_k = sum alpha_i^k = q^k + 1 - N_k` (any algebra
  text, e.g. Macdonald, *Symmetric functions*, I.2).
- **[Sturm]** C. Sturm (1835); Basu, Pollack, Roy, *Algorithms in real algebraic geometry*,
  Thm. 2.50. Already in `rustmath-polynomials/src/real_roots.rs`.
- **[Ful69]** W. Fulton, *Algebraic curves* (delta invariant, blow-ups, Noether's formula
  `delta = sum m(m-1)/2` over infinitely near points). **[Har77]** R. Hartshorne, *Algebraic
  geometry*, III.9 (flatness, Euler characteristic), III.11 (Zariski's connectedness), and
  Liu, *Algebraic geometry and arithmetic curves*, 7.5 and 10.1 (normal models, `p_a = p_a~ + delta`).
- **Weierstrass preparation and the A_k normal form** in characteristic not 2: Greuel, Lossen,
  Shustin, *Introduction to singularities and deformations*, I.2 (splitting lemma); the argument in
  Section 4.4 only uses preparation and completing the square, valid over any field of
  characteristic not 2.

MAGMA reference (the interface spec): Handbook ch. 114 (`rustmath-lava/docs/Chapter_114_Algebraic_Curves.md`):
`IsNode`, `IsCusp`, `IsOrdinarySingularity`, `Multiplicity`, `TangentCone`,
`IsAnalyticallyIrreducible` (114.4.3), `LPolynomial`, `ZetaFunction`,
`NumberOfPlacesOfDegreeOneECF` (114.8.1); ch. 42 (`Chapter_42_Algebraic_Function_Fields.md`): the
same `LPolynomial`/`ZetaFunction` for function fields. MAGMA computes these for arbitrary curves by
other methods; we implement only what genus 4 needs (naive fibre counting), with the MAGMA
conventions: `LPolynomial` is `L(T) = T^{2g} P(1/T)`, ascending.

Conventions in this document: `F[i][j]` is the coefficient of `T^i V^j`, `0 <= i <= 4`,
`0 <= j <= 23`. The fibre variable is T (degree 4), the base is V (degree 23), as in `curve.py`
with `fibre="V"`. `P_p(T)` is the characteristic polynomial of Frobenius, descending, monic, as in
HANDOFF 3.1. `p_a = (4-1)(23-1) = 66`, `g = 4`, `delta_0 = p_a - g = 62`.

## 2. What is reused, what is new

Reused (Section 2 of the brief, checked in the code at `0d277f0`):

| Need | Existing entry point |
|---|---|
| F_p polynomial arithmetic, gcd, factorization with multiplicity, mod-p inverse | `rustmath-polynomials/src/fp_factor.rs` (`mul`, `div_mod`, `gcd`, `extended_gcd`, `factor_with_multiplicity`, `mod_inv`) |
| Square-free decomposition over Z | `rustmath-polynomials/src/zx.rs` `squarefree_decomposition` |
| Sturm counts over Q | `rustmath-polynomials/src/real_roots.rs` `count_real_roots`, `count_real_roots_in` (half-open `(a, b]`) |
| Big integers, decimal parsing, Kronecker symbol | `rustmath-integers` `Integer::from_decimal_str`, `kronecker_symbol` |
| Rationals | `rustmath-rationals::Rational` |
| GF(p^n) cross-checks in tests, factoring in rare residue fields | `rustmath-finitefields` `FiniteField`, `FqPoly`, `poly_factor::factor` |
| Parallel loops | `rayon` (already a dependency of `rustmath-curves`) |
| JSON test data | `serde_json` as a dev-dependency (already in the lock file, used by `rustmath-groups`) |

Not used (facade or wrong, per the brief): the plane-curve layer of `rustmath-curves`
(`genus.rs:30`, `singularities.rs:104`, `riemann_roch.rs`, `differentials.rs`), the function-field
layer of `rustmath-rings` (its `count_points` returns `q^n + 1`), `rustmath-functionfields/src/genus.rs`.
The new modules do not import any of them.

New, because nothing exists (brief, "What to build"): a word-size `GF(p^k)` kernel with a
distinct-root count, the plane point count with Frobenius orbits, the singularity certificate, the
local classifier, the L-polynomial from counts, the exact Weil test, the mod 2 and mod 23 checks,
readers for the data files.

Why a kernel and where it lives. `FiniteFieldElement` stores coefficients as heap `Integer`s and
inverts by Fermat. Measured with the prototype (`wpf_proto oldff 31 4 200`, release, one thread): a
quartic fibre over `GF(31^4)` with `FqPoly::pow_mod` and `gcd` costs 4.0 ms, against 5.0 us
with the kernel. At p = 79, k = 4 there are 9.7 million fibres. The kernel goes into
`rustmath-finitefields` (new module `small_field.rs`), as the rules require, with tests against
`FiniteField`.

## 3. The equation question (decided)

Facts checked in PARI/GP (`work/mob.gp`, `work/hj.gp`, `work/n13.gp`):

- `G(T, W)`, the coefficient array `coefficients_T_then_W` of Le Lay's `data/optimal_23_4_Z.json`
  (CC0-1.0, sha256 `cfb4185b2800744d524d87b3f08f5459cd38d42afb3787c11fa1e2aa3f25ee84`, the value in
  his `SOURCE_NOTES.md`), satisfies exactly
  `F(T, (967W - 3)/(81 - 2267W)) (81 - 2267W)^23 = c G(T, W)` with `c` the (positive) content of
  the left side. The map `V = (967W - 3)/(81 - 2267W)` is an automorphism of P^1 over Q and `T` is
  unchanged, so G and F define the same curve in P^1 x P^1 (an isomorphism of plane models, not only
  a birational map), with the same function t. The Moebius map was found from the totally ramified
  fibres over `T = +-sqrt(-23)` and one root of the degree 7 factor of `F(0, V)`.
- Its determinant is `967*81 - 3*2267 = 71526 = 2 * 3 * 7 * 13 * 131`. For p not dividing it the
  map is invertible mod p, so `G mod p` and `F mod p` are isomorphic models: same counts, same
  singularities. At p = 7 `disc_T G` vanishes identically mod 7, at p = 13 `G mod 13` has a fibre
  component; at p = 131 it is the other way round (F has a fibre component, G is nodal with 62
  nodes).
- Two more integral models of the same curve, found by a 4 x p search over Moebius maps of
  determinant p:
  - `G7(T, V) = G(T, 7V + 3)` made primitive: nodal with 62 nodes at p = 7;
  - `G13(T, V) = G(T, (11V + 130)/(V + 13)) (V + 13)^23` made primitive: at 13 it has 61 nodes and
    the same A2 cusp as F (the cusp sits on `T = 0` and is a property of the map t, so no Moebius
    map in V removes it).

Decision: vendor G (and nothing from m23isgalois) into `rustmath-curves/tests/data/m23/`, with a
README that gives the source, licence, checksum and the Moebius identity. The models G7 and G13 are
computed at run time from G by `BihomogeneousForm::mobius_y` (Y = V). The M23 driver tries, at each p, the
fixed list `[G, G7, G13]` and uses the first model whose certificate is Decided; if none is, it
reports the best status (Section 4.5). F and calF are read only from `$M23ISGALOIS`:

| Without `M23ISGALOIS` | With it (`#[ignore]` tests, run with `--ignored`) |
|---|---|
| all X rows from G, G7, G13 (fast rows in debug, the full table opt-in in release); certificate, Weil, mod 2, mod 23 checks; small curves against PARI/GP | the exact identity F o M = c G; X rows recomputed from F (same values); the 7 rows of the L table; the anchor 45, 52, 57 at 53; the corrected L row at 31 |

An `#[ignore]` test that needs the variable panics with "set M23ISGALOIS to a checkout of
shaowuz/m23isgalois" when it is missing, so `--ignored` never passes vacuously.

The L covers have no CC0 plane model. Le Lay's data has canonical models of the covers, but
counting on a curve in P^3 is a different algorithm (WP-A, step A0). Follow-up F-6.

## 4. Algorithm

### 4.1 Plane point count by fibres (reference: `curve.py`, `count_points`, `fibre="V"`)

Let `X0_p = V(F mod p)` in P^1 x P^1 over F_p and `q = p^k`. Then

`#X0_p(F_q) = sum over v in P^1(F_q) of #{distinct roots of F(T, v) in P^1(F_q)}`.

Here `F(T, v)` is the binary quartic `sum_i c_i(v) T^i` with `c_i(V) = sum_j F[i][j] V^j`; a drop
of degree counts as one root at `T = infinity`; `v = infinity` uses the leading coefficients
`F[i][23]`. A fibre that vanishes identically is a vertical component: refused
(`DegenerateModel::FibreComponent`).

Frobenius orbits (as the reference): code elements of `GF(q)` by their base-p digits; process `v`
only if its code is the smallest in `{v, v^p, ..., v^(p^(k-1))}` and weight it by the orbit length.
The fibre counts of conjugate base points are equal because F has coefficients in F_p. The loop is
split into chunks of `2^14` codes for `rayon`; each chunk is stateless.

Evaluation of `c_i(v)`: compute `v^j`, `j <= 23` (22 products), then each `c_i(v)` as an F_p-linear
combination with lazy accumulation in `u64` (one reduction per coefficient). This is 3 times
cheaper than Horner in `GF(q)`.

### 4.2 Distinct roots of a quartic over GF(q) (new kernel)

`#{roots in GF(q)} = deg gcd(f, x^q - x)` for the monic affine part f of degree `d <= 4`.

- `h1 = x^p mod f` by square-and-multiply (`log2 p` squarings, multiplications by x are shifts).
- `x^q mod f` from `h1` by the semilinear p-power map `g^p = sum_j sigma(g_j) h1^j mod f`
  (`sigma` = Frobenius on coefficients, a precomputed k x k matrix over F_p; `h1^2, h1^3` are two
  products). Applying it `k - 1` times gives `x^q`. This replaced `k - 1` modular compositions and
  saved 30 percent (measured: 7.3 -> 5.0 us per fibre).
- `gcd(f, x^q - x)` by Euclid in `GF(q)[x]`; inverses by the norm: `a^-1 = (a^p ... a^(p^(k-1))) /
  N(a)` with `N(a)` in F_p.
- Arithmetic: elements are `[u64; K]` (const generic K <= 8), p < 2^16, Barrett reduction mod p,
  products of polynomials over `GF(q)` accumulated as `u64` bivariate convolutions and reduced once
  per coefficient. All bounds stay below `2^40`.

A `Vec` path of the same algorithm handles any fibre degree (used for the T-fibration with degree 23
fibres, Section 4.9).

### 4.3 Singular points: discriminant and residue fields

Let `D(V)` be the formal discriminant of `F(T, V)` as a binary quartic in T: the degree 6 polynomial
in `c_0..c_4` (16 terms), computed in `F_p[V]` with `fp_factor`. It is a binary form of degree
`6 * 23 = 138` in V; `V = infinity` is a root of multiplicity `138 - deg D`.

Lemma 4.3. Every singular point of `X0_p` over `F_p-bar` lies over a root of D in `P^1(F_p-bar)`.
Proof: at a singular point `(t0, v0)` both `F` and `F_T` vanish, so the binary quartic `F(T, v0)` has
a multiple root at `t0` (at `T = infinity` in the other chart), and a binary form has a multiple
root exactly when its formal discriminant vanishes. If `D = 0` identically every fibre has a multiple
root; for `p > 4` this means `X0_p` is not reduced, and the model is refused
(`DegenerateModel::DiscriminantVanishes`). QED.

Procedure: factor D over F_p (`fp_factor::factor_with_multiplicity`). For every irreducible factor
`g` (all of them, whatever the multiplicity, so no tameness assumption is needed) and for
`V = infinity` when `deg D < 138`:

1. `K = F_p[V]/(g)`, `v0 = V mod g`. Reduce `c_i`, `dc_i/dV`, `d^2 c_i/dV^2` mod g (no evaluation is
   needed: the residue of `c_i` is `c_i(v0)`).
2. `G = gcd(f, f_T, F_V(T, v0))` in `K[T]`. Its roots are the singular points with T affine.
   `deg G <= 2` (each singular point is a multiple root of a quartic).
   - `deg G = 1`: one point `tau` in K.
   - `deg G = 2` with nonzero discriminant: two points. If the discriminant is a nonsquare in K they
     form one closed point of degree `2 deg g` with residue field `K[T]/(G)`; otherwise two closed
     points of degree `deg g`. This case occurs for X (two nodes in one V-fibre at 13 and others).
   - `deg G = 2` with zero discriminant: one point `tau = -G_1/2`.
3. `T = infinity`: if `c_4(v0) = c_3(v0) = c_4'(v0) = 0`, the point `(infinity, v0)` is singular;
   use the chart `U = 1/T` (reverse the T-coefficients).
4. Node test (MAGMA `IsNode`): the Hessian discriminant `H = F_TV^2 - F_TT F_VV` at the point. A
   singular point with `H != 0` is an ordinary double point (characteristic not 2). Its branches
   are rational over the residue field `kappa` exactly when `H` is a square in `kappa`. Square class
   in `K` by the norm: `N_{K/F_p}(h) = Res(g, h)` (g monic), then the Legendre symbol mod p. For the
   two-point case without splitting G: with `w = h^((|K|-1)/2)` in `K[T]/(G)`, the trace
   `2 w_0 - G_1 w_1` equals `chi_1 + chi_2`; for the irreducible case the norm
   `h_0^2 - G_1 h_0 h_1 + G_0 h_1^2` from `K[T]/(G)` to K followed by `Res(g, .)`.
5. `H = 0`: non-nodal; classify (Section 4.4). When this happens in a split two-point fibre, find
   the two roots of G in K first (Tonelli-Shanks with `ResidueField::pow`; a nonsquare is found by
   scanning, which only finds a helper and decides nothing), then classify each point. In an
   irreducible two-point fibre the point lives in `K[T]/(G)`: build that field as a
   `ResidueField` over F_p from the minimal polynomial of `tau` (the resultant
   `Res_V(g(V), G(T, V))` factored over F_p) and classify there. Neither case occurs in the data;
   both are covered by constructed tests (follow-up F-3 lists the second).
6. The local data of every point is kept with its residue degree `d`: the delta-sum adds
   `d * delta`, and the counting corrections of 4.6 use `d`.

Residue-field arithmetic uses `fp_factor` polynomials reduced mod g (`i64`, fast). This is a small
private helper type `ResidueField` inside the certificate module, not a new numeric wrapper: it has
`mul`, `inv` (via `fp_factor::extended_gcd`), `norm` (resultant over F_p, Euclid) and `pow` with an
`Integer` exponent.

Measured: the certificate takes 2 to 14 ms per prime in release (all primes up to 127).

Consistency with characteristic 0 (PARI/GP, `work/disc.gp`): over Z,
`disc_T F = 2^152 3^14 23^2 r_14(V) s_62(V)^2` with `r_14` (the 14 ramification points of the
degree 4 map v; Riemann-Hurwitz: `2*4 - 2 = 4*(-2) + 14`) and `s_62` (the V-coordinates of the 62
nodes) squarefree and coprime. So at a good prime D mod p has a squarefree part of degree 14 and a
square part of degree 62, as the certificate finds. The certificate does not use this
factorization (it analyses every factor of D mod p), so it needs no data derived from F.
`disc_V F = 2^1652 3^626 23^228 131^506 T^8 (T^2+23)^22 h_62(T)^2` (Section 1 of the brief) is used
only in one cross-check test (Section 9.1).

### 4.4 Local classification of a non-nodal singular point

At a singular point with coordinates in its residue field `kappa` (chart chosen so that it is
affine), shift to the origin: `G(x, y) = F(x + tau, y + v0)`.

- **Hessian rank 1 (A_k, k >= 2; MAGMA `IsCusp` for k = 2).** Let `a = G_xx, b = G_xy, c = G_yy` at
  0 with `b^2 = a c`. If `a != 0` put `x = u - (b/a) y`, else swap x and y. Then `H(u, y)` has
  `H_uu(0) = a != 0`. Solve `H_u(phi(y), y) = 0` for `phi` in `kappa[[y]]` by Newton iteration and
  put `g(y) = H(phi(y), y)`. Then `ord g = k + 1` and the type is `A_k`. Precision starts at
  `y^16` and doubles while `g = 0` up to `y^128`; beyond that the point is `Unclassified`
  (`delta > 62` there anyway).
  Proof sketch: by Weierstrass preparation `H = W (u^2 + beta_1 u + beta_0)` with W a unit, and
  `u^2 + beta_1 u + beta_0 = (u + beta_1/2)^2 - Delta(y)` (characteristic not 2). On the polar
  curve `H_u = 0` one has `u + beta_1/2 = O(Delta)`, so `g = -W(0) Delta (1 + O(Delta))`:
  `ord Delta = ord g = k + 1` and `lead(Delta) = -2 lead(g)/a`. The branches are
  `u + beta_1/2 = +-sqrt(Delta)`.
  - `delta = floor((k + 1)/2)` (blow up `floor((k+1)/2)` times; [Ful69], Noether's formula).
  - k even: one branch (MAGMA `IsAnalyticallyIrreducible`), so the point contributes exactly one
    point of the normalization over any field containing `kappa`.
  - k odd: two branches, defined over `kappa` iff `-2 a lead(g)` is a square in `kappa`.
- **Hessian rank 0, ordinary m-fold point (MAGMA `IsOrdinarySingularity`).** Let `T_m` be the
  lowest nonzero homogeneous part of G. If `T_m` is squarefree as a binary form, the point is an
  ordinary m-fold point: one blow-up separates m smooth branches with distinct tangents, so
  `delta = m(m-1)/2` and the branches correspond to the roots of `T_m` in `P^1`. Record the degrees
  of the irreducible factors of `T_m` over `kappa` (for `kappa = F_p` with `fp_factor`, otherwise
  with `poly_factor::factor` over `FiniteField::with_modulus`).
- Anything else (a non-ordinary point of multiplicity >= 3): `Unclassified`.

Every type carries its **branch orbits**: the multiset of the sizes of the Frobenius orbits of the
branches over `kappa`. Node or A_k (k odd) with square class +1: `[1, 1]`; with -1: `[2]`; A_k (k
even): `[1]`; ordinary m-fold: the factor degrees of `T_m`.

Found in the data (all with `kappa = F_p`):

| model | p | point | type | delta |
|---|---|---|---|---|
| X (F or G) | 5 | `T = 3`, `V = infinity` (F) | A3, conjugate branches | 2 |
| X | 13 | `T = 0`, `V = infinity` (F) | A2 | 1 |
| X | 31, 41, 73 | `T = 0` | A2 | 1 |
| X | 107 | `T = 23` | A3, rational branches | 2 |
| L, s0 = 12 | 31 | `(10, 17)` and `(22, 2)` | ordinary triple point `[1, 2]`; A3 conjugate | 3; 2 |
| L, s0 = 15 | 79 | `(26, 8)` | A3, rational branches | 2 |
| L, s0 = 93 | 103 | `(3, 2)` | A3, rational branches | 2 |

The same points were found and classified independently in PARI/GP (`work/local.gp`,
`work/localL.gp`, `work/tc.gp`).

### 4.5 The certificate and its proof

Input: the primitive integral form F (content 1), a prime `p > 2`, and the genus g of the
normalization of `X0 = V(F)` over Q. For X and the six covers, `g = 4`: [HJLPPZ, Prop. 3.10]
(62 nodes), equivalently Riemann-Hurwitz for the branch data `(2A, 23A, 23B)`:
`2g - 2 = -2*23 + (23 - 15) + 22 + 22 = 6`. The genus is an input of the API, documented at the
call site.

**Theorem 4.5.** Let `X0_p = V(F mod p)` and suppose that (i) `X0_p` has finitely many singular
points over `F_p-bar` and (ii) the sum of their delta invariants is `p_a - g`. Then X has good
reduction at p. More precisely, the normalization `X~_p` of `X0_p` is a smooth projective
geometrically connected curve of genus g over F_p, it is the special fibre of a smooth proper model
of X over `Z_(p)`, and `#X~_p(F_{p^k})` are the point counts of the reduction of X. Its zeta
function satisfies the Weil conjectures with L of degree 2g, and `P_p` is the characteristic
polynomial of Frobenius on `H^1_et(X_{Q-bar}, Q_l)` for `l != p`.

Proof. Write `R = Z_(p)` and `C` for the smooth projective curve over Q (the normalization of
`V(F)`), of genus g.
1. Let `Y0` be the closed subscheme of `P^1 x P^1` over R defined by F. As F is primitive, F mod p
   is not zero, so `Y0` is flat over R (a relative Cartier divisor). Its generic fibre is integral,
   hence so is `Y0`. Its special fibre is `X0_p`, a divisor of class (4, 23), so `p_a(X0_p) = 66`.
2. Let `Y -> Y0` be the normalization. It is finite (R is excellent). `Y` is normal, integral, proper
   and flat over R, and its generic fibre is C. By flatness the Euler characteristic of the
   structure sheaf is constant in the family: `p_a(Y_p) = 1 - chi(O_{Y_p}) = g`.
3. `Y_p` is reduced. `Y` is normal of dimension 2, so it satisfies Serre's S2, and the Cartier
   divisor `Y_p` satisfies S1. Condition (i) forces `X0_p` to be reduced (a multiple component would
   consist of singular points), so its generic points are smooth points. Over a smooth point of
   `X0_p` the scheme `Y0` is smooth over R (flat with smooth fibre), hence normal, and `Y -> Y0` is an
   isomorphism there. The generic points of `Y_p` map to generic points of `X0_p` (the map is
   finite), so `Y_p` is reduced at its generic points (R0). R0 and S1 give reduced.
4. `Y_p -> X0_p` is finite and birational (an isomorphism over the dense smooth locus). So the
   normalization `X~_p` of `X0_p` is also the normalization of `Y_p`. For a reduced projective
   curve Z with normalization `Z~`, `p_a(Z) = p_a(Z~) + delta(Z)` (from
   `0 -> O_Z -> nu_* O_{Z~} -> (skyscraper of length delta) -> 0`). With (ii):
   `p_a(X~_p) = 66 - (66 - g) = g`, and then `delta(Y_p) = p_a(Y_p) - p_a(X~_p) = 0`. So `Y_p` is
   normal, hence regular, hence smooth over the perfect field F_p, and `Y_p = X~_p`.
5. `Y` is flat over R with smooth fibres, so it is smooth over R. It is proper and normal, and its
   generic fibre is geometrically integral, so `f_* O_Y = R`; Zariski's connectedness theorem makes
   `Y_p` geometrically connected. So C has good reduction at p and `X~_p` is its reduction.
6. Smooth proper base change identifies `H^1` of the special and generic geometric fibres
   Frobenius-equivariantly; Weil's theorem for the smooth curve `X~_p` gives the rest. QED.

The same statement and proof hold with R the local ring of a number field at a prime with residue
field F_p. This covers the six covers over L at the degree-one primes `(p, s - s0)`: calF has
integral power-basis coefficients that generate the unit ideal of `O_L` (header of
`polynomial_calF.m`), so it is primitive at every prime of L.

Remark. Steps 1 to 4 also show `delta(X0_p) >= p_a - g` always (`delta(Y_p) >= 0`). A delta-sum
larger than 62 therefore never certifies anything: the model (or the curve) is bad at p.

Delta of each type: node 1, `A_k` `floor((k+1)/2)`, ordinary m-fold `m(m-1)/2`.

**Certificates and statuses.**

- `Certificate::Nodal`: every singular point is a node and their number over `F_p-bar` is
  `p_a - g`. Status Decided. This is the mandatory scope.
- `Certificate::DeltaSum`: every singular point is classified (node, `A_k`, ordinary m-fold) and
  the delta-sum is `p_a - g`. Theorem 4.5 applies verbatim. Proposed status Decided (decision D1);
  if D1 is rejected the same computation reports Unresolved with the reason "non-nodal
  singularities classified exactly, delta-sum certified" and nothing else changes.
- Otherwise Unresolved, with one of: `NonNodal` (a non-nodal point and the caller asked for the
  nodal certificate only; the counts are given with the reference rule "assumed unibranch; Weil
  test passed" when the Weil test passes), `Unclassified { point }`, `DeltaMismatch { found,
  expected }`.
- Refusals are errors, not statuses: `D = 0` identically, a fibre component, `F = 0 mod p`, `p = 2`,
  `p` not prime, the field too large for the kernel.

### 4.6 From the plane count to the normalization

For `q = p^k` and each closed singular point P with residue degree `d_P`: if `d_P` divides k, its
`d_P` geometric points are F_q-rational and each was counted once in Section 4.1; each has
`b_P(k) = sum of the branch-orbit sizes e with e dividing k/d_P` rational branches. So

`N_k = #X0_p(F_q) + sum over P with d_P | k of d_P (b_P(k) - 1)`.

For a node this is the reference rule (`+1` rational branches, `-1` conjugate, and conjugate
branches become rational over even extensions of `kappa`). The reference rule for non-nodal points
is `b = 1` ("assumed unibranch"); it is right for A_k with k even and wrong otherwise. The
corrections come from the certificate, so the fibre loop does no Hessian work. Checked against the
reference implementation (which tests every repeated root in the loop): all 25 rows agree.

### 4.7 L-polynomial from the counts

`s_k = p^k + 1 - N_k`, `k = 1..g`. Newton's identities `m e_m = sum_{i=1..m} (-1)^(i-1) e_{m-i} s_i`
give `e_1..e_g`. Each must be an integer (`WeilError::NonIntegral` otherwise: wrong counts or bad
reduction). The functional equation fills `e_{g+i} = p^i e_{g-i}`. Then
`P_p(T) = sum (-1)^i e_i T^(2g-i)` and `L(T) = T^(2g) P_p(1/T)` (MAGMA `LPolynomial`). Integer
arithmetic in `rustmath-integers`; `i128` suffices for p < 2^20 but the API uses `Integer`.

### 4.8 Exact Weil test

Write `P(T) = T^g R(T + p/T)`: with `c_j` the coefficient of `T^(g+j)`, and `D_0 = 2`, `D_1 = x`,
`D_{j+1} = x D_j - p D_{j-1}` (so `T^j + (p/T)^j = D_j(T + p/T)`), `R(x) = c_0 + sum_{j>=1} c_j
D_j(x)`, a monic integer polynomial of degree g. The roots of P have absolute value `sqrt p` iff all
roots of R are real and lie in `[-2 sqrt p, 2 sqrt p]` (a root x of R gives the pair of roots of
`z^2 - x z + p`, which have modulus `sqrt p` iff `x^2 <= 4p` and x is real).

Decided exactly, multiplicities included, with rational endpoints only:

1. All roots of R real: for each factor `R_i` of the square-free decomposition over Z (`zx`),
   `count_real_roots(R_i) = deg R_i`.
2. All `x^2 <= 4p`: let `S(y) = E(y)^2 - y O(y)^2` where `R(x) = E(x^2) + x O(x^2)`; its roots are the
   `x_i^2`. Given 1, they are `>= 0`; require `count_real_roots_in(S_i, -1, 4p) = deg S_i` for each
   square-free factor `S_i` of S (closed at `4p`, which matters for real Weil numbers `+-sqrt p`).

The test returns a boolean, not a status: it is a finite exact computation. Tests: the 25 table rows,
`(T^2 + p)^g`, `(T^2 - p)^2 ...` with repeated roots, the non-Weil row at p = 5 in reference mode,
and polynomials with one root just outside the disc.

### 4.9 Derived checks

- **Mod 23.** `chi(p) = kronecker(-23, p)` (`Integer::kronecker_symbol`); check
  `P_p(chi) = P_p(chi p) = 0 mod 23`. For the covers over L at a degree-one prime the same check
  applies with the same `chi(p)` (Frobenius at a degree-one prime restricts to `chi(p)` on
  `Q(sqrt(-23))`); it holds at every certified L row and fails only at the uncorrected reference
  row at 31.
- **Mod 2.** Factor `P_p mod 2` with `fp_factor::factor_with_multiplicity(., 2)`, enumerate the
  products of the factors of degree g and keep those Q with `Q Q^rev = P_p mod 2`; report all of
  them (Q and `Q^rev` both appear when they differ). The table's `Q` is reproduced at every row.
- **Both fibrations.** `#X0_p(F_q)` is independent of the fibration, so summing over T with degree
  23 fibres (the `Vec` path) must give the same plane count. This checks the kernel at a second
  fibre degree; the node corrections are shared. A third, fully independent check for `k <= 2`:
  evaluate F at all points of `P^1 x P^1 (F_q)`.

### 4.10 Order of Frobenius on Jac(X)[2] (extension item 5)

The characteristic polynomial of Frobenius on `J[2] = T_2 J / 2` is `P_p mod 2`. If it is
squarefree, Frobenius is semisimple on `J[2]` and its order is the lcm over the irreducible factors
`f` of the order of x in `(F_2[x]/f)^*`: Decided. Otherwise Frobenius may have a unipotent part:
report `Unresolved { semisimple_order, divides: semisimple_order * 2^ceil(log2 e_max) }`. In the
table, `P_p mod 2` is squarefree at 11, 31, 79 and 97 (order 15 each; the brief predicts 15 at 11,
31, 79).

## 5. Degenerate cases

| Case | Where it happens | Treatment |
|---|---|---|
| `p = 2` | | `Err(CountError::CharacteristicTwo)`: Hessian and completing the square need p odd |
| `p = 3` | X: F mod 3 = `(V-1)^23 T (T-1)^3`, `D = 0`; G also | `Err(DegenerateModel::DiscriminantVanishes)`; follow-up F-1 |
| `p = 23` | wild ramification of t | not excluded by the code; the certificate decides. Not in the tables (the reference skips it); run and report |
| `F = 0 mod p` | impossible for a primitive form | `Err(ZeroForm)` |
| `D = 0` identically | X at 3; L at (5, 3) and (11, 2) | `Err(DegenerateModel::DiscriminantVanishes)` ("model degenerate at this prime") |
| vertical component `V = v0` | F at 131; G at 13 | `Err(DegenerateModel::FibreComponent { place })` from the certificate, and from the fibre loop if met there |
| horizontal component `T = t0` | not met | not refused explicitly: Theorem 4.5 shows such a model can never reach delta-sum `p_a - g` (step 5 forces irreducibility), so the status is Unresolved |
| fibre degree drop | many fibres | root at `T = infinity` counted once; multiple root at infinity handled in the `U = 1/T` chart |
| `V = infinity` | `deg D < 138` (X at 5, 13) | analysed as the place `V = 0` of the reversed form |
| two singular points in one fibre | X at 13 and others | `deg G = 2` case of 4.3 (split, irreducible or double) |
| node with conjugate branches | most primes | branch orbit `[2]`: `-1` over odd degree, `+1` over even degree |
| node or point of degree `d > 1` | residue degrees up to 62 | counted only when `d` divides k |
| non-nodal, classified | Section 4.4 table | `Certificate::DeltaSum` (D1) or Unresolved `NonNodal` |
| non-nodal, unclassified | not met | `Unresolved(Unclassified)`, counts with the unibranch rule on request |
| delta-sum above `p_a - g` | E2 test curve at p = 11 | `Unresolved(DeltaMismatch)` |
| counts not integral in Newton | not met at certified primes | `Err(WeilError::NonIntegral)` |
| not Weil | X at 5 in reference mode | reported, never Decided |
| `q >= 2^63` or `k > 8` | | `Err(FieldTooLarge)` |
| `p >= 2^16` | | `Err(PrimeTooLarge)`: the kernel bound (products below `2^40`); k = 4 there would need about `2^62` fibres anyway |
| malformed coefficient array | | `Err(CountError::Shape)` |

## 6. Public API

### 6.1 `rustmath-finitefields/src/small_field.rs` (new)

Header cites MAGMA ch. 21 (`FiniteField`, `Frobenius`, `Norm`, `IsSquare`) and the timing that
justifies it.

```rust
/// GF(p^K) on machine words: p < 2^16 (proved prime), K <= 8, elements [u64; K] little-endian.
pub struct SmallFq<const K: usize> { /* p, Barrett constant, modulus, Frobenius matrix, F_p tables */ }
pub type SmallFqElt<const K: usize> = [u64; K];

impl<const K: usize> SmallFq<K> {
    pub fn new(p: u64) -> Result<Self>;                       // Conway polynomial if tabulated, else the least irreducible
    pub fn with_modulus(p: u64, modulus: &[u64]) -> Result<Self>; // monic, irreducible (checked)
    pub fn characteristic(&self) -> u64;
    pub fn order(&self) -> u64;                               // p^K < 2^63, checked in new
    pub fn modulus(&self) -> &[u64];
    pub fn zero(&self) -> SmallFqElt<K>; pub fn one(&self) -> SmallFqElt<K>;
    pub fn from_fp(&self, c: u64) -> SmallFqElt<K>;
    pub fn add/sub/neg/mul(&self, ..) -> SmallFqElt<K>;
    pub fn frobenius(&self, a: &SmallFqElt<K>) -> SmallFqElt<K>;
    pub fn norm(&self, a: &SmallFqElt<K>) -> u64;
    pub fn inverse(&self, a: &SmallFqElt<K>) -> Option<SmallFqElt<K>>;
    pub fn is_square(&self, a: &SmallFqElt<K>) -> bool;
    pub fn code(&self, a: &SmallFqElt<K>) -> u64; pub fn from_code(&self, n: u64) -> SmallFqElt<K>;
}
/// Number of distinct roots in GF(q) of a nonzero polynomial (little-endian coefficients).
pub fn count_distinct_roots<const K: usize>(fq: &SmallFq<K>, f: &[SmallFqElt<K>]) -> usize;
```

### 6.2 `rustmath-curves/src/weil.rs` (new)

```rust
pub struct FrobeniusPolynomial { /* p: Integer, genus: usize, coeffs: Vec<Integer> (descending) */ }
pub enum WeilError { NoCounts, NonIntegral { index: usize, numerator: Integer, denominator: usize } }
pub enum TwoTorsionOrder { Decided(u64), Unresolved { semisimple_order: u64, divides: u64 } }

impl FrobeniusPolynomial {
    pub fn from_point_counts(p: &Integer, counts: &[Integer]) -> Result<Self, WeilError>;
    pub fn from_coefficients(p: &Integer, coeffs: Vec<Integer>) -> Result<Self, WeilError>; // checks functional equation
    pub fn coefficients(&self) -> &[Integer];          // 1, a1, ..., p^g
    pub fn l_polynomial(&self) -> Vec<Integer>;        // ascending L(T), MAGMA LPolynomial
    pub fn point_counts(&self, kmax: usize) -> Vec<Integer>;
    pub fn real_weil_polynomial(&self) -> Vec<Integer>; // R, ascending
    pub fn is_weil(&self) -> bool;                     // exact (4.8)
    pub fn eval_mod(&self, x: &Integer, m: &Integer) -> Integer;
    pub fn factor_mod(&self, l: i64) -> Vec<(Vec<i64>, u32)>;
    pub fn reciprocal_halves_mod2(&self) -> Vec<Vec<i64>>; // all Q with P = Q Q^rev mod 2
    pub fn two_torsion_frobenius_order(&self) -> TwoTorsionOrder;
}
```

### 6.3 `rustmath-curves/src/nodal_biprojective.rs` and `src/nodal_biprojective/` (new)

Submodules: `form.rs` (forms over Z and mod p), `fibres.rs` (4.1, 4.2 driver), `singular.rs` (4.3,
residue fields), `local.rs` (4.4), `certify.rs` (4.5, 4.6). The module doc states that it does not
use the plane-curve layer of the crate.

```rust
/// sum F[i][j] X^i Y^j over Z, bidegree (dx, dy); the curve in P^1 x P^1.
pub struct BihomogeneousForm { /* coeffs: Vec<Vec<Integer>> */ }
impl BihomogeneousForm {
    pub fn new(coeffs: Vec<Vec<Integer>>) -> Result<Self, CountError>;
    pub fn from_decimal_rows(rows: &[Vec<String>]) -> Result<Self, CountError>;
    pub fn bidegree(&self) -> (usize, usize);
    pub fn arithmetic_genus(&self) -> usize;                    // (dx-1)(dy-1)
    pub fn content(&self) -> Integer; pub fn primitive_part(&self) -> Self;
    pub fn transpose(&self) -> Self;
    pub fn mobius_y(&self, a: i64, b: i64, c: i64, d: i64) -> Self; // F(X, (aY+b)/(cY+d)) (cY+d)^dy, primitive
    pub fn equals_up_to_scalar(&self, other: &Self) -> bool;
    pub fn reduce(&self, p: u64) -> Result<FormModP, CountError>;
}
pub struct FormModP { /* p, coeffs mod p */ }
impl FormModP { pub fn new(p: u64, coeffs: Vec<Vec<u64>>) -> Result<Self, CountError>; }

pub enum Fibration { FibresInX, FibresInY }     // the fibre variable; the other one is the base
pub fn plane_point_count(f: &FormModP, k: u32, fib: Fibration) -> Result<u64, CountError>;

pub enum LocalType { Node, A(usize), OrdinaryMultiple(usize), Unclassified(String) }
pub struct SingularPoint { pub residue_degree: usize, pub location: PointLocation,
                           pub local: LocalType, pub delta: Option<usize>,
                           pub branch_orbits: Option<Vec<usize>> }
pub struct SingularLocus { pub points: Vec<SingularPoint>, pub disc_degree: usize,
                           pub disc_factor_degrees: Vec<(usize, u32)> }
pub fn singular_locus(f: &FormModP) -> Result<SingularLocus, CountError>; // fibres in X, X-degree 2..=4

pub enum Certificate { Nodal { nodes: usize }, DeltaSum { delta_sum: usize } }
pub enum Unresolved { NonNodal, Unclassified(SingularPoint), DeltaMismatch { found: usize, expected: usize } }
pub enum ReductionStatus { Decided(Certificate), Unresolved(Unresolved) }
pub struct CountOptions { pub fibration: Fibration, pub allow_delta_certificate: bool, // D1
                          pub unibranch_if_unresolved: bool, pub threads: Option<usize> }
pub struct NormalizationCounts { pub p: u64, pub counts: Vec<u64>, pub plane_counts: Vec<u64>,
                                 pub status: ReductionStatus, pub locus: SingularLocus,
                                 pub branch_rule: BranchRule }
pub enum BranchRule { Exact, AssumedUnibranch }
pub fn normalization_point_counts(f: &FormModP, genus: usize, kmax: u32, opts: &CountOptions)
    -> Result<NormalizationCounts, CountError>;

pub enum Degeneracy { DiscriminantVanishes, FibreComponent { place: String } }
pub enum CountError { NotPrime(u64), CharacteristicTwo, PrimeTooLarge(u64), FieldTooLarge { p: u64, k: u32 },
                      ZeroForm, DegenerateModel(Degeneracy), FibreDegree(usize), Shape(String) }
```

`normalization_point_counts` never returns counts with `BranchRule::AssumedUnibranch` and status
Decided. With `unibranch_if_unresolved = false` an unresolved model returns the plane counts and
the locus but `counts` empty.

### 6.4 M23 driver (example and tests only, not library API)

`rustmath-curves/examples/m23_zeta.rs`: reads the vendored G (and F, calF if `M23ISGALOIS` is set),
for each prime in a range tries `[G, G7, G13]` (or F, or calF at each degree-one prime of L),
prints `p, s0, model, N_1..N_4, a_1..a_4, status, Weil, mod 23, P mod 2, Q, 2-torsion order`,
optionally as JSON in the format of `zeta_X.json`. Shared loaders in `tests/common/m23.rs`
(parsers for `polynomial_F.m` (terms `<int>*T^i*V^j`) and `polynomial_calF.m` (power-basis
coefficients), reduction at `(p, s - s0)`).

## 7. Results of the prototype

Release build, 4 threads, exact branch counts (`EXACT=1`), F read from `M23ISGALOIS`. The CC0
models give the same values wherever they are defined: G, G7, G13 checked at k <= 3 for every p up
to 131 and at k = 4 for 7 (G7), 11 (G) and 13 (G13). Status: `Nodal` = Certificate::Nodal;
`DeltaSum (...)` = Certificate::DeltaSum with the listed non-nodal points (decision D1). Every row
listed passes the exact Weil test and the mod 23 check, and `P mod 2 = Q Q^rev` with the Q shown
(the reverse of Q is the other solution when it differs). In the reference mode (non-nodal points
counted once) the rows at 13, 31, 41, 73 are unchanged and p = 5 gives the table's refused row
`8, 36, 170, 560` (not Weil).

Order of Frobenius on `J[2]` (item 5, Section 4.10): Decided, equal to 15, at 11, 31, 79, 97 (`P mod 2`
squarefree); Unresolved elsewhere, with semisimple order 7 (`(x+1)^2(x^3+x+1)(x^3+x^2+1)`), 3
(`(x+1)^4(x^2+x+1)^2`), 5 (`(x^4+x^3+x^2+x+1)^2`) or 1 (`(x+1)^8`).

### 7.1 X

| p | N1 | N2 | N3 | N4 | (a1, a2, a3, a4) | P mod 2 | Q (up to reversal) | status | reference |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 7 | 37 | 169 | 561 | (1,6,20,16) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | DeltaSum (60 nodes, A3 conj) | differs: [8, 36, 170, 560] |
| 7 | 11 | 69 | 383 | 2433 | (3,14,46,138) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | Nodal | equal |
| 11 | 15 | 133 | 1368 | 14905 | (3,10,33,145) | `(x^4+x+1)(x^4+x^3+1)` | `x^4+x^3+1` | Nodal | equal |
| 13 | 11 | 161 | 2135 | 29089 | (-3,0,-12,188) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | DeltaSum (61 nodes, A2) | equal |
| 17 | 21 | 285 | 4923 | 82985 | (3,2,0,-130) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | Nodal | equal |
| 19 | 17 | 405 | 7265 | 130913 | (-3,26,66,74) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | Nodal | equal |
| 29 | 35 | 789 | 24773 | 708785 | (5,-14,16,1060) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | Nodal | equal |
| 31 | 33 | 993 | 29640 | 926713 | (1,16,-35,875) | `(x^4+x+1)(x^4+x^3+1)` | `x^4+x^3+1` | DeltaSum (61 nodes, A2) | equal |
| 37 | 43 | 1433 | 50881 | 1881609 | (5,44,254,3156) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | Nodal | equal |
| 41 | 42 | 1780 | 69408 | 2827728 | (0,49,162,1692) | `(x+1)^4(x^2+x+1)^2` | `x^4+x^3+x+1` | DeltaSum (61 nodes, A2) | equal |
| 43 | 44 | 1996 | 80192 | 3422024 | (0,73,228,3470) | `(x+1)^4(x^2+x+1)^2` | `x^4+x^3+x+1` | Nodal | equal |
| 47 | 47 | 2349 | 104321 | 4882281 | (-1,70,96,2934) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | Nodal | equal |
| 53 | 54 | 2804 | 148704 | 7881808 | (0,-3,-58,-2164) | `(x+1)^4(x^2+x+1)^2` | `x^4+x^3+x+1` | Nodal | equal |
| 59 | 72 | 3560 | 205056 | 12120780 | (12,111,648,3991) | `(x^4+x^3+x^2+x+1)^2` | `x^4+x^3+x^2+x+1` | Nodal | equal |
| 61 | 64 | 3740 | 226852 | 13855856 | (2,11,-24,2476) | `(x+1)^4(x^2+x+1)^2` | `x^4+x^3+x+1` | Nodal | equal |
| 67 | 80 | 4460 | 302192 | 20163724 | (12,57,584,8759) | `(x^4+x^3+x^2+x+1)^2` | `x^4+x^3+x^2+x+1` | Nodal | equal |
| 71 | 68 | 5048 | 358256 | 25432076 | (-4,11,92,4679) | `(x^4+x^3+x^2+x+1)^2` | `x^4+x^3+x^2+x+1` | Nodal | equal |
| 73 | 83 | 5541 | 390089 | 28393065 | (9,146,1428,12030) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | DeltaSum (61 nodes, A2) | equal |
| 79 | 77 | 6369 | 491636 | 38964585 | (-3,68,-663,7335) | `(x^4+x+1)(x^4+x^3+1)` | `x^4+x^3+1` | Nodal | equal |
| 83 | 91 | 6885 | 574375 | 47483041 | (7,22,902,12258) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | Nodal | new |
| 89 | 104 | 7942 | 707906 | 62766210 | (14,108,1576,22324) | `(x+1)^8` | `x^4+1` | Nodal | new |
| 97 | 101 | 9385 | 910820 | 88546849 | (3,-8,-651,2563) | `(x^4+x+1)(x^4+x^3+1)` | `x^4+x^3+1` | Nodal | new |
| 101 | 114 | 10456 | 1030926 | 104090928 | (12,199,2020,28200) | `(x+1)^4(x^2+x+1)^2` | `x^4+x^3+x+1` | Nodal | new |
| 103 | 96 | 10648 | 1092594 | 112536824 | (-8,51,-282,-2198) | `(x+1)^4(x^2+x+1)^2` | `x^4+x^3+x+1` | Nodal | new |
| 107 | 116 | 11482 | 1226360 | 131139882 | (8,48,652,19390) | `(x+1)^8` | `x^4+1` | DeltaSum (60 nodes, A3 rat) | new |
| 109 | 104 | 11986 | 1294118 | 141216370 | (-6,70,-652,18718) | `(x+1)^8` | `x^4+1` | Nodal | new |
| 113 | 128 | 13016 | 1444142 | 163058328 | (14,221,2594,29766) | `(x+1)^4(x^2+x+1)^2` | `x^4+x^3+x+1` | Nodal | new |
| 127 | 127 | 16501 | 2047933 | 260199313 | (-1,186,-336,31116) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | Nodal | new |

Reference column: "equal" means equal to `zeta_X.json` (with the exact branch rule; at 13, 31, 41,
73 the exact rule and the reference rule agree because the point is a cusp). p = 3: refused
(`D = 0`). p = 23: not run (follow-up F-5). p = 131: F has a fibre component and is refused; G is
nodal with 62 nodes: `N = 144, 17064, 2246976, 294472176`, `a = (12, 23, -672, -12864)`, Weil, mod 23
holds, `P mod 2 = (x+1)^4 (x^2+x+1)^2`, Q = `x^4+x^3+x+1`, status Nodal (new).

### 7.2 Covers over L at degree-one primes

| p | s0 | N1 | N2 | N3 | N4 | (a1, a2, a3, a4) | P mod 2 | Q (up to reversal) | status | reference |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 3 | | | | | | | | refused: discriminant vanishes identically mod p | refused |
| 7 | 5 | 7 | 45 | 343 | 2545 | (-1,-2,2,38) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | Nodal | equal |
| 11 | 2 | | | | | | | | refused: discriminant vanishes identically mod p | refused |
| 19 | 3 | 15 | 413 | 6849 | 130169 | (-5,38,-152,650) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | Nodal | equal |
| 31 | 12 | 43 | 1017 | 30265 | 923729 | (11,88,682,4438) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | DeltaSum (57 nodes, ord. triple [1,2], A3 conj) | differs: [44, 1014, 30266, 923726] |
| 41 | 8 | 40 | 1790 | 68866 | 2828954 | (-2,56,-128,2402) | `(x+1)^8` | `x^4+1` | Nodal | equal |
| 43 | 15 | 43 | 1829 | 78799 | 3422769 | (-1,-10,-226,1278) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | Nodal | equal |
| 47 | 13 | 54 | 2316 | 104076 | 4889096 | (6,71,438,5270) | `(x+1)^4(x^2+x+1)^2` | `x^4+x^3+x+1` | Nodal | new |
| 53 | 21 | 52 | 2862 | 150562 | 7889314 | (-2,28,508,-1024) | `(x+1)^8` | `x^4+1` | Nodal | new |
| 53 | 36 | 57 | 2925 | 147507 | 7890353 | (3,62,-280,512) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | Nodal | new |
| 53 | 41 | 45 | 2885 | 147972 | 7888001 | (-9,78,-761,4593) | `(x^4+x+1)(x^4+x^3+1)` | `x^4+x^3+1` | Nodal | new |
| 61 | 52 | 51 | 3909 | 226185 | 13840033 | (-11,154,-1516,12108) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | Nodal | new |
| 67 | 15 | 71 | 4661 | 302150 | 20149585 | (3,90,723,5045) | `(x^4+x+1)(x^4+x^3+1)` | `x^4+x^3+1` | Nodal | new |
| 67 | 42 | 62 | 4614 | 301154 | 20166970 | (-6,80,-278,6274) | `(x+1)^8` | `x^4+1` | Nodal | new |
| 71 | 37 | 56 | 5114 | 358256 | 25415418 | (-16,164,-1144,7086) | `(x+1)^8` | `x^4+1` | Nodal | new |
| 71 | 67 | 78 | 5210 | 358566 | 25433810 | (6,102,758,11934) | `(x+1)^8` | `x^4+1` | Nodal | new |
| 79 | 15 | 94 | 6510 | 493546 | 38940122 | (14,232,2502,23582) | `(x+1)^8` | `x^4+1` | DeltaSum (60 nodes, A3 rat) | new |
| 79 | 40 | 82 | 6316 | 494788 | 38952656 | (2,39,658,2568) | `(x+1)^4(x^2+x+1)^2` | `x^4+x^3+x+1` | Nodal | new |
| 79 | 46 | 70 | 6334 | 492232 | 38966242 | (-10,96,-896,10508) | `(x+1)^8` | `x^4+1` | Nodal | new |
| 89 | 57 | 94 | 8088 | 707974 | 62746864 | (4,91,1344,9280) | `(x+1)^4(x^2+x+1)^2` | `x^4+x^3+x+1` | Nodal | new |
| 89 | 69 | 94 | 8090 | 705808 | 62734682 | (4,92,626,3438) | `(x+1)^8` | `x^4+1` | Nodal | new |
| 97 | 76 | 103 | 9497 | 913861 | 88569977 | (5,56,634,13668) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | Nodal | new |
| 101 | 74 | 109 | 10305 | 1024111 | 104049033 | (7,76,-1646,-14600) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | Nodal | new |
| 101 | 98 | 115 | 10117 | 1027081 | 104049377 | (13,42,-1260,-18212) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | Nodal | new |
| 103 | 93 | 117 | 10897 | 1091211 | 112578169 | (13,228,1726,23860) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | DeltaSum (60 nodes, A3 rat) | new |
| 127 | 63 | 135 | 16457 | 2050377 | 260132721 | (7,188,1866,19142) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | Nodal | new |
| 127 | 74 | 130 | 16210 | 2047378 | 260120418 | (2,42,-254,-5846) | `(x+1)^8` | `x^4+1` | Nodal | new |

Anchor at 53: s0 = 21, 36, 41 give `N_1 = 52, 57, 45` (sorted 45, 52, 57, as in `test_core.py` and
[HJLPPZ, Prop. 3.14] for 45 and 52), all three Nodal.

The L row at 31: the reference value `44, 1014, 30266, 923726` comes back in the reference mode, with
`P mod 2 = (x^2+x+1)^4` and Q = `x^4+x^2+1` as in the brief, but `P(1) = 15` and `P(31) = 7` mod 23.
The exact value in the table is a new result, verified by the derived checks; it has no external
reference. The PARI/GP classification of both points is in `work/localL.gp` and `work/tc.gp`.

## 8. Cost

All numbers are measured with the prototype on this machine (4 cores, shared with other agents;
load average about 2 during most runs). Release profile unless stated.

### 8.1 Per fibre

| what | threads | cost per orbit representative |
|---|---:|---|
| kernel, p = 31, k = 4 | 1 | 5.0 to 5.4 us |
| kernel, p = 79, k = 3 | 1 | 2.9 us |
| kernel, p = 127, k = 3 | 1 | 3.1 us |
| kernel, p = 79, k = 4 | 4 | 2.0 us wall (19.7 s for 9.7 million fibres) |
| kernel before the semilinear p-power map (k - 1 compositions) | 1 | 7.0 to 7.5 us |
| existing `FiniteField` + `FqPoly` (`pow_mod`, `gcd`), p = 31, k = 4 | 1 | 4.0 ms |
| existing stack, p = 31, k = 1 | 1 | 122 us |

So the existing types are about 750 times too slow at k = 4; the kernel is justified. The k = 4
count dominates: about `p^4/4` fibres, so about `0.5 us * p^4` wall time on 4 threads (79: 15 s
predicted, 14.7 s measured; 127: 130 s predicted, 99 s measured).

### 8.2 Tables (release, 4 threads)

| run | time |
|---|---|
| certificate (discriminant, factorization, all fibres over its roots) | 2 to 14 ms per prime |
| X, p <= 79, k = 1..4 (the 18 rows of the brief and p = 5) | 60 s |
| X, p <= 127 | 504 s (8.4 min) |
| covers over L, the 7 rows of the brief (p <= 43) | 2.7 s |
| covers over L, all degree-one primes p <= 127 (27 rows, 2 refused) | 498 s (8.3 min) |
| X and L to 127 together, wall clock | 16 min 43 s (64 min CPU) |
| T-fibration (degree 23 fibres), p = 31, k = 3 | 0.56 s (V-fibration: 0.026 s) |
| T-fibration, p = 11, k = 4 | 0.61 s (V-fibration: 0.019 s) |
| X at p = 131 from G, k = 4 | 114 s |

Beyond 127 the cost grows like `p^4`: p near 200 would take about 13 min per prime at k = 4.
Follow-up F-4 (a resolvent-cubic root count) would cut the constant by about 2 to 3.

### 8.3 Debug (prototype equivalents of the fast tests of 9.1)

| test | threads | time |
|---|---:|---|
| G7 at 7, k = 1..4, certificate, L-polynomial, checks | 1 | 0.07 s |
| G at 11, k = 1..4 | 1 | 0.31 s |
| G at 17 and 19, k = 1..3 | 4 | 0.25 s |
| G13 at 13, k = 1..3 | 4 | 0.08 s |
| G at 31, k = 1..3 | 1 | 0.60 s |
| G at 5, k = 1..3; refusal at 3 | 4 | 0.02 s |
| both fibrations at 7 and 11, k = 1, 2 | 4 | 0.04 s |
| the 18 small-curve cases | 1 | 0.07 s |
| (not in the fast set) G at 19, k = 4 | 1 | 2.9 s |
| (not in the fast set) G at 29, k = 4 | 4 | 4.5 s |

The fast set takes about 2 s in debug. The prototype's dependency set builds from scratch in 10 s
(debug) and 31 s (release). A test build of `rustmath-curves` compiles its whole dependency tree
(symbolic, schemes, groups, numberfields, numerical and more); that was not measured here and is
the dominant cost of the fast tests. The prototype's target directory reached 477 MB.

## 9. Tests

All new files cite MAGMA ch. 114/42 (and ch. 21 for the kernel) and the papers. Reference values come
from data files or PARI/GP output pasted with the command in a comment.

### 9.1 Fast tests (normal suite, debug)

`rustmath-finitefields` (unit tests in `small_field.rs` and `tests/small_field.rs`):

- K = 1..4 for p in {5, 7, 31, 127, 65521}: products, inverses, Frobenius, norm and `is_square`
  agree with `FiniteField::with_modulus` on 200 seeded elements each.
- `count_distinct_roots` against `fp_factor` (K = 1) and against `poly_factor::roots` over
  `FiniteField` (K = 2, 3, 4) on 100 seeded polynomials of degree 1..6, including repeated roots
  and `f = x^q - x` factors.
- Refusals: p not prime, p >= 2^16, reducible modulus.

`rustmath-curves/src/weil.rs` (unit tests):

- Newton round trip: `point_counts(from_point_counts(N)) = N` for the 25 table rows.
- Exact Weil test, true cases: every certified table row; `(T^2 + p)^4` (R = `x^4`, a fourfold root);
  `(T^2 - p)^4` (R = `(x^2 - 4p)^2`, double roots on the boundary `+-2 sqrt p`, which checks the
  closed interval); `T^2 + aT + p` with `a = floor(2 sqrt p)`. False cases: the reference-rule row at
  5 (`8, 36, 170, 560`); `T^2 + aT + p` with `a = floor(2 sqrt p) + 1`; `T^4 + (2p+1) T^2 + p^2`
  (R = `x^2 + 1`, non-real roots); `T^2 + pT + p` for p >= 5.
- Mod 2 halves: the table's Q at 7, 11, 41, 59 (data file); 2-torsion order 15 at 11, 31, 79.

`rustmath-curves/tests/biprojective_small_curves.rs` (PARI/GP references from `gp/small_curves.gp`,
values listed in Section 9.4):

- smooth (4,2) curve, genus 3, p = 11: `hyperellcharpoly` of the discriminant;
- (4,2) curves with one node (rational branches: C; conjugate: B, C2, G, H), one A2 (D), one A3
  (rational: F; conjugate: E, I, I2): `hyperellcharpoly` of the normalization (branch classes
  from `gp/branch.gp`: `h(t0)` square or not);
- the same curves after `T -> 1/T` and `V -> 1/V` (singular point at infinity in each chart);
- `V^2 = T^3 + 3T + 5` as a (3,2) form (A2 at `(infinity, infinity)`), p = 7, 11, 13: `ellap`;
- `V^3 = T^3 + 2T + 3` as a (3,3) form (ordinary triple point at `(infinity, infinity)`),
  p = 7, 13, 19: `ellap` of `ellinit(ellfromeqn(...))`; p = 11: delta-sum 4 > 3, status
  `Unresolved(DeltaMismatch)` (bad reduction: the cubic acquires a double root mod 11);
- refusals: a form with a fibre component, a form with `D = 0`.

`rustmath-curves/tests/m23_zeta.rs` (vendored G, `zeta_X.json`; `h62.json` for one cross-check):

- G7 at p = 7 and G at p = 11, k = 1..4: equal to the table, `Certificate::Nodal { nodes: 62 }`,
  Weil, mod 23, mod 2 with the table's Q;
- p = 17, 19 at k = 1..3 against the table, certificate Nodal;
- p = 13 (G13), 31 at k = 1..3: certificate `DeltaSum` with one A2 and 61 nodes; with
  `allow_delta_certificate = false` the status is `Unresolved(NonNodal)` and the counts with
  `AssumedUnibranch` equal the table;
- p = 5 (G): `DeltaSum` with one A3 (conjugate) and 60 nodes; `N_1..N_3 = 7, 37, 169`; the reference
  rule gives `8, 36, 170` (the table's refused row);
- p = 3: `Err(DegenerateModel(DiscriminantVanishes))`;
- both fibrations agree at p = 7, 11 for k = 1, 2; brute force over `P^1 x P^1 (F_p)` agrees at
  p = 7, 11 (plane counts 11 and 15; PARI/GP `work/brute.gp` gives 11, 15 and 22 at 7, 11, 17, the
  same as both fibrations of the prototype);
- the T-coordinates of the 62 nodes at p = 11 are roots of `h62 mod 11` (data file; links the
  certificate to the discriminant identity of HANDOFF Section 1; checked in PARI/GP at 7, 11, 17 by
  `work/h62chk.gp`: 62 of 62 each).

### 9.2 Opt-in tests needing `M23ISGALOIS` (`#[ignore]`)

Command: `M23ISGALOIS=/home/user/external/m23isgalois cargo test -p rustmath-curves --test m23_zeta -- --ignored env_`

- `env_f_equals_g_by_mobius`: the exact Integer identity of Section 3;
- `env_l_table`: the 7 rows of `zeta_L.json` (refusals at 5 and 11; Nodal at 7, 19, 41, 43 with the
  table values; at 31 the reference-rule counts equal the table and fail mod 23, and the exact
  counts are `43, 1017, 30265, 923729`, status `DeltaSum`);
- `env_anchor_53`: 45, 52, 57, all Nodal;
- `env_x_from_f`: X rows 7..79 at k <= 3 from F equal those from G.

### 9.3 Opt-in release tests (`#[ignore]`)

Command: `cargo test --release -p rustmath-curves --test m23_zeta -- --ignored full_`

- `full_x_table_k4`: all 18 rows of `zeta_X.json` at k = 4 from `[G, G7, G13]`, plus p = 5 (Decided,
  new values). Measured 60 s on 4 threads.
- `full_both_fibrations_k3`: T-fibration at k = 3 for p <= 31.

The example `cargo run --release -p rustmath-curves --example m23_zeta -- 5 127 4` regenerates the
table to 127 (8.4 min) and prints statuses; with `M23ISGALOIS` set, `--cover L` does the L covers.

### 9.4 PARI/GP reference values for the small curves

From `gp/small_curves.gp` (`hyperellcharpoly`, `ellap`; a(T) = 1 unless stated; b, c ascending
lists are in the script):

| name | p | a | b | c | singular point of y^2 = b^2 - 4ac | normalization | `P` (PARI/GP) |
|---|---:|---|---|---|---|---|---|
| A | 11 | `[3,1,0,0,2]` | `[1,0,5,1]` | `[7,0,4,0,1]` | none (genus 3) | `y^2 = D` | `x^6 + 5x^4 - 6x^3 + 55x^2 + 1331` |
| B | 11 | `[1]` | `[5,1,0,3,1]` | `[0,0,9,2,6]` | node at 2, conjugate | genus 2 | `x^4 + 3x^3 + 14x^2 + 33x + 121` |
| C | 11 | `[1]` | same | `[7,5,9,2,6]` | node at 4, rational | genus 2 | `x^4 + 2x^3 + 4x^2 + 22x + 121` |
| C2 | 11 | `[1]` | same | `[8,5,9,2,6]` | node at 5, conjugate | genus 2 | `x^4 + 4x^3 + 13x^2 + 44x + 121` |
| D | 11 | `[1]` | same | `[1,8,3,2,6]` | A2 at 3 | genus 2 | `x^4 + 10x^2 + 121` |
| E | 11 | `[1]` | same | `[0,9,4,6,6]` | A3 at 1, conjugate | genus 1 | `x^2 - x + 11` |
| F | 11 | `[1]` | same | `[10,5,7,0,6]` | A3 at 6, rational | genus 1 | `x^2 + 2x + 11` |
| G | 13 | `[1]` | `[1,7,0,1,2]` | `[10,10,1,5,3]` | node at 0, conjugate | genus 2 | `x^4 + x^3 - 8x^2 + 13x + 169` |
| H | 13 | `[1]` | same | `[0,5,1,5,3]` | node at 1, conjugate | genus 2 | `x^4 + 7x^3 + 29x^2 + 91x + 169` |
| I | 13 | `[1]` | same | `[7,0,12,0,3]` | A3 at 2, conjugate | genus 1 | `x^2 - 6x + 13` |
| I2 | 13 | `[1]` | same | `[0,10,12,8,3]` | A3 at 3, conjugate | genus 1 | `x^2 - 5x + 13` |
| E1 | 7, 11, 13 | `V^2 = T^3 + 3T + 5` | | | A2 at (inf, inf) | genus 1 | `ellap = 1, 3, 5` |
| E2 | 7, 13, 19 | `V^3 = T^3 + 2T + 3` | | | ordinary triple point at (inf, inf) | genus 1 | `ellap = -5, 7, -8` (model `ellfromeqn` = `[0,0,27,0,-251]`) |
| E2 | 11 | same | | | the cubic has a double root mod 11 | | delta-sum 4 > 3: `Unresolved(DeltaMismatch)` |

The prototype reproduces every row (`wpf_proto small`), including the node, cusp, tacnode and
triple-point branch counts at infinity.

## 10. Files

New:

- `rustmath-finitefields/src/small_field.rs`, `rustmath-finitefields/tests/small_field.rs`;
  `rustmath-finitefields/src/lib.rs` (module line and re-export).
- `rustmath-curves/src/weil.rs`.
- `rustmath-curves/src/nodal_biprojective.rs` and `rustmath-curves/src/nodal_biprojective/{form,
  fibres, singular, local, certify}.rs`; `rustmath-curves/src/lib.rs` (two module lines).
- `rustmath-curves/tests/biprojective_small_curves.rs`, `rustmath-curves/tests/m23_zeta.rs`,
  `rustmath-curves/tests/common/m23.rs`.
- `rustmath-curves/tests/data/m23/`: `lelay_optimal_23_4_Z.json` (CC0, Le Lay, commit `3f6ab4b`),
  `zeta_X.json`, `zeta_L.json`, `h62.json` (johnjanik/M23 at `d05475b`), `README.md` (sources,
  licences, sha256, the Moebius identity, the L-31 note).
- `rustmath-curves/examples/m23_zeta.rs`.
- `rustmath-curves/Cargo.toml`: `serde_json` under `[dev-dependencies]`.

Not touched: `PROJECT_TRACKER.md`, `docs/` (a later step), the facade modules of `rustmath-curves`.

## 11. Plan with acceptance commands

Step 0. Baseline: `cargo test -p rustmath-finitefields` and `cargo check -p rustmath-curves` green at
`0d277f0` (record times).

Step 1. `small_field.rs` with its tests.
`cargo test -p rustmath-finitefields small_field` and `cargo test -p rustmath-finitefields`;
`cargo clippy -p rustmath-finitefields`. Commit 1.

Step 2. `weil.rs` with its unit tests. `cargo test -p rustmath-curves --lib weil`.

Step 3. `nodal_biprojective` (form, fibres, singular, local, certify) and the small-curve tests.
`cargo test -p rustmath-curves --test biprojective_small_curves`.

Step 4. Vendored data, `tests/m23_zeta.rs`, the example.
`cargo test -p rustmath-curves --test m23_zeta` (fast set, debug).

Step 5. Opt-in runs:
`cargo test --release -p rustmath-curves --test m23_zeta -- --ignored full_` and
`M23ISGALOIS=/home/user/external/m23isgalois cargo test --release -p rustmath-curves --test m23_zeta -- --ignored env_`;
`cargo run --release -p rustmath-curves --example m23_zeta -- 5 127 4` and the same with
`--cover L`. Paste the outputs and times into the report.

Step 6. `cargo test -p rustmath-curves` (whole crate, no regression), `cargo clippy -p rustmath-curves`,
`cargo fmt`. Commit 2. Delete `target/release` when done.

Acceptance: every row of both tables equal (reference rule where the reference used it), the three
derived checks at every certified row, 45, 52, 57 at 53, statuses as in Section 7, and the two
differences from the reference (p = 5 for X decided; L at 31 corrected) reported with their
evidence.

## 12. Follow-ups (not in this unit)

- **F-1. p = 3.** Both plane models are hopeless mod 3 (`F mod 3 = (V-1)^23 T (T-1)^3`; rescalings
  `V = c + 3^a W` up to `a = 4` stay degenerate, `work/p3b.gp`). Needs a 3-adically good model:
  the canonical model of WP-A A0 reduced mod 3, or a semistable model computation. No claim about
  good or bad reduction at 3 is made.
- **F-2. Le Lay's (8,23) model `Fint_coefficients_Z.json`.** Not used for item 1: it has 84 nodes
  but `p_a = 154`, so its other 66 units of delta sit in two non-nodal points at
  `(T, V) = (+-sqrt(-23), infinity)` (`disc_T` has degree 184 of 322, `work/f8.gp`). A count on it is
  never Nodal, and its fibres are octics (about 3 times the cost). The A2 classification removes
  the D flags on the (4,23) model instead. It stays useful as an independent cross-check at small
  k; not needed for acceptance.
- **F-3. A_k and ordinary points over residue fields of degree > 1, and the two-point fibre with a
  non-nodal point.** Implemented in the design (4.4 works over any `kappa`); every non-nodal point
  met so far has `kappa = F_p`, so tests cover only that case. Add a constructed test over `F_p^2`.
- **F-4. Faster quartic root count** (resolvent cubic and square classes, two to three
  exponentiations in `GF(q)` instead of about 10 polynomial products). Only if p > 127 is wanted.
- **F-5. p = 23** (wild ramification of t): run the certificate; expect bad reduction of the model.
- **F-6. CC0 route for the L covers** through Le Lay's canonical models (counting on a curve in
  P^3), so that the L table runs without `M23ISGALOIS`.
- **F-7. Q versus Q^rev** (extension item 4): decided by WP-A, not here.
- **F-8. Report the L-31 correction to the owner** (johnjanik/M23 `data/zeta_L.json` and the paper's
  table), and the good reduction at 5. For WP-A: good reduction of X at 5 makes `J[2]` unramified
  at 5 (Neron-Ogg-Shafarevich, 5 != 2), so the A8 field `K_W` is unramified at 5. That consequence
  is not checked here.

## 13. Decisions for the reviewer

- **D1 (status of the delta certificate).** The project rule says a count on a model with a
  non-nodal singular point is not Decided. That rule protects against the reference's unibranch
  assumption. Theorem 4.5 with the exact classification of 4.4 assumes nothing. Proposal: report
  `Decided(Certificate::DeltaSum)`, labelled separately from `Nodal`, behind
  `CountOptions::allow_delta_certificate` (default true for the M23 driver, false in the
  library default). If rejected: the same computation reports `Unresolved` with that reason, and
  the D rows, p = 5, 107 and the L rows at 31, 79, 103 stay Unresolved.
- **D2 (models).** Vendor only Le Lay's CC0 G; derive G7 and G13 at run time; read F and calF only
  from `M23ISGALOIS`.
- **D3 (the L-31 row).** Tests assert the reference value in reference mode and the corrected value
  in exact mode; the README of the data directory records the discrepancy.
- **D4 (two crates).** The kernel goes into `rustmath-finitefields` (commit 1), the rest into
  `rustmath-curves` (commit 2), on one branch.

## Appendix: prototype and scripts

Base directory `/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/m23/design/WPF/`
(target directory deleted; rebuild with `cargo build --release` inside `proto/`, about 31 s):

- `proto/src/gfq.rs` (kernel), `roots.rs` (root count, both paths), `count.rs` (fibre loop with
  orbits), `cert.rs` (discriminant, residue fields, nodes, two-point fibres), `local.rs` (A_k and
  ordinary points over F_p), `lpoly.rs` (Newton, exact Weil test, mod 2, mod 23), `main.rs`
  (commands `x`, `l`, `tfib`, `bench`, `oldff`, `small`; `EXACT=1` switches to exact branch counts).
- `x_table_F.txt`, `l_table.txt`, `x131_G.txt`, `tables.md`, `table_time.txt`: the raw outputs of
  Section 7 and 8; `scripts/tables.py` formats them; `scripts/debug_timing.sh` gives Section 8.3.
- `gp/small_curves.gp`, `gp/branch.gp`: the PARI/GP references of Section 9.4.
- `work/*.gp`: the PARI/GP checks quoted in Sections 3, 4 and 12 (`mob.gp`, `hj.gp`: the Moebius
  identity; `cert.gp`, `certL.gp`: the certificate; `local.gp`, `localL.gp`, `tc.gp`: local types;
  `disc.gp`, `f8.gp`: discriminants; `p3.gp`, `p3b.gp`: p = 3; `n7.gp`, `n13.gp`: G7, G13;
  `brute.gp`, `h62chk.gp`: cross-checks). `work/F.gp`, `work/coeffs*.json` hold F and calF parsed
  from m23isgalois; they are scratch only and must not be copied into RustMath.
