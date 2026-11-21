# Test data for the M23 curve (WP-F)

Used by `rustmath-curves/tests/m23_zeta.rs` and `rustmath-curves/examples/m23_zeta.rs`. The brief
is `docs/workstreams/m23_golay/HANDOFF.md`, Section 3.

## Files

| File | Source | Licence | sha256 |
|---|---|---|---|
| `lelay_optimal_23_4_Z.json` | Le Lay, `research-reflexivity/frontier-math-m23-cover-investigation`, `data/optimal_23_4_Z.json`, commit `3f6ab4b` | CC0-1.0 (`data/LICENSE` of that repository; the paper and code there are CC BY-NC-SA and are not used) | `cfb4185b2800744d524d87b3f08f5459cd38d42afb3787c11fa1e2aa3f25ee84` (the value in its `SOURCE_NOTES.md`) |
| `zeta_X.json` | `johnjanik/M23`, `data/zeta_X.json`, commit `2d11038` (main) | the owner's repository, copied as the brief allows | `77fdf78ee48071b84f7ab20c2fb475eb18db89a1d40d7712989572be9a039e56` |
| `zeta_L.json` | `johnjanik/M23`, `data/zeta_L.json`, commit `2d11038` (main) | as above | `d2bae9ef107ca6cfd9f507551583e07521aa7f0b3bd42f76dff2f8bb069b19d8` |
| `h62.json` | `johnjanik/M23`, `data/h62.json`, commit `d05475b` (unchanged at `2d11038`) | as above | `f7ebdf75335d9f05c11e351663448e6fc902845592cc30c242b31f47ae9b4441` |
| `a8_frobenius_predictions.json` | `johnjanik/M23`, `data/a8_frobenius_predictions.json`, commit `e0d25b8` (branch `claude/blissful-knuth-qla7ym`), written by `scripts/a8_frobenius_predictions.py` there | as above | `505b25ea9ff292a8c9047dd5616c29ecfb15e6987bc5656e9ddf077c9c436162` |

`a8_frobenius_predictions.json` holds the owner's 14 classes of `GL_4(2) = A_8` and, for the 29
X rows of REQUESTS 1.2, `#Jac(F_p)`, its 2-adic valuation, the order on `Jac[2]` where his rule
decides it, and the allowed cycle types. It is the reference of `c5_matches_the_owners_predictions`
(C-5). At `e0d25b8`, `zeta_X.json` and `zeta_L.json` are byte-identical to the vendored files of
`2d11038`.

The files are copied unchanged. `zeta_X.json` and `zeta_L.json` were first vendored from
`d05475b` (sha256 `a854d316...90e19f4` and `ed639be2...c0da7f`). After WP-F the owner corrected
the two rows that RustMath found wrong (follow-up F-8) and added a certificate to every row; the
files here are those of `2d11038`. What changed:

- `zeta_X.json`: the row at p = 5 is now `ok`, with counts `7, 37, 169, 561` and
  `P = [1, 1, 6, 20, 16, 100, 150, 125, 625]` (it was skipped, with `8, 36, 170, 560` and the
  `reason` "not a Weil polynomial"; that row no longer has a `reason` field). Every row gains
  `certificate { certified, singular_points, delta }`, the singular points found by the owner's
  classifier (`m23golay/curve.py` at `2d11038`, by blow-ups); the `model` field is dropped. The
  other counts are unchanged.
- `zeta_L.json`: the row at (31, 12) is corrected to `43, 1017, 30265, 923729`, with P, `P mod 2`
  and Q (it was `44, 1014, 30266, 923726`). Rows gain `certificate` and the mod 23 fields; (5, 3)
  and (11, 2) are `refused`, and new refused rows (23, 5) and (23, 15) appear. `check_53` is
  unchanged.

`rustmath_wpf_tables.json` and `rustmath_crosscheck.json` of `2d11038` are not vendored: the
first is RustMath's own output (no test needs it), and the three values of the second are quoted
in `full_x_owner_recount_83_97_107` with their source. Le Lay's file is attributed to Knabble,
Inc. (dba Reflexivity), who waived their rights under CC0 1.0.

Not here: HJLPPZ's `polynomial_F.m` and `polynomial_calF.m` (`shaowuz/m23isgalois`). That repository
has no licence, so the tests read them at run time from the directory named by the environment
variable `M23ISGALOIS` (the `env_` tests).

## The model G and HJLPPZ's F

`coefficients_T_then_W[i][j]` is the coefficient of `T^i W^j` of a primitive form `G(T, W)` of
bidegree (4,23). It is HJLPPZ's `F(T, V)` in another coordinate on the V-line:

    F(T, (967 W - 3)/(81 - 2267 W)) (81 - 2267 W)^23 = c G(T, W),   c = 2^23 3^23 131^23 > 0.

This identity was checked exactly in PARI/GP (WP-F design, Section 3, and its review) and is
checked again by the opt-in test `env_f_equals_g_by_mobius`. The map is an automorphism of the
V-line over Q, so G and F define the same curve in `P^1 x P^1`, with the same function T. Its
determinant is `967 * 81 - 3 * 2267 = 71526 = 2 * 3 * 7 * 13 * 131`. For p not dividing it the
two reductions are isomorphic models.

The tests also use two further integral models of the same curve, computed from G at run time:

- `G7 = G(T, 7V + 3)`, made primitive: nodal with 62 nodes at p = 7, where G degenerates;
- `G13 = G(T, (11V + 130)/(V + 13)) (V + 13)^23`, made primitive: at p = 13 it has 61 nodes and
  one ordinary cusp, like F (G has a fibre component at 13).

## Notes on the reference tables

- `zeta_X.json`, p = 5: the row at `d05475b` counted a tacnode once and got `8, 36, 170, 560`,
  which is not a Weil polynomial. The tacnode has two conjugate branches. Counting them gives
  `7, 37, 169, 561` (and `3197, 15709` at k = 5, 6, the values predicted by P), a Weil polynomial
  that passes the mod 23 and mod 2 checks. The owner corrected the row at `2d11038`. Status:
  Decided, delta-sum certificate (Lemma 9.1 of the owner's paper): X has good reduction at 5.
  The old values stay in the tests (`x_at_5_tacnode_with_conjugate_branches`) as what the
  assumed-unibranch rule gives.
- `zeta_L.json`, p = 31, s0 = 12: the row at `d05475b`, `44, 1014, 30266, 923726`, was wrong. The
  model has 57 nodes, an ordinary triple point (one rational and two conjugate tangents) and a
  tacnode with conjugate branches. That row counted both once. Its P is a Weil polynomial but
  fails the mod 23 check (`P(1) = 15`, `P(31) = 7` mod 23), and the count at k = 5 (`28604634`)
  differs from the value its P predicts (`28598074`). The exact branch counts give `43, 1017,
  30265, 923729`, which pass all checks, and the predicted `N_5 = 28604633` is the counted value.
  The owner corrected the row at `2d11038`. Status: Decided, delta-sum certificate (Lemma 9.1).
  The old values stay in the tests (`env_l_table`, `env_l31_k5`) as what the assumed-unibranch
  rule gives.
- `zeta_L.json`, p = 11, s0 = 2: the prime `(11, s - 2)` of L is ramified (e = 2, f = 1). The
  refusal is right regardless: the discriminant of the model vanishes identically there. The same
  holds at `(5, s - 3)`, which is unramified of degree one, and at `(23, s - 5)` and
  `(23, s - 15)`. The owner's reasons differ ("the degree in V drops mod p" at 5, "the
  characteristic divides the degree in V" at 23); RustMath refuses all four with
  `DiscriminantVanishes`.
