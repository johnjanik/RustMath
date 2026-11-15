# Test data for the M23 curve (WP-F)

Used by `rustmath-curves/tests/m23_zeta.rs` and `rustmath-curves/examples/m23_zeta.rs`. The brief
is `docs/workstreams/m23_golay/HANDOFF.md`, Section 3.

## Files

| File | Source | Licence | sha256 |
|---|---|---|---|
| `lelay_optimal_23_4_Z.json` | Le Lay, `research-reflexivity/frontier-math-m23-cover-investigation`, `data/optimal_23_4_Z.json`, commit `3f6ab4b` | CC0-1.0 (`data/LICENSE` of that repository; the paper and code there are CC BY-NC-SA and are not used) | `cfb4185b2800744d524d87b3f08f5459cd38d42afb3787c11fa1e2aa3f25ee84` (the value in its `SOURCE_NOTES.md`) |
| `zeta_X.json` | `johnjanik/M23`, `data/zeta_X.json`, commit `d05475b` | the owner's repository, copied as the brief allows | `a854d3169c4dc642c68ce6c369684752f2747584f328e157ebae6b42d90e19f4` |
| `zeta_L.json` | `johnjanik/M23`, `data/zeta_L.json`, commit `d05475b` | as above | `ed639be2b8e7a69b9e0a8554fad6ca81df89705db162b2f2e18fcf2364c0da7f` |
| `h62.json` | `johnjanik/M23`, `data/h62.json`, commit `d05475b` | as above | `f7ebdf75335d9f05c11e351663448e6fc902845592cc30c242b31f47ae9b4441` |

The files are copied unchanged. Le Lay's file is attributed to Knabble, Inc. (dba Reflexivity),
who waived their rights under CC0 1.0.

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

- `zeta_X.json`, p = 5: the reference counted a tacnode once and got `8, 36, 170, 560`, which is
  not a Weil polynomial. The tacnode has two conjugate branches. Counting them gives
  `7, 37, 169, 561` (and `3197, 15709` at k = 5, 6, the values predicted by P), a Weil polynomial
  that passes the mod 23 and mod 2 checks. Status: non-nodal singularities classified exactly,
  delta-sum certified (Unresolved under the project rule).
- `zeta_L.json`, p = 31, s0 = 12: the published row `44, 1014, 30266, 923726` is wrong. The model
  has 57 nodes, an ordinary triple point (one rational and two conjugate tangents) and a tacnode
  with conjugate branches. The reference counted both once. Its P is a Weil polynomial but fails
  the mod 23 check (`P(1) = 15`, `P(31) = 7` mod 23), and the count at k = 5 (`28604634`) differs
  from the value its P predicts (`28598074`). The exact branch counts give `43, 1017, 30265,
  923729`, which pass all checks, and the predicted `N_5 = 28604633` is the counted value.
- `zeta_L.json`, p = 11, s0 = 2: the prime `(11, s - 2)` of L is ramified (e = 2, f = 1). The
  refusal is right regardless: the discriminant of the model vanishes identically there. The same
  holds at `(5, s - 3)`, which is unramified of degree one.
