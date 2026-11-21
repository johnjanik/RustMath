# Test data for the Frobenius module Jac[2] (C-5, design C57)

Used by `rustmath-curves/src/weil.rs` and `src/weil/two_torsion.rs` (unit tests), `tests/two_torsion.rs` and
`tests/m23_zeta.rs` (`c5_bounds_equal_pari`, `c5_order_text_says_what_was_tried`). The requests are
`docs/workstreams/m23_golay/REQUESTS.md` 1.2 and C-5. Every file was made by the script next to it,
with PARI/GP 2.15.4 or GAP 4.12.1, in this directory. The scripts and outputs are those of the
design's prototype (scratch work, not kept: `docs/workstreams/m23_golay/c57/README.md`), rerun
for the implementation: every output was byte-identical, except `lr_gap.txt` (extended from
`|mu| = |nu|` to every split, and equal to the prototype's 187 lines on the equal splits) and
`gl42_classes.txt` (one line per class in a parseable format; the values are those of the
prototype's run and of the owner's `classes`).

| File | Made by | What | sha256 |
|---|---|---|---|
| `hyperelliptic_jac2.txt` | `gp -q hyper_truth.gp > hyperelliptic_jac2.txt` (seed `20261120`) | 3098 curves `y^2 = f(x)` over `F_p`, `3 <= p <= 47`, genus 1 to 4: `p g [P] [cycle type of f mod p] order [dim ker (F - 1)^j]`. `P` by `hyperellcharpoly`; the exact `Jac[2]` is the space of even subsets of the branch points modulo the whole set, with Frobenius acting by the cycle type | `c262604c53a55d4e941276e03d166b3dd5aef6a237bc0a7a1090b00d57442d28` |
| `lr_gap.txt` | `gap -q lr.g > lr_gap.txt` | every nonzero Littlewood-Richardson coefficient `c^lam_(mu, nu)` with `\|mu\|, \|nu\| >= 1`, `\|lam\| <= 8` (1197 lines), from the character tables of the symmetric groups | `3179942b024c2bde0efb8dc6863e9be7e6a00dabf70f87aa05ddc0e5ec8e11db` |
| `gl42_classes.txt` | `gap -q gl42_classes.g > gl42_classes.txt` | the 14 classes of `GL_4(2)`: characteristic polynomial, order, class size, `dim ker (g - 1)^j`, cycle types on the 15 points, the 15 hyperplanes and 8 letters (through `GL(4,2) = A_8`) | `3390e8e432bebcb8120f42afee8b37c404edb2d848d92f130541f02c2371f6f4` |
| `sp4_classes.txt`, `sp6_classes.txt`, `sp8_classes.txt` | `gap -q sp_classes.g` (from the design's review 1) | the invariant form and a representative of each conjugacy class of `Sp(4, 2)`, `Sp(6, 2)`, `Sp(8, 2)` (11, 30, 81 classes; the class sizes add up to the group orders) | `82d69e0d...`, `afce494a...`, `cdcb82df...` |
| `jac2_v2_pari.txt` | `gp -q v2.gp > jac2_v2_pari.txt` (reads `rows.gp`) | for the 54 counted rows of X and L (`rows.gp`: `a_1..a_4` of RustMath's tables after F-9), `N_max` and `v_2(Res(P, x^k - 1)) / v_2(Res(P, x^k + 1))` for every `k \| N_max` (`polresultant`) | `c3b8863e92603be84dd22b5e56d64041fc52d390978ae28b482b3c4625c1eb81` |
| `keyvals_pari.txt` | `gp -q keyvals.gp > keyvals_pari.txt` | `P(1)`, `P(-1)`, `Res(P, x^4 + 1)`, `Res(P, x^5 - 1)` at X 17, 59, 107, 109 and L (41, 8), (89, 69) | `dfa3a2fbf2c3c3192de022d448040ddfe6129122fc8822f16dd1fed64546cc4a` |
| `elliptic_pari.txt` | `gp -q elliptic.gp > elliptic_pari.txt` | `#E(F_p)`, `a_p`, `P(-1)` and the factorization of the cubic for the genus-1 examples of `degenerate_genus_0_and_1` | `cdf50bc838d985dd60159134b010b3f4ecf4cb6a010c42ecda0ed6b11f19b83e` |
| `large_nmax_pari.txt` | `gp -q large_nmax.gp 2>/dev/null > large_nmax_pari.txt` (C57 review 1, B1) | a genus-12 `P` with the functional equation and `P mod 2 = (x + 1)^2 f f*`, `f = x^11 + x^2 + 1` (`N_max = 4094`), over `q = 3` and `10^9 + 7`: `v_2(Res(P, x^k - 1)) / v_2(Res(P, x^k + 1))` for every `k \| 4094` and the two resultants modulo `10^30` at `k = 1, 23, 89` (`norm(Mod(x, P)^k -+ 1)`); for `large_n_max_bounds_are_cheap` and `base_change_of_genus_12` in `src/weil.rs` | `7c128ea4f1ffe6f394adca60b37d3e530fd64ef5aff6dbb3a8e4677d2504162c` |
| `skipped_ec_pari.txt` | `gp -q skipped_ec.gp > skipped_ec_pari.txt` (C57 review 2, n1) | the elliptic Weil polynomial `P = T^2 - 2T + q`, `q = 21 * 2^128 + 1` (the least prime `k 2^128 + 1`): `v_2` of `Res(P, x^k - 1)` and `Res(P, x^k + 1)` for `k = 1, 2` (128, 2, 130, 2) and the 2-adic valuations of the elementary divisors of multiplication by `x^k -+ 1` on `Z[x]/(P)` (`matsnf`): the untwisted bounds have a divisor of valuation 128 or more, so the analysis skips them; for `c5_order_text_says_what_was_tried` in `tests/m23_zeta.rs` | `f3ec0faf04c239acf67bd2ecc04cd41ddd7c57813ce93e4fb288024e59b39a15` |

The full sha256 of the three `sp*_classes.txt` files: `82d69e0ddab1de40203657d344dddda8eb70612c0110ba1bc611f19ff42721a1`,
`afce494ac5b01fb2c901dfb16959e06d1547e5a57ede90d7a6ceb9a8201b574a`,
`cdcb82dfa689882841db2933866417a0fa1081014d835d6f63fe757b24bf2d9b`.

`rows.gp` copies `a_1..a_4` from `docs/workstreams/m23_golay/wpf/followup_f9/F9_impl.md`
Section 3 (RustMath at `88f9174`); the rows with p <= 79 of X and the L rows at 7, 19, 31, 41, 43
equal the owner's `zeta_X.json` and `zeta_L.json` (`tests/data/m23/`).

Licence: these files are computed by RustMath's own scripts with PARI/GP and GAP; no third-party
data is copied.
