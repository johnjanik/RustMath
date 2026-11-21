# WP-F implementation report

Date: 2026-11-15. Worktree `/home/user/RustMath-wpf`, branch `claude/m23-wpf`, HEAD `aa485d0`
(base `origin/main` `0d277f0` plus the Section 2 docs commit). Design:
`scratchpad/m23/design/WPF.md` (no Revision section), approved by
`scratchpad/m23/design/WPF.review1.md` with D1 rejected and the required changes of its Section 3.
All work is uncommitted in the worktree (the task says: no commit, no push). Nothing outside the
worktree was touched except this scratch directory. The refreshed Section 2 under
`scratchpad/m23wt/` does not exist; the worktree's HANDOFF.md (Section 2 checked at `aa485d0`) was
used. The worktree had no earlier uncommitted changes.

## 0. Summary

- Both reference tables reproduce exactly. All 18 rows of `zeta_X.json` at k = 1..4 (opt-in
  release test `full_x_table_k4`, 83 s), the 7 rows of `zeta_L.json` (5 and 11 refused as in the reference; 7, 19, 41, 43 equal; 31
  equal under the reference rule and corrected under the exact rule; `env_l_table`), and the anchor 45, 52, 57 at 53 (`env_anchor_53`).
- Statuses follow review decision D1 (rejected): `Decided` only for the nodal certificate.
  The D rows 13, 31, 41, 73 (one ordinary cusp), p = 5 (a tacnode), 107 (a tacnode) and the L rows
  (31, 12), (79, 15), (103, 93) are `Unresolved: non-nodal singularities classified exactly,
  delta-sum certified`, with exact branch counts. Follow-up F-9 lets the owner promote them.
- Two differences from the reference, both with evidence: X at p = 5 has exact counts
  `7, 37, 169, 561` (Weil, mod 23 and mod 2 hold, and `N_5, N_6 = 3197, 15709` are predicted by
  P); the L row at (31, s - 12) is `43, 1017, 30265, 923729`, not the published
  `44, 1014, 30266, 923726` (the published P fails the mod 23 check, `(15, 7)`, and its
  prediction of `N_5` fails).
- New rows: X at 83..131 and the L covers at every degree-one prime up to 127 (Section 3).
  Release run times: X 5..131 at k <= 4 in 12 min 42 s, L 5..127 in 8 min 27 s (4 threads, shared).
  Every counted row is Weil (exact), passes mod 23, and was rechecked in PARI/GP.
- The equation F is not copied. The CC0 model G of Le Lay is vendored; `F o M = c G` is checked
  exactly at test time from `$M23ISGALOIS` (`env_f_equals_g_by_mobius`).
- Debug test suites: `rustmath-finitefields --test small_field` 3.2 s, `rustmath-curves --test
  biprojective_small_curves` 1.0 s, `--test m23_zeta` 7.3 s (fast set). The whole `rustmath-curves` suite passes.

## 1. What changed, per item

### 1.1 Kernel (`rustmath-finitefields`, design 4.2, 6.1)

- `rustmath-finitefields/src/small_field.rs` (new, 818 lines): `SmallFq<K>` (:55), `GF(p^K)` on
  `[u64; K]`, `p < 2^16` proved prime, `K <= 8`, `p^K < 2^63` (:138 `new`: Conway polynomial if
  tabulated, else the least irreducible in counter order :106; :162 `with_modulus`, checks
  irreducibility with `fp_factor`). Barrett reduction (:246), lazy accumulation, Frobenius by a
  precomputed matrix (:376), norm and inverse by the conjugates (:414, :420), `is_square` by the
  norm (:437), `code`/`from_code` (:443, :453). `count_distinct_roots` (:649): degree <= 4 by
  fixed-size arithmetic mod f with `x^p` by square-and-multiply and `x^q` by the semilinear
  Frobenius (:551), then `gcd(f, x^q - x)` (:585); `count_distinct_roots_any_degree` (:725) the
  vector path for any degree (used for the degree 23 fibres of the T-fibration).
- `rustmath-finitefields/src/lib.rs:34` (module), `:59` (re-exports `SmallFq`, `SmallFqElt`,
  `count_distinct_roots`).
- Tests: unit tests in `small_field.rs` (refusals: p = 1, 9, 65537, `65521^4`, K = 9, reducible and
  non-monic moduli; Frobenius order; roots of `x^q - x` factors) and
  `rustmath-finitefields/tests/small_field.rs` (new): arithmetic against `FiniteField::with_modulus`
  for K = 1..4 and p in {5, 7, 31, 127, 65521} (K <= 3 for 65521) on 200 seeded elements each
  (products, sums, differences, Frobenius, norm, `is_square`, inverses), plus K = 8 at p = 3;
  root counts against `fp_factor::factor_with_multiplicity` (K = 1), `poly_factor::roots` over
  `FiniteField` (K = 2, 3, 4) and brute force (q <= 400), on polynomials of degree 1..6 with
  repeated roots; both paths compared on every case; subfield polynomials `x^5 - x`, `x^25 - x`
  over `GF(25)`, `GF(5^3)`, `GF(5^4)`; degree 23 polynomials (vector path) against
  `poly_factor::roots` and brute force.

### 1.2 `rustmath-curves/src/weil.rs` (new, design 4.7, 4.8, 4.9, 4.10, 6.2)

`FrobeniusPolynomial` (:88) with `from_point_counts` (:112, Newton's identities with an
integrality check and the functional equation), `from_counts_u64`, `from_coefficients` (:163,
checks the functional equation), `coefficients`, `l_polynomial` (MAGMA `LPolynomial`, ascending),
`point_counts` (:209, counts predicted by P for any k), `predicts` (:246, the k > g check of
review 3.3), `real_weil_polynomial` (:251), `is_weil` (:281, exact: Sturm on each squarefree
factor of R and of `S(y) = E(y)^2 - y O(y)^2` on `(-1, 4p]`), `eval_mod`, `reduce_mod`,
`factor_mod` (`fp_factor`), `reciprocal_halves_mod2` (:347, all Q with `Q Q^rev = P mod 2`),
`two_torsion_frobenius_order` (:387). `WeilError` (:38: `InvalidInput`, `NonIntegral`,
`FunctionalEquation`), `TwoTorsionOrder` (:75: `DeterminedByP(n)`, `Unresolved {
semisimple_order, divides }`), `f2_poly_string`. Unit tests: Newton round trip, Weil, mod 23 and
mod 2 on the 18 X rows; Q at 7, 11, 41, 59 against the table; 2-torsion order 15 at 11, 31, 79
and the bound at 7; Weil edge cases at p in {5, 7, 11, 13, 101} (`(T^2 + p)^4`, `(T^2 - p)^4` with
double roots on the boundary, `T^2 + aT + p` with `a = floor(2 sqrt p)` and `a + 1`, `T^2 + pT + p`,
`R = x^2 + 1`, `R = x^2 - 1`, `R = x^2 - (4p +- 1)`); the reference row at 5 not Weil and the exact
row Weil; error cases; predictions beyond g on `y^2 = x^3 + 3x + 5` at 7 (PARI `ellap = 1`).

### 1.3 `rustmath-curves/src/nodal_biprojective.rs` and `nodal_biprojective/` (new, design 4.1 to 4.6, 6.3)

- `nodal_biprojective.rs`: module doc (method, theorem, genus as a trusted input, papers),
  `Degeneracy` (:69), `CountError` (:81, with `GenusInconsistent` of review 3.2), `check_prime`.
- `form.rs`: `BihomogeneousForm` (:13; `new`, `from_decimal_rows`, `bidegree`,
  `arithmetic_genus`, `content`, `primitive_part`, `transpose`, `mobius_y` :112,
  `equals_up_to_scalar`, `reduce`), `FormModP` (:197; `new`, `transpose`, `reverse_x`,
  `reverse_y`).
- `fibres.rs`: `Fibration` (:14), `plane_point_count` (:153): the fibre loop of `curve.py`
  (`fibre = "V"` for `FibresInX`), chunks of `2^14` codes on rayon, Frobenius-orbit weighting
  (:73), lazy evaluation of the fibre coefficients (:42), a vanishing fibre refused.
- `residue.rs` (new file, not in the design's list; it holds the residue-field helper the design
  places in `singular.rs`): the private trait `ResField` (:66) with `PolyField` (:196,
  `F_p[Y]/(g)` on `fp_factor` polynomials; norm by the resultant) and `QuadExt` (:291,
  `K[X]/(X^2 + g1 X + g0)` for two conjugate points in one fibre), Tonelli-Shanks `sqrt` (the
  nonsquare found by enumeration, which decides nothing), `charpoly` (the characteristic
  polynomial over F_p of the X-coordinate), polynomial helpers and a distinct-degree
  `factor_degrees` (:489) for tangent cones over any residue field.
- `singular.rs`: `YPlace`, `SingularPoint` (:35, with `residue_degree`, `y_place`,
  `x_at_infinity`, `coordinates`, `x_charpoly`, `local`, `delta`, `branch_orbits`;
  `rational_branches`, `describe`), `SingularLocus` (:102), the formal discriminant for fibre
  degree 2, 3, 4 (:141, :197), the per-place analysis (:305) with `deg G <= 3` (review 3.1: degree
  3 is checked to be `(X - tau)^3`, otherwise `Unclassified`, never an error with a false
  message), the split and irreducible two-point fibres, the chart at `X = infinity`, and
  `singular_locus` (:420) with the place `Y = infinity` when `deg D` drops.
- `local.rs`: `LocalType` (:27: `Node`, `A(k)`, `OrdinaryMultiple(m)`, `Unclassified`) and the
  classifier `classify` (:200), generic over the residue field: Hessian rank 2 node, rank 1
  `A_k` by Newton on the polar curve (precision 16 doubling to 128), rank 0 ordinary m-fold point
  by the tangent cone.
- `certify.rs`: `Certificate::Nodal { nodes, genus_hypothesis }` (:39, review 3.2), `Unresolved`
  (:51: `DeltaSumCertified { delta_sum, genus_hypothesis, non_nodal }`, `Unclassified`,
  `DeltaMismatch { found, expected, found_is_lower_bound }`), `ReductionStatus` (:74, `label`),
  `certify` (:133), `BranchRule` (:200), `CountOptions` (:210; no `allow_delta_certificate`,
  review D1), `NormalizationCounts` (:236; `frobenius_polynomial`, `predicted_counts_agree` :265),
  the branch corrections (:277), and `normalization_point_counts` (:308).
- `rustmath-curves/src/lib.rs:65-66` (two module lines, with a comment), `rustmath-curves/Cargo.toml`
  (`serde_json = "1"` under `[dev-dependencies]`), `Cargo.lock` (one line).

### 1.4 Tests, data and the driver

- `rustmath-curves/tests/biprojective_small_curves.rs` (new): 10 tests, PARI/GP references quoted
  per case (Section 2).
- `rustmath-curves/tests/m23_zeta.rs` (new): 8 fast tests, 5 `env_` tests, 3 `full_` tests
  (Section 2).
- `rustmath-curves/tests/common/m23.rs`, `tests/common/mod.rs` (new): loaders (G, G7, G13; F and
  calF from `$M23ISGALOIS` with parsers equivalent to `m23golay/data.py`; reduction of calF at
  `(p, s - s0)`; degree-one primes of L; the JSON tables; h62), the model selection of review 3.6
  (`status_rank`, `select_model`, `x_row`: Nodal > DeltaSumCertified > Unclassified >
  DeltaMismatch > refusal; counts of two counted models compared at `k <= min(kmax, 2)` under the
  exact rule), derived checks and row formatting.
- `rustmath-curves/tests/data/m23/` (new): `lelay_optimal_23_4_Z.json` (CC0, Le Lay commit
  `3f6ab4b`, sha256 `cfb4185b...`), `zeta_X.json`, `zeta_L.json`, `h62.json` (johnjanik/M23 at
  `d05475b`), `README.md` (sources, licences, checksums, the Moebius identity, G7 and G13, the
  notes on X at 5, L at (31, 12) with the k = 5 evidence, and the ramified prime (11, s - 2)).
- `rustmath-curves/examples/m23_zeta.rs` (new): `cargo run --release -p rustmath-curves --example
  m23_zeta -- <pmin> <pmax> <kmax> [--cover L] [--model F] [--reference-rule] [--threads N]`.

## 2. Tests and reference sources

| Test | What | Reference | Debug time |
|---|---|---|---|
| finitefields `small_field` (4 tests) and 3 unit tests | kernel vs `FiniteField`, `fp_factor`, `poly_factor::roots`, brute force | the canonical stack of the crate; brute force | 3.2 s |
| curves `weil` unit tests (7) | Newton, exact Weil test, mod 2, mod 23, 2-torsion | `zeta_X.json`; PARI `ellap` | 0.02 s |
| `biprojective_small_curves`: `smooth_genus_3` | curve A, genus 3, p = 11; `GenusInconsistent` with genus 2 | PARI `hyperellcharpoly` | |
| `nodes_with_rational_and_conjugate_branches` | B, C, C2 (p = 11), G, H (p = 13) | PARI `hyperellcharpoly`, `branch.gp` | |
| `cusp_and_tacnodes` | D (A2); E, F, I, I2 (A3, conjugate and rational); the reference rule equal for A2 and wrong for A3 | PARI | |
| `tacnode_tangent_to_the_fibre` | review 3.1 curve, `deg G = 3`; N_1 = 8, `P = T^2 - 4T + 11` | PARI `tangent.gp` | |
| `points_of_residue_degree_two` | J (node of degree 2, conjugate branches), J' (rational), J2 (two conjugate nodes in one rational fibre: `QuadExt`), K1 (two conjugate cusps in one rational fibre) | PARI `small_curves4.gp` (new) | |
| `split_fibre_with_a_cusp_and_a_node` | K2, p = 13: a cusp and a node in one fibre (Tonelli-Shanks split) | PARI `small_curves4.gp` | |
| `charts_at_infinity` | B, C, D, E, F after `T -> 1/T`, `V -> 1/V`, both | invariance | |
| `elliptic_curves_with_singular_points_at_infinity` | E1 (A2 at (inf, inf)) at 7, 11, 13; E2 (ordinary triple point) at 7, 13, 19; E2 at 11 `DeltaMismatch { found: 4, expected: 3 }` | PARI `ellap` | |
| `refusals`, `weil_polynomial_of_b_from_counts` | fibre component, `D = 0`, p = 2, 9, 65537, k = 9, fibre degree 5, `FieldTooLarge` at p = 65521, k = 4 (plane count and full count) | | |
| whole `biprojective_small_curves` | every case also checks both fibrations and brute force at k = 1, the T-fibration at k = 2, and the predicted counts beyond g | | 1.0 s |
| `m23_zeta` fast: `cc0_models_are_primitive_of_bidegree_4_23`, `reference_polynomials_pass_the_derived_checks` | G, G7, G13 primitive of bidegree (4,23); every ok row of `zeta_X.json` | `zeta_X.json` | |
| `x_at_7_k6_and_11_k5_nodal` | G7 at 7 (k <= 6), G at 11 (k <= 5): table, Nodal 62, predictions at k = 5, 6; order 15 at 11; nodes over the roots of h62 mod 11 | `zeta_X.json`, `h62.json` | |
| `x_at_17_and_19_k3_nodal`, `x_at_13_and_31_one_cusp` | table at k <= 3; one A2 and 61 nodes; the reference rule equal and Unresolved | `zeta_X.json` | |
| `x_at_5_tacnode_with_conjugate_branches` | 60 nodes + A3 conjugate; `7, 37, 169, 561, 3197, 15709`; P, Weil, mod 23, Q; the reference rule gives the table's `8, 36, 170, 560`, not Weil, predictions fail | `zeta_X.json` (row 5), review 1 Section 1 | |
| `x_at_3_and_23_refused`, `both_fibrations_and_brute_force` | refusals; T-fibration = V-fibration at 7, 11 (k = 1, 2); brute force | | |
| whole `m23_zeta` fast set | | | 7.3 s |
| `env_f_equals_g_by_mobius` | `F o M / content = G` exactly, and back | `polynomial_F.m` | release |
| `env_x_from_f_k3` | X rows 7..79 at k <= 3 from F equal the table | `zeta_X.json` | release |
| `env_l_table` | the 7 L rows at k = 4 (refusals, values, L-31 in both rules) | `zeta_L.json` | release |
| `env_anchor_53` | 52, 57, 45 at s0 = 21, 36, 41; sorted = `check_53` | HJLPPZ Prop. 3.14, `zeta_L.json` | release |
| `env_l31_k5` | (31, 12) at k = 5: exact `N_5 = 28604633` predicted; reference `28604634` not | review 1 Section 1 | release |
| `full_x_table_k4` | the 18 X rows at k = 4 with statuses, and p = 5 | `zeta_X.json` | release |
| `full_x_k5_31_41`, `full_both_fibrations_k3` | predictions at k = 5 at 31, 41; both fibrations at k = 3 for p <= 31 | | release |

PARI/GP scripts: the design's `scratchpad/m23/design/WPF/gp/small_curves.gp`, `branch.gp`, the
review's `review1/tangent.gp` (all rerun here with the same output), and the new
`scratchpad/m23/impl/gp/small_curves2.gp`, `small_curves3.gp`, `small_curves4.gp` (curves J, J', J2,
K1, K2).

## 3. Regenerated tables

Run: `target/release/examples/m23_zeta 5 131 4` (X) and `M23ISGALOIS=/home/user/external/m23isgalois
target/release/examples/m23_zeta 5 127 4 --cover L`, release, 4 threads, shared machine. Raw
output: `scratchpad/m23/impl/x_out.txt`, `l_out.txt`; formatted by `scratchpad/m23/impl/tables.py`.
"2-torsion order" is the order of Frobenius on `Jac[2]`: a value when `P mod 2` is squarefree,
"Decided" only on a Decided row ("conditional on the counts" otherwise), and the bounds
otherwise (extension item 5). Q is one of the pair `{Q, Q^rev}` (both are printed in the raw
output). Every row with counts has `P(chi) = P(chi p) = 0 mod 23` and passes the exact Weil test.

Updated 2026-11-20 by REQUESTS C-5 (records in `../../c57/`): the column "2-torsion order" now
gives the result of `two_torsion_frobenius_order` after C-5, read from the JSON files of the C-5
release runs (`../../c57/impl/C57.md` Section 4, which also gives the bounds, the types at the
eigenvalue 1 and the classes on `W`). The rank bounds `v2 #Jac(F_p^k)` of the Jacobian and
`v2 #Jac'(F_p^k)` of its quadratic twist decide more rows than `P mod 2` alone. An open order
gives its candidates (`Unresolved: 7 or 14`). Since F-9 every counted row is Decided
(`../followup_f9/F9_impl.md` Section 3), so every value marked Decided here is Decided for the
curve; the status column of these tables is still that of WP-F, before F-9. The other columns
are unchanged.

Independent checks of the regenerated rows:
- PARI/GP (`scratchpad/m23/impl/gp/check_rows.gp`, rows written by `rows.py` from the raw output):
  for every counted row, P from the counts by Newton's identities in PARI, integral, all roots of
  absolute value `sqrt p` to 50 digits (`polroots` at 200 digits), `P(chi) = P(chi p) = 0 mod
  23`, and the factorization of `P mod 2`. Result: all 54 counted rows (29 of X, 25 of the covers) pass every
  check (`gp_check_all.txt`).
- The prototype of the design (independent code, `design/WPF/tables.md`): the counts of all 53 rows of its tables are equal, and X at 131 equals its
  `x131_G.txt` (`compare_proto.py`). The prototype is a design artefact, so this is a consistency
  check, not an external reference.

### 3.1 X

| p | model | N1 | N2 | N3 | N4 | (a1, a2, a3, a4) | P mod 2 | Q (one of the pair) | Weil | mod 23 | status | 2-torsion order | reference | time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | G | 7 | 37 | 169 | 561 | (1,6,20,16) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | true | (0,0) | Unresolved: DeltaSumCertified (A3 conjugate branches) | Unresolved: 7 or 14 | differs: reference [8, 36, 170, 560] (not a Weil polynomial) | 0.0 |
| 7 | G7 | 11 | 69 | 383 | 2433 | (3,14,46,138) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | true | (0,0) | Decided: Nodal (62 nodes) | Unresolved: 7 or 14 | equal | 0.1 |
| 11 | G | 15 | 133 | 1368 | 14905 | (3,10,33,145) | `(x^4+x+1)(x^4+x^3+1)` | `x^4+x^3+1` | true | (0,0) | Decided: Nodal (62 nodes) | 15 (Decided; P mod 2 squarefree) | equal | 0.2 |
| 13 | G13 | 11 | 161 | 2135 | 29089 | (-3,0,-12,188) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | true | (0,0) | Unresolved: DeltaSumCertified (A2) | Unresolved: 7 or 14 | equal | 0.1 |
| 17 | G | 21 | 285 | 4923 | 82985 | (3,2,0,-130) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | true | (0,0) | Decided: Nodal (62 nodes) | 14 (Decided; v2 #Jac(F_p) = 1) | equal | 0.2 |
| 19 | G | 17 | 405 | 7265 | 130913 | (-3,26,66,74) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | true | (0,0) | Decided: Nodal (62 nodes) | Unresolved: 7 or 14 | equal | 0.2 |
| 23 |  |  |  |  |  |  |  |  |  |  | refused: model degenerate at this prime: the discriminant vanishes identically mod p |  | not in the reference table | 0.0 |
| 29 | G | 35 | 789 | 24773 | 708785 | (5,-14,16,1060) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | true | (0,0) | Decided: Nodal (62 nodes) | Unresolved: 7 or 14 | equal | 0.5 |
| 31 | G | 33 | 993 | 29640 | 926713 | (1,16,-35,875) | `(x^4+x+1)(x^4+x^3+1)` | `x^4+x^3+1` | true | (0,0) | Unresolved: DeltaSumCertified (A2) | 15 (Decided; P mod 2 squarefree) | equal | 0.6 |
| 37 | G | 43 | 1433 | 50881 | 1881609 | (5,44,254,3156) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | true | (0,0) | Decided: Nodal (62 nodes) | Unresolved: 7 or 14 | equal | 0.9 |
| 41 | G | 42 | 1780 | 69408 | 2827728 | (0,49,162,1692) | `(x+1)^4(x^2+x+1)^2` | `x^4+x^3+x+1` | true | (0,0) | Unresolved: DeltaSumCertified (A2) | Unresolved: 6 or 12 | equal | 1.4 |
| 43 | G | 44 | 1996 | 80192 | 3422024 | (0,73,228,3470) | `(x+1)^4(x^2+x+1)^2` | `x^4+x^3+x+1` | true | (0,0) | Decided: Nodal (62 nodes) | 12 (Decided; v2 #Jac(F_p) = 1) | equal | 2.0 |
| 47 | G | 47 | 2349 | 104321 | 4882281 | (-1,70,96,2934) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | true | (0,0) | Decided: Nodal (62 nodes) | Unresolved: 7 or 14 | equal | 2.5 |
| 53 | G | 54 | 2804 | 148704 | 7881808 | (0,-3,-58,-2164) | `(x+1)^4(x^2+x+1)^2` | `x^4+x^3+x+1` | true | (0,0) | Decided: Nodal (62 nodes) | Unresolved: 6 or 12 | equal | 3.7 |
| 59 | G | 72 | 3560 | 205056 | 12120780 | (12,111,648,3991) | `(x^4+x^3+x^2+x+1)^2` | `x^4+x^3+x^2+x+1` | true | (0,0) | Decided: Nodal (62 nodes) | 10 (Decided; v2 #Jac(F_p^5) = 4) | equal | 5.6 |
| 61 | G | 64 | 3740 | 226852 | 13855856 | (2,11,-24,2476) | `(x+1)^4(x^2+x+1)^2` | `x^4+x^3+x+1` | true | (0,0) | Decided: Nodal (62 nodes) | Unresolved: 6 or 12 | equal | 6.5 |
| 67 | G | 80 | 4460 | 302192 | 20163724 | (12,57,584,8759) | `(x^4+x^3+x^2+x+1)^2` | `x^4+x^3+x^2+x+1` | true | (0,0) | Decided: Nodal (62 nodes) | 10 (Decided; v2 #Jac(F_p^5) = 4) | equal | 10.0 |
| 71 | G | 68 | 5048 | 358256 | 25432076 | (-4,11,92,4679) | `(x^4+x^3+x^2+x+1)^2` | `x^4+x^3+x^2+x+1` | true | (0,0) | Decided: Nodal (62 nodes) | 10 (Decided; v2 #Jac(F_p^5) = 4) | equal | 13.4 |
| 73 | G | 83 | 5541 | 390089 | 28393065 | (9,146,1428,12030) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | true | (0,0) | Unresolved: DeltaSumCertified (A2) | 14 (Decided; v2 #Jac(F_p) = 1) | equal | 14.1 |
| 79 | G | 77 | 6369 | 491636 | 38964585 | (-3,68,-663,7335) | `(x^4+x+1)(x^4+x^3+1)` | `x^4+x^3+1` | true | (0,0) | Decided: Nodal (62 nodes) | 15 (Decided; P mod 2 squarefree) | equal | 19.5 |
| 83 | G | 91 | 6885 | 574375 | 47483041 | (7,22,902,12258) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | true | (0,0) | Decided: Nodal (62 nodes) | Unresolved: 7 or 14 | new | 23.8 |
| 89 | G | 104 | 7942 | 707906 | 62766210 | (14,108,1576,22324) | `(x+1)^8` | `x^4+1` | true | (0,0) | Decided: Nodal (62 nodes) | 8 (Decided; v2 #Jac(F_p) = 1) | new | 31.5 |
| 97 | G | 101 | 9385 | 910820 | 88546849 | (3,-8,-651,2563) | `(x^4+x+1)(x^4+x^3+1)` | `x^4+x^3+1` | true | (0,0) | Decided: Nodal (62 nodes) | 15 (Decided; P mod 2 squarefree) | new | 40.8 |
| 101 | G | 114 | 10456 | 1030926 | 104090928 | (12,199,2020,28200) | `(x+1)^4(x^2+x+1)^2` | `x^4+x^3+x+1` | true | (0,0) | Decided: Nodal (62 nodes) | Unresolved: 3 or 6 or 12 | new | 47.6 |
| 103 | G | 96 | 10648 | 1092594 | 112536824 | (-8,51,-282,-2198) | `(x+1)^4(x^2+x+1)^2` | `x^4+x^3+x+1` | true | (0,0) | Decided: Nodal (62 nodes) | 12 (Decided; v2 #Jac(F_p) = 1) | new | 43.5 |
| 107 | G | 116 | 11482 | 1226360 | 131139882 | (8,48,652,19390) | `(x+1)^8` | `x^4+1` | true | (0,0) | Unresolved: DeltaSumCertified (A3 rational branches) | Unresolved: 4 or 8 | new | 73.8 |
| 109 | G | 104 | 11986 | 1294118 | 141216370 | (-6,70,-652,18718) | `(x+1)^8` | `x^4+1` | true | (0,0) | Decided: Nodal (62 nodes) | 8 (Decided; v2 #Jac'(F_p^4) = 6) | new | 103.5 |
| 113 | G | 128 | 13016 | 1444142 | 163058328 | (14,221,2594,29766) | `(x+1)^4(x^2+x+1)^2` | `x^4+x^3+x+1` | true | (0,0) | Decided: Nodal (62 nodes) | 12 (Decided; v2 #Jac(F_p) = 1) | new | 80.8 |
| 127 | G | 127 | 16501 | 2047933 | 260199313 | (-1,186,-336,31116) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | true | (0,0) | Decided: Nodal (62 nodes) | 14 (Decided; v2 #Jac(F_p) = 1) | new | 118.0 |
| 131 | G | 144 | 17064 | 2246976 | 294472176 | (12,23,-672,-12864) | `(x+1)^4(x^2+x+1)^2` | `x^4+x^3+x+1` | true | (0,0) | Decided: Nodal (62 nodes) | Unresolved: 6 or 12 | new | 117.1 |

p = 3: refused by every model (discriminant vanishes identically mod 3; `x_at_3_and_23_refused`).
p = 23: refused likewise.

### 3.2 Covers over L at degree-one primes

| p | s0 | N1 | N2 | N3 | N4 | (a1, a2, a3, a4) | P mod 2 | Q (one of the pair) | Weil | mod 23 | status | 2-torsion order | reference | time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 3 |  |  |  |  |  |  |  |  |  | refused: model degenerate at this prime: the discriminant vanishes identically mod p |  | refused (BadFibre('fibre polynomial vanishes identically')) | 0.0 |
| 7 | 5 | 7 | 45 | 343 | 2545 | (-1,-2,2,38) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | true | (0,0) | Decided: Nodal (62 nodes) | Unresolved: 7 or 14 | equal | 0.1 |
| 11 | 2 |  |  |  |  |  |  |  |  |  | refused: model degenerate at this prime: the discriminant vanishes identically mod p |  | refused (BadFibre('fibre polynomial vanishes identically')) | 0.0 |
| 19 | 3 | 15 | 413 | 6849 | 130169 | (-5,38,-152,650) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | true | (0,0) | Decided: Nodal (62 nodes) | Unresolved: 7 or 14 | equal | 0.1 |
| 23 | 5 |  |  |  |  |  |  |  |  |  | refused: model degenerate at this prime: the discriminant vanishes identically mod p |  | not in the reference table | 0.0 |
| 23 | 15 |  |  |  |  |  |  |  |  |  | refused: model degenerate at this prime: the discriminant vanishes identically mod p |  | not in the reference table | 0.0 |
| 31 | 12 | 43 | 1017 | 30265 | 923729 | (11,88,682,4438) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | true | (0,0) | Unresolved: DeltaSumCertified (ordinary 3-fold point [1, 2]; A3 conjugate branches) | Unresolved: 7 or 14 | differs: reference [44, 1014, 30266, 923726] | 0.4 |
| 41 | 8 | 40 | 1790 | 68866 | 2828954 | (-2,56,-128,2402) | `(x+1)^8` | `x^4+1` | true | (0,0) | Decided: Nodal (62 nodes) | 8 (Decided; v2 #Jac'(F_p^4) = 6) | equal | 1.1 |
| 43 | 15 | 43 | 1829 | 78799 | 3422769 | (-1,-10,-226,1278) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | true | (0,0) | Decided: Nodal (62 nodes) | Unresolved: 7 or 14 | equal | 1.3 |
| 47 | 13 | 54 | 2316 | 104076 | 4889096 | (6,71,438,5270) | `(x+1)^4(x^2+x+1)^2` | `x^4+x^3+x+1` | true | (0,0) | Decided: Nodal (62 nodes) | 12 (Decided; v2 #Jac(F_p) = 1) | new | 1.7 |
| 53 | 21 | 52 | 2862 | 150562 | 7889314 | (-2,28,508,-1024) | `(x+1)^8` | `x^4+1` | true | (0,0) | Decided: Nodal (62 nodes) | 8 (Decided; v2 #Jac(F_p) = 1) | new | 2.7 |
| 53 | 36 | 57 | 2925 | 147507 | 7890353 | (3,62,-280,512) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | true | (0,0) | Decided: Nodal (62 nodes) | Unresolved: 7 or 14 | new | 3.0 |
| 53 | 41 | 45 | 2885 | 147972 | 7888001 | (-9,78,-761,4593) | `(x^4+x+1)(x^4+x^3+1)` | `x^4+x^3+1` | true | (0,0) | Decided: Nodal (62 nodes) | 15 (Decided; P mod 2 squarefree) | new | 2.8 |
| 61 | 52 | 51 | 3909 | 226185 | 13840033 | (-11,154,-1516,12108) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | true | (0,0) | Decided: Nodal (62 nodes) | Unresolved: 7 or 14 | new | 5.0 |
| 67 | 15 | 71 | 4661 | 302150 | 20149585 | (3,90,723,5045) | `(x^4+x+1)(x^4+x^3+1)` | `x^4+x^3+1` | true | (0,0) | Decided: Nodal (62 nodes) | 15 (Decided; P mod 2 squarefree) | new | 7.6 |
| 67 | 42 | 62 | 4614 | 301154 | 20166970 | (-6,80,-278,6274) | `(x+1)^8` | `x^4+1` | true | (0,0) | Decided: Nodal (62 nodes) | Unresolved: 4 or 8 | new | 7.6 |
| 71 | 37 | 56 | 5114 | 358256 | 25415418 | (-16,164,-1144,7086) | `(x+1)^8` | `x^4+1` | true | (0,0) | Decided: Nodal (62 nodes) | Unresolved: 4 or 8 | new | 9.5 |
| 71 | 67 | 78 | 5210 | 358566 | 25433810 | (6,102,758,11934) | `(x+1)^8` | `x^4+1` | true | (0,0) | Decided: Nodal (62 nodes) | Unresolved: 4 or 8 | new | 9.4 |
| 79 | 15 | 94 | 6510 | 493546 | 38940122 | (14,232,2502,23582) | `(x+1)^8` | `x^4+1` | true | (0,0) | Unresolved: DeltaSumCertified (A3 rational branches) | Unresolved: 2 or 4 or 8 | new | 14.5 |
| 79 | 40 | 82 | 6316 | 494788 | 38952656 | (2,39,658,2568) | `(x+1)^4(x^2+x+1)^2` | `x^4+x^3+x+1` | true | (0,0) | Decided: Nodal (62 nodes) | Unresolved: 6 or 12 | new | 14.5 |
| 79 | 46 | 70 | 6334 | 492232 | 38966242 | (-10,96,-896,10508) | `(x+1)^8` | `x^4+1` | true | (0,0) | Decided: Nodal (62 nodes) | 8 (Decided; v2 #Jac(F_p) = 1) | new | 14.3 |
| 89 | 57 | 94 | 8088 | 707974 | 62746864 | (4,91,1344,9280) | `(x+1)^4(x^2+x+1)^2` | `x^4+x^3+x+1` | true | (0,0) | Decided: Nodal (62 nodes) | Unresolved: 3 or 6 or 12 | new | 23.2 |
| 89 | 69 | 94 | 8090 | 705808 | 62734682 | (4,92,626,3438) | `(x+1)^8` | `x^4+1` | true | (0,0) | Decided: Nodal (62 nodes) | 8 (Decided; v2 #Jac'(F_p^4) = 6) | new | 23.2 |
| 97 | 76 | 103 | 9497 | 913861 | 88569977 | (5,56,634,13668) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | true | (0,0) | Decided: Nodal (62 nodes) | Unresolved: 7 or 14 | new | 36.3 |
| 101 | 74 | 109 | 10305 | 1024111 | 104049033 | (7,76,-1646,-14600) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | true | (0,0) | Decided: Nodal (62 nodes) | Unresolved: 7 or 14 | new | 37.8 |
| 101 | 98 | 115 | 10117 | 1027081 | 104049377 | (13,42,-1260,-18212) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | true | (0,0) | Decided: Nodal (62 nodes) | Unresolved: 7 or 14 | new | 38.3 |
| 103 | 93 | 117 | 10897 | 1091211 | 112578169 | (13,228,1726,23860) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | true | (0,0) | Unresolved: DeltaSumCertified (A3 rational branches) | 14 (Decided; v2 #Jac(F_p) = 1) | new | 53.4 |
| 127 | 63 | 135 | 16457 | 2050377 | 260132721 | (7,188,1866,19142) | `(x+1)^2(x^3+x+1)(x^3+x^2+1)` | `x^4+x^3+x^2+1` | true | (0,0) | Decided: Nodal (62 nodes) | Unresolved: 7 or 14 | new | 100.2 |
| 127 | 74 | 130 | 16210 | 2047378 | 260120418 | (2,42,-254,-5846) | `(x+1)^8` | `x^4+1` | true | (0,0) | Decided: Nodal (62 nodes) | Unresolved: 4 or 8 | new | 99.0 |

## 4. API

```rust
// rustmath-finitefields
pub struct SmallFq<const K: usize>;  pub type SmallFqElt<const K: usize> = [u64; K];
impl<const K: usize> SmallFq<K> {
    pub fn new(p: u64) -> Result<Self>;  pub fn with_modulus(p: u64, modulus: &[u64]) -> Result<Self>;
    pub fn characteristic, order, degree, modulus, reduce, zero, one, from_fp, is_zero,
           add, sub, neg, scale, mul, pow, frobenius, norm, inverse, is_square, code, from_code;
}
pub fn count_distinct_roots<const K: usize>(fq: &SmallFq<K>, f: &[SmallFqElt<K>]) -> Option<usize>;
pub fn count_distinct_roots_any_degree<const K: usize>(..) -> Option<usize>;   // small_field::

// rustmath_curves::weil
pub struct FrobeniusPolynomial;  pub enum WeilError;  pub enum TwoTorsionOrder;
impl FrobeniusPolynomial {
    pub fn from_point_counts(p: &Integer, counts: &[Integer]) -> Result<Self, WeilError>;
    pub fn from_counts_u64(p: u64, counts: &[u64]) -> Result<Self, WeilError>;
    pub fn from_coefficients(p: &Integer, coeffs: Vec<Integer>) -> Result<Self, WeilError>;
    pub fn p, genus, coefficients, l_polynomial, point_counts, predicts, real_weil_polynomial,
           is_weil, eval_mod, reduce_mod, factor_mod, reciprocal_halves_mod2,
           two_torsion_frobenius_order;
}
pub fn f2_poly_string(c: &[i64]) -> String;

// rustmath_curves::nodal_biprojective
pub struct BihomogeneousForm; pub struct FormModP; pub enum Fibration { FibresInX, FibresInY }
pub fn plane_point_count(f: &FormModP, k: u32, fib: Fibration) -> Result<u64, CountError>;
pub fn singular_locus(f: &FormModP) -> Result<SingularLocus, CountError>;
pub fn certify(locus: &SingularLocus, arithmetic_genus: usize, genus: usize)
    -> Result<ReductionStatus, CountError>;
pub fn normalization_point_counts(f: &FormModP, genus: usize, kmax: u32, opts: &CountOptions)
    -> Result<NormalizationCounts, CountError>;
pub enum Certificate { Nodal { nodes, genus_hypothesis } }
pub enum Unresolved { DeltaSumCertified { delta_sum, genus_hypothesis, non_nodal },
                      Unclassified(SingularPoint),
                      DeltaMismatch { found, expected, found_is_lower_bound } }
pub enum ReductionStatus { Decided(Certificate), Unresolved(Unresolved) }
pub enum BranchRule { Exact, AssumedUnibranch }
pub struct CountOptions { fibration, branch_rule, unibranch_if_unclassified, threads }
pub struct NormalizationCounts { p, genus, counts, plane_counts, status, locus, branch_rule }
pub enum LocalType { Node, A(usize), OrdinaryMultiple(usize), Unclassified(String) }
pub struct SingularPoint { residue_degree, y_place, x_at_infinity, coordinates, x_charpoly,
                           local, delta, branch_orbits }
pub struct SingularLocus { p, points, disc_degree, disc_formal_degree, disc_factor_degrees }
pub enum CountError { NotPrime, CharacteristicTwo, PrimeTooLarge, FieldTooLarge, ZeroForm,
                      DegenerateModel(Degeneracy), FibreDegree, Shape, GenusInconsistent }
pub enum Degeneracy { DiscriminantVanishes, FibreComponent { place } }
```

Guarantees: `normalization_point_counts` never returns `BranchRule::AssumedUnibranch` with a
Decided status (Decided implies no non-nodal point, and the rule is then reported as `Exact`).
An unclassified point gives no counts unless `unibranch_if_unclassified` is set. A bounded search
never decides: the classifier's precision cap gives `Unclassified`, the nonsquare scan only finds a
helper whose root is verified.

## 5. Deviations from the design, with the reason

1. **D1 rejected (review 2).** `Certificate` has only `Nodal`; the delta-sum result is
   `Unresolved::DeltaSumCertified { delta_sum, genus_hypothesis, non_nodal }` with the exact branch
   rule. `CountOptions::allow_delta_certificate` does not exist; no option gives Decided for a
   non-nodal model. Follow-up F-9.
2. **Options and statuses.** The status no longer depends on the options. `CountOptions` has
   `branch_rule` (`Exact` by default, `AssumedUnibranch` for the reference rule) and
   `unibranch_if_unclassified` (the design's `unibranch_if_unresolved`). `Unresolved::NonNodal` is
   dropped: a model with a non-nodal point is `DeltaSumCertified`, `Unclassified` or
   `DeltaMismatch`, and the rule its counts were made with is `NormalizationCounts::branch_rule`.
   The driver labels reference-rule rows "Unresolved: assumed unibranch; Weil test passed".
3. **Genus input (review 3.2).** `Certificate::Nodal { nodes, genus_hypothesis }`;
   `DeltaMismatch` is only for a delta-sum above `p_a - g`; below it is
   `Err(CountError::GenusInconsistent)` (tested on curve A with genus 2). With unclassified points
   the delta-sum is a lower bound (each unclassified point counts 1): above `p_a - g` it is
   `DeltaMismatch { found_is_lower_bound: true }`, otherwise `Unclassified`.
4. **`deg G <= 3` (review 3.1).** Degree 3 is checked to be `(X - tau)^3` and passed to the
   classifier; anything else is `Unclassified` with a true message. Tested on the review's curve
   (`tacnode_tangent_to_the_fibre`).
5. **Residue fields.** The design's private `ResidueField` became `residue.rs` with a trait and
   two fields: `F_p[Y]/(g)` and the quadratic extension `K[X]/(G)`. So a non-nodal point in an
   irreducible two-point fibre is classified in its own residue field (the design's F-3 case) and
   tested (curve K1: two conjugate cusps in one rational fibre), as is the split case with a
   non-nodal point (curve K2). The two-point node test uses the quadratic extension directly
   instead of the trace formula.
6. **`SingularPoint::x_charpoly`** (new field): the characteristic polynomial over F_p of the
   X-coordinate. It links the certificate to `h62` (test: the product over the 62 nodes at p = 11 is
   `h62 mod 11` made monic).
7. **Small API differences.** `count_distinct_roots` returns `Option<usize>` (`None` for the zero
   polynomial). `SmallFq` also has `reduce`, `scale`, `pow`, `degree`, `is_zero`;
   `count_distinct_roots_any_degree` is public (both paths are tested against each other).
   `BihomogeneousForm::mobius_y` returns `Result` (refuses `ad - bc = 0`) and divides by the
   positive content without changing the sign, so `F o M / content = G` checks `c > 0` exactly.
   `WeilError::NoCounts` is dropped (genus 0 gives `P = 1`); `InvalidInput` and
   `FunctionalEquation` are added. `TwoTorsionOrder::Decided` is `DeterminedByP` (review 3.4); the
   driver prints "(Decided)" only on a Decided row, "(conditional on the counts)" otherwise.
8. **Model selection (review 3.6).** Order Nodal > DeltaSumCertified > Unclassified > DeltaMismatch
   > refusal, first model wins on ties. When two models both give counts, they are compared at
   `k <= min(kmax, 2)` only (the k = 4 count would triple the cost), and only under the exact rule
   (the reference rule is not an invariant of X).
9. **p = 23 (review 3.5).** Refused: the discriminant vanishes identically for G, G7, G13 and F.
10. **Example.** No JSON output (design 6.4 "optionally as JSON"); options `--model F`,
    `--reference-rule`, `--threads` added. Follow-up.
11. **Tests added beyond the design:** the residue-degree-two curves J, J', J2, K1, K2 (review 3.7
    and 3.5), the k = 5, 6 predictions in the fast set and in `env_l31_k5`, `full_x_k5_31_41`
    (review 3.3), `GenusInconsistent`, the chart tests at infinity for five curves.

### Review notes folded in and skipped

All required changes of review Section 3 are in (3.1 to 3.7). Skipped or partial:
- 3.4 "Section 0 items 3 and 4 ... rephrase": the design document was not edited (not mine to edit);
  this report states the statuses after D1.
- 3.6 "assert that the counts are equal": done at `k <= 2` only (item 8 above).
- 3.7 "measure the debug test-harness compile": done (Section 6).

## 6. Measured times

Machine: 4 cores shared with other agents (load average 4 to 5 during the runs). Release runs use
4 rayon threads.

| What | Time |
|---|---|
| Debug build of the `rustmath-curves` test harness from an empty target directory (after the `rustmath-finitefields` test build, 21 s) | 33 s; debug target 2.7 GB |
| Release build of `--test m23_zeta` (all dependencies) | 1 min 34 s; `target/release` 320 MB |
| `cargo test -p rustmath-finitefields --test small_field` (debug) | 3.2 s (4 tests) |
| `cargo test -p rustmath-curves --test biprojective_small_curves` (debug) | 1.0 to 1.1 s (10 tests) |
| `cargo test -p rustmath-curves --test m23_zeta` (fast set, debug) | 7.3 s (8 tests; measured while a release table run used the 4 cores) |
| `cargo test -p rustmath-curves --lib weil` (debug) | 0.02 s (9 unit tests of `weil` and `residue`) |
| `cargo test -p rustmath-finitefields` (whole crate, debug, tests run) | 22.9 s wall including the incremental build |
| `cargo test -p rustmath-curves` (whole crate, debug) | 2 min 34 s (lib 308 passed, 9 ignored, 123 s, dominated by the existing Belyi tests; all test targets and doc tests pass) |
| `env_` tests (5, release, `--test-threads 1`) | 38.4 s |
| `full_x_table_k4` (18 rows at k = 4, p = 5, cross-model checks at k <= 2) | 82.9 s |
| `full_both_fibrations_k3` | 1.9 s |
| `full_x_k5_31_41` | 77.2 s |
| Example, X for p = 5..131 at k <= 4 | 12m41.658s |
| Example, L covers for p = 5..127 at k <= 4 | 8m27.023s |

Per prime (example, release, k = 1..4, certificate and cross-model checks included): p = 79: 19.5 s, p = 97: 40.8 s, p = 113: 80.8 s, p = 127: 118.0 s, p = 131: 117.1 s.
X for p = 5..127 took 645 s here (the design's prototype: 504 s at load 2); the difference is the
heavier load (other agents plus my own clippy and debug test runs in parallel) and the
cross-model checks at k <= 2.
At k = 4 about `p^4 / 4` orbit representatives: at p = 79 (9.7 million) 19.5 s, about 2.0 us
wall per fibre on 4 threads, as in the design (Section 8.1). The certificate (discriminant,
factorization, all places) takes milliseconds per prime in release.

Build hygiene: `cargo check -p rustmath-curves --tests` passes (14.9 s incremental).
`cargo clippy -p rustmath-finitefields --tests`: no warning in the new files (one module-level
`allow(clippy::needless_range_loop)` in `small_field.rs`, index loops mirror the formulas).
`cargo clippy -p rustmath-curves --tests --examples --no-deps`: no warning in the new files (one
`allow(clippy::type_complexity)` in `biprojective_small_curves.rs` for the case tables). Without
`--no-deps` clippy stops on deny-level lints in `rustmath-combinatorics` (`inherent_to_string_shadow_display`,
`while_immutable_condition`), and with `--no-deps` on three `approx_constant` errors in
`rustmath-curves/src/belyi/kernel_refine.rs:1777` and `:1801` (lib test). All pre-existing, not
touched. `rustfmt` was run on the new files only (`cargo fmt` would reformat many unrelated
files: the crates are not fmt-clean at the base). Zero `unsafe`; no em-dashes in the new prose.

## 7. Scope of the extensions, and what is left undone

| Item (HANDOFF 3.3) | Done here | Status |
|---|---|---|
| 1. Remove the D flags at 13, 31, 41, 73 | The non-nodal point is classified exactly: an ordinary cusp (A2, delta 1, one branch) on `T = 0`, with 61 nodes; delta-sum 62. The reference counts are the exact counts. | Not removed under the project rule: the rows are `Unresolved (DeltaSumCertified)`. Le Lay's (8,23) model cannot help: its fibre degree is 8 (the certificate needs 2 to 4, and the other ruling has degree 23), and it has two non-nodal points over Q, so it is never Nodal (design F-2). Promotion: F-9. |
| 2. p = 3 and 5 | p = 5: exact counts `7, 37, 169, 561` (k <= 6 checked), Weil, mod 23, mod 2; status DeltaSumCertified. p = 3: refused by every model. | 5: Unresolved pending F-9 (if promoted, X has good reduction at 5). 3: F-1. |
| 3. More primes, about 127 | X to 131, L to 127 (Section 3). | New results; statuses per row. |
| 5. Order of Frobenius on `Jac[2]` | `two_torsion_frobenius_order`; 15 where `P mod 2` is squarefree. | Decided (15) at X: 11, 79, 97 and at the covers (53, 41), (67, 15) (all Decided rows); 15 "conditional on the counts" at X: 31; bounds elsewhere. Since C-5 (2026-11-20, `../../c57/`): Decided at 15 rows of X and 8 over L (4 of the X rows, 11, 31, 79, 97, and 2 of the L rows, (53, 41), (67, 15), by `P mod 2` squarefree, the rule of WP-F; the rest by the rank bounds); the tables of Section 3 give the new values. |
| 4. Q versus Q^rev | not attempted (WP-A). | F-7. |

Follow-ups (with the reason):

- **F-1. p = 3.** `F mod 3 = (V - 1)^23 T (T - 1)^3`; G, G7, G13 are degenerate too. Needs a
  3-adically good model (the canonical model of WP-A step A0). No claim about the reduction at 3.
- **F-2. Le Lay's (8,23) model** as an independent cross-check at small k (plane counts only; the
  certificate would need fibre degree up to 8).
- **F-4. Faster quartic root count** (resolvent cubic), only if p well above 131 is wanted (cost
  grows like `p^4`: 131 took 117.1 s at k = 4).
- **F-5. p = 23**: refused for every model (D = 0). Needs another model.
- **F-6. CC0 route for the L covers** (Le Lay's canonical models in P^3), so that the L table runs
  without `M23ISGALOIS`.
- **F-7. Q versus Q^rev**: WP-A.
- **F-8. Report to the owner** the L-31 correction (`zeta_L.json` and the paper's table) and the
  exact counts at 5.
- **F-9. Owner decision: promote `DeltaSumCertified` to Decided.** Evidence: Theorem 4.5 (proof in
  `certify.rs` header and design 4.5, checked step by step by review 1), the exact classifier
  (tested on A2, A3 with both branch classes, ordinary triple points, cusps in conjugate pairs),
  and the k = 5, 6 predictions (X at 5, 31, 41; L at (31, 12)). If promoted: X has good reduction
  at 5, 13, 31, 41, 73, 107, and the covers at (31, 12), (79, 15), (103, 93).
- **F-10. JSON output of the example** in the format of `zeta_X.json` (design 6.4).
- **F-11.** A test of an ordinary m-fold point and an `A_k` over a residue field of degree > 1
  with odd k (K1 covers A2 over `F_121`; the quadratic-extension classifier is otherwise exercised
  only by nodes and cusps).

## 8. Handoffs

Tracker rows (`PROJECT_TRACKER.md`, a later step):
- `rustmath-finitefields/src/small_field.rs:55` `SmallFq`, `:649` `count_distinct_roots` [done:
  tested against `FiniteField`, `poly_factor::roots`, `fp_factor`, brute force]. Note in the
  finite-field section that it is a specialised kernel (p < 2^16, K <= 8), justified by the
  750x speed-up measured in the design (4.0 ms against 5 us per quartic fibre over `GF(31^4)`).
- `rustmath-curves/src/weil.rs:88` `FrobeniusPolynomial` [done: zeta_X.json rows, exact Weil
  edge cases, PARI `ellap`].
- `rustmath-curves/src/nodal_biprojective/fibres.rs:153` `plane_point_count` [done: brute force,
  both fibrations, PARI small curves].
- `rustmath-curves/src/nodal_biprojective/singular.rs:420` `singular_locus` and `local.rs:200`
  `classify` [done for nodes, A_k, ordinary m-fold points over F_p and F_p^2; others Unclassified].
- `rustmath-curves/src/nodal_biprojective/certify.rs:308` `normalization_point_counts` [done:
  both M23 tables, anchor at 53, PARI small curves]. Status semantics: Decided = nodal
  certificate relative to the genus input.
- The "Do not call" entry "Point counts, zeta functions and L-polynomials of curves of genus >= 2:
  absent" is now partly obsolete: present for curves in P^1 x P^1 given by a bihomogeneous form
  with fibre degree 2 to 4, p < 2^16, k <= 8.

MAGMA tracker (`docs/magma/`): ch. 114 `LPolynomial`, `ZetaFunction` (for these curves, from
counts), `NumberOfPlacesOfDegreeOneECF` (the counts `N_k`), `IsNode`, `IsCusp`,
`IsOrdinarySingularity`, `IsAnalyticallyIrreducible`, `TangentCone` (as used by the classifier):
partial, through `nodal_biprojective`; ch. 21 `Frobenius`, `Norm`, `IsSquare` for the word-size
kernel.

Workstream handoff (`docs/workstreams/m23_golay/HANDOFF.md`):
- Section 2: add the WP-F entry points above under "Usable as is"; remove "Point counts ... absent".
- Section 3.2: both tables reproduced; the L row (31, 12) corrected to `43, 1017, 30265, 923729`,
  `a = (11, 88, 682, 4438)`, Q `x^4+x^3+x^2+1` up to reversal; the p = 5 row of X filled with the
  exact counts and status DeltaSumCertified; the D flags explained (one ordinary cusp each).
- Section 3.3: items 1, 2, 3, 5 as in Section 7 above; F-1 to F-11.
- Section 8 step 1 (WP-F core) done; step 4 (extensions 1, 2) partly done, pending F-9 and F-1.
- Commands: the debug suite; the opt-in commands
  `cargo test --release -p rustmath-curves --test m23_zeta -- --ignored full_` and
  `M23ISGALOIS=... cargo test --release -p rustmath-curves --test m23_zeta -- --ignored env_`; the
  example.

Commit and push: not done here (the task forbids it). The user's request is to push WP-F to
`claude/m23-wpf`. Proposed split (design D4): commit 1 `rustmath-finitefields` (`src/small_field.rs`,
`src/lib.rs`, `tests/small_field.rs`); commit 2 `rustmath-curves` (the rest, with `Cargo.lock`).
`target/release` was deleted at the end (Section 6).
