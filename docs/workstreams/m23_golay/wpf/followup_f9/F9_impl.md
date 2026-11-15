# F-9 implementation report: the reduction certificate of Lemma 9.1

Work package WP-F of the M23 Golay workstream, follow-ups F-9, F-5, F-1 and F-8, after the owner's
decisions of 2026-11-15. Branch `claude/m23-wpf` (PR #581), worktree `/home/user/RustMath-wpf`,
base `d91f880` (clean at the start; no earlier attempt was found). Nothing is committed. The
owner's clone is still at `d05475b` (read with `git show origin/main:<path>` only).

Paths `f9/...` are in `/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/m23/f9/`.
The approved design is `f9/F9.md` (no Revision section); the minor notes m1 to m7 of
`f9/F9.review1.md` are folded in (Section 6).

## 0. Summary

- Lemma 9.1 of the owner's paper (`paper/golay_m23.tex` at `2d11038`, `\label{lem:reduction}`) is
  now the certificate, hypothesis by hypothesis. The missing hypothesis, absolute irreducibility
  of `F mod p`, is checked by a new module `nodal_biprojective/irreducible.rs` (criterion and
  proofs: design Section 2). A line of a ruling (including a line at infinity, so "keeps its
  bidegree") is a refusal; otherwise a fibre irreducible of full degree over the residue field of
  a closed point of degree <= 4, then `gcd(dx, dy)` or a smooth `F_p`-point; otherwise
  `Unresolved(IrreducibilityNotShown)`.
- Statuses: `Decided(Certificate::Nodal { .. })` and `Decided(Certificate::DeltaSum { .. })`, both
  with `IrreducibilityEvidence`. Labels `Decided: nodal (62 nodes)` and
  `Decided: delta-sum, Lemma 9.1 (61 nodes, A2)`. `Unresolved::DeltaSumCertified` is removed;
  `Unresolved::IrreducibilityNotShown(IrreducibilityGap)` is new. `certify(f, genus)` takes the
  form. `CountOptions::branch_rule` is removed; the reference rule is
  `NormalizationCounts::counts_under(BranchRule::AssumedUnibranch)`, so a Decided status always
  has exact counts (an `assert!`).
- Regression: the reducible reductions of the design (two (2,2) curves meeting in 8 nodes, a
  curve times its conjugate over `F_169`, a vertical line, the line `x = infinity`, a (2,0) form)
  were `Decided: nodal` before and are now Unresolved or refused.
- Tables rerun (release, X 5..131 and L to 127, k <= 4): every counted row is Decided; the
  delta-sum certificate at X 5, 13, 31, 41, 73, 107 and L (31, 12), (79, 15), (103, 93), the nodal
  one elsewhere; refusals unchanged (X 3 and 23; L (5, 3), (11, 2), (23, 5), (23, 15)). Counts
  unchanged from WP-F. Equal to the owner's main on every row, singular-point types included.
- Data: `zeta_X.json` and `zeta_L.json` vendored from the owner's main `2d11038`.
- F-5 closed (p = 23 refused for good), F-1 waits for WP-A A0, F-8 done by the owner, F-9 done.
- Acceptance: debug `rustmath-curves` (whole crate) and `rustmath-finitefields --test small_field`
  green; release `full_` (4) and `env_` (5) green; clippy clean on touched files; rustfmt on
  touched files; no `unsafe`; owner's clone at `d05475b`; `target-f9rel` deleted.

## 1. What changed, per item

### 1.1 Scope item 1: the certificate of Lemma 9.1 and the irreducibility check

New `rustmath-curves/src/nodal_biprojective/irreducible.rs` (672 lines, from the prototype
`f9/proto/irreducible.rs`):

- `:1-46` header: MAGMA ch. 114.8.1 `IsAbsolutelyIrreducible`, ch. 23 for the fibre
  factorization (m7), Lemma 9.1 with its label, the paper's criterion quoted, the criterion with
  its proof in three steps, the search bound stated exactly (m5: 64 points per ruling and
  degree, infinity not tried for p >= 64).
- `:49-52` `MAX_BASE_DEGREE = 4`, `MAX_FIBRES_PER_DEGREE = 64`; `:60` `BasePoint`, `:71`
  `IrreducibleFibre` (`residue_degree`, Display `x-fibre over y = 6`, `x-fibre over y = inf`,
  `y-fibre over x = 2`, `x-fibre over the point g(y) = [2, 0, 1] of degree 2`), `:110`
  `AbsoluteWitness::{Bidegree { gcd }, SmoothRationalPoint { x, y }}`, `:122`
  `IrreducibilityEvidence` (`#[non_exhaustive]`, m7; Display `... irreducible of full degree;
  gcd(dx, dy) = 1`, `...; gcd(dx, dy) = 2, settled by the residue degree 2`, `...; smooth F_p-point
  (5, 6)`), `:157` `IrreducibilityGap::{NoIrreducibleFibre, NoSmoothRationalPoint { fibre, gcd }}`
  with Display, `:185` `Irreducibility::{Absolute, NotShown}`.
- `:209` `check_bidegree` (a zero entry is `Err(Shape)`, on every path: m2); `:234`
  `check_line_components` (step 1; the place of a reducible gcd is its first irreducible factor,
  `:221` `line_place`, m4); `:270` `line_points_of_degree` (closed points in the order of the
  design); `:297` `fibre_irreducible` (distinct-degree factorization over `F_p[t]/(g)`); `:310`
  `search_fibre`; `:382` `smooth_rational_point` (exhaustive over the four charts, roots sorted
  ascending); `:431` `irreducibility_search` (steps 2 and 3, as design 2.7); `:481`
  `absolute_irreducibility` (public).
- Unit tests `:487-672`: `coprime_parts`, `closed_points_by_gauss` (6, 10, 40, 150 over `F_5`,
  78 over `F_13`; PARI `unit.gp`), `line_components_are_refused` (curves (iii), (iii'), (iii''),
  a quadratic vertical factor over `F_5`, a reducible gcd, (v)), `display_strings`,
  `smooth_points_and_fibres_at_infinity`.

`rustmath-curves/src/nodal_biprojective/certify.rs`:

- `:1-52` header: Lemma 9.1 statement, the check of each hypothesis, the DVR remark for the L
  covers, the statuses, the counts and the reference rule.
- `:68` `Certificate::Nodal { nodes, genus_hypothesis, irreducibility }` and
  `Certificate::DeltaSum { delta_sum, nodes, genus_hypothesis, non_nodal, irreducibility }`, both
  variants `#[non_exhaustive]` (only `certify` builds them, m7); `:94` `genus_hypothesis()`,
  `:106` `irreducibility()`.
- `:116` `Unresolved::{IrreducibilityNotShown(IrreducibilityGap), Unclassified, DeltaMismatch}`
  (`DeltaSumCertified` removed).
- `:141` `non_nodal_types` (the existing rendering; a closed point of degree > 1 adds
  ` at a point of degree d`), `:173` `ReductionStatus::certificate()`, `:182` `label()`.
- `:231` `certify(f, genus) -> Result<(ReductionStatus, SingularLocus), CountError>`: bidegree,
  `singular_locus`, `genus <= p_a`, line components, the delta tally (unchanged), then the
  irreducibility search (design 4.2, with m2).
- `:312` `BranchRule` doc; `:323` `CountOptions` without `branch_rule`; `:388`
  `NormalizationCounts::counts_under(rule)`; `:420` `apply_corrections`; `:449`
  `normalization_point_counts` (calls `certify(f, genus)`, exact rule unless an unclassified
  point under `unibranch_if_unclassified`, `assert!` that Decided never has assumed-unibranch
  counts).

`rustmath-curves/src/nodal_biprojective/form.rs`: `:176` `is_primitive`, `:183`
`reduce_primitive` (`Err(NotPrimitive { content })`), `:191` doc of `reduce`, `:213-222` doc of
`FormModP` (the hypotheses of Lemma 9.1 that are the caller's).

`rustmath-curves/src/nodal_biprojective.rs`: `:5-8` MAGMA list with `IsAbsolutelyIrreducible`;
`:30-44` method item 3 rewritten (Lemma 9.1, both certificates, the criterion); `:48-51` the
genus and primitivity; `:61` `mod irreducible`; `:72-75` re-exports
(`absolute_irreducibility, AbsoluteWitness, BasePoint, Irreducibility, IrreducibilityEvidence,
IrreducibilityGap, IrreducibleFibre, MAX_BASE_DEGREE, MAX_FIBRES_PER_DEGREE`); `:108`
`CountError::NotPrimitive { content: Integer }` with its Display; `:115-117` the
`GenusInconsistent` doc.

Primitivity: checked by `reduce_primitive` in the M23 driver (`tests/common/m23.rs` `x_row`);
for `calf_mod` (`tests/common/m23.rs:220-230`) the doc states why it holds (HJLPPZ's header of
`polynomial_calF.m`; `[O_L : Z[s]] = 1`, PARI `lfield.gp`) and that every counted L row is at an
unramified prime (m6).

### 1.2 Scope item 2: statuses and every caller

- `tests/common/m23.rs`: imports `:22-28`; `RefRow::singular_points` (`certificate.singular_points`)
  and its parser; `zeta_x`/`zeta_l` docs (`2d11038`); `status_rank` (Nodal 5, DeltaSum 4,
  IrreducibilityNotShown 3, Unclassified 2, DeltaMismatch 1); new `owner_types(&SingularLocus)`
  (labels `A_1`, `A_k`, `ordinary m-fold`, counted with residue degree); `ModelStatus` and
  `select_model` (returns status and locus, via `certify(f, GENUS)`); `x_row` (`reduce_primitive`;
  the cross-model count check between Decided models); `checks` now delegates to the new
  `checks_of_counts(p, counts)`; `format_row` appends ` | irreducibility: {evidence} | types
  {owner types}` to a Decided row.
- `examples/m23_zeta.rs`: header (statuses, `--reference-rule` now prints
  `reference rule N = [..]` from `counts_under`); `line()` helper.
- `tests/m23_zeta.rs`, `tests/biprojective_small_curves.rs`: Section 2.
- Documents: Section 8.

### 1.3 Scope item 3: the tables

Rerun in release with `--target-dir /home/user/RustMath-wpf/target-f9rel`: Section 3; the
comparison with the owner's main: Section 4.

### 1.4 Scope item 4: the vendored data

`rustmath-curves/tests/data/m23/zeta_X.json` (sha256 `77fdf78e...039e56`) and `zeta_L.json`
(`d2bae9ef...19d8`) replaced by those of `2d11038`, unchanged (checked with `sha256sum` against
`git show origin/main:data/...`). `README.md`: the file table (commit, sha256, the old sha256),
what changed, why `rustmath_wpf_tables.json` and `rustmath_crosscheck.json` are not vendored, the
notes on p = 5 and (31, 12) rewritten (the owner corrected them; the old values stay in the tests
as what the assumed-unibranch rule gives), and the refusal reasons at (5, 3) and (23, s0). No test
needs `rustmath_wpf_tables.json`, so it is not vendored; the three values of
`rustmath_crosscheck.json` are quoted in `full_x_owner_recount_83_97_107`.

### 1.5 Scope item 5: F-5, F-1, F-8

`x_at_3_and_23_refused` keeps its assertions; its comment says that 23 is final (F-5, the
owner's decision, Le Lay Section 5 "Constructing the harmonic lift",
`paper/harmonic-construction.tex`, `sec:harmonic-construction`, Lemma `lem:marked-distance`, the
good-reduction field `K(rho)`, `rho^15 = pi`; Section 4 "Reduction at 23" for the local
structure; cited, not copied: m1) and that 3 waits for WP-A A0 (F-1). HANDOFF Section 3 Status
and 3.3, and Section 8, say the same; F-8 is recorded as done by the owner at `2d11038`.

## 2. Tests

All reference values come from PARI/GP 2.15.4 scripts in
`docs/workstreams/m23_golay/wpf/gp/` (the F-9 design's scripts copied there and rerun: outputs
identical; one new script `f9_tests.gp`), from the owner's data and classifier at `2d11038`, or
from HJLPPZ Prop. 3.14.

### 2.1 New tests

| Test | What | Reference |
|---|---|---|
| `irreducible.rs` unit tests (5) | Section 1.1 | Gauss's formula and a direct count (`unit.gp`); the curves of `regress.gp` |
| `biprojective_small_curves.rs::reducible_reductions_are_never_decided` | (i) `E1 E2`, (4,4), genus 1: 8 nodes, delta 8 = `p_a - g`, `Unresolved(IrreducibilityNotShown(NoIrreducibleFibre))`, counts `[19, 351]`, plane `18`, `predicted_counts_agree() == Some(false)`; (ii) `E E^sigma`: `NoSmoothRationalPoint { x-fibre over y = 0, gcd: 4 }`, counts `[0, 290]`, 8 nodes; (iii) `(x - 3) G`: 2 nodes, delta 2, `certify` and `normalization_point_counts` give `FibreComponent { "x-place g(x) = [10, 1] (ascending)" }`; (iii') `x = infinity`; (iii'') `y = infinity` (unchanged refusal); (v) `x^2 + 11` of bidegree (2,0): `Shape` on every path | `regress.gp` (factor over `F_13` and `F_169`, `#E1 + #E2`, `2 #E(F_169)`, brute force 18 and 0) |
| `smooth_22_curve_certified_by_a_smooth_point` | (iv) `E1` alone, gcd 2: `Decided: nodal (0 nodes)`, evidence `x-fibre over y = 0 ...; smooth F_p-point (3, 6)` (checked on the curve), counts `9, 171, 2268` | `regress.gp` (charpoly `x^2 - 5x + 13`), `f9_tests.gp` (the first smooth point in the Rust order) |
| `absolute_irreducibility_settled_by_a_residue_degree` | review m3: S1, a smooth (2,2) curve at p = 3 with no irreducible rational fibre: `Decided: nodal (0 nodes)` with `Bidegree { gcd: 2 }` settled by the fibre over `y^2 + 1`, counts `6, 12, 18`; S2, a (4,2) form at p = 3 with a rational fibre and no smooth `F_3`-point: `Bidegree { gcd: 2 }` by the fibre over `y^2 + 2y + 2` | `f9_tests.gp` (one factor over `F_3` and `F_9`; charpoly `x^2 + 2x + 3`, brute force and `hyperellcharpoly`) |
| `primitive_forms` | `reduce_primitive` refuses content 2 and 13 (`NotPrimitive`), accepts the primitive part; Display | arithmetic |
| `m23_zeta.rs::x_certificates_equal_the_owners` (fast, debug, 6 s) | every row of `zeta_X.json`: the best model is Decided with `Bidegree { gcd: 1 }`, nodal iff only `A_1`, and its singular points equal the owner's `certificate.singular_points` | the owner's classifier, `2d11038` |
| `m23_zeta.rs::full_x_owner_recount_83_97_107` (release) | counts at 83, 97, 107 at k = 4; 107 `Decided: delta-sum, Lemma 9.1 (60 nodes, A3 rational branches)` | `data/rustmath_crosscheck.json` and the paper at `2d11038` |

### 2.2 Changed tests (the owner's decision changes the expected status; a comment at each)

- `biprojective_small_curves.rs`: `decided_nodal` now matches the certificate with its
  evidence and returns it; `delta_certified` became `decided_delta_sum` (Decided, delta-sum equal to
  the locus's, label prefix, exact rule). Cases D, E, F, I, I2, T3, K1, K2, E1 (3,2) at 7, 11, 13,
  E2 (3,3) at 7, 13, 19 and the A3/A15 review cases now expect `Decided(DeltaSum)` (were
  `Unresolved(DeltaSumCertified)`). `cusp_and_tacnodes`: the unibranch counts come from
  `counts_under`; the former `assert!(!u.status.is_decided())` is removed with a comment (the
  rule is no longer the rule of any status). Curve A asserts a smooth-point witness checked on the
  curve; E1 (3,2) asserts `Bidegree { gcd: 1 }`; labels asserted for D and K1;
  `unclassified_e6_points` adds the witness at 11 (`x-fibre over the point g(y) = [6, 1, 1] of
  degree 2`; PARI `f9_tests.gp`, `unit.gp`: no rational fibre at 11).
- `m23_zeta.rs`: `nodal62` (Nodal 62, `Bidegree { gcd: 1 }`, exact, label); `delta_sum62` replaces
  `non_nodal_types`; `reference_polynomials_pass_the_derived_checks` (19 rows, all ok);
  `x_at_13_and_31_one_cusp` (Decided delta-sum, label, owner types `{A_1: 61, A_2: 1}`, G13 at 13,
  unibranch counts equal by `counts_under`); `x_at_5_tacnode_with_conjugate_branches` (Decided
  delta-sum, label, owner types `{A_1: 60, A_3: 1}`, the witness fibre over `y^2 + 2` (PARI
  `m23_p5.gp`), the reference row of `2d11038`, and `counts_under(AssumedUnibranch)[..4] ==
  [8, 36, 170, 560]`, the `d05475b` row, not Weil, predictions fail); `x_at_3_and_23_refused`
  (comments only); `env_x_from_f_k3` (also Decided and the owner's types on F);
  `env_l_table` (the four refusals, owner types on every counted row, (31, 12) Decided delta-sum
  with its label and the corrected row, `counts_under` gives `44, 1014, 30266, 923726`, Weil,
  mod 23 `(15, 7)`); `env_l31_k5` (the reference rule from `counts_under`: `N_5 = 28604634`,
  predicted `28598074`); `full_x_table_k4` (19 rows; delta-sum at 5, 13, 31, 41, 73; owner types
  on every row).

## 3. Regenerated tables

Run: `target-f9rel/release/examples/m23_zeta 5 131 4` (X) and
`M23ISGALOIS=/home/user/external/m23isgalois target-f9rel/release/examples/m23_zeta 5 127 4 --cover L`,
release, 4 threads, shared 4-core machine (load average 5 to 6 during the runs: the Steiner job
ran beside). Raw output: `f9/impl/x_out.txt`, `f9/impl/l_out.txt`; formatted by
`f9/impl/tables.py`. The binary was built before two last edits that only rename private helpers
and doc links; after them the example was rebuilt and X and L at p <= 47 rerun: identical rows
(`f9/impl/x_final_5_47.txt`, `l_final_5_47.txt`). "2-torsion order" is the order of Frobenius on
`Jac[2]`, "Decided" only on a Decided row. Every counted row passes the exact Weil test and the
mod 23 check. Not in the table: X at 3 (refused by every model, `x_at_3_and_23_refused`).

Summary of the statuses:

- X, `Decided: nodal (62 nodes)`: 7 (G7), 11, 17, 19, 29, 37, 43, 47, 53, 59, 61, 67, 71, 79, 83, 89,
  97, 101, 103, 109, 113, 127, 131 (23 rows).
- X, `Decided: delta-sum, Lemma 9.1`: 5 `(60 nodes, A3 conjugate branches)`, 13 (G13), 31, 41, 73
  `(61 nodes, A2)`, 107 `(60 nodes, A3 rational branches)` (6 rows).
- X refused (`DiscriminantVanishes`, every model): 23 (and 3).
- L, `Decided: nodal (62 nodes)`: 22 rows; `Decided: delta-sum, Lemma 9.1`: (31, 12)
  `(57 nodes, ordinary 3-fold point [1, 2]; A3 conjugate branches)`, (79, 15) and (103, 93)
  `(60 nodes, A3 rational branches)`; refused: (5, 3), (11, 2), (23, 5), (23, 15).
- Irreducibility: every counted row by a fibre and `gcd(4, 23) = 1`; an `F_p`-rational x-fibre
  everywhere except X at 5 (the x-fibre over the root of `y^2 + 2`). The witnesses equal those of
  the independent GP version (`irred.gp` via `m23_cc0.gp`, `l_covers.py`, `m23_p5.gp`) on all 54
  counted rows.
- Order of Frobenius on `Jac(X)[2]`: 15 and Decided at X 11, 31, 79, 97 and at L (53, 41),
  (67, 15) (X 31 was "conditional on the counts" before).
- The reference rule (`--reference-rule`, `counts_under`): X at 5 `8, 36, 170, 560`, L (31, 12)
  `44, 1014, 30266, 923726` (the `d05475b` rows).

### 3.1 X

| p | model | N1, N2, N3, N4 | (a1, a2, a3, a4) | Weil | mod 23 | status | irreducibility | singular points (owner's labels) | 2-torsion order | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | G | 7, 37, 169, 561 | (1, 6, 20, 16) | true | (0, 0) | Decided: delta-sum, Lemma 9.1 (60 nodes, A3 conjugate branches) | x-fibre over the point g(y) = [2, 0, 1] of degree 2 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 60, "A_3": 1}` | Unresolved (multiple of 7, divides 14) | 0.1 |
| 7 | G7 | 11, 69, 383, 2433 | (3, 14, 46, 138) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 6 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 7, divides 14) | 0.1 |
| 11 | G | 15, 133, 1368, 14905 | (3, 10, 33, 145) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 2 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | 15 (Decided) | 0.2 |
| 13 | G13 | 11, 161, 2135, 29089 | (-3, 0, -12, 188) | true | (0, 0) | Decided: delta-sum, Lemma 9.1 (61 nodes, A2) | x-fibre over y = 6 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 61, "A_2": 1}` | Unresolved (multiple of 7, divides 14) | 0.1 |
| 17 | G | 21, 285, 4923, 82985 | (3, 2, 0, -130) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 7 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 7, divides 14) | 0.2 |
| 19 | G | 17, 405, 7265, 130913 | (-3, 26, 66, 74) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 3 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 7, divides 14) | 0.2 |
| 23 | | | | | | refused: model degenerate at this prime: the discriminant vanishes identically mod p | | | | 0.0 |
| 29 | G | 35, 789, 24773, 708785 | (5, -14, 16, 1060) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 10 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 7, divides 14) | 0.6 |
| 31 | G | 33, 993, 29640, 926713 | (1, 16, -35, 875) | true | (0, 0) | Decided: delta-sum, Lemma 9.1 (61 nodes, A2) | x-fibre over y = 2 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 61, "A_2": 1}` | 15 (Decided) | 0.7 |
| 37 | G | 43, 1433, 50881, 1881609 | (5, 44, 254, 3156) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 4 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 7, divides 14) | 1.2 |
| 41 | G | 42, 1780, 69408, 2827728 | (0, 49, 162, 1692) | true | (0, 0) | Decided: delta-sum, Lemma 9.1 (61 nodes, A2) | x-fibre over y = 1 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 61, "A_2": 1}` | Unresolved (multiple of 3, divides 12) | 1.8 |
| 43 | G | 44, 1996, 80192, 3422024 | (0, 73, 228, 3470) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 6 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 3, divides 12) | 2.4 |
| 47 | G | 47, 2349, 104321, 4882281 | (-1, 70, 96, 2934) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 5 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 7, divides 14) | 3.1 |
| 53 | G | 54, 2804, 148704, 7881808 | (0, -3, -58, -2164) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 4 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 3, divides 12) | 4.7 |
| 59 | G | 72, 3560, 205056, 12120780 | (12, 111, 648, 3991) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 3 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 5, divides 10) | 7.2 |
| 61 | G | 64, 3740, 226852, 13855856 | (2, 11, -24, 2476) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 4 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 3, divides 12) | 11.5 |
| 67 | G | 80, 4460, 302192, 20163724 | (12, 57, 584, 8759) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 3 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 5, divides 10) | 13.7 |
| 71 | G | 68, 5048, 358256, 25432076 | (-4, 11, 92, 4679) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 7 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 5, divides 10) | 17.5 |
| 73 | G | 83, 5541, 390089, 28393065 | (9, 146, 1428, 12030) | true | (0, 0) | Decided: delta-sum, Lemma 9.1 (61 nodes, A2) | x-fibre over y = 7 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 61, "A_2": 1}` | Unresolved (multiple of 7, divides 14) | 28.2 |
| 79 | G | 77, 6369, 491636, 38964585 | (-3, 68, -663, 7335) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 2 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | 15 (Decided) | 41.9 |
| 83 | G | 91, 6885, 574375, 47483041 | (7, 22, 902, 12258) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 2 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 7, divides 14) | 44.5 |
| 89 | G | 104, 7942, 707906, 62766210 | (14, 108, 1576, 22324) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 7 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 1, divides 8) | 39.6 |
| 97 | G | 101, 9385, 910820, 88546849 | (3, -8, -651, 2563) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 3 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | 15 (Decided) | 87.5 |
| 101 | G | 114, 10456, 1030926, 104090928 | (12, 199, 2020, 28200) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 5 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 3, divides 12) | 108.9 |
| 103 | G | 96, 10648, 1092594, 112536824 | (-8, 51, -282, -2198) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 1 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 3, divides 12) | 74.4 |
| 107 | G | 116, 11482, 1226360, 131139882 | (8, 48, 652, 19390) | true | (0, 0) | Decided: delta-sum, Lemma 9.1 (60 nodes, A3 rational branches) | x-fibre over y = 1 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 60, "A_3": 1}` | Unresolved (multiple of 1, divides 8) | 65.9 |
| 109 | G | 104, 11986, 1294118, 141216370 | (-6, 70, -652, 18718) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 1 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 1, divides 8) | 71.2 |
| 113 | G | 128, 13016, 1444142, 163058328 | (14, 221, 2594, 29766) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 4 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 3, divides 12) | 79.3 |
| 127 | G | 127, 16501, 2047933, 260199313 | (-1, 186, -336, 31116) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 3 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 7, divides 14) | 131.7 |
| 131 | G | 144, 17064, 2246976, 294472176 | (12, 23, -672, -12864) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 4 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 3, divides 12) | 244.2 |

### 3.2 Covers over L at degree-one primes

| p | s0 | model | N1, N2, N3, N4 | (a1, a2, a3, a4) | Weil | mod 23 | status | irreducibility | singular points (owner's labels) | 2-torsion order | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 3 | | | | | | refused: model degenerate at this prime: the discriminant vanishes identically mod p | | | | 0.0 |
| 7 | 5 | calF | 7, 45, 343, 2545 | (-1, -2, 2, 38) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 0 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 7, divides 14) | 0.1 |
| 11 | 2 | | | | | | refused: model degenerate at this prime: the discriminant vanishes identically mod p | | | | 0.0 |
| 19 | 3 | calF | 15, 413, 6849, 130169 | (-5, 38, -152, 650) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 5 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 7, divides 14) | 0.1 |
| 23 | 5 | | | | | | refused: model degenerate at this prime: the discriminant vanishes identically mod p | | | | 0.0 |
| 23 | 15 | | | | | | refused: model degenerate at this prime: the discriminant vanishes identically mod p | | | | 0.0 |
| 31 | 12 | calF | 43, 1017, 30265, 923729 | (11, 88, 682, 4438) | true | (0, 0) | Decided: delta-sum, Lemma 9.1 (57 nodes, ordinary 3-fold point [1, 2]; A3 conjugate branches) | x-fibre over y = 1 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 57, "A_3": 1, "ordinary 3-fold": 1}` | Unresolved (multiple of 7, divides 14) | 0.7 |
| 41 | 8 | calF | 40, 1790, 68866, 2828954 | (-2, 56, -128, 2402) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 0 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 1, divides 8) | 2.1 |
| 43 | 15 | calF | 43, 1829, 78799, 3422769 | (-1, -10, -226, 1278) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 1 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 7, divides 14) | 3.4 |
| 47 | 13 | calF | 54, 2316, 104076, 4889096 | (6, 71, 438, 5270) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 3 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 3, divides 12) | 4.1 |
| 53 | 21 | calF | 52, 2862, 150562, 7889314 | (-2, 28, 508, -1024) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 1 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 1, divides 8) | 6.7 |
| 53 | 36 | calF | 57, 2925, 147507, 7890353 | (3, 62, -280, 512) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 11 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 7, divides 14) | 6.7 |
| 53 | 41 | calF | 45, 2885, 147972, 7888001 | (-9, 78, -761, 4593) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 12 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | 15 (Decided) | 6.9 |
| 61 | 52 | calF | 51, 3909, 226185, 13840033 | (-11, 154, -1516, 12108) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 0 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 7, divides 14) | 12.3 |
| 67 | 15 | calF | 71, 4661, 302150, 20149585 | (3, 90, 723, 5045) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 1 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | 15 (Decided) | 19.6 |
| 67 | 42 | calF | 62, 4614, 301154, 20166970 | (-6, 80, -278, 6274) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 2 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 1, divides 8) | 18.1 |
| 71 | 37 | calF | 56, 5114, 358256, 25415418 | (-16, 164, -1144, 7086) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 1 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 1, divides 8) | 24.2 |
| 71 | 67 | calF | 78, 5210, 358566, 25433810 | (6, 102, 758, 11934) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 2 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 1, divides 8) | 21.2 |
| 79 | 15 | calF | 94, 6510, 493546, 38940122 | (14, 232, 2502, 23582) | true | (0, 0) | Decided: delta-sum, Lemma 9.1 (60 nodes, A3 rational branches) | x-fibre over y = 0 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 60, "A_3": 1}` | Unresolved (multiple of 1, divides 8) | 27.8 |
| 79 | 40 | calF | 82, 6316, 494788, 38952656 | (2, 39, 658, 2568) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 0 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 3, divides 12) | 19.8 |
| 79 | 46 | calF | 70, 6334, 492232, 38966242 | (-10, 96, -896, 10508) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 1 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 1, divides 8) | 19.7 |
| 89 | 57 | calF | 94, 8088, 707974, 62746864 | (4, 91, 1344, 9280) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 9 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 3, divides 12) | 33.2 |
| 89 | 69 | calF | 94, 8090, 705808, 62734682 | (4, 92, 626, 3438) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 0 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 1, divides 8) | 33.5 |
| 97 | 76 | calF | 103, 9497, 913861, 88569977 | (5, 56, 634, 13668) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 1 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 7, divides 14) | 40.9 |
| 101 | 74 | calF | 109, 10305, 1024111, 104049033 | (7, 76, -1646, -14600) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 4 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 7, divides 14) | 49.7 |
| 101 | 98 | calF | 115, 10117, 1027081, 104049377 | (13, 42, -1260, -18212) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 6 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 7, divides 14) | 47.7 |
| 103 | 93 | calF | 117, 10897, 1091211, 112578169 | (13, 228, 1726, 23860) | true | (0, 0) | Decided: delta-sum, Lemma 9.1 (60 nodes, A3 rational branches) | x-fibre over y = 1 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 60, "A_3": 1}` | Unresolved (multiple of 7, divides 14) | 53.6 |
| 127 | 63 | calF | 135, 16457, 2050377, 260132721 | (7, 188, 1866, 19142) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 6 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 7, divides 14) | 121.4 |
| 127 | 74 | calF | 130, 16210, 2047378, 260120418 | (2, 42, -254, -5846) | true | (0, 0) | Decided: nodal (62 nodes) | x-fibre over y = 6 irreducible of full degree; gcd(dx, dy) = 1 | `{"A_1": 62}` | Unresolved (multiple of 1, divides 8) | 114.4 |


## 4. Comparison with the owner's main `2d11038`

`f9/impl/compare_owner.py x_out.txt l_out.txt` (reads the owner's files with
`git show origin/main:<path>` in `/home/user/johnjanik/m23`, whose working tree stays at
`d05475b`). Output (`f9/impl/compare.txt`): 0 problems.

| Owner's file (`2d11038`) | Checked | Result |
|---|---|---|
| `data/zeta_X.json` (19 rows, all ok) | counts N1..N4, `P[1..4] = a`, `certificate.singular_points` against our types, status Decided | all equal; types `{A_1: 60, A_3: 1}` at 5, `{A_1: 61, A_2: 1}` at 13, 31, 41, 73, `{A_1: 62}` elsewhere, as ours |
| `data/zeta_L.json` (9 rows) | the four refused rows refused; the five counted rows: counts, a, singular points, Decided | all equal; (31, 12) `{A_1: 57, A_3: 1, ordinary 3-fold: 1}` and the corrected counts `43, 1017, 30265, 923729` |
| `data/rustmath_wpf_tables.json` (30 X, 29 L rows: RustMath's WP-F rows as the owner recorded them) | counts and a on every row; the refusal strings; the statuses under the renaming `Decided: Nodal` -> `Decided: nodal`, `Unresolved: DeltaSumCertified (T)` -> `Decided: delta-sum, Lemma 9.1 (n nodes, T)` | all equal |
| `data/rustmath_crosscheck.json` (the owner's recount at 83, 97, 107) | counts and a | equal (also asserted by `full_x_owner_recount_83_97_107`) |
| WP-F raw tables (`scratchpad/m23/impl/x_out.txt`, `l_out.txt`) | the same rows; counts unchanged; refusals unchanged | equal |

The owner's statement that his classifier finds the same singular points as RustMath's at every
prime up to 79 holds on every row of both of his tables (19 X rows, 5 counted L rows), here and in
the tests (`x_certificates_equal_the_owners`, `full_x_table_k4`, `env_l_table`, `env_x_from_f_k3`
on HJLPPZ's F). The owner's refusal reasons differ in wording: (5, 3) "the degree in V drops mod
p", (11, 2) "the discriminant vanishes identically mod p", (23, s0) "the characteristic divides the
degree in V"; RustMath refuses all four with `DiscriminantVanishes`. The owner's tables have no
row for X at 23 or 3.


## 5. Times

Shared 4-core machine; another job (the Steiner labelling) ran beside, load average 5 to 6.

| Step | Time |
|---|---|
| `cargo test -j 2 -p rustmath-curves` (debug, whole crate, warm build) | 1 min 49 s wall; lib 314 passed / 9 ignored (93.6 s), `biprojective_small_curves` 16 / 1 ignored, `m23_zeta` 9 / 9 ignored (7.8 s), the other targets green |
| `cargo test -j 2 -p rustmath-finitefields --test small_field` | 4 passed, 3.6 s |
| `m23_zeta` fast set alone (debug) | 8 to 17 s; `x_certificates_equal_the_owners` alone 6.3 s (below the 10 s of the design, so it stays in the fast set) |
| `biprojective_small_curves` (debug) | 1.1 to 2.2 s; `reducible_reductions_are_never_decided` included |
| `irreducible.rs` unit tests (debug) | 0.02 s |
| release build (`--target-dir target-f9rel`, from scratch) | 198 s |
| `full_` (4 tests in parallel: `full_x_table_k4`, `full_x_owner_recount_83_97_107`, `full_x_k5_31_41`, `full_both_fibrations_k3`) | 420 s, all pass |
| `env_` (5 tests, `M23ISGALOIS`) | 30.8 s (after a 29 s rebuild), all pass |
| `biprojective_small_curves -- --ignored` (release) | 1 test, pass |
| X table 5..131, k <= 4 | 1083 s (WP-F: about 800 s on a less loaded machine); 131 alone 244 s |
| L table 5..127, k <= 4 | 688 s |
| the GP scripts of `wpf/gp/` (rerun) | each below 0.2 s |
| clippy `--tests --examples --no-deps` | no warning in a touched file (the lib-test target fails on the pre-existing `approx_constant` errors in `belyi/kernel_refine.rs:1777,1801`, as at `d91f880`) |
| `cargo doc -p rustmath-curves --no-deps` | no warning in `nodal_biprojective` |

The irreducibility check costs about 1 ms per M23 form in debug (prototype, design 2.8) and is
invisible beside the plane count; the per-row times above are those of the WP-F tables.


## 6. Deviations from the design, and the review notes folded in

- m1 (F-5 citation): Le Lay's Section 5 is "Constructing the harmonic lift"
  (`sec:harmonic-construction`); the code comment, the HANDOFF and this report cite it so, with
  Section 4 "Reduction at 23" for the local structure.
- m2: `certify` checks the bidegree first, so `x^2 + 11` of bidegree (2,0) is `Err(Shape)` on every
  path (`certify`, `normalization_point_counts`, `absolute_irreducibility`).
- m3: a test of the `Bidegree { gcd: 2 }` branch (S1 with a full certificate, S2 with
  `absolute_irreducibility`), new PARI script `f9_tests.gp`.
- m4: a line component names the first irreducible factor of the gcd (a true place), as
  `singular.rs` does.
- m5: the `NoIrreducibleFibre` label states the bound ("no irreducible fibre among the first 64
  closed points of each degree <= 4 in either ruling"); the module header says 64 and that
  infinity is not tried for p >= 64.
- m6: the `calf_mod` doc says that the ramified primes 2, 11, 23 have no counted L row.
- m7: ch. 23 cited for the fibre factorization; `l_covers.py` docstring says `5 <= p <= 127`;
  `Certificate` variants and `IrreducibilityEvidence` are `#[non_exhaustive]`.
- Labels: a non-nodal closed point of degree > 1 is rendered with ` at a point of degree d`
  (K1: `Decided: delta-sum, Lemma 9.1 (0 nodes, A2 at a point of degree 2)`); the design's
  rendering had no degree. No M23 label changes (all their non-nodal points are rational).
- `select_model` returns `(status, locus)` pairs (`ModelStatus`), needed for `owner_types`.
- `checks_of_counts(p, counts)` in `tests/common/m23.rs`, for the derived checks of the
  `counts_under` values (the design used them without saying how).
- `x_row`'s cross-model count check now runs only between Decided models (before: Decided or
  `DeltaSumCertified`).
- `env_x_from_f_k3` also asserts that F is Decided with the owner's singular points (not in the
  design; it passes).
- Private helpers named so that the MAGMA candidate scan does not pick them up:
  `line_points_of_degree` (not `base_points`, which matched ch112 `BasePoints`), the unit-test
  helpers `test_form` and `component_place` (`form` and `place` matched ch143 `Form` and ch114
  `Place`).
- Doc links to the function `certify` use `fn@super::certify` (the module of the same name made
  the link ambiguous).
- Clippy: the column gcd loop uses `transpose()` (needless_range_loop), and `ModelStatus` names
  the complex type of `select_model`.

## 7. Follow-ups

- F-1: p = 3 waits for the model of WP-A step A0.
- F-5: closed.
- F-8: done by the owner at `2d11038`.
- F-9: done.
- Unchanged: F-2, F-4, F-6, F-7, F-10, F-11 (WP-F report Section 7).
- New, small: the irreducibility search has fixed bounds (4, 64). A curve that needs more is
  `Unresolved(IrreducibilityNotShown)`, never wrong. No case in the tables comes near them (at
  most 17 rational fibres tried, or 4 points of degree 2 at p = 5).

## 8. Handoffs

All edits are in the worktree, uncommitted (as instructed).

- `PROJECT_TRACKER.md`: `:2116` the certify row rewritten (Lemma 9.1, both certificates, the
  check order, `counts_under`, `reduce_primitive`, the new tests and data, the results; paths
  `certify.rs:449,231,388`); `:2117` new 🟢 row for `irreducible.rs:481,234,431,382`; `:2119` the
  singular-locus row (statuses renamed, `x_certificates_equal_the_owners`); `:2089` the curves
  tally (🟢 17, `#[ignore]` x 19, 10 of them WP-F); totals `:56`, `:60`, `:65`, `:3178`, `:3179`
  (632 / 331 / 281 / 84 / 72, 1400 rows; shares and defects unchanged) and the CURVES summary row
  `:195` (17 🟢); changelog row `:3191`.
- `docs/magma/status_overrides.csv` (CRLF kept): new `ch114.IsAbsolutelyIrreducible,partial,
  rustmath-curves/src/nodal_biprojective/irreducible.rs:481` (one-sided; true when the criterion
  proves it, a refusal for a line component, otherwise NotShown); notes of
  `ch114.NumberOfPlacesOfDegreeOneOverExactConstantField` ("Decided ... includes absolute
  irreducibility", path `certify.rs:449`), `.2` (path `:449`) and `ch114.LPolynomial` ("Decided
  (Lemma 9.1, nodal or delta-sum certificate)", path `certify.rs:363`). `python3
  tools/magma_tracker/build.py` run: `docs/magma/STATUS.md` ch. 114 reviewed 8 -> 9, partial
  8 -> 9 (overall reviewed 5188 -> 5189, partial 1250 -> 1251, missing 8000 -> 7999);
  `intrinsics.csv` only the four ch. 114 rows and the generated line references of
  `ch136.FrobeniusPolynomial` (`certify.rs:258` -> `:363`).
- `docs/workstreams/m23_golay/HANDOFF.md`: Section 3 Status block rewritten (statuses with both
  labels, the irreducibility check, every X and L row by certificate, the refusals with F-5
  final, the corrected rows, the 2-torsion order now Decided at X 31, extensions 1 and 2, the
  follow-ups, the commands); a note at the top of 3.2 (the tables are those of `d05475b`;
  `2d11038` corrected them); 3.3 items 1 and 2 with their status; Section 8 item 4 (done except
  p = 3, which moves to WP-A A0).
- `docs/workstreams/m23_golay/wpf/`: `design/F9.md`, `design/F9.review1.md` (copies of
  `f9/F9.md`, `f9/F9.review1.md`), `impl/F9.md` (this report), `README.md` (rows for them and for
  the F-9 GP scripts, how to make `G.gp`), `gp/` (the design's nine scripts with their rerun
  outputs, identical to the design's, and the new `f9_tests.gp` with its output).
- `rustmath-curves/tests/data/m23/README.md`: Section 1.4.
- For the owner (not done here, his repository): his `docs/RUSTMATH_HANDOFF.md` copy of the brief
  can take the new Status block; his `data/rustmath_wpf_tables.json` has the old status names
  (`Decided: Nodal`, `Unresolved: DeltaSumCertified (...)`), which the rerun maps one to one
  (Section 4).
- Commit and push: not done (not asked). Suggested message: "rustmath-curves: F-9, the reduction
  certificate of Lemma 9.1 with absolute irreducibility; data from M23 main 2d11038".

## 9. Minor items of review 1

Code review 1 (`f9/F9.creview1.md`) approved the change with three optional, cosmetic items
(its Section 7). I agree with all three and fixed them before the commit. None changes code
behaviour.

| Item | File | Fix |
|---|---|---|
| n1 | `rustmath-curves/tests/data/m23/README.md:20-25` | The "what changed" entry for `zeta_X.json` now says that the p = 5 row also loses its `reason` field ("not a Weil polynomial"). Checked with `git show` on both commits: at `d05475b` only the p = 5 row has `reason` and 18 of 19 rows have `model`; at `2d11038` no row has either. I also rewrapped the next paragraph, whose line on Le Lay's attribution was 118 columns after the F-9 edit. |
| n2 | `rustmath-curves/examples/m23_zeta.rs:26` | `use std::time::Instant;` moved up to the other `use` lines, above `fn line`. |
| n3 | `docs/workstreams/m23_golay/HANDOFF.md:602-612` | Section 3.3 items 1 and 2 rewrapped at 96 columns. The text is unchanged word for word (the script that rewrapped them asserts this). The same kind of over-long line, introduced by F-9, was also rewrapped in two paragraphs of `docs/workstreams/m23_golay/wpf/README.md`. |

Also added: `docs/workstreams/m23_golay/wpf/impl/F9.review1.md`, a copy of the code review (as
`impl/WPF.review1.md` is for WP-F), with a row in `wpf/README.md`; that README now says that the
review's `f9/rev1/...` paths are scratch work that is not kept. `impl/F9.md` is again a copy of
this report.

Reruns (all debug; no release build was needed, so `target-f9rel` was not created):

| Check | Result |
|---|---|
| sha256 of `tests/data/m23/zeta_X.json`, `zeta_L.json`, `h62.json`, `lelay_optimal_23_4_Z.json` | equal to the README; the first two equal `git show origin/main:data/...` (`2d11038`) |
| `rustfmt --edition 2021 --check rustmath-curves/examples/m23_zeta.rs` | clean |
| `cargo build -j 2 -p rustmath-curves --example m23_zeta`, then `m23_zeta 5 7 2 --reference-rule` | p = 5 `N = [7, 37]`, `Decided: delta-sum, Lemma 9.1 (60 nodes, A3 conjugate branches)`, reference rule `[8, 36]`; p = 7 `N = [11, 69]`, `Decided: nodal (62 nodes)` (as in Section 3) |
| `cargo clippy -j 2 -p rustmath-curves --example m23_zeta --no-deps` | no diagnostic in the example |
| `cargo test -j 2 -p rustmath-curves --test m23_zeta --test biprojective_small_curves` | 9 passed / 9 ignored; 16 passed / 1 ignored |

The owner's clone is still at `d05475b` with a clean tree. Nothing was committed.
