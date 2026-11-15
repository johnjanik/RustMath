# WP-F: the minor items of code review 1, applied

Date: 2026-11-15. Worktree `/home/user/RustMath-wpf`, branch `claude/m23-wpf`, HEAD `acea928`. All
changes are uncommitted: nothing was committed or pushed. Nothing in `/home/user/RustMath` was touched.
Scratch for this pass: `scratchpad/m23/impl/minors/` (gp script, its output, clippy and test logs).

**Not mine:** `git status` also shows `PROJECT_TRACKER.md`, `docs/magma/STATUS.md`,
`docs/magma/intrinsics.csv`, `docs/magma/status_overrides.csv` and
`docs/workstreams/m23_golay/HANDOFF.md` as modified. Their mtimes (08:45:18 to 08:45:37) are before
my first edit, so someone else made those changes in this worktree. They look like a parallel docs
pass (for example, the tracker now says `§21.4.4 IsSquare`). I did not touch them. Check them
before you commit.

## 1. Changes per item

Line numbers are the new ones.

1. **Unclassified path test (review 3.1).** New test `unclassified_e6_points`
   (`rustmath-curves/tests/biprojective_small_curves.rs:517`). It uses `F = Y^3 - X^4` (`cols[0][3] = 1`,
   `cols[4][0] = p - 1`) at p = 7, 11 and 13, the primes the review's `e6` bin used. It asserts:
   - By default: status `Unresolved::Unclassified`, `counts` empty, `plane_counts = p^k + 1`
     (k <= 3), brute force `p + 1`, and two points, both `LocalType::Unclassified` with `delta == None`.
   - With `unibranch_if_unclassified`: `branch_rule == AssumedUnibranch`, `!status.is_decided()`,
     the same status as the default, and counts `p^k + 1`.
   - `DeltaMismatch { found_is_lower_bound: true }` is also reached. With genus 6, the genus of a
     smooth (4,3) lift over Q, `p_a - g = 0`. Each of the two unclassified points counts at least 1.
     So the status is `DeltaMismatch { found: 2, expected: 0, found_is_lower_bound: true }`, with no
     counts.
2. **Costs and the reduction (review 3.2).**
   - `fibres.rs:152` (`plane_point_count` doc): about `q/k + 1` fibre root counts and one pass
     over the q base points. Measured about 2 us wall per quartic fibre on 4 threads at k = 4
     (p = 79: 9.7 million fibres in about 20 s). At p = 65521, k = 3 that is about 9.4e13 fibres,
     which would take years.
   - `certify.rs:312` (`normalization_point_counts` doc): the cost of `singular_locus` once, plus
     `plane_point_count` for each k up to kmax. The last k dominates. The bound `p^kmax < 2^63` does
     not make the call fast.
   - `singular.rs:418` (`singular_locus` doc): the cost is the factorization of the discriminant,
     of degree `(2dx - 2) dy <= 6 dy`. That is 138 for M23 (milliseconds), and up to 24576 at the
     bound `dy <= 4096` (far from interactive).
   - `fibres.rs:127-132`: the `collect` into `Result<Vec<u64>>` (q / 2^14 entries) and the `sum`
     are replaced by `.try_reduce(|| 0, |a, b| Ok(a + b))?`. The Ok value is the same u64 sum. On a
     vanishing fibre, one failing chunk's error is returned, which the collect also did.
3. **Panics (review 3.3).**
   - `small_field.rs:649`: `# Panics` on `count_distinct_roots` (degree above 4096; degrees above
     4 take the vector path).
   - `small_field.rs:731`: `# Panics` on `count_distinct_roots_any_degree` (degree above 4096; the
     `2^48` accumulator bound).
   - `weil.rs:388-395` (`two_torsion_frobenius_order`): a `Cost` paragraph (trial division up to
     `2^(d/2)`, about `2^31` steps near d = 62; instant for genus 4) and `# Panics` (a factor of
     `P mod 2` of degree above 62, possible from genus 32 on).
4. **GenusInconsistent in the driver (review 3.4).**
   - `tests/common/m23.rs:342`: new `assert_genus_consistent(result, what)`. It panics on
     `Err(CountError::GenusInconsistent)`, saying the error contradicts the genus hypothesis
     g = 4 of every certificate and is not a refusal.
   - `select_model` (`m23.rs:355`) calls it for every model (`:360-364`) and documents the panic.
   - The example's X path uses `x_row` and `select_model` from the same file (`#[path]`), so it is
     covered.
   - The example's L path has no model selection, but it printed this error as "refused: ...". I
     applied the same check there (`examples/m23_zeta.rs:57`) and in the matching test helper
     `l_row` (`tests/m23_zeta.rs:383`). This goes a little beyond the letter of the item. Revert
     those two lines if you want the change kept to `select_model` only.
5. **MAGMA chapter in the headers (review 3.5).** The digest is
   `rustmath-lava/docs/Chapter_114_Algebraic_Curves.md`.
   - `certify.rs:3-6` cites 114.5.1 `Genus`, `ArithmeticGenus`; 114.4.3 `IsNode`,
     `IsAnalyticallyIrreducible`; 114.8.1 `NumberOfPlacesOfDegreeOneECF`, `LPolynomial`.
   - `residue.rs:5-8` cites 114.4.1 (points over extensions), 114.5.1
     `HasSingularPointsOverExtension`, and ch. 21 `Norm`, `IsSquare`.
   - The style follows the other headers in `nodal_biprojective.rs` and `weil.rs`.
6. **F-11, the A_k part (review 3.6).**
   - New helper `monic_quadratic_in_x` (`biprojective_small_curves.rs:42`), struct `ReviewCase`
     (`:586`), `review_cases()` (`:619`, PARI values quoted above it), `check_review_case` (`:692`).
   - New tests `node_of_degree_3_tacnode_of_degree_2_and_a15` (`:710`, debug) and
     `node_of_degree_3_up_to_k6` (`:726`, `#[ignore]`, release).
   - Each case runs the existing `check` (P against PARI, Weil, predictions beyond g, both
     fibrations, brute force at k = 1). It also asserts the counts against PARI, one closed point,
     its residue degree, local type and branch orbits, and the status: `Decided` nodal (3 nodes)
     for the nodes, `DeltaSumCertified` with `[A(k)]` otherwise.
   - The file header (`:13-16`) names the sources.
   - Debug kmax: 3 for the nodes (k = 3 already separates the classes: plus or minus 3 points),
     4 for the A3 cases (class at k = 2, rational at k = 4), 3 for the A15 cases.
   - k = 6 for the nodes is in the release test. A first version ran k = 6 in debug and took 38 s.
     In release it takes 2.0 s for both nodes.
7. **IsSquare citation.** `small_field.rs:5`: `IsSquare (21.4.5)` became `(21.4.4)`. The digest
   has IsSquare in §21.4.4 "Predicates on Ring Elements", line 217. Frobenius and Norm are correctly
   §21.4.6.

No `unsafe` and no em-dashes or en-dashes in the added lines (checked with grep on the diff).

## 2. New tests and their reference values

Every reference value was rerun with PARI/GP 2.15.4.
- Rerunning `gp -q WPF.rev1/gp/adv2.gp` gives output identical to `cases.txt` (`diff`, saved as
  `minors/cases_rerun.txt`).
- My own script `minors/verify.gp` (output `minors/verify_out.txt`) starts only from p, b and c.
  It factors `D = b^2 - 4c` mod 13, takes the squarefree part H, and computes
  `hyperellcharpoly(Mod(1,13)*H)`, `N_k = 13^k + 1 - polsym(charpoly, 6)[k+1]` and the branch class
  `issquare(subst(D / S^ord, 'Y, ffgen(S)))`.

All curves are `X^2 + b(Y) X + c(Y)` at p = 13 (b and c ascending in Y):

| case | b / c | D mod 13 (PARI `factor`) | charpoly | N_1..N_6 | point, branches | status |
|---|---|---|---|---|---|---|
| node3_sq0 | [7,1,10,1,0,1] / [6,7,4,12,2,0] | (Y+5)(Y^3+2)^2(Y^3+8Y^2+Y+11) | x^2 + x + 13 | 15, 195, 2160, 28275, 372075, 4829760 | node, degree 3, orbits [2] (not a square in F_13^3) | Decided nodal (3) |
| node3_sq1 | [11,8,10,9,0,1] / [8,2,1,12,5,0] | (Y+5)(Y^3+2)^2(Y^3+8Y^2+4Y+9) | x^2 - x + 13 | 13, 195, 2236, 28275, 370513, 4829760 | node, degree 3, [1, 1] | Decided nodal (3) |
| A3_d2_sq0 | [12,3,1,3,11,8,1] / [7,10,12,11,7,4,0] | (Y+1)(Y+8)(Y^2+2)^4(Y^2+7Y+7) | x^2 + 4x + 13 | 18, 180, 2106, 28800, (371538, 4822740) | A3, degree 2, [2] | DeltaSumCertified [A3] |
| A3_d2_sq1 | [1,4,7,4,11,8,1] / [12,12,5,11,2,5,0] | (Y+1)(Y^2+2)^4(Y^3+2Y^2+11Y+6) | x^2 - 5x + 13 | 9, 171, 2268, 28899, (372069, 4826304) | A3, degree 2, [1, 1] | DeltaSumCertified [A3] |
| A15_sq1 | [12,2,9,8,12,9,5,7,12,1] / [3,10,2,12,12,8,5,12,11,0] | (Y-1)^16 (Y^2+Y+2) | 1 (genus 0) | 14, 170, 2198 (13^k + 1) | A15, degree 1, [1, 1] (h(1) = 4) | DeltaSumCertified [A15] |
| A15_sq0 | [11,12,10,0,3,8,0,1,6,1] / [7,2,3,11,12,7,11,0,12,0] | (Y-1)^16 (Y+6)(Y+9) | 1 (genus 0) | 14, 170, 2198 | A15, degree 1, [2] (h(1) = 5) | DeltaSumCertified [A15] |

- Values in brackets are quoted in the comment but not asserted.
- The debug test asserts N_1..N_3 for the nodes and N_1..N_4 for A3. The release test asserts
  N_1..N_6 for the nodes.
- E6 (`unclassified_e6_points`): the reference is that the curve is rational. `t -> (t^3, t^4)` is a
  bijection from P^1 with inverse `t = Y/X`, so `#X0(F_q) = q + 1`. The brute force in `verify.gp`
  gives p affine points plus (inf, inf), so p + 1, at p = 7, 11 and 13.

## 3. Acceptance results

All builds used `-j 2`. The machine is shared (load 1 to 4). The final runs were made after rustfmt.

| Command | Result | Time |
|---|---|---|
| `cargo test -p rustmath-finitefields --test small_field` | 4 passed | 2.96 s (3.1 s wall) |
| `cargo test -p rustmath-finitefields --lib small_field` | 3 passed | 0.01 s |
| `cargo test -p rustmath-curves --test biprojective_small_curves` | 12 passed, 1 ignored (`node_of_degree_3_up_to_k6`) | 1.05 s (1.9 s wall; first build after the edits 18.7 s) |
| `cargo test -p rustmath-curves --test m23_zeta` | 8 passed, 8 ignored | 5.53 s (6.6 s wall) |
| `cargo test -p rustmath-curves --lib -- weil residue nodal` | 12 passed (7 weil, 2 residue, 3 differentials residue) | 0.01 s |
| `cargo check -p rustmath-curves --tests --examples` | ok | 10.6 s first, 0.45 s incremental |
| `cargo test --release -p rustmath-curves --test m23_zeta -- --ignored full_both_fibrations_k3` | 1 passed | build 2 min 29 s, test 1.79 s |
| `cargo test --release -p rustmath-curves --test biprojective_small_curves -- --ignored node_of_degree_3_up_to_k6` (extra) | 1 passed | 2.00 s |
| `rm -rf target/release` | done (320 MB) | |
| `cargo clippy -p rustmath-curves --no-deps --lib --examples --test m23_zeta --test biprojective_small_curves` | no warning in any touched file; the `m23_zeta` and `biprojective_small_curves` test targets and the `m23_zeta` example have no warnings at all | log `minors/clippy_curves.log` |
| `cargo clippy -p rustmath-finitefields --tests --no-deps` | no warning in `small_field.rs` or `tests/small_field.rs` | log `minors/clippy_ff.log` |
| `rustfmt --edition 2021` on the 10 touched `.rs` files | applied (only new lines changed), `--check` clean | |

- Free disk before the release build was 12 GB, so no debug executables had to be deleted.
- The reduction is exercised by every count in these runs. The fast `m23_zeta` set asserts the
  `zeta_X.json` values up to k = 6 at 7 and k = 5 at 11. `full_both_fibrations_k3` compares both
  fibrations for p <= 31 at k <= 3. `biprojective_small_curves` asserts the PARI values.

## 4. New line numbers (after the edits, by grep)

| Item | File:line |
|---|---|
| `SmallFq` struct | `rustmath-finitefields/src/small_field.rs:55` |
| `SmallFq::new` | `rustmath-finitefields/src/small_field.rs:138` |
| `count_distinct_roots` | `rustmath-finitefields/src/small_field.rs:654` |
| `count_distinct_roots_any_degree` | `rustmath-finitefields/src/small_field.rs:735` |
| `FrobeniusPolynomial` struct | `rustmath-curves/src/weil.rs:88` |
| `from_point_counts` | `rustmath-curves/src/weil.rs:112` |
| `is_weil` | `rustmath-curves/src/weil.rs:281` |
| `reciprocal_halves_mod2` | `rustmath-curves/src/weil.rs:347` |
| `two_torsion_frobenius_order` | `rustmath-curves/src/weil.rs:396` |
| `plane_point_count` | `rustmath-curves/src/nodal_biprojective/fibres.rs:160` |
| `singular_locus` | `rustmath-curves/src/nodal_biprojective/singular.rs:426` |
| `classify` | `rustmath-curves/src/nodal_biprojective/local.rs:200` |
| `certify` | `rustmath-curves/src/nodal_biprojective/certify.rs:138` |
| `normalization_point_counts` | `rustmath-curves/src/nodal_biprojective/certify.rs:319` |
| `BihomogeneousForm` | `rustmath-curves/src/nodal_biprojective/form.rs:13` |
| `LocalType` | `rustmath-curves/src/nodal_biprojective/local.rs:27` |
| `Certificate` | `rustmath-curves/src/nodal_biprojective/certify.rs:44` |
| `Unresolved` | `rustmath-curves/src/nodal_biprojective/certify.rs:56` |
| `ReductionStatus` | `rustmath-curves/src/nodal_biprojective/certify.rs:79` |

Changed from the old numbers: `count_distinct_roots` 649 to 654; `count_distinct_roots_any_degree`
725 to 735; `two_torsion_frobenius_order` 387 to 396; `plane_point_count` 153 to 160;
`singular_locus` 420 to 426; `certify` 133 to 138; `normalization_point_counts` 308 to 319;
`Certificate` 39 to 44; `Unresolved` 51 to 56; `ReductionStatus` 74 to 79. The rest are unchanged.

## 5. For the report and the handoff

- F-11 can now be narrowed to "an ordinary m-fold point over F_(p^2)". The A_k part (odd k over a
  residue field of degree 2, the degree-3 node, the A15 precision doubling) is tested.
- The new release-only test runs with
  `cargo test --release -p rustmath-curves --test biprojective_small_curves -- --ignored`.
- Docs that cite `biprojective_small_curves` counts: there are now 12 debug tests and 1 ignored
  test (it was 10).

## 6. Fix-up after the pre-commit checks

Date: 2026-11-15. Same worktree and branch, still uncommitted. Inputs: the code check of these
fixes (`scratchpad/m23/impl/minors_codereview1.md`: approved, three minor items) and the docs check
(`scratchpad/m23/impl/minors_docsverify1.md`: six blockers, five minor items). I fixed all of them
and disagree with none. Scratch for this pass: `scratchpad/m23/impl/fixup/`.

### 6.1 Code (the three minor items of the code check)

1. **Overflow of the chunk sum** (`fibres.rs`). The reduction is now
   `.try_reduce(|| 0, |a, b| a.checked_add(b).ok_or_else(overflow))?` (`:134`), and the base
   point at infinity is added with `checked_add` too (`:140`). `overflow` (`:130`) gives
   `CountError::FieldTooLarge { p, k }`. The comment at `:127-129` says why every partial sum is
   at most the count. The `Errors` doc of `plane_point_count` (`:160-163`) adds the case: a count
   above `u64::MAX`, never a wrapped sum. The count is at most `min(dx, dy) (q + 1)`, so this needs
   `min(dx, dy) >= 3` and `q + 1 > 2^64 / min(dx, dy)`. `FormModP` accepts both degrees up to
   4096, so the threshold can be as low as about `2^52`, not only near `2^63`. The doc of the
   variant (`nodal_biprojective.rs:88`) now reads "`p^k >= 2^63` or `k > 8`, or a point count
   above `u64::MAX`". No test reaches it: it needs `q` above `2^52`.
2. **Overflow of the 2-torsion order** (`weil.rs`). `lcm` (`:463`) uses `checked_mul`, and so does
   the bound `divides` (`:415`). Both panic with the message `ORDER_OVERFLOW` (`:461`), "the order
   does not fit in a u64", in debug and in release, instead of the old debug panic and wrapped
   release value. The `# Panics` section (`:392-396`) now names this case, "possible from about
   genus 62 on". New unit test `weil::tests::two_torsion_order_above_u64_panics`
   (`#[should_panic(expected = "does not fit in a u64")]`). It builds the genus-67 polynomial with
   p = 3 whose reduction mod 2 is `f f^rev` for `f = x^31 + x^3 + 1`, `x^19 + x^5 + x^2 + x + 1`
   and `x^17 + x^3 + 1`, with the functional equation `c[2g - i] = 3^(g - i) c[i]`. PARI/GP
   (`two_torsion_overflow.gp`): all three are irreducible and not self-reciprocal, `2^d - 1` is
   prime for each, `factormod` gives degrees `[17, 17, 19, 19, 31, 31]`, and the order
   `(2^31 - 1)(2^19 - 1)(2^17 - 1) = 147572545148221521919` is above `2^64`, while the product of
   the first two is not. The test runs in 0.1 s in debug.
3. **gp scripts outside the repository.** The scripts behind the reference values of
   `tests/biprojective_small_curves.rs` are now in `docs/workstreams/m23_golay/wpf/gp/`, copied
   unchanged, each with its output rerun here (`*.out.txt`, PARI/GP 2.15.4):
   - `small_curves.gp`, `branch.gp` (from `design/WPF/gp/`);
   - `tangent.gp` (from `design/WPF/review1/`);
   - `small_curves2.gp`, `small_curves4.gp` (from `impl/gp/`; 4 reads 2, so run them from `gp/`);
   - `verify.gp` (from `impl/minors/`; `verify.out.txt` is identical to `verify_out.txt`);
   - `two_torsion_overflow.gp` (new, item 2).

   None holds HJLPPZ data: they hold only the small curves, are plain ASCII, and read nothing from
   `$M23ISGALOIS`. The test header (`biprojective_small_curves.rs:7-16`) cites that folder.
   It keeps its 10 lines, so the test line numbers of Section 4 still hold. The in-body citations
   `minors/verify.gp` became `verify.gp`.

   I did not copy `adv2.gp`. `verify.gp` holds the b and c vectors and reproduces every value.
   Also, the branch-class column that `adv2.gp` prints is wrong for the A3 rows. For
   `A3_d2_g1_sq0` it prints 1, while its own loop tag and `verify.gp` give 0 (`h(y0)` is not a
   square in F_169). The cause is `valuation(lift(S^2 H / Hs), lift(br))`, which returns 0 there,
   so `mk` tests `D` itself and not `D / S^4` (`fixup/adv2_sq.gp`). The test asserts `[2]`, after
   `verify.gp`. Its N_2 = 180 agrees with `hyperellcharpoly` and depends on that class. So the
   column of `WPF.rev1/gp/cases.txt` must not be used.

### 6.2 Docs

Blockers of the docs check, all in `PROJECT_TRACKER.md`:
- B1 (SmallFq row): `:654` and `:735`. The Limit now says that both paths have degree <= 4096 (the
  assert at `:744`), documented under `# Panics` (`:649`, `:731`).
- B2 (Frobenius mod l row): Location `weil.rs:347,397`, `two_torsion_frobenius_order (:397)`,
  `order_of_x_mod2 (:469, assert :471)`, `lcm (:463)`. The Limits now say the panic is documented
  (`:388-396`), with the new u64 case. The Tests list adds `two_torsion_order_above_u64_panics`.
- B3 (plane count row): Location `fibres.rs:164`. The cost is stated in the doc (`:154`). The chunk
  sums are combined by `try_reduce` with `checked_add` (`:127-134`), so an overflow gives
  `FieldTooLarge`.
- B4 (normalization counts row): line numbers `:319`, `:138`, `:282`, `:258` and `:270`. The count
  is "12 tests and the release-only `node_of_degree_3_up_to_k6`". The Limit now cites
  `unclassified_e6_points` and the cost doc at `:312`. One clause is added: `GenusInconsistent` is
  a fault in the M23 drivers (`assert_genus_consistent`, minor 4).
- B5 (singular locus row): `PolyField :201`, `QuadExt :296`. The Tests list adds
  `node_of_degree_3_tacnode_of_degree_2_and_a15`, `node_of_degree_3_up_to_k6` and
  `unclassified_e6_points`. The Limits keep only the ordinary m-fold point over F_(p^2) (F-11) and
  cite the cost doc at `:418`.
- B6 (Change Log row): 9 `#[ignore]` (5 `$M23ISGALOIS`, 4 release, one of them in
  `biprojective_small_curves.rs`; base count 9, not 6). The copied files now include the docs
  check, this report and `wpf/gp/`. The row also notes the `ch118.IsWeil` override.

Minor items:
- m1: the `ch118.IsWeil` note now says that the `rust_path` `weil.rs:281` and the inventory status
  `done` in `intrinsics.csv` come from the generated candidate (build.py keeps the generated path
  when an override gives none), so they are the false lead. The status stays `missing`.
  `build.py` is unchanged.
- m2, m4: the wpf README is rewrapped. It says that `WPF_minors.md` lists the new lines of the main
  items and that this section gives the lines moved again. It adds a `gp/` row and an origin table
  for the scripts. The remark that scripts are not copied now excepts `gp/`.
- m3: the HANDOFF Status block names the docs check, the minor-fix report and `wpf/gp/`. The
  Commands add the debug and release-only runs of `biprojective_small_curves`. A sentence records
  the two overflow guards. `fibres.rs:164` and `weil.rs:397` are cited (twice for weil). The
  follow-up paragraph is rewrapped at 100 columns and split in two. Only the existing
  `M23ISGALOIS` command line is longer.
- m5: the CSV notes of `ch114.SingularPoints`, `IsNode`, `IsCusp`, `IsOrdinarySingularity` and
  `NumberOfPlacesOfDegreeOneOverExactConstantField` cite the new tests. The CSV is still 5189 CRLF
  lines with 4 columns. It was written with python `csv` (`newline=''`, `lineterminator='\r\n'`),
  and an unchanged round trip had been byte-identical before the edit.

`python3 tools/magma_tracker/build.py` was rerun. Against HEAD, the same 25 `intrinsics.csv` rows
change as before. Two generator candidates moved with the `weil.rs` edit: `ch114.GCD` is now
`weil.rs:453` and `ch114.LCM` is now `:463`. `STATUS.md` is unchanged by this pass (reviewed 5188,
candidate 3453, missing 8000).

### 6.3 Line numbers moved by this pass

| Item | Before | Now |
|---|---|---|
| `two_torsion_frobenius_order` | `weil.rs:396` | `weil.rs:397` |
| `gcd`, `lcm` | `weil.rs:452`, `:460` | `weil.rs:453`, `:463` |
| `order_of_x_mod2` (assert) | `weil.rs:466` (`:468`) | `weil.rs:469` (`:471`) |
| `plane_point_count` | `fibres.rs:160` | `fibres.rs:164` |
| its Cost doc | `fibres.rs:152` | `fibres.rs:154` |
| chunk reduction | `fibres.rs:127-132` | `fibres.rs:127-134` |

Everything else in Section 4 holds, including `weil.rs:388` (Cost) and `:392` (`# Panics`).

### 6.4 Results

All builds used `-j 2`.

| Command | Result |
|---|---|
| `cargo test -p rustmath-curves --lib -- weil residue nodal` | 13 passed (8 weil incl. the new should_panic test, 2 residue, 3 differentials), 0.10 s; build 20 s |
| `cargo test -p rustmath-curves --test biprojective_small_curves --test m23_zeta` | 12 passed, 1 ignored; 8 passed, 8 ignored (1.24 s and 6.68 s; 16.6 s wall) |
| `cargo test --release -p rustmath-curves --lib --test biprojective_small_curves --test m23_zeta --no-run` | built in 2 min 39 s (12 GB free before) |
| `cargo test --release -p rustmath-curves --lib -- weil::tests` | 8 passed, 0.07 s |
| the same test with the old `lcm` and `s * two` restored for one run, release | `two_torsion_order_above_u64_panics` FAILED ("test did not panic"): the old code wrapped silently. File restored, `cmp` identical |
| `cargo test --release -p rustmath-curves --test biprojective_small_curves -- --include-ignored` | 13 passed (incl. `node_of_degree_3_up_to_k6`), 1.99 s |
| `cargo test --release -p rustmath-curves --test m23_zeta -- --ignored full_` | 3 passed (`full_both_fibrations_k3`, `full_x_k5_31_41`, `full_x_table_k4`), 154 s |
| `M23ISGALOIS=/home/user/external/m23isgalois cargo test --release -p rustmath-curves --test m23_zeta -- --ignored env_` | 5 passed, 34 s |
| `cargo clippy -p rustmath-curves --no-deps --lib --tests --examples` | no warning in a touched file; lib 73 warnings (the baseline) |
| `rustfmt --edition 2021 --check` on the 11 touched `.rs` files | clean |
| `gp -q <name>.gp` for the 7 scripts in `wpf/gp/` | all ran; `verify` identical to `minors/verify_out.txt` |
| grep of the added lines | no `unsafe`; the only em-dash is the old separator of the Puiseux bullet |

Free disk before the release build was 12 GB, so no debug executable was deleted.
`target/release` was deleted afterwards. `WPF_minors.md` is still identical in the scratch
directory and in `wpf/impl/`, with this section appended to both.

## Last edits before the commit

After the second round of checks (code approved; docs one blocker and nine optional items, all
applied): the overflow condition in the `plane_point_count` doc is stated with `m`, the degree of
the counted fibres (`dx` for `FibresInX`, `dy` for `FibresInY`), since a reducible form can exceed
`min(dx, dy) (q + 1)`; the `# Panics` doc of `two_torsion_frobenius_order` says "from genus 64 on"
(the order is below `2^(g + 1)`); the overflow test names its gp script. Each added a doc line, so
the lines move by one more: `plane_point_count` `fibres.rs:165`, `two_torsion_frobenius_order`
`weil.rs:398`, `lcm` `:464`, `order_of_x_mod2` `:470` (assert `:472`). The tracker rows and the
HANDOFF Status block cite the current lines.
