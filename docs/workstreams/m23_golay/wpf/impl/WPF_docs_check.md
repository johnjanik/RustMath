# WP-F documentation step: independent verification 1

Date: 2026-11-15. Object: the uncommitted documentation edits in `/home/user/RustMath-wpf`
(branch `claude/m23-wpf`, HEAD `aa485d0`) described in `scratchpad/m23/impl/WPF_docs_report.md`:
`PROJECT_TRACKER.md`, `docs/magma/status_overrides.csv` with the generated `intrinsics.csv` and
`STATUS.md`, `docs/workstreams/m23_golay/HANDOFF.md`, and `docs/workstreams/m23_golay/wpf/`.
No file in the worktree was edited. Scratch: `scratchpad/m23/impl/docsverify1/`. The refreshed
Section 2 under `scratchpad/m23wt/` does not exist, so the worktree's Section 2 was used.

## Verdict

**Approve.** No blockers. Every new or changed row matches the code and the tests. The counts
moved by exactly the deltas. The CSV kept CRLF and minimal quoting. The `build.py` output is
current. The Status block in HANDOFF.md is true. There are no em-dashes in the new prose. Six minor
items follow (Section 3). They can be fixed in the same commit, before the push.

## 1. What I checked, and how

### 1.1 PROJECT_TRACKER.md (7 new rows, tallies, scorecard, registry, metrics, change log)

- Every `file:line` cited in the 7 rows was read in the code. All of them point at the named item:
  - `small_field.rs`: :55, :138, :162, :376, :414, :420, :437, :443, :453, :649, :725.
  - `weil.rs`: :88, :112, :156, :163, :203, :209, :246, :251, :281, :319, :329, :340, :347, :387,
    and :459, the assert inside `order_of_x_mod2`.
  - `fibres.rs`: :14, :127, :153.
  - `singular.rs`: :35, :420.
  - `local.rs`: :27, :200.
  - `certify.rs`: :133, :253, :265, :277, :308.
  - `form.rs`: :13, :112, :175, :197.
  - `residue.rs`: :196, :291.
- The described behaviour matches the code. Points checked:
  - the Sturm test on R and on `S = E^2 - y O^2` over `(-1, 4p]`;
  - `NonIntegral` and `FunctionalEquation`;
  - `DeterminedByP` and `Unresolved { semisimple_order, divides }`;
  - the certificate logic: `Decided` only when every point is a node and there are p_a - g of them;
    `DeltaSumCertified`, `Unclassified`, `DeltaMismatch` (lower bound) and `GenusInconsistent`;
    `AssumedUnibranch` never with `Decided`;
  - chunks of 2^14 on rayon, and the `FieldTooLarge` limits (odd p < 2^16, k <= 8, p^k < 2^63);
  - the polar-curve precision, 16 doubling to 128;
  - the assert at degree 4096, and the panic of `order_of_x_mod2` above degree 62;
  - Conway or least irreducible in `new`, and the irreducibility check in `with_modulus` by
    `fp_factor`.
- Every test name cited exists. These were run here in debug and pass:
  - `m23_zeta`: 8 passed, 8 ignored;
  - `biprojective_small_curves`: 10 passed;
  - `weil` unit tests: 7 passed;
  - `residue` unit tests: 2 passed.

  The 8 opt-in tests pass in the implementer's logs (`env_tests.log`, `full_tests.log`). The test
  contents match the parentheticals, for example the K and p grid of
  `arithmetic_matches_finite_field`, the cases of `exact_weil_test_edge_cases`, and `ellap = 1` for
  `y^2 = x^3 + 3x + 5` at 7 (rerun in gp).
- The review's minor items 1, 2 and 3 are carried into the Limits of the right rows.
- Recount, by my own script (`docsverify1/recount.py`, run on the file before and after the edit):
  - Every section's Tally equals its row count. Before the edit the totals were
    624/331/281/84/72 = 1392; after, 631/331/281/84/72 = 1399.
  - Curves 🟢 10 -> 16, finitefields 🟢 23 -> 24.
  - The Module Registry totals are 631/331/281/84/72. CURVES 16/4/4/3/0, FINITEFIELDS 24/2/0/0/0.
  - The shares stay 45/24/20/6/5 %. Board Metrics: 1399, 631, defects 365 = 281 + 84.
- ASCII order within the 🟢 blocks holds. The Change Log row is at the top, and `updated:` is
  2026-11-15.

### 1.2 docs/magma

- `status_overrides.csv` has 5188 lines, all CRLF, and no bare CR. A `csv` round trip with
  `lineterminator='\r\n'` reproduces the file byte for byte, so minimal quoting is kept.
- The 8 new rows are all ch. 114, all `partial`. Their ids exist in `intrinsics.csv`, and `.2` is
  the `(C, m)` signature. Their notes agree with the code: the IsNode, IsCusp,
  IsOrdinarySingularity and Multiplicity mappings from `LocalType`, `counts[m - 1]`, and
  `frobenius_polynomial` then `l_polynomial`. Every test they cite exists.
- `build.py` was rerun on a tar copy of the worktree (`docsverify1/copy`). All 11 files in
  `docs/magma/` come out byte-identical, so the generated output is current.
- `intrinsics.csv` changed in 105 rows:
  - 8 rows changed status, candidate or missing to partial: the overrides.
  - `ch118.IsWeil` went from missing to candidate at `weil.rs:281`.
  - The other rows changed only `rust_path` or `match_scope`. These are the generator's new
    candidate paths at `weil.rs:187`, `:443`, `:451`, `residue.rs` and others, as the report
    says.
- `STATUS.md` reproduces the report's totals:
  - reviewed 5179 -> 5187, partial 1242 -> 1250, candidate 3455 -> 3454, missing 8006 -> 7999;
  - ch. 114: 8 reviewed, 8 partial, candidate 160 -> 158, missing 154 -> 148;
  - ch. 118: candidate +1, missing -1.

### 1.3 HANDOFF.md Status block (104 lines added, nothing removed)

- Every `file:line` and test name in it is correct (see 1.1).
- The tables:
  - The X table in `wpf/impl/WPF.md` 3.1 gives Decided at 7, 11, 17, 19, 29, 37, 43, 47, 53, 59,
    61, 67, 71 and 79 (14 primes), and Unresolved with one A2 at 13, 31, 41 and 73.
  - p = 5: 60 nodes and an A3 with conjugate branches, counts 7, 37, 169, 561.
  - New X rows: 83 to 131 are Decided except 107 (an A3 with rational branches).
  - L: (7, 5), (19, 3), (41, 8) and (43, 15) are Decided. (31, 12) gives 43, 1017, 30265, 923729
    with a = (11, 88, 682, 4438).
  - The 18 new L rows are Decided, and (79, 15) and (103, 93) are Unresolved.
  - Refusals: X at 3 and 23; L at (5, 3), (11, 2), (23, 5) and (23, 15).
  - The anchor at 53 is 52, 57, 45, all Decided.
- Independent PARI/GP check (`docsverify1/rows.gp`, output `rows_out.txt`). It covers all 54
  counted rows of the report tables, 29 of X and 25 of L. For each row, P from N_1..N_4 by
  Newton's identities is integral and equals the tabulated a. All 8 roots have
  `| |z|^2/p - 1 | < 1e-40` at 200 digits. And `P(chi) = P(chi p) = 0 mod 23`.
- `P mod 2` is squarefree exactly at X 11, 31, 79 and 97, and at L (53, 41) and (67, 15). This
  matches the 2-torsion paragraph: 15 is Decided at the Decided rows, and conditional at X 31.
- The vendored data: `zeta_X.json`, `zeta_L.json` and `h62.json` are byte-identical to
  `johnjanik/m23` at `d05475b`. `lelay_optimal_23_4_Z.json` is byte-identical to Le Lay's
  `data/optimal_23_4_Z.json`. The sha256 values in the data README are right.
- The commands and the example's arguments match `examples/m23_zeta.rs`.

### 1.4 docs/workstreams/m23_golay/wpf/

- The four copied documents are byte-identical to the scratch originals (`cmp`).
- The README index is accurate. Both review verdicts are as stated: the design was approved with
  D1 rejected, and the code was approved with six minor items.
- Licence: the 1129 distinct integers of 10 or more digits in the files of `m23isgalois`, and every
  20-digit substring of them, were searched for in all 34 changed or untracked files of the
  worktree. There is no hit.

### 1.5 Em-dashes

- New prose has no em-dash or en-dash in the tracker rows, the CSV notes, the Status block or the
  wpf documents.
- The one added tracker line with an em-dash is the absent bullet "Puiseux expansions ...". That
  dash is the pre-existing separator of every absent bullet. The text appended after it has none.

## 2. Not checked

- I did not rerun the opt-in tests (`env_`, `full_`), the release example or the whole
  `rustmath-curves` and `rustmath-finitefields` suites. I relied on the implementer's logs and on
  the code review for those. The PARI/GP check above covers the published numbers.

## 3. Minor items (none blocks approval)

1. **Provenance tag of the 8 new CSV rows.** They end in `[review 2026-11-06; partial in
   nodal_biprojective, ...]`. The README (`docs/magma/README.md`, Review status) says that tag
   records the provenance of the 2026-11-06 review. These rows were not part of that review. The
   precedent for later additions is `[added 2026-11-14; fixed <sha>]` (18 rows). Required change:
   use `[added 2026-11-15; partial in nodal_biprojective, curves in P^1 x P^1 over F_p, WP-F]`
   (and the `weil.rs` variant for LPolynomial). Then rerun `build.py`.
2. **Wrong section for IsSquare.** The finitefields row cites "MAGMA ch. 21 §21.4.5 IsSquare". In
   the digest `IsSquare` is in §21.4.4 (Predicates on Ring Elements), and §21.4.5 is the minimal and
   characteristic polynomial. The code header `small_field.rs:5` has the same slip. Required
   change: write "§21.4.4 IsSquare" in the row, and fix the code header too if the code commit is
   still open.
3. **Curves `#[ignore]` count.** The delta (+8) is right, but the base was stale. Before WP-F,
   `rustmath-curves` already had 9 `#[ignore]` attributes: `hypergeometric.rs:652`,
   `kernel_refine.rs:2459`, `modular_forms_hp.rs` x 5, `genus0_map.rs:1058` and `mp_svd.rs:1511`.
   The implementer's lib run shows "9 ignored". So the tally's 14 should be 17. Required change:
   write `#[ignore]` × 17 (8 of them the opt-in M23 tests ...). Optionally, note in the Change Log
   that the base 6 was stale.
4. **Status block, derived checks.** It says the derived checks were run "on every counted row" and
   cites `reference_polynomials_pass_the_derived_checks`. That test checks the reference rows of
   `zeta_X.json`. The counted rows are checked by `assert_matches_table` (in
   `x_at_7_k6_and_11_k5_nodal`, `full_x_table_k4` and `env_l_table`) and by the example for the new
   rows. Required change: cite those as well. The claim itself is true (my PARI run confirms it on
   all 54 rows).
5. **Status block, last paragraph.** "`p < 2^16` and `k <= 8`" leaves out "odd p" and
   "`p^k < 2^63`", which the code enforces (`CharacteristicTwo`, `FieldTooLarge`) and the tracker
   rows state. The Code bullet also omits `p^K < 2^63` for `SmallFq`. Required change: add both
   limits.
6. **`ch118.IsWeil` (optional).** It is now a candidate at `weil.rs:281` with `inventory_status`
   done. That function tests Weil polynomials, not Weil divisors on toric varieties. The report
   acknowledges this and adds no override. That is acceptable, since a candidate is a lead.
   Optional change: add an override `ch118.IsWeil,missing,,` with a note that `weil.rs:281` is the
   Weil test of a Frobenius polynomial, so the false lead does not carry a "done" inventory status.

Two conflicts are recorded in the Change Log and left for the owner, as is right for a docs step:
the new `#[ignore]`s against the Definition of Done, and B-06 (`rustmath-curves` off-limits).
