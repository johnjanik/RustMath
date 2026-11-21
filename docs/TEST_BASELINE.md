# Test baseline (2026-11-20, M23 units and S1)

`CARGO_INCREMENTAL=0 cargo test -j 2 --workspace --no-fail-fast` at commit `1b4b450` (the merge of
`main` at `88f9174` into `claude/gracious-gates-e6b6l7` at `fe53bf4`), rustc 1.97.0, 4-core cloud
container shared with another build, atlas data present, incremental compilation disabled. The run
took 66 minutes, including the rebuild of the changed crates.

**Result: 15,058 passed, 0 failed, 376 ignored, across 228 test targets.**

Compared with the 2026-11-15 baseline below (14,791 / 0 / 328, 203 targets), the difference is five
units (commits `f29c492..1b4b450`): WP-F (`a62da12`, `ec17666`) and its follow-up F-9 (`b5299c0`),
the Steiner labelling (`5874caf`) and WP-P (`0e245a7`) of the M23 workstream
(`docs/workstreams/m23_golay/`), and S1, the series engine and literal parser of rustmath-symbolic
(`4763f31`, `a4f7791`, `fe53bf4`; see `docs/workstreams/handoff-followups/`). They added 267 passing
tests, 48 ignored tests and 25 test targets:

| Target | Before | After | Change |
|---|--:|--:|---|
| `rustmath-curves` lib | 299 (+9 ignored) | 314 (+9 ignored) | WP-F +10 (`weil.rs`, nodal residues); F-9 +5 (`nodal_biprojective/irreducible.rs`) |
| `rustmath-curves --test biprojective_small_curves` | (new) | 16 (+1 ignored) | small curves in P^1 x P^1 against PARI/GP: WP-F 12 (+1 ignored), F-9 4 |
| `rustmath-curves --test m23_zeta` | (new) | 9 (+9 ignored) | point counts and Frobenius polynomials of the genus 4 curve X: WP-F 8 (+8 ignored), F-9 1 (+1 ignored) |
| `rustmath-finitefields` lib | 202 | 205 | WP-F: the word-size kernel `SmallFq` (`small_field.rs`) |
| `rustmath-finitefields --test small_field` | (new) | 4 | WP-F: `SmallFq` and `count_distinct_roots` against `FiniteField`, `fp_factor` and brute force |
| `rustmath-finitefields --test galois_ring_frobenius` | (new) | 11 | Steiner: Frobenius matrix, traces and orbit polynomials of Galois rings against PARI/GP |
| `rustmath-coding --test cyclic_golay` | (new) | 7 | Steiner: the two cyclic binary Golay codes of length 23 and their Steiner systems |
| `rustmath-combinatorics --test designs_exact` | (new) | 6 | Steiner: the exact t-design test (`t_design_index`, `is_steiner_system`) |
| `rustmath-galois` lib | 62 (+2 ignored) | 66 (+2 ignored) | Steiner: `integrality.rs` |
| `rustmath-galois --test steiner_labelling` | (new) | 13 (+10 ignored) | Steiner: the certified labelling on Le Lay's CC0 polynomial and small inputs |
| `rustmath-groups` lib | 548 | 558 | WP-P: BSGS enumeration, class functions, representations |
| `rustmath-groups` doctests | 78 (+4 ignored) | 79 (+4 ignored) | WP-P: `StabilizerChain::for_each_element` |
| `rustmath-groups --test m23_atlas` | (new) | 6 | WP-P: the ATLAS character table of M23 against GAP |
| `rustmath-groups --test m23_classes` | (new) | 5 (+2 ignored) | WP-P: M23 on the published generators, its classes and centralizers |
| `rustmath-groups --test m23_nielsen` | (new) | 4 (+3 ignored) | WP-P: the Nielsen classes (C, 23A, 23B), real structures, cohomology over F_2 |
| `rustmath-groups --test m23_three_point` | (new) | 3 (+1 ignored) | WP-P: the 57 three-point class vectors |
| `rustmath-symbolic` lib | 656 (+3 ignored) | 686 (+5 ignored) | S1: engine modules, parser, `expand`, `simplify`, `series.rs` |
| `rustmath-symbolic` doctests | 116 (+13 ignored) | 116 (+11 ignored) | S1: two `ignore` examples of the old `series.rs` removed |
| `rustmath-symbolic --test s1_cuts` | (new) | 16 (+2 ignored) | S1, request C-1: arguments that leave a branch cut (`a4f7791`, `fe53bf4`) |
| `rustmath-symbolic --test s1_depth` | (new) | 14 | S1: parser and consumer depth, `limit_with_error` (review 1, finish round 1) |
| `rustmath-symbolic --test s1_finish1` | (new) | 11 (+1 ignored) | S1: blockers of the finish reviews fin1 |
| `rustmath-symbolic --test s1_fuzz_data` | (new) | 7 | S1: the first 60 inputs of each of review 9's eight random runs, with PARI/GP references |
| `rustmath-symbolic --test s1_payload` | (new) | 2 (+1 ignored) | S1: review 2, result payload |
| `rustmath-symbolic --test s1_regressions` | (new) | 26 (+2 ignored) | S1: B1, B2 and the red team's holes, written before the fixes |
| `rustmath-symbolic --test s1_review2` | (new) | 2 | S1: review 2 minor items |
| `rustmath-symbolic --test s1_review3` | (new) | 7 (+7 ignored) | S1: review 3 blockers and minor items |
| `rustmath-symbolic --test s1_review4` | (new) | 4 (+2 ignored) | S1: review 4 B1 and minor items |
| `rustmath-symbolic --test s1_review4_memory` | (new) | 1 (+1 ignored) | S1: review 4 B2 |
| `rustmath-symbolic --test s1_review5` | (new) | 4 (+2 ignored) | S1: review 5 blockers |
| `rustmath-symbolic --test s1_review5_memory` | (new) | 2 (+2 ignored) | S1: review 5 B2 (peak memory) |
| `rustmath-symbolic --test s1_review6` | (new) | 4 | S1: review 6 minor items |
| `rustmath-symbolic --test s1_singular_points` | (new) | 9 | S1: the singular-point matrix of the inverse functions |
| `rustmath-jupyter` lib | 109 (+1 ignored) | 120 (+3 ignored) | S1: REPL `taylor` (engine, display, consumers, session memory), `expand`, LaTeX work list |

The 50 new ignored tests (48 net of the two removed doctests below) are opt-in runs:

- WP-F and F-9 (10, rustmath-curves): five tests of `m23_zeta` need the external `M23ISGALOIS`
  checkout (`shaowuz/m23isgalois`); four of `m23_zeta` and one of `biprojective_small_curves` are
  release runs (the stated times range from 2 s to 2.5 min).
- Steiner labelling (10, `steiner_labelling`): eight need `M23ISGALOIS` in release; two are release
  runs (the CC0 invariants at the second prime 67, the twelve inert primes below 900).
- WP-P (6): four release runs over the whole group (`full_census`, `full_family_11`,
  `full_c_23a_23b`, `full_57_rows`) and two dictionaries that need external checkouts
  (`env_hjlppz_dictionary` with `M23ISGALOIS`, `env_lelay_dictionary` with `LELAY_M23`).
- S1 (24): 18 release budget tests in the `s1_*` targets (`#[ignore = "release budget test: run
  with --release -- --ignored --test-threads=1"]`), one release timing of the re-parse of a result
  (`b3f1_reparse_of_a_result_within_the_bar`), two release calibrations of the work model
  (`series_const::tests::fold_work_calibration`, `parse_work_calibration`), and three child-process
  helpers that non-ignored tests run in a subprocess (`b2r5_memo_child`, and
  `s1r4_variable_chain_session_child`, `s1r5_session_child` in rustmath-jupyter).

Two ignored doctests went away: the S1 rewrite of `rustmath-symbolic/src/series.rs` dropped the
`ignore` examples of the old `Expr::series` and `Expr::series_with_big_o`. By name, no other test of
the 2026-11-15 run disappeared or changed status. In the files that existed at `f29c492`, 74 of
the 80 removed assertion lines come back unchanged up to `&` and `*` (mechanical: `matches!(&x, ..)`,
`**inner`), and the other six (old series tests that checked `!is_constant()` or compared with the
old `known_series` output) are replaced by exact coefficient checks; no other assertion was removed.

## Previous baseline (2026-11-15, open follow-ups round part 2)

`cargo test --workspace --no-fail-fast` at commit `f29c492`, rustc 1.97.0, 4-core cloud container,
atlas data present, incremental compilation disabled.

**Result: 14,791 passed, 0 failed, 328 ignored, across 203 test targets.**

Compared with the 2026-11-14 baseline below (14,386 / 0 / 333, 194 targets), the difference is eight
more units of the open follow-ups round (commits `4b7f034`..`36c2170` and `f29c492`; see
`docs/workstreams/handoff-followups/`). They added 405 tests and nine test targets:

| Target | Before | After | Change |
|---|--:|--:|---|
| `rustmath-finitefields` lib | 164 | 202 | proven characteristic, `Gfpn::try_new`, group orders, field square roots (`4b7f034`) |
| `rustmath-finitefields --test field_sqrt` | (new) | 9 | `IsSquare` and `SquareRoot` in GF(q) against gp |
| `rustmath-finitefields --test unit_group_orders` | (new) | 6 | orders and `FactoredOrder` against gp `znorder` and `fforder` |
| `rustmath-finitefields --test unit_group_orders_capped` | (new) | 1 | the capped Ring hook |
| `rustmath-finitefields --test unbound_constants` | (new) | 10 | unbound constants in polynomial containers |
| `rustmath-finitefields --test galois_ring_sentinel` | 16 | 17 | |
| `rustmath-finitefields --test sentinel_eq` | 7 | 9 | |
| `rustmath-finitefields --test matrix_forms_gfp` | 17 | 27 | peeled rational form, primary forms, Jordan form over GF(p) (`d3a133b`, `dc367a2`) |
| `rustmath-rings` lib | 1726 | 1787 | exact p-adic sentinels, field certificates, ring predicates (`fd081fa`) |
| `rustmath-category` lib | 152 | 160 | ring predicates of concrete parents |
| `rustmath-powerseries --test ring_predicates` | (new) | 6 | power series ring hooks |
| `rustmath-schemes` lib | 238 (+13 ignored) | 245 (+7 ignored) | six elliptic-curve p-adic tests un-ignored, one new (`fd081fa`) |
| `rustmath-matrix` lib | 254 | 289 | vector peeling, primary forms, multi-RHS solve (`d3a133b`, `dc367a2`) |
| `rustmath-combinatorics` lib | 1485 | 1509 | checked machine-word counting, factorials, HRR partition numbers (`2e000c9`) |
| `rustmath-combinatorics --test ch144_counting` | (new) | 14 | PARI fixtures for the counting functions |
| `rustmath-combinatorics --test ch145_number_of_partitions` | (new) | 6 | 71 PARI rows of p(n) and standard tableaux |
| `rustmath-combinatorics --test machine_word_counts` | (new) | 53 | overflow refusals and exact values |
| `rustmath-jupyter` lib | 64 | 109 (+1 ignored) | parser, conversions, growth guards, display budget, interrupt (`bd03278`, `36c2170`) |
| `rustmath-jupyter --test kernel_protocol` | (new) | 4 | the kernel's interrupt and limit replies |
| `rustmath-manifolds` lib | 258 | 308 | exact d, wedge, interior product, Lie derivative, zero test, checked counts (`f29c492`) |
| others | | +15 | doctests: combinatorics +2, finitefields +1, manifolds +2, matrix +1; polynomials lib +4, finitefields `ring_predicates` +1, algebras +1, core +1, numberfields +1, rationals +1 |

One `#[ignore]` test was added: `test_sqrt_mod_refusal_release` in rustmath-jupyter, a timing check
for release builds that its debug counterpart covers. Six ignore markers were removed in
rustmath-schemes (the elliptic-curve p-adic tests that panicked in the old static p-adic `one()`).
No existing assertion was weakened (each unit was reviewed for that before its commit).

## Previous baseline (2026-11-14, open follow-ups round part 1)

`cargo test --workspace --no-fail-fast` at commit `922024b`, rustc 1.97.0, 4-core cloud container,
atlas data present, incremental compilation disabled.

**Result: 14,386 passed, 0 failed, 333 ignored, across 194 test targets.**

Compared with the 2026-11-08 baseline below (14,173 / 0 / 330, 190 targets), the difference is the
first four units of the open follow-ups round (commits `80130e7`..`166ac34`; see
`docs/workstreams/handoff-followups/`). They added 213 tests and four test targets:

| Target | Before | After | Change |
|---|--:|--:|---|
| `rustmath-integers` lib | 174 | 233 | Lehmer gcd, exact roots, binomial, sigma, BPSW, ECPP, sieve, SIQS (`80130e7`, `7c4d057`, `0ceaeaa`) |
| `rustmath-integers --test ch18_primality` | (new) | 18 (+2 ignored) | PARI certificates, pseudoprimes, tampered certificates |
| `rustmath-integers --test ch18_quadratic_sieve` | (new) | 12 (+1 ignored) | SIQS against gp `factor`, 20 to 45 digits and named inputs |
| `rustmath-integers --test ch18_limited_factorization` | (new) | 14 | `factorization_with_limits`, MAGMA's third return value |
| `rustmath-integers --test ch18_signs_roots` | 21 | 38 | gp fixtures for isqrt, ispower roots, Lehmer xgcd, big binomials |
| `rustmath-integers --test ch18_residues` | 26 | 30 | residues with the BPSW test |
| `rustmath-integers --test ch18_factorization_drivers` | 6 | 9 | the drivers split psi_12 |
| `rustmath-integers` doctests | 38 | 45 | examples for the new APIs |
| `rustmath-polynomials` lib | 314 | 350 | checked `fp_factor` entries, PARI order, typed Zassenhaus errors (`166ac34`) |
| `rustmath-polynomials --test ring_predicates` | (new) | 10 | polynomial Ring hooks |
| `rustmath-groups --test pgalois` | 3 | 15 | prime check, primitive part, root at infinity |
| `rustmath-core` lib | 41 | 49 | i32/i64 gcd signs, `static_is_commutative` |
| others | | +13 | finitefields `prime_residues` +1, polynomials `ch23_berlekamp` +1 and `ch23_is_irreducible` +1, rationals +1, rings +1, numberfields +1, symbolic +6, jupyter +1 |

Three `#[ignore]` tests were added, all opt-in runs whose behaviour non-ignored tests also cover:
`ecpp_1024_release` (a 1024-bit proof timed in release), `print_certificates_for_pari` (prints
certificates for a one-off `primecertisvalid` check) and `qs_factors_gp_semiprimes_50_to_60_digits`
(about 30 s in debug, 4 s in release). No ignore marker was removed, and no existing assertion was
weakened (each unit was reviewed for that before its commit).

## Previous baseline (2026-11-08, final fixes)

`cargo test --workspace --no-fail-fast` at commit `9c24389`, rustc 1.97.0, 4-core cloud container,
atlas data present, incremental compilation disabled.

**Result: 14,173 passed, 0 failed, 330 ignored, across 190 test targets.**

Compared with the earlier 2026-11-08 baseline below (14,093 / 0 / 330, 189 targets), the difference is
the final follow-up fixes (commits `f0682b7`..`9c24389`; see `docs/HANDOFF.md`). They added 80 tests
and one test target:

| Target | Before | After | Change |
|---|--:|--:|---|
| `rustmath-polynomials` lib | 299 | 314 | `fp_factor` equal-degree factorization (`f0682b7`): PARI `factormod` rows, pinned output orders, panics on contract violations |
| `rustmath-finitefields` lib | 129 | 164 | Cantor-Zassenhaus certificates, square-free and characteristic checks, the uniform sampler (`7ce8117`) |
| `rustmath-integers` lib | 163 | 174 | factorization drivers: B-04, the ECM double push, the rho fallback (`6adaba2`) |
| `rustmath-integers --test ch18_factorization_drivers` | (new) | 6 | the same drivers against PARI `factor` |
| `rustmath-integers` doctests | 37 | 38 | a new example in the `factor_pollard_rho` docs (`prime.rs`) |
| `rustmath-jupyter` lib | 51 | 63 | REPL integer arguments and limits, T-250 (`874258b`) |

No `#[ignore]`, `ignore` or `no_run` marker was added or removed, and no existing assertion was
weakened (each unit was reviewed for that before its commit).

## Previous baseline (2026-11-08)

`cargo test --workspace --no-fail-fast` at commit `702d0a5`, rustc 1.97.0, 4-core cloud container,
atlas data present, incremental compilation disabled.

**Result: 14,093 passed, 0 failed, 330 ignored, across 189 test targets.**

Compared with the 2026-11-07 baseline below (13,950 / 0 / 330, 184 targets), the difference is the
follow-up fixes (commits `89b1c4e`..`702d0a5` and `14a05a2`; see `docs/HANDOFF.md`). They added 5
test targets: `ch23_berlekamp.rs` and `ch23_is_irreducible.rs` (polynomials), `ch18_signs_roots.rs`
(integers, with PARI/GP grids under `tests/data/` generated by `gen_signs_roots.gp`), and
`prime_residues.rs` and `galois_ring_sentinel.rs` (finitefields). One doctest in
`rustmath-polynomials/src/factorization.rs` that was marked `ignore` now runs. One `#[ignore]` was
added: `release_timing_dense_q` in `rustmath-matrix/src/companion.rs`, an opt-in timing report for
release builds, whose reference values are also checked by non-ignored tests. The ignored total is
unchanged.

An intermediate run at `06945db` had one failure, `rustmath-interfaces`
`util::tests::test_object_registry`: two tests raced on a global registry. Commit `14a05a2`
serialises them with a test-only lock; the assertions are unchanged.

## Previous baseline (2026-11-07)

`cargo test --workspace --no-fail-fast` at commit `8e25fe2`, rustc 1.97.0, 4-core cloud container,
atlas data present, from a clean target dir with incremental compilation disabled
(`~/.cargo/config.toml`, to stay inside the session's disk allowance).

**Result: 13,950 passed, 0 failed, 330 ignored, across 184 test targets.** The run took 37 minutes,
including the full build.

Compared with the 2026-11-06 baseline below (13,772 / 0 / 332, 174 targets), the difference is the
L0-L1 defect fixes (commits `8b29b5a`..`8e25fe2`; see `docs/magma/REVIEW_2026-11-06.md`). They added
10 test targets: `ring_predicates.rs` in five crates (core predicates per type), `ch18_residues.rs`
(integers, with PARI/GP grids under `tests/data/`), `square_roots.rs` (finitefields),
`ch23_quotrem_gcd.rs` (polynomials), `parent_finiteness.rs` (matrix) and `matrix_forms_gfp.rs`
(finitefields, Smith/rational forms over GF(p)). Two doctests that were marked `ignore` now run; no
`#[ignore]` or `ignore` marker was added. Each fix was reviewed for weakened assertions.

## Previous baseline (2026-11-06)

`cargo test --workspace --no-fail-fast` at commit `e0a9c23`, rustc 1.97.0, 4-core cloud container,
with the gitignored degree-24 atlas data present (`rustmath-groups/data/`, built by the environment's
setup script from GAP).

**Result: 13,772 passed, 0 failed, 332 ignored, across 174 test targets.** Every workspace crate and
test target compiles. The run took 53 minutes, starting from a partly built target dir.

## What changed since the 2026-11-05 baseline

The 2026-11-05 run (below) had 60 failures: 57 with the atlas data present, plus 3 environment
failures that disappeared once the data was built. Building the data exposed one more: a test that
asserted the generator atlas was absent. All 58 were fixed in 13 commits (`eb50f9b`..`e0a9c23`).
Each crate's fix was reviewed by an independent agent before it was committed. The review checked
for weakened assertions, unproven changes to expected values, and facades.

| Target | Before (pass/fail) | After (pass/fail) | Cause, in short |
|---|--:|--:|---|
| `rustmath-groups` lib | 468 / 18 | 548 / 0 | Code bugs, not stale tests. The libgap layer used the context-free `E::identity()`. The semidirect product multiplied `i32` components with `*`. `merge_elements` used the wrong exponents. Nilpotent Lie elements had a dimension-1 identity. About 20 facades are now honest or implemented (`contains` was `true` by default, `is_simple` was true for every group of order > 2, every dim >= 3 nilpotent group got the Heisenberg bracket). Reference values come from GAP. |
| `rustmath-groups` doctests | 64 / 8 | 78 / 0 | Stale examples, plus real bugs found while writing their assertions: `ArtinGroup::new` stored garbage relators, `misc::AdditiveAbelian` panicked on Z/2+Z/3+Z/5, `power` overflowed one squaring early, and the Mathieu, Alternating and Cactus constructors returned the wrong groups. |
| `rustmath-groups --test perm23_classifier` | 10 / 1 | 11 / 0 | The test assumed the generator atlas was absent. It now checks the real data when the file exists (25,000 groups; 24T1, PSL(2,23) = 24T7817, M24 = 24T24680, values from GAP) and checks the graceful error when it does not. |
| `rustmath-quivers` lib | 48 / 2 | 63 / 0 | `cartan_companion` computed BᵀB, not the Fomin-Zelevinsky companion. `type_d` used a non-standard orientation. `type_e` built A_n. `mutation_type` was a facade that returned Finite for almost every input. It is now a theorem-based classifier that returns `Err` when it cannot decide. |
| `rustmath-lieconformal` doctests | 19 / 13 | 32 / 0 | The examples did not import the trait. Writing real assertions exposed wrong ghost brackets (skew-symmetry) and `∂` implemented as a formal derivative. Several `lambda_bracket` methods returned empty maps; they now panic honestly. |
| `rustmath-trees` doctests | 29 / 5 | 34 / 0 | Stale examples. |
| `rustmath-numbertheory` doctests | 6 / 4 | 11 / 0 | Stale examples. `bernoulli_multimodular` was a facade for n > 20 and now implements the real multimodular algorithm. The i64 overflows in `bernoulli_number`/`bernoulli_mod_p` are fixed. `verify_bernoulli_mod_p` had the wrong formula. |
| `rustmath-modules` doctests | 4 / 2 | 6 / 0 | The examples used `num_bigint::BigInt`, which is not a `Ring`. The `echelon_form`, `span_basis` and `are_linearly_independent` facades are now honest. |
| `rustmath-algebras` doctests (`full` feature) | 110 / 2 | 112 / 0 | Type annotations were missing. Only a workspace build enables `full` (through `rustmath-jupyter`), so `cargo test -p rustmath-algebras` alone never showed these failures; use `--features full`. |
| `rustmath-stats` doctests | 5 / 1 | 6 / 0 | No type in the workspace could call `std`. It now works for `Real`, and for `Rational` with an exact root. |
| `rustmath-misc` doctests | 13 / 1 | 14 / 0 | A tracked, non-empty `output_dir` made `rename` fail. The doctests now write under the system temp dir and no longer modify `output.txt` or leave `.tmp_dir_*` behind. |
| `rustmath-curves` lib | 296 / 1 | 299 / 0 | The result was not reproducible: a rayon `fold`/`reduce` re-associated the high-precision sums differently on each run, and the free phase of the Jacobi singular vectors amplified that noise from about 1e-78 to about 1e-9. The assembly is now deterministic and the phase is canonical. The 1e-30 assertion is unchanged, and the two runs now agree exactly. |

Ignored: 334 → 332. The `#[ignore]` on `verify_bernoulli_mod_p`'s test was removed because the
formula is fixed. No `#[ignore]`, `ignore` or `no_run` was added.

Facades and other defects found, fixed or only recorded, are listed per row in
`PROJECT_TRACKER.md` (Change Log, 2026-11-06).

## Reproduce

```bash
cargo test --workspace --no-fail-fast 2>&1 | tee test_all.log
grep -E '^test .* FAILED$' test_all.log
grep -E '^test result:' test_all.log | awk '{p+=$4; f+=$6; i+=$8} END {print p, f, i}'
```

The first build compiles GMP/MPFR from source (`rug`), which takes several minutes. A cold full run
takes about an hour in a 4-core container. For one crate, use `cargo test -p <crate>`, but note
feature unification: `rustmath-algebras` needs `--features full` to match the workspace build.

## Previous baseline (2026-11-05)

At commit `9c7e23f` (belyi + integrate-lava-galois + main): **13,586 passed, 60 failed, 334
ignored.** Failures: `rustmath-groups` lib 18, doctests 8, `galois_narrowing` 3 (atlas data
absent); `rustmath-lieconformal` doctests 13; `rustmath-trees` 5; `rustmath-numbertheory` 4;
`rustmath-quivers` lib 2; `rustmath-modules` 2; `rustmath-algebras` 2; `rustmath-curves` lib 1;
`rustmath-stats` 1; `rustmath-misc` 1.
