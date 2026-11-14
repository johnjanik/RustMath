# Implementation report: Unit M1a (rustmath-matrix, bounded invariant factors / minimal polynomial / rational form)

Branch `claude/gracious-gates-e6b6l7`, on top of HEAD `fd081fa` (I3, I1, I2, P1, F1, R1 committed).
Design `fu2/design/M1a-matrix-rcf-bound.md` (no revision sections; approved by review 1, whose eight minor
notes m1-m8 are folded in, see below). Nothing committed. Work files and logs: `fu2/impl/M1a-work/`
(`acc_*.log`, `check_ws.log`, `clippy2.log`, `clippy_lib.log`, `adv30.log`, `old_paths.log`,
`old_wbw20.log`, `m2.gp`). No release build was made.

Line numbers below are in the working tree after this unit.

## What changed, per plan step

### Step 1: `rustmath-matrix/src/krylov.rs`
* `Echelon` is `pub(crate)` (`:189`), with `pub(crate)` `new`, `reduce` (`:247`), `push`, `coords`; new
  `len` (`:213`), `truncate` (`:219`), private `reduce_upto` (`:228`, shared by `reduce` and
  `reduce_prefix`), `reduce_prefix` (`:254`). The type docs say why truncation and prefix reduction are
  valid (rows only refer to earlier rows).
* `poly_mat_vec` (`:145`) moved here from `companion.rs` (`poly_at_matrix_times_vector`, unchanged
  Horner body); `companion.rs::image_by_horner` uses it.
* `#[cfg(test)] start_basis` (`:166`) for the `W·B·W⁻¹` tests.
* Module docs: "Combination" bullet (the main path needs no Smith form), "Cost" (no bound claimed for the
  Smith step, which now only runs in the fallback; the peeling handles r = n in O(n⁴)), "Exactness" (the
  three-stage chain, pointer to `frobenius.rs` "Exactness").
* Test `echelon_truncate_and_prefix` (`:604`).

### Step 2: `rustmath-matrix/src/frobenius.rs` (new, 697 lines) and `lib.rs:13`
* Header: MAGMA ch. 26 §26.11, §26.12.2 with the digest path; [Ste97] (factors χ_A); Hoffman-Kunze §7.2,
  Jacobson; [KG85]; Storjohann thesis ch. 9 (§9.4 step 5, Fig. 9.6; his O(n³) algorithm not implemented);
  Bach-Shallit §4.8. Facts (F1)-(F7) with proofs, the algorithm, the verification list, the cost (with
  the coprime-split term, review m7), the tight Θ(n⁴) cases (m3), "Over Q" (reworded, m5), "Exactness"
  (the Z-semantics argument, m1), termination for rounding impls.
* API as designed: `PeelDepth` (`:158`), `PeelLevel` (`:168`), `Peeling` (`:177`), `peel` (`:434`),
  `Peeling::{invariant_factors (:487), minimal_polynomial (:492), generators (:501)}`. Private: `Span`
  (`:183`, `spin` `:221`), `relation_polynomial`, `coefficient_polynomial`, `monic`, `gcd_monic`,
  `exact_quotient` (takes the error text), `coprime_split` (`:310`, cap deg f + 2 gcds), `maximal_vector`
  (`:334`). Ported from the prototype `exp/frob.rs` without `STATS`/`bump`/`spin_timed`; same logic
  (single span with `reduce_prefix`, S1' then S2, certificate, candidate dropping, r = 1 fast path).
  Additions over the prototype: the guard branch checks that the respun degree equals deg f; the r = 1
  path checks `deg R[0][0] == n`; `Span::len` debug-asserts that echelon and stored vectors agree.
* Unit tests: `coprime_split_examples` (`:552`), `merged_vector_has_lcm_annihilator` (`:586`),
  `peel_small_cases` (`:629`, extra: diag(2,2,3), zero 2×2, 1×1, 0×0, non-square, generators refused after
  `MinimalPolynomial` depth), `rounding_field_peel_terminates` (`:665`).
* `grep -n unsafe rustmath-matrix/src/frobenius.rs`: nothing.

### Step 3: `rustmath-matrix/src/companion.rs`
* New private `invariant_factors_peeled` (`:456`), `minimal_polynomial_peeled` (`:462`),
  `invariant_factors_fallback` (`:474`, Krylov Smith then classical), `rational_canonical_form_peeled`
  (`:637`), `certified_form` (`:850`, factored out of `rcf_from_presentation` `:801`, which now calls it).
* Public chains: `invariant_factors` (`:516`) peeled → fallback; `minimal_polynomial` (`:541`) own square
  check (same message as before), n = 0 → 1, peeled → last factor of the fallback;
  `rational_canonical_form` (`:615`) peeled → Krylov → classical.
* Docs: module docs (Algorithm, Complexity with measured times, Fallback), `invariant_factors`,
  `minimal_polynomial`, `rational_canonical_form` (Construction, Fallback text of design 2.2,
  Certificate unchanged, Complexity). The unused `[smith_normal_form]` link definition is gone.
* Tests: see "Tests" below.

### Step 4: `rustmath-matrix/src/polynomial_matrix.rs`
* Header last paragraph (`:19-31`): the design 2.2 text, with the measured times (m7 wording).
* `smith_normal_form` docs (`:181-184`): the sentinels of `PrimeField` and `FiniteFieldElement`.

### Step 5: `rustmath-finitefields/tests/matrix_forms_gfp.rs`
* Comments of `forms_sentinel_permutations_and_jordan` (`:456`) and `forms_sentinel_natural_families`
  (`:590`) name the three stages; 43 is called a lower bound (m2). No expected value changed.
* New `forms_sentinel_peeling_answers_where_the_smith_paths_err` (`:665`, m2), helper
  `start_basis_matrix` (`:684`), optional `forms_fixed_start_basis_worst_case_gf101` (`:700`).

### Step 6 (trackers, HANDOFF)
Not done here by instruction; see Handoffs.

## Tests and reference sources

All gp commands are PARI/GP 2.15 (`/usr/bin/gp`); I re-ran `exp/cd7chk.gp` and `exp/bref.gp` (13 lines,
all 1) and `M1a-work/m2.gp` (the 4×4 and H26E9 values, all 1).

| Test (file) | Reference |
|---|---|
| `krylov::echelon_truncate_and_prefix` | hand-checked linear combinations (u0 + 2u1, u0 + 2u1 + u3) |
| `frobenius::coprime_split_examples` | gp `lcm(...)` (comment) |
| `frobenius::merged_vector_has_lcm_annihilator` | gp `minpoly(matconcat(matdiagonal([...])))` |
| `frobenius::peel_small_cases` | gp `matfrobenius(matdiagonal([2,2,3]),1)` |
| `frobenius::rounding_field_peel_terminates` | none (no accuracy claim) |
| `companion::peeled_forms_conjdiag20_s7_was_81_s` | gp `cd7chk.gp` (3 lines) |
| `companion::peeled_forms_fixed_start_basis_worst_case_n20` | gp `bref.gp` (wbw20) |
| `companion::peeled_forms_r_equals_n_repeated_eigenvalues` | gp `bref.gp` (wbwrep20) |
| `companion::peeled_merges_arrow_n20` | gp `bref.gp` (arrow20); comment on the HEAD time vs the bound (m8) |
| `companion::peeled_multilevel_families` | gp `bref.gp` (derog28, nilp28, scalar40) |
| `companion::peeled_path_over_prime_field_r_equals_n` | gp `bref.gp` (gf101 wbw60) |
| `companion::peel_exact_fields_never_fall_back` | classical Smith oracle (69 over Q, 4×64 over GF(2,3,5,7)) |
| `companion::peeled_generators_are_clean` | structural (f(A)·g = 0, independence) |
| `companion::h26e9_invariant_factors_and_rational_form` (m4) | MAGMA H26E9 output (M1b design §0.1); gp `matfrobenius(a,1)`, `t*a*t^-1 == r` |
| `companion::peeled_sentinel_results_are_the_results_mod_every_p` (m1) | bound GF(2,3,5,7) classical oracle; 45 families + 600 random {-1,0,1} matrices |
| `companion::peeled_sentinel_answers_where_the_smith_paths_err` (m2) | gp `matfrobenius(A,1) = [x^4 - x^3]`, mod p values |
| `matrix_forms_gfp::forms_sentinel_peeling_answers_where_the_smith_paths_err` (m2) | same gp lines, through `check_forms_sentinel` (Z certificate, det T = ±1, GF(2..7)) |
| `matrix_forms_gfp::forms_fixed_start_basis_worst_case_gf101` | gp `matfrobenius(B,1)` with n = 40 mod 101 |

Changes to existing tests (no assertion weakened):
* `check_forms`: adds peeled IF and MP checks; the T comparison now compares the public T with
  `rational_canonical_form_peeled` (design behaviour change 2: for r ≥ 2 the public T is the peeled one),
  and a new separate check that `rational_canonical_form_krylov` is Ok with `A·T == T·C` and the right
  factors. This is the one changed assertion; it asserted HEAD's choice of T, which the design changes on
  purpose (T is not unique; both are certified). The old `invariant_factors_krylov == expected` stays.
* `check_forms_timed` → `check_forms_bounded(a, expected, 60)`, plus `invariant_factors_peeled == expected`.
* `invariant_factors_agree_with_smith_of_xi_minus_a`: the 69 matrices moved to `agreement_matrices()`
  (unchanged list, `assert_eq!(mats.len(), 69)`); adds the peeled factors and its certified T.
* `check_krylov_gfp`, `krylov_vs_classical_gfp`: add the peeled IF, MP, RCF (certified, T invertible); the
  64 matrices moved to `gfp_matrices::<P>()` (same generator).
* `forms_conjdiag20_q_small_r_rcf_timing`: 15 s bound kept, comment updated (peeled: 0.17 s).
* `release_timing_dense_q` (ignored): adds `CONJDIAG20_S7` and `WBW20`.

## Measurements (debug, idle container; tmp harnesses removed after use)

| Input | new IF | new RCF | Smith form of R on this tree (IF / RCF) |
|---|---|---|---|
| CONJDIAG20_S7 (n 20, r 2) | 0.05 s | 0.14 s | 3.2 s / 5.7 s (81-102 s IF at c79a582) |
| CONJDIAG20 (n 20, r 2) | 0.10 s | 0.17 s | |
| W·B·W⁻¹ n 14 / 20 / 30 (r = n) | 0.05 / 0.05 / 0.42 s | 0.07 / 0.10 / 0.78 s | n 14: 3.5 s / 19 s; n 20: 31.5 s / 101 s |
| W·B_rep·W⁻¹ n 20 | 0.18 s | 0.30 s | |
| arrow n 20 / 30 | 0.19 / 1.16 s | 0.27 / 1.43 s | n 20: 0.03 s / 3.0 s |
| derog28, nilp28 (r 7, s 7) | 0.29, 0.32 s | 0.32, 0.68 s | |
| GF(101) Gf W·B·W⁻¹ n 60 | 0.012 s | 0.024 s | 0.26 s (IF) |
| word-size GF(7) 2·I n 40 / 60 / 100 | 0.02 / 0.10 / 0.71 s | 0.03 / 0.11 / 0.75 s | 0.004 / 0.012 / 0.05 s; RCF 0.007 / 0.02 / 0.10 s |
| `PrimeField` GF(7) 2·I n 60 / 100 | 0.45 / 2.5 s | 0.85 / 4.3 s | (review: 0.18 s / 2.4 s at c79a582, n 100) |
| Q 3·I n 40 | 0.10 s | 0.18 s | 0.01 s / 0.10 s |
| review m5 family W·Q·J·Q⁻¹·W⁻¹ n 30, seeds 0..59 | up to 1.82 s | up to 2.61 s | |

The old-path times are much lower than at c79a582 (CONJDIAG20_S7 3.2 s against 81-102 s): the container is
idle now, and the integer arithmetic changed in this round (I3). I did not isolate the cause. The r = n
case is still far beyond the new path (n = 20: 31.5 s / 101 s against 0.05 s / 0.10 s). The review's
7.75 s on its m5 family was not reproduced (max 1.82 s over 60 seeds); the docs give my maximum and say it
is not a ceiling.

## API and behaviour changes; callers checked

* Public signatures unchanged. Crate-internal: `Echelon` and its methods `pub(crate)`, new
  `krylov::poly_mat_vec`, `frobenius` module.
* Values: unchanged over exact fields (unique invariants); `canonical_matrix` unchanged.
* `change_of_basis`: same T for r = 1; for r ≥ 2 another certified T (docs already say T is not unique).
* Speed: the table above; the scalar-like slowdown (r ≈ s ≈ n) is documented in the module docs, the
  `frobenius` cost section and the public docs (m3).
* Z semantics (m2): Err → Ok on inputs whose presentation has unit pivots but whose Smith stages meet a
  non-unit pivot, e.g. the 4×4 `[0,0,0,0; -1,1,0,0; 1,0,0,0; 0,-1,1,0]` (HEAD: Err for all three; now
  `x^4 - x^3`, correct by the m1 argument and checked mod 2, 3, 5, 7). `minimal_polynomial` can answer
  where `invariant_factors` errs (the first level suffices); documented. No Ok → Err change: the old chain
  is still behind the new stage. On the 45 natural families the peeling answers exactly the 19 with a
  unit-pivot presentation; the public count stays 43 (`forms_sentinel_natural_families` passes).
* `Real`: whichever stage passes its checks first answers; no accuracy claim, as before.
* Errors: same types and messages for non-square input; new internal messages are
  `InvalidOperation("internal: ...")`.
* Callers (grep of the workspace, excluding target): no production caller of `invariant_factors`,
  `minimal_polynomial`, `rational_canonical_form` or `RationalCanonicalForm` outside rustmath-matrix (the
  hits in modules, schemes, groups and numberfields are other functions). Test caller:
  `rustmath-finitefields/tests/matrix_forms_gfp.rs` (green). `characteristic_polynomial` (numberfields
  `rnfeq.rs`, jupyter) is untouched.

## Review 1 notes

| Note | Status |
|---|---|
| m1 Z-semantics argument | Done: `frobenius.rs` "Exactness" (both depths), pointer from `krylov.rs`; test on the 19 families plus 600 random matrices (107 answered), IF, MP and RCF (det T = ±1) |
| m2 Err → Ok change | Done: listed above, 4×4 test in companion (Zs) and in matrix_forms_gfp (PrimeField), "43" called a lower bound |
| m3 scalar-like Θ(n⁴) | Done, with n = 100 figures (word-size GF(7) both paths; PrimeField new path, old from the review) |
| m4 H26E9 | Done (`h26e9_invariant_factors_and_rational_form`, Gf<5>; also checks MAGMA's t·a = r·t) |
| m5 growth inside a level | Done: "Over Q" reworded; n = 30 figure of the review's family in the module docs ("not a ceiling") |
| m6 coprime-split bound | Done: (F4) states log2(1 + e_f) + 2 ≤ deg f + 2 gcds, final one included; the code comment says the same |
| m7 wording | Done: bound for the main path only (module docs and each public function), "81 to 102 s" used consistently, coprime-split term in the cost sum |
| m8 arrow margin | Done: comment says HEAD took about 2× the bound, so a pass alone does not show the fix |

Skipped: the optional parts of m3 (a diagonal-R shortcut, peeling in Krylov coordinates). Reason: not
required, and the slowdown stays within the bound and is documented.

## Deviations from the design

1. `peeled_generators_are_clean` lives in `companion.rs` tests, not `frobenius.rs`: DEROG12 and NILP15 are
   private constants of the companion test module. Same content.
2. Extra test `frobenius::peel_small_cases` (n = 0, 1, non-square, depth handling).
3. `cargo fmt -p rustmath-matrix` was not run crate-wide: `companion.rs`, `polynomial_matrix.rs` and many
   other files of the crate were not rustfmt-clean at HEAD, so a crate-wide run would reformat unrelated
   code. I ran rustfmt on `frobenius.rs`, `krylov.rs` (clean at HEAD) and `matrix_forms_gfp.rs` (clean at
   HEAD), and applied by hand only the rustfmt hunks that fall in my new code of `companion.rs`; the
   remaining rustfmt hunks there and in `polynomial_matrix.rs` are all pre-existing.
4. `cargo clippy -p rustmath-matrix --tests` exits with 2 errors at HEAD already (`clippy::erasing_op`,
   deny by default, in `integer_forms.rs:645,670` tests, untouched). No warning in any touched file
   (the one new warning, a range loop in `peeled_merges_arrow_n20`, was fixed). `cargo clippy -p
   rustmath-matrix` (lib) is clean for the touched files.
5. Timings in docs and comments are my measurements on this tree (idle container), not the design's; the
   c79a582 figures are kept where they describe the original defect.
6. The `check_forms` T comparison changed target as the design prescribes (see "Tests").

## Final test counts

| Command | Passed | Failed | Ignored |
|---|---|---|---|
| `cargo test -p rustmath-matrix --no-fail-fast` | 311 (was 295: +16 new) | 0 | 1 |
| `cargo test -p rustmath-matrix --lib peeled -- --nocapture` | 9 | 0 | 0 |
| `cargo test -p rustmath-finitefields --test matrix_forms_gfp --no-fail-fast` | 19 (was 17) | 0 | 0 |
| `cargo test -p rustmath-numberfields --no-fail-fast` | 167 | 0 | 1 |
| `cargo check --workspace --tests` | exit 0 | | |
| `grep -n unsafe rustmath-matrix/src/frobenius.rs` | nothing | | |

## Left undone

* Full `cargo test --workspace` (the orchestrator runs it).
* Trackers and HANDOFF (design section 3), by instruction.
* F1's container hand-off for `Matrix<R>`/`Vector<R>`/`SparseMatrix<F>` binding of unbound constants
  (F1 report, Handoffs) is outside this unit's design; not taken up (M1b binds its own inputs by `x + z`).

## Handoffs

* **Trackers (orchestrator; design section 3)**: `PROJECT_TRACKER.md` rows `:954` (invariant factors and
  minimal polynomial: peeling first, `companion.rs:456/462/516/541`, O((r+1)n³) field operations),
  `:956` (Krylov presentation: also the candidate source of the peeling; `Echelon` crate-shared),
  `:964` (rational form: `rational_canonical_form_peeled` `:637`, `certified_form` `:850`), `:967` (F[x]
  Smith form: fallback only); a new 🟢 row for `frobenius.rs` (`peel` `:434`, `Peeling::generators`
  `:501`, tests listed above). Absent bullets `:978` (minimal polynomial with a bound: now present;
  reword to "no bit-complexity bound over Q") and `:980` (O(n³) Frobenius form, Storjohann: still absent;
  an O(n⁴) bound is in place). Change Log entry.
* `docs/magma/status_overrides.csv`: `ch026.InvariantFactors` (`:3192`), `ch026.MinimalPolynomial`
  (`:3178`), `ch059.MinimalPolynomial` (`:493`), `ch026.RationalForm` (`:3190`): new path and file:line,
  the bound, the new timings; statuses unchanged. Then `python3 tools/magma_tracker/build.py`.
* `docs/HANDOFF.md`: remove the item at `:312-315` (both parts: the bound and the two doc nits); add the
  commit to the "fixed" list. New facts for HANDOFF: the r ≈ s ≈ n slowdown (documented, within bound);
  Err → Ok for some sentinel-only inputs; the old Smith path is much faster on this tree than at c79a582
  (cause not isolated).
* **M1b**: the contracts it uses are kept (`field_constants`, `linearly_independent`, `mat_vec`,
  `div_rem_field`, the documented behaviour of `invariant_factors` and `rational_canonical_form`).
  `poly_at_matrix_times_vector` is now `krylov::poly_mat_vec`. `Echelon` (with `truncate`,
  `reduce_prefix`) and `frobenius::peel`/`Peeling::generators` are available crate-wide; the
  `RationalCanonicalForm::change_of_basis` for r ≥ 2 is the peeled T (columns g, A·g, … of the clean
  generators, increasing factor order).
* **Pre-existing, not this unit**: `clippy::erasing_op` errors in `rustmath-matrix/src/integer_forms.rs`
  tests (`:645`, `:670`) make `cargo clippy -p rustmath-matrix --tests` fail.
