# Implementation report: Unit I2-integers-factorization

Design: `fu2/design/I2-integers-factorization.md` (Revisions 1 and 2 apply; revised body
authoritative), reviews 1 to 3 (review 3 approved). Base: HEAD `7c4d057` (I3 `80130e7` and I1
`7c4d057` on top of `c79a582`). No uncommitted changes existed at start. Nothing committed. All
five items are implemented; nothing is declared out of scope. No `rustmath-misc/output.txt` or
`.tmp_dir_*` change; `target/release` was deleted after the timing runs; the two temporary timing
harnesses (`tests/zz_tmp_*.rs`) were deleted (copies in `scratchpad/i2impl/`).

Handoffs from earlier reports: I1 hands I2 "use `is_probable_prime` for splitting decisions and
`try_is_prime` to certify factors; `sieve::for_each_prime` has the I2 contract (no survivor test
below 2^40); the QS psi12 test is not ignored; `prime::factor` / `prime_divisors` unchanged;
`fast_arith::pow_mod_u64` is `pub(crate)`". All used as stated. I3 has no handoff to I2; its
`quadratic_sieve.rs` edits were inside functions this rewrite deletes (design §6), so nothing of
them survives. Design assumptions about I1 hold at HEAD: `prime::{is_probable_prime, try_is_prime,
is_prime, is_prime_u64}`, `sieve::{for_each_prime, small_primes}`, `primality::PSI_12`.

## Per item: what changed

### Item 1: the quadratic sieve (SIQS)

- `src/quadratic_sieve.rs` (rewritten, 501 lines): module doc (MAGMA ch. 18 §18.10.3 / §18.10.1,
  Pomerance 1985, Silverman 1987, Alford-Pomerance 1995, Contini 1997, Crandall-Pomerance §6.1,
  Knuth §4.5.4, Cohen 1.5.1 / §10.4; algorithm; measured cost); `QsParams` (`:82`, the consts,
  `for_bits` `:139`, `validate` `:173`); `QsOutcome` (`:214`); `quick_paths` (`:231`, step 0);
  `quadratic_sieve_factor` (`:283`), `quadratic_sieve_factor_with_params` (`:333`, now
  `(n, &QsParams) -> Result<QsOutcome>`), `quadratic_sieve_factor_complete` (`:396`, now
  `limited_engine(n, 47, {rho 1, 1, 8191, qs Always, complete}, true)`, panics on an unresolved
  proof with the design's message naming `factorization_with_limits`).
- `src/quadratic_sieve/siqs.rs` (new): port of `$X/rev2/siqs_r3.rs` split as in design §1.4:
  word helpers (`mul_mod_u64`, `inv_mod_u64`, `sqrt_mod_u64` `:83`, `mod_small`, `splitmix64`,
  `log2_big`), `multiplier` (`:174`, 30 multipliers, first 300 odd primes from
  `sieve::small_primes()`), `FactorBase` / `BaseScan` / `factor_base` (`:238`, over
  `sieve::for_each_prime(3, 2^31)`), `ASelector` / `a_selector` (`:317`, the a-supply guard) /
  `choose_a` (`:362`, partial Fisher-Yates, 1000 repeats), `Polynomial` / `Roots` /
  `init_polynomial` (`:446`) / `next_polynomial` (`:508`, Gray code), `sieve` (`:546`),
  `sieve_init`, `large_prime_bound`, `Relation` / `check_candidate` (`:622`, exact check always
  asserted), `Added` / `RelationStore::add` (`:737`: duplicate test, cap test, insert |y|, store;
  cap 16 F), `parity_rows`, `square_roots` (`:806`, X^2 = Y^2 asserted), `RunStats`, `run` (budget
  checked before every polynomial and before every draw, stop at R rows inside the candidate loop).
- `src/quadratic_sieve/gf2.rs` (new): `left_nullspace` (dense elimination with history; the
  correctness argument is in the header).
- `src/lib.rs:32`: re-exports `QsOutcome, QsParams`.
- Removed (design 1.5): `legendre_symbol`, `tonelli_shanks` (ported as `sqrt_mod_u64`),
  `generate_factor_base`, `trial_factor`, `sieve_for_smooth_relations`, `build_exponent_matrix`,
  `gaussian_elimination_gf2`, `try_dependency`, `SmoothRelation`.

### Item 2: `trial_division_with_primes`

- `src/factorint.rs:436`: returns `LimitedFactorization`; no early break; each dividing entry is
  classified (`classify(p, true)`, composite entry panics "the list entry {p} is not prime"); the
  cofactor is classified with `classify(m, false)`; entries < 2 panic up front. Module doc
  (`:16`) gains the MAGMA `TrialDivision(n, B)` paragraph with review 3 m4's "close to" wording.

### Item 3: capped drivers and MAGMA's third return value

- `src/factorization.rs` (new, 769 lines): module doc (the four MAGMA points of design §3.2 with
  review 3 m1's cost statement); `LimitedFactorization` (`:85`, `is_complete`, `value`);
  `FactorizationLimits` (`:149`, cost/memory doc, `Default` = 10000 / 8191 / None / true);
  `factorization_with_limits` (`:183`); crate-internal `Class` / `classify` (`:214`), `QsPolicy`,
  `SplitPlan`, `split_composite` (`:258`), `has_at_most_digits` (`:307`), `coprime_refine`
  (`:326`), `trial_division_stage` (`:419`, two `for_each_prime` calls split at 2^40, early BPSW
  stop above 2^16, state in a small `TrialDivision` struct), `limited_engine` (`:463`).
- `src/prime.rs`: `POLLARD_RHO_DEFAULT_ITERATIONS` (`:1099`), `factor_pollard_rho_limited`
  (`:1139`, MAGMA `PollardRho(n, c, s, k : Proof)`); `brent_rho` (`:1219`) now takes x0, has an
  exact budget and returns `(Option<Integer>, u64)`; new `rho_rounds` (`:1288`);
  `split_odd_composite` `pub(crate)` (`:1168`); `split_brent_and_trial_division` calls
  `brent_rho(m, c, 2, budget).0`; `factor_pollard_rho` doc: "overshoot by 2x" clause gone, a
  pointer to the capped variants, and the `# Panics` sentence naming `factorization_with_limits`.
- `src/lib.rs`: `pub mod factorization;` and `pub use factorization::{factorization_with_limits,
  FactorizationLimits, LimitedFactorization};`.

### Item 4: the QS `legendre_symbol`

Deleted with the old `tonelli_shanks`. The factor base uses Euler's criterion on words, then
`sqrt_mod_u64` (word Tonelli-Shanks, `None` exactly for non-residues, panics for a non-prime p).
No error value is swallowed anywhere.

### Item 5: the ECM module doc

`src/ecm.rs:14-31`: the design's text (Lenstra 1987, Hasse interval, L_p[1/2, sqrt 2], what
stage 1 and the 100-prime stage 2 test, MAGMA GMP-ECM difference). `:337` the curve comment
("Suyama's u, v only seed a random point ... not Suyama's curve"). `ecm_factor_complete` doc
(`:509`) gains the sentence naming `factorization_with_limits`. No code change.

## Tests and reference sources

All reference values were recomputed with PARI/GP 2.15.4 in this session
(`scratchpad/i2impl/refs.gp`, `bits.gp`); every value matched the design. Brent step counts from
an independent Python copy of the budget rule (`scratchpad/i2impl/brent.py`).

- `tests/ch18_quadratic_sieve.rs` (new, 12 tests + 1 ignored): `qs_factors_gp_semiprimes_20_to_45_digits`
  (rows 20 and 30 also as -n), `qs_factors_named_inputs`, `qs_complete_matches_pari` (40-digit row,
  F7, -pqr, p^2 q, psi_12), `qs_complete_prime_powers` (`within(5)`), `qs_prime_gives_no_factor`,
  `qs_with_params_tiny_inputs`, `qs_quick_paths` (also -15 gives 3), `qs_out_of_polynomials`,
  `qs_a_supply_too_small_returns_promptly` (`needed: 63 / 64`), `qs_polynomial_budget_is_per_polynomial`
  (`needed: 73 / 93 / 133`), `qs_params_validation`, `qs_seed_sweep_24_digits`; ignored
  `qs_factors_gp_semiprimes_50_to_60_digits` with the design's reason string.
- `tests/ch18_limited_factorization.rs` (new, 14 tests): the 14 rows of design §3.5 table that
  are integration tests, each checked with the invariant helper (value, coprime, sorted, `is_prime`
  on primes, `is_probable_prime` on probable primes, `!is_probable_prime` on composites).
- Unit tests, `siqs.rs`: `sqrt_mod_u64_least_non_residue_above_100`,
  `sqrt_mod_u64_small_primes_exhaustive`, `sqrt_mod_u64_rejects_composite`,
  `factor_base_matches_kronecker`, `factor_base_reports_a_small_prime_factor`, `a_selector_guard`,
  `polynomial_roots_survive_gray_code` (also: the 2^(s-1) values of b are distinct and
  a c = b^2 - kN), `relations_are_exact`, `relation_store_drops_repeated_y`,
  `relation_store_caps_first_partials`, `run_stops_at_the_polynomial_budget` (b in {1, 3, 16, 17}:
  polynomials = b, a_values = ceil(b/16), s = 5, needed 835), and the extra
  `run_on_grid_rows_drops_no_partials` (20/30/35-digit rows factored, `dropped_partials == 0`, the
  design's risk check). `gf2.rs`: `gf2_left_nullspace_hand`, `gf2_left_nullspace_random` (rank by
  an independent column-echelon elimination of the transpose). `factorization.rs`:
  `coprime_refine_values`, `coprime_refine_random`, `trial_division_crosses_2_40` (found
  [precprime(2^40), nextprime(2^40)], m = P30, 2365 primes visited), extra
  `trial_division_stage_small_limits` (2 P30 with no limit: 6543 primes visited, gp
  `primepi(2^16)` = 6542), `has_at_most_digits_edges`, `split_composite_plans`,
  `engine_requeues_composite_divisors`, extra `classify_classes`. `prime.rs`: `brent_rho_budget`
  (31870 steps, (None, 1), (None, 8191), (None, 0)), extra `rho_rounds_moves_to_the_next_c` (15, 25;
  (10^9+7)(10^9+9) not split by 8191, split by 2^20). `factorint.rs`: the eight tests of design
  §2.4 plus `trial_division_with_primes_rejects_one`.
- Doc examples: module doc (20-digit row), `quadratic_sieve_factor` (15347 gives 103),
  `quadratic_sieve_factor_with_params` (270050147102784401), `quadratic_sieve_factor_complete`
  (1001, 12780049, the 30-digit row), `factorization` module doc (the trial-division-limit row),
  `factor_pollard_rho_limited` (2^64 + 1; p15 q15 unsplit), `trial_division_with_primes` (1225).

HEAD tests replaced (design fate table): `test_legendre_symbol`, `test_factor_base_generation`,
`test_trial_factor`, `test_gaussian_elimination` deleted with their functions or ported; the five
Tonelli tests ported to `sqrt_mod_u64_*` (same values; the composite-panic message changes with
the function name). Assertions that pinned buggy behaviour, corrected with a reference value:
`test_complete_factorization_composite_sieve_divisor` asserted `quadratic_sieve_factor(165731) ==
Some(3127)` (gp `factor(165731)` = [53,2; 59,1]; 3127 = 53·59 is the composite the old sieve
returned), now `test_complete_factorization_repeated_prime` with `Some(53)`;
`test_complete_factorization_sieve_failure` asserted `quadratic_sieve_factor(3127) == None` (gp
`factor(3127)` = [53,1; 59,1]), now `test_complete_factorization_3127` with `Some(53)`. Both
complete-driver assertions are kept. `test_quadratic_sieve_small` / `_larger` now assert the exact
divisor (stronger). `test_brent_rho` became `brent_rho_budget` with its two assertions kept and
step counts added. `trial_division_with_primes` tests: the old `-1` entry assertion becomes
`f.sign == -1` (API change); the old "cofactor 5 is in the list" becomes an exact list. Nothing
weakened.

## API and behaviour changes, callers checked

- `quadratic_sieve_factor_with_params(n, usize, usize) -> Option<Integer>` became
  `(n, &QsParams) -> Result<QsOutcome>`; new public `QsParams`, `QsOutcome`.
- `trial_division_with_primes` returns `LimitedFactorization` instead of `Vec<(Integer, u32)>`.
- New public: module `factorization` (`LimitedFactorization`, `FactorizationLimits`,
  `factorization_with_limits`), `prime::factor_pollard_rho_limited`,
  `prime::POLLARD_RHO_DEFAULT_ITERATIONS`.
- Behaviour: `quadratic_sieve_factor` finds factors (3127, 165731 and 15347 give 53, 53, 103;
  -15 gives 3, `None` at HEAD); the complete QS driver finishes 40-digit semiprimes and splits
  psi_12 via the sieve; `brent_rho`'s budget is exact.
- Callers: grep of `*.rs` outside `target/` for `quadratic_sieve_factor_with_params`,
  `trial_division_with_primes`, `brent_rho`, `split_odd_composite`, `QsParams`,
  `LimitedFactorization` and `quadratic_sieve`: no hit outside `rustmath-integers` (src, tests).
  `rustmath-numbertheory` glob-re-exports `prime::*`; no crate defines the new names; its tests
  pass. `cargo check --workspace --tests` passes.

## Deviations from the design (with evidence)

1. `init_polynomial(fb, m, a_factors, &mut Roots) -> Polynomial` reuses one `Roots` (incl. the
   s x F Bainv2 table) for every a-value instead of returning a fresh one; `next_polynomial` takes
   kN; `Polynomial` stores `a: BigInt` and `two_bl` (2 B_l) instead of `a_factors` / `bl`;
   `FactorBase` has no `k` field (unused). Internal shapes only.
2. `BaseScan` has a fourth variant `Exhausted` for the scan reaching 2^31 with fewer than F primes
   (design step 2 asks for `OutOfPolynomials { relations: 0, needed }` there; the design's enum
   sketch lists three variants).
3. Factor base: Euler's criterion on words first, then `sqrt_mod_u64` only for residues, so only
   about F primes pay its `is_prime_u64` precondition check (design: Euler + `sqrt_mod_u64`).
4. `sieve_init` and `large_prime_bound` are small helpers shared by `run` and the unit tests.
5. `brent_rho` counts the replay steps after a block gcd of m in its returned step count, as the
   design's prototype `driver_r1.rs` and review 3's `brent3.py` do; so the count can exceed
   `max_steps` by at most 128 recomputed terms (documented). New terms stay within the budget.
6. `trial_division_with_primes` checks every entry for >= 2 before dividing (an entry 1 loops
   forever, 0 divides by zero); the design's `# Panics` names this case.
7. `qs_with_params_tiny_inputs` uses `QsParams::for_bits(12)` (the design names no parameters);
   2207 gives `NoFactor` with >= 32 dependencies.
8. Extra tests listed above (`run_on_grid_rows_drops_no_partials`,
   `rho_rounds_moves_to_the_next_c`, `trial_division_stage_small_limits`, `classify_classes`,
   `trial_division_with_primes_rejects_one`).

## Review 3 minor notes

- m1 (ECM cost on large inputs, "a few percent"): folded into the `factorization.rs` module doc and
  the `FactorizationLimits` cost doc (hours to days from about 90 digits; a larger
  `pollard_rho_limit` reaches factors up to about 12 to 14 digits). Design §8's HANDOFF line is the
  orchestrator's (below).
- m2 (`cargo fmt -p rustmath-integers -- --check` fails at HEAD): folded as suggested. New files
  are rustfmt-clean (`rustfmt --edition 2021 --check`); in the touched files that were not clean
  at HEAD the diff count did not rise (prime.rs 8 = 8, factorint.rs 4 < 6, ecm.rs 10 = 10); the
  crate-wide check still fails only on pre-existing diffs (also integer.rs, sage_wrapper.rs,
  tests/ring_predicates.rs, untouched). A crate-wide format is a separate commit.
- m3 (memory of the set of used a-values): folded into the `QsParams` memory paragraph.
- m4 ("close to `TrialDivision(n, B)`"): folded into the `factorint` module doc; the tracker note
  is for the orchestrator.
- Note (F = 841 vs 802): the `QsParams` doc now says F = 802 for a 40-digit n of 131 bits and
  F = 3794 for 198 bits. Nothing skipped.

## Measurements

Release, the port (`scratchpad/i2impl/grid_release.txt`, gp grid `$X/semiprimes.txt`, every row
18 to 70 digits factored): 20 digits 5 ms, 25 11 ms, 30 12 ms, 35 29 ms, 40 61 ms, 45 0.15 s,
50 0.35 s, 55 1.0 s, 60 2.3 s, 65 8.1 s, 70 36 s (prototype: 6 ms, 18 ms, 14 ms, 43 ms, 77 ms,
0.15 s, 0.35 s, 1.1 s, 2.4 s, 8.7 s, 39 s): within the 1.5x acceptance. Robustness in release
(`scratchpad/i2impl/zz_tmp_robust.rs`): `$X/robust.txt` (581 inputs) x seeds 1..3 and
`$X/rev1/small.txt` (1445 inputs) x seeds 1..3 with `for_bits`: 0 failures; 16 s and 4.0 s per
seed (prototype 13 to 14 s and 3.3 s). p12 * P30 with `FactorizationLimits::default()`: 0.087 s.
`cargo test --release -p rustmath-integers --test ch18_quadratic_sieve -- --ignored`: 3.49 s
(acceptance < 5 s). Debug: `ch18_quadratic_sieve` 2.4 s wall / 5.3 s CPU (< 8 s),
`ch18_limited_factorization` 2.7 s wall / 5.5 s CPU (< 6 s), `ch18_factorization_drivers` 2.2 s.

## Acceptance commands

| command | result |
|---|---|
| `cargo fmt -p rustmath-integers -- --check` | fails on pre-existing diffs only (review 3 m2, above); new files clean |
| `cargo clippy -p rustmath-integers --tests` | no warning in new or rewritten code (remaining: pre-existing ecm.rs, integer.rs, sage_wrapper.rs, prime.rs thread_local and test `vec!`) |
| `cargo test -p rustmath-integers --no-fail-fast` | 400 passed, 0 failed, 6 ignored (lib 233; drivers 9; limited 14; primality 18 + 2 ign; quadratic sieve 12 + 1 ign; residues 30; signs_roots 38; ring_predicates 1; doc tests 45 + 3 ign). I1 had 349 / 5 |
| `cargo test --release -p rustmath-integers --test ch18_quadratic_sieve -- --ignored` | 1 passed, 3.49 s |
| `cargo test -p rustmath-numbertheory --no-fail-fast` | 46 passed, 0 failed |
| `cargo check --workspace --tests` | passes |
| `grep -rn unsafe rustmath-integers/src` | no new hit |

## Left undone

- Full `cargo test --workspace` (the orchestrator runs it).
- No edits to docs/*, PROJECT_TRACKER.md or docs/magma/* (design plan step 7 is below).

## Handoffs

- Trackers (orchestrator, design plan step 7; line numbers at `c79a582`, re-grep):
  `PROJECT_TRACKER.md` rows `:395` (QS, 🔴 -> 🟢: SIQS, measured table), `:399`
  (`trial_division_with_primes`, 🛑 -> 🟢: `LimitedFactorization`), `:402` (ECM: doc fixed;
  per-curve cost and success rate), `:406` and `:407` (⏳ -> 🟢; `:407` loses the
  `legendre_symbol` sentence), `:415` (rho driver: the capped `factor_pollard_rho_limited`), new
  rows for `factor_pollard_rho_limited` (`prime.rs:1139`), `factorization_with_limits`
  (`factorization.rs:183`), `quadratic_sieve/siqs.rs` and `quadratic_sieve/gf2.rs`; card T-223
  (`:254`) and blocker B-04 (`:3100`). `docs/magma/status_overrides.csv`: `ch018.MPQS` (SIQS,
  verified against gp 20 to 70 digits), `ch018.PollardRho` (`factor_pollard_rho_limited(n, c, s,
  k, proof)`, Brent, third value), `ch018.TrialDivision` (`trial_division_with_primes`; and
  `factorization_with_limits` with `pollard_rho_limit: 0, mpqs_limit: Some(0)` is close to
  `TrialDivision(n, B)`: perfect powers are split too, the cofactor is classified, `proof: false`
  skips ECPP; plus the m10 note on `factor_trial_division`), `ch018.Factorization`
  (`factorization_with_limits`: sign, Proof, MPQSLimit with "< 25 disables", `ECMLimit := 0`
  fixed, third value only with `mpqs_limit` set, `Bases` absent); then
  `python3 tools/magma_tracker/build.py`. `docs/HANDOFF.md:275-282`: remove the five I2 items and
  add the three hand-offs of design §8:
  1. `factorint::factor_trial_division(n, None)` (`factorint.rs:199`) stops at 10^6 above 2^128
     and appends a composite cofactor, against its doc; `factorization::trial_division_stage` can
     serve it.
  2. `prime::factor` and `prime::prime_divisors` are unbounded trial division (3.4 s release on
     100000007 * 1000000007; hours for a 12-digit second factor), used on 41 lines in 24 files of
     8 crates (the REPL's `sigma` at `rustmath-jupyter/src/repl.rs:2630` among them); rebuild on
     `factorization_with_limits(n, &FactorizationLimits::default())` or document the cost.
  3. `ECMLimit`: needs a new ECM (projective Montgomery/Edwards curves, Suyama parametrisation, a
     seeded sigma, full stage 2) before an `ecm_limit` in `FactorizationLimits` pays; without it a
     large composite with a medium factor costs the sieve's time for the whole number (hours to
     days from about 90 digits; review 3 m1).
- Later units of this round: none addressed. J1a/J1b (REPL) may want a `factor` command with
  limits on `factorization_with_limits` (never panics, reports unproven factors), but no current
  REPL arm calls the QS or the limited engine.
