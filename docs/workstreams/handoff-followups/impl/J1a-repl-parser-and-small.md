# J1a-repl-parser-and-small: implementation report

Branch `claude/gracious-gates-e6b6l7`, on top of HEAD `2e000c9` (C1). Nothing committed. Files changed:
`rustmath-rationals/src/rational.rs`, `rustmath-finitefields/src/integer_mod.rs` (CRLF kept),
`rustmath-finitefields/src/lib.rs`, `rustmath-jupyter/src/repl.rs`. Line numbers below are those of the
working tree at the end of this unit.

Starting point: the revision-2 prototype (`$S/r2/proto/src/repl.rs`, design scratch) was merged onto the
current `repl.rs` with a three-way merge (`git merge-file`, base = `repl.rs` at `c79a582`, ours = HEAD,
theirs = the prototype): no conflict (only I1 had touched `repl.rs` since `c79a582`). Everything the
prototype lacked was then written by hand, and the prototype parts were reviewed and cleaned (see
"Deviations").

Scratch: `/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/impl/J1a-work/`
(driver sources `drive.rs`, `drive_len.rs`, `timing.rs`, gp scripts `sq.gp`, `sq2.gp`, `s15.gp`, inputs
`s9.txt`, `s11.txt`, `s14.txt`, `n5.txt`, `s15.txt`, `sq.txt`, `rel_rows.txt`, release outputs
`timing_rel.out`, `rel_rows.out`, acceptance logs `acc_*.log`). All binaries deleted; `target/release`
deleted.

## Per item

### S11: `Rational::to_f64` and the REPL conversions

- `rustmath-rationals/src/rational.rs:163` `Rational::to_f64` now delegates to num-rational 0.4.2
  `BigRational::to_f64` (correctly rounded, never NaN; ±inf above f64::MAX, ±0.0 below the least
  subnormal). Doc rewritten. `NumericConversion::to_f64` keeps forwarding to it. Test
  `test_to_f64_correctly_rounded` (`:625`): the seven edge cases with Python `float(Fraction(n, d))`
  references (in the comment), plus 1/3 bit-identical to `1.0/3.0` and `NumericConversion` on -7/2.
- `repl.rs:10375` `exact_to_f64` (from the prototype, changed to call the fixed `Rational::to_f64`
  instead of num-rational directly, so rustmath-jupyter needs no new dependency).
- Sites rewritten: float literals that overflow are a ValueError in `eval_expr` (`:783`);
  `integrate_num` bounds (`:3099`, `what` = "integrate_num: lower bound"/"upper bound");
  `list_to_floats` (`:8966`), `list_to_complexes` (`:8974`), `value_to_f64` (`:9000`, `what` =
  "argument"; this covers the 3- and 4-argument `find_root`, the points); `eval_to_f64` (S9 item 10).
- Review 3 m2: `N`/`evalf` (`:1994`) refuses a non-finite result with a ValueError ("the value is
  outside the range of a double ...") instead of the text "inf"/"NaN". `N(r2)` gives 1.0000000000.
- Tests: `test_large_rational_to_float` (new, every row of design 4.4 plus review 3 m3:
  `integrate_num(x*1, x, 0, Rational(2*big + 1, big))` within 1e-12 of 2 (gp `intnum(x=0, 2, x)` = 2;
  the Simpson sum gives 2.0000000000000004), `integrate_num(x*1, x, 0, 2^1100)` ValueError naming the
  upper bound, `N(2^1100)` ValueError, `N(r2)`); prototype tests `test_median_large_rational`,
  `test_float_literal_range_and_underflow` kept.

### S9: parser

From the prototype (reviewed): `LexState` (`:10401`), `top_level_chars` (`:10440`),
`is_string_literal` (`:10447`), `is_binary_sign` (`:10469`), `ChainLevel`/`binary_chain`
(`:10284`/`:10293`), `is_exact_zero` (`:10316`), `group_inner` (`:10330`), `value_type_name` (`:11165`),
`negate` (`:9647`), `try_eval_binary_op` (`:9582`), the negative-exponent and Integer/Rational arms of
`apply_binary_op` (`:9664` ff.), grouping in `eval_expr` (`:847`), the function-call and string-literal
checks (`:834`), the splitters (`parse_args`, `parse_list_elements`, `parse_symbolic_args`,
`try_parse_sagemath_plot`, `eval_slope_field`, `split_at_depth_zero` `:5100`, `find_equation_operator`
`:9005`, `split_rhs_and_bounds` `:9013`, `parse_list_of_integers`, the `find_root` bounds),
`eval_to_f64` (`:5183`), `RealArith` (`:10277`), `float_arith` (`:5216`), `real_op` (`:5310`);
`eval_simple_arithmetic` and the textual `pi` substitution are deleted.

Added beyond the prototype:
- `parse_nested_list` (`:4829`): the row loop uses `LexState`, so brackets inside a string literal do not
  count (design 2.2 item 2; the prototype had not changed it).
- Review 3 m4: a nesting bound. `eval_expr` (`:724`) is now a wrapper that counts its own recursion in a
  new field `ReplContext::eval_depth` (`:561`) and refuses more than `repl_limits::EVAL_DEPTH` = 200
  levels (`:10858`) with a SyntaxError; the body is `eval_expr_nested` (`:736`). `float_arith` takes a
  `depth` parameter with the same bound. Runs of signs and parentheses are one step each, so the design's
  10^5-parenthesis and 100001-sign rows still evaluate. Alternating nesting `-(-(...))` costs two calls per
  level and is refused from 100 levels (it aborted the debug prototype at 600 to 1000 levels and the
  optimized one at 3000 to 5000).
- Cleanups of the prototype: the dead `depth` variables of the five splitter loops (rustc warnings), the
  unused `bytes` in `try_eval_binary_op`, the misplaced doc lines of `strip_inline_comment` and of
  `eval_slope_field`, the test helper `r1_text` renamed `eval_text` and moved with the prototype tests
  above the comment that belongs to `test_sqrt_mod_composite_moduli`; three clippy lints on new lines
  (`split(['e', 'E'])`, a collapsed `if`, `is_some_and`); "ch. 25 s25.1.2" written "§25.1.2".

Tests: the eleven prototype tests (`test_rational_products_kept` `:12624`,
`test_integer_rational_arithmetic`, `test_range_bounds_float_arith`, `test_range_bounds_exact_first`,
`test_float_arith_real_field` `:12709`, `test_negative_exponent_message`, `test_chart_checked_in_form_arms`,
`test_median_large_rational`, `test_value_type_name_by_variant`, `test_float_literal_range_and_underflow`,
`test_sign_and_group_runs` `:12900`) and new `test_parser_signs_and_precedence` (`:12927`, gp and Python
references in the comment), `test_parser_string_literals`, `test_lex_state`, `test_eval_depth_limit`
(`:12989`, runs on a 64 MB thread because debug frames are larger than release ones).

### S14: chart

`chart` (`:4676`) uses `chart_coordinate_names` (`:11128`) and passes dim(M) to `Chart::new`;
`check_chart` (`:11155`) in `coordinate_form` (`:4718`), `exterior_derivative` (`:4736`), `wedge`
(`:4772`). Existing tests rewritten as in design 5.3 (the prototype's `patch_s14_tests.py`, which targets
HEAD's text): `test_int_arg_ranges_named` (`:13778`, `C = chart(E3, "x,y,z")`, `Cb` dropped),
`test_form_component_limits` (`:13994`, `chart_names(100000)`, `C3 = chart(E3, "x,y,z")`,
`Cm = chart(M3, "a,b,c")`), `test_form_without_components` (`:14042`, `chart_names(2000)`,
`CS = chart(S, "a,b,c")`), helper `chart_names` (`:13988`). These are input changes forced by S14 (the
one-name charts are now refused); no assertion was weakened. New `test_chart_matches_dimension`
(`:12757`, Sage `Chart.__init__` message) and the prototype's `test_chart_checked_in_form_arms`.

### N5: plot options

`plot_int_options` (`:5513`, prototype) now used by all six commands: `plot_slope_field` (`:5619`),
`parametric_plot` (`:5857`, bare integer allowed), `contour_plot` (`:6071`), `vector_field` (`:6350`),
`implicit_plot` (`:6573`), `region_plot` (`:6889`). New constants `PARAMETRIC_POINTS`,
`VECTOR_FIELD_POINTS`, `IMPLICIT_POINTS` next to the prototype's three, moved into the J1a block of
`repl_limits`. `Plot3DOptions` (`:242`): the field `plot_points` is deleted; `parse_from_args` returns
`Result<Self, EvalError>`, every key and value is checked (ValueError naming the key); review 3 m1:
`parse_color` (`:294`) returns `None` for an unknown name or a `#`/`0x` value that is not 3 or 6 hex
digits, which is an error. Test `test_plot_options_checked` (`:12791`): every row of design 1.4 plus
the m1 colors, `plot_points=4, plot_points=5` (twice), and a bare argument to `contour_plot`.

### S10: pochhammer

`"falling_factorial" =>` (`:2412`) and `"rising_factorial" | "pochhammer" =>` (`:2440`); the rising arm
uses `func_name` in all messages. Help line `:10098`. Test `test_pochhammer_is_rising` (`:12846`, gp
`prod` references).

### S15: limits

- Graphs: `hash_set_bytes` (`:11083`), `graph_bytes` (`:11101`, checked `Option<u128>`),
  `check_graph_size` (`:11112`) in the six constructor arms (`:3242` ff.), `GRAPH_BYTES` (`:10872`).
  Small n handled without underflow (`path_graph(0)`, `path_graph(1)`, `star_graph(0)`).
- `prime_pi` (`:2787`): x > `PRIME_PI_X` = 6·10^9 is a ValueError (replaces I1's unreachable 2^64 text).
- `primes` (`:2807` calls `checked_primes`, `:10988`): `count_bound` (`:10923`, Montgomery-Vaughan),
  the sieve part below 2^64 (`try_prime_range`), and above 2^64 the prior check, pass 1
  (`is_probable_prime` per odd candidate, refusal at the first k_big that exceeds the model) and pass 2
  (`try_is_prime`; unresolved is a ValueError). `PSI_12` (`:10912`), `table_seconds` (`:10942`),
  `primes_t_test` (`:10965`), `primes_t_proof`, constants `PRIMES_COUNT`, `PRIMES_SPAN`,
  `PRIMES_SECONDS`, `PRIMES_STRONG_TEST_SECONDS`, `PRIMES_PROOF_SECONDS`. Help line `:10129` (half-open).
- Tests: `test_graph_limits` (`:13143`), `test_hash_set_bytes_model`, `test_prime_counting_limits`
  (`:13200`), `test_primes_time_model` (`:13232`, including the `count_bound` rows and the margin against
  HEAD's per-candidate cost). gp references in the comments (`primepi`, `primes`, `nextprime`,
  `binomial`, the count bound formula).

### sqrt_mod

- rustmath-finitefields: `num_square_roots_mod_prime_power` (`integer_mod.rs:1045`, with a doctest),
  re-exported in `lib.rs:46-49`; test `test_num_square_roots_mod_prime_power` (`:1515`, gp references;
  equality with `all_square_roots_mod_prime_power(..).len()` for 0 <= a < 64, p in {2,3,5,7}, k in 1..=3).
- REPL: `SqrtModLimits` (`:11586`), `SQRT_MOD_LIMITS` (`:11604`), `SqrtModPiece` (`:11616`),
  `sqrt_mod_trial_bound` (`:11628`, u64 integer arithmetic, review 3 m6), `sqrt_mod_factor` (`:11653`),
  `isqrt_cap` (`:11779`), `sqrt_mod_report` (`:11802`) = `sqrt_mod_report_with(.., &SQRT_MOD_LIMITS)`
  (`:11830`); constants `SQRT_MOD_*` in `repl_limits` (`:10892` ff.; `SQRT_MOD_PROOF_BITS =
  super::PROOF_BITS`, so it stays aligned with I1's `PROOF_BITS`); help line `:10081`. I1's prime branch,
  its separate Jacobi shortcut and the three `IntegerMod` calls are gone.
- Review 3 m5: phase 3 also stops when the modelled cost of the steps taken exceeds `trial_work`.
- Tests: `test_sqrt_mod_factored` (`:13273`, all rows of design 7.4 with gp references, including the
  restored 4 and 73 mod psi_12 rows), `test_sqrt_mod_factor_phases` (`:13300`, the three limit settings of
  design 7.4 plus a Jacobi row), `test_sqrt_mod_refusal_release` (`:13354`, `#[ignore]`, release: 1.02 s).
  I1's `test_sqrt_mod_large_prime_and_strong_pseudoprime` keeps every assertion; only its comment about the
  dropped psi_12 rows now points at the new test. The existing `test_sqrt_mod_*` tests pass unchanged,
  including `test_sqrt_mod_agrees_with_integer_mod` (every n <= 60 against `IntegerMod`).

### repl_limits documentation

The comment block above `repl_limits` (`:10710`) lists the J1a references (ch. 18 §18.4.1, §18.9.2,
§18.10.1, §18.12.1; ch. 19 §19.5.6; ch. 20 §20.1.2, §20.2.2, §20.4.2; ch. 25 §25.1.1-§25.1.2; ch. 49;
Montgomery-Vaughan 1973; Sorenson-Webster 2017); the `repl_limits` doc (`:10751`) describes the J1a
block, and each new constant has its measurement in its doc.

## API and behaviour changes, callers checked

- `Rational::to_f64` / `NumericConversion::to_f64`: same value when |num| and den fit in 53 bits,
  otherwise the correctly rounded double (at most 2 ulp from the old one, or a value where the old one was
  NaN, inf or 0). Callers: about 60 files in 24 crates (`grep -rl "to_f64()" | xargs grep -l Rational`); a
  grep for test assertions on exact `Rational` conversions found only the new test. The full workspace run
  (orchestrator) covers them.
- `rustmath_finitefields::num_square_roots_mod_prime_power`: new public function and re-export.
- `Plot3DOptions::parse_from_args` returns `Result`; field `plot_points` removed. No caller in the
  workspace (`grep -rn "Plot3DOptions\|parse_from_args"`: only `repl.rs`, which uses `default()`).
  No crate depends on rustmath-jupyter.
- REPL behaviour: as listed in design 2.3, 1.3, 3.2, 4.3, 5.3, 6, 7.3. Also: nesting deeper than 200
  `eval_expr` levels is a SyntaxError; `N` of a value beyond the double range is a ValueError; `sqrt_mod`
  of a prime n between psi_12 and 1024 bits gives no caveat (proven), above 1024 bits the caveat names
  `n`.

## Deviations from the design, with evidence

1. **MF1 and S1 are not in HEAD** (`git log c79a582..HEAD` has no MF1 or S1 commit, and no impl report
   exists for them). So there is no R^1 detour in `test_form_component_limits` to restore, and the
   library `exterior_derivative` does not check the chart dimension. The test rewrites were made against
   HEAD's text (the prototype's version), and `check_chart` covers all three REPL arms.
2. **Nesting bound (review 3 m4) as a recursion counter**, not as a bracket-depth limit in
   `top_level_chars`/`group_inner`: a bracket-depth limit would refuse the design's row
   `"("*100000 + "7" + ")"*100000` = 7, which costs no recursion. The counter bounds exactly what overflows.
   Nested calls (`abs(abs(...))`) still overflow a debug build's 8 MB stack at about 40 levels (measured on
   the debug driver), below the bound: their debug frames are about 240 KB. In release, by the design's
   measurement (2267 nested calls on 8 MB, about 3.6 KB per level), 200 levels need about 0.7 MB. The
   kernel evaluates on the main thread (`#[tokio::main]` runs its future there, 8 MB).
3. **Messages** that differ from the design text:
   - `primes` span refusal: "primes: the span b - max(a, 2) is {y}, above 8000000, so [a, b) may contain
     more than 1000000 primes (Montgomery-Vaughan bound), more than the REPL limit 1000000 (a memory
     bound)". The design's "spans {y} integers" read "spans an integer of 1329 bits integers" for
     `primes(0, 10^400)` (`show_int`).
   - Prior check: "testing the {n_odd} odd candidates of [max(a, 2^64), b), numbers of {B} bits, needs
     about {t} s (REPL limit 10 s)"; the design printed the bounds, which `show_int` turned into "[an
     integer of 1001 bits, an integer of 1001 bits)".
   - Pass 2 (review 3 m8): "has at least {k_big} BPSW probable primes that need a proof (at or above
     psi_12 = ...); proving them prime needs about {t:.1} s or more". One decimal, since t just above 10
     printed as "about 10 s" (release: `primes(10^30, 10^30 + 10^5)` refused at k = 1191).
   - `sqrt_mod` refusal prints n and the unfactored cofactor in full up to 1024 bits (`show_int` above).
     With `show_int` alone (full only up to 128 bits) the design's own release test, which expects the
     80-digit cofactor named, failed: "an integer of 261 bits".
   - The probable-prime caveat names "n" when the probable prime is n itself (for 2^1279 - 1 the design's
     `{q}` would repeat the 386-digit number).
4. **Pass 1 of `primes` calls `is_probable_prime` on every odd candidate**, as the design says. I1's
   faster pre-sieved enumeration (`for_each_big_probable_prime`, primes below 2^16) is private to
   rustmath-integers, and making it public would be a fifth crate. The model `t_test` matches
   `is_probable_prime`, which trial divides by the primes below 1000 (`primality.rs:401`).
5. **Time tables not refit.** Release on this container (`timing_rel.out`): one strong test 9.5e-5 s at
   512 bits (table 1.5e-4), 3.75e-3 s at 2048 (4.7e-3), 0.241 s at 8192 (0.26); `try_is_prime` means
   0.0104 s at 100 bits over 8 primes (table 0.008), 0.307 s at 512 bits over 6 (0.41), 2.41 s at 1024
   bits over 4 (3.3). Every entry is within a factor 1.6 and all but the 100-bit proof are
   overestimates (more refusals, never fewer). The release rows below agree with the model, so I kept the
   tables (refitting would also move the margin of `test_primes_time_model`).
6. **`exact_to_f64` calls `Rational::to_f64`** (the plan's intent; the prototype called num-rational
   because it could not edit the library).
7. The prime-pi refusal replaces I1's "exceeds 2^64" message (unreachable now). `value_to_f64` names its
   value "argument".

## Review 3 minor notes

All eight folded in: m1 (colors), m2 (`N`), m3 (`integrate_num` rows; `list_to_complexes`,
`value_to_f64`, `integrate_num` rewritten, so `fft([r2])` = `[1 + 0i]` and the 4-argument `find_root`
gives 1), m4 (as deviation 2), m5 (cost cap), m6 (u64 formula in `sqrt_mod_trial_bound` and the constant
docs), m7 (the `SQRT_MOD_PROOF_BITS` doc says 3.3 s mean, 4.3 s max; the "3.5 s vs 4.2 s" HEAD figure is
only in the design text, nothing in code to change), m8 (k_big wording; `prime_pi(6000000000)` took
11.3 s in release, inside the 7 to 14 s band, so `PRIME_PI_X` stays). None skipped.

## Release timing (design 8.2; `rel_rows.out`, 4 cores, load about 1)

| Call | Time | Check |
|---|---|---|
| `prime_pi(6000000000)` | 11.29 s, 279545368 (gp) | 7-14 s band |
| `primes(10^30, 10^30 + 20000)` | 1.70 s, 258 primes | model 2.2 s, within 10 s |
| `primes(2^512, 2^512 + 2000)` | 2.13 s, 7 primes | model 2.9 s |
| `primes(10^400, 10^400 + 10)`, `primes(2^1000, 2^1000 + 200)` | 0.001 s, 0.008 s, `[]` | < 0.1 s |
| `sqrt_mod(4, 2^127+1)` | 0.009 s | < 1 s |
| `complete_graph(3585)` | 1.03 s | < 2 s |
| `primes(10^30, 10^30 + 10^5)` | refused at k = 1191 in 0.13 s | |
| `primes(2^1024, 2^1024 + 20000)` | refused at k = 3 in 0.11 s | |
| `primes(2^1000, 2^1000 + 10^6)` | refused by the prior check (101 s modelled) | |
| `sqrt_mod(4, p20*q21)` | 0.046 s | |
| `test_sqrt_mod_refusal_release` | 1.02 s | < 5 s |

## Acceptance

| Command | Result |
|---|---|
| `cargo test -p rustmath-rationals --no-fail-fast` | 102 passed, 0 failed, 3 ignored (lib 68 incl. the new test) |
| `cargo test -p rustmath-finitefields --no-fail-fast` | 331 passed, 0 failed (lib 202 incl. the new test; doctests incl. the new one) |
| `cargo test -p rustmath-jupyter --no-fail-fast` | 89 passed, 0 failed, 1 ignored (the release test), 435.6 s in debug (`test_table_limits`, see Handoffs); HEAD had 64 tests, 26 are new (one ignored) |
| `cargo test --release -p rustmath-jupyter -- --ignored test_sqrt_mod_refusal_release` | 1 passed (1.02 s) |
| `cargo check --workspace --tests` | ok, no error |
| `grep -n "unsafe" rustmath-jupyter/src/repl.rs` | no hits (none in the other changed files either) |
| `cargo clippy --no-deps` on the three crates | no warning on a changed line (repl.rs: three fixed); `cargo clippy -p rustmath-jupyter` without `--no-deps` stops in rustmath-combinatorics (pre-existing deny lints) |
| `cargo test --workspace` | not run (the orchestrator runs it) |

Every new test value was rechecked here against its reference (gp scripts `sq.gp`, `sq2.gp`, `s15.gp`
and the one-liners in the test comments, Python `Fraction`) and run on the debug driver before it was
written into a test. The HEAD failures of the rows the design marks F were reproduced by the design and
its three reviews (outputs in the design scratch); I did not rerun them on HEAD.

## Left undone

- The full `cargo test --workspace` (the orchestrator runs it).
- Trackers and docs (`PROJECT_TRACKER.md`, `docs/HANDOFF.md`, `docs/magma/`): orchestrator, see Handoffs.

## Handoffs

### J1b (REPL values and limits)
- **`test_table_limits` takes about 7 minutes in a debug build** (448 s in the first full run here; C1's
  report shows the jupyter suite at 412-429 s already). The cause is C1's `stirling_second` (and the
  other band-row counts): `stirling2(4294967295, 1)` now runs `band_rows` over all n rows (it did not
  finish in 100 s on the debug driver), while `table_bytes` still returns `None` ("no table: the callee
  answers at once"). This is J1b's REPL limit recalibration (C1's handoff). The test passes.
- `value_type_name` (`repl.rs:11165`) is an exhaustive match: a new `RustMathValue` variant must be named
  there. `exact_to_f64` is the one exact-to-double helper. Integer / Integer is now a Rational, and
  Integer op Rational is computed in Q.
- `pochhammer` now shares the `rising_factorial` arm and its messages use `func_name`, so J1b's switch
  to `rising_factorial_integer` covers it.
- `eval_expr` counts its nesting (`eval_depth`); new recursive evaluators should go through `eval_expr`
  or take a depth like `float_arith`.

### Trackers and HANDOFF (orchestrator; design 8.1 item 4)
- `PROJECT_TRACKER.md`: jupyter rows for the REPL parser (S9 grammar, literal-aware scanner, Q
  arithmetic, real-field bounds, nesting bound), `sqrt_mod` (one factorization: trial division, I2 rho
  and sieve, ECPP; unfactored part refused), the REPL limits (graphs, prime_pi, primes two passes, plot
  options); rustmath-rationals `to_f64` (correctly rounded); rustmath-finitefields
  `num_square_roots_mod_prime_power`; the rustmath-symbolic `expand` row (`PROJECT_TRACKER.md:2159` at
  `c79a582`) from 🟢 to 🟡 with "`distribute_negation` (`expand.rs:250-263`) is wrong in both arms:
  `expand(-(x - 1))` = x - 1" (design revision 2, m2).
- `docs/HANDOFF.md`: remove N5, S9, the REPL part of S10 (pochhammer), S11, S14, S15 and the sqrt_mod
  bullet. Add design section 9 (all bullets still hold; S1 and MF1 did not land, so the
  `distribute_negation` and `--x` display items stay open), and: nested calls `abs(abs(...))` still
  overflow a debug build's stack at about 40 levels (below `EVAL_DEPTH`; an iterative evaluator or a
  larger eval thread would fix it); the `test_table_limits` slowness above.
- `docs/magma/`: `ch018.Modsqrt` if it points at the REPL (now factored by I2 within limits), then
  `python3 tools/magma_tracker/build.py`.
