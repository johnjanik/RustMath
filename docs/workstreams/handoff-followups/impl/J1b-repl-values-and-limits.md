# J1b-repl-values-and-limits: implementation report

Crate: `rustmath-jupyter` only (`Cargo.toml`, `src/repl.rs`, `src/kernel.rs`, `src/handlers.rs`,
`src/protocol.rs`, `src/install.rs`, new `tests/kernel_protocol.rs`; `Cargo.lock` gains the `rug`
line in the jupyter dependency list, no new crate). Line numbers are those of the working tree at
the end of this unit (HEAD `bd03278` plus this unit's uncommitted changes). Nothing committed.

Scratch: `$W = /tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/impl/J1b-work`
(merge inputs, gp scripts and outputs, test logs, `meas/` with the ladder driver, the logs of
the release runs, the fit script and the F2 scenario output). `$J` is the design scratch
`/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/J1b-repl-values-and-limits`.

## How the prototype was merged

`repl.rs`: three-way merge (`git merge-file`, base = `repl.rs` at `c79a582`, which is the
designer's `proto/src_head`; ours = HEAD with J1a; theirs = the r1 prototype `$J/r1/src/repl.rs`).
11 conflicts, resolved by script (`$W/resolve.py`): ReplContext fields and `new()` take both
sides; `eval_expr` gets the checkpoint before J1a's depth guard; the seven graph constructors and
`chart` keep J1a's code with `.into()`; the power arms take J1b's. J1a's own `Rational ** Integer`
arm (bounded by `OP_POW_EXP`) was deleted, replaced by J1b's. The prototype's measurement-only
`chart` arm was dropped (J1a's S14 arm kept). `kernel.rs`, `handlers.rs`, `protocol.rs` were
unchanged since `c79a582`, so the prototype versions were taken and then completed. Everything
the prototype lacked was written from the design: the Evaluator trait, the plot checkpoints, the
C1 arms, the limit re-measurement, the tests, and all review-2 notes.

Step 0 (merge check): `grep OP_POW_EXP|POW_EXP|STRING_REPEAT` found J1a's two Integer power arms
and its Rational power arm, the `pow` arm, the repetition arm, `test_negative_exponent_message`
(J1a) and the three HEAD tests of design table 2.5. C1's `number_of_partitions`,
`NUMBER_OF_PARTITIONS_MAX_N`, `falling_factorial_integer`, `rising_factorial_integer` are
public (`partition_number.rs:40, :78`, `factorials.rs:130, :149`) and are imported through
their modules. None of the removed constants is left.

## What changed, per item

### F2: values behind `Arc` (repl.rs)

- `RustMathValue` (`:176-213`): `Matrix`, `Polynomial`, `List`, `Graph`, `Plot.svg` (`Arc<str>`),
  `ChartVal`, `DiffFormVal`, `VectorFieldVal` are behind `Arc`. About 120 construction sites take
  `.into()`; `add_edge` clones the graph with `Graph::clone(g)` and wraps the result; polynomial
  `+ - *` clone through `(**p).clone()`.
- Assignment paths (`eval_line`, `:697-800`): single assignment (`:755`), tuple unpacking and the
  expression statement all follow value, checkpoint, display, checkpoint, store (review 2 m7: the
  tuple path now displays before its stores, the expression path no longer clones before the
  display). The value is moved into `last_result`; the variable gets one O(1) clone.
- Measured (release, `$W/meas/scen_final.txt`, the design's `scen_r1.sh`): chart of 1.5·10^6
  names 192 MB, `w = coordinate_form(M, C, 0)` 232 MB, `v = w`, `u = w`, `wedge(w, w, C)` add
  nothing; `L` (10^5 integers) 39 MB, `P = [L, L, L, L]` 65 MB peak, 22 MB resident;
  `identity(1000)`, `B = A` 42 / 47 MB; `complete_graph(2000)`, `H = G` 77 MB both. The design's
  table 1.5 holds.

### S12: growth guards (repl.rs)

- Powers: `checked_int_pow` (`:11687`, now `(cmd, b, k, e)` so the message names the exponent as
  written and says "a denominator of" for e < 0), `int_power_value` (`:11714`),
  `rational_power_value` (`:11730`), `int_pow_bits` (`:11759`). `**`/`^` (`:9811` region) and
  `pow` (`:1179`) call the same functions, so they agree for integer and rational bases; `pow`
  accepts a rational base and negative exponents.
- Integer and rational products (review 2 m3, folded): `checked_int_product` (`:11767`) refuses
  `a * b` above POW_BITS (at once when bits(a) + bits(b) - 1 exceeds it, else after the product
  when the product is one bit over); `check_rational_op` (`:11789`) bounds `+ - * /` of rationals
  by the unreduced size.
- Rational gcd (found, see Deviations 1): `check_rational_gcd` (`:11813`) refuses a reduction
  estimated above 10 s; called by `check_rational_op`, `Integer / Integer`, the `Rational(n, d)`
  constructor and `rational_power_value`.
- Matrix powers: `checked_matrix_power` (`:10174`), left-to-right binary powering with (a) the
  trace bound, (b) `check_matrix_product` before every product, (c) the up-front floor; the doc
  comment carries the argument (Knuth 4.6.3, Horn and Johnson 5.6.9). Negative exponents give the
  design's ValueError (it names MAGMA's invertible case).
- Products: `matrix_product_seconds` (`:11866`, with the fit tables), `ceil_log2`,
  `check_matrix_product` (`:11886`; the entry bound uses ceil(log2 k), review 2 nit),
  `check_scalar_product` (`:11909`), `check_poly_product` (`:11933`), wired into `Matrix * Matrix`,
  `Matrix * Integer`, `Integer * Matrix`, `scalar_mul`, `Polynomial * Polynomial`,
  `Polynomial * Integer`.
- Strings (`:10024`): repetition bounded by `STRING_BYTES / len` (any count for `""`, which is
  answered at once), concatenation by `STRING_BYTES`.
- `repl_limits`: `POW_EXP`, `OP_POW_EXP`, `STRING_REPEAT` removed; a J1b block (`:11180-11207`)
  with `POW_BITS`, `RESULT_BITS`, `PRODUCT_SECONDS`, `STRING_BYTES`, `OUTPUT_BYTES`,
  `LIBRARY_INT_DISPLAY_BITS`; the module doc no longer says "no timeout or interrupt" and no
  longer lists `matrix_power` as unbounded; the section header cites ch. 1, 3, 18, 20, 26, 49,
  144, 145 and the papers (`:10965`).

### D: the display budget (repl.rs, Cargo.toml)

- `rug = { version = "1.24", default-features = false, features = ["integer"] }` in
  `rustmath-jupyter/Cargo.toml` (the declaration of rustmath-rationals with one feature).
- `decimal_string` (`:11514`), `rational_text` (`:11524`, used by `to_display`, `matrix_inverse`,
  `rref` and `format_rational_matrix`), `poly_text` (`:11532`), `DisplaySize`, `int_text_bytes`,
  `display_size` (`:11569`), `expr_display_size` (`:11620`), `display_summary` (`:11657`);
  `to_display` (`:350`) measures first and `to_display_unchecked` formats; the matrix arm builds
  text and LaTeX in one pass; `bar_chart` integer labels and `expr_to_latex` leaves go through
  `decimal_string` (review 2 nit).
- Review 2 m2 folded: `display_size` skips zero polynomial coefficients. Review 2 nit folded:
  2 more bytes per numeric expression leaf.
- Measured (release): `sigma(2, 268435456)` 1.56 s (446 MB, the value); `x = sigma(2, 55700000)`
  2.40 s (printed), `55800000` 0.19 s (summarised); `[a, a]` with a 2.7·10^7-bit `a` 2.27 s; the
  `e = e + e` DAG after 40 doublings 0.01 s.

### S13: interrupt, time limit, a kernel that answers while it computes

- repl.rs: `InterruptHandle` (`:67`), fields and `interrupt_handle`, `set_time_limit`,
  `time_limit`, sticky `checkpoint` (`:624`), `eval`/`eval_cell` (`:665`), checkpoints before each
  line, in `eval_expr`, in the three store paths, in `checked_matrix_power`, and one per sample in
  the 13 plot functions and in `plot_function` (review 2 m1, folded; `:4013 5274 5764 5998 6227
  6496 6721 7038 7236 7305 7361 7452 7802 7872 7939`). `is_cancellation` (`:11506`): the two
  sites that swallow an evaluation error (`parse_symbolic_arg`, the plot fallback) pass a
  KeyboardInterrupt or TimeoutError on (review 2 nit). `time_limit(s)` command (`:987`; integer or
  float seconds up to one year, 0 removes, `time_limit()` shows). `divmod` uses `div_rem`
  (`:1999`): ZeroDivisionError "divmod: division by the zero polynomial".
- kernel.rs (rewritten from the prototype): module doc with the MAGMA ch. 1 1.2 and ch. 3 3.12
  citations and what an interrupt stops; `EVAL_STACK_BYTES` = 64 MiB (`:45`),
  `INTERRUPT_NOTICE_AFTER` = 2 s, `ABORT_WINDOW` (its doc says the abort order differs from
  ipykernel, review 2 nit); private `trait Evaluator` (`:62`) with `impl for ReplContext`;
  `EvalWorker::{spawn, spawn_with, submit}` (`:91`); `pub fn eval_guarded` (`:122`); one
  `tokio::select!` loop (`:185`) over shell (idle only), control, heartbeat, the job, SIGINT and
  the notice timer; queued `execute_request`s are answered `aborted` before the failed cell's
  outputs and reply; if the evaluation thread is gone the cell gets an InternalError and `run`
  returns (`:228`, the design's "exits run()"; the prototype kept looping).
- handlers.rs: `EvalOutcome` (`:54`), `begin_execute`, `finish_execute`, `aborted_reply`,
  `publish_stream`; `handle_execute` removed; `handle_complete` lost its `&ReplContext`.
- protocol.rs: `stop_on_error` defaults to true (`:126`). install.rs: `"interrupt_mode":
  "message"` (`:49`).

### C1 arms (repl.rs)

- `falling_factorial` (`:2527`) and `rising_factorial` / `pochhammer` (`:2560`): any Integer x,
  u64 k; k = 0 answers 1 first (review 2 nit), the zero products are answered at once, then
  `check_factorial_product` and C1's `falling_factorial_integer` / `rising_factorial_integer`.
- `partitions` (`:2719`): u64 n up to `PARTITIONS_N`, C1's `number_of_partitions`; the static
  assertion `PARTITIONS_N <= NUMBER_OF_PARTITIONS_MAX_N` (`:11207`).
- `factorial_product_seconds` and `check_factorial_product` (`:12099-12122`).

### Limits (section 6; repl.rs `repl_limits`)

Measured end to end in release with the real `ReplContext` (one process per input, VmHWM), the
library built once with every limit raised to its ladder top (a local edit, restored before any
other change; `$W/meas/ladders.log`), then confirmed at the final values (`$W/meas/confirm/`).
Rule (stated in the `repl_limits` doc, review 2 nit): the largest ladder value whose admitted
calls take at most about 10 s and 400 MB; the time bound wins over rounding.

| Limit | Before | After | Evidence (release, end to end) |
|---|---|---|---|
| FIBONACCI_N | 1300000 | 70000000 | 7·10^7 8.6 s / 87 MB; 8·10^7 10.5 s |
| BELL_N | 600 | 5000 | 5000 5.5 s; 6000 10.6 s |
| MOTZKIN_N | 4500 | 300000 | 300000 8.6 s |
| DERANGEMENTS_N | 1400 | 2000000 | 2·10^6 8.1-8.7 s / 54 MB |
| SCHRODER_N | 50000 | 250000 | 250000 9.4-9.7 s; 300000 14.0 s |
| PARTITIONS_N | 416 (usize) | 2·10^10 (u64) | 1.9·10^10 8.3 s, 2·10^10 - 1 8.4 s, 2·10^10 7.2-7.4 s |
| TABLE_BYTES model | (n+1)(k+1) table bytes | time model `stirling_seconds` + memory (w+1)(B/8+32) | 21 (n, k) per kind; constants (50, 1.05), (50, 1.05), (150, 1.5) ns; stirling1(6000, 3000) 5.9 s admitted, stirling1(7000, 3500) refused (10.7 s est.) |
| FACTORIAL_WORK | k^2 work model | removed; `factorial_product_seconds` for falling, rising, multinomial | falling(2^32-1, 1.74·10^6) 8.4 s, 2·10^6 7.3 s (summarised), 2.2·10^6 refused; rising(1, 3·10^6) 6.9 s; multinomial(3·10^6, [1.5·10^6] x 2) 0.19 s |
| MANIFOLD_COORDS | 1500000 | 2500000 | chart 343 MB, form 374 MB, `v = w` free |
| GRAPH_BYTES | 133·10^6 (3 copies) | 200·10^6 (one copy) | path_graph(1785714) 199 / 395 MB with add_edge; star_graph(1617192) 200 / 395 MB; Graph(4166666) 200 / 395 MB |
| MUL_STEP_NS | 34 (design) | 46 | identity(d)^2: 560 6.1 s, 563 8.5 s (admitted, est. 9.99 s), 564 refused; 575 10.6 s, 600 10.0 s |
| POLY_STEP_FACTOR | 1.0 | 0.79 (44 ns a coefficient product) | 15000 terms 6.5 s |
| POW_BITS, RESULT_BITS | new | 2^25 | 3^21170489 3.2 s; 3^(2^26 bits) 5.3 s (would also pass, not raised, see Deviations) |

Unchanged on purpose: TAYLOR_ORDER (S1 not landed; its doc now notes the 64 MiB thread),
FORM_COMPONENTS and DIFF_FORM_DEGREE (MF1 not landed; FORM_COMPONENTS doc says the peaks were
measured with HEAD's copies), the binomial family and PRIMES_COUNT (not re-measured; they only
refuse more than needed: catalan(316227) takes 0.03 s), SIGMA_*, EUCLIDEAN_DIM.

MF1 §4.4 item 4 (stale line references in the form arms and `check_form_components`) is done:
the comments name `DiffForm::exterior_derivative`, `DiffForm::wedge` through
`TensorField::tensor_product`, and `TensorField::components`. Items 2 and 3 need MF1's
`component_count` / `DegreeOverflow`, which are not in the tree.

## Tests

New tests in `repl.rs` (all pass in debug; the whole lib suite takes 6.5 s, down from about
7 minutes at HEAD because `test_table_limits` is gone):
`test_values_shared_not_copied` (`:14880`, also tuple unpacking), `test_value_semantics_kept`,
`test_int_pow_growth_guard`, `test_negative_and_rational_powers` (with the rational gcd guard),
`test_matrix_power_growth_guard` (gp values, the property test against `Matrix::pow` for four
matrices and e < 40, the model and size bound), `test_product_size_guards`,
`test_string_growth_guard`, `test_integer_display_limit`, `test_output_budget` (with the
700000-zero polynomial, review 2 m2), `test_pending_interrupt_stops_next_eval`,
`test_interrupt_from_another_thread`, `test_time_limit`, `test_matrix_power_checkpoint`,
`test_plot_checkpoint` (plot3d and plot_function), `test_divmod_by_zero_polynomial`,
`test_c1_arms` (`:15301`), `test_stirling_limits` (`:14707`, replaces `test_table_limits`).
kernel.rs: `test_eval_guarded_turns_panics_into_errors`, `test_eval_worker_survives_panic`
(stub Evaluator), `test_eval_worker_runs_the_repl`. install.rs: `kernel_json_interrupt_mode`.
`tests/kernel_protocol.rs` (new, 280 lines): `kernel_answers_while_busy`,
`kernel_survives_sigint` (unix), `kernel_aborts_queued_cells`,
`kernel_stops_on_shutdown_during_library_call`; 3.8 s, stable over four runs.

Reference values: PARI/GP 2.15.4 (`/usr/bin/gp`); every value is in `$W/ref_j1b.gp` / `.out`,
`$W/st.gp`, `$W/st2.gp` and the design's `ref_r1*.gp` (rerun: same values as the design's stored
outputs; ref_r1's stored file differs only in captured stderr). Each test row carries its gp
command in a comment.

Existing tests changed (each with a reference value or because the guarded model was removed):
- `test_int_args_truncation_fixed`: `matrix_power(.., 4294967296)`, `falling_factorial(4294967297,
  2)`, `rising_factorial(4294967295, 2)`, `partitions(417)` move from refusals to values (gp
  `[1,1;0,1]^4294967296`, `prod(i=0,1,4294967297-i)`, `prod(i=0,1,4294967295+i)`,
  `numbpart(417)`).
- `test_int_args_no_panic`: the `pow`/`**` messages are the POW_BITS ones;
  `matrix_power(.., 2^63)` is the value `[[1, 9223372036854775808], [0, 1]]` (gp); the repetition
  message is the STRING_BYTES one.
- `test_int_arg_ranges_named`: the `pow(2, -1)` and `2 ** 10001` rows moved to value tests; the
  table-model rows (stirling1(1250, 1249) and so on) and the work-model rows (multinomial,
  falling, rising) are replaced by refusals at the fitted boundaries, and the now-admitted calls
  are value rows (gp `abs(stirling(1250,1249,1))` = 780625, `stirling(1300,1299,2)` = 844350,
  `A(1100,1099)` = 1, `300000!/300000!` = 1). Review 2 m4 folded.
- `test_negative_exponent_message` (J1a): `2 ** -10001`, `(1/2) ^ -10001`, `2 ** 10001` are now
  values (gp `2^-10001 == 1/2^10001`, `(1/2)^-10001 == 2^10001`, `#digits(2^10001)` = 3011); the
  refusal moves to `2 ** -33554432` (gp `#binary(2^33554432)` = 33554433).
- `test_pochhammer_is_rising` (J1a): `pochhammer(-1, 2)` is 0 (gp `prod(i=0,1,-1+i)` = 0); the
  "errors name the command" check uses `pochhammer(1, -1)`.
- `test_graph_limits`: boundaries recomputed from `graph_bytes` at 200 MB; the message says
  "add_edge copies the graph". `test_repl_limit_models`: the `product_work` rows (function
  removed) replaced by `factorial_product_seconds` rows.
- `test_table_limits` replaced by `test_stirling_limits` (the design's rename; the value rows are
  kept, the `table_bytes` exactness rows go with the function).

Fails on HEAD: the new unit tests use new APIs (they do not compile on HEAD). The HEAD behaviours
they guard (refusals of `2 ** 10001`, `pow(2, 1001)`, `matrix_power(.., 10 ** 30)`,
`partitions(417)`, the unbounded `matrix_power(Matrix([[2]]), 4294967295)`, the divmod panic,
the dead kernel on SIGINT, no interrupt reply) were reproduced by the design and both reviews
(`$J/s13/run_final_head.txt`, review 1 and 2 "What I reproduced"); J1a's own tests at HEAD
asserted the refusals of the rows changed above. I did not rebuild HEAD.

## Results

| Command | Result |
|---|---|
| `cargo test -p rustmath-jupyter --no-fail-fast` | lib 109 passed, 0 failed, 1 ignored (J1a's release test); main 0; `kernel_protocol` 4 passed; doc 0 |
| `cargo test -p rustmath-jupyter --test kernel_protocol` | 4 passed (3.8 s; four runs) |
| `cargo check --workspace --tests` | ok |
| `cargo clippy -p rustmath-jupyter --no-deps --all-targets` | no warning on a changed line except the pre-existing `needless_return` on `return Ok(Plot {..})` lines that only gained `.into()` |
| `$J/mem/scen_r1.sh <release replmem>` | one copy at every step (table above) |
| ladders (my driver, the design's ladders plus `multinomial`; `remeasure_r1.sh` itself was not used because it rebuilds with the limits as committed) | `$W/meas/ladders.log`, `$W/meas/confirm/confirm.txt` |
| gp scripts | as above |
| `grep unsafe` | none; no em-dash added |

`target/release` was deleted after the timing runs; the release harness binaries in `$W/meas`
too.

## Deviations from the design, with evidence

1. **A rational gcd time guard (new).** The prototype's `pow(Rational(2, 3), 15000000)` (admitted
   by POW_BITS) ran for more than 120 s in release (`ladders.log`): `Rational::new` reduces
   n^k / d^k with rustmath-integers' Lehmer gcd, which is quadratic (measured `Rational(a, b)`
   with a, b of 2^20, 2^21, 2^22 bits: 1.8, 7.0, 27.7 s), and rustmath-rationals has no
   constructor that skips the reduction. The same holds for every rational `+ - * /` and for
   `Integer / Integer`. `check_rational_gcd` refuses when 1.7e-12 · bits(n) · bits(d) s exceeds
   10 s. It refuses some cheap inputs (structured parts whose Euclid ends in two steps, e.g.
   `Rational(2^2^21 + 1, 2^(2^21+1) + 3)` 0.1 s); the message says "estimate".
2. **Multinomial.** The design kept `FACTORIAL_WORK`/`product_work` for multinomial with a refit.
   C1's product of binomials is 40 times below the k^2 model (0.23 s for (3·10^6, [1.5·10^6] x 2),
   refused by the old model), so multinomial uses `factorial_product_seconds(n, bits(n))` (the
   n! product tree bounds the binomials' numerator factors) and `FACTORIAL_WORK`, `product_work`
   are removed.
3. **Matrix constants.** `MUL_STEP_NS` 46 instead of 34 and `POLY_STEP_FACTOR` 0.79 instead of
   1.0: one-word products of d >= 575 take 46-56 ns a step (cache), so 34 admitted
   `identity(600)^2` at 10.0 s and `identity(575)^2` at 10.6 s. With 46 the largest admitted is
   563 (8.5 s). The polynomial factor keeps the design's 44 ns per coefficient product, so the
   design's 15000 / 16000-term boundary is unchanged. The floor now refuses
   `[[0,1],[1,0]]` to the 2^20000000 (6.0 s, estimate 11.0 s): errs towards refusing.
4. **Rounding.** FIBONACCI_N 7·10^7 (not the design's 5·10^7), MOTZKIN_N 300000 (not 250000),
   DERANGEMENTS_N 2·10^6 (not 10^6), PARTITIONS_N 2·10^10 (not 10^10): the time bound wins over
   the 1-2-2.5-5 rounding, as review 2 asked the design to state; each value was confirmed at the
   final build (all under 10 s).
5. **POW_BITS stays 2^25.** 2^26 would meet the policy (3^e at 2^26 bits 5.3 s, summarised), but
   RESULT_BITS, the integer product bound and the design's tests are at 2^25; raising it is left
   to the trackers.
6. **Stirling2 with k = 1** is answered at once (it is 1 for n >= 1); C1's callee runs n rows,
   which is what made `test_table_limits` take about 7 minutes in debug (J1a's handoff).
7. **`x * (2 ** 4194303)` prints**: tested on the size estimate (`library_int_bits` = 4194304,
   under the cap) instead of printing it, which takes seconds in a debug build.
8. **Integration tests** use 2000 lines of `factorial(3000)` and SIGINT through `kill`, as
   designed; the semiprime test still holds because the REPL's `factor` is trial division
   (`rustmath-integers/src/prime.rs:634`), so the call is long.

## Review notes (review 2, the approving one)

Folded: m1 (plot_function checkpoint, test, nintegrate and find_root named in the help and the
kernel doc), m2 (zero coefficients, test), m3 (Integer and Rational products bounded, tests),
m4 (rows listed and replaced), m5 (constants fitted on 21 measured points per kind, centre-first,
the edge over-estimate stated in the doc), m6 (help text: `print` as a function returns the
printed text, summary included), m7 (the same order in all three store paths); nits: ceil(log2 k),
the one-word-operand over-estimate in the doc, +2 bytes per numeric leaf, cancellation passed on
at the swallowing sites, k = 0 first, `bar_chart` labels and the rational matrix texts through
`decimal_string`, the abort order stated as differing from ipykernel, "agree for integer and
rational bases", the rounding rule, `"a" * 16777216` expects the summary.
Skipped: the `remeasure_r1.sh` header nit (a design scratch file, not in the repository; my
driver `$W/meas/ladders.sh` has the right header).

## API and behaviour changes, callers

No crate depends on `rustmath-jupyter` and no file outside it names `RustMathValue` or
`rustmath_jupyter::` (grep). Public API: `RustMathValue` variants hold `Arc`s;
`repl::InterruptHandle`, `ReplContext::{interrupt_handle, set_time_limit, time_limit}`;
`kernel::{eval_guarded, EVAL_STACK_BYTES, INTERRUPT_NOTICE_AFTER, ABORT_WINDOW}`;
`handlers::{EvalOutcome, begin_execute, finish_execute, aborted_reply, publish_stream}`,
`handle_execute` removed, `handle_complete(msg)`. User-visible: queued cells are aborted after an
error; a panic is an InternalError; SIGINT interrupts; outputs above 16 MiB are summarised;
powers, products and strings are bounded by result size; negative and rational powers in `pow`;
`matrix_power` takes any non-negative exponent; the C1 arms take any integer; the limits above.

## Left undone, and why

- TAYLOR_ORDER re-measurement and the stack check (S1 did not land), the MF1 §4.4 items 2-3 and
  the DIFF_FORM_DEGREE removal (MF1 did not land), FORM_COMPONENTS re-measurement (same).
- The binomial family (BINOMIAL_BITS, BINOMIAL_WORK, CATALAN_N, narayana, delannoy,
  DELANNOY_SUM_CAP) and PRIMES_COUNT were not re-measured (time); they are safe over-estimates.
- `cargo test --workspace` (the orchestrator runs it). Trackers and docs (orchestrator).

## Handoffs

### Trackers (orchestrator; design 8.1 step 9)
- PROJECT_TRACKER.md, jupyter section: the REPL integer-arguments-and-limits row (growth guards
  on result size, the display budget with GMP conversion, the re-measured limits table above, the
  C1 arms, the Stirling time model, the rational gcd guard; drop `matrix_power` and "the REPL has
  no timeout" from its not-bounded list); the wire-protocol row (control and heartbeat served
  while busy, SIGINT handled, `interrupt_mode: message`, `stop_on_error` default true, aborted
  replies); new rows "REPL values shared through Arc (F2)", "REPL interrupt, time limit and panic
  isolation (S13)", "REPL display budget and GMP decimal conversion".
- docs/HANDOFF.md: remove F2 (REPL part), S12, S13, C1's two REPL arm notes and the
  `test_table_limits` slowness note of J1a; keep the TAYLOR / MF1 limit notes (not re-measured,
  S1 and MF1 absent); add design section 7 plus the items below.

### New open items (for HANDOFF)
- rustmath-rationals: every operation reduces with a quadratic Lehmer gcd and there is no
  constructor from coprime parts, so r^e and products of large fractions are refused by the REPL's
  gcd estimate although the result is known to be reduced (27.7 s at 2^22-bit parts).
- POW_BITS could be 2^26 (5.3 s measured); kept at 2^25 with RESULT_BITS.
- The binomial family and PRIMES_COUNT are conservative after I3 and F2 (catalan(316227) 0.03 s).
- `stirling_band`'s a-priori bounds are loose at the band edges for stirling2 with small k
  (stirling2(200000, 2): 0.15 s, estimate 1.3 s).
- Design section 7 bullets all hold (`Matrix::pow` extra squaring, negative matrix exponents,
  library calls are not interruptible, quadratic `Display for Integer` in rustmath-integers,
  Clifford/Jordan displays outside the budget, `print`/`vars` budget per value, plot samples of
  a DAG, list literals copy Integers, `det`/`rank`/`factor`/`nth_prime` unbounded); `Integer *
  Integer` is now bounded (that bullet is done).
- `kernel_stops_on_shutdown_during_library_call` relies on the REPL's `factor` being unbounded
  trial division; if a later unit bounds it, switch the test to another long library call.
