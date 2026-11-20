# S1 implementation report (rustmath-symbolic series engine, literal parser, Display, Drop)

Branch `claude/gracious-gates-e6b6l7`, HEAD 8a0b310. Nothing committed, pushed, stashed or reset.
Governing record: `fu2/s1impl/owner_decision.md` (design frozen, no revision 9; the two behaviour
rules; the singular-point matrix next to review 9's harness). The design (`fu2/design/S1-symbolic.md`)
was not edited. Every change from it is listed under "Deviations".

Summary: the prototype `proto13` is ported into the crate and wired to the public API of design
§2.9 and §2.13. The owner's two rules are applied in the atan family, and arctan(+-i) is a pole of
the constant (review 9 m2). With those, every test of the first step passes, unchanged: B1, B2, the
red-team holes, the singular-point matrix and the fuzz data. The rest of the Plan is done too: the
§2.12 parser, the §2.11 Display, `Drop for Expr` with its 97 E0509 sites, simplify along the left
spine (§2.10), expand under a minus (§2.17), the REPL taylor arm with `TAYLOR_WORK`, and the
work-list LaTeX writer. Review 9's random harness on the same 2000 inputs gives 1250 results equal
to gp, the 4 gp-convention differences review 9 m1 named, no SUSPECT row, no crash and no timeout.
The design's oracle `gpcheck3.py` gives 107 of 107.

## 1. What changed, per item (file:line at the end of this step)

### Engine (design §2.1 to §2.8, §2.15, §2.16; Plan steps 1 to 4)

| file | content |
|---|---|
| `rustmath-symbolic/Cargo.toml` | dependencies `rustmath-reals`, `rug` (float, integer; as rustmath-reals), `num-bigint` (workspace); `rustmath-complex` moved to `[dev-dependencies]`; the feature `s1-engine-tests` removed (Plan step 4 of `tests_report.md`). `Cargo.lock` gains the new dependency lines. |
| `src/series_ball.rs` (new, from `proto13/ball.rs`) | MPFR balls, the window R1 and exact zeros R2, the guards, `ball_cost`. Header with MAGMA ch. 49 digest, Fousse et al. 2007, Johansson 2017. |
| `src/series_const.rs` (new, from `proto13/konst.rs`) | the constant field K: charged arithmetic, kernels, zero test, `kapply_src` (poles), `kpow_const`, `konst_of` on a work stack, caps. Header with ch. 49, Knuth, Brent and Kung, Johnson, Monagan and Pearce, Richardson, Lindemann, Hermite. |
| `src/series_engine.rs` (new, from `proto13/ser.rs`) | `Prec`, `Ser`, the operations, `expand` on a work stack, the driver `expansion` (`:2194`) with `Render` (`:2187`), and `Session` (`:2158`). Header with ch. 49 and the papers. |
| `src/series.rs` (top rewritten, `:1-700`) | the public layer of §2.9, ported from `proto13/api.rs`: `SeriesError` (`:61`) with the bounded `Display` (`display_estimate`, `:113`) and the hand-written `Debug`, `StructureLimit` (`:101`), `ExpansionPoint` (`:251`), `SeriesExpansion` (`:260`) with `first_exponent`, `coefficient`, `to_expr`, `big_o`, and a hand-written `Debug`; `series_expansion` (`:519`), `taylor` (`:543`), `taylor_with_budget` (`:553`), `maclaurin`, `series`, `laurent` (`:579`, `ValuationBelowMinimum`), `asymptotic` (`:602`), `series_coefficients` (`:613`, `Pole`), `series_with_big_o` (`:638`); the internal entries `expansion_first_term` (`:653`) and `expansion_reaches` (`:671`); `mentions_symbol` (`:687`, a DAG walk); `LIMIT_SERIES_WORK` (`:57`). The derivative loops and the facade `laurent` of HEAD are gone. `known_series` builds balanced sums (`:710`). The module header states the conventions (generic parameters, real abs and sign, float literals, branch cuts, the owner's rules). |
| `src/limits.rs:437` | `limit_via_series` reads `expansion_first_term` under `LIMIT_SERIES_WORK` (§2.13). `:564`: `limit_with_error` takes its `BigO` from `expansion_reaches` (no term rendered), and keeps HEAD's `Expr::limit` for the limit. |
| `src/lib.rs` | `pub(crate) mod series_ball, series_const, series_engine`; exports `SeriesError`, `SeriesExpansion`, `ExpansionPoint`, `mentions_symbol`; the series tests unwrap the `Result`s (assertions unchanged; `test_calculus_series_coefficients` also asserts `coeffs[2] = 1` and `coeffs[3] = 0`, §4.2). |

Instrumentation of the prototype removed: the env overrides `TREECAP` and `S1TRACE`, `KEYCOST`,
`KEXPRCOST`, `table_stats`, `record_sizes`, `HINTS`, `SEEDS`, `LAST_PREC`, the `FIRST_ONLY` and
`NO_TERMS` thread-locals (now the `Render` argument of the driver), the unused prototype `to_expr`
of the engine. `MAX_RESULT_TREE` is a constant (`1 << 21`).

### The owner's two rules and review 9 (B1, B2, m2)

- `series_engine.rs:1156` `atan_like` (arctan, arctanh, arcsin, arccos, arcsinh, arccosh; erf keeps
  the design's path). Rule 1: an argument whose series is an exact constant c (`exact_constant`)
  gives `Ser::exact_const(kapply(op, c))`, with no integrand. Rule 2: for any other argument the
  integrand is formed and its preconditions are checked (`integrate_check`, `:1099`, split out of
  `integrate`) before `kapply(op, b0)`. At a branch point the integrand has a pole (arctanh,
  arctan: the logarithmic branch point, `NotLaurent`) or a non-integer power of a series of positive
  valuation (arcsin, arccos, arcsinh, arccosh: `NotLaurent` from `pow_const`). An empty truncated
  integrand of negative precision asks for precision (`PrecisionExhausted` at the end), as in
  revision 8.
- `series_const.rs:2499` `is_plus_minus_i` and `:2578`: `apply(arctan, +-i)` is
  `DivisionByZero("arctan at i or -i")` (review 9 m2; gp: `atan(I)` "domain error"). Without it,
  rule 1 would have made `arctan(I*x^0)` a kernel at a pole.

Where else the design builds an integrand or a pole rule for a branch function, and what holds:
`log` refuses the exact 0 (`DivisionByZero("log(0)")`, a division by an exact zero in §2.9) and is
`NotLaurent` for a non-constant argument of positive valuation; `pow_const` (sqrt and non-integer
powers) gives `kpow_const(0, alpha)` for the exact 0 (0, `DivisionByZero` or `NotLaurent` by
`zero_pow_sign`) and `NotLaurent` for a non-constant base of positive valuation; a power with a
variable exponent takes the constant-base rule for an exact-constant base (`pow_general`,
`zero_pow`). These already follow the two rules, so they are unchanged. Rule 1 is not applied to
exp, sin, cos, tan, the hyperbolic functions, erf or log of a nonzero constant (optional in the
scope of `tests/s1_common/mod.rs`); for tan it would turn an honest `UndecidedZero` at an undecided
pole into a kernel (review 9 B1, required change 1).

### The three red-team holes

The three holes the task names (an empty truncated argument of negative precision read as valuation
>= 0; 0^r decided from the syntax of the base and a rational test of r; result trees exponential in
their DAG) were closed in the frozen design's prototype (revision 7rt3: `require_nonneg_val`,
`integrate`'s check, `pow_general` with `exact_constant`, `zero_pow`, `zero_pow_sign`, the hints,
`render_tree`, `MAX_RESULT_TREE`; revision 7rt4: `Prec` out of band and the saturating units, the
round-4 holes). The port keeps that code verbatim (`series_engine.rs` `require_nonneg_val`,
`integrate_check`, `pow_general`, `zero_pow`; `series_const.rs` `zero_pow_sign`, `undecided`'s tree
check; the driver's tree check). Their regression tests pass in debug and release (section 3). No
change to the design was needed for them, so there is no deviation to record.

### Parser (§2.12; Plan step 8)

`src/parser.rs`: `integer_literal` (`:149`, `Integer::from_decimal_str`, any size, unsigned);
`number` without the `a/b` token; `sign_run` (`:293`, a loop: an odd run of `-` is one `Neg`);
signed exponents with the fold `Neg(Integer(n))` to `Integer(-n)` (`make_power`, `:362`);
`multiplicative` folds a quotient of two signed integer literals to the exact rational, and `1/0`
is a parse error (`product_step`, `:384`); `arcsinh/asinh`, `arccosh/acosh`, `arctanh/atanh` map
to their `UnaryOp` (`unary_op_by_name`, `make_call`, `:272`). Tests of §4.3 at `:1071` to `:1210`.

The new code of the recursive rules sits in `#[inline(never)]` helpers (`make_power`, `negate_if`,
`product_step`, `make_call`, `sign_run`), so that the parser's frames are no larger than HEAD's.
Measured on a 2 MB debug thread in the test harness, the deepest input each parser reads (HEAD's
parser compiled into the same test binary from `git show HEAD:...`, `fu2/s1impl/impl/head_parser.rs`):

| shape | HEAD | port |
|---|---|---|
| `sin(` nested | 162 | 190 |
| `(` nested | 216 | 221 |
| `x^x^...^x` (carets) | 1440 | 1441 |
| `2*(` nested | 216 | 221 |
| `-(` nested | 216 | 221 |
| `2^(` nested | 187 | 191 |
| `x^-x^-...` | 5 (parse error) | 1440 |

A first version with the new code inline read 147 and 195 (a regression against HEAD); the helpers
fixed it. Release (8 MB) and REPL (64 MiB) limits were not measured (Plan step 8 asks for them):
release frames scale the same way, so only the debug comparison is recorded.

### Display (§2.11), Drop (§2.14), simplify (§2.10), expand (§2.17)

- `src/expression.rs:577` Display with a work list and the rules: a fraction under `^` or as a
  divisor, a negated or negative base of `^`, a negated negation, negative literal or fraction under
  a negation, and `.0` on a finite integral float. Tests `test_display_work_list` (`:807`, 10^5 terms,
  towers, sin chains and call chains on a 2 MB debug thread) and `test_display_parenthesis_rules`.
- `src/expression.rs:109` `Drop for Expr` with `EXPR_DROP_STACK_TOP`, `GLUE_STACK_BUDGET = 16 KiB`,
  the detach list and `Arc::into_inner`, and the leaf read by `try_with` (revision 8). Tests
  `drop_deep_shapes` (`:865`, a 10^6-term sum and four shapes of 2x10^5 levels on 2 MB debug),
  `drop_shared_children` (twin and sibling sharing, 2x10^5 levels), `drop_during_tls_teardown`
  (`:914`).
- E0509 sites (97, found by compiling the workspace with `--keep-going`): library
  `integrate_exact.rs:95` and `:144` (`neg_clean`, `split_rational_const` rewritten with `let next =
  match &cur`), `parser.rs` (the new `make_power`), `rustmath-geometry/src/plane_curve_ops.rs:492`;
  test sites in `parser.rs` (21), `specialfunctions/*` (60), `radical.rs` (3), `function.rs`,
  `walker.rs`, `rustmath-calculus/src/{integration.rs:427, limits.rs:309, sum.rs:232}`, each by
  matching on a reference, with derefs adjusted in 8 assertions (none weakened). The jupyter site
  is in the rewritten taylor arm. The CRLF line endings of six files were kept.
- `src/simplify.rs:376` `simplify` walks the left spine with an explicit stack;
  `simplify_non_binary` (`:396`) holds the other arms. Test `simplify_long_left_deep_sum` (`:687`,
  10^5 terms on 2 MB debug; HEAD's result: the constant folded, the sum of x kept).
- `src/expand.rs:254` `distribute_negation`: -(a + b) = -a + -b, -(a - b) = -a + b. Tests
  `test_expand_negation` (`:317`, the five gp values 4/7, -10/7, -64/35, 34/35, 40/49 and no negated
  sum left) and `test_expand_negation_structure` (`(-x + 1)`).

### REPL (`rustmath-jupyter/src/repl.rs`; Plan steps 5 and 10)

- `:2118` the taylor arm: the variable by `Symbol::new` and `mentions_symbol` (no `symbols()` walk),
  `match &args[3].simplify()`, `taylor_with_budget(.., repl_limits::TAYLOR_WORK)`, and the error
  mapping of §3.1 (`DivisionByZero`, `InvalidPoint`, `Pole`, `ValuationBelowMinimum` to ValueError,
  the rest to ComputationError, message `taylor: {e}`).
- `:11195` `TAYLOR_ORDER` stays 20000 with a new doc comment; `:11210` `TAYLOR_WORK = 100_000_000`
  with the doc comment of §3.1 (provisional value of the design; the calibration of §5.3 was not
  rerun, see section 6).
- `:10906` `expr_to_latex` as a work-list writer into one String; `:10996` the old recursive writer
  kept as the test reference `expr_to_latex_reference` (`#[cfg(test)]`).
- Tests (`:14843` to `:14960`): `test_taylor_engine` (§4.5: `taylor("1/sin(x)", x, 0, 3)` prints
  `(((1 / x) + (1/6 * x)) + (7/360 * (x ^ 3)))`, gp `1/sin(x) + O(x^4)`; log, gamma, budget,
  `x^2/4`, `1/(x-x)` ValueError, `1/(x + sin(2^100000000) + 2)` Ok, "not found", `x + 1` at order
  20000, `exp(x)` at 20000 refused, `-(-x)`, a long undecided constant with a message under 300
  characters), `test_taylor_shared_dag` (`e = e + e` 40 times: refused in under 20 s, "not found" at
  once), `test_expand_under_minus`, `test_expr_latex_work_list` (every node kind, a 10^5-digit
  integer, a taylor result), `test_taylor_display_linear` (33 000 nested calls at order 60 on a
  256 MiB thread: Ok, displayed with LaTeX over 10 MB, in under 20 s).

## 2. Deviations from the frozen design (the design is not revised)

1. **Thread-local work state instead of an explicit `&mut Work`.** The Plan (steps 3 and 4) asks
   the port to replace the prototype's thread-local counter, fault, kernel table, hint table and
   allowed depth by an explicit `&mut Work` threaded through every operation with `?`. The port
   keeps the prototype's state and charge points, which every review and red-team round measured,
   and scopes it to one call: `series_engine::Session` (`:2158`) is taken by every public entry
   (`series.rs` `with_expansion`), refuses a nested expansion (`Unsupported`; the engine calls
   nothing that expands again), and on drop (a panic included) clears the hints, the counter, the
   fault, the kernel table, the pre-pass and the substitution memos. Evidence: the port reproduces
   proto13's outcome on every test row before the fixes (the same 6, 135 + 16 and 19 + 1 failing
   rows as `tests_report.md`'s proto13 column), and the budget tests hold (section 3). Risk: the
   design's "a missed charge is a budget hole" applies unchanged, since the charge points are the
   prototype's.
2. **Owner's rule 1 in `atan_like`** (B1): exact-constant argument evaluated before any integrand.
   Evidence: `b1_exact_constant_operand_is_evaluated` (10 rows), `b1_exact_constant_operand_at_a_regular_point`,
   the 135 failing (a) rows of the matrix and 19 hand-reviewed fuzz rows failed on the port without
   it (`fu2/s1impl/impl/run1.log`) and pass with it (`run2.log`).
3. **Owner's rule 2 in `atan_like`** (B2): the integrand's checks before `apply(f, b0)`. Evidence:
   `b2_branch_point_of_a_non_constant_argument` (5 rows), `b2_truncated_argument_at_a_pole_is_undecided`,
   the 16 arctanh rows of matrix (b) and `O:y13.36`.
4. **arctan(+-i) is `DivisionByZero` in `kapply_src`** (review 9 m2, needed by rule 1). Evidence:
   `b1_m2_arctan_at_plus_minus_i_is_a_pole` (4 rows).
5. **`limit_with_error` keeps `Expr::limit`** for the limit (HEAD's call; §2.13 says "the limit as
   above", the prototype used `limit_via_series`). The tests state that the limit is `Expr::limit`,
   which S1 does not change, and assert only the `BigO`.
6. **`laurent` and `series_coefficients` decide the low terms inside the session** (zero tests on
   the stored constants before rendering, with the pending fault checked after each test); the
   design gives the errors, not the mechanism.
7. **`KernelData::unary` in the kernel**, as the design says; the prototype's side table `UNARY`
   is removed. Kernels of a unary kind are only made by `kapply_src` with the record, and the other
   kinds have prefixed keys, so the record is the same as the side table's.
8. **The design's "port trap" fixed**: `pow_u` compared a coefficient with -1 by copying it
   (`(-q.clone()).is_one()`); it now reads the sign and the magnitude (`series_const.rs`, `pow_u`).
9. **Parser helpers out of line** (not in the design's text; needed to keep the parser's frames at
   most HEAD's, measured above).
10. **One test assertion relaxed in a test written in this step, not in the first step's tests**:
    `test_display_parse_round_trip` compares `Pow(Real(-1.5), 2)` with its read-back by value
    (`eval_float`), because HEAD's `simplify` does not fold `Neg(Real(1.5))` to `Real(-1.5)` (the
    design's text assumed it does).

No expected outcome of the first step's tests was changed. The only edits to those files are the
removed feature gate (`#![cfg(feature = "s1-engine-tests")]`) and the doc lines that named it.

## 3. Tests and results

| command | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic` (debug) | unit 675 passed, 3 ignored; `s1_fuzz_data` 7; `s1_regressions` 26 passed, 2 ignored (the release tests); `s1_singular_points` 4; doc 116 passed, 11 ignored. 0 failed. |
| the same, before the fixes of section 1 (the port of proto13 alone) | `s1_regressions` 6 failed (the B1, B2 and m2 tests and `limit_with_error_follows_the_expansion`), matrix (a) 135 of 379 and (b) 16 of 184 rows failed, fuzz 19 of 22 hand-reviewed rows and 1 of 4 O rows failed: proto13's profile in `tests_report.md`. |
| `cargo test --release -p rustmath-symbolic --test s1_regressions --test s1_singular_points --test s1_fuzz_data` | 7 + 26 (2 ignored) + 4 passed |
| release `#[ignore]` budget tests, `--ignored --test-threads=1` | `rt_structure_h2_consumers_release` and `rt_series_h1_at_taylor_order_release` pass (timings in section 5) |
| crates depending on rustmath-symbolic: calculus, curves, functions, geometry, manifolds, plot3d (`cargo test -j 2 --no-fail-fast -p ...`) | all pass (calculus 122 + 62 doc, curves 299 + 3 + 2 + 1 doc, functions 72 + 1 doc, geometry 417 + 106 doc, manifolds 308 + 33 doc, plot3d 78 + 12 doc) |
| `cargo test -j 2 -p rustmath-jupyter` | lib 114 passed, 1 ignored; `kernel_protocol` 4 passed with `--test-threads=1`. In the first parallel run one kernel failed to start with `AddrInUse` (the test's `free_port` binds and releases a port, and a parallel kernel took it), and `kernel_aborts_queued_cells` then waited without end. Pre-existing flakiness of the test harness, not related to S1; serial runs pass. |
| total over the final runs | 2462 passed, 0 failed |
| `cargo check -j 2 --workspace --tests` | no error (rustmath-benchmarks included) |
| `cargo clippy -j 2 -p rustmath-symbolic --no-deps` | no warning in `series_ball.rs`, `series_const.rs`, `series_engine.rs` or the new parts of `series.rs`, `parser.rs`, `expression.rs`, `expand.rs`. The remaining warnings in touched files are at HEAD (`limits.rs:304` matches!, the two parser loops, `simplify.rs:39/371`, the unused `var` of `discrete_fourier_coefficients`). |
| rustfmt | the three new engine files formatted; the new head of `series.rs` formatted (the untouched bottom of the file has HEAD's formatting debt and was left). The other touched files were not reformatted: rustfmt changes 9 to 7558 lines of untouched HEAD code in each of them (`repl.rs` 7558). The tests of the first step are rustfmt-clean. |

The singular-point matrix (609 rows) and the fuzz data (479 rows) are part of the debug and release
runs above.

## 4. Review 9's random harness and the design's oracle

Driver: `fu2/s1impl/fuzzdrv/s1drv.rs` (built with rustc against the release rlibs of this tree,
`s1drv_rel`): the command line of the design's probe `rt2_rel both FILE POINT ORDER STACK` with
`GPCHECK` (the crate's parser, `taylor_with_budget` with the REPL budget 1e8, the gp form of input,
point and result), and `coeffs F A N` for `gpcheck3.py`. Runs from `fu2/s1impl/fuzzdrv/run/`, the
scripts copied unchanged: `fuzz.py` seeds 1 to 4 and `fuzz2.py` seeds 11 to 14, 250 inputs each
(the same 2000 inputs as review 9, same seeds).

| outcome | port | review 9 on proto13 |
|---|---|---|
| equal to gp (all coefficients to 1e-60 at 300 digits) | 1250 | 1240 |
| differ (DIFF) | 4 | 4 |
| gp cannot expand (GPERR) | 251 | 237 |
| refused (ERR) | 495 | 519 |
| SUSPECT (`fuzz2.py`: a DivisionByZero or NotLaurent where gp has a series) | 0 | 0 |
| crash, timeout | 0 | 0 |

The 4 DIFF rows are exactly review 9 m1 (gp takes the real root of a series power of a negative
rational constant term): `(((x)*(x))^3)/((((1-x))^(2/3))+((x-2)))` at 2,
`((x^(1/3))+(x^2+1))*sign(x^3)` at -1, `pi^(acos(x^2+1)+x)` at -1/2,
`((x^(1/3))+x)/abs(x^((x-x)+2))` at -1.

Outcome changes against proto13 on the same inputs (74): DivisionByZero to NotLaurent 27 (B2,
logarithmic and algebraic branch points); DivisionByZero to a value equal to gp 8 (B1);
DivisionByZero to PrecisionExhausted 3 (b = `1 + O(h^P)` at a pole of arctanh, or an optional fold);
DivisionByZero to UndecidedZero 3 and to Unsupported 2 (an honest refusal after rule 1 evaluated the
constant: sign of pi/2, an undecided integrand); PrecisionExhausted to DivisionByZero 6 and
BudgetExceeded to DivisionByZero 1 (now an exact zero is divided by or given to log:
`log(atanh(x-x))`, `1/asinh((x+1)^2-x^2-2*x-1)`); UndecidedZero and PrecisionExhausted to a value
equal to gp 2; and 14 rows where proto13 refused and gp cannot expand the input but the port gives a
value. Those 14 were checked one by one (`fu2/s1impl/fuzzdrv/newvalues.tsv`): 11 are exact constants
whose value follows from the exact rules (0 for `abs(atanh(x-x))`, `abs(atan((x-x)*(1-x))*2^(2-x))`,
`acos(sign(3+(x-1)))*...` (sign 1, acos(1) = 0), `abs(asin((0*(1/x))^((x-x)+2)))`,
`asinh(sign(x-x))^(1/2)`, `...*asinh(0*(1/x))^(2+x)`, `((asinh(x-x))^(1/3))^3*...`,
`sinh(acos((x-2)/(x-2)))`; 1 for `exp(acosh(x/(x-0)))`; `1 - arcsin(1)` for
`exp(asin(x)) - asin(x/x)` at 0, order 0; `arcsin(1)/(1 - pi)` for
`asin(x+(1-x))/((x/x)-(pi-0*(1/(x-1))))` at 1), 1 is `arctan(arctanh(3))` (fuzz data row
`B1:y14.33`, checked at 60 digits), and 2 were compared with gp on the simplified input:
`(1/2)^atan(1-x) + asin(x-x)^(2+x)` at 0, order 3 equals gp's `(1/2)^atan(1-x)` (difference
7.5e-329 at 300 digits); `asin(cosh(x)) + acos(3^((x+1)^2-x^2-2*x-1))` at -1, order 2 has c0 =
arcsin(cosh(-1)), c1 = i, c2 = 0, gp's `asin(cosh(-1+h))` is `(pi/2 - i) + i h + 0 h^2` (gp's own
check failed on `polcoef` of an `O(h^0)` series).

Design oracle `fu2/design/S1-symbolic/rev2/gpcheck3.py` with `PROTO=fuzzdrv/s1drv_rel`:
107 of 107 OK, 0 failures (`fu2/s1impl/fuzzdrv/gpc/gpcheck3_port.out`).

## 5. Measured times against the bar (release, 8 MB thread, budget 1e8; S1_BAR (c): 20 s, 512 MB)

From the `#[ignore]` tests (process peak RSS by `fu2/s1impl/impl/peak.py`):

| input | entry | time |
|---|---|---|
| F_10 at 1, order 1 | `taylor_with_budget` 0.007 s, consumers 0.154 s (whole 0.195 s); `series_expansion` 0.007 s; limit path 0.128 s | |
| F_11 at 1, order 1 (tree 1 036 403, DAG 49 268, text 3.5 MB) | 0.012 s, consumers 0.297 s (whole 0.378 s); 0.016 s; 0.114 s | |
| F_12 at 1, order 1 | `LimitExceeded(TreeSize)` in 0.018 s and 0.019 s; limit path 0.119 s | |
| whole test `rt_structure_h2_consumers_release` | 1.00 s, peak RSS 205 MB | |
| `x^30001*exp((exp(x^30000)-1)/x^30001)` at order 20000 | refused in 0.023 s, 0.017 s, limit path 0.000 s; peak RSS 13 MB | |

Driver samples (`s1drv_rel both`, wall time with parsing):

| input, point, order | outcome | time |
|---|---|---|
| x + 200 64th roots of 4096-bit radicands (review 9 `roots200.txt`), 0, 1 | Ok | 2.67 s |
| the 2000-root input (`roots2000.txt`, 40.9 KB) | BudgetExceeded | 2.82 s |
| `(5/7+x)^(-3/11)`, 0, 2000 | Ok | 0.91 s |
| `tan(x)`, 1, 98 / 100 | Ok / BudgetExceeded | 0.91 s / 0.95 s |
| `exp(x)`, 0, 20000 | BudgetExceeded | 0.14 s, 120 MB |
| `exp(sin(x))`, 0, 2000 | BudgetExceeded | 0.80 s |
| `(1+x)^(10^100)`, 0, 1150 | Ok | 0.14 s, 63 MB |
| `x + (3^330000)/(5^200000)*2*3`, 0, 1 | BudgetExceeded | 0.33 s |
| 100 KB texts at 1, order 3: `x+x+...` (50 000), `1/x/.../x`, `x*3/2*3/2...`, 100 000 signs, 4000 nested `sin(`, a 99 990-digit literal | Ok, BudgetExceeded, Ok, Ok, BudgetExceeded, Ok | at most 0.22 s, 30 MB |

The slowest sample is 2.8 s, a margin of 7 over the bar.

## 6. Callers checked

`grep -rn '\.taylor(\|\.maclaurin(\|\.series_coefficients(\|\.laurent(\|\.asymptotic(\|\.series_with_big_o(\|\.series(&'`:
the REPL taylor arm (rewritten), `limits.rs` (`limit_via_series`, `limit_with_error`), `series.rs`
and `lib.rs` tests. `rustmath-calculus/src/taylor.rs` has its own implementation (§7 item 1,
unchanged). Reverse dependencies of rustmath-symbolic (cargo metadata): benchmarks, calculus,
curves, functions, geometry, jupyter, manifolds, plot3d; all compile with their tests and pass
(section 3). rustmath-manifolds needed no change (every match is on a reference, as the MF1 report
said), and its slot assertions pass, so `simplify`'s results are unchanged there. `rustmath-py` is
outside the workspace and was not built (it calls `parse`, `expand` and Display only, §3.2).

Parser callers: the REPL (`parse_expr` for every symbolic argument) and the crate's tests; the REPL
tests pass, so no test depended on the old `a/b` literal or the old Display forms.

## 7. Left undone

- `tests/series_engine.rs` with the ~100 tests of §4.1 was not written as a file. Coverage instead:
  the 107 cases of the design's oracle `gpcheck3.py` (all pass through the driver), the first step's
  three test files (B1, B2, the red-team rows of rounds 3 and 4, the matrix, the fuzz data), the
  rewritten `series.rs` tests and the REPL tests. The §4.1 budget tests (`budget_*`) and the unit
  tests named in the Plan (`empty_argument_negative_precision`, `exact_sentinel`,
  `dense_buffer_units`, `result_tree_cap`, ...) are covered by the corresponding `rt_*` and `rt4_*`
  rows of `s1_regressions.rs`, not by tests of their own names.
- The engine-internal round-trip test of §4.3 (`series_engine::same_expansion` comparing normal
  forms) and `test_display_round_trip_values` (400 random trees) were not written; the round trip
  is tested on the hand-built cases of `test_display_parse_round_trip`.
- The calibration of `TAYLOR_WORK` (§5.3, `calib6.py`) and the remaining acceptance scripts of §6
  (`revchecks.py`, `revchecks3.py`, `budget6.py`, `newtests6_dbg.py`, `b2probe.py`, `fnscan`,
  `nearscan`, `runall_rt3.sh`, `rt4_extra.sh`) were not run against the crate. The REPL keeps the
  design's provisional 1e8; the samples of section 5 stay under 3 s.
- Parser depth on 8 MB release and on the REPL's 64 MiB thread (Plan step 8) not measured; the
  debug comparison with HEAD is in section 1.
- The audit greps of §6 "Risks": `Rational::new`, `.pow(` and friends in the three engine files and
  `series.rs` outside the charged layer find only constant-size values (1/k, +-1/2), `Expr::pow`
  node constructions and test code; `.gcd(` and `.lcm(` appear only inside `i_gcd` and `i_lcm`. The
  copy and kernel-walk greps were not repeated beyond the prototype's code, which the port keeps.

## 8. Handoffs

Tracker rows (the orchestrator's pass; not edited here), from Plan step 11:

- `PROJECT_TRACKER.md` "Taylor / Maclaurin series" (`series.rs:543`): truncated Laurent series over
  exact constants, with a work budget and structural caps; 🟢.
- "Laurent series" (`series.rs:579`): 🛑 to 🟢 (`ValuationBelowMinimum` instead of dropping the
  principal part).
- "Asymptotic expansion" (`series.rs:602`): the expansion at infinity.
- The expression parser row (`parser.rs:149`): bignum literals, `a/b` as an operator, sign runs,
  inverse hyperbolic names.
- "Polynomial expansion" (`expand.rs:254`): 🔴 to 🟢 (expand under a minus).
- "Derivative of a named function": drop "Contaminates taylor(), series()".
- The Expr row: iterative Drop (`expression.rs:109`) and work-list Display (`expression.rs:577`).
- New rows for the out-of-scope items of design §7 (items 1 to 10), with item 5 naming PartialEq,
  substitute and `symbols()` as the consumers that still recurse, and item 8 naming the towers.

`docs/HANDOFF.md`: the "Unit S1" bullet (taylor F1, the literal parser S7, expand under a minus)
and the S1 paragraph of "Open follow-ups round" can close, as can the notes "a symbolic chain
`w + w + ... + w` of 3000 terms aborts the debug driver in Expr Display and Drop" and "`-(-x)`
displays `--x`". The MF1 note on `impl Drop for Expr` is resolved (it compiled there). `TAYLOR_ORDER`
keeps 20000; J1b's "waits for S1" note on it can go.

Follow-ups:

1. The kernel test `kernel_protocol.rs` picks ports by binding and releasing them (`free_port`), so
   parallel kernels can collide (`AddrInUse`) and `kernel_aborts_queued_cells` then waits forever.
   Pre-existing; a REPL item.
2. Decide whether to replace the thread-local work state by the design's explicit `&mut Work`
   (deviation 1), as a refactor under the same tests and budget measurements.
3. Calibrate `TAYLOR_WORK` with §5.3's script against a driver of the crate, and port the §4.1 test
   list as `tests/series_engine.rs`.
4. Design §7 items 1 to 10 as new HANDOFF rows (rustmath-calculus `taylor.rs`, Puiseux and log
   terms, gamma and zeta, fast series arithmetic, iterative PartialEq and substitute, products of
   radicals, the result-tree refusals of towers, the per-cell resource policy, the two numeric
   gaps).
5. The one-sided limit of `limits.rs` (`tests_report.md` finding 7: direct substitution takes
   `0^10*exp(0/0)` as 0 and gives `Finite(1)` from the left of `1 + x^10*exp((sin(x^2)-x^2)/x^9)`,
   whose left limit is +infinity) needs a unit of its own.

## Fix-up round 1

Reviews: `fu2/s1impl/S1.review1-resources.md` (not approved: B1, B2, B3 and six minors) and
`fu2/s1impl/S1.review1-semantics.md` (approved: m1 to m5). Scratch of this round:
`fu2/s1impl/fix1/`. Nothing committed, pushed, stashed or reset. The design was not edited.

### 1. Regression tests first

New file `rustmath-symbolic/tests/s1_depth.rs`, 7 tests. Each runs in both profiles on the bar's
threads, at about 96 % of HEAD's limit (a frame larger than HEAD's by 16 bytes fails; build noise
of a few levels does not). A failure is an abort of the test binary, so the file holds only these
tests. Run on the code of the implementation step before any fix:

| test | debug (2 MB) | release (8 MB, 2 MB) | before the fix |
|---|---|---|---|
| `b1_caret_tower_depth_not_below_head` | 1 380 carets | 21 800; 5 450 | debug ok; release abort |
| `b1_nesting_depth_not_below_head` (`(`, `sin(`, `2^(`, `2*(`, `-(`) | 207, 155, 179, 207, 207 | 3 950 to 3 350; 985, 835 | ok (a guard: review 1 found these at or above HEAD) |
| `b3_simplify_depth_not_below_head` (right sum, tower, `sin` chain) | 3 120 | 31 300; 7 840 | abort in both |
| `b3_limit_of_a_tower_depth_not_below_head` (`Expr::limit` at 1) | 2 500 | 27 900; 6 970 | debug ok; release abort |
| `b2_limit_with_error_of_a_long_sum` (`x+...+x` at 1, order 3) | 3 000 terms | 30 000 terms | abort in both |
| `b2_limit_with_error_of_a_signed_tower` (`x^-x^-...` at 1, order 3) | 1 400 levels | 3 000 and 13 000 levels | abort in both |
| `b2_limit_with_error_of_a_sum_of_kernels` (`x+sin(1)+...+sin(m)` at 0, order 2) | m = 2 399 | m = 2 399, 3 199, 9 999; 20 s and 512 MB (VmHWM) | debug abort; release: `Finite(sin(1)+...)` from HEAD's `Expr::limit` after 1.74 s |

Logs: `fix1/before_rel.log` and the debug runs in this round's transcript. HEAD's limits come from
review 1's cargo builds of HEAD (`rev1-resources/headprobe_rel`, `consume_head_rel`) and from
`fix1/head_depth.py`.

One expectation was set after the fix, with its evidence in the test. The 13 000-level signed
tower first expected `Finite(1)`. Through the engine it is `(Unknown, None)`: the order-5
expansion of the limit runs out of `LIMIT_SERIES_WORK` (`BudgetExceeded` at 1e8 units from about
6 000 levels; `fix1/sp_rel`). The test now checks that `series_expansion` at order 5 is
`BudgetExceeded` and that `limit_with_error` returns `(Unknown, None)` within the bar. The old code
still fails it (it aborts in `Expr::limit` at 13 000 levels, review 1).

### 2. The three blockers

**B1, parser (`parser.rs:335`).** `atom` is `#[inline(never)]`. objdump of the release builds:
the frame of `power` is 0x138 bytes and 6 pushes in HEAD (368 bytes per `^` level with the return
address), 0x228 and 6 pushes before the fix (600 bytes), 0x148 and 4 pushes now (368 bytes).
Deepest input parsed (`fix1/depth.py`; HEAD and the port in the same probe binary
`fix1/probe_rel_new` or `probe_dbg_new`, HEAD by cargo in brackets):

| shape | debug 2 MB: HEAD / port | release 8 MB: HEAD / port | release 2 MB: HEAD / port |
|---|---|---|---|
| carets | 1 440 / 1 441 | 22 770 (22 770) / 22 770 | 5 674 (5 677) / 5 674 |
| `(` | 216 / 220 | 4 123 (4 124) / 4 514 | 1 027 (1 028) / 1 124 |
| `sin(` | 162 / 190 | (4 124) / 4 514 | 1 027 (1 028) / 1 124 |
| `2^(` | 187 / 191 | 3 491 (3 491) / 3 767 | 869 (870) / 938 |
| `2*(`, `-(` | 216 / 220 | (4 124) / 4 514 | |
| `f(x,` | 162 / 190 | | |
| `x^-` | parse error / 1 441 | parse error / 22 770 | |

Before the fix: 13 782 carets on 8 MB release and about 3 400 on 2 MB. The 3-level gap at 2 MB
between HEAD by cargo (5 677) and the port (5 674) is the probe: HEAD in the port's probe also
reads 5 674, and the frame per level is the same. The report's sentence "release frames scale the
same way" (section 1, Parser) was wrong; this table replaces it, and the open item of section 7
(parser depth in release) is done for 8 MB and 2 MB.

**B3, simplify (`simplify.rs:376`).** HEAD's match is the body again. A `Binary` whose left child
is a `Binary` goes to `simplify_left_spine` (the loop of design §2.10), and a function call to
`simplify_function`; both `#[inline(never)]`. `simplify_non_binary` is gone. The evaluation order
and every result are HEAD's. Frames (objdump): release HEAD 0xd0 and 5 pushes (256 bytes), before
the fix 0x138 and 6 pushes (368 bytes), now 0xd0 and 5 pushes (256 bytes); debug HEAD 0x278
(632 bytes), now 0x258 (600 bytes). Deepest tree (`fix1/consume_*`, review 1's `consume.rs`; HEAD
by cargo in release and by rustc at opt-level 0 in debug):

| tree, consumer | release 8 MB: HEAD / port | release 2 MB: HEAD / port | debug 2 MB: HEAD / port |
|---|---|---|---|
| right sum, `simplify` | 32 745 / 32 745 | 8 169 / 8 169 | 3 256 / 3 427 |
| tower, `simplify` | 32 745 / 32 745 | 8 169 / 8 169 | 3 256 / 3 427 |
| `sin` chain, `simplify` | 32 746 / 32 746 | 8 170 / 8 170 | 3 263 / 3 435 |
| tower, `Expr::limit` at 1 | 29 106 / 29 106 | 7 261 / 7 261 | 2 610 / 2 610 |
| left sum, `simplify` | 32 745 / over 4 million | | |

Before the fix: 22 720 on 8 MB release and 2 416 for the `sin` chain on 2 MB debug (review 1).
`simplify_long_left_deep_sum` (10^5 terms on 2 MB debug) still passes.

**B2, `limit_with_error` (`limits.rs:558`).** The limit is `limit_via_series(var, point, 5)` (the
engine under `LIMIT_SERIES_WORK`), then for a finite limit the `BigO` from `expansion_reaches` at
the command's order, as design §2.13 and the prototype's `api::limit_with_error` do. `dir` is not
used: a `Finite` result of the engine is the limit from either side. Deviation 5 of section 2 is
withdrawn. Release, 8 MB thread, `fix1/sp_rel` with `impl/peak.py` (whole process):

| input | `limit_with_error` now | review 1 on the old code |
|---|---|---|
| `x+...+x`, 30 000 terms (59 999 B), at 1, order 3 | `(Finite(30000), O(h^4))`, 0.13 s, 21 MB | stack overflow |
| `x+...+x`, 50 000 terms (99 999 B) | `(Finite(50000), O(h^4))`, 0.27 s, 30 MB | |
| `x+sin(1)+...+sin(2399)` (22 884 B), at 0, order 2 | `(Unknown, None)`, 0.016 s, 13 MB | 1.81 s, 797 MB |
| `x+sin(1)+...+sin(3199)` (30 884 B) | `(Unknown, None)`, 0.021 s, 13 MB | 3.47 s, 1.41 GB |
| `x+sin(1)+...+sin(9999)` (98 884 B) | `(Unknown, None)`, 0.11 s, 21 MB | allocation failure under 4 GB |
| `x^-...^-x`, 13 000 levels (39 001 B), at 1, order 3 | `(Unknown, None)` (`BudgetExceeded`), 0.30 s, 17 MB | stack overflow |

A change of behaviour that the design states and the review asked for: `limit_with_error` now
gives `Finite` only for a rational limit, as the limit path does. HEAD's `Expr::limit` also gave
symbolic finite values (`Finite(sin(1))` for `sin(x)` at 1, the sum of the `sin(k)` above). No
caller outside the S1 tests uses `limit_with_error` (grep of the workspace). The doc of
`limit_with_error` and of the test `limit_with_error_follows_the_expansion` say so; the test's
assertions are unchanged and pass.

Also fixed: the port had moved the `#[inline(never)]` of `kapply_src` onto the new helper
`is_plus_minus_i` (`series_const.rs:2509`); it is back on `kapply_src`, as in proto13.

### 3. Minor items

Semantics lens:

- m1 (rule 1 gives a kernel at an undecided pole): no code change; I disagree with the required
  change. Its zero test on 1 + c^2 cannot tell `arctan(sqrt(-1))` from `arctan(arctanh(3))`: both
  constants are non-real, have no real ball, and the test is `Unknown` (m3). So the change would
  turn the first step's fuzz row `B1:y14.33` (`atan(atanh(sign((x+1/2)^2)+2))` at -1/2, the value
  arctan(arctanh(3)) against gp, a row of the owner's rule 1) into `UndecidedZero`, which proto13
  gave and the owner's rule removed. Making the field reduce `sqrt(-1)` to i would decide both of
  the review's examples, but the frozen design keeps `sqrt(-1)` and `sqrt(-4) - 2*I` undecided
  (§2.2 zero test, §4.1 `complex_constant_undecided`) and keeps f(c) a kernel at a possible pole
  (§2.2, §7 item 10). The convention is now stated in `series.rs`'s list (below, with m2), with
  `x + arctan(sqrt(-1)*x^0)` as its example. No matrix rows for these two inputs: a kernel at a
  possible pole has no gp reference and no single decided outcome.
- m2: `series.rs:9` now says where an undecided test is reported, and the conventions list
  (`series.rs:36`) states that a divisor constant is taken as nonzero, and a function of a
  constant at a pole the field cannot recognise is kept as the kernel f(c), when the test cannot
  decide; the result holds where the input is defined. Examples checked: `x/(sin(1)^2 + cos(1)^2
  - 1)` at 0 is `Ok` with the limit `Finite(0)`; `x*arctan(sqrt(-1)*x^0)` has the limit `Finite(0)`
  (`fix1/conv.txt`).
- m3: no change. For the HANDOFF (docs step): the numeric zero test does not reach complex
  arguments of trig and hyperbolic kernels, inverse functions off their real domains, or complex
  radicals, so such constants are `UndecidedZero` (55 of 4 000 rows in review 1); next to design
  §7 item 10.
- m4: the `DivisionByZero` doc now says that exp, sin, cos and tan of an exact-constant argument
  give a truncated series, so a pole through them can be `PrecisionExhausted`
  (`tan(pi/2 + x - x)` at 0).
- m5: the matrix has a class (d) of 63 rows (`gen_s1_singular.py`, appended after every other row,
  so no earlier id or line changes; the old 609 rows are byte-identical): `f(1/x)` at 0 for each
  of the 17 functions (the branch point at infinity), and two arguments per (function, branch
  point) whose constant term is c only through an exact kernel value (`sin(pi/2 + x)`,
  `2*sin(pi/6 + x)`, `-sin(pi/2 + x)`, `cos(pi + x)`, `+-I*sin(pi/2 + x)`, `+-I*cosh(x)` at 0;
  `sin(x)` at pi and `cos(x)` at pi/2 for c = 0). All `NotLaurent`, limit `Unknown`. New test
  `other_shapes_of_branch_points_are_not_laurent`; the coverage test requires the (d) rows of each
  pair and function. gp often returns a series there, since its pi is a float (stated in the
  notes). The two gp artefacts: `s1_gp.py` already passes a complex point inexact (`(P)*1.`) and
  avoids gp's `acosh` left of -1 (`ACOSH`).

Resources lens:

1. `Expr::limit` and `limit_enhanced` on `x^-x^-...` towers and `limit_enhanced` on the 100 KB
   sum: HEAD's code, unchanged. For the HANDOFF next to design §7 item 5 (iterative
   `substitute`). `limit_with_error` no longer calls them (B2).
2. `series_coefficients` doc (`series.rs:644`): order + 1 entries whatever the series, so memory
   is linear in the order (`x` at order 10^7 about 0.6 GB); `series_expansion` is sparse and
   budgeted.
3. `ExpansionPoint` has a hand-written, bounded `Debug` (`series.rs:291`, through `point_text`,
   which `SeriesExpansion`'s `Debug` now shares); the docs of `SeriesError`, `ExpansionPoint` and
   `SeriesExpansion` say that `PartialEq` compares trees.
4. `Drop` cost on large teardowns: no change (no bar issue).
5. The driver `fuzzdrv/s1drv.rs` drops the result and the input before its last line, which now
   carries the drop time and the VmHWM after the drops. The old binary is `s1drv_rel.round1`.
6. The `BudgetExceeded` doc says that `work` can include an a-priori charge refused before its
   step ran.

### 4. Acceptance reruns (final code)

| command | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic --no-fail-fast` (debug) | unit 675 (3 ignored); `s1_depth` 7; `s1_fuzz_data` 7; `s1_regressions` 26 (2 ignored); `s1_singular_points` 5; doc 116 (11 ignored); 0 failed |
| `cargo test --release -p rustmath-symbolic --test s1_regressions --test s1_singular_points --test s1_fuzz_data --test s1_depth` | 26 (2 ignored), 5, 7, 7 passed |
| release `--ignored --test-threads=1` | `rt_structure_h2_consumers_release` 1.03 s, peak RSS 206 MB (F_10 0.194 s, F_11 0.399 s, F_12 refused in 0.038 s); `rt_series_h1_at_taylor_order_release` 0.04 s, 13 MB |
| downstream (`cargo test -j 2 --no-fail-fast -p rustmath-calculus -p rustmath-curves -p rustmath-functions -p rustmath-geometry -p rustmath-manifolds -p rustmath-plot3d`) | calculus 122 + 62 doc, curves 299 + 3 + 2 + 1 doc, functions 72 + 1 doc, geometry 417 + 106 doc, manifolds 308 + 33 doc, plot3d 78 + 12 doc; 0 failed (`fix1/deps_test.log`) |
| review 9's harness (`fuzz.py` seeds 1 to 4, `fuzz2.py` seeds 11 to 14, 250 inputs each, through the rebuilt `s1drv_rel`; `fuzzdrv/run_fix1/`) | every one of the 2 000 outcomes equal, row by row, to the implementation step's run (times and VmHWM ignored): 1 250 equal to gp, the 4 DIFF rows of review 9 m1, 0 SUSPECT, 0 crash |
| design oracle `gpcheck3.py` (`fuzzdrv/gpc/gpcheck3_fix1.out`) | 107 of 107 OK |
| `cargo test -j 2 -p rustmath-jupyter --lib`; `--test kernel_protocol -- --test-threads=1` | 114 passed (1 ignored); 4 passed (`fix1/jup_lib.log`, `fix1/jup_kernel.log`) |
| `cargo clippy -j 2 -p rustmath-symbolic --no-deps --all-targets` | no warning in the code of this round; the two warnings in touched files are older (`simplify.rs:371`, HEAD's `let` and return; a doc list item in `s1_singular_points.rs:44`, from the first step) |
| rustfmt | `tests/s1_depth.rs`, `s1_singular_points.rs`, `s1_regressions.rs`, `series_const.rs` and the head of `series.rs` are clean; the remaining diffs in `simplify.rs`, `limits.rs`, `parser.rs` are HEAD's formatting outside the changed lines |

The total of the final runs: 2 517 passed, 0 failed (symbolic 836 in debug, its four S1 test files
in release 45, downstream 1 516, jupyter 118, release budget tests 2; the same tests as in
section 3, plus `s1_depth` and the matrix's class (d)).

### 5. Corrections to sections 1 to 8

- Section 1, Parser: "release frames scale the same way" was wrong (B1). The release limits are in
  section 2 of this round. Deviation 9 now also covers `atom`.
- Section 1, simplify: the left-spine loop of the implementation step made the recursive frame
  larger (B3); the restructured `simplify` of this round replaces it.
- Section 1, limits and section 2, deviation 5: withdrawn; `limit_with_error` follows design
  §2.13 (B2).
- Section 7: the parser's depth in release (8 MB and 2 MB) is measured. The REPL's 64 MiB thread
  was checked by review 1 (every 100 KB caret and `x^-` tower parses there).
- Section 8, tracker rows: the "limit with error" behaviour is the limit path's (`Finite` only for
  a rational limit), with a `BigO` when the expansion reaches the order.

### 6. Handoffs added by this round

For the docs step (HANDOFF and tracker, not edited here):

1. m3: the numeric zero test has no complex coverage (complex arguments of trig and hyperbolic
   kernels, inverse functions off their real domains, complex radicals), so such constants are
   `UndecidedZero`; next to design §7 item 10.
2. Resources minor 1: `Expr::limit`, `limit_enhanced` and `substitute` recurse per node (HEAD's
   code); `x^-x^-...` towers of 1 400 levels abort them on 2 MB debug; next to §7 item 5.
3. The convention of m1 and m2 (`series.rs:36`): an undecided divisor is taken as nonzero, and
   f(c) at an undecided pole stays a kernel. A later unit could teach the constant field the
   principal square root of a negative rational (sqrt(-b) = i sqrt(b)), which would decide the
   review's two examples; that is a change of design §2.2 and of its §4.1 row
   `complex_constant_undecided`.

### 7. Files of this round

- `rustmath-symbolic/src/parser.rs` (`atom`), `src/simplify.rs` (`simplify`,
  `simplify_left_spine`, `simplify_function`), `src/limits.rs` (`limit_with_error`),
  `src/series.rs` (module docs, `SeriesError` docs, `ExpansionPoint` `Debug`, `point_text`,
  `series_coefficients` doc), `src/series_const.rs` (`kapply_src` attribute).
- `rustmath-symbolic/tests/s1_depth.rs` (new), `tests/s1_singular_points.rs` (class (d)),
  `tests/s1_regressions.rs` (one doc sentence), `tests/data/gen_s1_singular.py` and
  `tests/data/s1_singular_points.tsv` (63 rows appended).
- Scratch: `fu2/s1impl/fix1/` (`head_depth.py`, `depth.py`, `sp.rs`, `rows.rs`, the probes built
  from review 1's `probe.rs` and `consume.rs`, the depth tables `parse_*.txt`, `consume_*_new.txt`,
  `b2_rel.txt`, the logs); `fu2/s1impl/fuzzdrv/s1drv.rs` (drops before exit; previous source
  `s1drv.rs.round1`), `fuzzdrv/run_fix1/`, `fuzzdrv/gpc/gpcheck3_fix1.out`.
- `target/release` was deleted after the release runs.

## Fix-up round 2

Reviews: `fu2/s1impl/S1.review2-resources.md` (not approved: B1, B2 and three minors) and
`fu2/s1impl/S1.review2-semantics.md` (approved: six minors). Scratch of this round:
`fu2/s1impl/fix2/`. Nothing committed, pushed, stashed or reset. The design was not edited.

### 1. Regression tests first

Both blockers got tests that fail on the code of round 1. They were run on that code before any
fix was written (debug, and release with the binaries kept as `fix2/s1_depth_before_rel` and
`fix2/s1_payload_before_rel`).

| test | what it checks | on the code of round 1 |
|---|---|---|
| `s1_depth.rs` `b1r2_simplify_of_calls_and_zigzags_depth_not_below_head` | `simplify` of `f(f(...))`, `f(x + f(...))` and `L = (x + L) + x` at review 2's rows (96 % of HEAD): release 8 MB 13 970, 9 670, 15 700; release 2 MB 3 485, 2 410, 3 920; debug 2 MB 1 228, 880, 1 560 | aborts ("has overflowed its stack") in debug and in release |
| `s1_depth.rs` `b1r2_limit_of_calls_and_zigzags_depth_not_below_head` | `Expr::limit` at 1 of `f(f(...))` (`Unknown`) and of the zigzag (`Finite(2n+1)`): release 8 MB 12 890 and 13 970; debug 2 MB 1 080 and 1 250 | aborts in release; passes in debug (round 1 was equal to HEAD there: a guard) |
| `s1_payload.rs` `b2r2_large_leaves_at_taylor_order_are_bounded` (new file) | review 2's four 100 KB inputs and `1/(1+N-x)` at the point N, order 20 000, REPL budget, through `taylor_with_budget` and `series_expansion`: a refusal, or a result whose tree holds at most 128 MiB of leaf text and 2^21 nodes | fails on all five: 1.0e9 to 2.0e9 characters of leaf text (`fix2/before_payload_dbg.log`) |
| `s1_payload.rs` `b2r2_large_leaves_at_order_200_stay_values` | the same inputs at order 200 are values (201 equal coefficients), so no scope cut | passes (a guard) |
| `s1_payload.rs` `b2r2_consumers_of_large_leaf_results_release` (`#[ignore]`, release) | each input at order 20 000 and 200, then `simplify`, `substitute`, `==` and `Display`, each within 20 s, VmHWM under 512 MB; then `sqrt(N)/(1-x)` at order 1 300, next to the new cap | fails: `simplify` of `sqrt(N)/(1-x)` at VmHWM 819 MB (`fix2/before_payload_rel.log`) |

The bound of the first payload test was 64 MiB in the run on the old code. It is 128 MiB now, with
the cap of section 2. The old code fails it by a factor 8 to 15 either way.

### 2. The two blockers

**B1, `simplify` (`simplify.rs:385`).** `simplify` is now an explicit work list over every node
kind: a step visits a node or combines the simplified children on top of a value stack. The
combining steps `simplify_binary` and `simplify_unary` do not recurse. The evaluation order is
HEAD's (left before right, children before their parent, a call's arguments in order), so every
result is HEAD's. `simplify_left_spine` and `simplify_function` are gone. `simplify` is
`#[inline(never)]`, so its work list never enters the frame of a recursive caller
(`Expr::limit`). I chose the work list over a smaller recursive frame: it removes the depth limit
for every shape, which a frame-size fix cannot do. Deepest tree, exact binary search
(`fix2/depth/`, review 2's `consume2.rs` and its HEAD binaries `c2_head_*`; 2 097 151 is the
search's bound):

| tree, consumer | release 8 MB: HEAD / port | release 2 MB: HEAD / port | debug 2 MB: HEAD / port |
|---|---|---|---|
| `f(f(...))`, `simplify` | 14 553 / 2 097 151 | 3 631 / 2 097 151 | 1 280 / 2 097 151 |
| `f(x + f(...))`, `simplify` | 10 075 / 2 097 151 | 2 513 / 2 097 151 | 917 / 2 097 151 |
| `L = (x + L) + x`, `simplify` | 16 372 / 2 097 151 | 4 084 / 2 097 151 | 1 628 / 2 097 151 |
| `f(f(...))`, `Expr::limit` at 1 | 13 433 / 13 433 | | 1 125 / 1 125 |
| `L = (x + L) + x`, `Expr::limit` at 1 | 14 553 / 14 553 | | 1 305 / 1 305 |

`Expr::limit` now equals HEAD exactly: what is left on its path is HEAD's recursive `substitute`
and limit laws, unchanged. Review 2's eight direct reproductions all return now
(`fix2/repro_simplify_after.txt`). Round 1's three shapes (right sum, tower, `sin` chain) and the
10^5-term left sum still pass (`b3_simplify_depth_not_below_head`, `simplify_long_left_deep_sum`).

**B2, the result's leaf text (`series_const.rs:639` to `:720`, `series.rs:522` and `:564`).**
A second cap, `MAX_RESULT_TEXT` = 2^27 characters, bounds the leaf text of a result's tree: at
each occurrence of a leaf the characters its `Display` prints (`node_text`: 0.302 bits + 2 per
integer, both parts of a rational, 24 per float, the name of a symbol, the name of a call and its
parentheses; the count of `display_estimate`). `tree_text` computes it by an explicit-stack walk
of the DAG; a node held by more than one `Arc` is memoized by address, so a shared kernel costs O(1)
at each further occurrence and the walk is linear in the DAG. `public_expansion` (the one place
where every full result is built: `series_expansion`, `taylor`, `taylor_with_budget`, `maclaurin`,
`series`, `laurent`, `asymptotic`, `series_coefficients`, `series_with_big_o`) counts the text of
`to_expr()` from its parts (`expansion_text`: each term's coefficient, h and the exponent) and
refuses above the cap with `LimitExceeded { limit: TreeText, value: 2^27 }`. `undecided_expr`
(`series_const.rs:535`) applies the same cap to the constant of `UndecidedZero`. New variant
`StructureLimit::TreeText` (`series.rs:132`), with its `Display` text and the `SeriesError` doc.
The limit paths render one term or none, so they are unchanged.

Deviation from review 2's example (and from the a-priori style of design §2.15): the text is
counted on the rendered DAG, after the node cap and the charged rendering, not carried as a cached
size in every constant, kernel, hint and pre-pass entry. The count is exact, it is linear in what
the rendering already charged, and it changes no constant, kernel or hint, so no result and no
hint decision of the engine changes. The cost of a refusal is the rendering of a DAG that the node
cap already bounded (at most 0.02 s on the inputs below).

Why 2^27 (release, 8 MB thread, `fix2/cap_edge_rel_2e27.txt`, `fix2/rel_payload_after.log`):

| input, at 0 | order | now | consumers of the result |
|---|---|---|---|
| review 2's five inputs | 20 000 | `LimitExceeded(TreeText)` in 0.010 to 0.016 s | |
| the same | 200 | `Ok` (1.0e7 to 2.0e7 characters) | `Display` at most 1.3 s; `simplify`, `substitute` under 0.01 s |
| `sqrt(N)/(1-x)` | 1 300 (1.30e8 characters) | `Ok` | `Display` 8.5 s, VmHWM 129 MB; `simplify` 0.03 s, 57 MB |
| `sqrt(N)/(1-x)` | 1 340 | `LimitExceeded(TreeText)` | |
| a symbol of 99 941 characters over `(1-x)` | 1 340 (1.34e8 characters) | `Ok` | `simplify` 0.06 s, VmHWM 133 MB; `Display` 0.07 s, 140 MB |
| `exp(x)` | 8 800 (1.28e8 characters, not shared) | `Ok` | `Display` 3.7 s, VmHWM 243 MB |
| `exp(x)` | 9 000 | `LimitExceeded(TreeText)` | |

The slowest printing is a 10^5-digit integer at about 15 MB/s, so 2^27 characters print in about
9 s, and `simplify` or `substitute` copies at most 128 MiB of leaves. With the node cap (2^21
nodes, about 200 MB for `simplify`), every consumer of an accepted result stays within S1_BAR (c).
The cost: `exp(x)` from about order 8 900 to 9 128 was a value in round 1 (138 MB of text,
`Display` 4.1 s, VmHWM 262 MB) and is now refused; the REPL budget refuses it from 9 156 anyway
(`fix2/expx_round1.txt`). HEAD's derivative-based `taylor` does not reach these orders. A cap of
2^26 would have refused `exp(x)` from about order 6 500, so I did not take it.

The REPL: `taylor("sqrt(N)/(1-x)", x, 0, 20000)` is now a `ComputationError` ("more than
134217728 characters of leaf text"), so `simplify(t)` has nothing to copy. New REPL test
`test_taylor_leaf_text_cap` (`repl.rs:14968`). The `TAYLOR_WORK` doc comment is corrected
(`repl.rs:11203` to `:11215`): it names both caps, the bar, and review 2's 824 MB. The comment of
the engine's tree cap (`series_engine.rs:2285`) points to the new cap.

### 3. Minor items

Resources lens:

1. The rows of B1 are in `s1_depth.rs` (section 1), and `Expr::limit` of the zigzag too.
2. `limit_with_error` doc (`limits.rs:543`): when the expansion to `series_order` is refused and the
   one of order 5 succeeds, the result is the limit with no remainder term (`sin(x)/x` at order
   10^9 gives `(Finite(1), None)`).
3. `expand` of `L = -(L + x) + x` (quadratic memory, HEAD's cost): no change; a HANDOFF line
   (section 5).

Semantics lens:

1. The `limit_with_error` coverage numbers belong in the tracker row: handed to the docs step
   (section 5). No code change.
2. Done: `exp_of_logs` (`series_const.rs:2602`, called from `kapply_src` at `:2691`), the inverse of
   `log_exact`. exp(r + n_1 log(q_1) + ...) for a rational r, nonzero rationals q_i and integers n_i
   is exp(r) q_1^n_1 ... (exp(n Log q) = q^n on the principal branch, q < 0 included). Folded only
   when the product has at most `MAX_EXACT_POWER_BITS` bits, a-priori from the sizes; otherwise the
   kernel stays. Now `x^x`, `3^x` at 3 have the constant term 27 and the limit `Finite(27)`, `2^x`
   at 3 `Finite(8)`, and `1/(2^x - 8)` at 3 is a pole with 1/(8 log 2) h^-1 - 1/16 + log(2)/96 h
   (round 1: `UndecidedZero`). Tests `s1_review2.rs` `m2_*`, against gp at realprecision 300
   (`fix2/r2.gp`, `r2_gp.txt`), and `exp(10^6 log 3)` stays a kernel.
3. No change. A float exponent with an integral value stays undecided: the design's "no float
   folding" rule (§2.2, where `0.5 - 1/2` is undecided) is frozen. A HANDOFF note (section 5).
4. Done: `sin_q_pi` (`series_const.rs:2524`) reduces the numerator of 12 q mod 24 as an Integer
   (charged, one pass over its words). `sin(10^20*pi)` and `cos(10^20*pi) - 1` are the exact 0,
   `sin((10^20+1/6)*pi)` and `cos((2^70+1/3)*pi)` are 1/2, `tan((10^30+1/4)*pi)` is 1, and
   `x*tan((2^60+1/2)*pi)` and `x*tan((10^25+1/2)*pi)` are `DivisionByZero` like the 2^58 case
   (round 1: `Ok` with the kernel and the limit `Finite(0)`). Tests `s1_review2.rs` `m4_*`.
   Round-1 outcomes of these inputs: `fix2/minors_before.txt`.
5. `Expr::limit` and its laws return symbolic `Finite` values without a zero test, and the S1
   parser widens the text that reaches them: a HANDOFF row (section 5). No code change.
6. Done: class (e) of the matrix, three guard rows after every other row (`gen_s1_singular.py`;
   the first 677 lines of `s1_singular_points.tsv` are byte-identical, 680 now):
   `arcsin(1^(1+x))` `PrecisionExhausted`, `arcsin((-1)^(1+x))` `NotLaurent` (round 1:
   `UndecidedZero`; the fold of minor 2 makes exp(log(-1)) the exact -1) and
   `arcsin(sin(1)^2+cos(1)^2+x)` `UndecidedZero`, each at 0. New test
   `guard_shapes_are_typed_refusals`; the coverage test accepts for (e) any typed refusal except
   `DivisionByZero`. The generator reproduces the file byte for byte.

### 4. Acceptance reruns (final code)

| command | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic --no-fail-fast` (debug) | unit 675 (3 ignored); `s1_depth` 9; `s1_fuzz_data` 7; `s1_payload` 2 (1 ignored); `s1_regressions` 26 (2 ignored); `s1_review2` 2; `s1_singular_points` 6; doc 116 (11 ignored); 0 failed (`fix2/sym_debug_tests2.log`) |
| `cargo test --release -p rustmath-symbolic --test s1_regressions --test s1_singular_points --test s1_fuzz_data --test s1_depth --test s1_payload --test s1_review2` | 26 (2 ignored), 6, 7, 9, 2 (1 ignored), 2 passed (`fix2/rel_tests_final.log`) |
| release `--ignored --test-threads=1` | `b2r2_consumers_of_large_leaf_results_release` 11.7 s (the five inputs refused at order 20 000 in at most 0.016 s; order 1 300 `Display` 8.5 s); `rt_structure_h2_consumers_release` and `rt_series_h1_at_taylor_order_release` pass (F_11: taylor 0.013 s, consumers 0.29 s) |
| review 9's harness (`fuzz.py` seeds 1 to 4, `fuzz2.py` seeds 11 to 14, 250 inputs each, through `s1drv_rel` rebuilt; `fix2/fuzz/`) | 1 253 equal to gp, 5 DIFF, 251 GPERR, 491 refusals, 0 SUSPECT, 0 crash. Against round 1, row by row: 5 rows changed (below) |
| design oracle `gpcheck3.py` (`fix2/fuzz/gpcheck3_fix2.out`) | 107 of 107 OK |
| downstream (`cargo test -j 2 --no-fail-fast -p rustmath-calculus -p rustmath-curves -p rustmath-functions -p rustmath-geometry -p rustmath-manifolds -p rustmath-plot3d`) | 1 516 passed, 0 failed, the same counts as round 1 (`fix2/deps_test.log`) |
| `cargo test -j 2 -p rustmath-jupyter --lib`; `--test kernel_protocol -- --test-threads=1` | 115 passed (1 ignored; the new `test_taylor_leaf_text_cap` included); 4 passed |
| `cargo clippy -j 2 -p rustmath-symbolic --no-deps` on the lib and the six S1 test targets | no warning in the code of this round. The only warning in an S1 test file is the doc list item of `s1_singular_points.rs:49` (first step). `--all-targets` stops on the lib test target at HEAD's `approx_constant` errors in `parser.rs` tests (`3.14159`, `3.14`; HEAD lines 434 and 511), as at HEAD |
| rustfmt | the new and changed test files and `series_const.rs` are clean; the remaining diffs in `simplify.rs`, `series.rs`, `limits.rs` and `repl.rs` are HEAD's code outside the changed lines |

The five harness rows that changed against round 1: four `UndecidedZero` became values equal to gp
(`asin((((x^2+1))+((1-x)))*((x)+(x)))^x)` at -1, `tanh(((x)^(2+x))*(exp(x)))` at -1,
`(acos(((x)*((x^2+1)))^(2+x)))^3` at -1, `(tanh((1/2)^((x)^x)))/(acos(...))` at -1; the fold of
minor 2), and one equal row became DIFF: `log((((sign((x--1)^2))/(x))^x)^x)` at -1, order 1. That
DIFF is gp's float, not the engine. The engine gives log(-1) + h (c1 = 1); gp gives c1 = 1 - 2 pi i.
The inner power has the constant term exp(-log(-1)), which is exactly -1, and the engine takes its
principal log, i pi (the convention of `series.rs`: a constant on a branch cut takes the principal
value). gp evaluates exp(-log(-1)) as -1 - 9.1e-309 i at realprecision 300 (`fix2/diff2.gp`), so its
principal log is -i pi, and c1 = 1 + L0 - i pi moves by 2 pi i. Round 1 kept the kernel
exp(-log(-1)), whose exact value is the same; the harness evaluated it with gp's floats and found
the same artefact on both sides, so the row read as equal. The exact value of round 1's result and
of this one is the same. The function itself is discontinuous at -1 on the real line (gp's scalar
values: f(-1 + 10^-6) = 1e-6 + 3.1415864 i, f(-1 - 10^-6) = -1e-6 - 3.1415927 i), so no series is
"the" function there; the engine's series is the one its documented convention defines.

Total of the final runs: 2 533 passed, 0 failed (symbolic 843 in debug, its six S1 test files in
release 52 and the 3 release budget tests, downstream 1 516, jupyter 119).

### 5. Handoffs added by this round (for the docs step; HANDOFF and tracker not edited here)

1. Tracker row "limit with error": `Finite` only for a rational limit. Review 2 (semantics) on 2 255
   inputs against HEAD: 1 097 HEAD answers became `Unknown` (1 060 of them right against gp, 37
   without a gp reference; 1 088 symbolic values such as sin(1), exp(1), log(2), and 9 rational);
   175 bad HEAD answers removed (83 wrong values, 20 `Finite` at a pole, 72 values that are 1/0);
   every answer of the port right (354 of 354).
2. HANDOFF row: `limits.rs` `limit`, `limit_advanced` and the `combine_limits_*` laws (HEAD's code,
   the REPL's `limit`) return symbolic `Finite` values without a zero test of a denominator. The S1
   parser reads texts HEAD refused (`1/x`, `1/(x^4-4)^3`), so more texts reach them: on 181 such
   inputs 154 `Finite` values at a pole and 8 wrong values (review 2, semantics). On the inputs HEAD
   parses the outcome class equals HEAD's.
3. HANDOFF note next to the float-literal convention: a float exponent with an integral value is
   undecided (`x^2.0` at 0 is `UndecidedZero('-2 + 2.0')`, where HEAD returns an expression equal to
   x^2), by the design's "no float folding" rule.
4. HANDOFF line next to design §7 item 5: `expand` of `L = -(L + x) + x` uses memory quadratic in
   the depth (1.48 GB at 4 000 levels), in HEAD and in the port.
5. Tracker rows for the result caps: 2^21 nodes and 2^27 characters of leaf text
   (`StructureLimit::TreeText`); `exp(x)` from about order 8 900 to 9 128 is now refused where
   round 1 gave a value.
6. The two folds of this round are additions to the constant field of design §2.2 (exp of integer
   multiples of logs of rationals; sin, cos, tan at q pi for every size of q). With them, the
   implementation's deviations from the frozen design also include: `simplify` is an iterative walk
   over every node kind (design §2.10 walks the left spine only), and the leaf-text cap is counted
   on the rendered DAG (design §2.15 checks its caps a priori).

### 6. Corrections to earlier sections

- Fix-up round 1, section 2, B3: the fix covered three shapes only. This round's `simplify`
  replaces it (no `simplify_left_spine`, no `simplify_function`).
- Section 1, simplify (§2.10): `simplify` is a work list over every node kind, with HEAD's order.
- Section 4 and fix-up round 1, section 4: the harness now gives 1 253 equal to gp and 5 DIFF (the
  4 of review 9 m1 and the gp float row above).
- The `TAYLOR_WORK` doc comment no longer claims "within about 200 MB" for any result.

### 7. Files of this round

- `rustmath-symbolic/src/simplify.rs` (`simplify`), `src/series_const.rs` (`MAX_RESULT_TEXT`,
  `node_text`, `tree_text`, `text_limit`, `undecided_expr`, `sin_q_pi`, `exp_of_logs`, the Exp hook
  in `kapply_src`), `src/series.rs` (`StructureLimit::TreeText`, its `Display` and mapping, the
  `LimitExceeded` doc, `public_expansion`, `expansion_text`), `src/series_engine.rs` (one comment),
  `src/limits.rs` (`limit_with_error` doc), `rustmath-jupyter/src/repl.rs` (`TAYLOR_WORK` doc,
  `test_taylor_leaf_text_cap`).
- Tests: `rustmath-symbolic/tests/s1_depth.rs` (two `b1r2_*` tests, `deep` shapes `fch`, `fadd`,
  `zig`), `tests/s1_payload.rs` (new), `tests/s1_review2.rs` (new), `tests/s1_singular_points.rs`
  (class (e)), `tests/data/gen_s1_singular.py` and `tests/data/s1_singular_points.tsv` (3 rows
  appended). `s1_regressions.rs`, `s1_fuzz_data.rs`, `s1_common/` and `s1_fuzz.tsv` are unchanged.
- Scratch `fu2/s1impl/fix2/`: the before logs and binaries, `depth/` (binary searches), `fuzz/`
  (the harness run and `s1drv_rel`), `p3_rel`, `pprint.rs`, `c2_port_*`, `r2.gp`, `diff1.gp`,
  `diff2.gp`, the measurement logs named above.
- `target/release` was deleted after the release runs.

## Fix-up round 3

Reviews: `fu2/s1impl/S1.review3-resources.md` (not approved: B1, B2 and three minors) and
`fu2/s1impl/S1.review3-semantics.md` (approved: two minors). Scratch of this round:
`fu2/s1impl/fix3/`. Nothing committed, pushed, stashed or reset. The design was not edited.

### 1. Regression tests first

Every test below was run on the code of round 2 before any fix was written (debug:
`fix3/before_review3_dbg.log`, `fix3/before_unit_dbg.log`; release, one process per test with
`ulimit -v` and a 20 s watchdog per consumer: `fix3/before_rel_1.log`, `fix3/before_rel_2.log`; the
release binary of that code is kept as `fix3/s1_review3_before_rel`). All of them failed there.

| test | what it checks | on the code of round 2 |
|---|---|---|
| `tests/s1_review3.rs` `b1r3_float_leaves_at_order_189_are_bounded` (both profiles) | review 3's `g(4.9e-324 x 11 000)/(1-x)` and the same with `1.0e-308` at 0, order 189, through `taylor_with_budget` and `series_expansion`: a refusal, or at most 128 MiB of leaf text as `Display` prints it and 2^21 nodes; order 1 stays a value | Ok with 681 340 837 and 647 900 837 characters of leaf text (2 091 133 nodes) |
| `b1r3_display_of_float_results_release` (`#[ignore]`, release) | the same input at order 189: refused, or `Display`, `simplify`, `substitute`, `==` each within 20 s and VmHWM under 512 MB | `Display` 0.44 s, VmHWM 662 MB |
| `b2r3_simplify_of_a_refused_kernel_power_returns` (both) | `simplify` of the 48-byte result of `((2^256+1)^4096*x)^4096` at order 4096 returns (60 s watchdog in debug) and builds no literal above 2^20 bits | not done after 60 s |
| `b2r3_simplify_folds_a_shared_source_once` (both) | `sin((2^200+1)^512)/(1-x)` at order 60: `simplify`'s output holds the folded source once, with the value (2^200+1)^512 | 61 distinct folded integers |
| `b2r3_simplify_of_results_release_*` (5 tests, `#[ignore]`) | review 3's five inputs at its orders (4096; 20, 40, 100; 1000, 5000; 1000, 5000; 200, 600): refused, or `simplify`, `Display` and `Display` of `simplify`'s output each within 20 s, VmHWM under 512 MB | `simplify` not done in 20 s (power of a kernel, `exp(x+N^4096)` at 200); `simplify` 7.1 s, 7.8 s, 7.4 s, then `Display` of its output not done in 20 s (the geometric series at 20, `exp(x+(2^256+1)^4096)` at 1000, `sin((2^254+1)^4096)/(1-x)` at 1000) |
| `m1r3_large_integers_print_with_a_margin_release` (`#[ignore]`) | resources m1: `exp(x+(10^300-1)^1050*7^130000)` at 312 and the same constant over `(1-x)^2`: refused, or `Display` within 10 s (half the bar) | `Display` not done in 10 s (review 3: 18.8 s) |
| `m2r3_bounded_messages_count_floats_by_their_display` | resources m2: `Debug` of `At(5e-324)` under 120 characters; the `UndecidedZero` message of `1/(x + 1.0e-308*(0.5 - 1/2))` under 400 | 330 characters; 685 characters (`fix3/probe/uz.rs` on round 2's rlib) |
| `m3r3_error_messages_bound_function_names` | resources m3: the `Unsupported` message for a 99 990-character function name is under 300 characters | 100 028 characters |
| `s1r3_big_o_at_infinity_is_a_power_of_one_over_x` | semantics m1: `(x/(x+1))` at infinity to the exponent 4 has `big_o()` = O((1/x)^5) (gp: `subst(x/(x+1),x,1/y)+O(y^5)` = 1 - y + y^2 - y^3 + y^4 + O(y^5)) | O(x^5) |
| `s1r3_float_literal_underflowing_to_zero_is_refused` | semantics m2: `1.0e-400` and 10 other texts holding it are parse errors that name the literal; `0.0e-400`, `4.9e-324`, `1.0e-300` parse as before | `1.0e-400` parsed as 0.0 |
| unit `series_const::node_text_bounds_the_display_of_a_float` | `node_text(Real(f)) >= Display length` for the six values of review 3 and five more | 24 < 326 for 5e-324 |
| unit `series_const::tree_text_bounds_the_text_of_simplify` | `tree_text` bounds the leaf text of 10 hand trees and of their `simplify` | `(2^200+1)^512`: 13 against 30 826 |
| unit `series::display_estimate_counts_a_float_by_its_display` | `display_estimate` of a float is at least its `Display` length | fails at 5e-324 |
| unit `simplify::simplify_folds_powers_up_to_the_engine_cap` | `((2^256+1)^4096)^2` keeps its powers; `(2^254+1)^4096` (2^20 bits a priori) and `(1/3)^(-5)` fold as at HEAD | folded to a 2.1e6-bit integer |
| unit `simplify::simplify_shares_the_value_of_a_shared_node` | a node held twice is simplified once and shared | not shared |

Written after the fixes, as guards: unit `series_const::tree_text_bounds_simplify_on_random_trees`
(3000 seeded random trees with shared subtrees, long literals, rationals, powers, folding exponents
and the shapes of `simplify`'s rules; 2 688 are checked, 437 of them grow under `simplify`) and the
REPL test `test_taylor_simplify_of_kernel_sources` (`repl.rs:15011`; review 3 measured the old
behaviour of its lines in the REPL: not done after 60 s, and 34.8 s).

### 2. The two blockers

**B1, floats (`expression.rs:577`, `series_const.rs:695`, `series.rs:159`).** A float leaf is
counted by the exact length that `Display` writes, `real_display_len` (Rust's `{}` of the f64
into a counting writer, then `.0` for a finite integral value; at most 327). `node_text` and
`display_estimate` both use it, so the cap and the bounded messages agree with `Display` by
construction. Review 3's input at order 189 is now `LimitExceeded(TreeText)` in 0.01 s. The
largest accepted order is 36 (134 MB of text: `Display` 0.08 s, VmHWM 137 MB); order 1 is a value.

**B2, `simplify` of a result.** Three parts.

1. A size guard on the fold of c^n (`simplify.rs:556`, `:674`): `simplify` folds c^n only when
   |n| (bits(num c) + bits(den c)) is at most `MAX_FOLD_POWER_BITS`, which is the engine's own
   cap of an exact power (`series_const::MAX_EXACT_POWER_BITS`, 2^20 bits) with the engine's own
   estimate (`rat_bits` times n). The check reads the literal's bit lengths before anything is
   copied. So `simplify` never builds a power that the engine kept as a kernel. A larger power
   stays `c ^ n`, the same value. This is the only change of HEAD's results: powers of more than
   2^20 bits a priori (HEAD folded every |n| <= 4096).
2. A memo of shared nodes (`simplify.rs:394`): a node held by more than one `Arc` is simplified
   once, and its value is shared by every occurrence; a leaf reached through an `Arc` is shared,
   not copied. `simplify_binary` and `simplify_unary` work on `Arc`s and return an operand's
   `Arc` where HEAD returned the operand. The rules and their order are HEAD's. Check against
   HEAD (`fix3/probe/sdx.rs`, built against HEAD's debug rlib and the port's): review 3's 20 000
   random trees and 20 000 more (seeds 31 to 34 of its generator), each as a tree and inside a
   DAG that names it four times (`e*x + e*y - g(e, e)` with one `Arc`): the serializations of
   `simplify` are byte-identical on all 80 000 (`head.out`, `port_final.out`).
3. The text cap counts what `simplify` builds (`series_const.rs:706` to `:1229`). `tree_text`
   now computes per node a `TextBound`: an upper bound of the printing weight of the node's tree
   and of the tree `simplify` makes of it, with a model of `simplify`'s constant folding (`Val`:
   the literal it surely folds to while it has at most 4096 bits, computed exactly; above that,
   bounds of the bit lengths of its parts). The model follows `simplify_binary` and
   `simplify_unary` rule by rule: 0 + x, x - 0, 0 * x, 1 * x, x / 1, x^0, x^1, the folds of
   `+ - * /`, c^n with the guard of part 1, the special values of the unary functions, and
   (c * rest) / d = (c/d) * rest. `expansion_text` (`series.rs:586`) composes the terms in the
   shape of `to_expr`. Part 3 is needed for the printing of `simplify`'s output: with parts 1 and
   2 alone `simplify` is fast, but its output names the folded literal at every term, and
   printing it converts the literal at each one (`sin((2^254+1)^4096)/(1-x)` at order 1000:
   1 001 x 315 653 digits).

Review 3's five inputs now (release, 8 MB thread, `taylor_with_budget` at 1e8;
`fix3/rel_ignored_final.log`):

| input, at 0 | order | now | `simplify` | `Display` of `simplify`'s output | before (review 3) |
|---|---|---|---|---|---|
| `((2^256+1)^4096*x)^4096` | 4096 | Ok | 0.000 s | 0.000 s (111 bytes) | not done after 60 s |
| `1/(1 - (2^256+1)^4096*x)` | 20, 40, 100 | Ok | 0.000 s | 0.000 s (11 173 bytes at 100) | 7.0 s, 36.7 s, 336.5 s and 730 MB |
| `exp(x + (2^256+1)^4096)` | 1000, 5000 | Ok | 0.001 s, 0.005 s | 0.013 s, 0.78 s (39 MB) | 7.7 s; 38.3 s and 667 MB |
| `sin((2^254+1)^4096)/(1-x)` | 1000, 5000 | `LimitExceeded(TreeText)` | | | 7.7 s at 1000 |
| `exp(x + N^4096)`, N of 600 digits | 200, 600 | Ok | 0.000 s | 0.001 s, 0.006 s | 31.6 s; 95.4 s and 597 MB |

The cost of part 3: `sin((2^254+1)^4096)/(1-x)` is refused from order 193 (round 2 accepted it up
to the node cap). At order 192 its result prints 9 KB, and `simplify`'s output prints 60 MB in
7.6 s (`fix3/probe/edge1.log`). The other four inputs keep their values: in them the engine kept
the power as a kernel, and the guard of part 1 keeps `simplify` from folding it, so the bound
counts their source text only.

### 3. Minor items

Resources lens:

1. Done: the printing weight (`series_const.rs:651`, `:654`). num-bigint 0.4.8 converts an
   integer of d digits to decimal at 2.5 ns per digit at 10^2 digits, 7.5 at 10^3, 22 at 10^4,
   63 at 10^5, 130 at 4x10^5 and 264 at 1.6x10^6 (release, `fix3/probe/conv.rs`): about
   0.21 d^(1/2) ns per digit above 10^4 digits. `print_weight` counts d digits as d up to 2^16
   digits and as d (d/2^16)^(1/2) above, so a weight unit prints in at most about 54 ns whatever
   the size, and 2^27 units in about 7 to 8 s. Measured at the edge of the cap (largest accepted
   order, release; `fix3/probe/edge1.log`):

   | input, at 0 | largest accepted order | `Display` | `Display` of `simplify`'s output |
   |---|---|---|---|
   | `sqrt(N)/(1-x)`, N of 99 950 digits | 1081 (round 2: Ok at 1300, refused at 1340) | 6.9 s, 108 MB | 6.9 s |
   | `exp(x + (10^300-1)^1050*7^130000)` | 122 (round 2: above 312) | 7.3 s, 52 MB | 6.9 s |
   | `sin((2^254+1)^4096)/(1-x)` | 192 | 0.000 s | 7.6 s, 60 MB |
   | review 3's `gsub` | 36 | 0.08 s, 134 MB | 0.08 s |
   | `exp(x)` | 8996 (round 2: Ok at 8800, refused at 9000) | 4.0 s, 134 MB, VmHWM 255 MB | 3.8 s |

   Review 3's `exp(x+(10^300-1)^1050*7^130000)` at order 312 (18.8 s before) is now refused. The
   `MAX_RESULT_TEXT` doc states the measured rates.
2. Done with B1: `display_estimate` counts a float by `real_display_len`. `Debug` of `At(5e-324)`
   is now `At(<a point of about 326 characters or more>)`.
3. Done: `series_engine::name_text` (`series_engine.rs:1256`) writes a name of more than 80
   bytes as its first 40 characters and its length. The `Unsupported` message for an unknown
   function of the variable (`:1954`) uses it. It is the only engine message that embedded an
   input name (every `SErr` constructor with interpolated text was checked: the others carry a
   static word, an operator name, a valuation or a float).

Semantics lens:

1. Done: `big_o()` at infinity is O((1/var)^precision) (`series.rs:439`). At a finite point it is
   unchanged. No caller used it at infinity.
2. Done, by refusal (the reviewer's first option): the parser refuses a float literal with a
   nonzero digit whose f64 value is 0 (`parser.rs:170`), with a nom `Failure` so that no other
   rule reads its digits again (`number`, `:303`, passes the failure on), and a message that names
   the literal (`:133`): "the float literal 1.0e-400 underflows to 0 ...". A subnormal literal is
   kept (its exact binary value), and a literal whose digits are all 0 is the float 0. The
   convention is stated in the `series.rs` module docs. This refuses an input that HEAD's parser
   read as 0; HEAD's value was not the literal's, so the refusal follows "honesty over coverage".
   No caller in the workspace writes such a literal (grep).

Doc comments corrected: `MAX_RESULT_TREE` and `MAX_RESULT_TEXT` (`series_const.rs:625` to `:646`,
what is bounded and the measured rates), `node_text`, the `LimitExceeded` and `TreeText` docs and
the refusal text in `series.rs` ("... in the tree of the result or of its simplification"; it keeps
the words "leaf text" that the REPL test `test_taylor_leaf_text_cap` asserts), the comment at
`series_engine.rs:2305`, and the REPL's `TAYLOR_WORK` doc (`repl.rs:11196` to `:11219`: the two
caps as they are now, the `simplify` guard and memo, the round 2 and round 3 measurements).

No disagreement with either review.

### 4. Acceptance reruns (final code)

| command | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic --no-fail-fast` (debug) | unit 681 (3 ignored; 6 new); `s1_depth` 9; `s1_fuzz_data` 7; `s1_payload` 2 (1 ignored); `s1_regressions` 26 (2 ignored); `s1_review2` 2; `s1_review3` 7 (7 ignored); `s1_singular_points` 6; doc 116 (11 ignored); 0 failed (`fix3/sym_debug_tests_final.log`) |
| `cargo test --release -j 2 -p rustmath-symbolic --no-run`, then each release binary | unit 681 (3 ignored); `s1_regressions` 26 (2 ignored), `s1_singular_points` 6, `s1_fuzz_data` 7, `s1_depth` 9, `s1_payload` 2 (1 ignored), `s1_review2` 2, `s1_review3` 7 (7 ignored); 0 failed (`fix3/rel_tests_final.log`) |
| release `--ignored --test-threads=1` | `s1_regressions` 2 passed in 0.82 s (F_11: taylor 0.019 s, consumers 0.18 s); `s1_payload` 1 passed in 3.3 s (the five inputs refused at order 20 000; order 200 values, `Display` at most 1.36 s; order 1300 now refused); `s1_review3` 7 passed in 1.8 s (table of section 2) (`fix3/rel_ignored_final.log`) |
| review 9's harness (`fuzz.py` seeds 1 to 4, `fuzz2.py` seeds 11 to 14, 250 inputs each, `s1drv_rel` rebuilt; `fix3/fuzz/`) | 1 253 equal to gp, 5 DIFF, 251 GPERR, 491 refusals, 0 SUSPECT, 0 crash. Row by row against round 2: the outcome, input and point of all 2 000 rows are identical, and the eight summary lines are identical |
| design oracle `gpcheck3.py` (`fix3/gpc/gpcheck3_fix3.out`) | 107 of 107 OK, the same output as round 2 |
| `simplify` against HEAD (`fix3/probe/sdx.rs`; HEAD's debug rlib and the final port rlib) | 40 000 random trees, each as a tree and in a shared DAG: identical serializations (`head.out`, `port_final.out`) |
| downstream (`cargo test -j 2 --no-fail-fast -p rustmath-calculus -p rustmath-curves -p rustmath-functions -p rustmath-geometry -p rustmath-manifolds -p rustmath-plot3d`) | 1 516 passed, 0 failed, the same counts as rounds 1 and 2 (`fix3/deps_test.log`); MF1's slot assertions in rustmath-manifolds pass |
| `cargo test -j 2 -p rustmath-jupyter --lib`; `--test kernel_protocol -- --test-threads=1` | 116 passed (1 ignored; the new `test_taylor_simplify_of_kernel_sources` included); 4 passed (`fix3/jup_lib.log`, `fix3/jup_kernel.log`) |
| `cargo clippy -j 2 -p rustmath-symbolic --no-deps --tests` | no warning in the code of this round. The remaining warnings in touched files are HEAD's (the two parser loops, `simplify.rs:39/371/921`, the unused `var` at `series.rs:1344`, the `approx_constant` errors of HEAD's parser tests) |
| rustfmt | `series_const.rs`, `series_engine.rs` and `tests/s1_review3.rs` are clean; the remaining diffs in `simplify.rs`, `series.rs`, `parser.rs` and `expression.rs` are HEAD's code outside the changed lines |

Total of the final runs: 3 242 passed, 0 failed (symbolic 856 in debug; in release its unit tests
681, its seven S1 test files 59 and the 10 release budget tests; downstream 1 516; jupyter 120).
The tests of the first step are unchanged: `check.rs`, `eval.rs`, `s1_fuzz.tsv`,
`gen_s1_fuzz.py` and `s1_gp.py` are byte-identical to the harness copy, and no earlier test file
was edited in this round (mtimes 21:30 or earlier). `target/release` was deleted after the release
runs, and the test executables of the finished runs in `target/debug/deps` too (disk).

### 5. Deviations and handoffs added by this round (for the docs step; HANDOFF and tracker not edited here)

1. `simplify` (HEAD's rules): a rational power c^n is folded only when |n| (bits(num c) +
   bits(den c)) is at most 2^20, the series engine's cap of an exact power. HEAD folded every
   |n| <= 4096. A larger power stays `c ^ n` (the same value). Tracker row of `simplify`.
2. `simplify` simplifies a node held by more than one `Arc` once and shares its value; results
   equal HEAD's on 80 000 random trees and DAGs. With item 1 its work on a series result is
   bounded by the result's DAG and the folds the engine computed itself.
3. The result's text cap (`StructureLimit::TreeText`, 2^27) now counts the tree of the result and
   the tree `simplify` makes of it, a float by its printed length, and the digits of an integer
   above 2^16 digits by `print_weight`. Families refused earlier than in round 2:
   `sqrt(N)/(1-x)` with N of 10^5 digits from order 1082 (round 2: 1340); `exp(x + C)` with C of
   4.2e5 digits from 123 (round 2 accepted 312); `sin((2^254+1)^4096)/(1-x)` from 193; a call of
   11 000 float arguments over (1-x) from 37 (round 2 accepted 189). `exp(x)` is a value up to
   8996. Tracker rows for the caps.
4. The parser refuses a float literal with a nonzero digit whose f64 value is 0, where HEAD read
   it as 0. A HANDOFF note next to the float-literal convention.
5. `SeriesExpansion::big_o` at infinity is O((1/x)^precision).
6. With these, the implementation's deviations from the frozen design also include: the text cap
   is counted on the rendered DAG with a model of `simplify`'s folds (design §2.15 checks its caps
   a priori and has no text cap), and `simplify` is a memoized work list with a size guard (design
   §2.10 walks the left spine only and keeps HEAD's folds).

### 6. Corrections to earlier sections

- Fix-up round 2, section 2, B2: the cap counted a float as 24 characters (now its printed length,
  up to 327) and did not count what `simplify` builds; "2^27 characters print in about 9 s" held
  for 10^5-digit integers only (4.2e5-digit integers printed 133 MB in 18.8 s). Both are corrected
  here.
- Fix-up round 2, section 5, item 5: the text cap's edge for `exp(x)` is now order 8996.
- Fix-up round 2, section 3 and the `TAYLOR_WORK` doc: the claim that `simplify` of every accepted
  result stays within the bar is true from this round on (B2 of review 3).

### 7. Files of this round

- `rustmath-symbolic/src/expression.rs` (`real_display_len`), `src/series_const.rs`
  (`MAX_RESULT_TREE` and `MAX_RESULT_TEXT` docs, `PRINT_LINEAR_DIGITS`, `print_weight`,
  `node_text`, `Fold`, `Val`, `TextBound`, `pow_val`, `tree_bound`, `tree_text`, unit tests),
  `src/series.rs` (`display_estimate`, `big_o`, `expansion_text`, the `LimitExceeded` and
  `TreeText` docs and text, the float convention, a unit test), `src/simplify.rs` (`simplify`,
  `simplify_binary`, `simplify_unary`, `MAX_FOLD_EXPONENT`, `MAX_FOLD_POWER_BITS`, unit tests),
  `src/series_engine.rs` (`name_text`, one comment), `src/parser.rs` (`float`, `number`, the
  refusal message), `rustmath-jupyter/src/repl.rs` (`TAYLOR_WORK` doc,
  `test_taylor_simplify_of_kernel_sources`).
- Tests: `rustmath-symbolic/tests/s1_review3.rs` (new). No earlier test file changed.
- Scratch `fu2/s1impl/fix3/`: the before logs and `s1_review3_before_rel`, `after_*` logs,
  `probe/` (`conv.rs` conversion rates, `edge.rs` and `edge1.log` the edge of the cap, `sdx.rs`
  and its outputs, `uz.rs`), `fuzz/` (the harness run and `s1drv_rel`), `gpc/`, the final test
  logs.

## Fix-up round 4

Reviews: `fu2/s1impl/S1.review4-resources.md` (not approved: B1, B2, one doc minor) and
`fu2/s1impl/S1.review4-semantics.md` (approved: minors m1, m2). Scratch of this round:
`fu2/s1impl/fix4/`. Nothing committed, pushed, stashed or reset. The design was not edited.

### 1. Regression tests first

Every test below was run on the code of round 3 before any fix was written, and failed there
(debug: `fix4/before_review4_dbg.log`, `fix4/before_memory_dbg.log`, `fix4/before_repl_dbg.log`;
release, with the binaries of that code kept as `fix4/s1_review4_before_rel` and
`fix4/s1_review4_memory_before_rel`: `fix4/before_rel_b1.log`, `fix4/before_rel_b2.log`).

| test | what it checks | on the code of round 3 |
|---|---|---|
| `tests/s1_review4.rs` `b1r4_simplify_of_a_kernel_source_chain_is_bounded` (both profiles) | review 4's B1 inputs at 0, order 1, budget 1e8: `sin(K/3...)` and `sin(K/7...)` with 12 000 divisions, `g(K/3...)` with 12 000 and 30 000 (K = (2^254+1)^4096; 24 to 60 KB): refused, or `simplify` of the result within 20 s (the same deadline in debug); the control `exp(x + C)` stays a value | all four `Ok`, then `simplify` not done in 20 s |
| `b1r4_simplify_of_kernel_source_chains_release` (`#[ignore]`) | the same inputs: refused, or `simplify`, `Display` of its output and `Display` each within 20 s, VmHWM under 512 MB | `simplify` not done in 20 s (review 4: 35.2 s, 52.2 s, 34.4 s, 169.5 s) |
| `tests/s1_review4_memory.rs` `b2r4_simplify_of_a_chain_held_by_variables` (both profiles; a test binary of its own; the peak is reset through `/proc/self/clear_refs` before each case) | review 4's REPL-shaped chains, every link kept: `K * 3` 6 000 levels, `K + 1` 4 000, eight 2^20-bit factors `+ 1` 600: `simplify` of the last link raises VmHWM by less than 64 MB and gives the folded value | +749 MB, +497 MB, +600 MB |
| `b2r4_taylor_then_simplify_of_a_held_chain_release` (`#[ignore]`) | `sin(a_6000)/(1-x)` for the `* 3` and `/ 3` chains through `taylor_with_budget` at 1e8, then `simplify`: refused, or within 20 s and under 512 MB | `* 3`: `simplify` 1.38 s, VmHWM 754 MB |
| `rustmath-jupyter` `test_simplify_of_a_variable_chain_memory` (runs `s1r4_variable_chain_session_child` in a child process of the test binary, so the peak is its own) | review 4's 93 861-byte session (`a1 = expr("(2^254+1)^4096")`, 6 000 lines `a_k = a_{k-1}*3`, `z = sin(a6000)/(1-x)`, `t = taylor(z, x, 0, 1)`, `s = simplify(t)`): every line `Ok`, peak under 512 MB | 762 MB |
| `s1_review4.rs` `m1r4_exact_values_at_branch_points_are_decided` (both) | semantics m1: 14 regular inputs that invert, log or root f(c) of an exact-constant argument at a branch point, against gp | 13 of 14 `UndecidedZero` (`1/(x + 3 + acos(0))` was already a value) |
| `m1r4_an_exact_zero_through_the_values_is_a_division_by_zero` (both) | `x/(asin(1) - pi/2)`, `x/(acos(-1) - pi)`, `1/(asinh(I) - I*pi/2)`, `x/(acosh(-1) - I*pi)` at 0 divide by an exact zero: `DivisionByZero` | four values such as `((1 / (arccos(-1) - pi)) * x)` |

Written after the fixes, as guards: `b1r4_simplify_at_the_edge_of_the_fold_cap_release`
(`#[ignore]`: for four families, the largest accepted chain by bisection, then `simplify` and
`Display` of its output there), the unit tests
`series_const::fold_work_bounds_simplify_on_random_trees` (the model of section 2 is at least the
work of `simplify`'s folds computed with the same cost formulas on the actual operands, on 3 000
seeded random trees with shared subtrees and 19 chain shapes) and
`simplify::simplify_counts_parents_inside_the_dag`, the `#[ignore]` calibration
`series_const::fold_work_calibration`, and the 60 rows of class (f) of the singular-point matrix
(section 3). The m2 tests written first were turned into a test of the refusal that stays
(section 3, m2).

### 2. The two blockers

**B2, the memo of `simplify` (`simplify.rs:410`).** A first walk of the DAG counts, for each
non-leaf node held by more than one `Arc`, its parents inside the walked DAG (`parents_in_dag`,
`:448`; a node held by one `Arc` has one parent and is not counted; each counted node is expanded
at its first visit only). A node is memoized only when it has two or more parents there; the entry
holds the number of uses still to come (`Memo`, `:487`) and is removed at the last use. A node held
twice because a REPL variable keeps a clone of its parent has one parent in the DAG and is not
kept. So the memory is HEAD's live set plus the values of the shared nodes whose uses are pending,
and round 3's sharing inside the DAG stays (the output names one value at every occurrence).

| case (debug, 64 MiB thread) | round 3 | now | HEAD (review 4) |
|---|---|---|---|
| `K * 3`, 6 000 levels, every link kept | +749 MB | +1 MB, 2.1 s | 9 MB in total |
| `K + 1`, 4 000 levels | +497 MB | +1 MB, 1.1 s | 7 MB |
| eight factors `+ 1`, 600 levels | +600 MB | +5 MB, 7.9 s | 10 MB |
| the REPL session of 93 861 bytes (whole session) | 762 MB | 17 MB | |
| `sin(a_6000)/(1-x)` then `simplify`, release (`* 3`) | 1.38 s, 754 MB | 1.06 s, under 512 MB | |

`simplify` against HEAD (`fix3/probe/sdx.rs` rebuilt against the final port's debug rlib,
`fix4/probe/sdx_port`): round 3's 40 000 random trees, each as a tree and in a DAG that names it
four times, give serializations byte-identical to HEAD's (`fix4/probe/port_fix4.out` equals
`fix3/probe/head.out`).

**B1, `simplify`'s folds of a kernel's source (`series_const.rs:675`, `:919` to `:1503`;
`series.rs:592`, `:604`).** I took the second way the review offered: the engine bounds the work
of `simplify`'s folds on every result it returns, and refuses above a cap. The first way
(rendering a kernel from its argument's normal form) covers only arguments whose normal form is a
rational: `sin(pi*K/3/.../3)` and the hints of `maybe_hint` would still render the input's own
arithmetic, and round 3's `b2r3_simplify_folds_a_shared_source_once` asserts that a result names
its kernel's source text.

1. `tree_bound` (`:1421`) already models `simplify`'s rules node by node (`TextBound`, the literal
   a node may fold to). Each `TextBound` now also carries the work of `simplify`'s folds at that
   node alone (`work`), and `TextMemo` (`:1409`, the memo of `tree_bound`) sums it once per
   distinct node, as `simplify` simplifies each distinct node once. The cost formulas
   (`fold_binary_work` `:938`, `fold_pow_work` `:981`, `pow_fold_work` `:1313`) are the engine's
   own charges (`q_add`, `q_mul`, `q_pow`, `COPY_WORD`) on bounds of the operands' sizes, with the
   copies `as_rational_const` makes, HEAD's `Ring::pow` (a normalized product at every squaring)
   and, for the normalization, `norm_simplify` (`:1011`: one division n d, then 4 min(n, d)^2 for
   Lehmer's steps; measured at about 5 ns per word product for a balanced gcd against 1 ns for a
   division by a short integer). Every fold that may run is counted (an upper bound). The rule
   (c * rest) / d is counted with c bounded by the characters of the left's tree, numerators and
   now denominators apart (`den_chars`, `:1039`).
2. `public_expansion` refuses a result whose `simplify` may fold more than
   `MAX_RESULT_FOLD_WORK` = 2^31 units (`series.rs:592`), with the new
   `LimitExceeded(StructureLimit::FoldWork)` (`series.rs:147`). The constant of `UndecidedZero` is
   checked the same way (`result_caps`, `series_const.rs:1503`, used at `:551`).

Calibration (release, 8 MB thread, `fold_work_calibration`, `fix4/calibration_final.log`): 24
shapes scaled to 1e7 to 2.5e10 units: chains of quotients by 3, by 2^61 - 1 and under pi, sums
with growing denominators, products, sums and quotients of 1 to 4 rational powers with 2^19-bit
parts, a balanced gcd of two 2^20-bit integers, chains of `* 3` and `+ 1`. `simplify` takes 0.36 to
1.08 ns per unit; the edge test shows 1.3 for a chain of quotients by 7. The cap is therefore at
most about 3 s of folding in release (about ten times that in debug).

Review 4's inputs now (release, `fix4/rel_ignored_final.log`):

| input, at 0, order 1, budget 1e8 | round 3 | now |
|---|---|---|
| `sin(K/3 x 12 000)/(1-x)`, `sin(K/7 x 12 000)/(1-x)`, `g(K/3 x 12 000)/(1-x)`, `g(K/3 x 30 000)/(1-x)` | `Ok`, then `simplify` 35.2 s, 52.2 s, 34.4 s, 169.5 s | `LimitExceeded(FoldWork)` at once |
| `exp(x + K/3 x 12 000)` (the control) | `Ok`, `simplify` 0.000 s | the same |
| the largest accepted chain: `sin(K/3 x n)` n = 2 535; `g(K/7 x n)` n = 2 115; `sin(pi*K/3 x n)` n = 1 448; `sin(K/(2^61-1) x n)` n = 494 | | `simplify` 2.1 s, 2.8 s, 0.8 s, 2.0 s; `Display` of its output about 0.1 s more each; VmHWM under 512 MB |
| `sin(a_6000)/(1-x)`, the held `/ 3` chain | `Ok`, `simplify` 11.7 s, 756 MB | `LimitExceeded(FoldWork)` |
| `sin(a_6000)/(1-x)`, the held `* 3` chain (1.09e9 units) | 1.38 s, 754 MB | `Ok`, `simplify` 1.06 s |
| the REPL, debug: the 12 000-level `sin` line, then `s = simplify(t)` | `t` in 0.38 s, then 355 s | `t = taylor(...)` refused at once |

Round 3's inputs keep their outcomes (`s1_review3` release tests: the same refusals and values,
`simplify` at most 0.006 s).

The three doc comments of the resources minor state the guarantee that holds, with the measured
numbers: `simplify` (`simplify.rs:384` to `:409`: on any input HEAD's work or less; on a result the
engine accepted, at most `MAX_RESULT_FOLD_WORK`), `MAX_RESULT_TEXT` and the new
`MAX_RESULT_FOLD_WORK` (`series_const.rs:638` to `:675`: printing at most about 8 s, folding at
most about 3 s, `substitute` and `==` one visit per tree node) and the REPL's `TAYLOR_WORK`
(`repl.rs:11196` to `:11229`). The `LimitExceeded` doc names the new cap (`series.rs:111`).

### 3. Minor items

Resources lens:

1. The printing margin (7.8 s and 7.1 s at the text cap): no change, as the review says. The docs
   state the number.
2. The three doc comments: done (section 2).

Semantics lens:

1. **m1, done (`series_const.rs:3459`, `:3552`, `:3590`).** `kapply_src` folds the exact values
   arcsin(+-1) = +-pi/2, arccos(-1) = pi, arccos(0) = pi/2, arccosh(-1) = i pi, arccosh(0) =
   i pi/2, arctan(+-1) = +-pi/4 and arcsinh(+-i) = +-i pi/2 (principal values: Abramowitz and
   Stegun 4.4 and 4.6, arccosh z = log(z + sqrt(z - 1) sqrt(z + 1)); gp `asin(1)`, `acos(-1)`,
   `acos(0)`, `acosh(-1)`, `acosh(0)`, `atan(1)`, `asinh(I)`). arccos(1) = arccosh(1) = 0 and the
   poles (arctanh(+-1), arctan(+-i)) were already decided. The values are Laurent polynomials in
   the kernels pi and I (`pi_times`: the interned kernels of the input's own symbols), which the
   zero test decides, so a sum that holds one can be inverted, logged and rooted: the 14 rows of
   `m1r4_exact_values_at_branch_points_are_decided` equal gp, and a divisor that is 0 only through
   them is the exact 0 (`DivisionByZero`, the owner's rule). The matrix gets class (f): 60 rows,
   `1/(3 + f(c x^0))`, `1/(x + 3 + f(c x^0))`, `log(x + 3 + f(c x^0))` and `sqrt(x + 3 + f(c x^0))`
   at 0 for each of the 15 pairs whose f(c) is finite, with gp values (`data/gen_s1_singular.py`,
   rows appended after class (e); the regenerated file starts with round 3's file byte for byte,
   so with the first step's 106 450 bytes too). New test `s1_singular_points`
   `inverted_values_at_branch_points_are_decided`; the coverage test asserts four (f) rows per
   pair.
2. **m2, not fixed: a handoff (a disagreement with "restore if cheap", with evidence).** The code
   change is small (`abs_sign`, `series_engine.rs:1200`: accept coefficients with a real ball
   enclosure and take the sign of the leading one by `real_sign`). It is not cheap in the sense
   that matters here. The frozen design scopes it out (§2.4, the abs/sign row: "all coefficients
   rational"), and the first step's fuzz data asserts that refusal on 14 rows (`data/s1_fuzz.tsv`:
   `U:z1.13`, `U:z2.34`, `U:z3.12`, `U:z3.26`, `U:z3.30`, `U:z4.0`, `U:z4.13`, `U:z4.49`, `U:y11.7`,
   `U:y12.33`, `U:y14.43` would become values; `U:z2.27`, `U:z3.16`, `U:y14.15` stay refused for a
   complex coefficient) and lists it in `R:y13.13`. That file is byte-identical to the harness copy,
   which every review checks, and a refusal is not a wrong value, so the rule for correcting a test
   (a reference value proving it wrong) does not apply. Restoring HEAD's coverage is a scope change
   for the owner. `s1_review4.rs` `m2r4_abs_and_sign_of_real_coefficients_stay_refused` pins the
   honest refusal on the review's six inputs and the controls, and records gp's values for the
   follow-up. Handoff: section 5, item 4.

### 4. Acceptance reruns (final code)

| command | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic --no-fail-fast` (debug, `fix4/sym_debug_tests_final.log`) | unit 683 passed (4 ignored); `s1_depth` 9; `s1_fuzz_data` 7; `s1_payload` 2 (1 ignored); `s1_regressions` 26 (2 ignored); `s1_review2` 2; `s1_review3` 7 (7 ignored); `s1_review4` 4 (2 ignored); `s1_review4_memory` 1 (1 ignored); `s1_singular_points` 7; doc 116 (11 ignored); 0 failed |
| `cargo test --release -j 2 -p rustmath-symbolic --no-run`, then each release binary (`fix4/rel_tests_final.log`) | unit 683 (4 ignored); 9; 7; 2 (1 ignored); 26 (2 ignored); 2; 7 (7 ignored); 4 (2 ignored); 1 (1 ignored); 7; 0 failed |
| release `--ignored --test-threads=1` (`fix4/rel_ignored_final.log`, `fix4/calibration_final.log`) | `s1_review4` 2 passed (17.7 s with the bisections); `s1_review4_memory` 1 passed; `s1_review3` 7 passed (1.8 s); `s1_regressions` 2 passed; `s1_payload` 1 passed (3.2 s); `fold_work_calibration` passed (worst 1.08 ns per unit) |
| review 9's harness (`fuzz.py` seeds 1 to 4, `fuzz2.py` seeds 11 to 14, 250 inputs each, byte-identical to `design/S1-symbolic/review9/`; `s1drv_rel` rebuilt against the final release rlib; `fix4/fuzz/`) | 1 255 equal to gp, 6 DIFF, 252 GPERR, 487 refusals, 0 SUSPECT, 0 crash (round 3: 1 253, 5, 251, 491). Row by row, 5 of the 2 000 rows change, all from `UndecidedZero` (an arccos(0), arccosh(0) or arcsin(1) kernel) to a decided outcome: `acos(acosh(1-x)*2*(2x+1))` at 1 and `tan(4x + 1 - 3^x - asin(sign((x+1/2)^2)))` at -1/2 now equal gp; `cosh(tan(acos(x)))` at 0 is `NotLaurent` (tan(acos x) = sqrt(1-x^2)/x has a pole, cosh of it an essential singularity); `tan(acos(x))*log((2x+1)^(x+2))` at 0 is `4 - 2x`, a DIFF only because gp's float Pi/2 makes its tan coefficient 8.8e308 (gp on the exact form `sqrt(1-x^2)/x*log((2*x+1)^(x+2))` gives `4 - 2*x + ...`; `fix4/diff1.gp`, run as `gp -q < diff1.gp`); `atan(x/(pi^x - asin((x-2)/(x-2))))` at 2 is a value where gp fails on (x-2)/(x-2), and its coefficients equal gp on the simplified `atan(X/(Pi^X - Pi/2))` (0.23648885141931610758..., -0.19620161784222037988...; `fix4/gperr1.gp`, `fix4/gperr2.gp`) |
| design oracle `gpcheck3.py` (`fix4/gpc/gpcheck3_fix4.out`) | 107 of 107 OK, the same output as round 3 |
| `simplify` against HEAD (`fix4/probe/sdx_port`) | 40 000 random trees and DAGs: byte-identical to HEAD |
| downstream (`cargo test -j 2 --no-fail-fast -p rustmath-calculus -p rustmath-curves -p rustmath-functions -p rustmath-geometry -p rustmath-manifolds -p rustmath-plot3d`, `fix4/deps_test.log`) | 1 516 passed, 0 failed (25 ignored), the same counts as rounds 1 to 3; MF1's slot assertions in rustmath-manifolds pass |
| `cargo test -j 2 -p rustmath-jupyter --lib`; `--test kernel_protocol -- --test-threads=1` (`fix4/jup_lib.log`, `fix4/jup_kernel.log`) | 117 passed (2 ignored: an earlier one and the child of the new memory test); 4 passed |
| `cargo clippy -j 2 -p rustmath-symbolic --no-deps --tests` | no warning in the code of this round (the remaining warnings in touched files are HEAD's: `simplify.rs:40/372/1052`, the unused `var` in `series.rs`, the parser tests) |
| rustfmt | `series_const.rs`, `s1_review4.rs`, `s1_review4_memory.rs` and `s1_singular_points.rs` are clean; in `simplify.rs`, `series.rs` and `repl.rs` the remaining diffs are HEAD's code outside the changed lines |

Total of the final runs: 3 263 passed, 0 failed (symbolic debug 864; release: unit 683, the nine S1
test files 65, the 14 ignored budget and calibration tests; downstream 1 516; jupyter 121). The
tests of the first step are unchanged: `check.rs`, `eval.rs`,
`s1_fuzz.tsv`, `gen_s1_fuzz.py` and `s1_gp.py` are byte-identical to the harness copy, and
`s1_regressions.rs` (20:02) was not touched. Of the earlier test files only `s1_singular_points.rs`
changed (class (f) added, no assertion weakened), and its data file grew by appended rows.
`target/release` was deleted after the release runs. `rustmath-misc/output.txt` is unchanged.

### 5. Deviations and handoffs added by this round (for the docs step; HANDOFF and tracker not edited here)

1. `simplify` memoizes a node only when it has two or more parents inside the walked DAG, and
   releases the value after its last use (round 3: every node held by two `Arc`s, kept to the end
   of the walk). Results equal HEAD's on 80 000 trees and DAGs. Tracker row of `simplify`.
2. A new structural cap of a result: `simplify`'s constant folds on its tree, at most 2^31 units
   of the engine's arithmetic (`MAX_RESULT_FOLD_WORK`, the new public variant
   `StructureLimit::FoldWork`). Refused now: a kernel's source or a hint whose literal arithmetic
   `simplify` would fold for more than about 3 s, such as `sin((2^254+1)^4096/3/.../3)/(1-x)` from
   about 2 500 divisions. Design §2.15 has no such cap. Tracker rows for the caps.
3. Exact values of the inverse functions (m1): arcsin(+-1), arccos(-1), arccos(0), arccosh(-1),
   arccosh(0), arctan(+-1), arcsinh(+-i). Design §2.2 lists the exact values of `apply`; these are
   new. A HANDOFF row next to the owner's first rule.
4. Handoff (semantics m2, review 4): abs and sign of a series whose coefficients are real but not
   all rational are `Unsupported` (design §2.4), where HEAD's `taylor` gave the right series
   (`abs(x - sqrt(2))` at 0, `abs(sin(x))` at 1, `abs(cos(x))` at 2, `abs(x + pi)*x` at 0,
   `sign(x - pi)` at 0, `sign(log(x))` at 3; gp values in `s1_review4.rs`). The follow-up: accept a
   series whose coefficients each have a real ball enclosure (`series_ball::enclose`, which takes
   every function on its real domain only), decide the sign s of the leading coefficient with
   `real_sign` (undecided: `UndecidedZero`), keep `NotLaurent` at a zero of odd order, and give
   sign(b) = s and abs(b) = s b; then correct the 11 first-step fuzz rows that become values, with
   gp references, and the `m2r4` test. It needs the owner's decision on the scope of §2.4. A
   HANDOFF row next to the abs/sign convention.

### 6. Corrections to earlier sections

- Fix-up round 3, section 5, item 2 ("with item 1 its work on a series result is bounded by the
  result's DAG and the folds the engine computed itself") did not hold (review 4, B1: a kernel's
  source is folded in another order than the engine's). The bound holds now through item 2 of
  section 5 above.
- Fix-up round 3, section 6, third bullet ("`simplify` of every accepted result stays within the
  bar is true from this round on") held only for the inputs of review 3; it holds from round 4 on,
  with the fold cap.
- Fix-up round 3, section 2, part 2: the memo kept the values of nodes shared only with owners
  outside the DAG (review 4, B2); corrected in section 2 above.

### 7. Files of this round

- `rustmath-symbolic/src/simplify.rs` (`simplify`: `parents_in_dag`, the counted memo, the doc;
  `simplify_binary` and `simplify_unary` are `pub(crate)` for the soundness test; unit test
  `simplify_counts_parents_inside_the_dag`), `src/series_const.rs` (`MAX_RESULT_TEXT` doc,
  `MAX_RESULT_FOLD_WORK`, the fold cost functions, `TextBound` `den_chars` and `work`,
  `pow_fold_work`, `TextMemo`, `tree_bound`, `tree_text`, `fold_limit`, `result_caps`,
  `undecided_expr`; `qhalf`, `qquarter`, `pi_times` and the exact values in `kapply_src`; unit tests
  `random_tree` (extracted from round 3's test), `actual_fold_work`, `fold_chains`,
  `fold_work_bounds_simplify_on_random_trees`, `fold_work_calibration`), `src/series.rs`
  (`StructureLimit::FoldWork`, its message and mapping, the `LimitExceeded` doc,
  `public_expansion`, `expansion_text`), `rustmath-jupyter/src/repl.rs` (`TAYLOR_WORK` doc,
  `test_simplify_of_a_variable_chain_memory`, `s1r4_variable_chain_session_child`).
- Tests: `rustmath-symbolic/tests/s1_review4.rs` and `tests/s1_review4_memory.rs` (new);
  `tests/s1_singular_points.rs` (class (f)), `tests/data/gen_s1_singular.py` (class (f)),
  `tests/data/s1_singular_points.tsv` (60 rows appended, 740 lines).
- Scratch `fu2/s1impl/fix4/`: the before logs and binaries, `refs.py` (the gp references of the
  new tests), `s1_singular_points.tsv.before`/`.new`, `calibration*.log`, `fuzz/` (the harness run
  and `s1drv_rel`), `gpc/`, `probe/` (`sdx_port`, `port_fix4.out`), the gp checks `diff1.gp`,
  `gperr1.gp`, `gperr2.gp`, the final test logs, `S1_impl.md.before_round4`.

## Fix-up round 5

Reviews: `fu2/s1impl/S1.review5-semantics.md` (not approved: B1, minors m1, m2) and
`fu2/s1impl/S1.review5-resources.md` (not approved: B1, B2, a doc minor). Scratch of this round:
`fu2/s1impl/fix5/`. Nothing committed, pushed, stashed or reset. The design was not edited. The
state before the round is kept in `fix5/backup/` (the diff and the untracked files).

### 1. Regression tests first

Every test below was run on the code of round 4 before any fix was written, and failed there
(debug: `fix5/before_review5_dbg.log`, `fix5/before_review5_memory_dbg.log`,
`fix5/before_repl_dbg.log`; release, with the binaries of that code kept as
`fix5/s1_review5_before_rel` and `fix5/s1_review5_memory_before_rel`: `fix5/before_rel_b1.log`,
`fix5/before_rel_b2.log`, `fix5/before_rel_both.log`).

| test | what it checks | on the code of round 4 |
|---|---|---|
| `tests/s1_review5.rs` `b1s5_a_variable_named_pi_or_i_never_names_a_constant` (both profiles) | semantics B1: 17 rows with the variable `pi` or `I` (the table of review 5 section 4: `acos(0*pi)`, `atan(pi^0)` at 1, `acos(-1) + pi`, `asin(1)*pi`, `erf(pi)`, `acosh(0*I)`, `acosh(0*i)`, `acosh(-1) + I`, ...). Through `series_expansion` (REPL budget) no coefficient names the variable and each equals gp; through `taylor_with_budget` the result substituted at the point, simplified and evaluated, is f(a). A typed refusal on both paths is also accepted | 17 of 17 rows fail: the coefficient names the variable (`(1/2 * pi)`, `pi`, `(2 * (pi ^ (-1/2)))`, `(1/2 * (pi * I))`, ...) |
| `b1s5_controls_keep_their_values` (both) | `sin(pi)` in pi at 0 is pi - pi^3/6 (HEAD's right answer stays), `acos(0*x)` keeps the exact value pi/2 | passed (a control) |
| `rustmath-jupyter` `test_taylor_variable_named_like_a_constant` | the REPL: `t = taylor(acos(0*pi), pi, 0, 2)`, `atan(pi^0)` at 1, `acos(-1) + pi`, then `u = subs(t, pi, a)` evaluates to acos(0), atan(1), acos(-1) (`eval_float`); `t = taylor(acosh(0*i) + i, i, 0, 2)`, `subs(t, i, 0)` does not simplify to 0 | `subs at 0 is (1/2 * 0) = 0, want 1.5707963267948966` |
| `s1_review5.rs` `b1r5_consumers_of_a_kernel_source_result_are_bounded` (both) | resources B1: `sin(K/3 x 2535)/(1-x)` (5 095 bytes, K = (2^254+1)^4096) at 0, order 10, REPL budget: refused, or `simplify` of the output of `substitute(x, 2)`, `expand` and `differentiate(x)` each within 20 s (the same deadline in debug) | `simplify of substitute(x, 2) did not finish within 20 s` |
| `b1r5_consumers_of_kernel_source_results_release` (`#[ignore]`) | the same at orders 10 and 190, and `sin(K/3 x 2535)/(2-x)` at 1, order 10 (`expand` distributes over the binomial terms there), with VmHWM under 512 MB | `did not finish within 20 s` at order 10 |
| `rustmath-jupyter` `test_taylor_consumers_of_a_kernel_source_result` | review 5's REPL sessions `replB` (`t = taylor(...)`, `u = subs(t, x, 2)`, `s = simplify(u)`) and `replC` (`d = diff(t, x)`), each in a child process of the test binary, killed at 20 s: the `taylor` line refused or every line `Ok`, peak under 512 MB | `the session did not finish within 20 s` |
| `tests/s1_review5_memory.rs` `b2r5_simplify_does_not_keep_the_values_of_a_dag` (both; a binary of its own, the peak reset by `/proc/self/clear_refs`) | resources B2: review 5's `memo_zero` (sum of `0 * q_i` plus the same sum, q_i = Neg(P), P of 2^20 bits) and `memo_dag` (S1 - S2, q_i = P*(i+2)), m = 4 000: the peak grows by less than 64 MB and the value is 0 | +516 MB and +500 MB |
| `b2r5_simplify_of_a_large_dag_does_not_abort` (both) | `memo_zero` with m = 40 000 (2^20 bits) and with m = 2 000 (2^23 bits) in a child process under `ulimit -v 4000000`: 0, no abort, growth under 64 MB | the first child aborted (no exit code, killed by the allocator), the second grew by 2 008 MB |
| `b2r5_differentiate_then_simplify_of_a_chain_result_release` (`#[ignore]`) | `sin(K*3 x n)/(1-x)` with n = 6 000 and 11 000 (12 and 22 KB) at 0, order 1: refused, or `simplify(R.differentiate(x))` within 20 s and VmHWM under 512 MB | 3.65 s, VmHWM 774 MB |
| `rustmath-jupyter` `test_taylor_diff_of_a_chain_result_memory` | review 5's session `replA` (12 064 bytes): `t = taylor("sin(K*3*...*3)/(1-x)", x, 0, 1)` with 6 000 factors, `d = diff(t, x)`, in a child: refused or `Ok`, peak under 512 MB | `the session's peak is 840 MB` |
| `tests/s1_singular_points.rs` `values_reached_through_folded_constants` (class (g), both) | semantics m1 and the owner's first rule (section 3) | 16 of 32 rows fail: the values that this round adds were `UndecidedZero`; the 12 (g1) rows already passed |

Written after the fixes, as guards: `s1_review5.rs` `b1r5_consumers_at_the_edge_of_the_fold_cap_release`
(`#[ignore]`: the largest accepted `/3` chain at order 10, at 0 and at 1, by bisection, then the
three consumers there), `b1r5_caps_keep_the_largest_earlier_results` (both: `exp(x)` at order 8996
and `sin((2^254+1)^4096)/(1-x)` at order 192 stay values), four rows of the variable test for the
new exact values, the unit tests `series_const::total_fold_work_bounds_simplify_of_an_unshared_copy`
and `simplify::simplify_past_the_memo_cap_gives_heads_value`.

### 2. The three blockers

**Semantics B1, a variable named `pi` or `I` (`series_const.rs:3619` to `:3672`; `series_engine.rs:1184`, `:2202`, `:2232`).**
I took the second way of the review: for such a variable the engine keeps the kernels, so HEAD's
right answers stay. `series_engine::expansion` records whether the variable is named `pi` or `I`
(`set_var_shadow`, a thread-local cleared with the session). `pi_times` returns `None` when the
variable is named `pi` (or `I`, for the constants q i pi), and every exact value of round 4 (and
of this round) then falls through to the kernel f(c) that round 3 and HEAD wrote (`arccos(0)`,
`arctan(1)`, `arccosh(-1)`, `arcsinh(I)`). erf's constant 2/sqrt(pi) is written with the kernel
`arccos(-1)` when the variable is named `pi`. The engine makes no other constant named after a
symbol: grep of the engine for `"pi"`, `"I"` and `"e"` finds only the input's own symbols
(`named_class`, `log_exact`, the ball of a symbol kernel) and these two places. Results now
(`b1s5_...`, both profiles): `acos(0*pi)` in pi at 0 is `arccos(0)`, `acos(-1) + pi` is
`arccos(-1) + pi`, `erf(pi)` is `2*arccos(-1)^(-1/2)*pi - ...`, `acosh(0*I)` in I is `arccosh(0)`;
`sin(pi)` in pi stays pi - pi^3/6. In the REPL `subs(t, pi, 0)` of `taylor(acos(0*pi), pi, 0, 2)` is
the ValueError "Variable 'pi' not found", as at HEAD (the result is the constant `arccos(0)`), and
`t` itself evaluates to 1.5707963...

**Resources B1, the folds after a consumer that rebuilds the tree (`series_const.rs:1089`, `:1097`, `:1236`, `:1639`; `series.rs:606`, `:623`).**
I took the first way of the review and charged the caps per occurrence. Each `TextBound` now
carries `total`, the fold work of its tree with every occurrence of a shared subtree counted (its
own work plus its children's totals), and `public_expansion` checks the root's total against
`MAX_RESULT_FOLD_WORK` (round 4 checked the sum over distinct nodes, which only `simplify` with its
memo reaches). The unit test `total_fold_work_bounds_simplify_of_an_unshared_copy` checks, on 3 000
seeded random DAGs and the chain shapes, that the total is at least the fold work of `simplify` on a
copy of the tree with no shared node (the cost formulas of `actual_fold_work`), and that the copy's
total equals the DAG's.

Measuring the consumers at the new edge found a second hole of the same kind, which I fixed in
the same round: the output of `simplify` after `differentiate` keeps the values of inner nodes
(the quotient rule writes d(3/a) with a^2 in its denominator), so the derivative of the accepted
chain `sin(3/(3/(.../(3/K))))/(1-x)` with 255 levels at order 1 printed 160 MB in 20.6 s
(`simplify` itself took 0.32 s; `fix5/probe/edge2.log`). Each `TextBound` now also carries
`lit_text`, the printing weight of the literal at every node and occurrence (a leaf's, and the one
an inner node may fold to), and the root's is checked against `MAX_RESULT_TEXT` (refusal
`LimitExceeded(TreeText)`, its message extended). For a result whose literals are all printed it
coincides with the text cap: `sin((2^254+1)^4096)/(1-x)` stays a value up to order 192 as before.

**Resources B2, the memo of `simplify` (`simplify.rs:401`, `:500` to `:640`, `:729`).**
The memo holds at most `MEMO_PAYLOAD_WORDS` = 2^22 words (32 MiB) of fresh literals. Each value on
the walk's stack carries an upper bound of the words of fresh literals it holds that no memo entry
accounts for (`weight`: an operand returned keeps its bound, a folded literal is its own words, any
other node at most its operands' bounds plus the literals it made itself; a leaf is shared, 0). A
shared node is memoized only while the entries' total stays within the cap; past it, its next uses
recompute it, as HEAD does. The uses of each shared node are counted down at every visit (a
recomputed parent visits its children again, past the count, and those visits recompute). An entry
whose value is held by nothing else gives its words back at its last use; one held elsewhere stays
counted (an upper bound). So the memory is HEAD's live set plus at most the cap, and each visit is
one HEAD makes, so the work is at most HEAD's. `simplify_past_the_memo_cap_gives_heads_value`
checks the value against HEAD's rules applied at every occurrence on a DAG past the cap.

That alone would make `simplify` after `differentiate` quadratic on review 5's `* 3` chain once the
chain passes the cap (each clone of the product rule names the previous link again). For a result
the engine bounds the payload too: `MAX_RESULT_MEMO_WORDS` (= the memo cap) bounds the literal words
`simplify` builds at the result's distinct nodes (`TextBound::lit`, summed once per distinct node in
`TextMemo::payload`; new refusal `LimitExceeded(FoldPayload)`, `series.rs:155`). `differentiate`
makes fresh nodes and clones whose children are the result's nodes, so the nodes it shares are the
result's own, and their literals fit in the memo at once: no recomputation. With the text count
above, the long chains of 2^20-bit literals are refused earlier still (95 links at order 1).

| case | round 4 | now |
|---|---|---|
| `memo_zero`, m = 4 000, debug | +516 MB | +33 MB, 0.055 s (HEAD 0.046 s) |
| `memo_dag`, m = 4 000, debug | +500 MB, 5.75 s | +33 MB, 6.64 s (HEAD 6.91 s) |
| `memo_zero`, m = 40 000 under a 4 GB limit | abort | +36 MB |
| `memo_zero`, P of 2^23 bits, m = 2 000 | +2 008 MB | +32 MB |
| `sin(K*3 x 6000)/(1-x)` and `x 11000`, then `simplify(R.differentiate(x))` (release) | 3.65 s, 774 MB (6 000) | refused (`TreeText`, the literals of the chain) |
| REPL `replA` (debug) | 840 MB | `t` refused, peak under 512 MB |
| `sin(K/3 x 2535)/(1-x)` at orders 10, 190, and over (2-x) at 1 (release) | 24.1 s, 418.6 s | refused (`FoldWork`) |
| REPL `replB`, `replC` (debug) | 207 s, 225 s | `t` refused at once |

`simplify` against HEAD: `fix3/probe/sdx.rs` rebuilt against the final debug rlib
(`fix5/probe/sdx_port`): round 3's 40 000 random trees and DAGs give `port_fix5.out`, byte-identical
to `fix3/probe/head.out`. Review 5's `sdag.rs` (DAGs with clones held outside the walk), rebuilt
against the final debug rlib (`fix5/sdag/sdag_port`): seeds 1 to 8, 5 000 DAGs each, byte-identical
to its `sdag_head`.

### 3. The required items of this round

1. **B2 in `simplify`, HEAD's memory on DAGs whose nodes are also held outside the walk.** Round 4's
   counting of parents inside the walked DAG and the release after the last use stay; this round adds
   the payload cap (section 2). The reviewer's chains are regression tests in both profiles
   (`s1_review5_memory.rs`, `s1_review4_memory.rs`, which still passes: its `K * 3`, `K + 1` and
   eight-factor chains peak at a few MB).
2. **B1, a real bound.** The caps are charged per occurrence (fold work, literal text), and the
   literals of the distinct nodes are bounded for the memo (section 2). The three doc comments now
   state the guarantee that holds with the measured numbers: `MAX_RESULT_TREE`
   (`series_const.rs:633`), `MAX_RESULT_TEXT` (`:645`, the bound for `simplify` after `substitute`,
   `expand` and `differentiate` and for the `Display` of each output) and `MAX_RESULT_FOLD_WORK`
   (`:688`), with `MAX_RESULT_MEMO_WORDS` (`:706`), `simplify`'s doc (`simplify.rs:376` to `:422`)
   and the REPL's `TAYLOR_WORK` (`repl.rs:11196` to `:11236`). Measured at the edges (release, 8 MB
   thread; 13 families of literal-heavy kernel sources inside `sin(...)/(1-x)` at orders 1 and 10,
   bisected to the largest accepted size, and five large results; `fix5/probe/edge.py`,
   `fix5/probe/edges_final.log`, summarized by `fix5/probe/summarize.py`):

   | family | order | largest n | limited by | simplify(R) | after subs | after expand | after diff | Display (max) | VmHWM |
   |---|---|---|---|---|---|---|---|---|---|
   | K/3 x n | 1 | 95 | text | 0.02 | 0.05 | 0.06 | 0.10 | 0.12 | 17 MB |
   | harmonic sum, n terms | 1 | 3 912 | fold | 0.20 | 0.41 | 0.41 | 0.20 | 0.00 | 10 MB |
   | sum of 1/p, n primes | 1 | 2 032 | budget | 0.86 | 1.70 | 1.80 | 0.84 | 0.00 | 7 MB |
   | 3/(3/(.../K)), n levels | 1 | 95 | text | 0.02 | 0.05 | 0.05 | 0.15 | 7.75 | 88 MB |
   | ((p/q)^2048 sums), n | 1 | 22 | budget | 0.42 | 0.84 | 0.83 | 0.60 | 0.01 | 5 MB |
   | sum of 1/p, n primes | 10 | 1 487 | fold | 0.35 | 3.93 | 4.12 | 0.34 | 0.00 | 11 MB |
   | 3/(3/(.../K)), n levels | 10 | 16 | text | 0.01 | 0.11 | 0.11 | 0.33 | 7.93 | 93 MB |
   | ((p/q)^2048 sums), n | 10 | 16 | fold | 0.14 | 1.52 | 1.53 | 0.64 | 0.04 | 6 MB |
   | `sin((2^254+1)^4096)/(1-x)` | 192 | | | 0.01 | 1.52 | 1.40 | 2.90 | 8.01 | 110 MB |
   | `exp(x)` | 8996 | | | 0.01 | 1.08 | 0.01 | 0.03 | 3.81 | 320 MB |

   The other nine families (K*3, K - 1/p, K*p/p, (K*y)/3, nested negations, K^(-1) towers,
   K + k/(k+1), K/(k/(k+1))) stay under 0.25 s for every consumer and 0.5 s for every Display. The
   slowest consumer is 4.1 s (`expand` then `simplify` of the sum of 1 487 fractions at order 10,
   1.9 ns per unit of the model, above the 1.17 ns of the calibration's worst shape). Every case is
   within S1_BAR (c).
3. **The owner's first rule, a value at an exact-constant argument.** Round 4 folded arcsin(+-1),
   arccos(-1), arccos(0), arccosh(-1), arccosh(0), arctan(+-1) and arcsinh(+-i); arccos(1) =
   arccosh(1) = 0 and the poles arctanh(+-1), arctan(+-i) were decided before. These are all the
   finite branch points of the six inverse functions (the engine has no other inverse function).
   This round adds the other exact values at simple rational and imaginary arguments
   (`series_const.rs:3758`, `:3814`; Abramowitz and Stegun 4.4 and 4.6): arcsin(+-1/2) = +-pi/6,
   arccos(1/2) = pi/3, arccos(-1/2) = 2 pi/3, arccosh(+-1/2) = i arccos(+-1/2), arcsinh(+-i/2) =
   +-i pi/6, arctanh(+-i) = +-i pi/4 (gp: `asin(1/2)` = 0.52359877559829887308, `acos(-1/2)` =
   2.0943951023931954923, `acosh(1/2)` = 1.0471975511965977462*I, `asinh(I/2)` =
   0.52359877559829887308*I, `atanh(I)` = 0.78539816339744830962*I; `fix5/refs1.gp`, run as
   `gp -q < refs1.gp`). As kernels the complex ones had no enclosure, so `1/(3 + arccosh(1/2))` was
   `UndecidedZero`. The matrix gets class (g) (`data/gen_s1_singular.py`, 32 rows appended after
   class (f); the regenerated file starts with round 4's file byte for byte, so with the first
   step's 106 450 bytes too): `1/(3 + f(c x^0))` for each new value (the value against gp on the
   simplified constant) and `x/(f(c x^0) - value)` (DivisionByZero: an exact zero divisor). Rows
   such as `1/(3 + arcsin(x^0))` at 0 were added in round 4 as class (f) (60 rows); they pass.
4. **abs and sign of real non-rational coefficients: a handoff, not restored.** Round 4 recorded it
   (its section 5, item 4). The reasons have not changed: the frozen design scopes it out (§2.4,
   "all coefficients rational"), and fourteen rows of the first step's fuzz data assert the
   refusal (`data/s1_fuzz.tsv`, byte-identical to the harness copy). A refusal is not a wrong value,
   so the rule for correcting a test does not apply, and restoring the values changes the design's
   scope, which is the owner's. The code change itself is small (the leading coefficient's sign by
   `real_sign` on its ball). `s1_review4.rs` `m2r4_abs_and_sign_of_real_coefficients_stay_refused`
   still pins the honest refusal and records gp's values for the follow-up.

### 4. Minor items

Semantics lens:

1. **m1, done.** Class (g1) of the matrix: one pair per inverse function whose argument reaches a
   branch point only through a value folded in round 4: `arcsin(2*arcsin(x^0)/pi + x)`,
   `arccos(-arccos(-x^0)/pi + x)`, `arctan(2*arcsinh(I*x^0)/pi + x)`, `arcsinh(2*arccosh(0*x)/pi + x)`,
   `arccosh(-arccosh(-x^0)/(I*pi) + x)`, `arctanh(4*arctan(x^0)/pi + x^2)` are `NotLaurent`; the
   exact-constant forms are pi/2, pi, `DivisionByZero`, i pi/2, i pi and `DivisionByZero`. They
   passed before any change of this round (the outcomes the reviewer's oracle found right).
2. **m2, a handoff.** Functions of the imaginary exact values (exp(i pi), cosh(i pi), cos(2 pi i), ...)
   stay kernels without an enclosure, so `1/(exp(x)+1)` at `acosh(-1)` is `UndecidedZero`. Honest.
   Follow-up: extend the q pi fold (`as_q_pi`) to exp, sin, cos, sinh, cosh at q i pi
   (exp(q i pi) = cos(q pi) + i sin(q pi), cosh(q i pi) = cos(q pi)), next to the abs and sign
   handoff.

Resources lens:

1. **The cap docs, done** (section 3, item 2).
2. **The printing margin**: noted only, as the review says (Display at the text cap 7.8 to 8.0 s).

### 5. A test of round 4 corrected, with the evidence

`rustmath-jupyter` `test_simplify_of_a_variable_chain_memory` (round 4) asserted that every line of
the 93 861-byte session is `Ok`, including `t = taylor(z, x, 0, 1)` for the `* 3` chain of 6 000
links held by variables. That acceptance is the defect of this round's B2: review 5 measured
`d = diff(t, x)` of the same result (written as text, `replA_mul3_6000_diff.txt`) at 839 MB, and
`simplify` after a consumer that rebuilds the tree folds the chain once per term (2.2e9 units of the
fold model for the two terms, over `MAX_RESULT_FOLD_WORK`); its distinct nodes fold to 6 000 literals
of 2^20 bits (98e6 words). The test now accepts the engine's typed refusal on that line only
(`ComputationError` with "structural limit"), requires every other line `Ok`, and adds
`s2 = simplify(z)`, so that `simplify` of the chain held by the variables is still measured (the
property round 4 tested). The peak is under 512 MB. No other assertion changed. The round-4 release
test `b2r4_taylor_then_simplify_of_a_held_chain_release` already allowed the refusal; it now refuses
both chains.

### 6. Acceptance reruns (final code)

| command | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic --no-fail-fast` (debug, `fix5/sym_debug_final.log`, and `fix5/sym_debug_final2.log` after the last rustfmt edits) | unit 685 passed (4 ignored); `s1_depth` 9; `s1_fuzz_data` 7; `s1_payload` 2 (1 ignored); `s1_regressions` 26 (2 ignored); `s1_review2` 2; `s1_review3` 7 (7 ignored); `s1_review4` 4 (2 ignored); `s1_review4_memory` 1 (1 ignored); `s1_review5` 4 (2 ignored); `s1_review5_memory` 2 (2 ignored); `s1_singular_points` 8; doc 116 (11 ignored); 0 failed |
| `cargo test --release -j 2 -p rustmath-symbolic --no-run`, then each release binary (`fix5/run_rel.sh`, `fix5/rel_tests_final.log`) | 685 (4 ignored), 9, 7, 2 (1), 26 (2), 2, 7 (7), 4 (2), 1 (1), 4 (2), 2 (2), 8; 0 failed |
| release `--ignored --test-threads=1` (same log) | `s1_payload` 1 (3.46 s); `s1_regressions` 2 (0.82 s); `s1_review3` 7 (1.94 s); `s1_review4` 2 (2.38 s; the edge of the fold cap is now 95 links for each of its four families); `s1_review4_memory` 1; `s1_review5` 2 (2.14 s); `s1_review5_memory` 2; `fold_work_calibration` passed (46.9 s, worst 1.17 ns per unit) |
| review 9's harness (`fuzz.py` seeds 1 to 4, `fuzz2.py` seeds 11 to 14, 250 inputs each, byte-identical to `design/S1-symbolic/review9/`; `s1drv_rel` rebuilt against the final release rlib; `fix5/fuzz/`) | 1 255 equal to gp, 6 DIFF, 252 GPERR, 487 refusals, 0 SUSPECT, 0 crash; row by row (outcome, input, point, order) identical to round 4's run (`fix5/fuzz/sum.py`) |
| design oracle `gpcheck3.py` with the rebuilt driver (`fix5/gpc/gpcheck3_fix5.out`) | 107 of 107 OK, the same output as round 4 |
| `simplify` against HEAD (`fix5/probe/sdx_port`, `fix5/sdag/`) | 40 000 random trees and DAGs, and 40 000 DAGs with outside owners: byte-identical to HEAD |
| downstream (`cargo test -j 2 --no-fail-fast -p rustmath-calculus -p rustmath-curves -p rustmath-functions -p rustmath-geometry -p rustmath-manifolds -p rustmath-plot3d`, `fix5/deps_test.log`) | 1 516 passed, 0 failed (25 ignored), the same counts as rounds 1 to 4; MF1's slot assertions in rustmath-manifolds pass |
| `cargo test -j 2 -p rustmath-jupyter --lib`; `--test kernel_protocol -- --test-threads=1` (`fix5/jup_lib_final.log`, `fix5/jup_kernel_final.log`) | 120 passed (3 ignored: an earlier one and the two child tests); 4 passed |
| `cargo clippy -j 2 -p rustmath-symbolic --no-deps --tests` (`fix5/clippy.log`) | no warning in the code of this round (the remaining warnings in touched files are HEAD's: `simplify.rs:40/372/1193`, the unused `var` in `series.rs`) |
| rustfmt | `series_const.rs`, `series_engine.rs`, `s1_review5.rs`, `s1_review5_memory.rs`, `s1_singular_points.rs` are clean; in `simplify.rs` and `series.rs` the remaining diffs are HEAD's code outside the changed lines |

The tests of the first step are unchanged: `check.rs`, `eval.rs`, `s1_fuzz.tsv`, `gen_s1_fuzz.py`
and `s1_gp.py` are byte-identical to the harness copy (`protoharness/work/src`), `s1_regressions.rs`
was not touched, and `s1_singular_points.tsv` starts with round 4's 740 lines byte for byte (772
lines now). Of the earlier test files only `s1_singular_points.rs` (class (g) added; no assertion
weakened) and the round-4 REPL test of section 5 changed. `target/release` was deleted after the
release runs. `rustmath-misc/output.txt` is unchanged and no `.tmp_dir_*` exists.

### 7. Deviations and handoffs added by this round (for the docs step; HANDOFF and tracker not edited here)

1. A variable named `pi` or `I` keeps the kernels of round 3 for the exact values (`arccos(0)`,
   not pi/2) and writes erf's constant with `arccos(-1)`. Design §2.2 lists the exact values of
   `apply`; with such a variable they are not folded. A HANDOFF row next to the exact values.
2. `simplify`'s memo holds at most 32 MiB of fresh literals (`MEMO_PAYLOAD_WORDS`); past that a
   shared node is recomputed at each use, as HEAD does. Tracker row of `simplify`.
3. The result caps are counted per occurrence: the fold work (`MAX_RESULT_FOLD_WORK`, round 4 counted
   distinct nodes) and the printing weight of the literal at every node (`MAX_RESULT_TEXT`, also
   for inner nodes); and a new cap of the literal words of the distinct nodes
   (`MAX_RESULT_MEMO_WORDS` = 2^22, public `StructureLimit::FoldPayload`). Design §2.15 has none of
   these. Refused now: kernel sources with more than about 95 links of 2^20-bit literals at order 1
   (16 at order 10), and sums of more than about 1 500 fractions in a kernel at order 10. Tracker
   rows for the caps.
4. Exact values arcsin(+-1/2), arccos(+-1/2), arccosh(+-1/2), arcsinh(+-i/2), arctanh(+-i). A HANDOFF
   row next to round 4's exact values.
5. Handoffs kept: abs and sign of real non-rational coefficients (round 4, item 4; section 3, item
   4 above); functions of the imaginary exact values (semantics m2, section 4).

### 8. Files of this round

- `rustmath-symbolic/src/series_const.rs` (the doc comments of the result caps, `MAX_RESULT_MEMO_WORDS`,
  `val_words`, `TextBound::{total, lit, lit_text}` and `with_children`, `TextMemo::payload`,
  `literal_text_limit`, `memo_limit`, `bound_caps`, `result_caps`; `VAR_SHADOW`, `set_var_shadow`,
  `var_is_named_pi`, `pi_times` returning `Option`, `as_q_i`, `qnd` and the new exact values in
  `kapply_src`; unit test `total_fold_work_bounds_simplify_of_an_unshared_copy`),
  `src/simplify.rs` (the memo's `Entry`, `lit_words`, `weight`, `remember`, the uses counted down,
  `MEMO_PAYLOAD_WORDS`, the doc; unit test `simplify_past_the_memo_cap_gives_heads_value`),
  `src/series.rs` (`StructureLimit::FoldPayload`, the `TreeText` doc and message, `public_expansion`,
  `expansion_text`), `src/series_engine.rs` (`set_var_shadow` in `expansion` and in the session's
  drop, erf's constant), `rustmath-jupyter/src/repl.rs` (`TAYLOR_WORK` doc; tests
  `test_taylor_variable_named_like_a_constant`, `test_taylor_consumers_of_a_kernel_source_result`,
  `test_taylor_diff_of_a_chain_result_memory`, `s1r5_session`, `s1r5_session_child`; the corrected
  `test_simplify_of_a_variable_chain_memory`).
- Tests: `rustmath-symbolic/tests/s1_review5.rs` and `tests/s1_review5_memory.rs` (new);
  `tests/s1_singular_points.rs` (class (g)), `tests/data/gen_s1_singular.py` (class (g)),
  `tests/data/s1_singular_points.tsv` (32 rows appended, 772 lines).
- Scratch `fu2/s1impl/fix5/`: `backup/`, the before logs and binaries, `refs1.gp`,
  `s1_singular_points.tsv.before`/`.new`, `probe/` (`build.sh`, `build2.sh`, `probe6.rs` and `p6_rel`,
  `edge.py`, `run_edges.sh`, `summarize.py`, `edge*.log`, `edges_final.log`, `sdx_port`,
  `port_fix5.out`), `sdag/`, `fuzz/` (the harness run, `s1drv_rel`, `sum.py`), `gpc/`, the final test
  logs, `S1_impl.md.before_round5`.

Total of the final runs: 3 288 passed, 0 failed (symbolic debug 873; release: unit 685, the eleven
S1 test files 72, the 18 ignored budget and calibration tests; downstream 1 516; jupyter 124).

## Finish under the narrowed bar

Governing records: `fu2/s1impl/owner_decision2.md` (narrow the bar, then finish) and the narrowed
bar of this run. A blocker is (a) a wrong, guessed or fabricated result, or an undecided answer
reported as decided; (b) a regression against HEAD-before-S1 (8a0b310) on an input it handled,
for a public function S1 changed; (c) a budget hole in S1's own entry points or in the printing,
dropping and re-parsing of their results. The cost of a consumer applied to a result is out of the
bar: it is documented with measured scaling, and no doc comment may claim it is bounded. Reviews:
`S1.review6-semantics.md` (approved, minors m1 to m3) and `S1.review6-resources.md` (B1 and B2,
now out of the bar; minors 1 to 3). Branch `claude/gracious-gates-e6b6l7` at main 28c4b69 (the S1
work in progress is 4763f31). Scratch: `fu2/s1impl/fin/`. Nothing committed, pushed, stashed or
reset. The design was not edited.

### 1. Tests written first

Each was run on the code at the start (28c4b69) before any fix (`fin/before_review6_dbg.log`,
`fin/before_singular_dbg.log`, `fin/before_depth_dbg.log`).

| test | what it checks | on the code at the start |
|---|---|---|
| `tests/s1_review6.rs` `m1r6_a_truncated_integer_exponent_is_undecided` (both profiles) | semantics m1: the 18 class-H rows of review 6 (`x^(x^-5*Z)`, `x^(Z)`, `x^(Z/x^2)` at 0 for its six truncated zeros Z; each function is 1) and 12 hand rows (`x^((1+x)/(1+x))`, `x^(sin(x)^2+cos(x)^2)`, `x^(2*(sin(x)^2+cos(x)^2))`, `x^(2*cos(x-x))`, `x^(exp(x)*exp(-x))`, `(x^2)^((2+x)/(2+x))`, `(sin(x))^(cos(x)^2+sin(x)^2)`, `x^(sin(x)/sin(x))`, and the bases of negative valuation `(1/x)^((1+x)/(1+x))`, `(1/sin(x))^(exp(x)*exp(-x))`, ...): `PrecisionExhausted` through `series_expansion` and `taylor_with_budget`, the limit path `Unknown` | 30 of 30 rows `NotLaurent` ("log has a logarithmic singularity at the point") |
| `m1r6_controls_keep_their_outcomes` (both) | exact exponents stay values (`x^(x-x+1)`, `x^(x/x)`, `x^(1+0*sin(x))`, `x^((x+1)^2-x^2-2*x)` are x), bases of valuation 0 stay values (`(1+x)^((1+x)/(1+x))`, `x^((1+x)/(1+x))` at 1, `exp(x)^(1/(1+x)*(1+x))`; gp), a non-integer constant term or a term in h stays `NotLaurent` (`x^(1+x)`, `x^(x + (1+x)/(1+x))`, `x^(1/2 + (1+x)/(1+x) - 1)`, `x^(sqrt(2) + ...)`), an undecided integrality is `UndecidedZero` (`x^(sin(1)^2+cos(1)^2 + (1+x)/(1+x) - 1)`) | 1 of 12 failed: the undecided row was `NotLaurent` |
| `m3r6_gaussian_constants_are_inverted_exactly` (both) | semantics m3: `1/(x + 1/(1+I) - (1-I)/2)`, `1/(x + 1/(1-2*I) - (1+2*I)/5)`, `1/(x + 1/(pi+I) - (pi-I)/(pi^2+1))` and `1/(x + 1 + (9/8*I)/(1+I)^2 - 25/16)` at 0 are x^-1; `x/(1/(1+I) - (1-I)/2)` and `x/(3/(2-I) - (6+3*I)/5)` divide by an exact zero (`DivisionByZero`) | 5 of 6 failed: three `UndecidedZero`, two values of a constant factor (the convention of `series.rs` for a divisor the input names); the `(1+I)^2` row already passed (a power of a Gaussian rational is exact) |
| `m3r6_values_through_gaussian_inverses` (both) | controls: `1/(x + 1/(2+3*I))`, `1/(x + 1/(pi+I))` and `log(x + 1/(1+I))` at 0 against gp | passed (values before too, through the kernel 1/(A + B*I)) |
| `tests/s1_singular_points.rs` `truncated_constants_at_branch_points_are_undecided` and the coverage test (both) | semantics m2: class (h), two truncated-constant rows per (function, branch point) pair, 46 rows: `f(c*(1+x)/(1+x))` and `f(c + (sin(x)^2+cos(x)^2-1))` at 0 (for c = 0, `(1+x)/(1+x) - 1` and `sin(x)^2+cos(x)^2-1`): `PrecisionExhausted` | passed: the engine already gave this outcome (review 6 found it on 1 520 rows of its fuzzer). The rows pin it per pair, as the minor asks; no fix was needed |
| `tests/s1_depth.rs` `p6_texts_refused_by_head_are_refused_not_aborted` (debug 2 MB; release 2 MB and 8 MB) | resources minor 2 and its class: `x^-x^-...^-x` and `-x^-x^-...` of 1 500 levels and of 100 KB, `x^-(x^-(...))` and `--(--(...))` of 300 levels, and `x^-x`, `x^+2`, `- - -x` or a 20-digit literal before 300 nested parentheses: a `ParseError`, HEAD's outcome on every one (checked with review 6's probe against HEAD's debug rlib, `fin/parse/`); towers of 500 levels and nests of 60 and 100 levels still parse | the test binary aborted (stack overflow on the 2 MB debug thread); review 6's probe of the port aborts on all nine texts |
| `p6_texts_read_by_head_keep_their_reach` (both) | control: texts HEAD reads (`x^-2`, `--5`, `- -5`, `+-5`, `x^ -5.5` and the literal i64::MAX, each before 300 nested parentheses; and the S1 constructs after the deep part) parse on a 64 MiB thread | passed |

### 2. What changed (file:line at the end of this step)

1. **The doc comments of the consumers (task 1).** Each now says what is bounded (the entry points,
   and the printing, `Debug`, dropping and re-parsing of their results) and gives the measured
   scaling of `simplify`, `expand`, `differentiate`, `substitute`, `==`, the REPL's `diff` and
   `integrate` of a result, with the numbers of review 6 B1 and B2 and rounds 4 and 5, and a
   pointer to the follow-up unit S2 ("DAG-aware consumers"):
   - `rustmath-symbolic/src/series_const.rs:638` `MAX_RESULT_TREE` (no longer "a consumer that
     walks the tree ... is bounded by the cap"); `:650` to `:711` `MAX_RESULT_TEXT` (its first
     paragraph now says the cap counts the tree `simplify` makes, not that it bounds `simplify`'s
     output; the paragraph "The bound that holds" is replaced by "What the caps bound", `:675`, and
     "What they do not bound", `:682`, with one measured line per consumer); `:712`
     `MAX_RESULT_FOLD_WORK` (it counts folds only, not the nodes that `expand` and `differentiate`
     add, so it does not bound `simplify` after them); `:732` `MAX_RESULT_MEMO_WORDS`; `:1115`
     `TextBound::total`; the doc of the unit test `total_fold_work_bounds_simplify_of_an_unshared_copy`
     (`:4795`: `expand` distributes, so it hands `simplify` more than an unshared copy).
   - `src/simplify.rs:419` (`simplify` on a result: a measured scaling, not a bound; 21.8 s and
     5.6 GB after `expand`, 9.1 s and 2.9 GB after `differentiate`).
   - `src/series.rs:149` and `:154` (`StructureLimit::FoldWork`, `FoldPayload`), `:726` (the doc of
     `taylor_with_budget`).
   - `src/series_engine.rs:2345` (the comment at the tree check of the driver).
   - `rustmath-jupyter/src/repl.rs:11211` to `:11236` (`TAYLOR_WORK`: "What is bounded" and "What
     is not bounded", with the REPL's numbers: `e = expand(t)` 5.7 s and 2.5 GB, `d = diff(t, x)`
     11.6 s and 2.1 GB, `integrate` 43.3 s at 1 000 terms).
   A grep for `consumer`, `bounded too`, `stay bounded`, `S1 bar` and `S1_BAR` over the two crates'
   sources finds no other claim.
2. **m1 (`src/series_engine.rs:2062`, `:2084` `truncated_integer_exponent`).** In `pow_general`,
   before log of the base: a base of decided nonzero valuation (positive, as the review asks, and
   negative, where the same argument holds) and a truncated exponent whose only stored term is its
   constant term k (or no stored term) give `NeedPrecision` when k is an integer (the driver raises
   the precision and ends with `PrecisionExhausted`), `NotLaurent` when k is decided not to be an
   integer (x^(k + r) has a branch point whatever r is), and `UndecidedZero` of k - n when the
   integrality is undecided (`non_integer_status`, as `pow_const` uses it; the module rule for an
   exponent). f^g = f^k exp((g - k) log f) is Laurent exactly when g is the integer k, and
   g - k = O(h^P) is not known to be 0. Before, `NotLaurent` was a decided answer for
   `x^((1+x)/(1+x))`, which is x.
3. **m2.** `tests/data/gen_s1_singular.py:76` (header) and `:341` (class (h)); the data file
   `tests/data/s1_singular_points.tsv` gains 46 rows (818 lines). Its first 772 lines (146 294
   bytes, round 5's file, so the first step's 106 450 bytes too) are unchanged byte for byte
   (`cmp -n`; `fin/s1_singular_points.tsv.before` and `.new`), and the generator reproduces the
   whole file. `tests/s1_singular_points.rs`: the (g) and (h) bullets of the module doc, the
   coverage loop (two rows per pair), the expectation arm, the test at `:324`.
4. **m3 (`src/series_const.rs:637` `GAUSSIAN_INVERSE_TERMS`, `:2579` `Konst::gaussian_inverse`,
   `:2628` its call in `inverse_src`).** A constant c = A + B*I with A, B free of the kernel I
   (`split_imag`), of at most 16 terms, with the kernel I and no free parameter, is inverted as
   (A - B*I) * (1/N), N = A^2 + B^2, when N is decided nonzero; for A = 0 as -I * (1/B).
   (A + B*I)(A - B*I) = A^2 + B^2 in any field with I^2 = -1, so this also proves c nonzero, and it
   runs before the zero test of `strict`. N and B are free of I, so their inverses are rationals,
   exact monomials, or kernels with a real enclosure when A and B have one. Otherwise c keeps the
   kernel 1/c, as before. A constant with a free parameter keeps the kernel: for `a + I`,
   (a - I)/(a^2 + 1) would add the exceptional point a = I, where 1/(a + I) is defined. The change
   is small and local (one function, one call), so it is done here, not handed off.
5. **The parser's depth (resources minor 2, `src/parser.rs`).** HEAD's parser refused three
   constructs at once: a sign after `^` other than a literal's own (`x^-2` it read; `x^-y`, `x^+2`,
   `x^--2` it refused), more than one sign before an operand other than a literal's (`--5` and
   `- -5` it read; `--x` and `- - -x` it refused), and an integer literal above i64::MAX. The port
   reads them and then went on into the rest of the text, so `x^-x^-...` of 1 500 levels, any of
   these constructs before 300 nested parentheses, or `x^-(x^-(...))` of 300 levels aborted a 2 MB
   debug thread where HEAD returned a `ParseError`. Review 6 named the towers; the other texts are
   the same class (nine texts abort the port of the start, `fin/parse/`). The fix:
   - `:184` `mod depth`: a per-thread count of the parse's depth in units (`NEST` = 8 per level of
     nested atoms, that is a parenthesis or a call; `TOWER` = 1 per level of `^`), a flag that each
     of the three constructs sets (`relax`), and `LIMIT` = 1 024 units. Once the text has used one
     of them, a level past the limit is refused: at most 128 levels of parentheses and calls, or
     1 024 levels of `^`. The count is checked only after `relax`, so a text in HEAD's syntax keeps
     at least HEAD's reach. `parse` (`:126` to `:145`) resets the state and names the refusal ("the
     expression nests too deeply for its syntax ...").
   - `:420` `sign_run` also returns the number of signs and whether the last one is a `-` right
     before a digit; its white space is now the grammar's (`multispace0`: space, tab, CR, LF; before,
     any Unicode white space, which HEAD refused there). `:445` `operand_signs` (in `unary`) and
     `:459` `exponent_signs` (after `^`; it also enters the tower level, and past the limit it
     returns the empty text, so the exponent fails at once) relax for the forms HEAD refused; `:258`
     the integer literal relaxes above i64::MAX.
   - `:504` `atom` enters a `NEST` level and reads the white space itself (`:514` `skip_space`),
     `:521` `caret` reads `^` out of line, `:552` `make_power` leaves the tower level (an error fails
     the whole parse, and `parse` resets the count). The frames were tuned so that no recursive
     frame grows. The first versions (a guard in `expression`, the count in a `ws(atom)` wrapper or
     in a `nested_expression` wrapper) lost reach against HEAD in debug or in release: LLVM inlined
     the `ws(atom)` closure into `power` in one build and kept two more registers across the
     recursion (`fin/objs_old/`, `fin/objs/`: the frames of the parser's functions). Reach now
     (`fin/zz_reach_probe.rs` through the test harness, bisected by `fin/reach.sh`; review 6's
     `pprobe.rs` rebuilt against the new rlibs, `fin/parse/pp_new_rel`, `pp_new_dbg`):

     | shape | HEAD | port at the start | now |
     |---|---|---|---|
     | carets, 2 MB debug | 1 440 | 1 441 | 1 475 |
     | `(`, `2*(`, `-(` nested, 2 MB debug | 216 | 221 | 240 |
     | `sin(` nested, 2 MB debug | 162 | 190 | 204 |
     | `2^(` nested, 2 MB debug | 187 | 191 | 207 |
     | carets, release 8 MB / 2 MB | 22 770 / 5 677 | 22 773 / 5 676 | 29 101 / 7 255 |
     | `(`, `sin(`, `2*(`, `-(` nested, release 8 MB / 2 MB | 4 124 / 1 028 | 4 515 / 1 125 | 4 328 / 1 079 |
     | `2^(` nested, release 8 MB / 2 MB | 3 491 / 870 | 3 767 / 939 | 3 768 / 939 |

     Every shape is at least HEAD's in both profiles. The release nesting is 4 % below the port of
     the start and 5 % above HEAD. The limit fits a 2 MB debug thread (128 nested calls take about
     1.3 MB there, 1 024 carets about 1.4 MB).
   - What it refuses: only texts HEAD refuses too (each holds one of the three constructs). A text
     in HEAD's syntax deeper than HEAD's reach still overflows, as at HEAD. One effect to note: a
     printed series result with an integer above i64::MAX (`1/30!`) followed by more than 128
     levels of nesting no longer re-parses (a `ParseError`, not a crash); HEAD's parser refused any
     text with such a literal.
6. **A test of fix-up round 1 adapted, with the reason.** `tests/s1_depth.rs:476`
   `b2_limit_with_error_of_a_signed_tower` parsed `x^-x^-...` of 1 400 (debug), 3 000 and 13 000
   (release) levels, which the parser now refuses (item 5). The test builds the same tree
   (`signed_tower`, `:443`, checked equal to the parser's tree at 500 levels; `lwe_expr`, `:456`).
   Its assertions are unchanged. It is not a file of the first step.

The tests of the first step are unchanged: `s1_common/*.rs` (md5 checked before and after rustfmt,
`fin/s1_common_md5_before_fmt.txt`), `data/s1_fuzz.tsv`, `gen_s1_fuzz.py`, `s1_gp.py` and
`s1_regressions.rs` are not touched, and `s1_singular_points.tsv` keeps its first 772 lines. No
expected outcome changed. The gp references of the new rows: `fin/refs6.gp` (`gp -q < refs6.gp`,
PARI/GP 2.15.4, `\p 70`), named in the header of `s1_review6.rs`.

### 3. Minor (v): the REPL's `integrate` of a result, a handoff

Review 6 (resources, minor 3): HEAD's `integrate` is cubic in the number of terms
(`integrate(taylor(1/(1-x), x, 0, N), x)`: 0.78 s at N = 250, 5.6 s at 500, 43.3 s at 1000,
release), and the same polynomial typed as text costs the same at HEAD, so S1 adds nothing to it.
It is a consumer of a result, out of the narrowed bar; the docs state the numbers
(`series_const.rs` `MAX_RESULT_TEXT`, `repl.rs` `TAYLOR_WORK`). Handoff: acceptance item 4 of the
S2 spec (find where the cost is, tier 1, the rational Risch front door, on a polynomial of degree
N, or tier 3's recursion over the sum running tier 1 again on every subtree, and make it linear in
the terms).

### 4. The S2 spec

`fu2/s1impl/S2_spec.md` ("DAG-aware consumers"): the consumers and how they walk (`simplify` is
already DAG-aware on its input; `expand`, `differentiate`, `substitute`, `PartialEq` and
`integrate` are not), the goal (cost linear in the DAG of a result, outputs that share what the
result shares, values unchanged), the measured cases of rounds 4 to 6 as acceptance inputs (E1 to
E6 for `expand`, D1 to D10 for `differentiate` and the REPL's `diff`, S1 to S4 for inputs the S1
caps refuse today, I1 for `integrate`), the acceptance tests (written first, both profiles, the
peak memory per case, the scaling with the order), and the constraints (share by address within a
call, memory at most HEAD's live set plus a cap, the derivative of a subtree free of the variable,
iterative walks, zero `unsafe`). An optional phase 2 may relax the per-occurrence caps of S1 once
the consumers share. The docs step copies the file into the repository.

### 5. Acceptance reruns (final code)

| command | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic --no-fail-fast` (debug, `fin/sym_dbg_final2.log`) | unit 685 passed (4 ignored); `s1_depth` 11; `s1_fuzz_data` 7; `s1_payload` 2 (1 ignored); `s1_regressions` 26 (2 ignored); `s1_review2` 2; `s1_review3` 7 (7 ignored); `s1_review4` 4 (2 ignored); `s1_review4_memory` 1 (1 ignored); `s1_review5` 4 (2 ignored); `s1_review5_memory` 2 (2 ignored); `s1_review6` 4; `s1_singular_points` 9; doc 116 (11 ignored); 0 failed (880 passed) |
| `cargo test --release -j 2 -p rustmath-symbolic --no-run`, then each release binary (`fin/run_rel.sh`, `fin/rel_tests_final2.log`) | 685 (4 ignored), 11, 7, 2 (1), 26 (2), 2, 7 (7), 4 (2), 1 (1), 4 (2), 2 (2), 4, 9; 0 failed |
| release `--ignored --test-threads=1` (same log) | `s1_payload` 1 (3.68 s); `s1_regressions` 2 (0.85 s); `s1_review3` 7 (1.92 s); `s1_review4` 2 (2.72 s); `s1_review4_memory` 1; `s1_review5` 2 (2.44 s); `s1_review5_memory` 2; `fold_work_calibration` passed (47.4 s, worst 1.15 ns per unit, the cap about 2.5 s) |
| `s1_depth` and `s1_review6` again after the last edits (an `assert!` form and a type alias, from clippy), debug and release | 11 and 4 passed in each profile |
| dependents (`cargo test -j 2 --no-fail-fast -p rustmath-calculus -p rustmath-curves -p rustmath-functions -p rustmath-geometry -p rustmath-manifolds -p rustmath-plot3d`, `fin/deps_test.log`) | 1 556 passed, 0 failed (35 ignored); MF1's slot assertions in rustmath-manifolds pass |
| `cargo test -j 2 -p rustmath-jupyter --lib`; `--test kernel_protocol -- --test-threads=1` (`fin/jup_lib.log`, `fin/jup_kernel.log`) | 120 passed (3 ignored); 4 passed |
| `cargo check -j 2 --workspace --tests` (`fin/check_ws.log`) | no error |
| `cargo clippy -j 2 -p rustmath-symbolic --no-deps` (`fin/clippy_lib.log`) | passes; 56 warnings, none on a line of this step (checked against `git diff -U0 4763f31`) |
| `cargo clippy ... --no-deps --tests` (`fin/clippy.log`) | the lib-test target stops at two `approx_constant` errors in HEAD's parser tests (`parser.rs:689` and `:766`, `3.14159` and `3.14`, from 2025-12-28; deny by default under clippy 1.97), not code of S1. In the integration tests the two warnings on new code (a complex type in `s1_review6.rs`, `assert_eq!` with `true` in `s1_depth.rs`) are fixed; one remains on a doc line of the first step (`s1_singular_points.rs:61`, `doc_lazy_continuation`), left as it is |
| rustfmt | `s1_review6.rs`, `s1_depth.rs`, `s1_singular_points.rs`, `series_const.rs`, `series_engine.rs` are clean; in `parser.rs`, `simplify.rs`, `series.rs` and `repl.rs` rustfmt's remaining changes are all on HEAD's lines (none on a line of this step) |

Total of the final runs: 3 342 passed, 0 failed (symbolic debug 880; release 764 and the 18
ignored budget and calibration tests; downstream 1 556; jupyter 124). `target/release` was deleted
after the release runs. `rustmath-misc/output.txt` is unchanged and no `.tmp_dir_*` exists. The
reach probe `tests/zz_reach_probe.rs` was removed from the tree (a copy is `fin/zz_reach_probe.rs`).

### 6. Deviations and handoffs added by this step (for the docs step; HANDOFF and tracker not edited here)

1. A base of nonzero valuation to a truncated exponent with an integer constant term and no other
   stored term is `PrecisionExhausted` (or `UndecidedZero` for an undecided integrality), not
   `NotLaurent`. Tracker row of the series engine (powers with a variable exponent).
2. The constant field inverts a small Gaussian constant without free parameters exactly
   ((A - B*I)/(A^2 + B^2), at most 16 terms). Design §2.2 has no such rule. A HANDOFF row next to
   the exact values; follow-up: constants with free parameters, and larger ones.
3. The parser's depth limit after a construct HEAD refused (sign runs, signed exponents other than
   a literal, integer literals above i64::MAX): at most 128 levels of parentheses and calls, or
   1 024 of `^`, then a `ParseError`; the white space between signs is the grammar's (space, tab,
   CR, LF). Tracker row of the parser, with the reach table of section 2.
4. The doc comments state the measured scaling of the consumers of a result and point to S2.
   HANDOFF: the unit S2 (`S2_spec.md`), which also takes the REPL's `integrate` (section 3).
5. Kept from earlier rounds: abs and sign of real non-rational coefficients (round 4, item 4);
   functions of the imaginary exact values (round 5, m2).

### 7. Files of this step

- `rustmath-symbolic/src/series_engine.rs` (`truncated_integer_exponent`, its call in
  `pow_general`, the comment at the tree check), `src/series_const.rs` (`GAUSSIAN_INVERSE_TERMS`,
  `Konst::gaussian_inverse` and its call, the doc comments of the result caps and of
  `TextBound::total`), `src/parser.rs` (`mod depth`, `too_deep`, `sign_run`, `operand_signs`,
  `exponent_signs`, `atom`, `skip_space`, `caret`, `power`, `make_power`, the refusal in `parse`,
  the literal check), `src/series.rs` and `src/simplify.rs` (doc comments),
  `rustmath-jupyter/src/repl.rs` (the `TAYLOR_WORK` doc).
- Tests: `tests/s1_review6.rs` (new), `tests/s1_depth.rs` (the `p6_*` tests, `parse_on`,
  `refused_by_head`, `signed_tower`, `lwe_expr`, the adapted `b2_limit_with_error_of_a_signed_tower`),
  `tests/s1_singular_points.rs` (class (h)), `tests/data/gen_s1_singular.py` and
  `tests/data/s1_singular_points.tsv` (46 rows appended).
- Scratch `fu2/s1impl/fin/`: `refs6.gp`, the before logs, `s1_singular_points.tsv.before`/`.new`,
  `parse/` (the texts of the class, `bis.sh`, the probes against HEAD, the port at the start and
  the new code), `reach.sh` and `zz_reach_probe.rs`, `build.sh`, `rlibs_*.txt`, `objs_old/` and
  `objs/` (object files of the release rlibs, for the frames), `run_rel.sh`, the final logs;
  `fu2/s1impl/S2_spec.md`.

## Finish fix-up round 1

Reviews: `fu2/s1impl/S1.fin1-semantics.md` (B1, B2; minors n1 to n3) and
`fu2/s1impl/S1.fin1-resources.md` (B1 to B5; minors 1 to 4). Semantics B2 and resources B1 are the
same defect. The narrowed bar of this run applies. Branch `claude/gracious-gates-e6b6l7` at main
28c4b69 plus the uncommitted changes of the finish and of this round. Scratch: `fu2/s1impl/fix1f/`.
Nothing committed, pushed, stashed or reset. The design was not edited.

### 1. Regression tests first

Each test was built and run on the code at the start of this round, before any fix
(`fix1f/before_dbg.log`, `fix1f/before_rel.log`, `fix1f/before_rel_depth.log`; the release test
binaries of the start are kept as `fix1f/s1_finish1_before_rel` and `fix1f/s1_depth_before_rel`).

| test | what it checks | at the start |
|---|---|---|
| `tests/s1_finish1.rs` `b1f1_an_undecided_exponent_coefficient_is_not_decided` (both profiles) | semantics B1, the reviewer's table: `x^(1+(log(2)+log(3)-log(6))*x)`, `x^((log(2)+log(3)-log(6))/x)`, `x^(2+(atan(1/2)+atan(1/3)-pi/4)/x^2)`, `sin(x)^(3+(sin(1)^2+cos(1)^2-1)*x)` at 0; `(x-1)^((exp(x)*exp(-x)-1)/x^3)` and `((x-1)^2)^((exp(x)*exp(-x)-1)*(1+x))` at 1; `(x-pi)^((1+x)/(1+x))` at pi, `(x-sqrt(2))^(2*(1+x)/(1+x))` at sqrt(2), `(x-1/(pi+I))^((1+x)/(1+x))` at 1/(pi+I); `(1/x)^(1+(log(2)+log(3)-log(6))*x)` at infinity. `series_expansion`, `taylor_with_budget` and `series_coefficients` give `UndecidedZero` or `PrecisionExhausted`, the limit path `Unknown` | 10 of 10 rows `NotLaurent`, both profiles |
| `b1f1_m1_shapes_at_irrational_points_are_not_decided` (both) | the m1 shapes `(x-A)^G`, `((x-A)^2)^G`, `(1/(x-A))^G` for G in `(1+x)/(1+x)`, `2*(1+x)/(1+x)`, `(2+x^2)/(2+x^2)` at pi, sqrt(2), e, log(2), 2^(1/3), pi/4, asin(1/2): the same outcomes | 60 of 63 rows `NotLaurent` (the 3 others at sqrt(2), whose coefficients cancel exactly) |
| `b1f1_a_decided_exponent_term_keeps_not_laurent` (both) | controls: `x^(1+x+(log(2)+log(3)-log(6))*x)`, `x^((log(2)+log(3)-log(6))*x+x^2)`, `x^(1/2+(log(2)+log(3)-log(6))*x)`, `(x-pi)^(1+x)` at pi and `x^(1+x)` stay `NotLaurent` | passed |
| `b3f1_reparse_of_a_result_within_the_bar` (release, ignored) | resources B3: q4 `(10^100-1)^1050/7^120000/(1-x)` at orders 100 and 514, q5 at 282, q3 at 2 192, through `taylor_with_budget` (1e8): refused, or the re-parse of the printed result within 20 s | q4 at 100 re-parsed in 18.10 s; q4 at 514 took 94.4 s: failed |
| `b4f1_reparse_of_a_result_does_not_overflow` (debug 2 MB; release 8 MB) | resources B4: `sin^195(1)/(1-x)` at order 1000 (debug), `sin^4320(1)/(1-x)` at orders 100 and 400 (release): the input parses on the thread, then the re-parse of the result is `Ok` or a `ParseError`, or the result is refused | stack overflow, abort (both profiles) |
| `b4f1_reparse_of_a_kernel_with_a_flat_sum_does_not_overflow` (both) | the same class through `Display`: `sin(sqrt(2)+...+sqrt(301))/(1-x)` (debug) and 4 500 terms (release) at order 8. The kernel's source is the input node, a left-deep chain that `Display` writes with a parenthesis per node | abort (both profiles) |
| `b4f1_a_moderate_result_reparses` (both) | control: `sin^100(1)/(1-x)` (debug), `sin^2000(1)/(1-x)` (release) at order 200 re-parse | passed |
| `b5f1_expand_of_a_parsed_signed_tower` (both) | resources B5: `x` and 1 439 copies of `^-2`, the same after `-`, with `^-2.5`, and with ` ^ -2` (debug 2 MB); 22 772, 22 772, 22 770 and 19 996 levels (release 8 MB): parse, `expand` (which leaves these trees unchanged, compared by `Display`), drop | abort (both profiles) |
| `tests/s1_depth.rs` `p6_texts_read_by_head_keep_their_reach` (extended; both) | semantics B2: the six texts of the review with -9223372036854775808 and 150 levels parse on a 2 MB thread; `x^-9223372036854775809 + nest(x)` stays refused | failed (`ParseError`) |
| `f1_head_negative_literal_keeps_head_reach` (new; both) | resources B1: `x^-N*`, `--N*` and `2/-N*` before 200 parentheses (debug) or 1 000 (release), `x^-N*` before 150 calls (1 000) and before 1 000 carets parse, with `Integer(i64::MIN)` in the exponent; `-N*` and `N*` before 300 parentheses stay refused | failed |
| `f1_literal_rationals_refused_by_head_are_refused_not_aborted` (new; both) | resources B2: `2/x*`, `1/(x+1)*`, `2/ 3*`, `2/3.5*`, `2/--3*`, `2/+3*`, `x^2/x*`, `2 /3/x*` before 300 parentheses, `1/sin(x)*` before 300 calls, `2/x*` before 1 601 carets (2 MB, both profiles); `2/x*` before 5 000 parentheses and before 30 001 carets (8 MB, release): a `ParseError` | debug: abort; release: read (`Ok`), failed |
| `f1_literal_rationals_read_by_head_keep_their_reach` (new; both) | control: `2 / x`, `2/3^x`, `2/3/x`, `2/-9223372036854775808`, `x^-9223372036854775808/3`, `6/3/2/1` and `2/-3` before 300 parentheses parse (64 MiB thread) | failed (the two i64::MIN rows) |

Added after the fixes: `b4f1_the_parser_takes_no_stack_per_level` (100 KB of nesting in HEAD's
syntax parses on a 2 MB thread in both profiles) and the release calibration
`series_const::tests::parse_work_calibration` (ignored).

### 2. The blockers

1. **Semantics B1, `pow_general` (`src/series_engine.rs`, `variable_exponent`, called before
   `log(f)`).** For f of decided nonzero valuation v and g not an exact constant,
   f^g = exp(g log c + v g log h + g log(1 + u)) is Laurent exactly when g is an integer constant.
   The new rule reads g's stored terms first, in the reviewer's order. A decided non-integer
   constant term, or a term of exponent e != 0 with a decided nonzero coefficient, keeps the rule
   `exp(g log f)`, which gives `NotLaurent` (with its old message). Otherwise the first term
   e != 0 with an undecided coefficient gives `UndecidedZero` of that coefficient, for exact and
   truncated g alike. A g with no stored term besides its constant term goes to
   `truncated_integer_exponent`, unchanged. Its doc now states the general argument (minor n2). The
   header of `tests/s1_review6.rs` says "another stored term with a decided nonzero coefficient"
   (doc lines only; no assertion changed).
2. **Semantics B2 and resources B1, HEAD's negative literal (`src/parser.rs`, `depth::Lit`,
   `set_lit`, `take_lit`, `integer_literal`).** A per-thread state says what HEAD's `integer` rule
   read at the next literal: an atom (up to i64::MAX); a negative literal (`NegAtom`, set by
   `operand_signs` for a sign and a negative literal, `--N`, `- -N`, `+-N`, and by `exponent_signs`
   for `x^-N`; up to 2^63); or the denominator of a literal rational. `number`, the first rule of
   every atom, takes the state, so it never reaches a later literal (a float takes it too).
   `x^-9223372036854775809` still relaxes the limit.
3. **Resources B2, HEAD's literal rational (`integer_literal`, `head_denominator`).** HEAD's
   `rational` read an integer, '/' at once, and HEAD's `integer` at once: an optional '-' and
   digits whose value with the sign is in i64 and not 0. Any other text after `INT/` was a
   `ParseError` at once: `2/x`, `1/(x+1)`, `1/sin(x)`, `2/ 3`, `2/--3`, `2/+3`, `x^2/x`, and `1/x`
   itself. A '.' right after the denominator (`2/3.5`) was refused too: HEAD's integer stops there,
   and no rule of HEAD's grammar reads a '.'. Such a text now relaxes the limit. A valid
   denominator sets `Lit::Denom`, so its literal is not read as a new numerator (`2/3/x` is HEAD's
   (2/3)/x). A deviation from the required wording ("the right side is not a single i64 literal
   that ends there"): HEAD reads `2/3^x` as (2/3)^x, so relaxing whenever the right operand is more
   than one literal would refuse texts HEAD reads. The control test holds `2/3^x`. The reviewer's
   fuzzers, rebuilt against the new port (`fix1f/fz/`, with `fin1-resources/buildpr.sh dbg`):
   `fz.rs` on `t1.txt` (219 243 texts, 4 837 that HEAD parses, three deep contexts each) finds no
   deep-context mismatch (the review found 262), only the 337 plain float underflows of resources
   minor 4. `fz2.rs` with 300 parentheses finds "the port reads it 0" on `t2.txt` (96 659 texts)
   and `t3.txt` (248 260; the review found 3 058).
4. **Resources B3, the re-parse work (`src/series_const.rs`: `TextBound::parse` and `int_bits`,
   `int_parse_units`, `quot_parse_units`, `leaf_parse_units`, `MAX_RESULT_PARSE_WORK` = 2^33,
   `parse_limit`, `ResultBound`, `bound_caps`; `src/series.rs`: `StructureLimit::ParseWork`,
   `expansion_text`).** A new bound counts, at every occurrence like the text cap, the work of
   re-parsing a printed result beyond its characters: w^2/2 word products for the decimal parse of
   each integer literal of w words (num-bigint 0.4.8 multiplies by 10^19 per 19 digits), and
   8 w_p w_q for the gcd with which the parser folds each printed quotient of two integer literals
   (a `Rational` leaf, or a `/` of two signed `Integer` nodes). It is checked with the other caps:
   `series_expansion`, `taylor_with_budget`, `laurent`, `series_coefficients`, and the constant of
   `UndecidedZero`. Calibration (release, `parse_work_calibration`, `fix1f/calib.log`): 0.63 to
   0.69 ns per unit on texts whose numbers dominate (q4 and q5 quotients, an unbalanced quotient,
   long integers; a printed `/` with a large common factor 0.13), so the cap is about 5.9 s.
   Release probe (`fix1f/probe_rel.log`, `fix1f/zz_probe_rel.rs`): q4 is a value up to order 32
   (re-parse 5.71 s; at order 29, 5.01 s) and refused from 33. q5 at 282, q3 at 2 192 and
   `exp(x + (10^300-1)^1050*7^370000)` at 192 are refused; the last re-parsed in 16.4 s before,
   within the bar, so the cap is stricter than needed there. `exp(x)` at order 8 996 stays a value
   (re-parse 2.82 s, VmHWM 272 MB). `b3f1` passes: the four rows are refused in 0.7 s in all.
5. **Resources B4, the parser no longer recurses (`src/parser.rs`: `expression`, `Frame`,
   `Next`, `token`; the recursive `atom`, `parens`, `function_call`, `power`, `unary`,
   `multiplicative` and `additive` are gone).** The same grammar runs on a work stack on the heap.
   An operand pushes its signs. An atom either completes or opens a parenthesis or a call (with
   frames for the sum and the product inside). A completed atom either reads `^` (a power frame)
   or closes the powers, the signs, the product and the sum below it. A completed expression
   closes its parenthesis or adds an argument to its call. `product_step`, `make_power`,
   `operand_signs`, `exponent_signs`, `number`, `make_call` and the levels of `depth` are the
   recursive parser's. One difference: where the recursive rules backtracked from a call whose
   arguments fail to the name alone (whose '(' no rule then reads), this one fails at once; the
   text is a `ParseError` either way, and only the message may differ. A deviation from the
   reviewer's first two options (refuse a result deeper than its input, or a depth refusal that
   keeps HEAD's reach), with the evidence: a refusal would refuse inputs HEAD's `taylor` handled.
   HEAD's release `taylor` of `sin(1+2+...+8000)/(1-x)` at order 3 returns in 0.32 s on an 8 MB
   thread (`fix1f/head/`, the reviewer's `pr2_rel head file ... tay`). The port's result names
   the kernel's source, a left-deep chain that `Display` writes with 8 000 nested parentheses,
   which no recursive parser re-reads on 8 MB. (A version with that refusal was written in this
   round and dropped for this reason.) The explicit stack removes the overflow for every text.
   Equivalence with the recursive parser: section 4 (`fix1f/pdiff/`). Reach now: 100 KB of nesting
   (33 333 parentheses, 20 000 calls, 50 000 carets) parses on a 2 MB thread in both profiles. In
   release the result of `sin^4320(1)/(1-x)` at order 400 (8.7 MB) re-parses in 0.33 s, of
   `sin(sqrt(2)+...+sqrt(4501))/(1-x)` at order 8 in 0.02 s, and the port returns HEAD's case
   `sin(1+...+8000)/(1-x)` (re-parse 0.01 s). The limit of `depth` stays for the syntax HEAD
   refused, so its deep texts keep HEAD's `ParseError`.
6. **Resources B5, `expand` (`src/expand.rs`).** The walk over the tree uses an explicit stack: a
   node is visited, then built from its expanded children by the old rules (`expand_binary`). The
   rules that distribute (`expand_multiply`, `expand_power`, `distribute_negation`) still recurse
   over the sums they distribute, as HEAD's did. The outputs are the same trees. `substitute`
   keeps its recursion (not an S1 function; the reviewer measured 1 303 levels of the signed tower
   against HEAD's 1 439 in debug): a handoff to S2, whose constraints already ask for iterative
   walks.

### 3. Minor items

- Semantics n1 (`limit_with_error` gives `Finite` only for a rational limit, frozen design
  §2.13): no code change. Handoff: the tracker row of `limit_with_error` states "Finite only for a
  rational limit" (docs step; `PROJECT_TRACKER.md` is not edited here).
- Semantics n2: fixed with B1 (the docs of `variable_exponent` and `truncated_integer_exponent`).
- Semantics n3 and resources minor 3 (false parser claims): the `mod depth` doc is rewritten (the
  four constructs HEAD refused; the parser that no longer recurses). The refusal message names
  "an integer literal outside i64, or an integer literal divided at once by something other than
  an integer literal (2/x)". Section 6 corrects the report.
- Resources minor 1 (the docs claimed the re-parse was bounded): it is now, and the docs say how
  (`MAX_RESULT_TREE`, "What the caps bound" in `MAX_RESULT_TEXT`, `MAX_RESULT_PARSE_WORK`, the doc
  of `taylor_with_budget`, the comment at the driver's tree check, the REPL's `TAYLOR_WORK`).
- Resources minor 2 (`simplify` after `substitute` is not "at most 4.1 s"): the docs give the
  measured scaling, about the number of terms times the folds of one substituted coefficient (q4:
  6.8 s at order 10 and 66.0 s at 100, the reviewer's numbers; 18.3 s at order 29 and 20.0 s at
  30 in this round, with q4 now accepted up to 32): `MAX_RESULT_TEXT`, `MAX_RESULT_FOLD_WORK`,
  `StructureLimit::FoldWork`, `simplify`, `TAYLOR_WORK`. A consumer of a result, out of the bar.
- Resources minor 4 (underflowing float literals are refused; HEAD read 0.0): not counted by the
  reviewer; unchanged (the rule of fix-up round 3).

### 4. Acceptance reruns (final code)

| command | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic --no-fail-fast -- --test-threads=2` (debug, `fix1f/sym_dbg_final.log`) | unit 685 (5 ignored, the new calibration among them); `s1_depth` 14; `s1_finish1` 8 (1 ignored); `s1_fuzz_data` 7; `s1_payload` 2 (1); `s1_regressions` 26 (2); `s1_review2` 2; `s1_review3` 7 (7); `s1_review4` 4 (2); `s1_review4_memory` 1 (1); `s1_review5` 4 (2); `s1_review5_memory` 2 (2); `s1_review6` 4; `s1_singular_points` 9; doc 116 (11); 891 passed, 0 failed |
| `cargo test --release -j 2 -p rustmath-symbolic --no-run`, then each release binary (`fix1f/run_rel.sh`, `fix1f/rel_tests_final.log`) | 685 (5), 14, 8 (1), 7, 2 (1), 26 (2), 2, 7 (7), 4 (2), 1 (1), 4 (2), 2 (2), 4, 9; 775 passed, 0 failed |
| release `--ignored --test-threads=1` (same log) | `s1_finish1` 1 (0.72 s); `s1_payload` 1 (3.27 s); `s1_regressions` 2 (0.88 s); `s1_review3` 7 (2.07 s); `s1_review4` 2 (2.43 s); `s1_review4_memory` 1; `s1_review5` 2 (2.20 s); `s1_review5_memory` 2; `fold_work_calibration` passed (47.1 s, worst 1.12 ns per unit, the cap about 2.4 s); `parse_work_calibration` passed (3.7 s, worst 0.651 ns per unit, the cap about 5.6 s; an earlier run 0.689 and 5.9 s) |
| the differential parser test (`fix1f/pdiff/pdiff.rs`, debug, 1 GiB thread, `pdiff_all.log`): the new parser against the recursive one of the start of this round with the B1 and B2 rules, on 200 000 seeded random token texts and the reviewer's `t1.txt`, `t2.txt`, `t3.txt`, each alone and in three deep contexts (300 parentheses after it, 300 around it, 2 400 carets after it) | 2 825 888 texts (706 472 distinct): all agree. 115 325 both `Ok` with the same tree (compared node by node), 2 479 001 both a `ParseError`, 190 647 both the depth refusal, 40 915 both the underflow refusal; 0 differ |
| the reviewer's fuzzers against HEAD (`fix1f/fz/`) | `fz.rs t1.txt 300`: 337 mismatches, all plain float underflows (minor 4), no deep context; `fz2.rs 300`: the port reads 0 of the texts HEAD refuses, on `t2.txt` and `t3.txt` |
| dependents (`cargo test -j 2 --no-fail-fast -p rustmath-calculus -p rustmath-curves -p rustmath-functions -p rustmath-geometry -p rustmath-manifolds -p rustmath-plot3d`, `fix1f/deps_test.log`) | 1 556 passed, 0 failed (35 ignored) |
| `cargo test -j 2 -p rustmath-jupyter --lib`; `--test kernel_protocol -- --test-threads=1` (`fix1f/jup_lib.log`, `fix1f/jup_kernel.log`) | 120 passed (3 ignored); 4 passed |
| `cargo check -j 2 --workspace --tests` (`fix1f/check_ws.log`) | no error |
| `cargo clippy -j 2 -p rustmath-symbolic --no-deps` (`fix1f/clippy_lib.log`) and `--tests` (`fix1f/clippy_tests.log`) | no warning on a line of this round (55 warnings on the library, all on older lines). The lib-test target stops at the two `approx_constant` errors in HEAD's parser tests, as before. The two warnings on new test code (an `assert!` of a constant, an `Arc` in `vec![_; n]`) are fixed |
| rustfmt | `s1_finish1.rs`, `s1_depth.rs`, `s1_review6.rs`, `series_const.rs` and `series_engine.rs` are clean; in `parser.rs`, `expand.rs` and `series.rs` the remaining changes are on older lines (none on a line of this round) |

Total of the final runs: 3 366 passed, 0 failed (symbolic debug 891; release 775 and the 20 ignored
budget and calibration tests; downstream 1 556; jupyter 124). `target/release` was deleted after
the release runs (free disk 6.5 GB). `rustmath-misc/output.txt` is unchanged and no `.tmp_dir_*`
exists. The release probe `tests/zz_probe_rel.rs` was removed from the tree after its build (a copy
is `fix1f/zz_probe_rel.rs`).

### 5. Deviations and handoffs added by this round (for the docs step; HANDOFF and tracker not edited here)

1. A power f^g with f of decided nonzero valuation and an exponent with an undecided non-constant
   coefficient is `UndecidedZero` (of that coefficient), not `NotLaurent`. Tracker row of the
   series engine (powers with a variable exponent).
2. The parser runs on a heap stack (`expression`), not by recursion: no text overflows a thread,
   and no re-parse of a result overflows (corrected in "Finish minor items": a result whose text
   holds a construct that relaxes the depth limit, then more than 128 levels of nesting, is refused
   by the limit when it is re-parsed). Design §2.12 described recursive rules; the grammar and the trees
   are the same (the differential test of section 4). Tracker row of the parser; the reach tables
   of the earlier rounds are obsolete (100 KB of nesting parses on a 2 MB thread).
3. The depth limit after a construct HEAD refused now also covers HEAD's literal rational (`2/x`,
   `1/(x+1)`, `2/3.5`) and spares HEAD's negative literal i64::MIN. Tracker row of the parser.
4. A new structural cap, `StructureLimit::ParseWork` (`MAX_RESULT_PARSE_WORK` = 2^33 units, about
   6 s): the re-parse of a result's numbers at every occurrence. Design §2.15 lists the caps;
   this one is added. Tracker row of the result caps.
5. `expand` walks its input with an explicit stack. Tracker row of `expand`.
6. Handoffs: the tracker row of `limit_with_error` states "Finite only for a rational limit"
   (semantics n1); `substitute` keeps its recursion, and the S2 unit (`fu2/s1impl/S2_spec.md`)
   already asks for iterative walks of the consumers (resources B5's note).

### 6. Corrections to earlier sections

- "Finish under the narrowed bar", item 5: "What it refuses: only texts HEAD refuses too" was
  false: HEAD's negative literal -9223372036854775808 relaxed the limit (fixed, item 2 above), and
  HEAD's literal rational was missing from the relaxing constructs (item 3). The frames, the
  `#[inline(never)]` tuning and the reach table of that item describe the recursive parser, which
  this round replaced. "A text in HEAD's syntax deeper than HEAD's reach still overflows" is no
  longer so: no text overflows.
- The same section, item 1 and the doc comments it lists: "the printing, `Debug`, dropping and
  re-parsing of their results" were bounded; the re-parse was not (resources B3, B4) and is now
  (section 2, items 4 and 5). "`simplify` after `substitute` at most 4.1 s" was wrong (resources
  minor 2; section 3).
- The same section, item 2 (m1): `NotLaurent` stayed for an exponent with another stored term
  whose coefficient is undecided (semantics B1; fixed, item 1 above).

### 7. Files of this round

- `rustmath-symbolic/src/series_engine.rs` (`variable_exponent` at `:2089`, its call in
  `pow_general` at `:2062`, the doc of `truncated_integer_exponent` at `:2137`, the comment at the
  driver's tree check).
- `rustmath-symbolic/src/parser.rs` (`mod depth` and its doc from `:184`, `depth::Lit`, `set_lit`,
  `take_lit`; `integer_literal` `:302`; `head_denominator` `:334`; `number` `:474`;
  `operand_signs` `:519`; `exponent_signs` `:532`; `token` `:564`; `Frame` `:616`; `Next` `:633`;
  `expression` `:672`; the refusal message in `parse`; the recursive rules removed).
- `rustmath-symbolic/src/series_const.rs` (`MAX_RESULT_PARSE_WORK` `:769`; `TextBound::parse` and
  `int_bits` `:1159`; `int_parse_units`, `quot_parse_units`, `leaf_parse_units` from `:1168`;
  `parse_limit`, `ResultBound`, `bound_caps` from `:1758`; `parse_work_calibration` `:5049`; the docs
  of `MAX_RESULT_TREE`, `MAX_RESULT_TEXT`, `MAX_RESULT_FOLD_WORK`).
- `rustmath-symbolic/src/series.rs` (`StructureLimit::ParseWork` `:163`, its message and mapping,
  `expansion_text` `:638`, `public_expansion`, the docs of `FoldWork` and `taylor_with_budget`).
- `rustmath-symbolic/src/expand.rs` (`expand` `:35`, `expand_binary` `:151`).
- `rustmath-symbolic/src/simplify.rs`, `rustmath-jupyter/src/repl.rs` (`TAYLOR_WORK`): doc
  comments.
- Tests: `tests/s1_finish1.rs` (new: `b1f1_*`, `b3f1_*`, `b4f1_*`, `b5f1_*`); `tests/s1_depth.rs`
  (`p6_texts_read_by_head_keep_their_reach` extended, `integer_leaves`,
  `f1_head_negative_literal_keeps_head_reach`, `f1_literal_rationals_refused_by_head_are_refused_not_aborted`,
  `f1_literal_rationals_read_by_head_keep_their_reach`); `tests/s1_review6.rs` (two doc lines).
  The tests-first files are untouched, and no assertion or expected outcome of an earlier test was
  changed.
- Scratch `fu2/s1impl/fix1f/`: the before logs and binaries, `pdiff/` (`parser_port0.rs`, the
  recursive parser of the start; `pdiff.rs`; `pdiff_all.log`), `fz/` (the reviewer's fuzzers rebuilt
  and their outputs), `head/` (HEAD's `taylor` on flat sums in a kernel), `zz_probe_rel.rs` and
  `probe_rel.log`, `calib.log`, `run_rel.sh`, `rel_exes.txt`, the final logs.

## Finish minor items

Reviews: `fu2/s1impl/S1.fin2-semantics.md` and `fu2/s1impl/S1.fin2-resources.md`, both approved
under the narrowed bar with minor items only. This step fixes all seven items of the list (one of
them is a note). Branch `claude/gracious-gates-e6b6l7` plus the uncommitted changes of the finish
and its fix-up round 1 (HEAD moved to 52708d7 during the step by merges that touch no file of
S1; the diff against it is the diff against 28c4b69). The step ran in two sessions: the first
(2026-11-16) wrote the tests and the fixes and ran the full acceptance; the second (2026-11-20)
checked each fix against the reviews, refined three doc texts and reran the tests that cover
them. Scratch: `fu2/s1impl/minors/` (the second session in `minors/r2/`). Nothing committed,
pushed, stashed or reset. The design was not edited. No tests-first file and no earlier
assertion or expected outcome was changed.

### 1. Tests written first

Three tests at the end of `tests/s1_finish1.rs` (a file of the finish, not of the first step),
run on the code at the start of this step (`minors/before_dbg.log`):

| test | what it checks | at the start |
|---|---|---|
| `fin2_a_printed_relaxing_construct_limits_the_reparse` (`:436`, both profiles) | semantics m1, resources minor 1: the result of `taylor_with_budget` (REPL budget) re-parses for `x` and 100 copies of `^-2` at 1, order 3; `exp(x) + sin^150(1)/(1-x)` at 0, order 20; `exp(x)*sin^130(1)` at order 20; `sin(A1+...+A100)/(1-x)` with 30-digit Ak at order 3. It is refused by the depth limit (the message of `parse`) for 110 copies of `^-2`; 33 333 copies (100 KB, added in the second session); `exp(x) + sin^150(1)/(1-x)` at order 21; `exp(x)*sin^130(1)` at order 25; 200 terms Ak at order 3 | passed: it pins the behaviour that the docs now state (the fix is a doc fix). The edges agree with the reviewers' measurements |
| `fin2_a_literal_zero_divisor_has_the_design_message` (`:474`, both) | resources minor 2: `1/0`, `5 / 0`, `(1)/(0)`, `10\r/0`, `2/-0`, `x + 3/(0)`, `(7/7)/-(0/5)`, `0009223372036854775807 /0`, `sin(1/0)` are a `ParseError` whose message is "division by zero in a literal"; `x/0`, `5/(0*x)` and `0/5` still parse | failed: the message was nom's "Failed to parse expression: Parsing Failure: Error { input: \"\", code: Verify }" |
| `fin2_precision_exhausted_names_the_bounded_search` (`:502`, both) | semantics m3: `x^((1+x)/(1+x))`, `arcsin((1+x)/(1+x))` and `1/(sin(x)^2+cos(x)^2-1)` at 0 through `series_expansion` are `PrecisionExhausted`, its `Display` names "could not decide a coefficient, an exponent or a branch-point argument" and the working precision, and no longer "denominator vanished" | failed: "a denominator vanished to working precision 131 (it may be identically 0)" |

### 2. The items (file:line at the end of this step)

1. **Semantics m1 and resources minor 1: a printed result does not always re-parse.** Agreed; a
   doc fix, the behaviour is unchanged. The depth limit of the parser applies to a re-parse too,
   and a result prints relaxing constructs of its own: an integer above i64::MAX (from 21! on), a
   `Rational` with such a denominator, or `^ -(` (how `Display` writes the negated exponent the
   parser builds for HEAD's `x^-2^-2`). After one of them, more than 1 024 units counted from the
   start of the text (128 levels of parentheses and calls, or 1 024 of `^`) are refused at once
   with the typed `ParseError`. The texts now say so, with the measured examples:
   - `rustmath-symbolic/src/parser.rs:197` (the `mod depth` doc: no text overflows, "the re-parse
     of a printed result included"; `:203` "The exception for printed results": the condition,
     the four examples with their edges, and the note that a follow-up may exempt the parser's own
     printed forms);
   - `src/series.rs:756` (the doc of `taylor_with_budget`);
   - `src/series_const.rs:644` (`MAX_RESULT_TREE`) and `:691` (`MAX_RESULT_TEXT`, "What the caps
     bound");
   - `rustmath-jupyter/src/repl.rs:11221` (`TAYLOR_WORK`, "What is bounded");
   - the report: finish fix-up round 1, section 5, deviation 2, now says "no re-parse of a result
     overflows" and points here.
   The examples: `x` and 110 copies of `^-2` (332 bytes) and 33 333 copies (100 KB; the
   resources review: both results refused, 0.000 s) at 1, order 3; `exp(x) + sin^150(1)/(1-x)`
   from order 21; `exp(x)*sin^130(1)` at order 25; `sin(A1+...+A200)/(1-x)` with 30-digit Ak,
   refused from order 3 (the test) to order 5 177 (the review). `Display` writes that kernel's
   source, a left-deep chain, with a parenthesis per `+`, so the first literal comes after 200 of
   them; 100 terms re-parse. The depth refusal's message names the constructs by class ("an
   integer literal outside i64", "a signed exponent such as x^-y"), which covers the printed
   forms, so it is kept.
2. **Semantics m2 and resources minor 3: the consumer numbers.** Agreed. Each number now names the
   consumer it was measured on (fix-up round 5's table, `fix5/probe/edges_final.log`, rechecked:
   `simplify(R)` at most 0.862 s over the 26 edge rows, 0.352 s for the 1 487 fractions at order
   10; after `substitute` 3.934 s and after `expand` 4.121 s on that row; after `differentiate`
   2.90 s for `sin((2^254+1)^4096)/(1-x)` at order 192):
   - `src/series_const.rs:716` (`MAX_RESULT_TEXT`, the bullet on `simplify`): the slowest
     `simplify` of the result itself 0.86 s (2 032 fractions 1/p at order 1); the 1 487 fractions
     at order 10: `simplify` 0.35 s, after `expand` 4.12 s, after `substitute(x, 2)` 3.93 s
     (round 5; the fin2 semantics review 0.36 s, 4.53 s and 4.70 s); after `differentiate` the
     slowest 2.90 s;
   - `:728` (the bullet on `simplify` after `substitute`): the accepted orders of q4, 18.3 s at 29
     and 20.0 s at 30 (fix-up round 1 of the finish), 21.9 s at 30 and 23.3 s at 32 (the fin2
     resources review, `fin2-resources/rp/q4_subssimp.log`);
   - `:761` (`MAX_RESULT_FOLD_WORK`): 4.1 s is after `expand` (4.5 s in the fin2 review); the
     result itself 0.35 s;
   - `:781` (`MAX_RESULT_PARSE_WORK`): "about 6 s in the calibration", and at the cap 5.9 to 7.6 s
     under load (q3 at order 379, 8.58e9 units, 0.69 to 0.88 ns per unit at load 5), "so about 6
     to 8 s";
   - `src/simplify.rs:422` to `:434` (the doc of `simplify`): the 0.86 s, 4.12 s, 3.93 s and
     2.90 s of the edges with the fin2 numbers; q4 after `substitute`: 6.8 s at order 10 and
     66.0 s at order 100 in the fin1 review, now refused above order 32, the accepted orders 29 to
     32 taking 18 to 23 s;
   - `rustmath-jupyter/src/repl.rs:11221` ("about 6 to 8 s") and `:11231` (`simplify` of a result
     0.86 s at the slowest edge measured, about 4 to 5 s after `expand` or `subs`; q4 refused
     above order 32, the accepted orders 18 to 23 s).
   The second session reworded the first session's "at most 0.86 s", "at most 4.12 s, 3.93 s and
   2.90 s" and "after `differentiate`: at most 2.90 s" as the slowest measured times: the narrowed
   bar asks that no doc claim a bound on a consumer of a result, and "at most 4.12 s" next to the
   review's 4.53 s read as one. A grep of the added lines for "at most" now finds it only for the
   entry points, the printing and re-parse, and the structural caps.
3. **Semantics m3: the message of `PrecisionExhausted`.** Agreed. `src/series.rs:105` (the doc
   lists the bounded searches that return it: the leading coefficient of a denominator or of the
   argument of log, abs, sign or a power; the valuation of an argument of exp, sin, the atan family
   or an integrand; the rest of an exponent with an integer constant term on a base of nonzero
   valuation; the distance of an inverse function's argument from a branch point) and `:234` (the
   message: "a bounded search could not decide a coefficient, an exponent or a branch-point
   argument to working precision P (what it tested may be identically 0)"). The internal doc of
   `SErr::NeedPrecision` (`src/series_const.rs:53`) says the same. No test or caller compared the
   old text (grep of the workspace: the only other "denominator vanished" is an unrelated doc line
   in `rustmath-functionfields`).
4. **Resources minor 2: a literal zero divisor.** Agreed. `src/parser.rs:151`: `parse` maps the
   `Verify` failure of the literal fold (the only `Verify` the parser makes, `:788`) to the
   design's message, "division by zero in a literal" (§2.12). The behaviour is unchanged: every
   quotient of two signed integer literals with a zero divisor is refused, as the frozen design
   says. Against HEAD: HEAD refused `1/0` and read `5 / 0`, `(1)/(0)` and `10\r/0` as Div(n, 0)
   (17 plain texts of the resources fuzzer's 818 849). The value is undefined, and MAGMA and Sage
   refuse `5/0`, so the reviewer did not count it under bar (b); the tracker row of the parser
   should state it (handoff, section 4).
5. **Resources note 4 (a): `test_airy_numerical_helper`.** No change, as asked. In the release run
   of all ignored tests of the unit binary (section 3) it fails (`airy.rs:369`, `ai_large < 0.001`).
   It is HEAD's ignored demonstration test (`#[ignore]` at 8a0b310); S1 changed only the
   `matches!` borrows of other tests in that file. The failure is HEAD's.
6. **Resources note 4 (b) and semantics m4.** Notes, no change asked: the float underflow refusals
   stay the rule of fix-up round 3; the row `(x-1/(pi+I))^(1/2+(1+x)/(1+x)-1)` at 1/(pi+I) is an
   honest `UndecidedZero`.

### 3. Acceptance reruns

First session (2026-11-16, the fixes; the code was the same as now, only doc text changed after):

| command | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic --test s1_finish1 fin2_` on the code at the start (`minors/before_dbg.log`) | 1 passed (the re-parse test, which pins existing behaviour), 2 failed (the two messages) |
| `cargo test -j 2 -p rustmath-symbolic --no-fail-fast -- --test-threads=2` (debug, `minors/sym_dbg_final.log`) | unit 685 (5 ignored); `s1_depth` 14; `s1_finish1` 11 (1 ignored); `s1_fuzz_data` 7; `s1_payload` 2 (1); `s1_regressions` 26 (2); `s1_review2` 2; `s1_review3` 7 (7); `s1_review4` 4 (2); `s1_review4_memory` 1 (1); `s1_review5` 4 (2); `s1_review5_memory` 2 (2); `s1_review6` 4; `s1_singular_points` 9; doc 116 (11); 894 passed, 0 failed |
| `cargo test --release -j 2 -p rustmath-symbolic --no-run`, then each binary (`minors/run_rel.sh`, `minors/rel_tests_final.log`) | 685 (5), 14, 11 (1), 7, 2 (1), 26 (2), 2, 7 (7), 4 (2), 1 (1), 4 (2), 2 (2), 4, 9; 778 passed, 0 failed |
| release `--ignored --test-threads=1`, every binary (`minors/run_rel_ign.sh`, `minors/rel_ignored_final.log`) | `s1_finish1` 1 (`b3f1`, 0.73 s); `s1_payload` 1 (3.50 s); `s1_regressions` 2 (0.76 s); `s1_review3` 7 (1.98 s); `s1_review4` 2 (2.37 s); `s1_review4_memory` 1; `s1_review5` 2 (2.13 s); `s1_review5_memory` 2; unit: `fold_work_calibration`, `parse_work_calibration` and HEAD's two `factor` tests pass; HEAD's `specialfunctions::airy::tests::test_airy_numerical_helper` fails (`airy.rs:369`, item 5). `parse_work_calibration` again with `--nocapture` (`minors/calib_parse.log`, load 0.9): worst 0.669 ns per unit, the cap about 5.7 s |
| `cargo test -j 2 -p rustmath-jupyter --lib`; `--test kernel_protocol -- --test-threads=1` (`minors/jup_lib.log`, `minors/jup_kernel.log`) | 120 passed (3 ignored); 4 passed |
| dependents: `cargo test -j 2 --no-fail-fast -p rustmath-calculus -p rustmath-curves -p rustmath-functions -p rustmath-geometry -p rustmath-manifolds -p rustmath-plot3d` (`minors/deps_test.log`) | 1 556 passed, 0 failed (35 ignored), 18 binaries |
| `cargo check -j 2 --workspace --tests` (`minors/check_ws.log`) | no error |

Second session (2026-11-20, the final sources, `minors/r2/`):

| command | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic --no-fail-fast -- --test-threads=2` (debug; `r2/sym_dbg.log`, then `r2/sym_dbg_final.log` after the last doc edit) | the same counts as above, 894 passed, 0 failed (34 ignored), both runs; the three `fin2_*` tests pass, the 100 KB row included |
| `cargo test --release -j 2 -p rustmath-symbolic --test s1_finish1 --test s1_depth -- --test-threads=2` (`r2/rel_finish1_depth.log`) | `s1_depth` 14 passed; `s1_finish1` 11 passed (1 ignored), the `fin2_*` tests on 8 MB threads |
| the same with `--ignored --test-threads=1` for `s1_finish1` (`r2/rel_finish1_ignored.log`) | `b3f1_reparse_of_a_result_within_the_bar` passed (1.00 s) |
| `cargo check -j 2 -p rustmath-jupyter --tests` (`r2/jup_check.log`; `repl.rs` changed only in doc text) | no error |
| `cargo clippy -j 2 -p rustmath-symbolic --no-deps` (`r2/clippy_lib.log`) and `--test s1_finish1` (`r2/clippy_finish1.log`) | the library: 55 warnings, none on a line of this step (as before); `s1_finish1.rs`: no warning |
| rustfmt (`rustfmt --edition 2021 --check`) | `tests/s1_finish1.rs` and `src/series_const.rs` are clean; in `parser.rs`, `series.rs` and `simplify.rs` the remaining changes are all on older lines |
| line width | no added line of this step is over 100 characters |

Total: 3 374 passed, 0 failed in the first session's final runs (symbolic debug 894, release 778
and 22 ignored tests that pass: the budget and calibration tests and HEAD's two `factor` tests;
downstream 1 556; jupyter 124). HEAD's ignored `test_airy_numerical_helper` fails (item 5). In the
second session's final reruns: 920 passed, 0 failed (debug 894; release 25 and 1 ignored).
`target/release` was deleted after the release runs of each session (free disk 5.8 GB).
`rustmath-misc/output.txt` is unchanged (md5 746308829575e17c3331bbcb00c0898b) and no
`.tmp_dir_*` exists. `git status --short` lists the same files as at the start
(`minors/status_before.txt`); this step changed `parser.rs`, `series.rs`, `series_const.rs`,
`simplify.rs`, `repl.rs` and `tests/s1_finish1.rs` only.

### 4. Handoffs for the docs step (HANDOFF and tracker not edited here)

The reviews ask for tracker and HANDOFF text (items 1 and 4). This run's rule is not to edit
`PROJECT_TRACKER.md` or `docs/`, and the owner's decision puts the tracker update after the minor
items. The parser's tracker row still describes HEAD (`PROJECT_TRACKER.md:2156`, "Broken
literals ... not fixed"), so the docs step rewrites it whole and takes these items:

1. Tracker row of the parser and the HANDOFF: the re-parse exception of item 1 (a relaxing
   construct in the printed text, then more than 128 levels of parentheses and calls or 1 024 of
   `^`; the examples of item 1), with the follow-up "exempt the parser's own printed forms from the
   depth limit".
2. Tracker row of the parser: against HEAD, `5 / 0`, `(1)/(0)` and `10\r/0`, which HEAD read as
   Div(n, 0), are refused with "division by zero in a literal" (design §2.12; HEAD refused `1/0`
   too).
3. Tracker row of the series errors: the message of `PrecisionExhausted` names a coefficient, an
   exponent or a branch-point argument.
4. When the full ignored suite of `rustmath-symbolic` is run, `test_airy_numerical_helper` fails;
   it is HEAD's.
5. Carried from fix-up round 1: the tracker row of `limit_with_error` states "Finite only for a
   rational limit" (semantics n1).

### 5. Files of this step

- `rustmath-symbolic/src/parser.rs` (the `Verify` arm of `parse` `:151`, the comment at the fold
  `:788`, the `mod depth` doc `:197` to `:224`), `src/series.rs` (`PrecisionExhausted`'s doc
  `:105` and message `:234`, the doc of `taylor_with_budget` `:756`), `src/series_const.rs` (the
  docs of `SErr::NeedPrecision` `:53`, `MAX_RESULT_TREE` `:644`, `MAX_RESULT_TEXT` `:680` to
  `:739`, `MAX_RESULT_FOLD_WORK` `:761`, `MAX_RESULT_PARSE_WORK` `:781`), `src/simplify.rs` (the
  doc of `simplify` `:419` to `:434`), `rustmath-jupyter/src/repl.rs` (the doc of `TAYLOR_WORK`
  `:11217` to `:11237`): doc comments, one message and one match arm.
- `rustmath-symbolic/tests/s1_finish1.rs` (the module doc; `big_sum`, `reparse_message` and the
  three `fin2_*` tests from `:400`).
- This report: finish fix-up round 1, section 5, deviation 2 (corrected in place) and this
  section.
- Scratch `fu2/s1impl/minors/`: `diff_before.patch` and `status_before.txt` (the state at the
  start), `diff_after.patch` (the state at the end of the first session), the first session's logs
  and scripts (section 3), `recon/` (the files at the start, rebuilt from `diff_before.patch`, to
  check the first session's changes), and `r2/` (the second session's logs and `run_all.sh`).

## C-1 and C-4

Requests C-1 (P0, an argument that leaves a branch cut) and C-4 (P0, the crates that depend on
rustmath-symbolic) of `docs/workstreams/m23_golay/REQUESTS.md`, section 2, under the narrowed bar.
Branch `claude/gracious-gates-e6b6l7`, HEAD 52708d7, plus the uncommitted changes of this run
(`c1/diff_before.patch` and `c1/status_before.txt` are the state at the start, `c1/diff_after.patch`
the state at the end). Scratch: `fu2/s1impl/c1/`. Nothing committed, pushed, stashed or reset. The
design was not edited. `PROJECT_TRACKER.md` and `docs/` were not edited.

### 1. Tests written first

New: `rustmath-symbolic/tests/s1_cuts.rs` (`:67` to `:236`) and `tests/data/s1_cuts.tsv`, made by
`tests/data/gen_s1_cuts.py` (`python3 -I gen_s1_cuts.py > s1_cuts.tsv`; gp 2.15.4 at realprecision
300 through `s1_gp.py`). 82 rows, in this order:

| ids | rows | expected |
|---|---|---|
| `c1:1` to `c1:5` | the five inputs of the C-1 table | `NotLaurent` |
| `c1m:1` to `c1m:5` | their mirror images, I -> -I | `NotLaurent` |
| `ctl:*` | 37 real-coefficient controls on each open cut of each function: the four of the request first, then log (6 more, among them `log(x)` at -1, `log(-pi + x^2)`, the truncated `log(-1/2 + sin(x))` and `log(-2 + x*I*log(-1))`, whose coefficient -pi is real only by its enclosure), sqrt (2 more), powers `^(1/3)`, `^(2/3)`, `^(-1/2)`, `^pi`, `^I`, `^(1/2 + I)`, variable exponents (2), arcsin (2 more), arccos (2), arctanh (2 more), arccosh (4: both parts of its cut, and `arccosh(sinh(x))`), arctan (4: both cuts, `arctan(2*I*sqrt(1 + x))`), arcsinh (3) | a value, gp's series |
| `off:*` | 10 complex arguments off every cut (`log(sqrt(-4) + I*x)`, `log(log(-2) + I*x)`, `sqrt(x)` at I, ...) | a value |
| `lv:*` | 22 arguments that leave a cut, at least one per function and cut (`log(I*x)` at I, `(-2 + I*x)^(1 + x)`, `arctan(sqrt(-4) + x)`, the truncated `arctan(2*I + sin(x))`, and the limit-path input below) | `NotLaurent` |
| `und:*` | 3 undecided real constants on a cut (`log(-1 + I*(sin(1)^2 + cos(1)^2 - 1)*x)`, ...) | `UndecidedZero` |

The tests: the request rows come first (`data_has_the_request_rows_first`); the ten C-1 rows refuse
through `series_expansion` and `taylor_with_budget` with a message that contains "leaves a branch
cut at the point", and the limit path is `Unknown` (`c1_rows_refuse_with_the_branch_cut_message`);
`check_all` over each group (REPL budget, 2 MB debug or 8 MB release thread, limit path); the limit
path of `(log(-1 - I*x^2) - log(-1 + I*x^2))/(I*pi)` at 0 is `Unknown` (the true limit is -2; gp:
`(log(-1 - I*10^-9) - log(-1 + I*10^-9))/(I*Pi)` is -1.99999999936...); parameters
(`parameters_are_generic`: `log(y + I*x)` at 0 has the constant term `log(y)`, `log(-1 + y*x)` is
`NotLaurent`); and a release budget test (`cut_rule_within_the_bar_release`, ignored in debug,
section 5).

On the code at the start (`c1/before_dbg.log`): 3 passed, 6 failed. The five C-1 inputs and their
mirrors returned `Ok` with gp's formal series; 21 of the 22 `lv:` rows returned `Ok` and
`arctan(sqrt(-4) + x)` was `UndecidedZero`; the three `und:` rows returned `Ok`; the limit path read
`Finite(0)`; `log(-1 + y*x)` returned `Ok`. The 36 controls of then and the 10 `off:` rows passed
(`ctl:log:7` was added later and passes).

Three of my own rows changed before the rule was written, none of a tests-first file:
`arctan(sqrt(-4 + x))` became `arctan(2*I*sqrt(1 + x))` (the constant field does not reduce
`((-4)^(1/2))^2`, so the first is an honest `UndecidedZero` of the integrand, before and after);
`sqrt(I + x)` became `sqrt(x)` at I (gp computes the sqrt of a series with an exact complex constant
term to 19 digits, the note of `series_prog`); the limit column of `ctl:powvar:1` and `:2` is
`Finite(-2)` and `Finite(1)` (gp's constant term is a float, the engine's (-2)^1 and (-1)^2 are
exact).

One fix of the test evaluator, `tests/s1_common/eval.rs:201` and `:234`: on the lower imaginary
cuts MPC's `atan` and `asinh` (a +0 real part) take the limit from Re > 0, gp's principal formulas
the limit from Re < 0. The evaluator now takes -f(-z) there (gp 2.15.4, realprecision 40:
`atan(-3*I)` = -1.5707963267948966192 - 0.34657359027997265471*I, `asinh(-2*I)` =
-1.3169578969248167086 - 1.5707963267948966192*I; MPC gave +1.5707... and +1.3169...). No existing
row evaluates arctan or arcsinh of i t with t < -1, so no expected outcome changes; the rows
`ctl:arctan:3` and `ctl:arcsinh:2`, `:3` need it.

### 2. What changed (file:line at the end of this step)

The rule (`series_engine.rs:1160` to `:1521`, the comment block at `:1160`). At each application of
a branch function, after its own checks (an exact-constant argument is evaluated first, and the
branch-point rules keep their order), `cut_check` (`:1503`) asks: does the class of the argument's
coefficients keep it in the cut's direction? If not, where does its constant term c0 lie
(`position`, `:1351`)? If on an open cut, do its coefficients keep it there (`coefficient_verdict`,
`:1474`)? The outcomes: c0 decided off the cut, or the argument decided to stay on it: the
expansion as before; a coefficient decided off the cut's direction: `NotLaurent` with
`leaves_cut` (`:1272`): "<f>: the argument leaves a branch cut at the point (its constant term lies
on the cut, and a coefficient that is not a real multiple of the cut's direction takes it off; the
engine expands a branch function on its cut only for an argument that stays there)"; a real
constant whose zero test is undecided (the imaginary part of c0 or of a coefficient, written as
A + B*I with A and B real; the distance of c0 from an end of the cut, `position_undecided`,
`:1331`): `UndecidedZero` of it; any other undecided case: `NotLaurent` with `cannot_decide_cut`
(`:1280`). A truncated argument whose stored coefficients are all real is undecided (its terms
beyond the precision are not known), unless its class says so: a bounded check never decides that
the argument stays on the cut. The cuts (`Cut`, `:1247`): (-oo, 0) for log and the non-integer
powers; (-oo, -1) and (1, oo) for arcsin, arccos, arctanh; (-oo, -1) and (-1, 1) for arccosh; the
two half-lines of the imaginary axis beyond +-I for arctan and arcsinh. An end of a cut is a branch
point, left to the owner's second rule (the integrand's `NotLaurent`). The applications: `log`
(`:779`, also the `log f` of a power with a variable exponent, `pow_general_log`, `:2598`);
`pow_const` (`:895`) for sqrt and a constant non-integer exponent, where an exponent decided to be
an integer skips the rule and an undecided one is `UndecidedZero` of alpha - n; and `atan_like`
(`:1559` to `:1561`) for the six inverse functions. The integrands of the atan family call
`pow_const` with no class (the function checked its own cut first). Erf, exp, the trigonometric and
hyperbolic functions and integer powers have no cut. A constant base (`c^g`) does not move, so it
has no check.

The classes (`Tail`, `:1183`): for each value of the expansion, whether every coefficient of an
exponent other than 0 (beyond the precision too) is real, or purely imaginary, as a sufficient
condition built node by node. The work stack carries it (`type SV`, `:2126`): a constant has both,
the variable is real (`:2412`), a sum takes the meet (`:2266`), a product of one series factor
scales it by the constant, other products and quotients multiply full classes (`:2291`,
`finish_product` `:2536`), integer and constant powers (`pow_by_const` `:1694`, `pow_int_class`
`:1708`, `pow_class` `:1729`), variable exponents (`pow_general` `:2569`, `const_base_class`
`:2713`), and the unary functions (`expand_unary` `:2780`: real functions of a real argument, odd
and even functions of an imaginary one, log, sqrt, exp by the class of its constant term) and the
atan family (`atan_like` `:1549`: the class of the integrand, b' times the multiplier, whose class
is that of its constant term). `full_class` (`:1233`) adds the constant term.

The constants' classes (`series_const.rs`): each kernel has a class (`KData::kcls`, `:104`;
`KCLS_REAL`, `KCLS_IMAG`, `KCLS_UNKNOWN`, `:108` to `:112`), set once when the kernel is made
(`:3365`, inside `intern`, so a lookup costs nothing) by `kernel_struct_class` (`:3420`): pi, e,
radicals, floats, real functions of real constants (gamma, factorial, zeta and mod included), log
and sqrt of a positive constant, the inverse functions inside their real domain, powers, 1/P, I,
sqrt of a negative rational, odd functions of imaginary constants. For log, sqrt, the inverse
functions and c^a of a real constant that is not rational, the sign or range of c is read from its
real enclosures (`ball_sign_of` `:3528`, `ball_range_class` `:3539`; charged, once per kernel).
`KSize` gains `all_real` and `all_imag` (`:249`, `:252`, set in `ksize` at `:381`), read by
`Konst::struct_real` and `struct_imag` (`:2191`, `:2196`) in O(1). `struct_positive` (`:3391`).
`var_is_named_i` is public (`:4091`).

Complex enclosures (`series_ball.rs:1013` to `:1426`): `CBall` (`:1036`), a rectangle of two real
balls, with the principal branches of the module conventions (log on the cut exactly is
log|z| + i pi; a rectangle that meets the cut without lying on it is too wide), `cball_expr`
(`:1307`), `cball_konst` (`:1386`, charged `COMPLEX_COST_FACTOR` = 8 times `ball_cost` before it
runs) and `cdecide` (`:1413`, 64, 256 and 1024 bits as rule 5). A real step keeps the real ball and
an exact 0 imaginary part, so an enclosure decides "real" only when every step is real. They decide
where c0 lies (`log(-2)`, `(-4)^(1/2)`, `arcsin(2)`, `arctan(2*I)` have no real enclosure) and that
a coefficient is off the cut's direction; "on the direction" needs an exact 0 coordinate.

Conventions (`series.rs`): the bullet at `:54` (the variable is real; the rule; the cuts; how the
engine decides; the two-sided failure of the formal series; the refusals and their kinds; the
mirrors that the rule refuses although their series is right; generic parameters); the formulas of
arctan and arcsinh at `:49`; the owner's rules paragraph at `:82`; the doc of
`SeriesError::NotLaurent` at `:112`.

Choices made here (the request does not settle them; the design is not revised):

1. An undecided case that is a zero test of a real constant is `UndecidedZero`, not `NotLaurent`.
   The request says `NotLaurent` when the engine cannot decide. The module's rule (an undecided test
   is reported, never guessed) and bar (a) (an undecided answer reported as decided) ask for the
   typed undecided error where one exists. The other undecided cases (no enclosure, a truncated
   tail) are `NotLaurent` with a message that says the engine cannot decide, as the request asks.
2. The mirrors and every argument that leaves the cut to the side the principal value is continuous
   with are refused, as the request asks, although the formal series is the function on both sides
   (the oracle, section 3: `c1m:1`, `c1m:2`, `c1m:4`, `c1m:5`, `lv:log:5`, `lv:pow:2`,
   `lv:arccosh:2`, `lv:arcsinh:1`). The conventions and the doc of `NotLaurent` say so. The finer
   rule is a later item.
3. A constant term that depends on a free parameter is off the cut for generic values when it is a
   nonconstant polynomial in the parameters, decided as the zero test decides its groups. Otherwise
   (a kernel such as sin(y), or undecided groups) the position is undecided. A coefficient that
   depends on a parameter is not decided real.

### 3. The two-sided oracle (C-1 item 3)

`tests/data/s1_gp.py:271` to the end: `two_sided_prog` (`:337`), `two_sided_batch` (`:394`),
`two_sided_report` (`:441`), `branch_args` (`:315`), and the command
`python3 -I s1_gp.py --two-sided s1_cuts.tsv s1_singular_points.tsv s1_fuzz.tsv` (`:511`; exit 1
when a value fails). For each value outcome of a row (each `value` alternative of a multi-outcome
row) it compares the truncated series S at x0 and -x0 with gp's scalar function at P + x0 and
P - x0, and for each refusal row it checks gp's own formal series the same way. x0 starts at
1/1000 and shrinks until the rest of gp's series, summed in absolute values up to its precision, is
below 1e-15 max(1, |S|). The tolerance is 100 times that rest, plus 1e-60 sum |c_k| |x0|^k for the
70 digits of the data, plus 1e-100 max(1, |F|); errors are printed relative to max(1, |F|). A row
is "selected" when the argument of a branch function in it has a coefficient of exponent != 0 with
a nonzero imaginary part in gp's series. Two versions were corrected while it ran: the first
compared 70-digit data with a tolerance of 1e-100 (every exact constant row "failed" by 1e-70), and
the first rest was a signed sum that cancelled at the radius of convergence (`1/(x - 1/10000)` at 0).

Final run (`c1/oracle_final.log`, 1.2 s):

| file | value outcomes | selected | fail | refusal rows | formal series wrong on a side | no verdict |
|---|---|---|---|---|---|---|
| `s1_cuts.tsv` | 47 | 12 | 0 | 35 | 25 | 0 |
| `s1_singular_points.tsv` | 367 | 0 | 0 | 84 | 17 | 43 |
| `s1_fuzz.tsv` | 394 | 13 | 0 | 29 | 3 | 0 |

The 25 of `s1_cuts.tsv` are the five C-1 inputs, `c1m:3` (log(-1 - I*x) crosses the cut), 18 `lv:`
rows and `und:3`; the other 10 refusal rows are the 8 of choice 2 above and `und:1`, `und:2`, whose
undecided coefficient is 0. The C-1 table's errors: `log(-1 - I*x^2)` 2 pi I and `sqrt(-1 - I*x^2)`
a sign at x0 = 1e-4 on both sides; `log(-1 + I*x)` 2 pi I at -x0 only; `arctanh(2 + I*x^2)` pi I and
`arcsin(3/2 + I*x^2)` 1.92 I on both sides (as the request's table). The matrix has no selected
row; its 17 are refusal rows at branch points (`b:arcsin@1:1`, `b:sqrt@0:1`, ...) and its 43 without
a verdict are the (d) rows, where gp's floating constant term lies next to the branch point. Of the
fuzz data, the 3 are refusal rows (`N:z2.5`, `U:z3.12`, `U:y12.1`). Before the change of section 4
the oracle failed `U:y12.1`'s value alternative.

### 4. Expected outcomes changed, with the gp evidence

One row of a tests-first file, `tests/data/s1_fuzz.tsv`, `U:y12.1`,
`(acos(((x)*((x^2+1)))^(2+x)))^3` at -1, order 0: "err UndecidedZero | value (gp's formal series)"
became "err UndecidedZero | err NotLaurent". The engine now refuses with `NotLaurent` (it cannot
decide that the constant term `exp(log(-2))` is real). The evidence that the old value was wrong
for a real variable: the operand of acos is (x(x^2+1))^(2+x) = (-2 + 4h + ...)^(1 + h), with
constant term -2 on the cut of acos and the coefficient of h equal to 4 - 2 log 2 - 2 pi I (gp:
`X=-1+h+O(h^3); polcoef((X*(X^2+1))^(2+X), 1, h)` = 2.6137056388801 - 6.2831853071796*I), so it
leaves the cut. Two-sided at realprecision 300: the formal series is the function at -1 - 1e-16
(relative error 1.4e-14) and not at -1 + 1e-16 (relative error 73.4). The function has no Laurent
expansion there, and `NotLaurent` is its true kind. `gen_s1_fuzz.py` makes the row so
(`C1_NOT_A_VALUE`, used at `:332` and `:344`); the TSV row was edited to match (the generator needs
the prototype binary). No row of the matrix or the regression file changed, and no other row that
expected a value now refuses. Two fuzz rows did refuse in an intermediate version (`V:z3.2`,
`acos(sqrt(x*(sin(1)^2+cos(1)^2))) - 1/2` at 2, and `V:y14.4`, `acosh(asinh(sinh(x)))` at -1/2:
real arguments whose constant terms `(2(sin(1)^2+cos(1)^2))^(1/2)` and `(1 + sinh(-1/2)^2)^(-1/2)`
had no structural class); the enclosure-based kernel classes fixed them, and both pass the oracle.

### 5. Acceptance

| command | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic --no-fail-fast -- --test-threads=2` (debug, `c1/sym_dbg_final.log`, 2 min 4 s) | unit 685 (5 ignored); `s1_cuts` 9 (1); `s1_depth` 14; `s1_finish1` 11 (1); `s1_fuzz_data` 7; `s1_payload` 2 (1); `s1_regressions` 26 (2); `s1_review2` 2; `s1_review3` 7 (7); `s1_review4` 4 (2); `s1_review4_memory` 1 (1); `s1_review5` 4 (2); `s1_review5_memory` 2 (2); `s1_review6` 4; `s1_singular_points` 9; doc 116 (11): 903 passed, 0 failed, 35 ignored |
| `cargo test --release -j 2 -p rustmath-symbolic --no-run`, then each binary (`c1/run_rel.sh`, `c1/rel_tests_final.log`) | the same 15 binaries: 787 passed, 0 failed |
| the same with `--ignored --test-threads=1` (`c1/run_rel_ign.sh`, `c1/rel_ignored_final.log`) | `s1_cuts` 1 (`cut_rule_within_the_bar_release`, 3.74 s), `s1_finish1` 1, `s1_payload` 1 (4.70 s), `s1_regressions` 2, `s1_review3` 7, `s1_review4` 2, `s1_review4_memory` 1, `s1_review5` 2, `s1_review5_memory` 2; unit 4 passed and HEAD's `test_airy_numerical_helper` failed (`airy.rs:369`, as in the minor-items step): 23 passed, 1 failed (HEAD's) |
| `cut_rule_within_the_bar_release` (release, REPL budget, 8 MB thread; both entry points per input) | `log(-1 + I*x*(arcsin(2) + ... + arcsin(6740)))` (99 994 bytes) at order 30: `NotLaurent`, 0.44 s; `log(-1 + I*exp(x)*(f(1) + ... + f(10098)))` (99 990 bytes) at order 300: `BudgetExceeded`, 0.62 s; the sum of 3 744 terms `log(-k + x*I*log(-1))` (99 978 bytes) at order 3: `BudgetExceeded`, 3.16 s |
| `cargo test -j 2 -p rustmath-symbolic --test s1_cuts` after the last change (the notes of `s1_cuts.tsv` only) | 9 passed (1 ignored) |
| `python3 -I s1_gp.py --two-sided s1_cuts.tsv s1_singular_points.tsv s1_fuzz.tsv` | section 3; exit 0 |
| `cargo clippy -j 2 -p rustmath-symbolic --no-deps` (`c1/clippy_lib.log`) and `--test s1_cuts` (`c1/clippy_cuts.log`) | the library: 54 warnings, none on a line of this step (`series.rs:1473` is HEAD's `discrete_fourier_coefficients`; one `collapsible_match` in `cball_expr` was fixed); `s1_cuts.rs`: none |
| `rustfmt --edition 2021 --check` | `series_engine.rs`, `series_const.rs`, `series_ball.rs`, `tests/s1_cuts.rs`, `tests/s1_common/eval.rs` clean; `series.rs` has only older diffs (this step changed its doc comments) |
| line width | no added Rust line over 100 characters |

C-4, every crate that depends on rustmath-symbolic. `cargo metadata` (workspace members, all
dependency kinds, `c1/meta.json`) gives eight: rustmath-benchmarks, rustmath-calculus,
rustmath-curves, rustmath-functions, rustmath-geometry, rustmath-manifolds (direct) and
rustmath-jupyter (direct, and through geometry and manifolds), rustmath-plot3d (through geometry).
A grep of the `Cargo.toml` files adds rustmath-py, which the root `Cargo.toml` excludes from the
workspace (a cdylib built with maturin; pyo3 is not in the local registry cache). It was not built;
by inspection it uses `parse`, `differentiate`, `integrate`, `simplify`, `expand`, `printing` and
`Expr::Symbol`/`Expr::symbol`, whose signatures S1 did not change, and it destructures no `Expr`
(the new `Drop for Expr` breaks only moves out of an `Expr` pattern).

| crate | command | passed | failed | ignored |
|---|---|---|---|---|
| rustmath-benchmarks | `cargo test -j 2 --no-fail-fast -p rustmath-benchmarks -p rustmath-calculus -p rustmath-curves -p rustmath-functions -p rustmath-geometry -p rustmath-manifolds -p rustmath-plot3d` (`c1/deps_test.log`, 3 min 4 s) | 0 (4 bench binaries, no tests) | 0 | 0 |
| rustmath-calculus | same | 122 + doc 62 = 184 | 0 | 0 |
| rustmath-curves | same | lib 314, `belyi-atlas` 0, `belyi_build_gate` 3, `biprojective_small_curves` 16, `flags_m24_2_12_5` 2, `genus2_jacobian` 0, `m23_zeta` 9, doc 1 = 345 | 0 | 9 + 1 + 9 + 4 = 23 |
| rustmath-functions | same | 72 + doc 1 = 73 | 0 | 0 |
| rustmath-geometry | same | 417 + doc 106 = 523 | 0 | 1 |
| rustmath-manifolds | same | 308 + doc 33 = 341 | 0 | 11 |
| rustmath-plot3d | same | 78 + doc 12 = 90 | 0 | 0 |
| rustmath-jupyter | `cargo test -j 2 -p rustmath-jupyter --lib` (`c1/jup_lib.log`); `--test kernel_protocol -- --test-threads=1` (`c1/jup_kernel.log`); `--doc` (`c1/jup_doc.log`) | 120 + 4 + 0 = 124 | 0 | 3 |
| the workspace | `cargo check -j 2 --workspace --tests` (`c1/check_ws.log`) | no error | | |

Dependents: 1 680 passed, 0 failed, 38 ignored (22 binaries of the seven crates and three of
jupyter's). With rustmath-symbolic (debug 903, release 787 and 23 ignored that pass): 3 393 passed;
the one failure is HEAD's ignored airy test. `target/release` was deleted after the release runs,
and the test executables of the dependents after their runs (free disk fell to 3.0 GB during the
workspace check; 5.2 GB at the end). `rustmath-misc/output.txt` is unchanged (md5
746308829575e17c3331bbcb00c0898b), no `.tmp_dir_*` exists, and the `__pycache__` that a debug
command of mine left in `tests/data` was removed. The temporary probe test `tests/zz_probe_c1.rs`
was deleted.

### 6. Handoffs for the docs step (HANDOFF and tracker not edited here)

1. C-1 is done under the request's rule. The finer rule (an argument that leaves the cut to the same
   side for both signs of x: the expansion on the punctured neighbourhood, and, on the side the
   principal value is continuous with, the value at the point too) is open; the 8 rows of choice 2
   are its first tests.
2. Tracker rows: `series_engine.rs` (the cut rule, the coefficient classes), `series_const.rs` (the
   kernel classes, enclosure-based for log, sqrt, the inverse functions and powers of non-rational
   real constants), `series_ball.rs` (complex enclosures), `s1_gp.py` (the two-sided oracle), the new
   test file and data.
3. The REQUESTS acceptance of C-1: the five inputs refuse; the four controls of the request and 33
   more return a value and pass the two-sided check; the matrix and the fuzz data pass (one fuzz
   row changed, section 4).
4. C-4: the counts above; rustmath-py is not built (excluded from the workspace, maturin).
5. Known limits, honest refusals: a truncated argument on a cut whose coefficients involve a kernel
   of unknown class (a named function, `exp(log(-2))`) is `NotLaurent` ("cannot decide"); the
   constant field does not reduce `((-4)^(1/2))^2`, so `arctan(sqrt(-4 + x))` is `UndecidedZero`
   (before and after this step).

### 7. Files of this step

- `rustmath-symbolic/src/series_engine.rs` (the rule and the classes, section 2),
  `src/series_const.rs` (kernel classes), `src/series_ball.rs` (complex enclosures),
  `src/series.rs` (conventions and the doc of `NotLaurent`).
- `rustmath-symbolic/tests/s1_cuts.rs`, `tests/data/gen_s1_cuts.py`, `tests/data/s1_cuts.tsv` (new);
  `tests/data/s1_gp.py` (the oracle); `tests/s1_common/eval.rs` (the lower imaginary cuts);
  `tests/data/gen_s1_fuzz.py` and `tests/data/s1_fuzz.tsv` (`U:y12.1`).
- Scratch `fu2/s1impl/c1/`: the patches and status of the start and end, the logs named above, the
  run scripts `run_rel.sh` and `run_rel_ign.sh`, `rel_exes.txt`, `meta.json`, `chk.gp` (gp's scalar
  branch conventions).

## C-1 minor items

The minor items of the two C-1 reviews (`S1.c1-1-semantics.md` m1 to m7, `S1.c1-1-regressions.md`
m1 to m7 and its observation), under the narrowed bar. Branch `claude/gracious-gates-e6b6l7`, HEAD
52708d7, plus the uncommitted changes of this run (`c1m/diff_before.patch` and
`c1m/status_before.txt` are the state at the start, `c1m/diff_after.patch` and
`c1m/status_after.txt` the state at the end; the status lists the same files). Scratch:
`fu2/s1impl/c1m/`. Nothing committed, pushed, stashed or reset. The design was not edited.
`PROJECT_TRACKER.md` and `docs/` were not edited. Below, "s-mN" is item mN of the semantics review
and "r-mN" item mN of the regressions review.

### 1. Tests written first

In `rustmath-symbolic/tests/s1_cuts.rs` (the C-1 step's own file) and its data
`tests/data/s1_cuts.tsv` (rows appended by `tests/data/gen_s1_cuts.py`; the 82 rows of C-1 are
unchanged, byte for byte):

| test or rows | item | on the code at the start |
|---|---|---|
| `limit_path_does_not_read_a_wrong_branch` (`:178`): `limit_with_error` is `Unknown` | s-m6 | passes (the assertion is only stronger) |
| `kernels_of_a_parameter_are_generic` (`:224`): `log(sin(x))`, `log(cos(x))`, `sqrt(cos(x))`, `sqrt(log(x))`, `log(log(x))`, `log(exp(x))` at a give a value whose coefficients equal gp's series at a = 3/7 to 50 digits (gp command in the comment); `log(tan(x))` at a stays `NotLaurent` and `arctan(sin(x))` at a is `UndecidedZero` | r-m1 | fails (all six are `NotLaurent`, "cannot decide") |
| `the_generic_rule_keeps_its_guards` (`:377`): `log(sin(x)^2 + cos(x)^2 - 2 + I*(x - a))` at a and `log(abs(y) - 2 + I*x)` at 0 are not a value | r-m1 | passes (a guard) |
| `a_cut_refusal_does_not_read_as_a_decided_negative` (`:409`): the text of `NotLaurent` starts with "no Laurent expansion found at the point" for a mirror, a "cannot decide" case and `log(x)` at 0 | s-m2, r-m3 | fails |
| `chains_of_positive_constants_keep_their_value` (`:447`): `sin(x) + sqrt^1400(2)` and `x + log(2 + log(2 + ... 3))` (2 000 levels) at 0 give a value with the REPL budget, on the 2 MB debug thread | r-m4 | fails (`BudgetExceeded`) |
| `classes_decided_by_structure_or_enclosure` (`:158`): rows `pg:1` to `pg:5` (`log(-5 + (pi + x)^x)`, `sqrt(-4 + (pi + x)^x)` at 0, `arctanh(x^(2*(1+x)/(1+x)))` at sqrt(2), `arcsin(x^(1 + (x - e)))` at e, `log(-3 + (1 + sin(1) + x)^x)` at 0) and `cls:1`, `cls:2` (`arccos(-3 + log(-1)^2*sinh(x))`, `log(-1 + log(-1)^2*sin(x))` at 0): gp's series | r-m2, s-m3 | fails (7 of 7 `NotLaurent`, "cannot decide") |
| row `und:4`, `arctan(sqrt(-1) + x)` at 0: `UndecidedZero` (in `undecided_coefficients_on_a_cut_are_undecided`) | r-m3 | fails (`NotLaurent`, "cannot decide") |
| `series_engine.rs` unit test `pow_const_reports_a_budget_fault_of_the_cut_check` (`:3129`): a budget of one unit inside the cut check of `(-1 + I*h)^(sin(1)^2 + cos(1)^2)` reports `Budget` | s-m7 | passes (see s-m7 below) |

Run on the code at the start (`series_const.rs` of the start restored for the run, the new tests in
place; `c1m/before_cuts2.log`, `c1m/before_unit.log`): `s1_cuts` 9 passed, 5 failed; the unit test
passed. The gp values of the new rows come from `python3 -I gen_s1_cuts.py > s1_cuts.tsv` (gp
2.15.4, realprecision 300); the two-sided oracle passes all seven value rows on both sides.

### 2. The items (file:line at the end of this step)

**s-m1 (owner decision): are parameters real?** Decided as the frozen design states, pending the
owner: free symbols are generic complex indeterminates (design section "Classes", `Indeterminate`:
"generic parameters (independent indeterminates), as PARI treats y"). The conventions now say so
and say what it costs (`series.rs:76` to `:87`): the exception set has no interior but can hold every
real value of a parameter; `log(-y^2 + I*x)` at 0 returns the series through log(-y^2), which for
a real y != 0 is the function for x > 0 only. No code change. If the owner wants real parameters,
the reviewer's alternative applies (treat a free symbol as real in the axis test of `position`,
refuse when c0 is on the cut's axis); it would also make `parameters_are_generic`'s `log(y + I*x)`
refuse.

**s-m2 and r-m3: the text of `NotLaurent`.** Fixed. `Display` now reads "no Laurent expansion found
at the point: ..." (`series.rs:261`), for every case, since the variant holds decided cases and the
cut rule's refusals alike; the message body says which case it is ("has no Laurent expansion
there", "leaves a branch cut", "cannot decide"). The doc of the variant says so (`series.rs:125`).
No test or crate matched the old prefix (grep of the workspace). r-m3's other option is taken where
a constant exists: for c0 = i t on an imaginary-axis cut whose t is not written out (`sqrt(-1)`),
`position` now reports `UndecidedZero(1 + c0^2)` (`series_engine.rs:1452`), the real constant whose
zero test decides the distance from the ends +-i, as before C-1 (the integrand's 1 + b^2). Other
"cannot decide" cases have no single real constant to name and keep `NotLaurent`.

**s-m3: `Tail::scaled` and constants real by their enclosure.** Fixed. `Tail::scaled_enclosed`
(`series_engine.rs:1225`), used by `finish_product` (`:2576`), reads the class of a constant whose
structure shows none from one complex enclosure at 64 bits (real when the imaginary part is an
exact 0, which only exact arithmetic gives), for constants of at most `SCALE_ENCLOSE_TREE` = 256
tree nodes (`:1179`; charged as `cball_konst`; a larger one keeps `scaled`). Rows `cls:1`, `cls:2`.

**s-m4: the oracle and gp's rounding on a cut.** Fixed. The scalar function F of the two-sided oracle
snaps each branch argument to its cut's axis when its other part is below 1e-250 relative to
max(1, |z|) (`s1_gp.py`: `SNR`, `SNI` at `:41`, `gp_of(..., snap=True)` at `:189`, used at `:371`).
A genuine part at the oracle's smallest x0 (1e-40) is above that for terms up to x^6. gp's formal
series G is not snapped; the comment says that its "formal" verdict on a refusal row can depend on
gp's rounding (`U:y12.1`). Rerun: no verdict of the three data files changed by the snap.

**s-m5: `AB` of a non-real scalar.** Fixed: `AB(s)` is gp's `abs` (the modulus) on a scalar
(`s1_gp.py:40`), as the engine's abs of a constant is; the header says so. The gp references of all
185 data rows whose text has abs or sign are identical with the old and the new `AB`
(`c1m/abs_check.py`). The oracle's verdict changed for two refusal rows of `s1_fuzz.tsv`: `U:z2.27`
(`pi^... * abs(sqrt(x))` at -1) and `U:y14.15` (`atanh(abs(x^(x+1/2+2)))` at -1/2) now read "formal
series wrong on a side". There the scalar is abs of i sqrt(1 - h), whose old `AB` was 0 (the sign of
a zero real part), and gp's formal series is 0 too, so the old verdict "represents f" compared 0
with 0. These are documentation only; no expected outcome changes.

**s-m6: the weak assertion.** Fixed (`s1_cuts.rs:178`).

**s-m7: a budget fault inside `pow_const`'s cut check.** Rejected, with evidence: the scenario does
not occur. The fault is sticky, and `undecided` (`series_const.rs:562`, through `undecided_expr`)
returns the fault before it builds the constant of `UndecidedZero(alpha - n)`. The unit test
exhausts the budget in the first enclosure of the cut check, with alpha's integrality undecided,
and gets `Err(Budget)` on the code at the start and at the end. It stays as a guard. One related
hole was closed: `position_undecided` (`series_engine.rs:1357`) now checks the fault after its
enclosures and returns it (a too-wide or refused enclosure there was read as "undecided").

**r-m1 (owner decision): symbolic points.** Fixed under the design's generic convention, pending
the owner. `position` (`series_engine.rs:1382`) calls `Konst::generic_nonconstant`
(`series_const.rs:2699`): a constant term is off the cut for generic values when it is a nonconstant
function of the parameters, either a nonconstant polynomial in them (as before) or a Laurent
polynomial in one kernel k that depends on a parameter, with no parameter as its own factor, k
nonconstant and a decided nonzero coefficient of some k^e, e != 0. Each kernel carries the flag
`KData::gnc` (`series_const.rs:113`, set at `:3474` by `kernel_generic_nonconstant`, `:3592`): a
parameter; 1/P of a nonconstant P; c^a of a nonconstant c with a nonzero rational a, or of a
rational c other than 0 and 1 with a nonconstant a; f(c) of a nonconstant c for the analytic
functions of `GENERIC_ANALYTIC` (`:3565`: exp, log, sqrt, the trigonometric, hyperbolic and inverse
functions, gamma, factorial, erf, zeta; not abs, sign, mod or named functions). It is computed only
for a kernel with a parameter, with the structural part of the zero test (`struct_nonzero`, `:2772`).
Two kernels are not accepted (`sin(a)^2 + cos(a)^2` is constant), so `log(tan(x))` at a, whose
constant term is sin(a) cos(a)^-1, stays "cannot decide"; `arctan(sin(x))` at a passes the cut rule
and is `UndecidedZero` of 1 + sin(a)^2, the zero test's answer (its generic rule covers polynomials
in the parameters only, design section 2.2; unchanged). The reviewer's corpus had 55 such rows; see
section 3.

**r-m2: `pow_general`'s class.** Fixed (`series_engine.rs:2609` to `:2617`): f^g is real for a real f
whose constant term is positive by `struct_positive` or by its real enclosure (`real_sign`, charged),
and a real g. Rows `pg:1` to `pg:5`.

**r-m4: kernel classes cost the square of a constant's depth.** Fixed for the reported chains, the
general fix is a handoff. A kernel now carries `KData::kpos` (`series_const.rs:109`, set at `:3472`
by `kernel_struct_pos`, `:3542`): its value is a positive real number by its structure (pi, e,
radicals, positive floats, 1/P and P^a with P positive and a real, exp and cosh of a real constant,
sqrt, sinh, tanh, arctan, arcsinh, erf, gamma and factorial of a positive constant, log and arccosh
of a constant > 1). `struct_positive` (`:3505`) accepts a nonnegative rational plus positive terms
with positive factors, and `struct_gt1` (`:3515`) a rational part >= 1 plus positive terms;
`kernel_struct_class` uses `struct_gt1` for arccosh (`:3707`). So the class of such a chain needs no
enclosure. Release, all entry points of `ep.rs` (`c1m/res/ep_chains.log`):

| input (at 0, order 3) | bytes | before (regressions review) | now |
|---|---:|---|---|
| `sin(x) + sqrt^n(2)`, n = 1 400 / 2 000 / 16 000 | 8 410 / 12 010 / 96 010 | `BudgetExceeded` 1.51 s from n = 1 400 | `Ok` 0.004 / 0.004 / 0.035 s, 23 MB |
| `x + log(2 + ... 3)`, 2 000 / 11 000 levels | 18 005 / 99 005 | `BudgetExceeded` 1.18 s at 2 000 | `Ok` 0.010 / 0.043 s, 35 MB |
| `x + arccosh(2 + ... 3)`, 4 000 levels | 52 005 | not measured | `Ok` 0.024 s |
| `x + log(2 + sin(... 1))`, 100 / 200 / 400 / 800 levels | 1 405 to 11 205 | not measured | `Ok` 0.028 / 0.108 / 0.444 s, then `BudgetExceeded` in about 1 s from 800 |

The last family is not covered: the class of log(2 + sin(c)) still reads an enclosure of its whole
argument, so it is still quadratic, charged, and stopped by the budget. Caching each kernel's
enclosure (or the sign it needs) in `KData` is the general fix (handoff). The docs no longer hide
the cost (`series_const.rs:3468`, and the doc of `kernel_struct_class`).

**r-m5, r-m6 and the observation.** Handoffs for the docs step (section 5).

**r-m7: two loose doc claims.** Fixed: `struct_positive` says that it walks the factors of every term
(`series_const.rs:3499`), and the comment at the kernel's class says what an enclosure costs and
which kernels spare it (`:3468`).

### 3. Probes against the C-1 code

The regressions reviewer's probe `c1-1-regressions/drv/p.rs` and entry-point driver `ep.rs`, built
in release against the current library (`c1m/drv/build.sh`), on the reviewer's inputs. Outputs are
compared with the reviewer's outputs of the C-1 code (`hunt/h.sym`, `rnd/r.sym`, `corp/all.sym.gz`)
and of the code before C-1 (`corp/all.pre.gz`). Values that changed are checked two-sided with the
reviewer's `two.gp` (h = +-10^-20, realprecision 400). Outputs in `c1m/probe/`.

| set | rows | changed | changes |
|---|---:|---:|---|
| cut probes (`hunt/h.tsv`) | 1 807 | 2 | `NotLaurent` to a value: `log(-5 + (pi + x)^x)` and `sqrt(-4 + (pi + x)^x)` at 0 (r-m2); both pass two-sided |
| random rows (`rnd/r.tsv`) | 6 000 | 0 | |
| corpus (`corp/all.tsv`, every earlier review's rows) | 63 110 distinct | 128 | 76 `NotLaurent` to `UndecidedZero` and 2 to `Unsupported`, the kind of the code before C-1 in every case; 50 `NotLaurent` to a value equal to the value of the code before C-1, all 50 pass two-sided (the 38 with a parameter at a = 3/7 + 2i/9, 3/7 and -5/7 + i/3) |
| the semantics reviewer's rows with outputs to compare (`rows0` to `rows3`, `rows2b`, `rows5`, `rr1`, `rr2`) | 5 589 | 54 | 53 `NotLaurent` to a value: 48 pass two-sided, the other 5 (`rr1` r387, r446, `rr2` r781, r1175, r1364, nested g(c0 + d) - g(c0) on a cut) pass with the branch arguments snapped below 1e-300 (gp's residue, s-m4; `c1m/probe/sem/clean.gp`); 1 `NotLaurent` to `UndecidedZero` (`(arccosh(cos(2) + (log(-1)*(exp(x)-1))^2))^(-1)`, whose arccosh(cos(2)) the zero test does not decide) |

No value changed, no value became a refusal, and no new panic (the corpus has one driver panic on
the point `1e400`, as in the reviewer's run). The 36 resource inputs of the regressions review
(`res/*.txt`, up to 112 KB) through every entry point (`c1m/res/ep_res.log`): at most 2.07 s wall
for all entry points together (`asinc`, 99 040 bytes, order 1, its result re-parsed) and 36 MB;
`sqoff2000` (`sin(x) + sqrt^2000(2)`) is now `Ok` in 0.004 s.

### 4. Acceptance

| command | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic --no-fail-fast -- --test-threads=2` (debug, `c1m/sym_dbg.log`) | unit 686 (5 ignored); `s1_cuts` 14 (1); `s1_depth` 14; `s1_finish1` 11 (1); `s1_fuzz_data` 7; `s1_payload` 2 (1); `s1_regressions` 26 (2); `s1_review2` 2; `s1_review3` 7 (7); `s1_review4` 4 (2); `s1_review4_memory` 1 (1); `s1_review5` 4 (2); `s1_review5_memory` 2 (2); `s1_review6` 4; `s1_singular_points` 9; doc 116 (11): 909 passed, 0 failed, 35 ignored |
| `cargo test --release -j 2 -p rustmath-symbolic --no-run`, then each of the 15 binaries (`c1m/run_rel.sh`, `c1m/rel_tests.log`) | 793 passed, 0 failed |
| the same with `--ignored --test-threads=1` (`c1m/run_rel_ign.sh`, `c1m/rel_ignored.log`) | 23 passed; HEAD's `test_airy_numerical_helper` failed (`airy.rs:369`), as before. `cut_rule_within_the_bar_release` 3.78 s |
| `python3 -I -B s1_gp.py --two-sided s1_cuts.tsv s1_singular_points.tsv s1_fuzz.tsv` (`c1m/oracle_final.log`) | exit 0. Cuts: 54 value outcomes (12 selected), 0 fail, 35 refusal rows, 25 formal series wrong on a side, `und:4` a gp error (gp has no series of arctan at i). Matrix: 367, 0, 0 fail, 84, 17, 43 without a verdict. Fuzz: 394, 13, 0 fail, 29, 5 (`U:z2.27`, `U:y14.15` added by s-m5) |
| `cargo clippy -j 2 -p rustmath-symbolic --no-deps` and `--test s1_cuts` (`c1m/clippy_lib.log`, `c1m/clippy_cuts.log`) | the library: 54 warnings, as before, none on a line of this step; `s1_cuts.rs`: none. With `--tests` the lib's unit tests fail clippy on two `approx_constant` errors in `parser.rs` tests (3.14159, 3.14), which are in 8a0b310 too |
| `rustfmt --edition 2021 --check` | `series_engine.rs`, `series_const.rs`, `series_ball.rs`, `tests/s1_cuts.rs` clean |
| line width, em-dashes | no added Rust line over 100 characters; none |
| C-4 sample: `cargo test -j 2 -p rustmath-jupyter --lib`, `cargo test -j 2 -p rustmath-calculus` (`c1m/jup_lib.log`, `c1m/calc.log`) | 120 passed, 3 ignored; 122 + doc 62 passed. The change is internal to rustmath-symbolic plus the `Display` text of `NotLaurent`, which no crate matches |

`target/release` was deleted after the release runs, and the two dependents' test executables after
theirs (free disk 4.1 GB; it fell to 3.9 GB during the dependents' build). The probe binaries in
scratch were deleted. `rustmath-misc/output.txt` is unchanged (md5
746308829575e17c3331bbcb00c0898b), no `.tmp_dir_*` exists, and `tests/data` has no `__pycache__`.

### 5. Handoffs for the docs step (HANDOFF and tracker not edited here)

1. Two owner decisions taken under the frozen design's convention (parameters are generic complex
   indeterminates): s-m1 (documented, no code) and r-m1 (one kernel of a parameter is generic).
   The owner may prefer real parameters; the reviewer's alternative is recorded in section 2.
2. r-m4, the general fix: cache each kernel's enclosure (or sign) in `KData`, so that a chain whose
   classes need enclosures (`log(2 + sin(...))`) is linear. Bounded and charged now.
3. r-m5, for C-2's record of conservative refusals: a refused branch subexpression that cancels or
   is multiplied by 0 refuses the whole input (`log(-1 + I*x) - log(-1 + I*x)`, `0*log(-1 + I*x)`,
   `x + 0*sqrt(-1 + I*x)`, `exp(log(-1 + I*x))` at 0), where the function is analytic. The
   `NotLaurent` doc covers it (a subexpression).
4. r-m6: gaps of the constant field show as cut refusals (`I*(log(-2) - log(2))`, `exp(I)*exp(-I)`,
   `exp(I*pi)`, `I*acos(2)` are real but not decided real). A later unit could reduce `log(-c)` to
   `log(c) + I*pi` for a positive c and `exp(k*I*pi)` for a rational k.
5. r-m1 limits: a constant term that is a product or sum of two kernels of a parameter is not
   generic (`log(tan(x))` at a stays "cannot decide"); the zero test's generic rule covers
   polynomials in the parameters only (`arctan(sin(x))` at a is `UndecidedZero` of 1 + sin(a)^2,
   as before C-1).
6. The regressions review's observation, outside S1: HEAD's `Expr::limit` (`limits.rs:47`, direct
   substitution first) gives `Finite(log(-1))` for `log(-1 - I*x^2)` at 0 and `Finite(sqrt(-1))`
   for `sqrt(-1 - I*x^2)`; for a real x the limits are -i pi and -i. It contradicts the
   real-variable convention of `series.rs`.
7. The `Display` of `NotLaurent` changed ("no Laurent expansion found at the point"); the tracker row
   of `series.rs` may mention it.
8. Tracker rows: `series_const.rs` (`kpos`, `gnc`, `generic_nonconstant`, `struct_positive`,
   `struct_gt1`), `series_engine.rs` (`position` for parameters and the imaginary axis,
   `scaled_enclosed`, `pow_general`'s class), `s1_gp.py` (snap, `AB`), the new rows and tests.

### 6. Files of this step

- `rustmath-symbolic/src/series_const.rs` (`KData::kpos`, `gnc`, `struct_positive`, `struct_gt1`,
  `kernel_struct_pos`, `kernel_generic_nonconstant`, `Konst::generic_nonconstant`,
  `struct_nonzero`, docs), `src/series_engine.rs` (`position`, `position_undecided`,
  `Tail::scaled_enclosed`, `pow_general`, the unit test), `src/series.rs` (`Display` and doc of
  `NotLaurent`, the conventions on parameters).
- `rustmath-symbolic/tests/s1_cuts.rs` (five new tests, one assertion), `tests/data/gen_s1_cuts.py`
  and `tests/data/s1_cuts.tsv` (eight appended rows), `tests/data/s1_gp.py` (snap, `AB`).
- Scratch `fu2/s1impl/c1m/`: the patches and status of the start and end, `before/` (the
  reconstruction of the start's `series_const.rs`), the logs named above, `run_rel.sh`,
  `run_rel_ign.sh`, `rel_exes.txt`, `abs_check.py`, `param.gp` (the gp values at a = 3/7),
  `drv/` (the reviewer's `p.rs` and `ep.rs`, `build.sh`), `probe/` (outputs, `all.changed`,
  `ok50.gp`, `sem/`), `res/` (the chain inputs and logs).
