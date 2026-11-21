# S1 tests written before the fix: report (round 3, after the second independent check)

Branch `claude/gracious-gates-e6b6l7`, HEAD 8a0b310 (nothing committed). Governing record:
`fu2/s1impl/owner_decision.md` (design frozen, no revision 9; B1, B2 and the red team's three holes
become regression tests first; a singular-point matrix next to review 9's random harness; the two
behaviour rules). No fix was written. No file under `src/` changed. The second independent check is
`fu2/s1impl/testcheck.md` (approved, no blocker, optional minors n1 to n9); the first is
`testcheck.round1.md`. Earlier rounds of this report are kept as `tests_report.round1.md` and
`tests_report.round2.md`.

## Round 3: what changed after the second check

Every minor was taken. n9 was taken in a changed form (below and finding 7).

| item | change |
|---|---|
| n1 | The intended refusals are listed in full, in the matrix doc comment (`s1_singular_points.rs`, "Intended refusals") and below. I ran HEAD's `taylor` on all 316 refusal rows of the matrix (a scratch binary linked against HEAD, `fu2/s1impl/r3/headp.rs`): it returns x on 39 of them, the 23 `b:*:sub` rows and the 16 rows `a:*:1+0x` and `a:*:7+0x` of the eight families whose f(c) is refused (arctan@+-I, arctanh@+-1, log@0, pow(-1/2)@0, pow(-1/2+I)@0 with `DivisionByZero`, powI@0 with `NotLaurent`). The 12 rows the checker named are among them; the other 4 and 3 `sub` rows come from the new families (n2). On the 84 refusal rows of the fuzz data, HEAD's value is right on two more of the same class (a constant factor 0 in front of a refused subexpression): `N:y12.39` (0, `NotLaurent`) and `U:z4.49` (0, `Unsupported`). The `B2_ROWS` doc comment points to the list. |
| n2 | Three new matrix families with a complex constant exponent at 0: `u^I`, `u^(1/2+I)`, `u^(-1/2+I)` (81 rows). The exact zero gives `NotLaurent`, the value 0 and `DivisionByZero` (design §2.5 `zero_pow_sign`: Re r = 0 with a decided imaginary part, Re r > 0, Re r < 0); a base of positive valuation is `NotLaurent` (§2.4 "non-integer powers": `x^(1/2 + I)` at 0); the neighbours 1/2 and -1/2 are against gp (the principal power, checked: gp's `(-1/2+h)^I` has c0 = exp(-pi - I log 2)). The coverage test allows `NotLaurent` in (a) for the family `powI@0` only. |
| n3 | Three more exact-constant shapes in every family, appended so that no earlier id changes: abs of a rational series `c*abs(-x^0)` (§2.4: abs of a rational series of even valuation is sign(c0) b), a power with an exact-constant exponent `c*x^(x-x)` at 0 (§2.5: an exact-constant exponent takes the constant-exponent rule, and §2.4 `pow_int` rule 1, n = 0, gives the exact 1, although the base x has valuation 1), and a nested inverse function `c + arcsin(x-x)` (the owner's rule: arcsin of the exact 0 is 0); for c = 0, `abs(x^0 - 1)`, `x^(x-x) - 1`, `arcsin(x-x)`. 69 rows over the 23 families. proto13 fails them as the checker found: `DivisionByZero("0 to a negative power")` (B1) for the first two shapes and `PrecisionExhausted(131)` for the nested one. |
| n4 | `rt4_series_h2_dense_buffer_units` sweeps six more recurrences divided by x^(2^k), k = 57 to 62: `(1)/(1-x)*(1)/(1+x)` (a product, from the design's sweep `EXP/rev7rt4/tools/rt4_extra.sh`), `(1+x)^100` and `(1+x)^1000` (Miller), `(2+x+sin(1)^2+cos(1)^2-1)^1000` (the design's sweep), `(sin(1)^2+cos(1)^2-1+x)^100` and `(x+sin(1)^2+cos(1)^2)^100` (the leads that §4.1 `large_power_symbolic_lead` gives for the binomial path). §2.4 `pow_int` rule 7 never returns an exact power for n > 64, so even `(1+x)^100` fills 2^k + 1 slots. proto13: `BudgetExceeded` on each, under 3 ms in release, the limit path `Unknown`. |
| n5 | `rt_structure_h2_consumers_release` times every public entry point per input, each call on its own 8 MB thread against 20 s with VmHWM after it (new helper `timed_release`, also used by `rt_series_h1_at_taylor_order_release`): for F_10 and F_11, `taylor_with_budget` with its consumers, `series_expansion` (`Ok`) and the limit path (`Unknown`); for F_12, `taylor_with_budget` and `series_expansion` (`LimitExceeded(TreeSize)`) and the limit path (`Unknown`). |
| n6 | The 7 E rows are in the data as class `E` with gp on the simplified input (in the note): `z3.22`, `z4.37`, `z4.58` (0), `z4.41` (2 + h/2), `z4.53` (5 - pi/2 + 26h), `y11.24` ((1 + sin 1)^pi - 2/9) and `y14.58` (1). New test `values_where_gp_fails_on_the_shape`. The limit column is `Finite(0)`, `Finite(2)` or `Unknown`; it accepts `Unknown` too for `z4.58` and `y14.58`, whose constant terms are sin(0) and cosh(0) (an optional fold). `z1.9` (log(log(K)), design §7 item 10) stays out. proto13 passes the 7 rows. |
| n7 | The 11 U `UndecidedZero` rows accept gp's value too (`err UndecidedZero \| value ...`, with a limit per outcome by the V rule: `Finite(q),Unknown` for a rational constant term, else `Unknown`). Each undecided constant is nonzero. `R:y11.25` accepts `NotLaurent` too (acos at -1 behind pi*exp(-log pi)). The same holds for `R:z2.56`, which the checker did not name: acos of sin(1)^2 + cos(1)^2 = 1 is 0, so the input raises the exact 0 to the power -1, and the true outcome is `DivisionByZero`; it now accepts `UndecidedZero` or `DivisionByZero`. The generator writes these from a new table `ALSO_ACCEPTED` (R rows) and a rule (U rows). |
| n8 | `R:y13.13` accepts `Unsupported` or `NotLaurent`. `R:y11.42` has two causes too (an undecided integrand factor 1 - log(-1/2)^2 on the left, and `((2x)^(-1))^x` at 0, a h log h term, on the right) and accepts `UndecidedZero` or `NotLaurent`. |
| n9 | New test `rt_series_h1_one_sided_limit`: `limit_enhanced` from the left of `x/x + x^10*exp((sin(x^2)-x^2)/x^9)` at 0 is `Unknown` or `Infinity` (the true limit is +infinity), never `Finite`; from the right `Unknown` or `Finite(1)`. A new wrapper `lim_dir` in `api.rs`. The red team's literal input does not work for this test (finding 7). With `x/x +`, HEAD's direct substitution is 0/0, so `limit_enhanced` reaches `limit_via_series`. The test fails on the prototype of revision 7rt2, which had the hole (`Finite(1)` from the left), and passes on proto13. |

Validation of round 3 (details in "Validation"): every value outcome of the new and changed rows
agrees with an independent check (the checker's `numcheck.py`, gp's `derivnum` of the scalar
function at 300 digits, no gp series code): 293 of 293 matrix value outcomes (largest difference
2.1e-64) and 17 of 18 fuzz value outcomes of the E and U rows (largest 2.9e-70); on the whole
fuzz file 394 of 395. The exception is `U:y12.1`, where `derivnum` takes the other side of a cut: acos's argument is
-2 - 2(log 2 + I pi) h + ..., which crosses the cut of acos at h = 0, so the real-variable function
is discontinuous at the point. The data value 14.660... - 36.709...*I is acos(-2)^3, the branch
whose value at the point is principal, which design §2.4 "Branch at the point" prescribes, and gp's
series gives it (gp's scalar `acos(-2)` = 3.1416 - 1.3170*I). `eval.rs` gives the same value for
the kernel forms a port may render (`arccos(exp(log(-2)))^3` and `arccos(-2)^3`; MPC's residue
of exp(log(-2)) is +1.8e-308*I, the principal side, as gp's). Both generators still reproduce
their files byte for byte.

## Round 2: what changed after the first check

| item | change |
|---|---|
| T1 (blocker) | `U:y14.33` and `U:y13.19` are now hand-reviewed B1 value rows (`B1:y14.33`, `B1:y13.19`) with gp's series and the limit path `Unknown`. Their values equal the checker's gp values to 60 digits (atan(atanh(3)) = 1.3649587741200140004... - 0.6785135872606646808...*I; (1/2)^(asin(2*asin(Pi))) = 0.0015586880147174909... + 0.6140204543302058253...*I). The generator's B detection now also flags an `UndecidedZero` of an integrand factor (`(1 + ...)` or `(-1 + ...)`) in an input with an atan-family function. Every flagged row must be reviewed by hand: in `OVERRIDES`, or in the new `INTEGRAND_NOT_CONSTANT` list (non-constant operand, the integrand is needed, the class stays). An unreviewed candidate stops the generator. The detection flags 10 rows: the two above and 8 with a non-constant operand (`y11.4`, `y11.25`, `y11.42`, `y12.1`, `y13.27`, `y14.10`, `y14.52`, `z4.5`, each with its reason). |
| T2 (blocker) | The scope of the owner's first rule is stated once, in `tests/s1_common/mod.rs` ("Scope of the owner's first rule"), and below. Review 9's scope: mandatory for the six inverse functions (and the Pow, log and inv rules the design already states for exact constants); optional for exp, sin, cos, tan, sinh, cosh, tanh, erf, log of a nonzero constant, and abs and sign of a non-rational exact constant. A row whose outcome depends on an optional fold accepts every honest outcome. New row kind: `check::Outcome` and `check_outcomes` (Rust rows), and in the data files `value ... \| err Kind` with one limit per outcome (`RowWant::OneOf`). `y13.36` accepts `Unsupported` or the value -2/3 (limit `Finite(-2/3)` or `Unknown`: with the fold the constant term may stay the kernel `(-2/3)/sign(arcsin(1))`, and proto13 gives `Unknown` there). `z2.10` accepts `Unsupported` or gp's value 1/sqrt(10) = 0.3162277660168379332.... `atanh(exp(x-x))` moved out of `B2_ROWS` into `b2_argument_with_an_optional_fold`: `PrecisionExhausted` or `DivisionByZero`. The two rows the generator had excluded for the same reason are now included: `O:y11.18` (`PrecisionExhausted` or asinh(pi/2), gp on `asinh(asin(1))` since gp segfaults on the input) and `O:y13.0` (`DivisionByZero` or `PrecisionExhausted`). These four data rows form the new class `O` and the test `rows_with_an_optional_fold`. |
| T2, found while fixing | Accepting `DivisionByZero` for `atanh(exp(x-x))` lost that row's catch of B2 (proto13 gives `DivisionByZero` there). New test `b2_truncated_argument_at_a_pole_is_undecided`: `atanh(1+(sin(x)^2+cos(x)^2-1))` at 0, a truncated `1 + O(h^P)` with no optional fold, expects `PrecisionExhausted` (proto13: `DivisionByZero("arctanh at 1 or -1")`; revision 8: `PrecisionExhausted(131)`; gp: "impossible inverse in div_ser: O(h^41)"). `B2_ROWS` also gained `atanh(1+x^100)` (`NotLaurent`; proto13 the B2 error, revision 8 `NotLaurent`). |
| T2, the same class in the limit column | Ten V rows have a constant term that is a rational q (gp, to the 70 printed digits), which the design's constant field keeps as a kernel, so the prototype's limit path says `Unknown` (`V:z2.57`, 2^(2^x) at 2: exp(4 log 2) = 16; `V:z2.46`, `V:z4.51`: sin(1)^2 + cos(1)^2). A port that renders q decides the true limit. These rows now accept `Finite(q),Unknown` (`LimWant::AnyOf`). The generator does this when gp's constant term equals a rational of denominator at most 10^6 to 10^-68 and its imaginary part is below 10^-250 (gp's rounding noise in `V:z2.32`). |
| m1 | Round 4's two holes are now regression tests, from design §4.1: `rt4_series_h1_exact_sentinel_values` (the values 1, 1, the control 1, `1/x^9223372036854775807 + 1`, and the row at infinity; a new sparse value check `check_value_sparse`, since `check_value` loops over every exponent), `rt4_series_h1_exact_sentinel_refusals` (the 7 refusal rows), `rt4_series_h2_dense_buffer_units` (the 6 text rows, 10 infinite recurrences divided by x^(2^k) for k = 57 to 62, and `exp(x)` through `series_expansion` at max_exponent 2^57 - 1, 2^60 - 1, 2^63 - 2), `rt4_series_h2_dense_buffer_without_a_budget` (`Unsupported` above 2^57 slots). gp cannot expand these ("overflow in valp()"); the references are gp's exponent sums, given in the comments. |
| m2 | `rt_structure_h2_consumers_release` requires `Ok` for F_10 and F_11 and `LimitExceeded(TreeSize)` for F_12 only (`Ok` for F_12 fails). |
| m3 | F_3 at 1, order 1 is compared with gp as log(c0) = 2040852384673.4210091860408671914... and c1/c0 = 82510257166906.907711695695352522..., to 60 digits. New `eval::eval_scaled` keeps a value as m * exp(t), so exp(2.04e12) is represented without changing MPFR's exponent range. gp command in the test comment. |
| m4 | New test `rt_structure_h2_consumers_on_the_debug_stack`: F_10 and F_11 at 1, order 1, then `simplify`, `substitute`, `==`, `Display` and `Drop` of each result, on the 2 MB debug thread. Not ignored: on proto13 it takes 0.26 s and 0.58 s in debug. |
| m5 | Each input of the release tests is timed on its own against 20 s (F_10, F_11, F_12 separately). The order-20000 input is timed through `taylor_with_budget`, `series_expansion` and the limit path separately. The doc comments say that VmHWM is the peak of the whole process, so the check is stricter than the bar's 512 MB per input. |
| m6 | The matrix has two new families, the power with a variable exponent at its branch point 0 (design §2.5): `u^(x+1)` and `u^(2+sin(x))`, 48 rows: exact-zero bases as values 0 (`(x-x)^(x+1)`, `(0*x)^(2+sin(x))`, ...), non-constant bases as `NotLaurent` (`x^(x+1)`, `sin(x)^(2+sin(x))`, ...), and the neighbours 1/2 and -1/2 against gp. The checker's example `sin(x)^(2+x)` is covered by `sin(x)^(x+1)` and `sin(x)^(2+sin(x))`. |
| m7 | The header of `gen_s1_fuzz.py` now says what it does: the class comes from the two outcomes, the generator does not compare the prototype's value with gp's (the tests do), and the R and U kinds are the prototype's kinds, reviewed by hand. The `s1_fuzz_data.rs` header says the same. |
| m8 | `eval.rs` states the convention the tests fix for a kernel on a branch cut: gp 2.15's scalar functions, which are the logarithmic formulas with the principal log and square root (arcsin z = -I log(I z + sqrt(1 - z^2)), arccos z = pi/2 - arcsin z, arctanh z = (log(1 + z) - log(1 - z))/2, not (1/2) log((1 + z)/(1 - z))), with gp's values. Finding 4 below says what the port must do. The design is frozen, so this is not a design edit. |
| m9 | Noted below ("Intended refusals") and in the doc comments of `B2_ROWS` and of the matrix's (b) rows. |
| m10 | Noted below and in the `Cargo.toml` comment. |

Validation after the fix-up, on the prototypes (details in "Validation"): every new or changed test
type-checks and runs against proto13; the new value outcomes are right; the new round-4 rows fail on
revision 8's prototype (which had the round-4 holes) and pass on proto13; the new B2 rows fail on
proto13 and pass on revision 8.

## Scope of the owner's first rule (stated once; `tests/s1_common/mod.rs` has the same text)

The owner's text is about the inverse functions ("for each inverse function and each of its branch
points"; "that keeps the inverse functions everywhere they are regular"). The tests take review 9's
scope (B1, required change 1) with the frozen design (§2.4, §2.5):

- Mandatory. An argument of arcsin, arccos, arctan, arcsinh, arccosh or arctanh whose series is an
  exact constant c gives `apply(f, c)` before any integrand: the value f(c) (a rational, or a kernel
  when c is not rational), or `DivisionByZero` at a pole of the constant (arctanh(+-1), arctan(+-I)).
  This holds at regular points too (`sqrt(arcsin(x - x))` is 0). The exact-constant rules the design
  already has hold too: the exponent and base rules of `Pow` (`(1/2)^(sign(x^2+1))` is the exact
  1/2), and log and inv of the exact 0.
- Exact constants come from the design's exact rules: `x^0`, a monomial divided by itself, sign and
  abs of a series with rational coefficients, a constant factor 0, exact polynomial arithmetic, a
  power with an exact-constant exponent, and the six functions of an exact constant. `x/x` at a
  nonzero point is not exact (review 9 m3).
- Optional. The same evaluation for exp, sin, cos, tan, sinh, cosh, tanh, erf, log of a nonzero
  constant, and abs and sign of a non-rational exact constant (§2.4 keeps those `Unsupported`).
  Where a row depends on such a fold, it accepts each honest outcome.

So the rows pass under review 9's scope and under the general reading of the rule (every function
of an exact constant evaluated). They fail only on a wrong value, a wrong refusal kind, a panic or a
budget hole. Rows elsewhere still pin the frozen design's constant field K: for example
`R:y11.25` (`acos` of an argument whose constant term is -pi*exp(-log(pi)) = -1) is `UndecidedZero`
because K cannot simplify exp(-log(pi)) to 1/pi. A port that extends K beyond the design would decide
such rows, and their expectations would need gp's reference then. Round 3 (n7) makes the rows
of the random harness that pin K accept the true outcome as well: `R:y11.25` (`NotLaurent`),
`R:z2.56` (`DivisionByZero`) and the 11 U `UndecidedZero` rows (gp's value).

## Files

In the repository (`/home/user/RustMath/rustmath-symbolic/`):

| file | content |
|---|---|
| `Cargo.toml` | feature `s1-engine-tests = ["dep:rustmath-complex"]` and the optional dependency `rustmath-complex`. The comment now says that `--all-features` does not compile the three test targets until the engine lands (m10). `Cargo.lock` gains the one dependency line. |
| `tests/s1_regressions.rs` | B1, B2 (review 9's tables, the controls, m2, m3, the optional-fold row, the truncated row), the owner's rule at regular points, `limit_with_error`, the three red-team round-3 holes (with the one-sided limit, round 3), the two round-4 holes, the debug-stack consumers, two `#[ignore]` release budget tests (`timed_release`). 28 tests (26 plus 2 ignored). |
| `tests/s1_singular_points.rs` | the singular-point matrix driver, 4 tests over `tests/data/s1_singular_points.tsv` (609 rows: (a) 379, (b) 184, (c) 46; 23 (function, branch point) pairs), with the list of intended refusals. |
| `tests/s1_fuzz_data.rs` | review 9's random harness as fixed data, 7 tests over `tests/data/s1_fuzz.tsv` (479 rows). |
| `tests/s1_common/mod.rs`, `api.rs` | the scope statement; `api.rs` is the only file that names the entry points (the design's, and HEAD's `limit_enhanced` as `lim_dir`, round 3). |
| `tests/s1_common/check.rs` | outcome checks: a value, a refusal kind, several accepted outcomes (`Outcome`, `check_outcomes`, `RowWant::OneOf`), a limit expectation that may list several results (`LimWant::AnyOf`, `limit_matches`), the sparse value check (`check_value_sparse`); through `series_expansion` and `taylor_with_budget` with the REPL budget (1e8) and the limit path; DAG-safe tree and DAG counts; VmHWM; the data-file reader. Each row runs on a thread with the bar's stack (2 MB debug, 8 MB release). |
| `tests/s1_common/eval.rs` | MPC evaluation (1024 bits) of the engine's coefficients, the branch-cut convention (m8), and `eval_scaled` (m3); comparison to 60 digits, scaled by max(1, abs(gp)). |
| `tests/data/s1_gp.py` | gp references (unchanged). |
| `tests/data/gen_s1_singular.py` | the matrix generator: the families of round 2, the three complex-exponent families and the three producer shapes of round 3. |
| `tests/data/gen_s1_fuzz.py` | the fuzz generator: corrected header, extended B detection, `INTEGRAND_NOT_CONSTANT`, the O rows, the `Finite(q),Unknown` limit; round 3: the E rows in `OVERRIDES`, `ALSO_ACCEPTED`, the value outcome of the U `UndecidedZero` rows (`value_limit`). |
| `tests/data/s1_singular_points.tsv`, `s1_fuzz.tsv` | the data (106 KB, 128 KB). Both generators reproduce their files byte for byte (checked after the change: `python3 -I gen_s1_singular.py \| cmp - s1_singular_points.tsv`, `python3 -I gen_s1_fuzz.py 2>/dev/null \| cmp - s1_fuzz.tsv`). Round 3 changed no earlier matrix row (the 468 rows of round 2 are byte-equal) and, in the fuzz data, only the 15 rows of n7 and n8; it added 141 matrix rows and 7 fuzz rows. |

In the scratchpad (`fu2/s1impl/`): `protoharness/` (runs the tests against a prototype; round 3
added `lim_dir` to its shims `api_proto.rs` and `api_proto10.rs`), `fixcheck/` (a scratch test of
the round-2 value outcomes), `s1_fuzz.log` (the generator's stderr: every candidate row, the kept
B1 candidates), `s1_fuzz.tsv.before` and `s1_singular_points.tsv.before` (the round-1 data),
`rlibs_complex_rel_new.txt`, and `r3/` (round 3: `headp.rs`, HEAD's taylor on a file of rows, with
`head_ref.txt` and `fz_head_ref.txt`, its output on the refusal rows; `limp2.rs`, HEAD's limit
strategies; `pp.py`, a prototype probe; `vt/`, the evaluator probe; `sp_round2.tsv` and
`fz_round2.tsv`, the round-2 data; the harness outputs `*_proto13.out`).

## How the tests are kept out of the build

Unchanged. Each test file starts with `#![cfg(feature = "s1-engine-tests")]`. Without the feature
the three test targets compile as empty crates. `cargo test -j 2 -p rustmath-symbolic` passes: 656
unit tests, 116 doc tests, the three new targets with 0 tests. With the feature, `cargo check -j 2 -p
rustmath-symbolic --features s1-engine-tests --tests` fails with exactly the missing API: E0432
`ExpansionPoint`, `SeriesError`, `SeriesExpansion`, `series::StructureLimit`; E0599
`series_expansion`, `taylor_with_budget`; no other error and no warning in the test files. While
the gate exists, `cargo test`, `check` or `clippy` with `--all-features` on rustmath-symbolic fails
the same way (m10); CI does not use the flag.

## How to enable them

1. Implement the design's API (§2.9, §2.13) with the owner's two rules, in the scope above. The
   wrappers in `tests/s1_common/api.rs` assume HEAD's signatures for `limit_via_series(&x, &a, 5)`,
   `limit_with_error(&x, &a, Direction::Both, order)` and `limit_enhanced(&x, &a, dir)` (`lim_dir`,
   round 3), and the design's `taylor_with_budget(var,
   point, order: usize, budget: u64)`, `series_expansion(var, &ExpansionPoint, max_exponent: i64,
   Option<u64>)` and the public field `SeriesExpansion::terms`.
2. `cargo test -j 2 -p rustmath-symbolic --features s1-engine-tests`.
3. Release budget tests: `cargo test --release -p rustmath-symbolic --features s1-engine-tests --test
   s1_regressions -- --ignored --test-threads=1`.
4. When the engine lands: move `rustmath-complex` to `[dev-dependencies]`, delete the feature and the
   three `#![cfg(...)]` lines.

## Validation against the prototypes (no implementation exists)

`fu2/s1impl/protoharness/run.sh TESTFILE [debug|release] [-- args]` (run it with `bash`) compiles
the test files with `rustc --test` against the prototype (`PROTO=rev8/proto12` selects revision 8,
`PROTO=rev7rt2/proto10 SHIM=api_proto10.rs` revision 7rt2). The release run needs `cargo build
--release -j 2 -p rustmath-symbolic -p rustmath-complex` first; in round 2 I rebuilt it (the rlib
paths equal `rlibs_complex_rel.txt`), ran, and deleted `target/release` afterwards.

On proto13 (revision 7rt4, the frozen design's prototype), round 3, debug (round 2 gave the same
in release):

| file | passes | fails (all expected) |
|---|---|---|
| `s1_regressions` | 20 (and the 2 release tests, run in debug with `--ignored`) | 6: `b1_exact_constant_operand_is_evaluated` (10 of 10 `DivisionByZero("0 to a negative power")`), `b1_m2_arctan_at_plus_minus_i_is_a_pole` (4 of 4 `Ok(arctan(I) ..)`), `b1_exact_constant_operand_at_a_regular_point` (`PrecisionExhausted`, `DivisionByZero`), `b2_branch_point_of_a_non_constant_argument` (5 of 5 `DivisionByZero("arctanh at 1 or -1")`), `b2_truncated_argument_at_a_pole_is_undecided` (the same error), `limit_with_error_follows_the_expansion` |
| `s1_singular_points` | 2 (coverage; the 46 regular points) | 2: (a) 135 of 379 rows (B1 in every exact shape, m2, and the nested shape's `PrecisionExhausted`), (b) 16 of 184 (every arctanh row, B2). Of the 141 new rows, 37 fail, each of the B1 class: 16 `DivisionByZero("0 to a negative power")` on `c*abs(-x^0)` and `c*x^(x-x)` at arcsin, arccos, arccosh +-1 and arcsinh +-I, and 21 `PrecisionExhausted(131)` on the nested shape `c + arcsin(x-x)` (every family but arctanh at +-1, where proto13's B2 error is the expected kind). The other rows of the complex-exponent families pass. |
| `s1_fuzz_data` | 5 (360 V, 61 R, 25 U, 7 E rows, row count) | 2: 19 of 22 hand-reviewed rows (as in round 2) and 1 of 4 O rows (`O:y13.36`) |

New tests that pass on proto13: `rt_series_h1_one_sided_limit`, `values_where_gp_fails_on_the_shape`,
the extended `rt4_series_h2_dense_buffer_units` (36 more rows), and the rewritten release test,
run in debug through `--ignored` to check its logic (each call under 1 s in debug: F_10 taylor with
consumers 0.33 s, series_expansion 0.03 s, limit path 0.91 s; F_11 0.70, 0.06 and 0.95 s; F_12
0.12, 0.12 and 0.99 s). I did not rebuild `target/release` this round; the release timings of
round 2 (F_10 0.009 s + 0.139 s, F_11 0.041 s + 0.277 s, F_12 refused) are the last release run.

On the prototype of revision 7rt2 (`PROTO=rev7rt2/proto10`, which had series H1):
`rt_series_h1_one_sided_limit` fails with "from the left: expected Unknown or Infinity, got
Finite(1)", with every other `rt_series_h1_*` test.

(Round 2.) The value outcomes of the changed rows, checked against proto13 on the same constants written
without x (`fu2/s1impl/fixcheck/tests/s1_valcheck.rs`, which reads the rows from the data file):
`atan(atanh(3))` at -1/2, `(1/2)^(asin(2*asin(pi)))` at -1 (order 3), `((1/3)/x + (x+1/2))/sign(asin(1))`
at -1/2 (the kernel `sign(arcsin(1))`; limit `Unknown`, accepted), `(x(2x+1))^(-1/2)*sign(1/(sin(1)^2+cos(1)^2))`
at 2 and `asinh(asin(1)) + 0*x` at -1: all five agree with the data at 60 digits, through
`series_expansion`, `taylor_with_budget` and the limit path. So the data, the evaluator's branch
sides and the sign kernel are consistent.

## The cases

gp is PARI/GP 2.15.4 at realprecision 300. `data/s1_gp.py series(text, point, order)` reproduces each
reference; the test comments give the gp command and output.

### 1. Review 9, B1 and B2 (`s1_regressions.rs`)

B1, value expected (review 9's ten rows; limit path in brackets): unchanged from round 1
(`arcsin(x^0)` pi/2 [Unknown], `arccos(x^0)` 0 [Finite(0)], `x + 0*arcsin(x^0)` x, `exp(x) +
0*arccos(x^0)`, `arccos(sign(x^2+1)) + x`, `x + arcsin(1 + 0*x)`, `arcsin((x-1)/(x-1)) + x` at 1,
`arccosh(x^0)` and `arccosh(-x^0)` at 2, `arcsinh(I*x^0)`). The controls, m2 and m3 are unchanged.
The owner's rule at regular points is unchanged (the scope makes it mandatory for the six
functions).

B2: `atanh(x)` at 1, `atanh(1+x)` at 0, `x + 0*atanh(1+x)` at 0, `cos(atanh(1-x))` at 2 and the new
`atanh(1+x^100)` at 0: `NotLaurent`. `atanh(exp(x-x))` at 0: `PrecisionExhausted` or
`DivisionByZero` (optional fold of exp). `atanh(1+(sin(x)^2+cos(x)^2-1))` at 0: `PrecisionExhausted`.
Controls `x + atanh(1)`, `atanh(x^0)`: `DivisionByZero`.

`limit_with_error_follows_the_expansion` covers all B1 and B2 rows, the optional-fold and the
truncated row included.

### 2. The red team's holes (`s1_regressions.rs`)

Red team round 3: series H1 (the nine rows, at infinity, the limit path, the one-sided limit
through `limit_enhanced` (added in this report's round 3, n9), the order-20000 row, the controls),
structure H1 (8 values, 6 refusals), structure H2 (F_22, F_26, F_30
at order 0; F_2 equal to gp; F_3 now compared with gp as log(c0) and c1/c0; F_12 refused; F_10 and
F_11 with their consumers on the 2 MB debug thread; the release tests as in m2, m5 and n5).

Red team round 4 (this report's round 2, m1): the rows of §4.1 `exact_sentinel` and
`dense_buffer_units` listed in the table above, with the design's expected kinds, on the bar's
stack; this report's round 3 (n4) added the Miller and binomial powers and the product to the
sweep, and (n5) timed every entry point of the release test.

### 3. The singular-point matrix (`s1_singular_points.rs`, 609 rows)

23 (function, branch point) pairs: arcsin, arccos, arccosh, arctanh at +-1; arctan, arcsinh at +-I;
log, sqrt, `u^(1/3)`, `u^(2/3)`, `u^(-1/2)`, `u^pi`, `u^(x+1)`, `u^(2+sin(x))`, `u^I`, `u^(1/2+I)`,
`u^(-1/2+I)` at 0. Each pair has (a) 12 or 13 exact shapes of the constant (abs, an exact-constant
exponent and a nested inverse function among them; plus the sums `x + f(..)`, `x + 0*f(..)` for two
shapes), (b) 7 non-constant shapes plus `x + 0*f(c + x)`, (c) the neighbours c + 1/2 and c - 1/2
against gp. Each row has one expected outcome, as the owner asked: (a) a value (247 rows) or, at a
pole of the constant, `DivisionByZero` (115), and for 0^I, which has no value, `NotLaurent` (17,
the third outcome of an exact-constant argument); (b) `NotLaurent` (184); (c) gp's series (46).

### 4. The random harness as data (`s1_fuzz_data.rs`, 479 rows)

The first 60 inputs of each of review 9's eight runs, replayed and checked against its output files.
Classes: V 360 (gp's series, 60 digits; the limit path `Finite(q)` on 107 rows, `Unknown` on 241,
`Finite(q)` or `Unknown` on 10, not checked on 2), R 61 (NotLaurent 48, DivisionByZero 9, and four
rows with two accepted kinds: UndecidedZero or DivisionByZero 1, UndecidedZero or NotLaurent 2,
Unsupported or NotLaurent 1), U 25 (UndecidedZero or gp's value 11, Unsupported 14), hand-reviewed
22 (14 B1, 6 B2, 2 N), O 4, E 7 (gp on the simplified input). Of the 8 inputs where gp fails and
the prototype gives a value, only `z1.9` is left out (log(log(K)), a kernel at an undecided pole,
design §7 item 10), listed in `s1_fuzz.log`.

## Intended refusals (m9, n1)

These rows expect a refusal where HEAD's taylor returns the correct value. Each has a constant
factor 0 in front of a subexpression the engine refuses: the engine expands each subexpression on
its own, and "a constant factor 0 ... their errors still surface" (design §2.5, §2.9; the owner's
second rule for the NotLaurent ones). It is intended. The final review should not read these rows
as regressions in the sense of S1_BAR (b). I checked HEAD's result on every refusal row of the
matrix (316) and of the fuzz data (84) with a scratch binary linked against HEAD
(`fu2/s1impl/r3/headp.rs`).

- Matrix, 39 rows where HEAD returns x:
  - the 23 rows `b:*:sub`, `x + 0*f(c + x)`, one per family, `NotLaurent`;
  - the 16 rows `a:*:1+0x` (`x + 0*f(c*x^0)`) and `a:*:7+0x` (`x + 0*f(c + 0*x)`) of the eight
    families whose f(c) is refused: arctan@I, arctan@-I, arctanh@1, arctanh@-1, log@0,
    pow(-1/2)@0 and pow(-1/2+I)@0 (`DivisionByZero`), and powI@0 (`NotLaurent`). The checker's 12
    are the first six families; HEAD gives x on `x + 0*arctan(I*x^0)` and `x + 0*arctanh(x^0)`.
- Regression file: `x + 0*atanh(1+x)` in `B2_ROWS` (`NotLaurent`; HEAD x).
- Fuzz data, 2 rows where HEAD returns 0: `N:y12.39` (`3^(acosh(x^2+1))` times an exact zero,
  `NotLaurent`) and `U:z4.49` (`atan(x-x)/(x/x)` times `sign(x^(x+1))` at 1/3, `Unsupported`, sign
  of a series with non-rational coefficients).

The doc comments of `B2_ROWS` and of the matrix say so.

## Findings while writing the tests (for the implementer)

1. The owner's first rule reaches past the branch points, and past the integrand's own failure
   modes. An atan-family function of an exact constant at a regular point must give the exact
   constant (`sqrt(arcsin(x-x))` is `PrecisionExhausted` on proto13), and an atan-family function of
   an exact complex or kernel constant must not test its integrand factor for zero
   (`atan(atanh(sign((x+1/2)^2) + 2))` is `UndecidedZero(1 + arctanh(3)^2)` on proto13, where the
   value is the kernel arctan(arctanh(3))). 15 rows test this class: the 7 of
   `b1_exact_constant_operand_at_a_regular_point` and 8 fuzz B1 rows.
2. gp's acosh of a series is not the principal function left of -1 (unchanged from round 1; the
   references use `log(u + sqrt(u+1)*sqrt(u-1))`).
3. gp's sqrt of a series with an exact complex constant term has 19 digits only; complex points go
   to gp as floats. gp segfaults on `asinh(asin(sign((x+1)^2)^x))` at -1; that row's value
   reference is gp on `asinh(asin(1))`.
4. The side of a kernel on a branch cut is open in the frozen design. The tests fix it to gp's
   scalar functions: arcsin z = -I log(I z + sqrt(1 - z^2)), arccos z = pi/2 - arcsin z, arctanh z =
   (log(1 + z) - log(1 - z))/2, arccosh z = log(z + sqrt(z + 1) sqrt(z - 1)), with the principal log
   and square root (gp: `asin(3/2)` = 1.5707963267948966192 - 0.96242365011920689500*I, `atanh(3/2)`
   = 0.80471895621705018730 - 1.5707963267948966192*I). For arcsin and arccos this is the only side
   consistent with the engine's next coefficient. For arctanh it is a convention: (1/2) log((1 +
   z)/(1 - z)) gives 0.805 + 1.571*I at 3/2. The port must render such a constant with this value
   (as the kernel, or as a formula equal to it), and state the convention in its own module docs
   before the port is reviewed; a different convention changes `c:arcsin@1:0`, `c:arccos@1:0`,
   `c:arctanh@1:0`, `c:arctanh@-1:1` and fuzz rows such as `B1:y14.33`, with their gp references.
   `eval.rs` cites this.
5. The singular-point matrix finds B1 in every exact shape (96 rows of arcsin, arccos, arccosh at +-1
   and arcsinh at +-I) and B2 in every arctanh row.
6. (Round 2.) A row that accepts `DivisionByZero` because of an optional fold no longer tells a
   correct fold from the B2 error. Keep at least one row of each blocker class whose outcome does not
   depend on an optional fold: here `atanh(1+(sin(x)^2+cos(x)^2-1))` for the truncated case of B2.
7. (Round 3, n9; outside S1, for the HANDOFF.) HEAD's one-sided limits of series H1's function are
   wrong in a way S1 cannot fix. `limit_enhanced(x, 0, Left)` of `1 + x^10*exp((sin(x^2)-x^2)/x^9)`
   is `Finite(1)` before any series is computed: its first strategy, direct substitution
   (`limits.rs:487-490`), simplifies `1 + 0^10*exp(0/0)` to 1. `Expr::limit(x, 0, Left)` gives
   `Finite(1)` too, and on `x/x + x^10*exp(..)` `Finite((1 + 0))` (it takes the limit of each
   summand). The true left limit is +infinity (gp: f(-0.1) = 2.4e62). Design §2.13 changes only
   `limit_via_series` and `limit_with_error`, so a test on the literal input would fail on a correct
   port. The S1 test uses `x/x + x^10*exp(..)` through `limit_enhanced`, whose substitution is 0/0, so
   that only its series strategy can decide (HEAD: `Unknown`; revision 7rt2: `Finite(1)`; proto13:
   `Unknown`). The substitution strategy of `limits.rs` (a factor 0 in front of an undefined value
   taken as 0) needs a unit of its own.
8. (Round 3.) A kernel on a branch cut can come from a constant whose exact value is on the cut but
   whose numerical value has a rounding residue off it: `exp(log(-2))` is -2 + 1.8e-308*I in MPC at
   1024 bits (and +3.8e-39*I in gp at 38 digits). `eval.rs` normalizes only exact zero parts, so it
   evaluates `arccos(exp(log(-2)))` on the side of the residue. Here that is the principal side, so
   `U:y12.1` holds for either rendering. A port that renders such constants should prefer the exact
   form (-2; the design's `exp_int_pow` already does so for the enclosures), so that its value does
   not depend on a residue.
