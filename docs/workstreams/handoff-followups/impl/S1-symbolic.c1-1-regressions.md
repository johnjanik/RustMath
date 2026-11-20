# S1 request C-1, review round 1, regressions lens

Reviewed: branch `claude/gracious-gates-e6b6l7`, HEAD 52708d7, with the uncommitted changes of this
run (`git diff` equals `c1/diff_after.patch`). The C-1 change alone is the difference between
`c1/diff_before.patch` applied to HEAD (rebuilt in `c1-1-regressions/pre/`) and the working tree:
`c1-1-regressions/c1_only.patch` (series_engine.rs, series_const.rs, series_ball.rs, series.rs,
s1_gp.py, eval.rs, gen_s1_fuzz.py, s1_fuzz.tsv) plus the new `tests/s1_cuts.rs`,
`tests/data/gen_s1_cuts.py` and `tests/data/s1_cuts.tsv`. Read first: `owner_decision.md`,
`owner_decision2.md`, `S1_impl.md` (all sections, "C-1 and C-4" in detail), `S1.review6-*.md`,
`tests_report.md`, and C-1 to C-4 of `docs/workstreams/m23_golay/REQUESTS.md`.

No repository file was edited. `git status --short` and the md5 of `git diff` are the same before
and after (`c1-1-regressions/status_*.txt`, `diff_md5_*.txt`). `rustmath-misc/output.txt` is
unchanged (md5 746308829575e17c3331bbcb00c0898b), no `.tmp_dir_*` exists, and `tests/data` has no
`__pycache__`. `target/release` was deleted after the release runs. The two test executables of my
C-4 sample were deleted. Free disk was 4.8 GB at the end. Paths below are relative to
`fu2/s1impl/c1-1-regressions/` unless they start with `rustmath-` or `c1/`.

## Verdict

Approve. No blocker under the narrowed bar, and the tests pass in both profiles.

- No wrong value. Every value the engine now returns on my probes passes a two-sided check
  against gp's scalar functions at a + h and a - h: 579 of 581 values of 1 807 cut probes (2 hold
  a parameter), 4 733 values of 6 000 random branch-heavy rows, and 29 817 of 29 953 values of a
  63 864-row corpus made of every earlier review's rows (the other 136 cannot be evaluated by gp,
  section 3).
- No value changed and no new value appeared against the code before C-1. Every changed outcome
  is a value that became a refusal, or a refusal of another kind. Of the values that became
  refusals, 922 of 1 464 were wrong on at least one side (section 4).
- No crash, abort or stack overflow on any probe, in release (8 MB thread) or debug (2 MB thread).
  The resource bar holds on S1's entry points (section 6).
- The five C-1 inputs and their mirrors refuse with the stated message at every order, and the
  limit path is `Unknown` for all ten. The changed expected outcome (`U:y12.1`) is justified with
  gp; I reproduced the evidence. The C-4 counts are true.
- Seven minor items (section 8). The one that matters most is m1: an expansion at a symbolic point
  of a branch function of a kernel (`log(sin(x))` at `a`) now refuses, where HEAD-before-S1 and the
  code before C-1 returned the generic series. It is an honest refusal from the implementer's
  choice 3, of the kind C-2 already records as conservative. It needs the owner's decision.

## 1. Tests

| command | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic --no-fail-fast -- --test-threads=2` (debug, `sym_dbg.json`, `sym_dbg.err`, 2 min 19 s user) | 903 passed, 0 failed, 35 ignored: unit 685 (5); `s1_cuts` 9 (1); `s1_depth` 14; `s1_finish1` 11 (1); `s1_fuzz_data` 7; `s1_payload` 2 (1); `s1_regressions` 26 (2); `s1_review2` 2; `s1_review3` 7 (7); `s1_review4` 4 (2); `s1_review4_memory` 1 (1); `s1_review5` 4 (2); `s1_review5_memory` 2 (2); `s1_review6` 4; `s1_singular_points` 9; doc 116 (11). The report's numbers. |
| `cargo test --release -j 2 -p rustmath-symbolic --no-run`, then each of the 15 binaries (`run_rel.sh`, `rel_tests.log`) | 787 passed, 0 failed |
| the same binaries with `--ignored --test-threads=1` (`run_rel_ign.sh`, `rel_ignored.log`) | 23 passed, 1 failed. The failure is `specialfunctions::airy::tests::test_airy_numerical_helper` (`airy.rs:369`), ignored at 8a0b310 as "just a demonstration" (fin2 reviews). `cut_rule_within_the_bar_release` passed in 3.84 s. |
| `python3 -I s1_gp.py --two-sided s1_cuts.tsv s1_singular_points.tsv s1_fuzz.tsv` (`oracle.log`) | exit 0, the report's table exactly: cuts 47 value outcomes (12 selected), 0 fail, 35 refusal rows, 25 formal series wrong on a side; matrix 367 (0), 0, 84, 17, 43 without a verdict; fuzz 394 (13), 0, 29, 3 |

Tests-first files: the only changed expected outcome is `U:y12.1` of `s1_fuzz.tsv` (section 5).
`tests/s1_common/eval.rs` gains the lower imaginary cut rule for arctan and arcsinh. It touches
only values i t with t < -1 and an exact zero real part, where MPC and gp differ. gp's own values
confirm it: `atan(-3*I)` = -1.5707963 - 0.3465736 i = -atan(3 i), `asinh(-2*I)` = -1.3169579 -
1.5707963 i = -asinh(2 i). No other test file of an earlier step changed (`c1_only.patch`).

## 2. The C-1 inputs and their mirrors

Release probe `drv/p_sym_rel` (`series_expansion` with the REPL budget 1e8, `taylor_with_budget`,
`limit_via_series` at order 5, `limit_with_error`), rows of `hunt/c1.tsv` and `hunt/ord.tsv`:

- All ten rows give `NotLaurent` with "the argument leaves a branch cut at the point" through
  `series_expansion` and `taylor_with_budget`. The limit path is `Unknown`, and `limit_with_error`
  is `Unknown` with no `BigO`. The code before C-1 returned `Ok` for all ten.
- The rule holds at every order I tried: 1, 0, -1 and -3 for the five inputs, `atan(2*I + x)`,
  `acosh(I*x)`, `log(-1 + I/x)` at infinity, `(-1 + I*x)^x`, `x^3*log(-1 + I*x)` and
  `log(-1 + I*x)/x^2`. A truncation below the constant term never skips the check.
- HEAD-before-S1's `taylor` returns the formal series for these inputs. That series is wrong for a
  real variable, so the refusal is no regression.

## 3. Hunting for a value on a cut (bar (a))

The two-sided checker (`two.gp`, `mkgp.py`): for each `Ok` row with terms c_e and precision P it
evaluates S(h) = sum c_e h^e at h = +10^-20 and -10^-20 (h = 1/x at infinity) at realprecision 400,
and compares it with gp's scalar function at a + h. A side passes when |S - F| <= 10^15 h^P
max(1, |F|). An error of order below P is then at least 10^5 times the bound. A row is ONESIDE when
only one side passes. Rows the bound flags get two more checks: the scaling test `scale2.gp`
(e(h)/e(h/10) >= 10^(P-1) at h = 10^-20, or e below 10^-340), and `scale3.gp` (the same at
h = 10^-150, realprecision 1 200) or a rerun at realprecision 3 000.

The checker catches the C-1 failures. On the same rows, the code before C-1 has 72 wrong values in
the random set (3 FAIL, 69 ONESIDE) and 559 in the cut probes.

**Cut probes** (`hunt/gen.py`, 1 807 rows, `hunt/h.tsv`). Constant terms on every open cut of every
branch function, written exactly, through pi and e, decided only numerically (`log(2)-1`, `cos(2)`,
`1-e`, `sin(4)`, `gamma(-1/2)`, `zeta(1/2)`, `2*cos(1/10)`, `cosh(1)`, `I*cosh(1)`, `I*log(4)`) and
complex-written (`I^2`, `(1+I)*(1-I)*(-1)`, `exp(I*pi)`, `sqrt(-4)*sqrt(-1)`, `I*acos(2)`,
`acos(2)^2`, `log(-1)/2`, `(-2)^(1/2)`). Coefficients in the cut's direction and off it, real only
after cancellation (`(1+I)*(1-I)`, `sqrt(-2)*sqrt(-3)`, `sqrt(-4)/sqrt(-1)`, `(sqrt(2)+I)*(sqrt(2)-I)`,
`I*acos(2)`, `exp(I)*exp(-I)`), complex only after cancellation (`x*(1+I) - x`). Powers with
exponents 1/3, pi, I, -1/2, 3/2, sqrt(2), 1/2 + I, x, and disguised integers. Points -1, -pi,
`log(2)-1`, I, 2*I, -I, 1/2, -2, -8, `sqrt(2)`, `exp(I*pi/3)`, `I*pi`, `pi + I/10`, infinity. Nested
functions (`log(log(x))` at 1/2, `log(-1 + I*sqrt(-1 + x))`, `asin(2*cosh(I*x))`,
`atan(2*I*exp(I*x))`, `log(-1 + I*x*tan(I*x))`). Undecided coefficients. Parameters.

Result: 581 values. 551 pass at once. The 28 rows at `1 +- 10^-30` (radius of convergence 10^-30)
pass the scaling test at h = 10^-60. Two rows hold a parameter and were not checked. Debug on a
2 MB thread gives the same outcomes (`hunt/h.sym.dbg`).

**Random rows** (`rnd/gen.py`, seeds 1 to 3, 6 000 rows of depth 1 to 3, cut-sensitive constants,
points 0, 1, -1, I, 1/2, infinity): 4 733 values, all pass. Debug on 2 MB: the same outcomes.

**Corpus** (`corp/all.tsv`): 63 864 distinct rows, from the earlier reviews' `.tsv` files
(rev1 to rev6, fin1, fin2; their fuzzers, matrices and hand rows) and the three data files of the
tests. The engine returns 29 953 values. The first check passes 27 549, flags 2 270 and cannot
evaluate 136. Of the flagged rows, 2 022 pass the scaling test, 162 pass at h = 10^-150 and 83 pass
at realprecision 3 000. All 248 are gp rounding or rows whose radius of convergence is below 10^-20
(the `P` rows of rev3 with offsets of 10^-40 to 10^-60, `(asin(x) - x - x^3/6)/x^5` cancellations).
Three rows (`(exp(x^50)-1-x^50)/x^100` and its two siblings) underflow in gp at every precision. Their
value 1/2 + O(x^3) is right: (e^t - 1 - t)/t^2 = 1/2 + t/6 + ... with t = x^50. The 136 rows gp
cannot evaluate are documented conventions of earlier steps (a division by an undecided zero kept
as a kernel, `x/(sqrt(-1)^2+1)`; `(1 + t)^(1/t)` underflowing in gp; `x^(2^62)/x^(2^62)`). None of
them changed with C-1.

No row gave a value that the two-sided check refutes. No undecided coefficient test decided:
`log(-1 + I*(sin(1)^2+cos(1)^2-1)*x)`, `log(-1 + x*I*(log(6)-log(2)-log(3)))`,
`arctan(2*I + (sin(1)^2 + cos(1)^2 - 1)*x)`, `(-1 + I*x)^(sin(1)^2+cos(1)^2)` and
`(-1 + I*x)^(log(4)/log(2))` are `UndecidedZero`. `log(-1 + (exp(I*pi)+1)*x)` and every row with
`I*acos(2)` or `exp(I)*exp(-I)` as a coefficient refuse.

I read the class code for soundness: `kernel_struct_class` (`series_const.rs:3420`), the
`Tail` rules of sums, products, quotients, powers, variable exponents, `expand_unary` and
`atan_like`, `cut_check` and the shortcut for an argument whose tail class keeps it in the cut's
direction (sound: such an argument moves parallel to the cut, so it never crosses it near the
point, whether or not c0 is on it). I found no unsound class.

## 4. Regressions

### 4.1 Against the code before C-1

`prelib_*`: the pre-C-1 tree (`pre/`, HEAD plus `c1/diff_before.patch`) built with rustc against
cargo's dependency rlibs (`buildpre.sh`). The same probe runs on both libraries (`drv/p_pre_*`).
`classify.py` compares the outcomes and classifies each value that became a refusal by gp's
two-sided verdict on the old value.

| set | rows | values now / before | value changed | new value | value to refusal | old value wrong on a side | old value right |
|---|---:|---:|---:|---:|---:|---:|---:|
| cut probes | 1 807 | 581 / 1 481 | 0 | 0 | 900 | 559 | 338 (+3 with a parameter) |
| random | 6 000 | 4 733 / 4 822 | 0 | 0 | 89 | 72 | 17 |
| corpus | 63 864 | 29 953 / 30 428 | 0 | 0 | 475 | 363 | 57 (+55 with a parameter) |

Refusal to refusal: in the corpus 225 rows went from `UndecidedZero` to `NotLaurent` (70 "leaves",
155 "cannot decide") and 2 from `Unsupported` to `NotLaurent`. I checked a sample of the 70: c0 is
decided on the cut and a coefficient is decided off its direction, for example
`atan(sqrt(-2) + x)` (c0 = i sqrt(2), coefficient 1), so "leaves" is right. Before, the integrand's
`1 + sqrt(-2)^2` was undecided.

The old values that were right and now refuse fall into five groups:

1. The mirrors and every argument that leaves the cut to the side the principal value is
   continuous with (170 cut probes with an even power of x, 30 corpus rows). The request asks for
   this refusal (implementer's choice 2), and "leaves" is a true statement.
2. A refused subexpression that cancels or is multiplied by 0: `log(-1 + I*x) - log(-1 + I*x)`,
   `0*log(-1 + I*x)`, `x + 0*sqrt(-1 + I*x)`, `exp(log(-1 + I*x))`,
   `asin(e + 0*asin(2 + (1+I)*x*x))/...`. The doc of `NotLaurent` covers it (a subexpression).
   This is the class of C-2's 39 matrix rows (minor m5).
3. A real coefficient the engine cannot decide real: `I*(log(-2) - log(2))` (it is -pi),
   `exp(I)*exp(-I)` (1), `I*acos(2)` (-1.3170), `exp(I*pi)`, a truncated argument with a complex
   class whose stored coefficients are real (`log(-3 + exp(I*x) + exp(-I*x))`). These are honest
   "cannot decide" refusals, as C-1 item 1 asks (minor m6).
4. A variable exponent whose base has a constant term that is not a positive rational:
   `log(-5 + (pi + x)^x)` and `sqrt(-4 + (pi + x)^x)` at 0, `atanh(x^(2*(1+x)/(1+x)))` at sqrt(2),
   `asin(x^(1 + (x - e)))` at e (18 corpus rows). Real inputs (minor m2).
5. A parameter in a kernel of the constant term: `log(sin(x))` at a (minor m1).

### 4.2 Against HEAD-before-S1 (8a0b310)

`drv/hd_*` runs HEAD's `taylor` (its parser) through `fin1-resources/headlib_*`.

- On the 166 cut probes whose correct old value is now refused, HEAD's `taylor` returns its
  formal series in at most 0.06 s, with no panic (`hunt/lost.head`). HEAD returns the same formal
  series on the rows that leave the cut, where it is wrong.
- Symbolic points (`par/par.tsv`, 26 rows): HEAD and the code before C-1 return the generic series
  of `log(sin(x))`, `sqrt(cos(x))`, `log(exp(x))`, `sqrt(log(x))`, `log(cos(x))`, `log(tan(x))` and
  `log(log(x))` at `a`. The port now gives `NotLaurent` ("cannot decide"). `log(x)`, `sqrt(x)`,
  `asin(x)`, `atan(x)`, `acosh(x)`, `x^(1/3)`, `x^x`, `log(x^2+1)` at `a`, `log(x)` at `sin(a)`,
  `log(sin(y) + x)` at 0 still give values (minor m1).
- No new crash, abort or stack overflow: the C-1 code adds no recursion (`cball_expr` uses an
  explicit stack, the class functions are not recursive). Deep constants of up to 100 KB run on a
  2 MB debug thread (section 6).
- HEAD's own failures stay failures: `sin(x) + sqrt^100(2)` at 0 needs more than 6 GB in HEAD's
  `taylor` (`res/h_*.tsv`, killed under `ulimit -v 6000000`), and `sqrt^400(2)` exhausted the 16 GB
  machine once (my mistake: I ran it without a cap; nothing else was lost, the corpus runs
  survived).

I count the refusals of section 4.1 as minor, not as bar (b) regressions. They are typed, honest
refusals of the kind the request asks for ("cannot decide" returns `NotLaurent`) or of the kind
C-2 records as conservative (the `x + 0*f(...)` rows, `sqrt(x^4)`). None is a crash, an abort, a
stack overflow or a wrong value. If the owner reads bar (b) as covering honest refusals of inputs
HEAD handled, then m1 and m2 are the items to reconsider.

## 5. The changed expected outcome `U:y12.1`

`(acos(((x)*((x^2+1)))^(2+x)))^3` at -1, order 0: "err UndecidedZero | value (gp's formal
series)" became "err UndecidedZero | err NotLaurent". My gp check (`uy12.gp`, realprecision 100):
the coefficient of h in the operand of acos is 2.6137056388801 - 6.2831853071796 i
(4 - 2 log 2 - 2 pi i, not real), so the operand leaves the cut of acos at -2. With the series
truncated at O(h^4), |S - F|/|F| at h = +10^-12 is 1.857 and at h = -10^-12 it is 1.4e-47. The
formal series is the function on one side only. The change is justified. The generator records it
in `C1_NOT_A_VALUE` (`gen_s1_fuzz.py`), with the evidence.

## 6. Resources (bar (c)) on S1's entry points

Release probe `drv/ep_sym_rel` (8 MB thread, REPL budget; `taylor_with_budget`,
`series_expansion` with `to_expr`, `Display`, `Debug` and the re-parse of the `Display`,
`limit_via_series`, `limit_with_error`), debug `drv/ep_sym_dbg` on a 2 MB thread, rows in `res/`:

| input | bytes | release | debug 2 MB |
|---|---:|---|---|
| `log(-1 + I*x*tan^3000(I))` | 15 013 | `NotLaurent` 0.15 s, 8 MB | 0.43 s |
| `log(-1 + x*tan^20000(I))` | 100 013 | `BudgetExceeded` 0.05 s, 19 MB | 0.22 s |
| `log(sinh^15000(-1) + I*x)` | 90 013 | `UndecidedZero` 0.03 s, 26 MB | 0.23 s |
| `atan(sinh^15000(2*I) + x)` | 90 013 | `UndecidedZero` 0.16 s, 30 MB | 0.52 s |
| `acosh(tanh^15000(1/2) + I*x)` | 90 016 | `NotLaurent` 0.07 s, 26 MB | 0.24 s |
| `(sin^15000(-3) + I*x)^(1/3)` | 75 016 | `NotLaurent` 0.09 s, 26 MB | 0.33 s |
| 1 500 terms `log(tan^10(k+I) + I*x)` | 99 060 | `UndecidedZero` 0.01 s | |
| 2 300 terms `asin(k+1 + sin(k)^2 + cos(k)^2 - 1 + x)`, order 1 / 3 | 99 040 | `Ok` 0.56 s, 468 KB of text, re-parse 0.57 s, 36 MB / `BudgetExceeded` 0.48 s | |
| `sin(x) + sqrt^2000(2)` | 12 010 | `BudgetExceeded` 1.51 s (before C-1: `Ok` 4 ms) | |
| `x + log(2 + log(2 + ... 3))`, 2 000 levels | 18 005 | `BudgetExceeded` 1.18 s (before C-1: `Ok` 7 ms) | |

All are within 20 s and 512 MB, and no stack overflowed. The last two rows are minor m4: the class
of each new kernel reads an enclosure of its whole argument, so a nested constant costs about the
square of its depth (`sin(x) + sqrt^n(2)`: 10, 49, 126, 391 and 945 ms for n = 100, 200, 400, 700,
1 000, then `BudgetExceeded` from n = 1 400). It is charged, so the budget stops it.

## 7. C-4

The report's counts match its logs. Recounted from `c1/deps_test.log`: 1 556 passed, 0 failed,
35 ignored over the seven crates (calculus 184, curves 345 with 23 ignored, functions 73, geometry
523 with 1, manifolds 341 with 11, plot3d 90, benchmarks 0), and jupyter 120 + 4 + 0 with 3
ignored: 1 680 passed, 38 ignored. My sample rerun: `cargo test -j 2 -p rustmath-jupyter --lib`
120 passed, 3 ignored (`jup_lib.log`, 39 s); `cargo test -j 2 -p rustmath-calculus` 122 + doc 62
passed (`calc.log`).

## 8. Minor issues

**m1. Expansions at a symbolic point of a branch function of a kernel now refuse.**
`series_engine.rs:1351` (`position`): a constant term that depends on a parameter is off the cut
only when it is a nonconstant polynomial in the parameters. So `log(sin(x))`, `log(cos(x))`,
`log(tan(x))`, `sqrt(cos(x))`, `sqrt(log(x))`, `log(log(x))` and `log(exp(x))` at `a`, and
`atan(sin(x))` at `a`, are `NotLaurent` ("cannot decide"). HEAD-before-S1 and the code before C-1
return the generic series. The corpus has 55 such rows. The refusal is honest and documented
(choice 3), and it guards against a constant that only looks parametric
(`sin(a)^2 + cos(a)^2 - 2`, which is -1 and on the cut). But the engine already decides
`sin(a) != 0` generically before the cut check, so it treats the same constant generically
elsewhere. A single kernel of a nonconstant parameter argument is nonconstant, so it is
generically off a half-line. The owner should decide whether this coverage loss is acceptable for
the PR (the C-1 semantics review's m1 asks the same question).

**m2. A variable exponent keeps the real class only for a positive rational base.**
`pow_general` (`series_engine.rs:2569`) gives `Tail::REAL` only when the base's constant term is a
positive rational, while `const_base_class` and `kernel_struct_class` accept a structurally
positive constant or a positive enclosure. So the real inputs `log(-5 + (pi + x)^x)`,
`sqrt(-4 + (pi + x)^x)` at 0, `atanh(x^(2*(1+x)/(1+x)))` at sqrt(2) and `asin(x^(1 + (x - e)))` at
e refuse with "cannot decide" (18 corpus rows; HEAD returns their series). Using `struct_positive`
or the enclosure sign of f(a) would keep them.

**m3. `NotLaurent` for an undecided case says "no Laurent expansion at the point".** The display of
`cannot_decide_cut` (`series_engine.rs:1280`) starts with a decided claim and then says that the
engine cannot decide. For m1, m2 and group 3 of section 4.1 the claim is false. The request asks
for `NotLaurent` here, and the doc of `SeriesError::NotLaurent` now includes the undecided case,
so I do not count it as bar (a). It does conflict with fin1-semantics B1, where `NotLaurent` for an
undecided exponent was a bar (a) blocker. 155 corpus rows moved from `UndecidedZero`, which named
the constant, to this message. A wording such as "undecided at the point" would be clearer. Owner's
call.

**m4. Kernel classes cost about the square of a constant's nesting depth** (section 6). Bounded by
the budget, but `sin(x) + sqrt^1400(2)` and 2 000 nested logs now hit the REPL budget in about
1.2 s, where the code before C-1 returned in milliseconds (HEAD cannot expand them at all). Caching
each kernel's enclosure, or the sign it needs, in `KData` would make it linear.

**m5. Add the cut refusals of cancelling subexpressions to the C-2 record**: `log(-1 + I*x) -
log(-1 + I*x)`, `0*log(-1 + I*x)`, `x + 0*sqrt(-1 + I*x)`, `exp(log(-1 + I*x))` at 0 now refuse,
where the function is analytic.

**m6. Gaps of the constant field show as cut refusals**: `I*(log(-2) - log(2))`, `exp(I)*exp(-I)`,
`exp(I*pi)`, `I*acos(2)` are real but not decided real, so `log(-1 + I*(log(-2)-log(2))*x)`,
`log(exp(x))` at `I*pi` and 159 cut probes refuse. Honest. A later unit could reduce `log(-c)` to
`log(c) + I*pi` for a positive c, and `exp(k*I*pi)` for a rational k.

**m7. Two doc claims are loose.** `struct_positive` (`series_const.rs:3391`, doc at `:3388`) says "O(1) for the
shapes it accepts" but walks the factors of the term. The comment "once per kernel: it may read
enclosures" (`:3364`) is true but hides that each read costs the depth of the argument (m4).

**Observation (outside S1).** HEAD's generic `Expr::limit` (`limits.rs:47`, direct substitution
first, unchanged by S1) gives `Finite(log(-1))` for `log(-1 - I*x^2)` at 0 and `Finite(sqrt(-1))`
for `sqrt(-1 - I*x^2)`. For a real x the limits are -i pi and -i. HEAD gives the same
(`drv/lm.rs`). It contradicts the new real-variable convention, so it belongs in the handoff, not in
this unit.

## 9. Files

- Report: this file.
- Scratch `fu2/s1impl/c1-1-regressions/`: `c1_only.patch`, `pre/` (the tree before C-1),
  `buildpre.sh`, `prelib_*`, `rlibs_*.txt`, `rel_exes.txt`, `run_rel*.sh`, the logs of section 1;
  `drv/` (`p.rs` probe, `ep.rs` entry points and resources, `hd.rs` HEAD's taylor, `lm.rs`, build
  script, binaries); `two.gp`, `scale2.gp`, `scale3.gp`, `mkgp.py`, `classify.py`; `hunt/`,
  `rnd/`, `par/`, `res/` (inputs, outputs, gp verdicts, `changed.txt`); `corp/` (`all.tsv`, the
  compressed outputs `all.sym.gz` and `all.pre.gz`, the gp verdicts, `ch.changed`); `uy12.gp`.
