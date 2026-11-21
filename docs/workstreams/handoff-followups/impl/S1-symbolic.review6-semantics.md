# S1 implementation, review 6, lens: numeric and series semantics

Reviewer of the uncommitted S1 change in `/home/user/RustMath` (branch `claude/gracious-gates-e6b6l7`,
HEAD 8a0b310) after fix-up round 5. No repository file was edited: `git status --short` lists the
same 35 entries before and after (`rev6-semantics/status_before.txt`). Scratch:
`fu2/s1impl/rev6-semantics/`. Paths below are relative to
`/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/s1impl/`.

## Verdict

Approved in this lens. No blocker under S1_BAR. The tests pass in debug and in release. The three
blockers of round 5 are fixed; I reproduced their inputs on the built code.

- A fuzzer of my own with a new oracle (contour integrals in gp, section 4) ran 13,200 rows in seven
  classes. It found no wrong coefficient, no value where the function has no Laurent series, no
  decided zero that is not zero, no wrong `Finite` limit, and no panic. Every row the oracle could
  not confirm directly was checked by hand or by a second method; all are gp rounding artifacts or
  points on a branch cut where the engine takes the principal side.
- The owner's two rules hold on all 6,204 rows of class B (each inverse function at each finite
  branch point, the constant written in 59 exact ways, 33 argument shapes): an exact constant gives
  f(c) or, at a pole of f(c), `DivisionByZero`; a non-constant argument gives `NotLaurent`; a
  truncated constant gives `PrecisionExhausted`. No `DivisionByZero` outside exact poles, no value
  for a non-constant argument, no `UndecidedZero` (every writing was decided).
- Three minor issues (section 12): a power x^r with r a truncated constant is `NotLaurent` where
  the exponent is only undecided (m1); the matrix has no truncated-constant row per pair (m2);
  Gaussian rationals are not inverted exactly (m3, honest refusals).

## 1. Tests

| command | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic --no-fail-fast` (debug; `rev6-semantics/debug_tests.log`, 51 s) | unit 685 passed, 4 ignored; `s1_depth` 9; `s1_fuzz_data` 7; `s1_payload` 2 (1 ignored); `s1_regressions` 26 (2 ignored); `s1_review2` 2; `s1_review3` 7 (7 ignored); `s1_review4` 4 (2 ignored); `s1_review4_memory` 1 (1 ignored); `s1_review5` 4 (2 ignored); `s1_review5_memory` 2 (2 ignored); `s1_singular_points` 8; doc 116 (11 ignored); 0 failed |
| the twelve release test binaries (`cargo test --release -j 2 -p rustmath-symbolic --no-run`, every artifact fresh; copies in `rev6-semantics/relbin/`; `rel_tests.log`) | 685 (4 ignored), 9, 7, 2 (1), 26 (2), 2, 7 (7), 4 (2), 1 (1), 4 (2), 2 (2), 8; 0 failed |
| release `--ignored --test-threads=1` (`rel_ignored.log`) | `s1_payload` 1 (3.33 s); `s1_regressions` 2 (0.77 s); `s1_review3` 7 (1.78 s); `s1_review4` 2 (2.27 s); `s1_review4_memory` 1 (0.05 s); `s1_review5` 2 (2.06 s); `s1_review5_memory` 2 (0.05 s); `fold_work_calibration` passed (46.2 s); 0 failed |
| `cargo test -j 2 -p rustmath-jupyter --lib -- taylor simplify_of_a_variable_chain` (`jup_taylor.log`) | 10 passed, among them `test_taylor_variable_named_like_a_constant`, `test_taylor_consumers_of_a_kernel_source_result`, `test_taylor_diff_of_a_chain_result_memory`; 0 failed |

## 2. Tests of the first step

Compared with the harness copy of the first step (`protoharness/work/src`) and with round 4's
untracked files (`fix5/backup/untracked_round4.tgz`, unpacked in `rev6-semantics/r4src/`):

- Byte-identical to the harness: `tests/s1_common/check.rs`, `tests/s1_common/eval.rs`,
  `tests/data/s1_fuzz.tsv`, `tests/data/gen_s1_fuzz.py`, `tests/data/s1_gp.py`.
- `tests/s1_regressions.rs` (mtime 2026-11-15 20:02, unchanged since round 1): without comment
  lines, `#[path]` lines and white space it differs from the harness copy in one character, the
  trailing comma rustfmt added (`Ok(())},);`).
- `tests/data/s1_singular_points.tsv` (772 lines): its first 106,450 bytes are the harness copy,
  its first 140,189 bytes are round 4's file, and it equals `fix5/s1_singular_points.tsv.new`.
  Round 5 appended the 32 (g) rows.
- Changed in round 5 against round 4: `s1_singular_points.rs` (the (g) coverage loops, two
  expectation arms, the test `values_reached_through_folded_constants`; no earlier line changed),
  `gen_s1_singular.py` and the data file. New: `s1_review5.rs`, `s1_review5_memory.rs`. The other
  test files are byte-identical to round 4's.

No expected outcome of the first step changed.

## 3. The blockers of round 5

| item | status | evidence |
|---|---|---|
| semantics B1 (a variable named `pi` or `I` names a constant) | fixed | 24 rows with the variable `pi`, 15 with `I`, 8 with `e` (`hand/vpi.tsv`, `vi.tsv`, `ve.tsv`): every value row passes the contour oracle with the variable printed as `XV`, so a constant written with the variable's name would fail it. The new round-5 values keep their kernels under the shadow (`asin(1/2)*pi` in pi is `arcsin(1/2)*pi`, `acosh(1/2)*pi` is `arccosh(1/2)*pi`), erf's constant is `2*arccos(-1)^(-1/2)`, `erf(pi + acos(-1))` passes. Undecided cases stay refused (`1/(sin(acos(-1)) + pi)`, `tan(acos(0) + pi)`: `UndecidedZero`). With the variable `I`, `acos(1/2)*I` is `(1/3)*pi*I`, right (pi is not shadowed). The REPL test `test_taylor_variable_named_like_a_constant` passes. |
| resources B1 (consumers after `substitute`, `expand`, `differentiate`) | fixed | `rev5-resources/cons/sinchain2535.txt` at orders 1, 10, 190 and `sinmul3_11000.txt` at order 1: `LimitExceeded(TreeText)` in 0.38 s for all four (release, `hand/chain.tsv`). `b1r5_*` pass in both profiles. |
| resources B2 (the memo of `simplify`) | fixed | `b2r5_*` pass in both profiles. A semantic check of the memo cap (section 10): DAGs past the 32 MiB cap give HEAD's structure exactly. |

## 4. My fuzzer: a contour-integral oracle

Driver `rev6-semantics/drv/r6.rs` (release `r6_rel`, debug `r6_dbg`, both linked against the
current rlibs): per row `series_expansion` with the budget 1e8, `taylor_with_budget`,
`limit_via_series(.., 5)`, `series_coefficients`, the gp form of the input and of every result.
Oracle `rev6-semantics/cauchy.gp`, runner `run6.py`.

The oracle is different from review 9's (gp's own series) and review 5's (residuals at real
points). It uses only gp's scalar functions. For an `Ok` row with terms c_k and precision P it
computes the Laurent coefficients by the trapezoidal rule on circles around the point,
a_k = (1/N) sum_j f(a + r w^j) (r w^j)^(-k), N >= 160 points, at 110 digits, on two circles r and
r/2 (r = 1/16, then 1/256 and 1/4096). When both circles agree, f is single-valued and analytic in
the annulus and the a_k are its Laurent coefficients; each engine coefficient must then equal a_k
to 1e-43 of its scale (a test with one wrong coefficient of relative size 1e-20 at k = 9 is
caught). When the circles disagree (a cut through the point), a residual test on rays of the four
half-planes, and on the real line, decides which side the engine expanded. A refused row is probed
the same way: `NotLaurent` must have circles that disagree or an infinite principal part;
`DivisionByZero` must make gp fail or give values that are unstable between 40 and 120 digits.
The taylor rendering is checked against the terms at three points, `Finite` limits against a_0
with no negative term, and `series_coefficients` against the terms. Infinity is h = 1/x.

| class (generator) | rows | result |
|---|---|---|
| B: owner's rules (`gen6.py B`) | 6,204 | section 5 |
| E: functions with zeros, poles and cuts at points written through the round-4 and round-5 values (`asin(1/2)`, `acos(-1/2)`, `3*asin(1/2)`, `-I*acosh(1/2)`, `atanh(I)/I`, `acosh(1/2)`, `2*atanh(I)`, ...; 49 functions) | 1,433 | 1,182 `Ok`: 1,021 pass on circles; 144 one-sided at a cut, each on the principal side; 17 with the argument on a cut at a critical point (`asin(2*sin(x))` at pi/2, `log(2*cos(x))` at pi, `acosh(2*cos(x))` at pi), all equal to the principal function on both real rays (`realray`). 251 refusals: `NotLaurent` 106 (all non-Laurent), `UndecidedZero` 120 (complex kernels, honest), `PrecisionExhausted` 25 |
| Z: exact-zero identities and near misses inside 1/(K+x), sqrt(K+x^2), log(K+x), asin(K+1+x), atanh(K+1+x), exp(1/(K+x)), ... (93 true zeros, 12 near misses of size 1e-50 to 7e-5) | 1,470 | no true zero decided nonzero; 61 decided zero, 32 undecided (`atan(1/2)+atan(1/3)-pi/4`, `log(2)+log(3)-log(6)`, `e-exp(1)`, `sqrt(2)*sqrt(3)-sqrt(6)`, `gamma(5)-24`, ...). The near misses are decided nonzero and right on 154 of 168 rows, checked with circles of radius |K|/32 (the pole is at -K), one-sided residuals scaled to |K|, four rows at 500 digits (gp's 110 digits lose the expanded form), and the eight rows `exp(1/(K+x))` exactly against gp's series of exp(1/(K+x) - 1/K) at 600 digits (`fz/near*.gp`); the other 14 rows are `exp(I*pi)+1+10^-30`, `UndecidedZero` (exp(i pi) is a complex kernel without an enclosure) |
| H: the red-team holes (`gen6.py H`) | 242 | section 8 |
| R: random trees of depth 2 to 4 over 16 functions, kernel constants (pi, I, sqrt(2), e, the new values) and 18 points (complex, through exact values, infinity) (`gen6r.py`, seeds 61 to 64) | 3,000 | 2,400 `Ok`: 2,237 pass on circles, 161 one-sided at a cut (principal side), 2 checked by hand (an order-0 row and a critical point on the real line); refusals: `NotLaurent` 268, `UndecidedZero` 316, `DivisionByZero` 10 (all gp failures: exact zeros), `PrecisionExhausted` 6 |
| the matrix (`tests/data/s1_singular_points.tsv`) | 767 | section 6 |
| hand rows (variable shadow 47, imaginary values 20, Gaussian rationals 14, negative max exponent 9, x^r 20, laurent and limits 20) | 130 | all values right |

The release driver (8 MB thread) and the debug driver (2 MB thread) give the same output on all
of classes B, E, Z, H, R and the matrix (fields 1 to 17, md5 equal; no panic, no stack overflow);
slowest row 0.16 s in release, 0.65 s in debug. The Display of
all 1,182 `Ok` results of class E re-parses to the same values (`fz/rt.gp`).

## 5. The owner's two rules (class B)

Each of arcsin, arccos, arccosh, arctanh at +-1 and arctan, arcsinh at +-I. The constant c is
written 20 ways for 1 (`sin(pi/2)`, `2*sin(pi/6)`, `tan(pi/4)`, `2*cos(acos(1/2))`,
`6*asin(1/2)/pi`, `-I*I`, `sqrt(3)*tan(pi/6)`, `2*asin(x^0)/pi`, `log(exp(1))`,
`cos(acosh(1/2)/I)*2`, `sin(acos(-1/2))*2/sqrt(3)`, `tan(3*atan(1))^2`, `acosh(-1)/(I*pi)`, ...),
15 for -1, 14 for I, 10 for -I. Shapes: 8 exact (`C`, `C*x^0`, `C+0*x`, `C+(x-x)^2`, `C*(x/x)`,
`C+x-x`, `C+sin(x-x)`, `C*cos(0*x)`), 8 truncated constants (`C*(1+x)/(1+x)`,
`C*exp(x)*exp(-x)`, `C+sin(x)^2+cos(x)^2-1`, `C+log(exp(x))-x`, `C+(sin(x)^2+cos(x)^2-1)/x^3`,
...), 17 non-constant (`C+x`, `C-x^2`, `C+x^3`, `C+x^7`, `C+I*x`, `C*cos(x)`, `C/(1-x)`,
`C+tan(x)^2`, `C+(sin(x)^2+cos(x)^2-1)/x^3+x`, ...).

| shape | outcome (all functions, both points, every writing) |
|---|---|
| exact | the value (774 rows; arcsin, arccos, arccosh, arcsinh) or `DivisionByZero` (354 rows; arctanh(+-1), arctan(+-I)); `C+sin(x-x)` and `C*cos(0*x)` are `PrecisionExhausted` (276 rows; truncated by the documented convention) |
| truncated constant | `PrecisionExhausted` (1,520 rows) |
| non-constant | `NotLaurent` (3,196 rows; the oracle confirms each is non-Laurent) |

Values: 654 pass on circles; the other 120 are exact constants where gp's float input sits next to
the branch point (gp's `acos(2*sin(Pi/6))` is 3.2e-58 at 100 digits and 1.1e-154 at 300, the
engine's 0 is exact; `acos(cos(Pi) + x - x)` is pi plus noise). Every `DivisionByZero` row makes gp fail or
diverge (230 and 124 rows).

## 6. The singular-point matrix

All 767 rows through my driver and oracle (`mat/`), release and debug 2 MB (same output): the 367
values pass (309 on circles, 26 on the principal side of a cut, and 32 that gp cannot evaluate
around the point: 31 `sign(x^2+1)` shapes at complex points, and `arccosh(-1 + ((x+1)^2-x^2-2x-1))`,
where gp's float noise of the exact zero sits on the cut; the engine's i pi is exact); the 127
`DivisionByZero` rows are exact zeros by gp (121 fail, 6 diverge); the 271 `NotLaurent` rows are
non-Laurent, except the 23 intended `b:*:sub` rows `x + 0*f(c+x)` (Laurent, `NotLaurent` by the
first step's design, listed in its doc comment); the two (e) guard rows are `UndecidedZero` and
`PrecisionExhausted` as the matrix expects.

Completeness: the 23 (function, branch point) pairs, with exact and non-constant rows each,
infinity per function, the guard rows (e), the inverted values (f) and the folded values (g). That
is what the owner asked for. The truncated constant, the third case the engine meets at a branch
point, has 3 rows (all arcsin, class e): minor m2.

## 7. Exact zeros, Decided versus Unresolved

- Class Z: no constant that is zero was decided nonzero; every undecided one stays
  `UndecidedZero` through `series_expansion`, `laurent`, `series_coefficients` and the limit path.
  The convention for a divisor the input itself names (`x/(exp(1) - e)` is `Ok`, the result holds
  where the input is defined, `series.rs:38`) holds as documented.
- `drv/lsc.rs` (20 rows, `hand/lsc.tsv`): `laurent` and `series_coefficients` give `UndecidedZero`
  (not `ValuationBelowMinimum` or `Pole`) when the low coefficient is undecided
  (`(sin(1)^2+cos(1)^2-1)/x + 1`, `(atan(1/2)+atan(1/3)-pi/4)/x^2 + x`), the value when it is an
  exact zero (`(asin(1/2)-pi/6)/x + 2` is 2, `(acosh(1/2)-I*pi/3)/x^2 + 3` is 3), and a decided pole
  as `Pole` (`1/(2*sin(x)-1)` at asin(1/2)). `limit_via_series` and `limit_with_error` give `Finite`
  only for a rational constant term of a series of valuation >= 0 (`tan(x)*(x-acos(0))` at
  acos(0) is -1, gp -1); `Unknown` otherwise.
- New exact values (round 5), checked algebraically and numerically: `acosh(1/2)^2 + pi^2/9`,
  `acosh(1/2)*acosh(-1/2) + 2*pi^2/9`, `atanh(I)^2 + pi^2/16`, `asinh(I/2)^3 + I*pi^3/216`,
  `acosh(-1/2)^4 - 16*pi^4/81`, `asinh(I/2) + asinh(-I/2)`, `atanh(I) + atanh(-I)`,
  `2*atanh(I) - asinh(I)`, `acosh(0) - asinh(I)` are decided zero; `asinh(I/2)^3 - I*pi^3/216` is
  decided nonzero; `tan(2*atanh(I)/I)` is a pole (`hand/iv.tsv`, all right).

## 8. The red-team holes (class H)

Six truncated zeros Z (`sin(x)^2+cos(x)^2-1`, `(1+x)/(1+x)-1`, `exp(x)*exp(-x)-1`,
`log(exp(x))-x`, `sqrt(1+x)^2-1-x`, `tan(x)*cos(x)-sin(x)`) inside 26 shapes of an empty truncated
argument of negative precision (`asin(1 + x^-5*Z)`, `log(x^-5*Z)`, `1/(x^-5*Z)`, `sign(x^-5*Z)`,
`atanh(1 + x^-5*Z)`, ...) and 36 shapes of 0^r (`(x-x)^Z`, `Z^(1/2)`, `0^(atan(1)-pi/4)`,
`0^(asin(1/2)-pi/6+x)`, `0^(acosh(1/2))`, `(atan(1/2)+atan(1/3)-pi/4)^(1/2)`, ...):

- A value appears only where it is decided: `exp(x^-5*Z)`, `1/(1 + x^-5*Z)`, `(x^-5*Z)^0`,
  `cosh(x^-3*Z)` are 1 and `asin(x^-5*Z)`, `atanh(x^-5*Z)`, `log(1 + x^-5*Z)` are 0 (the engine raises
  the precision until the argument is O(h^k), k > 0; true for the identically zero Z); the oracle
  confirms each.
- At a branch point or a pole the answer is `PrecisionExhausted` (`asin(1 + x^-5*Z)`,
  `atanh(1 + x^-5*Z)`, `acos(-1 + x^-4*Z)`, `1/(x^-5*Z)`, `log(x^-5*Z)`, `abs`, `sign`): never a
  value, `NotLaurent` or `DivisionByZero`.
- 0^r: `(x-x)^(1+x)` and `(x-x)^(1+Z)` are 0, `(x-x)^(-1+x)` and `0^(-asin(1/2))` are
  `DivisionByZero`, `0^(asin(1/2)+x-pi/6)`, `0^(acosh(-1))`, `0^(I*asin(1/2))` are `NotLaurent`,
  `0^(atan(1/2)+atan(1/3)-pi/4)` is `UndecidedZero`, `(x-x)^Z` and `Z^(1/2)` are
  `PrecisionExhausted`, `0^0` and `0^(atan(1)-pi/4)` are 1.
- One family is m1 below (x^Z).

## 9. Variable named pi, I or e

Section 3. The shadow is set in `series_engine::expansion` before the point is read, so a point
written through the values (`taylor(sin(pi), pi, asin(1/2), 3)`) keeps the kernel `arcsin(1/2)`
and the rendering `pi - arcsin(1/2)`; it passes.

## 10. simplify past the memo cap

`rev6-semantics/sbig/mcap.rs`, the same program against HEAD's rlib and the port's: m shared nodes
s_i = P_j*(i+2) with 2^19-bit literals P_j, each used in four places (products, a symbolic term,
a quotient by y, a difference). m = 50, 600 and 1,200 (39 MB and 78 MB of shared literals, past
the 32 MiB cap): the structural printouts (integers by bit length and residue) are identical
(hashes 433e3f5cd63a112, 128443199d7ba4e9, 364473ae61bbd25e). On random DAGs with big shared
leaves (`sbig.rs`) the port differs from HEAD only where a power passes `MAX_FOLD_POWER_BITS`
(HEAD folds `B^3` of a 2^21-bit base, the port keeps it): the round-3 deviation, same value.

## 11. Comparison with HEAD on the rows the port refuses

HEAD's `taylor` (`drv/head6.rs`, against HEAD's debug rlib) on the 7,240 refused rows of classes B,
E, Z, H and R, checked by residuals on four rays and by evaluation at 60 and 240 digits
(`headcmp/`): HEAD's result has a 0/0 or 1/0 term (undefined) or is wrong on all rows except 101
`UndecidedZero` rows of class R, whose constants are complex kernels (`cosh(I/2)`, `log(1+I)`,
`cos(1+I)`, `1/(1+I)` kept as a kernel). There HEAD returns the unsimplified symbolic value. These
are the design's documented gap (§2.2 rule 4: a non-real kernel has no enclosure, so the zero test
is `Unknown`), honest refusals, accepted in earlier rounds (review 5 m2). Not a blocker.

## 12. Minor issues

### m1. x^r with r a truncated constant is `NotLaurent`, where only the exponent is undecided

`x^((1+x)/(1+x))`, `x^(sin(x)^2+cos(x)^2)`, `x^(2*(sin(x)^2+cos(x)^2))`, `x^(2*cos(x-x))`,
`x^(exp(x)*exp(-x))` and `x^(x^-5*Z)` at 0 are `NotLaurent` ("log has a logarithmic singularity at
the point"); the functions are x, x, x^2, x^2, x and 1 (the oracle: 18 class-H rows consistent,
lowest exponent 0). `pow_general` (`series_engine.rs:2058`) takes log of the base as soon as the
exponent's series is not exact. Where the exponent is `k + O(h^P)` with an integer k, it is not
decided whether the exponent is the constant k, so the module doc's rule ("whether an exponent is
an integer ... an undecided test is reported", `series.rs:13`) asks for `PrecisionExhausted`. No
value is produced, and the refusal is rule 2 applied to log of the base, so below the bar. HEAD
returns 0 for `x^((1+x)/(1+x))` (wrong), so this is no regression. Suggested: in `pow_general`,
when the base has positive valuation and the exponent is truncated with an integer constant term
and no other stored term, return `NeedPrecision`.

### m2. The matrix has no truncated-constant row per (function, branch point) pair

The (e) class has three guard rows, all arcsin. My class B shows the outcome per pair
(`PrecisionExhausted` on 1,520 rows), but no test pins it. Suggested: one row per pair,
`f(c*(1+x)/(1+x))` at 0, expecting `PrecisionExhausted` (never a value, `NotLaurent` or
`DivisionByZero`).

### m3. Gaussian rationals are not inverted exactly

`1/(x + 1/(1+I) - (1-I)/2)` at 0 is `UndecidedZero` (the constant is 0), and 101 class-R rows are
undecided on constants such as `1 + (9/8*I)/(1+I)^2` (= 25/16). The inverse of a multi-term
constant is a kernel, and a complex kernel has no enclosure. Honest. An exact inverse of A + B*I
with rational A, B ((A - B*I)/(A^2 + B^2)) would decide them. A handoff next to review 5's m2.

### Notes

- gp artifacts met by the oracle: acos and acosh next to +-1 lose half the precision; `sign` of a
  complex number fails; `exp` of 1e30 overflows; a near miss of size 1e-30 needs circles below the
  pole at -K; the expanded form of K^4 needs 500 digits.
- `NotLaurent` on functions that are Laurent occurs only where rule 2 or the intended refusals of
  the first step apply (`tan(atan(x))` and `exp(acosh(x))*x` at infinity, `sqrt(x)^(-2)`,
  `tanh(log(x))`, `0*f(..)`); m1 is the one family where the cause is an undecided exponent.
- A variable `Symbol::new("i")` or `Symbol::new("π")` built through the API is not shadowed (the
  parser maps both names first, so the REPL cannot make them); the result's Display re-parses the
  constant and the variable to one symbol. API only, noted.

## 13. Reproduction

```
R=/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/s1impl/rev6-semantics
cd $R/drv && ./build.sh r6.rs rel r6_rel && ./build.sh r6.rs dbg r6_dbg && ./build.sh head6.rs head head6
cd $R/fz && for c in E B Z H; do python3 -I ../gen6.py $c 601 0 > $c.tsv; ../drv/r6_rel $c.tsv > $c.out; python3 -I ../run6.py $c.out $c; done
cd $R/rt && for s in 61 62 63 64; do python3 -I ../gen6r.py $s 750; done > rall.tsv && ../drv/r6_rel rall.tsv > rall.out && python3 -I ../run6.py rall.out rall
cd $R/mat && ../drv/r6_rel mat.tsv > mat.out && python3 -I ../run6.py mat.out mat
cd $R/hand && VAR=pi ../drv/r6_rel vpi.tsv > vpi.out && python3 -I ../run6.py vpi.out vpi   # also vi (VAR=I), ve (VAR=e), iv, neg, gauss, xr
cd $R/hand && ../drv/lsc_rel lsc.tsv
cd $R/sbig && ./mcap_port 600; ./mcap_head 600
cd $R/headcmp && ../drv/head6 refused.tsv > head.out   # then h*.gp and stab.gp
```

`rlib_rel/` holds the release rlibs the drivers link against (copied at 03:02, fresh against the
current sources); the release test binaries are in `relbin/`. `target/release` was deleted at the
end. `rustmath-misc/output.txt` is unchanged (md5 7463088...) and no `.tmp_dir_*` exists.
