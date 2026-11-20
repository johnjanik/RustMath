# S1 implementation, review 3, lens: numeric and series semantics

Reviewer of the uncommitted S1 change in `/home/user/RustMath` (branch `claude/gracious-gates-e6b6l7`,
HEAD 8a0b310) after fix-up round 2. No repository file was edited: `git status --short` lists the
same 35 entries before and after. Scratch: `fu2/s1impl/rev3-semantics/`. Paths below are relative
to `/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/s1impl/`.

## Verdict

Approve in this lens. I found no blocker under S1_BAR. The tests pass in debug and in release. Two
minor issues follow (section 10).

- My fuzzer (9 600 rows, built differently from review 9's harness and from the fuzzers of rounds
  1 and 2) found no wrong value. 5 657 rows equal gp at 300 digits on every coefficient. The 168
  rows that differed at 300 digits were rechecked at 1 200 digits and by hand: every one is a gp
  artefact (section 4). The public renderings (`taylor_with_budget`'s expression and the re-parsed
  `Display`) give the same value as the terms on every row.
- The owner's two rules hold on 1 818 more branch-point rows of my own and on 108 high-valuation
  probes: no non-constant argument at a branch point gives a value or `DivisionByZero`; every
  `DivisionByZero` divides by an exact zero or applies f at an exact pole of a constant.
- The two new folds of fix-up round 2 (`exp_of_logs`, `sin_q_pi` reduced as an Integer) are
  right on every input I tried, negative bases and multiples of pi up to 10^1000 included.
- `simplify` gives HEAD's result on 20 000 random trees (exact structural equality). `expand`
  differs from HEAD on 1 042 of them; the port keeps the value on all 914 that gp can evaluate,
  HEAD loses it on 747 (the "expand under a minus" defect that S1 fixes).
- `Display` then `parse` keeps the value on 16 882 random trees and on every fuzz result.
- Release (8 MB thread) and debug (2 MB and 64 MB threads) give byte-identical outputs on all
  9 600 fuzz rows and on all hand probes.
- The tests of the first step are unchanged in their expected outcomes and pass. The round-2
  blockers (resources lens, B1 and B2) are fixed; I reproduced both checks.

## 1. Tests

| command | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic --no-fail-fast` (debug; `rev3-semantics/debug_tests.log`) | unit 675 passed, 3 ignored; `s1_depth` 9; `s1_fuzz_data` 7; `s1_payload` 2 passed, 1 ignored; `s1_regressions` 26 passed, 2 ignored; `s1_review2` 2; `s1_singular_points` 6; doc 116 passed, 11 ignored; 0 failed |
| `cargo test --release -j 2 -p rustmath-symbolic --no-run`, then each release binary (`rev3-semantics/rel_tests.log`) | unit 675 (3 ignored); `s1_depth` 9; `s1_fuzz_data` 7; `s1_payload` 2 (1 ignored); `s1_regressions` 26 (2 ignored); `s1_review2` 2; `s1_singular_points` 6; 0 failed |
| release `--ignored --test-threads=1` | `s1_regressions`: 2 passed (0.90 s); `s1_payload` `b2r2_consumers_of_large_leaf_results_release`: passed (11.39 s) |
| `rustmath_jupyter` lib test binary (fresh, built after the last `repl.rs` change), filter `taylor` | 5 passed (`test_taylor_engine`, `test_taylor_leaf_text_cap`, `test_taylor_shared_dag`, `test_taylor_display_linear`, `test_taylor_order_checked`) |

`target/release` was deleted after use.

## 2. Tests of the first step

Compared with the harness copy of the first step (`protoharness/work/src`):

- Byte-identical: `tests/s1_common/check.rs`, `tests/s1_common/eval.rs`, `tests/data/s1_fuzz.tsv`,
  `tests/data/gen_s1_fuzz.py`, `tests/data/s1_gp.py`.
- `tests/s1_common/api.rs`: the harness copy is the prototype shim; the repository file keeps the
  first step's mtime (17:45). `mod.rs` differs only in the doc paragraph about the removed feature
  gate.
- `tests/s1_regressions.rs` (mtime 20:02, before review 2): against the harness copy, the header
  (`#[path]` lines turned into doc comments), doc comments and one rustfmt wrap. No assertion and
  no expected outcome changed.
- `tests/data/s1_singular_points.tsv`: its first 614 lines are byte-identical to the harness copy;
  its first 677 lines are byte-identical to round 2's file (`fix2/s1_singular_points.tsv.before`).
  Fix-up round 2 appended the 3 rows of class (e).
- `tests/s1_fuzz_data.rs` (mtime 18:08) is unchanged since round 2 checked it.
  `tests/s1_singular_points.rs` gained the (e) test and an (e) arm in the coverage test; the arms
  of (a) to (d) are unchanged, and rows are filtered by class prefix only.

## 3. The round-2 blockers and minors

| item | status | evidence |
|---|---|---|
| resources B1 (`simplify` depth) | fixed | review 2's probe `consume2.rs` rebuilt against the current debug rlib: `fch 1250`, `fadd 900`, `zig 1600`, and `fch`, `zig`, `fadd` at 20 000 levels all return on a 2 MB debug thread; HEAD's binary aborts at `fch 1400` (`b1/`). `s1_depth` passes in both profiles. `simplify` results equal HEAD's (section 7). |
| resources B2 (leaf payload) | fixed | review 2's five 100 KB inputs at order 20 000 through the release driver: `LimitExceeded` (more than 134 217 728 characters of leaf text) in 0.01 to 0.02 s, peak RSS 12 MB (`b2/`). At order 200 they are values: 201 terms, each the one shared coefficient. The release `#[ignore]` consumer test passes. |
| semantics n1, n5 | handoffs (`S1_impl.md`, fix-up round 2, section 5) | read |
| semantics n2 (`exp(k log q)`) | done | section 6 |
| semantics n3 (float exponent) | no change, design rule | `x^2.0` at 0 is still `UndecidedZero("-2 + 2.0")` (REPL probe) |
| semantics n4 (`sin_q_pi`) | done | section 6 |
| semantics n6 (guard rows) | done, class (e) | rows read; the test passes |

## 4. My fuzzer

`rev3-semantics/fz3.py` (generator), `run3.py` (driver run, gp reference, comparison), `h3.gp`
(gp helpers adapted from review 2's: `PP`, `CH`, `CMP`, `ACOSH`, `ABS`, `SGN`, `TAN`, `SQ`, `LG`).
Driver `drv/r3.rs`, built against the current debug rlib (`drv/build.sh`) and against the release
rlib (`drv/r3_rel`, built before `target/release` was deleted). Per row it runs
`series_expansion` with the REPL budget 1e8, `taylor_with_budget`, `Display` of the result then
`parse`, `big_o`, `coefficient`, `laurent(0, n)`, `series_coefficients(n)` and
`limit_via_series(.., 5)`.

How it differs from the earlier harnesses (review 9: random trees at six rational points; round 1:
pole constructions and Taylor-prefix cancellations; round 2: generic parameters, special points,
function and inverse, high orders, infinity):

- Each row is built from a template g, a point a and an exact value v of g(a) written in another
  form (56 templates: trig at q pi with q up to 10^30, `2^x`, `x^x`, `(-2)^x`, `x^(1/x)`, exp at
  `log(3)` and `2 log 2 + 1`, log at 8, e^2, 1/9, powers at `sqrt(3)`, `2^(1/3)`, `-sqrt(2)`, `I`,
  `1 + I`, radicals at 4, 8, -4, 27/8, the inverse functions at their exact values, the hyperbolic
  functions at logs, abs and `sign(x)*x` at negative points, gamma).
- Z rows use the exact v, so g - v vanishes at a: poles, removable singularities, powers of a
  vanishing difference, and branch functions at their branch point through that difference
  (`asin(1 + k*(g - v))`, ...), or through an exact zero (`c + (x - x)*(g - v)`).
- P rows perturb v by a tiny nonzero constant (10^-60, 2^-100, pi - 355/113,
  exp(pi sqrt(163)) - 640320^3 - 744, sin(10^20)/10^5, ...), so the zero test must not decide 0.
- The gp reference is built from the gp form of the crate's own parse of the input, with `CH` at
  every sum and difference, I given inexact, and the principal power `PP`.
- The renderings are checked at a numeric point a + 1/997 (or 997 at infinity).

Results, 16 seeds x 600 rows (`fz/g301` to `g316`, `.res` per row), 300 digits:

| class | rows |
|---|---|
| equal to gp on every coefficient, renderings equal | 5 657 (1 939 of them P rows) |
| differ at 300 digits (DIFF 74, CMPERR 79, RT not checked 15) | 168, all rechecked below |
| `NotLaurent`: gp fails 1 477, gp has a value 94 | 1 571 |
| `UndecidedZero`: gp has a value 915, gp fails 482 | 1 397 |
| `Unsupported` (gamma of a series, abs or sign of a non-rational series) | 353 |
| `DivisionByZero` | 242, each with an exact zero factor `(x - x)*(...)`: atanh(+-1), atan(+-I), log(0) |
| engine value, gp fails | 212 |
| panic, crash, timeout | 0 (slowest row 0.84 s in debug) |

The 168 rows, rechecked at 1 200 digits (`GPPREC=1200`, `fz/sus2.res`):

- 36 are equal at 1 200 digits: near-zero divisors (P rows such as `(g - v')/(g - v')`, whose gp
  series division loses about 60 digits per order) and five rows on a branch cut whose gp constant
  carries a tiny imaginary part of either sign. I checked the five by hand: for example
  `acosh((-2)^x + 1/3)` at 3 is `acosh(-23/3)` with c1 = -8 log(-2) (-26/3)^(-1/2) (-20/3)^(-1/2),
  the principal convention of `series.rs`.
- 38 are `acos(-1 + (x - x)*D)`: the engine gives acos(-1) = pi (owner's rule 1). gp's
  `acos(-1 + O(h^10))` returns `O(h^5)` and drops pi (`t.gp`).
- 79 are `asinh(+-I + (x - x)*D)`: the engine gives the kernel asinh(+-i) = +-i pi/2; gp's series
  asinh at +-I is a domain error.
- 15 are right on their terms and `taylor_with_budget`; their `Display` exceeds 200 KB, so the
  driver did not re-parse it.

The 94 `NotLaurent` rows where gp has a value are all Z rows with a non-constant argument at a
branch point; gp's float constant term is only near the branch point. The 212 rows with a value
where gp fails: 197 exact constants at a branch point (`sqrt(0 + 0*D)` = 0, `acosh(+-1)`), and 15
with `exp(x/3)` at multiples of pi up to 10^30, where gp overflows. I checked three of the 15 by
hand (`exp(x/3)` at (10^20 + 1/6) pi has c_k = exp((6 10^20 + 1) pi/18)/(3^k k!), and two others).

Other checks on the same rows:

- `laurent(0, n)` and `series_coefficients(n)` against gp's valuation on the 5 672 rows equal to
  gp: no `Ok` at a pole, no `ValuationBelowMinimum` or `Pole` at a regular point; the dense
  coefficients equal `coefficient(k)` on every row; `coefficient(precision)` is `None`.
- `limit_via_series`: 1 537 `Finite` values equal gp's constant term, none at a pole; 155 `Finite`
  values without a gp reference, all `Finite(0)` of `sqrt(0 + 0*D)` and `acosh(1 + 0*D)`.

Hand probes against gp (each row through the same runner): `h1.tsv` (50 rows on `exp_of_logs`:
`x^x` at 3, -3, -1/2, `(-2)^x`, `2^x 3^x / 6^x`, poles of `x^x - 1/27`, `(-1)^x - 1`,
`exp(x log(-1)) + 1`, `10^x - 10^2000`, ...): 38 equal, 12 honest refusals. `neg.tsv` (16 rows,
negative bases): 15 equal, 1 undecided (`1/(x^x + 1)` at -1, a complex leading coefficient,
round 1 m3). `big.tsv` (15 rows, poles of tan, 1/sin, 1/cos at (10^30 + 1/2) pi, 10^25 pi,
(2^70 + 1/2) pi, ...): 12 equal at 1 500 digits, 3 right by exact reasoning (gp's float sin at
10^25 pi is not 0). `inf1.tsv` (25 expansions at infinity): 12 equal, 12 `NotLaurent` where gp
also fails (`sqrt(x^2 + 1) - x`, `x^(1/x)`, `atan(x)`, ...), 1 `Unsupported` (`acot`).

## 5. The owner's two rules and the singular-point matrix

Tally of the branch-function rows of my fuzzer (Z rows, `asin`, `acos`, `atanh`, `acosh` at +-1,
`atan`, `asinh` at +-I, `log`, `sqrt` at 0):

| argument | outcome |
|---|---|
| non-constant, constant term the branch point through an exact value (1 280 rows) | `NotLaurent` 870, `UndecidedZero` 360 (the field cannot decide g(a) = v), `Unsupported` 50; no value, no `DivisionByZero` |
| exact constant at the branch point, `c + (x - x)*D` (538 rows) | value 330 (asin, acos, acosh at +-1, asinh at +-I, sqrt at 0); `DivisionByZero` 182 (atan at +-I, atanh at +-1, log at 0: poles of the constant); `Unsupported` 26 (D holds gamma of a series, which surfaces under the factor 0, design §2.5) |

High-valuation and truncated arguments (`hp/hp.tsv`, 108 rows; `rt.tsv`, 20 rows), order 1:

- `f(1 + x^k)` and `f(1 - x^k exp(x))` for asin, acos, atanh, acosh, k = 10 to 10^9; `atan` and
  `asinh` at I + x^k; `log(x^k)`, `sqrt(x^k (1 + x))`: `NotLaurent`, or `PrecisionExhausted` when
  the deviation lies past the working precision (`asin(1 - x^1000 exp(x))`), or `BudgetExceeded`
  (`atanh(1 + x^(10^9))`). Never a value, never `DivisionByZero`.
- `asin(1 + (sin(x)^2 + cos(x)^2 - 1))`, `atanh(exp(x^1000))`, `asin(cos(x^500))`,
  `acosh(1 + x^-5 (sin^2 + cos^2 - 1))`, `sqrt(x^-5 (sin^2 + cos^2 - 1))`,
  `abs(x^-5 (sin^2 + cos^2 - 1))`: `PrecisionExhausted` (a bounded search, never decided).
- Where the truncated argument is provably small enough, the exact value comes out:
  `exp(x^-5 (sin^2 + cos^2 - 1))` is 1, `log(1 + ...)` and `atanh(...)` are 0, `0^(... + 1)` is 0.
- Exact constants: `asin(1 + ((1+x)^3 - 1 - 3x - 3x^2 - x^3))` and `asin((x^-3 + 1) - x^-3)` are
  arcsin(1); `atanh(1 + 1/x - 1/x)`, `atanh(sign(x^2 + 1))`, `atanh(1 + 0^(x+1))` are
  `DivisionByZero`; `log(0^(x+1))` at 0 is `DivisionByZero("log(0)")` and at -2
  `DivisionByZero("0 to a power with negative real part")`.

Generic parameters (`par.tsv`): `asin(a/a + x)`, `asin(a - a + 1 + x)`, `asin(x^2/a^2)` at a,
`atanh(x)` at a/a are `NotLaurent`; `asin(a/a + 0*x)` is arcsin(1), `atanh(a/a + 0*x)` and
`atan(I*a/a)*x` are `DivisionByZero`; identities through a quotient stay undecided
(`atanh((a+1)/(a+1) + x)`).

The matrix (`tests/data/s1_singular_points.tsv`, 680 rows) has, for each of the 23 (function,
branch point) pairs: (a) exact constants, (b) non-constant arguments, (c) regular neighbours,
(d) `f(1/x)` at 0 and arguments through exact kernel values, and (e) three guard rows. That is
what the owner asked for, for every inverse function and every branch point; the engine has no
other inverse function (acot and the other reciprocal inverses are `Unsupported`). My rows above
are outside the matrix and agree with it.

## 6. The new folds of fix-up round 2

`exp_of_logs` (`series_const.rs:2602`). exp of r + n_1 log(q_1) + ... is exp(r) q_1^n_1 ...: exp
is a homomorphism and exp(log q) = q on every branch, so the fold is exact for q < 0 too. Checked
against gp (`h1.tsv`, `neg.tsv`): `x^x` at 3, -1, -3 (27, -1, -1/27), `(-2)^x` at 3, -1, -3,
`(-2/3)^x` at -3 (-27/8), `(1/(-5))^x` at -2 (25), `(-3)^(x^2)` at -1 (-3),
`exp(x log(-1/2) + 2)` at -3 (-8 e^2), the poles of `1/((-2)^x + 8)` at 3,
`1/((-2/3)^x + 27/8)` at -3, `1/(2^x - 2^(x-1) - 4)` at 3, `1/(10^x - 10^2000)` at 2000 (their
principal parts too). A negative exponent of a negative base normalizes its sign (`(-2)^x` at -1
is -1/2). Non-integer multiples stay kernels (`exp(1/2 log(-2))`, `exp((1/2) log 4)`), and
`exp(10^15 log(-1))` stays a kernel (the bit cap).

`sin_q_pi` (`series_const.rs:2524`). The numerator of 12 q is reduced mod 24 with the Euclidean
`Integer::modulo`, so negative q are right. `1/(sin(x) + 1/2)` at -(10^1000 + 1/6) pi has the
residue 2/sqrt(3); `1/(cos(x) - sqrt(3)/2)` at (10^1000 + 1/6) pi has -2; `1/(tan(x) + 1)` at
(-10^999 - 1/4) pi has 1/2; `x/(sin((10^1000 + 1/3) pi) - sqrt(3)/2)` and
`x*tan((10^1000 + 1/2) pi)` are `DivisionByZero`; `sin(x)` at (10^1000 + 1/12) pi keeps the
kernel (`qq.tsv`, `big.tsv`). Each checked by exact reasoning; gp agrees where its floats reach.

The leaf-text cap (`MAX_RESULT_TEXT`, `undecided_expr`) only refuses. No value changes.

## 7. Renderings, parser, simplify and expand

- `to_expr` and `taylor_with_budget`: equal to the terms at a + 1/997 on every fuzz and hand row.
  At infinity the term of exponent k is c/x^k (checked on `inf1.tsv`).
- `Display` then `parse`: equal on every fuzz row whose text is under 200 KB, and on 16 882 of
  20 000 random programmatic trees (`sd/dr_check.out`); the other 3 118 either hold `%`
  (`Mod`, which neither HEAD's parser nor the port's reads), a literal `n / 0` (a parse error by
  design), or no gp value at the sample point.
- Parser precedence against gp's own parser on 60 texts (`p/p1.tsv`): `-2^2` = -4,
  `2^-1/2` = 1/4, `2/3^2` = 2/9, `x^-1/2` = (x^-1)/2, `2^-3^2` = 2^-9, `-1^(1/2)` = -1,
  `(-1)^(1/2)` = i, `9223372036854775808/2`, a 29-digit literal: all equal to gp where gp parses
  (gp reads `--` as decrement, so 4 texts have no gp value).
- `simplify` (`sd/`, `drv/sd.rs` built against HEAD's rlib `librustmath_symbolic-4cbaba7850391989`
  and against the port's): equal to HEAD on all 20 000 random trees (node kinds: the six binary
  operators, 18 unary functions, one- and two-argument calls; integers, rationals, floats, x, y,
  pi, e, I).
- `expand`: differs from HEAD on 1 042 trees. At x = 37/100, y = 59/100 the port's result equals
  the input on all 914 that gp evaluates; HEAD's is wrong on 747 of them.

## 8. Limits and the REPL

- `limit_via_series` (section 4): every `Finite` value right, none at a pole.
- REPL probes (`repl/r1.txt`, `ReplContext::eval` on a 64 MiB thread): `taylor("x^x", x, 3, 3)`
  starts 27; `taylor("1/(2^x - 8)", x, 3, 2)` is the pole with 1/(8 log 2); `asin(1 + x - x)` is
  arcsin(1); `atanh(1 + x - x)`, `log(x - x)`, `1/(x - x)`, `atan(I + 0*x)` and
  `x*tan((2^60 + 1/2)*pi)` are `ValueError` (division by an exact zero); `asin(x)` at 1,
  `atanh(x)` at 1, `log(x)` at 0, `atan(x)` at I and `exp(1/x)` at 0 are `ComputationError`
  ("no Laurent expansion"); `tan(x)` at pi/2 prints `-1/(x - pi/2) + ...`; `-2^2*x` is `-4 * x`.

## 9. Release against debug

`fz/all.tsv` (the 9 600 fuzz rows) through the debug driver on 64 MB and 2 MB threads and the
release driver on 8 MB threads: 0 rows differ in any field but the time, 0 panics
(`fz/all_compare.txt`). The same holds for the hand probes (`hp`, `rt`, `par`, `neg`, `big`, `h1`,
`inf1`, `abs`, `fl`, `inf`). No debug run overflowed an integer, so the release build cannot wrap
on these inputs.

## 10. Minor issues

### m1. `SeriesExpansion::big_o` at infinity is O(x^P), not O(x^-P)

`series.rs:428` and `:433`: `big_o()` is `O(h^precision)` with the stored h, and at infinity the
stored h is the variable itself (`h_of`, `:445`). So `(x/(x+1)).series_expansion(x, Infinity, 4)`
has `big_o()` = O(x^5), while the remainder is O(x^-5) (`inf1.drv`, field BO). The struct and
method docs say h = 1/var at infinity. The statement O(x^5) is true as x goes to infinity, so it
is not a wrong value, but it is useless and contradicts the doc. No caller uses it at infinity
(`series_with_big_o` takes a finite point). Required (minor): at infinity return O(1/x^P), and add
a test.

### m2. A nonzero float literal can parse as the exact 0

`parser.rs:159` (`float`, HEAD's rule): `1.0e-400` parses as `Real(0.0)`. The engine then sees an
exact zero: `x/1.0e-400` at 0 is `DivisionByZero("division by the constant 0")`,
`atanh(1 + 1.0e-400)*x` is `DivisionByZero`, and `1/(x + 1.0e-400)` is `1/x` (`uf.tsv`). The
documented convention ("a float literal denotes its exact binary value") covers it, and HEAD's
parser does the same, so it is no regression. But the engine refuses the overflow case
(`1.0e400` is `Unsupported("non-finite float inf")`, `series_const.rs:3076`), and the same honesty
applies to underflow. Suggested: the parser refuses a float literal with a nonzero digit whose
f64 value is 0 (and, if wanted, a subnormal one), or the doc says that such a literal is 0.

### Notes (no change needed for the bar)

- A constant on a branch cut crossed by the argument (for example `acosh((-2)^x + 1/3)` at 3,
  `sqrt((-2)^x + 8 - t)` at 3) gives the series of the documented principal convention; the
  principal-branch function is discontinuous there along the real line. Fix-up round 2 named the
  same case for `log`. The convention is stated in `series.rs`.
- abs and sign at a non-real point are refused unless the series has rational coefficients;
  `abs((x - I)^2 + 1)` at I is `1 + (x - I)^2`, right along I + R (round 2's note).
- Complex leading coefficients stay undecided (`1/(x^x + 1)` at -1, `1/(x^x + 1/27)` at -3):
  round 1 m3, handed off.

## 11. Reproduction

```
R=/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/s1impl/rev3-semantics
cd $R/drv && ./build.sh r3.rs r3_dbg                    # the driver (current debug rlib)
cd $R/fz && python3 -I ../fz3.py 301 600 g301 && python3 -I ../run3.py g301   # one seed
cd $R/fz && GPPREC=1200 python3 -I ../run3.py sus2       # the 168 rows at 1 200 digits
cd $R && python3 -I run3.py h1; python3 -I run3.py neg; GPPREC=1500 python3 -I run3.py big
cd $R && python3 -I run3.py inf1; grep -P '^i1\t' inf1.drv | tr '\t' '\n' | grep -A1 '^BO$'   # m1
cd $R && ./drv/r3_dbg ser uf.tsv                         # m2
cd $R && ./drv/r3_dbg ser hp/hp.tsv; ./drv/r3_dbg ser rt.tsv; ./drv/r3_dbg ser par.tsv
cd $R && ./drv/r3_dbg parse p/p1.tsv                     # parser precedence (gp check: p/p1.gp)
cd $R/sd && python3 -I gen.py 1 5000 > t1.tsv            # random trees; sd.rs against HEAD and the port
```

Binaries deleted to save disk (rebuild with the rustc lines of `drv/build.sh`, using
`--extern rustmath_symbolic=` the HEAD rlib `librustmath_symbolic-4cbaba7850391989.rlib` or the
port's `librustmath_symbolic-704f099c312b95e8.rlib`): `sd_head`, `sd_port`, `dr_port`,
`repl_probe`, `c2_port_dbg`. Kept: `r3_dbg`, `r3_rel`.
