# S1 implementation, review 2, lens: numeric and series semantics

Reviewer of the uncommitted S1 change in `/home/user/RustMath` (branch `claude/gracious-gates-e6b6l7`,
HEAD 8a0b310) after fix-up round 1. No repository file was edited (`git status --short` lists the
same 35 entries before and after). Scratch: `fu2/s1impl/rev2-semantics/`. Paths below are relative
to `/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/s1impl/`.

## Verdict

Approve in this lens. I found no blocker under S1_BAR. The tests pass. Six minor issues follow.

- My own fuzzer (4000 inputs, different from review 9's and from round 1's) found no wrong value:
  3194 rows equal gp at 300 digits on every coefficient, 0 differ, 0 crash.
- The owner's two rules hold on 1653 more branch-point inputs of my own: no non-constant argument at
  a branch point gives a value or `DivisionByZero`; every `DivisionByZero` divides by an exact zero.
- Every `Finite` limit that S1's own entries return (`limit_via_series`, `limit_with_error`) is right:
  354 of 354 checked against gp, none at a pole.
- The tests of the first step are unchanged in their expected outcomes and pass.
- The round-1 blockers B1, B2 and B3 (resources lens) are fixed for the shapes they named. B2 is in
  my lens: `limit_with_error` now runs through the engine; I checked it in debug and in release. The
  round-2 resources review reports two new resource blockers (B3's fix leaves function chains and
  zigzags with a larger frame than HEAD); that is outside this lens.

## 1. Tests

| command | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic --no-fail-fast` (debug, my run; `rev2-semantics/debug_tests.log`) | unit 675 passed, 3 ignored; `s1_depth` 7; `s1_fuzz_data` 7; `s1_regressions` 26 passed, 2 ignored (the release budget tests); `s1_singular_points` 5; doc 116 passed, 11 ignored; 0 failed |
| release | not rebuilt by me (disk at 3.6 GB free; the round-2 resources review built release and reports all four S1 test files and the two `--ignored` budget tests passing). My release checks use the implementer's release binaries, built after the last source change (section 7). |

## 2. Tests of the first step

- `tests/data/s1_fuzz.tsv` is byte-identical to the round-3 snapshot `r3/fz_new.tsv` and to
  `protoharness/work/src/data/s1_fuzz.tsv`.
- `tests/data/s1_singular_points.tsv`: its first 614 lines are byte-identical to `r3/sp_new.tsv`; the
  63 rows of class (d) are appended after them (round-1 m5).
- `tests/s1_common/check.rs` and `eval.rs` are byte-identical to the harness copy. `api.rs` keeps the
  mtime 17:45 (first step). `mod.rs` differs from the harness copy only in the doc paragraph about the
  removed feature gate.
- `tests/s1_regressions.rs` against the harness copy, header excluded: doc comments only (the
  `b:*:sub` sentence, the `limit_with_error` sentence of round-1 B2, one section number, the removed
  `--features` flag in two run commands) and one rustfmt wrap. No assertion changed.
- `s1_singular_points.rs` and `s1_fuzz_data.rs` filter rows by class prefix only; no row is skipped.

## 3. The blockers and minors of round 1

| item | status | evidence |
|---|---|---|
| resources B1 (parser frame) | fixed | `s1_depth` B1 tests pass in my debug run; release per the round-2 resources review |
| resources B2 (`limit_with_error` unbounded) | fixed | `limits.rs:558` calls `limit_via_series(var, point, 5)` and `expansion_reaches`; `s1_depth` B2 tests pass in debug; release (`fix1/sp_rel`, built 19:57 after the `limits.rs` change): the 30 000-term sum 0.22 s and 21 MB (`Finite(30000)`, `O((x-1)^4)`), `x+sin(1)+...+sin(9999)` 0.16 s and 21 MB (`Unknown`), the 13 000-level `x^-x^-...` tower 0.34 s and 16 MB (`Unknown`) |
| resources B3 (simplify frame) | fixed for its three shapes | `s1_depth` B3 tests pass in debug; the round-2 resources review finds two other shapes |
| semantics m1 (kernel at an undecided pole) | not changed, convention now stated (`series.rs:36`) | I agree with the implementer's reason: a zero test of 1 + c^2 cannot tell `arctan(sqrt(-1))` from `arctan(arctanh(3))` |
| m2 (undecided divisor convention) | documented (`series.rs:9-15`, `:36`) | read |
| m3 (no complex zero test) | handoff | `S1_impl.md`, fix-up round 1, section 6 |
| m4 (`PrecisionExhausted` for some exact poles) | doc of `DivisionByZero` amended | read |
| m5 (matrix gaps, gp artefacts) | class (d), 63 rows | `other_shapes_of_branch_points_are_not_laurent` passes |

## 4. My fuzzer

`rev2-semantics/fz/fz2.py` (generator), `run2.py` (driver run, gp reference, classes),
`helpers.gp` (gp functions). Driver `rev2-semantics/drv/r2.rs`, built in debug against the crate's
rlibs (`drv/build.sh`): `series_expansion` with the REPL budget 1e8 on a 64 MB thread, then
`limit_via_series` (order 5), `limit_with_error` (order n), `laurent(0, n)` and
`series_coefficients(n)` (the last two only when the budgeted expansion succeeded, since they have
no budget by design).

How it differs from review 9's harness (random trees at six rational points) and from round 1's
(`fzsem.py`: pole constructions and Taylor-prefix cancellations at 16 points):

- A generic parameter `a` in the input and in the point (`a`, `-a`, `1+a`, `2*a`); gp uses a = 7/11.
- Special points where only the exact constant gives the right answer: pi/6, pi/4, pi/3, pi/2, pi,
  e, sqrt(2), 2^(1/3), log(2), with quotients of functions that vanish there exactly (poles and
  removable singularities).
- A function composed with its inverse at regular points (`sin(asin(..))`, `atanh(tanh(..))`, ...).
- High orders, 12 to 24, at rational points.
- Expansions at infinity.
- The gp reference is built node by node. Each node chops leading coefficients below 1e-250 (gp's
  float pi). Non-integer powers are `exp(r log b)` (gp's `t_FRAC` power takes the real root); `acosh`
  of a series is gp's scalar `acosh` plus the integral (gp's series `acosh` takes the wrong sign left
  of -1); `tan` is `sin/cos` with the constant of `cos` chopped first; `I` is given to gp inexact
  (gp's series functions work at about 19 digits on an exact complex constant term, confirmed:
  `asin(2 + I + h)` differs by 4e-20 from `asin(2 + 1.*I + h)`).
- Every coefficient from the lowest exponent of either side up to n is compared (relative 1e-60).

Results, 8 seeds x 500 inputs (`fz/g201` to `g208`, `.res` per row):

| class | rows |
|---|---|
| equal to gp on every coefficient | 3194 (338 with a pole, 396 at order 12 to 24) |
| differ | 0 |
| engine value, gp fails | 3: `((x+a)-(x+a))^(1/2)` = 0, `x*abs(abs(1))*sign((x+x)*(x-x))` = 0 at a, `sign(I)` kept as a kernel |
| `NotLaurent` | 326 (324 gp also fails; 2 gp has a series: `erf(asin(x^4 - x/x))` at 0 and `acosh(acos(atanh(log(x))))` at e, both rule 2) |
| `UndecidedZero` | 327 (323 gp has a value: generic-parameter kernels such as `sin(cos(3a))`, `exp(1)` against `e`, complex constants) |
| `Unsupported` | 122 (abs and sign of non-rational coefficients) |
| `DivisionByZero` | 15, each an exact zero divisor or f at an exact pole of a constant: `(x-x)^(-2)`, `x/(a*x - a*x)`, `log(x-x)`, `(x+sqrt(2))/(x-x)`, `x/(pi/6 - pi/6)`, `log(1)^(-2)`, `atanh(a/a)`, `atanh(sign(x+2/3))` at 0, `atanh(1)`, ... |
| `PrecisionExhausted`, `BudgetExceeded` | 5, 8 |
| panic, crash | 0 |

`laurent` and `series_coefficients` on the 3020 rows where both ran and gp has a reference: on the
277 rows where gp's valuation is negative, `ValuationBelowMinimum` and `Pole` on 271 and
`UndecidedZero` on 6; on the 2743 others `Ok`, except 20 `UndecidedZero`. The 26 undecided rows hold
a stored term below 0 whose coefficient the field cannot decide (it is 0 in gp on the 20), which is
the documented honest answer (`first_exponent` doc). Example: `(x - exp(1))*acos((-1)^a)
/(log(x^2) - 2)` at e, whose h^-1 coefficient is `e*exp(1)*c - e^2*c` with `exp(1)` and `e` separate
kernels.

## 5. The owner's two rules and the singular-point matrix

Rule 2 brute force (`rev2-semantics/rule2.tsv`, `.meta`, `.out`): 13 functions (arcsin, arccos,
arctan, arcsinh, arccosh, arctanh, log, sqrt, `u^(1/3)`, `u^(-1/2)`, `u^pi`, `u^(x+1)`, `u^I`) x each
branch point x 29 argument shapes (c + x, c - x^2, c + x^3, c - x^4, c + I*x, c*cos(x), c*exp(x^2),
c/(1+x), c + sin(x)^2, c + tan(x), c*cosh(x), c*sqrt(1+x), c + log(1+x), c + atan(x),
c*(1-x^2)^(1/2), c^(1+x), -(-c - x), ...) x 3 points (0, 1/2, -2): 1653 inputs.

| outcome | rows |
|---|---|
| `NotLaurent` | 1437 |
| a value | 110, all with c = 0 and a product shape (`0*cos(x)` is the exact 0): sqrt, the powers and `u^(x+1)` at 0 and 1/2 give 0 (rule 1) |
| `DivisionByZero` | 70, all the exact 0 again: `log(0*..)`, `(0*..)^(-1/2)`, `0^(x+1)` at -2 (Re r < 0) |
| `UndecidedZero` | 24: `c^(1+x)` with c = -1, +-I (`exp(log(-1))` is not folded to -1) |
| `PrecisionExhausted` | 12: `1^(1+x)` (`exp(0*..)` is `1 + O(h^P)`, not the exact 1) |

No non-constant argument at a branch point gives a value or `DivisionByZero`. The release driver
gives the same outcome kind on all 1653 rows (section 7).

The matrix (`tests/data/s1_singular_points.tsv`, 672 rows) has, for each of the 23 (function, branch
point) pairs: (a) exact constants in 16 or 17 shapes, (b) non-constant arguments in 8 shapes, (c) the
two regular neighbours, (d) `f(1/x)` at 0 and two shapes through an exact kernel value. That is what
the owner asked for, for every inverse function and every branch point, infinity included. The
`c^(1+x)` shapes above are honest refusals the matrix does not hold (minor n6).

Hand-built rows (`rev2-semantics/hand1.tsv`, `hand2.tsv`, `hand3.tsv`, `e.tsv`; outputs with
`drv/r2_dbg FILE`), each checked by hand or against gp:

- Generic parameters: polynomial identities decide zero (`1/(x + (a+I)*(a-I) - a^2 - 1)` is `1/x`,
  `1/(x*(a-a) + x^2)` is `1/x^2`, `1/(x + a/(a*a) - 1/a)` is `1/x`); identities through kernels stay
  undecided (`sqrt(a)^2 - a`, `exp(a)*exp(-a) - 1`, `sign(a)^2 - 1`, `(a+1)/(a+1) - 1`,
  `sqrt(a^2) - a`, which is not an identity for complex a). `x^a` at 0 is `NotLaurent` (generic
  non-integer). `(1+x)^a`, `1/(x^2 - a^2)` at +-a, `1/(sin(x) - sin(a))` at a, `1/(x^a - 1)` at 1
  are right term by term.
- Exact constants: 35 divisors `1/(x + c)` with c an exact zero (`sqrt(8) - 2*sqrt(2)`,
  `(27/8)^(2/3) - 9/4`, `tan(3*pi/4) + 1`, `log(exp(2)*e) - 3`, `sin(100*pi)`, ...) are `1/x`; the true
  zeros the field cannot prove stay `UndecidedZero` (`(2^(1/6))^3 - sqrt(2)`, `exp(I*pi) + 1`,
  `(-8)^(1/3) - 1 - I*sqrt(3)`, `4*atan(1) - pi`, `gamma(1/2)^2 - pi`, `zeta(-2)`,
  `factorial(3) - 6`, `exp(1) - e`); `(-27/8)^(2/3) - 9/4`, which is nonzero (principal root), is
  undecided, not decided wrong.
- Exact poles: `tan(x)` at pi/2, 3pi/2, -pi/2; `1/cos(x)` at 5pi/2; `1/(sin(x)-1)` at pi/2;
  `1/(log(x)-1)` at e; `1/(x^2-2)` at sqrt(2); `1/(x^3-2)` at 2^(1/3); `1/(x^2+1)` at +-I;
  `1/(x^4-4)` at sqrt(2)*I: right principal parts. Removable: `(tan(x)-1)/(x-pi/4)` at pi/4 is
  `2 + 2h + 8/3 h^2` and its limit `Finite(2)`; `(x^2-2)/(x^4-4)` at sqrt(2) gives `Finite(1/4)`.
- Near-zero constants decided nonzero correctly: `exp(pi*sqrt(163)) - 262537412640768744`,
  `pi - 3.141592653589793`, `sqrt(2) - 1.4142135623730951`, `exp(-10^6)`, `10^-100`.
- Constant poles: `gamma(0)`, `gamma(-2)`, `factorial(-1)`, `zeta(1)`, `log(0)`, `0^(-1)`,
  `0^(-pi)` are `DivisionByZero`; `0^0` is 1, `0^pi` and `0^(1+I)` are 0, `0^I` is `NotLaurent`,
  `0^a` is `UndecidedZero`.

## 6. The limit path against gp and against HEAD

Drivers `rev2-semantics/head/lim3.rs` (built against HEAD's library
`target/debug/deps/librustmath_symbolic-4cbaba7850391989.rlib`, built 15:43 before the port, and
against the port), run by `head/runlim.py` on the 2255 finite-point rows of seeds 201, 202, 204,
206, 208 (`head/limin.tsv`); values in gp form checked by `head/limcmp.py` against the gp reference
of section 4 (constant term when the valuation is >= 0; 0 when it is > 0).

| entry | Finite answers | right | wrong value | Finite at a pole | gp cannot evaluate the value (1/0) | no reference |
|---|---|---|---|---|---|---|
| port `limit_via_series` | 356 | 354 | 0 | 0 | 0 | 2 |
| port `limit_with_error` | 356 | 354 | 0 | 0 | 0 | 2 |
| HEAD `limit_via_series` | 210 | 206 | 0 | 0 | 0 | 4 |
| HEAD `limit_with_error` | 1563 | 1350 | 83 | 20 | 72 | 38 |
| port and HEAD `Expr::limit` (HEAD's code, the REPL's `limit`) | see below | | | | | |

- `limit_with_error` (round-1 B2 fix, design §2.13): of HEAD's answers, 1097 that are not bad
  become `Unknown` (1060 right against gp, 37 without a reference; 1088 symbolic such as `sin(1)`,
  `exp(1)`, `log(2)`, 9 rational), 123 bad ones become `Unknown`, 52 bad ones become right; 13 new
  answers. Minor n1.
- `limit_via_series`: 9 of HEAD's right answers become `Unknown` (minor n2); 153 new answers, right
  wherever gp has a reference.
- `Expr::limit` is unchanged by S1 and does not call `limit_via_series` (only `limit_enhanced` does,
  which has no caller). On the 2074 inputs HEAD's parser reads, the port's outcome class equals
  HEAD's on every row (the strings differ only by the new Display's parentheses, `2 ^ 1/3` against
  `2 ^ (1/3)`). It returns wrong `Finite` values in HEAD too: `(x^4 - 4)^(-3)` at sqrt(2) is
  `Finite((sqrt(2)^4 - 4)^(-3))` in both. Minor n5.

## 7. Release

The implementer's release driver `fuzzdrv/s1drv_rel` was built at 20:09:15, after the last source
change (`series_const.rs`, 20:06:48). Through it (`taylor_with_budget`, 1e8, 8 MB thread;
`rev2-semantics/rel/relcheck.py`): every fifth finite-point fuzz row (719) and all rows of
`rule2.tsv`, `hand1.tsv`, `hand2.tsv` (1803) give the same outcome kind as my debug driver, on every
row.

## 8. Minor issues

### n1. `limit_with_error` lost HEAD's symbolic limits (design §2.13, round-1 B2)

`limits.rs:558`. `limit_with_error(sin(x), x, 1, Both, 3)` was `(Finite(sin(1)), O((x-1)^4))` at
HEAD and is `(Unknown, None)` now; `exp(x)` at 1, `x^2 + pi` at 0, `log(x)` at 2, `atan(x)` at 1 the
same (`head/lwe1.tsv`). Measured on 2255 inputs: 1097 answers lost (1060 of them right against gp,
37 without a reference), 175 bad ones removed or corrected (HEAD's 83 wrong values, 20 `Finite` at a pole, 72 values that are 1/0). Why below the
bar: the frozen design states this behaviour ("the limit as above"), round-1 B2 required it or a
bounded `Expr::limit`, every answer now given is right, `Unknown` is honest, and no caller in the
workspace uses `limit_with_error` (grep). The implementer's report states the change. Required
(minor): carry the numbers into the tracker row ("limit with error": `Finite` only for a rational
limit), so the loss is visible.

### n2. `exp(k log q)` is not folded to q^k

`series_const.rs:2513` (`kapply_src`, the `Exp` arm). `x^x` at 3 has the constant term
`exp(3*log(3))`, `2^x` at 3 `exp(3*log(2))`, `x/x^x` at 2 `2/exp(2*log(2))`. The series are right
(gp agrees), but the limit path reads no rational, so `limit_via_series(x^x, x, 3)` is `Unknown`
where HEAD gives `Finite(27)`; 5 of the 9 lost answers of section 6 are this class (the other 4 are
one-sided or branch-point limits, `NotLaurent`). `1/(2^x - 8)` at 3 is `UndecidedZero` where it has
a pole. Honest refusals; the REPL's `limit` is not affected (`Expr::limit` substitutes first). The
design knows the kernel `exp(log(2) K)` of towers (§2.15). Suggested: in `kapply_src`, `Exp` of a
Q-linear combination of `log(q_i)` with integer coefficients (and an integer part times the kernel
`e`) is the exact rational product, the inverse of `log_exact`.

### n3. A float exponent with an integral value is undecided

`x^2.0`, `x^3.0 + x`, `sin(x)^2.0` at 0 are `UndecidedZero("-2 + 2.0")`. HEAD's taylor returns an
expression equal to x^2 (with explicit `0^0` terms; `head/tay1.tsv`). This follows from the design's
rule that a float literal is a `NonZero` kernel with no folding (§2.2; `0.5 - 1/2` undecided is
documented), so it is a design-level refusal, not a wrong value. Suggested: read a finite float
with an integral value below 2^53 as that integer where an exponent's integrality is tested, or let
the numeric test decide `Zero` when every ball on the path is exact (the exact dyadic balls of
revision 7rt4 already exist).

### n4. `sin_q_pi` overflows i64 on large multiples of pi

`series_const.rs:2436`: `NumericConversion::to_i64(twelfths.numerator())?.rem_euclid(24)`. For
|12 q| >= 2^63 the exact value is not found. So `x*tan((2^60 + 1/2)*pi)` at 0 is
`Ok(tan(..)*x)` with the limit `Finite(0)`, while `x*tan((2^58 + 1/2)*pi)` is
`DivisionByZero("tan at an odd multiple of pi/2")`; `sin(10^20*pi)` and `cos(10^20*pi) - 1` are
`UndecidedZero` where they are exactly 0 (`hand2.tsv` z8 to z14). proto13 has the same line. HEAD
gives `Finite(0)` for both tan inputs (`head/q.tsv`), and the class is the known cost of design §7
item 10, so it is below the bar. The design's own text says this arithmetic is exact for large q
(§2.2: "uses q_add and q_mul, since q can be large"). Fix: reduce the numerator mod 24 as an
`Integer` before converting.

### n5. HEAD's `Expr::limit` reaches more texts through the new parser

Not S1 code, but S1's parser now reads texts that HEAD's parser refused (`1/x`, `1/(x^4 - 4)^3`).
On 181 of my inputs HEAD gives a parse error; the port's `Expr::limit` (the REPL's `limit`) gives
154 `Finite` values at a pole there, such as `Finite(1/((sqrt(2)^4 - 4)^3))` for `1/(x^4 - 4)^3` at
sqrt(2), and 8 wrong values. The same facade exists at HEAD for texts it parses (`(x^4-4)^(-3)`,
`x/(x^2-2)` at sqrt(2): `head/v.tsv`), so no input HEAD handles regresses. S1's own limit entries
are clean. Required (minor): a HANDOFF row for `limits.rs` `limit`, `limit_advanced` and the
`combine_limits_*` laws (symbolic `Finite` values without a zero test of the denominator).

### n6. Guard rows the matrix could add

The matrix covers every function and branch point. Three honest refusals of non-constant or
non-folded shapes are not held by any test: `asin(1^(1+x))` at 0 (`PrecisionExhausted`),
`asin((-1)^(1+x))` (`UndecidedZero`, `exp(log(-1))`), `asin(sin(1)^2 + cos(1)^2 + x)`
(`UndecidedZero`). Rows with these outcomes would keep "never `DivisionByZero`" locked for them.

Coverage notes (no change needed for the bar): a single monomial with a generic parameter and a
nonzero kernel is undecided (`a*erf(-8/(3*pi))`), and `exp(1)` is a kernel distinct from `e`. Both
belong with round-1 m3's handoff. abs and sign at a non-real point are expanded along the real
direction (`abs((x-I)^2 + 1)` at I is `1 + (x-I)^2`); the limits that come out are right.

## 9. Reproduction

```
R=/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/s1impl/rev2-semantics
cd $R/drv && ./build.sh r2.rs r2_dbg                  # the driver (debug rlibs listed in ../rlibs_dbg.txt)
cd $R/fz && python3 -I fz2.py 201 500 g201 && python3 -I run2.py g201    # one fuzz seed
cd $R && ./drv/r2_dbg hand2.tsv | cut -f1-6           # n4 (z10 to z14), exact constants and poles
cd $R && ./drv/r2_dbg e.tsv | cut -f1-6               # n2
cd $R/head && ./tay_head tay1.tsv; cd $R && ./drv/r2_dbg head/tay2.tsv    # n3, HEAD and port
cd $R/head && python3 -I runlim.py ./lim3_port limin.tsv lim3_port.out && python3 -I limcmp.py   # section 6
cd $R/head && ./lim3_head v.tsv 0; ./lim3_port v.tsv 0  # n5
cd $R/.. && python3 -I $R/rel/relcheck.py fuzzdrv/s1drv_rel $R/fz/g20*.tsv  # release kinds
```
