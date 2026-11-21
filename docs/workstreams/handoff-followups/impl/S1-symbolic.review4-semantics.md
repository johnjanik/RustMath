# S1 implementation, review 4, lens: numeric and series semantics

Reviewer of the uncommitted S1 change in `/home/user/RustMath` (branch `claude/gracious-gates-e6b6l7`,
HEAD 8a0b310) after fix-up round 3. No repository file was edited: `git status --short` lists the
same 35 entries before and after. Scratch: `fu2/s1impl/rev4-semantics/`. Paths below are relative
to `/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/s1impl/`.

## Verdict

Approve in this lens. I found no blocker under S1_BAR. The tests pass in debug and in release.
Two minor issues and three notes follow (section 9).

- Two fuzzers of my own (13 800 rows, built differently from review 9's harness and from the
  fuzzers of rounds 1 to 3) found no wrong value. 8 608 rows agree with gp on every coefficient,
  26 059 rational coefficients of them exactly (not to a tolerance). Every row that differed from
  gp is a gp artefact, checked by hand (section 4).
- The owner's two rules hold on 2 158 more rows of my own (section 5): no non-constant argument
  at a branch point gives a value or `DivisionByZero` (986 `NotLaurent`, 75 `UndecidedZero`); every
  exact-constant argument gives f(c), or `DivisionByZero` at a pole of f(c) (422 rows), with one
  gap that is honest but weak (minor m1: 69 rows `UndecidedZero` once f(c) must be inverted).
- A probe of 117 constants built to trap wrong folds (radicals of negative numbers, logs of
  products, inverse-of-function compositions, exact trig values, `0^r`) found no wrong zero
  decision in either direction (section 6).
- `simplify` with round 3's memo and fold guard keeps HEAD's values; the guard changes HEAD's
  output only for powers above 2^20 bits, as documented (section 7).
- Release (8 MB thread) and debug (64 MB and 2 MB threads) give the same output on all 13 800 rows.
- The tests of the first step are unchanged in their expected outcomes and pass. The blockers of
  round 3 (resources B1, B2) are fixed; I reproduced every input of both on the built code.

## 1. Tests

| command | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic --no-fail-fast` (debug; `rev4-semantics/debug_tests.log`) | unit 681 passed, 3 ignored; `s1_depth` 9; `s1_fuzz_data` 7; `s1_payload` 2 passed, 1 ignored; `s1_regressions` 26 passed, 2 ignored; `s1_review2` 2; `s1_review3` 7 passed, 7 ignored; `s1_singular_points` 6; doc 116 passed, 11 ignored; 0 failed |
| `cargo test --release -j 2 -p rustmath-symbolic --no-run`, then each release binary (`rev4-semantics/rel_tests.log`) | unit 681 (3 ignored); `s1_depth` 9; `s1_fuzz_data` 7; `s1_payload` 2 (1 ignored); `s1_regressions` 26 (2 ignored); `s1_review2` 2; `s1_review3` 7 (7 ignored); `s1_singular_points` 6; 0 failed |
| release `--ignored --test-threads=1` | `s1_payload` 1 passed (3.32 s); `s1_regressions` 2 passed (0.76 s); `s1_review3` 7 passed (1.80 s) |
| `cargo test -j 2 -p rustmath-jupyter --lib -- taylor` (`rev4-semantics/jup_taylor.log`) | 6 passed (`test_taylor_engine`, `_order_checked`, `_leaf_text_cap`, `_simplify_of_kernel_sources`, `_display_linear`, `_shared_dag`) |

`target/release` was deleted after use, and the test executables of the finished runs in
`target/debug/deps` too (free disk fell under 4 GB).

## 2. Tests of the first step

Compared with the harness copy of the first step (`protoharness/work/src`):

- Byte-identical: `tests/s1_common/check.rs`, `tests/s1_common/eval.rs`, `tests/data/s1_fuzz.tsv`,
  `tests/data/gen_s1_fuzz.py`, `tests/data/s1_gp.py`.
- `tests/s1_regressions.rs` (mtime 20:02): with comment lines, `#[path]` lines and whitespace
  removed, the code differs from the harness copy in one character, a trailing comma that rustfmt
  added (`Ok(())},);`). No assertion and no expected outcome changed.
- `tests/data/s1_singular_points.tsv` (680 lines): its first 106 450 bytes are the harness copy,
  byte for byte, and it starts with round 2's file (`fix2/s1_singular_points.tsv.before`).
- `tests/s1_common/mod.rs` differs only in the doc paragraph about the removed feature gate.
- `s1_fuzz_data.rs` (18:08) and `s1_singular_points.rs` (21:25) have not changed since round 3
  checked them; round 3 of the fix-ups added only `tests/s1_review3.rs`.

## 3. The blockers and minors of round 3

| item | status | evidence (release, 8 MB thread, `taylor_with_budget` with 1e8; `rev4-semantics/r3b/repro.txt`) |
|---|---|---|
| resources B1 (float leaves) | fixed | `g(4.9e-324 x 11 000)/(1-x)` at 0: order 189 `LimitExceeded(TreeText)` in 0.012 s, 10 MB; order 36 Ok, Display 133 MB in 0.095 s (VmHWM 136 MB), `simplify`, `substitute`, `==` at once; order 37 refused. The `1.0e-308` input at 189: refused in 0.012 s. |
| resources B2 (`simplify` of a result) | fixed | `((2^256+1)^4096*x)^4096` at 4096: Ok, 48 bytes, `simplify` 0.000 s. The geometric series at 20, 40, 100: `simplify` 0.000 s, Display at most 4 873 bytes. `exp(x + (2^256+1)^4096)` at 1000 and 5000: `simplify` 0.000 s and 0.005 s, Display 0.83 s (38 MB). `sin((2^254+1)^4096)/(1-x)` at 1000 and 5000: refused (`TreeText`) in 0.011 s. `exp(x + N^4096)` at 200 and 600: `simplify` 0.000 s. Resources m1's `exp(x+(10^300-1)^1050*7^130000)` at 312: refused in 0.021 s. The values of the accepted results are right: 1/(1 - K x) is the sum of K^k x^k with K the kernel. |
| semantics m1 (`big_o` at infinity) | done | `1/(x-1)` at oo to 3: `big_o()` = `((1)/x)^(4)` |
| semantics m2 (float underflow) | done | `1.0e-400`, `1.0E-400`, `0.1e-400`, `9.9e-325`, `2.4703282292062327e-324` (rounds to 0) and `1.0e-4000000000000000000000` are parse errors that name the literal; `0.0`, `0.000e-500`, `00.0e-400` are 0.0; `4.9e-324`, `2.4703282292062328e-324` (rounds up), `1.0e-320` keep their binary value (`pf.tsv`) |

## 4. My fuzzers

Driver `drv/r4.rs` (adapted from round 3's `r3.rs`; it adds `SIMP`, the gp form of
`simplify(to_expr())`), built against the current debug rlib (`r4_dbg`) and the release rlib
(`r4_rel`, built before `target/release` was deleted). Per row: `series_expansion` with the budget
1e8, `taylor_with_budget`, `Display` then `parse`, `simplify`, `big_o`, `coefficient`,
`laurent(0, n)`, `series_coefficients(n)`, `limit_via_series(.., 5)`. Runner `run4.py` and gp
helpers `h4.gp` (adapted from round 3's; new: `CMPX`, which compares two rational coefficients
exactly and the others to 1e-60 relative). gp 2.15.4 at 300 digits.

### 4.1 `gen4.py` (seeds 401 to 416, 9 600 rows)

Four classes, none of them review 9's random trees at six rational points or the templates of
rounds 1 to 3:

- A: exact rational expansions at 0 to orders 8 to 30: compositions of the elementary functions
  on polynomials without constant term, Taylor-prefix differences (`(sin(x) - (x - x^3/6))/x^k`),
  poles from dividing by `x^k`, `sin(x)^k`, `tan(x)^k`, `(exp(x)-1)^k`, integer powers -3 to 3.
- B: constants on a branch cut, away from the branch point: asin, acos, atanh at real b0 with
  |b0| > 1 (2, -5/3, sqrt(5), -1-sqrt(2), pi, -e, 1+sqrt(3)), acosh at b0 < 1, atan and asinh at
  i t with |t| > 1, log, sqrt, `u^(1/3)` and `u^(1/2+I)` at b0 < 0; the argument reaches b0
  through a sum, a product, a composition or the point itself (`acosh(x)` at -sqrt(3)).
- C: a singular subterm S(u), u vanishing at a in {0, 1, -2, 1/3, 5/2} (40 shapes: poles,
  removable singularities, branch points, `exp(1/u)`, `u^u`, `u log u`, `sqrt(u^2)`, `abs`, `sign`,
  `(u^2)^(1/2)`, `exp(log(u))`, `(1+u)^(1/u)`, `u^0`, `0*log(u)`, ...) at a random place of a random
  outer tree.
- D: expansions at infinity of random trees of rational functions and 37 atoms in x and 1/x.

| outcome | rows |
|---|---|
| equal to gp on every coefficient (A 2 655, B 2 669, C 900, D 551; 26 059 rational coefficients compared exactly, on 2 242 rows); renderings `taylor_with_budget`, re-parsed `Display` and `simplify` equal at a + 1/997 | 6 775 |
| the same, renderings not compared (`Display` over 200 KB) | 26 |
| `NotLaurent`, gp fails | 2 357 |
| `NotLaurent`, gp has a value: all an even-valuation argument at a branch point (`asin(-1 + u^2)`, `acos(-1 + 2u^2)`, ...), where gp takes a formal root | 113 |
| `UndecidedZero`, gp has a value: 172 are 1/f(b0) with f(b0) complex on a cut (`1/acos(sqrt(5) + ...)`, `1/log(-e + ...)`), 8 complex constants in class A | 180 |
| `Unsupported` (abs or sign of non-rational coefficients, `cot`, gamma) | 64 |
| `BudgetExceeded`, gp has a value (class A at orders 20 to 30; release at most 0.65 s) | 40 |
| gp `e_TYPE2` on `u^0` (the engine gives 1, or c + 1; checked by hand) | 38 |
| `DivisionByZero` 4, `PrecisionExhausted` 2, `UndecidedZero` 1, gp fails | 7 |
| a coefficient different from gp, an engine value where gp fails, a panic, a timeout | 0 |

Limits: 1 678 `Finite` values of `limit_via_series` equal gp's constant term; 28 more are on the
`u^0` rows (all `Finite(1)`, `Finite(4/3)`, `Finite(3)`, checked by hand); none is `Finite` at a
pole. `laurent(0, n)` is `ValuationBelowMinimum` and `series_coefficients` `Pole` on all 3 134 rows
whose first stored exponent is negative and decided, `UndecidedZero` on the 10 whose leading
coefficient is undecided (as the `first_exponent` doc says); `Ok` on the 7 890 others, with dense
coefficients equal to `coefficient(k)`; `coefficient(precision)` is `None` on every row.

### 4.2 `gen4b.py` (seeds 501 to 507, 4 200 rows)

- E: expansions at irrational and complex points (sqrt(2), pi/3, e, log 2, I, 1+I, 2^(1/3), the
  golden ratio, -sqrt(3)) of random analytic trees, and poles placed there (`1/(sin(x) - sqrt(3)/2)`
  at pi/3, `1/(x^2 - 2)^2` at sqrt(2), `1/(exp(x) - 1)` at 2 pi i, `x/(sinh(x) - I)` at i pi/2,
  `log(x)/(x - e)` at e, `tan(x)` at pi/2, ...).
- F: the owner's two rules at random (section 5).

E (2 042 rows): 1 807 equal to gp (one more with a `Display` over 200 KB, not re-parsed); 164
`UndecidedZero` (complex constants); 4 `NotLaurent` where gp fails too; 66 differ from gp, all
`1/sin(x)^2` at pi, where gp's float `sin(Pi)` is 1e-308 instead of 0. The engine is right there:
`(1/sin(x)^2)*x` at pi has the terms pi h^-2 + h^-1 + pi/3 + h/3 + pi h^2/15 + h^3/15, which is
gp's `(Pi + h)/sin(h)^2` (sin(pi + h) = -sin(h); `fz/hE.drv`, row `E502_217`).

## 5. The owner's two rules

`gen4b.py` class F: f in {asin, acos, atanh, acosh, atan, asinh, log, sqrt, `u^(1/3)`,
`u^(-1/2)`} at each of its branch points (16 pairs), at a in {0, 1, -2, 1/2}, inside a random outer
function (`f + x`, `exp(x)*f`, `f^2`, `1/(3 + f)`, `f*f`, `sin(f)`).

- F1, an exact-constant argument in 11 shapes (`c + 0*G`, `c*x^0`, `c + (x - x)*G`,
  `c*sign(x^2+1)`, `c + 0^(x+5)*G`, `c*abs(x^0)`, `c + asin(x - x)*G`, `c*(x-a)/(x-a)`,
  `c + ((x-a)^2 - (x-a)^2)`, `c*(1 + 0*G)^3`, `c + ((x+1)^2 - x^2 - 2x - 1)*G`, with G a random
  analytic tree). Each row is compared with the engine on the reference text, the same outer
  function of f(c) written as a literal (`check_f.py`, `check_f2.py`).
- F2, a non-constant argument `c + k (x-a)^m h(x)` with m in 1..4, k in {1, 2, -1/3, I, 1+I}, h(a)
  nonzero.

| rows | outcome |
|---|---|
| F1 at a regular branch point (asin, acos, acosh at +-1, asinh at +-I, sqrt, cube root at 0): 675 | 452 the same terms as the reference; 154 the same values (gp, 1e-200), the kernel printed from another source node; 69 `UndecidedZero`, all `1/(3 + f(c))` (minor m1) |
| F1 at a pole of f(c) (atanh at +-1, atan at +-I, log at 0, `u^(-1/2)` at 0): 422 | 422 `DivisionByZero`, the same as the reference |
| F2: 1 061 | 986 `NotLaurent`; 75 `UndecidedZero` (atan and atanh, whose integrand's leading coefficient, such as 4i(2 + sin 1), the constant field cannot decide); 0 values; 0 `DivisionByZero` |

Hand probes (`br.tsv`, 76 rows; `lim.tsv`, 40 rows): `asin(sin(1)^2 + cos(1)^2 + x)` and the five
others at an undecidable branch point are `UndecidedZero`, never `NotLaurent` or a value;
`asin(cos(x^2))`, `acos(cos(x))` at 0, `asin(sin(x))` at pi/2, `acosh(cosh(x))` at 0 are
`NotLaurent`; at regular points `acos(cos(x))` at 1, `asin(sin(x))` at 2, `acosh(cosh(x))` at -2
give the slopes 1, -1, -1 (the real functions x, pi - x, -x); `log(exp(x))` at 4i keeps the kernel
`log(exp(4i))` (it is 4i - 2 pi i); `(x^3)^(1/3)` at -2 has the principal constant `(-8)^(1/3)`.
`(0*x)^r`: 0 for r = 1/3, 1/2 + I, x + 1, pi - 3, 1 + (sin(1)^2 + cos(1)^2 - 1); `DivisionByZero`
for -1/3, -1/2 + I, x - 1, 3 - pi; `NotLaurent` for I and x; `UndecidedZero` for
sin(1)^2 + cos(1)^2 - 1. `(1+x)/(1+x)`, `exp(x - x)`, `cos(x - x)` as arguments are
`PrecisionExhausted`, inside the scope that `s1_common/mod.rs` states. The limit path:
`sign(x^2)` is `Finite(1)`, `(x - x)^x` at 1 is `Finite(0)`, `0^x` at -1 and `(x-x)^(x-1)` at 1 are
`Unknown`, `(sin(x)^2 + cos(x)^2 - 1)/x^10` is `Finite(0)` at every order (the bound O(x) is
rigorous: every coefficient below the precision is an exact 0).

The matrix (`tests/data/s1_singular_points.tsv`) has the 23 (function, branch point) pairs of the
engine's inverse functions, with exact-constant and non-constant rows for each and a row at
infinity, which is what the owner asked for. Its exact-constant rows use f(c) in a sum or alone;
no row inverts it (m1).

## 6. Constant traps (`k/k.tsv`, 117 constants K, 468 rows)

Each K was put in `1/(x + K)`, `x/K`, `asin(1 + K + x)` and `sqrt(x^2 + K)` at 0. 100 K are exactly
0, 17 are nonzero. Among the zeros: `sqrt(-2)*sqrt(-3) + sqrt(6)`, `(-8)^(1/3) - 1 - I*sqrt(3)`,
`sqrt(-1/4) - I/2`, `1/sqrt(-4) + I/2`, `log(-1) - I*pi`, `exp(2*log(-3)) - 9`, `asin(sin(2)) - pi
+ 2`, `atan(tan(2)) - 2 + pi`, `cosh(acosh(1/2)) - 1/2`, `((-8)^2)^(1/6) - 2`, `sin(pi/12) -
(sqrt(6) - sqrt(2))/4`, `zeta(-2)`. Among the nonzeros, the traps of a careless fold:
`sqrt(-2)*sqrt(-3) - sqrt(6)`, `sqrt(1/(-4)) - 1/sqrt(-4)`, `log((-2)*(-3)) - log(-2) - log(-3)`,
`log(exp(4*I)) - 4*I`, `asin(sin(2)) - 2`, `atan(tan(2)) - 2`, `acos(cos(-1)) + 1`,
`asinh(sinh(3*I)) - 3*I`, `((-1)^3)^(1/3) + 1`, `((-8)^2)^(1/6) - (-8)^(1/3)`, `(-4)^(1/4) -
sqrt(2)*I`, `log(-e) - 1`.

Result: no zero decided nonzero, no nonzero decided zero. 37 zeros are decided (the pole 1/x,
`DivisionByZero`, `NotLaurent`), the other 63 are `UndecidedZero` (and `x/K` is `x/K`, the documented
convention for a constant the input divides by). 5 nonzeros are decided, with the right values
(`1/(x + asin(sin(2)) - 2)` starts 1/(pi - 4)); the other 12 are `UndecidedZero`. `q.tsv` (30 rational powers of negative and
complex bases) agrees: `(-8)^(2/3)` stays the principal power, never 4.

## 7. `simplify` after round 3

The memo is keyed by the address of a node held by more than one `Arc`, while the input is borrowed,
so equal keys mean the same node, and a node's value is a function of the node: the memo cannot
change a value. The fold guard only declines to fold. Against HEAD's rlib
(`librustmath_symbolic-4cbaba7850391989`, `drv/simp.rs`, 16 texts at the guard's edge): the same
output for `(-3)^5`, `(-2)^4096`, `(-1/3)^(-7)`, `(2^254+1)^4096 - (2^254+1)^4096` (0 in 0.22 s both),
`0^4096`, `(-1)^4095`; HEAD folds `(2^255+1)^4096 - (2^255+1)^4096` to 0 and
`((2^255+1)^4096)/((2^255+1)^4096)` to 1 in 0.23 s, the port keeps both (177 characters), the
documented deviation of fix-up round 3 (section 5, item 1). The only workspace callers that test
`simplify(..).is_zero()` are two MF1 test assertions in `rustmath-manifolds/src/lie_derivative.rs`
(341, 374), which pass (round 3 ran them). On the 8 582 fuzz rows whose renderings were compared
with gp, `simplify` of the result has the result's value at a + 1/997.

## 8. Release against debug

`fz/all_g.tsv` (9 600 rows) and `fz/all_h.tsv` (4 200 rows) through `r4_dbg` on 64 MB and 2 MB
threads and `r4_rel` on 8 MB threads (`all_h.tsv` also holds the 1 097 F1 reference rows): 0 rows
differ in any field but the time (the 2 MB run of `all_g` was made without the limit path, so
that field is left out of its comparison), 0 panics, 0 stack overflows. The slowest row took 3.72 s
in debug and 0.70 s in release (the series step; the T field of the driver).

## 9. Minor issues and notes

### m1. f(c) at a branch point is opaque to the zero test once it must be inverted

The owner's first rule gives arcsin(1), arccos(-1), arccosh(-1), arcsinh(I) as kernels. These have
no enclosure: the ball of arcsin and arccos needs |m| + r < 1 (`series_ball.rs:351`), arccosh
needs lo > 1 (`:386`), and arcsinh(I) is not free of I. A single kernel is decided nonzero by its
class, but a sum with it is not. So the value of the rule cannot be inverted, logged or rooted:

| input, at 0 | outcome | value |
|---|---|---|
| `1/(3 + asin(1 + 0*x))`, `1/(3 + asin(-1*x^0))` | `UndecidedZero(3 + arcsin(1))` | 1/(3 + pi/2) |
| `1/(x + 3 + asin(1))`, `log(x + 1 + asin(1))`, `sqrt(x + asin(1) + 1)`, `(x + 3 + asin(1))^(1/2)`, `1/(exp(x) + asin(1))` | `UndecidedZero` | regular |
| `1/(x + 2 + acos(-1))`, `1/(x + 1 + acosh(-1))`, `1/(x + 1 + asinh(I))` | `UndecidedZero` | regular |
| `1/(x + asin(1))`, `1/(x + 3 + asin(1/2))`, `1/(x + atan(1) + 1)`, `1/(x + 3 + I*pi)` | values | right |

69 of my F1 rows are of this kind. It is honest (undecided, never decided), and HEAD gives no right
answer to compare with (its taylor of `log(x + 1 + asin(1))` has `0/0` terms; its parser rejects
`1/(...)`), so it is below the bar. But the owner asked for "a value for the first", and these
inputs are regular. Suggested: fold the exact values asin(+-1) = +-pi/2, acos(-1) = pi,
acosh(-1) = i pi, asinh(+-i) = +-i pi/2 (Laurent polynomials in pi and I, which `const_status`
decides), or let the ball of arcsin and arccos take an exact +-1 argument (radius 0) to MPFR's
value. Add matrix rows that invert the value, such as `1/(3 + arcsin(x^0))` (gp:
`1/(3 + asin(1))` = 0.218780258078402138...). (`u.tsv`, `v.tsv`.)

### m2. abs and sign of non-rational real coefficients are refused where HEAD was right

At a regular point, `abs(x - sqrt(2))` at 0, `abs(sin(x))` at 1, `abs(cos(x))` at 2,
`abs(x + pi)*x` at 0, `sign(x - pi)` at 0 and `sign(log(x))` at 3 are `Unsupported` ("abs of a
series with non-rational coefficients"). HEAD's taylor gives the right series on all six (checked
by hand: `abs(cos(x))` at 2 is -cos 2 + sin 2 (x-2) + cos 2 (x-2)^2/2; `abs.tsv`). The refusal is
the frozen design's scope (§2.4, the abs/sign row of the function table) and is typed, so I do not
count it under S1_BAR (b), in the same way as the intended refusals of `tests_report.md`. HEAD's abs
is not reliable either (`abs(x)` at 0 gives 0). Suggested handoff: decide the sign of c0 with
`real_sign` and accept the series when every coefficient has a real enclosure; then
abs(b) = sign(c0) b and sign(b) = sign(c0). A row for the HANDOFF next to the abs/sign convention.

### Notes (no change needed for the bar)

- Complex constants without an explicit I (asin(2), log(-2), sqrt(-2), acosh(1/2)) have no real
  enclosure, so 1/f(b0) on a cut and the atan/atanh leading coefficients above are undecided
  (172 + 75 + 164 rows here). Round 1's m3 handoff covers it.
- The documented convention for an undecided divisor still holds: `x/(asin(1) - pi/2)` at 0 is the
  value `x/(asin(1) - pi/2)` although the divisor is 0; with m1's fold it would become
  `DivisionByZero`.
- gp artefacts met here, for the next reviewer: float `sin(Pi)` (the `1/sin(x)^2` rows), gp's
  `acos(-1 + O(h^n))` dropping pi (round 3), and `e_TYPE2` on a series to the power 0.

## 10. Reproduction

```
R=/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/s1impl/rev4-semantics
cd $R/drv && ./build.sh r4.rs dbg r4_dbg                 # (rel needs target/release; r4_rel kept)
cd $R/fz && python3 -I ../gen4.py 401 600 g401 && python3 -I ../run4.py g401
cd $R/fz && python3 -I ../gen4b.py 501 600 h501 && ../drv/r4_dbg ser h501_ref.tsv > h501_ref.drv \
  && python3 -I ../run4.py h501 && python3 -I ../check_f.py h501 && python3 -I ../check_f2.py h501
cd $R && NOLIM=1 ./drv/r4_dbg ser k/k.tsv                  # constant traps (k/klist.txt)
cd $R && NOLIM=1 ./drv/r4_dbg ser u.tsv; NOLIM=1 ./drv/r4_dbg ser v.tsv   # m1
cd $R && NOLIM=1 ./drv/r4_dbg ser abs.tsv                  # m2 (HEAD: drv/head_taylor.rs)
cd $R && ./drv/p4_rel <round-3 input> 0 <order> 8388608 tw,disp,simp   # round 3 blockers (r3b/)
```

The five comparison outputs of section 8 (`fz/all_*.rel`, `.dbg`, `.dbg2m`) were deleted after
the comparison (rerun `r4_dbg ser fz/all_g.tsv 2097152` and `r4_rel ser fz/all_g.tsv 8388608`).
Binaries deleted to save disk (rebuild with `drv/build.sh`, or with rustc against HEAD's rlib
`librustmath_symbolic-4cbaba7850391989.rlib` for the HEAD probes): `lim_dbg`, `simp_port`,
`simp_head`, `head_taylor`. Kept: `r4_dbg`, `r4_rel`, `p4_rel`. The `.gp` batch files were deleted
(`run4.py` writes them again).
