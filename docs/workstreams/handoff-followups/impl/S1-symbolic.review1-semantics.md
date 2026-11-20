# S1 implementation, review 1, lens: numeric and series semantics

Reviewer of the uncommitted S1 change in `/home/user/RustMath` (branch `claude/gracious-gates-e6b6l7`,
HEAD 8a0b310). No repository file was edited. Scratch: `fu2/s1impl/rev1-semantics/`.

## Verdict

Approve in this lens. I found no blocker under S1_BAR. The tests pass. Five minor issues follow.

- No wrong value in 4000 inputs of my own fuzzer against gp at 300 digits (3711 equal, 4 differences,
  each a proven gp artefact), in 50 hand-built cancellation and branch inputs, or in 80 zero-test inputs.
- The owner's two rules hold on every branch-point input I built: an exact constant is evaluated, a
  non-constant argument at a branch point is `NotLaurent`, and `DivisionByZero` appears only where the
  input divides by an exact zero or applies a function at an exact pole of a constant.
- The tests of the first step are unchanged in their expected outcomes and pass.
- One gap of rule 1 (m1): a constant equal to +-i or +-1 in a form the constant field cannot decide
  (`sqrt(-1)`, `I*sqrt(-1)`) gives a kernel at a pole. HEAD gives the same output, and the frozen
  design lists this as a known cost (§7 item 10), so it is below the bar.

## 1. Tests

| command | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic --no-fail-fast` (debug, my run) | unit 675 passed, 3 ignored; `s1_fuzz_data` 7; `s1_regressions` 26 passed, 2 ignored (release budget tests); `s1_singular_points` 4; doc 116 passed, 11 ignored. 0 failed. Log: `rev1-semantics/debug_tests.log`. |
| review 9's `fuzz.py` seed 1 (250 inputs) on the implementer's release driver `fuzzdrv/s1drv_rel` | 161 equal to gp, 32 GPERR, 57 refusals, 0 DIFF, 0 crash. The four columns are identical to the implementer's `fuzzdrv/run/fz_1.out`. |
| my fuzz rows through `s1drv_rel` (release) against my debug driver | the same outcome kind on 937 of 937 finite-point rows. |

`s1drv_rel` was built at 18:48:48 against the release rlibs. The last source change in the crate is at
18:46:42, so it is the current code. I did not rebuild `target/release` (disk at 3.7 GB free when I
checked; I deleted six superseded test executables in `target/debug/deps`, 0.2 GB).

### Tests of the first step

- `tests/data/s1_singular_points.tsv` and `s1_fuzz.tsv` are byte-identical to the round-3 snapshots
  (`fu2/s1impl/r3/sp_new.tsv`, `fz_new.tsv`) and to the copies in `protoharness/work/src/data/`.
- `tests/s1_common/check.rs`, `eval.rs`, `api.rs` are unchanged since the first step (`check.rs` and
  `eval.rs` equal the protoharness copies; `api.rs` has the mtime 17:45, before the first step ended).
  `mod.rs` differs only in the doc paragraph about the removed gate. `gen_s1_singular.py` differs only
  in two comments (17:51, inside the first step).
- `tests/s1_regressions.rs` against the protoharness copy (17:46, gate line dropped): the `--features`
  flag removed from three doc lines, one doc comment extended (the 39 intended refusals), one section
  number, and a rustfmt line break of one `timed_release` call. No expectation changed.
- `s1_singular_points.rs` and `s1_fuzz_data.rs` have no snapshot. They hold no gate line, their test
  names and row filters match `tests_report.md`, and all their expectations live in the unchanged
  data files.
- Library tests in `series.rs` and `lib.rs`: the removed assertions are structural comparisons with
  `known_series::exp` and `!is_constant()`. Each is replaced by a value check that is at least as
  strong (rational coefficients 1, 1, 1/2, 1/6; gp values at three rational points). No assertion is
  weaker.

## 2. My fuzzer (different from review 9's)

`rev1-semantics/fz/fzsem.py` (generator), `run.py` (driver and gp), `gphead.gp`. Driver
`rev1-semantics/drv/sem.rs`, built in debug against the crate's rlibs: `series_expansion` with budget
2e7 on a 64 MB thread, and `limit_via_series` at order 5.

How it differs from review 9's harness:

- Points: 0, 1, -2, 1/2, pi, pi/2, pi/3, pi/6, sqrt(2), I, 1+I, -I/2, log(2), e, 2^(1/3), and infinity.
  Review 9 used six rational points.
- Inputs: pole constructions (`E/u^k`, `cos(u)/sin(u)`, `1/u - 1/sin(u)`, `1/log(1+u) - 1/u`,
  `1/(exp(u)-1)`) and cancellations of known Taylor prefixes (`(f(u) - prefix)/u^k` for exp, sin,
  cos, atan, asin, tan, sinh, atanh, log1p, sqrt1p, with k - 1, k or k + 1), u a random series of
  valuation >= 1 at the point.
- Every coefficient from the lowest exponent of the engine or gp up to n is compared. Review 9's
  harness compared exponents 0 to n only, so a wrong principal part would have passed it.
- The gp reference is built node by node, and leading coefficients below 1e-250 are dropped at each
  node (gp's float pi makes sin(Pi) = 1e-300). Complex points are given to gp inexact (see m5).
- The limit path is checked: `Finite(q)` must equal gp's limit.

Results, 8 seeds x 500 inputs (`rev1-semantics/fz/f101..f108.res`, `fuzz2.log`):

| class | rows |
|---|---|
| equal to gp to 1e-60 (all coefficients, negative exponents included) | 3711 |
| engine value, gp fails (`1/sin(u)` at pi/3 with a truncated factor `cosh(x-x)`: gp "impossible inverse"; the engine's `1/h - h/3 - h^3/45` is right) | 1 |
| `NotLaurent`, gp fails | 208 |
| `NotLaurent`, gp has a series (gp picks a root of a series of even valuation: `((x+2)^2)^(3/2)` at -2, `sqrt(sqrt(2)/(x-pi/3)^2)`, `acos(cosh(F))` with F of valuation 1): rule 2 or the non-integer power rule, correct | 6 |
| `DivisionByZero`, gp fails (an exact zero divisor `(x-x)`, `((x-p)-(x-p))`, `(x-x)^(-2)`, `log(x-x)`, or `atan(I)`, `atanh((x-p)/(x-p))` at an exact pole) | 10 |
| `UndecidedZero` (honest; see m3), `PrecisionExhausted` | 59, 3 |
| DIFF | 4, all gp artefacts (below) |
| a `Finite` limit that disagrees with gp | 0 |
| panic, crash, timeout | 0 |

The four DIFF rows:

- `acosh(exp(x)*cos(x))` at pi, c1 differs by 2. The engine's c1 = e^pi/sqrt(e^(2 pi) - 1) is right:
  the branch through c0 = log(e^pi + sqrt(e^(2 pi) - 1)) + i pi has derivative 1/sinh(c0) =
  -1/sqrt(e^(2 pi) - 1). gp's series takes (u^2 - 1)^(-1/2) and the other sign left of -1
  (`acosh(-3/2 + h + O(h^3))` gives +0.894 h with constant 0.962 + i pi). The first step's `s1_gp.py`
  already works around this (its `ACOSH`).
- `tan(x)` at pi/2: gp's cos(Pi/2) is 1e-301, not 0. The engine's `-1/h + h/3 + h^3/45` is right.
- `sqrt((x+I)^(-1))` at 1 and `asinh(((x+I)*x)^3)` at 1: gp works at about 19 digits on a series with
  an exact complex constant term (`sqrt((1+2*I) + h + O(h^70))` has 19 significant digits). With the
  point given as `1.` both rows agree to 2.8e-309 and 1.8e-307.

## 3. Hand-built inputs

`rev1-semantics/deep.tsv` (50 rows), `z.tsv` (80), `br.tsv` (60), `g.tsv`, `e.tsv`, `u.tsv`, `p.tsv`;
outputs reproduced with `drv/sem_dbg FILE`.

- Deep cancellations and high poles, compared with gp: `(exp(x^k)-1-x^k)/x^(2k)` for k = 10, 50, 200,
  1000 (1/2); `1/sin(x)^30`; `1/x^30 - 1/sin(x)^30`; `1/(x-1)^20 - 1/sin(x-1)^20` at 1;
  `(sin x - x)^5/x^15`; `(sin(sin(sin x)) - x + x^3/2)/x^5`; `(exp(x)-1)^(-3) - x^(-3)`;
  `(log(1+x)/x)^(1/x)`; `cos(x)^(1/x^2)`; `exp(sin x)/x^3`; `(1+x)^(1/x)`; `x^x` at 1. All equal to gp.
- Branch values on cuts: `log(x)` at -1, `sqrt(x)` at -1 and -4, `x^(1/3)` at -8, `acos(x)` at 2,
  `asin(x)` at -2, `atanh(x)` at +-2, `log(-x)` at 1, `(x^2)^(1/2)` at +-1, `sqrt(x^2-1)` at 0: the
  principal values of the stated formulas, with consistent derivatives.
- At infinity: `(x^3+x)/(x^2-1)`, `x^2*(exp(1/x)-1-1/x)`, `x*log(1+1/x)`, `x*(sqrt(1+1/x)-1)`,
  `(1+1/x)^x` are right; `atan(x)`, `sin(x)`, `exp(-x)`, `abs(x)`, `sign(x)`, `x^(1/x)` are
  `NotLaurent`.
- `laurent`, `series_coefficients`, `series_with_big_o`: `ValuationBelowMinimum`, `Pole` and
  `UndecidedZero` exactly where the design says (`drv/api.rs`).
- The zero test never decided wrongly. It decides exact identities (`sin(pi)`, `cos(pi/3) -
  sin(pi/6)`, `sqrt(8) - 2*sqrt(2)`, `(pi+I)*(pi-I) - pi^2 - 1`, `log(e) - 1`, `log(exp(2)) - 2`,
  `tan(pi/3) - sqrt(3)`) and transcendence (`pi - 355/113`, `e^pi - pi^e`). It leaves undecided the
  true zeros it cannot prove (`sqrt(2)*sqrt(3) - sqrt(6)`, `log(4) - 2*log(2)`, `exp(log(2)) - 2`,
  `atan(1) - pi/4`, `sqrt(-4) - 2*I`, `0.5 - 1/2`, `exp(1) - e`) and does not use log(exp(c)) = c for a
  non-real c (`log(exp(3*I*pi)) - I*pi` undecided).
- Exact-zero rules: `1/((x+1)^2-x^2-2*x-1)`, `1/(0*x)`, `0/(x-x)`, `(x-x)^(-1/2)`, `log(x-x)`,
  `1/(x-x+sin(pi))` are `DivisionByZero`. `(x-x)^(1/2)` is 0, `(x-x)^0` and `(x-x)^(x-x)` are 1,
  `(x-x)^x` is `NotLaurent`, `(x-x)^(x+1)` is 0, `(x-x)^I` is `NotLaurent`.
- Owner's rules on shapes the matrix does not hold: a constant term reached through a kernel identity
  (`asin(sin(pi/2+x))`, `asin(2*sin(pi/6+x))`, `asin(sqrt(2)*sqrt(2)/2+x)`, `acos(cos(x))`,
  `asinh(I*cosh(x))`, `atan(I*cosh(x))`, `atanh(cos(x))`, `acosh(cosh(x))`, `acosh(exp(x))`), the
  branch point at infinity (`f(1/x)` at 0 for all nine functions, `f(x)` at infinity), and exact
  constants in more shapes (`asin(x^2/x^2)`, `asin(sign(x)^2)` at 1, `acosh(sign(x))` at -1,
  `asin((x^3)^0)`, `asinh(I^5*x^0)`, `acos(-(x/x))`): every non-constant one is `NotLaurent`, every
  exact one is f(c) or `DivisionByZero` at arctanh(+-1) and arctan(+-I).
- The limit path gives `Finite` only for a rational constant term, and the right value each time (sin
  x/x at 0 is 1, at pi 0; `(1-cos x)/x^2` is 1/2; `x/sin(x)^2`, `1/x`, `1/x^2` are `Unknown`).

Parser and Display (`rev1-semantics/pd/`): 58 hand cases (`-2^2` is -4, `2^-3^2` is 2^-9, `2^-1/2`
is 1/4, `2/3^2` is 2/9, `-1/-2` is 1/2, `2^--2` is 4, ...) and 3000 random texts against Python's
reading of the same text: 2166 equal, 0 different, 0 round-trip failures of `parse(Display(e))`; the
25 parse errors are the texts Python cannot evaluate either. A differential run of `simplify` and
`expand` against HEAD (3000 random expressions, `pd/sfz.py`): `simplify` gives HEAD's value on every
row; `expand` gives the input's value where HEAD's differs on 306 rows (HEAD's expand-under-minus bug);
the 5 other rows are inf against NaN on inputs that divide by 0, the same in HEAD.

## 4. The singular-point matrix

The matrix has the 23 (function, branch point) pairs: arcsin, arccos, arccosh, arctanh at +-1;
arctan, arcsinh at +-I; log, sqrt and seven power families at 0. Each pair has 16 or 17 rows of (a),
8 of (b) and 2 of (c), with one expected outcome per row, and the coverage test enforces at least two
of each. That meets the owner's request for the finite branch points. Three gaps, none failing today
except the first:

1. Exact constants in a form the constant field cannot identify: `atan(sqrt(-1)*x^0)`,
   `atanh(I*sqrt(-1)*x^0)`. This is m1.
2. The branch point at infinity (an argument with a pole): `asin(1/x)` at 0 and so on. All nine are
   `NotLaurent` today.
3. Non-constant arguments whose constant term is the branch point only through a kernel identity
   (`asin(sin(pi/2+x))`, `atanh(cos(x))`, `asinh(I*cosh(x))`). `c*exp(x)` covers part of this. All are
   `NotLaurent` today.

I recommend adding rows for 2 and 3 as regression guards, and for 1 with the outcome m1 settles.

## 5. Minor issues

### m1. Rule 1 gives a kernel at an undecided pole of arctan or arctanh

`series_engine.rs:1156` (`atan_like`) returns `kapply(op, c)` for an exact constant c.
`series_const.rs:2578` refuses arctan only when c is the kernel I or -I, and arctanh only at the
rationals +-1. For a constant the field cannot decide, the result is `Ok` with a kernel that has no
value. Reproduced in debug (`rev1-semantics/p.tsv`) and release (`s1drv_rel`):

| input at 0 | port | proto13 (`rev7rt4/bin/rt2_rel`) | HEAD |
|---|---|---|---|
| `x + atan(sqrt(-1)*x^0)` | `Ok(arctan(sqrt(-1)) + x)` | `UndecidedZero(1 + sqrt(-1)^2)` | a series with `0/(1 + sqrt(-1)^2)` terms |
| `x + atanh(I*sqrt(-1)*x^0)` | `Ok(arctanh(I*sqrt(-1)) + x)` | `UndecidedZero(1 + sqrt(-1)^2)` | `atanh(I*sqrt(-1)) + x` |
| `x*atan(sqrt(-1)*x^0)`, limit path | `Finite(0)` | `Unknown` | `Finite(0)` |

The constant field does not reduce `sqrt(-1)^2` to -1, so 1 + c^2 is undecided. proto13 reported
that through the integrand. Rule 1 skips the integrand, so the undecided pole now reads as a value.
Why it is below the bar: HEAD gives the same limit (`Finite(0)`), and design §2.2 keeps f(c) a kernel
at a non-rational argument that may be a pole ("a known cost", §7 item 10). The pure constant
`x + atanh(sin(1)^2+cos(1)^2)` was already `Ok` in the design. Review 9 m2 put the same class below the
bar. Required change (minor): in the rule-1 branch, for arctan and arctanh at a non-rational c, run
the zero test on 1 + c^2 (or 1 - c^2). Give `DivisionByZero` when it is zero and `UndecidedZero` when
it is undecided. Add `atan(sqrt(-1)*x^0)` and `atanh(I*sqrt(-1)*x^0)` to the matrix with that
outcome.

### m2. The public docs say an undecided zero test is never guessed

`series.rs:9-12` says "An undecided test is reported (`SeriesError::UndecidedZero`), never guessed".
But a constant the input divides by is inverted as a kernel when its test is undecided
(`series_const.rs:1509`, "under the convention that the input is defined"). Examples:
`x/(sin(1)^2+cos(1)^2-1)` at 0 is `Ok`, and its limit path is `Finite(0)`. `1/(x + 1/(sin(1)^2 +
cos(1)^2 - 1))` gives c - c^2 x with c = sin(1)^2 + cos(1)^2 - 1 (`rev1-semantics/u.tsv`). This is the
design's convention, and HEAD gives `0/c` terms. Required change: state it in the module's list of
conventions, next to the generic-parameter rule. For example: a constant the input divides by is
taken as nonzero when the test cannot decide, so the result holds where the input is defined.

### m3. The numeric zero test does not reach complex kernels

Many clearly nonzero constants come back `UndecidedZero`: 55 of my 4000 rows. Examples:
`cos(-I/2)`, `cosh(1+I)`, `sinh(I)`, `arcsin(e)^3`, `arccos(e)`, `arctanh(2*pi)`,
`1 + arccosh(1/2)^2`, `arccosh(1+I)`, `2 + (-8)^(1/3)` (`fz/*.res`, class
`REFUSE_UndecidedZero_GPVAL`; `z.tsv` row z37). This is honest and in the design's scope (real MPFR
balls). The only effect is that inputs at complex points or on cuts are often refused. I would note
it in the HANDOFF next to §7 item 10.

### m4. Some exact poles are reported as `PrecisionExhausted`

`tan(pi/2+x-x)`, `1/sin(pi+x-x)`, `atan(I*cosh(x-x))` and `asin(cos(x)^2+sin(x)^2)` at 0 give
`PrecisionExhausted(131)` (`z.tsv`, `m4.tsv`). This happens because exp, sin, cos and tan of an exact constant return a
truncated series (rule 1 is optional there, by the scope in `tests/s1_common/mod.rs`). The refusal is
honest. The `DivisionByZero` doc in `series.rs` names "tan(pi/2) as a constant", which is true only
for the pure constant. No change is needed beyond, at most, a word in that doc line.

### m5. Two gp artefacts for future gp-referenced tests

- gp's series functions work at about 19 digits when the constant term is an exact complex number.
  `sqrt((1+2*I) + h + O(h^70))` is accurate to 19 digits at realprecision 300. Give gp the point
  inexact (`(1+2*I)*1.`), as my harness does for complex points.
- gp's `acosh` of a series left of -1 takes the wrong sign (see section 2). The first step's
  `s1_gp.py` already avoids it.

## 6. Reproduction

```
S=/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/s1impl/rev1-semantics
cd $S && ./drv/sem_dbg p.tsv                        # m1 (debug)
echo 'x+atan(sqrt(-1)*x^0)' > pin.txt && GPCHECK=1 ../fuzzdrv/s1drv_rel both pin.txt 0 2 8388608   # m1 (release)
cd $S/fz && python3 -I fzsem.py 101 500 f101 && python3 run.py f101    # one fuzz seed
cd $S && ./drv/sem_dbg u.tsv                        # m2
```

The driver sources are `drv/sem.rs`, `drv/api.rs`, `drv/pd.rs`, `drv/simp.rs`, `drv/headlim.rs`.
They are built with `rustc --edition 2021 -L dependency=/home/user/RustMath/target/debug/deps
--extern rustmath_symbolic=<rlib from rlibs_dbg.txt>`. HEAD's library is
`target/debug/deps/librustmath_symbolic-4cbaba7850391989.rlib` (built 15:43, before the port).
