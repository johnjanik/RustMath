# S1 request C-1, review round 1, semantics lens

Branch `claude/gracious-gates-e6b6l7`, HEAD 52708d7, the uncommitted changes of this run (the working
tree equals `c1/diff_after.patch`). The C-1 change alone is the difference between
`c1/diff_before.patch` applied to HEAD and the working tree: `c1-1-semantics/c1_interdiff.patch`
(`series_engine.rs`, `series_const.rs`, `series_ball.rs`, `series.rs`, `s1_gp.py`, `eval.rs`,
`gen_s1_fuzz.py`, `s1_fuzz.tsv`) plus the new `tests/s1_cuts.rs`, `tests/data/gen_s1_cuts.py`,
`tests/data/s1_cuts.tsv`. Scratch: `fu2/s1impl/c1-1-semantics/`. No file of the repository was
edited. The `__pycache__` that my first import of `s1_gp.py` wrote in `tests/data` was removed
(later runs set `sys.dont_write_bytecode`). `rustmath-misc/output.txt` is unchanged (md5
746308829575e17c3331bbcb00c0898b); no `.tmp_dir_*` exists. I built no release target;
`target/release` (213 MB, created 13:34 by another session) was left alone. The three test
executables I built for the C-4 sample were deleted (free disk 4.3 GB at the end).

## Verdict

No blocker under the narrowed bar. The tests pass. Approve, with seven minor items. Item m1 (free
parameters) needs the owner's decision on what a parameter is.

## 1. What I ran

| what | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic --no-fail-fast -- --test-threads=2` (debug, `sym_dbg.log`) | 903 passed, 0 failed, 35 ignored (unit 685, `s1_cuts` 9, `s1_depth` 14, `s1_finish1` 11, `s1_fuzz_data` 7, `s1_payload` 2, `s1_regressions` 26, `s1_review2` 2, `s1_review3` 7, `s1_review4` 4, `s1_review4_memory` 1, `s1_review5` 4, `s1_review5_memory` 2, `s1_review6` 4, `s1_singular_points` 9, doc 116): the report's numbers |
| `python3 -I -B s1_gp.py --two-sided s1_cuts.tsv s1_singular_points.tsv s1_fuzz.tsv` (`oracle_rerun.log`) | exit 0, the report's table exactly: cuts 47 value outcomes (12 selected), 0 fail, 35 refusal rows, 25 formal series wrong on a side; matrix 367 (0), 0 fail, 84, 17, 43 without a verdict; fuzz 394 (13), 0 fail, 29, 3 |
| `c1/before_dbg.log` (tests first) | 3 passed, 6 failed on the code before the rule, as reported |
| C-4 sample: `cargo test -j 2 -p rustmath-jupyter --lib`, `-p rustmath-calculus`, `-p rustmath-manifolds --lib` (`jup_lib.log`, `calc.log`, `manifolds.log`) | 120 passed 3 ignored; 122 + doc 62; 308 passed 11 ignored: the report's counts |
| my probe driver `drv/p.rs` (rustc against the current debug rlib; `series_expansion` with the REPL budget, `taylor_with_budget`, `limit_via_series`, 8 MB thread) and two-sided checker `twoside.py` (gp 2.15.4, realprecision 200, the engine's own terms S against gp's scalar f at P +- 1e-12, or 1/(+-1e-12) at infinity) | 5 592 inputs, section 2 |
| HEAD-before-S1: `drv/h.rs` against `fin1-resources/headlib_dbg/librustmath_symbolic_head.rlib` (8a0b310's source, HEAD's `taylor`) | section 4 |

## 2. Hunting for an argument that leaves a cut and still gets a value

Inputs (`rows0.tsv` to `rows6.tsv`, `rr1.tsv`, `rr2.tsv`; generators in `gen/`):

- `rows1.tsv` (2 366): each of the eight branch functions at 4 to 9 constant terms on each of its open
  cuts (rationals, pi, sqrt(2), log(1/2), sin(4), cos(2), sin(1), I*sqrt(2), ...), plus 16 real
  directions (x, x^2, sin(x), cos(I*x)-1, sin(I*x)^2, x/I^2, I*x*I, (I*x)^2, x*abs(I),
  x*log(-1)/I, ...), 18 imaginary ones (I*x, sin(I*x), x*sqrt(-1), x*(-1)^(1/2), x*log(-1),
  (I*x)^3, tan(I*x), x*arcsin(2*I), x*arctan(I/2), x*sinh(I), erf(I*x), x*exp(I*pi/2), ...) and 16
  complex ones ((1+I)*x, exp(I*x)-1, x+I*x^2, x*exp(I), x*log(-2), x*arcsin(2), x*(-8)^(1/3),
  x*log(I), x + I*x - I*x, x*(1+I)*(1-I), x*(1+I)^2, ...); and 216 powers of (-1, -2, -pi) +
  direction to 12 exponents (1/3, 2/3, -1/2, pi, I, 1/2+I, sqrt(2), log(2), x, x^2, 1+x, I*x).
- `rows2.tsv` (91): truncation (`log(-1 + x + I*x^9)` at order 2, `log(-1 + x^2 + I*x^30)` at order
  3, `log(-1 + I*(sin(x) - x))` at orders 2 and 5, `x/(1 - I*x^8)`), nesting, and 60 coefficient
  constants whose class is decided structurally, by enclosure, or not at all (`log(-1)^2`,
  `sqrt(-2)*sqrt(-3)`, `((-2)^(3/2))^2`, `(-1)^I`, `I^I`, `arcsin(sin(5))`, `arccosh(1/sin(1))`,
  `sqrt(sin(1)^2 + cos(1)^2 - 1)`, `abs(1+I)`, `gamma(I)`, `erf(I)^2`, ...).
- `rows3.tsv` (103): other points (`log(x)` at -1 and -pi, `log(x^2)` at I and 2*I, `arctan(x)` at
  2*I, `arccosh(x)` at 0, -2, 1/2, `log(x)` at -1 + I, `log(cos(x) + I*sin(x))` at pi, ...),
  infinity (`log(-1 + I/x)`, `arctan(2*I + 1/x)`, `log(-x/(x+I))`, `arccosh(1/(x+I))`, ...),
  branch functions of branch functions (`log(-1 + I*(arctan(2*I + I*x) - arctan(2*I)))`,
  `arctan(I*arccosh(2 + x))`, `log(-1 + log(-1 + x) - log(-1))`, ...), variable and disguised
  integer exponents (`(-2 + I*x)^x`, `(-1 + I*x)^(sqrt(4))`, `(-1 + I*x)^(sin(1)^2 + cos(1)^2)`,
  `(-1 + I*x)^(x - x + 1/2)`, ...), free parameters.
- `rows4.tsv`, `rows6.tsv`: parameters, HEAD comparisons. `rows5.tsv`: non-integer powers of
  arguments with a pole at infinity (all refuse as branch points).
- `rr1.tsv`, `rr2.tsv` (3 000, seeds 1 and 2): random. A branch function at a random cut constant plus a
  random expression that vanishes at a random point (0, 1, -1/2, 2, infinity), with the cut-sensitive
  constants of the lists above, nested once at random (the inner part is g(c0 + d) - g(c0)), a random
  outer power or second summand.

Outcomes: 1 799 values, 3 734 `NotLaurent`, 59 `UndecidedZero`, no panic. The two C-1 messages split
as 1 924 "leaves a branch cut" and 181 "cannot decide" on the random rows.

Every value without a free parameter was checked two-sided. All pass, after three explained cases:

1. On-cut cancellation in gp, not in the engine. 33 random rows and `n2`, `n3`, `n11`, `n15`,
   `n16` of `rows3.tsv` "failed" by 2 pi I or similar on both sides. In each, an inner g(c0 + d) -
   g(c0) is exactly real (or exactly imaginary) by the on-cut conventions (`log(-1 + x) - log(-1)`
   is log(1 - x); `arcsin(2 + x) - arcsin(2)` is -i(acosh(2 + x) - acosh(2)); `arctan(2*I + I*x) -
   arctan(2*I)` is imaginary), and gp's floats leave an imaginary residue near 1e-200 of either
   sign, which puts the outer function on either side of its cut. With each branch argument in F
   snapped to its axis when the other part is below 1e-120 (`CLEAN=1 python3 -I twoside.py ...`,
   1e-120 is far below any true part at t = 1e-12), all 839 random values and all of `rows3.tsv`
   pass (`rcc1.txt`, `rcc2.txt`, `chk3c.txt`). `oncut.gp` checks five of them with real() of the
   inner part: errors 6.7e-50, 4.2e-50, 2.6e-51 at t = +-1e-12 (n2, n15, n16) and a t^2 residue of
   the truncation only (n3, n11).
2. `log(-1 + I/10^50 + I*x)` at 0 (`t33`): c0 is off the cut by 1e-50, so the function is analytic
   at 0 with real radius 1e-50. The series is its Taylor series: it passes at t = +-1e-60.
3. `log(-1 + x*abs(1+I))` at 0 (`t49`): the engine reads abs(1+I) as the modulus (CBall,
   `KCLS_REAL`); `s1_gp.py`'s scalar `AB` reads it as sign(Re)*(1+I). With gp's `abs` it passes
   (1.0e-48 both sides). See m5.

4. Values with a free parameter (4 rows of `rows3.tsv`, 10 of `rows4.tsv`, 1 of `rows6.tsv`) have no
   numeric value in gp. `param.gp` checks one of them at sample values of the parameter: see m1.

The five C-1 inputs and their mirrors refuse through both entry points with "leaves a branch cut at
the point", and the limit path is `Unknown` (`out0.txt`, 15 rows besides the 5 592). The four
controls of the request return gp's series and pass two-sided (errors 1e-62 to 1e-73). In
`rows1.tsv`, every real direction on a real-axis cut returns a value that passes (528 rows); on the
cuts of arctan and arcsinh, 170 of the 180 imaginary directions do, and the other 10
(`x*exp(I*pi/2)`, a coefficient that is exactly I but whose enclosure leaves its real part
undecided) refuse with "cannot decide". Every imaginary or complex direction on a real-axis cut,
and every real or complex one on an imaginary-axis cut, refuses, except the inputs whose
coefficient is real (or imaginary) after cancellation, which return a correct value:
`x + I*x - I*x`, `x*(1+I)*(1-I)` and, on the imaginary-axis cuts, `x*(1+I)^2` (41 rows, all pass).
The 107 values among the 216 powers pass too.

## 3. The rule, read against the code

I read the interdiff in full. The points I checked:

- The classes (`Tail`, `series_engine.rs:1183`) are sound in each rule I derived: sums take the
  meet; one series factor scales by the constant; products and quotients multiply full classes;
  b^n by parity; b^alpha and log through c0^alpha (1 + w)^alpha and log c0 + log(1 + w) with w real
  when b is all real or all imaginary; exp(b0 + u) for a real tail; odd and even functions of an
  imaginary argument; the atan family as (b' class) times the multiplier's class, with the right
  condition for arccosh (b real). A tail class is never claimed from a stored-coefficient scan.
- The kernel classes (`kernel_struct_class`, `series_const.rs:3420`) are sound: radicals are of
  n > 1; float kinds and function kinds (`@n:name`) cannot collide with `root*`; half-integer powers
  of negative constants are imaginary (all four signs checked); arctan(i t) is imaginary only for
  |t| < 1 and arcsinh(i t) only for |t| <= 1; the enclosure-based range tests use strict or exact
  bounds.
- The enclosures decide "in the cut's direction" only from an exact 0 coordinate, which only exact
  arithmetic produces (`CBall::mul`, `real`, the real step of `unary`); `log` of a rectangle that
  meets (-oo, 0] without lying on it is too wide; I checked `log`'s arg formulas in the three
  half-planes.
- `position` is right at the ends: an end is a branch point and goes to the integrand's
  `NotLaurent`; an undecided distance to an end is `UndecidedZero` of the difference.
- The integrands of the atan family call `pow_const` without a class. That is right: the cut set of
  each integrand's square root equals the function's own cut set (1 - b^2 <= 0 exactly when b is
  real with |b| >= 1, 1 + b^2 <= 0 exactly when b = i s with |s| >= 1, b - 1 or b + 1 <= 0 exactly
  when b <= 1). With a class in the cut's direction and c0 off the axis, b moves parallel to the
  axis and never meets the integrand's cut. On a cut, the integrand's principal branch matches the
  derivative of the module's on-cut values (checked by hand for arcsin, arccos, arctan, arcsinh,
  arccosh on both sides of each cut, and numerically by every control).
- Every branch application goes through `cut_check`: `log` (also `log f` of a variable exponent),
  `pow_const` with a class (sqrt and constant exponents), `atan_like`. Constant bases `c^g` are
  single-valued in g. Every public series entry point (`taylor`, `series`, `laurent`,
  `asymptotic`, `series_coefficients`, `series_with_big_o`, `limit_via_series`,
  `limit_with_error`) runs the engine. `limit_via_series` maps every error to `Unknown`, so the
  undecided `NotLaurent` never becomes a decided limit.

## 4. Regressions against HEAD-before-S1 (8a0b310)

HEAD's `taylor` returned the formal series for all ten C-1 rows (`drv/h_dbg rows0.tsv`): five
wrong, four mirrors right, and for `arctanh(2 - I*x^2)` and the control `arctanh(2 - x^2)` only
`arctanh(2)` (wrong). The four mirrors now refuse, as the request asks (choice 2 of the report; the
conventions and the doc of `NotLaurent` say so). HEAD also returned the right series where the rule
now refuses with "cannot decide" (`rows6.tsv`): `log(-1 + x*exp(I*pi))`,
`arccos(-3 + log(-1)^2*sinh(x))`, `log(-1 + x*((-8)^(1/3))^3)`, `log(-1 + I*(sin(x)^2 + cos(x)^2 -
1))`. These are honest refusals of undecided cases, the kind the request asks for, and the report
lists the shape among its known limits (section 6, item 5). I count them as minor (m2, m3), not as
bar (b): no crash, no abort, no wrong value. I found no input where C-1 turned a right value of HEAD
into a wrong one.

## 5. The changed expected outcome and the evaluator

`U:y12.1` (`(acos(((x)*((x^2+1)))^(2+x)))^3` at -1, value alternative replaced by `NotLaurent`) is
justified. gp (`fuzz_y12.gp`, `fuzz_y12b.gp`): the operand's h coefficient is
2.6137056388801093811 - 6.2831853071795864769*I, so the operand leaves the cut of acos at -2;
gp's `acos` is continuous from above there (imag(acos(-2 + 1e-30*I)) = -1.31696 = imag(acos(-2)));
the formal series is the function on one side only (relative errors 1.6e-61 and 1.86 at
t = -+1e-8, 1.6e-125 and 1.86 at 1e-16, realprecision 300). Which side depends on gp's rounding of
the constant term acos(exp(log(-2)))^3 (at realprecision 80 the other side passes, m4); the
conclusion, no Laurent expansion, holds either way. The evaluator change (`eval.rs`, arctan and
arcsinh on the lower imaginary cut as -f(-z)) matches the engine's formulas: arctan(-3i) =
-pi/2 - (i/2) log 2 and arcsinh(-2i) = log(2 - sqrt 3) - i pi/2 from the conventions of
`series.rs:49`; no earlier expected outcome depends on it.

## 6. Minor items

- **m1 (needs the owner's decision): free parameters.** `position` (`series_engine.rs:1351`)
  decides that a constant term which is a nonconstant polynomial in the parameters is off the cut,
  which holds for generic complex values. For a real parameter the result can be wrong on one side
  for every real value: `log(-y^2 + I*x)`, `log(-1 - y^2 + I*x)`, `log(-pi*y^2 - 1 + I*x)`,
  `sqrt(-y^2 - 1 + I*x)`, `arcsin(y^2 + 2 + I*x)`, `arctan(I*(2 + y^2) + x)` at 0 all return a
  value. gp (`param.gp`), the engine's series of `log(-1 - y^2 + I*x)` against
  `log(-1 - y^2 + I*t)`: |S - F| = 1.0e-37 at t = 1e-9 and 6.2832 at t = -1e-9 for y = 1/2, the same
  for y = 3; 1.3e-37 and 3.9e-40 on both sides for y = 1/2 + I/3 and 2 - I/5. This is the documented
  convention (`series.rs`, the C-1 bullet: "`log(y + I*x)` at 0 holds for y off (-oo, 0]"), it is
  consistent with the coefficient side (`log(-1 + y*x)` refuses, as generic complex y leaves the
  cut), and HEAD-before-S1 returns the same series (`h5`), so it is neither bar (a) under that
  convention nor bar (b). But the design's generic convention ("valid for a != b only") excuses an
  algebraic exception set, and here the exception set can be the whole real line. If parameters are
  meant to be real like the variable: treat a free symbol as real in the axis test of `position`,
  refuse ("cannot decide") when c0 is then on the cut's axis (real for log, the powers, arcsin,
  arccos, arctanh, arccosh; imaginary for arctan, arcsinh), and keep `Off` otherwise, so that
  `arctan(y + x)` and `log(-1 + I*y + I*x)` keep their values. If not, the bullet should say that
  the exception set may contain every real value (`log(-y^2 + I*x)`).
- **m2: the printed text of an undecided `NotLaurent`.** `SeriesError::NotLaurent` prints "no
  Laurent expansion at the point: ..." (`series.rs:247`). For the mirrors (`log(-1 + I*x^2)` is
  i pi + log(1 - i x^2) for all real x near 0) and for every "cannot decide" refusal of an analytic
  input (`log(-1 + x*exp(I*pi))`, `log(-1 + I*(sin(x)^2 + cos(x)^2 - 1))`), the prefix states a
  decided negative that is false; only the body says the engine cannot decide. The kind is the
  request's and its doc says so; a reworded prefix for these messages (or "no Laurent expansion
  found") would keep the text honest.
- **m3: over-refusal of enclosure-real constants in truncated arguments.** `Tail::scaled` reads only
  the structural classes, so a truncated argument scaled by a constant that is real only by its
  enclosure loses its class: `arccos(-3 + log(-1)^2*sinh(x))` at 0 refuses ("cannot decide"), and
  HEAD gave the right series. The exact-argument path decides the same constant real
  (`log(-1 + x*log(-1)^2)` returns a value that passes). A charged `cdecide` of the scaling constant
  in `finish_product` would recover these. Not required.
- **m4: the two-sided oracle and gp's rounding on a cut.** `s1_gp.py`'s oracle evaluates F (and gp's
  formal series G) with gp's floats. An exact on-cut cancellation leaves a tiny imaginary residue of
  random sign, so a nested branch function lands on either side of its cut: section 2 item 1 (38
  false failures in a plain check), and `U:y12.1`'s G0 changes side between realprecision 80 and 300.
  No row of the three data files is affected today. A note in the oracle's comment, or snapping
  residues below a threshold as `twoside.py` does with `CLEAN=1`, would keep future nested rows from
  failing falsely.
- **m5: `AB` of a non-real scalar in `s1_gp.py`.** `AB(s) = SG(s)*s` with `SG = sign(real(s))` for
  a scalar disagrees with the engine's abs of a non-real constant (the modulus: `CBall::unary`
  `Abs`, `KCLS_REAL` at `series_const.rs:3516`, and `eval.rs`'s `a.abs()`). No data row has abs of a
  non-real constant. Use gp's `abs` for a non-series argument, or say so in the comment.
- **m6: a weak assertion in the new test.** `limit_path_does_not_read_a_wrong_branch`
  (`tests/s1_cuts.rs:157`) asserts only that `limit_with_error` is not `Finite(0)`; it is `Unknown`
  (`limit_via_series` is `Unknown`), so `assert!(matches!(l, LimitResult::Unknown))` is the honest
  claim.
- **m7: `pow_const` replaces a sticky fault.** When `cut_check` fails inside `pow_const`
  (`series_engine.rs:895`) with a budget fault from `check()` (in `position` or
  `coefficient_direction`) and the exponent's integrality is undecided, the fault is reported as
  `UndecidedZero` of alpha - n. `FAULT` is sticky (`series_const.rs:588`), so nothing is computed
  under an exhausted budget; only the reported kind changes. Propagating any error that is not a
  cut verdict would keep `BudgetExceeded`.

## 7. Files

- `c1-1-semantics/c1_interdiff.patch` (the C-1 change alone), `before/` (HEAD plus
  `c1/diff_before.patch`).
- `drv/p.rs`, `drv/h.rs` (drivers; binaries `p_dbg`, `h_dbg`), `twoside.py` (two-sided check),
  `gen/gen1.py`, `gen/gen2.py` (generators).
- Inputs and outputs: `rows0.tsv` to `rows6.tsv`, `out0.txt` to `out5.txt`, `rr1.tsv`, `rr2.tsv`,
  `ro1.txt`, `ro2.txt`, `rc1.txt`, `rc2.txt` (plain), `rcc1.txt`, `rcc2.txt` (snapped), `chk1.txt`,
  `chk2.txt`, `chk3.txt`, `chk3c.txt`.
- gp scripts: `oncut.gp`, `param.gp`, `fuzz_y12.gp`, `fuzz_y12b.gp`.
- Logs: `sym_dbg.log`, `oracle_rerun.log`, `jup_lib.log`, `calc.log`, `manifolds.log`.
