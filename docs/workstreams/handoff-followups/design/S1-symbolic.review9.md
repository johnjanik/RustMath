# S1-symbolic design: adversarial review 9

Design reviewed: `fu2/design/S1-symbolic.md` (5259 lines: "Revision 7, red-team round 4", "Revision 8",
the earlier revision sections and the body), with its prototype `EXP/rev7rt4/proto13/` (EXP =
`fu2/design/S1-symbolic`). HEAD 6e03987, the same HEAD as review 8: rustmath-symbolic, the crates it
reads, the cited lines of `repl.rs` and HANDOFF's S1 bullet (`docs/HANDOFF.md:509-519`) are unchanged.

My files are in `EXP/review9/`: `fuzz.py` and `fuzz2.py` (random inputs against gp at 300 digits; the
second adds the inverse functions), their outputs `fz_1.out` to `fz_4.out` and `fy_11.out` to
`fy_14.out` (2000 inputs), `dbgrun.sh` and `dbgrun2.sh` (every one of those inputs on the 2 MB debug
probe), `repro9.sh` with its output `repro9.txt` (every row of B1, B2 and m2 through the three entry
points, on 2 MB debug, on revision 8 and on HEAD), `refs9.gp` (the gp references below), and the
radical families `r_*.txt`, `roots200.txt`, `roots2000.txt`. I built nothing; the binaries are those
of `EXP/rev7rt4/bin/` and `EXP/rev8/bin/`. Numbers are release on an 8 MB thread, budget 1e8, unless
stated.

Verdict: **not approved**. Two blockers, one class: a decided `DivisionByZero` ("the expression divides
by an exact zero", which the REPL reports as a ValueError) raised by a rule of the engine where the input
divides by no exact zero. B1: the atan family forms its integrand for an argument whose series is an
exact constant, and the integrand is singular at arcsin(+-1), arccos(+-1), arccosh(+-1) and
arcsinh(+-I), although the value is a constant (revisions 7rt1 to 8 give the same error). B2: revision 7rt4's pole rule
for arctanh(+-1) fires on the constant term of a non-constant argument, so a logarithmic branch point
that revision 8 reported as `NotLaurent` is now a `DivisionByZero` (a regression of this revision).
Review 8's blocker and its minors are resolved, and everything else I checked holds.

## What I verified (holds)

- **Review 8 B1** (the hint substituted a non-regular subtree). Review 8's five inputs on proto13
  (`FILE=.. DISPLAY=iter bin/proto13_rel coeffs @ ..`): `simple_sign` prints c0 `(1 + sin(1)) ^ (500 * 1)`,
  `zero_pole_exp` `(1 + sin(1)) ^ (0 + 500)`, `cancel_div` `1 + (1 + sin(1)) ^ (500 * 1)`, `ssign_4` and
  `sing_8` with the seeded exponents `2 * 1` and `2 + 0`; no `sign(0)`, `/ 0` or `(1 - 1)` in any of them.
  I read the flag code of `proto13/ser.rs` again: the `BaseConst` state carries the exponent's flag
  (`reg_g || seed_const(r, ..)`), every record ANDs its children's flags, and a seed renders the engine's
  own exact constant. A seed in `SUBST` survives the attempts of the driver (only the hints are cleared),
  which is sound: it is the value of the node, which does not depend on P. The gp values of the design's
  test hold (`refs9.gp`): W_4 gives 2.896978525182688290513575858664 and 18.258779796453404166973655565514,
  `(1+sin(1))^500` = 3.8233154824771471508e132.
- **Review 8 minors.** m1: `(1)/(x + sin((-3+1)^(-2)) - 1/4)` and `(1)/(x + sin((-2)^(-2)) - 1/4)` at 0
  are `Ok`; m2: `x + 0^pi` and `x + (x-x)^pi` are `x`, `x + 0^(-pi)` `DivisionByZero`, `x +
  0^(sin(1)^2+cos(1)^2-1)` `UndecidedZero`; m3: §2.14 reads `LEAF` with `try_with`; m4: §2.15 and §7
  item 8 say what the cap refuses and name the remedy.
- **The new numeric rules of revision 7rt4** (`proto13/ball.rs`), read and sampled at 64 bits
  (`bin/rt2_rel ball`): an integer literal of at most p bits and a dyadic rational with such a numerator
  are exact balls; the exact 0 propagates through `add` (exact negatives), `mul` (a zero midpoint), sin,
  sinh, tanh, atan, erf and the inverse functions at 0, log at 1, `sqrt`, `0^k` (k > 0) and `0^b` by the
  sign of b's ball; `1 - 1`, `sqrt(1-1)`, `(1-1)^(1/3)` give the exact 0 and `(1-1)^(-(2*2))`
  `None(Never)`. `exp(k log a)` with an exact integer k (`exp_int_pow`) is a^k for every nonzero a, a
  true identity for any branch of log: `exp((-2)*log(-3 + sin(1)))`, `exp(log(-7/3)*(3-1))`,
  `exp((2*(-5) - (-5))*log(1/2 + tanh(-2)))` and `exp(0*log(-2))` give balls that exclude 0 around gp's
  0.21462693234838571987, 49/9, -46.481595731122308409 and 1; `exp(4*log(0))` is `None(Never)`.
- **The precision out of band** (`proto13/ser.rs`: `Prec`, `shift`, `mul`'s cutoff and the counting
  loop, `inv`, `pow_int` items 2 and 5, `trunc`, `integrate`, the driver's `PREC_CLAMPED` stop): every
  clamp is upwards to `Abs(i64::MAX)`, a weaker true claim, and every downward overflow is an error;
  `mul` keeps `Exact` only when its cutoff dropped no pair, and the counting loop's early stop is
  followed by a charge that fails. The saturating `Work::charge` and `alloc` (`16 n` saturating, then the
  2^57 slot cap) hold.
- **Random inputs against gp.** 2000 random inputs of depth 2 to 4 (`fuzz.py`, `fuzz2.py`: the
  functions of §2.4 including abs, sign and the inverse functions, sums, products, quotients, integer,
  rational, signed and series exponents, constant bases to series, exact constants in the variable such
  as `x - x`, `(x+1)^2-x^2-2*x-1`, `0*((1)/(x-a))`, `(x-a)/(x-a)`, `sign(x^2+1)`, at the points 0, 1,
  -1/2, 2, 1/3, -1, orders 0 to 3; gp with `sign` and `abs` of a series defined by the leading
  coefficient): 1240 results equal gp's series to 1e-60 at 300 digits; 4 differ only by gp's convention
  (m1); gp cannot expand 237; the engine refuses 519. I read 70 of the `NotLaurent` refusals by hand
  (branch points, logarithmic singularities, poles inside exp and sin, abs and sign at zeros of odd
  order, `0^r` with r vanishing): each is right. The `DivisionByZero` refusals are right except the two
  classes below, which the generator met about 20 times (B1) and 25 times (B2). No input crashed in
  release, and all 2000 also ran on the 2 MB debug probe (`bin/rt2`) with exit code 0.
- **Cost samples** (release, `tools/run7.py bin/proto13m_rel`, with the fuzzers running on the 4 cores):
  x plus 200 64th roots of 4096-bit radicands `Ok` after 3.16 s (the design: 3.0 s), the 2000-root input
  (40.9 KB) refused after 3.58 s (3.25 s), `(5/7+x)^(-3/11)` at order 2000 `Ok` after 1.28 s, `tan(x)` at
  1, order 100, refused after 1.32 s. New radical families: 200 roots of degree 2, 4, 8, 16 and 32 of
  4096-bit radicands take 0.55 to 2.39 s for 9.72e7 units (the time grows with the degree at the same
  charge; degree 64 is the calibrated worst), degrees 127 to 4095 under 0.01 s, 200 64th roots of
  4096-bit perfect squares and of perfect 32nd powers (`r_sq*.txt`, 248 KB of digits) 2.7 s; a 99 KB
  literal radicand with degrees 2 to 10^6 under 40 ms. All within the bar, with a margin of 5 over 20 s.
- **gp references of §4.1** (`refs9.gp`): the W_4 values, `sin(1) - 1/2` = 0.34147098480789650665,
  `G_6(1) - 3` = 0.99845675697654779266 and the three `exp_integer_log` values are gp's.
- **HANDOFF items and callers**: unchanged since review 8 (the S1 bullet's four items are covered by
  §1 to §2.17; §3's callers and the MF1 coordination of §3.4 are as before).

## Blockers

### B1. arcsin, arccos, arccosh and arcsinh of an exact-constant argument are `DivisionByZero` where the value is a constant (§2.4 "arctan, arcsin, ..." row; `proto13/ser.rs::atan_like`)

§2.4 expands the atan family as `G = f(b0) + integral of Q` with `Q = b' (1 - b^2)^(-1/2)` (arcsin),
`b' (b - 1)^(-1/2) (b + 1)^(-1/2)` (arccosh), `b' (1 + b^2)^(-1/2)` (arcsinh), and so on. When the series
of the argument is an exact constant c (`x^0`, `x/x` at 0, `(x-1)/(x-1)` at 1, `sign(x^2+1)`, `1 + 0*x`),
b' is the exact 0, so Q is 0 and the value is f(c). But `atan_like` builds the factors of Q first, and at
c = +-1 (arcsin, arccos), c = +-1 (arccosh) and c = +-I (arcsinh) the factor is `pow_const` of the exact
0 to the power -1/2, which is `DivisionByZero("0 to a negative power")`. The function is a constant there:
arcsin(1) = pi/2, arccos(1) = 0, arccosh(1) = 0, arccosh(-1) = i pi, arcsinh(i) = i pi/2 (gp:
`asin(1)`, `acos(1)`, `acosh(1)`, `acosh(-1)`, `asinh(I)` print 1.5707963267948966192, 0, 0,
3.1415926535897932385*I, 1.5707963267948966192*I). The design's §2.4 says that "`pow_const` already
refuses arcsin at 1", which is right for a non-constant argument (a branch point), not for a constant one.

Reproductions (`EXP/review9/repro9.sh`, output `repro9.txt`; every row takes under 0.1 ms and 3 MB in
release, the same through `taylor_with_budget` and `series_expansion`, `Unknown (DivisionByZero)` through
the limit path, the same on the 2 MB debug thread (under 0.25 ms), and the same in revision 8):

| input, point, order | proto13 | HEAD's `taylor` (HEAD's parser) | correct |
|---|---|---|---|
| `arcsin(x^0)`, 0, 2 | `DivisionByZero("0 to a negative power")` | `arcsin(1)` plus `0/0` terms | arcsin(1) |
| `arccos(x^0)`, 0, 2 | the same | `0/0` terms | 0 |
| `x + 0*arcsin(x^0)`, 0, 3 | the same | `x` | x (gp: `X + 0*asin(X^0)` = `h + O(h^4)`) |
| `exp(x) + 0*arccos(x^0)`, 0, 3 | the same | `1 + x + x^2/2 + x^3/6` | exp(x) (gp: the same) |
| `arccos(sign(x^2+1)) + x`, 0, 2 | the same | `0/sqrt(1 - sign(1)^2)` terms | x |
| `x + arcsin(1 + 0*x)`, 0, 2 | the same | `0/0` terms | arcsin(1) + x |
| `arcsin((x-1)/(x-1)) + x`, 1, 2 | the same | `0/0` terms | arcsin(1) + x |
| `arccosh(x^0)`, 2, 1 | the same | `arccosh(1)` (HEAD reads arccosh as a named function) | 0 |
| `arccosh(-x^0)`, 2, 1 | the same | `arccosh(-1)` | arccosh(-1) = i pi |
| `arcsinh(I*x^0)`, 0, 1 | the same | `arcsinh(I)` | arcsinh(I) = i pi/2 |

This is over the bar under (a): `DivisionByZero` is a decided claim that "the expression divides by an
exact zero" (§2.9), the REPL maps it to "ValueError" (a property of the input, §3.1), and these inputs
divide by no exact zero; their expansion is a constant (plus x). It is (b) as well on the rows where
HEAD returns the correct series (`x + 0*arcsin(x^0)`, `exp(x) + 0*arccos(x^0)`, and through HEAD's parser
`arccosh(x^0)`, `arccosh(-x^0)`, `arcsinh(I*x^0)`, which the §2.12 parser now reads as unary operations).
It is red team 3 structure H1's class ("the rule is chosen from the syntax, not from the series", there
`log` of an exact zero base), which revision 7rt3 closed for `Pow` only; the unary operations still form
their recurrences for an exact-constant operand.

Required change:

1. §2.4 and §2.5 ("`Unary`"): an operand whose series is an exact constant c (`exact_constant`, the test
   the Pow rules already use) gives the exact constant `apply(op, c)` before any integrand is formed, at
   least in the atan family (the only recurrence that can fail on a constant operand; for exp, log, the
   trigonometric and hyperbolic functions the rule is optional, and for tan it would turn the honest
   `UndecidedZero` of `tan(c)` at an undecided pole c into a kernel under the convention of §7 item 10).
   `apply` decides the poles of the constant (arctanh(+-1), and arctan(+-I) by m2), so `arctanh(x^0)` and
   `arctan(I*x^0)` stay `DivisionByZero`. The change is O(1) per node; the result is exact, so the
   implementer reruns the gp oracle and the revision checks (outcomes that read `Exact` can change).
2. §4.1: a test `unary_exact_constant_operand` with the ten rows above, outcome and value asserted (gp:
   `asin(1)`, `acos(1)`, `acosh(1)`, `acosh(-1)`, `asinh(I)`; `X=h+O(h^4); X + 0*asin(X^0)` and
   `exp(X) + 0*acos(X^0)`), and the controls `arctanh(x^0)` and `arctan(I*x^0)` at 0 `DivisionByZero`,
   `arcsin(1 - x^2)` at 0 and `arccosh(x)` at 1 `NotLaurent` (branch points). Each value row fails on
   proto13.
3. State the class in §2.5 with the audit: no `DivisionByZero` without an exact zero (or an exact pole of
   a constant) in the input. The producers I checked in the prototype: `inv` and `log` of the exact 0,
   `pow_int` and `kpow_const` of the exact 0 to a negative power, `zero_pow`, `inverse_src` of a zero
   constant and the poles of `kapply`; only the atan family reaches one of them without such a zero (B1
   through its integrand, B2 through its constant term).

### B2. Revision 7rt4's pole rule for arctanh(+-1) turns logarithmic branch points into `DivisionByZero` (§2.2 "apply", red team 4 numeric m1; `proto13/konst.rs::kapply_src`, called by `ser.rs::atan_like` on b0)

Revision 7rt4 makes `apply(Arctanh, +-1)` a `DivisionByZero`, which is right for the constant arctanh(1)
(`x + atanh(1)`). But `atan_like` calls `kapply(op, b0)` on the constant term of every argument, before
it looks at the integrand. For a non-constant argument with b0 = +-1, `atanh(b)` has a logarithmic
singularity, `(1/2) log((1 + b)/(1 - b))`, and revision 8 reported the documented `NotLaurent("the
function has a logarithmic branch point there")` from the integrand's valuation; revision 7rt4 reports
`DivisionByZero("arctanh at 1 or -1")`:

| input, point, order | proto13 | revision 8 | HEAD's `taylor` |
|---|---|---|---|
| `atanh(x)`, 1, 2 (8 bytes) | `DivisionByZero("arctanh at 1 or -1")` | `NotLaurent("the function has a logarithmic branch point there")` | `atanh(1)` (a named function) |
| `atanh(1+x)`, 0, 2 | the same | the same `NotLaurent` | `atanh(1)` |
| `x + 0*atanh(1+x)`, 0, 2 | the same | the same `NotLaurent` | `x` |
| `cos(atanh(1-x))`, 2, 2 | the same | the same `NotLaurent` | `cos(atanh(-1))` |
| `atanh(exp(x-x))`, 0, 2 (b = `1 + O(h^P)`: b - 1 is not decided) | the same | `PrecisionExhausted(131)` | `atanh(1)` |

Every row takes under 0.1 ms, the same through the three entry points (the limit path `Unknown`) and
on the 2 MB debug thread. The input divides by no exact zero (and in the last row the engine does not
know whether b - 1 vanishes, an undecided answer reported as decided), so the claim of §2.9 and the REPL's
"ValueError" are false, and `x + 0*atanh(1+x)` (whose value is x, which HEAD returns) moves from an honest
refusal (`NotLaurent` says that a subexpression has no expansion, which is true) to a false one. This is
the class of B1 and red team 3 structure H1 under (a), and a regression of this revision against
revision 8. It does not reach the `zeta` and `gamma` poles, since those functions of the variable are
`Unsupported`, nor log, whose valuation check runs before `apply`.

Required change: in the atan family, a pole of f at the constant term of a non-constant argument is the
logarithmic branch point (`NotLaurent`), as for `log(x)` at 0; only an exact-constant argument (B1's
rule) reaches the `DivisionByZero` of `apply`. The simplest order: form the integrand and apply its
valuation test (and its `NeedPrecision` for an empty truncated `1 - b^2`) before `apply(op, b0)`, so that
`atanh(1 + x)` is `NotLaurent` and `atanh(exp(x - x))` (b = `1 + O(h^P)`) asks for precision, as in
revision 8. Mapping the pole of `apply` to `NotLaurent` is right only when `b - b0` has a decided
nonzero term. Add the rows to `constant_poles` (§4.1), the first four as `NotLaurent` and the last as
`PrecisionExhausted`, with `x + atanh(1)` and `atanh(x^0)` staying `DivisionByZero` (each new row fails
on proto13).

## Minor issues (fix in the revision)

- **m1. gp takes the real root for a series power of a negative rational constant term.** gp gives
  `(-1-h+O(h^2))^(2/3)` = `1 + 2/3*h` and `(-1+h+O(h^2))^(1/3)` = `-1 + 1/3*h`, but `(-1)^(2/3)` =
  `-0.5 + 0.866*I` and `(-9/16)^(-1/2)` = `4/3*I`, while the engine takes the principal value throughout
  (`exp(alpha log c)`; `(-9/16)^(-1/2)` = -4i/3). Four of my 2000 random inputs differ from gp only for
  that reason (`(((x)*(x))^3)/((((1-x))^(2/3))+((x-2)))` at 2, `((x^(1/3))+(x^2+1))*sign(x^3)` at -1,
  `pi^(acos(x^2+1)+x)` at -1/2, `((x^(1/3))+x)/abs(x^((x-x)+2))` at -1; with the powers written as
  `exp(alpha*log(.))` in gp they agree). The design's "Branch at the point" paragraph describes another gp
  artefact (a float constant term below the cut) but not this one; the port's gp-referenced tests at
  points where a base is a negative rational should write the power as `exp(alpha*log(.))` in gp, and
  §2.4 should say so.
- **m2. arctan(+-I) is a kernel at a pole.** `x + arctan(I)` at 0 is `Ok` with the kernel `arctan(I)`
  and the limit path gives `Finite(0)` for `x*arctan(I)` and `x*arctan(-I)` (gp: `atan(I)` is "domain
  error"), where `x*arctanh(1)` is `DivisionByZero` since revision 7rt4. I is decided exactly (the kernel
  `I`), so the rational branch of `apply` can refuse arctan at +-I as it refuses arctanh at +-1. HEAD
  gives the same output, so this is below the bar (red team 4 numeric m1's class), but B1's change needs
  it: without it, `arctan(I*x^0)`, refused today through the integrand, would become the kernel.
- **m3. Exact constants that the engine does not see.** `arcsin(x/x)` at 1 is `PrecisionExhausted(131)`:
  x/x at a nonzero point is `1 + O(h^P)`, since the inverse of an exact non-monomial is truncated, so
  B1's rule does not reach it and `1 - b^2` stays an empty truncated series. Honest (HEAD prints `0/0`
  terms); worth a sentence in §7 next to the other honest refusals.

## Summary for the orchestrator

Revision 7rt4 closes red team 4's holes and review 8's blocker and minors: the out-of-band precision, the
saturating units and the slot cap hold, the new exact balls and `exp(k log a)` are sound, 2000 random
inputs give 1240 results equal to gp at 300 digits (4 more differ only by a gp convention) and no crash
in release or on the 2 MB debug thread, and the sampled costs stay under 3.6 s. Two blockers stop
approval, both a decided `DivisionByZero` where the input divides by no exact zero, each cheap to fix:

- B1 (in every revision since 7rt1 at least): arcsin, arccos, arccosh, arcsinh of an argument whose
  series is an exact constant form a singular integrand: `arcsin(x^0)` at 0 is `DivisionByZero` for the value pi/2, and
  `x + 0*arcsin(x^0)` and `exp(x) + 0*arccos(x^0)`, which HEAD answers correctly, are refused the same
  way. Fix: an exact-constant operand gives `apply(op, c)`, at least in the atan family, with
  arctan(+-I) added to the poles of `apply` (m2).
- B2 (a regression of revision 7rt4): the new pole rule for arctanh(+-1) fires on the constant term of a
  non-constant argument, so `atanh(x)` at 1 and `x + 0*atanh(1+x)` at 0 are `DivisionByZero` where
  revision 8 said `NotLaurent` (a logarithmic branch point). Fix: that pole is `NotLaurent` unless the
  argument is an exact constant.

The minors are a gp convention for the tests (m1), arctan(+-I) as a pole of the constant (m2, needed by
B1's fix) and a sentence on `arcsin(x/x)` at 1 (m3).
