# S1-symbolic, red team 4: the rigorous numeric layer

Design: `fu2/design/S1-symbolic.md`, revision 8 (prototype `EXP/rev8/proto12/`, `EXP` =
`fu2/design/S1-symbolic`). Binaries: `EXP/rev8/bin/rt2_rel` (the §2.12 parser probe, which reads big
literals and asinh, acosh, atanh as unary functions), `rt2` (debug), `proto12m_rel`, and
`EXP/rev7rt2/bin/rt2_rel`, `EXP/rev7rt3/bin/rt2_rel` as controls. All runs are release, budget 1e8,
with a 4 GiB address-space cap and a timeout (`r.py`), unless stated. Another agent ran gp on the
machine during part of the runs (load 1.0 to 1.6), so rates are medians of many calls.

My files are in `EXP/redteam4-numeric/`:

- `src/rt4.rs`: the §2.12 parser linked against `proto12/ball.rs`. Modes: `rate FILE P REPS`, `ball FILE P`
  (the enclosure as exact `int*2^exp`), `balls FILE 0` (every line at 64, 256 and 1024 bits),
  `scan FILE P REPS` (one node per line: its own units against its time minus its arguments' time).
- `src/hlim.rs`: HEAD's `Expr::limit_via_series` and `taylor` on one input.
- `build.sh release src/rt4.rs rt4` rebuilds a probe in seconds (I deleted the binaries).
- `gen_rigor.py`, `gen_focus.py`, `check_rigor.py`: random constants, their balls, and a gp check of
  `|v - m| <= r` at 3500 digits.
- `r.py`: one call of an entry point with time and max RSS. `repro.sh` reruns every reproduction below
  (output `out/repro.txt`).
- `inputs/`, `out/`: every input and output named below.

Verdict: no hole over the review bar in this lens. Three holes below it, all reproduced. None gives a
wrong result through the design as written. One (m1) is a gap between the prototype and its own §2.2.

## Holes

### m1. arctanh(1), arctanh(-1) and zeta(1) are kernels in the prototype, not `DivisionByZero` (minor)

Section: §2.2 "apply(op, c)": "Poles of the constant are `Err(DivisionByZero)`: log(0), arctanh(1),
arctanh(-1), zeta(1), gamma and factorial at a pole." Prototype: `proto12/konst.rs::kapply_src`, the
rational branch. It returns `DivisionByZero` for log(0), gamma and factorial only. atanh(+-1) falls to
the `NonZero` class of "f(q) for rational q", and zeta(1) to `Opaque`. No test of §4.1 names these poles.

| input at 0, order 2 | taylor_with_budget, series_expansion | limit path | HEAD |
|---|---|---|---|
| `(1)/(x + atanh(1))` | `Ok`: `1/arctanh(1) - x/arctanh(1)^2 + x^2/arctanh(1)^3` | `Unknown` | the same terms plus `0/0` junk |
| `(1)/(x + atanh(-1)) + x` | `Ok`, the same form | `Unknown` | junk |
| `x + atanh(sin(pi/2))` | `Ok`: `arctanh(sin(pi/2)) + x` (sin(pi/2) folds to 1) | `Unknown` | |
| `x*atanh(1)` | `Ok`: `arctanh(1) x` | `Finite(0)` | `Finite(0)` (`bin/hlim_rel`) |
| `x + zeta(1)` | `Ok`: `zeta(1) + x` | `Unknown` | |
| control `(1)/(x + log(1-1))` | `DivisionByZero("log(0)")` | `Unknown (DivisionByZero)` | |

Each call takes under 0.3 ms, at most 4965 units, VmHWM 3 MB. gp: `atanh(1)` and `zeta(1)` are
"domain error". The zero test decides `arctanh(1)` nonzero and inverts it, and the limit path reports
the limit of x times a pole as 0. HEAD gives the same outputs, so this is no regression. The design body
already prescribes the error, so a port that follows the body is right. A port that copies `kapply_src`
keeps the defect, and no test would catch it.

Over the bar: no, against the design as written. Through the prototype the limit `Finite(0)` is a
decided answer for an undefined input.

Fix that closes the class "an enclosed function at a pole of a rational argument": in the rational
branch of `kapply_src`, `Arctanh if q = 1 or q = -1` and `Zeta if q = 1` return `DivisionByZero`, as log
at 0 does. Add to §4.1 a test `constant_poles` with the six rows above (`DivisionByZero` through the
three entry points; the limit path `Unknown (DivisionByZero)`).

### m2. Short-mantissa arguments run cosh, sinh and the integer-valued power below the stated rates (minor)

Section: §2.3, the scan tables and "The slowest median of either scan is now 2.5e7 units/s (log at 256
bits), 2.5 times the floor"; §2.3 rev 8 note "the slowest of eight such constants at 8.3e7 units/s".

The scans of §2.3 use rationals `n/d` with small d, `k/65537 2^e` and p-bit random mantissas. All of
them have full p-bit mantissas after rounding. A literal `M/2^k` with a small odd M has a short
mantissa, and `Ball::rounded` keeps it exact. Then `1 + x^2/2` (cosh), `1 + x` (exp inside sinh and
cosh) is exactly representable, the next term lies past the working precision, and MPFR's Ziv loop
retries two to four times. The ball calls MPFR twice for sinh and cosh (`cosh(|m| + r)` for L, and f(m)),
and both retry. The charge is the generic `2u`.

`bin/rt4_rel scan` (`inputs/scan1.txt`, 18 024 arguments `f(M/2^k)`, `f(1 +- M/2^k)`, `f(2^k + M)` for
16 functions; `inputs/dense_*.txt`, 20 090 more for cosh, sinh, exp, tanh, asinh, log, asin; then the
slowest rerun alone, median of 1001 calls, node units over the node's own time):

| node | p | node units | time | node rate | table of §2.3 |
|---|---|---|---|---|---|
| `cosh(3/2^130)` | 256 | 464 | 28.8 us | 1.61e7 | cosh 5.4e7 |
| `sinh(3/2^130)` | 256 | 464 | 26.4 us | 1.76e7 | sinh 3.5e7 |
| `cosh(1/2^31)` | 64 | 128 | 6.8 us | 1.89e7 | cosh 3.0e7 |
| `cosh(15/2^470)` | 1024 | 4688 | 248 us | 1.89e7 | cosh 1.0e8 |
| `(1 - 2^-30)^(-(5))` (the rev 8 path, `exp(k log abs(a))`) | 64 | 192 | 9.7 us | 1.98e7 | "8.3e7 or more" (whole expressions) |

The slowest cell of each function over both scans: cosh 1.9e7 / 1.6e7 / 1.8e7 at 64 / 256 / 1024
bits, sinh 2.1e7 / 1.7e7 / 2.7e7, acosh (at `1 + M 2^-k`) 2.1e7 / 3.1e7 / 4.6e7, asinh 2.1e7 / 2.6e7 /
3.1e7, asin (at `1 - M 2^-k`) 2.8e7 / 2.5e7 / 2.9e7, log (at `1 + M 2^-k`) 2.7e7 / 2.5e7 / 3.2e7. Every
other function is at 2.9e7 or more.

The floor of 1e7 units/s still holds, so any numeric work charged 1e8 units runs at most 6.3 s. End to
end the slow node is diluted by its leaf, the combining nodes, `to_expr` and the other precisions:
`inputs/coshmany.txt` (200 kernels `cosh(M/2^130)` plus `log(PQ - e)` and `zeta(3)`, in 64 groups,
10.7 KB) is `UndecidedZero` after 0.16 s with 2.7e7 units, 1.7e8 units/s, through
`taylor_with_budget` and the limit path. `inputs/coshgrp_13.txt` (8192 groups of
`cosh(3/2^130) - 1 - 9/2^261`) is `Ok` in 0.07 s (decided at 1024 bits).

Over the bar: no. It is a stated figure that is too high: the margin over the floor is 1.6, not 2.5.

Fix that closes the class "an argument at which MPFR's Ziv loop retries because the exact result is a
representable number plus a tail past the working precision": add `M 2^-k` (M odd below 2^8, k from
p/4 to p + 1) and `1 +- M 2^-k` to `fnscan` and `nearscan` for every function, and either restate the
slowest rate as 1.6e7 or charge sinh and cosh as two operations at `2p + 64` bits (their two MPFR
calls), which puts them at 4e7 or more in the table above.

### m3. An input-syntax rendering with an exactly-zero operand loses the enclosure the normal form has (minor)

Section: §2.3 (an integer literal is `Ball::rounded`, so its ball has radius `abs(m) 2^-p`; revision 8
`small_int_value`), §2.5 "Value at the point" (hints render the input with the point substituted, since
revision 7rt3). Prototype: `ball.rs::ball_expr` (Integer leaves), `Sqrt`, `Log`, `big_int_pow`
(`!a.excludes_zero()` gives `TooWide`).

The text `1 - 1` gives the ball `0 +- 2^-63`, never the exact 0, because every integer literal gets a
radius. A kernel made from an input node, or a hint, renders the input's syntax. Where that syntax
holds an exactly-zero subexpression under sqrt, a fractional power, log, or a non-literal integer power
with abs(k) >= 4, the ball crosses 0 at every precision (`TooWide` three times), and a decidable
constant is `UndecidedZero`. The normal form of the same constant is decided at 64 bits.

| input | point, order | outcome (tw, se, lim) | controls |
|---|---|---|---|
| `(1)/(x - 1 + G_6(x + (x-1)^(2*2)) - 3)`, G_k = `(1+sin(..))^2` k times (`inputs/hs_zero_6.txt`) | 1, 0 | `UndecidedZero` of the hint `(1 - 1) + G_6(1 + (1 - 1)^(2*2)) - 3`, 0.2 ms | without the term, or with `(x-1)^4`: `Ok`, 2 or 3 hints (`hs_plain_6`, `hs_lit_6`); revision 7rt2 (no hints): `Ok`; revision 7rt3: `UndecidedZero` |
| `(1)/(x + sin(sqrt(1-1) + 1) - 1/2)` | 0, 1 | `UndecidedZero`, `Unknown (Undecided)` | `(1)/(x + sin(1) - 1/2)`: `Ok` |
| `(1)/(x + sin((1-1)^(2*2) + 1) - 1/2)` | 0, 1 | `UndecidedZero` | `(1-1)^4` (literal exponent): `Ok` |
| `(1)/(x + sin((1-1)^(1/3) + 1) - 1/2)` | 0, 1 | `UndecidedZero` | |

Each call takes under 0.6 ms, at most 78 512 units, VmHWM 3 MB; the same on a 2 MB debug thread. gp:
`G_6(1) - 3 = 0.99845675697654779266`, `sin(1) - 1/2 = 0.34147098480789650665`. HEAD prints a series for
each: `1/(G_6(1) - 3)` for the hinted row, `1/(sin(1) - 1/2) - x/(sin(1) - 1/2)^2` for the
`(1-1)^(2*2)` row, and a `0/0` term in the x coefficient of the sqrt row.

Over the bar: no. It is an honest refusal of the same class as review 8 m1 ("a hint or a kernel made
from an input node renders syntax the ball cannot enclose, where the normal form has one"), which
review 8 rated below the bar. Revision 8 closed that class for negative bases with integer-valued
exponents only; zero bases, sqrt and log of an exact zero remain. It is a regression of revision 7rt3
against 7rt2 on the hinted row.

Fix that closes the class: when the enclosure of an input-syntax rendering (a hint or a kernel's source
node) is `TooWide` or `Never` at every precision, retry once on the normal form of the same constant
(the kernel's recorded `(op, c)` with c rendered by `to_expr`, bounded by `MAX_RESULT_TREE` and charged
by `ball_cost`). A cheaper partial fix: an exact ball (radius 0) for a literal whose conversion is exact
(an integer of at most p bits, a dyadic rational), so `1 - 1` is the exact 0, plus the exact-zero rules
`sqrt(0) = 0`, `0^k = 0` for k > 0 (also on the `exp(k log abs(a))` path), `0^q = 0` for rational q > 0,
`0^0 = 1`. Add the four rows above to `integer_valued_exponent_enclosure` (§4.1).

## Families that held

- **Rigor of every enclosure against gp** (`check_rigor.py`, gp at 3500 digits, every ball as exact
  `int*2^exp`). Eight seeds of `gen_rigor.py` (2000 random constants each, depth up to 4, 17 functions,
  `+ - * /`, literal integer exponents up to 2^62 and beyond i64, integer-valued non-literal exponents,
  rational and general exponents, leaves `1 +- M 2^-k`, `M 2^-k`, `2^k + M`, pi, e, `PQ - e`, convergent
  numerators of pi/2) and four seeds of `gen_focus.py` (1500 each: asinh, acosh, atanh, asin, acos, log,
  sqrt and erf of computed arguments within `4.3e-37` or `M 2^-k` of their guards, exp, sinh and cosh
  near `2^27`, sin, cos and tan near `k pi/2`, powers of negative and near-1 computed bases with
  integer-valued exponents of abs(k) <= 3 and >= 4, i64 and beyond-i64 literals). 44 517 enclosures
  checked, 0 outside the ball. The 34 rows where gp itself failed (`atanh` of an exact rational overflows
  gp's stack) were rechecked with real arguments: 0 bad.
- **Lipschitz bounds at rounded ends, by hand.** asin and acos round `1 - a^2` without factoring it, but
  `up()` has already moved a by 16 ulps, and that slack covers the rounding of `a^2` for every distance of
  a from 1. atanh and acosh use the factored forms, `1 - a` exact by Sterbenz. asinh's `min(1, 1/d)`
  holds with d shrunk. `inv`'s gap rounds by at most 2^-p relative, inside `up()`. Every `RealMPFR` value
  of `ball.rs` is at precision p, so the max-precision operators of `rustmath-reals` never double-round.
- **Escalation and classification.** `TOO_WIDE` is set only on a path that returns `None` at once, so the
  class never leaks into a later success. A charge refused inside `enclose` is `Never` with the fault set.
  `ball_cost` over-charges a Function's arguments and a literal exponent's leaf. `floor_small` runs at
  most three times per expansion (its outcome ends `pow_const`).
- **Grouping.** I-powers reduce (`I^2 + 1`, `(I pi)^2 + pi^2`, `(1+I)^2 - 2I`, `(pi+I)(pi-I) - pi^2 - 1`,
  `(e I)^4 - e^4` are the exact 0). Non-real kernels stay undecided (`I^(1/2)^2 - I`, `(I^2)^(1/2) - I`,
  `(-1)^(1/2) - I`, `sqrt(-1) - I`, `(pi I)^(1/2)^2 - pi I`). Radicals of negative bases keep the
  principal branch (`(-8)^(1/3) + 2` and `- 1`: `UndecidedZero`, not folded to -2). Exact values at 0 of
  the odd functions and `log(0)` hold.
- **Exponent window and guards end to end.** `exp(134217727) - 1` decided, `exp(2^27) - 1` undecided (the
  guard's boundary, honest), `exp(-134217727) - 1` and `sinh(134217727) - 1` decided,
  `(1 + 2^-200)^(2^226) - 1` decided both as a literal exponent and as `2^226`, `(1 + 2^-200)^(2^228)` and
  `(1 + 2^-100)^(2^226)` undecided (outside the window). `erf(15/2 +- 2^-60) - 1` decided.
- **Pow nodes with non-literal exponents** (`inputs/powscan.txt`, 340 at three precisions): slowest node
  rates 1.6e7, 2.2e7, 3.5e7 (m2), every enclosure in the gp check.
- **Memory and stack.** VmHWM at most 16 MB in every run; the ball code is iterative; no stack use grows
  with the precision or the operand size on the 2 MB debug thread.
