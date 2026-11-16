# S1-symbolic, red team 3: the rigorous numeric layer

Design: `fu2/design/S1-symbolic.md`, revision 7 red-team round 2 (prototype `EXP/rev7rt2/proto10/`,
`EXP` = `fu2/design/S1-symbolic`). Binaries: `EXP/rev7rt2/bin/proto10_rel`, `proto10m_rel` (the
counting build), `proto10` (debug), `rt2_rel` (the §2.12 parser probe, which reads `asinh`, `acosh`
and `atanh` as unary functions; HEAD's parser reads them as opaque calls), `nearscan_rel`,
`fnscan_rel` and `onecall_rel`. All runs are release with budget 1e8 unless stated. Each run had
`ulimit -v 4000000` (or `RLIMIT_AS` 4 GiB) and a timeout. The machine was idle (load 0.01).

My files are in `EXP/redteam3-numeric/`:

- `tools/r.py`: one call with time, outcome, units, VmHWM and heap peak (4 GiB cap, timeout).
- `src/rate.rs`: `ball_cost` against the median time of `ball_expr_why` at 64, 256 and 1024 bits,
  for one expression per line. It links `proto10/ball.rs` and is built with
  `EXP/rev7rt2/build.sh release`. I deleted the binary after the runs.
- `inputs/`: every input below. `nearscan_rt3.txt` is the rerun of `nearscan_rel`.

Verdict: I found no hole over the review bar in this lens. I found two holes below the bar and
reproduced both. Each one is an honest refusal of a constant that can be decided. Each is cheap: at
most 1 ms and 9.1e4 units. Neither gives a wrong result.

## Holes

### m1. asinh, acosh and atanh computed by log formulas lose the enclosure (minor)

Section: §2.3, "Operations" ("asinh, acosh, atanh via their log formulas"). Prototype:
`ball.rs`, the arms `Arcsinh`, `Arccosh` and `Arctanh`.

`log(x + sqrt(x^2 + 1))`, `log(x + sqrt(x^2 - 1))` and `log((1 + x)/(1 - x))/2` are rigorous ball by
ball. But they lose all the relative precision of the argument in three regimes, at every precision
the test reaches:

- asinh at x << -1: `x + sqrt(x^2 + 1)` cancels to about `1/(2|x|)`, while its radius is about
  `|x| 2^-p`. Past |x| = 2^511 the ball contains 0 even at 1024 bits, and log gives `TooWide`.
- asinh and atanh at |x| < 2^-p: `1 + x` rounds to 1, log returns `0 +- 2^-p`, and the value x is
  gone.
- asinh and acosh at |x| > 2^(2^27): x^2 leaves the window, which is `Never`.

sinh and tanh call MPFR directly and keep the argument's relative precision at the same tiny x. All
results below are from `rt2_rel both FILE 0 3 8388608`. The outcome is the same through
`taylor_with_budget`, `series_expansion` and the limit path (`ENTRY=se`, `ENTRY=lim`). VmHWM is 3 MB.

| input at 0 (`inputs/ahyp_*.txt`) | outcome, time, units | value (gp) |
|---|---|---|
| `(1)/(x + asinh(-2^600) + 400)` | `UndecidedZero(400 + arcsinh(-(2^600)))`, 0.32 ms, 88 148 | `asinh(-2^600)+400` = -16.58145551652713 |
| `(1)/(x + asinh(-10^200) + 400)` | `UndecidedZero`, 0.20 ms, 77 996 | -61.21016577936908 |
| `(1)/(x + asinh(1/10^400)*10^400 + 1/2)` | `UndecidedZero`, 0.20 ms, 90 563 | 1.5 |
| `(1)/(x + atanh(1/10^400)*10^400 + 1/2)` | `UndecidedZero`, 0.20 ms, 90 563 | 1.5 (`atanh(t) = t + t^3/3 + ...`) |
| `(1)/(x + asinh(2^(2^27)) - 93032640)` | `UndecidedZero`, 0.13 ms, 3789 | `(2^27 + 1) log 2 - 93032640` = 0.43750880779418 |
| `(1)/(x + acosh(2^(2^27)) - 93032640)` | `UndecidedZero`, 0.12 ms, 3789 | 0.4375088 (`acosh(y) = log(2y) - O(y^-2)`) |
| control: `(1)/(x + sinh(1/10^400)*10^400 + 1/2)`, `tanh` likewise | `Ok`, 0.11 ms, 9165 | `checks/coefcheck.py` against gp: 4 of 4 coefficients equal |
| control: `(1)/(x + asinh(2^600) - 400)` | `Ok`, 0.16 ms | |

HEAD's `taylor` prints an unchecked series on each of these inputs: `rt2_rel head` gives
`1/(arcsinh(-4149...) + 400) + ...`. This is the same kind of incompleteness as red team 2's m1, which
was rated minor. Over the bar: no.

Fix that closes the class: enclose the three functions with MPFR's own functions (`Float::asinh`,
`acosh` and `atanh` are rug inherent methods, as `erf` already is), with Lipschitz bounds:

- asinh: L = 1.
- acosh: require `m - r > 1` (as now), and L = `1/sqrt((m - r)^2 - 1)`.
- atanh: require `|m| + r < 1`, and L = `1/(1 - (|m| + r)^2)`.

Keep the exact zeros of R2 (asinh and atanh at 0, acosh at 1) and the current charge `6u`. Add the
three functions to `nearscan` at `L 2^-k`, at `-2^k` and at `1 + 2^-k`. A cheaper alternative that
covers only the first regime: use `asinh(x) = -asinh(-x)` for m < 0.

### m2. An integer exponent above 2^32 has no enclosure, even when the value is moderate (minor)

Section: §2.3, "Operations" ("Integer powers by binary powering (|k| <= 2^32...)"), and §2.16, the row
"integer power of a ball". Prototype: `ball.rs::ball_expr`, `k.unsigned_abs() > (1u64 << 32)` returns
`None`, which is `Never`.

A kernel `c^n` that the size caps keep takes its input node as its expression (§2.2). When the
exponent in the text is a literal above 2^32, the ball refuses it at every precision. When the same
exponent is written as an expression (`2^33`) or negated (`-8589934592`), it takes the
`exp(b log a)` path and is decided. Through `taylor_with_budget`, `series_expansion` and the limit path
(`tools/r.py` with `proto10m_rel`, and `rt2_rel`): 0.2 ms, 6449 units, VmHWM 4 MB, heap peak under
0.1 MB.

| input at 0 | outcome | value (gp, `exp(k log(1 + 10^-9)) - c`) |
|---|---|---|
| `(1)/(x + (1+(1)/10^9)^4294967297 - 5)` | `UndecidedZero(-5 + (1 + 1/10^9)^4294967297)`; limit path `Unknown` | 68.329816143353933 |
| `(1)/(x + (1+(1)/10^9)^8589934592 - 5)` | `UndecidedZero` | 5372.2619248635673 |
| `(1)/(x + (1-(1)/10^9)^8589934592 - 1/2)` | `UndecidedZero` | -0.49981403174973 |
| control: `^4294967296` (2^32) | `Ok` (1.03e4 units) | |
| control: `^(2^33)` and `^(-8589934592) - 1` | `Ok` (c0 of the first: `-1/(5 - (1 + 1/10^9)^(2^33))`) | 5372.26..., -0.99981403174813 |

HEAD prints `1/((1000000001/1000000000 ^ 4294967297) - 5) + ...` unchecked. Over the bar: no. This is
an honest refusal, and the outcome depends on how the same exponent is written.

Fix that closes the class: accept every exponent that fits in i64. Binary powering needs at most 64
squarings and 64 products, so charge `u (2 min(bits(k), 64) + 1)`. R1 still refuses a value outside
the window, and a wide ball only widens. For a literal exponent beyond i64, use `exp(k log |a|)`,
with the sign `(-1)^k` when the ball of a lies below 0, charged `3u`. §2.3 and the §2.16 row then
say 64 bits in place of 33.

## Families that held

- **Near-guard scan, rerun on an idle machine** (`nearscan_rel`, `redteam3-numeric/nearscan_rt3.txt`,
  6.2 s). The slowest median rates are close to the §2.3 table:
  - log at `1 + 2^-255`, 256 bits: 2.50e7.
  - sin at `round((k + 1/2) pi)`, 256 bits: 2.97e7.
  - log at `1 - 2^-1023`, 1024 bits: 4.86e7.
  - Everything else is at least 4.2e7.
- **fnscan subset** (`atan,sinh,log,exp`). One cell scanned at 1.66e7: sinh(77/9) at 64 bits, 7.7 us.
  `onecall_rel sinh 77 9 64 5000`, run three times, gives medians of 8.1 (a preempted run), 5.4 and
  5.5 us, so 2.35e7 to 2.39e7. That agrees with the table's 2.4e7. Twelve other sinh arguments n/9 at
  64 bits are at least 2.49e7, and atan(44/3) is 3.33e7.
- **Composite constants at three precisions** (`src/rate.rs`, `inputs/rate1.txt`, 33 expressions):
  - powers `exp(b log a)` with a = 1 +- 2^-k (k = 62, 254, 1021, 1022) and b up to `2^1020/3`;
  - chains of eight atans, of sinh, and of log and exp near 1;
  - tan, sin and cos at the convergents 355/113 and 103993/33102;
  - erf at 13/2, 53/4 and 106/4;
  - `sqrt(1 + 2^-1022)`.

  The slowest rate is 6.2e7 units/s (`(1 + 1/2^62)^(1/3)` at 64 bits). A composite charges its
  leaves and its adds as well, so its rate is above the single-operation floor.
- **End to end, the 64-bit floor through the point.** The point K is an alternating 7000-level chain
  `atan(sinh(...(1/3)...))` (`inputs/pt_chain_7000.txt`, 42 KB), with `f = sum (1)/(x - j)`, so that
  every term zero-tests `K - j` at 64 bits.
  - In `rt2_rel engine` (parse on 1 GiB, the call on 8 MB): 4000 terms are refused at 1e8 after
    1.90 s with VmHWM 23 MB, and 100 terms are `Ok` in 1.87 s with 9.2e7 units. That is about 5e7
    units/s, a quarter of the bar.
  - In `both` mode, 7000 levels overflow the §2.12 probe parser on 8 MB. That is the parser's known
    limit of 4438 levels (structure lens). The engine is not involved, and 4000 levels in `both` mode
    are refused at 1e8 after 1.92 s.
- **Memory of the 64-bit memo** (`inputs/sprod_k.txt`, `(1)/(x + (1+sin(1))*...*(1+sin(k)))`):
  - k = 12, 14 and 15 are `Ok` with 6.8e6 to 6.4e7 units and at most 1.52 bytes per unit.
  - k = 16 and 17 are refused at 1e8 after at most 0.92 s, with VmHWM at most 226 MB and at most 1.72
    bytes per unit.
  - The hashbrown resize transient (an old and a new table) is inside these figures. By count it is
    at most about 3.6 bytes per unit of the memo alone, for one moment, below the bar.
- **erf of computed arguments just below X_1024**: `erf(26 + 3 sin(1)/5)` (26.50) and its negative,
  in 4 to 32 indeterminate groups (`inputs/erfgrp_k.txt`). With 4 and 8 groups the result is
  `UndecidedZero` in at most 0.04 s. With 16 and 32 groups it is refused at 1.03e8 after at most 0.08
  s. erf is over-charged by about 10 times here.
- **Debug, 2 MB thread**: `(1)/(x + (sin(L) + 1)*(1+y1)*...*(1+y10))`, with L the 1023-bit
  convergent numerator of pi/2 (sin near -1 at 1024 bits, 1024 groups), gives `UndecidedZero` in
  0.31 s. No stack use grows with the precision. GMP's reentrant alloca puts blocks of at most 32 KB
  on the stack, and MPFR_ALLOCA_MAX keeps larger blocks off it.
- **Complex values never get a real ball.** These eleven inputs are exact zeros or complex constants:
  - `I*sqrt(-1) + 1`, `I*(-1)^(1/2) + 1`, `I*log(-1)/pi + 1`, `exp(I*pi) + 1`;
  - `I*asin(2) + log(2 + sqrt(3)) + pi/2`, `(-8)^(1/3) - 1 - I*sqrt(3)`;
  - `acosh(1/2) - I*pi/3`, `atanh(2) - log(3)/2 -+ I*pi/2`;
  - `abs(sqrt(-4)) - 2`, `(sin(2) - 1)^(1/2) - I*(1 - sin(2))^(1/2)`.

  Each is `UndecidedZero` in at most 0.3 ms. This includes the nonzero one of the atanh pair (honest).
- **Negative and large integer exponents of negative multi-term bases**: `(sin(1) - 2)^(-5000) + 1`,
  `^-5000`, `^5000 - 1`, `^(-5001) + 1` and `(sin(1) - 3/2)^(-5001) + 1/2` are all decided, with both
  parsers. They go through the inverse kernel and binary powering, so the negated exponent node does
  not reach `exp(b log a)`.
- **Values against gp** (`checks/coefcheck.py`): every coefficient is equal for
  - `(1)/(x + tan(2^100) + 1/2)`,
  - `(1)/(x + arcsin(1 - (PQ - e)) - 1)`,
  - `(1)/(x + log(PQ - e) + 84)`,
  - `(1)/(x + sin(1) + 1)`,
  - and the sinh control of m1.
- **Source review of the charge.**
  - `ball_cost` and `ball_expr` see the same distinct nodes: the `seen` set and the memo are both keyed
    by the Arc pointee address. The root is never a child.
  - `ball_cost` walks the arguments of a Function and the leaf of an integer exponent, which
    `ball_expr` does not evaluate. That only over-charges.
  - Every MPFR call is inside a charged node.
  - The prototype charges `to_expr` of A and B (`expr_units`) at each precision.
  - `floor_small` runs at most once per expansion: its outcome ends the expansion.
  - Tan's `3 trig` is 5 % below `sin_units + trig` at 1024 bits, which is covered by the measured
    7.7e7 to 8.4e7.
- **Rigor checks I made by hand.**
  - The Lipschitz bounds at a rounded end (exp, sinh and cosh at `round(|m| + r)`) hold, because
    `round(m + r) >= m + r - r/2` whenever `r >= ulp(m)/2`. That holds for every rounded ball, binade
    crossings included, and the second-order deficit `r^2/24` is covered by `2^(4-p)`.
  - The floor test of `non_integer` places the interval in (k, k + 1).
  - Rule 5 needs both A and B enclosed. An enclosure for a non-real node is impossible: log, sqrt,
    asin, acos and acosh are guarded, real powers require a > 0, and `I` and functions have no ball.
