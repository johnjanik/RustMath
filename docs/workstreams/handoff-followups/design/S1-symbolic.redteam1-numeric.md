# S1-symbolic, red team 1: the rigorous numeric layer

Design: `fu2/design/S1-symbolic.md`, revision 7. Prototype: `EXP/rev7/proto8/`, built with
`EXP/rev7/build.sh release` against the HEAD rlibs of `EXP/rev5/rlibs_rel.txt`. My copies of the
binaries, my probe sources and every output are in
`EXP/redteam1-numeric/` (`EXP` = `fu2/design/S1-symbolic`). Probe sources: `src/myscan.rs` (one
enclosure, cold call plus the median of 200 calls), `src/broad.rs` (medians near 1, -1, 0 and k pi/2
for 16 functions), `src/sinscan.rs` (sin and cos at `round_p((k + 1/2) pi)` and `round_p(k pi)`),
`src/erfrat.rs` (erf at every `n/d` below `X_p + 1`), `src/ratscan.rs` (big rational and integer
leaves, and `literal_abs_bound`), `src/rigor.rs` with `mkgp.py` (enclosures against gp), `run.py`
(`run7.py` with a 4 GiB address-space cap and a timeout). `build.sh src/NAME.rs` rebuilds a probe
into `bin/` (deleted to save disk); `proto8m_rel` is `EXP/rev7/build.sh` output. All timings are release, on a machine
shared with other runs. A rate is the units `ball_cost` charges for the node over the median time.

Verdict: no hole at the review bar in this lens. Two minor holes, both reproduced.

## Holes

### m1. sin near +-1 and log near 1 run 4 to 18 times slower than the scan table says (minor)

Section: §2.3, "Scan of every enclosed operation" (the rows sin and log, the column "longest call",
and the sentence "every operation charges at least 2.1e7 units/s in every run").

Cause, from the MPFR 4.2.2 sources in `gmp-mpfr-sys-1.7.1`:

- `sin.c` computes sin as `sqrt(1 - cos^2)`. When the result rounds to +-1 it doubles its working
  precision ("Check if near 1: m += m") until `1 - |sin x|` is resolved. For a p-bit midpoint within
  an ulp of `(k + 1/2) pi`, `1 - |sin x|` is about `2^-2p`, so MPFR works at about 2p bits, plus the
  exponent for the reduction. fnscan never put a midpoint there.
- `log.c` adds the cancellation to its Ziv precision. At `1 +- 2^-k` with k near p it works at about
  2p bits. asinh and atanh of tiny arguments and acosh near 1 reach the same path through their log
  formulas, and so does `a^b` with a near 1 (`exp(b log a)`).

Measured (Ball::unary, median of 200 calls):

| case | 64 bits | 256 bits | 1024 bits | table (64 / 256 / 1024, longest) |
|---|---|---|---|---|
| sin, midpoint `round_p((k + 1/2) pi)` | 5.2 us, 192 units, 3.7e7 | 33.1 to 33.7 us, 864 units, 2.6e7 | 206 to 352 us, 9312 units, 2.6e7 to 4.5e7; slowest of 400 k: 522 us | 4.5e7 / 1.0e8 / 4.7e8, 20 us |
| log(1 +- 2^-k), k = p - 2 | 2.9 to 3.1 us, 128 units, 4.2e7 | 18.4 to 19.2 us, 464 units, 2.4e7 to 2.5e7 | 91 to 100 us, 4688 units, 4.7e7 to 5.1e7 | 3.6e7 / 7.0e7 / 3.2e8, 15 us |
| asinh, atanh of `L 2^-1022` | | 21 to 22 us | 102 to 108 us, 13936 units, 1.3e8 | longest 45 us |

The slowest median rate I found anywhere is 2.4e7 (log at 256 bits). It is above the floor of 1e7
units/s that `TAYLOR_WORK` assumes, so the bound "a numeric test charged 1e8 units runs at most about
5 s" still holds within the bar. End to end the effect is diluted by the other precisions and nodes:

- `(1)/(x + (sin(L) + 1)*(1+y1)*(1+y2)*...*(1+y13))` with `BIGLIT` L the 1023-bit numerator of a
  convergent of pi/2 (`conv_1023.txt`, `sin(L) = -1 + 2^-2049` by gp): 8192 indeterminate groups,
  each tested at 64, 256 and 1024 bits. Refused at 1.0e8 after 1.18 s, VmHWM 15 MB, heap 10.2 MB.
  With 4096 groups: `UndecidedZero` after 0.93 s budgeted, 0.98 s unbounded.

Fix that closes the class: add to fnscan the arguments within an ulp of `(k + 1/2) pi` and of `k pi`
(sin, cos, tan, both signs), `1 +- 2^-k` and `-1 +- 2^-k` up to k = p - 1 (log, acosh, the base of a
real power), and `2^-k` up to p - 1 (asinh, atanh, sinh, tanh), take medians rather than minima, and
correct the table and the "longest call" column. If the margin above the floor is to stay at 2x,
either charge sin and cos as an operation at `3p + 64` bits (the reduction plus the doubled
precision) or enclose sin through cos of the reduced ball (`sin x = cos(x - pi/2)`, rigorous with the
pi ball), which avoids MPFR's square-root path.

### m2. `literal_abs_bound` is an uncharged O(n) conversion per erf node (minor)

Section: §2.3 "erf", §2.7 (`erf_units`), `ball.rs::literal_abs_bound`.

`literal_abs_bound` reads `NumericConversion::to_f64` of the literal. For a `Rational` this is
`BigRational::new_raw(num.clone(), den.clone()).to_f64()` (`rustmath-rationals/src/rational.rs:163`),
which copies both parts and divides them. It runs inside `ball_cost` once per erf node per walk, and no
unit pays for it. `ratscan` with `t/(t+2)`, `t = 3^630900` (2 x 999 976 bits): 611 us for the bound,
548 us for the leaf's own conversion (`Ball::rounded` of `with_val_rational`, charged 31 346 units at
64 bits). Kernel interning lets at most two erf kernels (`erf(q)` and `erf(-q)`) name one literal, and
each textual literal is its own node, so the bound at most doubles or triples the leaf's time: about
1.8e7 units/s in the worst case. Such a literal is reachable without a 100 KB input, through a point:
`erf(x)` at a point with a 2 x 1M-bit value renders the kernel argument as a `Rational` leaf.

Fix: compute the bound in O(1) from the bit lengths and the top 64 bits of the numerator and the
denominator (an upper bound is enough for a charge), or charge `bits/64` units for it in
`unary_units`.

## Families that held

- **Rigor of every enclosure against gp.** `rigor.rs` builds the ball of each of 240 adversarial
  rationals at 64, 256 and 1024 bits and applies all 17 unary functions; `mkgp.py` writes a gp check
  of `|f(q) - m| <= r` with m and r exact (`i*2^e`), at 1300 digits (more for large sin arguments),
  with series and tail bounds for magnitudes below 2^-1000 and the erfc bound for the cheap erf
  regime. Arguments: `1 +- 2^-k`, `-1 +- 2^-k`, `+-2^-k`, `2 + 2^-k` for k from 1 to 1100;
  `(k + 1/2) pi` and `(k + 1) pi` to 2^-1100 for k up to 104 348; erf at `X_p` and at MPFR's cutoffs
  (6.7, 6.71, 7.5, 13.7, 14, 26.6, 26.65, 27, 28) +- 2^-70; `2^27 - 1`, `2^27`, 709, 745, 10^6 and
  their negatives for exp, sinh, cosh; `(3/2)^e` and `(2/3)^e` up to e = 100 000. 9523 checks, 0
  outside their ball, no ball for a non-real value (2783 cases have none, as the guards say).
- **Exact zeros are never decided nonzero.** 26 constants that are exactly 0 but not in normal form,
  each as `(1)/(x + c)`: `atan(1/2) + atan(1/3) - pi/4`, `atan(1) - pi/4`, `asin(1/2) - pi/6`,
  `cosh(log 2) - 5/4`, `sinh(log 2) - 3/4`, `tanh(log 2) - 3/5`, `tan(atan 2) - 2`,
  `sin(asin(1/3)) - 1/3`, `cos(acos(1/3)) - 1/3`, `exp(log 3) - 3`, `exp(1) - e`, `sqrt(exp 2) - e`,
  `exp(1/2)^2 - e`, `log 6 - log 2 - log 3`, `erf(1/2) + erf(-1/2)`, `2^(1/2) 3^(1/2) - 6^(1/2)`,
  `cosh(1)^2 - sinh(1)^2 - 1`, `sin 1 + sin(-1)`, `sin 2 - 2 sin 1 cos 1`, `tan 1 - sin 1/cos 1`,
  `acos 0 - pi/2`, `sin^2 + cos^2 - 1` and the I forms `1/I + I`, `(2I)^-1 + I/2`, `I^-3 - I`:
  every one `UndecidedZero` (or exactly 0 where the normal form folds it). HEAD returns a series
  with `1/0` coefficients on all of them, so none is a regression. (The prototype links HEAD's
  parser, which reads asinh, acosh and atanh as opaque functions; those three arms were checked
  through `Ball::unary` by the rigor check instead.)
- **Tiny nonzero constants are decided right.** `exp(pi sqrt 163) - 262537412640768744`,
  `erf(7) - 1`, `erf(8) - 1` (decided at 256 bits, above `X_64`), `tanh(20) - 1`, `sin 355`,
  `log(1 + 10^-15) - 10^-15`, `1 + sin(N)` with the 255-bit pi/2 convergent N (about 2^-512):
  `Ok`, coefficients of the right sign and form. `1 + sin(N)` with the 1023-bit N (2^-2049) and
  `sin^2 1 + cos^2 1 - 1 + exp(-1000)` are `UndecidedZero` (honest).
- **erf charge.** Every `n/d` below `X_p + 1`: d up to 120 at 64 bits (37 281 arguments), 40 at 256
  bits, 10 at 1024 bits. Slowest median rates 3.7e7, 6.7e7 and 6.6e8; isolated slow cells were noise
  (rechecked with 200 calls: `erf(1/2)` 4.7 us, `erf(61/25)` 4.5 us).
- **Broad median scan** (`broad.txt`): 16 functions, 319 to 871 arguments each near 1, -1, 0, k pi/2
  and over exponents -80 to 80. Apart from m1, every median rate is at least 2.3e7 (sinh at 64 bits),
  close to the table.
- **Guards.** asin and acos at `+-(1 - 2^-k)` up to k = p - 2, exact and rounded: no ball or a fast
  one. acosh at `1 + 2^-k`: at least 3.4e7. exp, sinh, cosh, tanh at `2^-k` across the MPFR fast-path
  boundary k = p/2: at least 4e7. MPFR hard assertions (`MPFR_ASSERTN` in sin.c, cos.c, log.c,
  erf.c, mul.c) are all out of reach behind the trig and exp guards and the window.
- **Big literals.** Rational and integer leaves of 2 x 1M bits, balanced, lopsided (`1/t`, `t/7`) and
  within `1/t` of a rounding midpoint at each precision: 5.2e7 to 8.1e7 units/s; the conversion is
  linear.
- **Indeterminate groups.** 3000 groups of `zeta(3)`, `sin^2 1 + cos^2 1 - 1`,
  `atan(1/2) + atan(1/3) - atan(1)`, `erf(1/2) + erf(-1/2)`, `log 6 - log 2 - log 3` (53 to 113 KB):
  `UndecidedZero` in 0.03 s or refused at 1e8 within 0.28 s, VmHWM at most 15 MB.
- **Precision escalation tricks.** A term whose radius leaves the window at 256 or 1024 bits stops
  the escalation early, but `ball_cost` of the failing precision is charged before the walk, so the
  test never runs the costly precision uncharged.
- **Cold first calls** are 5 to 10 times the median (MPFR constant caches and first allocations).
  MPFR keeps the highest precision of pi and log 2 and grows it by at least 10 %, and every needed
  precision is at most about 3p, so this is a one-time cost per thread, not repeatable by the input.
- **Memory and stack.** The memo is at most about 2.2 bytes per unit at 64 bits and 0.2 at 1024; the
  largest numeric probe peaked at 15 MB VmHWM. Ball code is iterative; Tan, asinh, acosh and atanh
  call `unary` one level deep.
