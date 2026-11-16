# S1-symbolic, red team 2: the rigorous numeric layer

Design: `fu2/design/S1-symbolic.md`, revision 7 red-team round 1 (prototype `EXP/rev7rt1/proto9/`,
`EXP` = `fu2/design/S1-symbolic`). Binaries: `EXP/rev7rt1/bin/proto9_rel` and `proto9m_rel` (the
counting build), release, budget `TAYLOR_WORK` = 1e8 unless stated, each run under `ulimit -v 4000000`
and `timeout`. My probes, inputs and outputs are in `EXP/redteam2-numeric/`:

- `src/trigbig.rs`: sin, cos and tan through `Ball::unary` at integers N that are numerators of
  convergents of pi/2 with 26 to p bits (both parities of the denominator), at 64, 256 and 1024 bits,
  median of 200 and of 1001 calls (`out/trigbig.txt`, `out/trigbig_r2.txt`).
- `src/rigor2.rs`, `gen_rigor2.py`, `mkgp2.py`: 495 composite constants evaluated by `ball_expr` at the
  three precisions; every enclosure checked by gp (`rigor2_check.gp`, `rigor2_check_b.gp`, output
  `out/rigor2.txt`, `out/rigor2_gp.out`).
- `src/headprint.rs`: HEAD's `Expr::taylor`, printed.
- `inputs/sin60_groups*.txt`: the end-to-end group inputs.

`build.sh SRC NAME` rebuilds a probe against the HEAD rlibs of `EXP/rev5/rlibs_rel.txt` (the binaries
were deleted after the runs).

Verdict: no hole over the review bar in this lens. Two holes below it, both reproduced.

## Holes

### m1. The zero test stops at the first precision without an enclosure (minor)

Section: §2.2 rule 5 ("NonZero iff A and B both have real ball enclosures and one of them excludes
0, at one of the precisions 64, 256, 1024 bits") and §2.3 ("at each precision p of 64, 256, 1024");
§6 step 2 ports `ball.rs` as it is.

The body says a decision at any of the three precisions counts. The prototype returns `Unknown` at
the first precision where `enclose` gives `None` (`ball.rs::proves_nonzero_ab`, `_ => return false`;
`ball.rs::non_integer`, `_ => return (ZS::Unknown, nearest)`). But at 64 bits many enclosures fail
only because the ball is too wide: a domain check on a sub-value that cancels by more than about 60
bits (log, sqrt, `1/x`, fractional powers, asin and acos, acosh, atanh), tan whose 64-bit argument
has `r >= 1`, or the exp guard crossed by the radius alone. At 256 bits the same constant has an
enclosure that excludes 0. So the expansion is `UndecidedZero` where the design's own test decides
it, and where HEAD prints the correct series.

Reproduced (each in at most 0.9 ms, at most 4 MB VmHWM, a few thousand units; `PQ` is
`2922842896378005707/1075253811351460636`, a convergent of e with `PQ - e = 4.33e-37`):

| input at 0 | `taylor_with_budget`, `series_expansion` | limit path | 256-bit ball (`proto9_rel ball`) | gp |
|---|---|---|---|---|
| `(1)/(x + tan(10^18) + 1)`, order 5 | `UndecidedZero(1 + tan(10^18))` | `Unknown` | m = -7.388549680593688, r = 2.5e-56 | `tan(10^18)+1` = -7.38854968059368800013 |
| `(1)/(x + tan(2^100) + 1/2)` | `UndecidedZero` | `Unknown` | m = -1.282955149376719, r = 1.2e-44 | -1.28295514937671908862 |
| `(1)/(x + log(PQ - e) + 84)` | `UndecidedZero` | `Unknown` | m = 0.26931251352533764, r = 1.1e-40 | 0.26931251352533765287 |
| `(1)/(x + sqrt(PQ - e) - 1)` | `UndecidedZero` | `Unknown` | m = -1, r = 3.6e-59 | -0.99999999999999999934 |
| `(1)/(x + (1)/(PQ - e) + 1)` | `UndecidedZero` | `Unknown` | m = 2.3108701425424933e36, r = 2.5e-4 | 2310870142542493287101733017160886428.98 |
| `(1)/(x + (PQ - e)^(1/3) - 1)` | `UndecidedZero` | `Unknown` | m = -0.9999999999992436 | -0.99999999999924361747 |
| `(1)/(x + arcsin(1 - (PQ - e)) - 1)` | `UndecidedZero` | `Unknown` | m = 0.5707963267948967 | 0.57079632679489661830 |
| `x^(log(PQ - e) + 84)` | `UndecidedZero(84 + log(PQ - e))`, not `NotLaurent` | `Unknown` | (the 256-bit ball of alpha lies in (0, 1)) | 0.2693 |

The 64-bit ball of each is `None` (`ball None cost=611` and so on). HEAD's `taylor` returns
`1/(tan(10^18) + 1) - x/(tan(10^18) + 1)^2 + ...` and the like (`bin/headprint`). The composite rigor
run shows the same class on exact zeros: `atanh(1 - 2^-k)` composites have no 64-bit ball (the
denominator `1 - x` contains 0) and do have 256 and 1024-bit balls (lines 140, 147 and 156 of
`out/rigor2.txt`). A second effect: in `pow_int` an `Unknown` leading coefficient takes the binomial
path, so `(x + tan(10^18) + 1)^200` at order 5 is `Ok` but costs 1.84e7 units where Miller's
recurrence would apply.

Over the bar: no. Each outcome is an honest error, and reviews 2 (m3) and 3 (m4) rated the same kind
of incompleteness (a decidable constant reported as `UndecidedZero`) as minor. It can be argued as
(b), since HEAD's output is correct on these inputs; but HEAD is right there only because it never
checks.

Fix that closes the class: `enclose_real` returns why it has no ball, `Never` (a non-real or opaque
node, an indeterminate, a function, an integer exponent above 2^32, a value outside the window) or
`TooWide` (a domain or guard check failed on a ball that contains a point of the domain: `lo <= 0 <
hi` for log, sqrt and fractional powers; `|m| <= r` for `1/x` and tan's cosine; `r >= 1` at the trig
guard; `|m| < 1 <= |m| + r` for asin and acos; the exp guard passed by `|m|` alone; the same for
acosh and atanh). `proves_nonzero_ab` and `non_integer` stop on `Never` and go on to the next
precision on `TooWide`, as the text of rule 5 already says. The charge stays a-priori per
precision, so the extra cost is at most the 256 and 1024-bit charges that a decided constant pays
today. Add the inputs above to `ball_unit_tests` and to §4.1 (`Ok`, with the gp coefficient).

### m2. A float literal is decided by its exact binary value (minor, a semantics question)

Section: §2.2 ("`Real(f)`: 0.0 is zero, a finite f is a `NonZero` kernel (no float folding;
documented)"), §2.3 (`Real` is an exact ball, `Ball::exact`).

`(1)/(x + 0.1 - 1/10)` at 0 is `Ok` through `taylor_with_budget` and `series_expansion` (95 us, 4811
units): `-10/(1 + (-10*0.1)) - 100/(1 + (-10*0.1))^2 x`, that is `1/c - x/c^2` with `c = 0.1 - 1/10 =
1/180143985094819840` (the binary value of 0.1 minus 1/10). `log(x + 0.1 - 1/10)` is `Ok` too.
gp says `0.1 - 1/10` is `0.E-39` and refuses `1/(0.1 - 1/10)` ("impossible inverse in invr"); Sage
computes the same difference in RR and gets 0. HEAD prints the same expression as the design, so this
is no regression. The result is exact for the binary value, so it is not fabricated; but a user who
types 0.1 means one tenth, and the zero test then decides a value that exists only as a rounding
error of the literal. `0.5 - 1/2` is `UndecidedZero` (the float kernel is not folded), which is
consistent but surprising next to the 0.1 case.

Over the bar: no (exact under the design's semantics; HEAD gives the same).

Fix: state the semantics where users see it (the `taylor` doc comment and the REPL help): a float
literal denotes its exact binary value, so a difference of a float and a rational that rounds to it is
nonzero. Or, if floats are meant as approximations, give a `Real` leaf the radius of half an f64 ulp
(`|f| 2^-53`) in the ball code, so that `0.1 - 1/10` is `UndecidedZero` like any other constant the
test cannot separate from 0.

## Families that held

- **sin, cos, tan near k pi/2 at large exponents** (not in `nearscan.rs`, which takes k < 60). The
  arguments are convergent numerators N of pi/2 with 26 to p bits, as `Ball::rounded` gives them to
  the zero test (r = |N| 2^-p < 1, so MPFR is called). Slowest median rates over two runs, in units
  per second: sin 3.0e7 / 4.8e7 / 5.9e7 at 64 / 256 / 1024 bits (sin near +-1 at 64 bits with a
  55 to 60-bit N: 8.7 us in the first run, 5.2 us in the rerun, 264 units); cos near 0 4.2e7 / 9.4e7
  / 2.1e8; tan 3.5e7 / 6.1e7 / 8.0e7. sin near 0 at 1024 bits (MPFR's huge-cancellation path doubles
  its precision, then adds the cancellation, about 4p) runs at 1.7e8 and more. Every rate is above the
  2.5e7 that §2.3 gives as the slowest of the scans. The two runs differ by up to 1.7 times on the
  same arguments (machine noise), so I do not report the lower figures as a stale table.
- **Rigor of composite enclosures against gp** (`rigor2`). 495 constants: integer powers with
  exponents up to +-2^32 of bases near 1 (1 + 2^-20 to 1 + 2^-1000, 1 - 3^-40), of 3/2, -7/5 and of
  `sin(1)`, `cos(1/3)`, `1 + exp(-40)`; non-integer powers `a^b` with a near 1, 10^300, 10^-300 and b
  up to 2^26/3 and `sin(1)`; exp, sinh, cosh, tanh at rounded arguments next to the 2^27 guard and at
  `2^26 sin(1)`; 17 identities that are exactly zero (`sinh(asinh q) - q`, `tanh(atanh q) - q`,
  `cosh(acosh q) - q`, `exp(log q) - q`, `log(exp q) - q`, `sin(asin q) - q`, `cos(acos q) - q`,
  `tan(atan q) - q`, `atan q + atan(1/q) - pi/2`, `erf q + erf(-q)`, the log formulas of asinh and
  atanh, `sin 2q - 2 sin q cos q`, `tan q - sin q/cos q`, `exp q exp(-q) - 1`, `(q^(1/3))^3 - q`,
  `cosh^2 - sinh^2 - 1`) at 33 points next to the guards (`1 +- 2^-k` for k up to 1022, `2^-k`, `-3/7
  2^-k`, 355/113, 103993/33102, `2^61 + 1`, `10^15/3`, 7/2, 26, -27). 1396 enclosures at 64, 256 and
  1024 bits; 1375 checked by gp at 4500 digits, the rest at 1200 digits or by `sign` for erf of a
  huge argument: 0 outside its ball, and no exact zero has a ball that excludes 0. Slowest rate
  2.85e7 (a power of `1 + 2^-250` at 256 bits).
- **Lipschitz bounds at rounded ends.** exp, sinh and cosh evaluate L at `round(m + r)`. For |m| near
  2^26 at 64 bits a rounding down of the end costs a relative `|m| 2^-p`, more than the `2^(4-p)`
  inflation. It is still sound: every ball other than an exact one has `r >= ulp(m)/2`, so `m + r`
  rounds down by at most `r/2`, which the convexity slack of the mean-value bound (`r/2 - r^2/6`)
  covers. The rigor run includes these arguments.
- **Even roots of negative kernels.** `(sin(4)^2)^(1/2) + sin(4)`, `sqrt(sin(4)^2) + sin(4)`,
  `(sin(4)^4)^(1/4) + sin(4)`, `(sin(4)^2)^(3/2) + sin(4)^3` (all exactly 0): `UndecidedZero`; no
  power-of-a-power fold to the wrong branch. `((-2)^2)^(1/2) - 2` folds to `1/x`.
- **Float leaves.** Exact balls; `1.0e300*1.0e300 - 1.0e300^2` folds to `1/x`; `0.5 - 1/2` is
  `UndecidedZero` (see m2).
- **End to end, undecided groups at the slowest sin cell.** `(1)/(x + (sin(N)^2 + cos(N)^2 - 1) *
  (1+y1)*...*(1+y13))` with N = 1108341089274117551 (sin(N) = -1 + 1.79e-37 by gp), 8192 groups, each
  tested at the three precisions: refused at 1.0e8 after 0.14 s, heap peak 13.8 MB, VmHWM 19 MB;
  unbounded `UndecidedZero` after 0.76 s. With 1024 groups: `UndecidedZero` in 0.10 s. The
  1024-bit charges dominate, so the end-to-end rate is about 7e8 units/s.
- **MPFR 4.2.2 paths** (sources in `gmp-mpfr-sys-1.7.1`). sin.c: the "near 1" doubling and the "huge
  cancellation" doubling, both bounded by about 2p plus the reduction; cos.c: the `exps == 1` exit and
  the `cancel` step, at most about 2p; tanh.c: `set_one` for huge arguments and for `t = 1`; erf.c:
  the Taylor path only below its cutoff, which lies below `X_p`; acos.c: the supplement from `1 - |x|`;
  cache.c grows the pi and log 2 caches by at least 10 %. `MPFR_SINCOS_THRESHOLD` (23 540 bits) and
  `MPFR_EXP_THRESHOLD` (6522 bits) are above every precision the ball code reaches, and the hard
  assertions of exp3.c, log.c, atan.c, erf.c, sqrt.c, sin.c, cos.c and mul.c are out of reach behind
  the window and the guards. The rug and RealMPFR calls of `ball.rs` have no panic path for finite
  windowed values (the precision is at least 64, the shifts use `p.max(2)`, `to_integer` and
  `to_i64` are `Option`s).
- **`non_integer`'s floor.** `floor_small` is O((p/8)^2) bigint steps and uncharged, about 60 us at
  1024 bits. It runs at most once per expansion, because its outcome (`NotLaurent` or
  `UndecidedZero`) ends the expansion, and an integer alpha is decided before any numeric work.
