# S1-symbolic, red team 3: series arithmetic and charged cost

Design attacked: `fu2/design/S1-symbolic.md`, revision 7 red-team round 2 (all sections, with §2.1,
§2.4, §2.5, §2.6, §2.7, §2.15 and §2.16). Prototype: `EXP/rev7rt2/proto10/` (`EXP` =
`fu2/design/S1-symbolic`), the binaries of `EXP/rev7rt2/bin/` (`proto10_rel`, the counting build
`proto10m_rel`, the debug `proto10`, the §2.12 parser probe `rt2_rel`, `replshow_rel`), unchanged.
HEAD is 2e15d1c for S1's crates. Every run went through a public entry point (`taylor_with_budget`,
`series_expansion` then `to_expr`, the `limit_via_series` path with `LIMIT_SERIES_WORK`), budget 1e8,
release on the main thread unless stated, with a 4 GiB address-space cap and a 60 s timeout
(`tools/r.py`). Coefficients were checked against gp 2.15 (`tools/cc2.py`, which substitutes
`point + h` into gp's text, and `tools/coefcheck.py` from `EXP/rev7rt2/checks/`).

My files are in `EXP/redteam3-series/`: `tools/r.py` (one call), `tools/cc2.py` (gp check),
`tools/scan.py` (random compositions), `tools/repro.sh` (every reproduction below, output
`repro.txt`), `inputs/`, `gp/h1.gp` with its output `gp/h1.out`, `scan7.txt`, and `h1_fix.diff` (the
fix sketch for H1, built and checked as described below; the binary was deleted). `bin/` holds
symbolic links to `EXP/rev7rt2/bin/`.

Verdict: one hole over the bar (H1, a wrong decided result, bar (a)) and two below it (m1, m2). I found
no budget hole: no series family ran slower than 6.0e7 units/s, and no call used more than 1.4 bytes
per unit.

## H1 (over the bar, bar (a)): an empty truncated argument of negative precision is taken as valuation >= 0

Design sections: §2.4, the rows `exp`, `sin_cos`, `sinh_cosh` (and `tan`, `tanh` through them), and
"arctan ... erf"; §2.5, the rule `0^r` and `l^r = exp(r log l)`; §2.6, the list of `NeedPrecision`
producers; §2.1, `val` of an empty series. Prototype: `ser.rs::require_nonneg_val`, `integrate`,
`pow_const_base`.

What happens. A truncated series with no stored term, `O(h^q)`, has an undecided valuation of at
least q. The rows of `inv`, `pow_const` and `log` say "b not empty ... truncated empty:
`NeedPrecision`". The rows of `exp`, `sin_cos` and the atan family say only "vb >= 0", and the
prototype's `require_nonneg_val` returns `Ok` for every empty series. So `exp(O(h^q))` with q < 0
returns `O(h^q)`. That claims a valuation of at least q for a function whose argument may have a pole
of order up to -q, which has no Laurent expansion. The claim alone only makes the driver retry. But a
factor `h^N` lifts the precision to `q + N`; when that reaches the target, the driver returns a decided
expansion of a function with an essential singularity. The argument is empty at the first working
precision whenever it is a cancellation whose leading term lies at or above P, for example
`(sin(x) - x)/x^4` at P = 3.

Reproduction (`tools/repro.sh`; each `Ok` below takes at most 0.01 s, 2.7e3 to 6.6e5 units and 4 MB of
VmHWM; the same output through `taylor_with_budget` and `series_expansion`):

| input | point, order | prototype | gp |
|---|---|---|---|
| `x^10*exp((sin(x)-x)/x^4)` | 0, 2 | `0`, stored `[]` with precision 3 | `domain error in exp: valuation < 0` |
| `(1)/(1-x) + x^10*exp((sin(x)-x)/x^4)` | 0, 2 | `1 + x + x^2`, precision 3 | domain error |
| `x^10*sin(..)`, `cos`, `sinh`, `cosh`, `erf(..)`, `2^(..)` of the same argument | 0, 2 | `0` each | `domain error in sin` (and exp) |
| `(x-1)^10*exp((sin(x-1)-(x-1))/(x-1)^4)` | 1, 2 | `0` | domain error |
| `exp((sin((1)/x)-(1)/x)*x^4)/x^10` | oo, 2 | `0` | (the same function of 1/x) |
| `x^30001*exp((exp(x^30000)-1)/x^30001)` (38 bytes) | 0, 20000 (TAYLOR_ORDER) | `0`, precision 20001, 6.6e5 units, 4 ms | essential singularity at 0 |
| the limit path: `x^10*exp((sin(x^2)-x^2)/x^9)` | 0 | `Finite(0)` | the left limit is +infinity: f(-0.1) = 2.4e62, f(-0.05) = 1.1e566 (`gp/h1.out`) |
| the limit path: `1 + x^10*exp((sin(x^2)-x^2)/x^9)` | 0 | `Finite(1)`, also on the 2 MB debug thread | f(-0.1) = 2.4e62 |
| the limit path: `x^30001*exp((exp(x^30000)-1)/x^30001)` | 0 | `Finite(0)` | the right limit is +infinity |
| `0^((sin(x))/x)` | 0, 0 | `NotLaurent("0^r with r not positive at the point")` | r = 1 - x^2/6 + ..., so 0^r = 0 (the prototype says 0 at order 3) |

The §2.12 probe parser (`rt2_rel both`, one 8 MB thread) gives the same results. `limit_enhanced`
(`limits.rs:485`) calls `limit_via_series` for every direction, so a one-sided limit from the left of
the second limit row is also `Finite(1)`. HEAD's `taylor` prints the same wrong `0` and `1 + x + x^2`
(`rt2_rel head`), so this is no regression; it is a wrong result of the kind §1.4 sets out to remove.
The `atan`, `asin`, `acos`, `asinh`, `acosh`, `atanh` and `tan` rows are not affected: their inner
`inv` or `pow_const` meets the empty series and asks for precision. `erf` is affected through its
`exp(-b^2)`.

Over the bar: yes. A decided expansion (`0`, `1 + x + x^2`) and a decided limit (`Finite(0)`,
`Finite(1)`) of functions that have neither. The `0^r` row is the same defect in the other direction:
a decided `NotLaurent` for a function whose expansion is 0.

Fix that closes the class. State one rule in §2.1: an empty truncated series `O(h^q)` has an
undecided valuation of at least q, so it meets the precondition "valuation >= 0" only when q >= 0, and
no precondition on its leading coefficient (`NonZero`, a positive constant term, valuation 0). An
operation whose precondition an empty truncated operand does not meet returns `NeedPrecision`, as
`inv`, `log` and `pow_const` already do. Apply it to every row with such a precondition: `exp`,
`sin_cos`, `sinh_cosh`, the atan family with `erf`, `integrate` (an empty integrand of negative
precision may hold an `h^-1` term), and the rule `0^r` of §2.5. Add them to the `NeedPrecision` list of
§2.6. Tests for §4.1, with gp's domain error as the reference: the rows above are `NotLaurent` after the
retry, the limit path gives `Unknown`, `0^((sin(x))/x)` at order 0 is the exact 0, and
`exp((sin(x)-x)/x^2)` at order 2 stays `1 - x/6 + x^2/72`.

Checked: `h1_fix.diff` is this rule in the prototype (three guards in `require_nonneg_val`,
`integrate` and `pow_const_base`). Built against the same rlibs, it turns every row above into
`NotLaurent` (or `Unknown` on the limit path; `Unknown (PrecisionExhausted(134))` for the
`exp(x^30000)` row, an honest refusal), gives `0` for `0^((sin(x))/x)` at order 0, keeps
`exp((sin(x)-x)/x^2)` and `x^10*exp((sin(x)^2+cos(x)^2-1)/x^4)` (now `0 + O(x^3)` after one retry,
which is right), and passes the gp oracle `EXP/rev2/gpcheck3.py` with 0 failures
(`gpcheck3_fix.out`).

## m1 (below the bar): the loss-driven step crawls when a precision does not grow with P

Design sections: §2.6 (the loss-driven step `P += T - prec`), §2.4 "integer powers" item 5.

When `n vb >= P`, `pow_int` keeps the leading term with precision `n vb + 1`, which does not depend on
P. After a division by a high power of h the loss is small and constant, so each attempt raises P by
`T - prec` and the result's precision stays the same until P passes `n vb`. The 24 attempts end first.

| input | order | prototype |
|---|---|---|
| `((x+x^2)^1000)/x^999` (this is `x (1 + x)^1000`) | 1 | `Ok` |
| the same | 2, 5, 10, 40 | `PrecisionExhausted(27)`, `(102)`, `(227)`, `(977)` in 0.2 ms each |
| the same | 50 | `Ok`, 1.7e5 units |
| the same, the limit path | | `Unknown (PrecisionExhausted(102))` (the limit is 0) |
| `(sin(x)^1000)/x^999` | 10 | `PrecisionExhausted(227)` |
| `sin(x)^(1000000)/x^1000000`, `(x+x^2)^(-1000000)*(x+x^2)^1000000` | 2 | `PrecisionExhausted(51)` |

This is an honest refusal of a decidable input. HEAD prints `0/0` terms on the first input
(`rt2_rel head`), so it is no regression. Over the bar: no. Fix: grow P geometrically when an attempt
does not raise the precision. For example, take `P' = max(P + (T - prec), 2 P - T)` when `prec` is no
larger than the previous attempt's. Then 24 attempts reach about `2^23 T`, and the budget bounds their
cost.

## m2 (below the bar): `inv` computes its offsets without a check

Design sections: §2.1 ("Revision 7rt2 checked every `+`, `-` and `*` on an exponent ... the plain ones
left are bounded by a check just before them") and §2.16, the `mul` row's audit. Prototype:
`ser.rs:408`, `(e - vb) as usize`.

`(1)/((1)/x^2 + x^9223372036854775806)` at 0, order 3 (38 bytes): the debug build on a 2 MB thread
panics with `attempt to subtract with overflow` at `ser.rs:408`. Release wraps the difference, the cast
to `usize` gives back the true offset, and the result `x^2 + O(x^4)` is right. HEAD prints `1/0` terms.
So it is a debug-only crash on an input that HEAD does not handle. Over the bar: no; but the claim of
§2.1 is false, and a `cargo test` with this input would abort. Fix: compute the offset as
`e.checked_sub(vb)` (or `e.abs_diff(vb)`, a `u64`), drop any offset at or above n before the cast,
and recheck every `as usize` of a difference of exponents in the engine.

## Families tried that held

All release, budget 1e8, through `taylor_with_budget` unless stated.

- Recurrences at high orders. 250 random compositions of `exp`, `sin`, `cos`, `sinh`, `tan`, `log(1+.)`,
  `atan`, `erf`, `(1+.)^(1/k)`, `1/(1+.)`, `(1+.)^k` (k = 3, 65, 200, 1000), products, sums and
  quotients, over dense and sparse leaves, at orders 20 to 20 000 (`tools/scan.py`, seeds 1 and 7,
  `scan7.txt`): no call crashed or timed out; the slowest 1.66 s (`exp((1)/(1+(x^5)))` at
  20 000, 6.0e7 units/s with the counting build, 1.27 s and 7.9e7 units/s with the plain build), the
  most 1.36 bytes per unit. The bar needs below 5e6 units/s.
- The changed code of revision 7rt2: the sparse `log` (the 17 inputs of red team 2, `log`, `exp` and
  the REPL shapes: refused in at most 0.23 s or `Ok`), the checked span and the heap merge (`ovf_span`,
  `ovf_heap`: `Ok`, 3.6e3 and 1.5e5 units), the binomial path (`(x+sin(1)^2+cos(1)^2-1)^65` at 60 and
  `^1000` at 300: `Ok` in at most 0.05 s; the early break at an empty `u'^m` can only happen at m = 1,
  where the precision is already the slots), and the saturating `sincos_u`.
- Large, negative and huge powers: `(x+x^2)^(-10^6)`, `sin(x)^(-10^5)`, `(x^3+x^4)^(-3*10^5)`,
  `(exp(x)-1)^(-5*10^4)`, `((1)/x+1)^(10^6)`, `(1+x+x^2)^(-123456789)`, `(1+x)^(-10^6)*(1+x)^(10^6)` at
  orders 2 and 20 000: `Ok` or refused in at most 0.14 s, at most 136 MB of heap (1.36 bytes per unit).
- The charge of each recurrence step against its loops: every loop breaks at the first offset above k,
  and the zero constant has `words` 1, so each visited offset is charged at least one unit. The O(|b|)
  vector of `wbs` and the row loop of `mul` are uncharged but covered by the producer of the operands;
  in the binomial loop they total at most about S |u'| word reads, under 1e9 at the budget (about 1 s).
- Coefficients against gp (all equal, `tools/cc2.py` and `coefcheck.py`): `((1)/x^3+1+x)^70`,
  `((1)/(x^5(1+x)))^65`, `(2+x+x^2)^(-100)` at 30, `(1+x)^(1/3)(1-x)^(-7/5)` at 40,
  `exp(sin(x))log(1+x)/x^3`, `sin(x)^65/x^60`, `((x+x^2)^1000)/x^999` at 400, `((1)/x+2+x)^200`,
  `((1)/x+2+x)^(-70)` at 100, `(1+x)^1000 (1-x)^1000`, `(exp(x)-1)^70/x^75`, `(cos(x)-1)^(-33)`,
  `log(1+x^3+x^7)` at 60, `exp(x^5+x^11)/(1-x^13)` at 80, `(1+x^7)^(1/7)` at 100, `atan(x/(1-x))`,
  `asin(x)^3`, `cosh(x)/cos(x)`, `tanh(x)^2` at 30, `(1+x)^(1/x)`, `x^(x^x)` at 1,
  `(2+sin(x))^cos(x)`, `exp(exp(x)-1)`, `log(1+log(1+x))`, `(1-4x)^(-1/2)` at 60, `(1+x)^1000` at 120,
  `(3+x)^(-1000)`, `(x+2x^2)^(-65)`, `((1)/x+x)^65`, `sin(x)^(-70)`, `(exp(x)-1-x)^(-3)`,
  `cos(x)^(1/3)`, and at points 1, 2 and 3: `(x^2+1)/(x-1)^3`, `log(x)/(x-1)^2`, `exp(x)/(x-2)^5`,
  `(x-3)^70/(x^2-9)^68`, `sin(x-1)^3/(x^2-1)^2`, `atan(x)` at 1, `(x^2+3)^(1/3)` at 2.
- Precision bookkeeping, checked by reading `ser.rs` against §2.1 and §2.4: the precision of `mul`,
  `inv` (exact and truncated), `pow_int` items 2 to 7, `pow_const`, `log`, `exp`, `sin_cos`,
  `atan_like` with `deriv` and `integrate`, `trunc` keeping a first term, `scale` by 0 and the 0
  factor of a chain. Apart from H1, each matched the true precision.
