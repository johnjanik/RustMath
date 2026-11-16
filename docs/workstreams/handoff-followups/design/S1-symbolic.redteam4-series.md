# S1-symbolic, red team 4: series arithmetic and charged cost

Design attacked: `fu2/design/S1-symbolic.md`, revision 8 (all of it, with §2.1, §2.4, §2.6, §2.7,
§2.15, §2.16 and the Revision sections). Prototype: `EXP/rev8/proto12/` (`EXP` =
`fu2/design/S1-symbolic`), the binaries of `EXP/rev8/bin/` unchanged (`proto12_rel`, the counting
build `proto12m_rel`, the debug `proto12`, the §2.12 parser probe `rt2_rel`, HEAD's `taylor` in
`headsimp_rel`). HEAD is 6e03987 for S1's crates. Every run went through a public entry point
(`taylor_with_budget`, `series_expansion` then `to_expr`, the `limit_via_series` path with
`LIMIT_SERIES_WORK`), budget 1e8, release on the main thread unless stated, with a 4 GiB address-space
cap and a 60 s timeout (`tools/r.py`). References from gp 2.15.

My files are in `EXP/redteam4-series/`: `tools/r.py` (one call), `tools/fam.py` (a family table:
time, units, rate, VmHWM, heap peak), `tools/repro.sh` (every reproduction below, output
`repro.txt`; with `BIN`, `BIN2`, `DBG` set to the fix sketch, output `repro_fix.txt`), `inputs/`,
`gp/*.gp` with their outputs `gp/*.out`, `out/fam1.txt` to `out/fam3.txt`, `fix_sketch.diff` (the
fix sketch below, built and checked, binaries deleted) and `gpcheck3_fix.out` (the gp oracle on it).

Verdict: two holes over the bar. H1 is a wrong decided result (bar a). H2 is a release crash on
15-byte inputs that HEAD answers (bar b). I found no budget hole: no family ran slower than 7.5e7
units/s, and no call held more than 1.36 bytes of heap per unit.

## H1 (over the bar, bar a): the EXACT sentinel is a value that exponents and precisions can take

Design sections: §2.1 (`EXACT = i64::MAX`; "all exponent arithmetic ... checked or saturating";
"`EXACT` means that every term of the value is present"), §2.4 (the `mul` row, `inv`'s
`val = -vb`, `pow_int` items 2 and 5), §2.6 (the loss-driven step, "not capped"), §2.16 (`mul`,
`pow_int`). Prototype: `ser.rs` `cadd`, `padd`, `mul` (`out_prec`), `pow_int`, `inv`, `expansion`.

What happens. `EXACT` is `i64::MAX`, and `i64::MAX` is also a legal exponent and a legal result of a
checked precision sum. Five producers conflate them:

1. `pow_int` item 2 stores an exact monomial at exponent `n v = i64::MAX` with precision `EXACT`.
   Then `add` and `sum_series` drop it (`e >= prec`) and keep the sum `EXACT`.
2. `mul` with exact operands takes `cutoff = EXACT` and drops a pair whose sum is `i64::MAX`. Its
   `out_prec` is `cutoff` when a pair was dropped, so it is `EXACT` again. §2.1 says "`mul` keeps
   `EXACT` only when no pair was dropped"; with this cutoff it does not.
3. `padd(va, pb)` returns the finite sum `va + pb = i64::MAX`, which reads as `EXACT`. A truncated
   product then claims to be complete.
4. `pow_int` item 5 returns the kept term with precision `n vb + 1`, which is `EXACT` when
   `n vb = i64::MAX - 1`.
5. The loss-driven step of §2.6 can set the working precision to `i64::MAX` exactly
   (`((1)/(1-x^9223372036854775806) - 1)/x^9223372036854775806` at order 0: P = 1, then P =
   9223372036854775807). With P = `EXACT`, `inv` of an exact series would return precision `EXACT`,
   and `pow_int` item 4 an exact zero. Here the next attempt meets H2 first and panics.

Reproduction (`tools/repro.sh`, output `repro.txt`). Every `Ok` takes at most 0.004 s, at most
5 600 units and 3.6 MB of VmHWM. The same output comes through `taylor_with_budget` and
`series_expansion`. The §2.12 probe parser (`rt2_rel both`) prints the same results, and so does the
debug build on a 2 MB thread (`max1`, `max3`, `mulmax2`, `prec1`, `item5` checked). HEAD prints `0 ^ N` forms (for example
`(((0 ^ 4611686018427387904) * (0 ^ 4611686018427387903)) / (0 ^ 9223372036854775807))`), so this is
not a regression. It is a decided wrong value of the kind §1.4 sets out to remove.

| input (at 0) | order | prototype (stored terms, precision) | true value (gp) |
|---|---|---|---|
| `(x^(2^62)*x^(2^62-1))/x^(2^63-1)` (32 bytes) | 0 | `0`, `[]`, `EXACT`; limit path `Finite(0)` | 1 (`2^62 + (2^62-1) == 2^63-1`, `gp/max.out`) |
| `(x^4611686018427387904*x^4611686018427387903)/x^9223372036854775807` | 0 | `0`, `EXACT`; `Finite(0)` | 1 |
| `(x^9223372036854775807 + 1)/x^9223372036854775807` | 0 | `1/x^9223372036854775807`, `EXACT` | `1/x^9223372036854775807 + 1` |
| `((1)/(1-x^9223372036854775807) - 1)/x^9223372036854775807` | 0 | `0`, `EXACT`; `Finite(0)` | 1 (gp with N = 7: `1/(-x^7 + 1)`, value 1 at 0) |
| `(x^9223372036854775806*x)/x^9223372036854775807` | 0 | `0`, `EXACT`; `Finite(0)` | 1 |
| `((x^(2^62)+x^(2^62+1))*(x^(2^62-1)+x^(2^62)))/x^(2^63-1)` | 2 | `0`, `EXACT`; `Finite(0)` | `1 + 2x + x^2` |
| `((1+x)*x^(2^62)*x^(2^62-1))/x^(2^63-1)` | 2 | `0`, `EXACT`; `Finite(0)` | `1 + x` |
| `((1)/x^(2^62)*(1)/x^(2^62-1))*x^(2^63-1)` at oo | 0 | `0`, `EXACT` | 1 |
| `(x^(2^63-4)*((1)/(1-x)))/x^(2^63-2)` (35 bytes; producer 3) | 2 | `1/x^2 + 1/x + 1`, `EXACT` | `x^-2 + x^-1 + 1 + x + x^2 + O(x^3)` (`gp/prec.out`) |
| `(x^(2^63-4)*exp(x))/x^(2^63-2)` (producer 3) | 2 | `1/x^2 + 1/x + 1/2`, `EXACT` | `x^-2 + x^-1 + 1/2 + 1/6 x + 1/24 x^2 + O(x^3)` |
| `((x+x^2)^9223372036854775806)/x^9223372036854775806` (producer 4) | 1 | `1`, `EXACT` | `1 + 9223372036854775806 x + O(x^2)` (`gp/item5.out`) |
| `(((x+x^2)^9223372036854775806)/x^9223372036854775806 - 1)/x` | 0 | `0`, `EXACT`; `Finite(0)` | `9223372036854775806 + O(x)` |

Controls: one less in the exponent gives the right value
(`(x^4611686018427387903*x^4611686018427387903)/x^9223372036854775806` is `1`), and so do the same
functions with small exponents (`((x+x^2)^1000)/x^1000` is `1 + 1000 x`,
`(x^1000*((1)/(1-x)))/x^1002` at order 2 has all five terms with precision 3).

Over the bar: yes. Each row is a decided result (no O-term) that drops nonzero terms, and the limit
path returns `Finite(0)` for limits 1 and 2^63 - 2.

Fix that closes the class: the exactness of a series must not share its encoding with a finite
exponent or precision. The robust form is an out-of-band precision, `enum Prec { Exact, Abs(i64) }`,
with every exponent and precision sum checked. If the port keeps `i64::MAX` as the sentinel, §2.1 must
reserve it: an exponent sum that reaches `i64::MAX` is the overflow of §2.1 (above every cutoff, so
`O(h^P)` for a positive power and `Unsupported` where the result needs it); a finite precision sum at
or above `i64::MAX` is clamped to `i64::MAX - 1`, a weaker true claim, never `EXACT`; `mul` compares
its pair sums with a finite limit and returns a finite precision whenever a pair was dropped;
`pow_int` item 2 treats `n v = i64::MAX` as not fitting and item 5 needs `n vb + 1 < i64::MAX`; `inv`
refuses `-vb = i64::MAX`; the driver keeps P below `i64::MAX` (else `PrecisionExhausted`). Add the
rows above to §4.1 as `exact_sentinel` (the expected outcome: the gp value, or an honest refusal).

Checked: `fix_sketch.diff` is the second form in the prototype (`MAXF = i64::MAX - 1` as the largest
finite value; `cadd`, `padd`, `mul`, `inv`, `pow_int`, the driver). Built against the same rlibs, it
turns every row into an honest refusal (`Unsupported("a series exponent overflows i64")`,
`PrecisionExhausted`, or `BudgetExceeded` for producer 3), and the gp oracle `EXP/rev2/gpcheck3.py`
gives 107 OK, the same lines as proto12 (`gpcheck3_fix.out`). It also refuses the control at
exponent `2^63 - 2` (`PrecisionExhausted`): with `i64::MAX` reserved, an exact product term at
`i64::MAX - 1` has no finite precision above it. The out-of-band form keeps that control.

## H2 (over the bar, bar b): the units of a dense buffer of 2^60 slots or more wrap to 15

Design sections: §2.7 (`Work::charge`, the row "dense buffer of n slots: `16 + 16 n`, before the
allocation", and "the budget refuses them before the dense buffer is allocated"), §2.6 (P "not capped;
bounded by the attempt limit and the budget"), §2.16 (`alloc`, the memory row "dense buffer slot").
Prototype: `ser.rs::Ctx::alloc` (`n.saturating_mul(16)`) and `konst.rs:408` (`Work::charge` computes
`charge(16 + w)`).

What happens. For n of 2^60 slots or more, `16 n` saturates to `u64::MAX`, and the base `16 + w`
wraps to 15. The budget admits the buffer for 15 units, and `Vec::with_capacity(n)` panics with
`capacity overflow` in release. The debug build panics first, with "attempt to add with overflow" in
`Work::charge`. The design gives no rule for the arithmetic of the units, and its claim that the budget
refuses a large buffer before it is allocated relies on that arithmetic. The slot count comes from the
text: a pole order (the loss-driven step sets P = T + order), a valuation (`inv` of an exact series has
`P + vb` slots), the pole of a power (`pow_int`'s `P - n vb`), or the `order` of `series_expansion`.

Reproduction (`tools/repro.sh`). Each panics in 0.06 to 0.09 s through `taylor_with_budget`,
`series_expansion` and the limit path, except where noted. HEAD's `taylor` returns in 6 to 13 us:

| input (at 0, order 0) | bytes | slots | HEAD |
|---|---|---|---|
| `exp(x)/x^(2^61)` | 15 | `P = 2^61 + 1` in `exp_u` | `(1 / (0 ^ 2305843009213693952))` |
| `exp(x)/x^2305843009213693952` | 28 | the same | the same |
| `sin(x)/x^1152921504606846976` | 28 | `2 P` in `sincos_u` | `(0 / (0 ^ 1152921504606846976))` |
| `(1+x)^(1/3)/x^1152921504606846976` | 33 | `P` in `pow_const` | `((1 ^ 1/3) / (0 ^ ...))` |
| `((1)/x^100000000000000000 + 1)^65` | 33 | `P - n vb = 6.5e18 + 1` in `pow_int` (Miller) | `(((1 / (0 ^ ...)) + 1) ^ 65)` |
| `(1)/(x^2305843009213693952+x^2305843009213693953)` | 49 | `P + vb` in `inv` | `(1 / ((0 ^ ...) + (0 ^ ...)))` |
| `((1)/(1-x^9223372036854775806) - 1)/x^9223372036854775806` | 57 | the loss step sets `P = i64::MAX` (also H1, producer 5), then `inv` | (the limit path: `PrecisionExhausted(6)`, because its step overflows) |
| `exp(x)` through `series_expansion` at order `2^60 - 1` | 6 | `P = 2^60` | |

The boundary is exact: `exp(x)/x^(2^59)` and `exp(x)` at order `2^59 - 1` are `BudgetExceeded`
(about 9.2e18 units), `2^60` panics. On the 2 MB debug thread `exp(x)/x^(2^61)` and the `^65` row
panic in `Work::charge` ("attempt to add with overflow", `konst.rs:408`). The REPL's kernel catches the
unwind (`rustmath-jupyter/src/kernel.rs:126`), so the REPL line shows an error, but the library entry
points panic, a `cargo test` with these inputs fails, and the limit path panics inside `limit`.

Over the bar: yes, under (b), as red team 2's S-B2 was: a release crash on inputs of 15 to 57 bytes
that HEAD answers without error in microseconds.

Fix that closes the class: state in §2.7 and §2.16 that every units formula is computed with
saturating arithmetic and that `Work::charge` adds with `saturating_add` and compares the saturated
sum; and that `alloc(n)` refuses a slot count above `remaining / 16` before it casts or allocates
(`BudgetExceeded`), and without a budget a slot count whose buffer does not fit in `isize::MAX` bytes
(`LimitExceeded` or `Unsupported`), so no `Vec::with_capacity` can panic. Audit every `charge(a + b)`
of the engine for the same pattern. Tests: the rows above, asserted `BudgetExceeded`, on a 2 MB debug
thread.

Checked: the fix sketch adds `saturating_add` in `Work::charge` and the `isize` check in `alloc`. Every
row above is `BudgetExceeded` in at most 0.3 ms (the 57-byte row `PrecisionExhausted(1)`, through H1's
bound on P), on 8 MB release and on the 2 MB debug thread (`repro_fix.txt`), with the gp oracle
unchanged.

## Families tried that held

All release, budget 1e8, through `taylor_with_budget` unless stated; rates are units per second of
the plain release build, heap peaks from the counting build (`out/fam1.txt` to `out/fam3.txt`).

- Recurrences at `TAYLOR_ORDER` (20 000): `exp`, `tan`, `atan`, `erf`, `log(1+x)`, `(1+x)^(1/3)`,
  `(1+x/3)^(5/7)`, `asin`, `tanh`, `exp(sin(x))`, `(1)/(1-x-x^2)`, and `exp`, `log`, `sin`, `x^(1/3)`,
  `atan` at the point 3/7: `Ok` or refused in at most 1.24 s, at least 8.1e7 units/s (the Miller rows
  are the slowest), at most 1.36 bytes per unit and 120 MB of VmHWM.
- Multi-term symbolic coefficients: `exp`, `1/(1-.)`, `(1+.)^(1/3)` and `log(1+.)` of
  `x*(y1+...+y40)` at orders 10 to 100: refused in at most 0.39 s, 2.8e8 to 4.9e8 units/s, at most
  0.41 bytes per unit. Kernel coefficients (`exp` and `1/(1-.)` of `sin(1) x + ... + sin(59) x^59`,
  products of `exp(x sin(k))`, `sin(x (sin(1)+cos(1)))^3`): at least 4.0e8 units/s.
- Very sparse and wide inputs: products and powers of `sum x^(i^2)` (199 terms) and `sum x^(i^3)`
  (59 terms), the heap and dense paths of `mul`, and `1/(1+.)`, `exp`, `log(1+.)`, `(1+.)^(1/5)`,
  `sin` of the cubes at 20 000: at least 1.2e8 units/s, at most 1.0 byte per unit.
- High pole orders: `(1)/(x^100000*(1-x))` (`Ok`, 9.9e7 units, 0.13 s), `(1)/sin(x)^1000`,
  `(1)/(exp(x)-1)^300`, `(1)/(sin(x)-x)^100` at 1000: refused in at most 0.63 s.
- Large, negative and rational exponents: `(1+x)^(123456789/1000003)`, `(1+x+x^2)^(-123456789)`,
  `(2+x)^(-1/1000000007)`, `(x+x^2)^(-1000000)`, `(1+x)^(10^3000)`, `(x+sin(1))^(10^30)`,
  `(x+1+sin(1)^2+cos(1)^2-1)^1000` (the binomial path, `Ok` at 20 000 with 2.4e6 units),
  `(x+2)^(10^6)*(x+3)^(-10^6)`, and a 99 990-digit literal L in `(1+x)^(L/7)`, `(1+x)^(-L)`,
  `(2+x)^L`, `exp(x/L)`, `log(1+x/L)`, `1/(1-x/L)`, `sin(L x)` at orders 20 to 2000: each refused or
  `Ok` in at most 0.84 s; the L rows are refused at about 1.08e8 units in at most 5 ms.
- Coefficient growth: `exp(x/3)`, `(1+x/7)^(-1/3)`, `log(1+x/11)`, `sin(x/13) cos(x/17)`,
  `atan(x/19)`, `exp(x) (1+x/7)^(-1/3)`, `log(x)`, `1/x` and `x^(1/3)` at `10^30/7`,
  `log(1+log(1+x))`, `exp(exp(x)-1)` at 2000 and 20 000: the slowest is `log(1+log(1+x))` at 2000,
  1.34 s at 7.5e7 units/s.
- The charge of each loop against its work, read in `ser.rs`: every recurrence step charges at least
  one unit per visited offset and a base of 17, and the visit of a zero coefficient reads the shared
  zero constant; the `as_rational` copies in `Konst::mul` and the rendering are followed by a charged
  operation of at least their size; `abs` and `sign` copy each rational coefficient once, uncharged, but
  measured this is small: `sign(L*((1)/(1-x)))` with a 99 990-digit L at 20 000 (20 000 coefficients
  sharing L) takes 0.04 s and 1.1e7 units, two such summands 0.07 s.
- Coefficients against gp (`checks/gpres.py` with the §2.12 probe, 300 digits, all equal):
  `(1+x+sin(1)^2+cos(1)^2-1)^70`, `(x^2+x+1)^1000/(x^3+2)^997` at 1, `(1+x)^(10^20) (1-x)^(-10^20)`,
  `(x^3-x)^(1/3)` at 2, `(exp(x)+sin(x))^(7/3)`, `log(cos(x)+x^3)/(1+x)^(1/7)`,
  `atan(exp(x)-1)+asin(sin(x)/2)`, `x^x` at 3, `exp(x/(1-x)) log(1-x^2)`, `log(x)^3/(x-1)^2` at 1,
  `sinh(x)^7/(cosh(x)-1)^3`, `(2+x)^(x+1/2)`, `(1-4x)^(-1/2) (1+x^5)^(1/5)`, `(x+x^3/7)^65/x^60`,
  `(1+x)^(123456789/1000003)` at 40, `exp(1/(1-x))`, `cos(x+sin(1))^3`; branches (`coefcheck.py`):
  `log(x-2)`, `sqrt(x-4)`, `(-x-4)^(-3/2)`, `atan(x+2)`, `asin(x+2)`, `acos(x-3)`, `log(x-I)`,
  `(x-1)^x`, `(x^2-2)^(1/4)` at 0 equal. `(-1-x)^(1/3)` and `(x-8)^(1/3)` differ from gp's series,
  which takes the real cube root of a negative rational, while gp's scalar `(-1)^(1/3)` and the
  engine take the principal value (§2.2): a convention, not a defect.
- Exponents just below the sentinel: products and quotients with exponents up to `2^63 - 2` are
  right (the controls of H1), and every overflowing exponent sum other than `i64::MAX` itself is an
  honest `Unsupported`.
