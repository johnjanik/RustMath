# S1-symbolic design: adversarial review 3

Design reviewed: `fu2/design/S1-symbolic.md` (1626 lines, with "Revision 2" and "Revision 1"). HEAD c79a582.
My experiments: `fu2/design/S1-symbolic/review3/` (`head_probe.rs` built against the HEAD debug rlibs,
`gcd_cost.rs` and `disp_cost.rs` against the HEAD release rlibs, `refs3.gp`, `hv.gp`, `repro_blockers.sh`).
I ran the designer's revision 2 prototype (`S1-symbolic/rev2/bin/proto3`, `proto3_rel`) on new inputs, with
`BUDGET=100000000`, the provisional `TAYLOR_WORK`.

Verdict: **not approved**. Two blockers, both budget holes of the class of review 1 B3 and review 2 B2: the
cost model charges big-integer operations far below their cost, so short REPL lines run 13 to 76 s under
`TAYLOR_WORK` (the X2 policy and §5.3 promise about 10 s). Both fixes are local to the cost model. The
revision 2 fixes of review 2 (B1 underflow window, B2 sin/cos guard, m1 to m5) hold.

## What I verified (holds)

- HEAD defects, my own probe (`review3/bin/head_probe`, debug): `taylor(x^2, 0, n)` 3.3 ms / 172 ms / 1.40 s
  and 4 MB / 34 MB / 234 MB at n = 5 / 7 / 8; exp(x) at 10: 0.36 s, 140 MB; 2*x+1 at 16: 77 MB; sin(x) at
  10: 157 MB. `taylor(x^2, x, 10^30, 2)` panics at `rustmath-integers/src/integer.rs:90:29`. Parser:
  `x + 18446744073709551616`, `9223372036854775808`, `1/x`, `1/sin(x)`, `2^-x`, `1/0` are parse errors;
  `x^2/4 -> (x ^ 1/2)`, `2^3/4 -> (2 ^ 3/4)`, `2/3^2 -> 4/9`, `x/2/3 -> (x / 2/3)`, `2^-1/3 -> (2 ^ -1/3)`,
  `-2^2/3 -> -(2 ^ 2/3)`, `asinh(x)` stays a `Function`. All as in §1.
- gp references (`refs3.gp`, `hv.gp`, PARI/GP 2.15): tan, csc, cot, Bernoulli gf, H49E3 (`== 1`), log cos,
  exp sin, triple pole, `(8+x)^(1/3)`, `x^x` at 1, `x^y` at 1, `exp(I*x)`, `1/(x^2+1)` at I,
  `1/(x+sin(1)+I)`, the pi points, the asymptotic case, `1/(2+sin(x))` at 1, acosh at 2, `x^2` at 10^30,
  `1/(x+log(2))`, `1/(sin(x)^50-x^50)` (valuation -52, -3/25, -62/125, -9113/13125, 28 nonzero),
  `1/(x+x^2)^200` (203 nonzero, last coefficient as in §4.1), `1/sin(x)^200` ([-200, 1, 100/3], 102 nonzero),
  `1/(exp(x)-1)^150` ([-150, 1, -75, 11225/4], 153 nonzero), `(1+x)^10000`, the limit `1/2`, the
  removable case. Every value equals the design's tables. The prototype's coefficients for
  `1/(2+sin(x))` at 1, `1/(x+sin(1)+I)`, `1/(x^2+1)` at I and `1/(x+x^2)^200` match.
- Review 2 fixes, rerun: the five underflow and overflow constants give `UndecidedZero` in 2 ms;
  `1/(x + sin(2^100000000) + 2)` and the cos twin are `Ok`, the tan twin and the `2^1000000000` variant
  `UndecidedZero`, all in 2 to 4 ms (`rev2/revchecks3.out`, rerun matches). m1: `1/sin(x)^200` 1.32 s
  (1.96e7 units), `1/(exp(x)-1)^150` 2.24 s (3.37e7), release. m2: 40 kernels `sin(2^340000+k)` are refused
  with budget 1e6 and `Ok` with 2e6 (1.28e6 units, 15 ms). m3: `1/(x+log(e^3)-3)` is exactly `1/x`.
  m5: `(x + sin(2^100000))^1000` at 1000 uses 1.38e5 units in 3 ms.
- Ball code (`rev2/proto3/ball.rs`) read in full. Every MPFR result passes `chk` (window R1) and the exact
  zero rule R2 is applied to products, sums and f(m). The inverse radius stays an upper bound after its
  roundings (the 1 + 2^(4-p) inflation covers the three roundings of `r / (|m| gap)`). The zero rule for
  `log(1)` after a rounded `x + sqrt(x^2+1)` is sound: the exact log of the rounded midpoint is 0 and the
  radius carries the earlier rounding. The window argument of §2.3 is correct.
- Zero test and kernel classes re-derived: rules 1 to 5; the NonZero classes of f(q) for rational q
  (arccos, arccosh at q != 1, arctanh with poles at +-1, log at q != 1, gamma, erf); the generic
  convention; `log_exact` (all factors positive reals). I found no unsound `NonZero`.
- Cost after (prototype): release x^2 at 20000 28 us, exp(x) at 2000 15.5 ms (1.46e6 units); debug
  2*x+1 at 20000 85 us, exp(x) and sin(x) at 2000 39 ms and 42 ms. Matches §5.1.
- Callers (my grep of every `*.rs` outside `target`): `rustmath-jupyter/src/repl.rs:1969`,
  `rustmath-symbolic/src/limits.rs:430`, `:554`, the `lib.rs` tests. `rustmath-calculus` has its own
  `taylor`/`series_coefficients`. `BigO` users: `limits.rs:547`. Complete, as §3.1 says.
- `simplify_binary` (`simplify.rs:470-620`) calls nothing recursive on its operands, so the left-spine
  rewrite of §2.10 keeps every result and order.
- Parser and Display: I walked the grammar of §2.12 and the Display rules of §2.11 over every operator
  pair; each printed form re-parses to the same value. `rustmath-reals` adds no cycle and no new native
  link (rustmath-py and 15 workspace crates already depend on it).
- MAGMA digest: §49.1.3, §49.1.4, §49.4.7-8 and §49.5 exist; H49E3 is the Bernoulli example. The digest's
  `Sqrt` takes a series of even valuation as a formal series; the design refuses `sqrt(x^2)` at 0 as
  `NotLaurent` because it expands the principal function, which is not analytic there. That is the
  honest choice for a symbolic `taylor`, and it is documented and tested (`branch_point`).

## Blockers

### B1. Rational normalization is charged linearly or not at all, but costs a quadratic gcd (§2.7)

`rustmath_rationals::Rational::new` (`rational.rs:19-35`) always reduces by
`gcd(numerator, denominator)`, and `Mul`/`Add` (`:279-345`) build the unreduced product or sum and call
`Rational::new`. The gcd is Euclid with full divisions (`rustmath-integers/src/integer.rs:1529`), which is
quadratic when both parts are large: I measured 0.29 s for 1016 x 907 words and 1.32 s for
2031 x 1814 words (`review3/bin/gcd_cost`), about 2.8e6 word pairs per second in release. Also `q * 7`
costs the same 0.32 s as the gcd, because `7 * num` and `den` are both large.

The cost table of §2.7 charges `SeriesConst::mul` and `scale` with `wa + wb` when one side is a single
word, the recurrence step with `1 + sum words * words` (linear in the coefficient when the other factor is
small), and an exact power of a rational with `bits(q) * |n| / 64 + 1` (linear). None of these counts the
gcd of the result. Reproduction (`review3/repro_blockers.sh`, prototype release, budget 1e8):

| input | point, order | wall | units charged | units/s |
|---|---|---|---|---|
| `x + (3/5)^200000` | 0, 1 | 12.7 s | 15 644 | 1.2e3 |
| `x + (3^330000)/(5^200000)*2*3` (31 characters) | 0, 1 | **75.7 s**, `Ok` | 5.94e7 | 7.8e5 |
| `x + (3^82000)/(5^50000)*2*3*5*7*11*13*17*19*23*29` | 0, 1 | 15.0 s | 3.73e6 | 2.5e5 |
| `x + (3^82000)/(5^50000)*(sin(1)+...+sin(8))` | 0, 1 | 13.2 s | 3.76e6 | 2.8e5 |
| `(2/3+x)^(1/7)` | 0, 2000 | 5.7 s | 1.51e6 | 2.6e5 |
| `sqrt(1+x/3)` | 0, 2000 | 1.8 s | 7.9e5 | 4.3e5 |
| `x+(1+x/3)^(1/7)*0+(1+x/5)^(1/7)*0+(1+x/11)^(1/9)*0+(1+x/13)^(1/3)*0` (67 characters) | 0, 2000 | 14.3 s | 4.96e6 | 3.5e5 |

The slowest rate in §5.3 is 1.28e7 units/s, so these run 30 to 10000 times longer per unit than the
calibration assumes. `x + (3^330000)/(5^200000)*2*3*5*7*11*13*17*19*23*29` would run about 275 s and still
pass the budget (each `*k` is one more 25 s gcd charged about 16 000 units). HEAD returns all of these
inputs at once (it never folds the large powers). The Miller rows show that ordinary rational-coefficient
series are affected too: every step `g_k = (1/k) ...` multiplies a coefficient with a large numerator and
a large denominator by a small rational, which is charged linearly and normalized quadratically. The
calibration of §5.3 missed it because its inputs (exp, sin, tan, exp(sin(x)) at 0) have numerator 1 or
small coefficients at the calibrated orders. The §2.7 sentence "every coefficient word in memory was
charged when it was produced, so the budget bounds the work" is false for these inputs.

Required change:

1. Charge every rational normalization before it runs, by the size of the unreduced result: for each
   rational product, sum or power that `Rational::new` will reduce, charge
   `GCD_UNITS * words(numerator) * words(denominator)` (a-priori: `(nwa + nwb) * (dwa + dwb)` for a product,
   `(nwa + dwa + nwb + dwb) * (dwa + dwb)` for a sum, `n * nw * n * dw` for `q^n`), skipping it when the
   denominator is 1. With 2.8e6 word pairs per second measured, `GCD_UNITS >= 5` keeps the slowest rate of
   §5.3. Apply it in `SeriesConst::add`, `mul`, `scale`, `pow_u`/`pow_rat_i`, `inverse`, the recurrence
   steps and the binomial path. Charging per combined term also removes the over-charge of minor m1.
   (Avoiding the gcds instead, by cross-cancellation for products with a small rational (Knuth TAOCP 2,
   §4.5.1) and a coprime constructor for `q^n`, needs a constructor of `Rational` from coprime parts in
   `rustmath-rationals`, which has none today; the charge is the change that stays in this crate.)
2. Recalibrate §5.3 in release with inputs whose coefficients have large numerators and denominators:
   `(2/3+x)^(1/7)` and `sqrt(1+x/3)` at 2000, `x + (3/5)^200000`, `x + (3^330000)/(5^200000)*2*3`, and
   inputs at a transcendental point (`exp(sin(x))`, `tan(x)`, `1/(2+sin(x))` at 1). Set `TAYLOR_WORK` from
   the slowest measured rate, as §5.3 already prescribes.
3. Budget tests (debug, outcome asserted, "returns within 20 s"): `x + (3/5)^200000` and
   `x + (3^330000)/(5^200000)*2*3` at 0, order 1, budget 1e8: `BudgetExceeded` (or `Ok`) within 20 s; the
   67-character zero-multiplied sum at order 2000 with the REPL budget: within 20 s. Each fails on the
   revision 2 prototype.

### B2. `Integer::nth_root` of large radicands is not charged (§2.2 radicals, §2.7)

`pow_const` of a rational to a power p/d calls `rat_root` (`rev2/proto3/konst.rs:893`: `nth_root` of the
numerator and of the denominator, then `a.pow(n) == num`), and `dth_power_part` (`:916`) calls
`nth_root` again on the cofactor (the perfect-power test of §2.2). The design charges radical extraction
`bits(n)/64 * 6542 + 1` only for the trial division, and only when n has at most 4096 bits; the
`nth_root` calls have no row in the cost table and are never charged. num-bigint's `nth_root` is Newton
with full-size divisions, quadratic in the size.

Reproduction (prototype release, budget 1e8, point 0, order 1):

| input | wall | units |
|---|---|---|
| `x + (3^340000+1)^(1/2)` (radicand 539 000 bits) | 0.20 s | 41 223 |
| `x + ((3^340000+1)^(1/2))^0 + ... + ((3^340000+100)^(1/2))^0` (2593 characters) | **14.3 s** | 4.12e6 |
| the same with 80 terms `(3^340000+k)^(1/2)` (1752 characters) | 12.5 s | 3.30e6 |

That is 2.9e5 units/s, 45 times below the calibrated rate; filling the budget takes about 2400 terms and
6 minutes.

Required change: before each `nth_root`, charge an a-priori bound of its cost (for example
`GCD_UNITS * words(n)^2` with the constant of B1, measured), or skip `rat_root` and the cofactor test when
the radicand has more than 4096 bits and keep `n^(1/d)` as a radical kernel without reduction (sound,
only less canonical, like the products of radicals of §7 item 6). Add the cost row to §2.7 and the
100-term input above as a budget test (within 20 s).

## Minor issues (fix in the revision)

- **m1. `add` over-charges constants with many monomials.** `SeriesConst::add` charges `(wa + wb)^2`
  over all terms when any denominator is not 1 (`konst.rs:286-290`), but each monomial's rational sum is
  normalized on its own. At a transcendental point the coefficients have many monomials, and the REPL
  refuses small expansions in milliseconds: `1/(2+sin(x))` at 1 is `Ok` at order 30 (22 ms, 8.5e7 units)
  and refused at 50 (27 ms); `exp(sin(x))` at 1 refused at 50 (51 ms); `log(cos(x))` and `tan(x)` at 1
  refused at 100 (0.42 s, 0.41 s). The effective rate is 1e9 to 4e9 units/s. Honest, but gp does these at
  once, and §5.3 has no input at a non-rational point. B1's per-term charge fixes it.
- **m2. Exact integer powers and radical folds.** `bits(q) * |n| / 64 + 1` is linear while num-bigint's
  powering is superlinear: `3^340000` takes about 6 ms for 15 937 units (2.7e6 units/s, 5 times below the
  calibration; 20 of them in 0.12 s). The radical fold of `reduce_mono` (`konst.rs:355-372`) computes
  `n^(e div d)` uncharged: `(sqrt(3)^524288)^0` costs 3 ms for about 27 units (200 copies: 0.54 s,
  5 397 units). Both are bounded per term and need inputs of tens of KB to pass 10 s; charge them with
  the B1 model (result size times a measured superlinear factor) during the recalibration.
- **m3. `SeriesExpansion::valuation()` is not a decided valuation.** It returns the exponent of `terms[0]`,
  whose coefficient may be an undecided zero: for `(sin(1)^2+cos(1)^2-1)/x + 1` at 0 it would return
  `Some(-1)`. §2.1 documents `Laurent::val` as a lower bound; the public method must say the same, or be
  named `first_exponent`, or return `Result<Option<i64>, SeriesError>` with `UndecidedZero` (CLAUDE.md:
  Decided versus Unresolved).
- **m4. erf has no ball enclosure in the prototype.** §2.3 lists erf (L = 2, `Float::erf`), but
  `ball.rs::unary` has no `Erf` arm, so `1/(x + erf(1) - 1/2)` would be `UndecidedZero`. Port it as the
  design says and add `2 + erf(1)` to the ball unit tests.
- **m5. The round-trip test does not say how values are compared.** `test_display_parse_round_trip` asks
  that `parse(format!("{e}"))` "simplifies to the same value" for every engine output of §4.1, including
  outputs with kernels such as `(3 + sin(2^100000000))^1000000000`, where no numeric comparison works.
  State the check (for example equality of the two `SeriesConst::from_expr` normal forms in one `Work`).
- **m6. Error text is unbounded.** `SeriesError::UndecidedZero(Expr)` is displayed in the REPL message
  `taylor: {e}`, and J1b's `OUTPUT_BYTES` covers values, not error messages. A 407-character input,
  `1/(x + (sin(1)^2+cos(1)^2-1)*(3^200002*sin(2) + ... + 3^200021*sin(21)))`, gives a 5.7 MB `UndecidedZero`
  text in 0.7 s (release); 200 terms give about 57 MB. Bound the constant printed by `Display for
  SeriesError` (for example the first 200 characters and the total length), and keep the full `Expr` in the
  variant for programmatic use.

## Summary for the orchestrator

Covered and correct apart from the budget: truncated Laurent arithmetic over symbolic constants with a sound
three-valued zero test (F1 time and memory: x^2 at 20000 in 28 us against 234 MB at order 8 on HEAD; every
gp value I reran matches), balanced output, the left-spine `simplify` and the stack-gated `Drop` (F1
recursion), the expansion point of any size and the bignum, precedence-correct literal parser (S7), all
callers, tests with gp references. Review 2's blockers are fixed. Blockers: B1, rational normalization (a
quadratic Euclid gcd inside every `Rational` operation) is charged linearly or not at all, so
`x + (3^330000)/(5^200000)*2*3` runs 75.7 s and `x + (3/5)^200000` 12.7 s under `TAYLOR_WORK`, and Miller
series with rational coefficients run 45 times slower per unit than calibrated; B2, `nth_root` of large
radicands is uncharged (a 2.6 KB line runs 14.3 s). Both are fixed in the cost model (§2.7) with a
recalibration of §5.3; nothing is declared out of scope that should not be.
