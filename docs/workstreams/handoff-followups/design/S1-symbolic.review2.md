# S1-symbolic design: adversarial review 2

Design reviewed: `fu2/design/S1-symbolic.md` (1350 lines, with "Revision 1"). HEAD c79a582.
My experiments: `fu2/design/S1-symbolic/review2/` (`refs.gp`, `hv.gp`, `neg.gp`, `branch.gp`,
`underflow.gp`, `lim_head.rs` built against the HEAD release rlib). I also ran the designer's revision 1
prototype (`S1-symbolic/bin/proto2`, `bin/proto2_rel`) and review 1's HEAD probe (`review1/repro`) on new
inputs.

Verdict: **not approved**. Two blockers, both in the rigorous numeric zero test (§2.3), both reproduced in
the revision 1 prototype that the plan tells the implementer to port. Everything else holds up well: the
revision fixes all four blockers of review 1, and the engine agrees with gp on every case I added.

## What I verified (holds)

- HEAD defects (review 1's HEAD probe, rerun): `taylor(x^2, 0, 8)` 1.12 s, 233 MB; `taylor(x^2, x, 10^30, 2)`
  panics; `x^2/4 -> (x ^ 1/2)`, `2/3^2 -> 4/9`, `x/2/3 -> x/(2/3)`, `-2^2/3 -> -(2^(2/3))`; `1/x`, `2^-x` and
  `x + 18446744073709551616` are parse errors.
- Revision 1 fixes of review 1, checked in the prototype: B1 (`revchecks`: the four complex zero constants
  give `UndecidedZero`), B2 (checked i64 exponents; the three inputs give the stated structures; overflow is
  `Unsupported`), B3 (constant powers kept as kernels in 2 ms; `x^(-10^9)+1` sparse in 2 ms; the multinomial
  and product inputs are refused at once with budget 1e6), B4 (stack-gated `Drop`, `rev1/sym_v10`; I read the
  code: only uniquely owned children with grandchildren are moved, every node is dropped once, the glue
  descends at most two levels below a node in the iterative branch, shared children are safe).
- gp references, rerun by me: H49E3 Bernoulli check is 1; `1/(sin(x)^50-x^50)` valuation -52, first
  coefficients -3/25, -62/125, 28 nonzero terms; `1/(sin(x)-x)`, `x^x` at 1, `1/(x^2+1)` at I,
  `1/(x+sin(1)+I)`, `x^2` at 10^30, `acosh` at 2, `(x^2+x)/(x^2+1)` at infinity. The prototype matches each.
- New cases (gp `neg.gp`): `exp`, `log`, `sqrt`, `abs`, `^100` of `1/sin(x) - 1/x (+1)` at orders 0, 1, 3. The
  prototype matches gp at every order (the loss-driven retry handles the negative intermediate precision).
- Branch cuts (`branch.gp`): `acos`, `asin` at 2, `sqrt`, `log` at -2. The prototype's coefficients are
  gp's, and the series equals the principal function on the cut side where it is continuous.
- Algorithms re-derived: Miller's power recurrence (Knuth §4.7), exp, log, sin/cos recurrences, the inverse
  precision `pb - 2 vb`, the `mul` precision, `non_integer_status` (the two balls of A - k and A - k - 1 put A
  in (k, k+1)), the symbolic zero-test rules 1 to 4 and the exact trig table. All correct.
- After cost (prototype, release): x^2 and 2*x+1 at order 20000 in 17 us and 15 us; exp(x) at 2000: engine
  13 ms, Display 81 ms, 5.35 MB, depth 15; sin(x) at 2000: 12 ms. Debug exp(x) at 2000: 31 ms. Matches §5.1.
- Callers (my grep): `repl.rs:1969` is the only external caller of the series API; `limits.rs:430`, `:554`;
  `lib.rs` tests; the doc examples in `series.rs` are `ignore`. `limit_enhanced` (the only caller of
  `limit_via_series`) is called by nothing, and the REPL `limit` arm (`repl.rs:2044`) calls `Expr::limit`,
  which never reaches the engine, so the unbounded engine is not reachable from the REPL. rustmath-py and
  rustmath-benchmarks call no series function. Parser users: `repl.rs:14`, `rustmath-py/src/symbolic.rs:6`;
  no test string in the workspace depends on `a/b` literals or signed float exponents.
- E0509 enumeration (`dropchain2.txt`): compiled for symbolic, calculus, functions, geometry, manifolds,
  curves (lib and `genus2_jacobian`), plot3d, jupyter, lib and test targets. The two curves integration
  tests that failed to compile there do not use rustmath-symbolic; benchmarks has no by-value match on
  `Expr`; no doctest in the workspace matches on `Expr`. The list is complete.
- `rustmath-reals` has no cycle; `with_val_integer`, `with_val_rational`, `pi`, `e`, `from_float`,
  `as_float` exist and round correctly. Toolchain 1.97 has `ptr::addr`.
- Out of scope items are justified: `rustmath-calculus::taylor` is another crate with its own copy; Puiseux,
  gamma/zeta series and iterative Display/PartialEq/substitute are not named in the item.

## Blockers

### B1. Ball arithmetic ignores MPFR underflow: a zero constant is decided NonZero (§2.3)

§2.3 states "for a result r the rounding error is at most |r| 2^-p". This is false when MPFR underflows:
a result below the smallest representable magnitude (MPFR default emin = 1 - 2^30, so below about
2^-1073741824) becomes 0, the ball becomes [0, 0], and the true value is outside it. A product with a large
finite number then gives a ball that excludes the wrong value.

Reproduction (revision 1 prototype, release and debug):

```
taylor((1)/(x + (1/2)^1073741827 * 2^1073741822 - 1/32), x, 0, 1)
  -> val 0 terms [0: -32/(1 - 32*((1/2)^1073741827 * 2^1073741822)), 1: -1024/(...)^2]
taylor((1)/(x + exp(-744261120)*exp(744261117) - exp(-3)), x, 0, 2)
  -> val 0 terms [0: -1/(exp(-3) - exp(-744261120)*exp(744261117)), ...]
```

Both constants are exactly 0 ((1/2)^1073741827 * 2^1073741822 = 2^-5; exp(a) exp(b) = exp(a + b)), so the
function is 1/x. The engine divides by an exact zero and presents the result as a decided expansion. The
symbolic rules cannot see it (the two powers are distinct kernels), so rule 5 decides, from the balls
[0, 0] * 2^1073741822 - 1/32. gp at 77 digits gives the constant as 5.4e-79 (`underflow.gp`), that is 0 to
working precision. HEAD returns the same kind of garbage, but the design claims a rigorous enclosure, and
the whole soundness argument of §2.2 rule 5 rests on it.

Required change: make every ball operation account for the exponent range. Either (a) add the absolute
underflow error 2^(emin-1) to every computed radius (then the product above gets radius about 1/8 and the
test is `Unknown`), or (b) refuse the enclosure (return `None`) as soon as any midpoint or radius leaves a
safe exponent window, for example |exponent| > 2^29, so that no accepted enclosure ever underflowed or
overflowed. State the rule in §2.3, port it, and add both inputs above as `undecided_after_underflow`
tests (`UndecidedZero`).

### B2. The numeric zero test runs MPFR sin/cos/tan on huge arguments without a charge (§2.3, §2.7)

The numeric test is charged `words * (p/64 + 1)^2 * 8` units per precision, which depends on the size of
the expression, not on the exponent of the arguments. MPFR's sin, cos and tan reduce the argument modulo
2 pi with as many bits of pi as the argument's exponent. A 12-character kernel such as `2^100000000` is a
ball with exponent 10^8 (the kernel is not expanded, its ball is formed by binary powering).

Reproduction (prototype, release, `BUDGET=100000000`, the provisional `TAYLOR_WORK`):

| input (order 2 at 0) | prototype | HEAD |
|---|---|---|
| `(1)/(x + sin(2^1000000) + 2)` | `UndecidedZero` after 0.35 s | |
| `(1)/(x + sin(2^100000000) + 2)` | `UndecidedZero` after **111.7 s** | 258 us (debug) |
| `(1)/(x + sin(2^1000000000) + 2)` | still running after 30 s (killed) | |

The time is wasted: the ball of `2^100000000` has a radius of about 2^(10^8 - 60), so the sin ball has
radius far above 1 and the test answers `Unknown` anyway. The same path runs whenever `zero_status` needs
the numeric test, including the kernel fallback of `pow_const` (prototype `konst.rs`, `c.zero_status()`
before `mk_kernel`). So one short REPL line bypasses the budget by more than 10 times (the X2 policy is
about 10 s), and the item's "no REPL bound prevents it" is not fully fixed. This is the same class of hole
as review 1's B3.

Required change: before calling an MPFR function whose cost grows with the argument (sin, cos, tan, and
sin/cos inside tan), check the input ball: if its radius is at least pi, or the exponent of its midpoint
exceeds p + 64, do not call MPFR and use the trivial enclosure (sin, cos in [-1, 1] with midpoint 0 and
radius 1; tan: no enclosure). Then `sin(2^100000000) + 2` is enclosed in [1, 3] and decided `NonZero` at
once. Charge each MPFR call by max(p, exponent bits of the argument) if any other function is found with
the same behaviour. Tests: `taylor_with_budget("1/(x + sin(2^100000000) + 2)", x, 0, 2, 10^8)` and the
`2^1000000000` variant return `Ok` (or an error) within 1 s in debug.

## Minor issues (fix in the revision)

- **m1. The precision search fails on simple high-order poles.** `pow_int` step 5 returns `O(h^P)` when
  `n * vb >= P`, which forgets that the valuation is exactly `n * vb` when the lead is decided. The
  following inverses then meet an empty series, the `NeedPrecision` doubling stops at extra 128, and the
  answer is `PrecisionExhausted(131)`:

  | input at 0, order 2 | prototype | gp (`refs.gp`) |
  |---|---|---|
  | `1/sin(x)^200` | `PrecisionExhausted(131)` | valuation -200, `x^-200 + 100/3 x^-198 + ...` |
  | `1/(x+x^2)^200` | `PrecisionExhausted(131)` | valuation -200 |
  | `1/(exp(x)-1)^150` | `PrecisionExhausted(131)` | valuation -150 |
  | `sin(x)^(-200)` | correct | same series as the first row |

  Honest, but avoidable and visible next to the working `sin(x)^(-200)`. Suggested fix: when the leading
  coefficient is decided nonzero, keep the leading term (`c0^n h^(n vb) + O(h^(n vb + 1))`) instead of
  `O(h^P)`, and let `trunc(P)` keep a decided first term; the inverse then has a decided valuation and the
  loss-driven retry (`P += T - prec`, uncapped) finishes in one step. Add the three rows as tests.
- **m2. Kernel creation is not charged.** Kernel keys are Display strings, and Display of a large exact
  integer is quadratic. Forty kernels `sin(2^340000 + k)` take 313 ms for 8.5e5 charged units (2.7e6
  units/s, five times below the slowest calibrated rate of 1.45e7 units/s in §5.3); 40 sums of the same
  integers without kernels take 7 ms. A long input of such kernels therefore runs several times longer
  than `TAYLOR_WORK` implies. Charge kernel creation by its key length (quadratic in the digits of
  embedded integers), or build keys without decimal conversion.
- **m3. `log(e^k) = k` is stated but not ported.** §2.2 lists it among the exact values; the prototype
  returns `UndecidedZero(-3 + log(e^3))` for `1/(x + log(e^3) - 3)`. Implement it with a test, or delete
  the sentence.
- **m4. Drop guarantee wording.** The §2.14 row "10^6-term sum dropped after 520 frames ... ok" ran after
  shallow drops on the same thread (`rev1/workload.rs`, `shapes`), which had already set the recorded
  stack top. When a thread's first `Expr` drop happens deep in its stack, the glue may still use 16 KiB
  there. The design says this in the correctness list; say it in that table row and in the doc comment of
  `Drop for Expr`.
- **m5. `TAYLOR_ORDER` is not an output-size bound.** `(1+x)^(10^100)` at order 1400 is admitted
  (7.4e7 units, engine 0.21 s) and prints 95.8 MB (Display 6.1 s, release). The budget, not the order,
  bounds the output, because the units track coefficient words. Say so in the `TAYLOR_ORDER` and
  `TAYLOR_WORK` doc comments (J1b's 16 MiB summary rule relies on it).

## Summary for the orchestrator

Covered and sound apart from the two blockers: truncated Laurent arithmetic over symbolic constants (F1
memory and time; x^2 at 20000 in 17 us against 1.92 GB at order 9 on HEAD), balanced output and the
left-spine `simplify` (F1 stack at order 20000), stack-gated iterative `Drop` with all E0509 sites listed,
the work budget with pre-charging and sparse series (REPL bound), the expansion point of any size (S7),
bignum literals and `a/b` precedence with a value-preserving fold (S7), Display parenthesization, all
callers, and tests with gp values. Blockers: B1, MPFR underflow makes the "rigorous" ball test decide a
zero constant nonzero (`1/(x + (1/2)^1073741827*2^1073741822 - 1/32)` at 0 gives a finite series of 1/0
terms); B2, MPFR sin/cos/tan of huge kernels are not charged (`1/(x + sin(2^100000000) + 2)` runs 112 s
under the REPL budget). Both have small, local fixes in `series_ball.rs`.
