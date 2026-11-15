# S1-symbolic design: adversarial review 1

Design reviewed: `fu2/design/S1-symbolic.md` (922 lines, no Revision section). HEAD c79a582.
My experiments: `fu2/design/S1-symbolic/review1/` (`repro.rs` built against the HEAD debug rlib,
`refs_check.gp`). I also ran the designer's prototype binaries (`S1-symbolic/bin/proto`,
`bin/proto_rel`) on new adversarial inputs.

Verdict: **not approved**. Four blockers (two silent wrong-value or unsound-decision paths, one
budget hole that makes the new REPL bound bypassable and regresses HEAD, one feasible part of the
item declared out of scope). The core algorithm (truncated Laurent arithmetic over symbolic
constants with a three-valued zero test) is sound in the prototype and the cost claims hold.

## What I verified (holds)

- F1 cost at HEAD (my own program): x^2 at order 5/6/7/8: 3.9 / 7.7 / 33.6 / 233 MB; exp(x) at 10:
  140 MB, 0.35 s; 2*x+1 at 16: 77 MB; growth about 7x per order. Matches §1.1.
- Stack thresholds at HEAD (designer's `deep_head`, debug, 8 MB): drop ok 40000 / overflow 80000,
  simplify ok 10000 / overflow 20000, Display ok 5000 / overflow 10000. Matches §1.2.
- §1.4 garbage at HEAD reproduced: `(1)/sin(x)` gives `1/0` terms, `log(x)` gives `log(0)`,
  `abs(x)` gives `abs(0) + sign(0)*x`, `gamma(x)` at 1 gives `gamma(1)`, `zeta(x)` at 2 gives `zeta(2)`.
- S7: `taylor(x^2, x, 10^30, 2)` panics at `series.rs:34:70` (HEAD). Parser table §1.6 reproduced:
  `x^2/4 -> (x ^ 1/2)`, `2/3^2 -> 4/9`, `1/x`, `1/sin(x)`, `2^-x`, 2^63, 2^64 all parse errors;
  also `x/2/3 -> x/(2/3)` and `2^-1/3 -> 2^(-1/3)` (same cause, fixed by the same change).
- gp references of §4.1: I ran 34 of them (`review1/refs_check.gp`); all equal the design table
  (tan, tanh, sinh, cosh, csc, cot, Bernoulli gf, log cos, exp sin, removable, triple pole,
  binomials, atan/asin/asinh/atanh, acos, x^x at 1, generic y cases, exp(I x), 1/(x^2+1) at I,
  pi points, asymptotic, 1/(2+sin x) at 1, acosh at 2, erf, 10^30 point, limit values).
- Prototype vs gp: `gpcheck.py` reports 62/62, but one case is a false pass (minor m4). The other
  61 are genuine. Exact trig at 27 points (sin, cos, tan at 5pi/4, 7pi/6, -pi/3, 11pi/6, 3pi/4,
  -pi/6, 2pi/3, -5pi/2, 13pi/3) all correct, including the pole of tan at -5pi/2.
- Ball arithmetic (`proto/ball.rs`): Lipschitz bounds, rounding terms, domain checks (log/sqrt need
  lo > 0, asin needs |m|+r < 1, acosh needs lo > 1, atanh through a checked inverse, no enclosure
  for I, indeterminates, gamma, zeta, functions). A successful enclosure implies a real value. Sound.
- Prototype zero test (`proto/konst.rs:373-453`) is sound on my adversarial constants:
  `log(-1) - I*pi`, `sqrt(-1) - I`, `sqrt(-4) - 2*I`, `exp(I*pi/2) - I`, `e - exp(1)`,
  `pi - 4*atan(1)`, `arcsin(2) - pi/2 + I*log(2+sqrt(3))` all give `Undecided`.
- After-cost (prototype): debug x^2, 2*x+1, x at order 20000: 0.5 ms / 44 us / 20 us, 3.7 MB;
  exp(x) at 2000: engine 53 ms debug, 15.7 ms release, Display 0.9 s debug, 89 ms release, depth 15,
  5.35 MB output. Matches §5.1.
- Callers (my grep of the whole workspace, `rustmath-py` excluded from the workspace and not
  calling these): `repl.rs:1969`, `limits.rs:430`, `:554`, `lib.rs` tests; parser: `repl.rs:14`
  (`:1849`, `:4910`, `:4939`), `rustmath-py/src/symbolic.rs:6`. Complete. No string literal in any
  `*.rs` depends on the old Display forms `^ p/q`, `p/q ^`, `/ p/q`, `(-n ^`.
- `limit_via_series` new rule (§2.13) is correct: coefficients below the valuation are exact
  zeros, so valuation > 0 or an all-zero expansion with precision >= 1 means limit 0.
- MAGMA ch. 49 digest sections cited exist (§49.1.3, §49.1.4, §49.4.7-8, §49.5); H49E3 is the
  Bernoulli example.
- New dependency `rustmath-reals` has no cycle; `with_val_integer`, `with_val_rational`, `pi`, `e`,
  `as_float`, `from_float` exist (`rustmath-reals/src/mpfr.rs`).
- Coordination: I3 design adds the 10^30 test; J1b reads S1's `TAYLOR_ORDER`/`TAYLOR_WORK`.

## Blockers

### B1. The written zero-test rule 4 is unsound (§2.2, "Zero test", step 4)

Text: "write c = A + B*I ... Else the numeric test on A and on B: NonZero if either is proved
nonzero." A can contain non-real kernels that have no enclosure, while B is real and provably
nonzero. Counterexamples (value 0, rule says NonZero):

- `c = sqrt(-4) - 2*I`: A = `(-4)^(1/2)` (no enclosure, it is 2i), B = -2.
- `c = exp(I*pi/2) - I`: A = `exp(I*pi/2)` (no enclosure), B = -1.

An implementation of the text expands `1/(x + sqrt(-4) - 2*I)` as `1/c - x/c^2 + ...`: a division
by an exact zero presented as a result. The prototype is sound only because it does something else
(`const_status` runs the numeric test on the whole constant, and `proves_nonzero` refuses any
constant with an `I` kernel). The text also contradicts §2.3 ("only ... the constant is real").

Required: state the rule as implemented, or the sound strengthening: "NonZero if A and B both have
a real ball enclosure and one of them excludes 0; otherwise Unknown". Add tests
`1/(x + sqrt(-4) - 2*I)` and `1/(x + exp(I*pi/2) - I)` at 0: `UndecidedZero`.

### B2. Kernel exponents are i32 with unchecked arithmetic: silent wrong values (§2.2 `Monomial`)

`Monomial = Vec<(Kernel, i32)>`, and exponents add and multiply in `mul`, `pow_u` and the series
`pow_int` without overflow handling. Release prototype, within the REPL limits and the 1e8 budget:

```
(sin(1)*x)^4294967296 at 1, order 1   ->  1 + 4294967296*(x - 1)            (factor sin(1)^(2^32) lost)
(sin(1)*x)^3000000000 at 1, order 1   ->  1/sin(1)^1294967296 + ...          (exponent wrapped)
x + ((sin(1)^4096)^4096)^4096 at 0    ->  1 + x                              (constant became 1)
```

Debug: `panicked at proto/konst.rs:283:25: attempt to add with overflow`. HEAD returns these
inputs unevaluated (no wrong value). This is a fabricated value introduced by the new engine.

Required: checked exponent arithmetic (i64 with `checked_add`/`checked_mul`, or `Integer`
exponents); on overflow keep a kernel `c^n` (as already done for |n| > 4096) or return
`Err(Unsupported)`. Tests for the three inputs, run in debug and in `--release`.

### B3. The work budget does not bound constant arithmetic or dense allocation (§2.7)

The design charges only the engine helpers `cx.kadd/kmul/kscale`. Two paths escape:

(a) Constant construction (`SeriesConst::from_expr`, exact `pow_const` for |n| <= 4096,
`apply`, `inverse`) is not charged. A multi-term constant raised to an exact power expands
multinomially. Release prototype, `BUDGET=1000000`:

| input (point 0, order 1) | result | wall |
|---|---|---|
| `x+(sin(1)+cos(1)+exp(1))^50` | `Budget(574997889)` (575x over, checked only at the end) | 13.8 s |
| `x+(sin(1)+cos(1)+exp(1))^100` | not finished | > 120 s |
| `x+(sin(1)+cos(1))^4096` | not finished | > 30 s |

HEAD returns `x + (...)^4096` in 65 us.

(b) Dense vectors are allocated before any charge. `add` loops over the whole exponent span
(`proto/ser.rs` `add`, charge after the loop). `x^(-100000000)+1` at 0, order 1, budget 1e8:
2.35 GB peak, 1.2 s, **work charged 20**. With `x^(-1000000000)+1` the process needs about 24 GB and
aborts (OOM is not a catchable error, so the REPL process dies). HEAD: 49 us. The public
`SeriesExpansion.coefficients` vector has the same dense span.

So the F1 sub-item "no REPL bound prevents it" is not fixed: two one-line inputs hang or kill the
REPL under `TAYLOR_WORK`, and both regress HEAD.

Required: thread `&mut Work` through all `SeriesConst` arithmetic (including `from_expr`,
`pow_const`, `pow_u`, `apply`, `inverse`, `zero_status`'s numeric test) and pre-charge each
operation by an a-priori size (monomial pairs |A|*|B| for `mul`, the term count of a multinomial
power) before running it; cap exact constant powers by result size (fall back to a kernel `c^n`);
charge the exponent span before any dense allocation (or keep exact series sparse). Tests:
`taylor_with_budget` with budget 1e6 on the three (a) inputs and on `x^(-1000000000)+1` gives
`BudgetExceeded` in under 1 s in debug, without the large allocation.

### B4. Iterative Drop is feasible but declared out of scope (§7.1)

The item as assigned names it: "Fix ... the deep recursion (simplify and Drop of left-deep sums)".
The design's own analysis says it is feasible (`impl Drop for Expr` with `Arc::into_inner` onto a
heap stack; E0509 at 4 library sites and about 98 test sites, all fixed by matching on a
reference). The stated reasons ("not needed once taylor stops building deep trees", "Drop is not
the first recursion to fail") are not infeasibility. The design's own test
`simplify_long_left_deep_sum` has to `mem::forget` its input because dropping a 100000-term sum
overflows, so the deep-recursion fix is visibly half done.

Required: implement the iterative Drop in this unit (library sites `integrate_exact.rs:96`, `:148`,
`rustmath-geometry/src/plane_curve_ops.rs:489`, `rustmath-jupyter/src/repl.rs:1961`, plus the test
sites); remove `mem::forget` from the simplify test; add a test that builds and drops a 10^6-term
left-deep sum on a 1 MB thread in debug; add the touched crates (geometry, calculus, jupyter) to the
acceptance commands. If the designer keeps it out, it needs a genuine infeasibility argument,
which the design's own measurements contradict.

## Minor issues (fix in the revision)

- **m1. `NotLaurent` claims more than the engine knows.** §2.9 documents it as "No Laurent
  expansion exists at the point". The engine expands bottom-up, so it returns it for Laurent
  functions: `sqrt(x)^2`, `exp(log(x))`, `x^(x-x+2)` at 0 (prototype output). It also decides a
  "branch point" from an undecided exponent: `x^(sin(1)^2+cos(1)^2)` at 0 gives `NotLaurent`
  (the exponent is 1). `0^x` at 1 gives `DivisionByZero("log(0)")` in the prototype (the design text
  says `NotLaurent`), though 0^x = 0 near 1. No consumer acts on the variant, so no wrong value, but
  the message and the REPL "ValueError" mapping are wrong. Required: document `NotLaurent` as "a
  subexpression has no Laurent expansion at the point (the function itself may have one, e.g.
  sqrt(x)^2)"; in `pow_const` with vb != 0 require alpha decided non-integer (else `UndecidedZero`);
  route `Pow` whose exponent series is an exact constant to the constant-exponent path.
- **m2. Precision cap too tight.** `1/(sin(x)^50 - x^50)` at 0, order 3 gives
  `PrecisionExhausted(108)`; gp gives valuation -52. Honest, but avoidable: once an `inv` has a
  decided leading coefficient the extra precision is known exactly (T + 2v), so only the
  `NeedPrecision` growth needs a cap (or tie the cap to the budget). Add this input as a test.
- **m3. Display round trip of a negated base.** The engine emits kernels with a `Neg` base:
  `sqrt(x - y)` at 0 gives kernel `(-y)^(1/2)`, printed `(-y ^ 1/2)`; with §2.11 it prints
  `(-y ^ (1/2))`, which re-parses as `-(y^(1/2))`. Parenthesize any `Unary(Neg, _)` base of `Pow`,
  and add `sqrt(x - y)` to `test_display_parse_round_trip`.
- **m4. Harness false pass.** `gpcheck.py` case `erf(x)` at 1: gp has no `erf`, the `ours=[...]`
  line fails, the loop errors, and "OK" is printed (nterms=2 is gp's length of the undefined
  variable `ours`). The prototype's values are right: I checked them with
  `E(z) = (1-erfc(polcoef(z,0))) + 2/sqrt(Pi)*intformal(exp(-z^2)*deriv(z))`, giving
  [0.84270..., 0.41510..., -0.41510..., 0.13836..., 0.069184...]. Fix the harness (define erf through
  erfc, treat any gp error as FAIL) before using it as the porting oracle.
- **m5. Constant base with variable exponent.** §2.5 rewrites `l^r` as `exp(r * log(l))`. The
  `log(l)` is introduced by the engine, so an undecided `l` must use the strict test
  (`UndecidedZero`), not an `Opaque` `log(l)` kernel (`(sin(1)^2+cos(1)^2-1)^x` at 1 would expand in
  `log(0)`).
- **m6. Slow REPL test.** `taylor("(1+x)^10000", x, 0, 2000)` is refused after 7.6 s in debug
  (prototype, budget 1e8), not "fast". With B3's pre-charging pick an input refused at once.
- **m7. Plan order.** Step 2 (zero test with the numeric fallback) needs step 3 (`series_ball`).
  Step 5 changes `taylor` to `Result` while `limits.rs` (step 6) and `repl.rs` (step 11) are
  updated later, so "each step builds green" fails. Swap 2 and 3; merge 5 and 6 and the REPL arm.
- **m8.** "each under 10 ms" timing asserts in debug unit tests are flaky; drop them or use a loose
  bound.
- **m9.** `abs`/`sign` assume a real variable and a real point; say so in the doc comments. The
  message "essential singularity" is wrong for `atan(1/x)` (a jump, still not Laurent).
- **m10 (optional).** Radical kernels are not d-th-power free: `sqrt(8) - 2*sqrt(2)` and
  `sqrt(6) - sqrt(2)*sqrt(3)` stay `UndecidedZero` (sound, incomplete).
- **m11 (optional).** Port H49E3 more fully: `x/(exp(x)-1)` to order 50 against gp `bernfrac`.

## Summary for the orchestrator

Covered and sound in the prototype: the series engine replacing repeated differentiation (F1
memory/time), balanced output (F1 stack at order 20000), iterative simplify, the expansion point
(S7), the bignum literal parser and `a/b` precedence (S7), Display parenthesization, callers and
tests with gp values. Blockers: B1 unsound text rule for complex constants; B2 i32 exponent
overflow gives silent wrong values in release (`(sin(1)*x)^(2^32)` at 1 gives `1 + 2^32 (x-1)`);
B3 the work budget misses constant arithmetic and dense allocation (REPL hangs or OOMs on one-line
inputs where HEAD answers in microseconds); B4 iterative Drop is in the assigned item and feasible,
but declared out of scope.
