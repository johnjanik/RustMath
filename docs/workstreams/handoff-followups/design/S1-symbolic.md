# S1-symbolic: design (crate rustmath-symbolic)

Unit items (docs/HANDOFF.md "Open follow-ups", REPL split-outs of the X2 design):

- **F1.** `Expr::taylor` (`rustmath-symbolic/src/series.rs:24`) differentiates without simplifying, so memory
  grows exponentially with the order, and a debug build overflows the stack at order 20000
  (`TAYLOR_ORDER`, `rustmath-jupyter/src/repl.rs:10534`). No REPL bound prevents it. Fix the algorithm
  and the deep recursion (simplify and Drop of left-deep sums).
- **S7 (symbolic part).** The expansion point in `series.rs:34`, and the i64-only literal parser
  (`rustmath-symbolic/src/parser.rs:148-154`).
- Measure before/after for x^2, exp(x), sin(x), 2*x+1 at several orders; reference series from gp.

HEAD = c79a582 (same code as 9c24389). All line numbers below are at HEAD.

Experiments: `/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/S1-symbolic/`
(below: `EXP/`). The working prototype of the engine is `EXP/rev2/proto3/` (`konst.rs`, `ser.rs`,
`ball.rs`, `main.rs`, built with `EXP/rev2/build.sh debug|release`, rustc against the workspace rlibs
listed in `EXP/rev2/rlibs_dbg.txt` and `rlibs_rel.txt`; binaries `EXP/rev2/bin/proto3` (debug) and
`EXP/rev2/bin/proto3_rel`). `EXP/proto2/` is the revision 1 prototype and `EXP/proto/` the revision 0
one, kept for comparison (`EXP/rev2/bin/proto2_copy` and `proto2_rel_copy` are copies of the revision 1
binaries, for before/after runs). The gp
cross-check harness is `EXP/rev2/gpcheck3.py` (107 cases, 0 failures, output `EXP/rev2/gpcheck3.out`;
it contains the 82 cases of `EXP/gpcheck2.py`). Expected errors, exact structures and budget timings
are checked by `EXP/revchecks.py` (revision 1 checks, output on the revision 2 prototype
`EXP/rev2/revchecks_r2_base.out`, 0 bad) and `EXP/rev2/revchecks3.py` (revision 2 checks, output
`EXP/rev2/revchecks3.out`, 0 bad). The iterative Drop experiments are in `EXP/rev1/`. `EXP/rev2/run.py`
times one prototype call.

## Revision 2 (response to review 2)

Review: `fu2/design/S1-symbolic.review2.md`. Both blockers and all five minor issues are fixed in the
body below, and the prototype carries each fix (`EXP/rev2/proto3/`). Summary:

| issue | change | where | evidence |
|---|---|---|---|
| B1 MPFR underflow | Every MPFR value the ball code computes (midpoints, radii, Lipschitz bounds) must be 0 or have its exponent in a window [-W, W], W = 2^28; else there is no enclosure. MPFR's exponent range is [1 - 2^30, 2^30 - 1], so no product, sum or quotient of values in the window can underflow or overflow. A computed midpoint 0 is accepted only when the exact result is 0 (a factor is 0, the summands are exact negatives, the argument is a zero of f). exp, sinh and cosh require \|m\| + r <= 2^27. Every enclosure that is accepted is therefore free of underflow and overflow, and the radius bound of every operation holds. | §2.3 | `revchecks3`: `1/(x + (1/2)^1073741827*2^1073741822 - 1/32)` and `1/(x + exp(-744261120)*exp(744261117) - exp(-3))` at 0 give `UndecidedZero` (revision 1: a series of 1/0 terms); also the overflow twin `exp(744261120)*exp(-744261117) - exp(3)`, `(1/3)^536870912*3^536870912 - 1` and `2^(-300000000)*2^300000000 - 1`; each in 2 to 13 ms debug |
| B2 MPFR sin/cos/tan cost | MPFR sin and cos are called only when the ball has r < 1 and exponent(m) <= p + 64; otherwise sin and cos get the trivial enclosure [-1, 1] and tan none. The a-priori charge of the numeric test is `ball_cost`, a walk over the expression actually evaluated (every node, shared kernel subtrees once per occurrence), with sin, cos, tan charged at precision 2p + 64. The walk stops at the remaining budget. Measured first: only sin and cos (and tan through them) cost more with a larger exponent; exp, log, sqrt, atan, tanh, erf, sinh take microseconds at 2^(10^7) (`EXP/rev2/mpfr_cost/out.txt`). | §2.3, §2.7 | debug, budget 1e8: `1/(x + sin(2^100000000) + 2)` is `Ok` in 2 ms (revision 1: 111.7 s release); the `2^1000000000` variant is `UndecidedZero` in 2 ms (its kernel's ball leaves the window); `cos`, `tan` variants, the `pow_const` kernel fallback `x + (sin(2^100000000) + 3)^1000000000` and the non-integer test `x^(2^100000000*sin(1))` all return in 2 ms |
| m1 high-order poles | `pow_int` step 5 (n vb >= P) keeps the leading term `c0^n h^(n vb) + O(h^(n vb + 1))`; the working truncation `trunc(P)` keeps the first term when every term is at or above P; `mul` computes at least its first term (cutoff `min(prec, max(P, val + 1))`). The loss-driven retry then finds the precision. | §2.1, §2.4 | `gpcheck3` (release): `1/sin(x)^200`, `1/(x+x^2)^200`, `1/(exp(x)-1)^150`, `2/(3*sin(x)^200)`, `1/(sin(x)^200+x^300)` at 0, order 2, equal gp at every coefficient (revision 1: `PrecisionExhausted(131)`); the first three take 1.3 to 1.5 s, 15 to 17 ms and 2.3 to 3.0 s release (two runs) |
| m2 kernel creation | Kernel keys are a prefix code built in linear time with integers in hexadecimal (no decimal conversion); a kernel's creation is charged by its key length before the key is built; kernels are interned per expansion and compare by (class rank, id), so a comparison is O(1) whatever the key length. | §2.2, §2.7 | release: 40 kernels `sin(2^340000 + k)` take 15 ms and 1.28e6 units (8.6e7 units/s; revision 1: 504 ms for 8.5e5 units); with each kernel squared, 24 ms (revision 1: 786 ms) |
| m3 `log(e^k)` | Implemented and extended: `log(q e^k prod exp(r_i)^(j_i)) = log(q) + k + sum j_i r_i` for a rational q > 0, integers k, j_i and rationals r_i (all factors are positive reals). | §2.2 | `1/(x + log(e^3) - 3)` and `1/(x + log(exp(2)) - 2)` at 0 give exactly `1/x`; `1/(x + log(2*e) - 1)` gives the series of `1/(x + log(2))` (gp); `log(-e)` is not folded |
| m4 Drop wording | The table row and the doc comment of `Drop for Expr` state that the guarantee is relative to the shallowest drop seen on the thread, so a thread whose first `Expr` drop happens deep in its stack may use up to 16 KiB of glue frames below that point. | §2.14 | |
| m5 output size | `TAYLOR_ORDER` and `TAYLOR_WORK` doc comments say what each bounds. I agree that the order does not bound the output. I disagree in part with "the budget bounds the output": the units track the words of the coefficients in memory, but a kernel is stored once and printed once per coefficient that contains it. So the budget bounds the work and the memory of the result, not the printed text. The printed size is bounded by the REPL display budget (J1b, `OUTPUT_BYTES` = 16 MiB, estimated before formatting). | §3.1 | release: `(1+x)^(10^100)` at order 1400 uses 7.4e7 units and prints 95.8 MB (review 2); `(x + sin(2^100000))^1000` at order 1000 uses 1.4e5 units, 4 ms and 5.7 MB peak memory, and prints 30.4 MB in 1.7 s |

Also found and fixed while porting the review's fixes:

- `non_integer_status` converted floor(mid(A)) through decimal strings (rug to num-bigint), which is
  quadratic in num-bigint. It now converts through binary digits, and only when |mid(A)| < 2^p (a
  midpoint of magnitude at least 2^p has p significant bits, so it is itself an integer and the ball
  cannot lie strictly between two integers at that precision). When no precision gives a nearest
  integer, the error is `UndecidedZero(alpha)`.
- The `NotLaurent` message of a non-integer power formatted the exponent eagerly. No engine message
  formats a constant now ("a non-integer power of an argument of valuation v has a branch point
  there").
- The prototype built 2^-p by p multiplications; the ball code builds powers of two by shifts.
- The m1 change was rerun on review 2's cases of negative intermediate precision (`exp`, `log`, `sqrt`,
  `^100` of `1/sin(x) - 1/x (+1)` at orders 0, 1, 3) and branch cuts (`acos`, `asin` at 2, `sqrt`, `log`
  at -2): all 16 equal gp (`gpcheck3`). The harness now parenthesizes a negative base, which HEAD's
  Display prints without parentheses (the Display fix of §2.11 removes that).

Where I did not take the reviewer's literal wording:

- B1: the review offered (a) an absolute underflow term on every radius or (b) a refusal outside an
  exponent window. I took (b) and added the exact-zero rule. With (b) every operation sees operands
  of bounded exponent, which keeps the no-underflow argument of §2.3 to a few lines, and it refuses
  overflow by the same check.
- B2: the trivial enclosure is used from r >= 1, not from r >= pi. For r >= 1 the Lipschitz enclosure
  [sin m - r, sin m + r] has width at least 2 and always contains 0, so [-1, 1] is never wider.
  The `2^1000000000` variant ends in `UndecidedZero`, not `Ok`, as the review allowed.
- m1: the leading-term rule does not require the lead to be decided. The term `c0^n h^(n vb)` is exact
  whatever c0 is; an undecided c0 is reported by the strict inverse that needs it.
- m5: see the table (the budget bounds memory, not text).

## Revision 1 (response to review 1)

Review: `fu2/design/S1-symbolic.review1.md`. Every blocker and every minor issue is fixed in the body
below. Summary of the changes, with the evidence:

| issue | change | where | evidence |
|---|---|---|---|
| B1 zero test rule 4 unsound | Rule 4 now splits c = A + B*I and answers `NonZero` only when A and B **both** have real ball enclosures and one of them excludes 0 (a successful enclosure proves the value real). A non-real kernel such as `(-4)^(1/2)` or `exp(I*pi/2)` has no enclosure, so the answer is `Unknown`. This is sound and slightly stronger than the revision 0 prototype, which refused every constant with `I` (it now decides `1/(x + sin(1) + I)`). | §2.2 zero test, §2.3 | `revchecks`: `1/(x+sqrt(-4)-2*I)`, `1/(x+exp(I*pi/2)-I)`, `1/(x+sqrt(-1)-I)`, `1/(x+log(-1)-I*pi)` at 0 give `UndecidedZero`; `gpcheck2`: `1/(x+sin(1)+I)`, `1/(x+pi+I)` match gp |
| B2 i32 exponents overflow | Kernel exponents are `i64` with checked arithmetic everywhere (monomial products, powers, inverses). Series exponents (valuation, precision, shifts) are checked too. A constant power whose exponent product overflows stays a kernel `c^n`; an overflow inside a monomial product or a series exponent is `Err(Unsupported)`. | §2.1, §2.2 | debug `proto2`, no panic: `(sin(1)*x)^4294967296` at 1 gives `sin(1)^4294967296 + 4294967296*sin(1)^4294967296*(x - 1)`, `(sin(1)*x)^3000000000` likewise, `x + ((sin(1)^4096)^4096)^4096` gives `sin(1)^68719476736 + x`; all three match gp (`gpcheck2`); `(sin(1)^4*x)^(2^62)` at 1 keeps the kernel `(sin(1)^4)^(2^62)`; `x + sin(1)^(2^62)*sin(1)^(2^62)` and `(x^(-2^62))^4` are `Unsupported` |
| B3a constant arithmetic not charged | `&mut Work` is threaded through **every** `SeriesConst` operation (`from_expr`, `pow_const`, `pow_u`, `apply`, `inverse`, `mul`, `add`, `scale`, the numeric zero test, radical extraction), each returns `Result`, and each charges an a-priori size **before** it runs. Exact constant powers are capped by result size (terms and bits) and fall back to a kernel `c^n`. Large series powers use Miller's recurrence (cost independent of n) when the leading coefficient is decided, else the binomial theorem with capped constant powers. | §2.2, §2.4, §2.7 | `revchecks` (debug, budget 1e6): `x+(sin(1)+cos(1)+exp(1))^50`, `^100`, `x+(sin(1)+cos(1))^4096` return `Ok` with a kernel in 2 ms (revision 0: 13.8 s, over 120 s, over 30 s) |
| B3b dense allocation before any charge | Series are **sparse**: `Vec<(i64, SeriesConst)>` of structurally nonzero terms. The only dense buffers are the outputs of the recurrences, whose length is charged (16 units per slot) before allocation. `SeriesExpansion` is sparse too. | §2.1, §2.9 | `x^(-1000000000)+1` at 0 gives exactly `1 + 1/x^1000000000` in 2 ms, 4 MB (revision 0: 2.35 GB at 10^8, about 24 GB at 10^9) |
| B4 iterative Drop out of scope | Implemented in this unit: `impl Drop for Expr` with bounded stack use (§2.14). It costs nothing measurable on ordinary workloads: the drop glue recurses as before while the drop runs less than 16 KiB below the shallowest drop seen on the thread, and only deeper subtrees are torn down iteratively. E0509 sites fixed by matching on references: 4 library sites and 98 test sites (listed in §3.3). The `mem::forget` is gone from the simplify test. | §2.14, §3.3, §4.1, §6 | `EXP/rev1/drop_measure.txt`: a 10^6-term left-deep sum is built and dropped on a 1 MiB thread in debug (drop 0.38 s); right-deep, unary, function and zig-zag chains of 10^6 too; a drop that starts with less than 16 KiB of stack left; release cost equal to HEAD (parse+differentiate+simplify loop 106.0 ms vs 107.5 ms, min of 8) |
| m1 NotLaurent semantics | `NotLaurent` is documented as "a subexpression has no Laurent expansion at the point"; the REPL maps it to `ComputationError`. A non-integer power of a series of nonzero valuation needs the exponent **decided** non-integer (rational, a nonzero indeterminate part, or a ball strictly between two integers or off the real axis), else `UndecidedZero`. A `Pow` whose exponent expands to an exact constant uses the constant-exponent path. `0^r` is exact 0 when r(a) is a positive rational. | §2.4, §2.5, §2.9, §3 | `x^(x-x+2)` at 0 is `x^2`; `x^(sin(1)^2+cos(1)^2)` is `UndecidedZero`; `x^pi`, `x^y`, `x^(1/2+I)` are `NotLaurent`; `0^x` at 1 is `0` |
| m2 precision cap | The driver separates the two kinds of retry: a decided precision loss raises P by the exact missing amount, uncapped (bounded by 24 attempts and the budget); an undecided valuation (`NeedPrecision`) doubles the extra precision up to `max(4|T|, 128)`. Exact series with at most 64 terms are no longer truncated at P, so exact denominators of high valuation need no search. | §2.6 | `1/(sin(x)^50 - x^50)` at 0, order 3, gives valuation -52 equal to gp (0.12 s release); `1/(x^300*(1+x))` matches gp; `1/(sin(x)^2+cos(x)^2-1)` is `PrecisionExhausted` after 0.5 s debug |
| m3 Display of a negated base | A `Pow` base that is `Unary(Neg, _)` or a negative number is printed in parentheses. | §2.11, §4.3 | `sqrt(x - y)` added to the round-trip test |
| m4 harness false pass | `gpcheck2.py` maps `erf` to `ERF0(z) = 1 - erfc(z)` and treats any gp error as a failure (gp's own stack-growth notices excepted). Two float artifacts of gp (`sqrt(2)*sqrt(2)-2`, `sqrt(8)-2*sqrt(2)`) use exact aliases, as the pi points already did. | §4 | `erf(x)` at 0 and at 1 now genuinely pass |
| m5 constant base | `l^r` with l free of the variable: l decided zero gives the `0^r` rule, l undecided is `UndecidedZero(l)`, only a nonzero l gives `exp(r log l)`. | §2.5 | `(sin(1)^2+cos(1)^2-1)^x` at 1 is `UndecidedZero` |
| m6 slow REPL test | Replaced by `taylor("exp(x)*sin(x)", x, 0, 2000)`: the product is pre-charged at 5.2e9 units and refused at once (0.18 s debug with budget 1e8). `(1+x)^10000` at 2000 is now admitted (Miller: 11.5 ms release, 45 ms debug). | §4.5 | `revchecks`, `EXP/calib_rev1.txt` |
| m7 plan order | Ball arithmetic first (it works on `Expr`, no dependency on the constant field), then the constant field; the `Result` API, its callers in `limits.rs` and the REPL arm in one step; Display, Drop and the E0509 fixes in one step. | §6 | |
| m8 timing asserts | No timing assertion in unit tests. Only the budget tests carry a loose 20 s bound, which guards against a hang, not a speed. | §4 | |
| m9 abs/sign, messages | abs and sign documented as assuming a real variable and a real point (non-rational coefficients give `Unsupported`). Messages: "exp of an argument with a pole at the point has no Laurent expansion there" (also for arctan(1/x), sin(1/x) ...), "abs of an argument with a zero of odd order is not analytic there". | §2.4 | `revchecks` |
| m10 radicals | Radical kernels n^(1/d) are made d-th-power free by trial division with the primes below 2^16 (when n has at most 4096 bits) plus a perfect-power test of the cofactor. `sqrt(8) - 2*sqrt(2)` is now `Zero`. Products of radicals are not split (`sqrt(6) - sqrt(2)*sqrt(3)` stays `UndecidedZero`: sound, incomplete, documented). | §2.2 | `gpcheck2`: `1/(x+sqrt(8)-2*sqrt(2))` is `1/x` |
| m11 H49E3 | `x/(exp(x)-1)` to order 50 against gp (`vector(51,k,polcoef(s,k-1)) == vector(51,k,bernfrac(k-1)/(k-1)!)` is 1). | §4.1 | `gpcheck2` |

Where I did not take the reviewer's literal wording:

- B3 asked that the three constant-power inputs return `BudgetExceeded` with budget 1e6. With the
  size cap they now return the correct value at once, with the large power kept as a kernel
  (2 ms debug), which is better than a refusal and is what "fall back to a kernel" gives. The
  budget path is still tested, on inputs below the caps whose work exceeds the budget:
  `(x+sin(1)+cos(1)+exp(1)+pi)^64` (`BudgetExceeded` after 19 ms debug, budget 1e6) and
  `exp(x)*sin(x)` at order 2000.
- B4 asked for the classic always-iterative Drop. I measured three such variants first
  (`EXP/rev1/dropbench2.rs`, `EXP/rev1/workload.rs`): they cost 18 to 28 % on a
  parse+differentiate+simplify loop (+46 % instructions), because every internal node pays an
  `Arc::get_mut` and two moves. The design uses the stack-gated variant instead (§2.14): same
  guarantee, no measurable cost.

## 0. Summary of decisions

1. Replace repeated differentiation by **truncated Laurent series arithmetic over a field of
   symbolic constants** (MAGMA Handbook ch. 49, "Power, Laurent and Puiseux Series", §49.1.3-49.1.4
   precision model, §49.4.7-49.4.8, §49.5 transcendental functions; digest
   `rustmath-lava/docs/Chapter_49_Power_Laurent_and_Puiseux_Series.md`). Each node of the expression
   becomes a sparse series in h = x - a with exact coefficients and an absolute precision. Products
   are truncated Cauchy products; exp, log, powers, sin/cos, sinh/cosh, inverse trig/hyperbolic
   functions and erf use the classical first-order recurrences derived from their differential
   equations (Knuth, TAOCP vol. 2, 3rd ed., §4.7; Brent and Kung 1978). Cost is O(n^2) coefficient
   operations per node instead of exponential.
2. Coefficients live in a canonical normal form: Q-linear combinations of monomials in **kernels**
   (constants such as `exp(1)`, `sin(1)`, `2^(1/2)`, `y`, `pi`) with `i64` exponents and checked
   exponent arithmetic. Every division by a derived leading coefficient goes through a **sound
   three-valued zero test** (`Zero` / `NonZero` / `Unknown`): symbolic rules first, then a rigorous
   MPFR ball enclosure of the real and imaginary parts (via `rustmath-reals::RealMPFR`), which
   refuses every value outside a safe exponent window, so that no accepted enclosure has underflowed
   or overflowed, and which is charged by the expression it evaluates. `Unknown` gives
   `Err(SeriesError::UndecidedZero)`. Nothing is guessed. Kernels are interned per expansion and
   compare in O(1).
3. `taylor`, `maclaurin`, `series`, `laurent`, `asymptotic`, `series_coefficients` and
   `series_with_big_o` return `Result<_, SeriesError>`. Poles give Laurent terms (as Sage/Maxima
   `taylor` and PARI do). Branch points, log and essential singularities, unsupported functions
   (gamma, zeta, factorial, named functions, mod of the variable) are `Err`, not fabricated values.
   This also fixes the `laurent` facade (PROJECT_TRACKER.md:2109) and the "inherits the
   differentiate facade" note of :2163.
4. The output is a **balanced** sum (depth O(log n)), so no consumer (simplify, Display, Drop,
   PartialEq, substitute) recurses once per term. `simplify` becomes iterative along the left spine
   of binary chains, and `Drop for Expr` has bounded stack use for every tree shape (§2.14).
5. A deterministic **work budget** (`taylor_with_budget`) lets the REPL refuse expansions that would
   run too long. Every constant and series operation charges an a-priori cost before it runs, and
   every dense buffer is charged before it is allocated. Exact constant powers are capped by result
   size. The REPL taylor arm uses the budget, and `TAYLOR_ORDER` drops from 20000 to 2000 (a plain
   argument check). The budget bounds the work and the memory of the result; neither bounds the
   printed text, which the REPL display budget of J1b bounds (§3.1).
6. S7: the point test uses the engine (any size of integer). The literal parser reads bignum integers;
   `a/b` stops being a literal token (it was the cause of `x^2/4` parsing as `x^(1/2)` and `1/x`
   failing to parse); a value-preserving fold keeps `parse("1/3") == Rational(1/3)`.
7. Declared out of scope (section 7): the duplicate `rustmath-calculus::taylor` (other crate, same
   defect plus an i64 factorial overflow); Puiseux and logarithmic expansions; series of gamma and
   zeta; iterative Display, PartialEq and substitute (not in the item, which names simplify and
   Drop).

## 1. Defects confirmed at HEAD

All runs: standalone programs in `EXP/` linked against the workspace rlibs. Debug = the workspace
debug profile, main thread (8 MB stack). Release = `-O` against `target/release` rlibs. Memory is
VmHWM from `/proc/self/status`.

### 1.1 F1: exponential memory and time (`series.rs:24-58`)

`taylor` keeps `derivative = derivative.differentiate(var)` without simplification, substitutes the
point into the whole tree, and accumulates `result = result + term`. `differentiate` builds
`0*x + 2*1`-style trees whose size doubles at each product rule.

Measured with `EXP/measure_head.rs` (`bin/measure_head f 0 n`), point 0:

| f | n | debug time | debug peak | release time | release peak |
|---|---|---|---|---|---|
| x^2 | 7 | 0.15 s | 33 MB | | |
| x^2 | 8 | 1.46 s | 233 MB | 0.46 s | 232 MB |
| x^2 | 9 | 11.2 s | 1.92 GB | 3.92 s | 1.92 GB |
| exp(x) | 10 | 0.35 s | 140 MB | 0.22 s | 139 MB |
| exp(x) | 12 | 12.6 s | 4.64 GB | 7.86 s | 4.64 GB |
| exp(x) | 14 | out of memory at 6 GB | | | |
| sin(x) | 12 | 14.1 s | 5.23 GB | 9.08 s | 5.23 GB |
| 2*x+1 | 20 | 3.95 s | 1.18 GB | 2.41 s | 1.18 GB |
| 2*x+1 | 40 (debug), 24 (release) | out of memory at 6 GB | | out of memory at 6 GB | |

The growth factor is about 7 to 8 per order for x^2. The REPL accepts orders up to 20000.
`series_coefficients` (`series.rs:148-169`) has the same loop and the same growth.

### 1.2 F1: stack overflow at TAYLOR_ORDER, and the recursive consumers

`bin/measure_head x 0 20000` (debug): `thread 'main' has overflowed its stack` (order 10000: 0.5 s,
85 MB). The result is a left-deep sum of 20001 terms; `simplify` (`simplify.rs:376-400`) recurses
once per term.

Depth at which each recursive consumer overflows on a left-deep sum, debug, 8 MB stack
(`EXP/deep_head.rs`, sum of `x^k`):

| consumer | ok | overflows |
|---|---|---|
| Display (`expression.rs:437`) | 5000 | 10000 |
| simplify (`simplify.rs:376`) | 10000 | 20000 |
| substitute (`substitute.rs:26`) | 10000 | 20000 |
| PartialEq (derived) | 20000 | 40000 |
| Drop (drop glue through `Arc`) | 40000 | 80000 |

The REPL runs on tokio worker threads and `cargo test` on 2 MB threads, so the limits there are about
four times lower. The REPL also prints the result, so Display at about 2500 terms is the real limit.
Release builds are not safe either: the HEAD build of `EXP/rev1/workload.rs` aborts with a stack
overflow when it drops a 2*10^5-term sum on the 8 MB main thread.

### 1.3 F1: no REPL bound

`repl.rs:1943-1971` checks `0..=TAYLOR_ORDER` (20000) only. The doc comment at `repl.rs:10524-10533`
says that the order cannot bound the cost. That is true for the old algorithm and stays partly true
for the new one (the cost of dense rational series grows like n^2 times the coefficient size, see
§5), so a work budget is needed, and it must cover every operation (review 1, B3).

### 1.4 F1 (found): wrong output at singular points and for named functions

Same program, order 3 (the numerators are written `(1)/...` because HEAD cannot parse `1/sin(x)`, §1.6):

```
taylor(1/sin(x), x, 0, 3) = ((((1 / 0) + ((-1 / 0) * x)) + (((0 / 0) * (x ^ 2)) / 2)) + ...
taylor(sin(x)/x, x, 0, 3) = ((((0 / 0) + ((0 / 0) * x)) + ...
taylor(log(x), x, 0, 3)   = (((log(0) + ((1 / 0) * x)) + ...
taylor(sqrt(x), x, 0, 3)  = ((((1 / 0) * x) + ...          (the constant sqrt(0) = 0 folded away)
taylor(gamma(x), x, 1, 3) = gamma(1)                       (all higher terms silently 0)
taylor(zeta(x), x, 2, 3)  = zeta(2)
taylor(abs(x), x, 0, 3)   = (abs(0) + (sign(0) * x))       (|x| is not analytic at 0)
```

The gamma/zeta rows come from the `differentiate` facades (PROJECT_TRACKER.md:2106: d/dx of
`Function`, `Gamma`, `Zeta`, `Factorial`, `Mod` is 0; d/dx erf uses a rational approximation of
2/sqrt(pi), `differentiate.rs:197`). `laurent` (`series.rs:89-119`) returns
`taylor(num)/taylor(den)` or a plain Taylor series (tracker 🛑 :2109). `asymptotic` (`:129-142`)
calls `taylor` at t = 0 of `f(1/t)`, which has a pole for every non-constant polynomial.
`limit_via_series` (`limits.rs:428-443`) only ever sees these `1/0` forms.

### 1.5 S7: the expansion point (`series.rs:34`, also `:206`, `:233`)

`matches!(point, Expr::Integer(i) if i.to_i64() == 0)` uses the panicking inherent `Integer::to_i64`
(`rustmath-integers/src/integer.rs:88`):

```
taylor(x^2, x, 10^30, 2)  ->  thread 'main' panicked at rustmath-integers/src/integer.rs:90:29:
                              Integer value too large for i64
```

Unit I3 (implemented first) changes `Integer::to_i64` to return `Option<i64>` and edits these three
sites to `i.is_zero()` (I3 design §9.3, Appendix A rows `series.rs:34/206/233`), and adds a symbolic
test "x^2 taylor at 10^30 does not panic". S1 rewrites these functions anyway; the new code never
calls `to_i64` on an Integer to decide anything about the point, and must keep (and adapt) I3's test
to the `Result` API.

### 1.6 S7: the literal parser (`parser.rs:148-171`, `:286-298`)

`integer` parses `opt('-') digit1` with `str::parse::<i64>`; `rational` parses `integer '/'
integer` as one token and is tried by `number` before any operator. Probe `EXP/parse_head.rs`:

| input | HEAD | correct |
|---|---|---|
| `x + 18446744073709551616` | parse error | `x + 2^64` |
| `9223372036854775808` | parse error | 2^63 |
| `123456789012345678901234567890` | parse error | the integer |
| `1/x`, `1/sin(x)`, `1/(y+x)`, `exp(1/x)` | **parse error** (`rational` consumes `1/` then fails with `?`, no backtrack) | `1/x` ... |
| `x^2/4` | `(x ^ 1/2)`, i.e. x^(1/2) | (x^2)/4 |
| `2^3/4` | `2^(3/4)` | 2 |
| `2/3^2` | `(2/3)^2 = 4/9` | 2/9 |
| `3/4^2`, `x*2/3^2` | 9/16, x*4/9 | 3/16, 2x/9 |
| `x/2/3`, `2^-1/3` | x/(2/3), 2^(-1/3) | x/6, 1/6 |
| `2^-x` | parse error | 2^(-x) |
| `1/0` | parse error | parse error (kept) |

The first three rows are the HANDOFF item. The others have the same cause (the `a/b` literal token)
and the fix of the literal parser removes them; they are in scope as part of S7.
The REPL parses every symbolic argument with this parser (`repl.rs:14`, `parse_expr`), so
`taylor("x^2/4", x, 0, 3)` expands x^(1/2) today.

## 2. Algorithm

### 2.1 Model (MAGMA ch. 49 §49.1.3, §49.1.4)

A series is stored as a sparse list of terms `c_e h^e` plus `O(h^p)`, with coefficients in the
constant field K (§2.2) and absolute precision `p`, or `p = EXACT` for an exact finite expansion
(constants, the variable, sums and products of them that were not truncated). Since series
arithmetic has no carries, every stored coefficient is exact (ch. 49 overview). h is `x - a` for a
finite point a, and `1/x` for the point at infinity (`asymptotic`).

```rust
// rustmath-symbolic/src/series_engine.rs (pub(crate))
pub(crate) const EXACT: i64 = i64::MAX;
#[derive(Clone)]
pub(crate) struct Laurent {
    /// (exponent, coefficient): increasing exponents, every coefficient structurally nonzero,
    /// every exponent below `prec`
    pub terms: Vec<(i64, SeriesConst)>,
    /// absolute precision, or EXACT
    pub prec: i64,
}
impl Laurent {
    /// first exponent (prec when empty and inexact, 0 for the exact zero)
    pub fn val(&self) -> i64;
    pub fn coef(&self, e: i64) -> SeriesConst;          // binary search; zero when absent
    pub fn trunc(self, p: i64) -> Laurent;              // working truncation (below)
    pub fn trunc_hard(self, p: i64) -> Laurent;         // result truncation
}
```

Why sparse (review 1, B3b): a dense vector from the valuation to the precision made
`x^(-10^9) + 1` allocate about 24 GB for a two-term answer. With sparse terms, memory is proportional
to the number of nonzero terms, except in the outputs of the recurrences (§2.4), whose length is the
working span and is charged before allocation.

`val` is a lower bound of the true valuation, and equal to it when the first coefficient is decided
nonzero. All exponent arithmetic (sums of valuations, precision `p - 2v`, shifts `n * v`) uses
`checked_add` / `checked_mul` on `i64`; an overflow is `Err(Unsupported("a series exponent
overflows i64"))`. `EXACT` absorbs in precision sums.

Working precision P and truncation. Every operation truncates its result at P:

- `trunc(P)`: an inexact series drops its terms of exponent >= P and takes `prec = min(prec, P)`,
  **except that it keeps its first term** (review 2, m1): when every term has exponent >= P, the
  first term e0 stays and `prec = min(prec, e0 + 1)` (no overflow: e0 < prec <= EXACT). This is
  correct because the series' own precision exceeds e0, so its coefficients below e0 + 1 are known.
  It keeps a decided valuation above P instead of turning it into `O(h^P)`, which an inverse could
  only meet with `NeedPrecision`. An exact series is left exact when all its terms are below P **or
  when it has at most `EXACT_KEEP_TERMS = 64` terms**; a larger exact series is truncated like an
  inexact one. Keeping small exact series exact means that exact denominators of high valuation
  (`1/(x^300*(1+x))`, `1/(x^1000 + x^1001)`) need no precision search. The bound keeps the cost of
  exact arithmetic small: a product of two kept series has at most 64 * 64 pairs.
- `mul` of two exact series with at most `EXACT_KEEP_PAIRS = 4096` pairs is computed in full, then
  `trunc(P)`. Other products are computed only up to `min(prec, max(P, va + vb + 1))`: at least the
  first term, for the same reason.
- `trunc_hard(T)` (the final result) drops every term of exponent >= T.

Every operation computes at least the first term of its result when the result's precision allows
it; this costs one coefficient per operation.

The variable's own series is exact: `[(0, a), (1, 1)]` at a nonzero point, `[(1, 1)]` at 0,
`[(-1, 1)]` at infinity.

Why not `rustmath-powerseries`: its `PowerSeries<R: Ring>` and `LaurentSeries<R>` need a
context-free `R::zero()`, a two-valued `Ring::is_zero` and `Field::inverse` on R. The symbolic
constant field has only a three-valued zero test. `LaurentSeries::try_inverse`
(`rustmath-powerseries/src/laurent.rs:287`) finds the valuation with
`position(|c| !c.is_zero())`, so an undecided leading coefficient would be silently taken as zero or
nonzero. Implementing the core trait tower for symbolic constants would force that unsound
`is_zero`, and it would have no place for the work budget. The recurrences are the same textbook
ones; the engine below is specific to symbolic constants (absolute precision with a retry driver,
strict and generic inverses, a work budget).

### 2.2 The constant field K (`series_const.rs`)

```rust
pub(crate) struct SeriesConst { terms: BTreeMap<Monomial, Rational> }   // no zero coefficient stored
pub(crate) type Monomial = Vec<(Kernel, i64)>;                           // sorted, exponents != 0
#[derive(Clone)] pub(crate) struct Kernel(Arc<KernelData>);              // Eq by id, Ord by (class rank, id)
pub(crate) struct KernelData {
    id: u64,              // interning id, unique per key within one expansion (below)
    key: String,          // prefix code of (kind, class, expr), built in linear time (below)
    expr: Expr,           // canonical expression (argument canonicalized through SeriesConst::to_expr)
    class: KernelClass,
    has_indet: bool,      // expr contains an indeterminate
    inv_of: Option<SeriesConst>,      // Inverse kernels: kernel = 1 / P0
    root: Option<(Integer, u32)>,     // radical kernels: kernel = n^(1/d), n > 1 integer, d-th-power free (below)
}
pub(crate) enum KernelClass { Indeterminate, ImaginaryUnit, NamedConstant, NonZero, Opaque, Inverse }
pub(crate) enum ZeroStatus { Zero, NonZero, Unknown }
```

Every operation below takes `w: &mut Work` and returns `Result<_, SeriesError>`; the first thing it
does is charge its a-priori cost (§2.7). The signatures are written without `w` for brevity.

Kernel identity (review 2, m2; `EXP/rev2/proto3/konst.rs`, `make_key`, `intern`):

- The key is a prefix code of the kernel's canonical expression, written in one linear pass:
  `#` and a signed hexadecimal integer; `#` hex `/` hex for a rational; `r` and the 16 hex digits of
  `f64::to_bits` for a float; `$len:name` for a symbol; the operator character, `(`, left, `,`,
  right, `)` for a binary node; the `UnaryOp` name, `(`, argument, `)` for a unary node;
  `@len:name(` arguments `)` for a function. Each token is self-delimiting and each node has a fixed
  arity, so the code is injective by construction (it does not rely on Display's parenthesization).
  Integers are written in base 16 (`{:x}` of num-bigint, linear time), never in decimal, whose
  num-bigint conversion is quadratic. The key starts with the kernel kind (plain, `inv`, `root`) and
  the class, so `c^n` as a `NonZero` and as an `Opaque` kernel are two kernels (two names for one
  number are sound, only less canonical).
- Creation is charged before the key is built: `key_size(expr)` (a walk that adds the bytes each node
  writes, stopping at the remaining budget), then `key_size / 8 + 1` units.
- Kernels are **interned** in a table owned by `Work` (one table per expansion, shared by all
  attempts of the driver): `HashMap<String, Kernel>`; a new key gets the next id. Two kernels are
  equal iff their ids are equal, and they are ordered by (class rank, id), with the indeterminates
  first. Comparisons are O(1) whatever the key length, so a monomial product or a map insertion never
  scans a long key. The normal form is canonical within one expansion; the order of terms in the
  output follows the creation order, which is deterministic.

Classes:

- `Indeterminate`: any `Symbol` other than the variable, `pi`, `e`, `I`, `oo`. Free symbols are
  **generic parameters** (independent indeterminates), as PARI treats `y` in `1/(y+x)`, and as the
  crate's Risch normalizer already does (`risch.rs:73-81`, `:319-326`, generic-identity convention).
  Documented on every public function.
- `ImaginaryUnit`: `Symbol("I")` (the parser maps `i` and `I` to it, `parser.rs:227-232`). Monomials
  reduce `I^e` to `I^(e mod 4)` with sign: `I^2 = -1`.
- `NamedConstant`: `Symbol("pi")`, `Symbol("e")` (the parser's constants, `parser.rs:216-226`).
- `NonZero`: a constant proved nonzero (list below). `Opaque`: anything else.
- `Inverse`: `1/P0` for a constant P that the input divides by or that a decided division needs.
- `Symbol("oo")` anywhere gives `Err(SeriesError::InvalidPoint("infinity ..."))`.

Arithmetic: `add`, `sub`, `neg`, `scale(&Rational)`, `mul` (sparse product of maps, accumulated in a
`BTreeMap`, not by repeated `add`), `pow_u(u64)`, `inverse(strict: bool)`, plus `pow_const` and
`apply(UnaryOp)` below. Exponents are `i64`; every exponent sum or product is checked. Canonical
reductions inside `mul` (sound identities only):

- `I^e -> (+-1) I^(e mod 4)`.
- Radical kernels `n^(1/d)`: exponent e reduced to `e mod d` and the factor `n^(e div d)` moved
  into the rational coefficient (so `sqrt(2) * sqrt(2) = 2` and `1/sqrt(2) = sqrt(2)/2`), but only
  when `|e div d| * bits(n) <= MAX_FOLD_BITS = 2^20`; a larger fold keeps the unreduced exponent
  (still a true identity, so the normal form stays sound; it is only less canonical).
- `Inverse` kernels only ever carry positive exponents; inverting a monomial that contains
  `(1/P0)^e` multiplies by `P0^e` (exact).
- An exponent overflow in a monomial product is `Err(Unsupported("a kernel exponent overflows
  i64"))`.

Construction of constants from expressions free of the variable (`SeriesConst::from_expr`):
`Integer`/`Rational` exact; `Real(f)`: 0.0 is zero, a finite f is a `NonZero` kernel (no float
folding; documented), non-finite is `Err(Unsupported)`; `Symbol` by class; `+ - *` exact; `/` is
`mul(inverse(false))`; `^` is `pow_const`; `Unary` is `apply`; `Function(name, args)` and `Mod` are
`Opaque` kernels with canonical arguments. Each step charges before it runs, so a large constant in
the input is refused by the budget before any large computation (review 1, B3a).

`pow_const(c, alpha)` (c^alpha for constants):

- alpha an integer n, c = 0: `0^n` is 0 for n > 0, 1 for n = 0, `Err(DivisionByZero)` for n < 0.
- alpha an integer n, c != 0. Let b = c for n > 0 and b = `inverse(false)` of c for n < 0 (a single
  monomial in all cases except a multi-term c with n > 0). The power is **exact** when its a-priori
  size is small, and a kernel `c^n` otherwise (`NonZero` when c is decided nonzero, else `Opaque`):
  - b a single monomial `q * m`: exact when `|n| * B <= MAX_EXACT_POWER_BITS = 2^20`, where
    `B = bits(q) + sum over radical kernels of bits(n_rad) * e / d` (B = 0 for q = +-1 and no
    radical); the exponents of m are multiplied by n with `checked_mul` (an overflow gives the kernel
    `c^n`). So `2^100000` is an exact integer, `sin(1)^(2^36)` is the monomial `sin(1)^68719476736`,
    and `(sqrt(2))^(2^40)` stays a kernel (its fold would need 2^39 bits).
  - b with t >= 2 terms: exact when `|n| <= 4096`, the multinomial term count
    `C(|n| + t - 1, t - 1) <= MAX_EXACT_POWER_TERMS = 512`, and `|n| * (B + log2 t + 1) <= 2^16`
    bits; computed by binary powering with `mul` (each product pre-charged). So
    `(sin(1)+cos(1))^4096` and `(sin(1)+cos(1)+exp(1))^50` are kernels (review 1 inputs), and
    `(1+sqrt(2))^20` is expanded.
- alpha = p/d (d > 1), c rational > 0: `c^(1/d) = (num * den^(d-1))^(1/d) / den`; write
  `num * den^(d-1) = m^d * r` with r d-th-power free (below); then `c^(p/d) = (m * r^(1/d) / den)^p`
  with the radical kernel `r^(1/d)` (none when r = 1). Negative rational c, or d > 64: a `NonZero`
  kernel `c^(p/d)` (principal value).
- other alpha: `1^alpha = 1`; c decided zero: `Err(UndecidedZero(0^alpha))`; else a kernel
  `c^alpha` (`NonZero` when c is decided nonzero, else `Opaque`).

d-th-power free radicals (review 1, m10): `n = m^d * r` by trial division with the 6542 primes below
2^16 when n has at most 4096 bits (charged `bits(n)/64 * 6542 + 1` units first), then a cofactor that
is a perfect d-th power (`Integer::nth_root`) moves into m. So `sqrt(8) = 2*sqrt(2)` and
`sqrt(8) - 2*sqrt(2)` is `Zero`. Products are not split into prime radicals (that needs a full
factorization): `sqrt(6) - sqrt(2)*sqrt(3)` stays `Unknown` (sound, incomplete, documented).

`apply(op, c)` (the value f(c)):

- Exact values at rational q: exp(0) = 1, log(1) = 0, sin/tan/sinh/tanh/arcsin/arctan/arcsinh/
  arctanh/erf(0) = 0, cos(0) = cosh(0) = 1, arccos(1) = arccosh(1) = 0, abs and sign of q,
  sqrt via `pow_const(c, 1/2)` (so `sqrt(2)` and `2^(1/2)` are the same kernel). Poles of the
  constant are `Err(DivisionByZero)`: log(0), arctanh(1), arctanh(-1), zeta(1), gamma and factorial
  at a pole.
- Exact trig at rational multiples of pi (`c = q*pi`, a single monomial): sin, cos and tan for
  denominators 1, 2, 3, 4, 6 (values in Q(sqrt 2, sqrt 3); tan at an odd multiple of pi/2 is
  `Err(DivisionByZero)`). Other q: a `NonZero` kernel (sin(q pi) = 0 iff q is an integer, cos(q pi) =
  0 iff q - 1/2 is an integer).
- Exact logs (review 2, m3; `EXP/rev2/proto3/konst.rs::log_exact`): for c a single monomial
  `q * e^k * exp(r_1)^(j_1) * ... * exp(r_m)^(j_m)` with q a positive rational, k and j_i integers and
  r_i rationals, `log(c) = log(q) + k + j_1 r_1 + ... + j_m r_m` (`log(q)` is 0 for q = 1, else the
  `NonZero` kernel `log(q)`). Every factor is a positive real, so the principal log is additive. So
  `log(e^3) = 3`, `log(exp(2)) = 2`, `log(2*e) = 1 + log(2)`. A negative q or any other kernel is not
  folded (`log(-e)` stays a kernel; it is 1 + I pi).
- Classification of the remaining kernels f(c): for rational q, f(q) is `NonZero` for exp, log
  (q != 1), sin, cos, tan, sinh, cosh, tanh, arcsin, arccos (q != 1), arctan, arcsinh, arccosh
  (q != 1), arctanh, erf, gamma, factorial (pi is irrational, Lambert 1761, so sin q = 0 and
  cos q = 0 have no rational solution q != 0; the hyperbolic and inverse functions vanish only at the
  point listed; Gamma has no zeros). zeta(q) is `Opaque`. For non-rational c: exp(c) is always
  `NonZero`; sqrt(c) is `NonZero` when c is; f(c) for c a non-constant polynomial in indeterminates
  only is `NonZero` (generic); otherwise `Opaque`.

**Zero test** `zero_status(&self) -> Result<ZeroStatus, SeriesError>` (sound;
`EXP/rev2/proto3/konst.rs::zero_status`, `const_status`, `numeric`):

1. No terms: `Zero`.
2. One term `q * m`: `NonZero` if m has no `Opaque` kernel (a product of nonzero numbers, real or
   complex, is nonzero), else the numeric test (rule 5).
3. Several terms. If some non-indeterminate kernel depends on an indeterminate: the numeric test
   (which cannot evaluate indeterminates, so in practice `Unknown`). Else group the terms by their
   indeterminate monomial. With a non-empty indeterminate part, the constant is a Laurent polynomial
   in independent indeterminates over the constants, so it is `NonZero` iff some group coefficient
   is (by rules 4 and 5), else `Unknown`.
4. Constants (no indeterminates) whose kernels are only `I` and one single named constant (only pi,
   or only e): `NonZero`. Proof: write c = A(t) + B(t) I with A, B Laurent polynomials over Q and t
   = pi or e. t is real and transcendental (Lindemann 1882, Hermite 1873), so A(t) and B(t) are real
   and c = 0 forces A(t) = B(t) = 0, hence A = B = 0 as polynomials; but the normal form is not
   empty.
5. Numeric test (review 1, B1): split c = A + B I, where A collects the terms without the kernel `I`
   and B the terms with `I` (with `I` removed). The answer is `NonZero` iff **A and B both have real
   ball enclosures** (§2.3) and one of them excludes 0, at one of the precisions 64, 256, 1024 bits;
   otherwise `Unknown`. An enclosure exists only for an expression built from rationals, pi, e and
   real functions on balls inside their real domains, so its value is a real number. Then A and B
   are real, c = 0 iff A = B = 0, and a ball that excludes 0 proves one of them nonzero. A kernel
   that may be non-real (`(-4)^(1/2)`, `exp(I*pi/2)`, `log(-1)`, an `Inverse` of a complex constant)
   has no enclosure, so the answer is `Unknown`, whatever B is. This replaces revision 0's
   unsound rule "NonZero if either part is proved nonzero". An enclosure is also refused when any
   value computed for it leaves the exponent window of §2.3 (review 2, B1), so a constant whose
   evaluation would underflow or overflow in MPFR gets `Unknown`: `(1/2)^1073741827 * 2^1073741822 -
   1/32` and `exp(-744261120) * exp(744261117) - exp(-3)` (both exactly 0) are `Unknown`.

Soundness: every rewriting above is a true identity, so a zero normal form is a zero value
(`Zero` is sound). Each `NonZero` answer is one of: a product of a nonzero rational and kernels
proved nonzero; a nonzero polynomial in algebraically independent indeterminates (the generic
convention); a nonzero polynomial in one transcendental real and I; or rigorous real enclosures of
both parts with one excluding 0. `sin(1)^2 + cos(1)^2 - 1` stays `Unknown` (it is zero; the numeric
enclosure always contains 0). `sqrt(-4) - 2*I` and `exp(I*pi/2) - I` stay `Unknown`.

**Decided non-integer** `non_integer_status(alpha) -> Result<(ZeroStatus, Option<Integer>),
SeriesError>` (for m1 of review 1, used by `pow_const` of a series of nonzero valuation): `NonZero`
when alpha is decided not to be an integer: alpha a non-integer rational; or alpha has an
indeterminate group with a nonzero coefficient (then alpha - n is nonzero for every integer n,
generic convention); or alpha = A + B I with real enclosures where B excludes 0; or B is exactly 0
and, with k = floor(mid(A)), the balls of A - k and A - (k + 1) both exclude 0 (the enclosure of A is
an interval containing its midpoint, so it lies in (k, k + 1)). `Zero` when alpha is an integer (a
rational integer), with that integer. Otherwise `Unknown`, with the integer n nearest to mid(A) when
one was computed; the caller reports `UndecidedZero(alpha - n)`, or `UndecidedZero(alpha)` without
n. The floor is computed only when |mid(A)| < 2^p (a midpoint of magnitude at least 2^p has p
significant bits, so it is itself an integer, the ball contains it, and the test cannot succeed at
that precision), and it is converted from MPFR through
binary digits (`rug::Integer::write_digits`), never through a decimal string: the decimal parse of
num-bigint is quadratic, and a 2^28-bit midpoint would take hours (found while fixing review 2, B2).
`x^(2^100000000 * sin(1))` at 0 is `UndecidedZero(2^100000000 * sin(1))` in 2 ms.

**Inverse.** `inverse(strict)`: `Zero` is `Err(DivisionByZero)`. A single monomial without `Opaque`
kernels is inverted exactly (exponents negated with `checked_neg`). Otherwise a kernel `1/P0` is
created, where `P = c * P0`, `P0` has coprime integer coefficients and a positive leading coefficient
(`EXP/rev2/proto3/konst.rs`, so `1/(2+sin(1))` prints as such). `strict = true` (a derived leading
coefficient) requires `NonZero`, else `Err(UndecidedZero(c))`. `strict = false` is used only for
constants that the **input itself** divides by (a variable-free denominator): the input is assumed
defined, and an inverse of a defined constant is nonzero. This is the distinction between "the input
divides by a constant" (generic, like Risch) and "the expansion must know a valuation" (decided,
else refused).

`to_expr()`: a balanced sum of monomials; a monomial is `q * (numerator kernels)` divided by the
product of negative-exponent kernels and `Inverse` denominators; q = 1 and q = -1 print without the
factor.

### 2.3 Rigorous numeric enclosures (`series_ball.rs`)

The module works on `Expr` only (no dependency on the constant field):

```rust
pub(crate) fn enclose_real(e: &Expr, prec: u32) -> Option<Ball>;
/// A-priori cost of enclose_real(e, prec) in work units: a walk over the nodes that
/// enclose_real will visit (shared subtrees once per occurrence); it stops once the sum passes
/// `limit` and then returns a value above `limit`.
pub(crate) fn ball_cost(e: &Expr, prec: u32, limit: u64) -> u64;
```

`series_const.rs` calls them on `A.to_expr()` and `B.to_expr()` (rule 5) after checking that A and B
contain no `I` and no indeterminate kernel: first `w.charge(ball_cost(&e, p, w.remaining()))?`, then
`enclose_real(&e, p)`, at each precision p of 64, 256, 1024 (review 2, B2: the revision 1 charge
`words(A) * (p/64 + 1)^2 * 8` counted a kernel as one word, so `sin(2^100000000)` cost 2 words; the
walk counts the expression that is evaluated).

Midpoint-radius balls over `rustmath_reals::RealMPFR` (MPFR rounds every operation correctly to
nearest: Fousse et al., ACM TOMS 33(2), 2007; ball arithmetic as in Johansson, IEEE Trans. Comput.
66(8), 2017). For a result r the rounding error is at most `|r| 2^-p` **when no underflow or
overflow happens**; an input radius propagates through f with a Lipschitz bound L of f on the ball;
every computed radius is multiplied by `1 + 2^(4-p)` so that it stays an upper bound despite its own
rounding. Powers of two (2^-p, 2^(4-p)) are built exactly by shifts.

**Exponent range (review 2, B1).** MPFR's default exponent range is [emin, emax] = [1 - 2^30,
2^30 - 1] (checked: `rug::float::exp_min()`). A result below 2^(emin-1) in magnitude underflows to 0
or to 2^(emin-1), and one above 2^emax overflows to infinity; then the bound above fails. Revision 1
decided the exact zero `(1/2)^1073741827 * 2^1073741822 - 1/32` nonzero this way: the first power
underflowed to [0, 0]. Two checks make every accepted enclosure free of underflow and overflow:

- **R1, window.** Every MPFR value the module computes (midpoints, radii, Lipschitz bounds,
  intermediate terms of radii) must be 0 or have its exponent in [-W, W], W = 2^28; otherwise the
  operation returns `None` and there is no enclosure. Inputs are checked when they are built
  (integers, rationals, floats, pi, e).
- **R2, exact zeros.** A computed midpoint equal to 0 is accepted only when the exact result is 0:
  for a product, a factor's midpoint is 0; for a sum, the two midpoints are exact negatives
  (`a == -b`); for f(m), m is a zero of f (sin, tan, atan, tanh, sinh, erf, sqrt, asin, asinh,
  atanh, abs at 0; log, acos, acosh at 1; exp, cos, cosh never). For an integer or a rational input,
  the number is 0. Otherwise `None`.
- exp, sinh and cosh require `|m| + r <= 2^27` before MPFR is called (exp(2^27) has exponent about
  1.94 * 10^8 < W).

Why this suffices: with operands in the window, a nonzero exact product has exponent at least -2W - 1
= -2^29 - 1 > emin, a nonzero exact sum is a multiple of the unit in the last place of the operand
with the smaller exponent, so at least 2^(-W-p) > 2^emin, a quotient 1/m is at least 2^(-W-1), and
exp, sinh, cosh stay inside the window by their guard; log, sqrt, atan, tanh, erf, asin, acos and the trig functions of a windowed argument cannot
overflow. So the only possible underflow is a nonzero exact result that MPFR flushes to 0 or to
2^(emin-1): R2 refuses the first, R1 the second (its exponent is emin < -W). A representable result
outside the window is refused by R1 before any later operation uses it. Every value an accepted
enclosure was computed from is therefore a correctly rounded MPFR result, and the radius bounds hold.
R1 is conservative: values of magnitude below 2^-W or above 2^W (such as `2^(2^29)` or
`exp(-744261120)`) get no enclosure, and constants built from them stay `Unknown` (sound). The
five test inputs of §4.1 `undecided_after_underflow` are exactly zero and give `UndecidedZero`.

**Argument size of sin and cos (review 2, B2).** MPFR's sin and cos reduce the argument modulo 2 pi
with as many bits of pi as the argument's exponent: at p = 64, sin(1.5 * 2^(10^7)) takes 3.8 s,
while exp, log, sqrt, atan, tanh, erf and sinh take at most 10 us at any exponent
(`EXP/rev2/mpfr_cost/out.txt`). So MPFR sin and cos are called only when `r < 1` and the midpoint is
0 or has exponent at most `p + 64`. Otherwise sin and cos get the trivial enclosure [-1, 1]
(midpoint 0, radius 1) and tan gets none. For r >= 1 the Lipschitz enclosure [sin m - r, sin m + r]
has width at least 2 and contains 0, so [-1, 1] is never wider. A rounded ball whose midpoint has
exponent above p has r >= 1, so the second condition only matters for exact inputs (a float such as
`1e300`). Then `sin(2^100000000) + 2` is enclosed in [1, 3] and decided `NonZero` at once.

Operations:

- `+ - *`: the usual midpoint-radius formulas. `1/x` requires `|m| > r`, radius `r / (|m| (|m| - r))`.
- exp: L = exp(m + r). log, sqrt: require `m - r > 0`, L = 1/(m - r), 1/(2 sqrt(m - r)).
  sin, cos (guarded as above), atan, tanh: L = 1. erf: L = 2 (true bound 1.13). sinh, cosh:
  L = cosh(|m| + r). asin, acos: require `|m| + r < 1`, L = 1/sqrt(1 - (|m|+r)^2). asinh, acosh,
  atanh via their log formulas (acosh requires `m - r > 1`). tan = sin/cos (no enclosure when the
  guard fails). Integer powers by binary powering (|k| <= 2^32, every product checked by R1 and R2);
  other powers `exp(b log a)` (requires a > 0). pi and e: MPFR constants. Integers and rationals:
  `with_val_integer` / `with_val_rational` (correctly rounded; both convert through binary digits,
  in time linear in the size). Anything else (gamma, zeta, named functions, indeterminates, `I`, a
  log of a non-positive ball): no enclosure.
- erf uses `RealMPFR::from_float(x.as_float().clone().erf())` (inherent rug method, no new
  dependency); asinh, acosh, atanh use log formulas (`RealMPFR` has no direct methods).
- `ball_cost` units per node, with `u = (p/64 + 1)^2 * 8`: integer and rational leaves `u + bits/64 +
  1` (the conversion); floats, symbols, `+ - * /`, functions `u`; `^` with an integer exponent k
  `u * (2 * min(bits(k), 33) + 1)` (binary powering), other `^` `3u`; sin and cos `t = ((2p + 64)/64 +
  1)^2 * 8` (the argument reduction at exponent at most p + 64), tan `3t`; asinh, acosh, atanh `6u`;
  other unary functions `2u`. Every MPFR call of `enclose_real` is covered by its node's units, and
  each call costs at most a constant times the cost of an operation at precision 2p + 64.

New dependency: `rustmath-reals = { path = "../rustmath-reals" }` in `rustmath-symbolic/Cargo.toml`.
No cycle: rustmath-reals depends on core, integers, rationals, num-traits, rug only. This consumes an
existing numeric crate instead of wrapping rug privately.

Effect (prototype): `1/(2 + sin(x))` and `log(1 + sin(x))` at x = 1, `1/(x + e - pi)` at 0,
`1/(x + sin(1) + I)` at 0 expand; `1/(x + sin(1)^2 + cos(1)^2 - 1)` at 0 is
`Err(UndecidedZero(-1 + cos(1)^2 + sin(1)^2))`; `1/(x + sqrt(-4) - 2*I)` is `UndecidedZero`.
Revision 2 prototype (`EXP/rev2/revchecks3.out`, debug): the five underflow and overflow inputs of
§4.1 are `UndecidedZero` in 2 to 13 ms; with budget 1e8, `1/(x + sin(2^100000000) + 2)` and its
`cos` twin are `Ok` in 2 ms, the `tan` twin and `1/(x + sin(2^1000000000) + 2)` are `UndecidedZero`
in 2 ms (revision 1: 111.7 s release for the first, over 30 s for the last).

### 2.4 Series operations (`series_engine.rs`)

Context: `struct Ctx<'a> { var: &'a Symbol, xser: Laurent, p: i64, work: &'a mut Work }`. Every
`SeriesConst` operation receives `cx.work`. Sparse loops: only structurally nonzero terms are
visited. A recurrence that fills a dense buffer of n slots first calls `cx.alloc(n)`, which charges
`16 n` units, then allocates; its output is compacted to sparse terms. Before step k of any
recurrence, the step's cost `sum_j words(u_j) * words(g_{k-j})` (plus a base of 1) is charged.

| operation | precondition | result valuation and precision | recurrence |
|---|---|---|---|
| `add(a, b)` | | merge, prec = min(pa, pb), then `trunc(P)` | termwise |
| `neg`, `scale(k)` | | unchanged | termwise |
| `mul(a, b)` | | val = va + vb (checked), prec = min(va + pb, vb + pa) (EXACT absorbs); computed up to `min(prec, max(P, val + 1))` (at least the first term, review 2 m1), or in full for small exact operands (§2.1) | first pass: sum the pair costs `words * words` of every contributing pair and charge it; second pass: products, accumulated in a dense buffer when the span is at most `2 * pairs + 64` slots (charged), else in a `BTreeMap` |
| `inv(b)` | b not empty (exact empty: `DivisionByZero`; truncated empty: `NeedPrecision`); c0 `NonZero` (strict inverse) | val = -vb; prec = pb - 2 vb, or P when b is exact; an exact monomial inverts exactly | g0 = 1/c0, g_k = -g0 sum_{j=1..k} b_j g_{k-j} over the span `prec - val` |
| `pow_int(b, n)` | n any Integer | see "integer powers" below | |
| `pow_const(b, alpha)` | alpha a non-integer constant; b not empty; c0 `NonZero`; vb = 0, else see "non-integer powers" below | val 0, prec = min(pb, P) | w = b/c0; Miller: g0 = 1, g_k = (1/k) sum_{j=1..k} ((alpha+1) j - k) w_j g_{k-j} (Knuth §4.7 eq. (9)); times `pow_const(c0, alpha)` |
| `exp(b)` | vb >= 0 (vb < 0 with lead `NonZero`: `NotLaurent`; lead `Unknown`: `UndecidedZero`) | val 0, prec = min(pb, P) | b = b0 + u; E_0 = 1, E_k = (1/k) sum j u_j E_{k-j}; times `apply(Exp, b0)` |
| `log(b)` | b not empty; c0 `NonZero`; vb = 0 (else `NotLaurent` log singularity) | val 0, prec = min(pb, P) | w = b/c0; k L_k = k w_k - sum_{j=1..k-1} j L_j w_{k-j}; plus `apply(Log, c0)` |
| `sin_cos(b)`, `sinh_cosh(b)` | vb >= 0 | prec = min(pb, P) | b = b0 + u; S_0 = 0, C_0 = 1, S_k = (1/k) sum j u_j C_{k-j}; C_k = -(1/k) sum j u_j S_{k-j} (trig) or +(1/k) sum j u_j S_{k-j} (hyperbolic); sin(b0 + u) = sin b0 C + cos b0 S, cos(b0 + u) = cos b0 C - sin b0 S; sinh(b0 + u) = sinh b0 C + cosh b0 S, cosh(b0 + u) = cosh b0 C + sinh b0 S |
| `tan`, `tanh` | | | sin/cos via `mul(s, inv(c))` (a pole if cos b0 = 0 is decided, e.g. at pi/2) |
| arctan, arcsin, arccos, arcsinh, arccosh, arctanh, erf | vb >= 0 | prec = pb (integration adds one, the derivative loses one) | G = f(b0) + integral of Q, Q = b' / (1 + b^2), b' (1 - b^2)^(-1/2), -b' (1 - b^2)^(-1/2), b' (1 + b^2)^(-1/2), b' (b - 1)^(-1/2) (b + 1)^(-1/2) (principal branch, not (b^2 - 1)^(-1/2)), b' / (1 - b^2), (2/sqrt(pi)) exp(-b^2) b'. Q must have valuation >= 0 (else `NotLaurent` log branch point; `pow_const` already refuses arcsin at 1) |
| `abs`, `sign` | **real variable and real point assumed** (documented): all coefficients rational, vb even | abs: sign(c0) b; sign: exact constant sign(c0) | odd vb: `NotLaurent`; non-rational coefficients: `Unsupported` |
| `deriv`, `integrate` | integrate: no h^-1 term | prec - 1, prec + 1 | formal |

Integer powers `pow_int(b, n)` (in this order):

1. n = 0: exact 1. b exact zero: 0 for n > 0, `DivisionByZero` for n < 0.
2. b an exact monomial `c h^v`: exact `pow_const(c, n) h^(n v)` (n v checked), kept whatever P is.
3. n < 0: `pow_int(inv(b), -n)` (strict inverse of the lead).
4. b empty and truncated, `O(h^q)`: `O(h^(n q))` for q > 0 (capped at P), else `NeedPrecision`.
5. vb > 0 and n vb >= P (n compared as an `Integer`): the leading term
   `pow_const(c0, n) h^(n vb) + O(h^(n vb + 1))` (review 2, m1), or `O(h^P)` when n vb does not fit
   in i64. The term is exact whatever c0 is (an undecided c0 is reported by the strict inverse that
   needs it, not here), and its precision `n vb + 1` is at most `n vb + (pb - vb)`, the true
   precision of b^n. Revision 1 returned `O(h^P)`, which forgot the valuation: the inverse of
   `sin(x)^200` met an empty series, the `NeedPrecision` doubling stopped at extra 128, and
   `1/sin(x)^200` was `PrecisionExhausted(131)`.
6. n <= `POW_BINARY_MAX = 64`: binary powering with `mul` (exact canonical coefficients).
7. n > 64: write b = h^vb (c0 + u') with u' of valuation >= 1, and compute S = min(relative
   precision of b, P - n vb) slots (charged), then shift by n vb:
   - c0 decided `NonZero`: Miller's recurrence with alpha = n on w = u'/c0, times `pow_const(c0, n)`
     (size-capped, kernel fallback). Cost O(S^2) independent of n: `(1+x)^10000` at order 2000 takes
     11.5 ms (release; revision 0 binary powering: 12.9 s). When c0 has several terms the output
     contains `c0^n * (1/c0)^k`, which is correct but not reduced: `(x + sin(1) + cos(1))^4096` at 0
     has `4096 * (cos(1) + sin(1))^4096 / (cos(1) + sin(1))` as its x coefficient.
   - c0 `Unknown`: the binomial theorem, no division and no zero test:
     `sum_{m < S} C(n, m) pow_const(c0, n - m) u'^m`. Cost O(S^3); charged.

Non-integer powers of a series of nonzero valuation (review 1, m1): `non_integer_status(alpha)`:
`NonZero` gives `NotLaurent` ("a non-integer power of an argument of valuation v has a branch point
there"; no engine message formats a constant, since a constant may hold a 2^20-bit integer whose
num-bigint decimal conversion is quadratic); `Zero` (alpha is an integer after all) gives `pow_int`;
`Unknown` gives `UndecidedZero(alpha - n)`, or `UndecidedZero(alpha)` when no nearest integer n was
computed (|mid(alpha)| >= 2^1024). So `x^pi`, `x^y` and `x^(1/2 + I)` at 0 are `NotLaurent`, and
`x^(sin(1)^2 + cos(1)^2)` at 0 is `UndecidedZero(-1 + cos(1)^2 + sin(1)^2)`.

Messages of `NotLaurent` (review 1, m9): an argument of negative valuation inside exp, sin, cos,
sinh, cosh, arctan, arcsin, ... gives "<f> of an argument with a pole at the point has no Laurent
expansion there" (this covers the essential singularity of exp(1/x) and the jump of arctan(1/x));
log gives "log has a logarithmic singularity at the point"; abs and sign give "<f> of an argument
with a zero of odd order is not analytic there".

Not expanded (`Err(Unsupported)`): `Gamma`, `Factorial`, `Zeta`, `Function(..)` and `Mod` applied to
an expression in the variable. (Gamma's series needs polygamma constants; a later unit can add it.)
When the argument is free of the variable they are kernels.

### 2.5 Expanding an `Expr`

`expand(e, cx)`:

- `e` free of the variable: `SeriesConst::from_expr(e)` as an exact constant series.
- the variable: `cx.xser`.
- `Add`, `Sub`, `Mul`, `Neg`: as above.
- `Div(l, r)`: r free of the variable: scale by `from_expr(r).inverse(false)` (the input divides by
  that constant); else `mul(expand(l), inv(expand(r)))` (strict).
- `Pow(l, r)`:
  - r free of the variable: alpha = `from_expr(r)`; an integer gives `pow_int`, any other
    `pow_const`.
  - else expand r first. If its series is exact with no term other than h^0 (for example
    `x - x + 2`), alpha is that constant and the previous case applies (review 1, m1: `x^(x-x+2)`
    at 0 is `x^2`).
  - else l free of the variable (review 1, m5): `lc = from_expr(l)` and its strict zero status:
    `Zero`: `0^r` is the exact 0 when r has valuation >= 0 and a positive rational constant term
    (0^r = 0 where Re r > 0), else `NotLaurent("0^r with r not positive at the point")`; `Unknown`:
    `Err(UndecidedZero(lc))`; `NonZero`: `exp(expand(r) * apply(Log, lc))`.
  - else (both depend on the variable): `exp(expand(r) * log(expand(l)))`, so x^x at 0 is
    `NotLaurent`, at 1 it expands.
- `Unary`: the table of §2.4; `Sqrt` is `pow_const(b, 1/2)`.

The engine expands every subexpression on its own. So a function whose subexpression has no Laurent
expansion is refused even when the function itself has one: `sqrt(x)^2` and `exp(log(x))` at 0 are
`NotLaurent` (documented on `NotLaurent`; Puiseux series are out of scope, §7).

Recursion depth is the depth of the input expression (unchanged from today's `differentiate`).

### 2.6 Driver: working precision search

```rust
pub(crate) fn expansion(f, var, point: &ExpansionPoint, max_exp: i64, work: &mut Work)
    -> Result<Laurent, SeriesError>
```

Target T = max_exp + 1 (checked; overflow is `Unsupported("order too large")`). P starts at T. Each
attempt expands with precision P, all attempts charging the same `Work`:

- result `prec >= T`: return `trunc_hard(T)`.
- result `prec < T`: a decided precision loss (an `inv` with a decided leading coefficient loses
  exactly 2 v; cancellations). The missing amount is known: `P += T - prec`. Not capped (review 1,
  m2); bounded by the attempt limit and the budget.
- `NeedPrecision` (an `inv`, `log` or `pow_const` met an empty truncated series, whose valuation is
  undecided): double the extra precision, `P = T + max(2 (P - T), 8)`, while the extra stays at most
  `max(4 |T|, 128)`; beyond that, `Err(PrecisionExhausted { precision: P })`.
- any other error: return it.

At most 24 attempts; then `Err(PrecisionExhausted { precision: P })`. This is a bounded search: it
never claims the denominator is zero.

Examples (prototype): `1/(sin(x)^2 + cos(x)^2 - 1)` at 0, order 3 (identically zero denominator,
every rational coefficient cancels) is `PrecisionExhausted(132)` after 0.5 s debug, 0.08 s release;
`1/(x - x)` (exact zero) is `DivisionByZero`. `1/(sin(x) - x)` at order 3 needs one loss-driven
retry (P = 4, then 10). `1/(sin(x)^50 - x^50)` at order 3 (review 1, m2) runs the NeedPrecision
doubling 4, 12, 20, 36, 68 (the denominator becomes visible at 68 > 52), then one loss-driven step
to P = 108, and returns valuation -52 with the gp coefficients (1.1 s debug, 0.12 s release).
`1/(x^300*(1+x))` needs no search (the exact denominator is kept, §2.1). `1/sin(x)^200` at order 2
(review 2, m1): at P = 3, `pow_int` keeps the leading term `x^200 + O(x^201)`, the inverse has
precision -199, and two loss-driven steps (P = 205, then 403) give the result with valuation -200,
equal to gp (1.3 s release; revision 1: `PrecisionExhausted(131)`). `1/(x+x^2)^200` and
`1/(exp(x)-1)^150` take the same three attempts (15 ms and 2.3 s release; `S1TRACE=1
EXP/rev2/bin/proto3_rel measure f 0 2` prints the attempts). The loss-driven step undershoots once
here because the precision of the kept leading term (201) does not grow with P until P passes 200;
the next attempt corrects it.

The point must not contain the variable (`InvalidPoint`). The point is any constant: `10^30`,
`pi/2`, `I`, a symbol `a`, `exp(1)`.

### 2.7 Work budget and cost model

```rust
/// Per-expansion state that every constant operation needs: the work counter and the kernel
/// interning table (§2.2). One `Work` serves all attempts of the driver.
pub(crate) struct Work { used: u64, max: u64, kernels: HashMap<String, Kernel> }
impl Work {
    pub fn new(max: u64) -> Work;
    pub fn unbounded() -> Work;                       // max = u64::MAX; the counter still counts
    pub fn charge(&mut self, units: u64) -> Result<(), SeriesError>;  // Err(BudgetExceeded) once used > max
    pub fn remaining(&self) -> u64;                   // max - used (the limit of the a-priori walks)
    pub fn intern(&mut self, key: String, make: impl FnOnce(u64, String) -> KernelData) -> Kernel;
}
```

Every constant operation and every series step calls `w.charge(units)?` **before** it does the
work, with units an a-priori size in 64-bit words
(`words(c) = sum over terms of (bits(num) + bits(den))/64 + 1 + monomial length`, at least 1):

| operation | units charged before it runs |
|---|---|
| `SeriesConst::add`, `sub` | `w = wa + wb` if every denominator is 1, else `w * w` (the gcd of the rational normalization is quadratic); adding to zero is free |
| `SeriesConst::mul`, `scale` | `wa + wb` if one side is a single word, else `wa * wb` (all monomial pairs) |
| exact power of a rational or a single monomial | `bits(q) * |n| / 64 + 1` (the size cap of §2.2 applies first) |
| multi-term exact power | each `mul` of the binary powering, as above |
| radical extraction | `bits(n)/64 * 6542 + 1` |
| numeric zero test, per precision p | `ball_cost(A.to_expr(), p, remaining)` and the same for B (§2.3; review 2, B2) |
| kernel creation | `key_size / 8 + 1`, before the key is built (§2.2; review 2, m2) |
| series `add` | number of terms of both operands, plus the constant adds |
| series `mul` | the sum over contributing pairs of the constant `mul` units, in one charge before the second pass |
| recurrence step k | `1 + sum_j words(u_j) * words(g_{k-j})` (Miller: times `words(alpha + 1)`) |
| dense buffer of n slots | `16 n`, before the allocation |
| driver | nothing extra: all attempts charge the same counter |

The model is deterministic (tests can rely on it). Its quadratic gcd term is essential: a first
prototype counter that charged products only let `exp(sin(x))` at order 2000 run 115 s (release)
before reaching 7.5e7 units. With the model above the same call is refused after 3.9 s. The size
caps of §2.2 (exact constant powers) and §2.1 (exact series kept, exact products in full) are
structural and apply with or without a budget; they keep `taylor` (unbounded) from expanding huge
constants. `taylor` and friends use `Work::unbounded()`; `taylor_with_budget` and
`series_expansion(.., Some(budget))` return `Err(BudgetExceeded { work, budget })` at the first
charge that passes the budget (no partial result).

Inherent output size is not bounded by the caps: `1/(x^(10^9)*(1+x))` up to x^3 has 10^9 + 4 terms,
and `x^(-10^9)*sin(x)` up to x^1 has 5*10^8. Unbounded calls compute them; the budget refuses them
before the dense buffer is allocated (the buffer is charged first).

What the budget bounds (review 2, m5). Every coefficient word in memory was charged when it was
produced, so the budget bounds the work and the memory of the result. It does not bound the printed
text: a kernel is stored once (its `Expr` is shared through `Arc`) and printed once per coefficient
that contains it, and each 64-bit word of a coefficient prints as about 19 decimal digits.
Measured (release): `(1+x)^(10^100)` at order 1400 uses 7.4e7 units, 0.28 s and 105 MB, and prints
95.8 MB (review 2); `(x + sin(2^100000))^1000` at order 1000 uses 1.4e5 units, 4 ms and 5.7 MB peak
(35 MB with the text), and prints 30.4 MB in 1.7 s. The printed size is the REPL's concern: J1b estimates
a value's text before formatting it and shows a value above `OUTPUT_BYTES` (16 MiB) as a summary.

### 2.8 Output rendering

`Laurent -> Expr`: terms in increasing exponent, coefficient k:

- exponent 0: `c`; exponent e > 0: `h^e` (`h` when e = 1), `-h^e` when c = -1, `c * h^e` otherwise;
  exponent e < 0: `c / h^|e|`.
- h is `x` when the point is zero (decided by `SeriesConst::from_expr(point)` being the exact zero,
  not by matching `Expr::Integer`), `x - a` otherwise (a as the caller gave it), and `x` with
  exponent -e at infinity.
- the terms are combined by a **balanced** sum (`balanced_sum`: sums of at most 4 terms stay
  left-deep `((a + b) + c) + d`; longer ones split in halves). The sum has depth about log2 n plus
  the depth of a term (3); the whole result had depth 18 at order 20000 in the prototype.
- no `simplify` call on the output (it is already canonical).

Examples (prototype output):
`taylor(exp(x), x, 0, 5) = (((1 + x) + (1/2 * (x ^ 2))) + (((1/6 * (x ^ 3)) + (1/24 * (x ^ 4))) + (1/120 * (x ^ 5))))`,
`taylor(1/sin(x), x, 0, 5) = ((((1 / x) + (1/6 * x)) + (7/360 * (x ^ 3))) + (31/15120 * (x ^ 5)))`,
`taylor(exp(x), x, 1, 2) = ((exp(1) + (exp(1) * (x - 1))) + ((1/2 * exp(1)) * ((x - 1) ^ 2)))`,
`taylor(x^(-1000000000) + 1, x, 0, 1) = ((1 / (x ^ 1000000000)) + 1)`.
(These are the HEAD Display strings; §2.11 adds parentheses around rational exponents.)

### 2.9 Public API (`series.rs`)

```rust
/// Errors of the series engine (MAGMA ch. 49).
#[derive(Debug, Clone, PartialEq)]
pub enum SeriesError {
    /// A subexpression has no Laurent expansion at the point: a branch point (sqrt(x) at 0), a
    /// logarithmic singularity (log(x) at 0), an argument with a pole inside exp, sin, arctan, ...
    /// (exp(1/x) at 0), or abs/sign at a zero of odd order. The engine expands each subexpression
    /// on its own, so the function itself may still have one (sqrt(x)^2 and exp(log(x)) at 0).
    NotLaurent(String),
    /// gamma, factorial, zeta, a named function or mod applied to an expression in the variable,
    /// abs/sign of a series with non-rational coefficients, an exponent that overflows i64.
    Unsupported(String),
    /// The expansion must divide by this constant (or read its sign) and cannot decide whether it
    /// is 0. For a power whose exponent alpha must be known to be an integer or not, the constant
    /// is alpha - n with n the nearest integer, or alpha itself when no nearest integer was
    /// computed. The numeric test refuses values outside a safe exponent window, so a constant
    /// whose evaluation would underflow or overflow is reported here, never decided.
    UndecidedZero(Expr),
    /// The expression divides by an exact zero (1/(x - x), log(0), 0^-1, tan(pi/2) as a constant).
    DivisionByZero(String),
    /// The point contains the variable, or infinity (oo) appears.
    InvalidPoint(String),
    /// Bounded search: a denominator vanished to this working precision. It may be identically 0.
    PrecisionExhausted { precision: i64 },
    BudgetExceeded { work: u64, budget: u64 },
    /// laurent(): the expansion has a nonzero term of exponent `valuation` below `min_order`.
    ValuationBelowMinimum { valuation: i64, min_order: i64 },
    /// series_coefficients(): the expansion has a pole of this order (valuation < 0).
    Pole { valuation: i64 },
}
impl fmt::Display for SeriesError { ... }  // one plain sentence each
impl std::error::Error for SeriesError {}

#[derive(Debug, Clone, PartialEq)]
pub enum ExpansionPoint { At(Expr), Infinity }

/// A truncated Laurent expansion sum c_k h^k + O(h^precision), h = var - a (or 1/var at infinity).
#[derive(Debug, Clone, PartialEq)]
pub struct SeriesExpansion {
    pub var: Symbol,
    pub point: ExpansionPoint,
    /// (exponent, coefficient) in increasing exponent; canonical constants, none structurally 0;
    /// every exponent below `precision`. Sparse: x^(-10^9) + 1 has two entries.
    pub terms: Vec<(i64, Expr)>,
    pub precision: i64,          // max_exponent + 1
}
impl SeriesExpansion {
    pub fn valuation(&self) -> Option<i64>;             // exponent of terms[0]; None when no term
    pub fn coefficient(&self, k: i64) -> Option<Expr>;  // None for k >= precision; Some(0) when absent
    pub fn to_expr(&self) -> Expr;                      // §2.8
    pub fn big_o(&self) -> BigO;                        // O(h^precision)
}

impl Expr {
    pub fn series_expansion(&self, var: &Symbol, point: &ExpansionPoint, max_exponent: i64,
                            budget: Option<u64>) -> Result<SeriesExpansion, SeriesError>;
    /// All terms of exponent <= order; poles give negative exponents (as Sage/Maxima taylor).
    pub fn taylor(&self, var: &Symbol, point: &Expr, order: usize) -> Result<Expr, SeriesError>;
    pub fn taylor_with_budget(&self, var: &Symbol, point: &Expr, order: usize, budget: u64)
        -> Result<Expr, SeriesError>;
    pub fn maclaurin(&self, var: &Symbol, order: usize) -> Result<Expr, SeriesError>;
    pub fn series(&self, var: &Symbol, point: &Expr, order: usize) -> Result<Expr, SeriesError>; // = taylor
    /// Terms of exponent in [min_order, max_order]; Err(ValuationBelowMinimum) if a term below
    /// min_order is decided nonzero (instead of dropping the principal part), Err(UndecidedZero)
    /// if the terms below min_order cannot be decided.
    pub fn laurent(&self, var: &Symbol, point: &Expr, min_order: i64, max_order: i64)
        -> Result<Expr, SeriesError>;
    /// Expansion at infinity in powers of 1/var, terms (1/var)^k for k <= order.
    pub fn asymptotic(&self, var: &Symbol, order: usize) -> Result<Expr, SeriesError>;
    /// [c_0, ..., c_order] (dense; absent terms are 0); Err(Pole) when a term of negative exponent
    /// is decided nonzero, Err(UndecidedZero) when it cannot be decided.
    pub fn series_coefficients(&self, var: &Symbol, point: &Expr, order: usize)
        -> Result<Vec<Expr>, SeriesError>;
    pub fn series_with_big_o(&self, var: &Symbol, point: &Expr, order: usize)
        -> Result<(Expr, BigO), SeriesError>;
}
```

`order` above `i64::MAX - 1` is `Err(Unsupported("order too large"))`. `lib.rs` exports
`SeriesError`, `SeriesExpansion`, `ExpansionPoint`. Every public doc comment states the generic
convention for free symbols, and abs/sign state the real-variable assumption.

Header of each new file: MAGMA ch. 49 (digest path, §§ above), Knuth TAOCP 2 §4.7, Brent and Kung,
"Fast algorithms for manipulating formal power series", J. ACM 25(4) 1978, 581-595; series_ball.rs:
Fousse et al. 2007 and Johansson 2017; series_const.rs: Richardson 1968 (zero equivalence is
undecidable in general, which is why the test is three-valued), Lindemann 1882, Hermite 1873.

### 2.10 `simplify` along the left spine (`simplify.rs:376-400`)

```rust
pub fn simplify(expr: &Expr) -> Expr {
    // Binary chains are usually left-deep (a + b + c + ..., built by the parser and by
    // accumulating loops): walk the left spine with an explicit stack.
    let mut spine: Vec<(BinaryOp, &Arc<Expr>)> = Vec::new();
    let mut cur = expr;
    while let Expr::Binary(op, l, r) = cur {
        spine.push((*op, r));
        cur = l;
    }
    let mut acc = simplify_non_binary(cur);          // the old Integer/../Unary/Function arms
    while let Some((op, r)) = spine.pop() {
        acc = simplify_binary(op, acc, simplify(r));
    }
    acc
}
```

The evaluation order (left before right, inner before outer) and every result are unchanged; only
the recursion over left children goes. `simplify_binary` (`simplify.rs:470`) inspects its operands
only one level deep (`as_rational_const` matches `Integer`/`Rational`, `simplify.rs:450`), so a deep
accumulator costs nothing extra. Right-deep chains (power towers) still recurse.

### 2.11 Display: parenthesize rationals and negative bases under `^` and `/` (`expression.rs:444-454`)

The parser change (§2.12) makes `a/b` an operator, so today's Display strings `(x ^ 1/2)` and
`(x / 1/2)` would re-parse as `(x^1)/2` and `(x/1)/2`. The engine outputs such kernels (`2^(1/2)`).
Display prints:

- an `Expr::Rational` with denominator > 1 in parentheses when it is an operand of `Pow` or the
  right operand of `Div`: `(x ^ (1/2))`, `((3/4) ^ (-1/2))`, `(x / (1/2))`;
- the base of a `Pow` in parentheses when it is `Unary(Neg, _)` or a negative `Integer`,
  `Rational` or `Real` (review 1, m3): `((-y) ^ (1/2))`, `((-2) ^ 2)`, `((-1.5) ^ 2)`.

A negative exponent needs no parentheses (`x ^ -2` parses through the signed exponent rule). A
workspace grep finds no test string that depends on the old forms (a grep for `^ p/q`, `p/q ^`,
`/ p/q` and `(-n ^` inside string literals of `*.rs` files is empty). The last rule also fixes the
pre-existing round-trip errors `Pow(-2, 2)` displayed `(-2 ^ 2)` (parses as -4) and
`sqrt(x - y)` at 0, whose kernel `(-y)^(1/2)` displayed `(-y ^ (1/2))` (parses as `-(y^(1/2))`).

### 2.12 The literal parser (S7)

```rust
// Unsigned integer literal of any size.
fn integer_literal(input: &str) -> IResult<&str, Integer> {
    let (rest, digits) = digit1(input)?;
    Ok((rest, Integer::from_decimal_str(digits).expect("digit1 yields decimal digits")))
}
fn number(input: &str) -> IResult<&str, Expr>   // float (no sign) | integer_literal; no a/b token
fn unary(input)          // ('-' | '+') unary | power             ("--5" = Neg(Neg(5)))
fn power(input)          // atom ('^' exponent)?, exponent := ('-' | '+') exponent | power
                         // Neg(Integer(n)) as an exponent folds to Integer(-n): "x^-2" = Pow(x, -2) as today
fn multiplicative(input) // unary (('*' | '/') unary)*, with the fold below
```

Fold in `multiplicative`: after reading `l / r`, if both l and r are signed integer literals
(`Integer(n)` or `Neg(Integer(n))` as the parser built them) and r != 0, the result is the exact
rational (an `Integer` when it divides). r = 0 is a parse error ("division by zero in a literal"), as
today. The fold is value-preserving, so it changes no meaning, and keeps `parse("1/3") ==
Rational(1/3)` (`parser.rs:423` test), `"6/4"` = `Rational(3/2)`, `"1/-3"` = `Rational(-1/3)` as
today. New structures: `"-2/3"` is `Rational(-2/3)` (was `Neg(Rational(2/3))`, same value), `"2^-1"`
is `Pow(2, Integer(-1))` as today.

Also (one-line table entries, same function `function_call`, `parser.rs:254-276`): map `arcsinh` /
`asinh`, `arccosh` / `acosh`, `arctanh` / `atanh` to their `UnaryOp`. Display prints `arcsinh(..)`
(`expression.rs:472`) but the parser read it back as an opaque `Function`, so taylor refused it.

Results (new): every row of §1.6 gets the "correct" column; `1/0` stays an error.

### 2.13 Callers in the crate

- `limits.rs:428-443` `limit_via_series`: use `series_expansion(var, At(point), order, None)`:
  `Err(_)` gives `Unknown`; no term with precision >= 1, or a first exponent > 0, gives
  `Finite(0)`; first exponent 0 with an `Integer`/`Rational` coefficient gives `Finite(c0)`;
  anything else `Unknown` (a pole or a symbolic constant, as before). Correct because terms below
  the first stored one are exact zeros, and a Laurent expansion with valuation >= 0 is continuous at
  the point after removing the removable singularity.
- `limits.rs:554` `limit_with_error`: `series_with_big_o(..).ok().map(|(_, o)| o)`.
- `known_series::{exp, sin, cos, log_one_plus, binomial}` (`series.rs:247-354`): build their sums
  with `balanced_sum` (the same depth problem); `exp` keeps its `simplify`.

### 2.14 `Drop for Expr` with bounded stack use (`expression.rs`, review 1, B4)

The derived drop glue drops a node, then each child `Arc`, and the last owner of a child drops it in
turn: one stack frame chain per tree level. HEAD overflows at 80000 levels (debug, 8 MB), and a
release build overflows at 2*10^5 (§1.2).

```rust
thread_local! {
    /// Highest stack address seen on entry to `Drop for Expr` on this thread.
    static EXPR_DROP_STACK_TOP: Cell<usize> = const { Cell::new(0) };
}

impl Drop for Expr {
    /// Teardown with bounded stack use. While the drop runs less than GLUE_STACK_BUDGET bytes
    /// below the shallowest drop seen on this thread, the drop glue recurses as before. Deeper, the
    /// subtree is torn down iteratively: a uniquely owned child whose own children have children
    /// is moved to a heap list and a leaf is left in its place, so the glue descends at most two
    /// more levels below any node.
    ///
    /// The bound is relative to the shallowest `Expr` drop seen on the thread, not to the top of
    /// the thread's stack: a thread whose first `Expr` drop happens deep in its stack treats that
    /// point as the top, and the glue may use up to GLUE_STACK_BUDGET bytes below it before the
    /// iterative teardown starts. A thread with less than 16 KiB of stack left at its first drop
    /// can therefore still overflow there.
    fn drop(&mut self) {
        const GLUE_STACK_BUDGET: usize = 16 * 1024;
        if !has_children(self) { return; }                   // leaves: no other work
        let marker = 0u8;
        let here = std::ptr::from_ref(std::hint::black_box(&marker)).addr();  // compared, never dereferenced
        let used = EXPR_DROP_STACK_TOP
            .try_with(|top| { if here > top.get() { top.set(here) } top.get().saturating_sub(here) })
            .unwrap_or(usize::MAX);                          // TLS being destroyed: iterative
        if used < GLUE_STACK_BUDGET { return; }              // the glue recursion is shallow here
        let mut list: Vec<Expr> = Vec::new();
        detach(self, &mut list);
        while let Some(mut e) = list.pop() {
            detach(&mut e, &mut list);                       // e then drops with shallow children
        }
    }
}
fn has_children(e: &Expr) -> bool   // Binary, Unary, or Function with arguments
fn deep(a: &Arc<Expr>) -> bool      // some child of *a has children
fn detach(e: &mut Expr, list: &mut Vec<Expr>) {
    // for each child Arc `a` of e: if deep(a), and Arc::get_mut(a) is Some(inner) (this drop is the
    // only owner), push std::mem::replace(inner, Expr::Real(0.0)) onto list
}
```

Correctness:

- Only uniquely owned children are moved (`Arc::get_mut`, which also checks that no `Weak` exists;
  the workspace has no `Weak<Expr>`). A shared child is left in place; its last owner drops it later,
  through the same `Drop`, so it is torn down with the same guarantee.
- Every node is dropped exactly once: a moved node is owned by `list`, its old place holds a leaf
  `Real(0.0)` (no heap), and the moved node is dropped at the end of its loop iteration after its
  own deep children were moved.
- Stack bound: in the iterative branch, the glue below a node descends at most two levels (a child
  that is not moved has no grandchild with children), and each of those drops returns at once or
  starts its own loop at the same depth. Above the switch, at most 16 KiB of glue frames (about 80
  to 160 levels) are used.
- The address of a local is a safe proxy for the stack depth (`ptr::from_ref(..).addr()`, no
  `unsafe`). The stack grows downward on all Rust tier-1 targets. On a target where it grows
  upward, `saturating_sub` gives 0 and the drop is the derived glue, as at HEAD. A thread whose first
  `Expr` drop happens deep in its stack treats that point as the top, so it may use 16 KiB more
  before switching; documented.

Measured (`EXP/rev1/`, the variant is `rev1/sym_v10/src/expression.rs`, compiled into a copy of the
crate with `rev1/build_sym.py`; `drop_measure.txt`):

| check | HEAD | this design |
|---|---|---|
| left-deep sum of 10^6 terms, debug, 1 MiB thread | overflows | built 0.50 s, dropped 0.38 s |
| right-deep power tower, unary chain sin(sin(..)), function chain f(f(..), x), zig-zag `+`/`*`, each 10^6 deep, debug, 1 MiB | overflow | ok |
| 10^6-term sum dropped after 520 frames of about 1.6 KiB on a 1 MiB thread (less than 16 KiB left: 530 frames alone overflow), after shallow drops on the same thread had recorded the stack top (`rev1/workload.rs`, `shapes`); when a thread's first `Expr` drop happens deep in its stack, the glue may still use 16 KiB there (review 2, m4) | overflow | ok |
| a shared deep subtree (two owners), dropped one owner at a time | | ok |
| 3*10^4-term sum, drop only, release, min of 7 | 4.4 ms | 3.8 ms |
| 3000 x (parse, 3 x (differentiate, simplify)), release, min of 8 runs | 107.5 ms | 106.0 ms |
| same loop, 300 iterations, instructions (cachegrind) | 96.5 M | 103.4 M (+7 %) |
| HEAD `taylor(exp(x), 0, 10)` (allocation heavy), release | 188 ms | 179 ms |

Rejected variants (same directory): always-iterative with a `Vec` per drop (`v1`, `v3`), with a
thread-local list (`v8`), depth-counted recursion (`v9`): +18 % to +28 % wall time and +19 % to
+48 % instructions on the differentiate loop, because every internal node pays an `Arc::get_mut`
(an atomic compare-exchange) and two moves.

Consequences: implementing `Drop` forbids moving fields out of an owned `Expr` (E0509). The full
list of sites (found by compiling the workspace chain with this `Drop`, `EXP/dropchain2.txt`) is in
§3.3; every one is fixed by matching on a reference and cloning the `Arc` contents where a value is
needed.

## 3. Callers and behaviour changes (whole workspace)

### 3.1 Series API

`grep -rn "\.taylor(\|\.maclaurin(\|\.series_coefficients(\|\.laurent(\|\.asymptotic(\|\.series_with_big_o(\|\.series(&" --include=*.rs`:

| site | change |
|---|---|
| `rustmath-jupyter/src/repl.rs:1969` (REPL `taylor`) | `taylor_with_budget(&var, &args[2], order, repl_limits::TAYLOR_WORK)`; map errors: `DivisionByZero`, `InvalidPoint`, `Pole`, `ValuationBelowMinimum` to "ValueError" (properties of the input); `NotLaurent`, `Unsupported`, `UndecidedZero`, `PrecisionExhausted`, `BudgetExceeded` to "ComputationError" (the engine could not expand; review 1, m1); message `taylor: {e}`. The `match args[3].simplify()` at `:1961` becomes `match &args[3].simplify()` (an E0509 site, §3.3) |
| `rustmath-jupyter/src/repl.rs:10524-10534` | `TAYLOR_ORDER` 20000 to 2000; new `TAYLOR_WORK` (§5.3, provisional 100_000_000); doc comments rewritten with the new measurements, and they say what each constant bounds (review 2, m5). `TAYLOR_ORDER`: "An argument check: a clear refusal of absurd orders before any work. It bounds neither the work nor the output: `(1+x)^(10^100)` at order 1400 is admitted and prints 95.8 MB." `TAYLOR_WORK`: "Bounds the work of one call and the memory of its result (every coefficient word is charged when it is produced). It does not bound the printed text: a kernel is stored once and printed in every coefficient that contains it, so `(x + sin(2^100000))^1000` at order 1000 uses 1.4e5 units and prints 30.4 MB. The display budget (`OUTPUT_BYTES`) bounds the text." For plain series the order still sets the size: exp(x) prints 5.3 MB at 2000, 38 MB at 5000, 730 MB at 20000 (§5). J1b owns the final value of `TAYLOR_ORDER` (2000 or 5000, its §3 table) |
| `rustmath-symbolic/src/limits.rs:430`, `:554` | §2.13 |
| `rustmath-symbolic/src/lib.rs:291-311`, `:366-368`, `:433-465` | tests: `.unwrap()` the Results; assertions unchanged (`laurent(1/x, .., -1, 2)` is `1/x`, not constant; `asymptotic(x^2)` is `x^2`) |
| `rustmath-symbolic/src/series.rs:795-980` (tests) | rewritten against gp values (§4) |
| `rustmath-calculus/src/taylor.rs` | none (own implementation, see §7) |
| `rustmath-py` (outside the workspace) | does not call these functions |

### 3.2 Parser and Display

Parser change (`parse`) callers: `rustmath-jupyter/src/repl.rs:14` (`parse_expr` for every symbolic
REPL argument, also `:1849`, `:4910`, `:4939`), `rustmath-py/src/symbolic.rs:6`, and the crate's own
tests. REPL behaviour changes: `x^2/4`, `2^3/4`, `2/3^2` now mean what they say; `1/x`, `1/sin(x)`,
`exp(1/x)` now parse; huge integer literals parse.
`grep -nE '\^[0-9]+/[0-9]|[^a-z0-9_.][0-9]+/[a-z(]|\^-[0-9]' repl.rs` finds no REPL test using these
forms.

Display change: only `Pow` operands and `Div` right operands that are rationals, and `Pow` bases that
are negated or negative.

### 3.3 `Drop for Expr`: E0509 sites

Compiling the reverse-dependency chain of rustmath-symbolic (symbolic, calculus, functions,
geometry, manifolds, curves, plot3d, jupyter; libraries and their test targets) with the `Drop` of
§2.14 (`EXP/dropchain.py`, output `EXP/dropchain2.txt`) gives E0509 at exactly these sites. Each is
fixed by matching on `&expr` (and `(**inner).clone()` where an owned value is needed); the fixed
library versions compile (`EXP/dropcheck/src/integrate_exact.rs`, `EXP/dropcheck_geom/plane_curve_ops.rs`,
`EXP/dropcheck_jup/repl.rs`).

Library code (4 sites):

- `rustmath-symbolic/src/integrate_exact.rs:96` and `:148` (the second becomes `let next = match
  &cur { .. }; cur = next;`, returning `(c, cur)` in the non-rational arms);
- `rustmath-geometry/src/plane_curve_ops.rs:489` (`match &simplified`);
- `rustmath-jupyter/src/repl.rs:1961` (`match &args[3].simplify()`, part of the taylor arm rewrite).

Test code (98 sites, all `#[cfg(test)]` modules; `match x {` / `if let Expr::.. = x` on an owned
result becomes `match &x {` with derefs adjusted):

| file | lines |
|---|---|
| `rustmath-symbolic/src/parser.rs` | 454 488 508 528 584 605 626 646 711 735 771 784 870 880 890 900 908 919 936 953 973 |
| `rustmath-symbolic/src/specialfunctions/orthogonal_polys.rs` | 629 662 669 676 683 690 697 710 722 734 |
| `rustmath-symbolic/src/specialfunctions/other.rs` | 466 532 (twice) 558 619 634 641 648 |
| `rustmath-symbolic/src/specialfunctions/special.rs` | 318 324 332 339 345 353 361 370 |
| `rustmath-symbolic/src/specialfunctions/airy.rs` | 300 306 315 321 330 338 394 |
| `rustmath-symbolic/src/specialfunctions/generalized.rs` | 271 279 297 338 400 442 |
| `rustmath-symbolic/src/specialfunctions/min_max.rs` | 252 272 276 299 319 323 |
| `rustmath-symbolic/src/specialfunctions/transcendental.rs` | 315 328 336 343 350 365 |
| `rustmath-symbolic/src/specialfunctions/wigner.rs` | 373 387 404 418 432 446 |
| `rustmath-symbolic/src/specialfunctions/jacobi.rs` | 185 192 199 206 213 |
| `rustmath-symbolic/src/specialfunctions/hypergeometric.rs` | 296 304 313 |
| `rustmath-symbolic/src/radical.rs` | 318 338 367 |
| `rustmath-symbolic/src/specialfunctions/prime_pi.rs` | 255 283 |
| `rustmath-symbolic/src/function.rs` 380, `walker.rs` 685, `specialfunctions/piecewise.rs` 94, `specialfunctions/spike_function.rs` 68 | one each |
| `rustmath-calculus/src/integration.rs` 427, `limits.rs` 309, `sum.rs` 232 | one each |

No other crate has a site: functions, manifolds, curves (library and `tests/genus2_jacobian.rs`),
plot3d and jupyter compile unchanged; rustmath-benchmarks and rustmath-py do not pattern-match on
`Expr`; rustmath-symbolic has no `tests/` directory today. The line numbers are at HEAD; the
implementer recompiles (`cargo check --workspace --tests`) to catch any site added by the units
implemented before S1.

### 3.4 Coordination

I3 (before S1) changes `Integer::to_i64` to `Option` and adds the symbolic test for taylor at 10^30;
S1 adapts that test. J1a/J1b (after S1) should treat the REPL taylor arm, `TAYLOR_ORDER`,
`TAYLOR_WORK` and `test_taylor_order_checked` as already done by S1. Any unit implemented after S1
that adds a by-value match on an owned `Expr` gets E0509 at compile time, which says how to fix it.
J1b's display budget (`display_size`, `OUTPUT_BYTES`) is what bounds the printed text of a taylor
result: `TAYLOR_WORK` bounds the work and the memory, not the text (§2.7, review 2 m5). Its estimate
must count a shared subtree once per occurrence (as Display prints it), which J1b's DAG case already
requires; `(x + sin(2^100000))^1000` at order 1000 (30.4 MB of text from a result of 5.7 MB peak
memory) is a good check for it.

## 4. Tests

Reference values come from PARI/GP 2.15.4 (`/usr/bin/gp`). Helper used for all of them
(`EXP/refs.gp`):

```
L(F, a, n) = my(s = F(a + x + O(x^(n + 20))), v = valuation(s, x)); [v, vector(n - v + 1, i, polcoef(s, v + i - 1, x))];
```

It returns `[valuation, coefficients up to exponent n]` (raise the `20` when the valuation is very
negative, as for `1/(sin(x)^50 - x^50)`, where gp needs `O(x^(n+120))`). Outputs are in
`EXP/refs.txt` and `EXP/gpcheck2.out`. A test helper `fn coeffs(f, a, n) -> SeriesExpansion` calls
`series_expansion` and compares `Expr` values exactly (rational coefficients compare as
`Expr::Integer`/`Expr::Rational`; parameter cases substitute a rational for y and `simplify`;
transcendental cases evaluate with `eval_float` after substituting `pi`, `e` with floats, tolerance
1e-12; coefficients too small for f64, as `sin(1)^4294967296`, are compared as structures).

The porting oracle is `EXP/rev2/gpcheck3.py` (107 cases: the 82 of `EXP/gpcheck2.py`, 9 for review 2
(m1 poles, m3 logs, a huge trig kernel) and review 2's 16 cases of negative intermediate precision
and branch cuts). It runs the prototype and compares every
coefficient with gp in an 80-digit gp session: exactly for rational, parameter and `I` cases, to a
relative 1e-60 for transcendental constants. It maps gp's missing `erf` to `ERF0(z) = 1 - erfc(z)`
and counts any gp error as a failure (review 1, m4); gp's own float artifacts (`sin(Pi)`,
`sqrt(2)*sqrt(2)-2`, `sqrt(8)-2*sqrt(2)`) and gp's refusal of a series exponent (`x^(x-x+2)`) use exact
aliases, as do `log(e^3) - 3` and `log(exp(2)) - 2`, which gp evaluates as floats. The harness
parenthesizes a negative base, which HEAD's Display prints without parentheses. The implementer ports
these cases as tests and can run the harness against a small driver binary of the real crate.
`EXP/rev2/gp/m1.gp` (output `m1.out`) gives the reference values of the m1 tests below with
`L(F, a, n, extra) = my(s = F(a + x + O(x^(n + extra))), v = valuation(s, x), w = vector(n - v + 1,
i, polcoef(s, v + i - 1, x))); [v, #select(t -> t != 0, w), w[1..3], w[#w]]` (valuation, number of
nonzero coefficients up to x^n, the first three, the last).

No unit test asserts a duration (review 1, m8). The budget tests below carry one loose bound,
"returns within 20 s", which guards against a hang only. It also catches the review 2 B2 regression
(revision 1 took 111.7 s on `budget_huge_trig_kernels`; the fix takes 2 ms), so review 2's "within 1
s" is met by the measurement and guarded by the 20 s bound without a timing-sensitive assertion.

### 4.1 Engine (new file `rustmath-symbolic/tests/series_engine.rs`, plus unit tests in the modules)

Each test checks values. Each fails on HEAD: HEAD returns `1/0` forms, fabricated zeros, panics,
runs out of memory, or (for the new entry points) has no such function.

| test | input | expected (gp) |
|---|---|---|
| `exp_at_0` | exp(x), 0, 12 | `L(x->exp(x),0,12)` = [0, [1, 1, 1/2, 1/6, ..., 1/479001600]] |
| `sin_at_0` | sin(x), 0, 15 | [1, [1, 0, -1/6, 0, 1/120, ..., -1/1307674368000]] |
| `tan_at_0` | tan(x), 0, 15 | [1, [1, 0, 1/3, 0, 2/15, 0, 17/315, 0, 62/2835, 0, 1382/155925, 0, 21844/6081075, 0, 929569/638512875]] |
| `tanh_sinh_cosh_at_0` | tanh 9, sinh 7, cosh 6 | [1,[1,0,-1/3,0,2/15,0,-17/315,0,62/2835]], [1,[1,0,1/6,0,1/120,0,1/5040]], [0,[1,0,1/2,0,1/24,0,1/720]] |
| `csc_cot_at_0` (poles) | 1/sin(x), 1/tan(x), 0, 9 | [-1, [1, 0, 1/6, 0, 7/360, 0, 31/15120, 0, 127/604800, 0, 73/3421440]]; [-1, [1, 0, -1/3, 0, -1/45, 0, -2/945, 0, -1/4725, 0, -2/93555]] |
| `bernoulli_gf` | 1/(exp(x)-1), 0, 8 | [-1, [1, -1/2, 1/12, 0, -1/720, 0, 1/30240, 0, -1/1209600, 0]] |
| `bernoulli_h49e3` (m11) | x/(exp(x)-1), 0, 50 | coefficient of x^k is `bernfrac(k)/k!` for k = 0..50: gp `s = x/(exp(x+O(x^60))-1); vector(51,k,polcoef(s,k-1)) == vector(51,k,bernfrac(k-1)/(k-1)!)` is 1 |
| `log_cos`, `exp_sin` | 0, 10 / 0, 8 | [2, [-1/2, 0, -1/12, 0, -1/45, 0, -17/2520, 0, -31/14175]]; [0, [1, 1, 1/2, 0, -1/8, -1/15, -1/240, 1/90, 31/5760]] |
| `removable_singularity` | (sin(x)-x)/x^3, 0, 6; sin(x)/x, 0, 10 | [0, [-1/6, 0, 1/120, 0, -1/5040, 0, 1/362880]]; sin(x)/x from `L(x->sin(x)/x,0,10)` |
| `triple_pole_needs_retry` | 1/(sin(x)-x), 0, 3 | [-3, [-6, 0, -3/10, 0, -11/1400, 0, -17/126000]] |
| `high_valuation_denominator` (m2) | 1/(sin(x)^50 - x^50), 0, 3 | valuation -52, 28 nonzero terms (exponents -52, -50, ..., 2), first ones -3/25, -62/125, -9113/13125; all from gp `F(x)=1/(sin(x)^50-x^50); s=F(x+O(x^124)); vector(56,i,polcoef(s,-53+i))` (exact rationals) |
| `exact_denominator_no_search` | 1/(x^300*(1+x)), 0, 3; 1/(x^3*(1+x)), 0, 3 | [-300, [1, -1, 1, ..., ]] (304 terms, alternating +-1); [-3, [1, -1, 1, -1, 1, -1, 1]] |
| `binomial_series` | sqrt(1+x) 6; (8+x)^(1/3) 4; (1+x)^(-5) 8 | [0,[1,1/2,-1/8,1/16,-5/128,7/256,-21/1024]]; [0,[2,1/12,-1/288,5/20736,-5/248832]]; `L(x->(1+x)^-5,0,8)` |
| `large_integer_powers` | (1+x)^10000, 0, 30; (1+x+x^2)^100, 0, 20; (2+x)^(-70), 0, 10 | `L(x->(1+x)^10000,0,30)` (= binomial(10000, k)); `L(x->(1+x+x^2)^100,0,20)`; `L(x->(2+x)^(-70),0,10)` |
| `large_power_symbolic_lead` | (x+sin(1)+cos(1))^4096, 0, 2 (Miller); (sin(1)^2+cos(1)^2+x)^100, 0, 3 and (sin(1)^2+cos(1)^2-1+x)^100, 0, 3 (binomial, undecided lead) | numeric values of `L(x->(x+sin(1)+cos(1))^4096,0,2)`; `L(x->(1+x)^100,0,3)` = [0,[1,100,4950,161700]] numerically; all coefficients of the last are numerically 0 up to x^3 (it is x^100) |
| `inverse_trig_at_0` | atan, asin 9; asinh, atanh 9 (built with `Expr::arcsinh` etc.) | [1,[1,0,-1/3,0,1/5,0,-1/7,0,1/9]]; [1,[1,0,1/6,0,3/40,0,5/112,0,35/1152]]; [1,[1,0,-1/6,0,3/40,0,-5/112,0,35/1152]]; [1,[1,0,1/3,0,1/5,0,1/7,0,1/9]] |
| `acos_at_0_has_pi_over_2` | acos(x), 0, 3 | c0 = pi/2 exactly (`Expr` check), then [-1, 0, -1/6] |
| `x_pow_x_at_1` | x^x, 1, 5 | [0, [1, 1, 1, 1/2, 1/3, 1/12]] |
| `exact_constant_exponent` (m1) | x^(x-x+2), 0, 3 | `x^2` (gp refuses the series exponent; exact reference) |
| `zero_base` (m1) | 0^x, 1, 3 | exact 0 |
| `removable_at_1`, `pole_free_at_2` | (x-1)/(x^2-1) at 1, 4; 1/(1-x) at 2, 4 | [0,[1/2,-1/4,1/8,-1/16,1/32]]; [0,[-1,1,-1,1,-1]] |
| `generic_parameter` | 1/(y+x), 0, 4; x^y at 1, 3; exp(y*x)/(1-x) 0, 5 | [0,[1/y,-1/y^2,1/y^3,-1/y^4,1/y^5]]; [0,[1,y,1/2*y^2-1/2*y,1/6*y^3-1/2*y^2+1/3*y]]; `L(x->exp(y*x)/(1-x),0,5)` (compare after y = 3/7) |
| `imaginary_unit` | exp(I*x), 0, 6; 1/(x^2+1) at I, 3 | [0,[1,I,-1/2,-1/6*I,1/24,1/120*I,-1/720]]; [-1,[-1/2*I,1/4,1/8*I,-1/16,-1/32*I]] |
| `complex_constant_decided` (B1) | 1/(x+sin(1)+I), 0, 2; 1/(x+pi+I), 0, 4 | `L(x->1/(x+sin(1)+I),0,2)` = [0, [0.49264333476860154590 - 0.58545492793321861568*I, 0.10006001734936179013 + 0.57684093610746422169*I, -0.38700826930171954901 - 0.22559621214875578048*I]]; `L(x->1/(x+Pi+I),0,4)` |
| `exact_pi_points` | 1/sin(x) at pi, 5; tan(x) at pi/2, 5; sin(x) at pi/4, 2 | `L(x->-1/sin(x),0,5)` = [-1,[-1,0,-1/6,0,-7/360,0,-31/15120]]; `L(x->-1/tan(x),0,5)` = [-1,[-1,0,1/3,0,1/45,0,2/945]]; c = [sqrt(2)/2, sqrt(2)/2, -sqrt(2)/4] exactly (gp numeric `L(x->sin(x),Pi/4,2)`). (gp's own floating Pi gives 5.3e38 here, which is why exact pi matters) |
| `radical_reduction` (m10) | 1/(x+sqrt(8)-2*sqrt(2)), 0, 2; zero_status of sqrt(8) - 2*sqrt(2) | `1/x`; `Zero` |
| `asymptotic_rational` | (x^2+x)/(x^2+1) at oo, 5 | `subst((x^2+x)/(x^2+1),x,1/x)+O(x^6)` = 1 + x - x^2 - x^3 + x^4 + x^5: `1 + 1/x - 1/x^2 - 1/x^3 + 1/x^4 + 1/x^5` |
| `asymptotic_polynomial` | x^2 at oo, 3 | `x^2` (valuation -2) |
| `transcendental_coefficients` | exp(x) at 1, 4; 1/(2+sin(x)) at 1, 3; acosh(x) at 2, 3; erf(x) at 0, 7 | `L(x->exp(x),1,4)`; [0.35193039286572446133, -0.066919142861945019958, 0.064834789907861362567, -0.011083758395192097310]; [1.3169578969248167086, 0.57735026918962576451, -0.19245008972987525484, 0.096225044864937627418]; `2/sqrt(Pi)*intformal(exp(-x^2+O(x^9)))` = (2/sqrt(pi)) (x - x^3/3 + x^5/10 - x^7/42) |
| `big_point_s7` | x^2 at 10^30, 3 | [0, [10^60, 2*10^30, 1]] (gp `L(x->x^2,10^30,3)`); fails on HEAD (panic) |
| `symbolic_point` | exp(x) at a (symbol), 2 | `exp(a) + exp(a)*(x - a) + 1/2*exp(a)*(x - a)^2` (structure check) |
| `huge_exponents_b2` (debug and release) | (sin(1)*x)^4294967296 at 1, 1; (sin(1)*x)^3000000000 at 1, 1; x + ((sin(1)^4096)^4096)^4096 at 0, 1 | structures `[sin(1)^4294967296, 4294967296*sin(1)^4294967296]`, `[sin(1)^3000000000, 3000000000*sin(1)^3000000000]`, `[sin(1)^68719476736, 1]` (gp agrees numerically: `exponent(sin(1)^4294967296)` = -1069509430, `gpcheck2`); HEAD returns them unevaluated, revision 0 returned `1 + 4294967296*(x - 1)`, `1/sin(1)^1294967296 + ..`, `1 + x` |
| `exponent_overflow` | (sin(1)^4*x)^(2^62) at 1, 1; x + sin(1)^(2^62)*sin(1)^(2^62) at 0, 1; (x^(-2^62))^4 at 0, 1 | the first is `Ok` with the kernel `(sin(1)^4)^4611686018427387904` (the exponent product would overflow); the second is `Unsupported` (kernel exponent in a monomial product), the third `Unsupported` (series exponent); no panic in debug |
| `sparse_exact` (B3b) | x^(-1000000000)+1 at 0, 1 | terms `[(-1000000000, 1), (0, 1)]` (HEAD: garbage; revision 0: 24 GB) |
| `high_order_poles` (review 2, m1; debug-sized) | 1/(x+x^2)^200 at 0, order 2; 1/sin(x)^140 at 0, order 0 | `m1.out`: [-200, 203, [1, -200, 20100], 203359206710512721649984375195710539823858829997596852086963562829147609878495814974301634417563537134018932503818612000] (the coefficient of x^k is (-1)^(k+200) binomial(k+399, 199)); [-140, 71, [1, 0, 70/3], last as printed by `m1.out`]; every coefficient compared with gp. Revision 1: `PrecisionExhausted(131)` and `PrecisionExhausted(129)`. Debug: 0.15 s and 3.8 s |
| `high_order_poles_release` (review 2, m1; `#[cfg_attr(debug_assertions, ignore = "15 to 30 s in debug")]`, run by the `--release` acceptance command) | 1/sin(x)^200, 1/(exp(x)-1)^150, 2/(3*sin(x)^200), 1/(sin(x)^200+x^300), all at 0, order 2 | `m1.out`: [-200, 102, [1, 0, 100/3], ..]; [-150, 153, [1, -75, 11225/4], ..]; [-200, 102, [2/3, 0, 200/9], ..]; [-200, 102, [1, 0, 100/3], ..], last coefficients as printed; every coefficient compared with gp (`gpcheck3`). Revision 1: `PrecisionExhausted(131)`. Release: 1.3 s, 2.3 s, 1.4 s, 1.4 s |
| `exact_logs` (review 2, m3) | 1/(x+log(e^3)-3), 1/(x+log(exp(2))-2), 1/(x+log(2*e)-1), x+log(e), all at 0, order 2 | terms `[(-1, 1)]` (exactly 1/x) twice; the series of 1/(x + log(2)): c_k = (-1)^k log(2)^(-k-1), gp `L(x->1/(x+log(2)),0,2)`; `[(0, 1), (1, 1)]`. Revision 1: `UndecidedZero(-3 + log(e^3))` |

Errors (each a `matches!` on the variant):

| test | input | expected |
|---|---|---|
| `log_singularity` | log(x) at 0 | `NotLaurent` |
| `branch_point` | sqrt(x) at 0, x^(1/2), sqrt(x^2) at 0, asin(x) at 1, x^pi at 0, x^y at 0, x^(1/2+I) at 0 | `NotLaurent` |
| `subexpression_not_laurent` | sqrt(x)^2 at 0, exp(log(x)) at 0 | `NotLaurent` (documented: a subexpression) |
| `essential_singularity` | exp(1/x) at 0, atan(1/x) at 0 | `NotLaurent`, message contains "of an argument with a pole" |
| `abs_at_zero` | abs(x) at 0 (`abs(x-2)` at 0 is `2 - x`) | `NotLaurent`, message contains "zero of odd order" |
| `unsupported_functions` | gamma(x) at 1, zeta(x) at 2, bessel_j(0, x) at 0 | `Unsupported` (HEAD: `gamma(1)`, `zeta(2)`, 0) |
| `identically_zero_denominator` | 1/(x - x) | `DivisionByZero` |
| `undecidable_zero_denominator` | 1/(sin(x)^2 + cos(x)^2 - 1) at 0, order 2 | `PrecisionExhausted` (bounded search; 0.5 s debug) |
| `undecided_constant` | 1/(x + sin(1)^2 + cos(1)^2 - 1) at 0 | `UndecidedZero(c)` with c the canonical constant |
| `complex_constant_undecided` (B1) | 1/(x + sqrt(-4) - 2*I), 1/(x + exp(I*pi/2) - I), 1/(x + sqrt(-1) - I), 1/(x + log(-1) - I*pi), all at 0 | `UndecidedZero` (all four are 0; revision 0's text rule would have called the first two nonzero) |
| `undecided_exponent` (m1) | x^(sin(1)^2 + cos(1)^2) at 0 | `UndecidedZero` |
| `undecided_constant_base` (m5) | (sin(1)^2 + cos(1)^2 - 1)^x at 1 | `UndecidedZero` |
| `numeric_zero_test_decides` | 1/(2 + sin(x)) at 1 | `Ok` (ball test) |
| `undecided_after_underflow` (review 2, B1) | 1/(x + (1/2)^1073741827*2^1073741822 - 1/32) at 0, order 1; 1/(x + exp(-744261120)*exp(744261117) - exp(-3)) at 0, order 2; 1/(x + exp(744261120)*exp(-744261117) - exp(3)) (overflow); 1/(x + (1/3)^536870912*3^536870912 - 1); 1/(x + 2^(-300000000)*2^300000000 - 1) | `UndecidedZero` for each (all five constants are exactly 0, so the function is 1/x; gp at 77 digits gives the second constant as 5.4e-79, review 2 `underflow.gp`). Revision 1 returned a finite series of 1/0 terms for the first two (the other three were already `UndecidedZero` there, through its finiteness checks and radius growth; they guard the new checks). 2 to 13 ms debug |
| `non_integer_huge_midpoint` | x^(2^100000000*sin(1)) at 0, order 2 | `UndecidedZero(2^100000000*sin(1))` (no nearest integer at any precision; 2 ms debug) |
| `log_of_negative_not_folded` (m3) | 1/(x + log(-e) - 1) at 0 | `UndecidedZero` (log(-e) = 1 + I pi is not folded; the constant has no enclosure) |
| `point_contains_variable` | point = x + 1 | `InvalidPoint` |
| `laurent_min_order` | laurent(1/x^2, x, 0, -1, 2) | `ValuationBelowMinimum { valuation: -2, min_order: -1 }` |
| `series_coefficients_pole` | series_coefficients(1/x, x, 0, 3) | `Pole { valuation: -1 }` |

Budget (each with `taylor_with_budget`, debug, outcome asserted, plus "returns within 20 s"):

| test | input, budget | expected | prototype, debug |
|---|---|---|---|
| `budget_constant_powers` (B3a) | x+(sin(1)+cos(1)+exp(1))^50, x+(sin(1)+cos(1)+exp(1))^100, x+(sin(1)+cos(1))^4096 at 0, order 1, budget 1e6 | `Ok`, the power kept as a kernel (`gpcheck2` checks the values) | 2 ms each (revision 0: 13.8 s, over 120 s, over 30 s) |
| `budget_sparse` (B3b) | x^(-1000000000)+1 at 0, order 1, budget 1e6 | `Ok(1/x^1000000000 + 1)` | 1 ms, 4 MB |
| `budget_multinomial` | (x+sin(1)+cos(1)+exp(1)+pi)^64 at 0, order 1, budget 1e6 | `BudgetExceeded` | 19 ms |
| `budget_product_precharge` | exp(x)*sin(x) at 0, order 2000, budget 1e8 | `BudgetExceeded` (the product alone is charged 5.2e9) | 0.18 s |
| `budget_recurrence` | exp(sin(x)) at 0, order 300, budget 1e6 | `BudgetExceeded` | 0.9 s |
| `budget_huge_trig_kernels` (review 2, B2) | at 0, order 2, budget 1e8: 1/(x + sin(2^100000000) + 2); 1/(x + cos(2^100000000) + 2); 1/(x + tan(2^100000000) + 2); 1/(x + sin(2^1000000000) + 2); x + (sin(2^100000000) + 3)^1000000000 (order 1, the `pow_const` kernel fallback) | `Ok` with c0 = `1/(2 + sin(2^100000000))` (structure); `Ok`; `UndecidedZero`; `UndecidedZero` (the kernel's ball leaves the window); `Ok` with the kernel `(3 + sin(2^100000000))^1000000000` | 2 ms each (revision 1: 111.7 s release for the first, over 30 s for the fourth) |
| `budget_kernel_keys` (review 2, m2) | x + sin(2^340000+1) + ... + sin(2^340000+40) at 0, order 1, budget 1e6, then budget 2e6 | `BudgetExceeded` with 1e6 (the prototype charges 1.28e6, of which about 4.3e5 for the keys; without the key charge 8.5e5 would pass); `Ok` with 2e6. With the final model, put the low budget between the total without the key charge and the total | 15 ms release (revision 1: 504 ms) |

Zero test (unit tests in `series_const.rs`): `sqrt(2)*sqrt(2) - 2` is `Zero`; `sqrt(8) - 2*sqrt(2)`
is `Zero`; `I^2 + 1` is `Zero`; `pi^2 - 10` is `NonZero` (single named constant); `pi + I` is
`NonZero` (rule 4); `y + sin(1)` is `NonZero` (indeterminate group); `sin(1)^2 + cos(1)^2 - 1` is
`Unknown`; `2 + sin(1)` is `NonZero` (numeric); `sin(1) + I` is `NonZero` (both parts enclosed);
`sqrt(-4) - 2*I`, `exp(I*pi/2) - I` are `Unknown`; `sqrt(6) - sqrt(2)*sqrt(3)` is `Unknown`
(documented incompleteness); `sin(pi)` folds to 0; `exp(pi) - pi` is `NonZero` (numeric);
`log(-1)` has no enclosure. `non_integer_status`: `pi`, `y`, `1/2 + I`, `sqrt(2)` are decided
non-integers; `sin(1)^2 + cos(1)^2` is `Unknown` with nearest integer 1; `2^100000000 * sin(1)` is
`Unknown` without one.

Ball unit tests (in `series_ball.rs`; review 2, B1 and B2): `enclose_real` is `None` for
`(1/2)^1073741827`, `exp(-744261120)`, `exp(744261120)`, `2^(2^29)` (outside the window) and for
`tan(2^100000000)`; it is `[0 ± 1]` for `sin(2^100000000)` and `cos(2^100000000)`; R2 accepts the
zero midpoints of `sin(0)` (the exact ball [0, 0]) and `log(1)` (a ball around 0 of radius about
2^-p); `2 + sin(2^100000000)` excludes 0.
`ball_cost(sin(2^100000000), 64, u64::MAX)` is at least the trig unit `((2*64 + 64)/64 + 1)^2 * 8 =
128`, and `ball_cost` of a long sum returns as soon as it passes its limit.

Kernel unit tests (in `series_const.rs`; review 2, m2): creating `sin(2^340000 + 1)` charges at least
`key_size / 8` units (about 10^4) before the key exists; the key contains the integer in hexadecimal;
creating it twice in one `Work` gives the same kernel (same id); `sin(1)` and `sin(2)` compare
unequal; `Real(2.0)` and `Integer(2)` inside a kernel give different keys.

Large orders (unit tests; run on the default 2 MB test thread, debug):

| test | input | expected | HEAD |
|---|---|---|---|
| `taylor_order_20000_polynomials` | x^2, 2*x+1, x at order 20000 | `x^2`, `1 + 2*x`, `x` | x^2: 1.92 GB at order 9; x: stack overflow |
| `taylor_exp_order_2000_depth` | exp(x), 0, 2000 | coefficient of x^2000 is `1/2000!` (gp `1/2000!`); tree depth of the result <= 16; `format!("{}")` succeeds | out of memory |
| `simplify_long_left_deep_sum` | `1 + 1 + ... + 1` (100000 terms, built in a loop) simplified in a thread with a 1 MiB stack; the input is dropped normally (no `mem::forget`) | `Integer(100000)` | stack overflow |
| `drop_left_deep_sum_1e6` (B4) | a left-deep sum of 10^6 terms `x^(k mod 5)`, built and dropped in a thread with a 1 MiB stack | the thread joins without panic | stack overflow |
| `drop_other_shapes` (B4) | right-deep power tower, unary chain, function chain, zig-zag, each 10^5 deep, 1 MiB thread; a shared deep subtree dropped one owner at a time | joins without panic | stack overflow |

### 4.2 Existing tests to rewrite (`series.rs:795-980`, `lib.rs`)

- `test_taylor_polynomial`: `Ok(x^2)`.
- `test_maclaurin_exp`, `test_unified_series_function`: compare the coefficient list with gp
  (structural equality with `known_series::exp` no longer holds; known_series builds
  `x^2 / 2`, the engine `1/2 * x^2`).
- `test_series_coefficients`, `test_calculus_series_coefficients`: `[0, 0, 1, 0]` exactly (the old
  test never checked `coeffs[2]`).
- `test_taylor_around_nonzero`, `test_series_at_nonzero_point`: `1 + 2*(x-1) + (x-1)^2`.
- `test_laurent_positive_orders`: `laurent(exp(x), 0, 0, 3) == taylor(exp(x), 0, 3)`.
- `test_series_with_big_o`: BigO is `O(x^4)`.
- `test_asymptotic_expansion`: `1 + 1/x - 1/x^2 + O(..)` coefficients (the old test only checked
  "not constant").
- `lib.rs` tests: `.unwrap()`, assertions unchanged.

### 4.3 Parser and Display (`parser.rs` tests)

| test | input | expected | HEAD |
|---|---|---|---|
| `test_parse_big_integer` | `x + 18446744073709551616`, `-9223372036854775808`, `123456789012345678901234567890` | Integer values (`Integer::from_decimal_str`) | parse error |
| `test_parse_division_by_non_literal` | `1/x`, `1/sin(x)`, `exp(1/x)`, `1/(y+x)` | `Div(1, ..)` | parse error |
| `test_parse_rational_precedence` | `x^2/4`, `2^3/4`, `2/3^2`, `3/4^2`, `x*2/3^2`, `x/2/3`, `2^-1/3` | `Div(Pow(x,2),4)`; simplify gives 2, 2/9, 3/16, `x*2/9`, `x/6`, 1/6 (gp: `2^3/4`, `2/3^2`, `3/4^2`, `2^-1/3`) | x^(1/2), 2^(3/4), 4/9, 9/16, 4x/9, x/(2/3), 2^(-1/3) |
| `test_parse_signed_exponent` | `2^-1`, `x^-2`, `2^-x` | `Pow(2, Integer(-1))`, `Pow(x, Integer(-2))`, `Pow(2, Neg(x))` | third: error |
| `test_parse_literal_fold` | `1/3`, `6/4`, `1/-3`, `-2/3`, `1/0` | `Rational(1/3)`, `Rational(3/2)`, `Rational(-1/3)`, `Rational(-2/3)`, error | (as today except `-2/3` structure) |
| `test_parse_inverse_hyperbolic` | `asinh(x)`, `arccosh(x)`, `atanh(x)` | `Unary(Arcsinh, x)` etc. | `Function` |
| `test_display_parse_round_trip` | `Pow(x, 1/2)`, `Pow(3/4, -1/2)`, `Div(x, 1/2)`, `Pow(-2, 2)`, `Pow(Neg(y), 1/2)`, `Pow(Real(-1.5), 2)`, the engine output of `sqrt(x - y)` at 0 (review 1, m3), every engine output of §4.1 | `parse(format!("{e}"))` simplifies to the same value as `e` | `(x ^ 1/2)` re-parses as `(x^1)/2` under the new parser; `(-y ^ (1/2))` as `-(y^(1/2))` |

Existing parser tests stay green (checked against the grammar: `-17` is `Neg(17)`, `-1.5` is
`Neg(Real(1.5))`, `1.5e-3`, `2 ^ 3 ^ 2` right-associative, error cases unchanged). Their 21 by-value
matches are rewritten for E0509 (§3.3) without changing what they assert.

### 4.4 Limits

`limit_via_series`: `(1 - cos(x))/x^2` at 0 is `Finite(1/2)`; `(exp(x) - 1 - x)/x^2` at 0 is
`Finite(1/2)`; `x^3/x` at 0 is `Finite(0)`; `x*log(x)` is `Unknown`; `1/x` is `Unknown`
(gp: `limit` values from `L(..)[2][1]`). Existing limit tests stay green.

### 4.5 REPL (`rustmath-jupyter/src/repl.rs` tests)

- `test_taylor_order_checked` unchanged (it reads `TAYLOR_ORDER`).
- new `test_taylor_engine`: `taylor("1/sin(x)", x, 0, 3)` is `Ok` with the Display string
  `(((1 / x) + (1/6 * x)) + (7/360 * (x ^ 3)))` (HEAD: the argument does not parse);
  `taylor("log(x)", x, 0, 3)` is a ComputationError containing "logarithmic singularity";
  `taylor("gamma(x)", x, 1, 3)` is a ComputationError; `taylor("exp(x)*sin(x)", x, 0, 2000)` is a
  ComputationError (budget: the product is pre-charged at 5.2e9 units, far above any plausible
  `TAYLOR_WORK`; refused after 0.18 s in the debug prototype; review 1, m6);
  `taylor("x^2/4", x, 0, 2)` is `(1/4 * (x ^ 2))`; `taylor("1/(x-x)", x, 0, 2)` is a ValueError;
  `taylor("1/(x + sin(2^100000000) + 2)", x, 0, 2)` is `Ok` (review 2, B2: 2 ms in the debug
  prototype under the REPL budget; revision 1 ran 111.7 s in release).
  Do not put `exp(sin(x))` at a high order in a debug test: reaching the budget takes about 4 s in
  release and much longer in debug.

## 5. Cost: before and after

### 5.1 The four functions of the item (point 0)

Debug (revision 1 prototype `bin/proto2`, budget 1e8): x^2 and 2*x+1 take 36 us and 47 us at order
20000 (HEAD: 1.92 GB at order 9 for x^2, 1.18 GB at order 20 for 2*x+1); exp(x) 0.13 ms at 12 (HEAD
12.6 s, 4.64 GB) and 32 ms at 2000; sin(x) 0.13 ms at 12 (HEAD 14.1 s, 5.23 GB). Revision 0 debug
(`EXP/after_debug*.txt`): exp(x) 0.17 s at 5000, 2.4 s at 20000; sin(x) 2.0 s at 20000 (806 MB, the
coefficients themselves). Revision 2 debug (`EXP/rev2/after_debug_r2.txt`, `bin/proto3`, budget
1e8): x^2 and 2*x+1 80 us and 92 us at 20000; exp(x) 0.24 ms at 12 and 39 ms at 2000; sin(x) 0.23 ms
at 12 and 37 ms at 2000 (the same work units as revision 1; the small differences are noise of a
shared machine).

Release (`EXP/after_release.txt`, revision 0; the sparse revision 1 engine is within noise of it for
these inputs: exp(x) at 2000 takes 14.7 ms, sin(x) 14.1 ms, `EXP/calib_rev1.txt`):

| f | n | engine | to_expr | Display | output length | peak |
|---|---|---|---|---|---|---|
| x^2 | 9 / 20000 | 19 us / 7 us | | | 7 | 3.1 MB |
| 2*x+1 | 20000 | 10 us | | | 13 | 3.2 MB |
| exp(x) | 12 | 26 us | 7 us | 8 us | 268 | 3.2 MB |
| exp(x) | 1000 | 5.2 ms | 0.6 ms | 16 ms | 1.2 MB | 7.4 MB |
| exp(x) | 2000 | 13 ms | 1.7 ms | 83 ms | 5.3 MB | 17 MB |
| exp(x) | 5000 | 77 ms | 8 ms | 0.94 s | 38 MB | 86 MB |
| exp(x) | 20000 | 1.41 s | 78 ms | 39 s | 730 MB | 1.43 GB |
| sin(x) | 1000 / 5000 / 20000 | 4.3 ms / 73 ms / 1.25 s | | 7 ms / 0.47 s / 20 s | 0.6 / 19 / 365 MB | 6.8 MB / 65 MB / 1.0 GB |

The output size of exp(x) is inherent (sum of log10 k! digits for k <= n), and Display of big
integers dominates past order 2000. Hence `TAYLOR_ORDER = 2000` for the plain series. The order is
not an output bound in general (review 2, m5): other admitted inputs print far more at a lower order
(§2.7, §3.1), and the REPL display budget handles them.

Output depth: 18 at order 20000, so Display, simplify, Drop, substitute and PartialEq of the result
are all fine on 2 MB threads.

### 5.2 General functions (release, point 0)

| f | n | engine | work units (first model, no base charge) | gp (same series) |
|---|---|---|---|---|
| exp(sin(x)) | 100 / 200 / 300 | 0.06-0.11 s / 0.9-1.05 s / 4.0-4.2 s | 8.5e7 at 300 | 0.18 s at 300, 10.5 s at 1000 |
| tan(x) | 100 / 200 / 300 | 0.08 s / 1.0-1.2 s / 4.5-4.8 s | 1.1e8 at 300 | 15 ms at 300, 0.27 s at 1000 |
| 1/(exp(x)-1) | 200 | 1.04 s | 2.5e7 | |
| log(cos(x)) | 200 | 0.53-0.58 s | 8.1e6 | |
| sqrt(1+x+x^2) | 600 | 0.17 s | 1.2e6 | |
| exp(x)*sin(x) | 500 / 1000 | 2.7 s / 41 s | 4.1e8 at 500 | |
| (1+x)^10000 | 2000 | revision 0: 12.9 s; revision 1 (Miller): 11.5 ms | 6.4e5 | |
| 1/(1+x+x^2), atan(x) | 2000 | 3-4 ms | | |

Dense rational series cost O(n^2) operations on coefficients of O(n log n) bits; the gcd in every
rational addition dominates. gp is 20 to 300 times faster (GMP, fast algorithms). Faster series
arithmetic (Newton iteration for inverse/exp/log with fast multiplication, common-denominator
integer arithmetic) is a performance follow-up, not a correctness defect.

### 5.3 REPL budget calibration

Rates with the cost model of §2.7 (release, revision 1 prototype,
`BUDGET=75000000 bin/proto2_rel measure f 0 n`, `EXP/calib_rev1.txt`):

| f | n | result | units | time | units/s |
|---|---|---|---|---|---|
| exp(x) | 2000 | ok | 1.46e6 | 15 ms | 9.9e7 |
| sin(x) | 2000 | ok | 1.22e6 | 14 ms | 8.6e7 |
| exp(sin(x)) | 250 | ok | 3.75e7 | 2.17 s | 1.73e7 |
| tan(x) | 250 | ok | 4.93e7 | 2.52 s | 1.96e7 |
| sqrt(1+x+x^2) | 2000 | ok | 4.27e7 | 2.95 s | 1.45e7 |
| exp(x)*sin(x) | 300 | ok | 4.40e7 | 0.44 s | 9.9e7 |
| (1+x)^10000 | 2000 | ok | 6.4e5 | 12 ms | 5.5e7 |
| exp(sin(x)) | 2000 | refused | 7.5e7 | 3.88 s | 1.93e7 |
| tan(x) | 2000 | refused | 7.5e7 | 3.39 s | 2.21e7 |
| log(cos(x)) | 2000 | refused | 7.5e7 | 4.08 s | 1.84e7 |
| exp(x)*sin(x) | 2000 | refused at once (one charge of 5.2e9) | | 45 ms | |

The slowest rate is 1.45e7 units/s. Provisional `TAYLOR_WORK = 100_000_000` (about 7 s at that rate;
a refused call stops after at most about 7 s). The implementer reruns this table with the final code
(release) and sets `TAYLOR_WORK` to about 10 s times the slowest rate, rounded down, recording the
table in the doc comment (X2 policy: admitted calls at most about 10 s in release). The model
over-charges integer-only and product-heavy inputs; that errs towards refusing, with an honest
`BudgetExceeded`.

Revision 2 recheck (release, `EXP/rev2/calib_r2.txt`, measured while a gp job ran on another core):
exp(x) at 2000 1.46e6 units in 21 ms; sin(x) at 2000 1.22e6 in 17 ms; exp(sin(x)) at 250 3.75e7 in
2.27 s (1.65e7/s); tan(x) at 250 4.93e7 in 2.69 s (1.83e7/s); sqrt(1+x+x^2) at 2000 4.27e7 in 3.34 s
(1.28e7/s); (1+x)^10000 at 2000 6.4e5 in 11 ms; exp(sin(x)) at 2000 refused at 7.5e7 after 3.99 s.
The units are unchanged by revision 2 for these inputs (no kernel, no numeric test), and the rates
are within the noise of the table. New rates of review 2 inputs: the 40 `sin(2^340000 + k)` kernels
8.6e7 units/s (revision 1: 1.7e6/s, the uncharged Display keys); `1/(x + sin(2^100000000) + 2)` 2294
units in 0.13 ms (0.34 ms debug; revision 1: 111.7 s); `1/sin(x)^200` at order 2 1.96e7 units in 1.3 s
(1.5e7/s); `1/(exp(x)-1)^150` 3.37e7 in 2.3 s (1.46e7/s). The slowest rate stays near 1.3e7 to
1.45e7 units/s, so the provisional `TAYLOR_WORK = 100_000_000` stays at about 7 to 8 s.

### 5.4 Iterative Drop

See the table of §2.14: no measurable cost on ordinary workloads (+7 % instructions, wall time
within noise), and deep trees of any shape are dropped in bounded stack. A deep drop in the
iterative branch costs about the same as the glue (3*10^4-term sum: 3.8 ms vs 4.4 ms release).

## 6. Plan

Implementation order (each step builds and tests green; `cargo test -p rustmath-symbolic` after
each step, plus the crates named in the step):

1. `rustmath-symbolic/Cargo.toml`: add `rustmath-reals`.
2. `src/series_ball.rs` (new): `Ball`, `enclose_real(&Expr, prec) -> Option<Ball>` and
   `ball_cost(&Expr, prec, limit) -> u64` on `Expr` only, with the window R1, the exact-zero rule
   R2, the exp/sinh/cosh guard and the sin/cos guard of §2.3; powers of two by shifts.
   Unit tests: `2 + sin(1)`, `exp(pi) - pi`, `sin(1) + 1/3` enclosed away from 0;
   `sin(1)^2 + cos(1)^2 - 1` encloses 0; `log(-1)`, `(-4)^(1/2)`, `I`, `y`, `gamma(1/3)` have no
   enclosure; the ball unit tests of §4.1 (review 2). Port from `EXP/rev2/proto3/ball.rs` (its
   `enclose`, `proves_nonzero_ab`, `non_integer`, `floor_small` move to step 3, which owns the class
   checks and the charging; `enclose` charges `ball_cost` first).
3. `src/series_const.rs` (new): `Work`, `SeriesConst`, `Kernel`, classes, canonical reductions with
   checked `i64` exponents, `from_expr`, `apply`, `pow_const` with the size caps, d-th-power-free
   radicals, `inverse`, `zero_status` with the A + B I numeric rule, `non_integer_status`, `to_expr`,
   `balanced_sum`; every operation takes `&mut Work` and returns `Result`. Kernel keys by the linear
   prefix code, charged before they are built, and interned in `Work` (ids, O(1) comparisons);
   `log_exact`. Unit tests of §4.1 (zero test, non-integer status, kernel keys). Port from
   `EXP/rev2/proto3/konst.rs`, replacing its thread-local counter, fault flag and kernel table (a
   prototype emulation) by the explicit `&mut Work` and `?`.
4. `src/series_engine.rs` (new): sparse `Laurent`, `trunc` (keeping the first term) and
   `trunc_hard`, operations (the `mul` cutoff of §2.1), integer power paths (the leading term of step
   5, binary, Miller, binomial), `expand`, the `expansion` driver, rendering. Port from
   `EXP/rev2/proto3/ser.rs` (drop its `S1TRACE` print). Unit tests that build expressions with the
   `Expr` API (the parser cannot yet read `1/sin(x)`).
5. One step, because `taylor` changes its signature: `src/series.rs` (`SeriesError`,
   `ExpansionPoint`, sparse `SeriesExpansion`, the new public functions; delete the derivative loops
   `:24-58`, `:148-169` and the facade `laurent` body `:89-119`; `known_series` with
   `balanced_sum`; rewrite the tests of §4.2; adapt I3's 10^30 test as `big_point_s7`), `lib.rs`
   exports and test unwraps, `src/limits.rs` (§2.13, §4.4 tests), and the REPL taylor arm
   (`rustmath-jupyter/src/repl.rs:1943-1971`: `taylor_with_budget`, the error mapping of §3.1,
   `match &args[3].simplify()`, a provisional `TAYLOR_WORK = 100_000_000`).
   Then `cargo test -p rustmath-jupyter`.
6. `src/simplify.rs`: §2.10 and its test (with the input still forgotten until step 7 lands, or run
   the test from step 7 on).
7. One step, because `Drop` breaks compilation of the E0509 sites: `src/expression.rs` (Display
   §2.11, `Drop for Expr` §2.14 with its unit tests `drop_left_deep_sum_1e6`, `drop_other_shapes`);
   the 4 library sites and 98 test sites of §3.3 (`rustmath-symbolic`, `rustmath-calculus`,
   `rustmath-geometry`; the jupyter site is already gone in step 5); remove `mem::forget` from
   `simplify_long_left_deep_sum`. Then `cargo check --workspace --tests`.
8. `src/parser.rs`: §2.12 and §4.3 tests.
9. `tests/series_engine.rs` (new): §4.1 value, error and budget tests (they use the new parser);
   also run under `--release` (review 1 B2 structures, and `high_order_poles_release`, ignored in
   debug).
10. `rustmath-jupyter/src/repl.rs`: `TAYLOR_ORDER = 2000`, calibrate `TAYLOR_WORK` in release (§5.3),
    doc comments of both constants with what each bounds (§3.1, review 2 m5), §4.5 tests.
11. Docs: `PROJECT_TRACKER.md` rows :2163 (Taylor: truncated Laurent series arithmetic over symbolic
    constants, 🟢, file:line), :2109 (Laurent: 🛑 to 🟢), :2118 (asymptotic), :2162 (parser: bignum
    literals, `a/b` precedence fixed), :2106 (drop "Contaminates taylor(), series()"), the Expr row
    (iterative Drop); new rows for the out-of-scope items of §7; `docs/HANDOFF.md`: remove F1 and the
    symbolic part of S7, add §7's new items. `docs/magma/`: no row changes (the ch. 49 intrinsics map
    to rustmath-powerseries; the symbolic engine is not a MAGMA intrinsic).

Acceptance:

```
cargo test -p rustmath-symbolic --no-fail-fast
cargo test -p rustmath-symbolic --release --test series_engine      # B2 structures, high-order poles
cargo test -p rustmath-jupyter --no-fail-fast
cargo test -p rustmath-calculus -p rustmath-geometry -p rustmath-functions -p rustmath-manifolds -p rustmath-curves -p rustmath-plot3d --no-fail-fast
cargo check -p rustmath-benchmarks
cargo check --workspace --tests
cargo clippy -p rustmath-symbolic
python3 EXP/rev2/gpcheck3.py     # oracle (107 cases), against a driver built from the final crate if wanted
python3 EXP/revchecks.py; python3 EXP/rev2/revchecks3.py   # PROTO=<driver>: errors, structures, budget timings
```

Risks:

- The engine is about 2100 lines in the prototype. It passes 107 gp cross-checks
  (`python3 EXP/rev2/gpcheck3.py`, exact for rational, parameter and `I` cases, 1e-60 relative for
  transcendental constants), the 26 checks of `EXP/revchecks.py` (plus its 10 budget timings) and
  the 18 checks of `EXP/rev2/revchecks3.py`; keep all three as the oracle while porting.
- The prototype emulates `Result` threading with a thread-local fault flag. The port must charge at
  the same points; a missed charge is a budget hole. A grep for constant arithmetic calls that do not
  pass `w` finds them, since every operation takes `&mut Work`.
- Kernel keys are the prefix code of §2.2, injective by construction, so two different kernel
  expressions never merge. The interning table lives as long as one expansion; a kernel id must
  never be compared across two `Work`s (the API converts every constant to `Expr` before it
  returns, so none escapes).
- Ball soundness rests on R1 and R2 of §2.3 being applied to every MPFR value. Any new ball
  operation must pass its results through the same checked constructor; a raw `RealMPFR` operation
  whose result is not checked is a soundness hole (B1 of review 2). A grep for `RealMPFR` arithmetic
  outside the checked helpers finds them.
- Zero-test soundness rests on the rule list of §2.2. A reviewer should check each `NonZero` rule
  against its proof; `Opaque` is the safe default for anything new; a constant with a possibly
  non-real kernel never gets a numeric `NonZero`.
- Generic parameters: a result for `1/(x + a - b)` is valid for a != b only. This is the documented
  convention (gp, Maxima, the crate's Risch module), stated in every doc comment.
- The stack-gated `Drop` relies on stack addresses decreasing with depth; elsewhere it degrades to
  the HEAD behaviour. Units implemented after S1 must match on references (E0509 tells them).
- Display/parse: §2.11 changes printed forms of `Pow` with rational or negative operands; grep shows
  no test depends on them.
- REPL: lowering `TAYLOR_ORDER` to 2000 changes the error text of `taylor(.., 20001)` style calls
  (the test reads the constant).
- Build: `rustmath-symbolic` now links MPFR through rustmath-reals (already built in the workspace).

## 7. Out of scope (with honest interim behaviour)

1. **`rustmath-calculus/src/taylor.rs:35`** duplicates the old algorithm (same exponential growth) and
   overflows its i64 factorial at order 21: `taylor(5*x, "x", 0, 25)` panics with "attempt to multiply
   with overflow" (`taylor.rs:62`, debug; wraps in release). Its `laurent` (`:131`, tracker 🛑 :2189)
   is the same facade. It is another crate, not named in the item, and returns `Expr`; delegating to
   `Expr::taylor` needs a signature change there. New HANDOFF row.
2. **Puiseux and logarithmic expansions** (sqrt(x), log(x), x^x at 0, and so sqrt(x)^2 or exp(log(x))
   expanded bottom-up): MAGMA ch. 49 §49.1.2 Puiseux series need fractional exponents, and log terms
   are not series at all. Interim: `NotLaurent` (documented as "a subexpression has no Laurent
   expansion").
3. **gamma, zeta, factorial, named functions of the variable**: `Unsupported`. Needed: their series
   (polygamma constants) or derivative hooks (the `FunctionRegistry` derivative is not wired,
   tracker :2106).
4. **Faster series arithmetic** (§5.2): Newton iteration and fast multiplication; the budget keeps the
   REPL honest meanwhile.
5. **Iterative Display, PartialEq and substitute** for deep trees from other sources (they overflow at
   10000, 40000 and 20000 left-deep terms, §1.2). Not named in the item (which names simplify and
   Drop), and the engine's output is balanced, so taylor no longer produces such trees. Each needs its
   own explicit-stack rewrite (Display through a formatter work list). New HANDOFF row.
6. **Products of radicals** (`sqrt(6) - sqrt(2)*sqrt(3)`): needs a prime factorization of every
   radicand. Interim: `UndecidedZero` when such a constant must be inverted (sound). The same holds
   for large rational powers kept as kernels: `2^(-300000000) * 2^300000000 - 1` and
   `(1/2)^1073741827 * 2^1073741822 - 1/32` are 0, but the two powers are distinct kernels and their
   balls leave the window, so they are `UndecidedZero` (review 2 test inputs). Needed: kernels
   `b^n` over a canonical base b (a positive integer that is not a perfect power, as for radicals)
   with a folding rule back to a rational for small exponents, and `exp(q)` for rational q
   represented as `e^q`. Neither is needed for soundness.
7. Other defects seen but not touched: `differentiate` of erf uses a rational approximation of
   2/sqrt(pi) (`differentiate.rs:197`); `eval_float` returns None for `pi`/`e`.
