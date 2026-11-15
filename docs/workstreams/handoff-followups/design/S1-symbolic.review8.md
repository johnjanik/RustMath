# S1-symbolic design: adversarial review 8

Design reviewed: `fu2/design/S1-symbolic.md` (4689 lines: "Revision 7, red-team round 3" down to
"Revision 1" and the body), with its prototype `EXP/rev7rt3/proto11/` (EXP = `fu2/design/S1-symbolic`).
HEAD 6e03987. `git diff --stat 2e15d1c HEAD` touches only `docs/` (HANDOFF's build bullet, README,
TEST_BASELINE, the m23_golay handoff), so every fact the design rechecked at 2e15d1c stands, and the
HANDOFF lines it cites (`:461-463`, the S1 bullet at `:509-519`) have not moved.

My files are in `EXP/review8/`: `gen_sign.py`, `gen_sign2.py` and `gen_sing.py` (the inputs of B1),
`inputs/`, `randhint.py` (random hinted towers against gp) with its outputs `rh_*.out`, `src/headsimp.rs`
(HEAD's `taylor` and `simplify` on a file), `p8/` (proto11 with the fix of B1; `p8.diff` is the whole
change), `gpcheck3_r8.out` (the gp oracle on proto11) and `gpcheck3_p8.out` (on p8). The two binaries
were deleted after the runs; each rebuilds in a few seconds with the rustc line of `rev7rt3/build.sh`
(release, `src/headsimp.rs` or `p8/main.rs`). Numbers are release on an 8 MB thread, budget 1e8, unless
stated.

Verdict: **not approved**. One blocker: the rendering hint of revision 7rt3 is set on tower nodes whose
exponent is not regular, so a result can print a constant term whose value is wrong (B1). Everything
else I checked holds; the blockers of review 7 and the holes of red team 3 are closed.

## What I verified (holds)

- Review 7 B1 (erf charge): `review7/inputs/egbig_3700.txt` (12.0 KB) is refused after 0.22 s,
  `eg100_975_38.txt` after 0.17 s, `fz64_186.txt` after 0.06 s, at most 10 MB of VmHWM (revision 6:
  `Ok` after 32.1 s). Review 7 B2 (owned coefficients): `proto11/api.rs` stores `Arc<Expr>` coefficients,
  renders each distinct one once and builds `to_expr()` from the `Arc`s; `b2probe_rt3.txt` gives at most
  5 MB of heap for the long-name family through the three entry points (review 7: 983 MB and 2.35 GB).
- Oracles: `rev2/gpcheck3.py` on `bin/proto11_rel` gives 107 OK and 0 failures (`gpcheck3_r8.out`). The
  design's new gp references hold: `exp(4294967297 log(1+10^-9)) - 5` = 68.3298161434,
  `exp(8589934592 log(1+10^-9)) - 5` = 5372.2619249, `exp(8589934592 log(1-10^-9)) - 1/2` = -0.4998140317,
  `exp(-8589934592 log(1+10^-9)) - 1` = -0.9998140317, `asinh(-2^600) + 400` = -16.5814555165,
  `(2^27+1) log 2 - 93032640` = 0.4375088078, `1 - exp(36893488147419103233 log(1+10^-30))` =
  -3.6893488148e-11; `x^10*exp((sin(x+O(x^20))-x)/x^4)` is "domain error in exp: valuation < 0";
  `f(-0.1)` of `1 + t^10 exp((sin(t^2)-t^2)/t^9)` is 2.41e62.
- Red team 3, rerun by me on proto11: the zero bases `exp(x) + ((x+1)^2 - x^2 - 2*x - 1)^(x+1)` (0, 3),
  `cos(x) + (x*0)^(1+x^2)` (0, 3), `(x-x)^x` (1, 2), `x + 0^(2^x)` (1, 2), `x + 0^(x+sqrt(2))` (0, 2) give
  `1 + x + x^2/2 + x^3/6`, `1 - x^2/2`, `0`, `x`, `x`; `x + 0^((sin(1)^2+cos(1)^2-1)/x + 2)` is
  `UndecidedZero`; `x + (x*0)^(cos(4) + x)` is `DivisionByZero`; `((x+x^2)^1000)/x^999` at order 3 is
  `x + 1000 x^2 + 499500 x^3`, gp's `x (1 + x)^1000`. `redteam3_rt3.txt` shows the series H1 inputs
  `NotLaurent` through `taylor_with_budget` and `series_expansion` and `Unknown` through the limit path.
- The empty-operand rule of §2.1 against the code: `require_nonneg_val` (exp, sin_cos, the atan family
  with erf), `integrate`, `inv`, `log`, `pow_const`, `pow_int` items 2 and 4, `abs_sign` and `zero_pow`
  each refuse or ask for precision before reading a valuation or a leading coefficient they cannot
  trust; `mul` and `add` of `O(h^q)` give sound precisions; `trunc`, `pow_int` item 5 and the limit path
  read only exponents below which every term is exactly zero.
- The `0^r` rule (§2.5): with e0 = 0, `Re r(a) > 0` makes `0^r` identically 0 near the point, so the
  exact zero is right; `real_sign` returns a sign only from a rational or a ball that excludes 0, each
  enclosure charged by `enclose`. Zero bases with constant exponents: `(x-x)^(1/2)`, `0^(1/2)`,
  `(x*0)^(3/2)` give the exact 0, `(x-x)^(x-x)` gives 1, `0^(-1/2)` `DivisionByZero` (HEAD prints
  `0 ^ 1/2` and `0/0` terms).
- The inverse hyperbolic enclosures (`ball.rs:306-338`): the Lipschitz bounds `min(1, 1/d)`,
  `1/sqrt((lo-1)(lo+1))` and `1/((1-a)(1+a))` are the maxima of the derivatives on the balls, with the
  shrink factors on the safe side. Timed with `bin/onecall_rel` (300 calls): `atanh(1 - 10^-18)`,
  `acosh(1 + 10^-18)`, `asinh(-10^18/3)`, `asinh(3/(7 10^18))` at 1024 bits take 11 to 71 us for 13 936
  units (1.95e8 to 1.27e9 units/s), well above the 1e7 floor.
- Cost: the slowest calibration rows, rerun with the counting build: `x + (2^4095+1)^(1/64) + ...` with
  2000 roots (40.9 KB) refused after 3.0 s, `tan(x)` at 1, order 200, refused after 1.2 s (the design:
  2.72 s and 1.0 s without the counting allocator).
- The hint mechanism on regular nodes (§2.5 "Value at the point"): I re-derived that a node whose
  subtree is regular in the sense of §2.5 has a substitution equal to b0 (every operation's constant
  term is its function's principal value at the point, every division is by a decided nonzero value,
  `sign` only at valuation 0), and that a hint is a property of the constant's value, so any holder of
  the same `Arc` may use it. A random differential check backs this: `randhint.py` builds towers of 4 to
  7 levels `(c + f(...))^e` with f in sin, cos, atan, log, sqrt, tanh, asinh, c in 1, 2, -1, -3, 1/2, e in
  2, 3, 1/2, -1, 1/3, x, x+1, -2, 2/3, at the points 1, -1/2, 1/3, -2, 2, orders 0 and 1, and compares
  every coefficient with gp at 300 digits (`rev7rt3/checks/gpres.py`): 96 results, 0 different
  (`rh_1.out` to `rh_5.out`). The other 264 of the 360 inputs are errors, none a crash: 263
  `UndecidedZero` (mostly non-real constants, which have no enclosure; 257 of them are refused by
  revision 7rt2 too, the other 6 are m1) and one `NotLaurent`.
- The hint is never set at infinity (the variable is not regular there), and `subst_tree_estimate`
  saturates when there is no point node, so `subst_at`'s "substitution at infinity" error is not
  reachable through `maybe_hint` in the prototype.
- Drop (§2.14): `Arc::into_inner` hands the node to the last owner only, so a node held twice by one
  parent (`twin`) or shared with a sibling's subtree is torn down once, in the loop; every node with
  children is moved to the list, so the glue below a node drops leaves only.
- Callers and HANDOFF items: unchanged since review 7 (rustmath-symbolic and the crates it reads are
  untouched since 80130e7; `repl.rs` cited lines unchanged since f29c492).

## Blocker

### B1. A tower whose exponent expands to an exact constant ignores the exponent's regularity, so the hint substitutes a non-regular subtree and the result prints a wrong value (§2.5, `proto11/ser.rs:1293-1343`)

§2.5 defines a node as regular when "every child is regular" and when "every division in it is by a
value that is nonzero there", and only then renders its constant term as the node with the variable
replaced by the point. The tower record does not follow that rule on one path. `ETower::with_exponent`
sends an exponent whose series is an exact constant to `TSt::BaseConst(alpha)` and drops its
regularity flag `reg_g` (`ser.rs:1295-1298`); the `BaseConst` arm then sets the level's flag from the
base alone (`reg = reg_b && regular_val(&v) && !v.is_exact_zero()`, `ser.rs:1339-1342`), and
`level_done` calls `maybe_hint(self.node, &v)` with the whole tower node (`ser.rs:1309-1314`). An
exponent that depends on the variable, is not regular and still expands to an exact constant is easy to
write: `500*sign(x^2)` at 0 (the series of `sign(x^2)` is the exact constant 1 under the real-variable
convention, but the Unary record marks it not regular, since sign is not continuous at a zero),
`((1)/x)*0 + 500` (a pole times 0), `2*((x-1)/(x-1))` at 1 (a vanishing divisor that cancels). The
substitution then copies the exponent's subtree with x = a, which evaluates `sign(0)`, `1/0` or `0/0`.

Reproductions (proto11, `EXP/review8/inputs/`; `FILE=<file> DISPLAY=iter bin/proto11_rel coeffs @
<point> <order>`, i.e. through `series_expansion`; `taylor_with_budget` renders the same `Arc`s):

| input (bytes), point, order | stored constant term (proto11) | its value | correct constant term |
|---|---|---|---|
| `simple_sign.txt`: `x + (1+sin(1))^(500*sign(x^2))` (31), 0, 1 | `(1 + sin(1)) ^ (500 * sign((0 ^ 2)))` | 1 (gp: `sign(0) = 0`); fed back, `x + <it>` at 0 gives `1 + x` in proto11 itself | (1 + sin 1)^500 = 3.8233154825e132 (gp `(1+sin(1))^500`; for x != 0, sign(x^2) = 1) |
| `zero_pole_exp.txt`: `(1+sin(1))^(((1)/x)*0 + 500)` (29), 0, 1 | `(1 + sin(1)) ^ (((1 / 0) * 0) + 500)` | undefined (a division by zero; gp: "impossible inverse") | (1 + sin 1)^500 |
| `ssign_4.txt`: W_4 with W_0 = x, W_{k+1} = `(1+sin(W_k))^(2*sign((x-1)^2))` (109), 1, 1 | `((1 + sin(...(1 + sin(1)) ^ (2 * sign(((1 - 1) ^ 2)))...)) ^ (2 * sign(((1 - 1) ^ 2)))` | 1 (gp) | 2.8969785251826882905 (gp: `G(t)=my(s=t); for(i=1,4, s=(1+sin(s))^2); s; G(1)`); the x coefficient is right, 18.2587797965 |
| `cancel_div.txt`: `x + (1+sin(1))^(500*((x-1)/(x-1)))` (35), 1, 1 | `1 + ((1 + sin(1)) ^ (500 * ((1 - 1) / (1 - 1))))` (the `1 +` is x at 1) | undefined (0/0) | 1 + (1 + sin 1)^500 |
| `sing_8.txt`: the W_k tower with `exp` in place of `sin` and the exponent `(2+((1)/(x-1))*0)`, 8 levels, 1, 0 | `... ^ (2 + ((1 / (1 - 1)) * 0))` at every level | undefined | F_8(1), the normal form |

Each call takes under 0.06 s (`hints 1` to `hints 3` in the `measure` line). HEAD prints the same
substituted text for the first and third rows (HEAD's `taylor` substitutes the point everywhere;
`EXP/review8/src/headsimp.rs`), and HEAD's `simplify` keeps `sign(0)` and folds `(1/(1-1))*0` to 0, so
this is not a regression against HEAD. It is a wrong result returned as `Ok` (bar (a)): the engine's own
normal form of c0 is right (revision 7rt2 printed it), and the rendering replaces it by an expression of
another value, which the design's own engine reads back as 1. A kernel made from such a constant (for
example `sin(b0)` one level up, from 6 levels on: `hints 2`) carries the wrong text into the higher
coefficients, and an `UndecidedZero` constant is printed with it. The zero test stays sound only by
accident: the ball code has no enclosure for `sign` at 0, `1/0` or `0/0`, so a wrong hint gives `Unknown`
(I tried `(1)/((x-1) + sin(W_4) - sin(W_4 + sin(1)^2 + cos(1)^2 - 1 + ((1)/(x-1))*0))` at 1: it is
`UndecidedZero` where the right answer is `1/(x - 1)`; honest, but only because no wrong hint is
ball-evaluable). The design's tests do not cover this path: `result_tree_cap`, the hinted families of
§5.3 and the gp oracle use regular exponents only, and §6 step 4 says to port `ser.rs` as it is.

The class: "a node marked regular although a node of its subtree is not". I checked every other place
that sets the flag against the rule: the `+ -` record (`s.reg &= r`), the `* /` record (each series
factor's flag and a divisor of valuation 0; a zero constant factor keeps the factors' flags), the unary
record (the operand's flag, and `sign` at valuation 0), the constant-base path (`reg_g`), the
general path (`reg_f && reg_g`), the exact-constant base inside `pow_general` (the hint needs `reg_f`),
the variable (finite point only) and constants. Only the `BaseConst` path of a tower with an exponent
that depends on the variable drops a child's flag.

Required change:

1. §2.5: say that a tower level is regular only if its exponent is, also when the exponent expands to
   an exact constant (the constant-exponent rule decides the arithmetic, not the regularity). Port it:
   `TSt::BaseConst` carries the exponent's flag (true for a top exponent free of the variable) and the
   arm ANDs it in. `EXP/review8/p8.diff` is the whole change (four lines). On `p8` the five inputs above
   print their normal forms (c0 of `ssign_4` evaluates in gp to 2.896978525182688290513575858664,
   gp's value), no result contains a division by zero, the gp oracle gives 107 OK and 0 failures
   (`gpcheck3_p8.out`), and the hinted results are unchanged: `F_10` and `F_30` at 1, order 0, trees
   of 51 and 151 nodes, `F_10` at order 1 468 023 nodes, the same as proto11.
2. §4.1: a test `value_at_point_irregular_exponent`, outcome and value asserted: `x +
   (1+sin(1))^(500*sign(x^2))` at 0, order 1, has a constant term with no `sign` node whose enclosure
   contains gp's `(1+sin(1))^500` = 3.8233154825e132; `(1+sin(1))^(((1)/x)*0 + 500)` at 0 has no `1/0`
   (no `Div` node with a zero denominator in the result); the 4-level `W_4` at 1, order 1, has c0 =
   2.89697852518268829051 and c1 = 18.2587797964534041670 (gp, the commands above). Each fails on
   proto11.
3. Rerun `tools/hintscan.py` and the hinted families of §5.3 on the fixed prototype, and say in §2.5
   that the hint is a second rendering of the same value only because the flag is the AND over every
   node of the substituted subtree, so the soundness of the numeric zero test (which evaluates the
   hints) does not depend on the ball code refusing a wrong one.

## Minor issues (fix in the revision)

- **m1. A hint changes what the numeric zero test evaluates, and the ball code reads less of the
  input's syntax than of a normal form.** §2.5 says "The hint changes no normal form, no key and no
  zero test", but the zero test's numeric rule evaluates `to_expr` of the constant (`ball.rs::enclose`),
  and `to_expr` returns the hint, which is the input's own syntax. `ball_expr` treats only an
  `Expr::Integer` exponent as an integer power and sends every other exponent to `exp(r log a)`, which
  needs a > 0; the parsers give `(-2)` as `Neg(Integer 2)`, so `(-3 + 1)^(-2)` has no enclosure
  (`bin/proto11_rel ball '(-3 + 1)^(-2)' 64`: `None(Never)`), where the normal form renders `c^-2` with an
  `Integer` exponent. On the random towers of `randhint.py`, 6 of the 102 inputs that revision 7rt2
  (`rev7rt2/bin/rt2_rel`) expands are `UndecidedZero` in proto11, for example
  `(2 + cos((1 + atan(((-3) + atan((1 + log((1 + tanh(x))^(-2)))^2))^(-2)))^2))^(2/3)` at -2, order 0
  (`inputs/rh_5_35.txt`; the HEAD-parser binaries give the same: `proto10_rel` `Ok`, `proto11_rel`
  `UndecidedZero` of the hinted base), where HEAD prints the value. Honest, and below the bar by the
  precedent of red team 2 numeric m1, but a regression of this revision, and the same gap exists
  without hints for a kernel made from an input node: `(1)/(x + sin((-3+1)^(-2)) - 1/4)` and
  `(1)/(x + sin((-2)^(-2)) - 1/4)` at 0 are `UndecidedZero` in both revisions, where HEAD gives
  `1/(sin(1/4) - 1/4) - x/(sin(1/4) - 1/4)^2`. Fix: `ball_expr` takes an exponent whose own ball is
  exact and integral (radius 0, an integer midpoint that fits in i64: `Neg(Integer)`, `3 - 1`, `4/2`) by
  binary powering, as an `Integer` exponent; correct the sentence of §2.5 (a hint can change an
  `Unknown` of the numeric test, never its soundness once B1 is fixed); add the two inputs above to
  §4.1 with HEAD's values.
- **m2. `0^alpha` for a constant non-integer alpha is still undecided where the series rule decides.**
  `x + 0^(pi + x)` at 0 is `x` by the new `zero_pow`, but `x + 0^pi`, `x + (x-x)^pi` and
  `x + (x-x)^(x-x+pi)` (the exponent an exact constant, so `pow_const` of the exact zero) are
  `UndecidedZero(0^pi)` (§2.2 "other alpha: c decided zero: UndecidedZero(0^alpha)"). Honest, but the
  same constant gets two answers by the route it takes. `pow_const(0, alpha)` can use the rule of
  `zero_pow` with `real_sign` of alpha (positive real part: 0; negative: `DivisionByZero`; else
  undecided).
- **m3. The Drop fallback for a thread whose locals are being destroyed.** §2.14 takes the iterative
  branch when `EXPR_DROP_STACK_TOP.try_with` fails, but `detach` then clones the thread-local `LEAF`,
  whose `with` panics during that same teardown (and a panic in `Drop` there aborts). No `Expr` lives in
  a thread-local in the workspace today (`grep thread_local`: one `Cell<i64>` in rustmath-integers), so
  this is a port note: read `LEAF` with `try_with` and fall back to a fresh `Arc::new(Expr::Real(0.0))`.
- **m4. A refusal where HEAD and revision 7rt2 answer quickly.** `F_12` to `F_15` at 1, order 1, are now
  `LimitExceeded(TreeSize)` where HEAD answers in 0.2 ms (and revision 7rt2 in milliseconds, with a tree
  that consumers could not walk). §7 item 8 records it with the other honest refusals of the expanded
  normal form; I agree with that classification, but §2.15's row should say that the cap refuses these
  inputs although the engine's own work is small (2.4e6 units at k = 10), so the HANDOFF row names the
  product-preserving representation as the remedy for both the cap and the budget.

## Summary for the orchestrator

Revision 7rt3 closes red team 3's holes and review 7's blockers: the empty-operand rule covers every
operation that reads a valuation, `0^r` and zero bases are decided by the engine's own tests, the
inverse hyperbolic enclosures are MPFR's with correct Lipschitz bounds, the erf charge and the shared
`Arc` coefficients hold, the gp oracle gives 107 OK, and a random check of 96 hinted results against gp
found no wrong coefficient. One blocker stops approval:

- B1 (a wrong result): the tower record drops the exponent's regularity when the exponent expands to an
  exact constant, so the value-at-the-point hint substitutes a non-regular subtree. `x +
  (1+sin(1))^(500*sign(x^2))` at 0 (31 bytes) returns c0 = `(1 + sin(1)) ^ (500 * sign((0 ^ 2)))`, whose
  value is 1, not (1 + sin 1)^500 = 3.82e132, and `(1+sin(1))^(((1)/x)*0 + 500)` prints `1 / 0`. The fix
  is four lines (`EXP/review8/p8.diff`, checked: correct values, gp oracle 107 OK, hinted families
  unchanged), plus a test and a sentence in §2.5.

The minors are the numeric test reading the hints' input syntax (m1: `(-3 + 1)^(-2)` has no ball, so 6 of
102 random decidable towers become `UndecidedZero`; honest), the constant `0^alpha` (m2), a Drop port
note (m3) and the wording of the tree cap's refusals (m4).
