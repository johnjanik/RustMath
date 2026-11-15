# S1-symbolic design: adversarial review 7

Design reviewed: `fu2/design/S1-symbolic.md` (3266 lines, "Revision 6" to "Revision 1" and the body).
HEAD 0c2f368 (`git log c79a582..HEAD` ends at the tracker commit of part 2; rustmath-symbolic is
unchanged since 80130e7, and `git diff --stat 92906d4 HEAD` touches only `docs/` and `PROJECT_TRACKER.md`).
My experiments are in `fu2/design/S1-symbolic/review7/` (below `R7/`): `p7/` is the revision 6
prototype `rev6/proto7/` copied unchanged plus two drivers of mine (`main_se.rs`, `main_drop.rs`),
`build.sh` builds against the HEAD rlibs of `rev5/rlibs_*.txt`, `run7.py` is `rev6/tools/run6.py`,
`gen7.py` writes the new probe inputs to `inputs/`, `probes/` holds the MPFR timing programs,
`gpcheck3_r7.out` is the oracle run and `refs7.gp` the gp checks. Timings are release, budget 1e8
(the provisional `TAYLOR_WORK`), point 0, order 1, unless stated. The binaries were deleted after the
runs; `build.sh release p7/main.rs p7` rebuilds one in a few seconds.

Verdict: **not approved**. Two new blockers. B1 is a time hole: MPFR's erf costs up to about 1000
times what `ball_cost` charges it, and an 11.6 KB input is admitted after 31.7 s. B2 is a memory hole
in the public layer that the prototype does not model: `SeriesExpansion` stores owned `Expr`
coefficients, so every coefficient that is a bare kernel clones the kernel's top node, and a 100 KB
input holds 983 MB at 4.5e6 units. Revision 6 resolves the three blockers and the five minor issues
of review 6, and the algorithm, the oracles and the callers hold.

## What I verified (holds)

- Review 6 B1 to B3, on the rebuilt prototype with the counting allocator (`R7/fam_r6.py`):
  `fz(64, 186)` refused after 0.06 s, 4.4 MB of heap (revision 5: 15.5 s, 2.53 GB); `fz(64, 32)` refused
  after 0.03 s; `fz(64, 8)` `Ok` in 0.11 s, 38 MB (0.66 bytes per unit); `two(64, 4)` refused after
  0.20 s, 50 MB; `negz(64, 8)` refused after 0.29 s, 99 MB; `sparseg(60, 4)` refused after 0.14 s,
  32 MB; `sparseg(60, 1)` `Ok`, 126 MB (1.66); `negC(4096)` at order 560 refused, 177 MB of heap and
  183 MB VmHWM; `kterm` with 3 and 11 terms per coefficient refused at 52 to 56 MB. Every figure is
  within a few percent of §5.3.
- Review 6 m1: `invpow(80)` without a budget is `Ok` in 2.9 s with 19 MB; with the budget it is
  refused after 0.15 s. m2: `product_many` returns 0 before any round (`konst.rs:883-890`). m4 and m5:
  §2.4, §2.7 and the `scale` row of §2.16 now state the prototype's rules (`ser.rs:253-270`,
  `:296-300`).
- Oracles: `rev2/gpcheck3.py` on the rebuilt release prototype gives 107 OK and 0 failures
  (`R7/gpcheck3_r7.out`, 12 s). gp 2.15 (`R7/refs7.gp` and inline commands): tan to 15, the triple
  pole `1/(sin(x)-x)` ([-6, 0, -3/10, 0, -11/1400, 0, -17/126000]), `1/(sin(x)^50-x^50)` (valuation -52,
  -3/25, -62/125), H49E3 (`== 1`), `1/(2+sin(x))` at 1, `1/(x+sin(1)+I)`, acosh at 2. Every value
  equals §4.1.
- Expand under a minus (§1.7, §2.17, §4.6): HEAD's `distribute_negation` (`expand.rs:250-263`) is as
  quoted. `rev6/probes/expandneg.rs` rebuilt gives HEAD -4/7, 4/7, 34/35, -36/35, 16/49 and the fix 4/7,
  -10/7, -64/35, 34/35, 40/49; gp `x=3/7; y=2/5; print([-(x-1), -(x+1), -(x+y+1), -(x-y-1),
  -((x-1)*(x+1))])` gives `[4/7, -10/7, -64/35, 34/35, 40/49]`. The new recursion descends only where
  HEAD's did. The callers of `.expand()` are the five §2.17 lists (`rustmath-py` twice, the REPL arm
  `repl.rs:2054`, the benchmark, `simplify.rs:36`), and no test asserts the old forms (the only REPL
  test, `repl.rs:12943`, expands `(x + 1)^2`).
- Callers at HEAD: the taylor arm `repl.rs:2118-2145` with `symbols()` at `:2127`, `simplify()` at
  `:2135` and `taylor` at `:2143`; `TAYLOR_ORDER` at `:11087` with J1b's comment that the limit waits
  for S1; `limit_via_series` at `limits.rs:428`, called at `:509` with order 5.
- The numeric conversions that `ball_cost` charges linearly are linear: a rational of 2^20-bit parts
  converts in 0.75 ms at 64 bits (`R7/probes/ratconv.rs`), about 4.4e7 units/s.
- Drop order: on a 2 MB debug thread, `x + sin^300(1)` built with the `Expr` API expands and its
  result drops whether the kernel table is dropped before or after the result (`R7/p7/main_drop.rs`).
  A kernel chain is no longer than the input nesting that `expand` recursed through, so its drop is
  shallower than the expansion.

## Blockers

### B1. erf is charged like a cheap function, but MPFR's erf costs O(x^2) terms (§2.3, §2.7, §5.3)

`ball_cost` charges every unary function other than sin, cos, tan and the inverse hyperbolic ones
`2u + BALL_NODE`, with `u = (p/64 + 1)^2 * 8`: 96, 432 and 4656 units at 64, 256 and 1024 bits
(`ball.rs:383-430`). `Ball::unary(Erf)` calls MPFR's erf at the ball's precision (`ball.rs:214-218`).
MPFR's erf sums a Taylor series whose terms grow like e^(x^2) before they decrease, so its cost grows
with x^2 up to the point where erf(x) rounds to 1 (about x = sqrt(p ln 2)), and its Ziv loop often
retries at a higher precision. Measured with rug at HEAD (`R7/probes/erf_scan*.rs`, `fn_scan.rs`):

| precision | argument | time | units charged | units/s |
|---|---|---|---|---|
| 64 | 6.5 | 19.4 us | 96 | 4.9e6 |
| 256 | 13 | 187 us | 432 | 2.3e6 |
| 1024 | 10, 20, 26 | 0.25, 0.71, 1.3 ms | 4656 | 1.9e7 to 3.6e6 |
| 1024 | 975/38 (Ziv retries; 898 of 2936 rationals k/d in [23, 26.6] with d <= 40 take over 2 ms) | 6.4 ms | 4656 | 7.3e5 |

The other enclosed functions take at most 89 us at 1024 bits on the same scan (acos), at least
5e7 units/s. §2.3's claims "exp, log, sqrt, atan, tanh, erf and sinh take at most 10 us at any
exponent" (measured at 64 bits only) and "each call costs at most a constant times the cost of an
operation at precision 2p + 64" are false for erf.

Reproduction (`R7/gen7.py erfgroups n`): `1/(x + (y1 + ... + yn)*(erf(975/38) + erf(-975/38)) + y0)`.
The constant c0 has 2n + 1 terms. Rule 3 of the zero test groups them by indeterminate: each yi has
the coefficient `erf(975/38) + erf(-975/38)`, which is exactly 0, so its ball contains 0 at 64, 256
and 1024 bits and both erf nodes are evaluated at 1024 bits; y0's group (coefficient 1) is last and
decides `NonZero` (`konst.rs:1088-1100`). The result is correct: gp
`print(truncate(1/(x+(a+b+c)*(1-erfc(975/38)+(-1+erfc(975/38)))+d)+O(x^2)))` gives 1/d - x/d^2 up to
zeros of order 1e-327, which is the prototype's output for n = 3.

| input | length | outcome | wall | units | units/s |
|---|---|---|---|---|---|
| n = 100, erf(26) | 287 B | `Ok` | 0.27 s | 2.67e6 | 9.9e6 |
| n = 100, erf(975/38) | 295 B | `Ok` | 0.89 s | 2.67e6 | 3.0e6 |
| n = 3600, erf(975/38) | 11.6 KB | `Ok` | **31.7 s** | 9.61e7 | 3.0e6 |
| n = 3700, erf(975/38) | 12.0 KB | `Ok` | **32.1 s** | 9.88e7 | 3.0e6 |

VmHWM stays at 10 MB. The §5.3 rule would not catch it: its families contain no erf of a large
argument, and lowering `TAYLOR_WORK` to 10 s times 3e6 units/s would refuse most calibration rows.

Required change:

1. Charge erf by an a-priori bound of MPFR's cost that depends on the precision and on the argument,
   with a margin for the Ziv retries (for example `u * (2 + (|m| + r)^2)` capped where erf rounds to
   1), or replace erf of a large argument by a cheap rigorous enclosure (for x > 0,
   `1 - e^(-x^2)/(x sqrt(pi)) < erf(x) < 1`, one exp) and call MPFR only below a small bound on |m|.
   Say which in §2.3 and in the `ball_cost` row of §2.7.
2. Measure every enclosed function at 64, 256 and 1024 bits over a grid of arguments that includes
   values just inside each guard and rationals with long mantissas (`R7/probes/fn_scan.rs` and
   `erf_scan3.rs` are a start), and give the slowest units/s per function in §2.3.
3. Add the erf-group family to the §5.3 calibration and a budget test (debug, outcome asserted,
   "returns within 20 s"): n = 3700 at 0, order 1, budget 1e8 is `BudgetExceeded`, or `Ok` within the
   bar if the cheap enclosure is taken.

### B2. `SeriesExpansion` stores owned `Expr` coefficients, so each bare-kernel coefficient copies the kernel's top node (§2.8, §2.9)

§2.9 gives `SeriesExpansion { terms: Vec<(i64, Expr)>, .. }` and `to_expr(&self) -> Expr`, and §2.8
says that `SeriesExpansion::to_expr()` "renders what the expansion already paid for". Revision 6 made
the rendering refer to each kernel's `Arc<Expr>` so that no name is copied per term. But a coefficient
that is a single kernel with coefficient 1 renders as the kernel's own `Arc` (`konst.rs:1405-1411`,
`mono_expr`), which the kernel also holds, so an owned `Expr` of it is a clone of the top node:
a `Symbol` clones its name's `String`, a `Function` its name and its argument vector. Then
`to_expr(&self)` must clone each stored coefficient again to put it under `Arc`. Neither copy is
charged: `render_units` charges a bare-kernel coefficient `expr_units` = 140 units and `RENDER_TERM`
(`ser.rs:1252-1254`). The prototype renders from its internal series with shared `Arc`s
(`ser.rs:1260-1299`), so none of the measurements of §5.3 covers this layer.

Reproduction (`R7/p7/main_se.rs`: the prototype's expansion, then `terms` built with `to_expr_raw`,
which is `Arc::try_unwrap` or a clone of the top node, then the sum built with `Arc::new(c.clone())`
per term, as `series_expansion(..)?.to_expr()` would do). Y is a symbol of 99 990 characters, order
2000, budget 1e8:

| input | length | units | prototype's own renderer | after `SeriesExpansion` | after `to_expr()` |
|---|---|---|---|---|---|
| `Y/(1-x)` | 100.0 KB | 1.85e6 | 5 MB | 199 MB | **396 MB** |
| `(Y)/((1-x)*x^3000)` | 100.0 KB | 4.48e6 | 6 MB | **493 MB** | **983 MB** |
| `(Y)/((1-x)*x^10000)` | 100.0 KB | 1.06e7 | 9 MB | **1.18 GB** | **2.35 GB** |

Each call returns in 0.2 to 1.1 s. The same class reaches the library's budgeted path:
`limit_via_series` calls `series_expansion` with `LIMIT_SERIES_WORK = 1e8` (§2.13), so
`limit((Y)/((1-x)*x^10000), x, 0)` would hold about 1 GB. Coefficients that are integers or rationals
get one uncharged copy of their digits in `to_expr(&self)`: `(L)/((1-x)*x^1000)` with a 99 990-digit L
holds 251 MB in the prototype and about 375 MB with the layer (below the bar, but the same defect).

Required change:

1. Keep each coefficient of `SeriesExpansion` as the `Arc<Expr>` that `to_expr_arc_raw` built
   (`terms: Vec<(i64, Arc<Expr>)>`), and build `to_expr()` from those `Arc`s, one increment per term;
   `coefficient(k)` returns the `Arc` or documents its one clone for the caller. Or render
   `taylor_with_budget` from the internal series as the prototype does, and charge every owned
   coefficient of `series_expansion` its top node's size (cache the bytes of each kernel's top node in
   `KernelData`: a name's length, a function's argument count).
2. State in §2.8 and §2.16 ("Memory") which path renders, that no top node is cloned per term, and add
   the row for whatever copy remains.
3. Measure the long-name family above through the public entry points (`taylor_with_budget`,
   `series_expansion`, `limit`) with the port, and add a test: `Y/((1-x)*x^3000)` at 0, order 2000,
   through `taylor_with_budget` with budget 1e8, then a check that the result's coefficients share Y's
   node (`Arc::ptr_eq` on the leaves), or a memory check in the §5.3 script.

## Minor issues (fix in the revision)

- **m1. `TAYLOR_ORDER` 20000 to 2000 refuses inputs HEAD handles.** At HEAD the REPL accepts orders up
  to 20000, and `taylor(x+1, x, 0, 20000)` returns `(x + 1)` in 0.46 s release (`R7/p7/head.rs`, a
  64 MiB thread); its doc comment says f = x needs 320 MB at that order. With §3.1 the same call is a
  ValueError. The engine does not need the lower bound: at order 20000 with budget 1e8, `x+1` is `Ok`
  with 860 units, `1/(1-x)` `Ok` with 1.7e7 units and 14 MB, and `exp(x)` and `sin(x)` are refused after
  0.15 s and 0.11 s. Keep 20000 (the budget and J1b's display budget bound the rest), or state the
  refusal as a behaviour change with its reason. The §7 item 8 refusals (budget refusals where HEAD
  returns) are honest and were accepted by earlier reviews; this one is an argument check with no cost
  behind it.
- **m2. Two HANDOFF notes marked "(S1's area)" are not mentioned.** `docs/HANDOFF.md:449-451` (added by
  0c2f368 with the expand item): "a symbolic chain `w + w + ... + w` of 3000 terms aborts the debug
  driver in `Expr` Display and Drop" and "`-(-x)` displays `--x`". The Drop part is fixed by §2.14 and
  the Display part is §7 item 5; with §2.12's sign runs `--x` parses back to x, so it round-trips. Say
  so in §1 and §7, or fix the Display of a negated negation.
- **m3. The "parser stops first" argument is measured with HEAD's parser.** §2.5 and §2.15 compare the
  engine's depth with HEAD's parser, which the prototype links; the margin is 1.68 for the `/` chain on
  a 2 MB debug thread (212 against 357 levels). §2.12 rewrites `unary`, `power` and `multiplicative`.
  The implementer must measure the new parser's limits on the six shapes of §2.5 (2 MB debug, 8 MB
  release) and confirm that `taylor_with_budget` handles every depth it parses; add this to §6.
- **m4. Results of `q_mul` with a one-word operand are charged 2 units per word** (`q_mul(1, L)`:
  `OP_BASE + nb + 1 + (nb + 1)`, 4 bytes per unit for the result's digits), above the "at most 3 bytes
  per unit" of §2.16 ("results of the charged arithmetic by their operation"). On the families I tried
  the series charges dominate (`((L)/((1-x)*x^7000))*0` at order 2000 is refused at 1e8 with 195 MB),
  so correct the table row and the claim, or charge `COPY_WORD` per result word.

## Summary for the orchestrator

Revision 6 fixes review 6: the sparse product is a charged heap merge, constants are sorted slices
with a charged header, the rendering charges its nodes and shares h and the kernels' expressions, and
every family of review 6 is refused or admitted below 2 bytes per unit. The oracles (107 OK), the gp
values, the expand fix and the callers hold. Two new blockers stop approval:

- B1 (time): `ball_cost` charges erf like exp, but MPFR's erf at 1024 bits takes up to 6.4 ms for 4656
  units. An 11.6 KB input whose zero test evaluates `erf(975/38) + erf(-975/38)` once per indeterminate
  group is admitted after 31.7 s at 3.0e6 units/s.
- B2 (memory): the public `SeriesExpansion` holds owned `Expr` coefficients, so a bare-kernel
  coefficient clones the kernel's top node (a long symbol name) per term, and `to_expr(&self)` clones it
  again. A 100 KB input holds 983 MB at 4.5e6 units (2.35 GB at 1.1e7). The prototype renders with
  shared `Arc`s and does not model this layer.

The fixes are an argument-dependent erf charge (or a cheap rigorous enclosure for large arguments) with
a scan of every enclosed function, and `Arc<Expr>` coefficients in `SeriesExpansion`. The minors are
the unneeded `TAYLOR_ORDER` reduction (m1), two HANDOFF notes (m2), the parser depth measured on
HEAD's parser (m3) and one table row (m4).
