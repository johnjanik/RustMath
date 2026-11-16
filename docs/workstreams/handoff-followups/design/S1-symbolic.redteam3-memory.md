# S1-symbolic, revision 7rt2: red team 3 (memory and ownership)

Design attacked: `fu2/design/S1-symbolic.md` (revision 7 red-team round 2, all sections, with the
caps table of §2.15 and the Invariants and "Memory" tables of §2.16). Prototype:
`EXP/rev7rt2/proto10/` (EXP = `fu2/design/S1-symbolic`), its release binaries `bin/proto10m_rel`
(counting allocator), `bin/proto10_rel`, `bin/replshow_rel` and the debug `bin/proto10`, used
read-only. My probes are in `EXP/redteam3-memory/` (built with `EXP/redteam3-memory/build.sh`, the
rev7rt2 script with the output moved to `EXP/redteam3-memory/bin/`, against the same HEAD rlibs):

- `mr3.py`: one capped (RLIMIT_AS 4 GiB), timed run of `proto10m_rel` through `ENTRY=tw`
  (`taylor_with_budget`), `se` (`series_expansion`, then `to_expr`) or `lim` (`limit_via_series`,
  `LIMIT_SERIES_WORK`), with units, heap peak, cumulative allocation per unit and VmHWM.
- `fam1.py`, `fam2.py`: the families below (inputs in `inputs/`).
- `src/replline.rs` (`bin/replline_rel`): one REPL line with several taylor calls, built from
  proto10's `api::taylor_with_budget` (budget 1e8) on a 64 MiB thread, folded as the REPL folds a
  `+` chain (`repl.rs:9737-9743`, `(Expr, "+", Expr)` at `repl.rs:10117`); `MODE=nest` is
  `taylor(taylor(F, ..), ..)`.
- `src/twolines.rs` (`bin/twolines_rel`): consecutive lines with the previous value kept as `_`
  (`last_result`, `repl.rs:763`, `:799`).
- `src/head_lim.rs` (`bin/head_lim_rel`): HEAD's `Expr::limit_via_series` on one input.

HEAD is 2e15d1c. Numbers are release, budget 1e8 (`TAYLOR_WORK`), point 0, unless stated. Every run
had a timeout and an address-space cap (4 GiB, 6 GiB for the multi-call lines).

## Result

No hole over the bar for one call through any public entry point. Over 512 MB and 20 s only for a
REPL line (or cell) that calls taylor several times: every call is admitted by its own
`TAYLOR_WORK`, and the line keeps every result. J1b's REPL model bounds each single call
(`repl.rs:10982-10985`: "So these limits still bound each single call") and already admits retained
values of that size (EuclideanSpace, 385 MB), so I class it below the bar and leave the call to the
reviewer. Three more issues below the bar.

## m1 (below the bar): a family above the stated measured maxima, at the table's bound

What happens. A coefficient that is one monomial with every exponent +1 or -1 renders, per term,
only `Mul` nodes (a balanced product of the kernels' shared `Arc`s), a `Neg` or a `Div`. Each is one
96-byte chunk for `EXPR_NODE = 32`, the row "Expr node of a rendering" at exactly 3.0 bytes per
unit. The engine shares the coefficient across all terms (`scale` by a coefficient 1 returns the
same constant), so it adds almost nothing, and the rendering, which renders the same constant once
per term, is nearly all of the memory.

Inputs (`fam2.py`, `inputs/neg_*.txt`, `den3_*.txt`): `(-1)*(a*b*...)/(1-x)` and
`((1)/(a*b*...))/(1-x)` with N short distinct names.

| input | order | units | heap peak | bytes/unit (cumulative) | VmHWM | time |
|---|---|---|---|---|---|---|
| neg, N = 2700 (8.3 KB) | 1100 | 9.76e7 | 286.5 MB | 2.93 (3.62) | 291 MB | 0.23 s |
| neg, N = 2700 | 1120 | 9.94e7 | 291.7 MB | 2.93 (3.62) | 296 MB | 0.26 s |
| den3, N = 2700 | 1040 | 9.24e7 | 271.0 MB | 2.93 (3.63) | 276 MB | 0.23 s |
| neg, N = 1000 | 3000 | 9.91e7 | 289.6 MB | 2.92 | 294 MB | 0.23 s |
| neg, N = 8000 (29.5 KB) | 360 | 9.76e7 | 280.8 MB | 2.88 | 287 MB | 0.27 s |
| pow2 `(a^2*...)/(1-x)`, N = 2700 | 330 | 9.81e7 | 258.5 MB | 2.64 | 264 MB | 0.19 s |

The same through `series_expansion` + `to_expr` (ENTRY=se: 291.7 MB, 296 MB at order 1120). The
limit path at order 5 holds 7 MB. Debug on a 2 MB thread, order 1100 with HEAD's Display: `Ok`,
VmHWM 305 MB. The REPL line (`replshow_rel`, `LINEAR=iter`, 64 MiB): order 860 prints 16.46 MB of
text and 21.1 MB of LaTeX in 0.54 s, VmHWM 260 MB; from order 880 the estimate passes
`OUTPUT_BYTES` and the summary is shown (229 MB).

Correctness: gp `Vec(truncate((-1)*(a*b*c)/(1-x)+O(x^4)))` is `[-c*b*a, -c*b*a, -c*b*a, -c*b*a]`,
`(1/(a*b*c))/(1-x)` gives `1/(c*b*a)` four times, `(a^2*b^2*c^2)/(1-x)` `c^2*b^2*a^2` three times;
the prototype's `coeffs` give the same.

Over the bar? No (296 MB). But §0 item 5, §2.7, §2.16 and the `TAYLOR_WORK` comment state 2.61
bytes per unit, 261 MB with a budget and 265 MB for a REPL line as the measured maxima; this family
gives 2.93 and 296 MB. The claim (3 bytes per unit, about 300 MB) holds with no margin: 296 MB at
9.94e7 units.

Fix. State this family as the measured maximum and add it to §5.3 and the port's memory check. The
class is "a result that renders one shared constant once per term": `series_expansion` can memoize
`to_expr_arc_raw` by the constant's `Arc` address (one map entry per distinct coefficient), so every
term with the same constant shares one rendering. The Display text is unchanged; this input then
holds about 1 MB instead of 292 MB. For a margin below the table's bound, `EXPR_NODE = 36`
(2.67 bytes per unit) would do it, at the cost of a recalibration.

## m2 (below the bar per call; over 512 MB and 20 s per REPL line): results add up within one line

What happens. The result of an admitted call is most of its memory (the rendering), and the REPL
keeps results: in the accumulator of a `+` chain (`a.clone() + b.clone()`), in a list, in `_`, in a
variable. `TAYLOR_WORK` bounds each call, so a line of k calls holds k results.

Inputs and measurements (`bin/replline_rel`, `bin/twolines_rel`):

| line | text | calls | VmHWM | time |
|---|---|---|---|---|
| `taylor(F, x, 0, 1100) + taylor(F, x, 0, 1100)`, F = `inputs/neg_2700.txt` | 16.7 KB | 2 | 564 MB | 0.43 s |
| the same, 3 calls | 25 KB | 3 | 844 MB | 0.68 s |
| the same, 11 calls (`inputs/replline_sum11.txt`) | 91.7 KB | 11 | 3.08 GB | 2.6 s |
| `taylor((1)/(1-x), x, 0, 20000) + ...` | 13.2 KB | 400 | 2.26 GB | 6.1 s |
| the same at 100 KB (extrapolated: 5.6 MB and 14 ms per call) | 100 KB | about 3000 | about 17 GB | about 46 s |
| two lines `taylor(F, x, 0, 1100)`, the first kept as `_` | 8.3 KB each | 1 each | 564 MB at the second | 0.2 s each |
| `taylor(taylor(F, x, 0, 1100), x, 0, 5)` | 8.3 KB | 2 nested | 389 MB, the outer refused by the budget | 1.6 s |

Each call alone: 9.76e7 units, 284 MB (F) or 1.65e7 units, 11 MB (`1/(1-x)`), as measured above. At
HEAD neither line finishes (HEAD's `taylor` of F at order 1100 is exponential), so this is not a
regression. Values: gp `Vec(truncate(1/(1-x)+O(x^6)))` is `[1, 1, 1, 1, 1, 1]`, the prototype's
coefficients.

Over the bar? Per call, no: every call is within the budget. Per line, yes: a 91.7 KB line uses
3.08 GB, and the `1/(1-x)` line passes 20 s and exhausts this 16 GB container near 100 KB. J1b's
model is per call (interrupts and the optional `time_limit` stop a cell between calls; an
out-of-memory abort kills the kernel), and other commands give lines of the same kind (two
`EuclideanSpace(7000000)` values, 385 MB each by `repl.rs:11088`). So I class it below the bar, as a
REPL-wide limit that S1 makes reachable by taylor results of 300 MB. Sections: §3.1 (the
`TAYLOR_WORK` comment: "265 MB for a whole REPL line", "within the REPL's policy of about 400 MB per
admitted call"), §0 item 5, §2.7.

Fix. In S1's area: charge every taylor call of one cell to one `TAYLOR_WORK` (the REPL arm keeps the
units left in the cell and passes `min(TAYLOR_WORK, left)`, which needs the units back from
`series_expansion`), so the taylor calls of a cell hold at most about 300 MB and run at most about
3 s together, whatever their number. For the value kept as `_` into the next cell, bound the result
itself well below the call (a separate cap on `render_units`, for example 5e7 units, about 150 MB),
so the kept value and the next call stay below 512 MB. Either way the `TAYLOR_WORK` comment should
say that the result is most of a call's memory and outlives the call. The REPL-wide part (a
per-cell memory policy for every command) is J1b's.

## m3 (below the bar): the limit path renders every stored term and reads one

What happens. `limit_via_series` calls `series_expansion` at order 5, which charges `render_units`
and renders every stored coefficient, then reads only the first. When the first term is cheap and a
later one is large, the budget is spent on renderings no one reads.

| input (point 0) | entry | outcome | units | VmHWM | time |
|---|---|---|---|---|---|
| `1 + x*S_17` (S_k = `(1+cos(1))*...*(1+cos(k))`) | lim | `Finite(1)` | 7.97e7 | 145 MB | 0.41 s |
| the same | tw, order 0 | `Ok`, constant term 1 | 3.19e7 | 71 MB | 0.18 s |
| `1 + x*S_18` | lim | `Unknown (Budget(153714749))` | 1.54e8 | 123 MB | 0.33 s |
| the same | tw, order 0 | `Ok`, constant term 1 | 5.41e7 | 123 MB | 0.32 s |

The limit is 1 (gp: `subst(1 + x*s, x, 0)`). HEAD's `limit_via_series` takes 1.6 s and 925 MB for
S_12 and aborts at 4 GiB for S_18 (`bin/head_lim_rel`), so this is not a regression, and `Unknown`
is honest.

Fix. Give the limit path the first stored term without the rendering (an internal entry that runs
the driver and reads the engine's first term, rendering only that coefficient), or expand at order 0
first and go to order 5 only when the first stored exponent is not 0. §2.13.

## m4 (below the bar): an object without a row in the table "Memory"

`expand_var` copies the variable's series per occurrence (`cx.xser.clone()`, a vector of 1 or 2
terms: a 32- or 48-byte chunk) for 8 units, 4 to 6 bytes per unit; held in a `+ -` chain record it
is 136 bytes for 40 units at a nonzero point. §2.16 says every heap object has a row; this one has
none (§2.7 lists only its units). It is bounded by the input (at most one held per pending chain
item): `x+...+x` with 49 999 terms at 1 holds 18.1 MB of heap at 2.39e7 units (0.76 bytes per unit
overall). Fix: add the row (charge `16 + SER_TERM` per copied term), or list it with the pre-pass
map as input-bounded.

## Families tried that held

All release, budget 1e8 (`LIMIT_SERIES_WORK` for `lim`), through `tw`, `se` and `lim` where stated.

- Numeric zero test of a large constant (ball memo in MPFR memory, which the counting allocator does
  not see, so VmHWM is the check): `(1)/(x + S_k)` at orders 0 and 2: S_14 `Ok`, 3.05e7 units,
  VmHWM 58 MB; S_15 `Ok`, 6.42e7, 113 MB; S_16 refused at 1.12e8 after 0.99 s, 226 MB; S_17 and S_18
  refused after at most 0.42 s, at most 149 MB. VmHWM at most 2.3 bytes per admitted unit.
- Long monomials with exponent 2 (red team 2's family with 2-letter names): at most 2.64 bytes per
  unit, 264 MB; with exponent 3 in a series divisor the inverse is refused at 1.35e8 after 0.03 s.
- A series divisor with a long monomial leading coefficient (`(1)/((a*...*N)*(1-x))`): refused at
  1.33e8 after 0.02 s with 12 MB at every order (the recurrence charges words squared).
- The public layer's copies (code and runs): `SeriesExpansion` holds one `Arc` per term; `to_expr`
  adds at most 4 nodes per term; `se` and `tw` give the same heap peak on every family above. Per
  call the prototype copies the point's top node three times without a charge
  (`ExpansionPoint::At(point.clone())` in `taylor_with_budget`, the header, h) and the variable's
  name twice (the header, h), plus the top-node copy of a single shared coefficient: six copies
  where §2.8 lists four. All are per call and bounded by the arguments, so this is wording only.
- Drop order: the prototype drops the engine's series before the kernel table, so no kernel chain
  is released by a cascade; the design's `Work` owns the table and is declared first, so it drops
  last. The prototype's `UNARY` side table is cleared only at the next call (a prototype artifact;
  the port keeps `(f, c)` in `KernelData`).
- Variable chains: `x+...+x` and `x-...-x` with 10 000 and 49 999 terms at 0, 1 and 7: at most
  1.31 bytes per unit, VmHWM at most 27 MB.
- Nested call in one line: `taylor(taylor(F, ..), x, 0, 5)`: the outer call reads the 284 MB inner
  result and is refused by its budget, VmHWM 389 MB (below 512 MB).
- Stack: the neg family's 296 MB result on a 2 MB debug thread is built, printed by HEAD's
  recursive Display (depth 26) and dropped; the REPL display path (`display_size`, the work-list
  writers) is iterative.
- J1b's estimate stays above the Display text on these results (16.48 MB estimated, 16.46 MB
  written); the LaTeX was at most 1.3 times the estimate here.

## Files

- `EXP/redteam3-memory/mr3.py`, `fam1.py`, `fam2.py`, `inputs/`.
- `EXP/redteam3-memory/src/replline.rs`, `src/twolines.rs`, `src/head_lim.rs`, `build.sh`, `bin/`
  (three release binaries of about 5 MB; `build.sh release src/<f>.rs <name>` rebuilds each in a few
  seconds).
