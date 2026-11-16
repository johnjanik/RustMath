# S1-symbolic, revision 7: red team 1 (memory and ownership)

Design attacked: `fu2/design/S1-symbolic.md` (revision 7, all sections, including §2.15 caps and §2.16
Invariants and its table "Memory"). Prototype: `EXP/rev7/proto8/` copied unchanged to
`EXP/redteam1-memory/src/proto8/` (EXP = `fu2/design/S1-symbolic`), plus one probe of mine,
`replshow.rs`. Everything was built with `EXP/redteam1-memory/build.sh release SRC NAME` (the
revision 7 `build.sh` with the output moved to `EXP/redteam1-memory/bin/`, against the HEAD rlibs of
`EXP/rev5/rlibs_rel.txt`; `XFLAGS="--cfg memcount"` for the counting build). Every run had a timeout
and an address-space cap (`ulimit -v 4000000` for the probe, `RLIMIT_AS` 6 GiB in `mrun.py`). HEAD
is 2e15d1c; rustmath-symbolic and rustmath-jupyter's display code are as the design cites them. The
binaries were deleted after the runs; `build.sh` rebuilds each in about 4 s.

Lens: copies from the engine into `SeriesExpansion`, `Expr`, the Display text, the
`limit_via_series` path and the REPL taylor arm; the `Arc` sharing achieved; long names, long
literals, many terms; peak memory against the charged units through the public entry points.

## Hole H1: the REPL taylor line spends 32 s formatting the LaTeX of an admitted result

What happens. Through `taylor_with_budget` the result shares each kernel's expression across all
terms (§2.2, §2.8, revision 7 B2). That is what keeps the engine's memory small, but it also means
that a 99 KB input produces a result whose text is 100 times the input, for about 1.1e7 units. The
REPL then shows the result with HEAD's `to_display` (`rustmath-jupyter/src/repl.rs:349-385`). J1b's
`display_size` estimates the text at 16.67 MB, just below `OUTPUT_BYTES` = 16 MiB, so it is printed.
`format!("{}")` is linear (0.17 s for 10 MB). But `expr_to_latex` (`repl.rs:10883-10935`) builds
each node's text by `format!` over its children's finished strings, and the `Function` arm also
copies the arguments once more with `join`. So every subtree's text is copied once per ancestor:
the cost is O(text x depth), not O(text). A deep kernel printed in every term costs about
9 D^2 bytes of copying per occurrence.

Input (`EXP/redteam1-memory/inputs/fchain_33000.txt`, 99 009 bytes): `(F)/(1-x)` with
`F = f(f(...f(1)...))`, 33 000 nested calls of a one-letter function. The REPL line is
`taylor(<that text>, x, 0, 100)`.

Reproduction: `FILE=inputs/fchain_33000.txt bin/replshow_rel @ 0 100` (64 MiB thread, the REPL's
`EVAL_STACK_BYTES`). `replshow.rs` runs the prototype's `api::taylor_with_budget` with budget 1e8,
then HEAD's `display_size`, `format!("{}")` and `expr_to_latex`, copied verbatim from `repl.rs` at
2e15d1c (`decimal_string` is `to_string` for these small integers).

| step | time | size |
|---|---|---|
| parse (HEAD parser) | 0.045 s | |
| `taylor_with_budget`, budget 1e8 | 0.062 s | 1.12e7 units, `Ok`, 101 terms, each `F * x^k` with F shared |
| `display_size` | 0.069 s | 16 667 525 bytes, below `OUTPUT_BYTES`, so printed |
| `format!("{}")` | 0.17 s | 10 000 887 bytes |
| `expr_to_latex` | **31.96 s** | 33 331 691 bytes |
| whole line | **32.3 s** | VmHWM 188 MB |

Scaling (same probe, order 100): depth 5000 gives 0.77 s of LaTeX, 20 000 gives 12.1 s, 30 000
gives 26.3 s, 33 000 gives 32.0 s. Within the display budget the cost is about
`(OUTPUT_BYTES / 5D) * 9 D^2`, so it grows linearly with the depth D, which a 100 KB text bounds
at about 33 000. A `sin(sin(...))` chain of depth 20 000 at order 51 takes 2.1 s (Unary nodes are
estimated at 16 bytes, so fewer occurrences fit).

Correctness: the result is right. gp `print(truncate(ff/(1-x)+O(x^5)))` gives
`ff*x^4 + ff*x^3 + ff*x^2 + ff*x + ff`, which is the prototype's output for `f(f(1))/(1-x)` at
order 4 (`p8_rel series`), and `Vec(truncate(1/(1-x)+O(x^101))) == vector(101,i,1)` is 1.

HEAD on the same input (`HEADTAYLOR=1`, HEAD's `Expr::taylor` in the same probe): order 5 takes
2.2 s in `taylor` with VmHWM 1.0 GB, and its LaTeX 8.9 s (11.6 s in all); order 6 aborts at the
4 GB cap. So HEAD does not handle the input at order 100, and this is not a regression. It is a time
hole of the REPL taylor command, which the design's two budgets are meant to bound: the engine part
is 62 ms, the rest is HEAD's output conversion, which S1's shared results reach at 100 times the
input's text. Memory stays below the bar (188 MB).

Over the bar? Yes if the REPL `taylor` line counts as the entry point with its own budget
(`TAYLOR_WORK` and J1b's `OUTPUT_BYTES`), which is what §0 item 5, §3.1 and §3.4 promise to bound.
If only the time inside `taylor_with_budget` counts, it is a minor issue (62 ms).

Design sections: §3.1 (the `TAYLOR_WORK` doc comment: "The display budget (`OUTPUT_BYTES`) bounds
the text"), §3.4 ("J1b's display budget ... is what bounds the printed text of a taylor result"),
§2.7 ("The printed size is the REPL's concern"). The design measures only Display (text bytes and
`format!`), never the LaTeX that `to_display` builds next to it.

Fix that closes the class (S1 already edits `rustmath-jupyter/src/repl.rs` for the taylor arm):
make the LaTeX writer linear, as Display is. Write into one buffer, `fn latex_into(e: &Expr, out:
&mut String)`, pushing each node's prefix, its children and its suffix in place (no `format!` of
finished child strings, no `join`). Then the output cost is linear in the text, and J1b's estimate
bounds it (LaTeX is at most about 3.3 times the estimated Display text on these shapes). I wrote it
in `replshow.rs` (`LINEAR=1`): it gives the same string as `expr_to_latex` (`CHECKSAME=1` on the
depth 1000 chain and on `(sin(x)+sign(pi)+f(2,3)+(2/3)^x)/(1-x)`), and on the 99 KB input the LaTeX
takes 0.13 s instead of 32.0 s (whole line 0.52 s, VmHWM 139 MB). Its recursion depth is that of
Display, which the 64 MiB REPL thread already allows. Alternatively `display_size` could also bound
the sum over nodes of text times depth. Either way §3.1 and §3.4 should state that the display
budget bounds the output time only because the LaTeX writer is linear, and a REPL test should time
the depth 33 000 chain at order 100 (on a 64 MiB thread).

## Families tried that held

All release, budget 1e8 (`TAYLOR_WORK`, or `LIMIT_SERIES_WORK` for `lim`), point 0 unless stated,
heap peak from the counting allocator through the public entry points (`tw` =
`taylor_with_budget`, `se` = `series_expansion` then `to_expr`, `lim` = `limit_via_series`),
script `EXP/redteam1-memory/mrun.py`.

- Long-name coefficients, no copy per term (§2.8, B2 of review 7): `(Y)/(1-x)` with a 99 980-byte Y
  at order 2000: 1.89e6 units, 0.9 MB of heap (0.49 bytes per unit), VmHWM 4.6 MB.
- One shared constant rendered in every term, with a long literal and a long name:
  `(L + Y)/(1-x)` with a 49 990-digit L and a 49 990-byte Y at orders 500, 2000, 4000, 6000: 1.86 to
  1.88 bytes per unit, at most 128 MB of heap and 132 MB VmHWM (6.8e7 units), the same through
  `tw` and `se`; `lim` 0.2 MB. Each term re-renders the shared constant (charged by `expr_units`), so
  this is a constant factor, not a hole.
- Long literal coefficient `(L)/(1-x)`, L of 99 990 digits: order 2000 `Ok` at 7.47e7 units with
  167 MB (2.23 bytes per unit); at the budget edge order 2650 `Ok` at 9.9e7 units with 221 MB of
  heap and 225 MB VmHWM (2.23), below the stated 2.29 and 230 MB; orders 4000 and 8000 refused with
  at most 168 MB; `lim` gives `Finite(L)` with 0.6 MB.
- The top-node copy of a result that is one shared coefficient: `g(1, ..., 1)` with 49 990 arguments
  at order 5 copies the argument vector once in `to_expr()`: 3.2 MB at 1.85e6 units (1.72) through
  `tw`, `se` and `lim`.
- A long point, copied once per call (§2.16, "per call, not per term"): `exp(x)`, `sin(x)` and
  `1/(1-x)` at a 99 000-byte symbol or a 99 990-digit integer, order 2000: at most 10.2 MB (1.32).
  Through `lim` at 1.4e5 units the fixed copies of the point dominate (0.6 MB, 4.25 bytes per unit:
  about five copies of the 99 KB point, as §2.16 lists), bounded by the input, not by the budget.
- Cumulative allocation per unit (`heap_total` / units, which bounds what fragmentation can add to
  VmHWM): `sin(x)` at 2000: 3.27; `x + S_12*sin(x)`: 2.45; `1/(x + y1 + ... + y14999)` at 3: 2.39;
  a 49 KB name used twice: 2.13; `(Y)/(1-x)` at 2000: 1.78. So VmHWM cannot rise far above about
  3 bytes per unit.
- J1b's text estimate on S1 results: never below the real Display text (estimate 16.67 MB for 10.0 MB
  of text, 15.15 MB for 9.1 MB); `format!` is linear; integers in coefficients stay far below
  `LIBRARY_INT_DISPLAY_BITS`, so their quadratic decimal conversion stays below 1 s.
- Code reading of the copies: `to_expr_raw` clones a bare kernel's top node only in the numeric test
  (transient, and `ball_cost` charges the arguments it walks), in the `UndecidedZero` value (once per
  error) and in the expression of an `Inverse` kernel of a computed P0 that is a bare kernel (one per
  input node at most). Every error text is fixed except the function name of `Unsupported(f(...)
  of an expression in the variable)`, one copy per error. The kernel key's `kind` string (`$len:name`)
  is built before the key is charged, so a symbol occurrence allocates its name twice, once
  uncharged, but both are transient and the key's charge (1 unit per byte) covers the time
  (above 1e9 units per second).
- The `limit_via_series` path: it reads the first stored term only and copies one rational
  (`Finite(c.clone())`), as §2.13 says. The REPL `limit` arm calls `limit()`, which never reaches
  `limit_via_series` (`limits.rs:47-155`), so the REPL has no path that runs the engine more than once
  per line.

## Files

- `EXP/redteam1-memory/src/proto8/replshow.rs`: the REPL taylor arm end to end (prototype engine,
  HEAD's `display_size`, `format!` and `expr_to_latex`, the linear writer `latex_into`; `HEADTAYLOR=1`
  uses HEAD's `Expr::taylor`, `LINEAR=1` the fix, `CHECKSAME=1` compares the two LaTeX strings).
- `EXP/redteam1-memory/mrun.py`: one capped, timed run through `tw`, `se` or `lim` with the heap peak.
- `EXP/redteam1-memory/inputs/`: the inputs above (`fchain_*.txt`, `sinchain_20000.txt`,
  `LY_over.txt`, `Lonly.txt`, `Yonly.txt`, `gargs.txt`, ...).
