# S1-symbolic, revision 7rt1: red team 2 (memory and ownership)

Design attacked: `fu2/design/S1-symbolic.md` (revision 7 red-team round 1, all sections, with the
caps table of §2.15 and the Invariants and "Memory" tables of §2.16). Prototype: `EXP/rev7rt1/proto9/`
(EXP = `fu2/design/S1-symbolic`), its release binaries `bin/proto9m_rel` (counting allocator) and
`bin/replshow_rel`, plus two probes of mine in `EXP/redteam2-memory/src/` built against the same
HEAD rlibs with `EXP/redteam2-memory/build.sh` (the rev7rt1 script with the output moved to
`EXP/redteam2-memory/bin/`). Every run had a timeout and an address-space cap (4 GiB). HEAD is
2e15d1c. Numbers are release, budget 1e8 (`TAYLOR_WORK`, or `LIMIT_SERIES_WORK` for the limit path).

Lens: copies from the engine into `SeriesExpansion`, `Expr`, the Display text, the
`limit_via_series` path and the REPL taylor arm; the `Arc` sharing achieved; long names, long
literals, many terms; peak memory against the charged units through the public entry points.

## Result

No hole over the bar. One minor issue (a stated figure that a family exceeds).

## Minor m1: VmHWM and bytes per unit above the stated figures

What happens. A coefficient that is one monomial of N factors with exponent 2 renders, per term
and per factor, a `Mul`, a `Pow` and an `Integer` node. These are charged by `expr_units` at about
3 bytes per unit, the row "Expr node of a rendering" of the table "Memory". So the rendering
dominates the heap and the ratio rises towards that row's value, above the measured maximum that
§0, §2.7, §2.16 and the `TAYLOR_WORK` comment state (2.60 bytes per unit, VmHWM at most 256 MB).

Input: `(y0^2*y1^2*...*y7999^2)/(1-x)` (62 897 bytes), point 0. Script
`EXP/redteam2-memory/famA.py` and the edge runs of `mr.py` (`proto9m_rel`, `ENTRY=tw` and `se`).

| order | units | heap peak | bytes per unit | VmHWM | time |
|---|---|---|---|---|---|
| 100 | 9.31e7 | 236.8 MB | 2.54 | 245 MB | 0.22 s |
| 106 | 9.82e7 | 250.6 MB | 2.55 | 259 MB | 0.23 s |
| 107 | 9.91e7 | 252.9 MB | 2.55 | 261 MB | 0.22 s |
| 200 | refused at 1.79e8 | 6 MB | | 14 MB | 0.04 s |

With N = 2000 (14 897 bytes) at order 200: 2.61 bytes per unit (116.8 MB at 4.48e7 units). The
same through `series_expansion` + `to_expr`. The REPL line of the order-100 input
(`replshow_rel`, `LINEAR=iter`, 64 MiB thread): `taylor_with_budget` 0.17 s, `display_size` 15.2 MB
(printed), text 12.8 MB in 0.13 s, LaTeX 12.8 MB in 0.13 s, the line 0.48 s, VmHWM 265 MB.

Correctness: the result is right. gp `print(truncate((y0^2*y1^-2)/(1-x)+O(x^4)))` gives
`1/y1^2*y0^2*x^3 + 1/y1^2*y0^2*x^2 + 1/y1^2*y0^2*x + 1/y1^2*y0^2`, the prototype's terms; the same
for `(y0^2 + y1^2)/(1-x)` at order 2.

Over the bar? No. 261 MB is half the bar; the table "Memory" bounds every object at 3 bytes per unit,
so 1e8 units stay below about 300 MB plus the parsed input.

Sections: §0 item 5, §2.7 ("Measured ..."), §2.16 (the paragraph after the table "Memory"), §3.1
(the `TAYLOR_WORK` comment: "at most 256 MB of resident memory at 1e8 units").

Fix: state the bound of the table (at most 3 bytes per unit, so about 300 MB at 1e8 units plus the
parsed input) as the claim, and give this family (the rendering of long monomials with exponents
other than +-1, 2.55 to 2.61 bytes per unit, 261 MB of VmHWM at 9.91e7 units) as the measured
maximum; add it to the families of §5.3 so the port's check reruns it.

## Families tried that held

All release, budget 1e8, point 0 unless stated. `tw` = `taylor_with_budget`, `se` =
`series_expansion` then `to_expr`, `lim` = `limit_via_series` (order 5).

- Many terms with a shared bare-kernel coefficient: `(y)/(x^K*(1-x))` at order 20000. K = 50 000:
  `Ok`, 6.0e7 units, 29 MB (0.48 bytes per unit), 0.08 s; K = 200 000 and 10^6: refused at 1e8 in
  0.11 s with at most 40 MB; `lim` `Unknown` (pole, or the budget) with at most 40 MB.
- Multi-term coefficients with powers: `(y0^2 + ... + y_{N-1}^2)/(1-x)`, N = 2000 and 6000:
  at most 1.44 bytes per unit, 98 MB; N = 6000 at order 150 refused in 0.02 s.
- Negative exponents (the `den` vectors and a `Div` per term): `(y0^-2*...*y5999^-2)/(1-x)` at
  order 120: 8.3e7 units, 212 MB, 2.55 bytes per unit, VmHWM 219 MB, 0.21 s.
- The uncharged clone of an input divisor's node in `inverse_src` (`r.clone()`, 16 units; a
  Function's argument vector and name are copied): `x + (1)/(g(1,...,1))` with 49 000 arguments
  (98 KB): 6.6e6 units, 3.8 MB of heap, against 3.2 MB for `x + g(1,...,1)`. The copy is once per
  distinct kernel, so it is bounded by the input text (at most about 0.5 MB).
- The REPL display path (`replshow_rel`, `LINEAR=iter`): the LaTeX of a node is at most 3.33 times
  J1b's estimate (a zero-argument call `f()`), and about 2.4 times on the largest realistic shape:
  `(h(g(), ..., g()))/(1-x)` with 24 000 arguments (96 KB) at order 100: text 12.1 MB, LaTeX
  29.1 MB, the line 0.10 s, VmHWM 55 MB; at order 160 the estimate passes `OUTPUT_BYTES` and the
  summary is shown in 0.03 s.
- Integer conversion time in the printed text (a REPL Integer variable as the point, which
  `parse_symbolic_arg` passes as `Expr::Integer`; probe `EXP/redteam2-memory/src/bigpt.rs`):
  `exp(x)` at `2^4194303 + 1` (the largest integer `LIBRARY_INT_DISPLAY_BITS` admits) at order 6:
  engine 3.7 ms, 1.8e6 units; estimate 16.4 MB (printed); text 16.4 MB in 4.17 s (13 num-bigint
  conversions); LaTeX in 1.21 s (rug); the line 5.4 s, VmHWM 47 MB. At 2^21 bits, order 12: 3.9 s.
  The cost per printed byte grows with the bits of one integer, which J1b caps, so the text costs at
  most about 4.3 s per 16 MiB; `sin(x)` at the same point, order 12, is summarised in 14 ms.
- Large floats printed in every term: `(1.797...e308 written out)/(1-x)` at order 20000: LaTeX
  (`{:.6}`, exact) 6.7 MB in 0.37 s, the line 0.39 s.
- Code reading: no per-occurrence comparison of names against the variable (the pre-pass compares
  once per distinct node; `expand` reads the dependence map); a symbol occurrence's kind string is a
  transient copy covered by its key's charge; error texts copy at most one function name; the limit
  path copies one rational; the work lists of Display and LaTeX hold at most the depth plus the
  pending arguments, bounded by the input's argument count; `limit()` never reaches the engine, and
  `limit_enhanced` and `limit_with_error` run it once each.

## Files

- `EXP/redteam2-memory/mr.py`: one capped, timed run of `proto9m_rel` through `tw`, `se` or `lim`.
- `EXP/redteam2-memory/famA.py`, `famB.py`: the families above.
- `EXP/redteam2-memory/src/bigpt.rs` (`bin/bigpt_rel`): the REPL line with a big Integer point.
- `EXP/redteam2-memory/inputs/`: the inputs (`A_2000.txt`, `A_8000.txt`, `F6.txt`, `F9_h.txt`,
  `inv_g.txt`, `realbig.txt`, ...).
