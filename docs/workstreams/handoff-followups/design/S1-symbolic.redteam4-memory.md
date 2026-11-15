# S1-symbolic, revision 8: red team 4 (memory and ownership)

Design attacked: `fu2/design/S1-symbolic.md` (revision 8, all sections, with the Revision sections,
the caps table of §2.15 and the Invariants and "Memory" tables of §2.16). Prototype:
`EXP/rev8/proto12/` (EXP = `fu2/design/S1-symbolic`), its release binaries `bin/proto12m_rel`
(counting allocator), `bin/proto12_rel`, `bin/replshow_rel` and the debug `bin/proto12`, used
read-only. My scripts are in `EXP/redteam4-memory/`:

- `mr4.py`: one run of `proto12m_rel` with `RLIMIT_AS` 6 GiB and a timeout, through `ENTRY=tw`
  (`taylor_with_budget`, budget 1e8), `se` (`series_expansion`, then `to_expr`) or `lim`
  (`limit_via_series`, `LIMIT_SERIES_WORK`). It prints the time, the outcome, the units, the heap
  peak per unit, the cumulative allocation per unit and VmHWM per unit.
- `edge.py`: a bisection on the order (or a size) for the largest call the budget admits.
- `scan1.py`, `scan2.py`, `fam1.py`, `fam2.py`: the families below (inputs in `inputs/`).
- `src/head_lwe.rs` (built by `build.sh`, the rev8 script with the output in `EXP/redteam4-memory/bin/`):
  HEAD's `Expr::limit_with_error` on one input.

HEAD is 6e03987 (rustmath-symbolic as at 2e15d1c). Numbers are release on the main thread, point 0,
budget 1e8, unless stated. Every run had a timeout and an address-space cap.

## Result

No hole over the bar. I found no input, through `taylor_with_budget`, `series_expansion` with
`to_expr`, the limit path or the REPL taylor line, that comes near 512 MB or 20 s. The table "Memory"
holds: every object I checked in the code is charged at most about 3 bytes per unit, the counted heap
peak stays at or below 2.83 bytes per unit, and VmHWM tracks it. The tree cap (2^21 nodes) also stops
the results whose rendering dominates the memory at about 120 to 170 MB, before the budget. Two issues
are below the bar.

## m1 (below the bar): a stale measured maximum

What happens. The memo of revision 7rt3 renders each distinct coefficient once, by the constant's
address. When the coefficients are distinct constants (each a new scaling of one long monomial),
every term still renders its own copy of the monomial. Each copy is `L` `Mul` nodes of 96 bytes for
`EXPR_NODE = 32` units, the row "Expr node of a rendering" at 3.0 bytes per unit, and the engine's own
copy of the monomial adds 16 bytes per 6 units. So the heap per unit rises towards that row's value.

Inputs (`fam1.py`, `fam2.py`): `(-1)*(qa*qb*...)/(1-x)^2` and `(qa*qb*...)/(1-x)^2` with N short
distinct names, at the largest admitted order.

| input | order | units | heap peak | bytes per unit | VmHWM | time |
|---|---|---|---|---|---|---|
| N = 1000 (4.3 KB), both signs | 1043 | 4.16e7 | 117.8 MB | 2.83 | 122 MB (2.93 per unit) | 0.12 s |
| N = 4000 (19 KB) | 258 | 4.21e7 | 118.2 MB | 2.81 | 123 MB | 0.13 s |
| N = 8000 (39 KB) | 129 | 4.45e7 | 120.6 MB | 2.71 | 127 MB | 0.14 s |
| N = 300 | 3421 | 4.36e7 | 116.4 MB | 2.67 | 120 MB | 0.13 s |
| `(qa^2*...)/(1-x)^2`, N = 1000 | 521 | 6.05e7 | 159.5 MB | 2.64 | 164 MB | 0.14 s |

The next order of each is `LimitExceeded(TreeSize)`, so the tree cap binds before the budget. The
same numbers through `series_expansion`; the limit path `Unknown` (the constant term is not
rational) with at most 22 MB. Debug on a 2 MB thread (`STACK=2097152`, HEAD's recursive Display and
the work-list one): `Ok`, no overflow, at most 168 MB. The REPL line (`replshow_rel`, `LINEAR=iter`,
64 MiB thread): N = 1000 at 1043 prints 8.7 MB of text and 10.8 MB of LaTeX in 0.25 s, VmHWM 138 MB.

Correctness: gp `Vec(-(qa*qb*qc)/(1-x)^2+O(x^5))` is `[-qc*qb*qa, -2*qc*qb*qa, ..., -5*qc*qb*qa]` and
`Vec((qa^2*qb^2)/(1-x)^2+O(x^4))` is `[qb^2*qa^2, 2*qb^2*qa^2, 3*..., 4*...]`, the prototype's
`coeffs` (`DISPLAY=iter`).

Over the bar? No (122 MB). But §0 item 5 says "The largest measured heap peak is 2.6 bytes per unit",
§2.7 and §2.16 say 2.59 at a refusal and 2.29 for an admitted call, and the `TAYLOR_WORK` comment of
§3.1 says "a heap peak of at most 2.6 bytes per unit". This family gives 2.83 for an admitted call.
The bound of the table (3 bytes per unit) is not passed.

Fix. State 2.83 bytes per unit (`(-1)*(M_1000)/(1-x)^2` at order 1043) as the measured maximum, and
add the family to §5.3 and to the port's memory check. The class is "distinct coefficients that each
render a long monomial": the memo by address cannot share them, the row "Expr node of a rendering"
bounds them at 3.0, and the tree cap bounds them at about 2^21 nodes (about 200 MB of nodes). No code
change is needed.

## m2 (below the bar): `limit_with_error` renders every stored term to read only the precision

What happens. §2.13 gives `limit_with_error` the same budget "through `series_expansion`, then the
`BigO` of the result". The `BigO` is `O(h^precision)` with precision = order + 1 whenever the
expansion succeeds, so no stored term is read. But `series_expansion` charges `render_units`, checks
`render_tree` and renders every coefficient. This is red team 3's class m3 ("the limit path rendered
every stored term to read the first"), which revision 7rt3 fixed for `limit_via_series` only.

Inputs: `1 + x*S_k` with `S_k = (1+cos(1))*...*(1+cos(k))` (`inputs/xS_17.txt`, `xS_18.txt`), point 0,
order 5, `LIMIT_SERIES_WORK`. `ENTRY=se` is the call `limit_with_error` makes.

| input | entry | outcome | units | VmHWM | time |
|---|---|---|---|---|---|
| `1 + x*S_17` | se, order 5 | `LimitExceeded(TreeSize)` | | 71 MB | 0.19 s |
| the same | lim (first term) | `Finite(1)` | 3.19e7 | 71 MB | 0.18 s |
| `1 + x*S_18` | se, order 5 | `LimitExceeded(TreeSize)` | | 123 MB | 0.34 s |
| the same | lim | `Finite(1)` | 5.41e7 | 123 MB | 0.32 s |

So `limit_with_error` returns `(Finite(1), None)` where the error term `O(x^6)` is known. HEAD's
`limit_with_error` aborts at 4 GiB on both inputs (`bin/head_lwe_rel inputs/xS_17.txt 5`), so this is
not a regression, and `None` is honest. No caller in the workspace uses `limit_with_error` (grep:
only `limits.rs`).

Fix. Run the driver for `limit_with_error` without rendering, as the first-term entry does (charge
and bound no term, or the first only), and build `O(h^(order + 1))` from the precision. The class is
"an entry point that renders terms it does not read"; §2.13 and the row of §2.16 should say which
entries render.

## Families tried that held

All release, budget 1e8 (`LIMIT_SERIES_WORK` for `lim`), through `tw`, `se` and `lim` where stated.

- Budget edge of 28 families (`scan1.py`, `scan2.py`): `exp(x)`, `sin(x)`, `tan(x)`,
  `(1+x)^(1/3)`, `log(1+x)`, at 0, at 1/3 and at a symbol a; `(x+a)^k`, `(x+sin(1))^k`,
  `(1+x)^(10^100)`, `(1+x)^(10^1000)`, `(1)/(1-a*x-b*x^2)`, `(1)/(1-x-a*x^2-b*x^3)`,
  `(1)/((1-a*x)*(1-b*x)*(1-c*x))`, `exp(x+a*x^2)`, `log(1+x+a*x^2)`, `sin(x+a*x^2)`, `exp(a*x)*exp(b*x)`.
  The heap peak is at most 1.46 bytes per unit (`sin(x)` at a, order 7531) and VmHWM at most 146 MB.
  The cumulative allocation reaches 44.5 bytes per unit (`(1+x)^(1/3)` at 3906, gcd temporaries)
  with VmHWM 15 MB, so the uncharged transients are reused, not kept.
- Long literals: `(L*x)/(1-x)^2` with a 99 990-digit L at order 2000: `Ok`, 2.61 bytes per unit of
  heap, VmHWM 179 MB; at 3000 to 4000 refused with at most 259 MB of heap and 147 MB of VmHWM.
- The largest admitted VmHWM stays `x + S_19*sin(x)` at order 0: 230 MB, the same through `tw`,
  `se` and `lim`.
- A 50 KB symbol as the point: `exp(x)` at order 2000, 6 MB; the hinted `F_10` and `F_30` at that
  point, at most 6 MB; the limit path `Unknown`, or the budget, with at most 10 MB. The point is
  shared by every substitution and by h.
- A point with a large tree, `g(1,...,1)` with 49 000 arguments (98 KB): order 0 `Ok` with 16 MB;
  `exp(x)` there from order about 42 `LimitExceeded(TreeSize)` in 0.04 s, because h names the point
  in every term. HEAD's `taylor` aborts at 4 GiB from order 20 (`bin/head_rel`), so no regression.
- REPL line (`replshow_rel`): `(1)/(1-a*x-b*x^2)` at 654 prints 10.4 MB of text and 10.2 MB of
  LaTeX, 0.38 s, VmHWM 127 MB; the long-monomial families at the tree cap at most 175 MB.
- Debug on a 2 MB thread through the three entry points: the N = 16 000 monomial family (79 KB),
  `(qa^2*...)/(1-x)^2` with 3000 names, and `g(1,...,1)` as the input: no overflow, at most 168 MB.
- Code reading of the copies at the boundaries:
  - `seed_const` charges `SUBST_NODE + expr_units(c)` before it renders, once per input node and
    expansion (memo by address).
  - `maybe_hint` builds the substitution before its size test, but the nodes are charged and
    memoized.
  - The copies of `subst_or_node` and of the point for the substitutions are charged with
    `node_payload_units`, which counts the digits of an integer or rational and the name and
    arguments of a call.
  - The rendering memo of `series_expansion` holds one `Arc` per distinct constant.
  - `to_expr` builds at most 4 nodes per term.
  - The limit path copies one rational.
  - `exact_constant` allocates nothing.
  - On every path the series is dropped before the kernel table, and the hints are cleared before it.
  - The per-call copies of the point and the name are the same as red team 3 found (wording only).
- Not re-reported, as recorded in §7 item 9: a REPL cell that calls taylor many times keeps each
  output. `print(taylor(..))` lines also add their text to the cell's `stdout`. This is J1b's
  per-cell policy.
- `series_coefficients` copies a bare kernel's top node per stored coefficient (a 99 KB name per
  entry). It has no budget, as documented in §2.16, and HEAD's `taylor` fails earlier on such inputs.

Correctness samples against gp (`gp -q`, exact series): `Vec(1/(1-a*x-b*x^2)+O(x^6))`; the
x^30 coefficient of `(1+x)^(1/3)`, -2620049545720783352/984770902183611232881; the coefficient 12 of
`1/(1-x)` at 1/3, 1594323/8192; the x^6 coefficient of `1/(1-x-a*x^2-b*x^3)`; and the two families of
m1. Each equals the prototype's `coeffs`.

## Files

- `EXP/redteam4-memory/mr4.py`, `edge.py`, `scan1.py`, `scan2.py`, `fam1.py`, `fam2.py`,
  `fam1names.py`, `inputs/` (`nM1_*.txt`, `M1_*.txt`, `M2_*.txt`, `xS_17.txt`, `xS_18.txt`,
  `Gpt.txt`, `Yname.txt`, `L99990.txt`, `ab.txt`).
- `EXP/redteam4-memory/src/head_lwe.rs` and `build.sh` (`build.sh release src/head_lwe.rs head_lwe`
  rebuilds it in a few seconds; the binary was deleted after the runs).
