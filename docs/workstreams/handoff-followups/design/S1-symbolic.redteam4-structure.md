# S1-symbolic, revision 8: red team 4 (input structure and recursion)

Design attacked: `fu2/design/S1-symbolic.md`, revision 8 (all of it, with the Revision sections, §2.2,
§2.5, §2.10 to §2.16 and §7 in detail). Prototype: `EXP/rev8/proto12/` (EXP = `fu2/design/S1-symbolic`),
used read-only through `bin/proto12_rel`, `bin/proto12m_rel` (counting build), `bin/split_rel`,
`bin/rt2_rel` and `bin/rt2` (the §2.12 parser probe), `bin/dropm` and `bin/replshow_rel`. HEAD is
6e03987 (rustmath-symbolic as at 2e15d1c). Budget 1e8 (`TAYLOR_WORK`; `LIMIT_SERIES_WORK` on the limit
path). Every run had a timeout and an address-space cap of 4 GiB (`tools/r.py`). Release on an 8 MB
thread unless stated.

My files are in `EXP/redteam4-structure/`:

- `build.sh` (the rev8 build line, output in `redteam4-structure/bin/`; I deleted the binaries, each
  rebuilds in a few seconds).
- `src/headprint.rs`, `src/headgp.rs`: HEAD's `Expr::taylor` on a file, the result printed with HEAD's
  Display or in gp syntax (every node in parentheses, rationals as `(p/q)`), for value checks in gp.
- `probes/rt2t.rs`: red team 2's §2.12 probe against proto12 with `TREE_TOO=1` (the result's exact tree
  size, a memoized count by node address). `probes/rt2f.rs`: the same against `fix/`.
- `fix/`: proto12 with one change, the exact node count of a computed kernel (m1, below).
- `tools/treeest.py BIN FILE POINT ORDER`: bisects `TREECAP` to find the driver's computed result tree
  (`render_tree`) and prints it with the exact tree. `tools/fixsafe.py FILES..`: for each random input,
  the exact tree of the fixed build, then the same call with `TREECAP = tree - 1`, which must be
  refused (else the fixed estimate is below the tree), and the outcomes of both builds.
- `tools/gen_r8.py SHAPE N`: shapes that exercise the revision 8 code (seeds, `zero_pow_sign`,
  `small_int_value`, the tower's constant-exponent flag).
- `inputs/`, `out/` (gp scripts `chk_*.gp` and HEAD's results in gp form).

## Result

No hole over the bar in this lens, on the precedent the reviews set for honest refusals. Two holes are
below it. Both are regressions against HEAD on small inputs that HEAD answers correctly in
milliseconds, so a strict reading of (b) would count them; I give the facts for the reviewer to decide.
Every recursion of the engine I could find is bounded as §2.16 says, every cap is checked before the
work it bounds, and Drop, Display and PartialEq of the results held on 2 MB debug threads.

## m1 (below the bar: a constant factor of at most 3 in the tree cap; a regression on inputs HEAD answers): a computed kernel's cached tree counts 3 nodes above its arguments where its expression has 1

**What happens.** §2.2 defines `KernelData::tree` as "the input node's (from the pre-pass), else 3 + the
arguments' effective tree sizes", and the prototype does that (`proto12/konst.rs:1736`,
`mk_kernel_inner`: `args.iter().fold(3u64, ..)`). But a computed kernel's expression adds 1 node above
its arguments' renderings: `Unary(op, c)` for `sin(c)`, `exp(c)`, `log(c)`; `Binary(Pow, c, alpha)` for a
power kept as a kernel. (A radical `n^(1/d)` is 3 nodes in all, a float leaf 1, and an `Inverse`
kernel's expression is P0's rendering itself, its quotient node counted by the monomial.) So every
occurrence of a computed kernel is counted 2 nodes too many, and a chain of k nested computed kernels
is counted about 3k for a tree of k. `render_tree`, the check of `MAX_RESULT_TREE`, sums these cached
sizes, so the cap refuses results whose tree is well under 2^21 nodes. The design states the rule it
breaks (§2.2, `ConstSizes`): "An overestimate would set hints and refuse results for nothing."

**Inputs** (`tools/treeest.py`; `bin/split_rel engine`, `bin/rt2_rel engine` and `head`; heap peak from
`bin/proto12m_rel measure`):

| input (bytes), point, order | `render_tree` | exact tree | ratio | design (`taylor_with_budget`, `series_expansion`) | HEAD's `taylor` |
|---|---|---|---|---|---|
| `sin^1200(x+1+0*(y0+...+y499))` (8.4 KB, `sinDj_1200`), 0, 1 | 2 167 811 | 724 203 | 2.99 | `LimitExceeded(TreeSize)` in 0.05 s, 2.51e7 units, heap peak 1.9 MB, VmHWM 7.8 MB | 0.46 s, VmHWM 275 MB |
| `sin^2000(x+1)` (10 KB), 0, 1 | 3 348 921 | 2 010 865 | 1.67 | `LimitExceeded(TreeSize)` in 0.15 s, 6.7e7 units, 2.8 MB, 10.5 MB | 0.46 s, 386 MB |
| `sin^1700(x+1)` (8.5 KB), 0, 1 | 2 302 971 | 1 454 215 | 1.58 | `LimitExceeded(TreeSize)` in 0.10 s, 4.9e7 units | 0.28 s, 281 MB |
| `x^x^...^x`, 20 carets (41 bytes), 1/2, 2 | 2 334 746 | 1 309 568 | 1.78 | `LimitExceeded(TreeSize)` in 0.03 s, 1.32e7 units, 2.8 MB, 6.1 MB | 7 ms (10.6 ms with the parse), 9.6 MB |
| `cos(x+1)^...^(x+2)`, 100 levels (1.7 KB), 0, 1 | 2 812 370 | 1 763 358 | 1.59 | `LimitExceeded(TreeSize)` in 0.02 s, 1.14e7 units, 0.5 MB | 8.8 ms, 11 MB |

The `0*(y0+...+y499)` term of the first row is free of the variable and only raises the substitution
estimate, so no hint resets the chain and the ratio reaches its bound of 3. Without it the hints of
§2.5 reset the chain from time to time, and the ratio is 1.6 to 1.8. Other families: `x^x^...^x` with
4000 carets at 2, order 0: 1.41; `exp^1500(x+1)` at 0, order 1: 1.55; `(1+sin(x+1))^...^(x+2)` with 60 to
150 levels: 1.85; F_10 at 1: 1.16 at order 0, 1.13 at order 1.

Correctness. HEAD's results are right: `sin^1200(x+1)` (with and without the padding) gives c0 and c1
equal to gp's to 1e-78 (`out/chk_sinD.gp`: `s=1.; p=1.; for(k=1,1200, p*=cos(s); s=sin(s))`), and the
20-caret tower at 1/2 gives c0, c1, c2 equal to gp's to 1e-96 (`out/chk_towx20.gp`: `X=1/2+h+O(h^3);
s=X; for(i=1,20, s=X^s)`). Where the design answers it is right too (`checks/gpres.py`, 300 digits):
`sin^1000(x+1)` 2.7e-312, the 16-caret tower at 1/2, order 2, 1.9e-327, the cos tower with 40 and 70
levels 2.8e-309 and 2.2e-308.

**The fix, measured.** `fix/` counts 1 node above the arguments (2 for a radical, whose exponent is not
an argument). With it the four refused rows above are `Ok` through `taylor_with_budget` and
`series_expansion` (0.02 to 0.15 s, the same units), the 20-caret tower with a tree of 1 494 018 nodes.
The estimate equals the exact tree plus 8 to 68 nodes on every family above, and it is never below it:
on 635 random inputs of red team 3's generator (`r3_s2`, `r3_s3`, `r3_s4`, `r3b_s11`, `r3b_s12`;
`tools/fixsafe.py`; `out/fixsafe2.txt` holds the last 335, the first 300 went to the terminal) every call with `TREECAP = tree - 1` is refused, and every
outcome kind equals proto12's. The fixed build's values equal gp's (the 18-caret tower at 1/2, order 2,
1.2e-327; the cos tower with 70 levels, 2.2e-308).

One side effect needs the hint rule too. The over-count also fed the hint test of §2.5 (a hint when
the normal form's tree exceeds twice the substitution's estimate plus 64), so in proto12 some hints
fired only because the normal form was over-counted, and they made trees smaller. With the exact
count alone (`fix/`), `x^x^...^x` at 2, order 1, loses those hints: 100 carets give a tree of 1 404 999
nodes (proto12: 882 247) and kernel depth 202 (proto12: 123), and the last admitted tower falls from 118
carets to 110. A hint test on exact trees that needs only a margin, `normal form > estimate + HINT_MIN`
(`fix2/`, `ser.rs` `maybe_hint` and `hint_exp_arg`), recovers them and more: towers at 2 are admitted to
135 carets, every row of the table above is `Ok`, the gp oracle (`rev2/gpcheck3.py`) gives 0 failures,
the hint scan of the oracle and displaycheck corpora still finds one hinted input
(`(sin(1)^2+cos(1)^2+x)^100`, as `hintscan_r8.txt`), 246 random inputs (`r3_s2`, `r3b_s13`;
`out/fixsafe3.txt`) are never under-counted and keep their outcome kinds, and the values equal gp's
(the 100-caret tower at 1/2, order 1, 5.6e-309; `sin^500(x+1)` 5.4e-312).

**Over the bar.** No, by the bar's own text: a constant factor (at most 3) in a cap. But each row is an
input HEAD answers correctly in under 0.5 s and 400 MB, and the design refuses it although its
result's tree is under the cap. If the reviewer counts honest refusals under (b), this is over.

**Sections.** §2.2 (`KernelData::tree`, `ConstSizes`), §2.15 (the `MAX_RESULT_TREE` row: its checks
and its margin column, whose figures may be estimates or exact counts), §2.16 (the `mk_kernel` row).

**Fix that closes the class** ("a cached size that over-counts what it describes"). Count, per kind of
kernel, the nodes its expression builds above its arguments' renderings (1 for an operator or a
power, 3 in all for a radical, 1 for a float leaf, 0 for an `Inverse` kernel, whose `Div` node
`ConstSizes` counts), and keep the input node's tree for a kernel made from an input node. Then base
the hint test on the exact trees with a margin rather than a factor (`fix2/`: a hint when the normal
form's tree exceeds the substitution's estimate by more than `HINT_MIN`; the final check, a hint only
when its tree is smaller, stays), so that no hint depended on the over-count. Add a unit test that
builds one kernel of each kind and compares its cached tree with a memoized count of its expression,
and a property check in the random tests of §4.1: the exact tree of every result is at most
`render_tree` and at least `render_tree` minus 4 n minus the point's tree. State the cap's margins from
exact counts.

## m2 (below the bar by the precedent of review 8 m4; a false claim, and a refusal of classic inputs HEAD answers in milliseconds): towers and nested functions give result trees polynomially larger than HEAD's, and the cap refuses them from about 22 levels

**What happens.** §2.2, §2.15 and §7 item 8 say what the tree cap still refuses: "a normal form that
itself has exponentially many terms" (the x coefficient of F_k). Towers refuse much earlier, with
normal forms of few terms. At a point a, the value of a k-level tower `x^x^...^x` is a chain of
computed kernels `E_j = exp(log(a) E_(j-1))`, whose expressions nest: E_j has a tree of about j nodes.
The order-n coefficient, in the expanded normal form, names these kernels at every occurrence: at order
1 it has about k terms with up to k factors E_j each, so about k^2 kernel occurrences and a tree of
about k^3 nodes; at order 2 about k^4 (measured: 545 812 nodes at 16 carets, 879 609 at 18, 1 309 568 at
20). The distinct nodes stay few (59 458 at 20 carets, order 2). HEAD keeps the derivative nested,
`T_k (T_(k-1)' log x + T_(k-1)/x)`, and its result is about k^2. No hint applies: the hints of §2.5
replace a value at the point whose tree is much larger than its substitution, and here E_j's tree is
about the size of `a^a^...^a`. So the refusal is the same class as F_k (a representation without
shared products), but polynomial, and reached by 45-byte inputs.

**Inputs** (with the exact count of m1's fix, `bin/rt2f_rel`, unless stated; proto12 refuses each
refused row too, and from fewer levels; the exact tree with the cap raised, `TREECAP`):

| input (bytes), point, order | design | exact tree | HEAD |
|---|---|---|---|
| `x^x^...^x`, 21 carets, 1/2, 2 | `Ok` in 0.04 s, 1.88e7 units | 1 867 473 | 9.5 ms, 10 MB |
| the same with 22 carets (45 bytes) | `LimitExceeded(TreeSize)` in 0.05 s, 2.0e7 units (also with `fix2/`) | above 2^21 | 8.0 ms, 11 MB |
| 30 carets (61 bytes) | `LimitExceeded(TreeSize)` in 0.17 s, 8.5e7 units (near the budget) | | 36 ms, 22 MB |
| `x^x^...^x`, 200 carets (401 bytes), 2, 1 | `LimitExceeded(TreeSize)` in 0.13 s, 9.3e7 units | 6 216 611 (`fix2/`) | 21 ms, 18 MB |
| `x^x^...^x`, 20 carets, 2, 3 (proto12) | `BudgetExceeded` in 0.25 s | | 0.30 s, 184 MB |
| `cos(x+1)^...^(x+2)`, 150 levels (2.5 KB), 0, 1 | `LimitExceeded(TreeSize)` in 0.06 s, 3.9e7 units | 5 854 633 | 19 ms, 21 MB |
| `(1+sin(x+1))^...^(x+2)`, 150 levels, 0, 1 (proto12) | `LimitExceeded(TreeSize)` in 0.05 s, 4.0e7 units | 8 138 126 | 28 ms, 29 MB |

With `fix2/` (m1's exact count and margin hint test) the last admitted towers are 21 carets at 1/2,
order 2, and 135 carets at 2, order 1 (proto12: 19 and 118).

Every refusal takes at most 0.25 s and 26 MB of VmHWM, through `taylor_with_budget` and
`series_expansion`; the limit path (order 5) is refused by its budget. HEAD's result of the 150-level
cos tower equals gp's c0 and c1 to 1e-77 (`out/chk_ctow150.gp`). Where the design answers, its result
is right but much larger than HEAD's: the REPL text of the 12- and 16-caret towers at 1/2, order 2, is
684 KB and 2.4 MB (`bin/replshow_rel`, under `OUTPUT_BYTES`, so printed), HEAD's 31 KB and 67 KB.

**Over the bar.** No, by the precedent of review 8 m4, which rated the F_12 to F_15 refusals (HEAD
0.2 ms) as minor because §7 item 8 records the class. But §7 item 8 records only the exponential case,
so these refusals are not recorded, and `x^x^...^x` is a classic input. A strict reading of (b) would
count them.

**Sections.** §2.2 (the paragraph on kernel expressions: "What the cap still refuses is a normal form
that itself has exponentially many terms"), §2.15 (the `MAX_RESULT_TREE` row, "refuses (measured)"),
§5.3, §7 item 8.

**Fix that closes the class** ("a result whose tree is larger than its DAG by the number of times the
expanded normal form names each kernel"). In the design: say that the cap refuses any coefficient
whose expanded normal form names deep kernels many times, polynomially as well as exponentially
(towers: about k^(n+2) nodes at order n), add the tower families above to §5.3 and to §7 item 8 with
their measured limits (22 carets at order 2, about 120 at order 1), and name the remedy the HANDOFF row
already names: coefficients that keep products, or consumers memoized by `Arc` address with a cap on
the DAG instead of the tree. Until then the refusal is honest, and m1's fix moves the limit from 20 to
22 carets at order 2.

## Families tried that held

- **Recursion scan** (red team 3's `callgraph.py` on `proto12/konst.rs`, `ser.rs`, `ball.rs`, `api.rs`,
  `display.rs`, `replshow.rs`): the only self-calls are constant-depth (`kpow_const` once for a perfect
  power, `simplify_binary`'s Div arm, `Ball::unary`), log-depth (`balanced_arc`, `bal_depth`) or
  bounded (`small_int_value`: 8 levels, 64 nodes, `to_i64` is O(1) on a big literal). The large SCC the
  scan prints comes from name collisions (`inv`, `log`, `mul` in three files). `display_iter` and
  `latex_iter` are loops; `latex_head` is HEAD's writer, kept for comparison.
- **Parse and call on one 2 MB debug thread at the parser's limit** (`bin/rt2 both`, with
  `DISPLAY_TOO` and `EQ_TOO`): `x^x^...^x` with 1430 carets at 2, orders 0 and 1 (kernel depth 1750,
  result depth 1559, printed and compared), `sin^155(x+1)` at 0, order 1, the 100-level cos tower,
  through `taylor_with_budget`, `series_expansion` and the limit path: no overflow; outcomes `Ok`, the
  budget or the tree cap.
- **The revision 8 code at depth 150 on 2 MB debug** (`tools/gen_r8.py`): nested `sign(...(x^2+1))`,
  products of seeded factors `(x-x+k)` inside `sin`, towers of zero bases `(x-x)^...^(x+1)`,
  integer-valued exponents `(-2)^((-3+1)*(-1))` in nested kernels, seeded and `sign` towers at 1,
  sums of constant zero powers `(x-x)^(pi+k)`: no overflow, at most 1.7 s debug. Values: `x + sign^150`
  gives `1 + x`, the zero-base tower `1 + x`, the zero powers `x` (all the rule of §2.5), and the
  integer-exponent nest and the seeded product equal gp's (maxreldiff 0).
- **The same code at scale, release, 24 to 180 KB**: 5000 zero powers (89 KB) `Ok` in 28 ms with 8.9e6
  units, the limit `Finite(0)`; 3000 nested integer-exponent kernels (72 KB) `Ok` in 17 ms, kernel depth
  6005; 9000 seeded factors (72 KB) `Ok` in 29 ms; a 30 000-level zero-base tower (180 KB) `Ok` in 70 ms;
  3000 wide integer-exponent kernels under a divisor (98 KB) `Ok` in 57 ms; F_30 with a seeded summand
  `sign((x-1)^2) - 1` at every level: order 0 `Ok` in 0.4 ms, order 1 `BudgetExceeded` after 0.5 s.
- **Drop of engine results with the design's Drop** (`bin/dropm file:..`, modes `design` and `tls`, 2 MB
  debug): `sin^1500(x+1)` (depth 1514, tree 1.1e6), the 70-level cos tower, the 16-caret tower at 1/2,
  order 2, and the padded `sin^1000`: at most 17.3 KB of stack and 2.7 ms; in `tls` mode up to 2060
  fresh leaves, the same stack.
- **Tree accounting has no under-count**: on every family the computed `render_tree` is at least the
  exact tree (m1 is the opposite error). The hint and seed trees come from the substitution's exact
  counts, the point is counted per occurrence of h, and `ConstSizes::tree` matches `mono_expr`.
- **Caps before work** (code reading of proto12): the factor cap before each push in `mono_mul`; the
  term cap before each push in `merge2`, `sum_many` and the accumulator of `mul`; the kernel cap in
  `intern` before an id is given; the depth cap before the key is written; the tree cap in the driver
  before `render_units` and the rendering, and in `undecided` before the constant is rendered. The
  single-term `inverse` keeps the monomial's length, so it needs no factor check.
- **Address-keyed memos**: `DEP` and `SUBST` hold nodes of the borrowed input and the point; the hint
  table holds a clone of each constant it keys; the ball memo lives for one walk; `UNARY` is keyed by
  kernel id; the rendering memo of `series_expansion` by constants the Ser holds. No key can be reused
  by a new allocation while its entry lives. `SUBST` persists across attempts, which is sound: a
  regular subtree has no precision loss (every divisor has valuation 0), so its flag and its
  substitution do not depend on the attempt, and an exact constant is the same in every attempt.
- **Drop order of the engine's data**: the series and the hints are dropped before the kernel table
  (`api.rs`), the table newest first, and a kernel's `unary` and `inv_of` constants name older kernels
  only, so no drop recurses through a kernel chain.
- **Huge literals in integer-valued exponents of kernel nodes** (`(2+sin(k))^(L - L + 2)` with 1500- and
  45 000-digit L, 90 KB): `UndecidedZero` in under 1 ms, the class of red team 4 numeric m3 (an
  input-syntax rendering whose exact zero has a wide ball); HEAD's parser cannot read these literals.
- **§2.17**: the new `distribute_negation` recurses over the same nodes as HEAD's, so it adds no depth.
- **Invariants table**: every loop I found has a row, except trivial O(1) helpers (`exact_constant`,
  `regular_val`, `subst_tree_estimate`) and the per-visit child vectors of the pre-pass (`dep_pass4`'s
  `children`, three short allocations per node, bounded by the input's fan-out, uncharged). The port
  should name the latter in the pre-pass row.

## Files

`EXP/redteam4-structure/`: `build.sh`, `src/headprint.rs`, `src/headgp.rs`, `probes/rt2t.rs`,
`probes/rt2f.rs`, `probes/rt2f2.rs`, `fix/` (proto12 with the exact kernel count, `konst.rs:1736`),
`fix2/` (also the margin hint test, `ser.rs`), `tools/r.py`, `tools/fixsafe3.py`, `tools/hintscan_protofix*.py`,
`tools/treeest.py`, `tools/fixsafe.py`, `tools/fixsafe2.py`, `tools/gen_r8.py`, `inputs/` (`sinD_*.txt`,
`sinDj_*.txt`, `towx_*.txt`, `ctow_*.txt`, `expD_1500.txt`, `r8_*.txt`, `biglitpow_*.txt`), `out/`
(`chk_sinD.gp`, `chk_sinDj.gp`, `chk_towx20.gp`, `chk_ctow150.gp`, HEAD's results in gp form,
`fixsafe2.txt`, `fixsafe3.txt`, `hintscan_fix*.txt`, `gpcheck3_fix2.out`). `build.sh release <src> <name>`
rebuilds `headprint`, `headgp`, `rt2t`, `rt2f`, `rt2f2`, `protofix` (from `fix/main.rs`) and `protofix2`
(from `fix2/main.rs`) in a few seconds; I deleted the binaries.
