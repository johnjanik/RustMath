# S1-symbolic, revision 7rt2: red team 3 (input structure and recursion)

Design attacked: `fu2/design/S1-symbolic.md`, revision 7 red-team round 2 (all of it, with §2.2,
§2.5, §2.8, §2.10, §2.11, §2.14, §2.15, §2.16 and §7 in detail). Prototype: `EXP/rev7rt2/proto10/`
(EXP = `fu2/design/S1-symbolic`), used read-only through its binaries `bin/rt2_rel`, `bin/rt2`
(red team 2's §2.12 parser probe linked against proto10; `engine` parses on a 1 GiB thread and runs
the entry point on a STACK thread, `both` runs both on one thread, `head` runs HEAD's `Expr::taylor`).
Budget 1e8 (`TAYLOR_WORK`, or `LIMIT_SERIES_WORK` on the limit path). Every run had a timeout and an
address-space cap (4 GiB, 6 GiB for the largest tree walks). Release on an 8 MB thread unless stated.
HEAD is 2e15d1c.

My files are in `EXP/redteam3-structure/`:

- `build.sh` (the rev7rt2 script, output in `redteam3-structure/bin/`).
- `src/dropm.rs`: the design's `Drop for Expr` of §2.14 (the code of `EXP/rev1/sym_v10`, which §2.14
  prints) on a mirror enum with HEAD's field types, against the derived glue (HEAD) on the same DAG.
  It converts an engine result with its sharing (one new `Arc` per distinct allocation) and drops it on
  a STACK thread. The prototype links HEAD's `Expr`, and every probe forgets its results, so this is
  the first measurement of the drop the port ships.
- `src/consumers.rs`: a REPL-style line `consumer(taylor(F, x, a, n))` with HEAD's taylor or with
  proto10's `taylor_with_budget`, then HEAD's `simplify`, `substitute` or the derived `PartialEq`, or
  a memoized count of the result's tree nodes.
- `tools/gen3.py` and `tools/gpcheck3.py`: random inputs that stress the work stacks (towers with
  series and constant levels and negations between levels, `* /` chains with constant and series
  divisors and zero factors, mixed sums, calls), compared with gp on the probe's own parsed tree.
- `tools/shapes3.py`, `tools/r3.py`, `tools/bsdrop.py`, `tools/callgraph.py`; inputs in `inputs/`,
  outputs in `out/`.

## H1 (over the bar, (a) and (b)): `0^r` is decided from the syntax of the base, and from a rational test of r

**What happens.** §2.5 chooses the rule of `Pow(l, r)` from the pre-pass flag "depends on the
variable", not from the value of l. When l mentions the variable but its series is the exact zero
(`x*0`, `x - x`, `(x+1)^2 - x^2 - 2*x - 1`, a product with a zero factor), the rule "both depend on the
variable" applies: `exp(expand(r) * log(expand(l)))`, and `log` of the exact empty series is
`DivisionByZero("log(0)")`. The design already routes an exponent that expands to an exact constant
to the constant rule (review 1, m1: `x^(x-x+2)`), but not a base. The value is `0^r = 0` wherever
`Re r > 0` at the point, which the rule for a base free of the variable already returns.

The rule for `0^r` itself (`pow_const_base`, §2.5 "Zero: 0^r is the exact 0 when r has valuation >= 0
and a positive rational constant term, else NotLaurent") reads `val`, which §2.1 defines as a lower
bound when the lead is undecided, and accepts only a rational constant term. So an exponent whose
constant term is positive but not rational (`2^x` at 1 has `exp(log 2)`, `x + sqrt(2)` at 0) or whose
first stored term is an undecided zero gets `NotLaurent("0^r with r not positive at the point")`, a
decided claim that the expression has no Laurent expansion.

**Inputs** (release, 8 MB; `tools/r3.py bin/rt2_rel engine FILE POINT ORDER 8388608 tw|se|lim`; every
run 0.04 to 0.3 ms, at most 1.8e4 units, VmHWM 3 MB; the same outcome on 2 MB debug with `bin/rt2 both`;
`out/zerobase.txt`):

| input, point, order | `taylor_with_budget` and `series_expansion` | limit path | HEAD's `taylor` | correct (gp) |
|---|---|---|---|---|
| `exp(x) + ((x+1)^2 - x^2 - 2*x - 1)^(x+1)`, 0, 3 | `DivisionByZero("log(0)")` | `Unknown` | `1 + x + x^2/2 + x^3/6` | `exp(t+O(t^4))`: the same |
| `cos(x) + (x*0)^(1+x^2)`, 0, 3 | `DivisionByZero("log(0)")` | `Unknown` | `1 - x^2/2` | `cos(t+O(t^4)) = 1 - 1/2 t^2 + O(t^4)` |
| `(x-x)^x`, 1, 2 | `DivisionByZero("log(0)")` | `Unknown` | `0` | 0 |
| `1 + (x*0)^(2^x)`, 1, 2 | `DivisionByZero("log(0)")` | `Unknown` | `1` | 1 |
| `3^(1/3)*+-3^-((1+x)^+3^--(2/3)^(1/3)*0)^x`, 2, 1 (found by `gen3.py`) | `DivisionByZero("log(0)")` | `Unknown` | `(3^(1/3)) * -1` | `-3^(1/3)` |
| `x + 0^(2^x)`, 1, 2 | `NotLaurent("0^r with r not positive at the point")` | `Unknown` | `1 + (x - 1)` | x |
| `x + 0^((sin(1)^2+cos(1)^2-1)/x + 2)`, 0, 2 (the exponent is 2; its stored `x^-1` term is an undecided zero) | `NotLaurent(..)` | `Unknown` | `log(0)` and `0/0` terms | x |
| `x + 0^(x+sqrt(2))`, 0, 2 | `NotLaurent(..)` | `Unknown` | `log(0)` and `0/0` terms | x |

The random check found this class 3 times in about 1100 inputs (`out/r3_s*.out`, `out/r3b_s*.out`).

**Over the bar.** Yes. On the first six rows HEAD returns the correct series and the design returns an
error (b). The error is a false decided claim (a): `DivisionByZero` is documented as "the expression
divides by an exact zero" and the REPL maps it to "ValueError" (a property of the input, §3.1), and
`NotLaurent` says that a subexpression has no Laurent expansion. The `x^-1` row reports an undecided
valuation as a decided one, which the review bar names. It is cheap: none of these inputs costs more
than 0.3 ms.

**Sections.** §2.5 (the `Pow` rules, "else (both depend on the variable)", "Zero: 0^r ..."), §2.4
(`log`: exact empty is `DivisionByZero`), §2.9 (`DivisionByZero`, `NotLaurent` doc comments), §3.1
(the REPL's error map).

**Fix that closes the class.** Choose a rule of `Pow` from the series, not from the syntax, on both
sides: after expanding a base that depends on the variable, if its series is an exact constant
(`exact_constant(f)`, the test the design already applies to the exponent), take the rule for a
constant base with that constant. And decide `0^r` with the engine's own tests instead of a rational
test: the lead status of r (an undecided lead below exponent 0 is `UndecidedZero(lead)`, never a
decided valuation), then the sign of `Re r0` by `split_imag` and an enclosure of A (rule 5 of the zero
test). `Re r0` decided positive and valuation decided >= 0 gives the exact 0; a decided pole of r, or
`r0` decided 0, gives `NotLaurent`; `Re r0` decided negative gives `DivisionByZero("0 to a negative
power")`; anything undecided gives `UndecidedZero`. Add the rows above to §4.1 with gp values.

## H2 (over the bar under (b) if the consumers of a REPL value count): results whose trees are exponential in their DAG make every tree-walking consumer exponential

**What happens.** A kernel the engine makes from computed values renders its arguments' normal forms
and refers to each sub-kernel's expression through `Arc` (§2.2). When a normal form names a kernel more
than once (any exact power or product of a sum that contains it: `(1 + K)^2 = 1 + 2 K + K^2`), the
kernel's expression names the sub-kernel's expression twice, and a nest of k such levels has a tree of
about 2^k times its DAG. §2.2, §3.4 and §7 item 8 treat this as a text-size problem, which J1b's display
budget summarizes. But the result is a value of the REPL, and the REPL's other arms walk it as a tree
and copy it: `simplify` (`repl.rs:2045`) and `subs` (`:2085`, through `symbols()` and `substitute`),
measured below, and the other arms that walk an `Expr` (`expand` at `:2052`, `diff`; not measured),
and the derived `PartialEq` in Rust code. HEAD's results of the same inputs are
small trees, so these consumers were instant at HEAD.

**Inputs** `F_k = (1+exp((1+exp(...(1+exp(x))^2...))^2))^2` with k levels, at the point 1
(`inputs/nexp_k.txt`; `bin/consumers_rel MODE FILE 1 ORDER CONSUMER`, release, 64 MiB thread):

| k (bytes), order | HEAD: taylor, result tree | design: taylor, result tree (distinct nodes) | consumer of the result | HEAD | design |
|---|---|---|---|---|---|
| 10 (112), 1 | 0.12 ms, 605 nodes | 5.3 ms, 2.4e6 units, 1.04e7 nodes (22 613) | `simplify` | 18 us, 3 MB | 0.83 s, 960 MB |
| 10 (112), 1 | | | `substitute(x, 2)` | 15 us, 3 MB | 0.81 s, 960 MB |
| 12 (134), 1 | 0.15 ms | | `simplify` | 25 us, 3 MB | aborts at the 4 GiB cap |
| 14 (156), 1 | 0.21 ms, 1125 nodes | 0.16 s, 4.6e7 units, 2.68e9 nodes (491 637) | `simplify` | 35 us, 3 MB | aborts at the 4 GiB cap |
| 22 (244), 0 | 28 us, 111 nodes | 0.18 ms, 5.1e4 units, 4.19e7 nodes (177) | `simplify` | 3 us, 3 MB | 2.95 s and 3.84 GB uncapped; aborts at 4 GiB |
| 22 (244), 0 | | | `substitute` | 3 us | aborts at 4 GiB |
| 26 (288), 0 | | | `PartialEq` with a clone | 2 us | 1.7 s |
| 30 (332), 0 | 30 us, 151 nodes | 0.20 ms, 7.0e4 units, 1.07e10 nodes (241) | `PartialEq` with a clone | 2 us | 28.7 s |

`Expr` derives `PartialEq` without `Eq`, so `Arc<Expr>` has no pointer-equality shortcut, and a result
compared with its own clone walks the whole tree. HEAD's order-1 result is correct (k = 2: relative
difference 0 in both coefficients against gp's `f(1+t+O(t^2))`; the design's coefficients pass
`tools/gpcheck3.py` against gp too). The taylor call itself stays within the budget in every row; the
REPL prints its result as J1b's summary (its text passes `OUTPUT_BYTES`), where HEAD prints the answer.

**Over the bar.** Through `taylor_with_budget` alone, no: the call is fast and small. As a REPL line,
`simplify(taylor(F_12, x, 1, 1))` (about 160 bytes) aborts the kernel for lack of memory, and HEAD
answers it in 0.2 ms with 3 MB; that is (b). The consumers are HEAD's code, so the reviewer may hold
that only the line, not the entry point, regresses; red team 1's H2 (Display of deeper results) was
counted and fixed on the same reasoning.

**Sections.** §2.2 (kernel expressions of computed values), §2.8 (rendering), §2.15 (caps; the
consumers named there are judged by depth only), §3.4 (J1b's DAG case covers the display only), §7
items 5 and 8.

**Fix that closes the class.** Bound the tree, not only the depth and the DAG. Cache in each kernel
and each constant the size of its rendered tree (saturating; O(1) per creation from the arguments'
cached sizes, as the depth is cached: `1 + sum over factors of multiplicity x tree(kernel)` plus the
monomial and sum nodes), and add a fifth structural cap `MAX_RESULT_TREE` (for example 2^21 nodes,
about what HEAD's `simplify` copies in 200 MB at the measured 92 bytes per node), checked in
`mk_kernel` before a computed kernel's expression is built and in `render_units` before the result is
rendered, with `LimitExceeded(TreeSize)`. Then every consumer of a result is bounded by the cap times
its per-node cost, and the REPL text by J1b. The alternative, consumers memoized by `Arc` address
(`simplify`, `substitute`, `symbols`, `PartialEq` with `Arc::ptr_eq` and a visited pair set), is
outside S1's item. Either way §7 item 8 should say that the cost is not only text.

## m1 (below the bar, a false claim): §2.14's `Drop` is not bounded on DAGs that share a child within a node

**What happens.** In the iterative branch, `detach` moves only uniquely owned children. A child that
is shared at that moment stays in place; when the other references are dropped first, the last one
is dropped by the drop glue of the node itself, after `Drop::drop` returned. If the node is the root of
the teardown and its child is shared with its own other subtree (or held twice by the node), that
child is torn down by the glue one level deeper, and so on along a chain: the recursion is HEAD's.

**Inputs** (`bin/dropm SHAPE N STACK design|glue`, `tools/bsdrop.py`): `twin`, `N_{k+1} =
Binary(Mul, a, a)` with `a = Arc(N_k)`; `sib`, `N_{k+1} = Binary(Add, a, Arc(Unary(Neg, a)))`.

| shape | 2 MB debug: design / glue | 8 MB release: design / glue | stack per level (design, debug) |
|---|---|---|---|
| `twin` | 14 513 / 14 515 levels | 174 632 / 174 635 | 144 bytes |
| `sib` (2 nodes per level) | 14 455 / 7 257 | 174 455 / 87 317 | 146 bytes |
| unique chain, left-deep sum (controls) | at least 200 000 / 14 515 | | bounded, at most 17 406 bytes |

**Over the bar.** No. It is never worse than HEAD, the parser and the REPL's operators build no such
node (`Add for Expr` wraps two clones in two new `Arc`s), a workspace grep finds no
`Binary(op, a.clone(), a.clone())`, and the engine's results drop with at most 17 KB of stack (the
tower at 2 with 18 000 carets, depth 36 002, in 7 ms release; `F_14` at order 1 and `F_30` at order 0
in 28 ms and 1 ms on 2 MB debug). But §2.14 ("bounded for every shape") and the §2.16 row are false
for these DAGs.

**Fix.** In the iterative branch move every child `Arc`, shared or not, into a list of `Arc<Expr>`
(replace it with a clone of one static leaf `Arc`), and in the loop take each with `Arc::into_inner`
(stable since Rust 1.70): the last owner gets the node and detaches its children, any other owner only
decrements. Then the loop tears down every node whose last reference it holds, for every DAG.

## m2 (below the bar): a float literal at or above 2^53 prints as an integer of another value

**What happens.** Display writes `Expr::Real(f)` with Rust's `{}`, the shortest decimal that reads
back as f, without a decimal point for integral values. With §2.12's bignum literals that text reads
back as an `Integer`, which is not the float's exact binary value (§2.2: a float literal denotes its
exact binary value). `100000000000000000000000.0` is the f64 99999999999999991611392 and prints
`100000000000000000000000`.

**Inputs** (`bin/rt2_rel engine`, 0.1 ms): `(1)/(x + 100000000000000000000000.0 -
100000000000000000000000)` at 0, order 1, is `Ok` with c0 = `1/(-8388608)` (decided nonzero), and
prints `((-1 / (100000000000000000000000 + -100000000000000000000000)) + ...)`, which reads back as a
division by zero. `x + 100000000000000000000000.0 - 100000000000000000000000` prints a text that reads
back as `x`. Red team 2's `ROUNDTRIP` reports "same" on both, since the texts agree and only the
values differ. HEAD prints the same text, which HEAD's i64 parser refuses, so the new parser turns a
parse error into another value.

**Fix.** Print a `Real` so that §2.12 reads it as the same f64: Rust's `{:?}` (`1e23`, `2.0`) with an
exponent part in `number`, or a trailing `.0` when the shortest text has no `.`; add a float of 1e23 to
the round-trip test of §4.3.

## Families tried that held

- **Random structure through the work stacks** (`gen3.py`, 1 143 inputs: towers with series and
  constant levels and `^-`, `^--`, `^+` between levels, `* /` chains with constant and series divisors
  and zero factors, sums mixing constants and series, calls around all of these; points 1, 2, 1/2,
  3/2, 3; orders 0 to 4): 1 070 equal to gp (relative difference below 1e-40 per coefficient), 0 wrong,
  0 crashes, 52 that gp cannot evaluate (huge constant towers), 21 engine errors (budget, honest
  `UndecidedZero`, and the H1 class).
- **Nested shapes at 100 KB on 2 MB debug** (parse on a 1 GiB thread; `tools/shapes3.py`,
  `out/shapes_dbg.txt`; through `taylor_with_budget` and the limit path): `atan`, `asinh`, `erf`,
  `tanh`, `exp`, `sqrt(1+...)`, `log(1+...)`, `abs(1+...)`, alternating `sin(x+cos(1+...))`, variable-free
  calls `f(1, f(1, ...))`, `(1+...)^(1+x)`, `(x+1)/(2-...)`, `-(...)^-2`, `x*(1+...)+x`, `(2+...)^(1/2)`,
  `sin(1+...)*2-1` inside `x + ...`, `(x+1)^-(x+1)^-...`, `exp(-(...)^2*3+x)`, 7 137 to 19 960 levels:
  each `Ok`, refused by the budget or an honest error in at most 4.5 s debug and 143 MB
  (`Finite(14265)` for the `abs` nest is right). No stack growth.
- **Recursion scan** (`tools/callgraph.py`, the strongly connected components of proto10's call
  graph): only constant-depth self-calls (`kpow_const`, `pow_int`, `log_exact` to `kapply` and back once,
  `Ball::unary`, `atan_like`) and the log-depth `balanced_arc` and `bal_depth`. `konst_of` and `expand`
  are loops; `expand` reaches `konst_of`, never the reverse.
- **Drop of engine results with the design's `Drop`**: above (m1), bounded at about 17 KB.
- **Huge literals in structure** (50 000 to 99 990 digits; release, at most 21 ms and 8 MB):
  `(1+x)^L` at order 10 and `x^(1/L)` at 1 refused by the budget; `exp(L*x)`, `(x+L)^(1/2)`,
  `(L'+x)^(1/3)*(x+1/L')` `Ok`; `x + L/(L+1)` (the parser fold) `Ok`, and the limit is the exact
  rational; 90 nested `sqrt(L+...)` and `sin(L+...)` with 1000-digit L: `Ok` and an honest
  `UndecidedZero` (sin of a 3 322-bit argument cannot be reduced at 1024 bits); 89 terms
  `(L/(L+i))^x`.
- **Caps (code reading of proto10)**: the factor cap before each push in `mono_mul`
  (`konst.rs:943`), the term cap before each push in `merge2`, `sum_many` and the accumulator of `mul`
  (`:792`, `:868`, `:1060`), the kernel cap in `intern` before an id is given (`:459`), the depth cap
  before the key (`:1616`). Each comes before the work it bounds. The chain and tower spine walks
  (`chain_items`, `tower_levels`, `chain_items_e`, `tower_record`) allocate before their charge, but the
  charge follows at once and covers them, so on a DAG at most one spine is walked past the budget.
- **Pre-pass**: the address map is cleared at the start of every expansion (`ser.rs:1408`) and holds
  only nodes of the borrowed input and point, so no stale address can alias a new node.
- **Display, PartialEq and depth**: the depth limits of §2.15 hold as stated (`fcall` and `csum` give
  results of depth 19 882 and 36 266 that print with the work-list writer); PartialEq's time is H2.

## Files

`EXP/redteam3-structure/`: `build.sh`, `src/dropm.rs`, `src/consumers.rs`, `src/parse212.inc.rs` (red
team 2's §2.12 parser, included by both), `tools/gen3.py`, `tools/gpcheck3.py`, `tools/shapes3.py`,
`tools/r3.py`, `tools/bsdrop.py`, `tools/callgraph.py`, `inputs/` (`zb_*.txt`, `nexp_*.txt`, `sh_*.txt`,
`hl_*.txt`, `fl*.txt`, `r3_*.txt`, `r3b_*.txt`), `out/` (`zerobase.txt`, `shapes_dbg.txt`,
`r3_s*.out`, `r3b_s*.out`). The binaries were deleted; `build.sh release|debug src/<f>.rs <name>`
rebuilds each in a few seconds.
