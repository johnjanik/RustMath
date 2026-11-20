# S1 implementation, review 2: resources lens (memory, ownership, recursion, input structure)

Reviewed: the uncommitted S1 implementation in `/home/user/RustMath` (branch
`claude/gracious-gates-e6b6l7`, HEAD 8a0b310) after fix-up round 1, with `fu2/s1impl/S1_impl.md`
(including its "Fix-up round 1" part), `fu2/s1impl/tests_report.md`, `fu2/s1impl/owner_decision.md`
and the two round-1 reviews. No repository file was edited. Scratch:
`fu2/s1impl/rev2-resources/`.

Verdict: **not approved**. Two blockers under S1_BAR in this lens (B1, B2). The tests pass. The
three blockers of round 1 are fixed for the shapes they named; B3's fix leaves two node shapes
with a larger stack per level than HEAD (B1 below).

## How it was measured

- The port: the crate as cargo builds it, debug (`target/debug`) and release (`target/release`,
  built in this review: `cargo test --release -j 2 -p rustmath-symbolic --no-run`).
- HEAD: `git archive HEAD` of rustmath-symbolic and its six path dependencies, built by cargo in
  `rev2-resources/headws/` in release and debug (the same compiler and profiles).
- Probes are compiled with rustc against either crate (`build.sh SRC head|port dbg|rel OUT`), so
  the library code under test is cargo's in both cases. `consume2.rs` (programmatic trees through
  `simplify`, `expand`, `substitute`, `==`, `limit`), `parse2.rs` (the parser), `tay2.rs` (taylor
  of programmatic trees), `probe3.rs` (a text through `taylor_with_budget` with 1e8,
  `series_expansion`, `limit_via_series`, `limit_with_error` and the consumers of the result, with
  times and VmHWM), `repl_probe.rs` (lines through `ReplContext::eval` on a 64 MiB thread).
  Depth limits are exact binary searches (`depth2.py`, `pdepth.py`, `tdepth.py`).
- A failure is a process abort with "thread ... has overflowed its stack" (exit 134). Memory runs
  used `ulimit -v 4000000`.

## Tests

| run | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic --no-fail-fast -- --test-threads=2` (debug) | unit 675 passed, 3 ignored; `s1_depth` 7; `s1_fuzz_data` 7; `s1_regressions` 26 passed, 2 ignored; `s1_singular_points` 5; doc 116 passed, 11 ignored; 0 failed (`sym_debug_tests.log`) |
| release test binaries (`--no-run`, then each binary) | `s1_singular_points` 5, `s1_regressions` 26 (2 ignored), `s1_fuzz_data` 7, `s1_depth` 7; 0 failed; peak RSS at most 208 MB (`rel_tests.log`) |
| release `s1_regressions --ignored --test-threads=1` | both pass: F_10 0.162 s, F_11 0.366 s, F_12 refused in 0.019 s, order 20000 in 0.020 s; peak RSS 205 MB |
| `cargo test -j 2 -p rustmath-jupyter --lib -- taylor latex expand_under` | 6 passed (`jup_tests.log`) |

The first step's tests are unchanged in their expected outcomes. `tests/s1_common/check.rs`,
`eval.rs`, `data/s1_fuzz.tsv`, `data/gen_s1_fuzz.py` and `data/s1_gp.py` are byte-identical to the
harness copy of 17:46 (`protoharness/work/src`). `s1_regressions.rs` differs from it only in doc
comments and one rustfmt line wrap; the doc sentence of `limit_with_error_follows_the_expansion`
changed with the fix of round-1 B2, its assertions did not. `data/s1_singular_points.tsv` starts with
the 106 450 bytes of the 609 rows of the first step, byte for byte; the 63 rows of class (d) are
appended. Both generators reproduce their files byte for byte (`gen_s1_singular.py` and
`gen_s1_fuzz.py`, piped to `cmp`). `s1_singular_points.rs` filters rows by class prefix only, and
its coverage test now also requires the (d) rows.

## The blockers of round 1

| round 1 | status | evidence |
|---|---|---|
| B1, parser frame (caret towers) | fixed | table below; `s1_depth` B1 tests pass in both profiles |
| B2, `limit_with_error` unbounded | fixed | `s1_depth` B2 tests pass in both profiles; probes below |
| B3, `simplify` frame | fixed for the three shapes it named; **not** for function chains and zigzags | B1 of this review |

Parser, deepest input read (HEAD and the port built by cargo, `pd_*.txt`):

| shape | release 8 MB: HEAD / port | release 2 MB: HEAD / port | debug 2 MB: HEAD / port |
|---|---|---|---|
| `x^x^...^x` (carets) | 22 774 / 22 775 | 5 678 / 5 678 | 1 441 / 1 442 |
| `(` | 4 124 / 4 515 | 1 028 / 1 125 | 216 / 221 |
| `sin(`, `f(`, `f(x,`, `f(` with `,x)` | 4 124 / 4 515 | 1 028 / 1 125 | 162 / 190 |
| `x^(` | 3 492 / 3 768 | 870 / 939 | 188 / 191 |
| `x+(`, `(x+`, `x+(...)+x` | 4 124 / 4 515 | 1 028 / 1 125 | 216 / 221 |
| `2/(`, `--(` | parse error / 4 515 | parse error / 1 125 | parse error / 221 |

`limit_with_error` (release, 8 MB, `probe3`): the 22 700-level caret and `x^-` towers at 0, 1 and
2 return in at most 0.36 s and 47 MB per entry point (`towers_rel.txt`); on 2 MB debug the
1 400-level towers return in at most 1.6 s and 31 MB (`towers_dbg.txt`). Orders 10^9 and
2^62 - 1 return at once (`x`: `Finite(0)` with a `BigO`; `sin(x)/x`, `exp(x)`: `Finite(1)` without
one).

`simplify` of the shapes round 1 named (`d_*.txt`): right sum, tower and `sin` chain are equal to
HEAD in release (32 745 / 32 745 on 8 MB; 8 169 / 8 169 on 2 MB) and above it in debug (3 256 /
3 428). The left-deep sum is walked by a loop (the fix-up measured over 4 million terms; not remeasured here).

## Blockers

### B1. `simplify` recurses deeper per level than HEAD on function calls and zigzags

S1_BAR (b): inputs HEAD's `simplify` (and `Expr::limit`, which simplifies) handles now abort. This
is the class of round-1 B3, whose required change was "a `simplify` whose recursive frame is at
most HEAD's in both profiles".

The fix-up made `simplify`'s own frame equal to HEAD's (release 0xd0 + 5 pushes, 256 bytes per
call with the return address), but moved two paths into new frames that sit on the recursion
between two `simplify` frames:

- `simplify_function` (`simplify.rs:426`, `#[inline(never)]`): a `Function` node now costs
  `simplify` + `simplify_function` + the collect/fold frames, against HEAD's `simplify` + the
  collect/fold frames. `simplify_function`'s frame is 0x20 + 5 pushes (80 bytes) in release and
  0xa8 (176 bytes) in debug.
- `simplify_left_spine` (`simplify.rs:409`): a binary node whose left child is binary recurses into
  that child's right subtree from the spine function, whose frame is 0x128 + 6 pushes (352 bytes)
  in release, against the `simplify` frame (256 bytes) it replaces. In a zigzag
  `L = (x + L) + x` each level costs 256 + 352 = 608 bytes against HEAD's 2 x 256 = 512.

Deepest programmatic tree each build handles (`consume2.rs`, `d_rel8.txt`, `d_rel8b.txt`,
`d_rel2.txt`, `d_dbg2.txt`; levels, each a binary search):

| tree | consumer | release 8 MB: HEAD / port | release 2 MB: HEAD / port | debug 2 MB: HEAD / port |
|---|---|---|---|---|
| `f(f(...f(x)))` | `simplify` | 14 553 / 12 778 | 3 631 / 3 188 | 1 280 / 1 176 |
| `f(x, f(x, ...))` | `simplify` | 14 553 / 12 778 | | 1 280 / 1 176 |
| `f(x + f(x + ...))` | `simplify` | 10 075 / 9 191 | 2 513 / 2 293 | 917 / 874 |
| `f(sin(f(sin(...))))` | `simplify` | 10 075 / 9 191 | | 919 / 876 |
| `L = (x + L) + x` | `simplify` | 16 372 / 13 787 | 4 084 / 3 439 | 1 628 / 1 809 |
| `L = (x + L) * x`, `L = (x ^ L) + x` | `simplify` | 16 372 / 13 787 | 4 084 / 3 439 (`*`) | 1 628 / 1 809 |
| `f(f(...f(x)))` | `Expr::limit` at 1 | 13 433 / 12 778 | | 1 125 / 1 125 |
| `L = (x + L) + x` | `Expr::limit` at 1 | 14 553 / 13 785 | | 1 305 / 1 305 |
| `L = sin(x + L) + x` | `simplify` | 10 915 / 10 915 | | 1 085 / 1 142 |
| right sum, tower, `sin` chain | `simplify` | equal | equal | port higher |
| `L = (x + L) + x` | `substitute` | 14 554 / 14 554 | | |
| `f(f(...))` | `==` | 174 656 / 174 656 | | |

Direct reproductions, HEAD then the port, the same probe source (`repro_simplify.txt`):

```
c2_{head,port}_rel fch 13500 simplify 8388608   HEAD OK | port rc 134, has overflowed its stack
c2_{head,port}_rel zig 15000 simplify 8388608   HEAD OK | port rc 134
c2_{head,port}_rel fadd 9800 simplify 8388608   HEAD OK | port rc 134
c2_{head,port}_rel fch 13200 limit 8388608      HEAD OK | port rc 134   (Expr::limit at 1)
c2_{head,port}_rel zig 3800 simplify 2097152    HEAD OK | port rc 134
c2_{head,port}_rel fch 3400 simplify 2097152    HEAD OK | port rc 134
c2_{head,port}_dbg fch 1250 simplify 2097152    HEAD OK | port rc 134
c2_{head,port}_dbg fadd 900 simplify 2097152    HEAD OK | port rc 134
```

These trees are programmatic (as round-1 B3's were): the parser reads at most about 4 500 nested
calls on 8 MB release and 190 on 2 MB debug, below the port's `simplify` limits, so no text input of
the bar reaches them on the bar's threads.

Required: a `simplify` whose stack use per tree level is at most HEAD's for every node kind
(`Function`, a binary node with a binary left child, unary, binary), in both profiles, measured
against HEAD's cargo builds. Two ways that fit the design: keep HEAD's `Function` arm inline in
`simplify` (HEAD's frame was 0xd0 with it), and make the spine path's frame no larger than
`simplify`'s; or make `simplify` iterative over all node kinds (a post-order work list, as Display
is now), which removes the limits altogether while keeping HEAD's evaluation order. Add the rows of
the table to `tests/s1_depth.rs` at about 96 % of HEAD (release 8 MB: `f(f(...))` 13 970,
`f(x + f(...))` 9 670, zigzag 15 700, `Expr::limit` of `f(f(...))` 12 890; release 2 MB: 3 485,
2 410, 3 920; debug 2 MB: `f(f(...))` 1 228, `f(x + f(...))` 880, zigzag 1 560).

### B2. The result-tree cap counts nodes, not their payload: consumers of a 100 KB input's result exceed 512 MB

S1_BAR (c), through `taylor_with_budget` with the REPL's budget, in the class of red-team round 3
structure H2 (one of the owner's three holes): the consumers of a result (`simplify`,
`substitute`, Display) are bounded by bounding the result. The cap `MAX_RESULT_TREE` (2^21 nodes,
`series_const.rs:624`, checked by `render_tree`, `series_engine.rs:2380`) counts each node once,
whatever it holds. A coefficient that is one shared kernel holding a large leaf (an integer
literal, a long symbol or function name) is rendered once and named by every term, so the result is
a small DAG whose tree repeats the leaf at every term. `simplify` and `substitute` copy every leaf
they pass (`Integer`, `Symbol` and `Function` names are owned values), and Display prints it each
time. The REPL's own doc comment claims the cap keeps "`simplify`, `subs` and `==` of a result
within about 200 MB" (`repl.rs:11203`).

Measured in release on an 8 MB thread (`probe3`, `big1.txt` to `big6.txt`; each consumer in a fresh
process after `taylor_with_budget` with 1e8):

| input (bytes), at 0 | order | `taylor_with_budget` | `simplify` | `substitute` x = 2 | Display |
|---|---|---|---|---|---|
| `sqrt(N)/(1-x)`, N of 99 950 digits (99 962) | 20 000 | Ok, 0.017 s, 12 MB | 0.39 s, **818 MB** | 0.37 s, **818 MB** | **131 s, 1.9 GB** (2.0 GB of text) |
| the same | 2 000 | Ok, 0.002 s | 0.04 s, 85 MB | | 13.2 s, 213 MB |
| the same | 60 000 | Ok, 0.067 s, 28 MB | 4.6 s, **2.45 GB** | | |
| `sin(N)/(1-x)` (99 961) | 20 000 | Ok, 0.021 s | 0.38 s, **817 MB** | 0.13 s | |
| `a.../(1-x)`, a symbol of 99 941 characters | 20 000 | Ok, 0.018 s | 1.41 s, **1.93 GB** | 0.82 s, **1.93 GB** | |
| `f...(1)/(1-x)`, a function name of 99 931 characters | 20 000 | Ok, 0.024 s | 1.13 s, **1.93 GB** | 0.86 s, **1.93 GB** | |
| `1/(1+N-x)` at the point N, N of 49 990 digits (49 998 + 49 990) | 20 000 | Ok, 0.025 s | 0.50 s, 417 MB | 0.20 s, 416 MB | |
| `N/(1-x)` (99 956) | 20 000 | `BudgetExceeded` | | | |

`==` of each result with its clone takes under 0.04 s (it compares shared leaves without copying).
Order 20 000 is the REPL's `TAYLOR_ORDER`; through the library the budget allows order 60 000 for
`sqrt(N)/(1-x)` (order 200 000 is `BudgetExceeded`).

Through the REPL (debug, 64 MiB thread, `repl_sqrtN_dbg.txt`), two lines of 100 004 bytes in all:

```
t = taylor("sqrt(N)/(1-x)", x, 0, 20000)   Ok, 0.23 s, VmHWM 17 MB (the display limit refuses to print it)
simplify(t)                                Ok, 0.39 s, VmHWM 824 MB
```

The limit paths are not affected (they render the first term only: 0.012 s, 8 MB).

Required: bound the result by its payload as well as its node count, so that every consumer the
design names stays within the bar. For example, carry a payload-weighted size with each constant
and the point (the leaf bytes that `display_estimate` and the REPL's `expr_display_size` already
count: integer digits, both parts of a rational, symbol and function name lengths), sum it in
`render_tree` per term as the node count is summed, and refuse above a second cap
(`LimitExceeded(TreeSize)` or a new `StructureLimit`) chosen so that `simplify` and `substitute` of
a capped result stay under 512 MB and Display under 20 s. Regression tests: the four 100 KB rows of
the table at order 20 000 through `taylor_with_budget` with 1e8, then `simplify`, `substitute` and
Display of the result, in release on an 8 MB thread: either a refusal or each consumer within 20 s
with VmHWM under 512 MB. Correct the `TAYLOR_WORK` doc comment.

## Minor issues

1. `tests/s1_depth.rs` checks `simplify` on three shapes only (right sum, tower, `sin` chain).
   That is how B1 passed the fix-up's tests; the rows of B1's table belong there.
2. `limit_with_error` at a large order keeps the order-5 limit and drops the `BigO` when the
   second expansion is refused (`sin(x)/x` at order 10^9: `Finite(1)`, no `BigO`). This is what its
   doc says; noted only because `limit_via_series` at the same order gives `Unknown`.
3. `expand` of `L = -(L + x) + x` uses memory quadratic in the depth (1.48 GB at 4 000 levels, 5.2 GB
   at 10 273), equally in HEAD and the port (`distribute_negation` changed, the cost did not). Not
   S1's; a HANDOFF line next to design §7 item 5 at most.

## What held (no finding)

- `expand` with the new `distribute_negation`: `-(left sum)`, `-(left difference chain)`,
  `-(right sum)` and the left sum reach HEAD's depth exactly (release 8 MB 30 818; 2 MB 7 688;
  debug 1 918).
- Taylor of programmatic deep inputs on 2 MB debug, order 1 at 1 (`tay2.rs`, `td_dbg2.txt`; HEAD's
  `Expr::taylor` against `taylor_with_budget`, then Display of the result): HEAD returns up to
  1 120 levels of `f(f(...f(1)))/(1-x)`, 188 levels of `sin`, `exp`, negation, tower, right sum,
  left sum and right product chains, and 94 of the zigzag; the port returns at every depth the
  search tried, up to 520 192 for each shape (the search's bound).
- Every entry point on the deepest texts the parser now reads (22 700-level caret and `x^-`
  towers on 8 MB release, 1 400 levels on 2 MB debug): bounded, as above. Where the result is
  `Ok` (the 1 400-level towers at 1, debug), `simplify`, `substitute`, `==` and Display of it
  return at once.
- `Display` and `Drop` have no recursion over the depth (tests `test_display_work_list`,
  `drop_deep_shapes`, `drop_shared_children`, `drop_during_tls_teardown` pass in debug on 2 MB).
- The REPL taylor arm: `mentions_symbol` walks the DAG, the errors are mapped and bounded, the
  display limit refuses the 2 GB text of B2 before formatting it.
- `ExpansionPoint`'s `Debug` is bounded now (read: `point_text` formats the point only when
  `display_estimate`, which stops after 60, says it is short).
- The literal fold of the parser (`product_step`, a clone of the literal per step and the Lehmer
  gcd of `rustmath-integers`), release (`fold.txt`): a 49 990-digit literal followed by 25 000
  `/1` (99 990 bytes) parses in 0.33 s, by 16 600 `/-1` in 0.23 s, the quotient of two coprime
  49 990-digit literals in 0.05 s.

## Files

`fu2/s1impl/rev2-resources/`: `build.sh`, `consume2.rs` and `depth2.py` (B1), `parse2.rs` and
`pdepth.py` (parser), `tay2.rs` and `tdepth.py` (taylor depth), `probe3.rs` (entry points and
consumers on a text), `repl_probe.rs` (REPL lines), `in/` (the inputs of B2 and the towers), logs
`sym_debug_tests.log`, `rel_tests.log`, `jup_tests.log`, `d_rel8.txt`, `d_rel8b.txt`, `d_rel2.txt`,
`d_dbg2.txt`, `pd_rel8.txt`, `pd_rel2.txt`, `pd_dbg2.txt`, `repro_simplify.txt`, `td_dbg2.txt`,
`big1.txt` to `big6.txt`, `repl_sqrtN_dbg.txt`, `towers_rel.txt`, `towers_dbg.txt`, `fold.txt`. The probe
binaries `c2_*`, `p2_*_rel`, `p3_*`, `t2_*` are kept; `target/release` and the cargo build of HEAD
(`headws/target`) were deleted after use.
