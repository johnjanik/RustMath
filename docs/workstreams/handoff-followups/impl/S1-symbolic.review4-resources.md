# S1 implementation, review 4: resources lens (memory, ownership, recursion, input structure)

Reviewed: the uncommitted S1 implementation in `/home/user/RustMath` (branch
`claude/gracious-gates-e6b6l7`, HEAD 8a0b310) after fix-up round 3, with `fu2/s1impl/S1_impl.md`
(including "Fix-up round 3"), `fu2/s1impl/tests_report.md`, `fu2/s1impl/owner_decision.md` and the
two round-3 reviews. No repository file was edited (`git status --short`: the same 35 entries before
and after). Scratch: `fu2/s1impl/rev4-resources/` (paths below are relative to it unless they start
with `fu2/` or a crate name).

Verdict: **not approved**. Two blockers in this lens (B1, B2). The tests pass. The two blockers of
round 3 are fixed for every input that round named.

- B1 is a gap in round 3's fix of its B2. `simplify` of an accepted result can still take minutes.
  A kernel made from an input node renders that node, and `simplify` folds its arithmetic left to
  right. The engine evaluated the same constant in a different order (`product_many`, pairwise
  rounds), so its budget does not bound `simplify`'s folds.
- B2 is new in round 3: the memo of `simplify` keeps the value of every node whose `Arc` is held
  twice until the walk ends. On a DAG whose nodes are also held outside the walk (the REPL's
  variables), that is the sum of all intermediate values, where HEAD holds one.

## How it was measured

- The port as cargo builds it. Debug: `target/debug` (current: `librustmath_symbolic-704f099c312b95e8`
  rebuilt at 23:28 after the last source change). Release: `target/release` as the round-4
  semantics reviewer left it (`cargo test --release -j 2 -p rustmath-symbolic --no-run` reported
  every artifact fresh). That reviewer deleted `target/release` afterwards. My release probes were
  linked before that and are kept.
- HEAD: `librustmath_symbolic-4cbaba7850391989.rlib` (debug, 15:43, no `series_engine` symbol).
- Probes, compiled with rustc against cargo's rlibs (`build.sh SRC dbg|rel OUT`):
  - `probe5.rs` (`p5_rel`): reads a text input and runs steps on one thread of a given stack, each
    with its wall time and VmHWM: `tw` (`taylor_with_budget` at 1e8, the REPL's `TAYLOR_WORK`),
    `se`, `lim`, `lwe`, `disp`, `simp`, `sdisp` (Display of `simplify`'s output), `ssimp`, `subs`,
    `eq`, `exp` (expand), `simpin` (simplify of the input), `drop`.
  - `memo/chain.rs`, `memo/chain2.rs` (HEAD and port): builds a_1 = BASE and a_k = a_{k-1} op c
    the way the REPL does (a clone of the stored value under a new `Arc`), keeps every a_k alive,
    then `simplify(a_L)`.
  - `memo/chain_t.rs`, `memo/chain_t2.rs`: the same chain, then `sin(a_L)/(1-x)` through
    `taylor_with_budget` at 1e8, then `simplify` of the result.
  - `repl_probe.rs`: `ReplContext::eval` line by line on a 64 MiB thread (the kernel's
    `EVAL_STACK_BYTES`), debug, against the current jupyter lib.
  - `depthfam.py`, `depthrun.sh`: depth families; `fz/gen.py`, `fz/run.py`: a resource fuzzer.
- Memory runs used `ulimit -v 6000000` unless a smaller limit is named. A run is release on an 8 MB
  thread unless named otherwise.

## Tests

| run | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic --no-fail-fast -- --test-threads=2` (debug, `sym_debug_tests.log`) | unit 681 passed, 3 ignored; `s1_depth` 9; `s1_fuzz_data` 7; `s1_payload` 2 (1 ignored); `s1_regressions` 26 (2 ignored); `s1_review2` 2; `s1_review3` 7 (7 ignored); `s1_singular_points` 6; doc 116 (11 ignored); 0 failed |
| the eight release test binaries (`rel_tests.log`) | 681 (3 ignored), 9, 7, 2 (1 ignored), 26 (2 ignored), 2, 7 (7 ignored), 6; 0 failed |
| release `--ignored --test-threads=1` (`rel_ignored.log`) | `s1_payload` 1 passed (3.18 s); `s1_regressions` 2 passed (0.72 s); `s1_review3` 7 passed (1.72 s) |
| `cargo test -j 2 -p rustmath-jupyter --lib -- taylor latex expand_under` (`jup_tests.log`) | 8 passed (`test_taylor_simplify_of_kernel_sources` included) |

The tests of the first step are unchanged in their expected outcomes:

- Byte-identical to the harness copy (`fu2/s1impl/protoharness/work/src`): `s1_common/check.rs`,
  `s1_common/eval.rs`, `data/s1_fuzz.tsv`, `data/gen_s1_fuzz.py`, `data/s1_gp.py`.
- `s1_regressions.rs` differs from the harness copy only in the `#[path]` lines (now doc comments),
  doc comments and one rustfmt wrap. Its mtime is 20:02, as in rounds 2 and 3.
- `data/s1_singular_points.tsv` starts with the 106 450 bytes of the first step and with the whole
  file of round 2 (`fu2/s1impl/fix2/s1_singular_points.tsv.before`), byte for byte.
- No earlier test file changed in round 3 (mtimes 21:30 or earlier); `s1_review3.rs` is new.

## The blockers of round 3

Reproduced on the current release build (`r3b/round3_blockers_rel.txt`):

| round 3 | input, at 0 | now |
|---|---|---|
| B1 | `in/gsub.txt` at 189; `in/gfloat.txt` at 189 | `LimitExceeded(TreeText)` in 0.010 s |
| B1 | `in/gsub.txt` at 36 (largest accepted) | Display 133 MB in 0.12 s, VmHWM 135 MB |
| B2 | `((2^256+1)^4096*x)^4096` at 4096 | `simplify` 0.000 s, its output 111 bytes |
| B2 | `1/(1 - (2^256+1)^4096*x)` at 100 | `simplify` 0.000 s, output 11 173 bytes |
| B2 | `exp(x + (2^256+1)^4096)` at 5000 | `simplify` 0.004 s; Display of its output 38.7 MB in 0.82 s |
| B2 | `sin((2^254+1)^4096)/(1-x)` at 1000; at 192 | refused; at 192 Display of `simplify`'s output 60 MB in 7.8 s |
| B2 | `exp(x + N^4096)` at 600 | `simplify` 0.000 s |
| m1 | `exp(x + (10^300-1)^1050*7^130000)` at 312; at 122 | refused; at 122 Display 52 MB in 7.1 s |

`substitute`, `==` and Drop of these results take at most 0.03 s. The REPL lines of round 3
(`t = taylor(exp(x + (2^256+1)^4096), x, 0, 300)`, `s = simplify(t)`, and the same for the power of
a kernel) now take 0.009 s and 0.007 s (`repl_dag_dbg.txt`, debug).

## Blockers

### B1. `simplify` of a result re-folds a kernel's source left to right: 35 s from a 24 KB input, 169 s from a 60 KB input

S1_BAR (c), through `taylor_with_budget` with the REPL's budget, then `simplify` of the result. This
is the class of round 3's B2: a consumer of an accepted result passes the bar. Round 3 asked for the
property "`simplify` of any result that an entry point accepts stays within 20 s and 512 MB", and the
code claims it in three places:

- `simplify.rs:391`: "the work of `simplify` is bounded by the DAG of its input and by the folds the
  series engine computed itself";
- `series_const.rs:644` (`MAX_RESULT_TEXT`): "`Display`, `simplify`, `substitute` and `==` of a
  result, and `Display` of `simplify`'s output, stay within S1_BAR (c)";
- `repl.rs:11210` (`TAYLOR_WORK`): "`simplify`, `subs`, `==` and the printing of a result, and the
  printing of its `simplify`, stay within the S1 bar".

The mechanism:

1. A kernel made from an input node takes that node as its expression (`mk_kernel_src`,
   `series_const.rs:2520`). So `sin(C)` renders as the input text of C.
2. The engine evaluates a `* /` chain by pairwise rounds (`product_many`, `series_const.rs:1751`,
   from `chain_items`). For C = K/3/3/.../3 it computes 3^L by balanced products and divides once.
3. `simplify` folds the same chain left to right. Each level divides a rational with a 2^20-bit
   numerator by 3 and normalizes it with a gcd against the growing 3^k. The cost is quadratic in L.
4. Neither cap sees this. The text bound counts the folded literal once (637 KB of output), and the
   size guard of `Pow` passes (K = (2^254+1)^4096 is exactly 2^20 bits a priori).

Inputs `chainT/*.txt`, at 0, order 1, release, 8 MB thread (`p5_rel`):

| input (bytes) | `taylor_with_budget` | `simplify` of the result | Display of its output | `simplify` of the input |
|---|---|---|---|---|
| `sin((2^254+1)^4096` + `/3` x 8 000 + `)/(1-x)` (16 025) | Ok, 0.028 s, 9 MB | 17.9 s | 634 KB, 0.08 s | 18.0 s |
| the same with 12 000 `/3` (24 025) | Ok, 0.037 s, 11 MB | **35.2 s** | 638 KB, 0.08 s | 34.6 s |
| the same with 12 000 `/7` (24 025) | Ok, 0.050 s, 10 MB | **52.2 s** | 647 KB, 0.08 s | 51.8 s |
| `g(` the 12 000 `/3` chain `)/(1-x)` (24 023) | Ok, 0.038 s, 11 MB | **34.4 s** | 638 KB, 0.08 s | |
| the same with 30 000 `/3` (60 025) | Ok, 0.108 s, 21 MB | **169.5 s** | 655 KB, 0.08 s | |
| `exp(x + ` the 12 000 `/3` chain `)` (24 023) | Ok, 0.036 s, 11 MB | 0.000 s | 638 KB, 0.08 s | 34.5 s |

Notes:

- The last row is the control. There the kernel exp(C) is built from C's normal form (the input
  node `exp(x + C)` depends on x), so the result holds the folded literal and `simplify` has
  nothing to fold.
- `limit_via_series`, `limit_with_error` and `series_expansion` on the 12 000-level input return in
  0.035 s (`Unknown`, `Unknown`, Ok 2 terms).
- The REPL path (debug, 64 MiB thread, `chainT/repl_div3_12k.out`):
  `t = taylor("sin((2^254+1)^4096/3/.../3)/(1-x)", x, 0, 1)` is Ok in 0.38 s, then
  `s = simplify(t)` takes 355.5 s.
- `simplify` of the input itself takes as long, so the cost is the one HEAD's rules give on the
  user's own subexpression. It is not multiplied by the result. But the engine accepts the input in
  0.04 s, and the code states that `simplify` of the result is bounded by the engine's work.
  I count it as a blocker for that reason, as round 3 counted its B2. If the owner accepts a weaker
  statement ("`simplify` of a result costs at most what `simplify` of the input's subexpressions
  costs"), B1 becomes a doc correction; that is the owner's call.

Required (a property, not a mechanism): `simplify` of any result that an entry point accepts stays
within 20 s and 512 MB (release, 8 MB thread), or the three doc comments above state the weaker
bound that holds and the owner accepts it. Ways that fit:

1. Render a kernel from the normal form of its argument, not from the input node, when the input
   node holds literal arithmetic that `simplify` folds (or whenever the argument's normal form is a
   rational). The text cap then counts the literal, as it does for `exp(x + C)` (the control row).
2. Let `tree_bound` also bound the work of `simplify`'s folds (for example the sum over fold nodes
   of the operands' word sizes multiplied, which covers the gcd of `Rational::new`), and refuse a
   result above a measured cap.
3. Fold a `* /` or `+ -` chain of literals in `simplify` as the engine does (pairwise), so that its
   cost matches the engine's charge. This changes HEAD's `simplify` only in the order of exact
   operations (the value is the same), but it does not cover chains that mix literals with other
   nodes.

Regression test (release, 8 MB thread, `#[ignore]` like the other budget tests; written before the
fix): `sin((2^254+1)^4096` + `/3` x 12 000 + `)/(1-x)` and the same with `g(`, at 0, order 1,
through `taylor_with_budget` with 1e8: refused, or `simplify` of the result and Display of its
output each within 20 s with VmHWM under 512 MB.

### B2. The memo of `simplify` holds the value of every shared node: 754 MB where HEAD uses 9 MB; a 98 KB REPL session aborts

S1_BAR (b), a regression against HEAD on an input HEAD handles; and S1_BAR (c) on the REPL path
through `taylor` and then `simplify` of the result (761 MB).

`simplify` (`simplify.rs:431` to `:478`) memoizes every node whose `Arc` has a strong count above 1
(`:448`) and keeps the value until the walk ends (`:478`). The count also counts owners outside the
walked DAG. In the REPL, `a2 = a1 * 3` stores a clone of a1's node under a new `Arc`, so a1's
children are held by a1 and by that clone. A chain of L such assignments gives L nested nodes, each
held twice, and `simplify(a_L)` keeps the L intermediate literals. HEAD's recursive `simplify`
consumes each intermediate value as it combines, so it holds about one.

Library comparison, the chain built as the REPL builds it, every a_k kept alive, 64 MiB thread
(`memo/chain_runs.txt`, `chain8_runs.txt`, `chain2_mul3.txt`):

| a_1, op, L | HEAD (debug) | port (debug) | port (release) |
|---|---|---|---|
| `(2^254+1)^4096`, `+ 1`, 4 000 | 1.07 s, VmHWM 7 MB | 1.52 s, **501 MB** | 1.63 s, **501 MB** |
| `(2^254+1)^4096`, `* 3`, 6 000 | 1.97 s, 9 MB | 2.51 s, **754 MB** | |
| 8 factors `(2^254+k)^4096` (8.3e6 bits), `+ 1`, 600 | 7.76 s, 10 MB | 8.35 s, **604 MB** | 1.82 s, **604 MB** |
| the same, 5 600 | 21.1 s, 14 MB | **abort**: `memory allocation of 1040392 bytes failed` (4 GB limit) | **abort** (4 GB limit) |
| the `+ 1` chain at 1 000 to 4 000 with only a_L kept (`drop` mode) | 5 to 7 MB | 4 to 5 MB | 4 to 5 MB |

The last row shows that the memory comes from the outside owners, as described.

The REPL (debug, 64 MiB thread):

- `memo/r1000.txt` (15.8 KB: `a1 = expr("(2^254+1)^4096")`, `a2 = a1 + 1`, ..., then
  `b = simplify(a1000)`): 1.0 s, VmHWM 134 MB (250 lines: 41 MB; 500: 72 MB).
- `memo/big600.txt` (9.5 KB, the 8.3e6-bit a_1): `b = simplify(a600)` 8.1 s, VmHWM 608 MB.
- `memo/big5600.txt` (98 727 bytes): the 5 600 assignments run with VmHWM 12 MB; then
  `b = simplify(a5600)` aborts (rc 134, `memory allocation of 1040392 bytes failed`) under
  `ulimit -v 4000000`. HEAD's `simplify` of the same DAG takes 14 MB (library row above).
- Through the series engine, `memo/repl_mul3_6000.txt` (93 861 bytes: `a_k = a_{k-1}*3` up to
  6 000, `z = sin(a6000)/(1-x)`, `t = taylor(z, x, 0, 1)`, `s = simplify(t)`): `taylor` is Ok in
  0.19 s with VmHWM 16 MB; `s = simplify(t)` takes 3.6 s and **VmHWM 761 MB**. In release
  (`memo/chain_t2_rel`, library): `taylor_with_budget` Ok in 0.017 s, `simplify` of the result
  1.55 s, VmHWM 755 MB; with `/ 3` in place of `* 3`, 11.7 s and 756 MB.

The engine's own memos do not grow this way: `tree_bound` keeps a `TextBound` of at most 4 096-bit
literals per shared node, and the engine's walk stayed at 16 MB on the same DAG.

Required: `simplify` keeps HEAD's memory on a DAG whose nodes are also held elsewhere, and keeps
round 3's sharing within the walked DAG. A way that fits: count each node's parents inside the
walked DAG first (one DAG walk with a visited set by address), memoize only nodes with two or more
parents there, and remove a memo entry after its last use. The memory is then HEAD's live set plus
the values the output shares.

Regression tests (written before the fix):

- Both profiles, 64 MiB thread: build a_1 = `(2^254+1)^4096` and a_k = a_{k-1}.clone() * 3 up to
  6 000 as above, keep every a_k, and assert that `simplify(a_6000)` raises VmHWM by less than
  64 MB (HEAD: 9 MB in total) and equals the folded value.
- Release, 8 MB thread: `sin(a_6000)/(1-x)` through `taylor_with_budget` at 1e8, then `simplify` of
  the result: refused, or within 20 s with VmHWM under 512 MB.
- The REPL: the 93 861-byte session above, with `s = simplify(t)` under 512 MB.

## Minor issues

1. **The printing margin at the text cap.** Display of `simplify`'s output of
   `sin((2^254+1)^4096)/(1-x)` at order 192 (the largest accepted) takes 7.8 s, and Display of
   `exp(x + (10^300-1)^1050*7^130000)` at 122 takes 7.1 s. That is 2.5 to 2.8 times under the bar,
   as the `MAX_RESULT_TEXT` doc says. No change needed; noted for slower machines.
2. **The three doc comments of B1** (`simplify.rs:391`, `series_const.rs:644`, `repl.rs:11210`)
   must say what is guaranteed after the fix of B1 and B2.

## What held (no finding)

- **Depth.** Seventeen families at the parser's own limit (`depth_dbg2.txt`: 150 levels on a 2 MB
  debug thread, carets 1 300; `depth_rel8.txt`: 3 900 levels on an 8 MB release thread, carets
  21 000): `sin`, `sin(1+...)`, `exp`, `x+(...)`, `f(...)` of x, an opaque constant chain, a
  constant `sin` chain, continued fractions with and without a parameter, `sqrt`, `log`, `-(...)`,
  `2*(...)`, `atan`, `abs`, `x^x^...^x` and `2^2^...^2`. Every step (`tw`, `lim`, `lwe`, Display,
  `simplify`, its Display, `substitute`, `==`, expand, Drop) returns; the slowest is 1.45 s (debug,
  budget refusals). The 2 MB debug parser stops near 160 levels (HEAD's limit, `s1_depth.rs`), so
  deeper text never reaches the engine there. A result is as deep as its input plus a few levels,
  so the recursive `==` and `substitute` keep HEAD's reach.
- **REPL DAG values** (`repl_dag*_dbg.txt`, debug): `w = sin(1)` then 40 times `w = w + w`
  (2^40 leaves), `f = f*f` and `h = h*h` (2^40 and 2^30 factors), then `taylor` of `w*x`,
  `exp(w)*sin(x)`, `f`, `sin(f)`, `h`: each is `BudgetExceeded` in 0.56 to 1.39 s with VmHWM at
  most 69 MB. The engine charges its walk of the tree, so no exponential tree gets through. The
  REPL test `test_taylor_shared_dag` passes.
- **The engine's budget.** Sums of 2 to 32 distinct rational power folds of 2^20 bits
  (`gcd/k*.txt`, `sin(((2^254+k)/(2^253+3))^2048)`) are refused (1.35e8 units, 0.4 s); one fold
  costs 0.43 s in the engine and 0.46 s in `simplify`, so the charge matches the work there. Heavy
  families (`heavy/summary.txt`: a power of a 60-kernel sum, exp of a 39-kernel polynomial,
  `1/(1-x-x^2)` at 5 000, `exp(x)*sin(x)` at 1 200, 30 parameters, `tan(x)` at 2 000, a 2 000-call
  sum cubed): every step at most 0.78 s and 12 MB.
- **Resource fuzz** (`fz/`, 600 random inputs: big literals, powers at and above the cap, floats,
  parameters, opaque calls, 16 functions, ten points, orders 1 to 5 000): no crash, no step over
  20 s or 512 MB. The slowest is Display of `simplify`'s output, 117 MB in 5.9 s
  (`fz/out1.tsv.detail`).
- **`simplify` on the chains of B2** folds to the same value as HEAD (the same bit lengths), and
  with only the root alive its memory is HEAD's or less (the `drop` row of B2).
- **Display, Drop, the LaTeX writer** walk with explicit stacks. Drop of every result above took at
  most 0.015 s.
- **expand** does not enter call arguments, so a shared kernel source is not expanded per term
  (`expand/`: `sin((y+z+1)^12)/(1-x)` and `g((y+z+w+1)^8)/(1-x)` up to order 1 000 in 0.001 s).

## Files

`fu2/s1impl/rev4-resources/`:

- scripts: `build.sh`, `probe5.rs`, `repl_probe.rs`, `depthfam.py`, `depthrun.sh`, `runrel.sh`,
  `monitor.sh`, `fz/gen.py`, `fz/run.py`, `memo/chain.rs`, `memo/chain2.rs`, `memo/chain_t.rs`,
  `memo/chain_t2.rs`;
- inputs: `in/` (round 3's), `chainT/`, `memo/` (`r*.txt`, `big*.txt`, `ty*.txt`,
  `repl_mul3_6000.txt`, `mul3_text.txt`), `gcd/`, `heavy/`, `expand/`, `fz/in*.tsv`;
- logs: `sym_debug_tests.log`, `rel_tests.log`, `rel_ignored.log`, `jup_tests.log`,
  `r3b/round3_blockers_rel.txt`, `depth_dbg2.txt`, `depth_rel8.txt`, `repl_dag*_dbg.txt`,
  `memo/chain_runs.txt`, `memo/chain8_runs.txt`, `memo/chain2_mul3.txt`, `memo/big5600.out`,
  `memo/repl_mul3_6000.out`, `chainT/repl_div3_12k.out`, `fz/out*.tsv`;
- binaries kept: `p5_rel`, `memo/chain_port_rel`, `memo/chain_t_rel`, `memo/chain_t2_rel`,
  `memo/chain2_head_dbg`, `memo/chain2_port_dbg`. The REPL probe (134 MB) and the copied release
  test binaries were deleted; `repl_probe.rs` rebuilds with
  `rustc --edition 2021 -A warnings -L dependency=/home/user/RustMath/target/debug/deps --extern rustmath_jupyter=<the jupyter rlib of jup_rlibs_dbg.txt> repl_probe.rs`.

`target/release` was deleted by the round-4 semantics reviewer after both of us used it. I built no
release artifact of my own. The jupyter test executable I built in `target/debug/deps` was deleted
(free disk under 4 GB).
