# S1 implementation, review 5: resources lens (memory, ownership, recursion, input structure)

Reviewed: the uncommitted S1 implementation in `/home/user/RustMath` (branch
`claude/gracious-gates-e6b6l7`, HEAD 8a0b310) after fix-up round 4, with `fu2/s1impl/S1_impl.md`
(including "Fix-up round 4"), `fu2/s1impl/tests_report.md`, `fu2/s1impl/owner_decision.md` and the
two round-4 reviews. No repository file was edited (`git status --short`: the same 35 entries
before and after). Scratch: `fu2/s1impl/rev5-resources/` (paths below are relative to it unless
they start with `fu2/` or a crate name).

Verdict: **not approved**. Two blockers in this lens (B1, B2). The tests pass in both profiles. The
two blockers of round 4 are fixed for every input that round named.

- B1 is a gap in round 4's fix of its B1. The fold cap counts the work of `simplify`'s folds once
  per distinct node of the result's DAG. `substitute`, `expand` and `differentiate` rebuild the
  tree, so the next `simplify` folds a kernel's source once per term. A 5 KB input accepted in
  0.02 s gives 24 s at order 10 and 419 s at order 190 (release), and the REPL's `diff(t, x)` 27 s.
- B2 is a gap in round 4's fix of its B2. The memo of `simplify` keeps the value of every node with
  two or more parents in the DAG until its last use, also when the first use throws the value away.
  A DAG that HEAD simplifies in 0.45 s with 46 MB aborts the port at a 4 GB limit. Through the REPL,
  `diff(t, x)` of an accepted 12 KB result peaks at 839 MB.

## How it was measured

- The port as cargo builds it. Release: `cargo test --release -j 2 -p rustmath-symbolic --no-run`
  (1 min 4 s, `relbuild.log`), then `cargo build --release -p rustmath-symbolic --lib` (fresh) for
  the probes. Debug: `cargo test -j 2 -p rustmath-symbolic` (`sym_debug_tests.log`). The rlibs of
  each profile are in `port_rlibs_dbg.txt` and `port_rlibs_rel.txt`. `target/release` was deleted
  at the end.
- HEAD: `target/debug/deps/librustmath_symbolic-4cbaba7850391989.rlib` (debug, 15:43, the rlib that
  round 4 used for HEAD).
- Probes are compiled with rustc against cargo's rlibs (`build.sh SRC dbg|rel|head OUT`):
  - `cons/probe6.rs` (`p6_rel`, `p6_dbg`): a text input through `taylor_with_budget` at 1e8 (the
    REPL's `TAYLOR_WORK`), then consumers of the result on one thread of a given stack: `simp`,
    `subs` (`substitute(x, 2)`), `usimp` (`simplify` of that), `diff` (`differentiate(x)`), `dsimp`
    (`simplify` of that, which is what the REPL's `diff` does), `tree` (tree nodes), `eq`. Each
    step prints its wall time and VmHWM.
  - `cons/probe7.rs` (`p7_rel`): `expand`, `t*2`, `t - t` and a second `taylor` of a result, each
    followed by `simplify`.
  - `memo/memo_zero.rs`, `memo/memo_dag.rs`: library DAGs for the memo of `simplify` (HEAD and port).
  - `fold/bisect.py`: ten fold-heavy constant shapes inside `g(C)/(1-x)`, bisected to the largest
    accepted size, then `simplify` of the result there.
  - `depth/depthp.rs`: the depth of results against the depth of the input.
  - `repl/repl_probe.rs`: `ReplContext::eval` line by line on a 64 MiB thread (the kernel's
    `EVAL_STACK_BYTES`), debug, against the current jupyter lib (`jup_rlibs_dbg.txt`; the 134 MB
    binary was deleted, it rebuilds with the command in round 4's report).
- Release runs are on an 8 MB thread unless named otherwise. Memory runs use `ulimit -v 6000000`
  unless a smaller limit is named.

## Tests

| run | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic --no-fail-fast -- --test-threads=2` (debug, `sym_debug_tests.log`) | unit 683 passed, 4 ignored; `s1_depth` 9; `s1_fuzz_data` 7; `s1_payload` 2 (1 ignored); `s1_regressions` 26 (2 ignored); `s1_review2` 2; `s1_review3` 7 (7 ignored); `s1_review4` 4 (2 ignored); `s1_review4_memory` 1 (1 ignored); `s1_singular_points` 7; doc 116 (11 ignored); 0 failed |
| the ten release test binaries (`rel_tests.log`) | 683 (4 ignored), 9, 7, 2 (1 ignored), 26 (2 ignored), 2, 7 (7 ignored), 4 (2 ignored), 1 (1 ignored), 7; 0 failed |
| release `--ignored --test-threads=1` (`rel_ignored.log`) | `s1_payload` 1 passed (3.28 s); `s1_regressions` 2 passed (0.77 s); `s1_review3` 7 passed (1.85 s); `s1_review4` 2 passed (18.36 s); `s1_review4_memory` 1 passed (1.21 s); the unit `fold_work_calibration` passed |
| `cargo test -j 2 -p rustmath-jupyter --lib -- taylor latex expand_under simplify_of_a_variable` (`jup_tests.log`) | 9 passed (`test_simplify_of_a_variable_chain_memory` included) |

The release run of the unit binary with `--ignored` also runs `specialfunctions::airy::tests::
test_airy_numerical_helper`, which fails. It is HEAD's test, `#[ignore]` at HEAD ("Skip numerical
test - it's just a demonstration"), and the diff of `airy.rs` only adds `&` in `matches!` of other
tests. It is not an S1 test.

The tests of the first step are unchanged in their expected outcomes:

- Byte-identical to the harness copy (`fu2/s1impl/protoharness/work/src`): `s1_common/check.rs`,
  `s1_common/eval.rs`, `data/s1_fuzz.tsv`, `data/gen_s1_fuzz.py`, `data/s1_gp.py`.
- `s1_regressions.rs` (mtime 20:02, as in rounds 2 to 4): with comment lines, `#[path]` lines and
  white space removed, it differs from the harness copy in one character, a trailing comma that
  rustfmt added (`Ok(())},);`). No assertion changed.
- `data/s1_singular_points.tsv` (740 lines) starts with the 106 450 bytes of the first step and
  with the 124 043 bytes of round 3 (`fu2/s1impl/fix4/s1_singular_points.tsv.before`), byte for
  byte, and equals `fix4/s1_singular_points.tsv.new`.

## The blockers of round 4

Reproduced on the current release build:

| round 4 | input, at 0, order 1 | now |
|---|---|---|
| B1 | `chainT/div3_8k.txt`, `div3_12k.txt`, `div7_12k.txt`, `g_div3_12k.txt`, `div3_30k.txt` (16 to 60 KB) | `LimitExceeded(FoldWork)` in 0.03 to 0.07 s (`cons/r4b1_repro.txt`) |
| B1 control | `chainT/expdiv3_12k.txt` | Ok, `simplify` 0.000 s |
| B1 others | `atan_`, `root_div3_12k`, `add1_16_30k`, `div3_16_20k`, `mul3_16_40k` | `BudgetExceeded` in at most 0.15 s |
| B2 | round 4's REPL session `memo/repl_mul3_6000.txt` (93 861 bytes), debug | every line Ok; `s = simplify(t)` 3.4 s; VmHWM 17 MB for the session (round 4: 761 MB) (`repl/r4b2_session.out`) |
| B2 | the library chains (`K * 3` 6 000 levels and the others) | `s1_review4_memory` passes in both profiles |

## Blockers

### B1. The fold cap counts each distinct node once, but `substitute`, `expand` and `differentiate` un-share the result: `simplify` after them folds a kernel's source once per term (24 s at order 10, 419 s at order 190, from a 5 KB input)

S1_BAR (c), through `taylor_with_budget` with the REPL's budget, then a tree-walking consumer of the
result. This is the class of round 3's B2 and round 4's B1 (a consumer of an accepted result
passes the bar), and the class of red-team hole 3 that the owner asked to pin ("results ... which
make tree-walking consumers such as simplify, substitute and PartialEq exponential"). Here the
consumer is not exponential. It is linear in the number of terms, with a factor of seconds per term.

The mechanism:

1. A kernel made from an input node takes that node as its expression, and every coefficient of
   `sin(C)/(1-x)` names the same kernel `sin(C)`.
2. `tree_bound` (`series_const.rs:1421`) adds a node's fold work to `TextMemo::work` once per
   distinct node (`:1469`), and `expansion_text` (`series.rs:604`) checks that sum against
   `MAX_RESULT_FOLD_WORK` (`series.rs:592`). That bounds `simplify` of the result itself, which
   memoizes the shared node.
3. `substitute_by_name` (`substitute.rs:25`), `expand` (`expand.rs:26`) and `differentiate`
   (`differentiate.rs:24`) are HEAD's tree walks. They build a fresh node at every occurrence, so
   their output holds one copy of C per term, and the next `simplify` folds each copy. The text cap
   counts every occurrence and so allows about 190 terms whose folded literal has 2^20 bits.
4. The REPL's `diff` command is `differentiate` then `simplify` (`repl.rs:2024`). `subs` returns
   the substituted tree, which the user then simplifies.

Input `cons/sinchain2535.txt`: `sin((2^254+1)^4096` + `/3` x 2 535 + `)/(1-x)` (5 095 bytes; 2 535
is round 4's largest accepted chain), at 0, release, 8 MB thread (`cons/sinchain_rel.txt`,
`cons/sinchain_diff_rel.txt`, `cons/sinchain190_usimp.txt`, `cons/sinchain_orders.txt`; `probe7`):

| order | `taylor_with_budget` | tree nodes | `simplify(R)` | `simplify(R.substitute(x, 2))` | `simplify(R.expand())` | `simplify(R.differentiate(x))` |
|---|---|---|---|---|---|---|
| 1 | Ok, 0.017 s | 10 159 | 2.16 s | 4.53 s | | 7.32 s, VmHWM 327 MB |
| 10 | Ok, 0.016 s | 55 906 | 2.16 s | **24.1 s** | **23.7 s** | **27.1 s**, VmHWM 349 MB |
| 190 (largest accepted; 200 is `TreeText`) | Ok, 0.025 s | 970 846 | 2.17 s | **418.6 s** (with Display of its 60 MB output), VmHWM 179 MB | | |

`t*2` and `t - t` keep the sharing, and `simplify` of them takes 2.1 s at order 10. A second
`taylor` of the result is `BudgetExceeded`.

The REPL (debug, 64 MiB thread):

- `repl/replB_div3_2535_subs.txt` (5 154 bytes): `t = taylor("<the input>", x, 0, 10)` Ok in
  0.21 s, `u = subs(t, x, 2)` Ok in 0.04 s, then `s = simplify(u)` **207 s** (`repl/replB.out`).
- `repl/replC_div3_2535_diff.txt` (5 135 bytes): `t = taylor(...)` Ok in 0.20 s, then
  `d = diff(t, x)` **225 s**, VmHWM 379 MB (`repl/replC.out`).

For comparison, `simplify` of the input itself takes 2.1 s; the result multiplies the consumer's
cost by the number of terms. The doc comments claim the property only for `simplify` of the result
(`simplify.rs:384`-`:409`, `series_const.rs:638`-`:675`, `repl.rs:11196`-`:11229`), which holds, but
`MAX_RESULT_TREE`'s and `MAX_RESULT_TEXT`'s docs present the caps as the bound of tree-walking
consumers, and the fold cap is the one cap that is not counted per occurrence.

The fold model itself held on every shape I tried (below, "What held"); the gap is only the count
per distinct node.

Required (a property): for every result an entry point accepts, `simplify` of the output of
`substitute`, `expand` and `differentiate` of the result (and the REPL's `diff`), stays within 20 s
and 512 MB (release, 8 MB thread), or the result is refused. Ways that fit:

1. Count the fold work per tree occurrence, as the text is counted: a `TextBound` field with the
   subtree's total work (its own plus its children's, saturating), memoized like `text`, and the cap
   checked against the root's total. Any consumer that rebuilds the tree then folds at most the cap.
   (The input above would be refused from order 1: two occurrences of about 2^30.5 units.)
2. Or make `substitute`, `expand` and `differentiate` keep sharing (a memo by node address, and the
   same `Arc` for a subtree that does not change). This changes HEAD's code and does not cover
   other crates' tree walks, so (1) is the safer property.

Regression tests (written before the fix; release, 8 MB thread, `#[ignore]` like the other budget
tests): `cons/sinchain2535.txt` at 0 and orders 10 and 190 through `taylor_with_budget` with 1e8:
refused, or each of `simplify(R.substitute(x, 2))`, `simplify(R.expand())` and
`simplify(R.differentiate(x))` within 20 s with VmHWM under 512 MB. The REPL: the two sessions above,
each line within the same bounds (in a child process, as `test_simplify_of_a_variable_chain_memory`
does, with a debug deadline scaled as the other REPL budget tests).

### B2. The memo of `simplify` holds values that the output does not keep: an abort where HEAD takes 0.45 s and 46 MB; 839 MB for the REPL's `diff` of an accepted result

S1_BAR (b), a regression against HEAD on an input HEAD handles (an abort included); and S1_BAR (c)
on the REPL path through `taylor` and then `diff` of the result.

Round 4 required "the memory is then HEAD's live set plus the values the output shares". The fix
(`simplify.rs:448` `parents_in_dag`, `:487` `Memo`, `:488` `remember`) counts parents inside the DAG
and drops an entry after its last use. It holds a value from the first use to the last one, also
when the first use discards the value (`0 * q`), and the doc says so ("HEAD's live set plus the
values of the shared nodes whose uses are pending", `simplify.rs:400`). HEAD recomputes a shared node
at each occurrence and holds one value at a time.

Library DAGs (64 MiB thread; `memo/runs_zero1.txt`, `memo/runs_zero2.txt`, `memo/runs1.txt`). P is
one Integer leaf of 2^20 bits held by one `Arc`; q_i = `Arc(Neg(P))` (`memo_zero`) or
`Arc(Mul(P, i+2))` (`memo_dag`); the expression names each q_i twice:

| DAG | HEAD (debug) | port (debug) | port (release) |
|---|---|---|---|
| `memo_zero`: sum of `0 * q_i` + the same sum, m = 4 000 | 0.046 s, VmHWM +3 MB | 0.354 s, **+517 MB** | 0.270 s, **+517 MB** |
| the same, m = 40 000, `ulimit -v 4000000` | 0.454 s, VmHWM 46 MB in total | **abort** (`memory allocation of 131080 bytes failed`, rc 134) | **abort** (rc 134) |
| the same with P of 2^23 bits, m = 2 000 | 0.183 s, +3 MB | | 1.00 s, **+2 008 MB** |
| `memo_zero` as a tree (fresh q_i at each occurrence), m = 4 000 | | | 0.018 s, +0 MB |
| `memo_dag`: sum of q_i minus the same sum, m = 4 000 | 6.91 s, +3 MB | 5.89 s, **+502 MB** | 3.23 s, **+502 MB** |

The values agree (0 in every row).

The REPL (debug, 64 MiB thread, `repl/replA_mul3_6000_diff.txt`, 12 064 bytes):
`t = taylor("sin((2^254+1)^4096*3*...*3)/(1-x)", x, 0, 1)` with 6 000 factors 3 is Ok in 0.20 s
(VmHWM 13 MB), then `d = diff(t, x)` takes 8.6 s with **VmHWM 839 MB** (`repl/replA.out`). The
derivative names each link a_k of the chain twice (as the child of a_{k+1} in `cos(C)`, and through
the shallow clone `(**left).clone()` of the product rule in dC), and the first use is inside the
chain, so the memo holds all 6 000 links' values (2^20 bits each) until the second uses. The library
path in release (`cons/sinmul3_diff_rel.txt`): 6 000 factors, `simplify(R.differentiate(x))` 3.8 s,
VmHWM 774 MB; 11 000 factors (a 22 KB input, `taylor` Ok in 0.022 s), 7.7 s, **VmHWM 1 422 MB**. HEAD
would refold each link at every level of dC (a quadratic tree), so the REPL row is (c), not (b).

Required: `simplify` keeps HEAD's memory up to a constant on every input (the values the output
keeps plus a bounded payload), and keeps round 3's sharing for results. A way that fits: count the
words of the literal values held by the memo, and stop memoizing a literal value (recompute it at
its next use, as HEAD does) once that count passes a fixed cap such as 64 MB. A result never reaches
the cap: its distinct literals are bounded by `MAX_RESULT_TEXT` (2^27 units of printing weight,
under 56 MB of binary digits), so round 3's and round 4's tests keep their sharing. Recomputation
costs at most HEAD's time for that occurrence.

Regression tests (written before the fix):

- Both profiles, 64 MiB thread, in a test binary of its own with the peak reset as in
  `s1_review4_memory.rs`: `memo_zero` with m = 4 000 and P of 2^20 bits: `simplify` raises VmHWM by
  less than 64 MB (HEAD: 3 MB) and gives 0; and m = 40 000 (or P of 2^23 bits with m = 2 000) does
  not abort under the test's own limit.
- Release, 8 MB thread: `cons/sinmul3_11000.txt` at 0, order 1, through `taylor_with_budget` at 1e8,
  then `simplify(R.differentiate(x))`: refused, or within 20 s with VmHWM under 512 MB.
- The REPL: `repl/replA_mul3_6000_diff.txt`, peak under 512 MB (child process).

## Minor issues

1. **The cap docs.** `MAX_RESULT_TREE` (`series_const.rs:638`) and `MAX_RESULT_TEXT` (`:663`) say
   that tree-walking consumers are bounded by the caps. After the fix of B1 they should name the
   consumers measured (`substitute`, `expand`, `differentiate`, each then `simplify`) and the bound.
2. **The printing margin** (round 4's minor 1) is unchanged: Display of a result at the text cap
   takes up to about 8 s. Noted only.

## What held (no finding)

- **The fold model on one DAG.** Ten fold-heavy shapes inside `g(C)/(1-x)`, each bisected to its
  largest accepted size at order 0 (`fold/bisect1.txt`, `fold/bisect2.txt`): harmonic sums
  (4 590 terms), sums of 1/p (2 032), K - 1/p chains (283), alternating `*p` `/p` chains on K
  (1 551), `(K*y)/3/...` (the rule (c * rest)/d, 1 448), 1 953 nested negations, `K/(3^300)`
  chains (169), sums of 22 rational powers `(p/q)^2048`, `K + k/(k+1)` (621) and `K/(k/(k+1))`
  (1 285). `simplify` of the result at the edge takes at most 2.71 s, its Display at most 0.04 s.
  `sin((2^254+1)^4096)/(1-x)`: `simplify` of `substitute` takes 0.52 s at order 10 and 2.47 s at
  order 50 (most of it printing 16 MB), because one power fold is cheap (8 ms in release).
- **Depth.** For 16 nested families of 40 levels (`depth/`): the result is as deep as the input
  plus 18 levels for `sin`, `cos`, `exp`, `atan`, `sinh`, `cosh`, plus 4 for `sqrt`, 2 for `abs`;
  the others are refused (`UndecidedZero` or the budget). So the recursive `==`, `substitute` and
  `differentiate` keep HEAD's reach, as round 4 found.
- **The REPL.** The round 4 session of 93 861 bytes stays at 17 MB. The jupyter tests pass.
- **`t*2`, `t - t`, `simplify(t)`** keep the sharing (2.1 s at order 10 on the B1 input).
- **`substitute` and `==`** of a result take at most 0.09 s (order 190, VmHWM 94 MB).
- `git status --short` lists the same 35 entries; `rustmath-misc/output.txt` is unchanged and no
  `.tmp_dir_*` exists.

## Files

`fu2/s1impl/rev5-resources/`:

- scripts: `build.sh`, `run_rel_tests.sh`, `cons/probe6.rs`, `cons/probe7.rs`, `memo/memo_zero.rs`,
  `memo/memo_dag.rs`, `fold/bisect.py`, `depth/depthp.rs`, `repl/repl_probe.rs`;
- inputs: `cons/sinK.txt`, `cons/sinchain2535.txt`, `cons/gchain7_2115.txt`, `cons/sinmul3_6000.txt`,
  `cons/sinmul3_11000.txt`, `repl/replA_mul3_6000_diff.txt`, `repl/replB_div3_2535_subs.txt`,
  `repl/replC_div3_2535_diff.txt`, `depth/n_*.txt`;
- logs: `relbuild.log`, `rel_tests.log`, `rel_ignored.log`, `sym_debug_tests.log`, `jup_tests.log`,
  `cons/r4b1_repro.txt`, `cons/sinK_rel.txt`, `cons/sinchain_rel.txt`, `cons/sinchain_diff_rel.txt`,
  `cons/sinchain_orders.txt`, `cons/sinchain190_usimp.txt`, `cons/sinmul3_diff_rel.txt`,
  `memo/runs1.txt`, `memo/runs_zero1.txt`, `memo/runs_zero2.txt`, `fold/bisect1.txt`,
  `fold/bisect2.txt`, `repl/replA.out`, `repl/replB.out`, `repl/replC.out`,
  `repl/r4b2_session.out`;
- binaries kept (statically linked against the port's rlibs, so they run without `target/`):
  `cons/p6_rel`, `cons/p6_dbg`, `cons/p7_rel`, `memo/memo_zero_head_dbg`, `memo/memo_zero_port_dbg`,
  `memo/memo_zero_port_rel`, `memo/memo_dag_head_dbg`, `memo/memo_dag_port_dbg`,
  `memo/memo_dag_port_rel`, `depth/depthp_rel`, `depth/depthp_dbg`. `target/release` was deleted at
  the end.
