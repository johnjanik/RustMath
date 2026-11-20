# S1 implementation, review 6: resources lens (memory, ownership, recursion, input structure)

Reviewed: the uncommitted S1 implementation in `/home/user/RustMath` (branch
`claude/gracious-gates-e6b6l7`, HEAD 8a0b310) after fix-up round 5, with `fu2/s1impl/S1_impl.md`
(including "Fix-up round 5"), `fu2/s1impl/tests_report.md`, `fu2/s1impl/owner_decision.md` and the
two round-5 reviews. No repository file was edited (`git status --short`: the same 35 entries
before and after, `status_before.txt`, `status_after.txt`). Scratch: `fu2/s1impl/rev6-resources/`
(paths below are relative to it unless they start with `fu2/` or a crate name).

Verdict: **not approved**. Two blockers in this lens (B1, B2). The tests pass in both profiles. The
two blockers of round 5 are fixed for every input that round named.

Both blockers are the class of round 5's B1, which the owner asked to pin as red-team hole 3: a
consumer of an accepted result passes the bar because the result names a kernel's source at every
term. Round 5 counts two costs per occurrence (the folds of `simplify` and the literal text). It
does not count the output that HEAD's `expand` and `differentiate` build per occurrence, and that
output is not linear in the source.

- B1. `expand` distributes the source of every kernel at every term. A 110-byte input accepted in
  0.007 s gives `expand` 2.5 GB at order 1000 and `simplify` of that 21.8 s and 5.6 GB (release).
  HEAD's `expand` of the input: 0.004 s, 7 MB. The REPL's `e = expand(t)`: 2.5 GB.
- B2. `differentiate` builds 290 to 650 bytes per tree node of the result, so the node cap allows
  0.6 to 1.3 GB, and for a power tower its `simplify` keeps a quadratic tree per term. A 4.9 KB
  input gives the REPL's `d = diff(t, x)` 664 MB, and a power tower of the same size 2.1 GB in the
  REPL and 2.9 GB in release. HEAD's `diff` of the input: 16 MB (225 MB for the tower).

## How it was measured

- The port as cargo builds it. Release: `cargo test --release -j 2 -p rustmath-symbolic --no-run`
  (1 min 5 s, `relbuild.err`). Debug: `cargo test -j 2 -p rustmath-symbolic` (fresh, 0.09 s,
  `dbgbuild.err`). The rlibs used for the probes are in `port_rlibs_dbg.txt` and
  `port_rlibs_rel.txt`. The REPL probe links `target/debug/deps/librustmath_jupyter-2361d6f052b7be12.rlib`
  (02:57, after the last edit of `repl.rs` at 02:47 and of the symbolic sources at 02:53).
- HEAD: `target/debug/deps/librustmath_symbolic-4cbaba7850391989.rlib` (debug), the rlib rounds 4
  and 5 used for HEAD. HEAD's numbers below are debug numbers, so they overstate HEAD's cost.
- Probes are compiled with rustc against cargo's rlibs (`build.sh SRC dbg|rel|head OUT`):
  - `cons/probe8.rs` (`p8_rel`, `p8_dbg`; `probe8_head.rs`, `p8_head`): a text input through
    `taylor_with_budget` at 1e8 (the REPL's `TAYLOR_WORK`), or the input itself (`in`), then
    consumers on one thread of a given stack: `exp` (`expand`), `xsimp` (`simplify` of it),
    `diff` (`differentiate(x)`), `dsimp` (`simplify` of it, the REPL's `diff`), `subs`, `usimp`,
    `simp`, `eq`, `int` (`integrate(x)`), `tree` (tree nodes by a DAG walk). Each step prints its
    wall time and VmHWM. `xsimp` and `dsimp` print the tree nodes of the output.
  - `repl/repl_probe.rs` (`repl_probe`, debug): `ReplContext::eval` line by line on a 64 MiB
    thread (the kernel's `EVAL_STACK_BYTES`).
  - `parse/pprobe.rs` (`pp_rel`, `pp_dbg`, `pp_head`): `parse` of a file, then `simplify`,
    Display, `==`, Drop.
  - `dag/dagp.rs`: DAG inputs whose trees are exponential, through `taylor_with_budget`,
    `series_expansion` and `limit_via_series`, then consumers.
  - `memo/memo_zero.rs`, `memo/memo_dag.rs`: round 5's DAGs, rebuilt against the current rlibs.
- Release runs are on an 8 MB thread unless named otherwise. Memory runs use `ulimit -v 6000000`
  unless a smaller limit is named. The machine has 16 GB and no swap.

## Tests

| run | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic --no-fail-fast -- --test-threads=2` (debug, `sym_debug_tests.log`, 55 s) | unit 685 passed, 4 ignored; `s1_depth` 9; `s1_fuzz_data` 7; `s1_payload` 2 (1 ignored); `s1_regressions` 26 (2 ignored); `s1_review2` 2; `s1_review3` 7 (7 ignored); `s1_review4` 4 (2 ignored); `s1_review4_memory` 1 (1 ignored); `s1_review5` 4 (2 ignored); `s1_review5_memory` 2 (2 ignored); `s1_singular_points` 8; doc 116 (11 ignored); 0 failed |
| the twelve release test binaries (`rel_tests.log`) | 685 (4 ignored), 9, 7, 2 (1), 26 (2), 2, 7 (7), 4 (2), 1 (1), 4 (2), 2 (2), 8; 0 failed |
| release `--ignored --test-threads=1` (`rel_ignored.log`) | `s1_payload` 1 passed (3.20 s); `s1_regressions` 2 (0.79 s); `s1_review3` 7 (1.80 s); `s1_review4` 2 (2.23 s); `s1_review4_memory` 1 (0.05 s); `s1_review5` 2 (2.08 s); `s1_review5_memory` 2 (0.05 s; one is the child helper); the unit `fold_work_calibration` passed (45.7 s) |
| `cargo test -j 2 -p rustmath-jupyter --lib -- taylor latex expand_under simplify_of_a_variable` (`jup_tests.log`) | 12 passed, among them `test_taylor_consumers_of_a_kernel_source_result`, `test_taylor_diff_of_a_chain_result_memory`, `test_taylor_variable_named_like_a_constant`, `test_simplify_of_a_variable_chain_memory` |

## Tests of the first step

Unchanged in their expected outcomes:

- Byte-identical to the harness copy (`fu2/s1impl/protoharness/work/src`): `s1_common/check.rs`,
  `s1_common/eval.rs`, `data/s1_fuzz.tsv`, `data/gen_s1_fuzz.py`, `data/s1_gp.py`.
- `s1_regressions.rs` (mtime 2026-11-15 20:02, as in rounds 2 to 5): with comment lines, `#[path]`
  lines and white space removed, it differs from the harness copy in one character, the trailing
  comma rustfmt added (`Ok(())},);`). No assertion changed.
- `data/s1_singular_points.tsv` (772 lines): its first 106 450 bytes are the harness copy; it starts
  with round 4's file (`fu2/s1impl/fix5/s1_singular_points.tsv.before`, 140 189 bytes) byte for byte
  and equals `fix5/s1_singular_points.tsv.new`.
- Against round 4's state (`fu2/s1impl/fix5/backup/untracked_round4.tgz`, unpacked in `r4/`): only
  `s1_singular_points.rs` changed among the earlier test files, by additions only (class (g): the
  coverage loops, three expectation arms for `g:`, and the test `values_reached_through_folded_constants`).
  `s1_fuzz_data.rs`, `s1_depth.rs`, `s1_payload.rs`, `s1_review2.rs`, `s1_review3.rs`,
  `s1_review4.rs`, `s1_review4_memory.rs` and `s1_common/mod.rs` are unchanged since round 4.

## The blockers of round 5

Reproduced on the current release build (`cons/r5_repro.txt`, `memo/memo_runs.txt`):

| round 5 | input | now |
|---|---|---|
| B1 | `fu2/s1impl/rev5-resources/cons/sinchain2535.txt` at 0, orders 1, 10 and 190 | `LimitExceeded` (the literal text at every node) in 0.017 s |
| B1 control | `sinK.txt` (`sin((2^254+1)^4096)/(1-x)`) at order 192 | Ok; `simplify` of `substitute(x, 2)` 1.39 s, of `differentiate` 3.0 s, VmHWM 29 MB |
| B2 | `memo_zero`, m = 4 000, P of 2^20 bits (release) | 0, 0.047 s, VmHWM +33 MB (round 5: +517 MB) |
| B2 | `memo_zero`, m = 40 000, under `ulimit -v 4000000` (release and debug) | 0, +36 MB, no abort (round 5: abort) |
| B2 | `memo_zero`, P of 2^23 bits, m = 2 000 | 0, +32 MB (round 5: +2 008 MB) |
| B2 | `memo_dag`, m = 4 000, 2^20 bits (release) | 0, 3.76 s, +33 MB (round 5: +502 MB) |
| B2 | `sinmul3_6000.txt`, `sinmul3_11000.txt` at order 1 | refused (`LimitExceeded`, the literal text) in 0.016 s and 0.022 s |
| B2 REPL | `replA` | `test_taylor_diff_of_a_chain_result_memory` passes |

## Blockers

### B1. `expand` of a result distributes a kernel's source at every term: 2.5 GB from a 110-byte input, 5.6 GB and 21.8 s with `simplify`

S1_BAR (c), through `taylor_with_budget` with the REPL's budget, then a consumer of the result that
round 5 required to stay within the bar ("for every result an entry point accepts, `simplify` of
the output of `substitute`, `expand` and `differentiate` of the result ... stays within 20 s and
512 MB ..., or the result is refused"). The fix-up's docs claim it holds
(`MAX_RESULT_TREE` at `series_const.rs:633`: "a consumer that walks the tree node by node
(substitute, PartialEq, the walk of Display, expand, differentiate) is bounded by the cap times its
cost per node"; `MAX_RESULT_TEXT`'s "The bound that holds"; `repl.rs` `TAYLOR_WORK`: "so that
`subs`, `expand` and `diff`, which rebuild the tree, stay bounded too ... all within the S1 bar").

The mechanism: a kernel made from an input node renders as that node, and every coefficient of
`sin(C)/(1-x)` names it. HEAD's `expand` (`expand.rs:26`, unchanged by S1 except the negation rule)
distributes every product of sums, also inside a call's argument, and makes a fresh tree at every
occurrence. So `expand` of the result costs the order times `expand` of C. The engine itself
computes C as an expanded constant (2^12 monomials here, within the budget), so the work budget
does not see it; the text, fold and node caps count C's own size at every term, which is small.

Input `cons/prodsum12.txt`: `sin((y0+z0)*(y1+z1)*...*(y11+z11))/(1-x)` (110 bytes), at 0
(`cons/expand_table.txt`, `cons/expand_big.txt`):

| run | order | `taylor_with_budget` | `R.expand()` | `simplify(R.expand())` |
|---|---|---|---|---|
| HEAD (debug), the input itself | | | 0.004 s, VmHWM 7 MB | 0.152 s, VmHWM 23 MB (98 308 nodes) |
| port, release | 100 | Ok, 0.006 s, 6 MB | 0.29 s, VmHWM 257 MB | 1.87 s, **VmHWM 622 MB** |
| port, release | 400 | Ok, 0.007 s | 1.74 s, **1 007 MB** | 8.47 s, **2 433 MB** |
| port, release | 1 000 | Ok, 0.007 s | 4.01 s, **2 507 MB** | **21.8 s, 5 608 MB** (98 407 302 nodes) |
| port, release, `ulimit -v 4000000` | 3 000 | Ok, 0.009 s | **abort** (`memory allocation of 80 bytes failed`) | |
| port, release, `cons/prodsum8.txt` (74 bytes, 8 factors) | 20 000 (the REPL's largest order) | Ok, 0.020 s | 3.40 s, **3 132 MB** | |
| port, debug, 2 MB thread | 1 000 | Ok, 0.044 s | 3.69 s, **2 508 MB** | |

The REPL (debug, 64 MiB thread, `repl/replE_prodsum12_1000.txt`, `repl/replE.out`):

```
t = taylor(sin((y0+z0)*...*(y11+z11))/(1-x), x, 0, 1000)   Ok, 0.049 s, VmHWM 11 MB
e = expand(t)                                              Ok (not printed: over 16 MB), 5.69 s, VmHWM 2 512 MB
```

For comparison, the same consumers on the input cost 7 MB and 23 MB at HEAD; the result multiplies
them by the order. The input is not literal-heavy, so round 5's per-occurrence counts do not apply.
A literal-heavy variant (`cons/kprod4.txt`, `kprod6.txt`: `((2^254+1)^4096 + y_i)` factors) is
refused by the work budget (`cons/kprod.txt`), so the hole needs only symbols.

Required (a property): for every result an entry point accepts, `expand` of it and `simplify` of
that output stay within 20 s and 512 MB (release, 8 MB thread), or the result is refused. A way
that fits round 5's method: a `TextBound` field with an upper bound of the nodes HEAD's `expand`
builds for the subtree at every occurrence (saturating, memoized like `total`: a leaf 1; `+`, `-`,
`%` the children's sum plus 1; `*` from the summand counts and sizes of both sides, as
`expand_multiply` distributes them; `^` by an integer of at most 5 of a sum, the binomial terms; a
quotient, a unary node and a call, the children's sizes plus 1), checked at the root against a cap
that keeps `simplify(expand(R))` under 512 MB (about 9.9 million output nodes took 622 MB above,
so the cap is near 2^23 nodes or less). The cap docs then name `expand` with that bound.

Regression tests (written before the fix; release, 8 MB thread, `#[ignore]` like the other budget
tests, each consumer in a child process or with the peak reset as in `s1_review5_memory.rs`):
`cons/prodsum12.txt` at 0, orders 100 and 1000, and `cons/prodsum8.txt` at order 20000, through
`taylor_with_budget` with 1e8: refused, or `R.expand()` and `simplify(R.expand())` each within 20 s
with the peak under 512 MB. Both profiles: `cons/prodsum12.txt` at order 100 with the same memory
bound. The REPL: `repl/replE_prodsum12_1000.txt` in a child process, the `taylor` line refused or
every line `Ok` with the peak under 512 MB.

### B2. `differentiate` of a result builds 290 to 650 bytes per tree node at every term, and its `simplify` keeps a quadratic tree for a power tower: 610 MB to 2.9 GB from a 4.9 KB input

S1_BAR (c), the same path and the same property of round 5 ("`simplify` of the output of ...
`differentiate` of the result (and the REPL's `diff`) stays within 20 s and 512 MB"). The REPL's
`diff` is `differentiate` then `simplify` (`repl.rs:2024`).

The mechanism: HEAD's `differentiate` (`differentiate.rs`, unchanged) makes fresh nodes at every
occurrence of a kernel's source, although the source does not contain x: the product rule makes
about 5 nodes per level of a product, the quotient rule about 7, the power rule about 12 (with a
logarithm). `MAX_RESULT_TREE` (2^21) times that is 0.6 to 1.3 GB. For a power tower the simplified
derivative keeps a^b and log(a) at every level of every level's derivative, so it is quadratic as
a tree per term, and its DAG holds the order times that.

Inputs: `sin(S)/(1-x)` with S a product `y0*y1*...*y999` (`cons/prodchain1000.txt`, 4 901 bytes), a
quotient chain `y0/y1/.../y999` (`divchain1000.txt`), a power tower `y0^y1^...^y999`
(`powtower1000.txt`), the tower with 3 000 levels (`powtower3000.txt`, 16 901 bytes), at 0. Order
1040 is the largest under the node cap for the 1 000-level sources (1045 is
`LimitExceeded(TreeSize)`). Release, 8 MB thread (`cons/diff_table.txt`):

| input | order | `taylor_with_budget` | `R.differentiate(x)` | `simplify` of it (the REPL's `diff`) |
|---|---|---|---|---|
| `prodchain300` | 1 000 | Ok, 0.001 s | 0.13 s, 180 MB | 0.19 s, 181 MB |
| `prodchain1000` | 1 040 | Ok, 0.003 s, 5 MB | 0.50 s, **VmHWM 610 MB** | 0.55 s, **612 MB** |
| `divchain1000` | 500 | Ok, 0.003 s | 0.40 s, 495 MB | 1.50 s, **831 MB** |
| `divchain1000` | 1 040 | Ok, 0.003 s | 0.80 s, **1 023 MB** | 4.32 s, **1 706 MB** |
| `powtower1000` | 500 | Ok, 0.003 s | 0.50 s, **648 MB** | 3.66 s, **1 412 MB** (5.1e8 tree nodes) |
| `powtower1000` | 1 040 | Ok, 0.004 s, 7 MB | 1.08 s, **1 340 MB** | 9.12 s, **2 896 MB** (1.05e9 tree nodes) |
| `powtower3000` | 340 | Ok, 0.009 s | 1.21 s, **1 319 MB** | 8.69 s, **2 862 MB** (3.1e9 tree nodes) |
| `sinsum1000` (`sin(sin(y0)+...+sin(y999))/(1-x)`) | 500 | Ok | 0.24 s, 251 MB | 0.13 s, 252 MB |

HEAD (debug, 64 MiB thread) on the inputs themselves (`cons/head_inputs.txt`): `prodchain1000`
`differentiate` VmHWM 15 MB and its `simplify` 0.11 s, 16 MB; `divchain1000` 18 MB; `powtower1000`
1.55 s and 225 MB (HEAD's `simplify` copies the tree). The port's REPL on the input: `d0 =
diff(<powtower1000 input>, x)` 0.15 s, VmHWM 33 MB (`repl/replT0.out`).

The REPL (debug, 64 MiB thread):

```
repl/replD_prodchain1000_diff.txt (4 940 bytes), repl/replD.out:
t = taylor(sin(y0*...*y999)/(1-x), x, 0, 1040)   Ok, 0.35 s, VmHWM 40 MB
d = diff(t, x)                                  Ok, 4.99 s, VmHWM 664 MB
repl/replT_powtower1000_diff.txt (4 940 bytes), repl/replT.out:
t = taylor(sin(y0^...^y999)/(1-x), x, 0, 1040)   Ok, 0.30 s, VmHWM 35 MB
d = diff(t, x)                                  Ok (not printed: over 16 MB), 11.6 s, VmHWM 2 118 MB
```

Results without a repeated kernel source do not reach this: `exp(x*(y1+y2+y3+y4))` is refused by
the work budget from order 40, and at order 20 (150 659 tree nodes) every consumer stays under
114 MB (`cons/expy4.txt`, `cons/expy4.log`). So the cause is the source
repeated at every term, as in B1.

Required (a property): for every result an entry point accepts, `differentiate(x)` of it and
`simplify` of that output (the REPL's `diff`) stay within 20 s and 512 MB (release, 8 MB thread),
or the result is refused. Ways that fit:

1. Count per occurrence an upper bound of the nodes `differentiate` builds and of the tree its
   `simplify` keeps (a `TextBound` field as in B1: the rule's own nodes plus the children's, the
   power rule's output quadratic in a tower), and refuse past a cap that keeps the pair under
   512 MB. A lower `MAX_RESULT_TREE` alone does not do it: the tower's simplified derivative is
   quadratic per term.
2. Or give `differentiate` a path for a subtree that does not contain the variable (its derivative
   is 0, decided by one DAG walk before the recursion). Every kernel's source is free of x, so the
   result's derivative then costs one node per occurrence. This changes HEAD's output for such
   subtrees (`0*z + y*0` becomes `0`), so the callers' tests (rustmath-manifolds' slot assertions,
   rustmath-calculus) must be checked; it does not help B1.

Regression tests (written before the fix; release, 8 MB thread, `#[ignore]`, peak measured in a
child or after a reset): `cons/prodchain1000.txt` at 0, order 1040; `divchain1000.txt` at orders
500 and 1040; `powtower1000.txt` at orders 500 and 1040; `powtower3000.txt` at order 340: refused,
or `R.differentiate(x)` and `simplify` of it each within 20 s with the peak under 512 MB. The REPL:
`repl/replD_prodchain1000_diff.txt` and `repl/replT_powtower1000_diff.txt`, each in a child
process (as `test_taylor_diff_of_a_chain_result_memory` does), the `taylor` line refused or every
line `Ok` with the peak under 512 MB.

## Minor issues

1. **The cap docs.** `MAX_RESULT_TREE` (`series_const.rs:633`), the "bound that holds" paragraph
   of `MAX_RESULT_TEXT`, `simplify`'s doc (`simplify.rs:376` to `:422`) and the REPL's
   `TAYLOR_WORK` (`repl.rs:11196` to `:11236`) state that `expand` and `diff` of an accepted result
   stay within the bar. B1 and B2 contradict that. Fix the text with B1 and B2, naming the bound
   measured for each consumer.
2. **Signed exponent towers in the parser (not new).** `x^-x^-...^-x` and `-x^-x^-...`, which
   HEAD's parser refuses with a `ParseError`, abort the port's parser on a 2 MB debug thread from
   1 500 levels (4.5 KB): `parse/negt1500.txt`; `negt1440.txt` parses. That is HEAD's reach for
   carets on the same thread (HEAD overflows at 1 440 carets, `parse/car1440.txt`), and the REPL's
   64 MiB thread parses every 100 KB tower. Round 1 recorded the row and judged it below the bar,
   because HEAD cannot read such a text at all. I keep that reading; if the owner reads S1_BAR (b)
   as "HEAD returned an error, the port crashes", the fix is an iterative exponent rule or a depth
   counter that returns a `ParseError`.
3. **Note, not S1: HEAD's `integrate` is cubic in the number of terms.** `integrate` of
   `taylor(1/(1-x), x, 0, N)` takes 0.78 s at N = 250, 5.6 s at 500 and 43.3 s at 1000 (release,
   `cons/integrate.txt`). HEAD's `integrate` of the same polynomial typed as text costs the same
   (N = 250: HEAD debug 3.57 s, the port debug on the text 3.65 s and on the result 3.55 s), so the
   result adds nothing to HEAD's own hole on a 7 KB input. A handoff for the REPL's `integrate`.

## What held (no finding)

- **Round 5's fixes.** The table above: the per-occurrence counts refuse round 5's chains at once,
  and the memo of `simplify` stays at about 33 to 36 MB over the start on every DAG of round 5.
- **DAG inputs with exponential trees** (`dag/dagp.rs`, `dag/dag_rel.txt`, `dag/dag_rel2.txt`):
  constant DAGs c = c + c, c = c * c, c = sin(c) + c doubled 20 and 60 times inside
  `sin(c)/(1-x)` or `sin(c)*exp(x)`, an undecided divisor sin(c)^2 + cos(c)^2 - 1 built from them,
  and (1+x)^(2^k) by squaring: `taylor_with_budget`, `series_expansion` and `limit_via_series` give
  `BudgetExceeded` or `Unknown` in at most 0.13 s with VmHWM 4 MB. At 8 and 14 levels the results
  are Ok and `simplify`, `substitute`, `==`, `diff` and `expand` of them take at most 0.12 s and
  50 MB; the `UndecidedZero` message is 118 characters.
- **The parser on 100 KB texts** (`parse/parse_stress.txt`, 20 shapes, HEAD debug 2 MB, port debug
  2 MB and release 8 MB, each then `simplify`, Display, `==`, Drop): the port parses every shape
  HEAD parses. The port overflows only where HEAD also overflows on the same text (`(` nesting,
  carets, `sin(` and `f(` nesting, `(x^2)^2` nesting) or where HEAD refuses the syntax (minor 2).
  The port parses 100 KB of signs, digits, `7.../3...` quotients and float sums that HEAD refuses.
  `==` overflows on 50 000-term chains on the 2 MB debug thread in both HEAD and the port (the
  derived `PartialEq` of HEAD), and the results are as deep as their inputs plus at most 18 levels
  (round 5), so `==` keeps HEAD's reach.
- **Display and Drop** stay iterative; `Debug` of `Expr` is its Display, so the error values print
  without recursion.
- **`tree_bound`** is an explicit-stack DAG walk and `TextBound` holds only bounds (no literal
  values), so the round-5 fields add no copies.
- **The REPL.** The jupyter tests pass, including the round-5 child-process tests.
- `git status --short` lists the same 35 entries; `rustmath-misc/output.txt` is unchanged (md5) and
  no `.tmp_dir_*` exists.

## Files

`fu2/s1impl/rev6-resources/`:

- scripts: `build.sh`, `run_rel_tests.sh`, `run_rel_ignored.sh`, `cons/probe8.rs`,
  `cons/probe8_head.rs`, `repl/repl_probe.rs`, `parse/pprobe.rs`, `dag/dagp.rs`,
  `memo/memo_zero.rs`, `memo/memo_dag.rs`;
- inputs: `cons/prodsum8.txt`, `prodsum10.txt` to `prodsum16.txt`, `cons/prodchain300.txt`,
  `prodchain1000.txt`, `prodchain3000.txt`, `cons/divchain1000.txt`, `cons/powtower1000.txt`,
  `powtower3000.txt`, `cons/sinsum1000.txt`, `cons/kprod4.txt`, `kprod6.txt`, `cons/expy4.txt`,
  `cons/geo.txt`, `cons/poly250.txt`, `repl/replE_prodsum12_1000.txt`,
  `repl/replD_prodchain1000_diff.txt`, `repl/replT_powtower1000_diff.txt`,
  `repl/replT0_powtower1000_diff_input.txt`, `repl/replF_consumers.txt`, `parse/*.txt`;
- logs: `relbuild.err`, `dbgbuild.err`, `sym_debug_tests.log`, `rel_tests.log`, `rel_ignored.log`,
  `jup_tests.log`, `cons/expand_table.txt`, `cons/expand_big.txt`, `cons/kprod.txt`,
  `cons/diff_table.txt`, `cons/expy4.log`, `cons/head_inputs.txt`, `cons/r5_repro.txt`, `cons/integrate.txt`,
  `repl/replD.out`, `repl/replE.out`, `repl/replT.out`, `repl/replT0.out`, `memo/memo_runs.txt`,
  `parse/parse_stress.txt`, `dag/dag_rel.txt`, `dag/dag_rel2.txt`, `status_before.txt`,
  `status_after.txt`, `output_md5.txt`;
- binaries kept (statically linked, they run without `target/`): `cons/p8_rel`, `cons/p8_dbg`,
  `cons/p8_head`, `repl/repl_probe`, `parse/pp_rel`, `parse/pp_dbg`, `parse/pp_head`,
  `dag/dagp_rel`, `dag/dagp_dbg`, `memo/mz_rel`, `memo/mz_dbg`, `memo/md_rel`. `target/release` was
  deleted at the end.
