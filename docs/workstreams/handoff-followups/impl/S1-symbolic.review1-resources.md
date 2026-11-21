# S1 implementation, review 1: resources lens (memory, ownership, recursion, input structure)

Reviewed: the uncommitted S1 implementation in `/home/user/RustMath` (branch
`claude/gracious-gates-e6b6l7`, HEAD 8a0b310), with `fu2/s1impl/S1_impl.md`,
`fu2/s1impl/tests_report.md` and `fu2/s1impl/owner_decision.md`. No repository file was edited.
Scratch: `fu2/s1impl/rev1-resources/` (probes, logs, a cargo build of HEAD in `headws/`).

Verdict: **not approved**. Three blockers under S1_BAR in this lens (B1 to B3). The tests pass.

## How it was measured

- The port: the crate as cargo builds it (debug `target/debug`, release `target/release`, both built
  in this review). Probes are compiled with rustc against those rlibs (`build_probe.sh`).
- HEAD in release: `git archive HEAD` of rustmath-symbolic and its six path dependencies, built by
  cargo in `rev1-resources/headws/` (`headprobe.rs`, `consume.rs`).
- HEAD in debug: HEAD's `rustmath-symbolic/src` compiled with rustc and cargo's dev flags
  (`-C opt-level=0 -C debuginfo=2`). Its parser limits on a 2 MB thread are 216 (`(`), 162 (`sin(`)
  and 1440 (carets), the values the implementer measured with HEAD inside a cargo test binary, so the
  frames match cargo's.
- Every claimed failure below was reproduced on the built code. A failure is a process abort with
  "thread ... has overflowed its stack" (exit 134) or an allocation failure under a 4 GB
  address-space cap (`ulimit -v 4000000`, set so as not to disturb other work).

## Tests

| run | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic -- --test-threads=2` (debug) | unit 675 passed, 3 ignored; `s1_fuzz_data` 7; `s1_regressions` 26 passed, 2 ignored; `s1_singular_points` 4; doc 116 passed, 11 ignored; 0 failed (17 s) |
| release test binaries (`cargo test --release -p rustmath-symbolic --no-run`, then each binary) | 7, 4, 26 passed (2 ignored), 0 failed |
| release `--ignored --test-threads=1` | both pass: F_10 0.169 s, F_11 0.366 s, F_12 refused in 0.020 s, order 20000 in 0.029 s; process peak RSS 205 MB |
| `rustmath_jupyter` lib tests `taylor`, `latex`, `expand_under_minus` | 6 passed |
| downstream unit tests: rustmath-manifolds, rustmath-calculus (binaries of this tree) | 308 and 122 passed |

The first step's tests are unchanged in their expected outcomes. Evidence: both data files are
byte-identical to their generators' output (`python3 -I gen_s1_singular.py | cmp`, the same for
`gen_s1_fuzz.py`) and to the harness copy of 17:46 (`protoharness/work/src/data`); `check.rs` and
`eval.rs` are byte-identical to that copy; `s1_regressions.rs` differs from it only in doc comments
(two section numbers, the `b:*:sub` sentence, the removed feature flag in three run commands) and in
one rustfmt line wrap; `mod.rs` differs only in the doc lines that named the feature gate. The two
`#[ignore]` tests are the first step's release tests. No row filter or skip was added to
`s1_singular_points.rs` or `s1_fuzz_data.rs`.

## Blockers

### B1. The parser overflows on caret towers that HEAD parses (release)

S1_BAR (b), and (c) through the text input of `taylor_with_budget`.

`x^x^...^x` with 15 000 carets (30 001 bytes) parses on HEAD and aborts the port on an 8 MB release
thread:

```
probe_rel parse port carets15000.txt 8388608   -> thread '<unknown>' has overflowed its stack (rc 134)
headprobe_rel parse carets15000.txt 8388608    -> OK 6.0 ms   (HEAD built by cargo)
```

Deepest input read, 8 MB release thread (HEAD built by cargo for the carets rows):

| shape | HEAD | port |
|---|---|---|
| `x^x^...^x` (carets) | 22 770 (22 771 also OK, 23 000 fails) | 13 782 |
| `(` nested | 4 123 | 5 399 |
| `sin(` nested | 4 123 | 5 399 |
| `2^(` nested | 3 491 | 3 879 |
| `x^-x^-...` | parse error | 13 782 |

On a 2 MB release thread (the default of `std::thread::spawn`): HEAD reads 3 500 to 5 600 carets,
the port aborts from 3 500 (3 400 is the last that parses). In debug on 2 MB the port reads as deep
as HEAD or deeper (`(` 220 vs 216, `sin(` 190 vs 162, carets 1441 vs 1440), as the S1 report says.
On the REPL's 64 MiB thread both read every 100 KB caret tower.

Cause: in release `atom` is inlined into `power` (`parser.rs:331`, `:346`). The recursive frame of
`power` is 0x228 = 552 bytes against HEAD's 0x138 = 312 (objdump of the release binaries; HEAD keeps
`atom` out of line, 488 bytes, not live across the recursion). The S1 report measured debug only and
says "release frames scale the same way"; for carets they do not.

Checked fix (rustc rebuild of this tree in `rev1-resources/fixtry/`, which tracks the cargo build
within 3 levels): `#[inline(never)]` on `atom` gives carets 22 775, `(` 4 515, `sin(` 4 515, `2^(`
3 768 on 8 MB release, each at least HEAD's. Required: that fix or an equivalent (an iterative
`power`), measured with cargo builds in debug (2 MB) and release (8 MB and 2 MB) against HEAD for
every shape of the table, and a release regression test (for example 20 000 carets on an 8 MB thread
and 5 000 on a 2 MB thread). Correct the report's sentence.

### B2. `limit_with_error` is not bounded by its budget (deviation 5)

S1_BAR (c), through a public entry point with its own budget (the limit path of design §2.13).

The port gives `limit_with_error` the budget `LIMIT_SERIES_WORK` for the expansion, but first calls
HEAD's unbudgeted `Expr::limit` (`limits.rs:558`), which substitutes and simplifies recursively and
runs HEAD's strategies. Design §2.13 says "`limit_with_error`: the limit as above", the series rule
of `limit_via_series`; the port keeps `Expr::limit` (S1 report, deviation 5). Measured (point, order
in brackets):

| input | build, thread | `limit_with_error` | `limit_via_series` on the same input |
|---|---|---|---|
| `x+x+...+x`, 3 000 terms, 5 999 B [1, 3] | debug, 2 MB | stack overflow (2 000 terms: OK) | `Finite(3000)`, 19 ms |
| `x+x+...+x`, 30 000 terms, 59 999 B [1, 3] | release, 8 MB | stack overflow (20 000 terms: OK) | `Finite(30000)` |
| `x+sin(1)+...+sin(2399)`, 22 884 B [0, 2] | release, 8 MB | 1.81 s, peak RSS 797 MB | `Unknown`, 4 ms, 13 MB |
| `x+sin(1)+...+sin(3199)`, 30 884 B [0, 2] | release, 8 MB | 3.47 s, peak RSS 1.41 GB | `Unknown`, 6 ms, 13 MB |
| `x+sin(1)+...+sin(9999)`, 98 884 B [0, 2] | release, 8 MB | allocation failure under the 4 GB cap | `Unknown`, 24 ms |
| `x^-x^-...^-x`, 1 400 levels, 4 201 B [1, 3] | debug, 2 MB | stack overflow | `Finite(1)`, 84 ms |
| `x^-x^-...^-x`, 13 000 levels, 39 001 B [1, 3] | release, 8 MB | stack overflow (8 000: OK) | `Finite(1)` |

The memory grows quadratically with the number of kernels (357 MB at 1 600, 797 MB at 2 400). The
backtrace of the debug overflow is `limit_with_error` -> `Expr::limit` (`limits.rs:49`) ->
`substitute` -> `substitute_by_name` (2 600 frames). HEAD's `limit_with_error` fails the same way on
the sum and the sin sum (measured on the cargo build of HEAD), so this is not a regression; it is a
budget hole of an S1 entry point. The `x^-` rows are new: HEAD's parser rejects them, and the signed
exponent of §2.12 makes two nodes per level.

Required: compute the limit of `limit_with_error` through the engine as §2.13 says (the rule of
`limit_via_series` on `expansion_first_term`, then `expansion_reaches` for the `BigO`), or bound the
`Expr::limit` call otherwise. `limit_with_error_follows_the_expansion` asserts only that a `BigO`
comes exactly with a finite limit, so it stays valid; its doc sentence "Its limit is `Expr::limit`"
then changes. Add regression tests for the rows of the table (the sum on 2 MB debug, the sin sum
under 512 MB in release).

### B3. `simplify` recurses with a larger frame (release, and unary chains in debug)

S1_BAR (b): inputs HEAD's `simplify` handles now abort.

The new `simplify` (`simplify.rs:376`, design §2.10) walks the left spine with a loop, as intended,
but its recursive frame grew. Deepest programmatic tree each build handles (`consume.rs`, the same
source compiled against HEAD and against the port):

| tree | consumer | HEAD | port |
|---|---|---|---|
| `x+(x+(...))` (right sum) | `simplify`, release 8 MB | 32 640 | 22 720 |
| `x^(x^(...))` | `simplify`, release 8 MB | 32 640 | 22 720 |
| `sin(sin(...x))` | `simplify`, release 8 MB | 32 640 | 22 720 |
| `x^(x^(...))` | `Expr::limit`, release 8 MB | 29 056 | 22 720 |
| `sin(sin(...x))` | `simplify`, debug 2 MB | 3 248 | 2 416 |
| `x^(x^(...))`, right sum | `simplify`, debug 2 MB | 3 248 | 3 824 |
| `x^(x^(...))` | `substitute`, `==`, `differentiate`, `expand` | equal | equal |

Direct reproductions: a right-nested sum of 25 000 terms (`e = t + e` in a loop) simplifies on HEAD
and aborts on the port on an 8 MB release thread; 7 000 terms on a 2 MB release thread; `Expr::limit`
of a 25 000-level tower at 1 on 8 MB release; `sin` nested 3 000 times on 2 MB debug. The frame of
`simplify` in release is 312 bytes and 6 pushes against HEAD's 208 and 5: `simplify_non_binary` is
inlined into it and the spine vector lives in the recursive frame.

Checked fix (rustc rebuild, `fixtry/`): keep HEAD's match as the body of `simplify` and send only a
`Binary` whose left child is a `Binary` to an `#[inline(never)]` spine function. Release 8 MB: 32 704
for all three chains, `Expr::limit` 29 088, a left-deep sum of 2 093 056 terms; debug 2 MB: 3 100 to
3 108 (HEAD 3 248, so it needs tuning there). A bare `#[inline(never)]` on `simplify_non_binary`
does not help (22 752, and 14 960 for `sin` chains). Required: a `simplify` whose recursive frame is
at most HEAD's in both profiles, measured against HEAD, with a regression test (a right-nested sum of
25 000 terms on an 8 MB release thread, `sin` nested 3 000 times on a 2 MB debug thread). Through
text on the bar's threads this is reachable only once B1 is fixed (carets 22 721 to 22 770 in
release), so B1 and B3 should be fixed together.

## Minor issues

1. `Expr::limit` and `limit_enhanced` (HEAD's code, not changed by S1) abort on `x^-x^-...^-x` of
   1 400 levels (4.2 KB) on 2 MB debug and of 13 000 levels on 8 MB release, where HEAD's parser
   gave a parse error. Their depth limits per node are HEAD's (2 608 nodes for `substitute` on 2 MB
   debug); the signed exponent doubles the nodes per level. `limit_enhanced` also overflows on the
   100 KB sum on 8 MB release, as HEAD does. Worth a HANDOFF line next to design §7 item 5
   (iterative `substitute`).
2. `series_coefficients` (no budget, as its doc says) builds a dense vector of `order + 1` entries of
   64 bytes: `x` at order 10^7 takes 628 MB in 0.51 s.
3. `ExpansionPoint` derives `Debug`, which is `Display`, so it prints the point as a tree: a point
   made by 24 doublings of `1 + 1` prints 100 MB in 0.92 s (40 doublings would not finish). The
   hand-written `Debug` of `SeriesExpansion` bounds the point; the point's own `Debug` does not.
   `SeriesError` and `SeriesExpansion` derive `PartialEq`, which walks the trees.
4. `Drop for Expr` costs about 10 to 15 percent on large teardowns: dropping a balanced tree of
   2 097 151 nodes five times takes 0.67 to 0.79 s on HEAD and 0.73 to 0.91 s on the port (shallow
   and 100 frames deep, two runs each); a simplify workload of 200 000 small sums shows no difference
   (0.11 to 0.16 s on both). No bar issue.
5. The fuzz driver `fu2/s1impl/fuzzdrv/s1drv.rs` calls `std::process::exit(0)` inside the job, so
   its times and peak RSS leave out the drop of the input and of the result. My probes include
   them (the final drops of the 100 KB inputs take under 10 ms in debug), so the reported numbers
   stand, but the driver should drop before it exits.
6. At infinity, `1/(x+sin(1)+...+sin(9999))` is refused with "406478797 units charged, budget
   100000000": one a-priori charge was four times the budget. The work was not done (45 ms, 18 MB),
   so it is only a misleading count.

## What held (no finding)

- 35 shapes of up to 100 KB (sums, alternating sums, products, quotient chains, `*3/2` chains, sums
  of products, of `sin(x)`, of `x^2`, 10 000 kernels, an exact zero of 24 000 terms, `exp` of a
  sum, 99 990-digit literals as summand, base, exponent, root, binomial, divisor and inside `sin`,
  100 000 signs, a call with 49 990 arguments, a 99 990-character name, nested `sin`, `exp`, `atan`,
  `log`, `sqrt`, towers, a power chain) through `taylor_with_budget` at 1e8, then `Display`,
  `simplify`, `substitute`, `==` with a clone and with a second expansion, `series_expansion`, its
  `Debug`, `limit_via_series`, clone and drop, and the drop of the input: on 2 MB debug every call
  returns (slowest expansion 1.1 s, peak 31 MB); on 8 MB release all 35 complete, slowest step
  0.18 s, peak 31 MB. `rev1-resources/shapes_dbg.log`, `shapes_rel2.log` (`shapes_rel.log` is the
  first release run, which also called `limit_with_error` and `limit_enhanced`: B2, minor 1).
- Inputs shared as a DAG (40 doublings of `x`, of `1+x` as a product, of a constant inside `f`, of
  the point): `mentions_symbol` in microseconds; `series_expansion`, `taylor_with_budget` and
  `limit_via_series` refuse in at most 0.21 s with at most 4.2 MB.
- Programmatic depth 5 400 and 33 000 (chains of `f`, `sin`, `exp`, negation, towers, right sums,
  products, quotients, a zigzag) through the engine on 8 MB release: every expansion returns in at
  most 0.2 s. A result that keeps a 33 005-deep input kernel cannot be simplified or substituted on
  8 MB (HEAD's recursion, the input's own depth); `Display`, `==`, clone and drop work.
- Expansions at infinity with the budget on the shapes; constants such as `sin` of a 99 980-digit
  literal, `sin(2^(2^40))`, `3^3^3^3`, 6 000 radicals, 6 000 float literals: all within 0.15 s in
  release.
- The REPL: the taylor arm walks the DAG for the variable test, maps the errors, and its messages are
  bounded (`display_estimate`); `expr_display_size` still bounds the new `Display` (2 more bytes per
  numeric leaf, 16 per unary node, 332 per float); on the 64 MiB REPL thread the 100 KB caret and
  `x^-` towers parse, and `limit`, `simplify` and `Display` of them work in release.
- `Display` and `Drop` have no recursion over the depth; the TLS teardown case passes.

## Files

`fu2/s1impl/rev1-resources/`: `probe.rs` (`build_probe.sh debug|release`), `headprobe.rs`,
`consume.rs`, `consume_depth.py`, `parse_depth.py`, `shapes.py` and `runshapes.sh`, `dropbench.rs`,
`inf.rs`, `sc.rs`, `dbgpt.rs`, `fixtry/parser_atom_fix.diff` and `fixtry/simplify_fix2.diff` (the
two checked fixes, as diffs against the tree; the rebuilt rlibs were deleted), logs `pd_dbg_2mb.txt`,
`pd_rel_8mb.txt`, `consume_rel.txt`, `consume_dbg.txt`, `deep_rel.txt`, `shapes_*.log`,
`sym_debug_tests.log`. The release binaries (`probe_rel`, `headprobe_rel`, `consume_*_rel`,
`dropbench_*_rel`) are kept; `target/release` and the cargo build of HEAD (`headws/target`) were
deleted after use.
