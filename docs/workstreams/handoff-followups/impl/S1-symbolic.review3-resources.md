# S1 implementation, review 3: resources lens (memory, ownership, recursion, input structure)

Reviewed: the uncommitted S1 implementation in `/home/user/RustMath` (branch
`claude/gracious-gates-e6b6l7`, HEAD 8a0b310) after fix-up round 2, with `fu2/s1impl/S1_impl.md`
(including its "Fix-up round 2" part), `fu2/s1impl/tests_report.md`, `fu2/s1impl/owner_decision.md`
and the two round-2 reviews. No repository file was edited. Scratch:
`fu2/s1impl/rev3-resources/` (paths below are relative to it unless they start with `fu2/` or a
crate name).

Verdict: **not approved**. Two blockers under S1_BAR (c) in this lens (B1, B2). The tests pass. The
two blockers of round 2 are fixed for the shapes they named. B1 below is a gap in round 2's B2 fix:
the leaf-text cap counts a float as 24 characters, and a float prints up to 326.

Both blockers are in the class of round 2's B2 and of red-team round 3 structure H2 (one of the
owner's three holes): the consumers of a result (`simplify`, Display) must stay within the bar
because the result is bounded. The code states this as a property. `repl.rs:11205` (the
`TAYLOR_WORK` doc) says "`simplify`, `subs`, `==` and the printing of a result stay within the S1
bar (20 s, 512 MB)". `series_const.rs:624` (`MAX_RESULT_TREE`) says a tree-walking consumer is
"bounded by the cap times its cost per node". B1 and B2 are counterexamples, each measured on the
built code.

## How it was measured

- The port as cargo builds it: debug (`target/debug`, fresh) and release (`target/release`, built
  here with `cargo test --release -j 2 -p rustmath-symbolic --no-run`, 53 s; deleted after use).
- Probes are compiled with rustc against cargo's rlibs (`build.sh SRC dbg|rel OUT`; rlib lists
  `port_rlibs_dbg.txt`, `port_rlibs_rel.txt`):
  - `probe4.rs` (`p4b_rel`) reads a text input and runs steps on one thread of a given stack, each
    with its wall time and VmHWM. The steps are `taylor_with_budget` with 1e8 (the REPL's
    `TAYLOR_WORK`), `series_expansion`, `limit_via_series`, `limit_with_error`, Display into a
    String, `simplify`, `substitute` x = 2 and `==` of the result, and `simplify` of the input.
  - `consume2.rs` is review 2's depth probe. `c2_head_*` are review 2's HEAD builds; `c2_port_*`
    were rebuilt against the current code.
  - `repl_probe.rs` evaluates lines through `ReplContext::eval` on a 64 MiB thread (debug, against
    the jupyter lib built here).
  - `monitor.sh` samples VmHWM every 2 s and kills a run at a deadline.
- Memory runs used `ulimit -v 6000000`. Unless named otherwise, a run is release on an 8 MB thread.

## Tests

| run | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic --no-fail-fast -- --test-threads=2` (debug) | unit 675 passed, 3 ignored; `s1_depth` 9; `s1_fuzz_data` 7; `s1_payload` 2 passed, 1 ignored; `s1_regressions` 26 passed, 2 ignored; `s1_review2` 2; `s1_singular_points` 6; doc 116 passed, 11 ignored; 0 failed (`sym_debug_tests.log`) |
| the six release test binaries | 6, 26 (2 ignored), 7, 9, 2 (1 ignored), 2 passed; 0 failed (`rel_tests.log`) |
| release `--ignored --test-threads=1` | `s1_regressions`: both pass (F_10 0.19 s, F_11 0.36 s, F_12 refused); `s1_payload` `b2r2_consumers_of_large_leaf_results_release`: passes in 11.8 s (order 1 300 Display 8.6 s) |
| `cargo test -j 2 -p rustmath-jupyter --lib -- taylor latex expand_under` | 7 passed (`jup_tests.log`) |
| downstream unit binaries (calculus, functions, geometry, plot3d; fresh, run directly) | 122, 72, 417 (1 ignored), 78 passed; 0 failed |

The tests of the first step are unchanged in their expected outcomes:

- These files are byte-identical to the harness copy (`fu2/s1impl/protoharness/work/src`):
  `tests/s1_common/check.rs`, `eval.rs`, `data/s1_fuzz.tsv`, `data/gen_s1_fuzz.py` and
  `data/s1_gp.py`.
- `s1_regressions.rs` differs from the harness copy only in doc comments and one rustfmt wrap, as
  in round 2. Its mtime is 20:02 (round 1).
- `data/s1_singular_points.tsv` starts with the 106 450 bytes of the first step, byte for byte. It
  also starts with the 123 295 bytes of round 2's regeneration (677 lines). Round 2 appended three
  class (e) rows. `gen_s1_singular.py` reproduces the file byte for byte.

## The blockers of round 2

| round 2 | status | evidence |
|---|---|---|
| B1, `simplify` depth per level | fixed | see the reproductions below |
| B2, the leaf payload of a result | fixed for its five shapes; the count is wrong for floats | the five inputs below; B1 of this review |

B1 reproductions (`repro_simplify.txt`): review 2's eight direct reproductions return on the port.
These shapes abort HEAD and return on the port (`simplify`):

- release: `f(f(...))` at 10^6 levels (8 MB) and the zigzag at 10^6 levels (2 MB);
- debug, 2 MB: `f(x + f(...))`, `L = sin(x + L) + x`, `f(x, f(...))`, `f(sin(f(...)))`,
  `L = (x ^ L) + x` (2x10^5 levels each) and `L = -(L + x) + x` (10^5 levels).

Exact binary searches, HEAD / port, equal everywhere (`d_rel8.txt`, `d_dbg2.txt`):

| tree, consumer | release 8 MB | debug 2 MB |
|---|---|---|
| `f(f(...))`, `Expr::limit` at 1 | 13 433 / 13 433 | 1 125 / 1 125 |
| zigzag, `Expr::limit` at 1 | 14 553 / 14 553 | 1 305 / 1 305 |
| `f(f(...))`, `substitute` | 13 434 / 13 434 | |
| zigzag, `substitute` | 14 554 / 14 554 | |
| `f(f(...))`, `==` | | 3 530 / 3 530 |
| zigzag, `expand` | | 959 / 959 |
| right sum, `expand` | | 1 919 / 1 919 |

B2: review 2's four 100 KB inputs at order 20 000, and `1/(1+N-x)` at the point N, are
`LimitExceeded(TreeText)` through `taylor_with_budget` in at most 0.02 s.

## Blockers

### B1. A float leaf counts 24 characters and prints up to 326: Display of a 99 KB input's result takes 662 MB

S1_BAR (c), through `taylor_with_budget` with the REPL's budget. This is the class of round 2's B2:
a consumer of a result exceeds 512 MB.

`node_text` (`series_const.rs:643`) counts a `Real` leaf as 24 characters (`:649`). Its doc and the
doc of `MAX_RESULT_TEXT` (`:629`) call the count "an upper bound of the printed length". It is not
one. Display prints a float with `{}` (`expression.rs:608`), and Rust's `{}` for f64 never uses an
exponent:

- `1.0e300` prints 301 digits and `.0` (303 characters);
- `1.0e-308` prints 310 characters;
- `4.9e-324` prints `0.` and 323 zeros before the `5` (326 characters).

So the node cap (2^21 nodes) allows about 2^21 float occurrences. The text cap sees 24 x 2^21 = 50
MB of text, but Display prints up to 680 MB. An opaque call whose arguments are floats is one
shared kernel node. It names a float at almost every node, so the tree is almost all floats.

Input `in/gsub.txt`: `g(4.9e-324, 4.9e-324, ..., 4.9e-324)/(1-x)`, 11 000 arguments, 99 008 bytes,
at 0, through `taylor_with_budget` with 1e8:

| order | `taylor_with_budget` | Display into a String | `simplify` | `substitute` | `==` |
|---|---|---|---|---|---|
| 189 | Ok, 0.010 s, VmHWM 9 MB | **685 523 667 bytes, 0.43 s, VmHWM 662 MB** | 0.24 s, 214 MB | 0.14 s | 0.005 s |
| 190 | `LimitExceeded(TreeSize)` | | | | |

- `series_expansion` with `to_expr` gives the same result and the same 662 MB.
- With `1.0e-308` (`in/gfloat.txt`), order 189 prints 652 083 667 bytes, VmHWM 630 MB.
- The text cap counts this result at about 50 MB (189 x 11 000 x 24).

The REPL does not print it. Its own estimate `expr_display_size` counts 332 per float
(`repl.rs:11765`) and refuses the text before formatting it. The library's Display of the result is
the consumer that fails.

`display_estimate` (`series.rs:151`) has the same 24 (minor m2).

Required:

- Count a float leaf by a true upper bound of its Display length. This applies to `node_text` and
  to `display_estimate`. Either measure the formatted length, which `tree_text` memoizes per shared
  node, or use a constant bound of at least 327, as the REPL does with 332.
- Correct the two doc comments.
- Regression test (release, 8 MB thread): `in/gsub.txt` at 0, order 189, through
  `taylor_with_budget` with 1e8. Either it is refused, or Display of the result stays under 512 MB
  VmHWM and 20 s.
- Unit test: `node_text(&Expr::Real(f)) >= format!("{}", Expr::Real(f)).len()` for f in 5e-324,
  1e-308, f64::MIN_POSITIVE, -1e-300, 1e300 and f64::MAX.

### B2. `simplify` of a small result folds powers that the engine refused: over 60 s from a 24-byte input, 667 MB from another

S1_BAR (c), through `taylor_with_budget` with the REPL's budget, then `simplify` of the result. This
is the class of round 2's B2 and of structure H2: a tree-walking consumer of a result is not bounded
by the result's caps. The REPL path is affected too.

`simplify` (HEAD's rules, `simplify.rs:619` to `:640`) folds c^n for every rational c and
|n| <= `MAX_FOLD_EXPONENT` = 4096 (`:500`). Nothing bounds the size of c^n. The comment at `:623`
says "bounded exponent so simplify can never blow up memory", but only the exponent is bounded.

The engine refuses to compute a power whose estimated size passes `MAX_EXACT_POWER_BITS` (2^20
bits) and keeps it as a kernel. A kernel made from an input node renders as that node
(`mk_kernel_src`, `series_const.rs:2011`). A power of a kernel in a monomial renders as
`Pow(kernel node, e)` (`kpow`, `:1901`).

Both result caps count the few characters of that source text, not the integer that `simplify`
builds from it. `simplify` walks the tree, so it folds the source again at every occurrence. Under
a kernel power it then folds the folded value to the e-th power.

The engine itself takes no time on these inputs. Each row below is release, 8 MB thread, with
`taylor_with_budget` at 1e8 (`p4b_rel`; logs in this folder):

| input (bytes), at 0 | `simplify` of the input | order | `taylor_with_budget` | the result | `simplify` of the result |
|---|---|---|---|---|---|
| `((2^256+1)^4096*x)^4096` (24) | 0.008 s | 4096 | Ok, 0.000 s | 48 bytes: `(((((2 ^ 256) + 1) ^ 4096) ^ 4096) * (x ^ 4096))` | **not done after 60 s, VmHWM 378 MB when killed** (`powK_simp.txt`); it builds ((2^256+1)^4096)^4096, an integer of about 4.3e9 bits (537 MB) |
| `1/(1 - (2^256+1)^4096*x)` (25) | | 20 | Ok, 0.000 s | 951 bytes | 7.0 s, 45 MB |
| the same | | 40 | Ok, 0.000 s | 1 931 bytes | **36.7 s**, 140 MB (`geoK_40.txt`) |
| the same | | 100 | Ok, 0.000 s | 4 873 bytes | **336.5 s, VmHWM 730 MB** (`geoK_100.txt`) |
| `exp(x + (2^256+1)^4096)` (24) | 0.008 s | 1 000 | Ok, 0.003 s | 1.2 MB | 7.7 s, 132 MB |
| the same | | 5 000 | Ok, 0.051 s, 39 MB | 38 MB | **38.3 s, VmHWM 667 MB** (`exp2p256_5000.txt`) |
| `sin((2^254+1)^4096)/(1-x)` (26) | 0.008 s | 1 000 | Ok, 0.009 s | 48 KB | 7.7 s, 130 MB |
| `exp(x + N^4096)`, N = 10^599 written out with a last digit 7 (615) | 0.157 s | 200 | Ok, 0.000 s | | **31.6 s**, 206 MB |
| the same | | 600 | Ok, 0.001 s | | **95.4 s, VmHWM 597 MB** (`expNpow_600.txt`) |

Notes on the rows:

- In the last four inputs the source is folded once per occurrence. `sin((2^254+1)^4096)` is a
  power that the engine did compute exactly (just under its 2^20-bit cap), but the kernel renders
  its input node.
- In the first two inputs the exponent of a kernel power multiplies the fold: K^k becomes
  ((2^256+1)^4096)^k.
- `substitute`, `==` and Display of these results stay fast: 0.013 s, 0.003 s and 0.78 s at order
  5 000.
- `limit_via_series` and `limit_with_error` are not affected. They give `Finite(0)`, `Finite(1)`
  or `Unknown` at once.

The REPL path (debug, 64 MiB thread; `repl2_dbg.txt`, `repl3_dbg.txt`):

```
t = taylor(exp(x + (2^256+1)^4096), x, 0, 300)     Ok, 0.012 s, VmHWM 10 MB
simplify(exp(x + (2^256+1)^4096))                  Ok, 0.720 s
s = simplify(t)                                    Ok, 34.8 s, VmHWM 49 MB
t = taylor(((2^256+1)^4096*x)^4096, x, 0, 4096)    Ok, 0.001 s (prints the 48-byte result)
simplify(((2^256+1)^4096*x)^4096)                  Ok, 0.700 s
s = simplify(t)                                    not done after 60 s (killed)
```

Neither cap can catch this. The 48-byte result of the first row has 13 nodes.

Required: a property, not a particular mechanism. `simplify` of any result that an entry point
accepts stays within 20 s and 512 MB (release, 8 MB thread), as the `TAYLOR_WORK` doc claims. Ways
that fit:

1. The result caps count what `simplify` builds. `tree_text`'s DAG walk can also compute, per node,
   the size of the node's value when its subtree folds under `simplify`'s rules: integer and
   rational leaves under `+ - * /`, and `^` with an integer exponent of absolute value at most 4096.
   The sizes are max + 1, a + b, a + b and |n| a. The walk then counts that size in place of the
   subtree's text, with a cap chosen by measurement. num-bigint's powering is superlinear, so one
   fold of 2^27 digits would itself take minutes; a power of about 2^24 bits took about 1.5 s here.
2. Render a kernel so that `simplify` has nothing large to fold. When the engine computed a rational
   argument, render the argument from its normal form instead of the input node. In addition, give
   `simplify` the size guard that its comment claims: fold c^n only when bits(c) x |n| is at most a
   cap such as `MAX_EXACT_POWER_BITS`. This changes HEAD's `simplify` only for powers of over 2^20
   bits.
3. Make `simplify` fold each shared node once (memo by address, as `tree_text` does), together with
   the size guard of way 2. This removes the per-occurrence repetition of the last four inputs. The
   guard handles the first two. The values are HEAD's; only the sharing differs.

Regression tests (release, 8 MB thread, `#[ignore]` as the other budget tests; written before the
fix): the five inputs above at the orders of the table, through `taylor_with_budget` with 1e8. Each
must be refused, or `simplify` of the result must finish within 20 s with VmHWM under 512 MB.
Correct the `TAYLOR_WORK` doc comment and the `MAX_RESULT_TREE` comment to what is guaranteed.

## Minor issues

1. **The printing margin at the text cap is thin.** `MAX_RESULT_TEXT`'s doc says 2^27 characters
   print in about 9 s, at the rate of 10^5-digit integers (15 MB/s). num-bigint's decimal
   conversion is superlinear, so larger leaves print more slowly. The engine can build a leaf of
   about 425 000 digits under the budget. `exp(x + (10^300-1)^1050*7^130000)` (34 bytes) at 0,
   order 312, gives a result of 133 MB of text, just under the cap (each of 313 terms holds the
   424 850-digit integer). Display took 17.8 s and 18.8 s in two release runs (`expbig2_312.txt`),
   about 7 MB/s. That is 94 % of the bar, so a slower machine could cross it. Suggested: weight
   integer digits by the conversion's growth, or lower the cap for large integers. Also correct the
   doc's rate.
2. **`display_estimate` (`series.rs:151`) counts a float as 24 characters.** So the
   `UndecidedZero` message (limit 200) and `ExpansionPoint`'s Debug (limit 60) can print about 14
   times their limit when the constant holds floats. This is cosmetic; the text is bounded by that
   factor. It is fixed with B1.
3. **Some error strings embed an input name in full.** `series_engine.rs:1935`
   (`Unsupported("{name}(...) of an expression in the variable")`) is an example. A call with a
   100 KB name gives a 100 KB message in the REPL's `taylor:` error. The message is bounded by the
   input size and is cosmetic. Truncating names in messages, as `point_text` does, would match the
   bounded Display of the other variants.

## What held (no finding)

- `simplify` is a work list over every node kind, and its results are HEAD's. Its memory on a
  2^21-node result is 214 MB in 0.24 s (`in/gsub.txt`, order 189).
- `tree_text` is linear in the DAG.
  - A node held by more than one `Arc` is memoized. Every node reachable along two paths has a
    strong count of at least 2, so nothing is walked twice.
  - The memo is keyed by address while all the nodes are alive, so no key is reused.
  - `expansion_text` counts h, which holds the point, at every term, as `to_expr` names it.
  - The rendering memo of `public_expansion` keeps each coefficient's `Arc` alive during the
    count.
- The limit paths read only a rational first coefficient, or render nothing, so neither cap nor
  consumer is involved. On the blocker inputs they return at once with 4 to 9 MB.
- Display, Drop and `mentions_symbol` have no recursion over the depth. Drop of the 2M-node results
  took 0.003 s.
- `Expr::limit`, `substitute`, `==` and `expand` reach HEAD's depth exactly (the table above).
- The REPL's taylor arm maps the errors and has a bounded display guard. Its tests pass.
- The engine's `Session` guard clears the thread-local state on drop, on a panic too.

## Files

`fu2/s1impl/rev3-resources/`:

- scripts: `build.sh`, `probe4.rs`, `consume2.rs`, `depth2.py`, `repl_probe.rs`, `monitor.sh`;
- inputs: `in/` (`gsub.txt`, `gfloat.txt`, `powK.txt`, `geoK.txt`, `exp2p256.txt`, `sinK.txt`,
  `expNpow.txt`, `expbig2.txt`, `repl2.txt`, `repl3.txt`);
- logs: `sym_debug_tests.log`, `rel_tests.log`, `jup_tests.log`, `repro_simplify.txt`,
  `d_rel8.txt`, `d_dbg2.txt`, `powK_simp.txt` (and `.mon`), `geoK_40.txt`, `geoK_100.txt`,
  `exp2p256_5000.txt`, `expNpow_600.txt`, `expbig2_312.txt`, `repl2_dbg.txt`, `repl3_dbg.txt`
  (and `.mon`);
- binaries kept: `p4b_rel`, `c2_port_rel`.

`target/release` was deleted after use. The debug probe binaries were deleted to save disk; they
rebuild with `build.sh`.
