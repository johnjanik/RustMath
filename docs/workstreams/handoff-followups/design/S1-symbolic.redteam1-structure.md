# S1-symbolic, revision 7: red team 1 (input structure and recursion)

Design attacked: `fu2/design/S1-symbolic.md`, revision 7. I read all of it, with the focus on §2.5
(recursion depth), §2.12 (parser), §2.14 (Drop), §2.15 (caps, and "why these values"), §2.16
(invariants, fact 3) and §6 step 8.

Prototype: `EXP/rev7/proto8/`, copied unchanged to `EXP/redteam1-structure/src/proto8/`
(EXP = `fu2/design/S1-symbolic`). I added four probes there:

- `split.rs` runs the parser and the engine on separate threads, so the two stack limits are
  measured apart. `MODE` is `parse`, `engine` (parse on a 1 GiB thread, then `taylor_with_budget`
  on a STACK thread), `both` (both on one STACK thread, as a caller would), or `head` /
  `headboth` (HEAD's `Expr::taylor` in the same places). `DISPLAY_TOO`, `EQ2` and `DROP_TOO` add the
  result's Display, PartialEq against a second expansion, and Drop on the same thread. `ENTRY=se`
  and `ENTRY=lim` select the other two public entry points.
- `apishape.rs` builds inputs with the `Expr` API in the structure that §2.12's parser gives them
  (`a/b` is an operator and the fold applies only to literal / literal). The prototype links
  HEAD's parser, which cannot produce these trees.
- `dagdrop.rs` drops a DAG chain whose every node holds the previous one twice.
- `litparse.rs` times `Integer::from_decimal_str` and the Display of a literal.

Scripts: `bsearch.py` and `bsapi.py` binary-search the deepest input that is handled, with the
generators in `shapes.py`. `rrun.py` and `timed.py` run one call. Every run had a 4 GiB
address-space cap and a timeout. The builds use `build.sh` (the revision 7 script with its
output in `redteam1-structure/bin/`) against the HEAD rlibs of `EXP/rev5/rlibs_*.txt`. Budget:
1e8 (`TAYLOR_WORK`), or `LIMIT_SERIES_WORK` on the limit path. "Debug 2 MB" means the debug build on
a 2 MiB thread, and "release 8 MB" means the release build on an 8 MiB thread.

## H1 (blocker): right-deep `^` chains overflow the engine on text that HEAD's parser and HEAD's taylor handle

**What happens.** `konst_of_inner` converts `l ^ r` with `a = konst_of(l)?; b = konst_of(r)?`.
`expand_pow` likewise recurses into its exponent. So a right-deep power tower recurses once per
`^`, and each level holds the large `konst_of` + `konst_of_inner` frame: about 5.3 KB in debug
and 720 B in release.

The parser reads the same tower through `power -> power` alone, at about 1.46 KB per level in
debug and 370 B in release. §2.12 keeps that rule: `exponent := ('-' | '+')* power`. So the
design's argument fails for this shape. That argument appears in §2.5 ("on the thread that parsed
an input `taylor_with_budget` cannot overflow first") and in §2.16 fact 3. It was measured only on
nested calls, parentheses and `/` with parentheses, where the parser costs 10 KB or more per level.

**Inputs** (generator `shapes.py`, `tow1`, `xtow2`, `towx1`). Parser and engine limits are from
`bsearch.py`. The overflows are aborts ("fatal runtime error: stack overflow", rc -6).

| input | thread | parser reads | engine handles | HEAD's taylor |
|---|---|---|---|---|
| `x+1^1^...^1` | debug 2 MB | 1434 carets | 397 | 1400 and more (`x + 1`, 0.4 ms) |
| `x+2^2^...^2` | debug 2 MB | 1434 | 397 | 1400 and more |
| `x^2^2^...^2` at 1 | debug 2 MB | 1434 | 398 | 1430 and more |
| `x+sin(1)^sin(1)^...` | debug 2 MB | 1424 | 395 | 1400 and more |
| `x+1^1^...^1` | release 8 MB | 22 679 | 11 623 | 28 880 (parse on a large thread) |
| `x^2^2^...^2` at 1, `x+sin(1)^...` | release 8 MB | 22 679 | 11 613 | 22 600 and more |
| `x^x^...^x` at 1 | debug 2 MB | 1434 | 602 | 188 |
| `x^-x^-...^-x` at 1 (API-built, §2.12 reads it) | release 8 MB | | 16 862 | 3 826 |

Reproductions, timed with `timed.py`. Each one is parse and call on one thread, as a REPL line
or a test runs it:

- `inputs/tow1_400.txt`, 804 bytes, debug 2 MB:
  `bin/split both inputs/tow1_400.txt 0 1 2097152`. The prototype aborts after 7 ms with a stack
  overflow in `konst_of_inner` / `konst_of` (`gdb` backtrace). HEAD (`headboth`) returns `(x + 1)`
  in 0.4 ms. The same abort happens through `series_expansion` (`ENTRY=se`) and the limit path
  (`ENTRY=lim`). At 397 carets all three entry points work: `Ok`, 45 721 units; the limit path
  gives `Finite(1)`.
- `inputs/tow1_12000.txt`, 24 KB, release 8 MB: the prototype aborts after 15 ms with 14 MB RSS.
  HEAD returns `x + 1` in 10 ms. The aborts recur at 15 000, 20 000 and 22 000 carets (44 KB);
  HEAD answers in 14 to 19 ms.
- The REPL's own thread (64 MiB, `kernel.rs:45`), debug: `inputs/tow1_13000.txt` (26 KB) aborts the
  process after 83 ms. HEAD returns in 26 ms. So in a debug REPL one line of 26 KB kills the kernel.
  In release, 64 MiB covers 100 KB of text for this shape.
- `x^2^...^2` at 1, 500 carets (1 KB), debug 2 MB: aborts. HEAD returns in 0.8 ms (depth 502).
- `x^x^...^x` at 1, 700 carets, debug 2 MB: aborts. HEAD aborts too, so this one is a (c) hole
  only.

**Correctness** where it works: gp `print(truncate(x + 1^(1^(1^1)) + O(x^2)))` gives `x + 1`, which
is the prototype's result at 397 carets. The x^x^x row is checked under H2.

**Over the bar.** Yes. It is (b), a regression against HEAD on 0.8 KB to 45 KB of text in debug
and in release, and (c), a stack overflow through `taylor_with_budget`, `series_expansion` and the
limit path.

**Sections.** §2.5 ("Recursion depth", the depth table, "Parsed inputs stop in the parser
first"), §2.12 (the `^` rule kept recursive), §2.15 ("the parser reads at most about 160 nested
levels on a 2 MB debug thread"), §2.16 fact 3 and the rows `from_expr`, `expand`, `expand_pow`
("depth at most D"), and §6 step 8.

§6 step 8 cannot catch this. It measures "the six shapes of §2.5", which do not include a `^`
chain. Its remedy also cannot be applied: a parser nesting limit "below the engine's limit and not
below HEAD's parser limit" does not exist here, because HEAD's parser reads 1434 levels and the
engine handles 397.

**Fix that closes the class.** No engine recursion may be cheaper for the parser than for the
engine. The parser has three kinds of rule:

- loops (`+ -`, `* /`, sign runs);
- the `^` rule, which recurses cheaply;
- parentheses and calls, which recurse at 10 KB or more per level in debug.

So:

1. Read a right-deep `^` chain along its right spine with a loop, in `konst_of` and in
   `expand_pow`, the way §2.5 already reads the `+ -` and `*` left spines. Collect the bases
   b_1 ... b_k and the top exponent t. Convert or expand t first. Then fold
   `alpha = b_i ^ alpha` from i = k down to 1, with the exact-constant check at each step, and
   with `kpow_const_src(.., Some(node_i))` so that each kernel still takes its input node. Then
   only bases recurse, and a nested base needs parentheses in text.
2. Add the `^` tower shapes (`x + 1^...^1`, `x^2^...^2`, `x^x^...^x`, `x^-x^-...`) to the
   depth table of §2.5, to `deep_api_inputs`, and to the parser-against-engine check of §6 step 8.
   State in §2.16 fact 3 that the engine recurses only where the parser recurses through
   parentheses or a call.

## H2 (blocker under (b) if printing the result counts, as §2.15 says it does): results twice as deep as their input abort Display on release threads where HEAD's result prints

**What happens.** §2.15 caps kernel depth at max(1024, 4 D). It justifies leaving Display
recursive with "any result prints on the thread where its input could be parsed", because "the
parser reads at most about 160 nested levels on a 2 MB debug thread and about 4100 on an 8 MB
release thread".

With H1's `^` chains the parser reads 22 679 levels on an 8 MB release thread. The engine handles
`x^x^...^x` there, since `expand_pow` is small in release. Its value at a point a != 0, 1 is built
from computed kernels `exp(log(a) * K)`, two levels per input level. So the result has depth
2 D + 2. HEAD's result has depth D - 2.

**Inputs.** `inputs/towx_<n>.txt` is `x^x^...^x` with n carets. The point is 2, order 0, run with
`bin/split_rel both ... 2 0 <stack>`, and with `DISPLAY_TOO=1` for the Display.

| n (text) | thread | `taylor_with_budget` | Display of the result | HEAD |
|---|---|---|---|---|
| 16 000 (32 KB) | release 8 MB | `Ok`, 61 ms, 2.98e7 units, kernel depth 32 002, 16 001 kernels, VmHWM 26 MB | 256 KB in 5.7 ms | depth 15 998, prints 96 KB |
| 18 000 (36 KB) | release 8 MB | `Ok`, 72 ms, 3.35e7 units, depth 36 002, VmHWM 29 MB | **stack overflow, abort** | 7 ms, depth 17 998, prints 108 KB |
| 20 000 (40 KB) | release 8 MB | `Ok`, 73 ms, 3.72e7 units, depth 40 002 | **abort** | prints 120 KB |
| 4 500 (9 KB) | release 2 MB (a `cargo test --release` thread) | `Ok`, 15 ms, 8.4e6 units, depth 9002 | **abort** | 1.4 ms, depth 4498, prints 27 KB |
| 5 000, 5 500 | release 2 MB | `Ok` | **abort** | prints |

PartialEq of the depth 40 002 result against a second expansion holds on the 8 MB release thread
(`EQ2=1`: `true` in 2.9 ms). Only Display fails.

On a 2 MB debug thread the engine stops at about 600 levels first (H1), so the result has depth
about 1200 and prints. The REPL's 64 MiB thread prints it.

**Correctness.** For `x^(x^x)` at 2, order 1, the prototype gives c0 = `exp(log(2) exp(2 log(2)))`
and c1 = `16 (4 log 2 + 4 log^2 2 + 2)`. gp (`out/tow.gp`): `polcoef(f(2+x+O(x^2)),0)` = 16 and
`polcoef(..,1)` = 107.1104124466..., which equals 16 (4 log 2 + 4 log^2 2 + 2) to 38 digits. For
the 4-level tower, c0 = 65536 matches gp.

**Over the bar.** The engine call itself returns. The abort is in the result's Display, which
HEAD's result survives on the same thread. If (b) counts "taylor, then print" as the behaviour
HEAD handles, this is a regression abort. The design makes it count: §2.15 rests the decision to
leave Display recursive (§7 item 5) on the claim that results print where their inputs parse. If
only the series call counts, this is a minor issue. The §2.15 claim is false either way.

**Sections.** §2.15 (the kernel-depth row, "Which inputs could reach the kernel-depth cap", "Why
these values"), §2.2 (computed kernels render normal forms), §2.8 (result depth), and §7 items 5
and 8.

**Fix that closes the class.** Make the result's depth track the input's depth, not 4 D, and do
one of these:

- (a) Render the kernel of a value at the point from the input node with the point substituted.
  §7 item 8 already proposes this for the text. That bounds such a kernel by D + the point's depth.
- (b) Cap the depth of computed kernels at max(1024, D + 64), and accept `LimitExceeded` for the
  rest. This keeps the honesty rule but refuses towers that HEAD answers.
- (c) Make Display iterative (a formatter work list), which removes the dependence on depth
  altogether.

Then correct the §2.15 claim and add the 18 000-caret tower at 2, order 0, with its Display on an
8 MB release thread, to the depth tests.

## H3 (blocker, (c)): left-deep `/` chains, which §2.12's parser reads by a loop, overflow the engine

**What happens.** Under §2.12, `a/b` is an operator. `multiplicative` reads `x/2/2/.../2` and
`x*3/2*3/2...` by its loop into left-deep `Div` and `Mul`/`Div` trees. The fold applies only to a
literal over a literal, so after `1/2` the next `/2` stays a `Div`. The parser does not recurse on
these at all.

The engine's chain loops cover the `+ -` spine and the `*` spine only:

- `expand` reads `Div(l, r)` as `let a = expand(l)?`, one level per `/` (`gdb`: `ser::expand`
  repeated).
- `konst_of_inner`'s generic arm does `konst_of(l)` per `/`.

HEAD's parser turned `2/2` into one literal token, so the same text had half the `Div` levels.

**Inputs.** Built by `apishape.rs` in the §2.12 structure. Limits come from `bsapi.py`. These are
the deepest inputs handled, then the first overflow.

| input (text length at the overflow) | debug 2 MB | release 8 MB | HEAD's taylor on the same tree |
|---|---|---|---|
| `x/2/2/.../2` | 1290, fails at 1302 (2.6 KB) | 40 253, fails at 40 643 (81 KB) | 188 debug, 6 579 release |
| `x + (1/2/2/.../2)` (constant, `konst_of`) | 398, fails at 400 (0.8 KB) | 11 600, fails at 11 697 (23 KB) | 187 debug, 6 579 release |
| `x*3/2*3/2...` (pairs) | 424, fails at 427 (1.7 KB) | 14 913, fails at 15 010 (60 KB) | 94 debug, 3 290 release |
| `(1)/x/x/.../x` at 1 | 1290, fails at 1300 (2.6 KB) | | 188 debug |

Where it works the values are right. For example, `1/x^1290` at 1 gives `(1 + (-1290 * (x - 1)))`.

As HEAD parses it, the text `x+(1/2/.../2)` with 12 000 and 12 500 slashes (24 and 25 KB) is a
chain of 6000 and 6250 `Div` nodes. HEAD's taylor returns it on an 8 MB release thread in
4.6 to 4.7 s with 3.4 to 3.7 GB. The prototype on that HEAD parse takes 5 to 8 ms. The tree that
§2.12 gives the same text overflows the prototype (`apishape_rel divlc 12000 8388608`). So there is
also a narrow release regression, though HEAD's own run there exceeds the memory bar.

**Over the bar.** Yes, (c). A text of 0.8 KB (debug 2 MB) or 23 KB (release 8 MB) overflows
`taylor_with_budget`. `series_expansion` and the limit path share `expand` and `konst_of`. The
parser cannot stop first, because it has no recursion for this shape. Every run aborts within
0.2 s, at under 30 MB.

**Sections.** §2.5 (chains: "A chain of `*` is read the same way" leaves out `/`), §2.12 (the
structure change of `multiplicative`), §2.16 (rows `expand_div`, `from_expr`), §6 step 8, whose
nesting-limit remedy cannot apply to a loop, and §3.2.

**Fix.** Read `*` and `/` as one chain along the left spine in `expand_product` and in `konst_of`,
with a sign per factor:

- constant divisors go to `product_many` through `inverse_src(false, Some(r))`;
- series divisors go through `inv` and `mul` in order.

That is the same loop as today's `*` chain, so there is no recursion per factor. Together with
H1's `^` loop, every parser loop then has an engine loop. Add `x/2/.../2`, `x + 1/2/.../2` and
`x*3/2*3/2...` at 100 KB to `deep_api_inputs`.

## Families tried that held

- **Wide inputs** (release 8 MB and debug 2 MB, 66 to 110 KB):
  - `x*x*...*x` (50 000 factors);
  - `x + f(1, ..., 1)` (49 990 arguments) and `x + f(1, ..., 17 999)`;
  - `1/(x + y1 + ... + y16000)`;
  - `x + y1*...*y16000`;
  - `x + sin(1) + ... + sin(11000)`;
  - `x^(1 + ... + 1)`;
  - `x + sqrt(2) + ... + sqrt(8999)`;
  - `x - x - ... - x`;
  - `x*sin(1)*...*sin(1)`;
  - `(x+1)*...*(x+8999)`.

  Each is `Ok` within 0.45 s (debug) and 26 MB, or refused by the budget at 1e8 within 0.45 s. The
  only exceptions are the `/` chains of H3. With HEAD's parser, `x*3/2*3/2...` is a pure product
  chain and holds at 49 000 pairs.
- **Nested calls and parentheses** (debug 2 MB): `x + sin(...(1))` parses to 162 levels and
  expands to 397. `((...(x)...))` parses to 215 levels and expands beyond 2000. So the parser
  stops first, as §2.5 says.
- **Result consumers at depth**:
  - PartialEq of a depth 40 002 result against a second expansion, release 8 MB: 2.9 ms.
  - HEAD's drop glue on a DAG chain whose every node holds its child twice (shared children are
    left in place by §2.14 and dropped through the glue): 100 000 levels on release 8 MB, 10 000
    on debug 2 MB.

  The design's `Drop` is not in the prototype, so its stack-gated variant was not measured on
  such DAGs.
- **Huge literals**: `Integer::from_decimal_str` (§2.12's literal reader) of 99 990 digits takes
  11 ms release and 106 ms debug, and its Display 8 ms and 80 ms. Both are linear enough.
- **Caps (code reading)**:
  - The factor cap is checked in `mono_mul` before each push, the term cap in `merge2`,
    `sum_many` and `mul`'s accumulator before each new entry, the kernel cap in `intern` before
    creation, and the depth cap in `mk_kernel_inner` before the key is written. All four come
    before the work they bound.
  - The cached depth (`ksize`, `kpow_depth`, `bal_depth`) matches the shape `mono_expr` builds:
    numerator, quotient, coefficient node, `Pow` per non-unit exponent, the balanced sum. Radical
    kernels are depth 2, from one leaf argument. `inverse_src` adds 1 for c != 1.
  - The depth of an input-node kernel comes from the pre-pass. Every input node, including the
    point's nodes and function arguments, is in that map.
- **DAG and point**: the pre-pass is an iterative memoized walk. `konst_of` of a DAG point is
  charged per occurrence. No recursion over a kernel expression was found in the engine. The ball
  code, `display_estimate`, `mentions_symbol` and the key writer are iterative, and
  `balanced_arc` and `bal_depth` recurse log2 n.
- **Long monomials, many kernels, radicals**: none reached a cap through a 100 KB text on any probe
  above. Budget refusals came first, within 0.45 s.

## Files

All under `EXP/redteam1-structure/`:

- `src/proto8/split.rs`, `apishape.rs`, `dagdrop.rs`, `litparse.rs` (the probes);
- `shapes.py` (the generators), `bsearch.py`, `bsapi.py`, `rrun.py`, `timed.py` (the scripts);
- `out/tow.gp` (the gp checks);
- `inputs/` (the inputs above, such as `tow1_400.txt`, `tow1_12000.txt`, `towx_18000.txt`).

`build.sh release|debug SRC NAME` rebuilds a binary into `bin/` in a few seconds. The binaries were
deleted after the runs.
