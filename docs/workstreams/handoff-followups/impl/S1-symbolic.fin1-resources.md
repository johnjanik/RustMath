# S1 finish, review fin1: resources lens, under the narrowed bar

Reviewed: main 28c4b69 (the S1 work in progress 4763f31) plus the uncommitted changes of the finish
step (`git diff`: parser.rs, series.rs, series_const.rs, series_engine.rs, simplify.rs, repl.rs, the
test files), with `owner_decision.md`, `owner_decision2.md`, `S1_impl.md` (all sections, the last
one "Finish under the narrowed bar"), `S1.review6-semantics.md`, `S1.review6-resources.md` and
`tests_report.md`. The narrowed bar of this run applies. No repository file was edited
(`git status --short` and the md5 of `git diff` are the same before and after; `status_before.txt`,
`output_md5.txt`). Scratch: `fu2/s1impl/fin1-resources/` (paths below are relative to it).

Verdict: **not approved**. Five blockers in this lens. The tests pass in both profiles.

- B1 (bar (b), new in this step). The parser's depth limit is triggered by the literal
  9223372036854775808 after a minus, which HEAD read as i64::MIN. HEAD parses such texts with more
  than 128 levels of nesting; the port now refuses them.
- B2 (bar (b), not new). HEAD's literal-rational rule refused `2/x`, `1/(x+1)`, `1/sin(x)`, `2/ 3`,
  `2/3.5` at once. The port reads them without the depth limit, so a deep part after them aborts the
  thread where HEAD returned a `ParseError`. This is the class of review 6 minor 2, which the finish
  step fixed for three constructs; this is a fourth.
- B3 (bar (c)). Re-parsing a printed result is not bounded: 95 s and 111 s for 32-byte inputs at
  their largest accepted orders (the gcd of every printed big rational).
- B4 (bar (c)). Re-parsing a printed result overflows the stack on the thread where the input itself
  parsed (the result nests 9 to 12 levels deeper than its input).
- B5 (bar (b), not new). `expand` of the parsed signed tower `x^-2^-2^...^-2`, a text HEAD reads,
  aborts from 959 levels (2 MB debug) and 15 410 levels (8 MB release). HEAD handles 1 439 and
  22 772.

## How it was measured

- The port as cargo builds it. Release: `cargo test --release -j 2 -p rustmath-symbolic --no-run`
  (1 min 14 s, `relbuild.err`). Debug: the same without `--release` (fresh). The rlibs are in
  `rlibs_rel.txt` and `rlibs_dbg.txt`.
- HEAD-before-S1 (8a0b310): its `rustmath-symbolic/src` (`git archive 8a0b310`, `headsrc/`) built
  with rustc as `rustmath_symbolic_head` against cargo's current rlibs of its dependencies
  (`buildhead.sh`; `rustmath-core`, `-integers`, `-rationals` and `-polynomials` are unchanged
  between 8a0b310 and 28c4b69). Release with `-C opt-level=3 -C codegen-units=16`, debug with
  `-C opt-level=0 -C debuginfo=2` and debug assertions (`headlib_rel/`, `headlib_dbg/`).
- Probes link both libraries (`buildpr.sh`):
  - `pr.rs` (`pr_rel`, `pr_dbg`): `pr LIB MODE KIND N STACK STEPS`, a text shape of N levels
    (`text`), a file, or a tree built through the API (`api`), on one thread of STACK bytes; then
    `simp`, `exp`, `diff`, `subs`, `disp`, `dbg`, `eq`, `reparse`; `keep` forgets the input and the
    outputs (to measure an operation of HEAD apart from its recursive `Drop`). It prints `DONE ok`
    when every step ran. `pr2.rs` adds the series entry points (HEAD's `taylor`; the port's
    `taylor_with_budget` at 1e8; `limit_via_series`; `limit_with_error`).
  - `reach.sh`: the largest N in [1, HI] with `DONE ok` (bisection; an abort counts as a failure;
    `ulimit -v 4000000`, 120 s per run). HI is the N at which the text, or the printed tree, is
    100 KB.
  - `fz.rs`, `fz2.rs`: parser fuzzers (below). `rp.rs`: the largest order `taylor_with_budget`
    (1e8) accepts at 0, then `Display`, `Debug`, `parse` of the printed text, `Drop`. `ep.rs`: the
    four entry points with `Display`, `Debug`, re-parse and `Drop`. `cs.rs`: `simplify` of a result
    and after `substitute(x, 2)` (consumers, for the docs only). `dump.rs`, `shape.rs`, `showp.rs`:
    helpers.
- Release runs use an 8 MiB thread, debug runs a 2 MiB thread, unless named otherwise. The machine
  has 4 cores shared with two other jobs; times are wall times.

## Tests

| run | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic --no-fail-fast -- --test-threads=2` (debug, `sym_dbg_tests.log`, 1 min 19 s) | unit 685 (4 ignored); `s1_depth` 11; `s1_fuzz_data` 7; `s1_payload` 2 (1); `s1_regressions` 26 (2); `s1_review2` 2; `s1_review3` 7 (7); `s1_review4` 4 (2); `s1_review4_memory` 1 (1); `s1_review5` 4 (2); `s1_review5_memory` 2 (2); `s1_review6` 4; `s1_singular_points` 9; doc 116 (11); 0 failed |
| the 13 release test binaries of this build (`run_rel.sh`, `rel_tests.log`) | 685 (4), 11, 7, 2 (1), 26 (2), 2, 7 (7), 4 (2), 1 (1), 4 (2), 2 (2), 4, 9; 0 failed |
| release `--ignored --test-threads=1` (same log) | `s1_payload` 1 (3.40 s); `s1_regressions` 2 (0.75 s); `s1_review3` 7 (2.06 s); `s1_review4` 2 (2.38 s); `s1_review4_memory` 1; `s1_review5` 2 (2.10 s); `s1_review5_memory` 2; `fold_work_calibration` passed (47.3 s) |
| `cargo test -j 2 -p rustmath-jupyter --lib -- --test-threads=2` (`jup_lib.log`) | 120 passed, 3 ignored, 0 failed |

## Blockers

### B1. A literal HEAD read as i64::MIN turns the depth limit on: HEAD-read texts with more than 128 levels are refused

Bar (b), the parser. New in this step (the round-5 port parses these texts:
`../rev6-resources/parse/pp_dbg parse/minint_par200.txt` gives `Ok`).

`integer_literal` (`parser.rs:255` to `:261`) calls `depth::relax()` for every digit string above
9223372036854775807. HEAD's `integer` rule (`recognize(pair(opt(char('-')), digit1))`, then
`parse::<i64>`) read the minus with the digits, so `-9223372036854775808` was a literal HEAD read
wherever its integer rule saw the minus: after `^` (`x^-9223372036854775808`, also
`x ^  -0009223372036854775808`), as the second sign of an operand (`--N`, `- -N`, `+-N`) and as the
denominator of its literal rational (`2/-N`). After one of these the port refuses any level past
128 parentheses or calls (or 1 024 carets) with "the expression nests too deeply for its syntax:
... an integer literal above 2^63 - 1 ...". HEAD parses the same texts.

The largest n that parses (`reach_minint.txt`; HEAD's tree forgotten, the port's dropped):

| text | 2 MB debug: HEAD / port | 8 MB release: HEAD / port |
|---|---|---|
| `x^-9223372036854775808*` + n parentheses | 215 / 127 | 4 124 / 127 |
| `--9223372036854775808*` + n parentheses | 215 / 127 | 4 123 / 127 |
| `x^-9223372036854775808*` + n nested `sin(` | 162 / 127 | 4 124 / 127 |
| `x^-9223372036854775808*` + a tower of n carets | 1 438 / 1 016 | 22 772 / 1 016 |

At n = 150 and 200 HEAD parses and simplifies these texts in under 0.01 s; the port refuses them
(`pr_dbg head|port text minint 200 2097152 simp`, and the variants `mminint`, `divminint`,
`minint0`, `minintsin`). The control `9223372036854775807*` + 150 parentheses parses in both.

Fuzzing (`fz.rs`, `fuzz/t1.txt`, `fuzz/t1.out`): 219 243 distinct random texts of up to 9 tokens;
HEAD parses 4 837. Each was put in three deep contexts (`(T)*` + 300 parentheses, 300 parentheses
around T, `(T)*` + 2 400 carets), on a 256 MiB thread. 262 contexts that HEAD parses are refused by
the port, and every one of them holds a minus (with optional white space) right before
`0*9223372036854775808`. No other HEAD-read construct turns the limit on.

Required: the limit is turned on only by texts HEAD refused. Here: 2^63 written right after the
minus that HEAD's integer rule took (one sign after `^`, the second sign of an operand, the
denominator right after `INT/`) is HEAD's syntax; above 2^63, or 2^63 elsewhere, is not. Regression
tests (both profiles): the four rows above at n = 200 (debug) and 1 000 (release) parse, with
`Integer(-9223372036854775808)` in the tree; the controls `-9223372036854775808*` and
`9223372036854775808*` + 300 parentheses stay a `ParseError` (HEAD refused them at the literal).

### B2. Texts HEAD refused through its literal-rational rule abort the port

Bar (b), the parser: HEAD returned a `ParseError` at once, the port aborts (the reading the finish
step adopted for review 6 minor 2, and the one this run's task names for the signed towers). Not
new: the round-5 port aborts on the same text (`../rev6-resources/parse/pp_dbg
parse/divx_par300.txt`).

HEAD's `number` tried `rational`: an integer, then `char('/')` with no white space, then `integer`
with `?`. So `2/x`, `1/(x+1)`, `1/sin(x)`, `2/-x`, `2/ 3`, `2/--3`, `2/+3` failed the whole atom,
and `2/3.5` left `.5` unread: all are a `ParseError` at HEAD before it reads any further. The port
reads them (S1 §2.12: `multiplicative` folds a quotient of two literals), and `operand_signs` and
`integer_literal` do not call `relax` for them, so the rest of the text is read without the limit.

| text | HEAD | port |
|---|---|---|
| `2/x*` + 300 parentheses (1.2 KB), 2 MB debug | `ParseError` | stack overflow, abort |
| `1/(x+1)*`, `2/ 3*`, `2/3.5*`, `2/--3*` + 300 parentheses, 2 MB debug | `ParseError` | abort |
| `1/sin(x)*` + 300 nested `sin(`, 2 MB debug | `ParseError` | abort |
| `2/x*` + a tower of 1 601 carets, 2 MB debug | `ParseError` | abort |
| `2/x*` + 5 000 parentheses (10 KB), 8 MB release | `ParseError` | abort |
| `2/x*` + a tower of 30 001 carets (60 KB), 8 MB release | `ParseError` | abort |

(`parse/div*.txt`.) Fuzzing (`fz2.rs`, `fuzz/t2.out`, `fuzz/t3.out`): 248 260 distinct random texts
in the context `(T)*` + 300 parentheses on a 256 MiB thread. HEAD refuses 234 535 of them. The port
refuses 5 136 by the depth limit and reads 3 058 to the end, so on a 2 MB debug thread past its
reach of 240 levels it aborts on each of these 3 058. All 2 313 classes of them hold a digit right
before `/`; the other constructs HEAD refused are covered.

The doc of `mod depth` (`parser.rs:182` to `:183`) says "A text deeper than HEAD's reach that uses no such
construct overflows as HEAD's parser does". For these texts HEAD did not overflow.

Required: the literal-rational construct turns the limit on too: a `/` right after an integer
literal (no white space) whose right side is not one optionally negative i64 literal that ends
there. The fuzz driver `fz2.rs` (no class left) is a check of the whole relax set. Regression tests
(both profiles): the six rows above give a `ParseError` or a value, never an abort; the control
`2 / x*` + 200 parentheses (HEAD's syntax) parses.

### B3. Re-parsing a printed result costs up to 111 s: the gcd of every printed big rational

Bar (c), the re-parsing of a result of `taylor_with_budget` with the REPL's budget.

The parser folds a quotient of two integer literals into a `Rational` (`product_step`,
`Rational::new`, which takes a gcd), and num-bigint's decimal parse is quadratic. A result prints
its coefficients as `p/q` at every term. The text cap counts the printing weight (one unit per
character up to 65 536 digits, d^(3/2) above), which bounds `Display`, not the parse. One quotient
of the two literals of row q4 parses in 0.164 s, their product in 0.018 s, one literal in 0.013 s
(`parse/q4_div.txt`, `q4_mul.txt`, `q4_lit.txt`).

`rp.rs` at 0, release, 8 MB thread, the largest accepted order (`rp/*.log`):

| input (bytes) | order | `taylor_with_budget` | `Display` | `parse(Display)` |
|---|---|---|---|---|
| q4 `(10^100-1)^1050/7^120000/(1-x)` (32) | 100 | Ok, 0.16 s | 1.35 s, 20.8 MB | **18.4 s** |
| q4 | 200 | Ok | 2.70 s, 41.5 MB | **35.5 s** |
| q4 | 514 (the largest) | Ok, 0.15 s | 7.09 s, 106 MB | **95.1 s**, VmHWM 168 MB |
| q5 `(10^150-1)^1050/7^180000/(1-x)` (32) | 282 (the largest) | Ok, 0.39 s | 7.68 s, 87 MB | **111.0 s**, VmHWM 139 MB |
| q3 `(10^30-1)^1050/7^37000/(1-x)` (27) | 2 192 (the largest) | Ok, 0.02 s | 4.78 s, 134 MB | **37.5 s**, VmHWM 211 MB |
| `exp(x + (10^300-1)^1050*7^130000)` (expbig2) | 122 | Ok | 7.35 s, 52 MB | 19.3 s |
| `exp(x + (10^300-1)^1050*7^370000)` (e4) | 192 | Ok | 7.77 s, 61 MB | 16.4 s |
| `(10^300-1)^1050/7^370000/(1-x)` (q1) | 192 | Ok | 7.71 s, 61 MB | 16.7 s |
| `sqrt(N)/(1-x)` (sqrtN, round 3) | 1 081 | Ok | 7.32 s, 108 MB | 9.6 s |
| `exp(x)` | 8 996 | Ok, 0.13 s | 3.87 s, 134 MB | 2.9 s, VmHWM 383 MB |

Rationals with big parts computed by the engine are bounded by the budget (`exp(A/B*x)` and
`1/(1-A/B*x)` with 1 000-digit A, B: orders 31 and 36, re-parse 0.18 s and 0.27 s), because the
engine pays the gcd once per coefficient. A kernel constant is computed once and printed at every
term, so the parse pays it at every term. Integer-only texts stay just under the bar (19.3 s).

Required: `parse(&format!("{r}"))` of every accepted result within 20 s and 512 MB (release, 8 MB
thread), or the result is refused. A way that fits the caps: a parse weight per occurrence (for a
printed rational p/q, the gcd, about words(p) x words(q) units, plus the quadratic parse of each
literal; for an integer, its parse), counted at every occurrence like the text cap and checked
against a cap calibrated in release. Regression tests (release, `#[ignore]`, as the other budget
tests): q4 at orders 100 and 514 and q5 at 282, through `taylor_with_budget` with 1e8: refused, or
the re-parse within 20 s.

### B4. Re-parsing a printed result overflows the stack where the input parsed

Bar (c), the re-parsing of a result: stack overflow of a 2 MB debug thread and of an 8 MB release
thread.

The rendering adds levels around a kernel's source: the balanced sum of the terms and the product
with x^k. So a result nests 9 to 12 parentheses deeper than its input, and an input the parser
reads near its reach gives a result whose re-parse overflows the same thread (`ep/sinn_*.txt`,
`ep.rs`, steps `tw,disp,reparse`):

| input (bytes) | thread | order | `taylor_with_budget` | `Display` (paren depth) | re-parse |
|---|---|---|---|---|---|
| `sin^150(1)/(1-x)` (757) | 2 MB debug | 1 000 | Ok | 771 KB | Ok, 0.50 s |
| `sin^185(1)/(1-x)` (932) | 2 MB debug | 1 000 | Ok | 946 KB | Ok, 0.61 s |
| `sin^195(1)/(1-x)` (982) | 2 MB debug | 1 000 | Ok, 0.007 s | 996 KB (207; input 195) | **stack overflow, abort** |
| `sin^200(1)`, `sin^204(1)` over `(1-x)` | 2 MB debug | 1 000 | Ok | Ok | **abort** |
| `sin^4300(1)/(1-x)` (21.5 KB) | 8 MB release | 100, 400 | Ok | 2.2 MB, 8.6 MB | Ok, 0.14 s, 0.54 s |
| `sin^4320(1)/(1-x)` (21.6 KB) | 8 MB release | 100 | Ok, 0.006 s | 2.2 MB (4 329; input 4 320) | **abort** |
| `sin^4320(1)/(1-x)` | 8 MB release | 400 | Ok | 8.7 MB | **abort** |

`sin^n(1)` is n nested `sin(` around 1. The abort is in `parse`: `dump.rs` wrote the printed text
(`ep/sinn_195_o1000.out`, `ep/sinn_4320_o100.out`) and `pr_dbg`/`pr_rel port file` abort on it.

Required: re-parsing a result never overflows a thread on which its input parsed. Ways: the engine
refuses a result whose printed nesting, in the parser's frame units, passes the input's; or the
parser turns a depth past a safe bound into a `ParseError` without losing HEAD's reach (B1 and B2
show how delicate the conditional limit is); or a parser with an explicit stack. Regression tests:
`sin^195(1)/(1-x)` at order 1 000 on a 2 MB debug thread and `sin^4320(1)/(1-x)` at order 100 on an
8 MB release thread: the re-parse returns `Ok` or a `ParseError`, or the result is refused.

### B5. `expand` of a parsed signed tower aborts where HEAD's did not

Bar (b), `expand` applied to an input (its parse). Not new: the round-5 port aborts too
(`../rev6-resources/parse/pp_dbg parse/nlit1000.txt 2097152 exp`; `pp_head` gives 7 001 characters).

The port reads `x^-2^-2^-2` as `Pow(x, Neg(Pow(2, Neg(Pow(2, -2)))))`, two nodes per level; HEAD
read `Pow(x, Pow(-2, Pow(-2, -2)))`, one node per level (`showp.rs`; the values differ too, a
matter for the semantics lens). `expand` uses one frame per node, and its reach on equal trees is
HEAD's (table "What held"), so on the port's tree it stops at half the levels. The largest number of
levels for which parse, `expand` and `Drop` complete (`reach_nlit_exp.txt`):

| text | 2 MB debug: HEAD / port | 8 MB release: HEAD / port |
|---|---|---|
| `x^-2^-2^...^-2` | 1 439 (its parse reach) / **958** | 22 772 (its parse reach) / **15 409** |
| `-x^-2^-2^...` | 1 439 / **958** | 22 772 / **15 408** |
| `x^-2.5^-2.5^...` | 1 439 / **958** | 22 770 / **15 408** |
| `x ^ -2 ^ -2 ...` (`textops.log`) | ok at 1 439 / **abort** | ok at 19 996 / **abort** |

HEAD's numbers include HEAD's recursive `Drop` of the input and the output. Texts of 2.9 KB
(debug) and 46 KB (release) abort. `substitute` (not an S1 function) also loses reach on this tree
(1 303 levels against HEAD's 1 439 in debug); `simplify`, `Display`, `==` and `differentiate` keep
it.

Required: `expand` of the parse of a text HEAD read does not abort where HEAD's `expand` of HEAD's
parse succeeded (an explicit-stack `expand`, or a negation and power arm that adds no frame per
level). Regression tests: `x` followed by 1 439 copies of `^-2` on a 2 MB debug thread and 22 772
on an 8 MB release thread, and the three variants above: parse and `expand` complete.

## Minor issues

1. **The docs claim the re-parsing is bounded.** `MAX_RESULT_TREE` (`series_const.rs:639`: "The
   walk of `Display`, `Drop` and the re-parsing of the printed text are linear in it"),
   `MAX_RESULT_TEXT` (`:676` to `:681`: "the dropping and the re-parsing of their results ...
   `Drop` and the re-parsing of the text are linear in the tree. The reviews of rounds 4 to 6 found
   these within 20 s and 512 MB ... on every input they tried"), `taylor_with_budget`
   (`series.rs:727` to `:728`: "The structural caps bound the printing, the dropping and the re-parsing of
   the result"), the comment in `series_engine.rs:2346` and the REPL's `TAYLOR_WORK`
   (`repl.rs:11217` to `:11218`: "What is bounded ...: the `taylor` call itself, and the printing, dropping and
   re-parsing of its result"). B3 and B4 contradict them, and no review measured the re-parsing of
   a result: review 6's parse stress parsed inputs (`grep` of the six resources reviews). Fix the
   text with B3 and B4.
2. **The docs misstate `simplify` after `substitute`.** "`simplify` of a result, and of its `subs`,
   at most 4.1 s at the caps" (`repl.rs:11224` to `:11225`), "`simplify` after `substitute` at most 4.1 s"
   (`MAX_RESULT_TEXT`, `series_const.rs:700`), "folds for at most 4.1 s in the measurements at the edges of the
   caps" (`simplify.rs:422`), and `StructureLimit::FoldWork` "the folds of a copy of the tree
   without its sharing (as `substitute` makes it)" (`series.rs:149`). The fold cap does not count
   the folds that the substituted value makes (the round-5 text said "plus the new folds of the
   substituted value"; this step dropped it). q4 `(10^100-1)^1050/7^120000/(1-x)` (32 bytes):
   `simplify(substitute(R, x, 2))` takes 6.8 s at order 10, 34.1 s at 50, 66.0 s at 100 and 348.0 s
   at 514, its largest accepted order (VmHWM 49 MB; `cs_q4.log`). This is a consumer of a result,
   out of the bar; the docs must give the measured scaling, about the number of terms times the
   cost of the substituted coefficient's folds, not "at most 4.1 s".
3. **Two parser claims are false (with B1 and B2).** `S1_impl.md`, finish item 5: "What it
   refuses: only texts HEAD refuses too" (B1). The `mod depth` doc: "A text deeper than HEAD's
   reach that uses no such construct overflows as HEAD's parser does" (B2). The refusal message
   names "an integer literal above 2^63 - 1" for `x^-9223372036854775808` (B1).
4. Note, not counted: the port refuses float literals that underflow (`1.5e-325`, `1.5e-352`), which
   HEAD read as 0.0. The fuzzer meets 337 such plain texts. This is the rule of fix-up round 3
   (review 3, semantics m2), a decision of the semantics lens.

## What held (no finding)

- **The parser on HEAD's syntax.** Reach of 21 shapes, HEAD (tree forgotten) against the port (tree
  dropped), up to 100 KB of text (`reach_all.txt`). Debug 2 MB: carets 1 438 / 1 472, `x^-2`
  towers 1 439 / 1 472, parentheses 215 / 240, `sin(` 162 / 204, `2^(` 187 / 206, `2*(`, `-(`,
  `- (`, `--5*(`, `- -5*(`, `+-5*(`, `x^-2*(`, `x^-2.5*(`, `(...)^-2`, `(...)^x` 215 / 240,
  `f(x,` 162 / 204. Release 8 MB: carets 22 772 / 29 098, parentheses 4 123 / 4 328, `sin(` 4 123 /
  4 328, `2^(` 3 491 / 3 767, the rest 4 123 or 4 124 / 4 328; with white space (`x ^ x`,
  `x ^ -2`, `( x )`, `x^-2.5`) both reach 100 KB in release. The port's reach is at least HEAD's on
  every shape, so the frames hold.
- **`simplify` on inputs** (`reach_api_simp.log`, `simp_times.txt`): on 19 API-built shapes (chains
  of `-`, `sin`, `+`, `-`, `*`, `/` and `^` on either side, nested negated sums, Horner forms,
  powers of sums) the port reaches 100 KB of printed text on both threads; HEAD stops at 1 624 to
  3 258 levels in debug. In release the port is as fast or faster (at most 0.004 s against HEAD's
  0.003 to 0.011 s) and uses less memory. On the 21 parsed shapes at HEAD's parse reach `simplify`
  and `Display` complete at the port (`textops.log`).
- **`expand` on API inputs** (`reach_api_exp.log`, `reach_api_exp2.log`): the port's reach equals
  HEAD's on all 19 shapes (1 915 or 1 916 frames in debug, 958 for two-node levels; 30 817 for the
  negation chain in release, 100 KB for the others); times equal (Horner form of 3 000 levels:
  1.53 s and 969 MB at HEAD, 1.47 s and 969 MB at the port; nested negated sums 0.64 s at both; the
  negated difference chain 0.68 s at HEAD, 0.001 s at the port). The only `expand` regression is
  B5, from the parsed tree.
- **The series entry points on deep inputs** (`reach_ser.log`, `reach_ser2.log`; API-built chains,
  at 0; HEAD's `taylor` against the port's `taylor_with_budget` at 1e8, order 3; `limit_via_series`
  order 5; `limit_with_error` order 3). The port reaches 100 KB of printed text on every shape and
  both threads. HEAD's largest depths, debug 2 MB / release 8 MB: `taylor` on `sin` chains 92 / 109,
  negation and sum chains 188 / 12 774, products 62 / 91, towers 37 / 52, quotients 31 / 51, `exp`
  chains 91 / 91; `limit_via_series` on `sin` chains 20 / 20, sums 188 / 12 774, products 19 / 19,
  towers 9 / 9; `limit_with_error` as `taylor`. No regression of the entry points.
- **The new engine paths of this step** (m1 `truncated_integer_exponent`, m3 `gaussian_inverse`):
  23 inputs (`ep/new_paths.txt`, 41 KB: truncated integer exponents alone, summed 300 times, nested,
  under a 200-level tower, with an undecided integrality over 1 000 kernels; Gaussian constants of
  2 to 16 terms with square roots, 300-digit parts, a near-zero norm against a 2 000-digit
  rational, 200 summed inverses) through `taylor_with_budget`, `series_expansion` with `to_expr`,
  `limit_via_series` and `limit_with_error`, then `Display`, `Debug`, re-parse and `Drop`, at orders
  8, 200 and 20 000 (`ep/new_paths_rel*.log`): at most 2.7 s per input for all steps in release,
  VmHWM at most 122 MB; on a 2 MB debug thread at order 8 at most 15.5 s per input, no overflow
  (`ep/new_paths_dbg_8.log`). `gaussian_inverse` builds A^2 + B^2 and the product through `mul` and
  `q_mul`, which charge before they work; the budget overshoot is at most 27 % (one a-priori charge).
- **`Display`, `Debug` and `Drop` of results.** `Display` at the text cap: 7.1 to 7.8 s (the docs
  say about 8 s), `Debug` the same (4.0 s for `exp(x)` at 8 996). `Drop` of a result and of its
  re-parsed tree at most 0.013 s. The re-parse memory stays under 211 MB.
- **The consumer numbers in the docs for `expand` and `differentiate`** match review 6 (110-byte
  input, 4.0 s and 2.5 GB, then 21.8 s and 5.6 GB; the 4.9 KB tower, 1.1 s and 1.3 GB, then 9.1 s
  and 2.9 GB; the REPL's 5.7 s and 2.5 GB, 11.6 s and 2.1 GB), and the docs say these are not
  bounded and point to S2.
- `git status --short` lists the same entries before and after, and the md5 of `git diff` is
  unchanged; `rustmath-misc/output.txt` is unchanged (md5) and no `.tmp_dir_*` exists.
  `target/release` was deleted at the end.

## Files

`fu2/s1impl/fin1-resources/`:

- scripts and probes: `buildhead.sh`, `buildpr.sh`, `reach.sh`, `reach_all.sh`, `reach_api.sh`,
  `reach_api2.sh`, `reach_ser.sh`, `textops.sh`, `run_rel.sh`, `pr.rs`, `pr2.rs`, `fz.rs`, `fz2.rs`,
  `rp.rs`, `ep.rs`, `cs.rs`, `dump.rs`, `shape.rs`, `showp.rs`;
- HEAD's source and libraries: `headsrc/`, `headlib_rel/`, `headlib_dbg/`;
- inputs: `parse/*.txt` (B1, B2, B5 texts), `fuzz/t1.txt` to `t3.txt`, `rp/*.txt` (B3), `ep/*.txt`
  (B4 and the new paths);
- logs: `relbuild.err`, `dbgbuild.err`, `sym_dbg_tests.log`, `rel_tests.log`, `jup_lib.log`,
  `reach_all.txt`, `reach_minint.txt`, `reach_api_exp.log`, `reach_api_exp2.log`,
  `reach_api_simp.log`, `reach_nlit_exp.txt`, `reach_ser.log`, `reach_ser2.log`, `textops.log`, `simp_times.txt`,
  `fuzz/t1.out` to `t3.out`, `rp/*.log`, `ep/*.log`, `cs_q4.log`, `status_before.txt`,
  `status_after.txt`, `output_md5.txt`;
- binaries (statically linked): `pr_rel`, `pr_dbg`, `pr2_rel`, `pr2_dbg`, `fz_rel`, `fz2_rel`,
  `rp_rel`, `ep_rel`, `ep_dbg`, `cs_rel`, `dump_rel`, `dump_dbg`, `shape_rel`, `showp_rel`.
