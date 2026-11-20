# S1 finish, review fin2: resources lens, under the narrowed bar

Reviewed: main 28c4b69 (the S1 work in progress 4763f31) plus the uncommitted changes of the finish
and of finish fix-up round 1 (`git diff`: parser.rs, expand.rs, series.rs, series_const.rs,
series_engine.rs, simplify.rs, repl.rs, the test files), with `owner_decision.md`,
`owner_decision2.md`, `S1_impl.md` (to "Finish fix-up round 1"), `S1.review6-*.md`,
`tests_report.md` and the round-1 reviews `S1.fin1-resources.md` and `S1.fin1-semantics.md`. The
narrowed bar of this run applies. No repository file was edited: `git status --short` and the md5 of
`git diff` are the same before and after (`status_before.txt`, `status_after.txt`,
`diff_md5_*.txt`). Scratch: `fu2/s1impl/fin2-resources/` (paths below are relative to it).

Verdict: **approved**. No blocker in this lens. The five resources blockers of round 1 are fixed,
and I found no new one. The tests pass in both profiles. Four minor issues follow, two of them doc
claims that are false.

## How it was measured

- The port as cargo builds it. Release: `cargo test --release -j 2 -p rustmath-symbolic --no-run`
  (1 min 7 s, `relbuild.err`, `relbuild.json`). Debug: the same without `--release`, which cargo
  reported fresh for the current sources (`dbgbuild.json`). The rlibs are in `rlibs_rel.txt` and
  `rlibs_dbg.txt`.
- HEAD-before-S1 (8a0b310): the round-1 libraries `../fin1-resources/headlib_rel/` and
  `headlib_dbg/` (HEAD's `src` built with rustc against cargo's rlibs of its dependencies; the
  dependency rlibs have the same names in this build, and the probes link against both).
- Probes (`buildpr.sh rel|dbg SRC OUT`): `pr.rs` and `reach.sh` (round 1's text, file and API
  shapes, with `simp`, `exp`, `disp`, `eq`, `reparse`); `fz3.rs` (a new differential parser
  fuzzer); `ep2.rs` (S1's four entry points on one input, then `Display`, `Debug`, the re-parse of
  each result with its tree dropped, and `Drop`, each with time and VmHWM); `maxord.sh` (the largest
  order `taylor_with_budget` accepts); `cons.rs` (consumers of a result, for the docs only);
  `ab.rs` with `old_expand.rs` (4763f31's recursive `expand` against the new one on the same tree);
  `headtay.rs` (HEAD's `taylor` against the port); `show.rs`.
- Release runs use an 8 MiB thread, debug runs a 2 MiB thread. The machine has 4 cores shared with
  the WP-P build and the semantics reviewer; the load average was 1.4 to 5.7. Times are wall times.

## Tests

| run | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic --no-fail-fast -- --test-threads=2` (debug, `sym_dbg_tests.log`, 1 min 24 s) | unit 685 (5 ignored); `s1_depth` 14; `s1_finish1` 8 (1); `s1_fuzz_data` 7; `s1_payload` 2 (1); `s1_regressions` 26 (2); `s1_review2` 2; `s1_review3` 7 (7); `s1_review4` 4 (2); `s1_review4_memory` 1 (1); `s1_review5` 4 (2); `s1_review5_memory` 2 (2); `s1_review6` 4; `s1_singular_points` 9; doc 116 (11); 891 passed, 0 failed |
| the 14 release test binaries of this build (`run_rel.sh`, `rel_tests.log`) | 685, 14, 8, 7, 2, 26, 2, 7, 4, 1, 4, 2, 4, 9; 775 passed, 0 failed |
| release `--ignored --test-threads=1` (same log) | `s1_finish1` 1 (`b3f1`, 0.72 s); `s1_payload` 1; `s1_regressions` 2; `s1_review3` 7; `s1_review4` 2; `s1_review4_memory` 1; `s1_review5` 2; `s1_review5_memory` 2; `fold_work_calibration` and `parse_work_calibration` passed. The unit binary's ignored `specialfunctions::airy::tests::test_airy_numerical_helper` fails (minor 4: not S1) |

The REPL diff changes doc lines only (`git diff 4763f31 -- rustmath-jupyter/src/repl.rs` has no
code line), so I did not rerun the jupyter tests.

## The blockers of round 1

| round 1 | status | measured on this build |
|---|---|---|
| B1: the literal i64::MIN turned the depth limit on | fixed | `x^-N*`, `--N*`, `x/2/-N*`, `x^-0..0N*` before 200 parentheses, `x^-N*` before 200 calls and before 1 400 carets parse on a 2 MB debug thread (`b1/b1_dbg.log`). The fuzzer finds no HEAD-read text refused by the depth limit (below) |
| B2: texts HEAD refused through its literal rational aborted | fixed | the ten texts of round 1 (`2/x*` before 300 and 5 000 parentheses and 1 600 and 30 000 carets, `1/(x+1)*`, `2/ 3*`, `2/3.5*`, `2/--3*`, `1/sin(x)*` before 300 calls, 300 parentheses around `2/x`) are a `ParseError` at once on 2 MB debug and 8 MB release (`b2.log`) |
| B3: re-parsing a printed result took up to 111 s | fixed | `MAX_RESULT_PARSE_WORK` refuses q4 from order 33. At the largest accepted orders: q4 (32) re-parse 5.89 s; q3 `(10^30-1)^1050/7^37000/(1-x)` (379, exactly at the cap) 5.94, 6.58 and 7.57 s in three runs; `exp(x + (10^300-1)^1050*7^370000)` (63) 6.07 s; `exp(x)` (8 996) 3.45 and 3.94 s (`rp/rp_edge.log`, `rp/maxord.log`) |
| B4: re-parsing a result overflowed the stack | fixed | `sin^195(1)/(1-x)` at order 1 000 on 2 MB debug: re-parse `Ok` in 0.67 s; `sin^4320(1)/(1-x)` at 400 on 8 MB release: 0.42 s; `sin^20000(1)/(1-x)` (100 KB) at 100 in release (10 MB of text): 0.47 s, VmHWM 211 MB; at 30 on 2 MB debug: 1.67 s (`b4/`) |
| B5: `expand` of a parsed signed tower aborted | fixed | `x^-2^...`, `-x^-2^...`, `x^-2.5^...`, `x ^ -2 ...` with 1 439 levels (2 MB debug) and 22 772, 22 772, 22 770, 19 996 levels (8 MB release): parse, `expand`, `simplify`, `Display`, `==` and `Drop` complete, as at HEAD (`b5.log`) |

## What held (no finding)

- **The parser against HEAD.** `fz3.rs` (`fuzz/`): 818 849 distinct texts, each plain and in three
  deep contexts (`(T)*` + 300 parentheses, 300 parentheses around T, `(T)*` + 2 400 carets), on a
  512 MiB thread: 3.27 million parses per library. Two new corpora from `gen_fuzz.py` (seeds 7 and
  11, 300 000 texts each: literals at the i64 edges with leading zeros, literal rationals, floats,
  sign runs, `^-`, calls, tab and CR white space; `u1.out`, `u2.out`) and round 1's `t1.txt`
  (`t1.out`). The port never panics. Its slowest parse is 0.0099 s. HEAD parses a text and the port
  refuses it in two classes only, the same in every context: an underflowing float literal (1 707
  plain texts; round 1's minor 4, the rule of fix-up round 3) and a literal zero divisor with white
  space or parentheses (17 plain texts, minor 2). No HEAD-read text meets the depth limit.
- **Pathological texts** (`pp/pp.log`): 18 texts of 100 KB (100 000 `(`, sign runs with and
  without spaces, `1/1/.../1`, a chain of 20-digit quotients, a `^-3` tower, a call with 50 000
  arguments, 25 000 nested calls with two arguments, a 100 000-digit literal and float, unclosed
  carets, a deep text with an error at its end): at most 0.017 s in release and 0.145 s on a 2 MB
  debug thread, no abort.
- **simplify and expand on HEAD-syntax inputs.** Round 1's 21 text shapes at HEAD's parse reach
  (`textops.log`): parse, `simplify`, `expand`, `Display` and `==` complete on both threads for all
  42 runs (round 1: `expand` aborted on 8). `expand` on round 1's API shapes (`reach_api_exp.log`):
  the port reaches 100 KB of printed text on all of them in both profiles (HEAD: 1 915 or 1 916 in
  debug); the nested negated sum reaches 4 341 levels in debug (HEAD 958). Times on inputs are
  HEAD's within the noise: Horner form of 3 000 levels 1.48 and 1.65 s at HEAD, 2.10 and 1.57 s at
  the port, 969 MB at both; nested negated sums 0.82 s and 0.64 to 0.72 s; products of 16 and 18
  sums 0.22 to 0.39 s at both. The new explicit-stack `expand` against 4763f31's recursive one on
  the same result tree (`ab.rs`, e1 below at order 1 000): 2.7 to 5.9 s each across four runs, the
  same 2.5 GB and the same text.
- **S1's entry points on 100 KB text inputs** (`ep/`, `ep_rel_o3.log`, `ep_dbg_o3.log`,
  `ep_dbg_o3b.log`). The parser now reads deep texts, so I ran 40 inputs of about 100 KB that only
  the API could build before: nests of `sin`, `exp`, `atan`, `sinh`, `tan`, `erf`, `atanh`,
  `sqrt(1+`, `log(1+`, `abs(1+`, `acosh(2+`, `gamma(1+`, `sign(2+`, `2^(`, `x^(`, `-(`, `(x+`,
  `(x*`, `1/(1+`, `f(`, `sin(exp(`, towers, flat sums, products and quotients of 50 000 terms,
  constant kernels of depth 20 000 and flat kernel sums of 17 000 terms over `(1-x)`, at 0, 1 or
  infinity, order 3. Each runs `taylor_with_budget` (1e8), `series_expansion` with `to_expr`,
  `limit_via_series` and `limit_with_error`, then `Display`, `Debug`, the re-parse and `Drop` of each
  result, and the input's `Drop`. All 40 complete on an 8 MB release thread (at most 2.35 s per
  input for everything) and on a 2 MB debug thread (at most 12.1 s per input). The outcomes are
  values, `BudgetExceeded`, `Unsupported` or `UndecidedZero`.
- **The re-parse at the caps.** Memory (`mem/mem_edge.log`): `sin(A1+...+Am)/(1-x)` with m
  integers of 30 to 250 digits at their largest accepted orders (5 177 to 10 229; up to 2 million
  nodes and 69 MB of text): re-parse at most 0.90 s, VmHWM at most 311 MB. `exp(x)` at 8 996:
  `Display` 5.0 and 5.5 s, `Debug` 4.8 and 5.4 s, VmHWM 383 MB with both strings held. Big powers
  stay kernels (`3^661000` prints as `(3 ^ 661000)`), so they add no parse work (`rp/`).
- **The limit path.** A limit is a rational whose size the budget bounds: `(10^76+1)^4096` gives a
  311 297-digit integer, printed in 0.07 s and re-parsed in 0.12 s; the larger quotients I tried
  exceed `LIMIT_SERIES_WORK` (`lim/`).
- **The new exponent check** (`variable_exponent`): `(x-pi)^((1+x)/(1+x))`,
  `(x-pi)^(exp(x)*exp(-x))`, `(x-pi)^(sin(x)^2+cos(x)^2)` at pi up to order 5 000 and an undecided
  coefficient over 1 000 terms at 0: at most 0.41 s, the budget stops the rest (`ve/`).
- **HEAD does not handle what the parse cap refuses.** HEAD's `taylor` of q4 in HEAD's syntax
  (`(10^100-1)^1050 / 7^120000 / (1-x)`): 26.4 s at order 5, and an allocation failure under a
  6 GB limit at 33 and 40 (`rp/`, `headtay.rs`). The port accepts q4 up to order 32.
- **The docs of the consumers.** No added doc line claims that a consumer of a result is bounded
  (a grep of the added lines for "bounded", "bounds" and "at most" near the consumers). The measured
  numbers hold within the noise: `simplify(substitute(R, x, 2))` for q4 is 6.84 s at order 10 (the
  docs: 6.8 s) and 21.9 s at 30 (the docs: 20.0 s); `expand` of the e1 result
  (`sin((y0+z0)*...*(y11+z11))/(1-x)`, 111 bytes) at 1 000 is 4.4 to 8.6 s and 2.5 GB at load 4 to
  5 (the docs: 4.0 s and 2.5 GB), at 100 0.25 s and `simplify` of that 1.0 s, 449 MB (`cons/`,
  `rp/q4_subssimp.log`).

## Minor issues

1. **A printed result does not always re-parse, against two claims.** The depth limit applies to
   the re-parse of a result too. A result prints relaxing constructs of its own: an integer above
   i64::MAX (a factorial), a `Rational` with such a denominator, or `^ -(`, which is how `Display`
   writes the negated exponent the S1 parser builds for HEAD's `x^-2^-2`. After one of them, any
   level past 1 024 units counted from the start of the text is refused. Measured (`rt/`,
   `mem/mem_edge.log`, `ep_rel_o3.log`):
   - `x` followed by 110 copies of `^-2` (332 bytes, HEAD syntax) at 1, order 3: the result
     re-parses as "the expression nests too deeply for its syntax" (100 copies: `Ok`);
   - `exp(x)*sin(sin(...(1)...))` with 130 calls (659 bytes, HEAD syntax): `Ok` at order 20 and a
     `ParseError` at order 25, once 1/21! is printed;
   - `sin(A1+...+A200)/(1-x)` with 30-digit Ai at order 5 177;
   - `x` and 33 333 copies of `^-2` (100 KB) at 1: both results.

   The refusal is immediate, so it is no budget hole under bar (c), and round 1 accepted a
   `ParseError` for B4. But `parser.rs:192` to `:193` says "a printed result re-parses on any
   thread", and `S1_impl.md:1915` (fix-up round 1, deviation 2) says "every result re-parses". The
   message names constructs the user did not write. Required: correct both texts, state the
   condition (a relaxing construct in the printed text, then more than 128 levels of parentheses
   and calls or 1 024 of `^`) with the examples above, and add it to the tracker row of the parser
   and the HANDOFF (a follow-up may exempt the parser's own printed forms).
2. **`5 / 0` is refused where HEAD read it, with a message the design does not give.** HEAD's
   `rational` rule refused `1/0`. With white space or parentheses (`5 / 0`, `(1)/(0)`, `10\r/0`,
   `07---09223372036854775808  /0`), HEAD read `Div(n, 0)`. The port folds every quotient of two
   signed literals and refuses a zero divisor: 17 plain fuzz texts. This is the frozen design §2.12
   ("r = 0 is a parse error (\"division by zero in a literal\"), as today"; the "as today" holds for
   `1/0` only). The value is undefined, and MAGMA and Sage refuse `5/0` too, so I do not count it
   against bar (b). The message is nom's: "Failed to parse expression: Parsing Failure: Error {
   input: \"\", code: Verify }" (`parser.rs:759` to `:761`; `parse` maps only `ErrorKind::Float`
   at `:149`). Required: map that failure to the design's message, and state the change against
   HEAD in the tracker row of the parser.
3. **Two doc numbers.** `MAX_RESULT_PARSE_WORK` and the REPL's `TAYLOR_WORK` say the parse cap is
   "about 6 s". At the cap I measured 5.9 to 7.6 s (q3 at order 379 is 380 x 2.258e7 = 8.58e9 units,
   0.69 to 0.88 ns per unit at load 5): say "about 6 to 8 s". `simplify.rs` (the doc of `simplify`)
   and the REPL's `TAYLOR_WORK` cite "66.0 s at order 100" for `simplify` after `substitute` of q4,
   an order the engine now refuses; the largest accepted order, 32, takes 23.3 s. The doc of
   `MAX_RESULT_TEXT` already says this; the other two should too.
4. Notes, not counted. (a) The release `--ignored` run of the unit binary fails
   `specialfunctions::airy::tests::test_airy_numerical_helper`. It is HEAD's ignored demonstration
   test of `airy_ai_numerical`, which S1 does not change (S1 changes only the `matches!` borrows of
   that file's other tests). The implementer runs the S1 ignored tests by name. (b) The underflow
   class (1 707 plain fuzz texts) is the semantics lens's rule of fix-up round 3, as round 1 noted.

## Housekeeping

- `git status --short` and the md5 of `git diff` are unchanged. `rustmath-misc/output.txt` is
  unchanged (md5 746308829575e17c3331bbcb00c0898b). No `.tmp_dir_*` exists.
- `target/release` was deleted at the end, after I checked that no process used it (the semantics
  reviewer had finished; the running cargo build is the WP-P one in `/home/user/RustMath-wpp`).
  Free disk at the end: 5.5 GB.

## Files

`fu2/s1impl/fin2-resources/`:

- scripts and probes: `buildpr.sh`, `reach.sh`, `reach_api.sh`, `textops.sh`, `run_rel.sh`,
  `run_ep.sh`, `maxord.sh`, `gen_fuzz.py`, `gen_ep.py`, `pr.rs`, `fz3.rs`, `ep2.rs`, `cons.rs`,
  `ab.rs`, `old_expand.rs`, `headtay.rs`, `show.rs` (and round 1's `pr2.rs`, `fz.rs`, `fz2.rs`,
  `rp.rs`, `ep.rs`, `dump.rs`, `showp.rs`, copied);
- inputs: `fuzz/u1.txt`, `fuzz/u2.txt`, `ep/*.txt` and `*.pt`, `pp/*.txt`, `mem/*.txt`, `rp/*.txt`,
  `rt/*.txt`, `b4/*.txt`, `cons/*.txt`, `lim/*.txt`, `ve/*.txt`, `z/*.txt`;
- logs: `relbuild.err`, `sym_dbg_tests.log`, `rel_tests.log`, `b1/b1_dbg.log`, `b2.log`, `b5.log`,
  `fuzz/u1.out`, `fuzz/u2.out`, `fuzz/t1.out`, `pp/pp.log`, `textops.log`, `reach_api_exp.log`,
  `ep_rel_o3.log`, `ep_dbg_o3.log`, `ep_dbg_o3b.log`, `mem/mem_edge.log`, `rp/maxord.log`,
  `rp/rp_edge.log`, `rp/q4_subssimp.log`, `cons/e1.log`, `status_before.txt`, `status_after.txt`,
  `diff_md5_before.txt`, `diff_md5_after.txt`, `output_md5.txt`;
- binaries (statically linked): `pr_rel`, `pr_dbg`, `fz3_rel`, `ep2_rel`, `ep2_dbg`, `cons_rel`,
  `ab_rel`, `headtay_rel`, `show_rel`.
