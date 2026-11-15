# J1a-repl-parser-and-small: REPL parser, plot options, chart, limits, sqrt_mod

Unit J1a of the HANDOFF follow-ups. Main crate: `rustmath-jupyter` (one file, `src/repl.rs`). Two small
library changes outside it, each justified below: `rustmath-rationals` (`Rational::to_f64`, the root cause of
S11) and `rustmath-finitefields` (one public count function for sqrt_mod). Branch
`claude/gracious-gates-e6b6l7`, HEAD `c79a582` (code identical to `9c24389`). All line numbers are HEAD's.
Units I3, I1, I2, S1, MF1 and C1 edit `repl.rs` before J1a is implemented, so the implementer must locate
each site by its function or match arm, not by the number.

## Revision 2

Changes after review 2 (`J1a-repl-parser-and-small.review2.md`). I agree with both blockers and all seven minor
issues; nothing is disputed. For B1 the revision goes one step beyond the required change (exact subexpressions
stay exact inside a text with a constant), for the reason given in its row. The revision-2 prototype is `$S/r2/`
(table below): revision 1's `repl.rs` plus `$S/r2/patch_r2.py`; 74 of 74 tests pass (the 68 of revision 1 and
6 new ones, 8.2 s in debug). The `primes` change of B2 needs I1's functions, so it is specified and its model is
measured, but it is not in the prototype (I1 is not implemented yet).

| Issue | What changed | Sections |
|---|---|---|
| B1 (blocker): `eval_to_f64` computed in doubles first, so texts that HEAD evaluated exactly lost digits (`10^20 + 1 - 10^20` read as 0, a `find_root` root outside the interval), and an inf or an underflow could be absorbed into a finite bound (`3 * 10^300 / 10^400 * 10^100` read as 0) | `eval_to_f64` evaluates the text exactly first (`eval_expr`): an Integer or a Rational is rounded once (`exact_to_f64`), a Float is taken as it is. Only a text whose value is not an exact number (it uses pi, e, tau, a decimal literal or a Float variable) goes to `float_arith`. Beyond the required change, `float_arith` now follows MAGMA's coercion rule (ch. 25 §25.1.2, `Chapter_25_Real_and_Complex_Fields.md:56-58`): integers and rationals combine exactly through `apply_binary_op`, and only an operation with a real operand is done in doubles. Without this, the defect of B1 (a) would remain inside texts with a constant: `(10^20 + 1 - 10^20) * pi` would be 0, and `10^400/10^399 * pi` an overflow; now they are pi and 10 pi. `float_arith` returns an error for a non-finite result, a division by zero, and a product, quotient or power of nonzero operands (or a converted exact value) below the normal range. A sum or difference cannot underflow: when it lies below 2^-1022 it is exact (argument in 2.2 item 10). So `pi - pi` stays 0, HEAD's value. Measured (`$S/r2/f64cmp/`, HEAD and revision-2 `eval_to_f64` on 7000 random bound texts, against gp at 100 digits): all 5084 exact texts give the correctly rounded double (HEAD gave 3340 values, 252 of them off by HEAD's double rounding). On the 942 texts with pi where HEAD gave a value, 912 give HEAD's double bit for bit (among them all 756 that HEAD rounded correctly), 27 give the correctly rounded value where HEAD was off, 1 differs where both are off, and 2 are ZeroDivisionErrors where HEAD returned ±inf (a division by `pi - pi`). The review's rows reproduce: three "No root found" (regression rows) and two roots 2. The sentences "HEAD's value bit for bit" are replaced by these measured statements. Texts with a real operand are computed in the real field of 53 bits, so cancellation after a real operand stays (`10^20 + pi - 10^20` = 0, as MAGMA gives at 53 bits). This is documented in the function and tested. | 0, 2.2 items 6 and 10, correctness, 2.3, 2.4 |
| B2 (blocker): the `primes` estimate charged an ECPP proof for every prime that the Montgomery-Vaughan bound allows, so it refused calls HEAD answered correctly in under 0.1 s (`primes(10^400, 10^400 + 10)` = `[]`) | Two passes above 2^64. Pass 1 runs I1's `is_probable_prime` (BPSW) on the odd candidates. It is refused beforehand only when `n_odd · t_test(B)` exceeds 10 s. Pass 2 proves the k probable primes with `try_is_prime`. The call is refused as soon as `n_odd · t_test(B) + k · t_proof(B)` exceeds 10 s, with k the count found so far (named in the message). Both models are recalibrated from measurements (release): one strong test by size (`$S/r2/tcost.rs`), HEAD's `primes` per odd candidate (`$S/r2/t_head_primes.txt`), I1's prototype ECPP times (100 bits measured here: 7.9 ms mean; 128 to 3072 bits from I1 §2.3.7). Pass 1 refuses nothing that HEAD finished in 30 s: HEAD spends 3.2 to 9.5 times `t_test` per odd candidate at every size measured. New estimates (`$S/r2/est2.py`, counts from gp): `primes(10^400, 10^400 + 10)` 0.002 s, `primes(2^1000, 2^1000 + 200)` 0.02 s, `primes(2^512, 2^512 + 70)` 0.002 s, `primes(10^30, 10^30 + 20000)` 2.2 s (258 primes), `primes(2^512, 2^512 + 2000)` 2.9 s (7 primes); all are admitted. `primes(2^1000, 2^1000 + 10^6)` is refused by pass 1 (100 s), `primes(2^1024, 2^1024 + 20000)` by pass 2 at the third probable prime (gp: 29 primes, about 96 s of proofs). The test of 6.4 and the release check of 8.2 are changed as asked, and the last paragraph of 6.3 is rewritten. | 0, 6.3, 6.4, 8.1, 8.2, 8.3 |
| m1: the prototype had `check_chart` only in `coordinate_form` | The revision-2 prototype adds it to `exterior_derivative` and `wedge` as 5.2 specifies. Test `test_chart_checked_in_form_arms` (5.4 rows) passes: both give the 5.2 message for a 4-name chart on R^3, and work with the 3-name chart. | 5.2, 5.4 |
| m2: `distribute_negation` in rustmath-symbolic is wrong in both arms | Confirmed (`rustmath-symbolic/src/expand.rs:250-263`: -(a + b) becomes (-a) - (-b), and -(a - b) becomes (-b) - (-a)); `expand(-(x - 1))` displays `(-1 - -x)`, which is x - 1. Listed in section 9 (S1's area, S1's design does not touch it), added to the HANDOFF list and to the tracker downgrade of 8.1 item 4. | 8.1, 9 |
| m3: the stack statement holds only for numeric chains | Qualified in section 9: a symbolic chain `w + w + ... + w` still aborts the debug driver at 3000 terms (2000 pass, measured on the revision-2 driver), through the recursive display and drop of the left-nested `Expr`. | 9 |
| m4: `2 ** -10001` printed `got 10001` for the exponent | The negative-exponent checks (Integer and Rational base) name the bounded quantity: `**: \|exponent\| must be an integer in 0..=10000 (REPL limit), got 10001`. The positive case keeps HEAD's text. Test `test_negative_exponent_message`. | 2.2 items 8 and 9, 2.4 |
| m5: graph limit doc | 6.1 and the `repl_limits` doc say that an assignment holds three copies (about 400 MB at the limit) and a bare expression two, so refused bare expressions start at about 266 MB. | 6.1 |
| m6: `median([r2, 1, 2])` panics on HEAD | Mentioned in 4.1 (HEAD: `rustmath-stats/src/statistics.rs:60`, `partial_cmp` of NaN unwrapped, a kernel kill), row `median([r2, 1, 2])` = 1 in 4.4 (F). It passes on the prototype. | 4.1, 4.4 |
| m7: `value_type_name` formatted the whole value with `Debug` | A `match` on the variant returning `&'static str` (exhaustive, so a new variant forces an update). Test `test_value_type_name_by_variant`. | 2.2 item 6 |
| Found while revising | (1) Revision 1's `float_arith` read a leading sign by recursion, one call per sign. It now reads the run in one loop, as `try_eval_binary_op` does: `"-"*100001 + "pi"` gives -pi in a 2 MB test thread. (2) The first draft of the underflow rule also refused a product with a zero operand. The random comparison found it (`pi + 12157665459056928801 * (pi - pi) - pi / (pi)`, HEAD and gp 2.14159...). The rule now asks for nonzero operands, and that row is a test. (3) In revision 1's prototype the doc line `/// Strip inline comments from a line ...` of `strip_inline_comment` ended up above `ChainLevel`. The implementer keeps it on its function. | 2.2 item 10, 2.4 |

## Revision 1

Changes after review 1 (`J1a-repl-parser-and-small.review1.md`). I agree with every issue; nothing is
disputed. The revision prototype is `$S/r1/` (table below): the original prototype plus six patch files,
68 of 68 tests pass (the 63 existing ones and 5 new revision tests), and every review input was rerun on
HEAD and on the revision.

| Issue | What changed | Sections |
|---|---|---|
| B1 (blocker): the left-associative `*`/`/` split turned `1/2 * 1/3` into a TypeError | Option (a). `apply_binary_op` gets Integer/Rational arms: Integer op Rational and Rational op Integer for `+ - * /` are done in Q; Integer / Integer is the rational quotient (MAGMA ch. 18 §18.4.1 "Division `/` gives a rational result", `Chapter_18_Ring_of_Integers.md:125`; ch. 20 §20.1.2 coercion, "the result lies in the larger of Q and R", `Chapter_20_Rational_Field.md:44-46`); Rational ** Integer for any sign. Every `/` with an Integer or Rational zero divisor is a ZeroDivisionError. This also fixes a HEAD panic that the review did not list: `h = 1/2; r = 0/5; h / r` panics at `rational.rs:354` (`Div` unwraps `Rational::new`). Measured: on every review input (`$V/t1.txt`..`t9.txt`, 221 commands) the revision differs from HEAD only where HEAD failed or was wrong; the rational products of `$V/t3.txt` are identical to HEAD. The sentence in 2.3 is corrected; section 9 no longer lists mixed Integer/Rational arithmetic. | 2.1, 2.2 items 8-9, 2.3, 2.4, 9 |
| B2 (blocker): `eval_simple_arithmetic` reads `-a+b` as `-(a+b)` | `eval_simple_arithmetic` and the textual `pi` substitution are deleted. `eval_to_f64` is rewritten on a new `float_arith`, which uses `top_level_chars`, `is_binary_sign` and the precedence of `try_eval_binary_op`, reads constants and numeric variables as whole names, and falls back to exact evaluation. Found on the way, same function: `pi2 = 5` is read as 3.14159...2 (the text replacement), a variable named `e` is ignored when it stands alone, and the bounds `1/0` and `1e400` make `find_root` answer `inf`. All fixed and tested. (Revision 2, B1, replaces the order of evaluation: exact first, then `float_arith` with exact subresults.) | 2.1, 2.2 item 10, 2.3, 2.4 |
| B3 (blocker): `graph_bytes` overflows u128 | `hash_set_bytes` and `graph_bytes` return `Option<u128>` with checked arithmetic; `None` (overflow) is "above the limit". Checked in a debug build (`$S/r1/gb/gb2.rs`): `complete_graph(2^62)` and `complete_graph(2^64-1)` give `None` and are refused, no panic. The sentence in 6.1 is corrected. | 6.1, 6.4 |
| m1: `count_bound` domain | `count_bound(y: &Integer) -> Option<u64>`: exact Integer comparison first; `None` for y > 8·10^6 (the bound there already exceeds 10^6); floats only on y <= 8·10^6. Test `primes(0, 10^400)` refused. | 6.3, 6.4 |
| m2: I3 also edits rustmath-rationals | Sentence fixed; merge note in 4.2 and 8.3. | 4.2, 8.3 |
| m3: non-finite float literals | A float literal that overflows (`1e400`) is a ValueError in `eval_expr`; `exact_to_f64` also refuses a non-finite Float; `eval_to_f64` checks its result. | 2.2 item 10, 4.2, 4.4 |
| m4: underflow | The `exact_to_f64` doc says that a nonzero value below the least subnormal rounds to 0.0; test `mean([Rational(1, 10^400)])` = 0. | 4.2, 4.4 |
| m5: phase 3 re-sieves | Stronger than asked: when phase 3 finds no factor of a composite that phase 2 already classified and sieved, there is no second I2 call at all. | 7.2 |
| m6: Jacobi shortcut | The odd-exponent condition is dropped. | 7.2 |
| m7: `SqrtModLimits` wording | "An ordinary private struct". | 7.2 |
| m8: debug test memory | `test_graph_limits` builds `path_graph(10^5)`; the boundary is checked on `graph_bytes`. | 6.4 |
| m9: psi_12 rows | `sqrt_mod(4, psi_12)` and `sqrt_mod(73, psi_12)` are restored with gp values. | 7.4 |
| m10: `--x` display | Listed in section 9. | 9 |
| Follow-on of B1 | Integer / Integer is now a Rational, so `plot_points=50/2` would have been refused with "got 25". `plot_int_options` takes a Rational with denominator 1 as its integer (MAGMA `Z ! a`). | 1.2, 1.4 |
| Found while revising | In the prototype, a run of leading signs and a run of redundant parentheses recursed once per character, so `"-"*722 + "1"` and 1000 nested parentheses aborted a debug build (stack overflow). Both runs are now read in one step (`$S/r1/patch_r1b.py`, `patch_r1c.py`). The `+ -` and `* /` chains are now folded in a loop (`patch_r1d.py`); HEAD recursed once per operator and a debug build aborted at a 732-term sum. 100001 signs, 10^5 parentheses and a 100001-term sum evaluate in a 2 MB test thread. HEAD's remaining stack aborts (35 nested calls, 815 nested lists in debug) are listed in section 9. | 2.2 items 5, 7; 9 |

Scratch directory `$S = /tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/J1a-repl-parser-and-small`:

| Path | Content |
|---|---|
| `$S/drive.rs` | REPL driver over the real `ReplContext` (one command per line, prints value or error, time, `HWM=1` prints VmHWM). Built as `drive_dbg` against the HEAD debug rlib: `rustc --edition 2021 -o drive_dbg drive.rs -L /home/user/RustMath/target/debug/deps --extern rustmath_jupyter=<the newest librustmath_jupyter-*.rlib>` (binary deleted, 117 MB). |
| `$S/s9_head.txt`, `$S/s9_more.txt`, `$S/n5.txt`, `$S/bigchart.txt` (deleted, 13 MB; regenerate with the python one-liner in section 5.4) | Inputs used for the reproductions below. |
| `$S/proto/` | A scratch copy of `repl.rs` built with the exact `--extern` list cargo uses (`proto/externs.txt`, `proto/build.sh`, `proto/build_lib.sh`). `patch_s9.py`, `patch_s9b.py`, `patch_s14.py`, `patch_s14_tests.py`, `patch_n5.py` are the prototype edits (exact-string replacements on HEAD's `repl.rs`). `proto/src/repl.rs` is HEAD's file with all five applied; with them the 63 existing REPL tests pass. Binaries deleted; `./build.sh t && ./t -q` rebuilds and runs in about 10 s (`./build_lib.sh` builds the driver `proto/drive` on the patched copy). |
| `$S/s11_check.rs`, `$S/s11_dump.rs` | `Rational::to_f64` old against num-rational 0.4.2; 3000 random rationals checked against Python `int/int` (correctly rounded): 0 mismatches. |
| `$S/td_cost.rs` | Release cost of HEAD's trial-division step and of a rho step by bit size. |
| `$S/graph_cost.rs` | Release mirror of `rustmath-graphs` `Graph` and the five generators: time and VmHWM. |
| `$S/list_cost.rs` | Memory of a REPL list of n Integers with the REPL's copies and the display text. |
| `$S/head_primes.rs` | HEAD release cost of `prime_pi` and `prime_range`. |
| `$S/sieve_i1.rs` | I1's segmented sieve prototype (`../I1-integers-primality/sieve_rs.rs`) with a timing `main`. |
| `$S/ref.gp`, `$S/ref.out` | PARI/GP 2.15.4 reference values (run `gp -q ref.gp < /dev/null`). |
| `$S/r2/` | Revision 2. `proto/` = `$S/r1/proto/src/repl.rs` plus `patch_r2.py` (B1 `eval_to_f64`, `float_arith`, `real_op`, `RealArith`; m1 `check_chart` in the two form arms; m4 `\|exponent\|`; m7 `value_type_name`; 6 tests). Rebuild: `cp ../r1/proto/src/repl.rs proto/src/repl.rs && python3 patch_r2.py proto/src/repl.rs && (cd proto && ./build.sh proto_test && ./build.sh lib && ./proto_test -q)` (74 passed). `f64cmp/` = HEAD's and the revision's `repl.rs` with a public hook `f64_of` = `eval_to_f64` (`f64cmp/build.sh`), `gen.py` (random bound texts), `all.txt` (7000 texts), `head.out`, `new.out`, `cmpref.py` (classification against gp at 100 digits). `tcost.rs`, `tcost_rel.out` (release cost of one strong test and of 167 remainders by size, and the ECPP prototype times), `t_head_primes.txt`, `t_head_primes_rel.out` (HEAD release `primes` above 2^64, per odd candidate), `drive_t.rs` (a driver that prints the time and the length of a list; built as `drive_t_rel` on HEAD's release rlib like `$S/r1/rebuild.sh`), `t_head_rows.txt`, `t_head_rows_rel.out` (HEAD release on the rows of 6.3), `est2.py`, `est2.out` (the two-pass model of 6.3, counts from gp), `r2.gp` (gp values of this revision), `fuzz/` (the review's fuzzed bounds `fya_*.txt` on the revision, `cmp.py`), `symchain_*.txt` (the m3 rows), `r2_t*.out` (the review-1 inputs on the revision: identical to revision 1 except the m4 message). Binaries deleted (about 90 MB each); the commands above rebuild them. |
| `$W = $S/review2/` | Reviewer 2's scratch: `r2_g.txt`, `r2_d.txt` (the B1 rows), `r2_b.txt` (the `expand` rows), `r2_median.txt`, `fya_*.txt` (fuzzed bounds) with HEAD's outputs `fyh_*.out`, `est.py` (revision 1's estimate), `head_repl.rs` (HEAD's file). |
| `$V = $S/review1/` | Reviewer 1's scratch: inputs `t1.txt`..`t9.txt`, `gb.rs` (the unchecked model that panics), `chk.gp`. |
| `$S/r1/` | Revision 1. `proto/` = `$S/proto/src/repl.rs` plus `patch_r1.py` (B1 arms, `is_exact_zero`, `group_inner`, `exact_to_f64` with num-rational standing in for the fixed `Rational::to_f64`, the finite float literal, `eval_to_f64`/`float_arith`, `list_to_floats`), `patch_r1b.py` (sign runs), `patch_r1c.py` (O(n) `group_inner`), `patch_r1d.py` (folded chains, `binary_chain`), `patch_r1_tests.py` (5 tests), `patch_r1e.py` (plot options take an integral Rational); apply in this order. `proto/build.sh` builds the test binary (`./build.sh && ./proto_test -q`: 68 passed) and `./build.sh lib` the driver `proto/drive`. `drive_head` = `drive.rs` on HEAD's debug rlib `librustmath_jupyter-5152ef5a4115ff6f.rlib`, `drive_rel` on the release rlib. Inputs `t_b1.txt`, `t_b2.txt`, `t_cl.txt`, `deep_*.txt`, `sum_*.txt`; `head_t*.out`/`r1_t*.out` = the review's `$V/t*.txt` on HEAD and on the revision before `patch_r1d.py`, `r1d_t*.out` after it (identical to `r1_t*.out`). `gb/gb2.rs` = the checked graph model. `r1.gp`, `psi.gp` = gp values (`gp -q r1.gp < /dev/null`). The large binaries are deleted (about 100 MB each); `rebuild.sh` rebuilds `drive_head`, `drive_rel`, `proto/drive` and `proto/proto_test` in under a minute. |

## 0. Summary

| Item | At HEAD | Fix |
|---|---|---|
| N5 plot options | `plot_points=` / `contours=` are clamped silently; unparsable values, unknown options and typos are ignored (6 plot commands and `Plot3DOptions::parse_from_args`). Reproduced: `plot_points=1000` draws 50 x 50, `plot_points=abc`, `=2.5`, `plot_pionts=25` all draw the default, `contours=100` draws 30, `cmap="jet"` is ignored. | One option parser `plot_int_options`: exact option names, values evaluated, `int_arg` range check, ValueError otherwise. Ranges: the old maxima stay as upper bounds (they bound the SVG size), the lower bounds become the smallest value the code can draw (2 grid points, 1 contour). `Plot3DOptions::parse_from_args` returns `Result` and loses its unused `plot_points`. |
| S9 parser | No string-literal awareness in any splitter; `-` is treated as binary even after an operator; char indices used as byte indices. Reproduced: `2 * -3`, `"ab" * -5`, `2 ** -1`, `3 - -2`, `-x`, `2 * 1e-3`, `"a-b"`, `s = "x,y,z"`, `chart(E3, "x,y,z")`, `(2+3)*4` are SyntaxErrors; `-2**2` gives 4; `π*2` **panics** the kernel (`repl.rs:9597`); `"é" + "a"` and `f("é", 2)` fail. The float evaluator of plot ranges and `find_root` bounds (`eval_simple_arithmetic`) reads `-3+2` as -5 and `-pi-1` as -2.14, and its text substitution reads a variable `pi2` as 3.14159...2: silent wrong ranges and false "No root found". `1/2 / (0/5)` **panics** (`rational.rs:354`). | One lexical scanner `LexState` (quotes with backslash escapes, bracket depth, byte offsets) used by every splitter; binary-operator splitting by precedence with unary-sign detection (Python/Sage/PARI rules); grouping parentheses; runs of signs and of parentheses in one step; `Integer ** negative` gives the rational (MAGMA, Sage, PARI `2^-1 = 1/2`); Integer/Rational arithmetic in Q and ZeroDivisionError for every exact zero divisor (MAGMA ch. 18, ch. 20), so the left-associative split keeps every rational product HEAD computed; `eval_to_f64` rewritten on the same rules: exact evaluation first, and for a text with a real operand `float_arith`, MAGMA's coercion into the real field of doubles with exact subresults (revision 2). Revision-2 prototype passes the 63 existing and 11 new tests. |
| S10 pochhammer | `pochhammer` is an alias of `falling_factorial` (`:2348`): `pochhammer(3, 4) = 0`. | Alias of `rising_factorial`, the Pochhammer symbol (a)_n = a(a+1)...(a+n-1) of MAGMA ch. 49 (`Chapter_49_Power_Laurent_and_Puiseux_Series.md:292`), Sage and the workspace's own `rustmath-calculus/src/pochhammer.rs`. `pochhammer(3, 4) = 360`. |
| S11 to_f64 | `num.to_f64() / den.to_f64()` at `:3018`, `:3024`, `:5180-5181`, `:8878-8879`, `:8898-8899`, `:8926-8927`, and in `Rational::to_f64` itself (`rustmath-rationals/src/rational.rs:156-160`). Reproduced: `mean([(10^400+1)/10^400])` = NaN, `fft([..])` = `[NaN + 0i]`, `find_root(x - 1, x, 0, (2·10^400+1)/10^400)` = 0 (the root is 1), `mean([1/10^310])` = 0 (true 1e-310), `mean([2^1100])` = inf. | `Rational::to_f64` delegates to num-rational 0.4.2 `Ratio<BigInt>::to_f64` (correctly rounded, already a dependency). The REPL converts every exact number through one helper `exact_to_f64` that refuses a value outside the double range (ValueError instead of inf or NaN); a float literal that overflows (`1e400`) is a ValueError too. |
| S14 chart | `chart(M, names)` (`:4579-4595`) builds a chart with as many coordinates as names; `m` is unused. A chart can also be used with a manifold of another dimension (`coordinate_form`, `exterior_derivative`, `wedge`). | `chart` passes dim(M) to `Chart::new` (whose own check then works) and checks names: exactly dim(M), identifiers, distinct (Sage: "the list of coordinates must contain dim elements"). `check_chart` in the three arms that take a chart. The tests that relied on one-name charts are rewritten (after MF1's version of them). |
| S15 limits | `Graph`/`complete_graph`/`cycle_graph`/`path_graph`/`star_graph`/`wheel_graph` (`:3166-3201`), `prime_pi` (`:2703`), `primes` (`:2718`) have no limit. | Graphs: a measured memory model (within 2 % of the real REPL peak) in checked u128 arithmetic (an overflow counts as above the limit), refused above 133 MB per copy (400 MB at the peak of an assignment). `prime_pi`: x <= 6·10^9 (about 10 s on I1's sieve). `primes(a, b)`: a proven bound on the number of primes (Montgomery-Vaughan) <= 10^6, decided by an exact comparison for spans above 8·10^6, and above 2^64 two passes with a time model each (revision 2): BPSW on every odd candidate, then an ECPP proof (`try_is_prime`) for each of the k probable primes actually found, refused when the modelled time exceeds 10 s; an unresolved ECPP proof is a ValueError, not a panic. Help text: `primes(a, b)` is half-open. |
| sqrt_mod | A composite n is factored by trial division, up to three times (`:10998-11014`). Reproduced: `sqrt_mod(4, 2^127+1)` and `sqrt_mod(4, p20·q21)` do not finish (killed after 20 s; the doc comment estimates 7.5e18 divisions for the first). | n is factored once: trial division to 10^4, I2's `factorization_with_limits` (Pollard rho, quadratic sieve up to 60 digits) for cofactors of at most 4096 bits, then streamed trial division with a budget that covers everything HEAD could factor in 10 s; ECPP proofs (I1) for probable primes up to 1024 bits. Roots per prime power from rustmath-finitefields, combined by CRT. An unfactored part is a ValueError that names it; an unproven large prime keeps HEAD's caveat. |

Nothing is declared out of scope. Defects found on the way and not in this unit are listed in section 9.

Dependencies on earlier units (all implemented before J1a): I1 (`prime::try_is_prime`,
`Integer::try_sqrt_mod_prime`, `sieve::for_each_prime`, the sieve-backed `prime_pi` and `prime_range`), I2
(`factorization_with_limits`, `FactorizationLimits`, `LimitedFactorization`), MF1 (its rewrite of
`test_form_component_limits`, section 4.3 of `MF1-manifolds.md`), C1 (whatever it does to
`rising_factorial`; S10 only moves a name). If one of them landed under other names, map them; section 8
lists the fallbacks.

---

## 1. N5: plot options clamp silently and ignore what they cannot parse

### 1.1 Defect at HEAD

| Site | Command | Code | Behaviour |
|---|---|---|---|
| `:5472-5485` | `plot_slope_field` | `part.starts_with("plot_points")`, `parse::<usize>()`, `n.min(50).max(5)`; a bare integer part is also n_points | clamped to 5..=50; unparsable value or unknown option: default 20 silently |
| `:5720-5733` | `parametric_plot` | same, `min(1000).max(10)` | clamped to 10..=1000 |
| `:5944-5961` | `contour_plot` | `contours`/`n_contours` `min(30).max(3)`; `plot_points` `min(100).max(20)` | clamped; any other option ignored |
| `:6231-6242` | `vector_field` | `min(30).max(5)` | clamped |
| `:6462-6473` | `implicit_plot` | `min(200).max(50)` | clamped |
| `:6786-6797` | `region_plot` | `min(200).max(50)` | clamped |
| `:231-275` | `Plot3DOptions::parse_from_args` (pub; no caller in the workspace) | `plot_points` `clamp(10, 100)` (`:255`), `opacity` `clamp(0.0, 1.0)` (`:248`), unknown keys ignored (`:268`) | clamped; and the `plot_points` field (`:202`) is read by no generator (the 3D grids are fixed at `:7002`, `:7124`, `:7210`) |

`starts_with` also accepts `plot_pointsX=5`. Reproduction (`$S/n5.txt`, `$S/drive_dbg`):

```
plot_slope_field(x - y, (x, -2, 2), (y, -2, 2), plot_points=1000)  -> Slope field with 2500 arrows   (50 x 50)
plot_slope_field(..., plot_points=abc)                               -> Slope field with 400 arrows    (default 20)
plot_slope_field(..., plot_points=2.5)                               -> Slope field with 400 arrows
plot_slope_field(..., plot_pionts=25)                                -> Slope field with 400 arrows
plot_slope_field(..., plot_points=3)                                 -> Slope field with 25 arrows     (5 x 5)
contour_plot(x^2 + y^2, (x, -2, 2), (y, -2, 2), contours=100)        -> Contour plot with 30 levels
contour_plot(..., cmap="jet")                                        -> Contour plot with 10 levels
contour_plot(..., contours=5, n_contours=6)                          -> Contour plot with 6 levels
```

### 1.2 Fix

New `ReplContext` method (prototype `$S/proto/patch_n5.py`, compiled and run):

```rust
/// The integer options of a plot command (N5): `parts` are the arguments after the positional
/// ones. Each is `name=value` with `name` one of `options` (exact match), and `value` evaluates
/// to an integer inside that option's range; `bare` names the option that a bare integer
/// argument sets (the old positional n_points form). Anything else is an error: nothing is
/// clamped, defaulted or ignored. An option given twice is an error.
fn plot_int_options<S: AsRef<str>>(&mut self, cmd: &str, parts: &[S],
    options: &[(&str, RangeInclusive<usize>)], bare: Option<&str>)
    -> Result<HashMap<String, usize>, EvalError>
```

- The `=` is the first top-level `=` of the part (`top_level_chars`, section 2), so a value may be an
  expression (`plot_points=2*10`, `plot_points=n`).
- No `=` and `bare` is None: ArgumentError `"{cmd}: unexpected argument {part}; options are given as name=value"`.
- Unknown name: ValueError `"{cmd}: unknown option {name} (options: a, b)"`.
- The value is `self.eval_expr(value)?`; an Integer goes through `int_arg::<usize>(cmd, name, &n, range, "")`
  (message `"{cmd}: {name} must be an integer in a..=b, got v"`); a Rational with denominator 1 is taken as
  its numerator (revision 1: S9 item 9 makes `n/2` and `10/2` Rationals, and `got 5` for the value 5 would
  be a confusing refusal; MAGMA coerces such a rational into Z too, `Z ! a`, ch. 18 §18.2.2, `Chapter_18_Ring_of_Integers.md:61`); any other value
  gives the same message with its display text (`got abc`, `got 2.5`, `got 5/2`).
- Twice: ValueError `"{cmd}: option {name} is given twice"`.

Ranges, new constants in `repl_limits` (comment: "plot options (N5): the old clamp maxima, now refused
above; they bound the number of SVG elements"):

| Command | Option(s) | Range | Default (unchanged) | bare |
|---|---|---|---|---|
| `plot_slope_field` | `plot_points` | `2..=SLOPE_FIELD_POINTS` (50) | 20 | `plot_points` |
| `parametric_plot` | `plot_points` | `2..=PARAMETRIC_POINTS` (1000) | 200 | `plot_points` |
| `contour_plot` | `contours`, `n_contours` | `1..=CONTOURS` (30) | 10 | none |
| `contour_plot` | `plot_points` | `2..=CONTOUR_POINTS` (100) | 50 | none |
| `vector_field` | `plot_points` | `2..=VECTOR_FIELD_POINTS` (30) | 15 | none |
| `implicit_plot`, `region_plot` | `plot_points` | `2..=IMPLICIT_POINTS` (200) | 100 | none |

Lower bounds: every grid divides by `n - 1` (`:5496`, `:5741`, `:5973`, `:6251`, `:6481`, `:6805`) and the
marching squares loops run over `n - 1` cells, so 2 is the least value that draws; the contour level formula
`(i + 0.5) / n` (`:6055`) needs n >= 1. The old minima (5, 10, 3, 20, 5, 50) were cosmetic; a value between
2 and the old minimum is now drawn as asked. `contour_plot` refuses `contours` together with `n_contours`
(ValueError `"contour_plot: give contours or n_contours, not both"`).

`Plot3DOptions` (`:192-276`): delete the field `plot_points` (`:201-202`, `:220`): no generator reads it,
so accepting it would be the same silent ignore. `parse_from_args` becomes
`pub fn parse_from_args(args: &[&str]) -> Result<Self, EvalError>`: an argument without `=`, an unknown key
(accepted: `color`/`rgbcolor`, `opacity`, `mesh`, `interactive`, `axes`, `shading`), an `opacity` that does not
parse as f64 or lies outside [0, 1], and a boolean that is not one of `True/true/1/False/false/0` are
ValueErrors naming the key. Its doc says no command calls it yet. The `_ =>` arm at `:268` goes.

Correctness: the options either take the value the user wrote or the call fails with the range; the drawing
code receives a value in its range as before.

### 1.3 Callers and behaviour changes

`grep -n "plot_points\|contours" repl.rs`: the six evaluators and `Plot3DOptions`; nothing else.
`grep -rn "Plot3DOptions\|parse_from_args"` over the workspace: only `repl.rs` (`:7730`, `:8268`, `:8695`
use `Plot3DOptions::default()`, unchanged). No crate depends on rustmath-jupyter.

Changes: out-of-range values, unknown options, unparsable values and a repeated option are now errors (they
were clamped or ignored); values between 2 and the old minimum are drawn as asked; expressions, variables and
integral rationals (`50/2`) are accepted as values. The existing plot tests (`:11430-11560`, implicit_plot etc.) pass no options and are
unchanged.

### 1.4 Tests (all new; F = fails on HEAD)

`test_plot_options_checked`:

| Command (after `x = var("x")`, `y = var("y")`) | Expected | |
|---|---|---|
| `plot_slope_field(x - y, (x, -2, 2), (y, -2, 2), plot_points=25)` | description `Slope field with 625 arrows` | P |
| same with bare `25` | `625 arrows` | P |
| same with `plot_points=3` | `9 arrows` (HEAD 25) | F |
| same with `plot_points=1000` | ValueError `plot_slope_field: plot_points must be an integer in 2..=50, got 1000` | F |
| same with `plot_points=abc`, `plot_points=2.5`, `plot_points=5/2` | ValueError, `got abc`, `got 2.5`, `got 5/2` | F |
| same with `plot_points=50/2` (revision 1) | `625 arrows` (an integral Rational) | F (HEAD: default 20, 400 arrows) |
| same with `plot_pionts=25` | ValueError `plot_slope_field: unknown option plot_pionts (options: plot_points)` | F |
| `contour_plot(x^2 + y^2, (x, -2, 2), (y, -2, 2), contours=5)` | `Contour plot with 5 levels` | P |
| same, `plot_points=2*10` | Ok | P |
| same, `contours=100` | ValueError `contour_plot: contours must be an integer in 1..=30, got 100` | F |
| same, `cmap="jet"` | ValueError `contour_plot: unknown option cmap (options: contours, n_contours, plot_points)` | F |
| same, `contours=5, n_contours=6` | ValueError `give contours or n_contours, not both` | F |
| `parametric_plot((x, x^2), (x, 0, 1), plot_points=1001)` | ValueError `... in 2..=1000 ...` | F |
| `vector_field((y, -x), (x, -1, 1), (y, -1, 1), plot_points=31)` | ValueError `... in 2..=30 ...` | F |
| `implicit_plot(x^2 + y^2 - 1, (x, -2, 2), (y, -2, 2), plot_points=201)`, same for `region_plot` | ValueError `... in 2..=200 ...` | F |
| `implicit_plot(..., plot_points=2)` | Ok (HEAD silently 50) | F |
| `Plot3DOptions::parse_from_args(&["opacity=1.5"])`, `(&["plot_points=40"])`, `(&["foo=1"])`, `(&["mesh=maybe"])` | Err each; `(&["color='red'", "opacity=0.5", "mesh=True"])` Ok with color `#ff6666`, opacity 0.5, mesh true | F (does not compile on HEAD: different return type) |

No external reference is needed: the expected values are counts of grid points (n^2 arrows) and the messages.

---

## 2. S9: parser defects

### 2.1 Defect at HEAD

Every splitter tracks bracket depth and ignores string literals; the binary-operator splitter treats every
`-` as binary; three scanners mix char indices and byte offsets (`try_eval_binary_op`, the function-call
check, `eval_simple_arithmetic`).

| Site | What |
|---|---|
| `try_eval_binary_op` `:9534-9664`: `find_op_outside_parens` (`:9539-9575`) | Right-to-left over `Vec<char>`, quotes ignored. The `+`/`-` loop (`:9578-9590`) splits at the rightmost `+`, then at the rightmost `-`, whatever precedes it (a unary sign after `*`, `**`, `^` or `-`, or the exponent sign of `1e-3`). The returned position is a **char index** used to slice the `&str` (`expr[..pos]`, `:9581`, `:9597`), which panics on non-ASCII input. `*` is tried before `/` (`:9593`, `:9605`): `a * b / c` becomes `a * (b / c)`. |
| `eval_expr` `:687-807` | Tuple split `:694-707` (`split_at_depth_zero`). Function-call check `:755-785`: `chars().enumerate()` index compared with `args_part.len() - 1` (bytes), quotes ignored. String literal `:787-793`: starts and ends with a quote, not checked to be one literal. No grouping parentheses: `(2+3)` is a SyntaxError (the function-call check needs a name). No unary minus on a non-literal: `-x` is a SyntaxError. |
| `parse_args` `:4712-4744`, `parse_list_elements` `:4802-4836`, `parse_symbolic_args` `:4865-4899`, `try_parse_sagemath_plot` `:4974-5004`, `eval_slope_field` `:5405-5428`, `split_at_depth_zero` `:5057-5075`, `split_rhs_and_bounds` `:8953-8971`, `find_equation_operator` `:8935-8951` | Depth only; a comma, bracket or `==` inside a string literal counts. |
| `parse_list_of_integers` `:4853` | `inner.split(',')`: not even depth-aware (`Polynomial([gcd(4, 6), 1])` fails). |
| `find_root` bounds `:3144` | `bounds_start.split(',')`: same. |
| `eval_to_f64` `:5142-5186` and `eval_simple_arithmetic` `:5189-5253` (revision 1) | The float evaluator of every plot range, `find_root` bound and point coordinate (55 calls of `eval_to_f64`, `:3148-7694`, all in `repl.rs`). `eval_to_f64` first replaces the text `pi` by `3.141592653589793` everywhere (`:5154-5156`), also inside names; a bare `pi`, `e`, `tau` is the constant even when a variable has that name (`:5146-5151`). `eval_simple_arithmetic` applies a leading `-` to the whole rest (`:5197-5202`), splits at the rightmost `+` or `-` whatever precedes it, compares a byte index with `chars().nth(i - 1)` (`:5217`), and divides by zero to inf. Its result is not checked to be finite. |
| `apply_binary_op` `:9666-9968` (revision 1) | No arm mixes Integer and Rational, and there is no Integer / Integer arm (the literal `a/b` with two i64 parts is read as a Rational at `:737-748`, any other quotient of integers is a TypeError). `(Rational, "/", Rational)` calls `Rational`'s `Div`, which unwraps `Rational::new` and **panics** on a zero divisor (`rustmath-rationals/src/rational.rs:354`). |

Reproduction (`$S/s9_head.txt`, `$S/s9_more.txt`, HEAD debug driver):

```
2 * -3              -> SyntaxError: Cannot parse: 2 *            (PARI 2*-3 = -6)
"ab" * -5           -> SyntaxError: Cannot parse: "ab" *
2 ** -1, 2 ^ -1     -> SyntaxError: Cannot parse: 2 **           (PARI 2^-1 = 1/2)
-2 ** 2, -2^2       -> 4                                         (PARI -2^2 = -4; Python -2**2 = -4)
-x                  -> SyntaxError: Cannot parse: -x             (also when x = 5)
3 - -2              -> SyntaxError: Cannot parse: 3 -            (5)
2 * 1e-3            -> SyntaxError: Cannot parse: 1e
"a-b", "a+b"        -> SyntaxError: Cannot parse: "a
s = "x,y,z"         -> SyntaxError: Cannot parse: "x
chart(E3, "x,y,z")  -> SyntaxError: Cannot parse: "x
'x,y'               -> SyntaxError: Cannot parse: 'x
(2+3)*4             -> SyntaxError: Cannot parse: (2+3)          (20)
x + y - z           -> (x + (y - z))                             (left associative: ((x + y) - z))
π*2                 -> thread 'main' panicked at repl.rs:9597:32: end byte index 1 is not a char boundary
"é" + "a"           -> SyntaxError: Cannot parse: + "a"
f("é", 2)           -> SyntaxError: Cannot parse: f("é", 2)
```

Revision 1 additions (`$S/r1/t_head1.txt`, `$S/r1/t_b2.txt`, `$S/r1/t_panic.txt`, HEAD debug driver
`$S/r1/drive_head`):

```
w = var("w")
find_root(w^2 == 4, -3+2, 3)       -> No root found      (the bound became -5; gp -3+2 = -1, solve(w=-1,3,w^2-4) = 2)
find_root(w^2 == 9, -pi-1, 0)      -> No root found      (the bound became -(pi-1) = -2.14; gp root -3)
find_root(w^2 == 9, -1-3, 0)       -> No root found      (the bound became 2; gp root -3)
find_root(w^2 == 2, -2^2, 0)       -> 1.4142135623842478 (the bound became +4; gp -2^2 = -4, root -1.414...)
pi2 = 5; find_root(w^2 == 16, 0, pi2)  -> No root found  (pi2 read as 3.14159...2; gp root 4)
e = 5;   find_root(w^2 == 16, 0, e)    -> No root found  (the variable e is ignored; root 4)
find_root(w^2 == 4, 0, 1/0)        -> inf                (a root at infinity; 1/0 is not a number)
find_root(w^2 == 4, 0, 1e400)      -> inf
find_root(w^2 == 2, 0, 2pi)        -> 1.414213562377824  (2pi is not REPL syntax; read as 23.14159...)
h = 1/2; r = 0/5; h / r            -> thread 'main' panicked at rustmath-rationals/src/rational.rs:354:33: DivisionByZero
1/2 * 3, 8/4 * 2, a/b (a = 8, b = 4) -> TypeError        (gp 3/2, 4, 2)
```

The panics kill the kernel (`kernel.rs`/`handlers.rs` have no `catch_unwind`).

### 2.2 Fix

Grammar (Python, Sage and PARI agree on every rule used here):

```
sum     := product (('+' | '-') product)*          left associative
product := unary (('*' | '/') unary)*              left associative
unary   := ('-' | '+') unary | power               -2**2 = -(2**2)
power   := atom (('**' | '^') unary)?              right associative; 2**-1, 2^3^2 = 2^9
atom    := literal | name | call | '(' expr ')' | '[' list ']'
```

The REPL keeps its split-based evaluator (rewriting it as a tokenizer and Pratt parser would touch every
command); it gets one lexical scanner and the precedence rules above. New items next to
`strip_inline_comment` (`:10236`), exact code in `$S/proto/patch_s9.py`:

```rust
/// Bracket depth and string-literal state of a left-to-right scan of REPL source text (S9).
/// A string literal starts with `"` or `'` outside a literal and ends at the next unescaped copy
/// of the same quote; inside it a backslash escapes the next character, as in
/// `strip_inline_comment`. Brackets inside a literal do not count.
#[derive(Clone, Copy, Debug, Default)]
struct LexState { depth: i32, quote: Option<char>, escaped: bool }

impl LexState {
    /// Advances over `ch`. True when `ch` lies at bracket depth 0 outside every string literal
    /// and is neither a bracket nor a quote: the characters at which a splitter may cut.
    fn top_level(&mut self, ch: char) -> bool;
}

/// The byte offset and the character of every top-level character of `s`.
fn top_level_chars(s: &str) -> Vec<(usize, char)>;

/// Whether `s` is exactly one string literal: it opens with a quote, and that literal closes at
/// the last character of `s`.
fn is_string_literal(s: &str) -> bool;

/// Whether the '+' or '-' at byte `pos` of `s` is a binary operator: the previous non-blank
/// character ends an operand (alphanumeric, '_', '.', a closing bracket or a closing quote), and
/// the sign is not the exponent sign of a float literal (1e-3, 2.5E+10: directly after an 'e'
/// that ends a decimal mantissa which is not the tail of a name, and followed by a digit).
fn is_binary_sign(s: &str, pos: usize) -> bool;
```

Changes:

1. `split_at_depth_zero` (`:5057`) iterates `top_level_chars(s)` (byte offsets; `start = i + ch.len_utf8()`).
2. The six depth loops (`parse_args`, `parse_list_elements`, `parse_symbolic_args`,
   `try_parse_sagemath_plot`, `eval_slope_field`, and the row loop of `parse_nested_list` (`:4762-4791`,
   brackets inside a literal are pushed and not counted)) call `lex.top_level(ch)` and cut at a `,` only when
   it returns true. `split_rhs_and_bounds` and `find_equation_operator` use `top_level_chars` (`==` is two
   consecutive top-level `=`). `parse_list_of_integers` (`:4853`) and the `find_root` bounds (`:3144`) use
   `split_at_depth_zero`.
3. Function-call check (`:760-781`): `char_indices` with a `LexState`; the call is valid when the `(` after the
   name closes exactly at the last byte, outside any literal.
4. String literal (`:787-793`): `if is_string_literal(expr)`. The value is still the raw text between the
   quotes (escapes are not translated, as at HEAD).
5. `try_eval_binary_op` (`:9534-9664`) rewritten on `top_level_chars(expr)`:
   1. if there is a top-level binary `+`/`-` (`is_binary_sign`): split at every one of them and fold the
      operands from the left with `apply_binary_op` (`a - b + c` = `(a - b) + c`);
   2. else, if there is a top-level `/` or a `*` that is not part of `**`: the same for that level
      (left associative: `a * b / c` is `(a * b) / c`; item 9 makes this give the same values as HEAD's
      `a * (b / c)`);

      Both levels use one helper (revision 1, `$S/r1/patch_r1d.py`):

      ```rust
      /// The two left-associative precedence levels of the S9 grammar.
      enum ChainLevel { Sum, Product }
      /// The chain of `s` at one level: every top-level binary `+`/`-` (`is_binary_sign`), or every
      /// top-level `*` that is not part of `**` and every `/`, with the operands between them
      /// (trimmed). None when the level has no operator or an operand is blank ("3 -": the caller
      /// then reports the text as unparsable). `top` is `top_level_chars(s)`.
      fn binary_chain<'a>(s: &'a str, top: &[(usize, char)], level: ChainLevel)
          -> Option<(Vec<&'a str>, Vec<&'static str>)>;
      ```

      Folding in a loop evaluates the operands in the same order as the recursive split (left subtree
      first) and builds the same tree, so values, errors and symbolic display are those of a
      rightmost split; but the number of terms adds no recursion. HEAD (and the unrevised prototype)
      recursed once per operator, and a debug build overflowed its 8 MB stack at a sum or product of 732
      terms (`$S/r1/sum_*.txt`; release: 10682); the revision evaluates `"1" + "+1"*100000` in 0.1 s (debug
      driver) and in the 2 MB stack of a test thread;
   3. else a run of leading signs (`-`, `+`, blanks between them), read in one loop: the parity of the `-`
      signs decides; evaluate the rest once, then `self.negate(&v)` if the count is odd. A run of any length
      adds one recursion (revision 1: the prototype recursed per sign, and 722 signs overflowed the 8 MB
      stack of a debug build; now `"-"*100001 + "1"` gives -1 in a 2 MB test thread);
   4. else the leftmost `**` (two consecutive top-level `*`) or `^`; the right operand may be signed.
   A candidate with an empty operand is not split (the caller reports "Cannot parse", as now).
   Integer and float literals such as `-5` and `-1.5e-3` are still read before (`:723-735`).
6. New `fn negate(&self, v: &RustMathValue) -> Result<RustMathValue, EvalError>`: `Integer`, `Rational`,
   `Float`, `Complex`, `IntegerMod` by `Neg` (all implement it: `rustmath-rationals/src/rational.rs:369`,
   `rustmath-complex/src/complex.rs:522`, `rustmath-finitefields/src/integer_mod.rs:516`), `Symbol` and `Expr`
   to `Expr(-e)` (`rustmath-symbolic/src/expression.rs:538`), `Matrix`, `Polynomial`, `CliffordElem` through
   the existing `Integer(-1) * v` arms; anything else TypeError `"bad operand type for unary -: {type}"`.
   Helper `fn value_type_name(v: &RustMathValue) -> &'static str`: a `match` on the variant that returns
   its name (`"Integer"`, `"String"`, `"List"`, ...; revision 2, m7: the first version formatted the whole
   value with `Debug` to read the name, which builds tens of MB for a list of 10^6 entries). The match is
   exhaustive, so a variant added later must be named here.
7. Grouping parentheses in `eval_expr`, after the list-literal check and before the identifier fallback
   (`$S/proto/patch_s9b.py`, revision 1 `$S/r1/patch_r1c.py`): `if let Some(inner) = group_inner(expr)
   { return self.eval_expr(inner) }` with

   ```rust
   /// The text inside the grouping parentheses around `s` (S9): Some(e) when s = "(..(e)..)" with
   /// j >= 1 pairs that each enclose all of e, e is not blank and has no top-level comma (a tuple
   /// such as "(x, 0, 1)" is not a group; "((x, 0, 1))" gives "(x, 0, 1)"). All j pairs go in one
   /// pass over the depth profile (LexState), so a run of any length costs O(len) and no recursion.
   fn group_inner(s: &str) -> Option<&str>;
   ```

   Algorithm: let p_0 < ... < p_(k-1) be the `(` of the leading run of `(` and blanks, q_0 > ... > q_(r-1)
   the `)` of the trailing run. The pair (p_t, q_t) encloses everything iff the bracket depth (after each
   character, `LexState`, literals skipped) stays above t on every character strictly between the two runs.
   So j = min(k, r, D) with D the least depth on that stretch (j = min(k, r) when the stretch is empty);
   the text must end at depth 0 outside a literal, else None. If the text inside the j-th pair has a
   top-level comma, j decreases by one (the tuple keeps its own parentheses). Inside text that is blank
   gives None. Checked cases (`$S/r1/t_b2.txt` and the test in 2.4): `((a)+(b))` gives `(a)+(b)`;
   `(1)+(2)` and `(a)(b)` give None (D = 0); `(")")` gives `")"`; `(1))`, `((1)`, `()`, `(())` give None.
   `(x, 0, 1)` stays a SyntaxError as an expression (plot commands read their ranges from the raw text).
8. `apply_binary_op`, new arm before `(Integer(a), "**" | "^", Integer(b))`: for b < 0,
   `a^b = 1 / a^|b|` in Q (`Rational::new(1, a^|b|)`), with |b| checked by the existing
   `int_arg::<u32>(op, "|exponent|", &(-b), 0..=OP_POW_EXP, "REPL limit")` (revision 2, m4: the name says
   that |b| is bounded, so `2 ** -10001` reads `**: |exponent| must be an integer in 0..=10000 (REPL
   limit), got 10001`; the positive case keeps HEAD's `exponent`), and a = 0 a ZeroDivisionError
   (PARI `0^-1`: error `e_INV`). MAGMA, Sage and PARI give `2^-1 = 1/2`; without this arm the parse fix
   would turn `2 ** -1` into an exponent-range error.
9. (Revision 1, B1.) Integer and Rational arithmetic in `apply_binary_op`, a block before the Rational arms
   (exact code `$S/r1/patch_r1.py`). MAGMA: "Division `/` gives a rational result" (ch. 18 §18.4.1,
   `Chapter_18_Ring_of_Integers.md:125`); `a / b` for integers "form the rational number a/b in reduced form"
   (ch. 20 §20.2.2, `Chapter_20_Rational_Field.md:88`); coercion between Z and Q puts the result in Q (§20.1.2,
   `:44-46`); `a ^ k` on rationals (§20.4.2, `:157`). Sage and PARI agree (gp `1/2*3` = 3/2, `8/4*2` = 4).

   ```rust
   // A zero divisor, Integer or Rational, for every pair of exact numbers (first, so that Rational's
   // panicking Div is never reached with 0).
   (Integer(_) | Rational(_), "/", Integer(_) | Rational(_)) if is_exact_zero(right) =>
       Err(EvalError::new("ZeroDivisionError", "division by zero")),
   (Integer(a), "/", Integer(b)) => Rational::new(a, b) as a Rational value,
   (Integer(a), "+" | "-" | "*" | "/", Rational(_)) => self.apply_binary_op(&Rational(from_integer(a)), op, right),
   (Rational(_), "+" | "-" | "*" | "/", Integer(b)) => self.apply_binary_op(left, op, &Rational(from_integer(b))),
   (Rational(a), "**" | "^", Integer(b)) => numerator and denominator to the power |b|
       (int_arg::<u32>(op, if b < 0 { "|exponent|" } else { "exponent" }, &|b|, 0..=OP_POW_EXP,
       "REPL limit")), swapped for b < 0;
       0 to a negative power is the ZeroDivisionError "{op}: 0 has no inverse (exponent {b})",
   ```

   New helper `fn is_exact_zero(v: &RustMathValue) -> bool` (Integer zero, or Rational with numerator zero).
   The result of Integer / Integer is a Rational also when the quotient is integral (`8/4` displays `2`,
   as the literal `8/4` does at HEAD, `:737-748`). No `Rational::new(...).unwrap()`: an `Err` maps to
   ZeroDivisionError. Values the REPL keeps as Integer are unchanged: only pairs with a Rational or a `/`
   between two Integers reach these arms.
10. (Revision 1, B2; revision 2, B1.) `eval_to_f64` (`:5142-5186`) is rewritten and `eval_simple_arithmetic`
   (`:5189-5253`) is deleted (its only caller is `eval_to_f64`), with the textual `pi` substitution. Exact code:
   `$S/r2/patch_r2.py`.

   ```rust
   /// A real number written as text: a plot range, a find_root bound, a point coordinate (S9,
   /// S11). The text is evaluated exactly first (`eval_expr`): an Integer or a Rational is rounded
   /// once to the nearest double (`exact_to_f64`, section 4), a Float is taken as it is. Only a text
   /// whose value is not an exact number (it uses pi, e, tau, a decimal literal or a Float
   /// variable) goes to `float_arith`. The result is a finite double or an error.
   fn eval_to_f64(&mut self, s: &str) -> Result<f64, EvalError>;

   /// A value inside `float_arith`: an exact number (Integer or Rational), or an element of the
   /// real field of doubles.
   enum RealArith { Exact(RustMathValue), Real(f64) }

   /// The value of an arithmetic text in the real field of doubles, by MAGMA's coercion rule
   /// (ch. 25 §25.1.2): integers and rationals combine exactly (`apply_binary_op`, the REPL's own
   /// arithmetic), and an operation with a real operand rounds the exact operand to the nearest
   /// double and is done in IEEE double arithmetic, the real field of 53 bits. Real operands are
   /// the constants pi/PI, e/E, tau/TAU (a variable of the same name wins), decimal literals and
   /// Float variables. So `10^400/10^399 * pi` is 10 pi, while `10^20 + pi - 10^20` is 0, as in
   /// MAGMA at this precision: cancellation after a real operand is part of the semantics. An
   /// exact power with a non-integral rational exponent is real (`2^(1/2)`, as in MAGMA).
   /// Precedence, associativity and unary signs as in `try_eval_binary_op`.
   /// Ok(None): `s` uses something else (a call, another variable, a string). Err: the value
   /// overflows or underflows a double, a division by zero, or an error of the exact arithmetic.
   fn float_arith(&self, s: &str) -> Result<Option<RealArith>, EvalError>;

   /// One operation of `float_arith`: exact when both operands are exact (a power with a
   /// non-integral rational exponent excepted), else in doubles. `whole` names the text in errors.
   fn real_op(&self, l: RealArith, op: &str, r: RealArith, whole: &str) -> Result<RealArith, EvalError>;
   ```

   `eval_to_f64(s)`: `eval_expr(s)`. If that is Ok with an Integer, Rational or Float, the result is
   `exact_to_f64` of it (a value outside the double range is its ValueError). Otherwise `float_arith(s)`:
   Some(Real(f)) gives f, Some(Exact(v)) gives `exact_to_f64(v)`, Err(e) gives e, and None gives the error of
   `eval_expr` (`2pi`: SyntaxError) or, if `eval_expr` gave a value that is not a number (an Expr in a
   variable), HEAD's TypeError `"Cannot convert to number"`. HEAD already called `eval_expr` on every text that
   its double path could not read; the texts that path read (literals, `pi`, `+ - * /`) have no side effect
   in `eval_expr`, so the exact-first order adds none. `float_arith` itself does not call `eval_expr`.

   `float_arith` steps, all on `top_level_chars(s)`: (1) the `+ -` chain, else (2) the `* /` chain
   (`binary_chain`, folded from the left with `real_op`); (3) a run of leading signs, read in one loop (an
   exact value by `negate`, a real one by `-f`); (4) the leftmost `**` or `^`; (5) `group_inner`; (6) atoms: a
   name bound in `self.variables` to an Integer or Rational gives Exact, to a finite Float gives Real, to
   anything else None; else `pi`/`PI`, `e`/`E`, `tau`/`TAU` give the constants (`std::f64::consts`, each the
   correctly rounded double, as MAGMA's `Pi(R)` at 53 bits); else a run of ASCII digits gives an Exact Integer;
   else a decimal literal (first character a digit or `.`, every character in `0-9 . e E + -`, then
   `str::parse::<f64>`, so `inf`, `nan` and `2pi` are not literals) gives Real; else None. A decimal literal
   that is not finite is the ValueError of m3, one whose mantissa has a nonzero digit but parses to 0 or a
   subnormal is a ValueError (underflow). Names are whole atoms, so `pi2` is the variable pi2 and `2pi` is not
   a number.

   `real_op(l, op, r)`: both Exact gives `apply_binary_op` (so `ZeroDivisionError`, the exponent limit and the
   Q arithmetic of item 9 apply), with a Rational exponent of denominator 1 read as its Integer; Exact ^ Exact
   with a non-integral rational exponent, and every pair with a Real operand, convert the operands to doubles
   (an Exact one by `exact_to_f64`; a nonzero exact value that rounds to 0 or a subnormal is a ValueError,
   underflow) and compute `+ - * /` or `powf`. Division by 0.0 is a ZeroDivisionError. A result that is not
   finite is a ValueError `"{s} is not a finite real number in double precision ({x:e} {op} {y:e} gives {z})"`. A
   product of two nonzero operands, a quotient of a nonzero dividend, or a power of a nonzero base that gives
   0 or a subnormal is a ValueError `"{s} underflows double precision (...)"`. A sum or a difference is never
   refused for underflow: every double is an integer multiple of 2^-1074, so the exact sum of two doubles is
   one too; when its magnitude is below 2^-1022 it is k·2^-1074 with |k| < 2^52, which is a double (0 or a
   subnormal), and the correctly rounded IEEE sum returns it exactly. So `pi - pi` is 0 (HEAD's value), and a
   zero from cancellation is exact. The errors replace the silent results of revision 1: `pi * 10^-400 *
   10^400` (0), `1e308 * 10 * pi` (inf), `pi / (pi - pi)` (HEAD: inf).

   Semantics and accuracy. For a text without a real operand the result is the correctly rounded double of
   its exact value (the conversion of S11 is correctly rounded). For a text with one it is MAGMA's value in
   `RealField(53 : Bits := true)` (ch. 25 §25.1.1, §25.1.2) whenever no intermediate leaves the double range,
   up to powers: IEEE `+ - * /` round to nearest even exactly as MPFR at 53 bits, the exact operands are
   coerced with one correct rounding, and `powf` (the platform `pow`) is not guaranteed to be correctly
   rounded. Measured against gp at 100 digits on 7000 random bound texts (`$S/r2/f64cmp/`; every row
   classified by `cmpref.py`):

   | Texts | HEAD | Revision 2 |
   |---|---|---|
   | 5084 without a real operand | 3340 values (252 not correctly rounded, double rounding), 1744 errors | all 5084 correctly rounded |
   | 942 with pi, HEAD a value | 756 correctly rounded, 184 not, 2 ±inf | 912 HEAD's double bit for bit (all 756 that HEAD rounded correctly among them); 27 correctly rounded where HEAD was off; 1 off where HEAD was off too; 2 ZeroDivisionErrors where HEAD returned ±inf (a division by `pi - pi`) |
   | 1910 with pi, revision 2 a value | | 1518 correctly rounded, 384 within 3 ulps, 8 further off by cancellation after a real operand (for example `43/47/8^22 + pi - pi` gives 0, while `pi - pi + 43/47/8^22` gives 1.2399e-20, the exact value: the small term is lost when it meets pi first, as at 53 bits in MAGMA; `$S/r2/f64cmp/ex.out`) |

   So in the sample no value that HEAD rounded correctly changes (3088 of 3088 without a real operand, 756 of
   756 with pi), and the review's rows reproduce (`$S/r2/proto/drive` on
   `$W/r2_g.txt`, `$W/r2_d.txt`): `find_root(2*w == 1, 10^20 + 1 - 10^20, 3)`, its `a + 1 - a` form and
   `2^53 + 1 - 2^53` are "No root found" again, and both absorbed-overflow forms give the root 2.

Correctness: the top-level characters are exactly the characters outside literals and brackets, so a split
never cuts a literal or a nested call. For an operator string without literals, splitting at the lowest
precedence operator, rightmost for left-associative and leftmost for right-associative operators, yields the
parse tree of the grammar above (standard; e.g. Aho, Lam, Sethi, Ullman, *Compilers*, 2nd ed., §4.1.2 and
§2.2.6). Cutting at every operator of the lowest level and folding from the left gives the same tree as
cutting at the rightmost one and recursing on the left part. A sign is binary exactly when it follows a
complete operand, which is the token rule of all three reference systems. Byte offsets come from `char_indices`, so every slice is on a char boundary. A run of
signs is equivalent to the nested unary rule (`--x` = `-(-x)`), and a run of enclosing parentheses to the
nested atom rule. `float_arith` applies the same tree, in Q until a real operand meets an exact one and in
doubles from there (item 10); Z and the arms of item 9 apply it in Q,
where `*` and `/` are associative in the sense `(a * b) / c = a * (b / c)`, so the left-associative split
gives HEAD's value for every product HEAD computed. Item 9's arms are the inclusion Z in Q followed by
field operations in Q, so they agree with the Integer arms wherever both apply (`a / b` with b | a gives the
rational a/b, displayed as the integer).

### 2.3 Callers and behaviour changes

Every caller of the splitters is inside `repl.rs`. Changes:

- Inputs with operators, commas or brackets inside a string literal parse (they were SyntaxErrors or split
  wrongly). `chart(E3, "x,y,z")` reaches `chart` with one string (S14 then applies).
- `-2**2`, `-2^2`: 4 becomes -4. `x + y - z` displays `((x + y) - z)` instead of `(x + (y - z))`;
  `a * b / c` is evaluated as `(a * b) / c`. With item 9 every value HEAD computed stays the same; this was
  measured, not assumed: the review's inputs `$V/t1.txt` to `$V/t9.txt` (221 commands) on HEAD and on the
  revision prototype (`$S/r1/head_t*.out`, `$S/r1/r1_t*.out`) differ only where HEAD gave an error (now a
  value, or another error) or a wrong value (`-2^2`, the `find_root` bounds); `$V/t3.txt` (the rational
  products `1/2 * 1/3`, `2/3 * 3/4`, `1/2 + 1/3 * 3/4`, `r * 3 / 2`, `e1 * 2/3`) is identical to HEAD. The only
  other values with a `* Rational` arm, CliffordElem (`a1 * 1/2`, `a1 * 2 / 3 + 1/2`), give the same elements
  (`$S/r1/t_cl.txt`). Values whose type has no `/ Integer` arm (Matrix, Polynomial, IntegerMod, Expr, Float,
  Complex) were TypeErrors for `v * 1/2` at HEAD (`* Rational`) and stay TypeErrors (`/ Integer`), with another
  message.
- Integer / Integer (`a / b` with integer variables, `10^20/3`, `2^3/4`), Integer op Rational, Rational op
  Integer and Rational ** Integer now evaluate (HEAD: TypeError). `8 / 0`, `(1/2) / 0` and `h / z` with
  `z = 0/5` are ZeroDivisionErrors (HEAD: TypeError, or the panic at `rational.rs:354`).
- `-x`, `2 * -3`, `3 - -2`, `(2+3)*4`, `2 ** -1` and non-ASCII expressions now evaluate; `π*2` no longer
  panics.
- Ranges and bounds (`eval_to_f64`, all 55 callers): `-a+b` is `(-a)+b`, `-2^2` is -4, a name is a whole
  identifier (`pi2`), a variable named `pi`, `e` or `tau` always wins over the constant (HEAD: a bare name was
  always the constant; inside a longer text `pi` was the constant and `e`, `tau` the variable), `2*e` and `pi*n` (n numeric) are read (HEAD:
  TypeError), and a bound that is not a finite number is an error instead of inf (`1/0`, `0/0`:
  ZeroDivisionError; `1e400`, `10^400`: ValueError). `2pi` is a SyntaxError (HEAD: 23.14159...). Revision 2:
  a text without a real operand is the correctly rounded double of its exact value, so texts that HEAD read in
  doubles (`1/3 + 1/7`, `100000000000000000000 + 3 - 100000000000000000000`) can change: by HEAD's rounding error
  in 252 of 3340 random texts, and from 0 to 3 in the second example. A text with pi, e, tau, a decimal literal
  or a Float variable is computed in the real field of doubles, with exact subresults kept exact up to the
  first real operand. It gives HEAD's double wherever HEAD's was correctly rounded (912 of 942 random texts
  bit for bit; the other 30 are texts HEAD rounded wrongly or evaluated to ±inf; table in 2.2 item 10). Overflow, underflow and division by a zero after a real
  operand are errors (HEAD: inf, 0 or a silent wrong bound).
- `parse_symbolic_arg` (`:4924-4935`) tries `eval_expr` on text that ends with `)`; grouping makes `(x + 1)`
  evaluate there (an `Expr`, the same expression the parser fallback built).
- No existing test changes (revision-2 prototype: the 63 existing tests pass with S9 and revisions 1 and 2 applied).

### 2.4 Tests (new; values: `gp -q ref.gp < /dev/null` in `$S/ref.out`, Python for `3 - -2`)

`test_parser_signs_and_precedence`:

| Command | Expected | Reference | HEAD |
|---|---|---|---|
| `2 * -3` | `-6` | gp `2*-3` | F |
| `3 - -2` | `5` | Python `3 - -2` | F |
| `--5` | `5` (unary minus twice) | Python `--5` | F (SyntaxError `Cannot parse: -`) |
| `-2 ** 2`, `-2^2` | `-4` | gp `-2^2` | F (4) |
| `2 ** -1`, `2 ^ -1` | `1/2` | gp `2^-1` | F |
| `-2**-2` | `-1/4` | gp `-2^-2` | F |
| `0 ** -1` | ZeroDivisionError | gp `0^-1` error | F |
| `2^3^2`, `2**3**2` | `512` | gp `2^3^2` | P |
| `1 - 2 - 3`, `2 - 3 + 4` | `-4`, `3` | gp | P |
| `(2+3)*4`, `2*(3-1)^2`, `-(2+3)` | `20`, `8`, `-5` | gp | F |
| `x = 5` then `-x`, `-x^2`, `-x * 2` | `-5`, `-25`, `-10` | | F |
| `y = var("y")`, `z = var("z")`, then `y + z - y` | `((y + z) - y)` | left associativity | F (`(y + (z - y))`) |
| `2 * 1e-3` | TypeError (no Float arithmetic, section 9), not a SyntaxError; assert `e.name == "TypeError"` | | F (SyntaxError `1e`) |
| `x1e = 7` then `x1e-3` | `4` (a name, not an exponent) | | P |
| `-"a"` | TypeError `bad operand type for unary -: String` | | F |
| `-+-5`, `"-"*100001 + "1"` (built with `format!`) | `5`, `-1` (one step for a run of signs, in a test thread) | Python | F (SyntaxError) |

`test_rational_products_kept` (revision 1, B1; regression rows that pass on HEAD and must keep passing):

| Command | Expected | gp | HEAD |
|---|---|---|---|
| `1/2 * 1/3` | `1/6` | `1/2*1/3` = 1/6 | P |
| `2/3 * 3/4` | `1/2` | `2/3*3/4` = 1/2 | P |
| `1/2 + 1/3 * 3/4` | `3/4` | `1/2+1/3*3/4` = 3/4 | P |
| `r = Rational(1, 3)` then `r * 3 / 2` | `1/2` | `r=1/3; r*3/2` = 1/2 | P |
| `e1 = 1/2` then `e1 * 2/3` | `1/3` | | P |

`test_integer_rational_arithmetic` (revision 1, B1; gp values from `$S/r1/r1.gp`):

| Command | Expected | gp | HEAD |
|---|---|---|---|
| `2**-2 * 4` | `1` | `2^-2*4` = 1 | F (SyntaxError) |
| `8 / 4 * 2`, `8 * 2 / 4`, `2 * 3 / 4` | `4`, `4`, `3/2` | `8/4*2`, `8*2/4`, `2*3/4` | F (TypeError) |
| `a = 8`, `b = 4`, `a / b` | `2` | | F |
| `h = 1/2`, `h + 1`, `1 - h`, `h ^ 2`, `(1/2)^2`, `h ** 0` | `3/2`, `1/2`, `1/4`, `1/4`, `1` | | F |
| `(2/3)^-2`, `(-2/3)^-3` | `9/4`, `-27/8` | `(2/3)^-2`, `(-2/3)^-3` | F |
| `10^20/3`, `10^400/10^399` | `100000000000000000000/3`, `10` | same | F |
| `1/0`, `a / 0`, `z0 = 0/5` then `h / z0`, `1/2 / 0`, `(0/1)^-1` | ZeroDivisionError each (`z0` itself is `0`) | `1/0`: error `e_INV` | F (TypeError; `h / z0` **panics**) |

`test_range_bounds_float_arith` (revision 1, B2; `w = var("w")`; `|v - expected| < 1e-9` on the Float text):

| Command | Expected | gp | HEAD |
|---|---|---|---|
| `find_root(w^2 == 4, -3+2, 3)` | `2` | `solve(w=-1,3,w^2-4)` | F (No root found) |
| `find_root(w^2 == 9, -pi-1, 0)` | `-3` | `solve(w=-Pi-1,0,w^2-9)` | F |
| `find_root(w^2 == 9, -1-3, 0)` | `-3` | `solve(w=-4,0,w^2-9)` | F |
| `find_root(w^2 == 4, 0, 2*-1+5)` | `2` | `2*-1+5` = 3 | F (SyntaxError) |
| `find_root(w^2 == 2, -2^2, 0)` | `-sqrt(2)` | `solve(w=-4,0,w^2-2)` | F (+1.414...) |
| `find_root(w^2 == 2, 0, 2*e)` | `sqrt(2)` | `solve(w=0,2*exp(1),w^2-2)` | F (TypeError) |
| `pi2 = 5`, `find_root(w^2 == 16, 0, pi2)` | `4` | `solve(w=0,5,w^2-16)` | F (No root found) |
| `e = 5`, `find_root(w^2 == 16, 0, e)` | `4` | | F |
| `find_root(w^2 == 4, 0, 10^400/10^399)` | `2` | | F (TypeError) |
| `find_root(w^2 == 4, 0, 1e400)`, `(.., 10^400)` | ValueError each | | F (`inf`, `inf`) |
| `find_root(w^2 == 4, 0, 1/0)`, `(.., 0/0)` | ZeroDivisionError each (revision 2: exact first) | gp `1/0`: error | F (`inf`, `0`) |
| `find_root(w^2 == 2, 0, 2pi)` | SyntaxError | | F (1.414...) |

`test_range_bounds_exact_first` (revision 2, B1; `w = var("w")`; gp values from `$S/r2/r2.gp` and the review):

| Command | Expected | Reference | HEAD |
|---|---|---|---|
| `find_root(2*w == 1, 10^20 + 1 - 10^20, 3)` | ValueError `No root found ...` (the bound is 1; the root 0.5 lies outside [1, 3]) | gp `10^20+1-10^20` = 1 | P (regression row; revision 1 returned 0.4999...) |
| `a = 10^20`, `find_root(2*w == 1, a + 1 - a, 3)` | same | | P (regression row) |
| `find_root(2*w == 1, 2^53 + 1 - 2^53, 3)` | same | | P (regression row) |
| `find_root(w^2 == 4, 1, 3 * 10^300 / 10^400 * 10^100)` | `2` (`\|v - 2\| < 1e-9`) | exact value 3; `solve(w=1,3,w^2-4)` | F (TypeError; revision 1: No root found) |
| `find_root(w^2 == 4, 1, 10^-200 * 10^-200 * 10^300 * 10^100 * 3)` | `2` | exact value 3 | F (SyntaxError; revision 1: No root found) |

`test_float_arith_real_field` (revision 2, B1; unit test on `ctx.eval_to_f64`, exact `f64` equality where the
operation order fixes the double):

| Text | Expected | Reference | HEAD |
|---|---|---|---|
| `10^400/10^399 * pi`, `(10^20 + 1 - 10^20) * pi` | `10.0 * PI`, `PI` | gp `10^400/10^399*Pi`, `(10^20+1-10^20)*Pi` | F (TypeError, SyntaxError) |
| `pi/2`, `-pi`, `pi - pi` | `FRAC_PI_2`, `-PI`, `0.0` | | P (regression rows: HEAD's text substitution read them in doubles) |
| `pi + 12157665459056928801 * (pi - pi) - pi / (pi)` | `PI - 1.0` (a zero operand is not an underflow) | gp 2.14159... | P (regression row) |
| `3^40 * pi` | `12157665459056928801.0 * PI` (3^40 rounded once) | gp `3^40*Pi` = 38194432490975628713.208... | F (TypeError) |
| `2^(1/2)` | within 1e-15 of `sqrt(2)` | | F (SyntaxError) |
| `10^20 + pi - 10^20` | `0.0` (the documented 53-bit semantics) | MAGMA at 53 bits; gp at 38 digits gives pi | F (TypeError) |
| `pi * 10^-400 * 10^400`, `pi * 1e-200 * 1e-200 * 1e300`, `pi^-1000`, `1e308 * 10 * pi`, `pi * 2^1100` | ValueError each (underflow, underflow, underflow, overflow, out of range) | | F (errors or `inf`) |
| `pi / (pi - pi)`, `pi + 1/0` | ZeroDivisionError each | | F (`inf`; TypeError) |
| `(-pi)^(1/2)` | ValueError (NaN) | | F |
| `"-"*100001 + "pi"` (built with `format!`) | `-PI` (one step for a run of signs) | | F |
| `0.1 + 0.2`; `x = 1.5`, `x * pi` | `0.1 + 0.2` (HEAD's double); `1.5 * PI` | Python `0.1 + 0.2` | P; F |

`test_negative_exponent_message` (revision 2, m4): `2 ** -10001` gives `ValueError: **: |exponent| must be an
integer in 0..=10000 (REPL limit), got 10001`; `(1/2) ^ -10001` the same with `^:`; `2 ** 10001` keeps
`**: exponent must be an integer in 0..=10000 (REPL limit), got 10001`. F for the first two (HEAD:
SyntaxError, and the Rational arm does not exist).

`test_value_type_name_by_variant` (revision 2, m7): `-"a"` gives `TypeError: bad operand type for unary -:
String`, `-[1, 2]` gives `... List`. F (HEAD: SyntaxError).

`test_sign_and_group_runs` (revision 1; all in the default 2 MB test thread): `"-"*100001 + "1"` gives `-1`;
`"("*100000 + "7" + ")"*100000` gives `7` (HEAD: SyntaxError; the unrevised prototype overflowed the stack at
about 1000); `"1" + "+1"*100000` gives `100001` (F: HEAD's debug build aborts at 732 terms); `"2" + "*2"*1000 + "/2"`
gives the decimal digits of 2^1000; `w - w + w - w` displays `(((w - w) + w) - w)`; `-+-5` gives `5`;
`((a)+(b))` gives `(a + b)`, `(1)+(2)` gives `3`, `(")")` gives `")"`, `((1)+(2))*((3))` gives `9`; `((1, 2))` and
`(())` are SyntaxErrors.

`test_parser_string_literals`:

| Command | Expected | HEAD |
|---|---|---|
| `"a-b"`, `"a+b"`, `"x^2"` | `"a-b"`, `"a+b"`, `"x^2"` | F |
| `s = "x,y,z"` then `s` | `"x,y,z"` | F |
| `'x,y'` | `"x,y"` | F |
| `[ "a,b", "c" ]` | `["a,b", "c"]` | F |
| `print("x,y", 1)` | text `"x,y" 1` | F |
| `"ab" * -5` | `""` (String times a negative count, as `"ab" * c` with c < 0 at `:9852`) | F |
| `"é" + "a"`, `"é"*2` | `"éa"`, `"éé"` | F |
| `π*2` | Ok, `(2 * π)`; the test asserts `is_ok()` (HEAD panics) | F (panic) |
| `Polynomial([gcd(4, 6), 1])` | `x + 2` | F |
| `x = var("x")` then `find_root(x^2 == 2, 0, 2)` | `1.414213562...` (`|v - sqrt(2)| < 1e-9`) | P |

Unit test `test_lex_state`: `top_level_chars("f(\"a,b\", [1, 2]), c")` has exactly one `,` (the last);
`is_string_literal("\"a\" + \"b\"")` false, `("\"a\\\"b\"")` true, `("'x'")` true, `("\"x")` false;
`is_binary_sign("1e-3", 2)` false, `("x1e-3", 3)` true, `("2 * -3", 4)` false, `("a - b", 2)` true.

---

## 3. S10: pochhammer aliases the falling factorial

### 3.1 Defect

`repl.rs:2348`: `"falling_factorial" | "pochhammer" => {`. So `pochhammer(3, 4)` = 3·2·1·0 = 0. The ch. 144
digest lists `NumberOfPermutations(n, k)` = falling factorial (`Chapter_144_Enumerative_Combinatorics.md:37`)
and no Pochhammer intrinsic; the ch. 49 digest defines "(a)_n = a(a+1)···(a+n−1) is the Pochhammer symbol"
(`Chapter_49_Power_Laurent_and_Puiseux_Series.md:292`, `HypergeometricSeries`). Sage's `rising_factorial`
docstring calls the rising factorial the Pochhammer symbol, and the workspace's own
`rustmath-calculus/src/pochhammer.rs:1-16` defines (x)_n = x(x+1)...(x+n-1). Only the REPL disagrees.

### 3.2 Fix

Move the name: `"falling_factorial" =>` and `"rising_factorial" | "pochhammer" =>`. Inside the rising arm use
`func_name` instead of the literal `"rising_factorial"` in every `int_arg` and error message, so
`pochhammer(...)` errors name `pochhammer`. Whatever C1 does to `rising_factorial` (wider n, Integer-valued)
applies to `pochhammer` automatically. Help text (`:10057-10058`): add
`pochhammer(x, n)       - Pochhammer symbol (x)_n = x(x+1)...(x+n-1), the rising factorial (MAGMA ch. 49)`.

Behaviour change: `pochhammer(n, k)` for k >= 1 changes from the falling to the rising factorial. No test
uses `pochhammer` (`grep -n pochhammer repl.rs`: only `:2348`).

### 3.3 Tests

`test_pochhammer_is_rising`: `pochhammer(3, 4)` = `360` (HEAD `0`, F), `pochhammer(5, 3)` = `210` (HEAD `60`,
F), `pochhammer(10, 5)` = `240240` (F), `pochhammer(0, 0)` = `1`, `falling_factorial(3, 4)` = `0` (P),
`pochhammer(3, 4)` equals `rising_factorial(3, 4)`. gp: `prod(i=0,3,3+i)` = 360, `prod(i=0,2,5+i)` = 210,
`prod(i=0,4,10+i)` = 240240, `prod(i=0,3,3-i)` = 0 (`$S/ref.out`).

---

## 4. S11: to_f64 of large rationals gives NaN

### 4.1 Defect

`Rational` keeps its numerator and denominator as `Integer`; `Integer::to_f64` (num-bigint 0.4.8) returns
`Some(±inf)` above 2^1024 (never `None`). The quotient of two such values is inf/inf = NaN, inf/den = inf for
a finite rational such as 10^400/(3·10^399), and num/inf = 0 for 1/10^310 (a representable subnormal).
Below 2^1024 the two roundings and the division can be off by 2 ulp (`$S/s11_check.rs`: 33772 of 100000 random
120-bit rationals differ from the correctly rounded value, at most 2 ulp; all with num or den >= 2^53).

Sites: `repl.rs:3018`, `:3024` (`integrate_num` bounds, `unwrap()`), `:5179-5182` (`eval_to_f64`, `unwrap_or`),
`:8877-8880` (`list_to_floats`), `:8897-8900` (`list_to_complexes`), `:8925-8928` (`value_to_f64`), and the
library function `rustmath-rationals/src/rational.rs:156-160` (`Rational::to_f64`, also behind
`NumericConversion::to_f64` at `:552-554`), which the REPL uses at `:10288` (`try_eval_to_f64`). The integer
branches' `ok_or_else("Integer too large ...")` (`:8874`, `:8893`, `:8922`, `:5176`) never fire, because
`to_f64` is never `None`; huge integers become inf silently.

Reproduction (HEAD debug driver):

```
big = 10^400;  r2 = Rational(big + 1, big)
mean([r2])                                    -> NaN        (gp: (10^400+1)/10^400 * 1. = 1.000...; nearest double 1.0)
variance([r2, 1])                             -> NaN        (0.0)
fft([r2])                                     -> [NaN + 0i] ([1 + 0i])
mean([Rational(1, 10^310)])                   -> 0          (1e-310, a subnormal double)
mean([2^1100])                                -> inf        (not a double)
x = var("x"); find_root(x - 1, x, 0, Rational(2*big + 1, big))  -> 0   (the root is 1; b became NaN)
median([r2, 1, 2])                            -> panic at rustmath-stats/src/statistics.rs:60:44 (revision 2, m6)
```

The last row kills the kernel: `median` sorts with `partial_cmp(..).unwrap()`, and the NaN of `r2` has no
order. With the fixed conversion the list is [1.0, 1.0, 2.0] and the median is 1 (`$S/r2/drive_head` and
`$S/r2/proto/drive` on `$W/r2_median.txt`). After J1a the REPL hands no NaN to rustmath-stats
(`list_to_floats` converts through `exact_to_f64`, which refuses a non-finite Float). Other callers of
rustmath-stats can still make `median` panic on a NaN (`statistics.rs:60`, and the same `unwrap` at `:129`,
`:136`); that is listed in section 9.

### 4.2 Fix

**Library (root cause), `rustmath-rationals/src/rational.rs:155-160`:**

```rust
/// The double nearest to this rational (round to nearest, ties to even), from num-rational's
/// `Ratio<BigInt>::to_f64` (0.4.2: an exact quotient of 55 or 56 bits with remainder, so the
/// result is correctly rounded also when the numerator or the denominator is beyond the double
/// range). A value whose magnitude rounds above f64::MAX gives ±inf, one that rounds below the
/// least subnormal gives ±0.0. Never NaN, never `None` (the denominator is positive).
pub fn to_f64(&self) -> Option<f64> {
    use num_traits::ToPrimitive;
    num_rational::BigRational::new_raw(self.numerator.as_bigint().clone(), self.denominator.as_bigint().clone())
        .to_f64()
}
```

num-rational is already a dependency of rustmath-rationals (`Cargo.toml:13`, default features include
`num-bigint`). Its algorithm (`num-rational-0.4.2/src/lib.rs:1557-1647`): fast path when both parts fit in
±2^53 (one IEEE division, bit-identical to HEAD there); otherwise shift so the quotient has 55 or 56 bits,
round to nearest even with the remainder as sticky bit, scale with an exact `ldexp`. Checked: 3000 random
rationals with up to 1500-bit parts, negative, overflowing and subnormal results, against Python `n/d` (exact
integer true division, correctly rounded): 0 mismatches (`$S/s11_dump.rs`). Table of the edge cases
(`$S/s11_check.rs`): (10^400+1)/10^400 -> 1.0, 10^400/10^399 -> 10.0, 1/10^310 -> 1e-310, 1/10^400 -> 0.0,
-(10^500)/(3·10^499) -> -3.3333333333333335, 2^1100/3 -> inf (correct: above f64::MAX).

This is a second crate in this unit. It is the root cause (the REPL's own division at six sites is a copy of
it), it is five lines, and it consumes an existing numeric crate as CLAUDE.md asks. Unit I3 also edits
rustmath-rationals, in other functions: `rational.rs:530` (`NumericConversion::to_i64` becomes
`i64::try_from(&self.numerator).ok()`, with its test `numeric_conversion_to_i64_is_checked`) and
`sage_wrapper.rs:244-249`, `:423`, `:457` (binomial and `nth_root`). I3 lands first; J1a changes only
`Rational::to_f64` (`:155-160`) and adds its test to the same test module, so the merge is textual only
(`NumericConversion::to_f64` at `:552-554` keeps forwarding to `Rational::to_f64`).

**REPL, one helper** (new, next to `show_int`):

```rust
/// The double nearest to an exact number (S11), for `what` in error messages. Integer and
/// Rational round once to nearest (`Integer::to_f64`, `Rational::to_f64`, both correctly
/// rounded). A value whose magnitude rounds above f64::MAX is a ValueError
/// "{what}: {v} is outside the range of a double", never inf or NaN. A nonzero value below the
/// least subnormal (about 4.9e-324) rounds to 0.0 (or -0.0): that is the IEEE result, the same as
/// Python's float(Fraction(1, 10**400)), and not an error. A finite Float passes through; a
/// non-finite Float is the same ValueError. Other values: `None`, so each caller keeps its own
/// TypeError.
fn exact_to_f64(v: &RustMathValue, what: &str) -> Option<Result<f64, EvalError>>
```

`v` in the message: `show_int` for an Integer, `"a rational with a {b}-bit numerator"` for a Rational with more
than 128 bits, else its display. Sites rewritten to use it: `integrate_num` bounds (`:3015-3026`, `what` =
"integrate_num: lower bound"/"upper bound"), `list_to_floats`, `list_to_complexes`, `value_to_f64` (each: `what` =
the old message prefix), and `eval_to_f64` (`:5174-5185`), which S9 item 10 rewrites around `float_arith` and
`exact_to_f64` (`what` = the text). `Complex(re, im)` (`:983-993`) already refuses non-finite integers;
unchanged. `try_eval_to_f64` (`:10287-10288`) keeps `Option` (an inf there is filtered by the plotting code)
and becomes correct through the library fix.

**Float literals (revision 1, m3).** `eval_expr` reads a float literal with `str::parse::<f64>` (`:730-735`),
which gives inf for `1e400` and `-inf` for `-1e400`. A literal whose value is not finite becomes a ValueError
`"the float literal {text} is outside the range of a double"`. Underflow (`1e-400`) stays 0.0, the IEEE
result. With `exact_to_f64` refusing non-finite Floats and `eval_to_f64` checking its result, no REPL
conversion to a double hands inf or NaN to a numeric routine. (A numeric routine can still overflow on
finite input: `mean([1e308, 1e308])` gives `inf` at HEAD and after J1a, because the sum overflows inside
`rustmath-stats`. That is a separate defect, listed in section 9; a Float inf produced that way is refused
where it is converted again.)

### 4.3 Callers and behaviour changes

- `Rational::to_f64` / `NumericConversion::to_f64` for `Rational`: identical values when |num| and den are at
  most 2^53; otherwise the correctly rounded value (up to 2 ulp different from HEAD) and no NaN. Callers: 62
  files in the workspace call some `.to_f64()` on a type that may be `Rational` (`grep -rln Rational | xargs grep
  -ln '\.to_f64()'`); `SageRational::to_f64` (`sage_wrapper.rs:592`) forwards. The only callers of the old
  num/den pattern itself are elsewhere (`rustmath-schemes/src/elliptic_curves/heegner.rs`,
  `rustmath-rings/src/qqbar.rs`, `rustmath-numberfields/src/kummer.rs`, `rustmath-combinatorics/src/derangements.rs`,
  2 each at most); they are not changed here (section 9).
- REPL: big rationals give their value; integers and rationals outside the double range give a ValueError
  in `mean`, `median`, `mode`, `variance`, `std`, `correlation`, `covariance`, `fft`, `ifft`, `find_root`,
  `integrate_num`, the plot ranges and the point constructors (all callers of the four helpers, `:2849-3644`,
  `:5090-7454`), where HEAD returned inf or NaN.

### 4.4 Tests

rustmath-rationals, new `test_to_f64_correctly_rounded` in `rational.rs`: the seven edge cases above (expected
values from Python `n/d`, e.g. `float(Fraction(10**400+1, 10**400))` = 1.0, `1/10**310` = 1e-310 (prints
`1e-310`), `(-(10**500))/(3*10**499)` = -3.3333333333333335, `2**1100/3`: OverflowError, so `f64::INFINITY`);
plus `1/3` bit-identical to `1.0/3.0` (fast path). F for the first five on HEAD.

REPL `test_large_rational_to_float`:

| Command | Expected | HEAD |
|---|---|---|
| `big = 10^400`, `r2 = Rational(big + 1, big)`, `mean([r2])` | Float, `text.parse::<f64>() == 1.0` | F (NaN) |
| `variance([r2, 1])` | `0` | F (NaN) |
| `fft([r2])` | `[1 + 0i]` | F |
| `mean([Rational(1, 10^310)])` | parses to `1e-310` exactly | F (0) |
| `mean([2^1100])` | ValueError containing `is outside the range of a double` | F (inf) |
| `x = var("x")`, `find_root(x - 1, x, 0, Rational(2*big + 1, big))` | `|v - 1| < 1e-9` | F (0) |
| `mean([Rational(1, 10^400)])` (revision 1, m4) | parses to `0.0` (Python `float(Fraction(1, 10**400))` = 0.0) | P (by accident: 1/inf) |
| `1e400`, `-1e400` (revision 1, m3) | ValueError `the float literal 1e400 is outside the range of a double` | F (`inf`, `-inf`) |
| `1e-400` | `0` | P |
| `median([r2, 1, 2])` (revision 2, m6; the prototype has it as `test_median_large_rational`) | parses to `1.0` | F (panic, `statistics.rs:60`) |

(`Rational(a, b)` and `big + 1` parse at HEAD; the tests need no S9 feature. The prototype test
`test_float_literal_range_and_underflow` in `$S/r1/patch_r1_tests.py` runs the last three rows and
`mean([Rational(1, 10^310)])`.)

---

## 5. S14: chart never compares the number of names with dim(M)

### 5.1 Defect

`repl.rs:4579-4595`: `let dim = coord_names.len(); Chart::new("user_chart", dim, coord_names)`. The manifold `m`
is bound and unused, so `Chart::new`'s own count check (`rustmath-manifolds/src/chart.rs:53-58`) compares the
names with themselves. `chart(E3, "t")` is a one-coordinate chart on R^3 (`$S/s9_head.txt`: `C1 = chart(E3, "t")`
-> `Chart(t)`). Empty names (`"x,,z"`) and repeated names (`"x,x,y"`) are accepted; `coordinate_symbols()`
(`chart.rs:133-137`) then makes two coordinates the same symbol. A chart also carries no manifold, and
`coordinate_form(M, C, i)`, `exterior_derivative(w, C)`, `wedge(a, b, C)` never compare C with M; after MF1 the
library's `exterior_derivative` refuses a chart of the wrong dimension (`MF1-manifolds.md` §4.2), the other two
do not.

Sage (`sage/manifolds/chart.py`, `Chart.__init__`, fetched to `$S/sage_chart.py:392-396`):
`raise ValueError("the list of coordinates must contain {} elements".format(self._manifold.dim()))`.

### 5.2 Fix (prototype `$S/proto/patch_s14.py`)

```rust
/// The coordinate names of `chart(M, names)` (S14): `names` separated by commas, exactly dim(M)
/// of them (Sage `Chart`: "the list of coordinates must contain dim elements"), each a valid
/// identifier, pairwise distinct (each names a coordinate symbol). For dim(M) = 0 the string is
/// empty.
fn chart_coordinate_names(names: &str, dim: usize) -> Result<Vec<String>, EvalError>;

/// The REPL check that `chart` has one coordinate per dimension of the manifold it is used on
/// (S14): a chart is made for one dimension, but REPL values can be combined freely.
fn check_chart(cmd: &str, chart: &Chart, dim: usize) -> Result<(), EvalError>;
```

Messages: `"chart: the manifold has dimension 3, so the chart needs 3 coordinate names, got 1"`,
`"chart: coordinate name \"\" is not an identifier"`, `"chart: coordinate name x appears twice"`,
`"{cmd}: the chart has 3 coordinates, but the manifold has dimension 4"`.

- `chart`: `let names = chart_coordinate_names(coords, m.dimension())?;
  Chart::new("user_chart", m.dimension(), names)` (the library check now agrees; its error, unreachable, is
  mapped to ValueError `"chart: {e}"`). The names arrive as one string literal with commas, which S9 makes
  possible.
- `coordinate_form`: `check_chart("coordinate_form", chart, m.dimension())?` after the `MANIFOLD_COORDS` check
  (`:4626-4630`) and before `index_arg` (so the size limit still answers first for a huge manifold).
- `exterior_derivative` and `wedge` (3-argument form): `check_chart(cmd, chart, form.manifold().dimension())?` as
  the first statement of the typed match arm (before `missing_components_error`); for `wedge` the form is the
  first one (a second form on another manifold is the library's error, as now). Both checks are in the
  revision-2 prototype (`$S/r2/patch_r2.py`; review 2, m1: the earlier prototype had only the
  `coordinate_form` one), so every row of 5.4 is verified.

### 5.3 Callers and behaviour changes

`chart` with a wrong number of names, a non-identifier or a repeated name is now a ValueError; a mismatched
chart in the three arms is a ValueError. Existing tests that build one-name charts must change (they were
written around S9 and S14, see the comments at `:12210-12211`). Against the code MF1 leaves (MF1 §4.3 already
replaced the R^3 block of `test_form_component_limits` by an R^1 block because the library refuses the
one-name chart on R^3):

| Test (HEAD line) | Old | New |
|---|---|---|
| `test_int_arg_ranges_named` `:12210-12220` | `C = chart(E3, "t")`; `Cb = chart(Mbig, "t")` | `C = chart(E3, "x,y,z")`; comment "A chart has one coordinate name per dimension (S14); the names are one string literal with commas (S9)"; drop `Cb`, use `coordinate_form(Mbig, C, 0)` with the comment "The size check comes before the chart check, so the R^3 chart shows the limit" |
| `test_form_component_limits` `:12352` | `C = chart(M, "t")`, M of dimension 100000 | `ctx.eval(&format!("C = chart(M, \"{}\")", chart_names(100000)))` with a test helper `fn chart_names(n: usize) -> String` returning `"x0,x1,...,x{n-1}"` (a 589 KB command; the whole test runs in 0.46 s in debug, measured) |
| same test, MF1's R^1 block | `C3 = chart(E3, "t")`, R^1 detour | restore the R^3 block with `C3 = chart(E3, "x,y,z")`: `da = exterior_derivative(a, C3)`, `exterior_derivative(da, C3)` Ok, `wedge(a, b, C3)` Ok, `wedge(da, b, C3)` Ok (MF1 §4.4 item 5 asks for exactly this) |
| same test `:12378` | `Cm = chart(M3, "t")` | `Cm = chart(M3, "a,b,c")` |
| `test_form_without_components` `:12402`, `:12416` | `CN = chart(N, "t")` (N of dimension 2000), `CS = chart(S, "t")` | `chart_names(2000)`, `CS = chart(S, "a,b,c")` |

With these, the prototype passes all 63 existing tests at HEAD (`$S/proto/patch_s14_tests.py`; MF1's R^1 detour
does not exist at HEAD, so the prototype keeps the original R^3 block, which is what J1a restores).

### 5.4 Tests

`test_chart_matches_dimension`:

| Command | Expected | HEAD |
|---|---|---|
| `E3 = EuclideanSpace(3)`, `chart(E3, "x,y,z")` | `Chart(x, y, z)` | F (SyntaxError, S9) |
| `chart(E3, "t")` | ValueError `chart: the manifold has dimension 3, so the chart needs 3 coordinate names, got 1` | F (Ok) |
| `chart(E3, "x,y,z,w")` | ValueError `... got 4` | F |
| `chart(E3, "x,,z")` | ValueError `chart: coordinate name "" is not an identifier` | F |
| `chart(E3, "x,x,y")` | ValueError `chart: coordinate name x appears twice` | F |
| `chart(E3, "x,1y,z")` | ValueError `... "1y" is not an identifier` | F |
| `M0 = Manifold(0, "P")`, `chart(M0, "")` | Ok, `Chart()` | P (HEAD displays the same, but builds a chart with one empty name) |
| `chart(M0, "t")` | ValueError `... dimension 0, so the chart needs 0 coordinate names, got 1` | F (Ok) |
| unit: `chart_coordinate_names("", 0)`, `("", 1)`, `(" x , y ", 2)` | `Ok([])`, Err, `Ok(["x", "y"])` | F (no such function) |
| `E4 = EuclideanSpace(4)`, `C3 = chart(E3, "x,y,z")`, `coordinate_form(E4, C3, 0)` | ValueError `coordinate_form: the chart has 3 coordinates, but the manifold has dimension 4` | F |
| `a = coordinate_form(E3, C3, 0)`, `C4 = chart(E4, "p,q,r,s")`, `exterior_derivative(a, C4)`, `wedge(a, a, C4)` | ValueError `exterior_derivative: the chart has 4 coordinates, but the manifold has dimension 3`, same for `wedge`; with `C3` both are Ok (prototype test `test_chart_checked_in_form_arms`) | F |
| `chart(E3, "r,θ,φ")` | Ok (`is_valid_identifier` accepts alphabetic Unicode) | F |

(`bigchart.txt` for the timing note: `python3 -c "n=100000; print('M = Manifold(%d, \"M\")'%n); print('C = chart(M, \"'+','.join('x%d'%i for i in range(n))+'\")')"`.)

---

## 6. S15: no size limit on graph constructors, prime_pi and primes

Policy (`repl_limits` doc `:10446-10462`, X2): a refused call would take more than about 10 s in a release
build or a few hundred MB (about 400 MB); an admitted call needs at most a few hundred MB; no new refusal of a
call that HEAD computed correctly within those bounds.

### 6.1 Graph constructors (`:3166-3201`)

**Defect.** `Graph(n)`, `complete_graph(n)`, `cycle_graph(n)`, `path_graph(n)`, `star_graph(n)`,
`wheel_graph(n)` take any usize from `eval_as_usize` (`:8908`). `rustmath-graphs` stores
`Vec<HashSet<usize>>` (`graph.rs:7-20`) and adds every edge to two sets (`:52-58`), so memory is the bound.
`complete_graph(10^6)` asks for about 5·10^11 edges (an allocation failure aborts the kernel);
`star_graph(usize::MAX)` overflows `n + 1` (`generators/basic.rs:663`, a panic in debug).

**Measurement.** `$S/graph_cost.rs` (release mirror; time is never the bound: complete_graph(4000), 8·10^6
edges, 1.1 s) and the real REPL (`$S/drive_dbg`, VmHWM). An assignment holds three copies (the value, the
variable and `last_result`, `:648-650`), a bare expression two. A per-copy model from the hash table layout
(std `HashSet` = hashbrown: an empty set is 48 bytes in the `Vec` and allocates nothing; a set of d entries has
4 buckets for d <= 3, 8 for d <= 7, else the power of two >= 8d/7, 9 bytes per bucket plus control bytes and
the allocator header):

```rust
/// Bytes of a std HashSet<usize> table holding d entries (0 for d = 0: no allocation). None when
/// the value does not fit in u128 (revision 1: checked arithmetic, never a panic).
fn hash_set_bytes(d: u128) -> Option<u128> {
    match d {
        0 => Some(0),
        1..=3 => Some(64),
        4..=7 => Some(96),
        _ => 8u128.checked_mul(d)?.div_ceil(7).checked_next_power_of_two()?.checked_mul(9)?.checked_add(32),
    }
}
/// The memory of one rustmath-graphs `Graph` with `v` vertices whose degrees come in
/// (count, degree) groups: 48 bytes per vertex plus each vertex's table. None when the model
/// overflows u128, which is far above every limit.
fn graph_bytes(v: u128, groups: &[(u128, u128)]) -> Option<u128> {
    let mut b = v.checked_mul(48)?;
    for &(c, d) in groups {
        b = b.checked_add(c.checked_mul(hash_set_bytes(d)?)?)?;
    }
    Some(b)
}
```

| Command | v | degree groups | model x3 | REPL peak, `G = ...` |
|---|---|---|---|---|
| `path_graph(10^6)` | n | (2, 1), (n-2, 2) | 336 MB | 334 MB |
| `complete_graph(3000)` | n | (n, n-1) | 332.5 MB | 331 MB |
| `Graph(2·10^6)` | n | none | 288 MB | 287 MB |
| `star_graph(10^6)` | n+1 | (n, 1), (1, n) | 392.6 MB | 389 MB |
| `complete_graph(3000)` without assignment | | | 221.6 MB (x2) | 223 MB |

Also `cycle_graph(n)`: (n, 2); `wheel_graph(n)`: v = n+1, (n, 3), (1, n). Edge counts for the message:
n(n-1)/2, n, n-1, n, 2n, 0.

**Fix.** `repl_limits::GRAPH_BYTES: u128 = 133_000_000` ("one copy; an assignment holds three, about 400 MB").
Each arm computes `graph_bytes` after its existing domain check (`n < 3` for cycle and wheel), with v and
the degrees as u128 (so `n + 1` for star and wheel cannot overflow), and admits the call only for
`Some(b)` with b <= GRAPH_BYTES. `None` means the model overflowed u128 and is refused like any other
value above the limit. (Revision 1, B3: the first version computed in plain u128 and claimed no overflow;
for `complete_graph` the model is about v · 9 · 2^(log2(8v/7)+1), so 2^132 for v = 2^64 - 1, and the debug
build of the review panicked with "attempt to multiply with overflow" at n = 2^62. The checked version,
`$S/r1/gb/gb2.rs` in a debug build: `complete_graph(2^32)` 3.3·10^20 bytes, refused; `complete_graph(2^62)`
and `complete_graph(2^64-1)` None, refused; no panic.) Refusal message:
`"{cmd}: a graph with {v} vertices and {e} edges needs about {bytes} bytes per copy (REPL limit 133000000, a memory bound; an assignment holds three copies)"`,
and for None the same text with `needs more than 2^128 bytes per copy`. The edge count e is computed in
u128 (n(n-1) < 2^128 for every usize n, so n(n-1)/2 does not overflow).

Admitted maxima (`$S/r1/gb/gb2.rs`): `Graph` 2,770,833; `path_graph`/`cycle_graph` 1,187,500; `star_graph` and
`wheel_graph` 1,018,978; `complete_graph` 3585 (3585 · (48 + 9·4096 + 32) = 132.4 MB; at 3586 the tables double to 8192
buckets, 264.7 MB per copy). Every refused graph needs more than about 400 MB at the peak of an assignment,
every admitted one at most about 400 MB, so no call that stayed within the policy is refused. A bare
expression holds two copies, so a refused bare expression needs more than about 266 MB (`Graph(3000000)`:
288 MB at HEAD, refused; still "a few hundred MB", the edge of the policy). The `repl_limits` doc says so
(revision 2, m5): "GRAPH_BYTES bounds one copy: an assignment holds three (about 400 MB at the limit), a bare
expression two, so the smallest refused bare expression needs about 266 MB".

Correctness of the model is empirical (std does not promise its table layout); the doc comment says so and
names the measurement. If a future std changes the layout, the refusal point moves, nothing breaks.

### 6.2 prime_pi (`:2703-2715`)

**Defect.** `prime_pi(n)` has no bound. HEAD tests every candidate (`prime.rs:356`): 2.26 s at 10^5 and 23 s at
10^6 in release (`$S/head_primes.rs`). After I1 the callee is a segmented sieve (I1 §2.5): linear time, small
memory; I1's prototype in release (`$S/sieve_i1.rs`): `prime_pi(10^9)` 1.65 s, `prime_pi(4·10^9)` 7.03 s (values
50847534 and 189961812 match gp `primepi`). Above 2^64 I1's `prime_pi` panics by design.

**Fix.** `repl_limits::PRIME_PI_X: u64 = 6_000_000_000` (10.5 s by the measured slope 1.75 s per 10^9; gp
`primepi(6*10^9)` = 279545368). x < 2 still gives 0 (any negative x). Above the limit: ValueError
`"prime_pi: x = {x} is above the REPL limit 6000000000 (the sieve takes about 10 s there)"` (`show_int` for x).
The 2^64 panic is unreachable. HEAD's 10 s reach was about 4·10^5, so nothing it computed is refused.

### 6.3 primes (`:2718-2733`)

**Defect.** `primes(a, b)` (alias `prime_range`) lists every prime in [a, b) with no bound: HEAD release 0.3 ms
per prime near 10^6 (22.7 s for [0, 10^6)), and each listed prime costs about 375 bytes at the peak (a list of
10^6 Integers with an assignment and the display text: 375 MB, `$S/list_cost.rs`). After I1, `prime_range` sieves
below 2^64 and runs `is_prime` (a proof, ECPP above psi_12) on each candidate above 2^64; `is_prime` **panics**
when ECPP is unresolved (I1 §0: about 1.5 % of 768-bit and 3 % of 1024-bit primes).

**Fix.** With lo = max(a, 2) and y = b - lo (empty list when y <= 0, as now):

1. Output bound (memory). The number of primes in [lo, lo + y) is at most
   `count_bound(y) = min(ceil(y/2) + 1, floor(2y / ln y))` for y >= 2 (the second term: H. L. Montgomery,
   R. C. Vaughan, *The large sieve*, Mathematika 20 (1973) 119-134, Corollary 2: π(x + y) − π(x) < 2y/log y for
   y > 1), and at most 1 for y = 1. Refuse when it exceeds `PRIMES_COUNT = 1_000_000`:
   `"primes: [a, b) may contain up to {c} primes (Montgomery-Vaughan bound), more than the REPL limit 1000000 (a memory bound)"`.
   Domain (revision 1, m1): y is an `Integer` of any size, so the bound is never computed in floats on y
   itself:

   ```rust
   /// An upper bound on the number of primes in [x, x + y) for every x >= 0 (S15; Montgomery and
   /// Vaughan 1973, Corollary 2). None when y > PRIMES_SPAN = 8_000_000 (an exact Integer
   /// comparison): there the bound already exceeds PRIMES_COUNT, since count_bound(8·10^6) =
   /// 1006608 and the bound increases with y. Some(c) for y <= 8·10^6, computed in f64 on y as
   /// u64 (exact below 2^53, no overflow, no NaN). count_bound(y) = 0 for y <= 0.
   fn count_bound(y: &Integer) -> Option<u64>
   ```

   `None` is refused with `"primes: [a, b) spans {y} integers and may contain more than 1000000 primes (Montgomery-Vaughan bound), more than the REPL limit 1000000 (a memory bound)"`
   (`show_int` for y). This admits spans up to 7,943,969 exactly (the largest y with count_bound(y) <= 10^6;
   `count_bound(7.6·10^6)` = 959374; the true count near 0 is about half of that). This bound is only a
   memory bound: the time model of step 3 uses the probable primes actually found (revision 2, B2). It also
   bounds the list of probable primes that step 3 keeps between its passes. HEAD's
   10 s reach is a span of about 4·10^5 (22.7 s for [0, 10^6), 2.3 s for [10^6, 1.1·10^6), release), so
   nothing it computed is refused.
2. The part below 2^64 goes to `prime_range` (I1's sieve; at most 7.95·10^6 candidates, under 1 s).
3. The part [lo', b) with lo' = max(lo, 2^64) is enumerated by the REPL itself, in two passes (revision 2,
   B2). n_odd is the number of odd c in [lo', b), and B = bits(b - 1), the size of every candidate.
   1. Prior check: `n_odd · t_test(B) > PRIMES_SECONDS` (10.0) is refused before any test, with
      `"primes: testing the {n_odd} odd candidates of [{lo'}, {b}) needs about {t:.0} s (REPL limit 10 s)"`
      (`show_int` for the bounds).
   2. Pass 1: each odd c goes through I1's `prime::is_probable_prime(&c)`. False is a proof that c is not
      prime (I1 §2.1: "`false` is always a proof"), so c is dropped. True pushes c onto a list of probable
      primes. After each push, with k the length of the list, the call is refused as soon as
      `n_odd · t_test(B) + k_big · t_proof(B) > PRIMES_SECONDS`, with
      `"primes: [a, b) has at least {k} BPSW probable primes above 2^64; proving them prime needs about {t:.0} s or more (REPL limit 10 s)"`.
      Here k_big counts the probable primes >= psi_12. Below psi_12, `is_probable_prime` runs the twelve
      strong tests to the bases 2..37, which decide there (Sorenson and Webster, Math. Comp. 86 (2017) 985-1003;
      I1 §2.1), so those need no proof. Refusing at the first k that exceeds the
      limit, instead of after the whole pass, keeps a refused call short (`primes(2^1024, 2^1024 + 20000)`
      stops at the third probable prime, about 1060 candidates). k only grows, so the result is the same
      as a check after the pass, and it depends only on the input, not on the machine.
   3. Pass 2: each listed c >= psi_12 goes through `prime::try_is_prime(&c)` (I1). Ok(true) keeps c, Ok(false)
      drops it (a BPSW pseudoprime, never seen), and Err(e) is a ValueError
      `"primes: {c} is a BPSW probable prime that could not be proven prime ({e})"`. This replaces I1's
      panicking `prime_range` path for the REPL. `try_is_prime` repeats BPSW before ECPP; that costs less
      than 1 % of a proof.

   The answer does not depend on the models: a BPSW failure is a proof of compositeness, every listed prime
   is proven, and a refusal answers nothing. Memory: the list holds at most `count_bound(y)` <= 10^6 entries
   (step 1); above psi_12, at most 10 / t_proof(B) entries (1250 at 100 bits).

   Time models (new `repl_limits` constants, release, this container; provisional until I1 is implemented,
   re-timed in 8.2):

   ```rust
   /// One strong (Miller-Rabin) test by bit size: num-bigint modpow, `$S/r2/tcost.rs`.
   pub const PRIMES_STRONG_TEST_SECONDS: [(u64, f64); 9] = [(64, 8.4e-6), (128, 1.4e-5), (256, 3.4e-5),
       (512, 1.5e-4), (1024, 7.3e-4), (2048, 4.7e-3), (4096, 4.5e-2), (8192, 2.6e-1), (16384, 2.2)];
   /// One ECPP proof with its verification: I1's prototype (`ecpp_rt_rel one`), 100 bits measured
   /// here (27 primes above 10^30, mean 7.9 ms), 128 to 3072 bits the means of I1 §2.3.7.
   pub const PRIMES_PROOF_SECONDS: [(u64, f64); 9] = [(100, 0.008), (128, 0.024), (256, 0.070),
       (512, 0.41), (768, 1.7), (1024, 3.3), (1536, 20.6), (2048, 75.8), (3072, 957.0)];
   ```

   `table_seconds(table, B)` interpolates log-linearly between two points, gives the first value below the
   table, and extends the last segment's power law above it. Then
   `t_test(B) = 2e-6 + T_strong(B) · (0.25 + 40 / B)` and `t_proof(B) = T_proof(B)`.
   The factor is an upper bound on I1's work per odd candidate: trial division by the 167 odd primes below
   1000 (1.2 µs at 64 bits to 0.25 ms at 16384 bits, `tcost.rs`; covered by the 2e-6 term below 1000 bits
   and by T_strong above), one strong test for the 16.3 % that pass it (Mertens: the product of (1 - 1/p) over
   the odd primes below 1000), and for the primes, a fraction of about 2.885 / B of the odd candidates, at
   most 12 more strong tests (34.6 / B). This is at most 0.163 + 34.6 / B, so below 0.25 + 40 / B.

   No new refusal by the prior check (step 3.1). HEAD's `prime_range` above 2^64 runs `is_prime`, which is
   `is_prime_miller_rabin(n, 20)`: every odd candidate gets at least the base-2 strong test, and a prime gets
   240 (the 12 bases 20 times, `prime.rs:55-58`). Measured HEAD release cost per odd candidate
   (`$S/r2/t_head_primes.txt`), against `t_test`:

   | Height | HEAD per odd candidate | t_test | HEAD / t_test |
   |---|---|---|---|
   | 2^64 | 61 µs | 9.4 µs | 6.5 |
   | 2^100 | 74 µs | 9.6 µs | 7.7 |
   | 2^128 | 70 µs | 9.9 µs | 7.1 |
   | 2^256 | 150 µs | 15.8 µs | 9.5 |
   | 2^512 | 295 µs | 51 µs | 5.8 |
   | 2^1024 | 1.31 ms | 0.21 ms | 6.2 |
   | 2^2048 | 4.75 ms | 1.27 ms | 3.7 |
   | 2^4096 | 37 ms | 11.7 ms | 3.2 |

   So a call refused by the prior check took HEAD more than 32 s. Pass 2 refuses only answers with probable
   primes >= psi_12 whose proofs exceed the limit. HEAD returned those lists unproven, and after I1 each
   needs an ECPP proof. That is a consequence of I1's proof semantics: those answers were not decided at HEAD.
   An answer without such primes is never refused by pass 2, since k_big = 0 there. So the review's rows are
   answered again:

   | Call (`$S/r2/est2.py`, counts from gp) | Answer (gp) | HEAD release (`$S/r2/t_head_rows_rel.out`) | Model |
   |---|---|---|---|
   | `primes(10^400, 10^400 + 10)` | `[]` (next prime at +69) | 0.007 s | 0.002 s, admitted |
   | `primes(2^1000, 2^1000 + 200)` | `[]` (next prime at +297) | 0.062 s | 0.02 s, admitted |
   | `primes(2^512, 2^512 + 70)` | `[]` (next prime at +75) | 0.005 s | 0.002 s, admitted |
   | `primes(2^512, 2^512 + 100)` | 1 prime | 0.036 s (unproven) | 0.42 s, admitted |
   | `primes(10^30, 10^30 + 20000)` | 258 primes | 0.64 s (unproven) | 2.2 s, admitted |
   | `primes(2^512, 2^512 + 2000)` | 7 primes | 0.34 s (unproven) | 2.9 s, admitted |
   | `primes(2^2048, 2^2048 + 400)`, `primes(2^4096, 2^4096 + 100)` | `[]`, `[]` | 0.95 s, 1.87 s | 0.25 s, 0.58 s, admitted |
   | `primes(2^1000, 2^1000 + 10^6)` | | about 5 min (0.62 ms per candidate, extrapolated) | 101 s: refused by the prior check |
   | `primes(2^1024, 2^1024 + 20000)` | 29 primes | about 10 to 13 s (1.03 to 1.31 ms per candidate in two runs, extrapolated; unproven) | refused at k = 3 (2.1 s + 3 · 3.3 s) |
   | `primes(10^30, 10^30 + 10^5)` | 1389 primes | 3.5 s (unproven) | refused at k = 1191 (1389 proofs: about 11 s) |

   Largest spans the prior check admits: about 2.1·10^6 at 2^64 and at 10^30 (step 1 bounds the span by
   7.94·10^6 in any case), 3.9·10^5 at 2^512, 4.8·10^4 at 10^400, 1.6·10^4 at 2^2048, 1700 at 2^4096. Pass 2
   admits up to 1250 probable primes at 100 bits, 24 at 512 bits, 3 at 1024 bits; a single probable prime of
   1309 bits or more is refused (one proof is modelled above 10 s).

Help text `:10089`: `primes(a, b)         - List of primes in [a, b) (half-open, as Sage prime_range)`;
HEAD's help says `[a, b]` but `prime_range` is half-open (`prime_range(11, 29)` = [11, 13, 17, 19, 23], gp
`primes([11, 28])`).

### 6.4 Tests (S15)

`test_graph_limits`: `G = path_graph(100000)` Ok (`Graph with 100000 vertices and 99999 edges`; 11.2 MB per
copy by the model, so the debug test stays small while other tests run in parallel; revision 1, m8). The
boundaries are checked on the function, without building the graphs: `graph_bytes(1187500, &[(2, 1), (1187498, 2)])
= Some(133000000)` admitted and `path_graph(1187501)` refused by the REPL; `graph_bytes(3585, &[(3585, 3584)])
= Some(132444240)` admitted, `complete_graph(3586)` refused with the message above; `Graph(2770834)` refused,
`graph_bytes(2770833, &[]) = Some(132999984)`; `star_graph(18446744073709551615)` refused (HEAD: overflow panic
in debug, F); `complete_graph(1000000)` refused (HEAD: abort, F); `complete_graph(4611686018427387904)` (2^62)
and `complete_graph(18446744073709551615)` refused with `needs more than 2^128 bytes` (revision 1, B3; the unrevised model
panicked in debug); `graph_bytes(1 << 62, &[(1 << 62, (1 << 62) - 1)]) == None`; `complete_graph(5)` still
`Graph with 5 vertices and 10 edges` (gp `binomial(5,2)` = 10). Unit test `test_hash_set_bytes_model`: the four
measured rows of 6.1 within 3 % of `3 · graph_bytes` (constants in the test with their source).

`test_prime_counting_limits`: `prime_pi(10^6)` = `78498` (gp `primepi(10^6)`; HEAD 23 s release, so the test also
checks I1's speed); `prime_pi(-5)` = `0`; `prime_pi(6000000001)` ValueError (F: HEAD runs for hours);
`primes(11, 29)` = `[11, 13, 17, 19, 23]` (gp `primes([11,28])`); `primes(2^64, 2^64 + 100)` =
`[18446744073709551629, 18446744073709551653, 18446744073709551667, 18446744073709551697, 18446744073709551709]`
(gp `primes([2^64, 2^64+99])`); `primes(10^30, 10^30 + 300)` =
`[1000000000000000000000000000057, 1000000000000000000000000000099, 1000000000000000000000000000211, 1000000000000000000000000000231, 1000000000000000000000000000271]`
(gp `primes([10^30, 10^30+299])`; five 100-bit ECPP proofs, about 0.3 s in debug); `primes(0, 10^8)` refused
(span above 8·10^6; count bound 10857362, F); `primes(0, 10^400)` refused with `more than 1000000 primes`
(revision 1, m1; HEAD runs forever, F). Revision 2 (B2), the two passes of 6.3 step 3:
`primes(10^400, 10^400 + 10)` = `[]` and `primes(2^1000, 2^1000 + 200)` = `[]` (gp `nextprime(10^400) - 10^400`
= 69, `nextprime(2^1000) - 2^1000` = 297; regression rows, P on HEAD; the first revision refused both);
`primes(2^1000, 2^1000 + 10^6)` refused by the prior check, message starting `primes: testing the 500000 odd
candidates` (F: HEAD runs about 5 min); `primes(2^1024, 2^1024 + 20000)` refused by pass 2, a ValueError
containing `BPSW probable primes above 2^64` and `REPL limit 10 s` (gp: 29 primes, the first at +643, +1081,
+2113; the refusal comes at the third, so the debug test scans about 1060 candidates and proves nothing; the
test does not pin k, which depends on the provisional `t_proof`; F: HEAD answers unproven). Unit test
`test_primes_time_model`: `table_seconds` returns the table values at its points (`PRIMES_PROOF_SECONDS` at 512
bits: 0.41), interpolates between them (between 1024 and 1536 bits it is increasing and within the two
values), gives 0.008 below 100 bits, and grows above 3072 bits; and for each row of the HEAD table of 6.3
(height 2^h, HEAD's measured seconds per odd candidate, constants in the test with their source),
`3 · t_test(h + 1)` is at most HEAD's time, which is the margin behind "no new refusal". Unit test of `count_bound`: `count_bound(1) = Some(1)`, `(2) = Some(2)`,
`(10^6) = Some(144764)` (floor(2·10^6/ln 10^6); gp `floor(2*10^6/log(10^6))`), `(7943969) = Some(1000000)`,
`(7943970) = Some(1000001)`, `(8000000) = Some(1006608)`, `(8000001) = None`, `(10^400) = None`, `(-3) = Some(0)`,
and `count_bound(y) >= π(y)` for y = 10, 100, 10^4 (gp `primepi`: 4, 25, 1229; the bounds are 6, 43, 2171).

---

## 7. sqrt_mod for a composite n factors by trial division, up to three times

### 7.1 Defect

`sqrt_mod_report` (`:10949-11015`): the composite branch calls `IntegerMod::sqrt`, `num_sqrts` and `all_sqrts`
(`:10998-11014`); each factors n with `rustmath_integers::prime::factor` (`rustmath-finitefields/src/integer_mod.rs:347`,
`:396`, `:433`), HEAD's trial division by every odd number (`rustmath-integers/src/prime.rs:415-456`), which runs
until the candidate passes the square root of what is left. So a composite with a large prime cofactor never
finishes. Reproduced: `sqrt_mod(4, 2^127+1)` and `sqrt_mod(4, 1000000000000000005490000000000000001989)`
(= p20·q21, gp `nextprime(10^19)*nextprime(10^20)`) were killed after 20 s; the doc comment (`:10943-10947`)
estimates 7.5e18 divisions for 2^127+1. The help line (`:10043-10045`) says so. After I1, the prime branch calls a
proving `is_prime` (ECPP, slow above 1024 bits, a panic when unresolved), and I1 adds a Jacobi shortcut to the
composite branch; neither changes the factoring.

HEAD's reach, for the "no new refusal" rule: one trial-division step costs about 55 + bits/11 ns in release
(`$S/td_cost.rs`: 61 ns at 64 bits, 103 ns at 512, 834 ns at 8192, 9.4 µs at 10^5 bits), so in 10 s HEAD gets to
odd candidates up to `T_old(bits) = 2·10^10 / (55 + bits/11)` (3.3·10^8 for small n, 2.1·10^6 at 10^5 bits). HEAD
finishes exactly when every prime factor but the largest is at most that, and the largest is at most its square
(the loop also runs to the square root of the last cofactor).

### 7.2 Fix

MAGMA ch. 18 §18.10.1 (`Factorization` with `TrialDivisionLimit` 10000, `PollardRhoLimit` 8191, `MPQSLimit`,
`Proof`, and its third return value; digest `Chapter_18_Ring_of_Integers.md:334`), §18.12.1 (`Modsqrt`, `:417`),
ch. 19 §19.5.6 (`Sqrt`, `AllSqrts` in Z/mZ with the optional `Factorization` parameter "to speed up computation
when m is large", `Chapter_19_Integer_Residue_Class_Rings.md:188-189`; worked example H19E4). The REPL does what
that parameter allows: it factors m once and hands the factorization to the per-prime-power solvers.
Algorithms: per prime power Tonelli-Shanks and Hensel lifting as now (Cohen GTM 138, Alg. 1.5.1 and §1.6), CRT
combination (Cohen §1.3.3), factoring by I2 (Brent's rho [Bre80], SIQS) and proofs by I1 (BPSW, ECPP [AM93]).

**Library addition (rustmath-finitefields, `integer_mod.rs` next to `all_square_roots_mod_prime_power`):**

```rust
/// The number of square roots of `a` modulo p^k (closed form, no listing): p^⌊k/2⌋ for
/// a ≡ 0; for a ≡ p^(2e)·u, p ∤ u, j = k − 2e: 2·p^e (odd p) or 2^e, 2·2^e, 4·2^e (p = 2, j = 1, 2,
/// >= 3) when u is a square mod p^j; else 0. 1 for k = 0. Same contract and panics as
/// [`square_root_mod_prime_power`].
pub fn num_square_roots_mod_prime_power(a: &Integer, p: &Integer, k: usize) -> Integer {
    let k = checked_prime_power_args("num_square_roots_mod_prime_power", p, k);
    if k == 0 { return Integer::one(); }
    prime_power_sqrts(a, p, k).count()
}
```

Re-exported from `lib.rs:44-47`. This is the count `IntegerMod::num_sqrts` already uses (`:433`); without it the
REPL would duplicate the closed form. One line of logic in F1's crate (F1 is implemented earlier and does not
touch these functions).

**REPL.** Constants in `repl_limits`:

```rust
/// sqrt_mod's factorization (MAGMA Factorization parameters, ch. 18 §18.10.1).
pub const SQRT_MOD_TRIAL: u64 = 10_000;          // phase 1 (MAGMA TrialDivisionLimit default)
pub const SQRT_MOD_RHO_STEPS: u64 = 8_191;       // phase 2 (MAGMA PollardRhoLimit default)
pub const SQRT_MOD_I2_MAX_BITS: u64 = 4_096;     // phase 2 only on cofactors this small: its BPSW
                                                 // and rho cost grow as bits^3 and bits^2
pub const SQRT_MOD_QS_DIGITS: u32 = 60;          // SIQS: 2.2 s release at 60 digits (I2 §1.3)
pub const SQRT_MOD_TRIAL_WORK: u64 = 20_000_000_000; // phase 3 bound T(bits) = WORK / (55 + bits/11)
pub const SQRT_MOD_PROOF_BITS: u64 = 1_024;      // ECPP proofs up to this size (I1: <= 5.4 s release)
```

and an ordinary private struct `SqrtModLimits` with the same six fields. Production code uses the constant
`SQRT_MOD_LIMITS` built from the constants (through `sqrt_mod_report_with(.., &SQRT_MOD_LIMITS)`, below); the
unit tests build smaller limits.

```rust
/// One coprime factor of n for sqrt_mod.
enum SqrtModPiece {
    Prime { p: Integer, e: u32 },                    // p proven prime
    Probable { q: Integer, e: u32, why: String },    // BPSW probable prime, not proven (why)
    Composite { c: Integer, e: u32, why: String },   // proven composite or untested, not split (why)
}

/// The factorization of |n| behind sqrt_mod, within `lim`.
fn sqrt_mod_factor(n: &Integer, lim: &SqrtModLimits) -> Vec<SqrtModPiece>
```

Algorithm of `sqrt_mod_factor` (m = |n| > 1):

1. **Phase 1.** Trial division of m by the primes below `lim.trial` (I1 `sieve::for_each_prime(2, lim.trial, ..)`,
   stopping when p^2 > m): strip each p^e, record `Prime`.
2. **Phase 2.** Cofactor m: if m = 1, done. If the phase 1 loop stopped at p^2 > m, or m < lim.trial^2, record
   `Prime(m)`: every prime below its square root was tried (lim.trial^2 = 10^8 < psi_12, so this is exact). Else if bits(m) <= `lim.i2_max_bits`: `factorization_with_limits(&m, &FactorizationLimits {
   trial_division_limit: 0, pollard_rho_limit: lim.rho_steps, mpqs_limit: Some(lim.qs_digits), proof: false })`;
   its `primes` are `Prime`, its `probable_primes` go to step 4, its `composites` to phase 3. Else (bits(m) >
   `i2_max_bits`): (m, 1) goes to phase 3 untested.
3. **Phase 3.** For each composite or untested (c, e): streamed trial division by the primes from `lim.trial`
   (`for_each_prime(lim.trial, lim.trial_work / 55 + 1, f)`), where `f` stops when p^2 > c_cur or
   p > T(bits(c_cur)) = `lim.trial_work / (55 + bits(c_cur)/11)`; the bound grows as c_cur shrinks, as HEAD's cost
   per step did. Found primes are `Prime` with exponent e·f. The cofactor c': 1, done; if the loop stopped at
   p^2 > c', `Prime(c', e)` (a proof by trial division; c' < 3.7·10^8 squared < psi_12); else if c' = c and
   c came from phase 2's composites, `Composite { why: "no prime factor up to {T}, not split by Pollard rho
   ({rho_steps} steps) or the quadratic sieve (up to {qs_digits} digits)" }` with no second I2 call (revision 1,
   m5: I2 already proved c composite and already ran rho and the sieve on it; a second call would repeat the
   BPSW test and, below 60 digits, up to 2.2 s of sieving); else if bits(c') <= `i2_max_bits` (c' is a proper
   cofactor, or c was untested in phase 2):
   `factorization_with_limits(&c', &{ trial 0, rho 0, mpqs Some(qs_digits), proof false })` (classifies c' by
   BPSW and sieves it if small enough), whose composites become
   `Composite { why: "no prime factor up to {T}, not split by the quadratic sieve" }`; else
   `Composite { why: "{bits} bits with no prime factor up to {T}; testing it exceeds the REPL limit (4096 bits)" }`.
4. **Proofs.** For each probable prime q: if bits(q) <= `lim.proof_bits`, `prime::try_is_prime(&q)` (I1; it caches a
   success, so the contract checks of the finitefields functions below cost nothing): Ok(true) -> `Prime`, Ok(false)
   -> `Composite { why: "a BPSW pseudoprime" }`, Err(e) -> `Probable { why: format!("ECPP unresolved: {e}") }`;
   else `Probable { why: "above the REPL proof limit of 1024 bits" }`.

`sqrt_mod_report(a, n)` (replaces `:10949-11015`; the output formats of HEAD stay):

1. n <= 0: error as now; n = 1: `"0; square roots of 0 mod 1: [0]"`. b = a mod n.
2. Odd part m of n (m = n / 2^v2): if m > 1 and `b.kronecker_symbol(&m) == -1`, no root (a proof: a square is a
   square mod m, and the Jacobi symbol of a unit square is 1).
3. pieces = `sqrt_mod_factor(n, &SQRT_MOD_LIMITS)`.
4. A piece with odd base q (any exponent e >= 1) and `(b mod q).kronecker_symbol(&q) == -1`: no root (proof: a
   root mod q^e is a root mod q, and the Jacobi symbol of a square mod q is 0 or 1; revision 1, m6 dropped
   the needless "odd e" condition). A
   `Prime { p, e }` with `square_root_mod_prime_power(&b, &p, e)` None: no root (proof). Output
   `"no square root: {b} is not a square mod {n}"`.
5. A `Composite`: ValueError
   `"sqrt_mod: n = {n} is not factored completely within the REPL limits: {c} ({why}); the square roots mod n need its factorization"`
   (`show_int` for n and c).
6. A `Probable { q, e }` with e >= 2: ValueError `"sqrt_mod: n has the factor {q}^{e}, and {q} is a BPSW probable
   prime that was not proven prime ({why}); the roots mod {q}^{e} need a proof"`. With e = 1: b mod q = 0 gives the
   roots [0]; else `Integer::try_sqrt_mod_prime(&(b mod q), &q)` (I1; no primality test, the root is squared back):
   Ok(Some(r)) gives [min(r, q-r), max(r, q-r)], Ok(None) no root (Jacobi -1, a proof), Err(_) proves q composite:
   ValueError `"sqrt_mod: {q} passed BPSW but Tonelli-Shanks failed, so it is composite; factor n to decide"`.
7. Count N = product of `num_square_roots_mod_prime_power(&b, &p, e)` over the primes and 1 or 2 per probable
   prime. r = CRT (`rustmath_integers::crt_two`) of the least root per piece, in [0, n) (the same r as
   `IntegerMod::sqrt`, by uniqueness of the CRT). If N <= 16: every root from
   `all_square_roots_mod_prime_power` per prime power (each list has at most 16 entries, since every count is
   >= 1), combined by CRT as in `IntegerMod::all_sqrts` (`integer_mod.rs:386-420`), sorted:
   `"{r}; square roots of {b} mod {n}: [..]"`; else `"{r}; {b} has {N} square roots mod {n} (not listed)"`.
8. If any `Probable` piece took part: append
   `" (unproven: {q} is a BPSW probable prime of {bits} bits that was not proven prime ({why}); the list is complete if it is prime)"`,
   one clause per such q. This keeps HEAD's caveat for a large prime n (HEAD: "n is only a probable prime").

`SQRT_MOD_PSI12` and HEAD's prime branch go (the factorization covers prime n: phase 1 then phase 2 classify n).
Doc comment rewritten with the phases, the costs and the references. Help line (`:10043-10045`):
`"... A composite n is factored by trial division, Pollard rho and the quadratic sieve within REPL limits; an unfactored part is reported, not guessed"`.

**Correctness.** x^2 ≡ b (mod n) iff x^2 ≡ b (mod q^e) for every piece, since the pieces are pairwise coprime
(I2's invariant; phase 3 splits a composite piece into divisors of it) and their product is |n| (each phase keeps
the product: stripped primes times the cofactor). So the roots mod n are the CRT combinations of the roots mod the
pieces, N is the product of the counts, and "no root" mod one piece is a proof for n. The per-prime-power results
are complete for a proven prime (the finitefields module docs); for a probable prime the result is complete if it
is prime, which the text says. A composite piece makes the answer undecidable here, and the command says that
instead of answering.

**No new refusal.** If HEAD finished in 10 s, every prime factor of n but the largest is at most T_old(bits) and the
largest is at most T_old^2. Phase 3's bound T(bits(c_cur)) uses the same cost model on a cofactor no larger than
HEAD's at the same point, so it reaches every such prime; the cofactor left is then below the square of the last
prime tried and is recorded as prime. A factor below 10^4 is found by phase 1 in any case. So every n that HEAD
could do in 10 s is factored completely.

**Cost** (release): phase 1 under 1 ms below 4096 bits, at most 1229 remainders on a huge n; phase 2 at most
8191 rho steps per composite (10 µs a step at 4096 bits: 0.08 s) and the sieve on pieces of at most 60 digits
(2.2 s at 60 digits, I2 §1.3), BPSW only on numbers of at most 4096 bits; phase 3 at most about
π(T)·c(bits) = 2·10^10 ns / ln T, 1.0 to 1.5 s for every size, constant memory (I1's segmented sieve); step 4 at most
one expensive proof in practice (the large cofactor; I1 release: 3.9 s mean, 5.4 s max at 1024 bits). Worst admitted
call about 10 s. HEAD was 1 to 3 times its trial division.

### 7.3 Callers and behaviour changes

Only the `sqrt_mod` arm (`:2124-2162`) calls `sqrt_mod_report`. `IntegerMod::sqrt`/`num_sqrts`/`all_sqrts`
are unchanged (other callers: finitefields and its tests). Changes: composites with large prime factors are
answered or refused with a reason (HEAD hung); a prime n between psi_12 and 2^1024 is proven (no caveat); above
1024 bits the caveat stays and no ECPP runs (after I1 the REPL would have tried one). Existing tests
`test_sqrt_mod_composite_moduli` (`:11656`), `test_sqrt_mod_*` (`:11621-11800`, as I1 leaves them) keep their texts:
the root r is the same CRT root and the lists are the same sorted lists.

### 7.4 Tests

`test_sqrt_mod_factored` (values: `$S/ref.out`, gp):

| Command | Expected | HEAD |
|---|---|---|
| `sqrt_mod(4, 2^127+1)` | `113427455640312821154458202477256070488; square roots of 4 mod 170141183460469231731687303715884105729: [2, 56713727820156410577229101238628035241, 113427455640312821154458202477256070488, 170141183460469231731687303715884105727]` (gp `factor(2^127+1)` = 3 · 56713727820156410577229101238628035243; roots by `chinese` over {1, 2} mod 3 and {2, p-2} mod p; r = `chinese(Mod(1,3), Mod(2,p))`) | F (hangs) |
| `sqrt_mod(4, 1000000000000000005490000000000000001989)` | `2; square roots of 4 mod 1000000000000000005490000000000000001989: [2, 208067940552016986281146496815286624620, 791932059447983019208853503184713377369, 1000000000000000005490000000000000001987]` (the 40-digit semiprime goes through SIQS: about 0.5 s in debug, I2) | F (hangs) |
| `sqrt_mod(3, 1000000000000000005490000000000000001989)` | `no square root: 3 is not a square mod 1000000000000000005490000000000000001989` (gp `kronecker(3, m)` = 1, so the Jacobi shortcut does not decide; `issquare(Mod(3, a))` = 0) | F (hangs) |
| `sqrt_mod(4, 2^1279-1)` | starts `2; square roots of 4 mod 1040793219...` with the list [2, 2^1279 - 3], ends with `(unproven: ... is a BPSW probable prime of 1279 bits that was not proven prime (above the REPL proof limit of 1024 bits); the list is complete if it is prime)` | P in substance (HEAD caveat text differs) |
| `sqrt_mod(4, 15)`, `sqrt_mod(1404, 2340)` | unchanged texts of `test_sqrt_mod_composite_moduli` | P |
| `sqrt_mod(4, 318665857834031151167461)` (psi_12, revision 1, m9) | `2; square roots of 4 mod 318665857834031151167461: [2, 3193322321766, 318665857830837828845695, 318665857834031151167459]` (gp `factor(318665857834031151167461)` = 399165290221 · 798330580441, both prime; roots by `chinese` over {2, p-2} and {2, q-2}, `$S/r1/psi.gp`; phase 2 splits psi_12 with I2's sieve, which takes composites below 25 digits too, I2 design m3) | F (HEAD: the two roots [2, psi_12 - 2] with the probable-prime caveat; I1 dropped the row because trial division would need about 2·10^11 divisions) |
| `sqrt_mod(73, 318665857834031151167461)` (m9) | `75322749390115309241067; square roots of 73 mod 318665857834031151167461: [75322749390115309241067, 116429215029920790230196, 202236642804110360937265, 243343108443915841926394]` (gp: `issquare(Mod(73,p))` = `issquare(Mod(73,q))` = 1; r = `chinese(Mod(79487054939,p), Mod(173837379523,q))`, the CRT of the least roots) | F (HEAD: ValueError "composite") |
| `sqrt_mod(41, 318665857834031151167461)` | `no square root: 41 is not a square mod 318665857834031151167461` (gp `kronecker(41, psi_12)` = -1) | P |

The report takes its limits as a parameter: `fn sqrt_mod_report_with(a, n, lim: &SqrtModLimits)`, and
`sqrt_mod_report(a, n)` is `sqrt_mod_report_with(a, n, &SQRT_MOD_LIMITS)`.

`test_sqrt_mod_factor_phases` (unit, small limits). n = 1009 · 1013 · (2^89 - 1) = 632659779567127515219534076208987
(gp `factor`), 109 bits; 2^89 - 1 = 618970019642690137449562111 is above psi_12, so it is a BPSW probable prime that
step 4 proves (I1: 12 ms in debug). Limits `SqrtModLimits { trial: 100, rho_steps: 0, i2_max_bits: 4096,
qs_digits: 0, trial_work: 2000 * 64, proof_bits: 1024 }`: phase 1 finds nothing below 100; phase 2 calls I2 with
neither rho nor sieve, so n is only classified (composite); phase 3 runs with T = 128000 / (55 + 109/11) = 2000 and
finds 1009 and 1013; the cofactor 2^89 - 1 is above 2000^2 and goes to I2 again (probable prime), and step 4 proves it.
Expected pieces: `Prime` 1009, 1013, 618970019642690137449562111, each with e = 1.

`sqrt_mod_report_with(9, n, same limits)`: N = 8, r = 3 (the least roots are 3 mod each prime), text
`3; square roots of 9 mod 632659779567127515219534076208987: [3, 89309327224184831862184968309860, 95033561965840430253318518712382, 184342889190025262115503487022239, 448316890377102253104030589186748, 537626217601287084966215557496605, 543350452342942683357349107899127, 632659779567127515219534076208984]`
(Python: CRT over the roots 3 and -3 mod each prime; each squares to 9 mod n).

With `trial_work: 1010 * 64` (T = 1010): phase 3 finds 1009, then stops at 1013 > T, and 1013 · (2^89 - 1) =
627016629898045109236406418443 is a proper cofactor, so I2 classifies it again: composite. Expected: ValueError
starting
`sqrt_mod: n = 632659779567127515219534076208987 is not factored completely within the REPL limits: 627016629898045109236406418443 (no prime factor up to 1010, not split by the quadratic sieve)`.

With `trial_work: 500 * 64` (T = 32000 / (55 + 9) = 500; revision 1, m5): phase 3 finds no prime factor of the
phase-2 composite n, so c' = c and no second I2 call is made. Expected: ValueError ending
`: 632659779567127515219534076208987 (no prime factor up to 500, not split by Pollard rho (0 steps) or the quadratic sieve (up to 0 digits)); the square roots mod n need its factorization`
(the `why` text tells the two paths apart).

`test_sqrt_mod_refusal_release` (`#[ignore]`, release only: phase 3 runs to T = 2.5·10^8): `sqrt_mod(4, p40 * q40)`
with p40 = `nextprime(10^39)`, q40 = `nextprime(2*10^39)` (gp): a ValueError naming the 80-digit cofactor, in under
5 s in release.

finitefields, `test_num_square_roots_mod_prime_power`: (0, 2, 64) -> 2^32 (p^⌊k/2⌋); (9, 2, 4) -> 4 (the roots
3, 5, 11, 13 of the existing doc example); (2, 2, 3) -> 0; with p = 1000003: (13·p^2, p, 3) -> 2p = 2000006 (gp
`issquare(Mod(13,1000003))` = 1; v = 2, e = 1, j = 1) and (2·p^2, p, 3) -> 0 (gp `issquare(Mod(2,1000003))` = 0);
(5, 7, 0) -> 1; and it equals `all_square_roots_mod_prime_power(a, p, k).len()` for every 0 <= a < 64,
p in {2, 3, 5, 7}, k in 1..=3.

---

## 8. Plan

### 8.1 Files, in implementation order

1. `rustmath-rationals/src/rational.rs`: `to_f64` via num-rational (section 4.2), doc; test
   `test_to_f64_correctly_rounded`. The ch. 20 digest (`Chapter_20_Rational_Field.md`) has no conversion to
   floating point (only `Qround`, the other direction), so the doc and the test cite num-rational 0.4.2
   (`ratio_to_f64`) and Python's exact `int / int` as the reference.
2. `rustmath-finitefields/src/integer_mod.rs`: `num_square_roots_mod_prime_power` and its test; `lib.rs` re-export.
3. `rustmath-jupyter/src/repl.rs`, in this order (each step keeps the crate's tests green):
   1. S11 (first, because the new `eval_to_f64` of step 2 uses its helper): `exact_to_f64` (calling the fixed
      `Rational::to_f64`; the prototype calls num-rational directly only because it cannot edit the library),
      the finite check of float literals in `eval_expr`, and the sites `integrate_num`, `list_to_floats`,
      `list_to_complexes`, `value_to_f64`; tests of 4.4.
   2. S9: `LexState`, `top_level_chars`, `is_string_literal`, `is_binary_sign`, `value_type_name`; the splitters;
      the function-call and string-literal checks; `ChainLevel`, `binary_chain`, `try_eval_binary_op` (folded
      chains, sign runs), `negate`; `group_inner` and grouping; the negative-exponent arm; `is_exact_zero` and the
      Integer/Rational arms (item 9); `RealArith`, `eval_to_f64`, `float_arith`, `real_op`, and the deletion of
      `eval_simple_arithmetic` (item 10). References, applied in this order to HEAD's text: `$S/proto/patch_s9.py`,
      `patch_s9b.py`, then `$S/r1/patch_r1.py`, `patch_r1b.py`, `patch_r1c.py`, `patch_r1d.py`, `patch_r1e.py`,
      `patch_r1_tests.py`, then `$S/r2/patch_r2.py` (`patch_r1.py` also carries the S11 helper and the
      float-literal check of step 1; take them from there; `patch_r2.py` replaces revision 1's `eval_to_f64` and
      `float_arith` and `value_type_name`, and changes the two `|exponent|` names). The implementer adapts them
      to the file after I3/I1/S1/MF1/C1. Tests of section 2.4 (`patch_r1_tests.py` and `patch_r2.py` have
      eleven of them in prototype form).
   3. S14: `chart_coordinate_names`, `check_chart`, the four arms; the test rewrites of section 5.3 (reference
      `patch_s14.py`, `patch_s14_tests.py`, and `$S/r2/patch_r2.py` for the `exterior_derivative` and `wedge`
      checks); tests of 5.4.
   4. N5: `plot_int_options`, the six evaluators, `Plot3DOptions`; the new `repl_limits` constants; tests of 1.4
      (reference `patch_n5.py` for two of the six, and `$S/r1/patch_r1e.py` for the integral Rational).
   5. S10: the `pochhammer` move and help line; test 3.3.
   6. S15: `hash_set_bytes`, `graph_bytes` (checked, `Option<u128>`), the six graph arms, `GRAPH_BYTES`;
      `PRIME_PI_X`; `count_bound` (`Option<u64>`, `PRIMES_SPAN`), the `primes` rewrite with its two passes above
      2^64 (`is_probable_prime`, then `try_is_prime`), `PRIMES_COUNT`, `PRIMES_SECONDS`,
      `PRIMES_STRONG_TEST_SECONDS`, `PRIMES_PROOF_SECONDS`, `table_seconds`, `t_test`, `t_proof`; help line;
      tests of 6.4. The model's Python twin is `$S/r2/est2.py` (same tables and formulas).
   7. sqrt_mod: constants, `SqrtModLimits`, `SqrtModPiece`, `sqrt_mod_factor`, the new `sqrt_mod_report`, doc
      comment, help line; tests of 7.4.
   8. The `repl_limits` doc comment (`:10446-10462`): list the new limits (graphs, prime_pi, primes, plot options,
      sqrt_mod) and how each was measured. The file-level comment above it: add MAGMA ch. 18 §18.4.1 (integer
      division gives a rational), §18.10.1, §18.12.1, ch. 19 §19.5.6, ch. 20 §20.1.2, §20.2.2, §20.4.2 (arithmetic
      in Q), ch. 25 §25.1.1-§25.1.2 (coercion into a real field, the semantics of `float_arith`), ch. 49
      (Pochhammer), Montgomery-Vaughan 1973, and Sorenson-Webster 2017 (the twelve bases below psi_12).
4. Trackers: `PROJECT_TRACKER.md` jupyter rows (REPL parser, sqrt_mod, limits), and the rustmath-symbolic row
   "Polynomial expansion (distribute, binomial power expansion for 0<=n<=10)" (`PROJECT_TRACKER.md:2159`) from
   🟢 to 🟡 with the note "`distribute_negation` (`expand.rs:250-263`) is wrong in both arms:
   `expand(-(x - 1))` = x - 1" (revision 2, m2; section 9). `docs/HANDOFF.md`: remove N5, S9, S10's pochhammer
   part, S11, S14, S15 and the sqrt_mod bullet; add section 9's findings (the `distribute_negation` bullet
   under S1's area). `docs/magma/` overrides for `ch018.Modsqrt` if it points at the REPL, then
   `python3 tools/magma_tracker/build.py`.

### 8.2 Acceptance

```bash
cargo test -p rustmath-rationals --no-fail-fast
cargo test -p rustmath-finitefields --no-fail-fast
cargo test -p rustmath-jupyter --no-fail-fast
cargo test --release -p rustmath-jupyter -- --ignored test_sqrt_mod_refusal_release
cargo check --workspace --tests
cargo test --workspace --no-fail-fast          # Rational::to_f64 is used everywhere; compare with the
                                               # baseline at c79a582: 14,173 passed, 0 failed, 330 ignored
grep -n "unsafe" rustmath-jupyter/src/repl.rs   # no hits
```

Release timing check (a scratch driver over the real `ReplContext` built with `--release`, like `$S/drive.rs`):
`prime_pi(6000000000)` between 5 and 20 s (else move `PRIME_PI_X` to the 10 s point and update the help text, the
doc and the test); `primes(10^30, 10^30 + 20000)` (258 primes) and `primes(2^512, 2^512 + 2000)` (7 primes)
admitted, each within 10 s and within a factor of 3 of its model time (2.2 s and 2.9 s; compute it with
`$S/r2/est2.py` on the refitted tables); `primes(10^400, 10^400 + 10)` and `primes(2^1000, 2^1000 + 200)` give
`[]` in under 0.1 s; one strong test at 512, 2048 and 8192 bits and one `try_is_prime` at 100, 512 and 1024
bits timed against `PRIMES_STRONG_TEST_SECONDS` and `PRIMES_PROOF_SECONDS` (else refit the tables, and check
again, on the same machine, that HEAD's cost per odd candidate of 6.3 stays at least three times `t_test`,
the margin test of 6.4); `sqrt_mod(4, 2^127+1)` under
1 s; `complete_graph(3585)` under 2 s.

### 8.3 Risks

- **Other units edit `repl.rs` first.** Line numbers move; the prototype patches target HEAD's text. MF1 changes
  `test_form_component_limits` (its §4.3); J1a replaces MF1's R^1 detour by the R^3 block. I1 rewrites
  `sqrt_mod_report` partly and its test (`test_sqrt_mod_large_prime_and_strong_pseudoprime`); J1a replaces the
  function again and keeps I1's test values that still hold (`sqrt_mod(41, psi_12)` gives "no square root" by the
  Jacobi shortcut, `is_prime(psi_12)` false), and restores the 4 and 73 mod psi_12 rows that I1 dropped
  (section 7.4).
- **I3 also edits rustmath-rationals** (`rational.rs:530`, `sage_wrapper.rs`), in other functions; I3 lands first
  and the J1a change to `Rational::to_f64` merges textually (section 4.2).
- **Integer / Integer is now a Rational.** The only code that reads the success of `eval_expr` as a signal is
  `parse_symbolic_arg` (`:4928`); it uses an `Expr`, `Symbol` or `Integer` result and otherwise falls through to
  the symbolic parser, as it did on the old TypeError, so `(4)/(2)` there still goes to the parser. No other
  caller of `apply_binary_op` exists (`grep -n "apply_binary_op(" repl.rs`: `try_eval_binary_op` and `negate`).
- **I1/I2 names.** If `sieve::for_each_prime` does not exist, use `sieve::primes_in_range` in windows of 2^20; if
  `factorization_with_limits` is missing, sqrt_mod falls back to phases 1 and 3 only (no rho, no sieve), which
  still covers HEAD's reach and refuses the rest honestly; if `try_is_prime` is missing, every probable prime is
  `Probable` with the caveat. If `is_probable_prime` is missing (I1 §2.1 makes it public), pass 1 of `primes` uses
  `prime::is_prime_miller_rabin(&c, 1)`: a false there is a strong-test witness, so a proof, and the probable
  primes go to pass 2 as designed; its 12 strong tests per prime are covered by the 40 / B term of `t_test`.
- **Second and third crate.** If the reviewers want J1a confined to rustmath-jupyter, the alternative for S11 is
  to add `num-rational` to rustmath-jupyter and convert there, leaving `Rational::to_f64` broken for the rest of the
  workspace (not recommended); for sqrt_mod, compute the count in the REPL from the closed form (duplicate logic).
- **Workspace effect of `Rational::to_f64`.** Values change only where |num| or den exceeds 2^53, by at most 2 ulp
  or from NaN/inf/0 to the right value. A test that pinned a double-rounded value would need its expected value
  updated to the correctly rounded one (verify with Python `n/d`).
- **Model constants** (graph bytes, `PRIMES_STRONG_TEST_SECONDS`, `PRIMES_PROOF_SECONDS`, the trial-division cost)
  are measured on this container; the graph model matched the REPL within 2 %, the strong-test table comes from
  num-bigint's modpow (which I1's BPSW uses), the proof table from I1's prototype, and both are re-timed in 8.2.
  The refusals of `primes` depend on these tables only near the limit: a call the tables put just below 10 s
  may take up to a few times longer on a slower machine, and the answer never depends on them.
- **Parser scope.** The evaluator stays split-based; precedence and unary rules are now right, but operators the
  REPL never supported (`%`, comparisons inside expressions) are unchanged.

---

## 9. Found on the way, not in this unit (for HANDOFF)

- The REPL has no Float arithmetic: `2 * 1e-3`, `1e5 - 1`, `1.5 * 2` are TypeErrors (`apply_binary_op`,
  `:9666-9968`, has no `Float` arm). Range bounds are the exception: `eval_to_f64` computes them with
  `float_arith`, in the real field of doubles (so `10^20 + pi - 10^20` is 0 there, as in MAGMA at 53 bits; a
  bound evaluator with higher precision, MPFR intervals as in MAGMA's real fields, would remove that). (Mixed Integer/Rational arithmetic was in this list before revision 1; J1a now adds it, S9
  item 9.) Symbolic values have no arm with a Rational and no division by an Integer: `x / 2`, `x * 1/2`,
  `x^2/2`, `x^(1/2)` are TypeErrors, at HEAD and after J1a. MAGMA, Sage and PARI coerce.
- `-(-x)` displays `--x` (the `Display` of `Unary(Neg, Unary(Neg, x))` in rustmath-symbolic), which reads like
  a decrement. A library display issue (S1's area); `negate` could also return `x` for a double negation.
  Reachable since S9 adds unary minus on expressions.
- rustmath-symbolic `expand` gives wrong results under a minus (review 2, m2; S1's area, not covered by S1's
  design). `distribute_negation` (`rustmath-symbolic/src/expand.rs:250-263`) maps -(a + b) to (-a) - (-b) and
  -(a - b) to (-b) - (-a), so both arms negate only one term: `expand(-(x - 1))` displays `(-1 - -x)`, which is
  x - 1 (correct: 1 - x), and `expand(-(x - 1)*(x + 1))` is x^2 - 1 (correct: 1 - x^2; `$W/r2_b.txt`, HEAD and
  the prototype alike). The fix is -(a + b) = (-a) + (-b) and -(a - b) = (-a) + b, with tests from Sage
  `expand(-(x - 1))` = `-x + 1`. `PROJECT_TRACKER.md:2159` rates `expand` 🟢; 8.1 item 4 downgrades it. The
  symbolic parser reached it at HEAD (`expand(-(x - 1))` above); S9's unary minus makes `g = -(x + 1);
  expand(g)` reachable from `eval_expr` too.
- `median`, `min` and `max` in rustmath-stats unwrap `partial_cmp` (`statistics.rs:60`, `:129`, `:136`), so a
  NaN panics. After J1a the REPL passes no NaN (section 4.1); other callers can.
- Deeply nested input aborts the kernel with a stack overflow, which no `catch_unwind` can catch. At HEAD in a
  debug build (8 MB main thread, `$S/r1/drive_head`): 35 nested calls `abs(abs(...))` (a frame chain of about
  240 KB per call level), 815 nested lists; in release (`$S/r1/drive_rel`): 2267 nested calls, 6460 nested
  lists. After J1a: runs of signs, runs of parentheses and long `+ -`/`* /` chains no longer recurse per
  character or term (S9 items 5 and 7), so HEAD's abort at a 732-term numeric sum (debug) is gone. That holds for
  numeric chains only (review 2, m3): a symbolic chain `w + w + ... + w` builds a left-nested `Expr` whose
  display and drop recurse once per term, and 3000 terms still abort the debug driver (8 MB main thread), while
  2000 pass (`$S/r2/symchain_*.txt` on `$S/r2/proto/drive`); an iterative `Display` and `Drop` for `Expr`
  would fix it (S1's area). Nesting that
  alternates operators and brackets recurses once per level, like nested lists (`-(-(...))` and `(1+(1+...))`
  abort at 547 levels in a debug build; nested calls still at 35, lists at 904). A later unit can add a
  bracket-depth limit (CPython's parser refuses more than 200 nested parentheses) and run `eval` on a thread
  with a large stack.
- `mean([1e308, 1e308])` gives `inf`: the sum overflows inside `rustmath-stats` on finite input (HEAD and
  after J1a).
- The REPL `factor(n)` command (`:921-933`) uses HEAD's trial division (`Integer::factor`) and hangs like sqrt_mod
  did; the same `sqrt_mod_factor` (or I2's driver directly) would fix it. `sigma(n, k)` factors the same way.
- String literals are not unescaped (`"a\"b"` keeps the backslash), as at HEAD.
- `integrate_num(x, x, 0, 1)` with a bare symbol x is a TypeError ("first argument must be an expression",
  `:3001-3004`); `x*1` works.
- The num/den division of S11 also appears in `rustmath-schemes/src/elliptic_curves/heegner.rs`,
  `rustmath-rings/src/qqbar.rs`, `rustmath-numberfields/src/kummer.rs`, `rustmath-combinatorics/src/derangements.rs`;
  each should call `Rational::to_f64` (correct after this unit) or the num-rational conversion.
- `Plot3DOptions` is never used by a command, and the 3D plots have fixed grids (30, 30x30, 20^3).
- Graph commands other than the constructors (`edges`, adjacency output, `show_graph`) are not bounded either;
  their output is at most the graph's size, which the constructor limit now bounds.
