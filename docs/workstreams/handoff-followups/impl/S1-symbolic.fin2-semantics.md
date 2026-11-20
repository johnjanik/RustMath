# S1 finish, review fin2, lens: numeric and series semantics

Reviewer of S1 as it stands: main 28c4b69 (the S1 work in progress is 4763f31) plus the uncommitted
changes of the finish and of its fix-up round 1 (`git diff`: `expand.rs`, `parser.rs`, `series.rs`,
`series_const.rs`, `series_engine.rs`, `simplify.rs`, `repl.rs` docs, the test files; untracked
`tests/s1_finish1.rs`, `tests/s1_review6.rs`). Bar: the narrowed bar of this run. No repository file
was edited: `git status --short` and the md5 of `git diff` (d33aec2d...) are the same before and
after (`fin2-semantics/status_before.txt`, `status_after.txt`, `diff_md5_before.txt`). Scratch:
`fu2/s1impl/fin2-semantics/` (paths below are relative to it).

## Verdict

Approved in this lens. No blocker under the narrowed bar. The tests pass in debug and in release.

- The two blockers of round 1 in this lens are fixed, and so are the five of the resources round 1
  (reproduced on the built code, section 2).
- The new power rule (`variable_exponent`) decides `NotLaurent` only when it is right. On 4 752
  rows of my own (class V) and 2 000 random trees (class R), every `NotLaurent` from the rule is
  non-Laurent by the contour oracle, and every refusal of a function that is Laurent is
  `UndecidedZero` or `PrecisionExhausted`, never a decided answer.
- The owner's two rules hold where the argument reaches the branch point through an undecided
  zero coefficient (class W, 1 080 rows: all `UndecidedZero`), and the 46 truncated-constant rows
  of the matrix pass in both profiles.
- The parser that no longer recurses gives the conventional value on 20 000 random texts checked
  in gp, with the i64 boundary literals, the literal rationals, calls and white space. `expand`
  keeps the value of 6 000 random trees.
- Four minor issues (section 9): a printed result can fail to re-parse (a typed `ParseError`) while
  a doc says every result re-parses; the docs give 4.1 s for `simplify` of a result where that is
  `simplify` after `expand`; the `PrecisionExhausted` message names a denominator for the new
  exponent and branch-point cases; one row lost a correct decided `NotLaurent` (now honest
  `UndecidedZero`).

## 1. Tests

| command | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic --no-fail-fast -- --test-threads=2` (debug, `debug_tests.log`, 78 s) | unit 685 (5 ignored); `s1_depth` 14; `s1_finish1` 8 (1 ignored); `s1_fuzz_data` 7; `s1_payload` 2 (1); `s1_regressions` 26 (2); `s1_review2` 2; `s1_review3` 7 (7); `s1_review4` 4 (2); `s1_review4_memory` 1 (1); `s1_review5` 4 (2); `s1_review5_memory` 2 (2); `s1_review6` 4; `s1_singular_points` 9; doc 116 (11). 891 passed, 0 failed |
| the 14 release test binaries (built 08:54 by the concurrent resources review, after the last source edit at 08:31; copies in `relbin/`; `rel_tests.log`) | 685, 14, 8, 7, 2, 26, 2, 7, 4, 1, 4, 2, 4, 9 passed: 775, 0 failed |
| release `--ignored --test-threads=1`, same binaries | `s1_finish1` 1 (`b3f1_*`, 0.78 s); `s1_payload` 1 (3.29 s); `s1_regressions` 2 (0.79 s); `s1_review3` 7 (1.82 s); `s1_review4` 2 (2.24 s); `s1_review4_memory` 1; `s1_review5` 2 (2.22 s); `s1_review5_memory` 2; unit `fold_work_calibration` and `parse_work_calibration` passed. The unit binary also ran three ignored tests from before S1: the two `factor` tests pass, and `specialfunctions::airy::tests::test_airy_numerical_helper` fails (`ai_large < 0.001`). It is `#[ignore]` "just a demonstration" at 8a0b310, and S1 changed only the `matches!` borrow lines of that file (E0509). Not counted |

The tests of the first step are not in the diff (`s1_regressions.rs`, `s1_fuzz_data.rs`,
`s1_common/`, `data/s1_fuzz.tsv`). `s1_singular_points.rs` and its data gain lines only (one comment
line is extended), and in `s1_depth.rs` the one changed test (`b2_limit_with_error_of_a_signed_tower`)
builds the tree it parsed before, with the same assertions (the finish step, reviewed in round 1).

## 2. The blockers of round 1

| item | status | evidence (built code) |
|---|---|---|
| semantics B1: `NotLaurent` for f^g with an undecided exponent coefficient | fixed | fin1's `b1.tsv`: the 7 rows are `UndecidedZero` (`x^((1+x)/(1+x))` stays `PrecisionExhausted`, `x^(1+x)` stays `NotLaurent`); fin1's grids rerun (`out/dbg_*.out` against fin1's outputs, fields 1 to 17): `pts` 100 rows, `hand1` 24, `M` 6, `G` 2 and `b1` 7 moved from `NotLaurent` to `UndecidedZero`, and no other field of any row changed (`hand2`, `hand3`, `hand4`, `tc`, `near`, `s` identical). fin1's oracle had found all of the moved `pts`, `M` and `b1` rows Laurent |
| semantics B2 = resources B1: the literal -9223372036854775808 relaxed the depth limit | fixed | `parse/pmin.rs` (fin1's probe): the six texts parse at 150 and 200 levels on a 2 MB debug thread with HEAD's trees (`--N` is 9223372036854775808 at both; HEAD prints it as `--9223372036854775808`); `x^-9223372036854775809 + nest(x)` stays refused. Values through the engine equal gp (`out/N.tsv`): `x^-9223372036854775808` at 1 is `1 - 2^63 h + 42535295865117307937533511947398414336 h^2` (gp `binomial(-2^63, 2)`), `2/-9223372036854775808` is `-1/4611686018427387904` |
| resources B2: HEAD's literal rational refused at once, the port aborted | fixed | `r1/rb.rs parse` on fin1's `div*.txt`: the eight 2 MB debug texts and the two 8 MB release texts (`2/x*` with 5 000 parentheses, with 30 001 carets) are a `ParseError`, HEAD's outcome |
| resources B3: re-parse up to 111 s | fixed | `r1/rb.rs rt` (release, 8 MB): q4 `(10^100-1)^1050/7^120000/(1-x)` at order 32 is a value whose re-parse takes 5.98 s (wall 7.4 s for all steps); orders 33, 100, 514 and q5 at 282 are `LimitExceeded` ("units of work to re-parse") in at most 0.22 s |
| resources B4: re-parse overflowed the stack | fixed | `sin^195(1)/(1-x)` at order 1000 (2 MB debug) re-parses in 0.47 s; `sin^4320(1)/(1-x)` at 100 and 400 (8 MB release) in 0.08 s and 0.41 s; `Display` of each re-parse equals the first `Display` |
| resources B5: `expand` of the parsed signed tower aborted | fixed | the four towers at 1 439 levels (2 MB debug) and 22 772 (8 MB release) parse and expand; `expand` leaves them unchanged (`Display` equal) |

## 3. Method

- Driver `drv/r8.rs`: fin1's `r7.rs` unchanged, compiled against this tree's debug rlib
  (`librustmath_symbolic-b2048a251f3765f1.rlib`, built 08:49, fresh for the sources) and the
  release rlib (`-77a3e0a7ff0c5cd6`, 08:54). Per row: `series_expansion` (budget 1e8),
  `taylor_with_budget` (1e8) and the re-parse of its `Display`, `limit_via_series(.., 5)`,
  `series_coefficients`, all printed in gp form. Release on an 8 MB thread and debug on a 2 MB
  thread give the same fields 1 to 17 (md5) on every file of this review. No panic.
- Oracle: review 6's contour-integral script `cauchy.gp` with `run6.py` (gp 2.15.4), as fin1. A
  `NotLaurent` row must have circles that disagree or an infinite principal part
  (`NONLAURENT_OK`); a refused row with agreeing circles is `CONSISTENT` (the function has a
  Laurent expansion there); an `Ok` row's coefficients must equal the contour coefficients (`PASS`).
  `rt.py` compares a result with its re-parse at a point.
- New inputs:

| file | rows | content |
|---|---|---|
| `out/V.tsv` (`gen/gen_v.py`) | 4 752 | f^g with f of nonzero valuation (h, h^2, 1/h, sin h, h(1+h), h^3/(2+h), tan(h)^2, 1/sin(h)^2, e^h - 1) at 0, 1, -2/3, pi, sqrt(2), I, 1+I and infinity; g = k + U h, k + U/h, k + U h + h^2, (k+U) + U h, k(1+x)/(1+x) + U h, k + U sin h, U h, k + 1/h + U h, 1/2 + U h, k + N h, k + U h + N h^2, k + h - h, k + a h, k + I h, k + U h^5, k + 0 h; U one of six exact zeros the engine cannot decide (`log(2)+log(3)-log(6)`, Machin's, `sin(1)^2+cos(1)^2-1`, `exp(1)*exp(-1)-1`, `sqrt(2)*sqrt(3)-sqrt(6)`, `exp(I*pi)+1`), N a decided nonzero constant |
| `out/W.tsv` (`gen/gen_w.py`) | 1 080 | the owner's rules: asin, acos, acosh, atanh at +-1 and atan, asinh at +-I, of c e^(U h), c + U h, c (1+U h)^2, c/(1-U h), c h^(U h), c 2^(U/h), c + sin(U h)^2, c h^(1+U h)/h, c cos(U/h), c + U/h, at 0, pi and infinity: each argument is the branch point c exactly |
| `out/Z0.tsv` | 20 | 0^g, 1^g and (x-x)^g with variable g |
| `out/R.tsv` (`gen/gen_r2.py`, seeds 71, 72) | 2 000 | random trees after review 6's `gen6r.py`, the point chosen first so that leaves vanish there, undecided zeros among the constants, and 30 % of the inner nodes powers with a variable exponent |
| `parse/q21`, `q22` (`gen_parse2.py`) | 8 000 | review 5's parser fuzz with the i64 boundary literals (2^63 - 1, 2^63, 2^63 + 1, with leading zeros, 2^64), x = 3/7, y = -5/11, dyadic floats, and literal quotients (2/3, 2/-3, 2/3/4, 2/3^2) |
| `parse/c41` to `c45` (`gen_call.py`) | 15 000 | calls (sin, cos, exp, sqrt, atan, ln, arcsinh, f(a,b), g(), h(a)) nested with white space (space, tab) |
| `expand/ex31`, `ex32` (`gen_ex.py`) | 6 000 | random trees of sums, differences, products, quotients, negations, powers with exponents -3 to 11, and calls |

## 4. The power rule of fix-up round 1 (`series_engine.rs:2089`, `variable_exponent`)

Read against the code: for f of decided nonzero valuation and g not an exact constant, a stored
term of g with e != 0 and a decided nonzero coefficient, or a decided non-integer constant term,
keeps `exp(g log f)` and so `NotLaurent`; else the first undecided coefficient is reported. Both
decisions are sound: v g_e h^e log h cannot cancel, and a non-integer constant exponent of a base
of nonzero valuation is discontinuous under the principal log. A stored coefficient is never a
decided zero (`Ser` keeps no structurally zero coefficient, and `zero_status` returns `Zero` only
for the empty constant), so no row can fall through to `NotLaurent` with every coefficient zero.
An exact-constant g never reaches the rule (`with_exponent`, `:1700`).

Measured (`out/rel_V.out`, `out/Vsel.res`):

| expectation | outcome | oracle (1 489 rows: all V rows at 0, pi, I, infinity for NL; all exact rows; a quarter of the U rows at pi, 1+I, infinity) |
|---|---|---|
| NL (a decided term or a non-integer constant term) | 1 872 `NotLaurent` | 1 008 checked, all `NONLAURENT_OK` |
| U (the function is f^k, Laurent) | 2 688 `UndecidedZero` | 289 checked, all `CONSISTENT` |
| V (an exact exponent) | 192 values | 192 `PASS` |

No row has another outcome. The limit path gives `Unknown` on every refused row, and a value row
gives `Finite` only with the right constant (`L=ok`). `0^(1+U x)` is 0 (right for every U), and
`0^(U x)`, `0^(1+U/x)`, `(x-x)^(U x)` are `UndecidedZero` (`out/Z0.tsv`).

Random trees (`out/R.res`, 2 000 rows; 1 218 values, 461 `UndecidedZero`, 260 `NotLaurent`, 52
`PrecisionExhausted`, 7 `DivisionByZero`, 2 `BudgetExceeded`):
- values: 1 146 `PASS`; 54 one-sided at a cut on a half-plane; 2 with no matching half-plane, both
  on a cut through the point (`(x^(1+(x+1/2)))^x` at -1/2, `acosh(1/(1/x)^2)` at infinity), which
  equal gp's principal function on both real rays (`out/noside.gp`), the documented convention for
  a constant on a cut; 16 rows the oracle marks `BAD` are functions that are identically 0 (for
  example `(sin(1)^2+cos(1)^2-1)*(x-pi)` at pi), whose coefficients the engine returns as the exact
  constants (sin(1)^2 + cos(1)^2 - 1, ...): gp evaluates them to 1e-116 at 110 digits where the
  contour sum is 0. The coefficients are right.
- `NotLaurent`: 235 `NONLAURENT_OK`; 24 `CONSISTENT`, each an essential singularity (exp, sin, cos,
  sinh of a pole, `pi^x` at infinity; the oracle cannot see an infinite principal part) or a
  non-Laurent subexpression (`x^(2+x)/x^x`, `sinh(log(x-pi))`, `((x^(-1/2))^(-1))^(-2)`), the
  documented convention of `SeriesError::NotLaurent` (`series.rs:81`); 1 gp overflow
  (`sin(sqrt(sin(x)))` at infinity, essential).
- `DivisionByZero`: 7, all exact zero divisors (gp fails).

## 5. The owner's two rules and the singular-point matrix

- Class W: all 1 080 rows are `UndecidedZero`, with `Unknown` limits and refused coefficients. The
  argument is the branch point exactly, so a value would be right and `NotLaurent` wrong; the
  engine decides neither (`out/rel_W.out`).
- The 46 class (h) rows (`tests/data/s1_singular_points.tsv`) pass in both profiles; fin1's 224
  truncated-constant rows at pi, sqrt(2), e and 0 (`out/in_tc.tsv`) give the same outcomes as in
  round 1 (`UndecidedZero` or `PrecisionExhausted`, never a value, `NotLaurent` or
  `DivisionByZero`). fin1's `hand4` (rules at branch points written through Gaussian inverses) is
  unchanged.

## 6. The parser that no longer recurses

- `q21`, `q22` (8 000 texts): the port's value equals gp's conventional value on 7 547 (7 303
  exactly, 244 to 1e-30 with floats); it refuses 38, and 415 parse to a tree whose gp value fails,
  all where gp's own value fails (a division by zero). HEAD reads 3 476 of them; the port reads all
  but two, `(7/7)/-(0/5)` and `0009223372036854775807 /0`, a literal divided by a literal 0, which
  the design refuses ("division by zero in a literal", §2.12; gp fails too). Where both read a
  text, they agree except on 2 rows where HEAD's `a/b` token is not the conventional reading.
- `c41` to `c45` (15 000 texts with calls): no port value differs from gp in `c44`, `c45`; in
  `c41` to `c43` the 57 differences are all rows where my generator wrote `1/2` without
  parentheses after `/`, `*` or `^` (a fault of the reference; `c44`, `c45` fix it). HEAD reads no
  text the port refuses.
- Re-parse round trips (`rt.py`): 698 of 700 values of fin1's class G, 497 of 497 of class M and
  168 of 168 of class V re-parse to the same value at a point; the other 2 are gp failures, as in
  round 1.

## 7. `expand` (`expand.rs:35`)

`expand(e)` equals e in gp at two rational points on 5 916 and 5 908 checks of `ex31` and `ex32`
(exact `EQ`, or `EQF` within 1e-40 where sqrt, sin or exp make floats); the other 76 and 86 are
points where e itself has no value in gp (`BOTHERR`). No check differs. The walk builds each node
with the rules of the recursive version (`expand_binary`, `distribute_negation`), and the power
rule still reads the unexpanded exponent, as before.

## 8. Doc comments on the consumers of a result

The docs touched by this run (`MAX_RESULT_TREE`, `MAX_RESULT_TEXT`, `MAX_RESULT_FOLD_WORK`,
`MAX_RESULT_MEMO_WORDS`, `MAX_RESULT_PARSE_WORK`, `StructureLimit::FoldWork`, `ParseWork`,
`taylor_with_budget`, `simplify`, the driver's comment, the REPL's `TAYLOR_WORK`) say that the
consumers of a result are not bounded and give measured numbers. A grep for "bounded" in the
touched sources finds no claim that a consumer of a result is bounded. Two numbers are wrong
(minor m2).

## 9. Minor issues

**m1. A printed result can fail to re-parse, and the docs say it cannot.** The depth limit is
turned on by an integer literal outside i64, so a result that prints such a literal before more
than 128 levels of nesting is a `ParseError` on re-parse. `exp(x) + sin^150(1)/(1-x)` (767 bytes;
`sin^150(1)` is 150 nested `sin(` around 1) at 0, order 21 or more (21! > 2^63 - 1): the result
prints in 0.00 s, and `parse` of its `Display` is "the expression nests too deeply for its syntax"
on a 2 MB debug and on an 8 MB release thread (`r1/bigdeep.txt`, `r1/rb.rs rt`). At order 20 it
re-parses. `exp(x/2) + sin^150(1)/(1-x)` at 25 and 35 is the same (debug). Out of the bar: the refusal is
typed and immediate (no time, memory or stack hole), and HEAD read no literal above 2^63 - 1. But
`parser.rs:193` ("a printed result re-parses on any thread") and the report (`S1_impl.md:1915`,
"every result re-parses") are false for it; the finish step's own note (`S1_impl.md:1656`) says so.
Required: state the exception in the `mod depth` doc and in `taylor_with_budget`'s doc, and correct
the report and the tracker row of the parser.

**m2. Two consumer numbers misstate which consumer costs what.** `series_const.rs:706` to `:707`
(`MAX_RESULT_TEXT`) and `simplify.rs:422` give "`simplify` of a result ... at most 4.1 s (a sum of
1 487 fractions 1/p in a kernel at order 10)". The 4.1 s is `simplify` after `expand` (the table of
fix-up round 5, `S1_impl.md:1389`: `simplify(R)` 0.35 s, after `substitute` 3.93 s, after `expand`
4.12 s; `MAX_RESULT_FOLD_WORK`'s own doc says "after `expand`"). Measured here (`cons/cs.rs`,
release, `sin(1/2+1/3+...+1/p_1487)/(1-x)` at order 10, the largest accepted size; 1 488 is refused
by `FoldWork`): `simplify(R)` 0.36 s, `simplify(expand(R))` 4.53 s,
`simplify(substitute(R, x, 2))` 4.70 s. Also `simplify.rs:428` still gives "66.0 s at order 100"
for q4 after `substitute`, an input now refused above order 32 (`MAX_RESULT_TEXT` says so; the
`simplify` doc does not). Neither doc claims a bound. Required: attribute the numbers to the right
consumer and note the refusal.

**m3. The message of `PrecisionExhausted` names a denominator for the new cases.** `series.rs:105`
and `:225`: "a denominator vanished to working precision P (it may be identically 0)". The finish
returns it for a truncated integer exponent (`x^((1+x)/(1+x))` at 0, `truncated_integer_exponent`)
and for a truncated constant at a branch point (the class (h) rows, `arcsin((1+x)/(1+x))`), where
no denominator is involved. Required: a message that names the bounded search ("a coefficient, an
exponent or a branch-point argument could not be decided to working precision P").

**m4 (note, no change asked).** fin1's row G P:884,
`(x-1/(pi+I))^(1/2+(1+x)/(1+x)-1)` at 1/(pi+I), was a correct decided `NotLaurent` and is now
`UndecidedZero`: at this point the constant term 1/2 + (a truncated zero) has an undecided
integrality and the other coefficients are undecided, so the new rule reports one. Honest, weaker.

**n1 (carried from round 1).** `limit_with_error` gives `Finite` only for a rational limit (frozen
design §2.13); the handoff to the tracker row stands.

## 10. Reproduction

```
F=/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/s1impl/fin2-semantics
# drivers (debug against target/debug, release against target/release, which is now deleted;
# the built binaries r8_rel, rb_rel, cs_rel, pmin_rel are statically linked and kept)
cd $F/drv && ./build.sh r8.rs dbg r8_dbg
cd $F/out && python3 -I ../gen/gen_v.py > V.tsv && ../drv/r8_rel V.tsv > rel_V.out    # class V
cd $F/out && NPROC=2 python3 -I ../run6.py Vsel.out Vsel                               # oracle on V
cd $F/out && python3 -I ../gen/gen_w.py > W.tsv && ../drv/r8_rel W.tsv | cut -f3 | sort | uniq -c
cd $F/out && ../drv/r8_rel R.tsv > rel_R.out && NPROC=2 python3 -I ../run6.py rel_R.out R  # 14 min
cd $F/out && gp -q noside.gp                       # the two cut rows on the real rays
cd $F/parse && python3 -I runp.py q21 && python3 -I runp.py c44                     # parser vs gp, HEAD
cd $F/expand && ./ex_dbg ex31.tsv > ex31.out && python3 -I runex.py ex31.out        # expand values
cd $F/parse && ./pmin_dbg 150 && ./pmin_head 150                                     # round-1 B2
cd $F/r1 && ./rb_rel rt q4.txt 32 8388608 && ./rb_rel rt q4.txt 33 8388608           # round-1 B3
cd $F/r1 && ./rb_dbg rt bigdeep.txt 25 2097152                                       # m1
cd $F/cons && ./cs_rel pharm1487.txt 10                                              # m2
gp -q < ../fin1-semantics/b1_refs.gp     # the gp references cited by tests/s1_finish1.rs (re-run: 0.E-213, 9.9E-212, 0.0178)
```

Housekeeping: `rustmath-misc/output.txt` is unchanged (md5 746308829575e17c3331bbcb00c0898b), no
`.tmp_dir_*` exists, `git status --short` and the md5 of `git diff` are unchanged.
`target/release` had already been deleted by the concurrent resources review when this review
ended; the release test binaries used are copied in `relbin/`. Free disk at the end: 5.9 GB.
