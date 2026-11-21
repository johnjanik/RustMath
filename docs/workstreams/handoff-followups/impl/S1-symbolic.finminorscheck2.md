# S1 finish, check of the minor-item fixes (round 2 of the check)

Independent reviewer of the section "Finish minor items" of `fu2/s1impl/S1_impl.md`: the fixes of
the seven items that the reviews fin2 (`S1.fin2-semantics.md` m1 to m4, n1;
`S1.fin2-resources.md` minors 1 to 4) listed. Bar: the narrowed bar of this run. Branch
`claude/gracious-gates-e6b6l7`, HEAD 52708d7 (main; the merges after 28c4b69 touch no file of
rustmath-symbolic or rustmath-jupyter: `git diff --stat 28c4b69 HEAD` lists only
`docs/workstreams/m23_golay/REQUESTS.md`), plus the uncommitted changes. No repository file was
edited. Scratch: `fu2/s1impl/finminors2/` (paths below are relative to it).

## Verdict

Approved. No blocker under the narrowed bar.

- The step changes code in two places only: a match arm in `parse` (`parser.rs:151`) that maps
  the literal fold's `Verify` failure to the design's message, and the `Display` text of
  `SeriesError::PrecisionExhausted` (`series.rs:232` to `:236`). Every other changed line is a
  doc comment or a comment (mechanical filter of the step's diff, section 1).
- Both code changes map an `Err` to an `Err` with another message. No `Ok`/`Err` outcome and no
  error kind changed, so no undecided answer became decided and nothing regressed against
  8a0b310 or against the state before the step.
- The tests pass: `rustmath-symbolic` debug 894 passed, 0 failed; the six dependents 1 556 passed,
  0 failed; `rustmath-jupyter` lib 120 and `kernel_protocol` 4; release `s1_finish1` 11 (and
  the ignored budget test) and `s1_depth` 14 on 8 MB threads.
- Two nits in the new doc text (section 3): three byte counts are one too high, and the short form
  of the depth condition ("more than 128 levels of parentheses and calls, or 1 024 of `^`") is not
  exact for a text that mixes both, which the docs' own first example does.

## 1. What the step changed

The state before the step was rebuilt from `minors/diff_before.patch` (md5 d33aec2d..., the same
as the md5 of `git diff` that the fin2 reviewers recorded) applied to HEAD (`before/`). Against it,
five tracked files differ: `parser.rs`, `series.rs`, `series_const.rs`, `simplify.rs` and
`rustmath-jupyter/src/repl.rs` (`step_a.diff`, `step_b.diff`). `expand.rs`, `series_engine.rs`,
`s1_depth.rs`, `s1_singular_points.rs` and the two data files are identical to the state before
the step. The untracked `tests/s1_review6.rs` was last written on 2026-11-16 08:00, before the
step began (09:39). No copy of `tests/s1_finish1.rs` from before the step exists; its tests before
`:400` match the table of the finish fix-up round 1 (rows, orders and expected outcomes checked
by reading), and the step adds `big_sum`, `reparse_message` and the three `fin2_*` tests at the end.

The non-comment lines of the step's diff (`grep '^[+-]' | grep -v '^[+-]\s*//'`):

```
+        Err(nom::Err::Failure(e)) if e.code == ErrorKind::Verify => Err(ParseError {
+            message: "division by zero in a literal".to_string(),
+        }),
-                "a denominator vanished to working precision {precision} (it may be identically 0)"
+                "a bounded search could not decide a coefficient, an exponent or a branch-point \
+                 argument to working precision {precision} (what it tested may be identically 0)"
```

## 2. Each fix

**Item 1 (semantics m1, resources minor 1): a printed result does not always re-parse.** Fixed as
asked, a doc fix. The false claims are gone: `parser.rs` (the `mod depth` doc) now says "no text
overflows its thread's stack, the re-parse of a printed result included" and adds "The exception
for printed results" with the condition and the examples; `taylor_with_budget` (`series.rs:756`),
`MAX_RESULT_TREE` (`series_const.rs:644`), `MAX_RESULT_TEXT` (`:691`) and the REPL's
`TAYLOR_WORK` (`repl.rs:11222`) state the exception; the report's deviation 2 of the finish
fix-up round 1 (`S1_impl.md:1915`) is corrected in place. A grep of both sources for "re-parses",
"reparses" and "round trip" finds no other such claim. The tracker row and the HANDOFF are left to
the docs step (handoff items 1 and 2 of the section), as this run's rules require.

I reproduced every example of the doc with my own driver (`drv/r1.rs`, against the current debug
rlib, 2 MB thread, `taylor_with_budget` with budget 1e8, then `parse` of `Display`; `drv/r1.log`):

| input | at, order | result text | re-parse |
|---|---|---|---|
| `x` and 100 copies of `^-2` (301 bytes) | 1, 3 | 4 305 bytes | `Ok`, 0.007 s |
| `x` and 110 copies (331 bytes) | 1, 3 | 4 725 bytes | refused by the depth limit, 0.001 s |
| `x` and 33 333 copies (100 000 bytes) | 1, 3 | 1.40 MB (taylor 0.90 s) | refused, 0.0005 s |
| `exp(x) + sin^150(1)/(1-x)` (766 bytes) | 0, 20 / 21 | 16 426 / 17 222 bytes | `Ok` / refused, 0.013 s |
| `exp(x)*sin^130(1)` (658 bytes) | 0, 20 / 25 | 14 314 / 17 809 bytes | `Ok` / refused, 0.010 s |
| `sin(A1+...+A100)/(1-x)`, `A200` (30-digit Ak) | 0, 3 | 14 045 / 28 045 bytes | `Ok` / refused, 0.0003 s |

The refusal is the typed `ParseError` of `parse` ("the expression nests too deeply for its
syntax"), immediate, on a 2 MB debug thread; the release test `fin2_a_printed_...` runs the same
rows on 8 MB threads (section 4). This is no budget hole under bar (c), and HEAD's parser refused
every relaxing construct (an integer above i64::MAX, `^ -(`), so it read none of these texts.

**Item 2 (semantics m2, resources minor 3): the consumer numbers.** Fixed. Each number now names
its consumer and its source, and I checked them against the logs:
`fix5/probe/edges_final.log` (26 edge rows): the largest `simplify` of a result itself is 0.862 s
(`pharm` at order 1, n = 2 032); `pharm` at order 10, n = 1 487: `simplify` 0.352 s, after
`substitute` 3.934 s, after `expand` 4.121 s; the largest `simplify` after `differentiate` 2.900 s
(`sin((2^254+1)^4096)/(1-x)` at order 192). `fin2-resources/rp/q4_subssimp.log`: q4 after
`substitute`, 21.864 s at order 30 and 23.268 s at order 32. `fin2-resources/rp/rp_edge.log` and
the review's table: q3 at order 379 re-parsed in 5.94, 6.58 and 7.57 s, so "about 6 to 8 s" is
right. The 66.0 s at order 100 now comes with "the engine now refuses this input above order 32"
in `simplify.rs:431` and `repl.rs:11236`. A grep of the step's added lines for "at most",
"bounded" and "bound" finds no claim that a consumer of a result is bounded: the hits are the
parser's limit, the "bounded search" of `PrecisionExhausted`, and "does not bound" or "not a bound".

**Item 3 (semantics m3): the message of `PrecisionExhausted`.** Fixed. The variant doc
(`series.rs:105`) lists the searches that return it, and they match the sites of
`SErr::NeedPrecision` in `series_engine.rs` (`inv`; `require_nonneg_val` for exp, sin/cos,
sinh/cosh and the atan family; `log`; `pow_const` and `pow_int` of an empty base; `integrate_check`;
`abs_sign`; `truncated_integer_exponent`; `zero_pow`), and the driver's `PrecisionExhausted(p)`.
Probe (`drv/pe.rs`, `drv/pe.log`): `1/(exp(x-x)-1)`, `atanh(1+(sin(x)^2+cos(x)^2-1))`,
`tan(pi/2+x-x)`, `x^((1+x)/(1+x))`, `arcsin((1+x)/(1+x))`, `log(sin(x)^2+cos(x)^2-1)` and
`exp(1/(sin(x)^2+cos(x)^2-1))` at 0 are still `PrecisionExhausted` through `series_expansion` and
`taylor_with_budget`, and print the new text. The REPL's `taylor` formats the error with `{e}`, so
it shows the new text too. No test or caller in the workspace compared the old text (grep for
"denominator vanished": one unrelated doc line in `rustmath-functionfields`).

**Item 4 (resources minor 2): a literal zero divisor.** Fixed. The literal fold is the only source
of a `Failure` of kind `Verify` (`parser.rs:790`; the parser uses no nom `verify` or `cut`, and
`product_step` returns `None` only when `Rational::new` refuses a zero denominator). The arm sits
after the depth arm and before the float arm, and maps an `Err` to an `Err`. Corpus check
(`drv/gen.py`, 14 865 texts: literal quotients of 13 literals, with white space, parentheses,
signs, i64 edges and leading zeros, in sums and calls, and 6 000 random trees; `drv/p1.rs` against
the current rlib and against HEAD-before-S1's rlib `librustmath_symbolic-4cbaba7850391989`,
`drv/ana.py`): no text still prints nom's "code: Verify"; 2 915 texts give "division by zero in a
literal", and each has a literal zero divisor; HEAD refused 1 765 of them and read 1 150 as a
division by zero (`5 / 0` as `5 / 0`, `(-0) / (0)` as `-0 / 0`). That class is the one the fin2
resources review did not count under bar (b) (the value is undefined; MAGMA and Sage refuse
`5/0`); it comes from the frozen design §2.12 and predates this step, which only names it. The
controls `x/0`, `5/(0*x)` and `0/5` still parse (`drv/c2.cmp`).

**Item 5 (resources note 4 (a)).** No change, as asked. Confirmed: at 8a0b310
`specialfunctions::airy::tests::test_airy_numerical_helper` is `#[ignore]` ("just a
demonstration"), and S1 changes only `matches!(x, ...)` to `matches!(&x, ...)` in that file.

**Item 6 (resources note 4 (b), semantics m4).** Notes, no change. Rechecked:
`(x-1/(pi+I))^(1/2+(1+x)/(1+x)-1)` at 1/(pi+I) is `UndecidedZero` (`drv/pe.log`), honest.

**Item 7 (n1, carried).** A handoff to the tracker row of `limit_with_error` (item 5 of the
section's handoffs). No code.

## 3. Minor issues (nits, no change required for approval)

**n1. Three byte counts are one too high.** `x` and 110 copies of `^-2` is 331 bytes (the docs and
the test comment say 332; 100 copies: 301, the comment says 302), `exp(x) + sin^150(1)/(1-x)` is
766 bytes (767), `exp(x)*sin^130(1)` is 658 bytes (659), measured with `wc -c` on the texts the
test builds (`drv/t110.txt`, `s150.txt`, `s130.txt`). The fin2 resources review counted its input
files with a newline. In `parser.rs:214`, `:216` and `:218`, the test comments
`s1_finish1.rs:439`, `:444` and `:447`, and the report's item 1.

**n2. The short form of the depth condition is not exact for mixed nesting.** `taylor_with_budget`
(`series.rs:758`), `MAX_RESULT_TREE` (`series_const.rs:644`, "more than 128 levels of nesting"),
`MAX_RESULT_TEXT` (`:694`) and `TAYLOR_WORK` (`repl.rs:11224`) say a printed result is refused when
it "nests more than 128 levels of parentheses and calls (or 1 024 of `^`)". The limit is 1 024
weighted units, 8 per parenthesis or call and 1 per `^` (`depth::NEST`, `TOWER`), so a text that
mixes both is refused sooner. The first example of these docs is such a text: the printed result
of `x` and 110 copies of `^-2` at 1 nests at most 115 parentheses and about 110 carets
(`drv/t110.res`), and it is refused. The `mod depth` doc states the units exactly ("more than
`LIMIT` units counted from the start of the text"), and the test's doc says "or 1 024 units in
all", so a reader who follows the pointer finds the rule. Suggested wording for the short form:
"more than 1 024 units of nesting (8 per parenthesis or call, 1 per `^`: 128 levels of
parentheses, or 1 024 of `^`)". The same wording would fit the tracker row.

## 4. Runs

| command | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic --no-fail-fast -- --test-threads=2` (debug, `sym_dbg.log`, 3 min 5 s) | unit 685 (5 ignored); `s1_depth` 14; `s1_finish1` 11 (1); `s1_fuzz_data` 7; `s1_payload` 2 (1); `s1_regressions` 26 (2); `s1_review2` 2; `s1_review3` 7 (7); `s1_review4` 4 (2); `s1_review4_memory` 1 (1); `s1_review5` 4 (2); `s1_review5_memory` 2 (2); `s1_review6` 4; `s1_singular_points` 9; doc 116 (11): 894 passed, 0 failed |
| `cargo test -j 2 --no-fail-fast -p rustmath-calculus -p rustmath-curves -p rustmath-functions -p rustmath-geometry -p rustmath-manifolds -p rustmath-plot3d` (`deps_test.log`) | 18 binaries, 1 556 passed, 0 failed, 35 ignored |
| `cargo test -j 2 -p rustmath-jupyter --lib`; `--test kernel_protocol -- --test-threads=1` (`jup_lib.log`, `jup_kernel.log`) | 120 passed (3 ignored); 4 passed |
| `cargo test --release -j 2 -p rustmath-symbolic --test s1_finish1 --test s1_depth -- --test-threads=2` (`rel_finish1_depth.log`) | `s1_depth` 14 passed; `s1_finish1` 11 passed (1 ignored), the three `fin2_*` tests included (build 53.5 s) |
| the same, `s1_finish1 -- --ignored --test-threads=1` (`rel_finish1_ignored.log`) | `b3f1_reparse_of_a_result_within_the_bar` passed (1.02 s) |
| rustfmt `--check` of `parser.rs`, `series.rs`, `s1_finish1.rs` | no diff on a line of this step (`s1_finish1.rs` clean) |
| width and dashes of the added lines | none over 100 characters, no em-dash |

`cargo tree -i rustmath-symbolic --workspace` lists the dependents tested above and
`rustmath-benchmarks` (not run: the step changes no signature and no public item, only two
message texts and docs). `rustmath-py` is outside the workspace.

## 5. Housekeeping

`git status --short` and the md5 of `git diff` (586f9ef4...) are the same at the start and at the
end (`status_before.txt`, `status_after.txt`, `diff_md5_before.txt`, `diff_md5_after.txt`).
`rustmath-misc/output.txt` is unchanged (md5 746308829575e17c3331bbcb00c0898b), no `.tmp_dir_*`
exists. `target/release` (169 MB) was deleted after the release runs, when no cargo or rustc of
this checkout was running. Free disk at the end: 5.6 GB.

## 6. Files

`fu2/s1impl/finminors2/`: `before/` (the files before the step, rebuilt), `step_a.diff`,
`step_b.diff`, `diff_now.patch`; `drv/` (`build.sh`, `p1.rs`, `gen.py`, `corpus.txt`, `cur.out`,
`head.out`, `ana.py`, `ana.log`, `c2.txt`, `c2.cmp`, `r1.rs`, `r1.log`, the input texts and
`t110.res`, `pe.rs`, `pe.log`); the logs of section 4.
