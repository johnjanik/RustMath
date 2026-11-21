# S1 finish, review fin1, lens: numeric and series semantics

Reviewer of S1 as it stands: main 28c4b69 (the S1 work in progress is 4763f31) plus the uncommitted
changes of the finish run (`git diff`: `parser.rs`, `series.rs`, `series_const.rs`,
`series_engine.rs`, `simplify.rs`, `repl.rs` docs, the test files). Bar: the narrowed bar of this
run. No repository file was edited (`git status --short` is the same before and after:
`fin1-semantics/status_before.txt`, `status_after.txt`). Scratch:
`fu2/s1impl/fin1-semantics/` (paths below are relative to it unless they start with `fu2/` or a
crate name).

## Verdict

Not approved. Two blockers in this lens. The tests pass in debug and in release.

- **B1 (bar (a)).** A power f^g with f of nonzero valuation is `NotLaurent`, a decided answer,
  whenever the exponent's series has a stored non-constant term, also when that term's
  coefficient is undecided and in fact 0. The function is then f^k, a Laurent series. The fix of
  review 6 m1 covers only the exponent whose single stored term is the constant term. The same m1
  input moved to an irrational point is still wrong: `(x-pi)^((1+x)/(1+x))` at pi is `NotLaurent`,
  and the function is x - pi. Measured: 100 of 390 rows of my grid at irrational points, all
  Laurent by the contour oracle.
- **B2 (bar (b)).** The parser's new depth limit refuses texts that HEAD parses. The literal
  -9223372036854775808 (i64::MIN), written as HEAD's negative literal (`x^-9223372036854775808`,
  `--9223372036854775808`, `+-9223372036854775808`), sets the "relaxed" flag. Any text with it and
  more than 128 nested levels after it is then a `ParseError`. HEAD's debug parser reads the same
  text on a 2 MB thread at 150 and 200 levels.

The rest of my lens holds:

- The owner's two rules hold at branch points reached through the new exact Gaussian values.
- The 46 new truncated-constant rows (class (h)) are what they claim. The outcome
  `PrecisionExhausted` holds at symbolic points too.
- The Gaussian inverse (m3) gives exact values that equal gp on 686 rows. It decides 12 of 13
  exact-zero identities, and the 13th stays undecided. It decides no complex-kernel constant
  wrongly.

## 1. Tests

| command | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic --no-fail-fast` (debug; `debug_tests.log`, 79 s) | unit 685 (4 ignored); `s1_depth` 11; `s1_fuzz_data` 7; `s1_payload` 2 (1 ignored); `s1_regressions` 26 (2 ignored); `s1_review2` 2; `s1_review3` 7 (7 ignored); `s1_review4` 4 (2 ignored); `s1_review4_memory` 1 (1 ignored); `s1_review5` 4 (2 ignored); `s1_review5_memory` 2 (2 ignored); `s1_review6` 4; `s1_singular_points` 9; doc 116 (11 ignored). 880 passed, 0 failed |
| the 13 release test binaries (built 06:44 to 06:45, after the last source edit at 06:24; copies in `relbin/`; `rel_tests.log`) | 685, 11, 7, 2, 26, 2, 7, 4, 1, 4, 2, 4, 9 passed; 0 failed |
| release `--ignored --test-threads=1` of the integration binaries (same log) | `s1_payload` 1 (3.34 s), `s1_regressions` 2, `s1_review3` 7, `s1_review4` 2, `s1_review4_memory` 1, `s1_review5` 2, `s1_review5_memory` 2; 0 failed. The unit binary's ignored tests (`fold_work_calibration`) were not rerun |

The tests of the first step are untouched in the diff: `s1_regressions.rs`, `s1_fuzz_data.rs`,
`s1_common/`, `data/s1_fuzz.tsv`. `s1_singular_points.rs` only gains the (h) coverage loop, its
arm and its test. `s1_singular_points.tsv` only gains 46 lines. In `s1_depth.rs`, the test
`b2_limit_with_error_of_a_signed_tower` builds the tree instead of parsing it, and its assertions
are unchanged.

## 2. Method

- Driver `drv/r7.rs`: review 6's `r6.rs` unchanged. It is compiled against this tree's rlibs: debug
  `target/debug/deps/librustmath_symbolic-b2048a251f3765f1.rlib`, which cargo reports fresh for
  the current sources, and release `rlib_rel/`, copied from `target/release/deps` at 06:46. Per
  row it runs `series_expansion` at budget 1e8, `taylor_with_budget` at 1e8 with its Display
  re-parsed, `limit_via_series(.., 5)` and `series_coefficients`, and prints each result in gp
  form.
- Oracle: review 6's contour-integral script `cauchy.gp` with `run6.py` (gp 2.15.4). It computes
  Laurent coefficients on two circles. `CONSISTENT` on a refused row means that the function is
  single-valued and analytic in the annulus, so it has a Laurent expansion there. The Display round
  trip compares `W` with `parse(Display(W))` at a point (`rt.py`).
- Debug on a 2 MB thread and release on an 8 MB thread give identical output (fields 1 to 17, md5)
  on all 8 files. There was no panic. The slowest row took 1.02 s in release.

| file | rows | content |
|---|---|---|
| `out/M.tsv` (`gen/gen_m.py`) | 1 802 | review 6 m1: bases of nonzero and of zero valuation, at 0, 1 and infinity, with exponents that are truncated integers, truncated with negative precision, decided non-integers, undecided integers, and non-constant |
| `out/pts.tsv` | 390 | m1 shapes (`(x-a)^g`, `((x-a)^2)^g`, `(1/(x-a))^g`, `sin(x-a)^g`, ...) for g in `(1+x)/(1+x)`, `2*(1+x)/(1+x)`, `sin(x)^2+cos(x)^2`, `exp(x)*exp(-x)`, `(2+x^2)/(2+x^2)`, `x/x`, at 13 points (0, rationals, I, 1+I, pi, sqrt(2), e, log(2), 2^(1/3), pi/4, asin(1/2)) |
| `out/hand1.tsv`, `hand2.tsv`, `b1.tsv` | 112, 66, 10 | an undecided zero U (`log(2)+log(3)-log(6)`, `atan(1/2)+atan(1/3)-pi/4`, `sin(1)^2+cos(1)^2-1`, `exp(1)*exp(-1)-1`) as a coefficient in powers, branch functions, log, 1/, sqrt, exp, the trigonometric and hyperbolic functions, erf, abs and sign |
| `out/G.tsv` (`gen/gen_g.py`) | 886 | m3: 31 Gaussian constants in 20 shapes, 13 exact-zero identities in 17 shapes, and 5 points written through the inverses in 9 shapes |
| `out/hand3.tsv`, `hand4.tsv`, `near.tsv`, `tc.tsv` | 87, 102, 5, 224 | complex kernels inside A + B*I; the owner's rules at branch points reached through Gaussian inverses; near misses of size 1e-40 to 1e-70; truncated constants at the branch points at pi, sqrt(2), e and 0 |
| `parse/pmin.rs` | 10 texts | B2, against HEAD's debug rlib `librustmath_symbolic-4cbaba7850391989.rlib` and the port |

## 3. B1: a power with an undecided exponent coefficient is reported as `NotLaurent`

`rustmath-symbolic/src/series_engine.rs:2056` (`pow_general`) and `:2084`
(`truncated_integer_exponent`). For f of nonzero valuation, `truncated_integer_exponent` handles
only two exponent shapes: one with no stored term, and one whose only stored term is the constant
term (`[(0, k)]`). Every other shape (`_ => None`) goes to `log(&f)`, which is `NotLaurent` for a
base of nonzero valuation. That covers two cases:

- an exact exponent with a non-constant term;
- a truncated exponent with more stored terms.

The rule does not ask whether the non-constant coefficients are decided nonzero. When they are
undecided and in fact 0, the exponent is the integer k and the function is f^k, which is a
Laurent series. `NotLaurent` then reports an undecided answer as decided. The module rule of
`series.rs:13` says "whether an exponent is an integer ... an undecided test is reported, never
guessed". The fix's own doc comment (`:2083`) and the header of `tests/s1_review6.rs` call
`NotLaurent` for `x^((1+x)/(1+x))` "a decided answer" under S1_BAR (a). The same header keeps the
gap on purpose: "a decided non-integer constant term, or another stored term, keeps `NotLaurent`".

Reproduced on the built code (`out/b1.tsv`, release `drv/r7_rel` on 8 MB and debug `drv/r7_dbg`
on 2 MB, identical). `series_expansion`, `taylor_with_budget` and `series_coefficients` all give
`NotLaurent` ("log has a logarithmic singularity at the point"):

| input, point | the function (U = 0 exactly) | engine |
|---|---|---|
| `x^(1+(log(2)+log(3)-log(6))*x)`, 0 | x | `NotLaurent` |
| `x^((log(2)+log(3)-log(6))/x)`, 0 | 1 | `NotLaurent` |
| `x^(2+(atan(1/2)+atan(1/3)-pi/4)/x^2)`, 0 | x^2 | `NotLaurent` |
| `sin(x)^(3+(sin(1)^2+cos(1)^2-1)*x)`, 0 | sin(x)^3 | `NotLaurent` |
| `(1/x)^(1+(log(2)+log(3)-log(6))*x)`, infinity | x | `NotLaurent` |
| `(x-1)^((exp(x)*exp(-x)-1)/x^3)`, 1 | 1 | `NotLaurent` |
| `((x-1)^2)^((exp(x)*exp(-x)-1)*(1+x))`, 1 | 1 | `NotLaurent` |
| `(x-pi)^((1+x)/(1+x))`, pi | x - pi | `NotLaurent` |
| `(x-sqrt(2))^(2*(1+x)/(1+x))`, sqrt(2) | (x - sqrt(2))^2 | `NotLaurent` |
| `(x-1/(pi+I))^((1+x)/(1+x))`, 1/(pi+I) | x - 1/(pi+I) | `NotLaurent` |
| controls: `x^((1+x)/(1+x))` at 0; `x^(1+(log(2)+log(3)-log(6)))` at 0; `x^(1+x)` at 0 | x; x; not Laurent | `PrecisionExhausted`; `UndecidedZero`; `NotLaurent` |

The last three m1 rows are review 6's m1 input moved to a point that is not rational. At pi the
series of `(1+x)/(1+x)` is `pi/(1+pi) + 1/(1+pi) + (...)h + ...`. Its constant term is 1 but
undecided, and every higher coefficient is an undecided 0, so the exponent has more than one
stored term. Grid `out/pts.tsv`:

- at 0, at 1/3, 2 and -5/7, and at I and 1+I: no row is `NotLaurent`. The outcomes are
  `PrecisionExhausted` or `UndecidedZero`, plus 5 values at 0;
- at pi, e, log(2), 2^(1/3), pi/4 and asin(1/2): 15 of 30 rows each are `NotLaurent`;
- at sqrt(2): 10 of 30 rows are `NotLaurent`.

The oracle finds all 100 `NotLaurent` rows `CONSISTENT` (`out/pts.res`), with the lowest exponent
k·v of f^k. In class M, the 6 rows at point 1 with `exp(x)*exp(-x)-1` are the same defect. The
other 113 `NotLaurent:CONSISTENT` rows of M are at infinity with `sin(x)`, `exp(x)` or
`sqrt(1+x)` in the input. That is the documented convention: a subexpression with no Laurent
expansion at the point.

Every other rule meets an undecided coefficient honestly (`out/hand1.tsv`, `hand2.tsv`, 178 rows).
`asin`, `acos`, `atanh`, `acosh`, `atan` and `asinh` at a branch point plus U x give
`UndecidedZero`, and so do `log(U x + x^2)`, `1/(U x + x^2)`, `sqrt(U x + x^2)`, `exp(U/x)`,
`sin(U/x)`, `tan(U/x)`, `erf(U/x)`, `(1+x)^(U/x^2)` and `2^(U/x)`. The truncated constants at the
branch points at pi, sqrt(2) and e (`out/tc.tsv`, 224 rows) are `UndecidedZero` or
`PrecisionExhausted`, never `NotLaurent`. Only `pow_general` with a base of nonzero valuation
decides.

gp (`b1_refs.gp`, `gp -q < b1_refs.gp`, `\p 200`):

- the four U are 0 to 211 digits (binary exponents -703 and -704);
- `x^(1+(log(2)+log(3)-log(6))*x) - x` is 0.E-213 on |x| = 1/16;
- `(x-1)^((exp(x)*exp(-x)-1)/x^3) - 1` is 9.9E-212 on |x - 1| = 1/16;
- the control `x^(1+x) - x` is 0.018 there.

The identities are exact: log 6 = log 2 + log 3, Machin's atan(1/2) + atan(1/3) = pi/4,
sin^2 + cos^2 = 1, and e·e^-1 = 1.

Required change. In `pow_general`, for f of decided nonzero valuation and g not an exact constant,
check g's stored terms before taking `log(&f)`:

1. A decided non-integer constant term gives `NotLaurent`, as now.
2. A non-constant term with a decided nonzero coefficient gives `NotLaurent`: the log path, which
   is correct then.
3. Otherwise, the first non-constant term with an undecided coefficient gives `UndecidedZero` of
   that coefficient. This applies to exact and to truncated g.
4. Only then does the present `truncated_integer_exponent` logic run.

Add these as regression tests, in both profiles:

- the B1 table rows: never `NotLaurent`, a value or `DivisionByZero`;
- `x^(1+x+(log(2)+log(3)-log(6))*x)` stays `NotLaurent`, because its h coefficient 1 + U is decided
  nonzero.

Correct the header of `s1_review6.rs` ("another stored term keeps `NotLaurent`") to say "another
stored term with a decided nonzero coefficient".

## 4. B2: the depth limit refuses a text that HEAD parses

`rustmath-symbolic/src/parser.rs:257` (`integer_literal`). `depth::relax()` runs for any digit
string above i64::MAX, with leading zeros ignored, so it also runs for 9223372036854775808.
HEAD's `integer` parsed `-9223372036854775808` as i64::MIN, with the sign inside the literal. It
read that literal wherever its atom saw a '-' directly before the digits: after `^`, or as the
second sign of an operand (`--N`, `+-N`, `- -N`). Once relaxed, any level past 1 024 units is
refused: 128 parentheses or calls, or 1 024 carets. That holds whether the level is in the rest of
the text or comes after the literal inside a deep nest.

Reproduced (`parse/pmin.rs`, `pmin_head` against HEAD's debug rlib, `pmin_dbg` and `pmin_rel`
against the port; `nest` = 150 nested parentheses around x or y):

| text | HEAD, debug 2 MB, 150 and 200 levels | port, debug 2 MB, 150 levels | port, release 8 MB, 150 levels |
|---|---|---|---|
| `x^-9223372036854775808 + nest(x)` | Ok `((x ^ -9223372036854775808) + x)` | ParseError "the expression nests too deeply for its syntax ..." | ParseError |
| `--9223372036854775808 + nest(x)` | Ok | ParseError | ParseError |
| `+-9223372036854775808 + nest(x)` | Ok | ParseError | ParseError |
| `2*--9223372036854775808*nest(y)` | Ok | ParseError | ParseError |
| `nest(x^-9223372036854775808 + y)` | Ok | ParseError | ParseError |
| `x^-09223372036854775808 + nest(x)` | Ok | ParseError | ParseError |
| controls: `x^-9223372036854775807 + nest(x)`, `nest(x)`, `x^-9223372036854775808` | Ok | Ok | Ok |
| `x^-9223372036854775809 + nest(x)` | ParseError | ParseError | ParseError |

At 126 levels the port reads every row. This is a regression for a public function that S1
changed (bar (b)). It was introduced by this run's fix; at 4763f31 there was no limit. The parser
doc (`parser.rs:189`, "only texts HEAD refuses too") and the report's "what it refuses" are false
for it. `p6_texts_read_by_head_keep_their_reach` tests the literal i64::MAX but not HEAD's
negative literal i64::MIN.

Required change. Do not relax for the magnitude 2^63 when the digits are those of a negative
literal in HEAD's sense. `sign_run` already computes this: the last sign is a '-' directly before
a digit. Pass it to `integer_literal`, for example as a per-thread flag set by `operand_signs` and
`exponent_signs` when `literal` is true and cleared by `integer_literal`. Add the five Ok texts
above, at 150 levels on a 2 MB debug thread, to `p6_texts_read_by_head_keep_their_reach`, and
keep `x^-9223372036854775809 + nest(x)` refused.

## 5. What holds in this lens

**The owner's two rules** (`out/hand4.tsv`, 102 rows; `G.tsv` class Z; `tc.tsv`):

- At branch points written through Gaussian inverses (`(1+I)/(1-I)` is I, `-2/(1-I)+1` is -I,
  `2/(1+I)+I` and `1/(1+I)+1/(1-I)` are 1, `-(2+I)/(3-I)*(1-I)` is -1):
  - an exact constant argument gives the value. Examples: `asin(1)` is pi/2, `acos(1)` is 0,
    `acosh(-1)` is i pi, `asinh(I)` is i pi/2, `atanh(I)` is i pi/4, `atan(1)` is pi/4.
  - an exact constant at a pole of the constant gives `DivisionByZero`: `atan(+-I)` and
    `atanh(+-1)`. gp fails on all 8 such rows.
  - a non-constant argument at a branch point gives `NotLaurent`. The oracle finds all 26 such
    rows non-Laurent.
  - a truncated constant at a branch point gives `PrecisionExhausted`.
  - at the regular points I and -I, asin, acos, acosh and atanh have values. `asin(I + x)` is
    `asin(I) + x/sqrt(2) + ...`. Of the 62 values, 60 pass the oracle and 2 are on the principal
    side of a cut.
- The exact zeros of class Z put at a branch point (`atan(I+D+x)`, `atanh(1+D+x)`) are
  `NotLaurent`, and `asin(1+D)+x` is `pi/2 + x`.

**The singular-point matrix, class (h)** (`tests/data/s1_singular_points.tsv`, the last 46 rows):

- there are two rows for each of the 23 (function, branch point) pairs, as the coverage test
  asserts. Each argument is the constant c up to a truncated zero;
- the expectation is `PrecisionExhausted` and the limit `Unknown`. The test passes in both
  profiles;
- the first 772 lines of the file are unchanged (the diff adds 46 lines only);
- at pi, sqrt(2) and e the same shapes are `UndecidedZero` or `PrecisionExhausted`, never a value,
  `NotLaurent` or `DivisionByZero` (`tc.tsv`). The only exception is c = 0 under a non-truncated
  `0*`, which is the exact 0: `log` gives `DivisionByZero` and `sqrt` gives 0, which is rule 1.

**m3, the Gaussian inverse** (`series_const.rs:2579`):

- of the 700 class G values, 686 pass the oracle. 13 of the other 14 have a pole or a branch
  point at distance 1e-30 from 0 (the constants `1/(10^30+I)` and `1/(1+10^-30*I)`), inside the
  oracle's circle. I checked the 5 pole rows and `log(c + x)` against gp's exact series instead
  (`out/tiny.gp`, `out/tiny2.gp`). For example, `1/(x + 1/(10^30+I))` gives
  `[10^30 + I, -(10^60 - 1) - 2·10^30 I, ...]` in both. The 14th row is `x/D` with D = 0 under
  the divisor convention, where gp fails.
- 12 of the 13 exact-zero identities are decided zero: `2/(1+I)-(1-I)`, `1/(pi+I)-(pi-I)/(pi^2+1)`,
  `1/(sqrt(2)+I)+1/(sqrt(2)-I)-2*sqrt(2)/3`, `1/(2^(1/3)+I*3^(1/2))-(2^(1/3)-I*3^(1/2))/(2^(2/3)+3)`
  and others. `1/(x+D)` is x^-1, `x/D` is `DivisionByZero` (gp fails), and `x^(1+D)` is x.
  `1/(cos(1)+I*sin(1)) - cos(1) + I*sin(1)` stays `UndecidedZero`, which is honest: its N is
  `cos(1)^2+sin(1)^2`.
- the near misses (`near.tsv`) are exact. `1/(x + 1/(1+I) - (1-I)/2 + 10^-40*I)` is
  `-10^40 I + 10^80 x`, as gp gives.
- constants with complex kernels in A or B (`hand3.tsv`, 87 rows): `sqrt(-1)`, `(-1)^(1/3)`,
  `(-8)^(1/3)`, `exp(I*pi/2)`, `log(-1)`, `asin(2)`, `acos(2)`, `atanh(2)`, `acosh(-2)`,
  `sqrt(I)`, `(3+4*I)^(1/2)`, `log(I)` and `exp(I)`. N is never decided, so they keep the kernel
  1/c and are `UndecidedZero` or a value under the documented divisor convention. No wrong value
  was found. `(-1)^(1/3) - 1/2 - I*sqrt(3)/2` is one example: it is 0 under the principal branch,
  and it would give a wrong finite inverse if the cube root had a real enclosure.
- the Display round trip agrees with the value at a point on 698 of 700 class G rows, 62 of 62
  rows of `hand4` and 5 of 5 rows of `pts` (`rt_*.res`). gp itself fails on the other 2: an exp
  overflow, and a divisor that is 0 under the convention.

**m1 at rational, Gaussian and zero points:** every exponent of the m1 class is
`PrecisionExhausted` or `UndecidedZero`, as the fix intends. Decided non-integer exponents stay
`NotLaurent`, and the oracle confirms all 678 `NotLaurent:NONLAURENT_OK` rows of M. No row of M
gives a value that the oracle rejects (500 `PASS`, 29 on the principal side of a cut).

**Doc comments on consumer costs:** I read the changed docs of `MAX_RESULT_TREE`,
`MAX_RESULT_TEXT`, `MAX_RESULT_FOLD_WORK`, `MAX_RESULT_MEMO_WORDS` and `TextBound::total`, of
`simplify` (`simplify.rs:405`), of `StructureLimit::FoldWork` and `FoldPayload`, of
`taylor_with_budget`, of the driver's tree-check comment, and of `TAYLOR_WORK` in `repl.rs`. Each
gives measured numbers and says the consumers are not bounded. A grep for "bounded" in the
touched files found no remaining claim that a consumer of a result is bounded. The REPL diff
changes doc lines only.

## 6. Minor issues

- **n1 (carried, review 2 n1, frozen design §2.13).** `limit_with_error` loses HEAD's symbolic
  limits. `limit_with_error(pi + x, x, 0, Both, 3)` was `(Finite(pi + 0), Some)` at HEAD and is
  `(Unknown, None)` now. `exp(x)` at 1 is the same: HEAD gave `Finite(exp(1))`. Measured with
  `lim/lwe.rs` against HEAD's debug rlib and the port. Review 2 classed this below the bar: the
  design states it, `Unknown` is honest, and HEAD's `Expr::limit` aborted (review 1 B2). I keep
  that classification. The tracker row should still say "`Finite` only for a rational limit".
- **n2.** The doc comment of `truncated_integer_exponent` (`series_engine.rs:2074` to `:2083`)
  states the rule correctly ("it has a logarithmic term v (g - k) log h otherwise"). The function
  applies it only to the single-term case. This is fixed with B1.
- **n3.** `parser.rs:189` ("A text without these constructs keeps at least HEAD's reach") and the
  report's "What it refuses: only texts HEAD refuses too" do not hold for B2's texts. This is
  fixed with B2.

## 7. Reproduction

```
F=/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/s1impl/fin1-semantics
cd $F/drv && ./build.sh r7.rs rel r7_rel && ./build.sh r7.rs dbg r7_dbg
cd $F/out && ../drv/r7_rel b1.tsv | cut -f1,3,5          # B1 table
cd $F/out && ../drv/r7_rel pts.tsv > pts.out && NPROC=2 python3 -I ../run6.py pts.out pts   # 100 NotLaurent:CONSISTENT
cd $F && gp -q < b1_refs.gp                                # the constants are 0; the functions are x and 1
cd $F/parse && ../drv/build.sh pmin.rs head pmin_head && ../drv/build.sh pmin.rs dbg pmin_dbg && ./pmin_head 150 && ./pmin_dbg 150   # B2
cd $F/out && python3 -I ../gen/gen_m.py > M.tsv && ../drv/r7_rel M.tsv > M.out && NPROC=2 python3 -I ../run6.py M.out M
cd $F/out && python3 -I ../gen/gen_g.py > G.tsv && ../drv/r7_rel G.tsv > G.out && NPROC=2 python3 -I ../run6.py G.out G
cd $F/lim && ../drv/build.sh lwe.rs head lwe_head && ../drv/build.sh lwe.rs rel lwe_rel && ./lwe_head && ./lwe_rel   # n1
```

Housekeeping:

- `rustmath-misc/output.txt` is unchanged (md5 746308829575e17c3331bbcb00c0898b), and no
  `.tmp_dir_*` exists.
- `target/release` was not deleted. The concurrent fin1-resources reviewer was building into it at
  07:00 (`cargo test --release` and its `pr_rel` probes were running). The release rlibs and test
  binaries I used are copied to `rlib_rel/` and `relbin/`.
- Free disk at the end: 7.9 GB.
