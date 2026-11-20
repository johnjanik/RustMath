# S1 implementation, review 5, lens: numeric and series semantics

Reviewer of the uncommitted S1 change in `/home/user/RustMath` (branch `claude/gracious-gates-e6b6l7`,
HEAD 8a0b310) after fix-up round 4. No repository file was edited: `git status --short` lists the
same 35 entries before and after. Scratch: `fu2/s1impl/rev5-semantics/`. Paths below are relative to
`/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/s1impl/`.

## Verdict

Not approved in this lens. One blocker (B1). The tests pass in debug and in release. The blockers of
round 4 (resources B1, B2) are fixed; I reproduced their inputs on the built code.

- **B1.** When the expansion variable is named `pi` or `I` (the parser maps `i` to `I` too), the
  constants that the engine makes itself (the exact values of fix-up round 4, and erf's 2/sqrt(pi))
  are printed with the variable's name. `taylor(acos(0*pi), pi, 0, 2)` returns `1/2 * pi`, where the
  answer is the constant pi/2; then `subs(t, pi, 0)` gives 0 instead of 1.5707963... HEAD returns
  `arccos(0)` there. A wrong result (S1_BAR a) and a regression against HEAD on inputs HEAD handles
  (S1_BAR b). It is new in round 4 for the inverse functions (`pi_times`).
- Apart from B1 I found no wrong value. A fuzzer of my own with a numerical oracle (4,800 rows,
  section 5) and a parser fuzzer (7,500 texts, section 7) found none; every row that the oracle
  flagged is a branch cut crossed at the point or gp's own rounding, checked one by one.
- The owner's two rules hold on all my rows: no non-constant argument at a branch point gives a value
  or `DivisionByZero`; every `DivisionByZero` (227 fuzz rows, 115 matrix rows) is a division by an
  exact zero, confirmed by gp at two precisions.
- `simplify` on 40,000 random DAGs with shared nodes and outside owners is structurally identical to
  HEAD's (section 8).

## 1. Tests

| command | result |
|---|---|
| `cargo test -j 2 -p rustmath-symbolic --no-fail-fast` (debug; `rev5-semantics/debug_tests.log`, 44 s) | unit 683 passed, 4 ignored; `s1_depth` 9; `s1_fuzz_data` 7; `s1_payload` 2 (1 ignored); `s1_regressions` 26 (2 ignored); `s1_review2` 2; `s1_review3` 7 (7 ignored); `s1_review4` 4 (2 ignored); `s1_review4_memory` 1 (1 ignored); `s1_singular_points` 7; doc 116 (11 ignored); 0 failed |
| the ten release test binaries (`rev5-semantics/rel_tests.log`) | 683 (4 ignored), 9, 7, 2 (1), 26 (2), 2, 7 (7), 4 (2), 1 (1), 7; 0 failed |
| release `--ignored --test-threads=1` (`rev5-semantics/rel_ignored.log`) | `s1_payload` 1 (3.38 s); `s1_regressions` 2 (0.78 s); `s1_review3` 7 (1.87 s); `s1_review4` 2 (17.9 s, with the bisections); `s1_review4_memory` 1 (1.12 s); `fold_work_calibration` passed (46.8 s); 0 failed |
| `cargo test -j 2 -p rustmath-jupyter --lib -- taylor simplify_of_a_variable_chain` (`rev5-semantics/jup_taylor.log`) | 7 passed (`test_taylor_engine`, `_order_checked`, `_leaf_text_cap`, `_simplify_of_kernel_sources`, `_display_linear`, `_shared_dag`, `test_simplify_of_a_variable_chain_memory`), 0 failed |

The release tree was fresh when I ran it (built from the current sources; cargo reported every
artifact fresh). It was deleted afterwards by the other reviewer of this round; my release drivers
were linked before that and are kept.

## 2. Tests of the first step

Compared with the harness copy of the first step (`protoharness/work/src`):

- Byte-identical: `tests/s1_common/check.rs`, `tests/s1_common/eval.rs`, `tests/data/s1_fuzz.tsv`,
  `tests/data/gen_s1_fuzz.py`, `tests/data/s1_gp.py`.
- `tests/s1_regressions.rs` (mtime 20:02): without comment lines, `#[path]` lines and whitespace, it
  differs from the harness copy in one character, a trailing comma rustfmt added (`Ok(())},);`).
- `tests/data/s1_singular_points.tsv` (740 lines): its first 106,450 bytes are the harness copy, and
  it starts with round 3's file (`fix4/s1_singular_points.tsv.before`, 680 lines), byte for byte.
  Round 4 appended the 60 (f) rows.
- `tests/s1_common/mod.rs` differs only in the doc paragraph on the removed feature gate.
- `tests/s1_singular_points.rs` changed in round 4: class (f) added (a test, the coverage loop, the
  expectation arm `("f:", Value)`). I read the whole file; no earlier assertion was weakened.

No expected outcome of the first step changed.

## 3. The blockers and minors of round 4

| item | status | evidence (`rev5-semantics/r4b/`) |
|---|---|---|
| resources B1 (`simplify` of a kernel source, 35 s to 170 s) | fixed | debug probe (round 4's `probe5.rs` rebuilt against the current rlib): `sin(K/3 x 12 000)/(1-x)`, `sin(K/7 x 12 000)`, `g(K/3 x 12 000)`, `sin(K/3 x 30 000)` at 0, order 1: `LimitExceeded(FoldWork)` in 0.36 to 0.65 s; the control `exp(x + K/3 x 12 000)` is Ok, `simplify` 0.000 s (`r4b_b1_dbg.txt`). Release: `b1r4_simplify_of_kernel_source_chains_release` and `b1r4_simplify_at_the_edge_of_the_fold_cap_release` pass. |
| resources B2 (`simplify` memo held 754 MB) | fixed | `chain_t2` rebuilt (debug, 64 MiB thread): the held `* 3` chain of 6,000 links, `sin(a_6000)/(1-x)`: `taylor_with_budget` Ok, `simplify` 2.2 s, VmHWM 11 MB (round 4: 754 MB); the `/ 3` chain is refused (`FoldWork`) (`r4b_b2_dbg.txt`). Release: `b2r4_taylor_then_simplify_of_a_held_chain_release` passes. |
| semantics m1 (exact values at branch points) | done | the 60 (f) rows and the 14 + 4 rows of `m1r4_*` pass; my oracle agrees on all (f) rows (section 6); the values are the principal ones (arcsin(+-1) = +-pi/2, arccos(-1) = pi, arccos(0) = pi/2, arccosh(-1) = i pi, arccosh(0) = i pi/2, arctan(+-1) = +-pi/4, arcsinh(+-i) = +-i pi/2; gp `asin(1)`, `acos(-1)`, `acos(0)`, `acosh(-1)`, `acosh(0)`, `atan(1)`, `asinh(I)`). But the fold introduced B1. |
| semantics m2 (abs and sign of real coefficients) | handoff | `m2r4_abs_and_sign_of_real_coefficients_stay_refused` pins the typed refusal; a scope decision for the owner, as the report says. No change in this lens. |

## 4. Blocker

### B1. A variable named `pi` or `I` collides with the constants the engine makes

S1_BAR (a), a wrong result, and (b), a regression against HEAD on inputs HEAD handles.

The engine accepts any symbol as the variable. With the variable `pi`, the input's own `pi` is the
variable (`sin(pi)` in `pi` at 0 is `pi - pi^3/6`, right). But the engine also creates the constant
pi itself, as the kernel of the symbol `pi`: `pi_times` (`series_const.rs:3459`, used by the exact
values of round 4 at `:3553` and `:3590` to `:3597`) and erf's 2/sqrt(pi) (`series_engine.rs:1186`).
With the variable `I`, the same holds for the constant i of `pi_times(.., true)` (arccosh(-1) = i pi,
arccosh(0) = i pi/2). The coefficients are right inside the engine (a kernel and the variable are
distinct there), but `to_expr`, `taylor_with_budget` and the REPL render both as the same symbol, so
the returned expression is a different function of the variable.

Measured on the built code: `drv/r5_rel` (release, 8 MB thread) and `drv/r5_dbg` (debug, 2 MB
thread) give the same output (`hand/vpi_rel.out`, `hand/vi_rel.out`, `hand/vpi_dbg.out`); HEAD is
`drv/head_taylor` against HEAD's rlib (`hand/vpi_head.out`, `hand/vi_head.out`).

| input, variable, point, order | port (`taylor_with_budget`) | right answer | HEAD |
|---|---|---|---|
| `acos(0*pi)`, pi, 0, 2 | `(1/2)*pi` | the constant pi/2 | `arccos(0)` |
| `acos(pi - pi)`, pi, 0, 2 | `(1/2)*pi` | pi/2 | `arccos(0)` |
| `atan(pi^0)`, pi, 1, 2 | `(1/4)*pi` | pi/4 | `arctan(1)` |
| `atan(pi/pi)`, pi, 1, 2 | `(1/4)*pi` | pi/4 | `arctan(1)` |
| `pi*acos(pi*0)`, pi, 1, 2 | `(1/2)*pi + (1/2)*pi*(pi - 1)` | (pi/2) + (pi/2)(h) with h = pi_var - 1 | `arccos(0) + arccos(0)*(pi - 1)` |
| `acos(0*pi) + sin(pi)`, pi, 0, 3 | `(1/2)*pi + pi - (1/6)*pi^3` | pi/2 + pi_var - pi_var^3/6 | `arccos(0) + pi - 1/6*pi^3` |
| `acos(-1) + pi`, pi, 0, 2 | `pi + pi` | the constant pi plus the variable | `0/0` terms (HEAD's bug) |
| `asin(1)*pi`, pi, 0, 2 | `(1/2)*pi*pi` | (pi/2) pi_var | `0/0` terms |
| `erf(pi)`, pi, 0, 3 | `2*pi^(-1/2)*pi - (2/3)*pi^(-1/2)*pi^3` | (2/sqrt(pi)) pi_var - ... | `500/443*pi - 500/1329*pi^3` (a float approximation) |
| `acosh(-I^0)`, I, 0, 2 | `pi*I` | i pi | `acosh(-1)` |
| `acosh(0*I)`, I, 0, 2 | `(1/2)*(pi*I)` | i pi/2 | `acosh(0)` |
| `acosh(-1) + I`, I, 0, 2 | `pi*I + I` | i pi + I_var | `acosh(-1) + I` |
| `acosh(0*I) + sin(I)`, I, 0, 3 | `(1/2)*(pi*I) + I - (1/6)*I^3` | i pi/2 + I_var - I_var^3/6 | `acosh(0) + I - 1/6*I^3` |

Through the REPL (debug, `ReplContext::eval` on a 64 MiB thread, `drv/repl_probe`;
`hand/repl_pi.out`, `hand/repl_pi2.out`, `hand/repl_i.out`):

```
t2 = taylor(acos(0*pi), pi, 0, 2)          -> (1/2 * pi)
subs(t2, pi, 0)                            -> (1/2 * 0)        (acos(0) = 1.5707963...)
t3 = taylor(atan(pi^0)*pi, pi, 1, 2)       -> ((1/4 * pi) + ((1/4 * pi) * (pi - 1)))
subs(t3, pi, 1)                            -> 1/4              (atan(1)*1 = 0.7853981...)
t = taylor(acos(-1) + pi, pi, 0, 2)        -> (pi + pi)
subs(t, pi, 0)                             -> (0 + 0)          (acos(-1) = 3.1415926...)
t = taylor(acosh(0*i) + i, i, 0, 2)        -> ((1/2 * (pi * I)) + I)
subs(t, i, 0)                              -> ((1/2 * (pi * 0)) + 0)   (acosh(0) = 1.5707963...*I)
```

The variable `i` is ordinary input: the parser maps `i` to the symbol `I` (`parser.rs:230`), and
`taylor(1/(1-i), i, 0, 3)` is right (`1 + I + I^2 + I^3`). The module doc (`series.rs:32`) says that
free symbols other than the variable are generic parameters; nothing says the variable may not be
`pi` or `I`, and nothing refuses it.

Scope. HEAD handles the first, second, third, fourth, fifth, sixth, tenth to thirteenth rows
right (it keeps the kernel, `arccos(0)`, `arctan(1)`, `acosh(-1)`); round 3 did too (a kernel
`arccos(0)`). Round 4's exact values made them wrong. The erf row is older (proto13), and HEAD's
value there is only a float approximation. The limit path is not affected (it reads the constant
term inside the engine, and `Finite` only for a rational one). `series_expansion`'s `terms` hold the
constant as `pi`, which a caller cannot tell from the variable either.

Required (a property): no result of a public entry point (`taylor_with_budget`, `series_expansion`
and `to_expr`, `taylor`, `laurent`, `series_coefficients`, the REPL arm) writes a constant with the
name of the variable. Ways that fit: refuse a variable named `pi`, `e` or `I` with a typed error
(for example `InvalidPoint("the variable has the name of a constant")`; `e` is not produced by the
engine today, so it may stay allowed), which turns HEAD's right `taylor(sin(pi), pi, 0, 3)` into a
typed refusal; or, for such a variable, keep the round-3 kernels (`arccos(0)`, `arccosh(-1)`, ...) and
write erf's constant without the symbol (such as `2/sqrt(arccos(-1))`).

Regression tests (written before the fix, both profiles): the rows of the table above through
`taylor_with_budget` with the REPL budget: a typed refusal, or a result `r` with
`r.substitute(var, point)` (simplified and evaluated numerically) equal to the function at the
point, f(a): pi/2 for `acos(0*pi)` at 0, pi/4 for `atan(pi^0)` at 1, pi for `acos(-1) + pi` at 0
(gp: `acos(0)` = 1.5707963267948966192, `atan(1)` = 0.78539816339744830962, `acos(-1)` =
3.1415926535897932385), i pi/2 for `acosh(0*I)` at 0 (gp: `acosh(0)` = 1.5707963267948966192*I),
2/sqrt(pi) for the first derivative of `erf(pi)` (gp: `2/sqrt(Pi)` = 1.1283791670955125739); and a
REPL test with `t = taylor(acos(0*pi), pi, 0, 2)` and `subs(t, pi, 0)`.

## 5. My fuzzer: a numerical residual oracle

Driver `drv/r5.rs` (from round 4's `r4.rs`; per row `series_expansion` with the budget 1e8,
`taylor_with_budget`, the re-parsed `Display`, `simplify`, `big_o`, `limit_via_series(.., 5)`, the
gp form of the input as the crate parsed it). Generator `gen5.py`, runner `run5.py`, gp helpers
`h5.gp`.

The oracle differs from review 9's and from rounds 1 to 4 (which compared gp's series): it uses only
gp's scalar functions. For an Ok row with terms S(h) and precision P, it evaluates
r(h) = |f(a + h) - S(h)| / |h|^P at h = +-10^-e, +-10^(2-e), +-10^(4-e) (exact rationals; e such
that the smallest |h|^(P+m) is about 1e-150, m the pole order), at 600 digits. A wrong coefficient at
exponent k < P makes r grow 100^(P-k)-fold per step; the row passes when r grows at most 30-fold
per step, on both sides. So gp's float `Pi` at a pole (`tan(x)` at pi/2) does not matter, and the
check covers points where gp's series fails. A row that passes on one side only is a branch cut
crossed at the point; each was checked against the principal value at the point. `DivisionByZero`
rows: f is evaluated at a + 1/7, a - 1/11, a + 1/1000 at 150 and 300 digits; an exact zero divisor
makes gp fail or the value unstable between the two precisions.

Classes (`gen5.py`, seeds 501 to 506 and 511 to 516, 400 rows each):

- G: compositions around round 4's exact values reached through exact-constant arguments (`asin(c)`,
  `acos(-c)`, `acosh(z)`, `atan(-c)`, `asinh(I*c)`, ... with c and z seven exact-1 and six exact-0
  shapes), inside 32 outer shapes (`tan(C/2 + G)`, `1/(C - C2 + x)`, `(1 + G)^C`, `log(C + 2 + G)`, ...)
  at 0, 1/3, -1/2.
- P: expansions at points written through those values (`asin(1)`, `2*atan(1)`, `acos(-1)/6`,
  `acosh(-1)`, `asinh(I)`, `2*acosh(-1)`, ...) of 40 functions with poles, zeros and cuts there.
- Q: the owner's rules where the branch point of an outer function is reached through the values:
  `asin(2*C/pi + G)`, `atanh(4*C/pi + G)`, `acosh(C/(I*pi) + G)`, `atan(C/(I*pi/2) + G)`,
  `log(C - V + G)`, `(C - V + Z)^(x-1)`, ... with G vanishing at 0 and Z an exact zero.
- R: random analytic trees at pi/6, pi/4, pi/3, 2pi/3, -pi/4, i pi/3 written as `asin(1/2)`,
  `atan(1)`, `acos(-1/2)`, `acosh(-1)/3`, ...

| outcome (4,800 rows) | rows |
|---|---|
| Ok, the oracle passes on both sides (11 of them printed on the next gp line, read by hand) | 3,720 |
| Ok, passes on one side only: atan, asinh at a point of their cut (i pi/2, 2i, -2i, i pi/3 with Im > 1) crossed by the real direction; the side is the one of the principal value at the point (arctan(iy) = pi/2 + ..., asinh(-2i) = -log(2 + sqrt 3) - i pi/2) | 17 |
| Ok, flagged: all five `(atan(+-1) -+ pi/4 + Z)^(1/3)`, where the engine gives 0 (no term) and gp's float atan(1) - Pi/4 is 1e-600 | 5 |
| `DivisionByZero`, confirmed exact zero by gp | 227 |
| `NotLaurent` (all checked by shape: sqrt, log, cube roots and `u^(x+1)` of a series of positive valuation, `asin(1 + G)`, `atanh(1 + G)`, `atan(-i + G)`, `asinh(i + G)`, `acosh(+-1 + G)`, `log(sin(x))` at pi, `sqrt(1 - sin(x))` at pi/2, `log(1 + cos(x))` at pi, `x^(pi/2)`) | 238 |
| `UndecidedZero` (complex kernels without an enclosure: cos(2 pi i), 1 + exp(i pi), cosh(i pi/2), ...) | 331 |
| `PrecisionExhausted` (`cos(x - x)` and its kin are truncated series by the documented convention, so `asin(cos(x-x)*x^0)` is a truncated 1) | 256 |
| `BudgetExceeded` 5, `LimitExceeded(TreeSize)` 1 (order 7 and 8 at irrational points with kernel coefficients) | 6 |
| a wrong coefficient, a value where the function has no Laurent series, a panic | 0 |

- The owner's two rules: no Q row with a non-constant argument at a branch point gives a value or
  `DivisionByZero`; every `DivisionByZero` row is `x/(F(c) - v)`, `1/(F(c) - v + Z)`, `log(F(c) - v
  + Z)`, `(F(c) - v + Z)^(-1/2)` or `^(x-1)`, `atanh(4*atan(c)/pi + Z)` or `atan(asin(c)/(i pi/2) + Z)`:
  a division by an exact zero or an exact pole of a constant.
- The limit path: on the 1,870 Ok rows of seeds 501 to 506, 362 `Finite` values, each equal to the
  constant term (or 0 with no constant term) of a series of valuation >= 0; no `Finite` on any
  refused row or at a pole.
- `taylor_with_budget` renders the same text as `to_expr` on all 1,870 rows; the re-parsed `Display`
  and `simplify` of the result equal S(1/997) at a + 1/997 on 1,799 of the 1,800 rows checked (the
  last was not printed, the gp line split; `fz/rt.gp`).
- Release (`r5_rel`) and debug on 2 MB threads (`r5_dbg`) give the same output on the 2,400 rows of
  seeds 501 to 506 (`fz/all_g.rel`, `fz/all_g.dbg2m`); slowest row 1.4 s in debug.

Hand probes (`fz/hreg.tsv`, `fz/trap.tsv`, `hand/*.tsv`):

- Regular and cut points of the six inverse functions (30 rows: `atan(1 + x)`, `acos(x)` at 0,
  `acosh(x)` at 0, -1/2, -2, 2i, -2i, -1/2 - i, `asin(x)` at -2, `atanh(x)` at 2 and -3, ...): all
  pass the oracle (four one-sided at imaginary-axis cuts, the principal side).
- Branch traps with the new values (30 rows): `asin(sin(acos(-x^0)))` is 0, `atan(tan(3*atan(x^0)))`
  is -pi/4 (not 3pi/4), `asin(sin(acos(-x^0) + x))` is -x, `1/(cos(acosh(0*x)*2/I) + 1 + x)` is 1/x;
  `1 + cosh(i pi)`, `exp(2 pi i) - 1`, `log(exp(3 pi i)) - 3 pi i` (which is -2 pi i, not 0) and
  `sqrt(exp(2 pi i)) + 1` are `UndecidedZero`, never decided; `log(exp(3*acosh(-x^0)))` keeps the
  kernel. No wrong zero decision in either direction.
- Powers (20 rows): `(acos(-x^0)/pi - 1)^r` is 0 for r = x + 1, `DivisionByZero` for x - 1,
  `NotLaurent` for x and for I + x; `0^(acos(-x^0) - pi)` is 1, `0^(acosh(-x^0))` `NotLaurent`,
  `x^(2*acos(-x^0)/pi)` is x^2, `x^(acos(0*x))` `NotLaurent`.
- Powers of I at the poles (14 rows): `atan(I^3*x^0)`, `atan(1/I*x^0)`, `atan(I^(-1)*x^0)` are
  `DivisionByZero`; `asinh(1/I*x^0)` is -i pi/2. The documented convention keeps
  `atan(sqrt(-1)*x^0)` as a kernel (`series.rs:39`); that holds.
- Kernels of exact-constant arguments are rendered from the constant, not from the input node:
  `asinh(x/x) + x` is `asinh(1) + x`, `sin(x/x) + x` is `sin(1) + x`.

## 6. The singular-point matrix

All 485 rows of classes (a), (c) and (f) through my driver and oracle (`fz/mat.tsv`): the 353 value
rows pass on both sides, the 115 `DivisionByZero` rows are exact zeros by gp, the 17 `NotLaurent`
rows are the family 0^I. This checks the matrix's gp references with an oracle that does not use
gp's series.

Completeness: the matrix has the 23 (function, branch point) pairs, an exact-constant and a
non-constant row for each, the branch point at infinity per function, the guard rows (e) and the
inverted values (f). That is what the owner asked for. One shape is missing (minor m1).

## 7. Parser fuzz

`gen_parse.py` (seeds 11 to 13; 7,500 texts): random trees of small and up to 60-digit integers with
`+ - * / ^`, small and negative exponents and towers, sign runs of 1, 3 and 5 minus signs, random
spaces and redundant parentheses, printed with conventional precedence (`-a^b = -(a^b)`,
`a^-b^c = a^(-(b^c))`, a sign after `* / ^` binds the next factor). gp evaluates the fully
parenthesised tree. Port and HEAD: `parse`, then `simplify` (`drv/pdrv.rs`).

| | port | HEAD |
|---|---|---|
| equal to gp | 7,213 | 2,984 |
| different from gp | 0 | 7 (HEAD's `a/b` token: `9 /1/420966` is 3,788,694, `2  / 2/9` is 9) |
| parse error where gp has a value | 0 | 4,220 |
| gp fails (division by zero): parse error (`1/0` refused, as designed) / a division-by-zero form | 154 / 129 | |
| gp fails, a value: `(...)^0` and `0*(...)` of an undefined operand folded by HEAD's `simplify` (HEAD gives the same) | 4 | |

36 edge cases by hand (`ps/edge.tsv`: `4/-2*3`, `2^-3^2`, `-2^-2^2`, `3^-2^-1`, `2---3`, `- - -2`,
`12/-3/-2`, `-12/3*-2`, `2^-1/2`, `1/2^-1`, ...) equal gp.

## 8. `simplify` against HEAD on DAGs

`drv/sdag.rs`, the same seeded generator compiled against HEAD's rlib and the port's: random DAGs of
depth 2 to 8 with a pool of shared nodes and clones held outside the walk (REPL variables), leaves
x, y, pi, small, large and rational literals; printed by a structural printer (Display changed by
design). Seeds 1 to 8, 5,000 DAGs each: the outputs are byte-identical (`sdag/port_*.out`,
`sdag/head_*.out`). Round 4's memo by in-DAG parents does not change a value.

## 9. Minor issues and notes

### m1. The matrix has no row where the constant term reaches the branch point through a folded value

Before round 4, `asin(2*asin(x^0)/pi + x)` was `UndecidedZero` (2 arcsin(1)/pi was a kernel). Now it
is `NotLaurent`, and `asin(2*asin(x^0)/pi)` is pi/2, `atanh(4*atan(x^0)/pi)` is `DivisionByZero`. My
Q class shows these outcomes are right, but no test pins them. Suggested: rows in (d) or a new class,
one pair per inverse function, such as `asin(2*asin(x^0)/pi + x)` (`NotLaurent`),
`atanh(4*atan(x^0)/pi + x^2)` (`NotLaurent`), `atan(2*asinh(I*x^0)/pi + x)` (`NotLaurent`), and the
exact-constant forms (`asin(2*asin(x^0)/pi)` = pi/2, gp `asin(1)`; `atanh(4*atan(x^0)/pi)`,
`DivisionByZero`).

### m2. Functions of the new imaginary exact values stay opaque

`cosh(i pi)`, `exp(i pi)`, `exp(2 pi i)`, `sinh(i pi/2)`, `cos(2 pi i)` are kernels without an
enclosure, so `1/(exp(x) + 1)` at `acosh(-1)` (= i pi), `tanh(x)` and `1/cosh(x)` at `asinh(I)`,
`log(cos(x))` at `2*acosh(-1)` are `UndecidedZero` (190 P rows, 94 R rows). Honest, so below the bar.
The fold of sin, cos and tan at q pi (`as_q_pi`) could take exp, sin, cos, sinh, cosh at q i pi the
same way (exp(q i pi) = cos(q pi) + i sin(q pi), cosh(q i pi) = cos(q pi)). A handoff next to m2 of
round 4.

### Notes

- gp artefacts met by the oracle: `acos` and `asin` near +-1 lose half the precision (my first run
  at 300 digits flagged 25 rows, all right at 600); a float exact zero to the power 1/3 is 1e-200;
  `sinh(sinh(exp(4 pi/3)))` overflows.
- `x/(c)` with an undecided c and `atan(sqrt(-1)*x^0)` follow the documented convention of
  `series.rs:37` (the result holds where the input is defined).
- The six one-sided rows of the matrix-free fuzz are all at points of the imaginary-axis cuts of
  atan and asinh; the engine takes the principal value at the point, which is the documented
  convention.

## 10. Reproduction

```
R=/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/s1impl/rev5-semantics
cd $R/drv && ./build.sh r5.rs dbg r5_dbg          # rel needs target/release (r5_rel kept)
cd $R/drv && ./build.sh head_taylor.rs head head_taylor; ./build.sh pdrv.rs dbg pdrv_dbg; ./build.sh pdrv.rs head pdrv_head
cd $R && VAR=pi ./drv/r5_rel ser hand/vpi.tsv 8388608; VAR=I ./drv/r5_rel ser hand/vi.tsv 8388608   # B1
cd $R && VAR=pi ./drv/head_taylor hand/vpi.tsv; VAR=I ./drv/head_taylor hand/vi.tsv                 # HEAD
cd $R && ./drv/repl_probe hand/repl_pi2.txt; ./drv/repl_probe hand/repl_i.txt                        # REPL
cd $R/fz && ./runall.sh 501 502 503                        # gen5.py + r5_rel + run5.py (600 digits)
cd $R/fz && DRV=../drv/r5_dbg python3 -I ../run5.py mat     # the matrix rows
cd $R/ps && python3 -I ../gen_parse.py 12 3000 q12 && python3 -I ../runparse.py q12
cd $R && for s in 1 2 3 4 5 6 7 8; do ./drv/sdag_port $s 5000 > a; ./drv/sdag_head $s 5000 > b; cmp a b; done
cd $R/r4b && ./p5_dbg ../../rev4-resources/chainT/div3_12k.txt 0 1 8388608 tw,simp,sdisp; OP=mul3 ./chain_t2_dbg 6000 67108864
```

`repl_probe` links `target/debug/librustmath_jupyter.rlib` (fresh at the time). Binaries kept:
`r5_dbg`, `r5_rel`, `head_taylor`, `pdrv_dbg`, `pdrv_head`, `sdag_port`, `sdag_head`, `repl_probe`,
`r4b/p5_dbg`, `r4b/chain_t2_dbg`.
