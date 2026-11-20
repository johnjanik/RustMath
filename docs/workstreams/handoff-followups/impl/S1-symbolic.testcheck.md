# S1 tests written first: independent check, round 2

Checked: `fu2/s1impl/tests_report.md` (round 2) and the files it names: `rustmath-symbolic/Cargo.toml`,
`tests/s1_regressions.rs`, `tests/s1_singular_points.rs`, `tests/s1_fuzz_data.rs`, `tests/s1_common/*`
and `tests/data/*`. The governing record is `fu2/s1impl/owner_decision.md`. Branch
`claude/gracious-gates-e6b6l7`, HEAD 8a0b310. I edited none of the checked files. My scratch is
`fu2/s1impl/testcheck/r2/`. The round-1 check is kept as `fu2/s1impl/testcheck.round1.md`.

Verdict: **approved**. No blocker. Round 1's T1 and T2 are resolved, and every round-1 minor was
handled. I found no wrong expected value and no missing branch point. The minors below are
optional improvements.

## What I ran

- Build-green. `cargo test -j 2 -p rustmath-symbolic --test s1_regressions --test s1_singular_points
  --test s1_fuzz_data` passes, with 0 tests in each target (the `s1-engine-tests` gate). `git status`
  shows only `Cargo.lock`, `rustmath-symbolic/Cargo.toml` and `rustmath-symbolic/tests/`. No em-dash
  and no `unsafe` in the new files. Each header cites the ch. 49 digest and the papers.
- An independent value check (`r2/numcheck.py`, my own code). It uses no series code of gp. For each
  value outcome I built the real-variable function from gp scalar functions with my own parser of the
  §2.12 grammar. The inverse functions use the principal log formulas of `eval.rs`. Every non-integer
  power is `exp(a log b)`. `sign` and `abs` act on the real value. Then gp's `derivnum` at
  realprecision 300 gives the Taylor coefficients at P + sqrt(2)·10^-150. For a pole of order v, I
  expanded f·(x-P)^(-v). I compared each coefficient at 1e-60, scaled by max(1, |gp|). A perturbation
  test showed that the check catches a change at digit 58.
  - Matrix: 228 of 228 value outcomes agree, 188 (a) and 40 (c). The largest difference is 2e-64, and
    it comes from my numeric noise on an exact-zero shape.
  - Fuzz: 377 of 377 value outcomes agree. That is 360 V rows, 14 B1 rows and the value outcomes of
    O:y13.36, O:z2.10 and O:y11.18. The largest difference is 4.8e-70. This includes the two T1 rows,
    `atan(atanh(3))` and `(1/2)^(asin(2*asin(pi)))`, and the review-9 m1 rows that need the
    principal power.
- Other gp checks (`r2/f23*.gp`, `r2/t1.gp`). F_2 at 1 (c0, c1), log(c0) and c1/c0 of F_3 agree
  with the data to 1e-70 relative. This uses log F_3 = 2 log(1 + exp(F_2)) and exp(-F_2(1)) <
  10^(-4·10^11). Also checked: f(-0.1) = 2.41e62 and f(-0.05) = 1.12e566 for series H1, the values
  -3^(1/3) and pi/2 + 1, and 1/sqrt(10). The cut conventions of `eval.rs` match gp at 3/2 and -3/2
  for asin, acos, atanh and acosh, and at 1/2 and -1/2 for acosh.
- Sample. `r2/replay.py` runs review 9's own `fuzz.py` and `fuzz2.py` code (exec of their
  `leaf`/`expr`) for seeds 1-4 and 11-14. All 472 data rows equal the replay in id, input, point and
  order. They also equal the first columns of `fz_*.out` and `fy_*.out`. The 8 left-out rows are the
  E rows.
- Both generators reproduce their files byte for byte: `python3 -I gen_s1_singular.py` and
  `TMPIN=... python3 -I gen_s1_fuzz.py`. The matrix's 420 round-1 rows are unchanged. 48 rows are new.
  Every fuzz change is the one the report lists.
- Prototypes, through a copy of the harness in `r2/ph/`, debug, with 2 MB rows.
  - On proto13: `s1_regressions` has 19 passes, 6 failures and 2 ignored tests. `s1_singular_points`
    fails on 98 of 268 (a) rows and 16 of 160 (b) rows, and all (c) rows pass. `s1_fuzz_data` fails
    on 19 of 22 hand-reviewed rows and 1 of 4 O rows (y13.36). Every failure is B1, B2 or m2, as the
    report says.
  - On proto12 (revision 8): the four `rt4_*` tests fail (sentinel precision, Ok(0), the
    capacity-overflow panic). The B2 tests pass. `b2_controls` fails on `x + atanh(1)`. So the
    new tests catch the round-4 holes, and the B2 rows catch a regression of 7rt4.
- Limit column. I recomputed the expected limit of every V row from gp's series with the rules of
  §2.13. 241 are Unknown and 107 are Finite(q). In the 6 Finite rows where gp prints decimals, the
  prototype gives the rational, by the design's log(e^k) rule and others. The 10 rows that accept
  `Finite(q),Unknown` are rational. The 2 `-` rows are sign kernels that a port may fold.

## Round 1 items

| item | status |
|---|---|
| T1 (y14.33, y13.19) | resolved: B1 value rows, gp's values (I checked them independently). I reread every UndecidedZero row (3 R, 11 U). None is an atan-family function of an exact constant. The integrand-factor detection plus `INTEGRAND_NOT_CONSTANT` covers the class (8 rows, each operand checked non-constant by hand). |
| T2 (scope) | resolved. The scope is stated once in `s1_common/mod.rs` and in the report. It is mandatory for the six inverse functions and the Pow, log and inv rules, and optional for the rest. That matches the owner's text, which is about inverse functions. I checked every R, U, O and B row for a dependence on an optional fold. The only ones are the four O rows and `atanh(exp(x-x))`, and each accepts every honest outcome. No Unsupported row is abs or sign of an exact constant. The truncated row `atanh(1+(sin(x)^2+cos(x)^2-1))` keeps B2's truncated case pinned to PrecisionExhausted, and proto13 fails it. |
| m1 to m10 | handled as the report says. I checked m2 (F_12 only refused), m3 (F_3 values), m4 (debug consumers pass on proto13), m5 (per-input timing), m6 (48 rows, values checked), m8 (convention stated in `eval.rs`), m9 and m10 (documented). |

## Coverage (holds)

| item | where | expected outcome, checked |
|---|---|---|
| B1, 10 rows of review 9 | `b1_exact_constant_operand_is_evaluated` | values pi/2, 0, x, exp(x), x, pi/2 + x, pi/2 + 1 + h, 0, i pi, i pi/2; limit Finite only for a rational c0 |
| B1 controls, m2, m3 | `b1_controls`, `b1_m2_*`, `b1_m3_*` | DivisionByZero at arctanh(1), arctan(+-I); NotLaurent at branch points; m3 PrecisionExhausted or the value |
| B1 at regular points | `b1_exact_constant_operand_at_a_regular_point`, 8 fuzz B1 rows | exact 0 values; DivisionByZero only for log(0), 1/0 |
| B2, 5 rows + `atanh(1+x^100)` | `B2_ROWS`, `B2_OPTIONAL_FOLD`, `B2_TRUNCATED`, `b2_controls` | NotLaurent; PrecisionExhausted or DivisionByZero (optional exp fold); PrecisionExhausted; DivisionByZero for constants |
| Hole: empty truncated argument of negative precision | `rt_series_h1_*` (5 tests) | every row of red team 3 series H1 (9 inputs, infinity, the limit path, order 20000, controls) |
| Hole: 0^r from syntax | `rt_structure_h1_*` | the 8 inputs of red team 3 structure H1 and the 6 refusals of §4.1 `zero_power_rule` |
| Hole: trees exponential in their DAG | `rt_structure_h2_*` (5 tests) | F_22, F_26, F_30 at order 0 with consumers; F_2 and F_3 equal gp; F_10 and F_11 Ok with consumers on 2 MB debug; F_12 TreeSize |
| Round-4 holes | `rt4_*` (4 tests) | §4.1 `exact_sentinel` and `dense_buffer_units` rows with the design's kinds; I checked the kinds against proto13 at the tests' orders |
| Matrix | 20 (function, branch point) pairs, 468 rows | arcsin, arccos, arccosh, arctanh at +-1; arctan, arcsinh at +-I; log, sqrt, u^(1/3), u^(2/3), u^(-1/2), u^pi, u^(x+1), u^(2+sin(x)) at 0. These are all the finite branch points of the design's inverse and branch functions. Gamma, zeta and factorial are Unsupported; erf, exp and the trigonometric functions are entire or have poles only. (a) has 9 or 10 exact shapes plus sums, as values or DivisionByZero at a pole of the constant. (b) has 7 shapes plus `x + 0*f`. I verified each (b) input by hand to be non-Laurent as a real-variable function. No shape has an even valuation with an even-integer power, where the function would be Laurent (`sqrt(x^4)`). (c) has 2 neighbours each, cuts included. |
| Random harness | 472 rows | the first 60 of each of review 9's 8 runs; refusal kinds reviewed. Each DivisionByZero has an exact zero divisor or an exact pole. Each NotLaurent is a genuine non-Laurent point. Each U row is an honest limit of the design's constant field or of its real-only enclosures (§2.3). |

Tolerance (60 digits, scaled by max(1, |gp|)): justified. gp works at 300 digits with 31 spare terms
and a `serprec` check. The data is within 5e-70 of my independent derivnum values. No nonzero
coefficient in either file lies between 1e-250 and 1e-40, so the absolute floor of 1e-60 masks no
true coefficient. The values below 1e-250 are gp's rounding noise on exact zeros.

Budget tests: they measure the bar. Each input is timed on its own against 20 s in release. VmHWM is
checked against 512 MB, which is stricter than the bar. Release threads have 8 MB and debug threads
2 MB, and every row runs on the bar's stack.

## Minor issues (optional)

- n1. The report's "Intended refusals" lists the 20 `b:*:sub` rows and `x + 0*atanh(1+x)`. It omits
  12 more rows of the same class: the matrix rows `a:*:1+0x` and `a:*:7+0x` at the poles
  (arctan@+-I, arctanh@+-1, log@0, pow(-1/2)@0). They expect DivisionByZero, and HEAD's taylor
  returns x (I checked `x + 0*arctan(I*x^0)` and `x + 0*arctanh(x^0)`; HEAD also returns x on 7
  `sub` rows). This follows §2.5 ("a constant factor 0 ... their errors still surface"). It is
  below the bar by the precedent of review 8 m4 and review 9. The final review should get the full
  list.
- n2. The pow families have no complex constant exponent. `(x-x)^I` at 0 is NotLaurent by §2.5's
  `zero_pow_sign` (proto13 agrees), which is a third outcome for an exact-constant argument. The
  matrix's (a) class (a value or DivisionByZero) does not show it. `(x-x)^(1/2+I)` is 0 and
  `(x-x)^(-1/2+I)` is DivisionByZero. Three rows would complete the family.
- n3. The (a) shapes do not use three of the producers the scope statement lists: abs of a rational
  series, a power with an exact-constant exponent, and a nested inverse function. Examples are
  `arcsin(abs(-x^0))`, `arcsin(x^(x-x))` and `arcsin(1 + arcsin(x-x))`. proto13 gives
  DivisionByZero, DivisionByZero and PrecisionExhausted(131). Fuzz B1:y11.45 and the regular-point
  rows cover parts of this class.
- n4. In `rt4_series_h2_dense_buffer_units`, the sweep k = 57 to 62 has no binomial power (c0
  Unknown) and no `pow_int` Miller power (n > 64). Only the `^65` text row covers the Miller path, at
  one size. §4.1 lists "Miller and binomial powers, inv".
- n5. `rt_structure_h2_consumers_release` times only `taylor_with_budget` for F_10 to F_12. The bar
  names `series_expansion` and the limit path too. For F_12 they run only in the debug test, with
  no time bound.
- n6. The 8 left-out E rows include 7 values that are easy to check by hand: z3.22, z4.37, z4.58
  (0, from an exact zero base to `sin(x)+1`, structure H1's class), z4.41 (2 + h/2), z4.53
  (5 - pi/2 + 26h), y11.24 and y14.58 (1). Each could be added with gp on the simplified input, as
  B1:z1.43 is. z1.9 (`log(log(K))`, §7 item 10) is right to leave out.
- n7. Some refusal rows pin the frozen design's constant field where the true answer differs.
  R:y11.25 is UndecidedZero, but the true answer is NotLaurent: acos at -1 behind pi·exp(-log pi).
  The 11 U UndecidedZero rows have a gp value. The V limits were relaxed for this reason
  (`Finite(q),Unknown`), but these rows were not. A port that decides more would fail them with a
  correct answer. The report says so for y11.25. Accepting both outcomes would be more robust.
- n8. R:y13.13 has two refusal causes. Its numerator is sign of non-rational coefficients
  (Unsupported). Its denominator is acos(x) at 1 (NotLaurent). The expected kind follows the
  design's left-to-right `* /` chain, so the row depends on the evaluation order.
- n9. Red team 3 series H1 noted that `limit_enhanced` gave Finite(1) from the left as well. The
  limit tests check only the two-sided `limit_via_series`.

## Files

- `fu2/s1impl/testcheck/r2/numcheck.py`: the derivnum value check. It writes `*.tsv.gp`, the gp
  programs it runs.
- `fu2/s1impl/testcheck/r2/replay.py`: the replay of review 9's random sequences.
- `fu2/s1impl/testcheck/r2/f23.gp`, `f23b.gp`, `t1.gp`: the F_2/F_3, constant and convention checks.
- `fu2/s1impl/testcheck/r2/sp_regen.tsv`, `fz_regen.tsv`: the regenerated data (identical).
- `fu2/s1impl/testcheck/r2/ph/`: the harness copy, with outputs `s1_*.out` (proto13) and
  `reg_proto12.out` (revision 8). I deleted its binaries.
